"""Sample-specific control constructions for Step 1A (protocol §7).

All controls share delimiter embeddings, position ordering, attention mask,
prefix length and cast (receiver embedding dtype/device) with the Matched group.

- Mismatched: donor Z_j = A_theta(H_j) (adapter applied FIRST), interpolated
  along the sequence axis with F.interpolate(mode="linear", size=L_i,
  align_corners=False) in float32 when L_j != L_i, then rescaled in float32 to
  ||Z_i||_F, then cast to receiver dtype/device. Never modifies H_j before the
  adapter; never pads by repeating the last state. Donor norm zero -> hard error.
  Donor selection: per task type, episodes sorted by SHA-256(draw + "\0" +
  game_file); unique cyclic shift k chosen by (max exact-length pairs, min sum
  |L_i - L_{i+k}|, min k). n < 2 -> hard error.
- Zero: all-zero message, same shape/mask.
- Norm-matched Random: iid Gaussian in float32 scaled to ||Z_i||_F, seeded
  deterministically from (draw, episode_id).
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from step1.common import LATENT_EXPECTED_DIM


def _sha256_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def select_cyclic_shift(rows: List[Dict], draw: int) -> Tuple[int, List[Tuple[str, str]]]:
    """rows: episode rows of one task type with keys episode_id, game_file, L_i.

    Returns (k, pairs) where pairs[i] = (target_episode_id, donor_episode_id).
    """
    n = len(rows)
    if n < 2:
        raise ValueError(f"cyclic shift needs n >= 2, got {n}")
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{draw}\0{r['game_file']}".encode()).hexdigest())

    def shift_score(k: int):
        exact = sum(1 for a, b in zip(ordered, ordered[k:] + ordered[:k]) if a["L_i"] == b["L_i"])
        total_diff = sum(abs(a["L_i"] - b["L_i"]) for a, b in zip(ordered, ordered[k:] + ordered[:k]))
        return (-exact, total_diff, k)

    k = min(range(1, n), key=shift_score)
    pairs = [(r["episode_id"], (ordered[(idx + k) % n])["episode_id"])
             for idx, r in enumerate(ordered)]
    return k, pairs


def build_mismatched_mapping(rows_by_type: Dict[str, List[Dict]], draw: int) -> Dict:
    """rows_by_type: task_type -> rows with episode_id, game_file, L_i.

    Returns {"draw":..., "shifts": {...}, "pairs": {...}, "mapping_sha256": ...}.
    """
    mapping: Dict[str, Dict] = {}
    shifts = {}
    for task_type in sorted(rows_by_type):
        rows = sorted(rows_by_type[task_type], key=lambda r: r["episode_id"])
        k, pairs = select_cyclic_shift(rows, draw)
        shifts[task_type] = k
        for target, donor in pairs:
            mapping[target] = {
                "target_episode_id": target,
                "donor_episode_id": donor,
                "task_type": task_type,
                "draw": draw,
                "shift_k": k,
            }
    blob = json.dumps({"draw": draw, "shifts": shifts, "pairs": mapping},
                      ensure_ascii=False, sort_keys=True)
    return {
        "draw": draw,
        "shifts": shifts,
        "pairs": mapping,
        "mapping_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
    }


def make_mismatched_message(adapter, H_donor: torch.Tensor, L_i: int,
                            norm_target: float, emb_dtype: torch.dtype,
                            device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """Z_j -> interpolate -> rescale to norm_target -> cast. Returns (Z, info)."""
    with torch.no_grad():
        Zj = adapter.process_hidden_states(H_donor.to(device, dtype=torch.float32)).float()
    L_j = Zj.shape[0]
    interpolated = L_j != L_i
    interp_params = None
    if interpolated:
        Zj = F.interpolate(Zj.t().unsqueeze(0), size=L_i, mode="linear",
                           align_corners=False).squeeze(0).t().contiguous()
        interp_params = {"mode": "linear", "size": L_i, "align_corners": False}
    donor_norm = float(Zj.norm())
    if donor_norm == 0.0:
        raise RuntimeError("donor norm is zero; refusing (protocol: hard error)")
    scale = norm_target / donor_norm
    Z = (Zj * scale).to(device=device, dtype=emb_dtype)
    info = {
        "L_donor": L_j, "L_target": L_i, "interpolated": interpolated,
        "interp_params": interp_params,
        "donor_norm_float32": donor_norm,
        "target_norm_float32": norm_target,
        "scale_factor": scale,
    }
    return Z, info


def make_zero_message(L_i: int, emb_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.zeros(L_i, LATENT_EXPECTED_DIM, dtype=emb_dtype, device=device)


LATENT_EXPECTED_DIM = 896


def make_random_message(L_i: int, draw: int, episode_id: str, norm_target: float,
                        emb_dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    g = torch.Generator()
    g.manual_seed(_sha256_int(f"{draw}\0{episode_id}") % (2 ** 63))
    noise = torch.randn(L_i, LATENT_EXPECTED_DIM, generator=g, dtype=torch.float32)
    noise_norm = float(noise.norm())
    scale = norm_target / noise_norm if noise_norm > 0 else 0.0
    Z = (noise * scale).to(device=device, dtype=emb_dtype)
    info = {
        "draw": draw, "episode_id": episode_id, "L_target": L_i,
        "noise_norm_float32": noise_norm, "target_norm_float32": norm_target,
        "scale_factor": scale,
    }
    return Z, info


@torch.no_grad()
def matched_norm_target(adapter, H_i: torch.Tensor, device: torch.device) -> float:
    Zi = adapter.process_hidden_states(H_i.to(device, dtype=torch.float32)).float()
    return float(Zi.norm())


@torch.no_grad()
def make_matched_message(adapter, H_i: torch.Tensor, emb_dtype: torch.dtype,
                         device: torch.device) -> Tuple[torch.Tensor, Dict]:
    Zi = adapter.process_hidden_states(H_i.to(device, dtype=torch.float32))
    return Zi.to(emb_dtype), {"L_target": int(H_i.shape[0])}


