"""Sender latent extraction for Step 1A (protocol §5).

For every episode, the frozen Sender selected by --model-profile
greedily decodes the plan (do_sample=false, num_beams=1, max_new_tokens=256,
stop at EOS). For each generated token t we save the last-layer hidden state
that was used to predict t (one-to-one alignment: h_{i,t} <-> token y_t,
including the state that predicts EOS; no padding is ever included).

Implementation: incremental decoding with KV cache. At decoding step t the
forward over the previous token yields hidden_states[-1][:, -1]; its argmax is
y_t. So (hidden_t, y_t) are aligned by construction. Smoke artifacts store
per-position token ids, decoded text, argmax agreement and EOS flags.

Artifacts (under --out-dir):
  latents/{role}/{episode_id}.pt          float32 [L_i, hidden_size] tensor
  latents/{role}/{episode_id}.debug.json  per-position alignment record
  latents_manifest.jsonl                  L_i, shape, SHA-256 per episode
  plan_texts/{role}/{episode_id}.txt      plan text from the same decoding run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from step1.common import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILES,
    SENDER_MAX_NEW_TOKENS,
    get_model_profile,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)
from step1.serialization import load_locked_tokenizer, sender_messages


def load_locked_sender(model_profile: str = DEFAULT_MODEL_PROFILE, device: str = "cuda"):
    from transformers import AutoModelForCausalLM

    profile = get_model_profile(model_profile)
    model = AutoModelForCausalLM.from_pretrained(
        profile["model_id"], revision=profile["model_revision"], torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    if model.config.hidden_size != profile["hidden_size"]:
        raise AssertionError(
            f"sender hidden size {model.config.hidden_size} != profile {profile['hidden_size']}"
        )
    model.eval()
    return model


@torch.no_grad()
def extract_latent(model, tok, messages, device: str = "cuda", max_new_tokens: int = SENDER_MAX_NEW_TOKENS,
                   eos_token_id: int | None = None):
    """Greedy decode with per-token hidden-state extraction.

    Returns dict with tokens (incl. EOS if produced), hidden float32 [L,hidden_size],
    per-position debug rows, plan text.
    """
    if eos_token_id is None:
        eos_token_id = tok.eos_token_id

    input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    input_ids = input_ids.to(device)

    out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    hidden = out.hidden_states[-1][:, -1, :]

    tokens = []
    hidden_rows = []
    debug_rows = []
    position = 0
    while True:
        next_id = int(torch.argmax(logits, dim=-1).item())
        hidden_rows.append(hidden[0].detach().to(torch.float32).cpu())
        tokens.append(next_id)
        debug_rows.append({
            "position": position,
            "token_id": next_id,
            "decoded": tok.decode([next_id]),
            "is_eos": next_id == eos_token_id,
            "hidden_norm_l2": float(hidden[0].detach().to(torch.float32).norm().item()),
        })
        position += 1
        if next_id == eos_token_id or position >= max_new_tokens:
            break
        out = model(input_ids=torch.tensor([[next_id]], device=device),
                    past_key_values=past, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        hidden = out.hidden_states[-1][:, -1, :]

    H = torch.stack(hidden_rows, dim=0)  # [L, hidden_size] float32
    plan_ids = [t for t in tokens if t != eos_token_id]
    plan_text = tok.decode(plan_ids)
    return {
        "tokens": tokens,
        "hidden": H,
        "debug_rows": debug_rows,
        "plan_text": plan_text,
        "input_len_prefill": int(input_ids.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-inputs", required=True, help="episode_inputs.json from gen_trajectories")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES),
                        default=DEFAULT_MODEL_PROFILE)
    args = parser.parse_args()
    profile = get_model_profile(args.model_profile)

    with open(args.episode_inputs, "r", encoding="utf-8") as f:
        episode_inputs = json.load(f)

    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_locked_tokenizer(args.model_profile)
    model = load_locked_sender(args.model_profile, device=device)

    lat_root = os.path.join(args.out_dir, "latents")
    plan_root = os.path.join(args.out_dir, "plan_texts")
    manifest_path = os.path.join(args.out_dir, "latents_manifest.jsonl")
    done_ids = set()
    if os.path.exists(manifest_path):
        for row in read_jsonl(manifest_path):
            existing_profile = row.get("model_profile", DEFAULT_MODEL_PROFILE)
            if existing_profile != args.model_profile:
                raise ValueError(
                    f"existing latent manifest contains profile {existing_profile!r}, "
                    f"requested {args.model_profile!r}; use a separate --out-dir"
                )
            done_ids.add(row["episode_id"])

    manifest_rows = []
    skipped = 0
    for i, row in enumerate(rows):
        eid = row["episode_id"]
        if eid in done_ids:
            skipped += 1
            continue
        info = episode_inputs.get(eid)
        if info is None:
            print(f"[ERROR] no episode_inputs for {eid}", file=sys.stderr)
            return 2
        messages = sender_messages(info["task_description"], info["initial_observation"])
        result = extract_latent(model, tok, messages, device=device)
        role = row["role"]
        os.makedirs(os.path.join(lat_root, role), exist_ok=True)
        os.makedirs(os.path.join(plan_root, role), exist_ok=True)

        pt_path = os.path.join(lat_root, role, f"{eid}.pt")
        torch.save(result["hidden"].contiguous(), pt_path)
        dbg_path = os.path.join(lat_root, role, f"{eid}.debug.json")
        write_json(dbg_path, {
            "episode_id": eid,
            "role": role,
            "task_type": row["task_type"],
            "game_file": row["game_file"],
            "sender_messages": messages,
            "prefill_len": result["input_len_prefill"],
            "rows": result["debug_rows"],
        })
        plan_path = os.path.join(plan_root, role, f"{eid}.txt")
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(result["plan_text"])

        H = result["hidden"]
        manifest_rows.append({
            "episode_id": eid,
            "role": role,
            "task_type": row["task_type"],
            "game_file": row["game_file"],
            "L_i": int(H.shape[0]),
            "dim": int(H.shape[1]),
            "shape": list(H.shape),
            "dtype": "float32",
            "model_profile": args.model_profile,
            "model_id": profile["model_id"],
            "model_revision": profile["model_revision"],
            "tensor_sha256": sha256_file(pt_path),
            "plan_text_sha256": sha256_text(result["plan_text"]),
            "plan_chars": len(result["plan_text"]),
            "hidden_min": float(H.min()), "hidden_max": float(H.max()),
            "hidden_mean": float(H.mean()), "hidden_std": float(H.std()),
        })
        if H.shape[1] != profile["hidden_size"]:
            print(f"[ERROR] {eid}: dim {H.shape[1]} != {profile['hidden_size']}", file=sys.stderr)
            return 2
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_rows[-1], ensure_ascii=False, sort_keys=True) + "\n")
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"[{i + 1}/{len(rows)}] extracted {eid} L_i={H.shape[0]}")

    print(f"[SUMMARY] extracted={len(manifest_rows)} skipped_existing={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
