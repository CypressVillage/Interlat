"""Step 1A adapter and Receiver-side injection (protocol §5, §7).

Step1Adapter is exactly the upstream full-length `process_hidden_states`
chain (core_training/hidden_model/custom_model.py:16-177):
  pre_ln(LN eps=1e-6) -> 8-head self-attention (dropout 0.1) -> residual ->
  post_ln(LN eps=1e-6) -> AdaptiveProjection(Linear-GELU-LN-Linear + 2 scales)
with NO clamp anywhere (upstream wrapper.forward clamps; we never call it).
Trainable parameters = hidden_mha + pre_ln + post_ln + adaptive_proj
= 4,827,650. Root-level `scale`/`output_scale` alias registrations do not
exist here; the two underlying Parameters stay inside adaptive_proj.

ReceiverWithLatent splices [<bop>; A_theta(H_i); <eop>] once after the first
user turn at embedding level; continuous positions get attention mask 1 and
labels -100. Delimiter embeddings are initialized to the mean of the original
input-embedding vocabulary; all token embeddings (tied LM head included) are
then frozen. The ONLY training forward is the single CE forward.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from step1.common import DEFAULT_MODEL_PROFILE, LATENT_EXPECTED_DIM, get_model_profile
from step1.serialization import RenderedInput, bop_eop_ids, render_with_labels

_ADAPTER_MODULES = ("hidden_mha", "pre_ln", "post_ln", "adaptive_proj")
# params live as "<owner>.<module>.<...>"; owner is "adapter" on ReceiverWithLatent
_ADAPTER_PARAM_PREFIXES = tuple(
    [f"adapter.{m}." for m in _ADAPTER_MODULES] + [f"{m}." for m in _ADAPTER_MODULES]
)


def _supervised_token_ce(lm_head, hidden_states: torch.Tensor,
                         labels: torch.Tensor):
    """Compute causal CE without materializing vocabulary logits for masked tokens."""
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    n_tokens = int(mask.sum())
    if n_tokens == 0:
        raise ValueError("no supervised tokens in batch")
    logits = lm_head(shift_hidden[mask])
    loss = torch.nn.functional.cross_entropy(logits.float(), shift_labels[mask])
    return loss, logits, n_tokens


class Step1Adapter(nn.Module):
    def __init__(self, hidden_size: int = LATENT_EXPECTED_DIM, num_heads: int = 8):
        from core_training.hidden_model.custom_model import (
            AdaptiveProjection as UpstreamAdaptiveProjection,
        )

        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.adaptive_proj = UpstreamAdaptiveProjection(hidden_size)
        self._init_mha_weights()

    def _init_mha_weights(self):
        nn.init.xavier_uniform_(self.hidden_mha.in_proj_weight, gain=1.0 / math.sqrt(3))
        nn.init.xavier_uniform_(self.hidden_mha.out_proj.weight, gain=1.0)
        if self.hidden_mha.in_proj_bias is not None:
            nn.init.constant_(self.hidden_mha.in_proj_bias, 0.0)
            nn.init.constant_(self.hidden_mha.out_proj.bias, 0.0)

    def process_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """No clamp. Returns output in the input dtype."""
        orig_dtype = hidden_states.dtype
        normed = self.pre_ln(hidden_states)
        attn_output, _ = self.hidden_mha(normed, normed, normed)
        attn_output = self.post_ln(normed + attn_output)
        projected = self.adaptive_proj(attn_output)
        return projected.to(orig_dtype)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_for_eval_export(self) -> Dict:
        return {
            "hidden_mha": self.hidden_mha.state_dict(),
            "pre_ln": self.pre_ln.state_dict(),
            "post_ln": self.post_ln.state_dict(),
            "adaptive_proj": self.adaptive_proj.state_dict(),
            "scale": float(self.adaptive_proj.scale.detach().item()),
            "output_scale": float(self.adaptive_proj.output_scale.detach().item()),
        }

    def load_eval_export(self, state: Dict):
        self.hidden_mha.load_state_dict(state["hidden_mha"])
        self.pre_ln.load_state_dict(state["pre_ln"])
        self.post_ln.load_state_dict(state["post_ln"])
        self.adaptive_proj.load_state_dict(state["adaptive_proj"])


class ReceiverWithLatent(nn.Module):
    """Frozen base Qwen + trainable Step1Adapter + once-only latent injection."""

    def __init__(self, base_model, tok, adapter: Step1Adapter, device: str = "cuda"):
        super().__init__()
        self.base_model = base_model
        self.tok = tok
        self.adapter = adapter
        self.device = device
        self.bop_id, self.eop_id = bop_eop_ids(tok)

        emb = base_model.get_input_embeddings()
        vocab = emb.weight.shape[0]
        tok_len = len(tok)
        if tok_len > vocab:
            mean_vec = emb.weight.detach().mean(dim=0)
            base_model.resize_token_embeddings(tok_len)
            new_emb = base_model.get_input_embeddings()
            with torch.no_grad():
                for row in range(vocab, tok_len):
                    new_emb.weight[row] = mean_vec.to(new_emb.weight.dtype)
        self.bop_emb_row = self.bop_id
        self.eop_emb_row = self.eop_id
        self.base_model.config.use_cache = False

    # ------------------------------------------------------------- freezing
    def freeze_base_and_assert(self, expected_param_count: int | None = None) -> Dict:
        base = self.base_model
        for p in base.parameters():
            p.requires_grad_(False)
        for p in self.adapter.parameters():
            p.requires_grad_(True)

        trainable = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        for name, _p in trainable:
            if not name.startswith(_ADAPTER_PARAM_PREFIXES):
                raise AssertionError(f"trainable param outside adapter whitelist: {name}")
        total = sum(p.numel() for _n, p in trainable)
        if expected_param_count is None:
            expected_param_count = 6 * self.adapter.hidden_size ** 2 + 12 * self.adapter.hidden_size + 2
        if total != expected_param_count:
            raise AssertionError(f"trainable params {total} != {expected_param_count}")

        emb_params = sum(p.numel() for n, p in base.named_parameters()
                         if "embed_tokens" in n and p.requires_grad)
        lm_head_params = sum(p.numel() for n, p in base.named_parameters()
                             if "lm_head" in n and p.requires_grad)
        base_params = sum(p.numel() for n, p in base.named_parameters() if p.requires_grad)
        if emb_params != 0 or lm_head_params != 0 or base_params != 0:
            raise AssertionError(
                f"base/embeddings/lm_head must have 0 trainable params, got "
                f"embed={emb_params}, lm_head={lm_head_params}, base={base_params}"
            )
        return {
            "trainable_names": sorted({n.split(".")[0] for n, _p in trainable}),
            "trainable_total": total,
            "base_trainable": base_params,
            "embed_trainable": emb_params,
            "lm_head_trainable": lm_head_params,
        }

    # ------------------------------------------------------------- building
    def build_training_batch(self, batch: List[Dict], latent_key: str = "Z") -> Dict:
        """batch items: {"rendered": RenderedInput, "Z": torch.Tensor [L,896] float32}

        `latent` is the message to inject: raw H_i ("raw") or already-processed
        Z_i = A_theta(H_i) ("matched"/"mismatched"/"zero"/"random").
        Returns input_embeds [B,T,D], attention_mask, labels [B,T].
        """
        emb_layer = self.base_model.get_input_embeddings()
        emb_device = emb_layer.weight.device
        emb_dtype = emb_layer.weight.dtype
        bop_emb = emb_layer.weight[self.bop_emb_row].to(self.device)
        eop_emb = emb_layer.weight[self.eop_emb_row].to(self.device)

        per_sample = []
        for item in batch:
            r: RenderedInput = item["rendered"]
            H = item[latent_key]
            if H.dim() != 2 or H.shape[1] != self.adapter.hidden_size:
                raise ValueError(f"latent shape {tuple(H.shape)} invalid")
            if latent_key == "Z":
                with torch.no_grad() if not self.adapter.training else torch.enable_grad():
                    Z = self.adapter.process_hidden_states(H.to(self.device))
                stats = self._stats(Z)
                Z_emb = Z.to(emb_dtype)
            else:
                stats = self._stats(H.to(self.device))
                Z_emb = H.to(device=self.device, dtype=emb_dtype)

            ids = torch.tensor(r.input_ids, device=emb_device)
            tok_embs = emb_layer(ids).to(self.device)  # [n, D]
            inj = r.injection_index
            full = torch.cat([
                tok_embs[:inj],
                bop_emb.unsqueeze(0),
                Z_emb,
                eop_emb.unsqueeze(0),
                tok_embs[inj:],
            ], dim=0)
            labels = torch.tensor(r.labels, device=self.device, dtype=torch.long)
            pad_labels = torch.full((2 + H.shape[0],), -100, device=self.device, dtype=torch.long)
            full_labels = torch.cat([labels[:inj], pad_labels, labels[inj:]], dim=0)
            per_sample.append((full, full_labels, stats, H.shape[0]))

        max_len = max(s[0].shape[0] for s in per_sample)
        B = len(per_sample)
        D = per_sample[0][0].shape[1]
        embed_list, label_list, mask_list = [], [], []
        stats_out = []
        for b, (full, full_labels, stats, L) in enumerate(per_sample):
            n = full.shape[0]
            pad_n = max_len - n
            embed_list.append(torch.nn.functional.pad(full, (0, 0, 0, pad_n)))
            label_list.append(torch.nn.functional.pad(full_labels, (0, pad_n), value=-100))
            mask_list.append(torch.nn.functional.pad(torch.ones(n, dtype=torch.long), (0, pad_n)))
            stats_out.append({**stats, "L": L})
        input_embeds = torch.stack(embed_list, dim=0)
        labels = torch.stack(label_list, dim=0)
        attention_mask = torch.stack(mask_list, dim=0).to(self.device)
        return {
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            "stats": stats_out,
        }

    def build_no_injection_batch(self, batch: List[Dict]) -> Dict:
        """Natural No-Comm forward inputs: same renders, no delimiters/latent."""
        emb_layer = self.base_model.get_input_embeddings()
        emb_device = emb_layer.weight.device
        embed_list, label_list, mask_list = [], [], []
        for item in batch:
            r: RenderedInput = item["rendered"]
            ids = torch.tensor(r.input_ids, device=emb_device)
            full = emb_layer(ids).to(self.device)
            labels = torch.tensor(r.labels, device=self.device, dtype=torch.long)
            n = full.shape[0]
            embed_list.append(full)
            label_list.append(labels)
            mask_list.append(torch.ones(n, dtype=torch.long, device=self.device))
        max_len = max(s.shape[0] for s in embed_list)
        embeds = [torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[0])) for e in embed_list]
        labels = [torch.nn.functional.pad(l, (0, max_len - l.shape[0]), value=-100) for l in label_list]
        masks = [torch.nn.functional.pad(m, (0, max_len - m.shape[0])) for m in mask_list]
        return {
            "input_embeds": torch.stack(embeds, dim=0),
            "attention_mask": torch.stack(masks, dim=0),
            "labels": torch.stack(labels, dim=0),
            "stats": [],
        }

    def forward(self, input_embeds, attention_mask, labels):
        """THE single training forward. Returns loss and supervised-token logits."""
        out = self.base_model.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        loss, logits, _n_tokens = _supervised_token_ce(
            self.base_model.lm_head, out.last_hidden_state, labels
        )
        return loss, logits

    @torch.no_grad()
    def supervised_nll(self, batch_out: Dict) -> Dict:
        """Same-calibration response-token NLL + token count for a built batch."""
        loss, _logits = self.forward(
            batch_out["input_embeds"], batch_out["attention_mask"], batch_out["labels"]
        )
        n_tokens = int((batch_out["labels"][:, 1:] != -100).sum())
        return {
            "nll": float(loss),
            "n_tokens": n_tokens,
        }

    @staticmethod
    def _stats(t: torch.Tensor) -> Dict:
        tf = t.detach().to(torch.float32)
        return {
            "min": float(tf.min()), "max": float(tf.max()),
            "mean": float(tf.mean()), "std": float(tf.std()),
            "norm_f": float(tf.norm()),
        }


def load_frozen_receiver(tok, adapter: Optional[Step1Adapter] = None, device: str = "cuda",
                         model_profile: str = DEFAULT_MODEL_PROFILE,
                         attn_implementation: str = "eager",
                         offload_input_embeddings: bool = False):
    """Load the locked base model, resize for <bop>/<eop>, wrap, freeze, assert."""
    from transformers import AutoModelForCausalLM

    profile = get_model_profile(model_profile)

    base = AutoModelForCausalLM.from_pretrained(
        profile["model_id"], revision=profile["model_revision"], torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).to(device)
    if base.config.hidden_size != profile["hidden_size"]:
        raise AssertionError(
            f"base hidden size {base.config.hidden_size} != profile {profile['hidden_size']}"
        )
    if adapter is None:
        adapter = Step1Adapter(
            hidden_size=profile["hidden_size"], num_heads=profile["adapter_num_heads"]
        ).to(device)
    if adapter.hidden_size != profile["hidden_size"]:
        raise AssertionError(
            f"adapter hidden size {adapter.hidden_size} != profile {profile['hidden_size']}"
        )
    receiver = ReceiverWithLatent(base, tok, adapter, device=device)
    if offload_input_embeddings:
        if base.config.tie_word_embeddings:
            raise ValueError("cannot offload tied input embeddings independently")
        base.get_input_embeddings().to("cpu")
    freeze_report = receiver.freeze_base_and_assert(profile["expected_param_count"])
    return receiver, freeze_report
