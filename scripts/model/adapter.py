"""Shared paper-aligned adapter and Receiver-side embedding injection.

Step1Adapter is exactly the upstream full-length `process_hidden_states`
chain (upstream core_training/hidden_model/custom_model.py:16-177):
  pre_ln(LN eps=1e-6) -> 8-head self-attention (dropout 0.1) -> residual ->
  post_ln(LN eps=1e-6) -> AdaptiveProjection(Linear-GELU-LN-Linear + 2 scales)
with NO clamp anywhere (upstream wrapper.forward clamps; we never call it).
Adapter computation stays float32, with no unused attention-weight output.
Trainable parameters = hidden_mha + pre_ln + post_ln + adaptive_proj
= 6D² + 12D + 2 (4,827,650 at D=896; 77,113,346 at D=3584).
Root-level `scale`/`output_scale` alias registrations do not
exist here; the two underlying Parameters stay inside adaptive_proj.

ReceiverWithLatent splices [<bop>; A_theta(H_i); <eop>] once at the rendered
injection index, inside the first user turn. Continuous positions get attention mask 1 and
labels -100. Delimiter embeddings are initialized to the mean of the original
input-embedding vocabulary; all token embeddings (tied LM head included) are
then frozen. The convenience forward computes CE; the three-branch paper
trainer uses runtime and its own loss functions for its additional objectives.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from scripts.common import HIDDEN_SIZE, ADAPTER_HEADS

class AdaptiveProjection(nn.Module):
    """Upstream projection retained locally without the legacy training pipeline."""

    def __init__(self, hidden_size):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.proj[0].weight, mean=0, std=0.02)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, x):
        residual = x * self.scale
        x = self.proj(residual)
        return (residual + x) * self.output_scale


class Step1Adapter(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_heads: int = ADAPTER_HEADS):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.adaptive_proj = AdaptiveProjection(hidden_size)
        self._init_mha_weights()

    def _init_mha_weights(self):
        nn.init.xavier_uniform_(self.hidden_mha.in_proj_weight, gain=1.0 / math.sqrt(3))
        nn.init.xavier_uniform_(self.hidden_mha.out_proj.weight, gain=1.0)
        if self.hidden_mha.in_proj_bias is not None:
            nn.init.constant_(self.hidden_mha.in_proj_bias, 0.0)
            nn.init.constant_(self.hidden_mha.out_proj.bias, 0.0)

    def process_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Paper operator: float32 computation, no clamp or attention-weight output."""
        normed = self.pre_ln(hidden_states.float())
        attn_output, _ = self.hidden_mha(normed, normed, normed, need_weights=False)
        attn_output = self.post_ln(normed + attn_output)
        return self.adaptive_proj(attn_output)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
