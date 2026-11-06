from dataclasses import dataclass
import hashlib
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.adapter import Step1Adapter
from common import (RECEIVER_MODEL, RECEIVER_REVISION,
    HIDDEN_SIZE, ADAPTER_HEADS, ADAPTER_PARAM_COUNT)

BOP, EOP = '<bop>', '<eop>'
TEMPLATE = 'Appendix H explicit ChatML; content+im_end+newline assistant labels; injection before first user im_end'


@dataclass
class RenderedInput:
    input_ids: list[int]
    labels: list[int]
    injection_index: int  # Inside the first user turn, before its im_end.
    turn_spans: list[tuple[int, int, str]]
    template_hash: str


def render(tok, messages, generation=False):
    """Render Appendix-H ChatML and supervised positions for training/generation."""
    ids, labels, spans = [], [], []
    injection = None
    for msg in messages:
        start = len(ids)
        header = tok.encode('<|im_start|>'+msg['role']+'\n', add_special_tokens=False)
        content = tok.encode(msg['content'], add_special_tokens=False)
        end = tok.encode('<|im_end|>\n', add_special_tokens=False)
        if msg['role'] == 'user' and injection is None:
            injection = start+len(header)+len(content)
        ids.extend(header+content+end)
        labels.extend([-100]*len(header) +
                      (content+end if msg['role'] == 'assistant' else [-100]*(len(content)+len(end))))
        spans.append((start, len(ids), msg['role']))
    if generation:
        header = tok.encode('<|im_start|>assistant\n', add_special_tokens=False)
        ids.extend(header)
        labels.extend([-100]*len(header))
    if injection is None:
        raise ValueError('messages must contain a user turn')
    return RenderedInput(ids, labels, injection, spans, hashlib.sha256(TEMPLATE.encode()).hexdigest())

_ADAPTER_MODULES = ("hidden_mha", "pre_ln", "post_ln", "adaptive_proj")
# params live as "<owner>.<module>.<...>"; owner is "adapter" on ReceiverWithLatent
_ADAPTER_PARAM_PREFIXES = tuple(
    [f"adapter.{m}." for m in _ADAPTER_MODULES] + [f"{m}." for m in _ADAPTER_MODULES]
)

def bop_eop_ids(tok):
    ids = tuple(tok.convert_tokens_to_ids(token) for token in (BOP, EOP))
    if any(value is None or value == tok.unk_token_id for value in ids) or ids[0] == ids[1]:
        raise ValueError('distinct <bop>/<eop> special tokens are required')
    return ids


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
        # Tokenizers can allocate new IDs inside a model's padded vocabulary.
        # Initialize the delimiter IDs explicitly even when no resize is needed.
        mean_vec = emb.weight.detach().mean(dim=0)
        if tok_len > vocab:
            base_model.resize_token_embeddings(tok_len)
        new_emb = base_model.get_input_embeddings()
        with torch.no_grad():
            for row in set(range(vocab, tok_len)) | {self.bop_id, self.eop_id}:
                new_emb.weight[row] = mean_vec.to(new_emb.weight.dtype)
        self.delimiter_init_report = {
            "tokenizer_length": tok_len, "original_embedding_rows": vocab,
            "ids": [self.bop_id, self.eop_id],
            "mean_norm": float(mean_vec.float().norm()),
            "equals_mean": all(torch.equal(new_emb.weight[i], mean_vec.to(new_emb.weight.dtype))
                               for i in (self.bop_id, self.eop_id)),
        }
        assert self.delimiter_init_report["equals_mean"]
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
        """batch items: {"rendered": RenderedInput, "Z": torch.Tensor [L,D]}.

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
        """CE-only forward. The three-branch trainer computes its objectives separately."""
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


def tokenizer():
    tok = AutoTokenizer.from_pretrained(RECEIVER_MODEL, revision = RECEIVER_REVISION)
    tok.add_special_tokens({'additional_special_tokens': [BOP, EOP]})
    tok.pad_token = tok.eos_token
    return tok


def load_receiver(checkpointing=False):
    tok = tokenizer()
    base = AutoModelForCausalLM.from_pretrained(
        RECEIVER_MODEL,
        revision = RECEIVER_REVISION,
        torch_dtype = torch.bfloat16,
        attn_implementation = 'sdpa'
    ).cuda()
    receiver = ReceiverWithLatent(base, tok, Step1Adapter(HIDDEN_SIZE, ADAPTER_HEADS).cuda())
    report = receiver.freeze_base_and_assert(ADAPTER_PARAM_COUNT)
    if checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    receiver.eval()
    return receiver, tok, report
