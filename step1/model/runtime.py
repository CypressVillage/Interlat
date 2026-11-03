"""Frozen Base runtime for the independently authorized paper-alignment variant.

Rendering lives in serialization; the Receiver is always the locked 7B Base.
"""
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from step1.adapter import ReceiverWithLatent, Step1Adapter
from step1.common import (RECEIVER_MODEL as BASE, RECEIVER_REVISION as BASE_REVISION,
    HIDDEN_SIZE, ADAPTER_HEADS, ADAPTER_PARAM_COUNT)
from step1.serialization import BOP, EOP, render


def tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    tok.add_special_tokens({'additional_special_tokens': [BOP, EOP]})
    tok.pad_token = tok.eos_token
    return tok


def load_receiver(checkpointing=False):
    tok = tokenizer()
    base = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    receiver = ReceiverWithLatent(base, tok, Step1Adapter(HIDDEN_SIZE, ADAPTER_HEADS).cuda())
    report = receiver.freeze_base_and_assert(ADAPTER_PARAM_COUNT)
    if checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    receiver.eval()
    return receiver, tok, report


def load_rows(path, tok, latent_dir=None, limit=0):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    for row in rows:
        row['rendered'] = render(tok, row['messages'])
        if latent_dir is not None:
            row['plan'] = json.loads((Path(latent_dir)/(row['id']+'.json')).read_text())
            # Public collection decodes plans with skip_special_tokens=True and
            # training tokenizes that content, not the Sender's turn terminator.
            # Keep original cache intact; omit the terminal EOS on BOTH channels.
            tokens = row['plan']['token_ids']
            if row['plan']['ended_eos']:
                assert tokens[-1] == tok.convert_tokens_to_ids('<|im_end|>')
            row['plan']['communication_token_ids'] = tokens[:-1] if row['plan']['ended_eos'] else tokens
            assert row['plan']['communication_token_ids'], 'empty plan content'
            row['latent_path'] = str(Path(latent_dir)/(row['id']+'.pt'))
    return rows


def latent(row):
    h = torch.load(row['latent_path'], map_location='cpu', weights_only=True)
    assert list(h.shape) == [len(row['plan']['token_ids']), HIDDEN_SIZE]
    assert torch.isfinite(h).all()
    return h[:len(row['plan']['communication_token_ids'])].cuda().float()


def adapted(receiver, row):
    return receiver.adapter.process_hidden_states(latent(row))


def text_message(receiver, row):
    return receiver.base_model.get_input_embeddings()(
        torch.tensor(row['plan']['communication_token_ids'], device='cuda'))


def build(receiver, rendered, message):
    item = {'rendered': rendered}
    if message is None:
        return receiver.build_no_injection_batch([item])
    return receiver.build_training_batch([{**item, 'injected': message}], latent_key='injected')


def supervised_hidden(receiver, rendered, message):
    batch = build(receiver, rendered, message)
    out = receiver.base_model.model(inputs_embeds=batch['input_embeds'],
        attention_mask=batch['attention_mask'], use_cache=False, return_dict=True)
    labels = batch['labels'][:, 1:]
    mask = labels != -100
    return out.last_hidden_state[:, :-1][mask], labels[mask]


@torch.no_grad()
def nll(receiver, row, message):
    h, y = supervised_hidden(receiver, row['rendered'], message)
    total = 0.
    for s in range(0, len(y), 64):
        logits = receiver.base_model.lm_head(h[s:s+64]).float()
        total += float(torch.nn.functional.cross_entropy(logits, y[s:s+64], reduction='sum'))
    return {'nll': total/len(y), 'tokens': len(y)}


def parameter_fingerprint(model):
    """Full tensor checksum, including the initialized frozen delimiters."""
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
