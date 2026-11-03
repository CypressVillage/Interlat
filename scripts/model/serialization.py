"""Appendix-H ChatML and supervised positions shared by training and generation."""
from dataclasses import dataclass
import hashlib

BOP, EOP = '<bop>', '<eop>'
TEMPLATE = 'Appendix H explicit ChatML; content+im_end+newline assistant labels; injection before first user im_end'


@dataclass
class RenderedInput:
    input_ids: list[int]
    labels: list[int]
    injection_index: int  # Inside the first user turn, before its im_end.
    turn_spans: list[tuple[int, int, str]]
    template_hash: str


def bop_eop_ids(tok):
    ids = tuple(tok.convert_tokens_to_ids(token) for token in (BOP, EOP))
    if any(value is None or value == tok.unk_token_id for value in ids) or ids[0] == ids[1]:
        raise ValueError('distinct <bop>/<eop> special tokens are required')
    return ids


def render(tok, messages, generation=False):
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
