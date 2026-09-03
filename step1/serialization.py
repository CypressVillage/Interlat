"""Step 1A input serialization: locked tokenizer + apply_chat_template only.

Protocol §4: Sender and Receiver inputs are rendered with the locked
tokenizer's own `apply_chat_template`; no handwritten ChatML, no raw-text
tokenization. `<bop>`/`<eop>` are added as special tokens; the latent message
is injected once after the first user turn, before the first assistant header
(at embedding level, by the model wrapper).

Receiver message structure (fixed for all groups):
  user turn 1: "The task is: {task_description}\nInitial observation: {initial_observation}"
  assistant turn k: "Thought: {thought}\nAction: {action}"
  user turn k>=2: "Observation: {obs}"

Sender message (single user turn, fixed):
  "Please provide a general plan to solve this task.

   The task is: {task_description}
   Initial observation: {initial_observation}"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from step1.common import MODEL_ID, MODEL_REVISION

BOP, EOP = "<bop>", "<eop>"


def load_locked_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    added = []
    for special in (BOP, EOP):
        tid = tok.convert_tokens_to_ids(special)
        if tid is None or tid == tok.unk_token_id:
            added.append(special)
    if added:
        tok.add_special_tokens({"additional_special_tokens": added})
    return tok


def bop_eop_ids(tok) -> Tuple[int, int]:
    bop_id = tok.convert_tokens_to_ids(BOP)
    eop_id = tok.convert_tokens_to_ids(EOP)
    if bop_id is None or eop_id is None or bop_id == tok.unk_token_id or eop_id == tok.unk_token_id:
        raise RuntimeError("<bop>/<eop> missing from tokenizer")
    return bop_id, eop_id


# ------------------------------------------------------------------ messages
def sender_user_content(task_description: str, initial_observation: str) -> str:
    return (
        "Please provide a general plan to solve this task.\n\n"
        f"The task is: {task_description}\n"
        f"Initial observation: {initial_observation}"
    )


def sender_messages(task_description: str, initial_observation: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": sender_user_content(task_description, initial_observation)}]


def receiver_first_user_content(task_description: str, initial_observation: str) -> str:
    return (
        f"The task is: {task_description}\n"
        f"Initial observation: {initial_observation}"
    )


def receiver_step_user_content(observation: str) -> str:
    return f"Observation: {observation}"


def receiver_assistant_content(thought: str, action: str) -> str:
    return f"Thought: {thought}\nAction: {action}"


# ------------------------------------------------------------------ rendering
@dataclass
class RenderedInput:
    input_ids: List[int]
    labels: List[int]              # -100 for non-assistant-response tokens
    injection_index: int           # token index AFTER first user turn segment
    turn_spans: List[Tuple[int, int, str]]  # (start, end, role) per turn segment
    template_hash: str


def render_with_labels(tok, messages: List[Dict[str, str]], supervision: bool = True) -> RenderedInput:
    """Render `messages` via apply_chat_template and compute per-token labels.

    Labels: -100 everywhere except the *content + turn terminator* of assistant
    messages (the assistant header itself is masked). Increments between
    consecutive prefixes must be exact suffix extensions (Qwen2 template is
    append-only; verified here, not assumed).

    injection_index = token length after the first user turn (start of the
    next segment). With messages[0] = user, the latent is spliced at that
    embedding index once.
    """
    if not messages or messages[0]["role"] != "user":
        raise ValueError("messages must start with a user turn")

    prefix_ids: List[int] = []
    spans: List[Tuple[int, int, str]] = []
    labels: List[int] = []

    for k, msg in enumerate(messages):
        sub = messages[: k + 1]
        ids = tok.apply_chat_template(sub, tokenize=True, add_generation_prompt=False)
        if ids[: len(prefix_ids)] != prefix_ids:
            raise AssertionError(
                "chat template is not prefix-stable at turn "
                f"{k} (role={msg['role']}); cannot compute spans"
            )
        seg = ids[len(prefix_ids):]
        start = len(prefix_ids)
        if msg["role"] == "assistant" and supervision:
            # header = segment of an assistant turn with empty content minus terminator
            term_ids = tok(tokenizer_terminator_text(tok), add_special_tokens=False)["input_ids"]
            empty_ids = tok.apply_chat_template(
                messages[:k] + [{"role": "assistant", "content": ""}],
                tokenize=True, add_generation_prompt=False,
            )
            empty_seg = empty_ids[len(prefix_ids):]
            header_len = len(empty_seg) - len(term_ids)
            if header_len < 0 or header_len > len(seg):
                raise AssertionError(f"bad header length at assistant turn {k}")
            seg_labels = [-100] * header_len + seg[header_len:]
        else:
            seg_labels = [-100] * len(seg)
        spans.append((start, start + len(seg), msg["role"]))
        labels.extend(seg_labels)
        prefix_ids = ids

    injection_index = spans[0][1]

    return RenderedInput(
        input_ids=prefix_ids,
        labels=labels,
        injection_index=injection_index,
        turn_spans=spans,
        template_hash=_template_hash(tok),
    )


_TERMINATOR_CACHE: Dict[int, str] = {}


def tokenizer_terminator_text(tok) -> str:
    """Per-message terminator appended by the Qwen2 chat template (strictly verified)."""
    key = id(tok)
    if key not in _TERMINATOR_CACHE:
        _TERMINATOR_CACHE[key] = verify_terminator(tok)
    return _TERMINATOR_CACHE[key]


def verify_terminator(tok) -> str:
    """Strict check that the template appends exactly '<|im_end|>\\n' per message."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": "A"}], tokenize=False, add_generation_prompt=False
    )
    if not text.endswith("A<|im_end|>\n"):
        raise AssertionError(f"unexpected chat template suffix: {text[-40:]!r}")
    return "<|im_end|>\n"


def _template_hash(tok) -> str:
    import hashlib
    tpl = getattr(tok, "chat_template", "") or ""
    return hashlib.sha256(tpl.encode("utf-8")).hexdigest()[:16]


def decode_supervised_span(tok, rendered: RenderedInput) -> List[str]:
    """Debug helper: decoded supervised tokens (labels != -100)."""
    out = []
    for i, lab in enumerate(rendered.labels):
        if lab != -100:
            out.append(tok.decode([rendered.input_ids[i]]))
    return out
