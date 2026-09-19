"""Step 1A input serialization: locked tokenizer + apply_chat_template only.

Protocol §4: Sender and Receiver inputs are rendered with the locked
tokenizer's own `apply_chat_template`; no handwritten ChatML, no raw-text
tokenization. `<bop>`/`<eop>` are added as special tokens; the latent message
is injected once after the first user turn, before the first assistant header
(at embedding level, by the model wrapper).

Receiver message structure (fixed for all groups):
  system: "You are a helpful assistant."
  user turn 1: upstream ALFWorld instruction + fixed 1-shot format demo
               + task + initial observation
  assistant turn k: "Thought: {thought}\nAction: {action}"
  user turn k>=2: "Observation: {obs}"

Sender message (single user turn, fixed):
  upstream ALFWorld instruction +
  "Please provide a general plan to solve this task.

   The task is: {task_description}
   Initial observation: {initial_observation}"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from step1.common import DEFAULT_MODEL_PROFILE, get_model_profile
from transformers import AutoTokenizer

BOP, EOP = "<bop>", "<eop>"
RECEIVER_SYSTEM_CONTENT = "You are a helpful assistant."
ALFWORLD_RECEIVER_INSTRUCTION = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. toggle {obj} {recep}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""

ALFWORLD_ONE_SHOT_DEMO = """Observation: You are in the middle of a room. Looking quickly around you, you see a countertop 1 and a cabinet 1.
Thought: I should inspect a visible receptacle to find the target object.
Action: go to countertop 1"""
ALFWORLD_SINGLE_ACTION_RULE = (
    "For each turn, output exactly one Thought line followed by exactly one Action line. "
    "Do not propose or output a second action in the same response."
)


def load_locked_tokenizer(model_profile: str = DEFAULT_MODEL_PROFILE):
    profile = get_model_profile(model_profile)
    tok = AutoTokenizer.from_pretrained(
        profile["model_id"], revision=profile["model_revision"]
    )
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
        f"{ALFWORLD_RECEIVER_INSTRUCTION}\n"
        "---\n\n"
        "Please provide a general plan to solve this task.\n\n"
        f"The task is: {task_description}\n"
        f"Initial observation: {initial_observation}"
    )


def sender_messages(task_description: str, initial_observation: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": sender_user_content(task_description, initial_observation)}]


def receiver_first_user_content(task_description: str, initial_observation: str) -> str:
    return (
        f"{ALFWORLD_RECEIVER_INSTRUCTION}\n"
        "---\n"
        "Here is a one-step format example.\n\n"
        f"{ALFWORLD_ONE_SHOT_DEMO}\n"
        f"\n{ALFWORLD_SINGLE_ACTION_RULE}\n"
        "---\n\n"
        "Now, it's your turn and here is the task.\n"
        f"The task is: {task_description}\n"
        f"Initial observation: {initial_observation}\n"
        "Now, you are given a step-by-step plan to complete this task as follow:"
    )


def receiver_initial_messages(task_description: str, initial_observation: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": RECEIVER_SYSTEM_CONTENT},
        {"role": "user", "content": receiver_first_user_content(task_description, initial_observation)},
    ]


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
    next segment). The latent is spliced at that embedding index once.
    """
    if not messages or not any(msg["role"] == "user" for msg in messages):
        raise ValueError("messages must contain a user turn")

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

    first_user_index = next(i for i, msg in enumerate(messages) if msg["role"] == "user")
    injection_index = spans[first_user_index][1]

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
