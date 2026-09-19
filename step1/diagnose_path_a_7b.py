"""Minimal 7B No-Comm ALFWorld check for the advisor's path-A fallback."""

from __future__ import annotations

import argparse
import os
import re
from types import SimpleNamespace

from step1.common import get_model_profile, read_jsonl, write_json, write_jsonl

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from step1.eval_rollout import rollout_episode
from step1.serialization import BOP, EOP

MODEL_PROFILE = "qwen2.5-7b"

LEGAL_ACTION_RE = re.compile(
    r"^(?:go to .+|take .+ from .+|put .+ (?:in|on) .+|open .+|close .+|"
    r"toggle .+ .+|clean .+ with .+|heat .+ with .+|cool .+ with .+)$"
)


def load_model_and_tokenizer(device: str):
    profile = get_model_profile(MODEL_PROFILE)
    tok = AutoTokenizer.from_pretrained(
        profile["model_id"], revision=profile["model_revision"]
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": [BOP, EOP]})

    model = AutoModelForCausalLM.from_pretrained(
        profile["model_id"],
        revision=profile["model_revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    model.config.use_cache = True
    return model, tok


def format_check(output: str) -> dict:
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    action_lines = [line for line in lines if line.startswith("Action:")]
    action = action_lines[0][len("Action:"):].strip() if len(action_lines) == 1 else None
    return {
        "nonempty_lines": len(lines),
        "action_line_count": len(action_lines),
        "exact_two_line_format": (
            len(lines) == 2
            and lines[0].startswith("Thought:")
            and len(action_lines) == 1
            and lines[1] == action_lines[0]
        ),
        "legal_action_grammar": bool(action and LEGAL_ACTION_RE.fullmatch(action)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--alfworld-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)[: args.limit]
    model, tok = load_model_and_tokenizer(args.device)
    receiver = SimpleNamespace(base_model=model)

    results = []
    for index, row in enumerate(rows, start=1):
        result = rollout_episode(
            receiver,
            tok,
            os.path.join(args.alfworld_data, row["game_file"]),
            "no_comm_7b",
            None,
            args.device,
            max_steps=args.max_steps,
        )
        result.update({
            "episode_id": row["episode_id"],
            "source_split": row["source_split"],
            "task_type": row["task_type"],
            "game_file": row["game_file"],
            "format_checks": [format_check(output) for output in result["outputs"]],
        })
        results.append(result)
        print(
            f"[{index}/{len(rows)}] {row['episode_id']} success={result['success']} "
            f"steps={result['n_steps']} reason={result['reason']}",
            flush=True,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "no_comm_7b.jsonl"), results)
    checks = [check for result in results for check in result["format_checks"]]
    profile = get_model_profile(MODEL_PROFILE)
    write_json(os.path.join(args.out_dir, "summary.json"), {
        "model_id": profile["model_id"],
        "model_revision": profile["model_revision"],
        "device": args.device,
        "n_episodes": len(results),
        "max_steps": args.max_steps,
        "total_generations": len(checks),
        "single_action_rate": (
            sum(check["action_line_count"] == 1 for check in checks) / len(checks)
            if checks else 0.0
        ),
        "exact_two_line_rate": (
            sum(check["exact_two_line_format"] for check in checks) / len(checks)
            if checks else 0.0
        ),
        "legal_action_grammar_rate": (
            sum(check["legal_action_grammar"] for check in checks) / len(checks)
            if checks else 0.0
        ),
        "parse_error_count": sum(result["reason"] == "parse_error" for result in results),
        "success_count": sum(result["success"] for result in results),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
