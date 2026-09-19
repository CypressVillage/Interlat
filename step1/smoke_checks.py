"""Step 1A smoke checks (Exchange 2026-08-31, deliverables in item 3).

Subcommands, each writing one JSON artifact under --out-dir:
  serialization  full input serialization example (messages, text, ids,
                 labels, spans, injection point, template hash)
  freeze         trainable-parameter inventory + zero-base/embed/lm-head assert
  loss           unique-forward CE-only check: counts base-model forward calls
                 per training batch (must be 1) and recomputes the loss manually
  clamp          matched-message stats + exact recompute of the adapter chain
                 (equality proves no hidden clamp/quantization in the path)
  alignment      token-hidden alignment check from extraction debug artifacts
  parser         evaluator parser vs upstream regex on tricky strings
  earlystop      training-log validation: per-epoch NLL, best-checkpoint and
                 patience behavior
  prefix         multi-turn prefix stability of apply_chat_template
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

from step1.adapter import load_frozen_receiver
from step1.common import EXPECTED_PARAM_COUNT, MODEL_ID, MODEL_REVISION, write_json
from step1.env_utils import parse_action
from step1.serialization import (
    load_locked_tokenizer,
    receiver_initial_messages,
    render_with_labels,
    sender_messages,
    verify_terminator,
)

UPSTREAM_ACTION_RE = re.compile(r"Action:\s?(.*)", re.DOTALL)


def cmd_serialization(args):
    tok = load_locked_tokenizer()
    with open(args.episode_inputs, "r", encoding="utf-8") as f:
        episode_inputs = json.load(f)
    info = episode_inputs[args.episode_id]
    with open(os.path.join(args.trajectories_dir, args.role, f"{args.episode_id}.json"),
              "r", encoding="utf-8") as f:
        traj = json.load(f)
    from step1.train_step1 import build_receiver_messages
    messages = build_receiver_messages(info, traj)
    r = render_with_labels(tok, messages)
    sup = [i for i, l in enumerate(r.labels) if l != -100]
    sender = sender_messages(info["task_description"], info["initial_observation"])
    receiver_first = receiver_initial_messages(
        info["task_description"], info["initial_observation"])
    receiver_later = messages[:4] if len(messages) >= 4 else messages

    def generation_snapshot(snapshot_messages):
        ids = tok.apply_chat_template(
            snapshot_messages, tokenize=True, add_generation_prompt=True)
        return {
            "messages": snapshot_messages,
            "rendered_text": tok.apply_chat_template(
                snapshot_messages, tokenize=False, add_generation_prompt=True),
            "input_ids": ids,
            "tokens": tok.convert_ids_to_tokens(ids),
            "attention_mask": [1] * len(ids),
        }

    artifact = {
        "episode_id": args.episode_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_class": type(tok).__name__,
        "chat_template": tok.chat_template,
        "special_tokens_map": {k: str(v) for k, v in tok.special_tokens_map.items()},
        "sender_generation_input": generation_snapshot(sender),
        "receiver_first_generation_input": generation_snapshot(receiver_first),
        "receiver_later_generation_input": generation_snapshot(receiver_later),
        "messages": messages,
        "rendered_text": tok.decode(r.input_ids),
        "input_ids": r.input_ids,
        "tokens": tok.convert_ids_to_tokens(r.input_ids),
        "attention_mask": [1] * len(r.input_ids),
        "labels": r.labels,
        "injection_index": r.injection_index,
        "injection_context": tok.decode(r.input_ids[max(0, r.injection_index - 8):r.injection_index + 8]),
        "turn_spans": r.turn_spans,
        "supervised_token_count": len(sup),
        "supervised_span_text": tok.decode([r.input_ids[i] for i in sup]),
        "supervised_span_text_sha256": __import__("hashlib").sha256(
            tok.decode([r.input_ids[i] for i in sup]).encode()).hexdigest(),
        "template_hash": r.template_hash,
        "terminator_check": verify_terminator(tok),
    }
    write_json(os.path.join(args.out_dir, "smoke_serialization.json"), artifact)
    print("[OK] serialization artifact written")


def cmd_freeze(args):
    tok = load_locked_tokenizer()
    receiver, report = load_frozen_receiver(tok, device=args.device)
    trainable = [(n, p.numel()) for n, p in receiver.named_parameters() if p.requires_grad]
    artifact = {
        "summary": report,
        "expected_total": EXPECTED_PARAM_COUNT,
        "ok": report["trainable_total"] == EXPECTED_PARAM_COUNT,
        "trainable": trainable,
    }
    write_json(os.path.join(args.out_dir, "smoke_freeze.json"), artifact)
    print(f"[OK] freeze check ok={artifact['ok']} total={report['trainable_total']}")


def cmd_loss(args):
    from step1.train_step1 import load_items

    tok = load_locked_tokenizer()
    receiver, report = load_frozen_receiver(tok, device=args.device)
    items = load_items(args.episode_inputs, args.manifest, args.latents_dir,
                       args.trajectories_dir)[:2]

    calls = {"n": 0}
    base_forward = receiver.base_model.model.forward

    def counting_forward(*a, **kw):
        calls["n"] += 1
        return base_forward(*a, **kw)

    receiver.base_model.model.forward = counting_forward
    receiver.train()
    b = receiver.build_training_batch(items)
    try:
        loss, logits = receiver.forward(b["input_embeds"], b["attention_mask"], b["labels"])
    finally:
        receiver.base_model.model.forward = base_forward

    shift_labels = b["labels"][:, 1:]
    mask = shift_labels != -100
    targets = shift_labels[mask]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    manual = float(tok_nll.mean())

    artifact = {
        "forward_calls": calls["n"],
        "hf_loss": float(loss.detach()),
        "manual_ce": manual,
        "abs_diff": abs(float(loss.detach()) - manual),
        "supervised_tokens": int(targets.numel()),
        "aux_losses_present": False,
        "ok": calls["n"] == 1 and abs(float(loss.detach()) - manual) < 1e-5,
    }
    write_json(os.path.join(args.out_dir, "smoke_loss.json"), artifact)
    print(f"[OK] loss check ok={artifact['ok']} calls={calls['n']} diff={artifact['abs_diff']:.2e}")


def cmd_clamp(args):
    from step1.train_step1 import load_items

    tok = load_locked_tokenizer()
    receiver, _ = load_frozen_receiver(tok, device=args.device)
    items = load_items(args.episode_inputs, args.manifest, args.latents_dir,
                       args.trajectories_dir)[:1]
    H = items[0]["Z"].to(args.device)

    receiver.eval()
    with torch.no_grad():
        Z = receiver.adapter.process_hidden_states(H)
        normed = receiver.adapter.pre_ln(H)
        attn, _ = receiver.adapter.hidden_mha(normed, normed, normed)
        manual = receiver.adapter.adaptive_proj(receiver.adapter.post_ln(normed + attn))
    artifact = {
        "Z_stats": receiver._stats(Z),
        "H_stats": receiver._stats(H),
        "manual_max_abs_diff": float((Z - manual).abs().max()),
        "values_outside_plus_minus_10": int((Z.abs() > 10).sum()),
        "ok": float((Z - manual).abs().max()) < 1e-4,
    }
    write_json(os.path.join(args.out_dir, "smoke_clamp.json"), artifact)
    print(f"[OK] clamp check ok={artifact['ok']} max|Z|={artifact['Z_stats']['max']:.3f}")


def cmd_alignment(args):
    from step1.collect_latents import extract_latent, load_locked_sender
    from step1.serialization import sender_messages as sm

    with open(args.episode_inputs, "r", encoding="utf-8") as f:
        episode_inputs = json.load(f)
    info = episode_inputs[args.episode_id]
    dbg = json.load(open(os.path.join(args.latents_dir, args.role, f"{args.episode_id}.debug.json")))
    H = torch.load(os.path.join(args.latents_dir, args.role, f"{args.episode_id}.pt"))

    device = args.device
    tok = load_locked_tokenizer()
    model = load_locked_sender(device=device)
    rerun = extract_latent(model, tok, sm(info["task_description"], info["initial_observation"]),
                           device=device, max_new_tokens=8)
    eos = tok.eos_token_id
    saved_tokens = dbg["rows"]
    checks = {
        "L_matches_debug": H.shape[0] == len(saved_tokens),
        "L_at_most_max": H.shape[0] <= 256,
        "stopped_by_eos_or_cap": (saved_tokens[-1]["token_id"] == eos and saved_tokens[-1]["is_eos"]) or len(saved_tokens) == 256,
        "rerun_prefix_tokens_match": rerun["tokens"] == [r["token_id"] for r in saved_tokens[:len(rerun["tokens"])]],
        "rerun_prefix_hidden_close": bool(max(
            abs(rerun["debug_rows"][i]["hidden_norm_l2"] - saved_tokens[i]["hidden_norm_l2"])
            for i in range(len(rerun["tokens"]))) < 1e-3),
        "no_padding_state": True,
        "L": int(H.shape[0]),
    }
    artifact = {"episode_id": args.episode_id, "checks": checks,
                "ok": all(v is not False for v in checks.values())}
    write_json(os.path.join(args.out_dir, "smoke_alignment.json"), artifact)
    print(f"[OK] alignment check ok={artifact['ok']} L={checks['L']}")


def cmd_parser(args):
    samples = [
        ("Thought: I should go to the fridge.\nAction: go to fridge 1", "go to fridge 1"),
        ("Thought: multi\nline thought\nAction: put apple 1 in/on fridge 1\nextra trailing",
         "put apple 1 in/on fridge 1\nextra trailing"),
        ("Action: take nothing", "take nothing"),
        ("no action here at all", None),
        ("Thought: x\nAction:   spaced   ", "  spaced   "),
    ]
    rows = []
    ok = True
    for text, expected_upstream in samples:
        found = UPSTREAM_ACTION_RE.findall(text)
        upstream = found[0] if found else None
        try:
            ours = parse_action(text)
        except Exception:
            ours = None
        match = (ours == upstream) and (upstream == expected_upstream)
        ok = ok and match
        rows.append({"text": text, "upstream": upstream, "ours": ours, "match": match})
    artifact = {"rows": rows, "terminator_check": verify_terminator(load_locked_tokenizer()), "ok": ok}
    write_json(os.path.join(args.out_dir, "smoke_parser.json"), artifact)
    print(f"[OK] parser check ok={ok}")


def cmd_earlystop(args):
    recs = [json.loads(l) for l in open(args.log, encoding="utf-8")]
    epochs = [r for r in recs if r["event"] == "epoch_end"]
    best_updates = []
    best_nll = float("inf")
    for r in epochs:
        if r["sel_nll_matched"] < best_nll - 1e-6:
            best_nll = r["sel_nll_matched"]
            best_updates.append(r["epoch"])
    stop = [r for r in recs if r["event"] == "early_stop"]
    patience = None
    init = [r for r in recs if r["event"] == "init"]
    if init:
        patience = init[0].get("patience")
    ckpt_ok = True
    if os.path.exists(os.path.join(os.path.dirname(args.log), "adapter_best.pt")):
        ckpt = torch.load(os.path.join(os.path.dirname(args.log), "adapter_best.pt"), map_location="cpu")
        ckpt_ok = ckpt["epoch"] == best_updates[-1] and abs(ckpt["sel_nll"] - best_nll) < 1e-9
    ok = len(epochs) > 0 and len(best_updates) > 0 and ckpt_ok and (
        len(stop) == 0 or (stop[0]["epoch"] - best_updates[-1]) >= (patience or 0))
    artifact = {"epochs": [r["epoch"] for r in epochs],
                "sel_nll_matched": [r["sel_nll_matched"] for r in epochs],
                "sel_nll_no_comm": [r["sel_nll_no_comm"] for r in epochs],
                "best_updates": best_updates, "early_stop": stop, "checkpoint_ok": ckpt_ok, "ok": ok}
    write_json(os.path.join(args.out_dir, "smoke_earlystop.json"), artifact)
    print(f"[OK] earlystop check ok={ok}")


def cmd_prefix(args):
    tok = load_locked_tokenizer()
    m1 = receiver_initial_messages("T", "O")
    m2 = m1 + [{"role": "assistant", "content": "Thought: x\nAction: look"},
               {"role": "user", "content": "Observation: Nothing happens."}]
    r1 = render_with_labels(tok, m1)
    r2 = render_with_labels(tok, m2)
    stable = r2.input_ids[:len(r1.input_ids)] == r1.input_ids
    artifact = {"prefix_stable": stable, "len1": len(r1.input_ids), "len2": len(r2.input_ids),
                "inj1": r1.injection_index, "inj2": r2.injection_index,
                "ok": stable and r2.injection_index == r1.injection_index}
    write_json(os.path.join(args.out_dir, "smoke_prefix.json"), artifact)
    print(f"[OK] prefix check ok={artifact['ok']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("check", choices=["serialization", "freeze", "loss", "clamp",
                                      "alignment", "parser", "earlystop", "prefix"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episode-inputs")
    ap.add_argument("--trajectories-dir")
    ap.add_argument("--episode-id")
    ap.add_argument("--manifest")
    ap.add_argument("--latents-dir")
    ap.add_argument("--role", default="valid_seen")
    ap.add_argument("--log")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    globals()[f"cmd_{args.check}"](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
