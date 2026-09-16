"""Step 1A matched-group training (protocol §6, §8).

Trains ONLY the Step1Adapter on Matched Latent messages ([<bop>;A(H_i);<eop>]
after the first user turn) with a single CE forward per batch. Base model,
Sender, token embeddings (incl. tied LM head) and all normalizations stay
frozen; the freeze report is asserted at startup and logged.

Optimizer: AdamW lr 5e-5, weight decay 0.01, batch size 2, grad accum 8,
max 10 epochs, Adam eps default, no grad clipping (not preregistered).
Shuffling uses a torch.Generator seeded with the training seed.
Model selection: response-token NLL on the 332-episode checkpoint-selection
split after each epoch; early stop with patience 2. Best adapter is exported
in the eval-side compatible format (hidden_mha_state.pt).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict

import torch

from step1.adapter import ReceiverWithLatent, Step1Adapter, load_frozen_receiver
from step1.common import (
    TRAINING_SEEDS,
    read_jsonl,
    write_json,
)
from step1.serialization import (
    load_locked_tokenizer,
    receiver_assistant_content,
    receiver_initial_messages,
    receiver_step_user_content,
    render_with_labels,
)


def build_receiver_messages(info: Dict, traj: Dict) -> list:
    """Full multi-turn receiver conversation for one expert trajectory.

    system + user1 (instruction + task + initial observation) -> assistant (Thought+Action) ->
    user (Observation) -> assistant -> ... No trailing observation after the
    final action. This is the training sample; CE covers all assistant turns.
    """
    actions = traj["actions"]
    thoughts = traj["thoughts"]
    observations = traj["observations"]
    if not traj.get("won"):
        raise ValueError(f"{traj.get('episode_id')}: trajectory not won; refusing to train on it")
    if not (len(actions) == len(thoughts) == len(observations)):
        raise ValueError(
            f"{traj.get('episode_id')}: trajectory length mismatch "
            f"a={len(actions)} t={len(thoughts)} o={len(observations)}"
        )
    msgs = receiver_initial_messages(info["task_description"], info["initial_observation"])
    n = len(actions)
    for i in range(n):
        msgs.append({"role": "assistant",
                     "content": receiver_assistant_content(thoughts[i], actions[i])})
        if i < n - 1:
            msgs.append({"role": "user",
                         "content": receiver_step_user_content(observations[i])})
    return msgs


def load_items(episode_inputs_path: str, manifest_path: str, latents_dir: str,
               trajectories_dir: str) -> list:
    with open(episode_inputs_path, "r", encoding="utf-8") as f:
        episode_inputs = json.load(f)
    tok = load_locked_tokenizer()
    items = []
    for row in read_jsonl(manifest_path):
        eid = row["episode_id"]
        info = episode_inputs[eid]
        with open(os.path.join(trajectories_dir, row["role"], f"{eid}.json"),
                  "r", encoding="utf-8") as f:
            traj = json.load(f)
        messages = build_receiver_messages(info, traj)
        rendered = render_with_labels(tok, messages)
        if not any(l != -100 for l in rendered.labels):
            raise ValueError(f"{eid}: no supervised tokens in rendered conversation")
        H = torch.load(os.path.join(latents_dir, row["role"], f"{eid}.pt"), map_location="cpu")
        if H.dtype != torch.float32 or H.shape[1] != 896:
            raise ValueError(f"{eid}: latent dtype/shape invalid: {H.dtype} {tuple(H.shape)}")
        items.append({
            "episode_id": eid,
            "rendered": rendered,
            "Z": H.contiguous(),
            "task_type": row["task_type"],
        })
    return items


def batches(items: list, bs: int, gen: torch.Generator):
    idx = torch.randperm(len(items), generator=gen).tolist()
    for s in range(0, len(idx), bs):
        yield [items[i] for i in idx[s:s + bs]]


def ordered_items(items: list):
    return sorted(items, key=lambda x: x["episode_id"])


@torch.no_grad()
def eval_nll(receiver, items: list, bs: int, no_comm: bool = False) -> dict:
    receiver.eval()
    total_nll, total_tok = 0.0, 0
    for s in range(0, len(items), bs):
        chunk = items[s:s + bs]
        if no_comm:
            b = receiver.build_no_injection_batch(chunk)
        else:
            b = receiver.build_training_batch(chunk)
        r = receiver.supervised_nll(b)
        total_nll += r["nll"] * r["n_tokens"]
        total_tok += r["n_tokens"]
    receiver.train()
    return {"nll": total_nll / max(total_tok, 1), "n_tokens": total_tok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode-inputs", required=True)
    ap.add_argument("--trajectories-dir", required=True)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--selection-manifest", required=True)
    ap.add_argument("--latents-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.seed not in TRAINING_SEEDS:
        print(f"[WARN] seed {args.seed} not in official TRAINING_SEEDS {TRAINING_SEEDS}")

    t0 = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tok = load_locked_tokenizer()
    receiver, freeze_report = load_frozen_receiver(tok, device=args.device)
    assert isinstance(receiver, ReceiverWithLatent)

    train_items = load_items(args.episode_inputs, args.train_manifest, args.latents_dir,
                             args.trajectories_dir)
    sel_items = load_items(args.episode_inputs, args.selection_manifest, args.latents_dir,
                           args.trajectories_dir)
    print(f"[DATA] train={len(train_items)} selection={len(sel_items)}")

    trainable = [p for p in receiver.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "training_log.jsonl")

    def log(rec: dict):
        rec = {"elapsed_s": round(time.time() - t0, 1), **rec}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        print("[LOG]", json.dumps(rec, ensure_ascii=False))

    log({
        "event": "init", "seed": args.seed, "epochs": args.epochs, "bs": args.bs,
        "accum": args.accum, "lr": args.lr, "wd": args.wd, "patience": args.patience,
        "freeze_report": freeze_report,
        "adapter_init": {
            "scale": float(receiver.adapter.adaptive_proj.scale.detach()),
            "output_scale": float(receiver.adapter.adaptive_proj.output_scale.detach()),
            "param_count": receiver.adapter.param_count(),
        },
    })

    gen = torch.Generator().manual_seed(args.seed)
    best = {"nll": float("inf"), "epoch": -1}
    no_improve = 0
    receiver.train()
    for epoch in range(args.epochs):
        ep_loss, nb, opt_steps = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for bi, chunk in enumerate(batches(train_items, args.bs, gen)):
            b = receiver.build_training_batch(chunk)
            loss, _logits = receiver.forward(b["input_embeds"], b["attention_mask"], b["labels"])
            (loss / args.accum).backward()
            ep_loss += float(loss.detach())
            nb += 1
            if (bi + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_steps += 1
        if nb % args.accum != 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_steps += 1

        sel_matched = eval_nll(receiver, ordered_items(sel_items), args.bs, no_comm=False)
        sel_nocomm = eval_nll(receiver, ordered_items(sel_items), args.bs, no_comm=True)
        rec = {
            "event": "epoch_end", "epoch": epoch, "train_loss_mean": ep_loss / max(nb, 1),
            "opt_steps": opt_steps,
            "sel_nll_matched": sel_matched["nll"], "sel_nll_no_comm": sel_nocomm["nll"],
            "nll_drop_pct": 100.0 * (sel_nocomm["nll"] - sel_matched["nll"]) / sel_nocomm["nll"],
        }
        log(rec)

        if sel_matched["nll"] < best["nll"] - 1e-6:
            best = {"nll": sel_matched["nll"], "epoch": epoch}
            no_improve = 0
            torch.save({
                "adapter_state": receiver.adapter.state_dict(),
                "eval_export": receiver.adapter.state_for_eval_export(),
                "epoch": epoch, "sel_nll": sel_matched["nll"], "seed": args.seed,
            }, os.path.join(args.out_dir, "adapter_best.pt"))
        else:
            no_improve += 1
            if no_improve > args.patience:
                log({"event": "early_stop", "epoch": epoch, "best": best})
                break

    log({"event": "done", "best": best})
    write_json(os.path.join(args.out_dir, "train_summary.json"), {
        "seed": args.seed, "best": best,
        "config": {k: getattr(args, k) for k in vars(args)},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
