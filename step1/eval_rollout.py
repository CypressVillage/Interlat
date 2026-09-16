r"""Step 1A evaluator rollouts (protocol §4, §7).

Drives the frozen Receiver (with trained adapter) in real ALFWorld TextWorld
games with a 20-step cap and greedy decoding, mirroring the eval-side agent:
incremental embedding cache, injection once after the first user turn, ChatML
rendering via apply_chat_template, greedy max_new_tokens=100, stop at
<|im_end|>, action extraction with the same upstream regex
(`Action:\s?(.*)`, DOTALL). Parse failure terminates the episode as a failure
with reason "parse_error" (recorded, never silently retried).

Groups: no_comm, matched, raw, zero, mismatched (per draw), random (per draw).
Zero/random/mismatched norms are anchored to ||Z_i||_F of the CURRENT adapter
(matched output). Mismatched mapping is built per task type and draw with the
preregistered cyclic-shift rule and saved with its SHA-256.

Outputs (under --out-dir):
  results/{group}.jsonl                      one record per episode(+draw)
  mismatched_mapping_draw{d}.json            full mapping artifact
  eval_summary.json                          counts per group
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

from step1.adapter import load_frozen_receiver
from step1.common import (
    EVAL_MAX_STEPS,
    MISMATCH_DRAWS,
    RANDOM_DRAWS,
    RECEIVER_MAX_NEW_TOKENS,
    load_manifest,
    read_jsonl,
    write_json,
)
from step1.controls import (
    build_mismatched_mapping,
    make_matched_message,
    make_mismatched_message,
    make_random_message,
    make_zero_message,
    matched_norm_target,
)
from step1.env_utils import make_plain_tw_env, parse_action, process_ob, split_task_and_initial_observation, strip_intro_text
from step1.serialization import (
    bop_eop_ids,
    load_locked_tokenizer,
    receiver_initial_messages,
    render_with_labels,
)

UPSTREAM_ACTION_RE = re.compile(r"Action:\s?(.*)", re.DOTALL)


@torch.no_grad()
def greedy_generate(receiver, cache_state, max_new_tokens: int, im_end_id: int, device: str):
    """Manual greedy loop over the KV cache. Returns (token_ids, hit_terminator).

    cache_state must carry prefill's last-position logits ("logits"); the loop
    then feeds one token embedding per step.
    """
    model = receiver.base_model
    logits = cache_state["logits"]
    past = cache_state["past"]
    ids, hit = [], False
    for _ in range(max_new_tokens):
        nxt = int(torch.argmax(logits, dim=-1).item())
        if nxt == im_end_id:
            hit = True
            break
        ids.append(nxt)
        emb = model.get_input_embeddings()(torch.tensor([[nxt]], device=device))
        out = model(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    cache_state["past"] = past
    cache_state["logits"] = logits
    return ids, hit


@torch.no_grad()
def prefill(receiver, prefix_embeds, device: str):
    out = receiver.base_model(inputs_embeds=prefix_embeds, use_cache=True)
    return {"past": out.past_key_values, "logits": out.logits[:, -1, :]}


@torch.no_grad()
def append_tokens(receiver, cache_state, token_ids, device: str):
    if not token_ids:
        return
    emb = receiver.base_model.get_input_embeddings()(
        torch.tensor([token_ids], device=device))
    out = receiver.base_model(inputs_embeds=emb, past_key_values=cache_state["past"],
                              use_cache=True)
    cache_state["past"] = out.past_key_values
    cache_state["logits"] = out.logits[:, -1, :]


def rollout_episode(receiver, tok, game_file_abs: str, group: str, latent: torch.Tensor | None,
                    device: str, max_steps: int = EVAL_MAX_STEPS) -> dict:
    bop_id, eop_id = bop_eop_ids(tok)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    env = make_plain_tw_env(game_file_abs)
    rec = {"group": group, "success": False, "n_steps": 0, "reason": None,
           "actions": [], "hit_terminator": []}
    try:
        state = env.reset()
        obs = strip_intro_text(state["feedback"])
        task_description, initial_observation = split_task_and_initial_observation(obs)
        rec["task_description"] = task_description
        rec["initial_observation"] = initial_observation

        user1 = receiver_initial_messages(task_description, initial_observation)
        ids1 = tok.apply_chat_template(user1, tokenize=True, add_generation_prompt=True)
        r = render_with_labels(tok, user1)
        if ids1[:r.injection_index] != r.input_ids[:r.injection_index]:
            raise AssertionError("prefix instability between render and rollout")

        emb_layer = receiver.base_model.get_input_embeddings()
        emb_dtype = emb_layer.weight.dtype
        ids_t = torch.tensor(ids1, device=device)
        if latent is None:
            prefix = emb_layer(ids_t)
        else:
            inj = r.injection_index
            bop_emb = emb_layer.weight[bop_id]
            eop_emb = emb_layer.weight[eop_id]
            prefix = torch.cat([
                emb_layer(ids_t[:inj]),
                bop_emb.unsqueeze(0),
                latent.to(device=device, dtype=emb_dtype),
                eop_emb.unsqueeze(0),
                emb_layer(ids_t[inj:]),
            ], dim=0)
        cache = prefill(receiver, prefix.unsqueeze(0), device)

        messages = list(user1)
        prev_len = len(ids1)
        won = False
        done = False
        for _ in range(max_steps):
            gen_ids, hit = greedy_generate(receiver, cache, RECEIVER_MAX_NEW_TOKENS, im_end_id, device)
            rec["hit_terminator"].append(hit)
            text = tok.decode(gen_ids)
            m = UPSTREAM_ACTION_RE.findall(text)
            action = parse_action(text) if m else None
            if action is None or action.strip() == "":
                rec["reason"] = "parse_error"
                rec["raw_output"] = text
                break
            rec["actions"].append(action)
            messages.append({"role": "assistant", "content": text})

            state, _reward, done = env.step(action)
            obs = process_ob(state["feedback"])
            won = bool(state["won"])
            rec["n_steps"] += 1
            if won or done:
                break
            messages.append({"role": "user", "content": f"Observation: {obs}"})

            ids_k = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            # greedy loop already fed ids1 + gen_ids embeddings into the cache
            # (the im_end terminator breaks before being fed), so continue from there
            prev_len = len(ids1) + len(gen_ids)
            expected = list(ids1) + list(gen_ids)
            if prev_len != len(expected) or ids_k[:prev_len] != expected:
                raise AssertionError("chat template prefix drifted during rollout")
            append_tokens(receiver, cache, ids_k[prev_len:], device)
            ids1 = ids_k
        else:
            rec["reason"] = "max_steps"
        if won:
            rec["success"] = True
            rec["reason"] = None
        elif rec["reason"] is None:
            rec["reason"] = "done_not_won" if done else "max_steps"
        return rec
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode-inputs", required=True)
    ap.add_argument("--gate-manifest", required=True)
    ap.add_argument("--latents-dir", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--groups", default="matched,no_comm")
    ap.add_argument("--max-steps", type=int, default=EVAL_MAX_STEPS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alfworld-data", required=True,
                    help="ALFWorld data root; game_file paths are joined against it")
    ap.add_argument("--mismatch-draws", default=",".join(map(str, MISMATCH_DRAWS)))
    ap.add_argument("--random-draws", default=",".join(map(str, RANDOM_DRAWS)))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    mismatch_draws = [int(x) for x in args.mismatch_draws.split(",") if x]
    random_draws = [int(x) for x in args.random_draws.split(",") if x]

    with open(args.episode_inputs, "r", encoding="utf-8") as f:
        episode_inputs = json.load(f)
    rows = load_manifest(args.gate_manifest)
    if args.limit:
        rows = rows[: args.limit]

    tok = load_locked_tokenizer()
    receiver, freeze_report = load_frozen_receiver(tok, device=args.device)
    ckpt = torch.load(args.adapter_path, map_location=args.device)
    receiver.adapter.load_eval_export(ckpt["eval_export"])
    receiver.eval()
    print(f"[LOAD] adapter from {args.adapter_path} (epoch={ckpt.get('epoch')}, "
          f"sel_nll={ckpt.get('sel_nll')})")

    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    device = args.device

    def latent_for(eid: str) -> torch.Tensor:
        row = next(r for r in rows if r["episode_id"] == eid)
        return torch.load(os.path.join(args.latents_dir, row["role"], f"{eid}.pt"),
                          map_location="cpu")

    for group in groups:
        out_path = os.path.join(args.out_dir, "results", f"{group}.jsonl")
        done_ids = {r["episode_id"] for r in read_jsonl(out_path)} if os.path.exists(out_path) else set()

        if group == "mismatched":
            by_type = {}
            for r in rows:
                if r["episode_id"] in done_ids:
                    continue
                by_type.setdefault(r["task_type"], []).append(
                    {**r, "L_i": int(latent_for(r["episode_id"]).shape[0])})
        else:
            by_type = None

        for draw in (mismatch_draws if group == "mismatched" else
                     random_draws if group == "random" else [None]):
            if group == "mismatched":
                mapping = build_mismatched_mapping(by_type, draw)
                write_json(os.path.join(args.out_dir, f"mismatched_mapping_draw{draw}.json"), mapping)
            n_ok = 0
            for i, row in enumerate(rows):
                eid = row["episode_id"]
                if eid in done_ids:
                    continue
                info = episode_inputs[eid]
                game_abs = os.path.join(args.alfworld_data, info["game_file"])
                H = latent_for(eid) if group != "no_comm" else None
                norm_target = matched_norm_target(receiver.adapter, H, torch.device(device)) if H is not None else 0.0
                if group == "matched":
                    Z, _ = make_matched_message(receiver.adapter, H, torch.bfloat16, torch.device(device))
                elif group == "raw":
                    Z = H.to(torch.bfloat16)
                elif group == "zero":
                    Z = make_zero_message(int(H.shape[0]), torch.bfloat16, torch.device(device))
                elif group == "mismatched":
                    p = mapping["pairs"][eid]
                    donor_path = os.path.join(args.latents_dir, row["role"], f"{p['donor_episode_id']}.pt")
                    H_donor = torch.load(donor_path, map_location="cpu")
                    Z, _cinfo = make_mismatched_message(
                        receiver.adapter, H_donor, int(H.shape[0]), norm_target,
                        torch.bfloat16, torch.device(device))
                elif group == "random":
                    Z, _rinfo = make_random_message(int(H.shape[0]), draw, eid,
                                                    norm_target, torch.bfloat16, torch.device(device))
                else:  # no_comm
                    Z = None
                rec = rollout_episode(receiver, tok, game_abs, group, Z, device,
                                      max_steps=args.max_steps)
                rec.update({"episode_id": eid, "game_file": row["game_file"],
                            "task_type": row["task_type"],
                            "draw": draw, "source_split": row["source_split"]})
                if group == "mismatched":
                    rec["donor_episode_id"] = p["donor_episode_id"]
                    rec["shift_k"] = p["shift_k"]
                n_ok += 1 if rec["success"] else 0
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                print(f"[{i + 1}/{len(rows)}] {group} draw={draw} {eid} "
                      f"success={rec['success']} steps={rec['n_steps']} reason={rec['reason']}")
            print(f"[GROUP] {group} draw={draw}: {n_ok} successes")

    write_json(os.path.join(args.out_dir, "eval_summary.json"), {
        "groups": groups, "adapter": args.adapter_path, "max_steps": args.max_steps,
        "n_episodes": len(rows),
        "freeze_report": freeze_report,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
