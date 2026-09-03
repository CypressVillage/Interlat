"""Generate canonical expert trajectories for Step 1A training/eval episodes.

Protocol §6: each game must have exactly one canonical trajectory that passes
the terminal-success check, or the formal run must error out (never silently
drop). Trajectories are produced by the ALFWorld handcoded expert
(`AlfredExpert`, HANDCODED) driven per game file; `won` after the final action
is the terminal-success check.

Outputs (under --out-dir):
  episode_inputs.json        episode_id -> {task_description, initial_observation,
                             game_file, source_split}  (+ file SHA-256 in meta)
  trajectories/{role}/{episode_id}.json

Thought provenance: the handcoded expert yields actions only; Thought text is
a deterministic template of the action (see env_utils.thought_for_action).
valid_unseen is refused unless --allow-reserved-inputs (inputs only, no tasks).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from step1.common import episode_id, sha256_text, write_json
from step1.env_utils import (
    build_receiver_messages,
    make_tw_env,
    process_ob,
    split_task_and_initial_observation,
    strip_intro_text,
    thought_for_action,
    tw_state_fields,
)


def run_one_episode(game_file_abs: str, expert_max_steps: int = 200):
    """Run the handcoded expert on one game. Returns dict with actions/obs/won.

    AlfredExpert protocol: at reset the wrapper always yields plan ["look"]
    (it only observes the initial feedback); real expert commands start after
    the first step, once prev_command is set. We consume that initial "look"
    and then follow act()-derived commands. A later fallback to "look" means
    the expert produced a non-admissible command — treat as failure.
    """
    env = make_tw_env(game_file_abs)
    try:
        state = env.reset()
        fields = tw_state_fields(state)
        obs0 = strip_intro_text(fields["feedback"])
        task_description, initial_observation = split_task_and_initial_observation(obs0)

        actions: list = []
        observations: list = []  # post-action processed observations
        exception = None
        won = False
        done = False
        first = True
        for _ in range(expert_max_steps):
            plan = fields["expert_plan"]
            if not plan:
                exception = "expert_plan_empty"
                break
            action = plan[0]
            if action == "look" and not first:
                exception = "expert_fallback_look"
                break
            first = False
            actions.append(action)
            state, _reward, done = env.step(action)
            fields = tw_state_fields(state)
            observations.append(process_ob(fields["feedback"]))
            won = fields["won"]
            if done or won:
                break
        else:
            exception = "expert_max_steps_reached"
        return {
            "task_description": task_description,
            "initial_observation": initial_observation,
            "actions": actions,
            "observations": observations,
            "won": won,
            "exception": exception,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alfworld-data", required=True)
    parser.add_argument("--manifest", required=True, help="manifest .jsonl to process")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="process only first N rows (smoke)")
    parser.add_argument("--allow-reserved-inputs", action="store_true",
                        help="permit valid_unseen inputs (inputs only, still no task results)")
    args = parser.parse_args()

    from step1.common import read_jsonl

    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if rows and rows[0]["source_split"] == "valid_unseen" and not args.allow_reserved_inputs:
        print("[ERROR] valid_unseen is reserved; pass --allow-reserved-inputs explicitly", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    traj_root = os.path.join(args.out_dir, "trajectories")
    inputs_path = os.path.join(args.out_dir, "episode_inputs.json")
    episode_inputs = {}
    if os.path.exists(inputs_path):
        with open(inputs_path, "r", encoding="utf-8") as f:
            episode_inputs = json.load(f)

    failures = []
    n_ok = 0
    for i, row in enumerate(rows):
        eid = row["episode_id"]
        role_dir = os.path.join(traj_root, row["role"])
        os.makedirs(role_dir, exist_ok=True)
        out_path = os.path.join(role_dir, f"{eid}.json")
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("won"):
                n_ok += 1
                episode_inputs[eid] = prev["episode_inputs"]
                continue
        game_abs = os.path.join(args.alfworld_data, row["game_file"])
        try:
            result = run_one_episode(game_abs)
        except Exception as e:  # noqa: BLE001 — record all expert failures
            result = {
                "task_description": None,
                "initial_observation": None,
                "actions": [],
                "observations": [],
                "won": False,
                "exception": f"{type(e).__name__}: {e}",
            }
            traceback.print_exc()
        if not result["won"]:
            failures.append({"episode_id": eid, "game_file": row["game_file"],
                             "reason": result["exception"] or "terminal-success check failed"})
        record = {
            "episode_id": eid,
            "role": row["role"],
            "source_split": row["source_split"],
            "task_type": row["task_type"],
            "game_file": row["game_file"],
            "task_description": result["task_description"],
            "initial_observation": result["initial_observation"],
            "actions": result["actions"],
            "observations": result["observations"],
            "thoughts": [thought_for_action(a) for a in result["actions"]],
            "won": result["won"],
            "n_env_steps": len(result["actions"]),
            "exception": result["exception"],
            "messages": build_receiver_messages(
                result["task_description"], result["initial_observation"],
                result["actions"], result["observations"],
            ) if result["won"] else None,
        }
        record["content_sha256"] = sha256_text(json.dumps(
            {k: record[k] for k in ("episode_id", "game_file", "task_description",
                                    "initial_observation", "actions", "observations",
                                    "won")}, sort_keys=True))
        record["episode_inputs"] = {
            "task_description": result["task_description"],
            "initial_observation": result["initial_observation"],
            "game_file": row["game_file"],
            "source_split": row["source_split"],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1)
        episode_inputs[eid] = record["episode_inputs"]
        n_ok += 1 if result["won"] else 0
        print(f"[{i + 1}/{len(rows)}] {eid} won={result['won']} steps={len(result['actions'])}")

    write_json(inputs_path, episode_inputs)

    summary = {
        "manifest": args.manifest,
        "processed": len(rows),
        "won": n_ok,
        "failures": failures,
    }
    write_json(os.path.join(args.out_dir, "trajectory_summary.json"), summary)
    print(f"[SUMMARY] processed={len(rows)} won={n_ok} failures={len(failures)}")
    if failures:
        for f_ in failures[:20]:
            print(f"  FAIL {f_['episode_id']} {f_['game_file']}: {f_['reason']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
