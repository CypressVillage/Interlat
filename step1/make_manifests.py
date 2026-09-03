"""Generate the four frozen data manifests for Step 1A.

Usage:
  python -m step1.make_manifests --alfworld-data ~/.cache/alfworld/json_2.1.1 --out-dir manifests

Outputs (per protocol §3):
  train-fit.jsonl + .meta.json
  train-checkpoint-selection.jsonl + .meta.json
  valid_seen-step1-gate.jsonl + .meta.json
  valid_unseen-reserved.jsonl + .meta.json

Rules:
  - enumeration MUST replicate the ALFWorld evaluator's episode enumeration
    (protocol §3): AlfredTWEnv.collect_game_files semantics — skip paths
    containing 'movable' or 'Sliced', require traj_data.json with a known
    task_type, require game.tw-pddl to exist and carry "solvable": true.
    Raw disk globbing is not acceptable;
  - verified canonical counts (2026-09-02, A10): train 3553 / valid_seen 140 /
    valid_unseen 134. valid_seen/valid_unseen match the frozen protocol; the
    frozen train target 3321 is NOT reproducible (upstream hardcodes
    N_TASKS=3321 in eval/alfworld/eval_agent/tasks/alfworld.py:83 with an
    order-dependent trim). The train count is therefore a CLI choice,
    default canonical 3553, and any deviation from a frozen number is
    recorded in the meta artifact for the smoke report;
  - train split: 332 checkpoint-selection / rest train-fit, proportional
    stratification over the six task types, split seed 20260831,
    largest-remainder allocation, ties broken by UTF-8 lexicographic order
    of task type;
  - within each stratum episodes are ordered by game_file, then by
    SHA-256("20260831\0" + game_file);
  - episode ID = first 16 hex of SHA-256(source_split + "\0" + game_file);
  - manifests sorted by (source_split, task_type, game_file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from step1.common import (
    ALFWORLD_TASK_TYPE_NAMES,
    N_CHECKPOINT_SELECTION,
    N_TRAIN_CANONICAL,
    N_TRAIN_OFFICIAL,
    N_VALID_SEEN,
    N_VALID_UNSEEN,
    ROLE_BY_SPLIT,
    SPLIT_SEED,
    TASK_TYPES,
    check_no_collision,
    episode_id,
    sha256_file,
    sort_manifest_rows,
    task_type_from_game_file,
    write_json,
    write_jsonl,
)


def list_games(json_root: str, source_split: str) -> list:
    """Canonical ALFWorld enumeration (AlfredTWEnv.collect_game_files).

    - walk split dir; require traj_data.json in the directory
    - skip paths containing 'movable' or 'Sliced'
    - traj_data task_type must be one of the six canonical types
    - game.tw-pddl must exist, contain key 'solvable' with value true
    Returns game.tw-pddl paths relative to json_root (slash-separated).
    """
    split_dir = os.path.join(json_root, source_split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing ALFWorld split dir: {split_dir}")
    games = []
    skip_stats = {"movable_or_sliced": 0, "unknown_task_type": 0,
                  "missing_game": 0, "missing_solvable_key": 0, "unsolvable": 0}
    for dirpath, _dirnames, filenames in os.walk(split_dir, topdown=False):
        if "traj_data.json" not in filenames:
            continue
        rel_dir = os.path.relpath(dirpath, json_root).replace(os.sep, "/")
        if "movable" in rel_dir or "Sliced" in rel_dir:
            skip_stats["movable_or_sliced"] += 1
            continue
        game_rel = f"{rel_dir}/game.tw-pddl"
        game_abs = os.path.join(json_root, game_rel)
        if not os.path.exists(game_abs):
            skip_stats["missing_game"] += 1
            continue
        with open(os.path.join(dirpath, "traj_data.json"), "r", encoding="utf-8") as f:
            traj = json.load(f)
        task_type = traj.get("task_type")
        if task_type not in ALFWORLD_TASK_TYPE_NAMES:
            skip_stats["unknown_task_type"] += 1
            continue
        with open(game_abs, "r", encoding="utf-8") as f:
            gamedata = json.load(f)
        if "solvable" not in gamedata:
            skip_stats["missing_solvable_key"] += 1
            continue
        if not gamedata["solvable"]:
            skip_stats["unsolvable"] += 1
            continue
        games.append(game_rel)
    print(f"[ENUM] {source_split}: kept={len(games)} skipped={skip_stats}")
    games.sort()
    return games


def largest_remainder_quota(counts: dict, total: int) -> dict:
    """Proportional quotas per task type summing exactly to `total`.

    Remainder seats go to the largest fractional parts; ties broken by
    UTF-8 lexicographic order of task type (earlier type wins).
    """
    n_total = sum(counts.values())
    quota = {}
    fractional = []
    allocated = 0
    for t in sorted(counts):
        exact = total * counts[t] / n_total
        quota[t] = int(exact)
        allocated += quota[t]
        fractional.append((exact - quota[t], t))
    remaining = total - allocated
    # sort by (-fraction, type) so ties resolve by lexicographic type
    fractional.sort(key=lambda x: (-x[0], x[1]))
    for _i in range(remaining):
        frac, t = fractional[_i]
        quota[t] += 1
    if sum(quota.values()) != total:
        raise AssertionError("largest-remainder allocation failed")
    return quota


def build_train_split(games: list, out_rows_by_role: dict, meta: dict) -> None:
    counts = {t: 0 for t in TASK_TYPES}
    for g in games:
        counts[task_type_from_game_file(g)] += 1
    quota = largest_remainder_quota(counts, N_CHECKPOINT_SELECTION)

    strata = {t: [] for t in TASK_TYPES}
    for g in games:
        strata[task_type_from_game_file(g)].append(g)

    for t in TASK_TYPES:
        # order by game_file, then SHA-256(split_seed \0 game_file)
        ordered = sorted(strata[t], key=lambda g: (g, hashlib.sha256(f"{SPLIT_SEED}\0{g}".encode()).hexdigest()))
        take = quota[t]
        for i, g in enumerate(ordered):
            role = "train-checkpoint-selection" if i < take else "train-fit"
            out_rows_by_role[role].append({
                "episode_id": episode_id("train", g),
                "source_split": "train",
                "role": role,
                "task_type": t,
                "game_file": g,
                "game_file_sha256": None,  # filled by caller
            })
        meta["quota"][t] = take
        meta["count_by_type"][t] = counts[t]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alfworld-data", required=True, help="path to json_2.1.1 directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-count", type=int, default=N_TRAIN_CANONICAL,
                        help="expected canonical train episode count after evaluator "
                             "enumeration (default 3553). The protocol-frozen 3321 is "
                             "not reproducible from the data; deviations from frozen "
                             "numbers are recorded in the meta artifacts.")
    args = parser.parse_args()

    expected_counts = {
        "train": args.train_count,
        "valid_seen": N_VALID_SEEN,
        "valid_unseen": N_VALID_UNSEEN,
    }
    frozen_counts = {"train": N_TRAIN_OFFICIAL, "valid_seen": N_VALID_SEEN,
                     "valid_unseen": N_VALID_UNSEEN}

    all_rows = {role: [] for roles in ROLE_BY_SPLIT.values() for role in roles}
    meta = {
        "split_seed": SPLIT_SEED,
        "quota": {},
        "count_by_type": {},
        "splits": {},
        "frozen_protocol_counts": frozen_counts,
        "count_deviations_from_frozen": {
            k: {"frozen": v, "actual": expected_counts[k]}
            for k, v in expected_counts.items() if frozen_counts[k] != v
        },
    }

    for source_split, expected_n in expected_counts.items():
        games = list_games(args.alfworld_data, source_split)
        if len(games) != expected_n:
            print(f"[ERROR] {source_split}: found {len(games)} games, expected {expected_n}", file=sys.stderr)
            return 2
        if source_split == "train":
            build_train_split(games, all_rows, meta)
        else:
            for role in ROLE_BY_SPLIT[source_split]:
                for g in games:
                    all_rows[role].append({
                        "episode_id": episode_id(source_split, g),
                        "source_split": source_split,
                        "role": role,
                        "task_type": task_type_from_game_file(g),
                        "game_file": g,
                        "game_file_sha256": None,
                    })
        meta["splits"][source_split] = {"games": len(games), "expected": expected_n}

    written = []
    for role, rows in all_rows.items():
        for r in rows:
            r["game_file_sha256"] = sha256_file(os.path.join(args.alfworld_data, r["game_file"]))
        rows = sort_manifest_rows(rows)
        check_no_collision(rows)
        path = os.path.join(args.out_dir, f"{role}.jsonl")
        sha = write_jsonl(path, rows)
        role_meta = {
            "role": role,
            "rows": len(rows),
            "sha256": sha,
            "count_by_task_type": {
                t: sum(1 for r in rows if r["task_type"] == t) for t in TASK_TYPES
            },
            "count_deviations_from_frozen": meta["count_deviations_from_frozen"],
        }
        if role.startswith("train"):
            role_meta["quota"] = meta["quota"]
            role_meta["train_count_by_type"] = meta["count_by_type"]
        mpath = os.path.join(args.out_dir, f"{role}.meta.json")
        write_json(mpath, role_meta)
        written.append((path, len(rows), sha))
        print(f"[OK] {path}: {len(rows)} rows, sha256={sha}")

    total_train = sum(n for p, n, _ in written if "train" in p)
    if total_train != args.train_count:
        print(f"[ERROR] train rows total {total_train} != {args.train_count}", file=sys.stderr)
        return 2
    if args.train_count != N_TRAIN_OFFICIAL:
        print(f"[NOTE] train count {args.train_count} deviates from frozen protocol "
              f"{N_TRAIN_OFFICIAL}; recorded in meta artifacts (advisor confirmation "
              f"required before formal runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
