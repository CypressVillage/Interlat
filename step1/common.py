"""Step 1A shared constants, hashing, and manifest utilities.

Protocol: notes/interlat-step1-experiment-protocol.md (collab repo).

Seeds (frozen):
  - split seed:            20260831
  - training seeds:        0, 1, 2 (Yellow extension: 3, 4)
  - mismatched draws:      1101..1105
  - random-message draws:  2101..2105
  - bootstrap/power seed:  20260820
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

# A10 has no direct huggingface.co access; all weights are pre-cached. Fail
# fast instead of hanging on hub retries (must run before transformers import).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------------------------------------------------------------- frozen seeds
SPLIT_SEED = "20260831"
TRAINING_SEEDS = [0, 1, 2]
TRAINING_SEEDS_YELLOW_EXTENSION = [3, 4]
MISMATCH_DRAWS = [1101, 1102, 1103, 1104, 1105]
RANDOM_DRAWS = [2101, 2102, 2103, 2104, 2105]
BOOTSTRAP_SEED = "20260820"

# ---------------------------------------------------------------- frozen scale
N_TRAIN_OFFICIAL = 3321
# Canonical evaluator enumeration (AlfredTWEnv semantics: skip movable/Sliced,
# require solvable) verified on 2026-09-02: train 3553 / valid_seen 140 /
# valid_unseen 134. The frozen 3321 is an upstream hardcoded trim
# (eval/alfworld/eval_agent/tasks/alfworld.py:83) not reproducible from data.
N_TRAIN_CANONICAL = 3553
N_VALID_SEEN = 140
N_VALID_UNSEEN = 134
N_CHECKPOINT_SELECTION = 332
EVAL_MAX_STEPS = 20
SENDER_MAX_NEW_TOKENS = 256
RECEIVER_MAX_NEW_TOKENS = 100
LATENT_EXPECTED_DIM = 896

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

EXPECTED_PARAM_COUNT = 4_827_650

# ---------------------------------------------------------------- roles
ROLES = {
    "train-fit": "train",
    "train-checkpoint-selection": "train",
    "valid_seen-step1-gate": "valid_seen",
    "valid_unseen-reserved": "valid_unseen",
}
ROLE_BY_SPLIT = {
    "train": ("train-fit", "train-checkpoint-selection"),
    "valid_seen": ("valid_seen-step1-gate",),
    "valid_unseen": ("valid_unseen-reserved",),
}

# canonical task types (UTF-8 lexicographic order used for tie-breaking)
TASK_TYPES = [
    "look_at_obj",
    "pick_and_place",
    "pick_clean_then_place",
    "pick_cool_then_place",
    "pick_heat_then_place",
    "pick_two_obj",
]

# json_2.1.1 directory prefix -> canonical task type
_TASK_TYPE_PREFIXES = [
    ("look_at_obj_in_light", "look_at_obj"),
    ("pick_and_place_simple", "pick_and_place"),
    ("pick_two_obj_and_place", "pick_two_obj"),
    ("pick_clean_then_place_in_recep", "pick_clean_then_place"),
    ("pick_cool_then_place_in_recep", "pick_cool_then_place"),
    ("pick_heat_then_place_in_recep", "pick_heat_then_place"),
]

# task_type strings as stored in traj_data.json (AlfredTWEnv filter compares these)
ALFWORLD_TASK_TYPE_NAMES = [src for src, _dst in _TASK_TYPE_PREFIXES]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_tensor_file(path: str) -> str:
    """SHA-256 of a serialized tensor file (bytes-level, for manifests)."""
    return sha256_file(path)


def episode_id(source_split: str, game_file: str) -> str:
    """Internal episode ID: first 16 hex chars of SHA-256(split + "\0" + game_file)."""
    return hashlib.sha256(f"{source_split}\0{game_file}".encode("utf-8")).hexdigest()[:16]


def task_type_from_game_file(game_file: str) -> str:
    """game_file is relative to json_2.1.1/; task type = first path component prefix."""
    first = game_file.replace(os.sep, "/").split("/")[1] if "/" in game_file.replace(os.sep, "/") else ""
    for prefix, canonical in _TASK_TYPE_PREFIXES:
        if first.startswith(prefix):
            return canonical
    raise ValueError(f"Cannot determine task type from game_file: {game_file!r} (dir={first!r})")


def write_jsonl(path: str, rows: List[Dict]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
    return sha256_file(path)


def check_no_collision(rows: List[Dict], key: str = "episode_id") -> None:
    seen = set()
    for row in rows:
        eid = row[key]
        if eid in seen:
            raise ValueError(f"Collision on {key}={eid}")
        seen.add(eid)


def sort_manifest_rows(rows: List[Dict]) -> List[Dict]:
    """Manifests are sorted by source_split, task_type, game_file (UTF-8 lexicographic)."""
    return sorted(rows, key=lambda r: (r["source_split"], r["task_type"], r["game_file"]))


def load_manifest(path: str) -> List[Dict]:
    return read_jsonl(path)
