"""Configuration and file utilities for the frozen-Receiver workflow."""
from __future__ import annotations

import hashlib
import json
import os

# Run commands from the project's code/ directory.
CODE_ROOT = os.getcwd()
DATA_ROOT = f'{CODE_ROOT}/data'
ETO_DATA = f'{DATA_ROOT}/eto/alfworld_sft.json'
ALFWORLD_GAMES = f'{DATA_ROOT}/json_2.1.1'
PREPARED_DIR = f'{DATA_ROOT}/prepared'
DEFAULT_RUN_DIR = f'{DATA_ROOT}/runs/frozen_receiver_v1'

# Use locally cached models.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# Sender and Receiver models version
SENDER_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
SENDER_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
RECEIVER_MODEL = 'Qwen/Qwen2.5-7B'
RECEIVER_REVISION = 'd149729398750b98c0af14eb82c78cfe92750796'

# Adapter configuration
HIDDEN_SIZE = 3584
ADAPTER_HEADS = 8
ADAPTER_PARAM_COUNT = 77_113_346

# training configuration
TRAIN_SEED = 0
DATA_SPLIT_SEED = 42
GLOBAL_BATCH = 16
N_VALID_SEEN = 140
SENDER_MAX_NEW_TOKENS = 5000
RECEIVER_MAX_NEW_TOKENS = 100
EVAL_MAX_STEPS = 20

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, sort_keys=True, indent=2)
    return sha256_file(path)
