"""Configuration and file utilities for the frozen-Receiver workflow."""
from __future__ import annotations

import os

# Sender and Receiver models version
SENDER_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
SENDER_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
RECEIVER_MODEL = 'Qwen/Qwen2.5-7B'
RECEIVER_REVISION = 'd149729398750b98c0af14eb82c78cfe92750796'

# Run commands from the project root directory.
CODE_ROOT = os.getcwd()
DATA_ROOT = f'{CODE_ROOT}/data'
ETO_DATA = f'{DATA_ROOT}/eto/alfworld_sft.json'
ALFWORLD_GAMES = f'{DATA_ROOT}/json_2.1.1'

PREPARED_DIR = os.environ.get('PREPARED') or f'{DATA_ROOT}/prepared'
DEFAULT_RUN_DIR = os.environ.get('RUN') or f'{DATA_ROOT}/runs/frozen_receiver_v1'

# Original data remains the input for replay auditing and success filtering.
# Change the training source here, not in individual preparation scripts.
ETO_AUDIT_DIR = f'{DEFAULT_RUN_DIR}/replay'
ETO_AUDIT_DATA = f'{ETO_AUDIT_DIR}/replay.jsonl'
SUCCESSFUL_ETO_DIR = f'{DEFAULT_RUN_DIR}/successful_eto'
SUCCESSFUL_ETO_DATA = f'{SUCCESSFUL_ETO_DIR}/alfworld_sft_successful.json'
TRAIN_SOURCE = SUCCESSFUL_ETO_DATA
VALID_SEEN_DATA = f'{DEFAULT_RUN_DIR}/valid_seen.jsonl'

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
