"""Prepare the configured successful ETO source with upstream conversation merging.

Paper's 5% selection ratio, upstream seed-42 shuffle. No evaluation tasks run.
"""
import argparse
import json
import random
from pathlib import Path
from configs.config import config_for_parser, load_config
from utils.hashing import sha256_file, write_json


def convert(row, games=None):
    games = games if games is not None else load_config()['data']['source']['alfworld_games']
    turns = row['conversations']
    if len(turns) < 4:
        raise ValueError(f"{row['id']}: unexpected initial exchange")

    initial_roles = [turn['from'] for turn in turns[:3]]
    if initial_roles != ['human', 'gpt', 'human']:
        raise ValueError(f"{row['id']}: unexpected initial exchange")

    for i, turn in enumerate(turns):
        expected_role = 'human' if i % 2 == 0 else 'gpt'
        if turn['from'] != expected_role:
            raise ValueError(f"{row['id']}: nonalternating dialogue")

    first = turns[0]['value'] + '\n' + turns[2]['value']
    first += '\nNow, you are given a step-by-step plan to complete this task as follow: '
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': first}
    ]

    for turn in turns[3:]:
        role = 'assistant' if turn['from'] == 'gpt' else 'user'
        messages.append({
            'role': role,
            'content': turn['value']
        })

    games = Path(games).resolve()
    game = (Path(row['game_file']) / 'game.tw-pddl').resolve()
    if not game.is_relative_to(games):
        raise ValueError(f"{row['id']}: game resolves outside ALFWorld root: {games}")
    if not game.is_file():
        raise FileNotFoundError(f"{row['id']}: missing ALFWorld game: {game}")

    return {
        'id': row['id'],
        'game_file': str(game),
        'sender_task': turns[2]['value'],
        'messages': messages
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    experiment = config_for_parser(ap)
    data = experiment['data']['split']
    TRAIN_SOURCE = str(Path(experiment['data']['successful']['output_dir']) / 'alfworld_sft_successful.json')
    ALFWORLD_GAMES = experiment['data']['source']['alfworld_games']
    DATA_SPLIT_SEED = data['seed']
    ap.add_argument('--audit', default=str(Path(experiment['data']['replay']['output_dir']) / 'replay.jsonl'), help='complete replay audit')
    ap.add_argument('--out-dir', default=data['output_dir'], help='new prepared-data directory')
    arg = ap.parse_args()

    out = Path(arg.out_dir)
    with open(TRAIN_SOURCE) as stream:
        raw = json.load(stream)
    records = list(map(json.loads, Path(arg.audit).read_text().splitlines()))
    audit = {r['id']: r for r in records}
    if len(audit) != len(records):
        raise ValueError('duplicate audit trajectory IDs')
    ids = [r['id'] for r in raw]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('training source must contain nonempty, unique trajectory IDs')
    if not set(ids).issubset(audit):
        raise ValueError('training source contains IDs missing from replay audit')
    for trajectory_id in ids:
        record = audit[trajectory_id]
        if (record.get('success') is not True or record.get('reason') != 'won'
                or 'error' in record or 'close_error' in record):
            raise ValueError(f'{trajectory_id}: training source contains an unsuccessful trajectory')
    rows = [ convert(r, ALFWORLD_GAMES) for r in raw ]
    out.mkdir(parents = True, exist_ok = False)

    random.Random(DATA_SPLIT_SEED).shuffle(rows)
    n = max(1,int(len(rows)*data['selection_fraction']));splits = {'selection':rows[:n],'fit':rows[n:]}
    hashes = {}
    for split,rs in splits.items():
        for r in rs:
            r['split'] = split
            r['current_environment_audit'] = audit[r['id']]
        path = out/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii = False) + '\n' for r in rs))
        hashes[split] = sha256_file(str(path))
    write_json(
        str(out/'manifest.json'),
        {
            'games': str(Path(ALFWORLD_GAMES).resolve()),
            'source': str(Path(TRAIN_SOURCE).resolve()),
            'source_sha256': sha256_file(TRAIN_SOURCE),
            'audit_sha256':sha256_file(arg.audit),
            'counts':{ k:len(v) for k, v in splits.items() },
            'sha256':hashes,
            'shuffle_seed':DATA_SPLIT_SEED,
            'selection_fraction':data['selection_fraction'],
            'experiment': {'settings': experiment, 'arguments': vars(arg)},
            'rounding':'floor',
            'filtering':'successful replay only; no additional filtering during preparation',
            'action_rewriting':'none',
            'scope':'user-authorized paper-alignment variant, not existing Step1 split',
            'limitations': [
                'Published ETO source is verified; exact final Interlat training artifact remains unverified.',
                'Selection ratio from paper; shuffle seed from upstream; exact paper split unavailable.',
                'Uses current-environment successful replay records and preserves original supervision text.',
                'Task success does not imply all intermediate observations match.'
            ]
        })
    print({k:len(v) for k,v in splits.items()})


if __name__ == '__main__':main()
