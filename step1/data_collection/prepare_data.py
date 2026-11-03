"""Prepare unfiltered published ETO records with upstream conversation merging.

Paper's 5% selection ratio, upstream seed-42 shuffle. No evaluation tasks run.
"""
import argparse
import json
import random
from pathlib import Path
from step1.common import DATA_SPLIT_SEED, sha256_file, write_json


def convert(row):
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

    return {
        'id': row['id'],
        'game_file': row['game_file'],
        'sender_task': turns[2]['value'],
        'messages': messages
    }


def main():
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument('--source', required = True)
    ap.add_argument('--audit', required = True)
    ap.add_argument('--out-dir', required = True)
    arg = ap.parse_args()

    out = Path(arg.out_dir)
    out.mkdir(parents = True, exist_ok = False)
    raw = json.loads(Path(arg.source).read_text())
    audit = { r['id']: r for r in map(json.loads,Path(arg.audit).read_text().splitlines()) }
    rows = [ convert(r) for r in raw ]
    assert len({r['id'] for r in rows}) == len(rows) == len(audit)

    random.Random(DATA_SPLIT_SEED).shuffle(rows)
    n = max(1,int(len(rows)*.05));splits = {'selection':rows[:n],'fit':rows[n:]}
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
            'source_sha256': sha256_file(arg.source),
            'audit_sha256':sha256_file(arg.audit),
            'counts':{ k:len(v) for k, v in splits.items() },
            'sha256':hashes,
            'shuffle_seed':DATA_SPLIT_SEED,
            'selection_fraction':.05,
            'rounding':'floor',
            'filtering':'none',
            'action_rewriting':'none',
            'scope':'user-authorized paper-alignment variant, not existing Step1 split',
            'limitations': [
                'Published ETO source is verified; exact final Interlat training artifact remains unverified.',
                'Selection ratio from paper; shuffle seed from upstream; exact paper split unavailable.',
                'Preserves known unsuccessful replay records and original supervision text.'
            ]
        })
    print({k:len(v) for k,v in splits.items()})


if __name__ == '__main__':main()
