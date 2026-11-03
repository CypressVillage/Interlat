"""Enumerate all valid_seen initial states, without teacher actions or unseen games."""
import argparse
import json
from pathlib import Path
from step1.env_utils import make_plain_tw_env, strip_intro_text, list_games
from step1.common import N_VALID_SEEN, sha256_file, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True)
    ap.add_argument('--games', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    assert not out.exists()
    instruction = json.loads(Path(a.source).read_text())[0]['conversations'][0]['value']
    games = [Path(a.games)/p for p in list_games(a.games, 'valid_seen')]
    assert len(games) == N_VALID_SEEN, len(games)
    rows = []
    for i, game in enumerate(games):
        env = make_plain_tw_env(str(game.resolve()))
        try:
            task = strip_intro_text(env.reset()['feedback'])
        finally:
            env.close()
        first = instruction+'\n'+task+'\nNow, you are given a step-by-step plan to complete this task as follow: '
        rows.append({'id': f'seen_{i:03d}', 'game_file': str(game.resolve()),
            'game_sha256': sha256_file(str(game)),
            'sender_task': task, 'split': 'valid_seen',
            'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'},
                         {'role': 'user', 'content': first}]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(str(out.with_suffix('.manifest.json')), {'count': len(rows),
        'sha256': sha256_file(str(out)), 'source': sha256_file(a.source),
        'scope': 'All 140 valid_seen initial states. No expert actions; valid_unseen untouched.'})


if __name__ == '__main__':
    main()
