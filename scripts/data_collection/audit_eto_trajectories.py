"""Replay published ETO actions in the current ALFWorld environment, without filtering.

Only observation comparisons normalize whitespace and remove evaluator prefixes.
Actions use the evaluator parser unchanged. No model, GPU, or 20-action cap is used.
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

from scripts.alfworld_utils import (
    make_plain_tw_env,
    parse_action,
    process_ob,
    strip_intro_text,
)
from scripts.common import ETO_DATA, ALFWORLD_GAMES, DEFAULT_RUN_DIR, sha256_file, write_json


def normalize(text):
    return ' '.join(text.split())


def replay(row, games):
    record = {
        'id': row['id'],
        'game_file': row.get('game_file'),
        'success': False,
        'actions_executed': 0,
        'initial_matches': None,
        'observations_compared': 0,
        'observation_mismatches': 0,
        'reason': None,
    }
    env = None
    try:
        game = (Path(row['game_file']) / 'game.tw-pddl').resolve()
        if not game.is_relative_to(games.resolve()):
            raise ValueError('game resolves outside --games')
        record['game_sha256'] = sha256_file(str(game))

        turns = row['conversations']
        if len(turns) < 4 or [turn['from'] for turn in turns[:3]] != ['human', 'gpt', 'human']:
            raise ValueError('unexpected initial exchange')
        for i, turn in enumerate(turns):
            expected_role = 'human' if i % 2 == 0 else 'gpt'
            if turn['from'] != expected_role:
                raise ValueError(f'nonalternating dialogue at turn {i}')

        env = make_plain_tw_env(str(game))
        state = env.reset()
        actual = strip_intro_text(state['feedback'])
        expected = turns[2]['value']
        record['initial_matches'] = normalize(actual) == normalize(expected)
        if not record['initial_matches']:
            record['initial_observation_mismatch'] = {'expected': expected, 'actual': actual}

        for turn_index in range(3, len(turns), 2):
            action = parse_action(turns[turn_index]['value'])
            if not action:
                raise ValueError(f'empty action at turn {turn_index}')
            state, _, done = env.step(action)
            record['actions_executed'] += 1
            record['success'] = bool(state['won'])

            if turn_index + 1 < len(turns):
                expected = turns[turn_index + 1]['value'].removeprefix('Observation:').strip()
                actual = process_ob(state['feedback'])
                record['observations_compared'] += 1
                if normalize(actual) != normalize(expected):
                    record['observation_mismatches'] += 1
                    record.setdefault('first_observation_mismatch', {
                        'turn': turn_index,
                        'action': action,
                        'expected': expected,
                        'actual': actual,
                    })

            if record['success'] or done:
                record['terminal_turn'] = turn_index
                record['reason'] = 'won' if record['success'] else 'done_not_won'
                break
        else:
            record['reason'] = 'trajectory_exhausted'
    except Exception as error:
        record['reason'] = 'execution_error'
        record['error'] = f'{type(error).__name__}: {error}'
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                record['close_error'] = f'{type(error).__name__}: {error}'
                record['reason'] = 'execution_error'
                record.setdefault('error', record['close_error'])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out-dir', default=f'{DEFAULT_RUN_DIR}/replay', help='new directory; existing outputs are never overwritten')
    parser.add_argument('--limit', type=int, default=0, help='audit only the first N records; 0 means all')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')

    source = Path(ETO_DATA).resolve()
    games = Path(ALFWORLD_GAMES).resolve()
    if not (games / 'train').is_dir():
        parser.error(f'ALFWORLD_GAMES must contain the train/ directory: {games}')
    rows = json.loads(source.read_text())
    if not isinstance(rows, list) or not rows:
        parser.error(f'ETO_DATA must contain a nonempty JSON array: {source}')
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        parser.error(f'duplicate trajectory IDs in ETO_DATA: {source}')
    source_count = len(rows)
    if args.limit:
        rows = rows[:args.limit]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    config = {
        'source': str(source),
        'source_sha256': sha256_file(str(source)),
        'source_records': source_count,
        'games': str(games),
        'limit': args.limit,
        'script_sha256': sha256_file(__file__),
        'environment_helpers_sha256': sha256_file(str(Path(__file__).parents[1] / 'alfworld_utils.py')),
        'versions': {name: version(name) for name in ('alfworld', 'textworld')},
        'comparison': 'Collapse whitespace; strip initial intro and apply process_ob to step feedback.',
        'actions': 'Original actions parsed with evaluator parse_action; no rewriting or step cap.',
        'filtering': 'none',
    }
    write_json(str(out / 'config.json'), config)

    results = []
    with (out / 'replay.jsonl').open('w') as stream:
        for row in rows:
            record = replay(row, games)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            results.append(record)
            if len(results) % 100 == 0 or len(results) == len(rows):
                print(f"replayed {len(results)}/{len(rows)} "
                      f"successes {sum(r['success'] for r in results)} "
                      f"errors {sum('error' in r for r in results)}", flush=True)

    summary = {
        'source_sha256': config['source_sha256'],
        'records': len(results),
        'complete_source': len(results) == source_count,
        'successes': sum(r['success'] for r in results),
        'failures_without_error': sum(not r['success'] and 'error' not in r for r in results),
        'errors': sum('error' in r for r in results),
        'initial_matches': sum(r['initial_matches'] is True for r in results),
        'zero_observation_mismatch_records': sum(
            'error' not in r and r['observations_compared'] > 0 and r['observation_mismatches'] == 0
            for r in results
        ),
        'comparison': config['comparison'],
        'note': 'Current-environment replay, not model evaluation or proof of exact paper data provenance; no records filtered.',
        'replay_sha256': sha256_file(str(out / 'replay.jsonl')),
    }
    write_json(str(out / 'summary.json'), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
