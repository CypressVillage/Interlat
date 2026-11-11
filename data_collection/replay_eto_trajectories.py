"""Replay published ETO actions in the current ALFWorld environment, without filtering.

Only observation comparisons normalize whitespace and remove evaluator prefixes.
Actions use the evaluator parser unchanged. No model, GPU, or 20-action cap is used.
"""
import argparse
import json
import os
from importlib.metadata import version

from alfworld_utils import (
    make_plain_tw_env,
    parse_action,
    process_ob,
    strip_intro_text,
)
from configs.config import config_for_parser
from utils.hashing import sha256_file, write_json


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
        game = os.path.join(row['game_file'], 'game.tw-pddl')
        games_root = os.path.realpath(games)
        if os.path.commonpath([os.path.realpath(game), games_root]) != games_root:
            raise ValueError('game resolves outside ALFWORLD_GAMES')
        record['game_sha256'] = sha256_file(game)

        turns = row['conversations']
        if len(turns) < 4 or [turn['from'] for turn in turns[:3]] != ['human', 'gpt', 'human']:
            raise ValueError('unexpected initial exchange')
        for i, turn in enumerate(turns):
            expected_role = 'human' if i % 2 == 0 else 'gpt'
            if turn['from'] != expected_role:
                raise ValueError(f'nonalternating dialogue at turn {i}')

        env = make_plain_tw_env(game)
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
    experiment = config_for_parser(parser)
    ETO_DATA = experiment['data']['source']['eto_path']
    ALFWORLD_GAMES = experiment['data']['source']['alfworld_games']
    parser.add_argument('--out-dir', default=experiment['data']['replay']['output_dir'], help='new directory; existing outputs are never overwritten')
    parser.add_argument('--limit', type=int, default=0, help='audit only the first N records; 0 means all')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')

    with open(ETO_DATA) as stream:
        rows = json.load(stream)
    ids = [ row['id'] for row in rows ]
    if len(set(ids)) != len(ids):
        parser.error(f'duplicate trajectory IDs in ETO_DATA: {ETO_DATA}')
    source_count = len(rows)
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(args.out_dir, exist_ok=False)
    config = {
        'experiment': {'settings': experiment, 'arguments': vars(args)},
        'source': ETO_DATA,
        'source_sha256': sha256_file(ETO_DATA),
        'source_records': source_count,
        'games': ALFWORLD_GAMES,
        'limit': args.limit,
        'script_sha256': sha256_file(__file__),
        'environment_helpers_sha256': sha256_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'alfworld_utils.py')),
        'versions': {name: version(name) for name in ('alfworld', 'textworld')},
        'filtering': 'none',
    }
    write_json(os.path.join(args.out_dir, 'config.json'), config)

    results = []
    with open(os.path.join(args.out_dir, 'replay.jsonl'), 'w') as stream:
        for row in rows:
            record = replay(row, ALFWORLD_GAMES)
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
        'replay_sha256': sha256_file(os.path.join(args.out_dir, 'replay.jsonl')),
    }
    write_json(os.path.join(args.out_dir, 'summary.json'), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
