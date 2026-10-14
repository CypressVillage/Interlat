"""Replay saved failed train episodes, logging expert proposals before filtering."""
import argparse
import json
from pathlib import Path
import random

from step1.env_utils import make_tw_env, tw_state_fields


def replay(record):
    random.seed(0)
    env = make_tw_env(str(Path.home() / '.cache/alfworld/json_2.1.1' / record['game_file']))
    proposals = []
    agent = env._handcoded_expert
    original = agent.act

    def traced(state, *args, **kwargs):
        action = original(state, *args, **kwargs)
        proposals.append(dict(proposed=action, admissible=action in state['admissible_commands'],
                              admissible_commands=list(state['admissible_commands']),
                              feedback=state['feedback']))
        return action

    agent.act = traced
    steps = []
    try:
        state = env.reset()
        initial = state['feedback']
        for i in range(200):
            fields = tw_state_fields(state)
            if fields['won']:
                break
            action = fields['expert_plan'][0]
            if action == 'look' and i > 0:
                break
            steps.append(action)
            state, _, done = env.step(action)
            if done:
                break
        return dict(episode_id=record['episode_id'], game_file=record['game_file'],
                    initial_feedback=initial, actions=steps, proposals=proposals,
                    won=bool(state['won']))
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--limit', type=int, default=6)
    args = ap.parse_args()
    root = Path(args.run_dir)
    rows = [json.loads(p.read_text()) for p in sorted((root / 'episodes/trajectories').rglob('*.json'))]
    failed = [r for r in rows if not r['won']][:args.limit]
    results = [replay(r) for r in failed]
    (root / 'expert_diagnostic.json').write_text(json.dumps(results, indent=2))
    for row in results:
        invalid = [p['proposed'] for p in row['proposals'] if not p['admissible']]
        print(row['episode_id'], 'won=', row['won'], 'invalid proposals=', invalid)


if __name__ == '__main__':
    main()
