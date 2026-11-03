"""ALFWorld enumeration, observations and action parsing for the current workflow."""
import json
import os
import re

ALFWORLD_TASK_TYPE_NAMES = [
    'look_at_obj_in_light',
    'pick_and_place_simple',
    'pick_two_obj_and_place',
    'pick_clean_then_place_in_recep',
    'pick_cool_then_place_in_recep',
    'pick_heat_then_place_in_recep',
]


def process_ob(ob: str) -> str:
    """Upstream evaluator convention: strip 'You arrive at loc ...' prefix."""
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ') + 2:]
    return ob


def strip_intro_text(obs: str) -> str:
    return '\n'.join(obs.split('\n\n')[1:])


def parse_action(llm_output: str) -> str:
    """Preserve the registered DOTALL parser; only clean boundary whitespace/backticks."""
    matches = re.findall(re.compile(r'Action:\s?(.*)', re.DOTALL), llm_output)
    if not matches:
        raise ValueError("no 'Action:' in llm_output")
    return matches[0].strip(' \t\r\n`')


def list_games(json_root: str, source_split: str) -> list:
    """Canonical evaluator enumeration, unchanged from the old manifest helper."""
    split_dir = os.path.join(json_root, source_split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f'Missing ALFWorld split dir: {split_dir}')
    games = []
    skip_stats = {'movable_or_sliced': 0, 'unknown_task_type': 0,
                  'missing_game': 0, 'missing_solvable_key': 0, 'unsolvable': 0}
    for dirpath, _dirnames, filenames in os.walk(split_dir, topdown=False):
        if 'traj_data.json' not in filenames:
            continue
        rel_dir = os.path.relpath(dirpath, json_root).replace(os.sep, '/')
        if 'movable' in rel_dir or 'Sliced' in rel_dir:
            skip_stats['movable_or_sliced'] += 1
            continue
        game_rel = f'{rel_dir}/game.tw-pddl'
        game_abs = os.path.join(json_root, game_rel)
        if not os.path.exists(game_abs):
            skip_stats['missing_game'] += 1
            continue
        with open(os.path.join(dirpath, 'traj_data.json'), encoding='utf-8') as stream:
            traj = json.load(stream)
        if traj.get('task_type') not in ALFWORLD_TASK_TYPE_NAMES:
            skip_stats['unknown_task_type'] += 1
            continue
        with open(game_abs, encoding='utf-8') as stream:
            game = json.load(stream)
        if 'solvable' not in game:
            skip_stats['missing_solvable_key'] += 1
            continue
        if not game['solvable']:
            skip_stats['unsolvable'] += 1
            continue
        games.append(game_rel)
    print(f'[ENUM] {source_split}: kept={len(games)} skipped={skip_stats}')
    return sorted(games)


def make_plain_tw_env(game_file_abs: str):
    """Evaluator environment with canonical demangling, no handcoded expert."""
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=['gamefile'])
    inner = textworld.start(game_file_abs, request_infos=request_infos)
    env = AlfredInfos(AlfredDemangler(shuffle=False, env=inner))
    env.load(game_file_abs)
    return env
