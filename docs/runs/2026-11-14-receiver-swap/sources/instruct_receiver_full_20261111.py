"""Bounded receiver-swap diagnostic, explicitly not a matched training run."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
from unittest.mock import patch

import numpy as np
import torch

from alfworld_utils import process_ob
from configs.config import CODE_ROOT, load_config
import eval.eval_adapter as evaluator
from eval.turns import PROTOCOL, action_turn
from model.receiver import load_receiver
from model.runtime import adapted, load_rows
from utils.hashing import sha256_file, write_json


def select_tasks(rows, count):
    """First canonical task of each type; never select based on outcomes."""
    if count < 1:
        raise ValueError('num-tasks must be positive')
    selected = []; seen = set()
    for row in rows:
        kind = evaluator.task_type(row)
        if kind not in seen:
            selected.append(row); seen.add(kind)
        if len(selected) == count:
            return selected
    raise ValueError('num-tasks exceeds the number of available task types')


def select_all_tasks(rows, expected):
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError('full evaluation requires the complete unique configured task set')
    if any(r['split'] != 'valid_seen' for r in rows):
        raise ValueError('full evaluation is restricted to valid_seen')
    return rows


def verify_runtime_sources(provenance):
    for path, digest in provenance['code'].items():
        if sha256_file(CODE_ROOT/path) != digest:
            raise ValueError('evaluation source changed during run: '+path)


def write_state(out, state):
    temporary = out/'state.json.tmp'
    write_json(str(temporary), state)
    temporary.replace(out/'state.json')


def tracked_run(out, total, run):
    """Never leave a failed/terminated job described as completed or running."""
    write_state(out, {'status': 'initializing', 'pid': os.getpid(), 'completed': 0, 'total': total})
    try:
        return run()
    except BaseException as error:
        state = json.loads((out/'state.json').read_text())
        state.update({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            'error': repr(error)})
        write_state(out, state)
        raise


def verify_swap(source, target):
    """Permit only Receiver identity and diagnostic output directory changes."""
    source = copy.deepcopy(source); target = copy.deepcopy(target)
    for settings in (source, target):
        settings['model']['receiver'].pop('name_or_path')
        settings['model']['receiver'].pop('revision')
        settings['evaluation'].pop('output_dir')
    if source != target:
        raise ValueError('receiver diagnostic must not change other experiment settings')


def traced_rollout(receiver, tok, row, message, config, out):
    history = copy.deepcopy(row['messages'][:2]); turn = 0
    make_env = evaluator.make_plain_tw_env
    generate = evaluator.generate_turn

    class LoggedEnv:
        def __init__(self, path):
            self.env = make_env(path)

        def reset(self):
            state = self.env.reset()
            write_json(str(out/'reset.json'), {k: state.get(k) for k in
                ('feedback', 'admissible_commands', 'won')})
            return state

        def step(self, action):
            result = self.env.step(action)
            observation = process_ob(result[0]['feedback'])
            history.append({'role': 'user', 'content': 'Observation: '+observation})
            print(json.dumps({'id': row['id'], 'action': action,
                'observation': observation, 'won': result[0]['won']}), flush=True)
            return result

        def close(self):
            self.env.close()

    def logged_generate(receiver, tok, past, logits, max_tokens):
        nonlocal turn
        turn += 1
        stem = 'turn_'+str(turn).zfill(2)
        write_json(str(out/(stem+'_input.json')), {'text_messages': copy.deepcopy(history),
            'communication': None if message is None else 'injected_message.pt',
            'injection': 'Initial user turn before im_end; retained through KV history'})
        raw, past, stop, count = generate(receiver, tok, past, logits, max_tokens)
        write_json(str(out/(stem+'_output.json')), {'raw_output': raw,
            'stop_reason': stop, 'generated_tokens': count})
        print(json.dumps({'id': row['id'], 'turn': turn, 'raw_output': raw,
            'stop_reason': stop}), flush=True)
        try:
            accepted = action_turn(raw, final=stop == 'eos')
        except ValueError:
            accepted = None
        if accepted:
            history.append({'role': 'assistant', 'content': accepted[0]})
        return raw, past, stop, count

    with patch.object(evaluator, 'make_plain_tw_env', LoggedEnv), \
         patch.object(evaluator, 'generate_turn', logged_generate):
        return evaluator.rollout(receiver, tok, row, message,
            config['max_new_tokens'], config['max_steps'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(CODE_ROOT/'configs/step1_instruct_diagnostic.yaml'))
    parser.add_argument('--source-config', default=str(CODE_ROOT/'configs/step1_full_20261109.yaml'))
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--num-tasks', type=int, default=None)
    scope.add_argument('--all-tasks', action='store_true', help='all configured valid_seen tasks, two groups')
    parser.add_argument('--out-dir')
    args = parser.parse_args()
    config = load_config(args.config); source = load_config(args.source_config)
    verify_swap(source, config)
    checkpoint = Path(source['training']['output_dir'])/'best.pt'
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (ckpt['base'], ckpt['revision']) != (
            source['model']['receiver']['name_or_path'], source['model']['receiver']['revision']):
        raise ValueError('checkpoint does not match the declared source Receiver')
    raw_rows = [json.loads(line) for line in Path(config['data']['eval_tasks']['output_path']).read_text().splitlines()]
    selected = (select_all_tasks(raw_rows, config['evaluation']['num_tasks']) if args.all_tasks
                else select_tasks(raw_rows, args.num_tasks if args.num_tasks is not None else 3))
    out = Path(args.out_dir or config['evaluation']['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    provenance = {'scope': 'Receiver-swap diagnostic, full set' if args.all_tasks else 'Receiver-swap diagnostic, small subset',
        'limitation': 'No training; transferred Base adapter is not trained for Instruct; no semantic-understanding claim',
        'arguments': vars(args),
        'source_receiver': source['model']['receiver'], 'target_receiver': config['model']['receiver'],
        'config': config, 'protocol': PROTOCOL, 'checkpoint_sha256': sha256_file(checkpoint),
        'adapter_source': {'base': ckpt['base'], 'revision': ckpt['revision'], 'step': ckpt['step']},
        'selection': 'All canonical valid_seen tasks' if args.all_tasks else 'First canonical task of each distinct type, before outcomes',
        'ids': [r['id'] for r in selected], 'groups': ['no_comm', 'matched'],
        'prompt': 'Unchanged, including original action spelling and plan introduction',
        'data_sha256': sha256_file(config['data']['eval_tasks']['output_path']),
        'latents_config_sha256': sha256_file(Path(config['data']['collection']['eval_latents'])/'config.json'),
        'config_sha256': sha256_file(args.config), 'source_config_sha256': sha256_file(args.source_config),
        'code': {p: sha256_file(CODE_ROOT/p) for p in ['eval/eval_receiver_swap.py',
            'eval/eval_adapter.py', 'eval/turns.py', 'model/runtime.py', 'model/receiver.py', 'alfworld_utils.py']}}
    write_json(str(out/'config.json'), provenance)
    shutil.copyfile(__file__, out/'eval_receiver_swap.py')
    print('OUTPUT '+str(out), flush=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    tracked_run(out, 2*len(selected), lambda: evaluate(config, ckpt, selected, out, provenance))


def evaluate(config, ckpt, selected, out, provenance):
    started = time.time()
    verify_runtime_sources(provenance)
    receiver, tok, _ = load_receiver(config=config['model'])
    if receiver.base_model.config.hidden_size != config['model']['adapter']['hidden_size']:
        raise ValueError('target Receiver hidden size is incompatible with the adapter')
    receiver.adapter.load_state_dict(ckpt['adapter'], strict=True)
    receiver.eval()
    rows = load_rows(config['data']['eval_tasks']['output_path'], tok, config['data']['collection']['eval_latents'])
    lookup = {row['id']: row for row in rows}
    summary = []; all_records = {'no_comm': [], 'matched': []}
    for group in ('no_comm', 'matched'):
        for chosen in selected:
            verify_runtime_sources(provenance)
            row = lookup[chosen['id']]
            write_state(out, {'status': 'running', 'pid': os.getpid(), 'group': group,
                'task': row['id'], 'completed': len(summary), 'total': 2*len(selected),
                'elapsed_s': time.time()-started})
            seed = config['evaluation']['seed']
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            directory = out/group/row['id']; directory.mkdir(parents=True)
            with torch.no_grad():
                if group == 'matched' and sha256_file(row['latent_path']) != row['plan']['sha256']:
                    raise ValueError('Sender payload checksum mismatch: '+row['id'])
                message = None if group == 'no_comm' else adapted(receiver, row)
                if message is not None:
                    torch.save(message.to(receiver.base_model.get_input_embeddings().weight.dtype).cpu(),
                        directory/'injected_message.pt')
                start = time.time()
                record = traced_rollout(receiver, tok, row, message, config['evaluation'], directory)
            record.update({'group': group, 'seconds': time.time()-start})
            all_records[group].append(record)
            write_json(str(directory/'episode.json'), record)
            entry = {k: record[k] for k in ('id', 'group', 'success', 'reason', 'n_steps', 'seconds')}
            entry.update({'admissible_actions': sum(record['admissible']),
                'task': row['sender_task'], 'raw_outputs': record['raw_outputs']})
            summary.append(entry)
            write_json(str(out/'summary.json'), summary)
            elapsed = time.time()-started
            write_state(out, {'status': 'running', 'pid': os.getpid(), 'group': group,
                'last_completed_task': row['id'], 'completed': len(summary), 'total': 2*len(selected),
                'elapsed_s': elapsed, 'rough_remaining_s': elapsed/len(summary)*(2*len(selected)-len(summary))})
            print('DONE '+json.dumps(entry), flush=True)
    verify_runtime_sources(provenance)
    checkpoint = Path(config['training']['output_dir'])/'best.pt'
    if sha256_file(checkpoint) != provenance['checkpoint_sha256']:
        raise ValueError('source checkpoint changed during run')
    write_json(str(out/'comparison.json'), {
        'scope': provenance['scope'], 'limitation': provenance['limitation'],
        'groups': {group: evaluator.summarize(records) for group, records in all_records.items()},
        'matched_minus_no_comm': evaluator.paired(all_records['matched'], all_records['no_comm'], config['evaluation']),
        'elapsed_s': time.time()-started})
    write_json(str(out/'complete.json'), {'diagnostic_only': True, 'episodes': len(summary),
        'artifacts_sha256': {str(p.relative_to(out)): sha256_file(p)
            for p in out.rglob('*') if p.is_file() and p.name not in ('state.json', 'state.json.tmp')}})
    write_state(out, {'status': 'completed', 'completed': len(summary), 'total': 2*len(selected),
        'elapsed_s': time.time()-started})


if __name__ == '__main__':
    main()
