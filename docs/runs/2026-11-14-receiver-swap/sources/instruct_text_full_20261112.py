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
from model.runtime import adapted, load_rows, text_message
from utils.hashing import sha256_file, write_json

GROUPS = ('no_comm', 'matched', 'mismatched', 'random', 'text')


def validate_groups(groups):
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError('groups must be nonempty, unique supported conditions')
    return list(groups)


def load_reference(directory, provenance, expected_groups=('no_comm', 'matched')):
    """Reuse completed baselines only when their task/model/runtime inputs match."""
    previous = json.loads((directory/'config.json').read_text())
    state = json.loads((directory/'state.json').read_text())
    complete = json.loads((directory/'complete.json').read_text())
    if state['status'] != 'completed' or complete['episodes'] != len(expected_groups)*len(provenance['ids']):
        raise ValueError('reference must contain all complete expected groups')
    for key in ('config', 'protocol', 'checkpoint_sha256', 'data_sha256', 'latents_config_sha256', 'ids'):
        if previous[key] != provenance[key]:
            raise ValueError('reference provenance mismatch: '+key)
    if set(previous['groups']) != set(expected_groups):
        raise ValueError('reference does not contain the expected groups')
    for path, digest in previous['code'].items():
        if path != 'eval/eval_receiver_swap.py' and provenance['code'].get(path) != digest:
            raise ValueError('reference core runtime changed: '+path)
    for path, digest in complete['artifacts_sha256'].items():
        if sha256_file(directory/path) != digest:
            raise ValueError('reference artifact checksum mismatch: '+path)
    return {group: [json.loads((directory/group/ident/'episode.json').read_text())
                    for ident in provenance['ids']] for group in expected_groups}


def message_for(receiver, row, donor, group, seed):
    if group == 'no_comm':
        return None, {'group': group, 'target': row['id']}
    if group == 'matched':
        return adapted(receiver, row), {'group': group, 'target': row['id'], 'donor': row['id']}
    if group == 'text':
        return text_message(receiver, row), {'group': group, 'target': row['id'],
            'control': 'Original Sender plan token embeddings; no adapter; same bop/eop boundaries; terminal Sender EOS excluded',
            'communication_token_ids': row['plan']['communication_token_ids'],
            'sender_plan_text': row['plan']['text']}
    if group == 'mismatched':
        return adapted(receiver, donor), {'group': group, 'target': row['id'], 'donor': donor['id'],
            'control': 'Raw same-task-type donor; length and norm are not matched'}
    if group == 'random':
        messages, info = evaluator.messages_for(receiver, row, donor, seed)
        return messages['random'], {'group': group, 'target': row['id'],
            'control': 'Gaussian, target length and norm matched, float64 recalibrated before common bf16 cast',
            'random': info['random']}
    raise ValueError('unsupported group: '+group)


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
    scope.add_argument('--all-tasks', action='store_true', help='all configured valid_seen tasks')
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=['no_comm', 'matched'])
    parser.add_argument('--reference-dir', type=Path, help='completed matched/no_comm baseline run for paired comparisons')
    parser.add_argument('--controls-reference-dir', type=Path, help='completed mismatched/random run to include in comparisons')
    parser.add_argument('--out-dir')
    args = parser.parse_args()
    groups = validate_groups(args.groups)
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
        'arguments': {**vars(args), 'reference_dir': str(args.reference_dir) if args.reference_dir else None,
            'controls_reference_dir': str(args.controls_reference_dir) if args.controls_reference_dir else None},
        'source_receiver': source['model']['receiver'], 'target_receiver': config['model']['receiver'],
        'config': config, 'protocol': PROTOCOL, 'checkpoint_sha256': sha256_file(checkpoint),
        'adapter_source': {'base': ckpt['base'], 'revision': ckpt['revision'], 'step': ckpt['step']},
        'selection': 'All canonical valid_seen tasks' if args.all_tasks else 'First canonical task of each distinct type, before outcomes',
        'ids': [r['id'] for r in selected], 'groups': groups,
        'prompt': 'Unchanged, including original action spelling and plan introduction',
        'data_sha256': sha256_file(config['data']['eval_tasks']['output_path']),
        'latents_config_sha256': sha256_file(Path(config['data']['collection']['eval_latents'])/'config.json'),
        'config_sha256': sha256_file(args.config), 'source_config_sha256': sha256_file(args.source_config),
        'code': {p: sha256_file(CODE_ROOT/p) for p in ['eval/eval_receiver_swap.py',
            'eval/eval_adapter.py', 'eval/controls.py', 'eval/turns.py', 'model/runtime.py', 'model/receiver.py', 'alfworld_utils.py']}}
    reference = {}
    if args.reference_dir:
        if set(groups) & {'no_comm', 'matched'}:
            raise ValueError('do not rerun baseline groups while reusing reference baselines')
        reference = load_reference(args.reference_dir, provenance)
        provenance['reference'] = {'directory': str(args.reference_dir.resolve()),
            'config_sha256': sha256_file(args.reference_dir/'config.json'),
            'complete_sha256': sha256_file(args.reference_dir/'complete.json'),
            'core_runtime_identical': True,
            'runner_change': 'Only orchestration/control support added; core rollout, model, prompt and input hashes match'}
    if args.controls_reference_dir:
        if set(groups) & {'mismatched', 'random'}:
            raise ValueError('do not rerun control groups while reusing their reference run')
        reference.update(load_reference(args.controls_reference_dir, provenance, ('mismatched', 'random')))
        provenance['controls_reference'] = {'directory': str(args.controls_reference_dir.resolve()),
            'config_sha256': sha256_file(args.controls_reference_dir/'config.json'),
            'complete_sha256': sha256_file(args.controls_reference_dir/'complete.json'),
            'core_runtime_identical': True}
    write_json(str(out/'config.json'), provenance)
    shutil.copyfile(__file__, out/'eval_receiver_swap.py')
    print('OUTPUT '+str(out), flush=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    tracked_run(out, len(groups)*len(selected), lambda: evaluate(config, ckpt, selected, out, provenance, reference))


def evaluate(config, ckpt, selected, out, provenance, reference=None):
    started = time.time()
    verify_runtime_sources(provenance)
    receiver, tok, _ = load_receiver(config=config['model'])
    if receiver.base_model.config.hidden_size != config['model']['adapter']['hidden_size']:
        raise ValueError('target Receiver hidden size is incompatible with the adapter')
    receiver.adapter.load_state_dict(ckpt['adapter'], strict=True)
    receiver.eval()
    rows = load_rows(config['data']['eval_tasks']['output_path'], tok, config['data']['collection']['eval_latents'])
    lookup = {row['id']: row for row in rows}
    groups = provenance['groups']; total = len(groups)*len(selected)
    summary = []; all_records = dict(reference or {})
    donors = evaluator.mapping(rows, config['evaluation']['seed']) if set(groups) & {'mismatched', 'random'} else None
    if donors:
        write_json(str(out/'mismatched_mapping.json'), donors)
    for group in groups:
        all_records[group] = []
        for chosen in selected:
            verify_runtime_sources(provenance)
            row = lookup[chosen['id']]
            write_state(out, {'status': 'running', 'pid': os.getpid(), 'group': group,
                'task': row['id'], 'completed': len(summary), 'total': total,
                'elapsed_s': time.time()-started})
            seed = config['evaluation']['seed']
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            directory = out/group/row['id']; directory.mkdir(parents=True)
            with torch.no_grad():
                donor = lookup[donors['pairs'][row['id']]['donor_episode_id']] if donors else row
                if group != 'no_comm':
                    for source in (row, donor):
                        if sha256_file(source['latent_path']) != source['plan']['sha256']:
                            raise ValueError('Sender payload checksum mismatch: '+source['id'])
                message, info = message_for(receiver, row, donor, group, seed)
                if message is not None:
                    torch.save(message.to(receiver.base_model.get_input_embeddings().weight.dtype).cpu(),
                        directory/'injected_message.pt')
                    info.update({'injected_shape': list(message.shape),
                        'injected_message_sha256': sha256_file(directory/'injected_message.pt')})
                write_json(str(directory/'message.json'), info)
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
                'last_completed_task': row['id'], 'completed': len(summary), 'total': total,
                'elapsed_s': elapsed, 'rough_remaining_s': elapsed/len(summary)*(total-len(summary))})
            print('DONE '+json.dumps(entry), flush=True)
    verify_runtime_sources(provenance)
    checkpoint = Path(config['training']['output_dir'])/'best.pt'
    if sha256_file(checkpoint) != provenance['checkpoint_sha256']:
        raise ValueError('source checkpoint changed during run')
    comparisons = ({group: evaluator.paired(all_records['matched'], records, config['evaluation'])
                    for group, records in all_records.items() if group != 'matched'}
                   if 'matched' in all_records else {})
    report = {
        'scope': provenance['scope'], 'limitation': provenance['limitation'],
        'groups': {group: evaluator.summarize(records) for group, records in all_records.items()},
        'matched_minus_controls': comparisons, 'elapsed_s': time.time()-started}
    if 'no_comm' in comparisons:
        report['matched_minus_no_comm'] = comparisons['no_comm']
    if 'text' in all_records:
        report['text_minus_controls'] = {group: evaluator.paired(all_records['text'], records, config['evaluation'])
            for group, records in all_records.items() if group != 'text'}
    write_json(str(out/'comparison.json'), report)
    write_json(str(out/'complete.json'), {'diagnostic_only': True, 'episodes': len(summary),
        'artifacts_sha256': {str(p.relative_to(out)): sha256_file(p)
            for p in out.rglob('*') if p.is_file() and p.name not in ('state.json', 'state.json.tmp')}})
    write_state(out, {'status': 'completed', 'completed': len(summary), 'total': total,
        'elapsed_s': time.time()-started})


if __name__ == '__main__':
    main()
