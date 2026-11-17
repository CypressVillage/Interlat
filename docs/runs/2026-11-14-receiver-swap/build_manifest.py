"""Archive existing receiver-swap records; never run training or evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[2]
RUN = ROOT / 'data/runs/frozen_receiver_full_20261109'
RUNS = (
    ('instruct_receiver_full_20261111', {'no_comm': 6, 'matched': 52}),
    ('instruct_controls_full_20261112', {'mismatched': 61, 'random': 17}),
    ('instruct_text_full_20261112', {'text': 9}),
)
# Freeze the history search at the refs inspected during this archive, rather
# than letting future archive commits change their own provenance report.
HISTORY_TIPS = (
    'f566bc2a63d1688495f8bb1cabf4c96da083651c',
    'f91e8281b11315fcb24074225fe2b333ec527e2b',
    '8ba396d45e23c20a11f2313e7ef4f9f10524f22c',
    '991b2080afe9969d4ebe93102f196d5a972b9a96',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def record(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
            'size_bytes': path.stat().st_size}


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()


def reconstructed_command(arguments):
    parts = ['uv', 'run', '--no-sync', 'python', '-m', 'eval.eval_receiver_swap']
    for key, value in arguments.items():
        if value is None or value is False:
            continue
        parts.append('--' + key.replace('_', '-'))
        if value is not True:
            parts.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return shlex.join(parts)


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT)


def audit_history(tasks, outputs):
    """Check recorded prompts and source bytes, without executing old code."""
    configs = {name: read_json(RUN / name / 'config.json') for name, _ in RUNS}
    names = {Path(path).name for config in configs.values() for path in config['code']}
    blobs = {}
    for line in git('rev-list', '--objects', *HISTORY_TIPS).decode().splitlines():
        if ' ' not in line:
            continue
        oid, path = line.split(' ', 1)
        if Path(path).name in names:
            content = git('cat-file', 'blob', oid)
            blobs.setdefault(hashlib.sha256(content).hexdigest(), []).append(
                {'blob_oid': oid, 'path': path})
    sources = {}
    for name, config in configs.items():
        sources[name] = {}
        for path, sha in config['code'].items():
            matches = [dict(item) for item in blobs.get(sha, [])]
            for match in matches:
                # log --find-object can include removals; select a commit that
                # actually contains the blob at its recorded path.
                commits = git('log', '--format=%H', '--find-object=' + match['blob_oid'],
                              *HISTORY_TIPS, '--', match['path']).decode().splitlines()
                match['representative_commit'] = None
                for commit in commits:
                    tree = git('ls-tree', commit, '--', match['path']).decode()
                    if match['blob_oid'] in tree:
                        match['representative_commit'] = commit
                        break
            item = {'sha256': sha, 'git_blob_matches': matches,
                    'note': 'Matching bytes do not establish the actual execution commit.'}
            if path == 'eval/eval_receiver_swap.py':
                snapshot = RUN / name / 'eval_receiver_swap.py'
                require(digest(snapshot) == sha, name + ': historical source changed')
                destination = 'sources/' + name + '.py'
                outputs[destination] = snapshot.read_bytes()
                item['archived_snapshot'] = destination
            sources[name][path] = item

    prompt_records = []
    examples = []
    for task in sorted(tasks, key=lambda row: row['id']):
        path = RUN / 'eval_latents' / (task['id'] + '.json')
        meta = read_json(path)
        expected = ('<|im_start|>user\n\n    Please provide a general plan to solve this task.\n\n\n'
                    '    The task is: ' + task['sender_task'] +
                    '\n    <|im_end|>\n<|im_start|>assistant\n')
        require(meta['id'] == task['id'] and meta['prompt'] == expected,
                'unexpected saved Sender prompt: ' + task['id'])
        message_path = RUN / RUNS[2][0] / 'text' / task['id'] / 'message.json'
        message = read_json(message_path)
        tokens = meta['token_ids'][:-1] if meta['ended_eos'] else meta['token_ids']
        require(message['communication_token_ids'] == tokens and
                message['sender_plan_text'] == meta['text'],
                'Text group is not linked to the saved Sender plan: ' + task['id'])
        complete = read_json(RUN / RUNS[2][0] / 'complete.json')
        relative = str(message_path.relative_to(RUN / RUNS[2][0]))
        require(digest(message_path) == complete['artifacts_sha256'][relative],
                'Text message provenance mismatch: ' + task['id'])
        prompt_records.append({**record(path), 'id': task['id'],
                               'prompt_sha256': hashlib.sha256(meta['prompt'].encode()).hexdigest(),
                               'text_message': record(message_path)})
        if len(examples) < 2:
            examples.append({'id': task['id'], 'prompt': meta['prompt'],
                             'sender_task': task['sender_task']})
    outputs['sender_prompt_examples.json'] = encode(examples)
    outputs['historical_audit.json'] = encode({
        'history_search_tips': list(HISTORY_TIPS),
        'search_scope': 'All ancestors of the local refs inspected on 2026-11-17; not remote-only or dangling objects.',
        'source_matches': sources,
        'sender_prompt_audit': {
            'checked_tasks': len(tasks), 'all_match_plan_request_plus_task_only': True,
            'includes_environment_rules_or_legal_action_templates': False,
            'examples': 'sender_prompt_examples.json', 'metadata': prompt_records,
            'text_group_uses_same_saved_plan_for_all_tasks': True,
            'sender_execution_commit': None,
            'limitation': 'Saved prompts and cache configuration establish recorded inputs; '
                          'the original Sender command and exact source commit were not retained here.'},
        'group_message_sources': {
            'matched': 'Adapter applied to target episode Sender states.',
            'mismatched': 'Adapter applied to mapped donor states; raw donor length and norm, not controlled interpolation.',
            'random': 'Generated Gaussian message, target length/norm matched; not a Sender plan.',
            'text': 'Same cached Sender plan token embeddings, terminal EOS excluded; no Adapter.',
            'no_comm': 'No injected message; historical prompt behavior is not replaced by the current no_comm fix.'},
    })


def build():
    outputs = {}
    artifacts = []

    def archive(source, destination):
        outputs[destination] = source.read_bytes()
        info = record(source)
        info['archive_path'] = destination
        artifacts.append(info)
        return info

    task_path = RUN / 'valid_seen.jsonl'
    tasks = [json.loads(line) for line in task_path.read_text().splitlines()]
    ids = {row['id'] for row in tasks}
    require(len(tasks) == len(ids) == 140, 'expected 140 unique tasks')
    require(all(row['split'] == 'valid_seen' for row in tasks), 'unexpected split')
    task_hash = digest(task_path)
    checkpoint = RUN / 'train/best.pt'
    checkpoint_hash = digest(checkpoint)
    sender_path = RUN / 'eval_latents/config.json'
    sender_hash = digest(sender_path)
    sender = read_json(sender_path)
    archive(sender_path, 'configs/sender_eval.json')
    archive(RUN / 'train/config.json', 'configs/adapter_training.json')
    artifacts.extend(record(path) for path in (
        task_path, checkpoint, RUN / 'train/summary.json',
        RUN / 'train/training.jsonl', RUN / 'valid_seen.manifest.json',
        ROOT / 'data/prepared/manifest.json',
        ROOT / 'data/runs/frozen_receiver_v1/successful_eto/manifest.json',
        ROOT / 'data/runs/frozen_receiver_v1/latents/config.json',
    ))
    training_summary = read_json(RUN / 'train/summary.json')
    require(training_summary['checkpoint_sha256']['best.pt'] == checkpoint_hash,
            'checkpoint differs from training summary')
    runs = []
    episode_results = []
    for name, expected in RUNS:
        directory = RUN / name
        config = read_json(directory / 'config.json')
        require(config['data_sha256'] == task_hash, name + ': task hash mismatch')
        require(config['checkpoint_sha256'] == checkpoint_hash, name + ': checkpoint mismatch')
        require(config['latents_config_sha256'] == sender_hash, name + ': Sender config mismatch')
        require(set(config['ids']) == ids and len(config['ids']) == 140, name + ': task IDs mismatch')
        archive(directory / 'config.json', 'configs/' + name + '.json')
        rows = read_json(directory / 'summary.json')
        require(len(rows) == 140 * len(expected), name + ': unexpected result count')
        comparison = read_json(directory / 'comparison.json')
        results = {}
        for group, successes in expected.items():
            selected = [row for row in rows if row['group'] == group]
            require(len(selected) == 140 and {row['id'] for row in selected} == ids,
                    name + ': incomplete/duplicate group ' + group)
            require(sum(bool(row['success']) for row in selected) == successes,
                    name + ': unexpected success count ' + group)
            require(comparison['groups'][group]['successes'] == successes,
                    name + ': comparison differs ' + group)
            results[group] = {'successes': successes, 'episodes': 140,
                              'success_rate': successes / 140}
            for row in sorted(selected, key=lambda item: item['id']):
                episode_results.append({key: row[key] for key in
                                        ('id', 'group', 'success', 'n_steps', 'reason')})
        for filename in ('summary.json', 'complete.json', 'comparison.json', 'validation.json'):
            artifacts.append(record(directory / filename))
        snapshot = directory / 'eval_receiver_swap.py'
        require(digest(snapshot) == config['code']['eval/eval_receiver_swap.py'],
                name + ': saved entrypoint hash mismatch')
        artifacts.append(record(snapshot))
        for key in ('reference', 'controls_reference'):
            if key in config:
                reference = config[key]
                for filename in ('config.json', 'complete.json'):
                    require(digest(Path(reference['directory']) / filename) ==
                            reference[filename.replace('.json', '_sha256')],
                            name + ': reference hash mismatch')
        runs.append({
            'id': name, 'results': results,
            'config_archive': 'configs/' + name + '.json',
            'recorded_arguments': config['arguments'],
            'command_reconstructed_from_arguments': reconstructed_command(config['arguments']),
            'command_status': 'Reconstructed, not an original shell transcript; launcher/environment unknown.',
            'evaluation_seed': config['config']['evaluation']['seed'],
            'max_steps': config['config']['evaluation']['max_steps'],
            'receiver_max_new_tokens': config['config']['evaluation']['max_new_tokens'],
            'source_receiver': config['source_receiver'],
            'target_receiver': config['target_receiver'],
            'adapter_source': config['adapter_source'],
            'recorded_code_sha256': config['code'],
            'saved_entrypoint': record(snapshot),
        })
    mapping_path = RUN / RUNS[1][0] / 'mismatched_mapping.json'
    mapping = read_json(mapping_path)
    require(set(mapping['pairs']) == ids, 'mapping does not cover all tasks')
    require(len({pair['donor_episode_id'] for pair in mapping['pairs'].values()}) == 140,
            'mapping donors are not a permutation')
    for target, pair in mapping['pairs'].items():
        require(pair['target_episode_id'] == target and pair['donor_episode_id'] in ids
                and pair['donor_episode_id'] != target, 'invalid mapping pair')
    archive(mapping_path, 'mismatched_mapping.json')
    outputs['episode_results.json'] = encode(episode_results)
    audit_history(tasks, outputs)
    manifest = {
        'schema_version': 1,
        'report_date': '2026-11-14',
        'archive_date': '2026-11-17',
        'scope': 'Historical Receiver-swap diagnostic; not Step 0 or Gate 0 acceptance.',
        'issue': 'https://github.com/wang-xijun/collab-zhaobc/issues/1#issuecomment-6874622924',
        'paths': {'relative_to': 'Interlat repository root (latent-communication/code)',
                  'archive_paths_relative_to': 'this manifest directory',
                  'local_artifact_root_at_archive': str(ROOT),
                  'availability': 'Local filesystem only; no remote storage or backup verified.'},
        'code_provenance': {
            'local_baseline_commit': 'f566bc2a63d1688495f8bb1cabf4c96da083651c',
            'historical_run_commit': None,
            'audit': 'historical_audit.json',
            'note': 'Local baseline is not claimed as the historical execution commit. '
                    'Per-file hashes and entrypoint snapshots are archived; see the pinned-history audit.'},
        'training_seed': read_json(RUN / 'train/config.json')['seed'],
        'sender_evaluation_cache': {key: value for key, value in sender.items() if key != 'experiment'},
        'mismatched_mapping': {
            'archive_path': 'mismatched_mapping.json', 'file_sha256': digest(mapping_path),
            'recorded_internal_mapping_sha256': mapping['mapping_sha256'],
            'note': 'File SHA-256 and the recorded internal mapping digest are different quantities.',
            'draw': mapping['draw'], 'pairs': len(mapping['pairs'])},
        'runs': runs,
        'artifacts': artifacts,
        'archived_generated_files': {
            name: {'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content)}
            for name, content in outputs.items()},
        'limitations': [
            'Adapter trained against Base Receiver; evaluated against Instruct Receiver without retraining.',
            'Single training seed; these historical results do not establish sample-specific communication utility.',
            'Original shell transcripts, exact execution timestamps, GPU and runtime package lock are not established by this archive.',
            'No full replay, tensor-content audit, or semantic validity audit was performed during archiving.',
            'Per-file matches to pinned Git history are recorded separately; actual execution commits remain unknown.',
        ],
    }
    outputs['run_manifest.json'] = encode(manifest)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify archive without writing')
    args = parser.parse_args()
    outputs = build()
    for name, content in outputs.items():
        path = ARCHIVE / name
        if args.check:
            require(path.is_file() and path.read_bytes() == content, 'archive differs: ' + name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(('Verified' if args.check else 'Generated') + f' {len(outputs)} archive files; 700 episode results, 140 mapping pairs.')


if __name__ == '__main__':
    main()
