"""Prepare the full locked train split, then profile exactly one seed-0 epoch.

Run from code/: CUDA_VISIBLE_DEVICES=1 uv run python -m step1.profile_epoch
Artifacts remain in the ignored data/ directory. Failed canonical trajectories
stop the pipeline; no samples are dropped and no gate/unseen tasks are evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--gpu-budget-gib', type=float, default=30.)
    args = ap.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'status.json'
    if status_path.exists():
        raise RuntimeError('Use a fresh output directory; existing run will not be overwritten')
    source = Path('data/step1_a100/manifests')
    manifests = [source / 'train-fit.jsonl', source / 'train-checkpoint-selection.jsonl']
    counts = [sum(1 for line in p.read_text().splitlines() if line.strip()) for p in manifests]
    if counts != [3221, 332]:
        raise RuntimeError(f'Unexpected locked split counts: {counts}')
    status = dict(seed=0, epochs=1, counts=counts, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  gpu_budget_gib=args.gpu_budget_gib, started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  manifest_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests},
                  stages=[])
    status['source_sha256'] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path('step1').glob('*.py')
    }
    (root / 'working_tree.diff').write_text(subprocess.check_output(['git', 'diff', 'HEAD'], text=True))
    (root / 'profile_epoch_source.py').write_text(Path(__file__).read_text())

    def save():
        status_path.write_text(json.dumps(status, indent=2))

    save()
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    episodes = root / 'episodes'
    latents = root / 'latents_out'
    started = time.monotonic()
    try:
        for phase in ('trajectories', 'latents'):
          for manifest in manifests:
            commands = [
                ('trajectories', ['-m', 'step1.gen_trajectories', '--alfworld-data',
                 str(Path.home() / '.cache/alfworld/json_2.1.1'), '--manifest', str(manifest),
                 '--out-dir', str(episodes)]),
                ('latents', ['-c',
                 'import torch,runpy; torch.cuda.set_per_process_memory_fraction('
                 f'{args.gpu_budget_gib}*1024**3/torch.cuda.get_device_properties(0).total_memory); '
                 'runpy.run_module("step1.collect_latents",run_name="__main__")',
                 '--episode-inputs', str(episodes / 'episode_inputs.json'), '--manifest', str(manifest),
                 '--out-dir', str(latents), '--model-profile', 'qwen2.5-7b'])]
            for name, command in commands:
                if name != phase:
                    continue
                stage = dict(name=name, manifest=str(manifest), command=[sys.executable, *command],
                             state='running', elapsed_s=0)
                status['stages'].append(stage)
                save()
                t = time.monotonic()
                with (root / f'{manifest.stem}_{name}.log').open('w') as log:
                    result = subprocess.run(stage['command'], env=env, stdout=log, stderr=subprocess.STDOUT)
                stage.update(elapsed_s=time.monotonic() - t, returncode=result.returncode,
                             state='done' if result.returncode == 0 else 'failed')
                save()
                if result.returncode:
                    raise RuntimeError(f'{name} failed for {manifest}; see stage log')

        # Hook existing batch/validation routines without changing training math.
        wrapper = '''import json,time,torch,numpy as np
from pathlib import Path
from step1 import train_step1 as train
np.random.seed(0)
torch.cuda.set_per_process_memory_fraction(BUDGET*1024**3/torch.cuda.get_device_properties(0).total_memory)
path=Path(PROFILE)
def record(event, **values):
    torch.cuda.synchronize()
    with path.open('a') as f: f.write(json.dumps(dict(event=event,time=time.time(),**values))+'\\n')
original_batches=train.batches
def batches(*a, **kw):
    record('training_start')
    for i,chunk in enumerate(original_batches(*a, **kw)):
        yield chunk
        if (i+1)%25==0:
            record('batch_progress',batches=i+1,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    record('training_batches_end')
train.batches=batches
original_eval=train.eval_nll
def evaluate(*a, **kw):
    record('validation_start',no_comm=kw.get('no_comm',False))
    result=original_eval(*a, **kw)
    record('validation_end',no_comm=kw.get('no_comm',False),**result)
    return result
train.eval_nll=evaluate
record('process_start')
raise SystemExit(train.main())
'''.replace('BUDGET', repr(args.gpu_budget_gib)).replace('PROFILE', repr(str(root / 'timing.jsonl')))
        command = [sys.executable, '-c', wrapper, '--episode-inputs', str(episodes / 'episode_inputs.json'),
                   '--trajectories-dir', str(episodes / 'trajectories'), '--train-manifest', str(manifests[0]),
                   '--selection-manifest', str(manifests[1]), '--latents-dir', str(latents / 'latents'),
                   '--out-dir', str(root / 'train_s0_epoch1'), '--seed', '0', '--epochs', '1',
                   '--bs', '2', '--accum', '8', '--patience', '2', '--model-profile', 'qwen2.5-7b',
                   '--gradient-checkpointing', '--attn-implementation', 'sdpa']
        stage = dict(name='training', command=command, state='running')
        status['stages'].append(stage)
        save()
        t = time.monotonic()
        with (root / 'train.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        stage.update(elapsed_s=time.monotonic()-t, returncode=result.returncode,
                     state='done' if result.returncode == 0 else 'failed')
        if result.returncode:
            raise RuntimeError('Training failed; see train.log')
        status['state'] = 'done'
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        status['total_elapsed_s'] = time.monotonic()-started
        save()


if __name__ == '__main__':
    main()
