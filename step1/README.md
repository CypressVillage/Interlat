# Step 1A implementation (protocol revision, 2026-08-31)

Implements the revised Step 1A protocol
(`latent-communication/notes/interlat-step1-experiment-protocol.md`) and the
Exchange 2026-08-31 advisor requirements. All frozen constants live in
`step1/common.py` (seeds, split sizes, model revision, expected param count).

## Pipeline

Run from the repo root on the GPU server (`PYTHONPATH=.` or `python -m`).
The venv must be `~/Interlat/.venv` (has alfworld + torch).

```bash
# 1) four ordered manifests + SHA-256 (needs ~/.cache/alfworld/json_2.1.1)
python step1/make_manifests.py --alfworld-data ~/.cache/alfworld/json_2.1.1 --out-dir data/step1/manifests

# 2) canonical expert trajectories (handcoded expert, terminal-success check)
python step1/gen_trajectories.py --alfworld-data ~/.cache/alfworld/json_2.1.1 \
  --manifest data/step1/manifests/train_fit.jsonl --out-dir data/step1/episodes
python step1/gen_trajectories.py --alfworld-data ~/.cache/alfworld/json_2.1.1 \
  --manifest data/step1/manifests/valid_seen_gate.jsonl --out-dir data/step1/episodes
# (valid_unseen is refused unless --allow-reserved-inputs: inputs only, no results)

# 3) Sender latents (greedy <=256, per-token aligned hidden states, float32)
python step1/collect_latents.py --episode-inputs data/step1/episodes/episode_inputs.json \
  --manifest <manifest> --out-dir data/step1/latents_out

# 4) matched-adapter training (single CE forward, freeze asserts, early stopping)
python step1/train_step1.py --episode-inputs ... --train-manifest ... \
  --selection-manifest ... --latents-dir ... --out-dir data/step1/train_s0 --seed 0

# 5) evaluator rollouts (20-step cap, greedy, upstream parser; groups selectable)
python step1/eval_rollout.py --episode-inputs ... --gate-manifest ... \
  --latents-dir ... --adapter-path data/step1/train_s0/adapter_best.pt \
  --alfworld-data ~/.cache/alfworld/json_2.1.1 --groups matched,no_comm \
  --out-dir data/step1/eval_s0
```

## Smoke checks (Exchange deliverable 3)

```bash
python step1/smoke_checks.py serialization --episode-inputs ... --episode-id <eid> --out-dir data/step1/smoke
python step1/smoke_checks.py freeze  --out-dir data/step1/smoke
python step1/smoke_checks.py loss    --episode-inputs ... --manifest <train subset> --latents-dir ... --out-dir data/step1/smoke
python step1/smoke_checks.py clamp   --episode-inputs ... --manifest <train subset> --latents-dir ... --out-dir data/step1/smoke
python step1/smoke_checks.py alignment --episode-inputs ... --latents-dir <latents_out> --episode-id <eid> --out-dir data/step1/smoke
python step1/smoke_checks.py parser  --out-dir data/step1/smoke
python step1/smoke_checks.py prefix  --out-dir data/step1/smoke
python step1/smoke_checks.py earlystop --log data/step1/train_s0/training_log.jsonl --out-dir data/step1/smoke
python step1/power_check.py --out-dir data/step1/smoke
```

Three-seed log heads for smoke: run `train_step1.py` with `--seed 0,1,2` on a
small head-subset of the manifests (e.g. `head -n 4`), then compare
`training_log.jsonl` init records (planned vs actual seeds).

## Known implementation notes / deviations

- Expert trajectories use the ALFWorld handcoded expert; `Thought` text is a
  deterministic per-action-verb template (expert yields actions only).
- Protocol serialization drops the upstream big instruction block and ICL
  (frozen decision); base-model format-collapse risk is shared by all groups
  and must be reported, not silently patched.
- Adapter runs in float32 (base model bf16); latents are stored float32 and
  cast to the embedding dtype at injection.
- Upstream `forward()` (clamp, plan-similarity, random-contrast, text-plan and
  random-message forwards) is never imported into the training path; only
  `AdaptiveProjection` is reused.
