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
- Receiver serialization restores the upstream ALFWorld instruction, strict
  `Thought`/`Action` response format, action list, generic system message, and
  latent-plan lead-in. The upstream ICL file is empty, so no demonstration is
  added. Training and rollout use the same registered prompt builder.
- Adapter runs in float32 (base model bf16); latents are stored float32 and
  cast to the embedding dtype at injection.
- Upstream `forward()` (clamp, plan-similarity, random-contrast, text-plan and
  random-message forwards) is never imported into the training path; only
  `AdaptiveProjection` is reused.

## 2026-09-09 smoke diagnosis (not formal evidence)

Baseline code before these fixes: `6fdf71b`.
The old `~/step1_smoke/eval_s0/results/*.jsonl` records all stopped at
step zero with `parse_error`. They do not validate multi-turn KV-cache use.
The old `prefix` smoke check only compares tokenizer prefixes, not the
actual rollout loop.

Fixed `eval_rollout.py`: refresh cached logits after generation and appended
observations, advance the token-history reference each turn, and stop when
the environment is done even if the task was not won. CPU regression tests
exercise the actual loop with deterministic model/environment doubles:

```bash
PYTHONPATH="$HOME/Interlat" "$HOME/Interlat/.venv/bin/python" -m unittest step1.test_rollout_cache
```

The tests failed on the old implementation (stale logits, prefix drift,
terminal handling) and pass after the patch. These tests do not establish
real-model task success or full cached-versus-uncached numerical equivalence.

An independent GPU format diagnostic used only the first two entries of
`~/step1_smoke/manifests/subset-train4.jsonl`, no adapter injection, no
environment rollout, and no training. It compared the registered input
against one explicit ALFWorld/Thought/Action instruction. Both inputs on
both examples still produced no `Action:` line and ended with EOS before
the 100-token cap. This rules out cap truncation for those four outputs;
it does not establish that every instruction strategy will fail or that
Receiver SFT is necessary. The registered serialization remains unchanged.

```bash
PYTHONPATH="$HOME/Interlat" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$HOME/Interlat/.venv/bin/python" -u "$HOME/Interlat/step1/diagnose_receiver_format.py"
```

Evidence: `~/step1_smoke/diagnosis_20260909/format_diagnostic.json`
(full messages, token IDs, outputs, split and episode IDs) and
`format_retry.log`. The first diagnostic attempt, `format.log`, failed
because the diagnostic did not catch the parser's expected ValueError;
this was corrected before the completed run. Existing artifacts were not
overwritten. No gate or reserved episodes were run in this diagnosis.

Remaining blocker: a usable, fixed Receiver action-format interface has
not been demonstrated. Any change to prompts, decoding constraints, model
checkpoint or backbone training needs an explicit protocol decision and
consistent application across training and all comparison groups. Do not
start the formal three-seed run on the strength of these checks.
