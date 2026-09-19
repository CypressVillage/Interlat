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
- Sender and Receiver serialization include the upstream ALFWorld instruction.
  Receiver serialization also includes the fixed one-step format demonstration
  required by the 2026-09-17 advisor response, a single-action rule, the generic
  system message, and the latent-plan lead-in. Training and rollout use the
  same registered prompt builder.
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

At the time of this diagnosis, a usable Receiver action-format interface had
not been demonstrated. The later upstream-prompt smoke below supersedes that
format finding but not the policy-quality concern.

## 2026-09-16 upstream-prompt smoke diagnosis (not formal evidence)

After restoring the upstream ALFWorld instruction and applying it consistently
to training and rollout serialization, two train episodes were run in the
`no_comm` group with the frozen `Qwen2.5-0.5B-Instruct` Receiver and a 20-step
cap. All 40 generations contained a parseable `Action:` and all reached the
chat terminator, so the earlier step-zero format collapse was not reproduced.

Neither episode succeeded. In the CD task, the Receiver visited `shelf 1` and
then hallucinated `shelf 2` through `shelf 20` instead of taking the observed
CD. In the book/desklamp task, it repeatedly alternated between `go to desk 1`
and the invalid `open desklamp 1` action. This smoke test supports only that the
restored prompt fixes the action-format interface on these examples. It does
not show usable ALFWorld policy quality, latent-message benefit, or formal
Step 1A performance. The current blocker is Receiver policy competence rather
than `Thought`/`Action` serialization. Do not start the formal three-seed run
on the strength of these checks.

## 2026-09-16 latent-path smoke diagnosis (not formal evidence)

The Sender was run on two train episodes and produced one float32 last-layer
hidden state per generated token (`[256, 896]` for both examples). Injecting
these states directly as the `raw` group, or through a randomly initialized
adapter, caused step-zero `parse_error` outputs. This confirms that unaligned
Sender states are not valid Receiver token embeddings.

A deliberately non-protocol two-example overfit smoke then trained only the
adapter for ten optimizer steps (`lr=1e-3`; the same two examples were used for
training and measurement). Teacher-forced NLL fell from `1.1560` for
`no_comm` to `0.8577` for `matched`, showing that gradients pass through the
injected Sender states and that the adapter can alter Receiver predictions.
The resulting closed-loop rollouts were parseable but solved `0/2` episodes;
both reached the 20-step cap while repeating search actions. The Sender plans
were also generic natural-language advice rather than executable ALFWorld
plans. Artifacts are under `data/step1_latent_smoke/` and are ignored by git.

This smoke verifies execution of the Model A hidden-state extraction, adapter,
and Model B injection path. It does not establish generalization, causal use of
the latent message, task success, or formal Step 1A performance.

## 2026-09-17 advisor path-B check (not formal evidence)

The registered serialization now includes the complete ALFWorld instruction
for both endpoints and one fixed single-step `Thought`/`Action` demonstration
for the Receiver. A two-episode, five-step check used the previously trained
two-example diagnostic adapter and latents only to test whether both injected
and non-injected Receiver paths preserve the action interface. These old
artifacts are not evidence for the revised Sender prompt or task performance.

No-Comm entered all five environment steps in both episodes, but one episode
produced multiple `Action:` lines in three responses. Because the upstream
regular expression is DOTALL, those extra lines were included in the action
sent to the environment rather than rejected. Matched entered five steps in
one episode; the other produced two parseable actions and then omitted the
`Action:` line on its third response. Both groups repeatedly chose invalid or
irrelevant actions and solved `0/2` episodes.

This does not meet the advisor's path-B requirement that the 0.5B Receiver
stably emit one legal action and sustain normal multi-turn interaction. Per the
advisor response, the next model check is the 7B fallback, not full training.
That fallback cannot run in the current environment yet: only the 0.5B model is
cached and Hugging Face access is configured offline. In addition, replacing
the 896-dimensional 0.5B endpoint with the 3584-dimensional 7B endpoint changes
the current full-width adapter from 4,827,650 to 77,113,346 trainable parameters,
so the adapter architecture and resource estimate require confirmation before
formal implementation.

Artifacts are under `data/step1_path_b_smoke/`. The exact current Sender and
Receiver serialization snapshot is
`data/step1_path_b_smoke/checks/smoke_serialization.json`; rollout records are
under `data/step1_path_b_smoke/eval_single_action_rule/results/`.

## 2026-09-19 dual-7B fallback smoke (not formal evidence)

The fallback was rerun with `Qwen/Qwen2.5-7B-Instruct` at both endpoints using
the locked revision `a09a35458c702b33eeacc393d103063234e8bc28`. Two train
episodes produced Sender latent sequences of shapes `[58, 3584]` and
`[61, 3584]`. The corresponding full-width adapter has 77,113,346 trainable
parameters. On the available A10, SDPA, gradient checkpointing, compact
supervised-token CE, input-embedding CPU offload, and the non-caching CUDA
allocator were sufficient for one optimizer step, but a second step ran out of
memory after AdamW state allocation.

The one-step checkpoint changed the two-example teacher-forced selection NLL
from `1.534769` for No-Comm to `1.534079` for Matched, a `0.045%` reduction.
This is too small and too under-trained to establish latent alignment. In a
five-step Matched rollout, one episode emitted `"0"` at the first step and
failed parsing; the other emitted five parseable actions but searched invalid
locations and reached the step cap. Both failed the task (`0/2`).

This smoke verifies that the dual-7B extraction, adapter, checkpoint, injection,
and multi-turn cache path execute end to end. It does not support a task-success
or latent-benefit claim, and the full-width adapter does not fit a useful
multi-step AdamW training run on the current single A10. Artifacts are under
`data/step1_path_a_7b/` and are ignored by git.
