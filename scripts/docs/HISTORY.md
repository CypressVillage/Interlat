# Step 1A implementation history (protocol revision, 2026-08-31)

This is the previous README, retained as a historical execution/evidence record.
Commands, machine paths and status statements below describe their original runs,
not the current environment or authorization. Some module paths predate the
directory reorganization; see [the current README](../README.md) for current
entry points and migration rules. No old run artifacts have been moved or rewritten.

Current cleanup note: only the frozen-Receiver workflow remains. The old CE-only
tools, retired diagnostic wrappers, one-off pipeline runner, and all Step 1 test
files have been removed. Historical test results and commands below are not
claims about a current retained test suite. Current names are `runtime.py`,
`prepare_data.py`, `prepare_eval.py`, `collect_latents.py`,
`train_step1.py`, `eval_adapter.py`, and `summarize_results.py`; consult the README
for the live workflow. The old `run_paper_v1.py` orchestration is represented by
its source snapshot and the recorded commands/artifacts, not another active runner.

## Read-only diagnosis of the step-20 malformed held-out reply

`step1/diagnose_failed_prefix.py` compares checkpoints 0/10/20 on the already
observed failing development episode `8781397fa27dc88c`. Completed artifacts:
`data/step1_7b_prefix_diagnosis_20261023_retry/diagnosis.json` and
`scale_probe.json`. The first attempt timed out during preparation and its
directory is retained separately. No weights were updated.

Training and rollout prefix embeddings are exactly equal. At the first
supervised token, full teacher-forced forward and rollout prefill agree on
argmax; fp32-converted bf16 logits have maximum differences 0.03125/0.03125/
0.015625 for steps 0/10/20. Cached and full-prefix recomputation agree on
greedy argmax over the first five positions along the same token history
(not bitwise logits, and not a whole-rollout cache equivalence claim).

Probability of the expected first token `Thought` is 0.999956 at step 0,
0.992357 at step 10, and 0.000021946 at step 20. The step-20 preferred token
is `preci` (p=0.102921), reproducing the repeated malformed output before any
environment action. No-Comm, delimiter-only, and same-length zero messages
all produce a normal first response containing an Action. This localizes
this observed failure to the learned nonzero prefix's effect rather than a
parser failure or a discrepancy that arises only after decoding begins.

Matched prefix Frobenius norms grow 8.965 -> 13.782 -> 16.356, with finite
values. Scaling the step-20 prefix to the step-10 norm still produces `preci`;
scaling to the step-0 norm produces a malformed leading header but eventually
an Action. Therefore simple uniform norm growth alone does not explain the
failure. These are posthoc inference-only probes, not a new method or a
validated repair. They do not establish excessive learning rate, numerical
overflow, semantic learning, or general failure of frozen endpoints.

## Reinitialized 13-fit / 4-held-out diagnostic after delimiter repair

User-authorized run: `data/step1_7b_fixed_13fit4val_20261023/`, launched with
`uv run --no-sync python -u -m step1.diagnose_two_sample_fit --split-trial
--source data/step1_7b_20261023/diagnostic_s0
--alfworld-data data/step1_05b_20261023/alfworld/json_2.1.1
--out-dir data/step1_7b_fixed_13fit4val_20261023` (offline HF environment).

Fresh 7B adapter, seed 0, frozen backbones and repaired delimiter means;
same success-only 13/4 task split and cached 7B Sender latents. No checkpoint
is loaded. Uses bs2/accum8, AdamW lr5e-5/wd.01, SDPA and gradient checkpointing.
Twenty fixed updates (one shuffled traversal of 13 fit tasks per update),
without early stopping. As in the previous 13/4 trainer, seven microbatch
losses are divided by eight, including the last singleton microbatch;
this is explicitly not an exact 16-example effective batch.

At updates 0/1/5/10/20, save token-weighted train and held-out NLL, checkpoint,
and five-action real rollouts for **all 13 train and 4 held-out tasks**, with
`split` recorded in each JSONL row. No-Comm rollouts are saved at update 0;
its unchanged frozen-model NLL is measured on both subsets at each snapshot.
Evaluation preserves training RNG state. The four held-out tasks never enter
the gradient loop. They are repeatedly inspected development tasks, not an
untouched final test. Excluded expert failures and original-supervision
provenance remain unresolved; this is not formal Step 1A evidence.

Completed 20-update result: train NLL at steps 0/1/5/10/20 was
1.0396/1.0792/0.9584/0.9183/0.8889; held-out NLL was
1.1425/1.2297/1.0929/0.9640/0.9747. Frozen No-Comm references were
1.0220 (train) and 1.1450 (held-out). Step 10 has the lowest held-out NLL
among the five measured snapshots, not necessarily among every update.
Train parse-error episode counts were 0/7/2/2/0 out of 13; held-out counts
were 0/2/2/0/1 out of 4. At step 20 one held-out task emitted repeated
`preci` text without an Action. Another retained a trailing `---` in its
parsed action; single-Action counts therefore do not establish action
validity. At steps 10/20 held-out task success was 0/4 under the five-action
cap; these are not formal 20-action success measurements. Loss improvement
does not establish stable generation or task benefit. Summary and full
outputs are saved under the run directory; no further training was launched.

## 2026-10-23 delimiter initialization repair and two-sample fit diagnostic

`ReceiverWithLatent` now explicitly initializes both delimiter IDs to the
original embedding-matrix mean even when the tokenizer allocates those IDs
inside an already padded vocabulary. Previously the mean-initialization
branch was skipped for 7B (151,667 tokenizer entries versus 152,064 embedding
rows), leaving near-zero reserved rows. Base embeddings remain frozen.
Regression coverage includes both reserved-row and vocabulary-growth cases.

User-authorized diagnostic command (from `code/`):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -u -m step1.diagnose_two_sample_fit \
  --source data/step1_7b_20261023/diagnostic_s0 \
  --alfworld-data data/step1_05b_20261023/alfworld/json_2.1.1 \
  --out-dir data/step1_7b_delimiter_fit_20261023
```

Uses the two training tasks evaluated in the preceding GB10 run (not the
historical A100 pair), seed 0, a fresh 7B adapter, batch 2 / accumulation 1,
AdamW lr 5e-5 / weight decay .01, and exactly ten updates without early
stopping. Nonfinite losses/gradients abort execution. Steps 0/1/5/10 save
checkpoints, teacher-forced NLL on the **same two fit samples**, and real
five-step Matched rollouts. Step 0 also runs No-Comm rollouts; its frozen
baseline NLL is measured at every snapshot. No independent-validation or
task-generalization claim is made. Evaluation preserves training RNG state.

Outputs include `run_manifest.json`, `diagnostic_log.jsonl`,
`outputs_step{0,1,5,10}.jsonl`, and `adapter_step{0,1,5,10}.pt` under the new
output directory, with source snapshot and environment inventory. Changing
the sample set and update schedule as well as fixing delimiters means a
comparison with the previous 13/4 run cannot isolate the fix's causal effect.

Completed observation: delimiter equality-to-mean assertion passed. Fit NLL
at steps 0/1/5/10 was 1.52145 / 1.52807 / 1.30699 / 1.21527; the frozen
No-Comm reference was 1.46134. Matched single-Action reply counts were
10/10, 5/6, 10/10, 10/10 respectively. At step 1 the alarmclock episode
failed parsing at its first reply; the other episode completed five actions.
Both episodes completed five actions at steps 0, 5 and 10. All bounded
rollouts were unsuccessful under the five-action cap. The step-1 transient
collapse followed by recovery supports further investigation of optimization
stability, not a conclusion that latent semantics were learned. In particular,
the untrained adapter at step 0 already preserved action formatting.
All ten updates completed with finite losses/gradients. These are same-sample
fit observations, not validation or a causal estimate of delimiter repair.

## 2026-10-23 GB10 0.5B diagnostic trial

The latest advisor authorization (2026-10-12, Issue #1) permits **7B seed 0**
full training after parser cleanup/material preparation. A user-requested
0.5B run is a separate diagnostic, not that formal experiment. Do not infer
permission for seeds 1/2 or Step 1B from this trial.

From `code/`, using the prepared GB10 environment described in
[`ENVIRONMENT.md`](../../ENVIRONMENT.md):

```bash
uv run --no-sync python -m unittest step1.test_action_parser step1.test_rollout_cache step1.test_serialization step1.test_model_profiles step1.test_training_schedule
uv run --no-sync python -u -m step1.run_05b_trial \
  --alfworld-data data/step1_05b_20261023/alfworld/json_2.1.1 \
  --out-dir data/step1_05b_20261023/new_trial
```

The runner selects 16 fit / 6 separate checkpoint-selection tasks in canonical
task-type round-robin order, uses seed 0, bf16 frozen endpoints, the 4,827,650
parameter adapter, CE only, AdamW 5e-5, bs2/accum8, at most 20 epochs and
patience=2. It then runs two **training** tasks for five steps per group
(No-Comm and Matched). It generates all four canonical manifests but does not
evaluate gate or reserved tasks. Run output directories must be new.

The first preparation attempt failed on 5/22 handcoded-expert trajectories
(`expert_fallback_look`) and stopped before training. The explicit flag
`--diagnostic-success-only` allows only a non-formal diagnosis on the remaining
13 fit / 4 selection tasks, preserving all failure records and the canonical
manifests. **This must never be used to silently trim the formal dataset.**
The original-paper supervision source and all game mappings still require
verification before full-data training; this runner uses newly generated
handcoded-expert trajectories with templated Thoughts, not verified original
paper supervision. A partial final accumulation window is divided by the
configured accumulation factor, as in the existing trainer; the 13-example
diagnostic is not an exact 16-example effective batch.

Artifacts: `data/step1_05b_20261023/trial_s0/` (initial stopped attempt) and
`data/step1_05b_20261023/diagnostic_s0/` (explicit diagnostic). The run manifest
records commands, code revision/patch, package inventory, subset hashes and
exclusions. Logs/checkpoints stay out of git. Cosmetic action cleanup strips
only boundary whitespace/backticks; it does not repair semantic errors or
multiple Action lines. The trainer now stops after exactly two consecutive
non-improving epochs (previously three), defaults to 20 epochs, and also seeds
NumPy. CPU regressions cover the early-stop schedule and success/failure at
action 20.

Completed diagnostic observation: 20 epochs / 20 optimizer updates, training
loop elapsed 123 s; best adapter at epoch 20 (zero-based index 19). On the four
independent selection trajectories, response-token NLL fell from 0.7792 after
the first update to 0.6812 at the best checkpoint; No-Comm was 0.9012 (24.4%
lower NLL for the final Matched adapter). Early stopping did **not** trigger;
its patience logic was exercised separately by the CPU regression, not by
this improving real run. Peak training allocation/reservation was
3,290,552,832 / 7,216,300,032 bytes.

Best-checkpoint five-step rollouts on two training tasks completed without
parse-error termination, but both groups solved 0/2 tasks. Matched had exactly
one `Action:` in all 10 outputs, including incomplete commands such as
`go to 1` and repeated irrelevant search. No-Comm had exactly one `Action:` in
9/10 outputs; its other output contained two actions and was passed as a
multi-line string by the legacy DOTALL parser. Thus this verifies bounded
execution and lower teacher-forced NLL, **not** policy competence, stable
single-action behavior in all groups, task benefit, or formal Step 1A validity.
Do not extend full-data/three-seed training on the strength of these results.

### Same-day 7B diagnostic follow-up

The user clarified the requested next model is the advisor-specified
`Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`, not a 20B model.
The model-selectable runner keeps the legacy 0.5B entry point compatible:

```bash
uv run --no-sync python -u -m step1.run_step1_trial \
  --model-profile qwen2.5-7b \
  --alfworld-data data/step1_05b_20261023/alfworld/json_2.1.1 \
  --out-dir data/step1_7b_20261023/diagnostic_s0 \
  --diagnostic-success-only
```

This uses the same canonical round-robin task selection and explicit expert
failure handling as the 0.5B diagnosis, while extracting **new 3584-dimensional
7B latents** and training a **new 77,113,346-parameter adapter** from seed 0.
No 0.5B latent/checkpoint is reused with the 7B endpoints. Train settings remain
20 epochs maximum, bs2/accum8, AdamW 5e-5, patience=2, frozen bf16 backbones,
SDPA and gradient checkpointing. Results and logs are separate from 0.5B.
The strict preparation attempt at `data/step1_7b_20261023/` does not permit
excluding failed expert trajectories and therefore is distinct from the
explicit diagnostic child run. No full-data training, new seed, or gate task
evaluation is authorized by this diagnostic command.

Completed 7B observation: four epochs / four optimizer updates, 511 s including
model/data startup. Selection NLL by epoch was 1.6593, 1.3869, 2.2645, 2.2844;
the best adapter is epoch 2, and patience=2 stopped after epoch 4. No-Comm NLL
was 1.1450, so even the best Matched checkpoint was worse on these four
selection trajectories. Only the step-1 snapshot was reached; the planned
step-5/10 snapshots were not produced. Peak allocated/reserved memory was
19,131,107,840 / 21,237,858,304 bytes; there was no OOM.

With the best adapter, No-Comm completed five steps on each of the two train
tasks and emitted exactly one `Action:` in 10/10 responses. Matched emitted
no `Action:` on either task and stopped before the first environment action
(`parse_error`, 0/2 parseable responses). Both groups solved 0/2 tasks under
the five-step cap. The pipeline completed, but **the Matched action interface
failed**; this is not a successful 7B smoke gate or evidence for scaling up.
The prior A100 two-example run used different data/training conditions; this
trial does not identify hardware, optimization, or injection as the cause.
See `data/step1_7b_20261023/diagnostic_s0/diagnostic_summary.json`,
`train_s0/training_log.jsonl`, and `eval_train2/results/*.jsonl` for the exact
observation. Diagnose before further training; do not silently change the
learning rate, adapter architecture, or freezing boundary.

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

## 2026-10-07 A100 2x10-step smoke (not formal evidence)

Environment rebuilt on a new A100 server per `ENVIRONMENT.md`
(uv-managed Python 3.10.22, torch 2.4.1+cu121, transformers 4.46.3, alfworld
0.3.5, vendored textworld 1.6.1). ALFWorld `json_2.1.1` re-downloaded from the
official 0.2.2 release assets; canonical enumeration reproduced
train 3553 / valid_seen 140 / valid_unseen 134. Locked
`Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`
re-downloaded and loaded offline. All runs on one idle A100 80GB
(`CUDA_VISIBLE_DEVICES=1`); the other GPU carried an unrelated service.

Deviations from the 2026-09-20 plan, recorded up front: the original A10 smoke
artifacts were not available on this machine, so the two episodes were rebuilt
from freshly generated manifests (latent shapes `[56, 3584]` / `[60, 3584]`
versus the A10 `[58, 3584]` / `[61, 3584]`); sample identity with the A10 run
is NOT verified. Effective batch composition was bs2 x accum1 (same effective
size as the A10 bs1 x accum2), seed 0 assumed, and rollouts used SDPA.

Training: 2 train episodes = 2 selection episodes, 10 optimizer steps
(lr 5e-5, AdamW, gradient checkpointing, SDPA), checkpoints at steps 1/5/10.
Selection NLL (same two episodes): No-Comm `1.5559`; Matched by step:
`2.0619` (1), `1.8538` (2), `1.7474` (3), `2.2957` (4), `1.4477` (5),
`1.4338` (6), `1.2042` (7, best), `1.2378` (8), `1.2341` (9), `1.3735` (10).
Training peak VRAM 17.16 GB allocated / 17.61 GB reserved. Patience-2 early
stop triggered after epoch 9; step-10 checkpoint saved before stop.

Matched 5-step rollouts from the step-5 and step-10 checkpoints (2 episodes
each, 5-step cap): 20/20 responses contained exactly one parseable `Action:`,
zero `parse_error`, and both episodes completed all 5 environment steps at
both checkpoints; task success 0/2 (recorded only). Rollout peak VRAM
15.9 GB allocated / 16.37 GB reserved. Cosmetic quirks observed and logged:
leading ```` ``` ```` / `<|im_start|>` artifacts in some first-step outputs,
and one parsed action retaining a trailing code fence
(`go to desk 1\n``` `) that the environment rejected as invalid.

This run verifies the approved A100 execution plan end to end (multi-step
AdamW on 7B, checkpoint snapshots, format gate rollouts) but remains
exploratory: it is not the same two episodes as the A10 smoke, it does not
establish latent benefit, and formal single-seed, 3-seed, and Step 1B runs
stay paused pending advisor review. Artifacts and `run_manifest.json` are
under `data/step1_a100/` and are ignored by git.

## Historical epoch-2-3 continuation commands

The one-off `continue_paper_epochs.py` wrapper was removed during cleanup.
The commands below preserve its fixed paths and options, not a reusable run
recommendation or permission to restart training. This cleanup did not run them.
Use the original experiment's source snapshot for reproduction: current code
hashes have changed, and strict resume checks must not be bypassed. All paths
are relative to `latent-communication/code/`.

The historical training module name below was `step1.train_paper_adapter`.
Its implementation now lives at the sole current entry point `step1.train_step1`;
the old module has been removed. The historical commands are not rewritten,
and the current entry point still enforces source-hash checks on resume.
Likewise, the historical `step1.eval_paper_adapter` implementation is now
`step1.eval_adapter`; both the old name and the older CE-only `eval_rollout.py`
entry point have been removed. Historical commands below require their original
source snapshot, not a mixture of old arguments and current evaluation code.

The wrapper loaded `train/epoch_01.pt` and completed epochs 2 and 3 (three total,
not three additional epochs), then evaluated the new training directory.
`--resume-epoch` loads adapter weights and **reinitializes AdamW**; it does not
restore the full optimizer/RNG state. `--max-hours 0` disabled the wall-clock
limit for this specific invocation; it is not a change to the v1 default budget.
The original wrapper required both output directories to be absent before
starting and never overwrote previous logs.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -u -m step1.train_paper_adapter \
  --fit data/interlat_alignment_20261026/prepared/fit.jsonl \
  --selection data/interlat_alignment_20261026/prepared/selection.jsonl \
  --latents data/interlat_frozen_receiver_v1_20261026/latents \
  --out-dir data/interlat_frozen_receiver_v1_20261026/train_epochs_2_3 \
  --epochs 3 --max-hours 0 \
  --resume-epoch data/interlat_frozen_receiver_v1_20261026/train/epoch_01.pt

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -u -m step1.eval_paper_adapter \
  --data data/interlat_frozen_receiver_v1_20261026/valid_seen.jsonl \
  --latents data/interlat_frozen_receiver_v1_20261026/eval_latents \
  --train-dir data/interlat_frozen_receiver_v1_20261026/train_epochs_2_3 \
  --out-dir data/interlat_frozen_receiver_v1_20261026/eval_epochs_2_3 \
  --selection data/interlat_alignment_20261026/prepared/selection.jsonl \
  --train-latents data/interlat_frozen_receiver_v1_20261026/latents
```

The wrapper recorded commands in `continuation_commands.jsonl`, subprocess
stdout/stderr in `train_epochs_2_3.log` and `eval_epochs_2_3.log`, and a final
`continuation_complete.json` containing the finish timestamp and training/eval
summaries, all under `data/interlat_frozen_receiver_v1_20261026/`. Existing
artifacts are retained unchanged; this document does not assert their completion
or recreate wrapper metadata when the commands are run directly.
