# Frozen Receiver v1: execution specification

## Status and authorization

User-authorized local research experiment, 2026-10-26. This changes the historical
Step 1A CE-only protocol; it is **not advisor approval**, an equivalent replication
of Interlat performance, or a change to the research brief. No remote publication
is authorized. The existing detached checkout and previous dirty files are retained.

Question: can a full-length adapter alone make a frozen Qwen2.5-7B Base consume
latent plans stably and improve useful ALFWorld behavior? Both agents see the goal
and initial observation. This does not test transmission of previously unknown facts.

## Sources and adopted choices

- Paper v5 §3.2, §5, Appendix H: <https://arxiv.org/html/2511.09149v5>.
- Upstream reference: `66a89cb4d4097b2f86cbe48ed9851d6e8578f821`.
- Data/model revisions and prior audits: `PAPER_ALIGNMENT.md`.
- Receiver: `Qwen/Qwen2.5-7B`, revision
  `d149729398750b98c0af14eb82c78cfe92750796`, entirely frozen after delimiter setup.
- Sender: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, offline and frozen.
- All 3,119 published ETO trajectories, unchanged, including the 757 current-environment
  replay failures. Existing seed-42, floor-5% split: 2,964 fit / 155 selection.
- Sender exact original general-plan prompt and ChatML serialization. Adopt public
  collector defaults: sampling temperature .7, top-p .9, top-k 50, repetition penalty
  1, one beam, 5,000 new-token cap. These defaults are not a verified paper-run config.
  Task-specific RNG derived from SHA256("42:" + id) gives deterministic resumability.
  No quality filtering or rewriting; EOS/cap, text, IDs and checksums are saved.
- Full per-token last-layer states immediately **before** each generated token,
  including EOS in the immutable offline cache. Runtime communication excludes a
  terminal Sender EOS and its corresponding hidden vector on both channels; this
  follows the public decoded-plan content and keeps one-to-one positions without
  putting an extra ChatML end-of-turn control token inside the textual plan.
  Capped messages lose no position. The public collector computes corrected `hidden_seq` but saves the
  older `hidden_state` path; v1 deliberately follows the paper's position alignment.
  Storage is bf16 (lossless for bf16 activations), adapter computation float32.
- Appendix-H ChatML, merged original instruction/initial observation, no synthetic
  one-shot demonstration. Inject `<bop> message <eop>` **inside** first user turn,
  before its `<|im_end|>`. Explicit rendering is shared by training and generation;
  assistant content plus terminator/newline are supervised; all other tokens masked.
  Tokenize header/content/terminator as separate segments to make boundaries exact.
- Full-width upstream adapter operator, 77,113,346 trainable parameters. Attention
  does not return unused weights. Random initialization, no old adapter reuse.
  Explicit mean initialization of frozen delimiter rows, including reserved rows.
- Three branches: mixed matched, mixed donor, all-text teacher. Sample uniformly
  `r in {0,.1,...,1}`; text prefix then latent suffix. Donor uses the same r and its
  own text/latents. All objectives use the mixed matched distribution during training;
  evaluation and selection use pure latents. This combination is an explicit choice
  where the paper does not fully specify curriculum application to auxiliary branches.
- Loss: task CE + lambda_sep * negative standard JS + lambda_align *
  (KL(matched || text) + normalized-logit cosine distance). Alpha=beta=1.
  Both matched and donor adapter branches receive gradients. Teacher is frozen/no-grad.
  Every branch is aligned by the same **causally predicted response token**, never
  absolute post-injection position. Full vocabulary, no top-k loss approximation.
  Chunked checkpointed LM-head computation preserves exact objectives and gradients.
- Logical batch 16, serialized microbatch 1 with gradient accumulation. Donor is a
  nonzero cyclic shift within that logical batch (different task ID). Final 4-example
  batch is retained and normalized by its actual supervised-token count.
- AdamW peak lr 1e-5, weight decay 0, betas (.9,.999), eps 1e-8, max grad norm 1.
  Paper 3% linear warmup; cosine decay and wd=0 from public `train.sh`.
  Reproduction choice: lambda_sep increases linearly .1→1 and lambda_align decreases
  .2→.1 over planned updates. Paper gives ranges, not this exact curve or alpha/beta.

## Fixed budget and checkpoint rule (before training)

- One seed (0), at most **3 complete epochs = 558 optimizer updates**, 8,892 fit
  presentations, or **24 hours** from the start of optimization, whichever first.
  Wall budget checked after complete optimizer updates; validation time after the
  last update is outside the optimization budget. No loss-triggered extension.
- 17 warmup updates. Selection NLL on all 155 at initialization and each completed
  epoch/budget endpoint. Lowest response-token NLL wins, including initialization;
  ties retain earlier checkpoint. Preserve initial/best/last and every 25th update.
- Throwaway two-fit-task one-update GPU smoke precedes the real training. Its adapter
  is discarded. It checks all three objectives, gradients, freezing and serialization.
- Collection budget: all 3,119 train/selection and 140 valid_seen tasks, at most 5,000
  generated tokens each (16,295,000 upper bound); no regeneration for plan quality.
- Evaluation: all **140 canonical valid_seen games**, independently held out from fit
  and selection. Use canonical evaluator enumeration (raw glob gives 187). Never read
  expert actions for these games. valid_unseen remains unused.
- Best checkpoint: Matched, No-Comm, raw Mismatched, length/norm-controlled Mismatched,
  Zero, norm-matched Gaussian Random, Text. Initial and last: Matched. At most 1,260
  episodes, 20 actions each, greedy 100 generated tokens per turn (2,520,000 upper bound).
  Stop at im_end or Base EOS; capped turns retain their raw text and are not retried.
  All groups share instruction, initial observation, parser and termination handling.
  No-Comm removes delimiters/message only; it retains the common plan-introduction text.
- Mismatch: fixed same-task-type, no-self permutation chosen by existing deterministic
  length-aware cyclic rule, draw 0. Raw mismatch keeps donor length/norm; controlled
  mismatch interpolates adapted donor to target length and rescales to target Frobenius
  norm. Random draw 0 matches target length/norm. Zero has the same length/mask.
- Diagnostics: best-checkpoint teacher-forced NLL on selection for all seven groups;
  selection is not independent evidence. Global cycle supplies selection donors.
- No checkpoint/config choices based on valid_seen results. Report success counts,
  paired differences/bootstrap intervals, parsing, action admissibility, capped turns,
  per-episode raw responses, training curve/gradient/message norms. One seed and one
  perturbation draw cannot establish across-seed stability; zero discordant pairs can
  make bootstrap intervals misleadingly degenerate. Execution errors are separate
  from behavioral failures and invalidate affected comparisons.

## Platform and traceability

GB10 ARM, existing uv Python 3.10 environment, torch 2.9.1+cu128,
transformers 4.46.3. BF16 Base, SDPA and native PyTorch accumulation/checkpointing,
not paper FlashAttention-2/DeepSpeed. Existing capability 12.1 warning and cusparselt
platform discrepancy remain recorded; successful operations do not verify all CUDA
features. TextWorld ARM adaptation is documented in [`ENVIRONMENT.md`](../../ENVIRONMENT.md). No `uv sync`.

Artifacts: `data/interlat_frozen_receiver_v1_20261026/` (ignored). Commands run from
code with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -m ...`.
Record code hashes/snapshot, preexisting git state/diff, environment inventory,
model/data revisions, per-task text/latent IDs/checksums, all update logs/checkpoints,
and full Base parameter checksums before/after optimization.

This is a bounded frozen-Receiver variant, not the paper's jointly updated actor
and adapter, three-seed performance reproduction. A negative result addresses this
budget/configuration, not the impossibility of frozen-Receiver latent communication.

## Preflight observations

The two-fit-task GPU smoke completed one three-branch update with finite losses and
gradients, and the full frozen Base SHA-256 was unchanged. This does not establish
task performance. The initial full-prefill vs incremental BF16 alignment check had
one position below its .995 cosine threshold (minimum .98618, mean .999875, relative
L2 .01476). Rather than simply loosen the threshold, an independent forced-token
replay of the original padded eight-task batch was run: all 3,829 prediction-position
vectors were **bitwise identical** to the saved vectors. This verifies position
mapping for that batch and identifies the full-prefill discrepancy as numerical/path
dependence, not a shifted extraction. Original failed approximate-check evidence is
retained. Final collection audits require checksum/finite/shape checks for every task
plus exact same-batch replay; full-prefill discrepancies remain recorded diagnostics.

An evaluator integration preflight uses only the first fit task and a throwaway
smoke checkpoint, one step per group; no validation/test tuning is performed.
It checks bitwise training/generation prefix equality and control norms. The first
norm check exposed a 6.9e-5 relative discrepancy from CPU/GPU float32 norm reductions
over millions of random coordinates. The new evaluator (not historical controls)
now recalibrates both normalized controls using a common float64 norm reference,
then records their actual bf16-injection norm ratio. The pre-cast tolerance is 1e-6;
the common bf16 cast must retain a norm ratio within 1e-3. The failed preflight log
is retained as `data/interlat_alignment_20261026/evaluator_smoke.log`.

Before formal training, a source-level serialization discrepancy was corrected:
the initial runtime included the Sender's terminal `<|im_end|>` in the text teacher,
whereas `alfworld_collection.py:260-262` decodes with `skip_special_tokens=True` and
`data_processor.py:394` tokenizes the plan content. Final runtime excludes the terminal
EOS and its paired latent vector; stored cache, sampling, data and task labels do not
change. Initial `early_smoke/` and `evaluator_smoke.json` retain the superseded
EOS-inclusive smoke evidence. New content-span smoke evidence is separate. No claim
that EOS caused all observed malformed output is made. This is a documented
pre-training alignment correction, not outcome-based tuning on validation/test.

The content-span GPU smoke has now completed (`content_smoke/`): one update on two
fit tasks, finite three-branch losses/gradients, and unchanged full Base checksum.
Its paired-prefix/control-norm evaluator smoke also completed for seven groups on
one fit task (`content_evaluator_smoke.json`). This is an **implementation pass but
not a behavioral pass**: five groups produced no parsable action, and No-Comm plus
the controlled mismatch sent inadmissible multiline actions. All seven reached the
100-token turn cap. Excluding EOS did not remove the Text group's repetition on
this task. These outcomes remain recorded; they neither select hyperparameters nor
replace the preregistered full-budget training and independent evaluation.
