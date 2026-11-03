# Step 1: Frozen-receiver latent communication

This branch retains only the current Step 1 workflow in `scripts/`: public ETO data,
Sender latent collection, adapter training with a frozen Receiver, and ALFWorld
evaluation. It is not a full reproduction of the upstream joint-training method.

- [Workflow and commands](scripts/README.md)
- [Environment setup and platform caveats](ENVIRONMENT.md)
- [Execution specification](scripts/docs/FROZEN_RECEIVER_V1.md)
- [Upstream alignment and provenance](scripts/docs/PAPER_ALIGNMENT.md)

Run commands from this directory using `uv`. `pyproject.toml` and `uv.lock`
remain the dependency sources; `setup.py` is the setuptools shim. The legacy
collection, joint-training, compression, evaluation, and shell-script pipelines
have been removed, including their CLI entry points and promotional assets.
The projection required by Step 1 is now self-contained in
`scripts/model/adapter.py`, with its upstream initialization and computation retained.

Local `.venv/`, `vendor/`, and pretrained model caches are retained. On 2026-11-03,
the user requested removal of historical local experiment artifacts, including
prepared splits, latent caches, adapter checkpoints, logs, and environment probes.
Only the original datasets remain under `data/`: ETO at
`data/eto/alfworld_sft.json` and ALFWorld at `data/alfworld/json_2.1.1/`.
Historical documents retain their original paths as provenance; those local
artifacts are no longer present. New splits and experiment outputs must be
regenerated. No new experiments were started by this cleanup.
