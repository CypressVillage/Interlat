# Step 1: Frozen-receiver latent communication

This branch retains only the current `step1/` workflow: public ETO data,
Sender latent collection, adapter training with a frozen Receiver, and ALFWorld
evaluation. It is not a full reproduction of the upstream joint-training method.

- [Workflow and commands](step1/README.md)
- [Environment setup and platform caveats](ENVIRONMENT.md)
- [Execution specification](step1/docs/FROZEN_RECEIVER_V1.md)
- [Upstream alignment and provenance](step1/docs/PAPER_ALIGNMENT.md)

Run commands from this directory using `uv`. `pyproject.toml` and `uv.lock`
remain the dependency sources; `setup.py` is the setuptools shim. The legacy
collection, joint-training, compression, evaluation, and shell-script pipelines
have been removed, including their CLI entry points and promotional assets.
The projection required by Step 1 is now self-contained in
`step1/model/adapter.py`, with its upstream initialization and computation retained.

Local `.venv/`, `vendor/`, data, model caches, and experiment outputs are not
deleted. Historical artifacts must be checked against their original code
revision; do not rewrite their manifests or bypass source-hash checks to reuse
them with this layout. No new experiments are implied by this cleanup.
