# Code Workspace Instructions

- Treat this directory as the working directory for all code, dependency, test, and experiment commands.
- This project uses `uv`; do not use the system `python`, `python3`, or `pip` for project work.
- Run Python commands as `uv run python ...` and tools as `uv run <tool> ...` from this directory.
- Restore the locked environment with `uv sync --frozen`. Read `ENVIRONMENT.md` before rebuilding it.
- The project environment is `.venv/` and uses uv-managed Python 3.10. The tracked dependency sources are `pyproject.toml` and `uv.lock`.
- `vendor/textworld-1.6.1/` is required by the lock file but intentionally ignored by git. Follow `ENVIRONMENT.md` to rebuild it when absent.
- A failed system `python` import does not show that the project environment is missing; verify with `uv run python` first.
