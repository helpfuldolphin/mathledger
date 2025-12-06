# Repository Guidelines

## Project Structure & Module Organization
- `backend/` hosts FastAPI services: `api/` for schemas, `orchestrator/` for queue coordination, `generator/` for job synthesis, `logic/` for normalization flows, and `worker.py` for runtime execution. Lean proofs live in `backend/lean_proj`.
- `scripts/` exposes operational entry points such as migrations, nightly jobs, and policy tooling; extend these rather than adding ad-hoc scripts.
- Frontend assets live under `ui/` (Svelte dashboard) and `templates/` (shared HTML/email); deployment defaults reside in `infra/`, `config/`, and `docs/`. Tests sit in `tests/` and `tests/integration`, with durable outputs in `artifacts/` and `metrics/`.

## Build, Test, and Development Commands
- Run `uv sync` to install Python dependencies, then `uv run python scripts/run-migrations.py` before touching database-backed flows.
- Use `make api` and `make worker` to launch the backend services with the expected Postgres/Redis environment.
- For quick feedback, execute `uv run pytest -m "not slow"`; add `-m integration` when Postgres is available. The full smoke lives in `powershell -File scripts/sanity.ps1`.
- Frontend workflows start with `cd ui && npm install && npm run dev`, and conclude with `npm run build`.

## Coding Style & Naming Conventions
- Target Python 3.11, four-space indents, and typed function signatures. Modules and functions stay `snake_case`; classes use `CamelCase`.
- API payload keys mirror the Pydantic schemas in `backend/api`. Svelte components follow `PascalCase.svelte`.
- Prefer succinct, purposeful comments; keep PowerShell filenames in `Verb-Noun.ps1`.

## Testing Guidelines
- Tests rely on Pytest markers (`unit`, `integration`, `slow`) managed via `pytest.ini`; tag new tests accordingly.
- Share the database fixture from `tests/conftest.py` to avoid autocommit mismatches.
- Regenerate canned data with `scripts/patch_*` helpers when schemas change, and run `run_tests.py` after migrations to mirror CI.

## Commit & Pull Request Guidelines
- Commit subjects follow `component: action` (e.g., `backend: tighten worker retries`) capped at 72 characters; include a brief rationale in the body when needed.
- Work from focused branches (`mvdp-*`, `main` for release). Pull requests link issues, list validation commands (Pytest, sanity PowerShell), and attach screenshots or payload samples for UI/API changes. Call out migration or data-backfill steps explicitly.
