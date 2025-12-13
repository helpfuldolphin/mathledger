# MathLedger Makefile - Windows-friendly via PowerShell
# Usage: make <target>
#
# SECURITY NOTICE:
# - worker, api, enqueue-sample, db-stats targets require DATABASE_URL and REDIS_URL
#   to be set in the environment. They will fail if not configured.
# - For First Organism tests, use 'make first-organism-up' and load .env.first_organism

.PHONY: help worker api enqueue-sample db-stats clean check-local qa-metrics-lint first-organism-up first-organism-down first-organism-validate first-organism-test

# Docs maintenance (optional; run manually before publishing system law docs)
# first-light-system-law-index:
# 	python tools/generate_system_law_index.py
# first-light-system-law-index-check:
# 	python tools/generate_system_law_index.py --check

# Default target
help:
	@echo "MathLedger Build Commands:"
	@echo ""
	@echo "  Development:"
	@echo "    make worker         - Run the worker process (requires DATABASE_URL, REDIS_URL)"
	@echo "    make api            - Run the FastAPI server (requires REDIS_URL)"
	@echo "    make enqueue-sample - Enqueue 2-4 sample jobs (requires REDIS_URL)"
	@echo "    make db-stats       - Show database statistics (requires DATABASE_URL)"
	@echo "    make clean          - Clean temporary files"
	@echo ""
	@echo "  Quality Assurance:"
	@echo "    make check-local    - Run local development guardrails"
	@echo "    make qa-metrics-lint - Run metrics v1 linter QA tool"
	@echo "    make vibe-check     - Run Vibe Compliance Check (VCP 2.1)"
	@echo ""
	@echo "  First Organism (Secure Integration):"
	@echo "    make first-organism-validate - Validate .env.first_organism configuration"
	@echo "    make first-organism-up       - Spin up secure Postgres/Redis stack"
	@echo "    make first-organism-down     - Tear down the First Organism stack"
	@echo "    make first-organism-test     - Run First Organism integration tests"
	@echo "    make fo-cycles               - Run First Organism baseline cycles (1000 cycles)"
	@echo ""
	@echo "  NOTE: Set DATABASE_URL and REDIS_URL before running dev targets."
	@echo "        For First Organism, copy ops/first_organism/first_organism.env.template"
	@echo "        to .env.first_organism and configure secure credentials."

# Run Vibe Compliance Check
vibe-check:
	python tools/vibe_check.py

# Run the worker process (requires env vars)
worker:
	@echo "Starting worker process..."
	@echo "NOTE: Ensure DATABASE_URL and REDIS_URL are set in environment"
	python backend/worker.py

# Run the FastAPI server (requires env vars)
api:
	@echo "Starting FastAPI server..."
	@echo "NOTE: Ensure DATABASE_URL, REDIS_URL, CORS_ALLOWED_ORIGINS, LEDGER_API_KEY are set"
	uv run uvicorn interface.api.app:app --host 0.0.0.0 --port 8000 --reload

# Enqueue sample jobs (requires env vars)
enqueue-sample:
	@echo "Enqueuing sample jobs..."
	@echo "NOTE: Ensure REDIS_URL is set in environment"
	python backend/generator/propgen.py --depth 2 --atoms p q --dry-run
	python backend/generator/propgen.py --depth 2 --atoms p q

# Show database statistics (requires env vars)
db-stats:
	@echo "Database statistics:"
	@echo "NOTE: Ensure DATABASE_URL is set in environment"
	python backend/tools/db_stats.py

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	powershell -Command "if (Test-Path 'tmp\*.json') { Remove-Item 'tmp\*.json' -Force }"
	powershell -Command "if (Test-Path 'backend\lean_proj\ML\Jobs\job_*.lean') { Get-ChildItem 'backend\lean_proj\ML\Jobs\job_*.lean' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 500 | Remove-Item -Force }"

# Run local development guardrails
check-local:
	@echo "Running local development guardrails..."
	python tools/ci-local/branch_guard.py
	@echo "OK: local guardrails passed"

# Run metrics v1 linter QA tool
qa-metrics-lint:
	python tools/metrics_lint_v1.py > NUL 2>&1 || python tools/metrics_lint_v1.py

# First Organism targets
first-organism-validate:
	@echo "Validating .env.first_organism configuration..."
	python tools/validate_first_organism_env.py

first-organism-up:
	@echo "Starting secure First Organism dependencies..."
	@echo "NOTE: Ensure .env.first_organism exists with secure credentials"
	docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism up -d
	@echo "Waiting for services to be healthy..."
	docker compose -f ops/first_organism/docker-compose.yml ps

first-organism-down:
	@echo "Tearing down First Organism stack (preserving volumes)..."
	docker compose -f ops/first_organism/docker-compose.yml down
	@echo "To remove volumes for clean state: docker compose -f ops/first_organism/docker-compose.yml down -v"

first-organism-test:
	@echo "Running First Organism integration tests..."
	@echo "NOTE: Ensure .env.first_organism is loaded and stack is running"
	uv run pytest -v -m "first_organism" tests/integration/

# Run First Organism cycles (baseline mode)
fo-cycles:
	@echo "Running First Organism baseline cycles..."
	@echo "NOTE: Ensure DATABASE_URL is set in environment"
	uv run python experiments/run_fo_cycles.py --mode=baseline --cycles=1000 --out=results/fo_baseline.jsonl
