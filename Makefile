# MathLedger Makefile - Windows-friendly via PowerShell
# Usage: make <target>
#
# SECURITY NOTICE:
# - worker, api, enqueue-sample, db-stats targets require DATABASE_URL and REDIS_URL
#   to be set in the environment. They will fail if not configured.
# - For First Organism tests, use 'make first-organism-up' and load .env.first_organism

.PHONY: help worker api enqueue-sample db-stats clean check-local qa-metrics-lint first-organism-up first-organism-down first-organism-validate first-organism-test lean-setup lean-check verify-mock-determinism verify-mock-determinism-verbose verify-lean-single evidence-pack evidence-pack-verify

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
	@echo "  Lean Toolchain:"
	@echo "    make lean-setup              - Build Lean project with Mathlib (~2GB, 10-30 min first time)"
	@echo "    make lean-check              - Verify Lean toolchain is correctly installed"
	@echo "    make verify-mock-determinism - Verify pipeline determinism using MOCK harness (no Lean)"
	@echo "    make verify-lean-single      - Verify a single proof with REAL Lean (requires lean-setup)"
	@echo ""
	@echo "  Evidence Pack (External Audit):"
	@echo "    make evidence-pack        - Generate and verify CAL-EXP-3 evidence pack (ONE COMMAND)"
	@echo "    make evidence-pack-verify - Verify existing evidence pack only"
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

# Lean toolchain setup - REQUIRED before verification
lean-setup:
	@echo "============================================================"
	@echo "  LEAN TOOLCHAIN BOOTSTRAP"
	@echo "============================================================"
	@echo ""
	@echo "Building Lean project with pinned Mathlib..."
	@echo "  Lean version:   v4.23.0-rc2"
	@echo "  Mathlib commit: a3e910d1569d6b943debabe63afe6e3a3d4061ff"
	@echo ""
	@echo "NOTE: First run will download ~2GB of Mathlib cache."
	@echo "      This may take 10-30 minutes depending on network speed."
	@echo ""
	cd backend/lean_proj && lake exe cache get && lake build ML
	@echo ""
	@echo "============================================================"
	@echo "  LEAN SETUP COMPLETE"
	@echo "============================================================"
	@echo ""
	@echo "Verification toolchain is ready. You can now run:"
	@echo "  - make lean-check              (verify installation)"
	@echo "  - make verify-lean-single      (verify a proof with REAL Lean)"
	@echo "  - make verify-mock-determinism (test pipeline determinism, no Lean)"
	@echo "  - make worker                  (starts verification worker)"
	@echo ""

# Verify Lean toolchain is correctly installed
lean-check:
	@echo "Checking Lean toolchain..."
	@echo ""
	cd backend/lean_proj && lake env lean --version
	@echo ""
	@echo "Checking ML module builds..."
	cd backend/lean_proj && lake build ML 2>&1 | head -5
	@echo ""
	@echo "OK: Lean toolchain verified"

# ===========================================================================
# MOCK DETERMINISM VERIFICATION
# ===========================================================================
# WARNING: This target uses MOCK/SYNTHETIC mode. It does NOT invoke real Lean.
# For real Lean verification, use: make verify-lean-single PROOF=<path>
#
# What this verifies:
#   - Pipeline produces identical H_t across runs with same seed
#   - Cryptographic chaining (R_t, U_t, H_t) is deterministic
#   - No timestamp/UUID leakage in outputs
#
# What this does NOT verify:
#   - Lean type-checking of proofs (use verify-lean-single for that)
#   - Real proof validity (mock mode uses synthetic artifacts)
verify-mock-determinism:
	@echo "============================================================" 1>&2
	@echo "  MOCK DETERMINISM VERIFICATION" 1>&2
	@echo "  WARNING: This does NOT invoke real Lean verification." 1>&2
	@echo "  For real Lean: make verify-lean-single PROOF=<path>" 1>&2
	@echo "============================================================" 1>&2
	@echo "" 1>&2
	@echo "Mode: MOCK (synthetic artifacts, no Lean)" 1>&2
	@echo "Runs: 2 (identical seed)" 1>&2
	@echo "" 1>&2
	@ML_LEAN_MODE=mock uv run python scripts/verify_core_loop.py --runs 2
	@echo "" 1>&2
	@echo "============================================================" 1>&2
	@echo "  VERIFICATION COMPLETE (MOCK MODE)" 1>&2
	@echo "============================================================" 1>&2

# Verbose mock verification with detailed output (for debugging)
verify-mock-determinism-verbose:
	ML_LEAN_MODE=mock uv run python scripts/verify_core_loop.py --runs 3 --verbose --pretty

# ===========================================================================
# REAL LEAN VERIFICATION
# ===========================================================================
# Verifies a single proof file with REAL Lean 4 type-checking.
# Requires: make lean-setup (one-time, ~2GB download)
#
# Usage:
#   make verify-lean-single PROOF=backend/lean_proj/ML/Jobs/job_example.lean
#
# This is the ONLY target that actually invokes Lean.
verify-lean-single:
ifndef PROOF
	@echo "ERROR: PROOF parameter required" 1>&2
	@echo "Usage: make verify-lean-single PROOF=<path-to-lean-file>" 1>&2
	@echo "" 1>&2
	@echo "Example:" 1>&2
	@echo "  make verify-lean-single PROOF=backend/lean_proj/ML/Jobs/job_test.lean" 1>&2
	@exit 1
else
	@echo "============================================================" 1>&2
	@echo "  REAL LEAN VERIFICATION" 1>&2
	@echo "============================================================" 1>&2
	@echo "" 1>&2
	@echo "File: $(PROOF)" 1>&2
	@echo "" 1>&2
	cd backend/lean_proj && lake build $(shell basename $(PROOF) .lean)
	@echo "" 1>&2
	@echo "============================================================" 1>&2
	@echo "  LEAN VERIFICATION COMPLETE" 1>&2
	@echo "============================================================" 1>&2
endif

# ============================================================================
# Evidence Pack Generation (External Audit)
# ============================================================================
# ONE COMMAND for external evaluators to generate, verify, and receive
# a compliance verdict for CAL-EXP-3 / First-Light evidence packs.

# Generate and verify evidence pack (auto-discovers P3/P4 artifacts)
evidence-pack:
	@echo "============================================================"
	@echo "  CAL-EXP-3 / FIRST-LIGHT EVIDENCE PACK"
	@echo "============================================================"
	@echo ""
	@echo "This is the ONE COMMAND for external evaluators."
	@echo "It will: discover artifacts → generate pack → verify integrity"
	@echo ""
	@uv run python scripts/generate_and_verify_evidence_pack.py \
		--json-report results/first_light/compliance_report.json

# Verify existing evidence pack only (no generation)
evidence-pack-verify:
	@uv run python scripts/generate_and_verify_evidence_pack.py --verify-only
