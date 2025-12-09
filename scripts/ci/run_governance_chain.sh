#!/bin/bash
#
# Governance Chain CI Runner
#
# Standalone script for running governance chain checks without GitHub Actions.
# Can be run locally or in any CI environment.
#
# Author: Manus-B (Ledger Replay Architect & PQ Migration Officer)
# Phase: IV - Consensus Integration & Enforcement
# Date: 2025-12-09

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Configuration
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:test@localhost:5432/mathledger_test}"
START_BLOCK="${START_BLOCK:-0}"
END_BLOCK="${END_BLOCK:-1000}"
REPLAY_MODE="${REPLAY_MODE:-sliding_window}"
GOVERNANCE_POLICY="${GOVERNANCE_POLICY:-strict}"
FAIL_ON_BLOCK="${FAIL_ON_BLOCK:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-./reports}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "GOVERNANCE CHAIN CI RUNNER"
echo "============================================================"
echo "Database: $DATABASE_URL"
echo "Block Range: $START_BLOCK - $END_BLOCK"
echo "Replay Mode: $REPLAY_MODE"
echo "Governance Policy: $GOVERNANCE_POLICY"
echo "Output Dir: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Job 1: Schema Migration Check
echo "Job 1/7: Schema Migration Check"
echo "-----------------------------------------------------------"
if python3 scripts/ci/validate_migrations.py \
    --database-url "$DATABASE_URL" \
    --migrations-dir migrations \
    --check-monotonicity \
    --check-hash-law-invariants \
    --check-backward-compatibility \
    --output "$OUTPUT_DIR/migration_validation.json"; then
    echo -e "${GREEN}✓ Schema migration check passed${NC}"
else
    echo -e "${RED}✗ Schema migration check failed${NC}"
    exit 1
fi
echo ""

# Job 2: Epoch Backfill Dry Run
echo "Job 2/7: Epoch Backfill Dry Run"
echo "-----------------------------------------------------------"
if python3 scripts/backfill_epochs.py \
    --database-url "$DATABASE_URL" \
    --dry-run \
    --epoch-size 100 \
    --start-block "$START_BLOCK" \
    --end-block "$END_BLOCK" \
    --hash-version sha256-v1 \
    --output "$OUTPUT_DIR/epoch_backfill.json"; then
    echo -e "${GREEN}✓ Epoch backfill dry run passed${NC}"
else
    echo -e "${RED}✗ Epoch backfill dry run failed${NC}"
    exit 1
fi
echo ""

# Job 3: Replay Verification
echo "Job 3/7: Replay Verification ($REPLAY_MODE)"
echo "-----------------------------------------------------------"
if python3 scripts/replay_verify.py \
    --database-url "$DATABASE_URL" \
    --mode "$REPLAY_MODE" \
    --start-block "$START_BLOCK" \
    --end-block "$END_BLOCK" \
    --fail-fast \
    --output "$OUTPUT_DIR/replay_verification_${REPLAY_MODE}.json"; then
    echo -e "${GREEN}✓ Replay verification passed${NC}"
else
    echo -e "${RED}✗ Replay verification failed${NC}"
    exit 1
fi

# Check replay success rate
if python3 scripts/ci/check_replay_success_rate.py \
    --report "$OUTPUT_DIR/replay_verification_${REPLAY_MODE}.json" \
    --min-success-rate 1.0; then
    echo -e "${GREEN}✓ Replay success rate: 100%${NC}"
else
    echo -e "${RED}✗ Replay success rate < 100%${NC}"
    exit 1
fi
echo ""

# Job 4: Drift Radar Scan
echo "Job 4/7: Drift Radar Scan"
echo "-----------------------------------------------------------"
if python3 scripts/ci/drift_radar_scan.py \
    --database-url "$DATABASE_URL" \
    --start-block "$START_BLOCK" \
    --end-block "$END_BLOCK" \
    --scan-types schema,hash-delta,metadata,statement \
    --governance-policy "$GOVERNANCE_POLICY" \
    --output "$OUTPUT_DIR/drift_signals.json" \
    --evidence-pack "$OUTPUT_DIR/evidence_pack.json"; then
    echo -e "${GREEN}✓ Drift radar scan completed${NC}"
else
    echo -e "${RED}✗ Drift radar scan failed${NC}"
    exit 1
fi

# Check governance signal
if python3 scripts/ci/check_governance_signal.py \
    --evidence-pack "$OUTPUT_DIR/evidence_pack.json" \
    --fail-on-block "$FAIL_ON_BLOCK"; then
    echo -e "${GREEN}✓ Governance signal: OK/WARN${NC}"
else
    echo -e "${RED}✗ Governance signal: BLOCK${NC}"
    exit 1
fi
echo ""

# Job 5: Attestation Integrity Sweep
echo "Job 5/7: Attestation Integrity Sweep"
echo "-----------------------------------------------------------"
if python3 scripts/ci/attestation_integrity_sweep.py \
    --database-url "$DATABASE_URL" \
    --start-block "$START_BLOCK" \
    --end-block "$END_BLOCK" \
    --check-dual-roots \
    --check-merkle-trees \
    --check-domain-separation \
    --output "$OUTPUT_DIR/attestation_integrity.json"; then
    echo -e "${GREEN}✓ Attestation integrity sweep passed${NC}"
else
    echo -e "${RED}✗ Attestation integrity sweep failed${NC}"
    exit 1
fi

# Check integrity violations
if python3 scripts/ci/check_integrity_violations.py \
    --report "$OUTPUT_DIR/attestation_integrity.json" \
    --max-violations 0; then
    echo -e "${GREEN}✓ No integrity violations${NC}"
else
    echo -e "${RED}✗ Integrity violations detected${NC}"
    exit 1
fi
echo ""

# Job 6: PQ Activation Readiness Audit
echo "Job 6/7: PQ Activation Readiness Audit"
echo "-----------------------------------------------------------"
if python3 scripts/ci/check_pq_migration_code.py \
    --check-dual-commitment-support \
    --check-sha3-support \
    --check-cross-algorithm-validation \
    --check-epoch-transition-support \
    --output "$OUTPUT_DIR/pq_readiness.json"; then
    echo -e "${GREEN}✓ PQ activation readiness check passed${NC}"
else
    echo -e "${RED}✗ PQ activation readiness check failed${NC}"
    exit 1
fi
echo ""

# Job 7: Governance Gate
echo "Job 7/7: Governance Gate"
echo "-----------------------------------------------------------"
if python3 scripts/ci/aggregate_governance_results.py \
    --reports-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/governance_summary.json"; then
    echo -e "${GREEN}✓ Governance results aggregated${NC}"
else
    echo -e "${RED}✗ Governance aggregation failed${NC}"
    exit 1
fi

# Check governance pass/fail
if python3 scripts/ci/check_governance_gate.py \
    --summary "$OUTPUT_DIR/governance_summary.json" \
    --fail-on-any-error; then
    echo -e "${GREEN}✓ Governance gate PASSED${NC}"
else
    echo -e "${RED}✗ Governance gate FAILED${NC}"
    exit 1
fi
echo ""

# Print summary
echo "============================================================"
echo "GOVERNANCE CHAIN SUMMARY"
echo "============================================================"
cat "$OUTPUT_DIR/governance_summary.json" | python3 -m json.tool
echo "============================================================"
echo -e "${GREEN}✓ All governance checks passed${NC}"
echo "============================================================"

exit 0
