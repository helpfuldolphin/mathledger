#!/bin/bash
# Policy Drift Guard - install with:
#   git config core.hooksPath .githooks
#   chmod +x .githooks/policy_drift_guard.sh
#
# This pre-commit hook lints the active policy, records a snapshot, and
# blocks commits if the latest drift comparison reports BLOCK status.

set -euo pipefail

echo "========================================"
echo "Policy Drift Guard (Pre-commit)"
echo "========================================"

POLICY_DIR="${POLICY_DIR:-artifacts/policy}"
LEDGER_PATH="${POLICY_LEDGER:-$POLICY_DIR/policy_hash_ledger.jsonl}"

if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python"
fi

if [ ! -d "$POLICY_DIR" ]; then
    echo "Policy directory not found at $POLICY_DIR"
    echo "Skipping policy drift guard."
    exit 0
fi

echo ""
echo "[1/3] Linting policy artifacts..."
$PYTHON_CMD scripts/policy_drift_linter.py --policy-dir "$POLICY_DIR" --lint

echo ""
echo "[2/3] Recording pre-commit snapshot..."
NOTES="pre-commit $(git rev-parse --short HEAD 2>/dev/null || echo uncommitted)"
$PYTHON_CMD scripts/policy_drift_linter.py \
    --policy-dir "$POLICY_DIR" \
    --ledger "$LEDGER_PATH" \
    --snapshot \
    --notes "$NOTES" \
    --quiet

echo ""
echo "[3/3] Checking for policy drift..."
if $PYTHON_CMD scripts/policy_drift_linter.py --ledger "$LEDGER_PATH" --drift-check; then
    echo ""
    echo "Policy drift guard passed."
    exit 0
fi

echo ""
echo "POLICY DRIFT BLOCKED: review the summary above."
echo "To approve intentional policy changes:"
echo "  1. Ensure delta_log.jsonl attests the new hash."
echo "  2. Re-run the guard after updating the ledger with a reviewed snapshot."
echo "  3. If you must bypass temporarily (not recommended), use: git commit --no-verify"
exit 1
