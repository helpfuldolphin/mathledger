#!/bin/bash
# PHASE II — NOT USED IN PHASE I
# File: .githooks/check_curriculum_drift.sh
#
# Pre-commit hook for curriculum drift detection.
#
# Installation:
#   git config core.hooksPath .githooks
#   chmod +x .githooks/check_curriculum_drift.sh
#
# Or copy to .git/hooks/pre-commit and make executable.
#
# This hook:
#   1. Records a curriculum snapshot (origin=pre-commit)
#   2. Runs the CI drift guard
#   3. Fails the commit if blocking drift is detected
#

set -e

echo "========================================"
echo "Curriculum Drift Check (Pre-commit)"
echo "========================================"

# Configuration
CONFIG_FILE="${CURRICULUM_CONFIG:-config/curriculum_uplift_phase2.yaml}"
LEDGER_FILE="${CURRICULUM_LEDGER:-artifacts/phase_ii/curriculum_hash_ledger.jsonl}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  Config file not found: $CONFIG_FILE"
    echo "   Skipping curriculum drift check."
    exit 0
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv not found. Trying python directly..."
    PYTHON_CMD="python"
else
    PYTHON_CMD="uv run python"
fi

echo ""
echo "📸 Recording curriculum snapshot..."
$PYTHON_CMD experiments/curriculum_hash_ledger.py \
    --snapshot \
    --config "$CONFIG_FILE" \
    --origin=pre-commit \
    --notes="Pre-commit snapshot for $(git rev-parse --short HEAD 2>/dev/null || echo 'uncommitted')"

echo ""
echo "🔍 Checking for curriculum drift..."
$PYTHON_CMD scripts/check_curriculum_drift.py \
    --config "$CONFIG_FILE" \
    --ledger "$LEDGER_FILE"

DRIFT_EXIT_CODE=$?

if [ $DRIFT_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Curriculum drift check passed."
    exit 0
elif [ $DRIFT_EXIT_CODE -eq 1 ]; then
    echo ""
    echo "🚫 COMMIT BLOCKED: Curriculum drift detected."
    echo ""
    echo "   To bypass this check (NOT RECOMMENDED):"
    echo "   git commit --no-verify"
    echo ""
    echo "   To approve the drift:"
    echo "   1. Review the changes above"
    echo "   2. Run: $PYTHON_CMD experiments/curriculum_hash_ledger.py \\"
    echo "          --snapshot --config $CONFIG_FILE --origin=manual \\"
    echo "          --notes=\"Approved: <your reason>\""
    echo "   3. Re-run: git commit"
    echo ""
    exit 1
else
    echo ""
    echo "⚠️  Drift check encountered an error (exit code: $DRIFT_EXIT_CODE)"
    echo "   Allowing commit to proceed (fail-open)."
    exit 0
fi

