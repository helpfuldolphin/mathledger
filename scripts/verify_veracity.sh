#!/bin/bash
# Veracity Quick Check Script
# Claude A — Veracity Engineer
#
# Quick verification that veracity is maintained.
# Usage: ./scripts/verify_veracity.sh

set -e

echo "======================================================================="
echo "VERACITY QUICK CHECK"
echo "======================================================================="
echo ""

# Run preflight scanner
python tools/preflight_lean_jobs.py --quiet

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "[PASS] Veracity maintained"
    echo "       malformation=0.00%"
    exit 0
elif [ $EXIT_CODE -eq 1 ]; then
    echo ""
    echo "[FAIL] Veracity regression detected"
    echo ""
    echo "Run for details:"
    echo "  python tools/preflight_lean_jobs.py"
    exit 1
else
    echo ""
    echo "[ABSTAIN] Scanner unavailable"
    exit 2
fi
