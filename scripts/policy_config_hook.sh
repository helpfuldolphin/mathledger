#!/bin/bash
# Policy Config Drift Guard
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <baseline-policy> <candidate-policy>" >&2
  exit 2
fi
OLD="$1"
NEW="$2"
echo "[policy-drift-lint] Comparing $OLD -> $NEW"
uv run python scripts/policy_drift_lint.py --old "$OLD" --new "$NEW" --text
