#!/bin/bash
# Governance validation CI script
# Usage: ./scripts/validate-governance.sh
# Exit codes: 0=LAWFUL, 1=UNLAWFUL

set -e

echo "=================================="
echo "GOVERNANCE VALIDATION (CI)"
echo "=================================="
echo ""

# Check for required files
GOVERNANCE_CHAIN="artifacts/governance/governance_chain.json"
DECLARED_ROOTS="artifacts/governance/declared_roots.json"

if [ ! -f "$GOVERNANCE_CHAIN" ]; then
    echo "❌ ERROR: Governance chain not found: $GOVERNANCE_CHAIN"
    echo "   Run: python backend/governance/export.py"
    exit 1
fi

if [ ! -f "$DECLARED_ROOTS" ]; then
    echo "❌ ERROR: Declared roots not found: $DECLARED_ROOTS"
    echo "   Run: python backend/governance/export.py --db-url \$DATABASE_URL"
    exit 1
fi

# Run validation
echo "📋 Validating governance artifacts..."
echo "   - $GOVERNANCE_CHAIN"
echo "   - $DECLARED_ROOTS"
echo ""

if uv run python backend/governance/validator.py \
    --governance "$GOVERNANCE_CHAIN" \
    --roots "$DECLARED_ROOTS"; then

    echo ""
    echo "✅ VERDICT: LAWFUL"
    echo "   All provenance seals validated successfully."
    echo ""

    # Generate verdict if requested
    if [ "$1" = "--generate-verdict" ]; then
        echo "📄 Generating governance_verdict.md..."
        # Verdict already exists, just confirm
        if [ -f "governance_verdict.md" ]; then
            echo "   ✓ governance_verdict.md present"
        fi
    fi

    exit 0
else
    echo ""
    echo "❌ VERDICT: UNLAWFUL"
    echo "   Provenance seal violations detected."
    echo "   See errors above for details."
    echo ""
    exit 1
fi
