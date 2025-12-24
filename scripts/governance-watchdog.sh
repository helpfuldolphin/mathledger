#!/bin/bash
# Documentation Governance Watchdog - First Light Integration
#
# Runs documentation governance drift radar to detect:
# - Uplift claims without "integrated-run pending" disclaimer
# - TDA enforcement claims before runner wiring complete
# - Contradictions to Phase I-II disclaimers
#
# Exit Codes:
#   0 - PASS (no violations)
#   1 - FAIL (critical violations)
#   2 - WARN (non-critical violations)
#   3 - ERROR (infrastructure failure)
#   4 - SKIP (no files to scan)

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Default parameters
MODE="${1:-watchdog}"
OUTPUT="${2:-artifacts/drift}"
FAIL_ON_WARN="${FAIL_ON_WARN:-false}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors helper
color_echo() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Header
color_echo "$CYAN" "\n╔════════════════════════════════════════════════════════════╗"
color_echo "$CYAN" "║  Documentation Governance Watchdog - First Light           ║"
color_echo "$CYAN" "║  The organism does not move unless the Cortex approves.    ║"
color_echo "$CYAN" "╚════════════════════════════════════════════════════════════╝\n"

# Ensure output directory exists
OUTPUT_PATH="$PROJECT_ROOT/$OUTPUT"
mkdir -p "$OUTPUT_PATH"

# Check prerequisites
if ! command -v python &> /dev/null; then
    color_echo "$RED" "💥 ERROR: Python is not installed or not in PATH"
    exit 3
fi

RADAR_SCRIPT="$SCRIPT_DIR/radars/doc_governance_drift_radar.py"
if [ ! -f "$RADAR_SCRIPT" ]; then
    color_echo "$RED" "💥 ERROR: Radar script not found: $RADAR_SCRIPT"
    exit 3
fi

# Run radar
echo "Running documentation governance radar..."
color_echo "$GRAY" "  Mode:   $MODE"
color_echo "$GRAY" "  Output: $OUTPUT_PATH\n"

DOCS_PATH="$PROJECT_ROOT/docs"

# Run and capture exit code
set +e
python "$RADAR_SCRIPT" \
    --mode "$MODE" \
    --docs "$DOCS_PATH" \
    --output "$OUTPUT_PATH"
EXIT_CODE=$?
set -e

# Print summary
echo ""
color_echo "$CYAN" "════════════════════════════════════════════════════════════"

case $EXIT_CODE in
    0)
        color_echo "$GREEN" "✅ PASS: No governance violations detected"
        color_echo "$GRAY" "   The organism maintains narrative integrity."
        SUCCESS=true
        ;;
    1)
        color_echo "$RED" "❌ FAIL: Critical governance violations detected"
        color_echo "$RED" "   ⛔ ORGANISM NOT ALIVE"
        color_echo "$GRAY" "   No document may imply 'organism alive' until First Light completes."
        SUCCESS=false
        ;;
    2)
        color_echo "$YELLOW" "⚠️  WARN: Non-critical governance violations detected"
        if [ "$FAIL_ON_WARN" = "true" ]; then
            color_echo "$YELLOW" "   Treating warnings as failures (FAIL_ON_WARN=true)"
            SUCCESS=false
        else
            color_echo "$GRAY" "   Review recommended but not blocking."
            SUCCESS=true
        fi
        ;;
    3)
        color_echo "$RED" "💥 ERROR: Infrastructure failure"
        SUCCESS=false
        ;;
    4)
        color_echo "$GRAY" "⏭️  SKIP: No files to scan"
        SUCCESS=true
        ;;
    *)
        color_echo "$RED" "❓ Unknown exit code: $EXIT_CODE"
        SUCCESS=false
        ;;
esac

color_echo "$CYAN" "════════════════════════════════════════════════════════════\n"

# Show report locations
REPORT_PATH="$OUTPUT_PATH/doc_governance_drift_summary.md"
if [ -f "$REPORT_PATH" ]; then
    echo "📄 Detailed report:"
    color_echo "$GRAY" "   $REPORT_PATH\n"
    
    # Show first few violations if any
    if [ $EXIT_CODE -eq 1 ] || [ $EXIT_CODE -eq 2 ]; then
        color_echo "$YELLOW" "Preview (first 20 lines):"
        head -n 20 "$REPORT_PATH" | while IFS= read -r line; do
            color_echo "$GRAY" "   $line"
        done
        echo ""
    fi
fi

# Exit with appropriate code
if [ "$SUCCESS" = "true" ]; then
    exit 0
else
    if [ "$FAIL_ON_WARN" = "true" ] && [ $EXIT_CODE -eq 2 ]; then
        exit 1
    else
        exit $EXIT_CODE
    fi
fi
