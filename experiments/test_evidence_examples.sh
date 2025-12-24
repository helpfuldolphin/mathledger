#!/bin/bash
# Test script for evidence fusion examples
# This script demonstrates the full workflow with all three scenarios

set -e

echo "=========================================="
echo "TDA-Aware Evidence Fusion - Examples"
echo "=========================================="
echo

# Create output directory
mkdir -p /tmp/evidence_examples

echo "1. Testing OK scenario (all aligned)..."
python3 experiments/evidence_fusion.py \
  experiments/sample_evidence_data.json \
  /tmp/evidence_examples/fused_ok.json \
  --hss-threshold 0.7
echo
echo "Running precheck..."
python3 experiments/promotion_precheck.py /tmp/evidence_examples/fused_ok.json
echo
echo "✓ OK scenario complete"
echo
echo "=========================================="
echo

echo "2. Testing WARN scenario (hidden instability)..."
python3 experiments/evidence_fusion.py \
  experiments/sample_evidence_warn.json \
  /tmp/evidence_examples/fused_warn.json \
  --hss-threshold 0.7
echo
echo "Running precheck..."
python3 experiments/promotion_precheck.py /tmp/evidence_examples/fused_warn.json 2>&1 | head -40
echo
echo "✓ WARN scenario complete"
echo
echo "=========================================="
echo

echo "3. Testing BLOCK scenario (uplift/TDA conflict)..."
python3 experiments/evidence_fusion.py \
  experiments/sample_evidence_block.json \
  /tmp/evidence_examples/fused_block.json \
  --hss-threshold 0.7
echo
echo "Running precheck..."
python3 experiments/promotion_precheck.py /tmp/evidence_examples/fused_block.json 2>&1 | head -40 || echo "(Expected exit code 1)"
echo
echo "✓ BLOCK scenario complete"
echo
echo "=========================================="
echo "All examples complete!"
echo "Output files saved to /tmp/evidence_examples/"
echo "=========================================="
