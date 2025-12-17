"""
Frozen doc samples for DOC ↔ TEST ↔ IMPL triangle lock tests.

================================================================================
GOVERNANCE: THIS FILE IS THE ONLY SOURCE OF FROZEN SAMPLES
================================================================================

SINGLE SOURCE OF TRUTH:
- These constants are the canonical samples documented in First_Light_External_Verification.md
- Tests assert implementation output matches these samples exactly
- INLINE LITERALS ARE FORBIDDEN in test files; import from here

SHADOW MODE CONTRACT:
- Non-gating: format mismatch is a test failure, not a runtime gate
- Changes require coordinated update (see procedure below)

Doc Section References:
- NCI_P5_BREACH_SAMPLE: §11.10 "Golden Bundle Entry: nci_p5 Signal"
- NOISE_VS_REALITY_MARGINAL_SAMPLE: §11.9 "Golden Bundle Entry: noise_vs_reality Signal"

================================================================================
UPDATE PROCEDURE
================================================================================

WHO: Any developer changing warning format in implementation code.

WHEN: Before merging any PR that modifies:
  - backend/health/nci_governance_adapter.py::build_nci_status_warning()
  - backend/topology/first_light/noise_vs_reality_integration.py::format_advisory_warning()

HOW (3-step coordinated update):
  1. UPDATE IMPLEMENTATION: Change the format function
  2. UPDATE THIS FILE: Change the corresponding *_SAMPLE constant
  3. UPDATE DOCS: Change the sample in First_Light_External_Verification.md

VERIFICATION:
  pytest tests/health/test_nci_ggfl_alignment.py::TestWarningFormatLock
  pytest tests/first_light/test_noise_vs_reality_integration.py::TestAdvisoryWarningFormatLock

If tests fail after implementation change, the triangle is broken.
Fix by updating this file AND the doc section in lockstep.

================================================================================
"""

# =============================================================================
# NCI P5 Warning Samples (§11.10)
# =============================================================================

# Frozen sample: "NCI BREACH: {pct}% consistency (confidence {pct}%)"
# Input: slo_status=BREACH, recommendation=NONE, global_nci=0.72, confidence=0.65
NCI_P5_BREACH_SAMPLE = "NCI BREACH: 72% consistency (confidence 65%)"

# Input parameters that produce NCI_P5_BREACH_SAMPLE
NCI_P5_BREACH_INPUT = {
    "slo_status": "BREACH",
    "recommendation": "NONE",
    "global_nci": 0.72,
    "confidence": 0.65,
}


# =============================================================================
# Noise-vs-Reality Warning Samples (§11.9)
# =============================================================================

# Frozen sample: "VERDICT: factor=value [source_abbrev]"
# Input: verdict=MARGINAL, top_factor=coverage_ratio, top_factor_value=0.72, p5_source=p5_real_validated
NOISE_VS_REALITY_MARGINAL_SAMPLE = "MARGINAL: coverage_ratio=0.72 [real]"

# Input parameters that produce NOISE_VS_REALITY_MARGINAL_SAMPLE
NOISE_VS_REALITY_MARGINAL_INPUT = {
    "verdict": "MARGINAL",
    "top_factor": "coverage_ratio",
    "top_factor_value": 0.72,
    "p5_source": "p5_real_validated",
}
