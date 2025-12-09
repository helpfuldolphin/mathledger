# ==============================================================================
# PHASE II — SYNTHETIC TEST DATA ONLY
# ==============================================================================
#
# This module provides synthetic noise generation for stress-testing the
# U2 uplift analysis infrastructure.
#
# NOT derived from real derivations; NOT part of Evidence Pack.
# Must NOT generate or simulate uplift conclusions.
#
# Contents:
#   - noise_models.py: Drift and correlation noise models
#   - generate_synthetic_logs.py: Basic deterministic JSONL log generator
#   - generate_synthetic_logs_v2.py: Enhanced generator with drift/correlation
#   - scenario_suite.py: 12 ready-made synthetic scenarios
#   - synthetic_slices.yaml: Legacy slice definitions
#   - tests/: Validation tests (20 original + 18 new = 38 tests)
#   - generated/: Isolated output directory for synthetic data
#
# Usage:
#   # Basic generation
#   from experiments.synthetic_uplift.generate_synthetic_logs import (
#       generate_synthetic_logs,
#       load_synthetic_config,
#   )
#
#   # Enhanced generation with noise
#   from experiments.synthetic_uplift.generate_synthetic_logs_v2 import (
#       generate_synthetic_logs_v2,
#   )
#   from experiments.synthetic_uplift.scenario_suite import (
#       load_scenario,
#       list_scenarios,
#   )
#
# ==============================================================================

SAFETY_LABEL = "PHASE II — SYNTHETIC TEST DATA ONLY"

__all__ = ["SAFETY_LABEL"]

