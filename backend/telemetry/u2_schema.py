# PHASE II — U2 UPLIFT EXPERIMENT
# backend/telemetry/u2_schema.py
# Pure Python validation for U2 Telemetry Schemas

import re
from typing import Dict, Any, List, Set

# --- Schema Definitions (as Python constants) ---

# Per-Cycle Event Schema (`u2-cycle-v1`)
CYCLE_EVENT_REQUIRED_KEYS: Set[str] = {
    "ts", "run_id", "slice", "mode", "cycle", "metric_type", "metric_value", "H_t"
}
CYCLE_EVENT_OPTIONAL_KEYS: Set[str] = {"success", "R_t", "U_t", "is_l2_event"}
VALID_SLICES: Set[str] = {"U2_env_A", "U2_env_B", "U2_env_C", "U2_env_D"}
VALID_MODES: Set[str] = {"baseline", "rfl"}
HEX_64_CHAR_PATTERN = re.compile(r"^[a-f0-9]{64}$")

# Experiment Summary Schema (`u2-summary-v1`)
SUMMARY_REQUIRED_KEYS: Set[str] = {
    "slice", "mode", "n_cycles", "p_base", "p_rfl", "delta", "CI", 
    "baseline_run_id", "rfl_run_id"
}

def _validate_keys(data: Dict[str, Any], required: Set[str], optional: Set[str] = None):
    """Checks for required keys and unknown keys."""
    optional = optional or set()
    keys = set(data.keys())
    
    missing_keys = required - keys
    if missing_keys:
        raise ValueError(f"Missing required keys: {sorted(list(missing_keys))}")

    unknown_keys = keys - required - optional
    if unknown_keys:
        raise ValueError(f"Unknown keys found: {sorted(list(unknown_keys))}")

def validate_cycle_event(event: Dict[str, Any]):
    """
    Validates a U2 per-cycle metric event against the `u2-cycle-v1` schema.
    Raises ValueError or TypeError on validation failure.
    
    :param event: A dictionary representing the metric event.
    """
    if not isinstance(event, dict):
        raise TypeError("Event must be a dictionary.")

    _validate_keys(event, CYCLE_EVENT_REQUIRED_KEYS, CYCLE_EVENT_OPTIONAL_KEYS)

    # Type and value validation for required fields
    if not isinstance(event['ts'], str): # Simplified ISO check
        raise TypeError(f"ts must be a string, got {type(event['ts'])}")
    if not isinstance(event['run_id'], str): # Simplified UUID check
        raise TypeError(f"run_id must be a string, got {type(event['run_id'])}")
    if not isinstance(event.get('slice'), str):
        raise TypeError(f"slice must be a string, got {type(event['slice'])}")
    if not isinstance(event.get('mode'), str):
        raise TypeError(f"mode must be a string, got {type(event['mode'])}")
    if not isinstance(event.get('cycle'), int) or event.get('cycle', -1) < 0:
        raise TypeError(f"cycle must be a non-negative integer, got {event.get('cycle')}")
    if not isinstance(event.get('metric_type'), str):
        raise TypeError(f"metric_type must be a string, got {type(event['metric_type'])}")
    if not isinstance(event.get('metric_value'), (int, float)):
        raise TypeError(f"metric_value must be a number, got {type(event['metric_value'])}")
    if not (isinstance(event.get('H_t'), str) and HEX_64_CHAR_PATTERN.match(event['H_t'])):
         raise ValueError("H_t must be a 64-character hex string.")

    # Type and value validation for optional fields
    if 'success' in event and not isinstance(event['success'], bool):
        raise TypeError(f"success must be a boolean, got {type(event['success'])}")
    if 'R_t' in event and not (isinstance(event['R_t'], str) and HEX_64_CHAR_PATTERN.match(event['R_t'])):
        raise ValueError("R_t must be a 64-character hex string.")
    if 'U_t' in event and not (isinstance(event['U_t'], str) and HEX_64_CHAR_PATTERN.match(event['U_t'])):
        raise ValueError("U_t must be a 64-character hex string.")

def validate_experiment_summary(summary: Dict[str, Any]):
    """
    Validates a U2 experiment summary against the `u2-summary-v1` schema.
    Raises ValueError or TypeError on validation failure.
    
    :param summary: A dictionary representing the experiment summary.
    """
    if not isinstance(summary, dict):
        raise TypeError("Summary must be a dictionary.")

    _validate_keys(summary, SUMMARY_REQUIRED_KEYS)

    # --- Type and value validation ---
    if not isinstance(summary.get('slice'), str):
        raise TypeError(f"slice must be a string, got {type(summary['slice'])}")
    if not isinstance(summary.get('mode'), str):
        raise TypeError(f"mode must be a string, got {type(summary['mode'])}")
    
    # n_cycles validation
    n_cycles = summary.get('n_cycles')
    if not isinstance(n_cycles, dict) or 'baseline' not in n_cycles or 'rfl' not in n_cycles:
        raise TypeError("n_cycles must be a dict with 'baseline' and 'rfl' keys.")
    if not isinstance(n_cycles['baseline'], int) or not isinstance(n_cycles['rfl'], int):
        raise TypeError("n_cycles values must be integers.")

    # Probability validation
    for key in ['p_base', 'p_rfl']:
        val = summary.get(key)
        if not isinstance(val, (int, float)):
            raise TypeError(f"{key} must be a number, got {type(val)}")
    
    # Delta validation
    delta = summary.get('delta')
    if not isinstance(delta, (int, float)):
        raise TypeError(f"delta must be a number, got {type(delta)}")

    # CI validation
    ci = summary.get('CI')
    if not isinstance(ci, dict) or 'lower_bound' not in ci or 'upper_bound' not in ci:
        raise TypeError("CI must be a dict with 'lower_bound' and 'upper_bound' keys.")
    if not isinstance(ci['lower_bound'], (int, float)) or not isinstance(ci['upper_bound'], (int, float)):
        raise TypeError("CI bounds must be numbers.")

    # ID validation
    if not isinstance(summary.get('baseline_run_id'), str):
        raise TypeError("baseline_run_id must be a string.")
    if not isinstance(summary.get('rfl_run_id'), str):
        raise TypeError("rfl_run_id must be a string.")
