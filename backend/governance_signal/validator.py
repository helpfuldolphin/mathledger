# backend/governance_signal/validator.py
import json
import os
from typing import Dict, Any, Tuple

try:
    import jsonschema
except ImportError:
    # This allows the module to be imported even if jsonschema is not installed,
    # though the validation function will fail.
    jsonschema = None

SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "schema.json")

def _load_schema() -> Dict[str, Any]:
    """Loads the Governance Signal JSON schema from file."""
    with open(SCHEMA_FILE, "r") as f:
        return json.load(f)

def validate_signal(signal_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates a signal dictionary against the Governance Signal JSON schema.

    Returns a tuple of (is_valid, message).
    """
    if jsonschema is None:
        return False, "jsonschema library is not installed. Please `pip install jsonschema`."

    try:
        schema = _load_schema()
        jsonschema.validate(instance=signal_data, schema=schema)
        return True, "Signal is valid."
    except jsonschema.exceptions.ValidationError as e:
        return False, f"Signal is invalid: {e.message}"
    except FileNotFoundError:
        return False, f"Schema file not found at {SCHEMA_FILE}"
    except Exception as e:
        return False, f"An unexpected error occurred during validation: {e}"
