# backend/governance_signal/canonical.py
import json
from typing import Dict, Any

def to_canonical_json(signal_data: Dict[str, Any]) -> str:
    """
    Converts a signal dictionary to a canonical JSON string.

    The canonical form is defined as:
    - JSON format.
    - Keys sorted alphabetically.
    - No insignificant whitespace.
    - UTF-8 encoding.

    This ensures that the same logical signal always produces the exact same
    byte string for signing.
    """
    # Create a deep copy to avoid modifying the original dict
    data_copy = json.loads(json.dumps(signal_data))

    # The signature must not be part of the payload that is signed
    if "cryptographicMetadata" in data_copy and "signature" in data_copy["cryptographicMetadata"]:
        del data_copy["cryptographicMetadata"]["signature"]

    return json.dumps(data_copy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
