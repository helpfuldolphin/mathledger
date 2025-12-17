import json
import copy
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union

try:
    import jsonschema
except ImportError:
    jsonschema = None

from . import nl_ir_failure_codes as codes

# --- Constants ---
SCHEMA_FILE_PATH = Path(__file__).parent.parent.parent / "schemas" / "nl_ir.schema.json"
NL_IR_SCHEMA_VERSION = "1.0.0" # To be updated when schema changes

# Mock Ontology for validation purposes. In a real system, this would be a
# managed, external resource.
KNOWN_ONTOLOGY_TYPES = {
    "animal:Canidae",
    "Person",
    "SystemAction.AccountFreeze"
}

# --- Cached Schema ---
_schema = None

def _load_schema() -> Dict[str, Any]:
    """Loads the IR schema from file, caching it for subsequent calls."""
    global _schema
    if _schema is None:
        if not SCHEMA_FILE_PATH.exists():
            raise FileNotFoundError(f"Schema file not found at {SCHEMA_FILE_PATH}")
        with open(SCHEMA_FILE_PATH, 'r') as f:
            _schema = json.load(f)
    return _schema

def _validate_semantics(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Performs semantic checks on a structurally valid IR payload."""
    errors = []
    
    if payload.get("disambiguation_notes"):
        errors.append({
            "code": codes.AMBIGUITY_DETECTED,
            "message": "The NLU processor flagged ambiguities that require resolution.",
            "path": "disambiguation_notes"
        })

    grounding_context = payload.get("grounding_context", {})
    for entity_id, context in grounding_context.items():
        schema_ref = context.get("schema_ref")
        if schema_ref and schema_ref not in KNOWN_ONTOLOGY_TYPES:
            errors.append({
                "code": codes.ONTOLOGY_MISMATCH,
                "message": f"Schema reference '{schema_ref}' not found in known ontology.",
                "path": f"grounding_context.{entity_id}.schema_ref"
            })
            
    return errors

def validate_ir(payload: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Union[str, int]]]]:
    """
    Validates a dictionary payload against the nl_ir.schema.json and performs
    additional semantic checks.
    """
    if jsonschema is None:
        return (False, [{"code": codes.DEPENDENCY_MISSING, "message": "jsonschema library is not installed.", "path": None}])
        
    all_errors = []
    
    try:
        schema = _load_schema()
        validator = jsonschema.Draft7Validator(schema)
        schema_errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
        
        for error in schema_errors:
            path = ".".join(map(str, error.path)) if error.path else "root"
            error_code = f"SCHEMA_{error.validator.upper()}_FAILED"
            all_errors.append({"code": error_code, "message": error.message, "path": path})

    except FileNotFoundError as e:
        all_errors.append({"code": codes.SCHEMA_NOT_FOUND, "message": str(e), "path": None})
        return False, all_errors
    except Exception as e:
        all_errors.append({"code": codes.UNEXPECTED_VALIDATION_ERROR, "message": str(e), "path": None})
        return False, all_errors

    if not all_errors:
        all_errors.extend(_validate_semantics(payload))

    return not all_errors, all_errors

def attach_nl_evidence(
    evidence_pack: Dict[str, Any], 
    nl_ir_payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Produces a new evidence pack with an NL IR governance block and data attachment.
    This function is non-mutating and returns a deep copy of the evidence pack.
    """
    # 1. Non-mutation: Create a deep copy of the original pack
    new_pack = copy.deepcopy(evidence_pack)

    # 2. Validate the IR payload
    is_valid, findings = validate_ir(nl_ir_payload)

    # 3. Create a deterministic hash of the payload
    payload_bytes = json.dumps(nl_ir_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()

    # 4. Ensure governance and data keys exist
    if 'governance' not in new_pack:
        new_pack['governance'] = {}
    if 'data' not in new_pack:
        new_pack['data'] = {}
        
    # 5. Attach the raw payload to the data section
    new_pack['data']['nl_ir'] = {'payload': nl_ir_payload}

    # 6. Attach the governance block
    new_pack['governance']['nl_ir'] = {
        'schema_version': NL_IR_SCHEMA_VERSION,
        'payload_hash': payload_hash,
        'findings': findings
    }
    
    return new_pack
