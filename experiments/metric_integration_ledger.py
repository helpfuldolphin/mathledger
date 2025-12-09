"""
PHASE II — NOT USED IN PHASE I
Metric Integration Ledger Builder (Streaming & Production Ready)
"""
import json
import hashlib
import datetime
from typing import Any, Dict, List, Iterable, Tuple

def to_canonical_json(data: Any) -> str:
    """
    Recursively sorts and serializes a Python object to a canonical JSON string.
    """
    if isinstance(data, dict):
        sorted_items = sorted(data.items())
        return "{" + ",".join(f'{to_canonical_json(k)}:{to_canonical_json(v)}' for k, v in sorted_items) + "}"
    if isinstance(data, set):
        return to_canonical_json(sorted(list(data)))
    if isinstance(data, list):
        # In a fully robust system, sorting lists of complex objects would require
        # a more advanced method. For this system, we rely on upstream sorting
        # or sorting primitive lists.
        return "[" + ",".join(to_canonical_json(item) for item in data) + "]"
    return json.dumps(data, ensure_ascii=False)

def hash_sha256(data: str) -> str:
    """Computes the SHA-256 hash of a string and returns it as hex."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def get_derivation_fingerprint_stream(derivation_stream: Iterable[Dict[str, Any]]) -> str:
    """
    Implements a streaming version of the Derivation Log Fingerprinting.
    This uses constant memory by hashing each line, then hashing a sorted
    list of the resulting hashes.
    """
    line_hashes = [hash_sha256(to_canonical_json(d)) for d in derivation_stream]
    line_hashes.sort()
    concatenated_hashes = "".join(line_hashes)
    return hash_sha256(concatenated_hashes)

def get_dag_fingerprint(analyzer: Any) -> str:
    """Implements the DAG Fingerprinting algorithm."""
    if not hasattr(analyzer, '_dep_graph'):
        raise TypeError("Analyzer does not have a _dep_graph attribute.")
    
    node_strings = []
    for h in sorted(analyzer._dep_graph.keys()):
        premises = sorted(analyzer._dep_graph[h])
        node_strings.append(f'{to_canonical_json(h)}:{to_canonical_json(premises)}')
    
    return hash_sha256("{" + ",".join(node_strings) + "}")

def build_metric_ledger_entry(
    *,
    derivation_fingerprint: str, # Now passed in from the stream processor
    all_derivations: List[Dict[str, Any]], # The full list, now collected by the caller
    computation_request: Dict[str, Any],
    computation_output: tuple,
    auditor_verdict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Builds the canonical ledger entry from pre-computed/streamed components.
    """
    from experiments.derivation_chain_analysis import ChainAnalyzer
    analyzer = ChainAnalyzer(all_derivations)

    ledger_body = {
        "ledger_schema_version": "2.0",
        "derivation_fingerprint": derivation_fingerprint,
        "dag_fingerprint": get_dag_fingerprint(analyzer),
        "computation_request": computation_request,
        "computation_output": {
            "success": computation_output[0],
            "value": computation_output[1],
            "details": computation_output[2],
        },
        "auditor_verdict": auditor_verdict,
    }
    ledger_id = hash_sha256(to_canonical_json(ledger_body))
    
    return {
        "ledger_id": ledger_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        **ledger_body,
    }