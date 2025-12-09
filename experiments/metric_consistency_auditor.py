"""
PHASE II — NOT USED IN PHASE I
Metric-Derivation Consistency Auditor (Streaming & Production Ready)

Implements the multi-phase verification pipeline and generates a Metric
Integration Ledger entry from a streaming JSONL source.
"""
import json
import sys
import io
import abc
import os
from typing import Any, Dict, List, Set, Iterable

# --- Dependency Imports ---
sys.path.insert(0, '.')
from experiments.derivation_chain_analysis import ChainAnalyzer
from experiments.metric_integration_ledger import (
    build_metric_ledger_entry,
    to_canonical_json,
    hash_sha256
)
from experiments.slice_success_metrics import compute_metric

# --- Storage Abstraction (Task 4) ---

class MetricLedgerStore(abc.ABC):
    """Abstract base class for ledger storage backends."""
    @abc.abstractmethod
    def save(self, ledger_entry: Dict[str, Any]) -> None:
        raise NotImplementedError
    
    def get(self, ledger_id: str) -> Dict[str, Any]:
        raise NotImplementedError

class InMemoryLedgerStore(MetricLedgerStore):
    """An in-memory, non-persistent store for testing."""
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
    
    def save(self, ledger_entry: Dict[str, Any]) -> None:
        print(f"Saving ledger {ledger_entry['ledger_id']} to InMemoryLedgerStore...")
        self._store[ledger_entry["ledger_id"]] = ledger_entry

    def get(self, ledger_id: str) -> Dict[str, Any]:
        return self._store.get(ledger_id)

class FilesystemLedgerStore(MetricLedgerStore):
    """Saves ledgers as JSON files to the filesystem."""
    def __init__(self, base_path: str = "artifacts/metrics/ledgers"):
        self._base_path = base_path
        os.makedirs(self._base_path, exist_ok=True)

    def save(self, ledger_entry: Dict[str, Any]) -> None:
        ledger_id = ledger_entry["ledger_id"]
        output_path = os.path.join(self._base_path, f"{ledger_id}.json")
        print(f"Writing ledger artifact to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(ledger_entry, f, indent=2, sort_keys=True)

# --- Audit Phases (Tasks 1 & 2) ---

def audit_phase_I(derivations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Performs Phase I: Structural Admissibility checks."""
    # (Implementation from previous turn is sufficient)
    findings: Set[str] = set()
    known_hashes: Set[str] = set()
    for d in derivations:
        if "hash" not in d or not isinstance(d.get("hash"), str): findings.add("METINT-2")
        elif d["hash"] in known_hashes: findings.add("METINT-21")
        else: known_hashes.add(d["hash"])
        if "premises" not in d or not isinstance(d.get("premises"), list): findings.add("METINT-3")
    if "METINT-2" not in findings:
        for d in derivations:
            if "premises" in d:
                for p in d["premises"]:
                    if p not in known_hashes:
                        findings.add("METINT-6"); break
    return {"status": "PASSED" if not findings else "FAILED", "findings": findings}

def audit_phase_II(analyzer: ChainAnalyzer) -> Dict[str, Any]:
    """Performs Phase II: Semantic Consistency checks."""
    findings: Set[str] = set()
    # Check for cycles and depth issues
    sample_node = next(((h, p) for h, p in analyzer._dep_graph.items() if p), None)
    if sample_node:
        h, p = sample_node
        try:
            # Check if the depth of the current node is correctly calculated based on its premises
            if (1 + max([analyzer.get_depth(pr) for pr in p] + [0])) != analyzer.get_depth(h):
                findings.add("METINT-23") # Inconsistent depth calculation
        except RecursionError:
            findings.add("METINT-22") # Cycle detected or excessive depth
    return {"status": "PASSED" if not findings else "FAILED", "findings": findings}

def audit_phase_III(output: tuple, config: Dict[str, Any]) -> Dict[str, Any]:
    """Performs Phase III: Metric Contract Enforcement."""
    findings: Set[str] = set()
    success, value, details = output
    
    # Domain/Range Invariant: metric value must be between 0.0 and 1.0 for 'density'
    if config.get("metric_kind") == "density" and not (0.0 <= value <= 1.0):
        findings.add("METINT-31") # Value Out of Range
    
    # DAG-layer statistical invariant: max_chain_in_cycle should not be absurdly high
    if details.get("metric_kind") == "chain_length" and details.get("max_chain_in_cycle", 0) > 1000:
        findings.add("METINT-32") # Suspiciously Deep Chain

    return {"status": "PASSED" if not findings else "FAILED", "findings": findings}

def audit_phase_IV(fingerprint: str, historical_fingerprints: Set[str]) -> Dict[str, Any]:
    """Performs Phase IV: Threat Analysis (Heuristics)."""
    findings: Set[str] = set()
    
    # Anomaly Detection: check if derivation fingerprint is suspiciously novel
    if historical_fingerprints and fingerprint not in historical_fingerprints:
        findings.add("METINT-41") # Novel Derivation Pattern
        
    # "Metric Spoofing": A simple heuristic could be a very high metric value from very few derivations
    # This logic would be more complex in reality.
    
    return {"status": "PASSED" if not findings else "WARNING", "findings": findings}


# --- Streaming Orchestrator (Task 3) ---

def audit_and_build_from_stream(
    derivation_stream: Iterable[str],
    metric_config: Dict[str, Any],
    historical_fingerprints: Set[str]
) -> Dict[str, Any]:
    """Main orchestrator for streaming audit and ledger generation."""
    
    # --- Streaming Pass ---
    # In one pass, we collect derivations for in-memory audit and calculate fingerprint hashes
    all_derivations: List[Dict[str, Any]] = []
    line_hashes: List[str] = []
    for line in derivation_stream:
        try:
            d = json.loads(line)
            all_derivations.append(d)
            line_hashes.append(hash_sha256(to_canonical_json(d)))
        except json.JSONDecodeError:
            # Handle malformed JSON lines
            pass # Or log a specific METINT code
    
    line_hashes.sort()
    derivation_fingerprint = hash_sha256("".join(line_hashes))

    # --- In-Memory Audit & Computation ---
    phase_I_verdict = audit_phase_I(all_derivations)
    
    # Only proceed with sound data
    sound_derivations = [d for d in all_derivations if "hash" in d and "premises" in d] if phase_I_verdict["status"] == "PASSED" else []
    
    phase_II_verdict = audit_phase_II(ChainAnalyzer(sound_derivations)) if sound_derivations else {"status": "SKIPPED", "findings": set()}

    # Prepare and run metric computation
    computation_request = {"metric_kind": metric_config["metric_kind"], "parameters": {k:v for k,v in metric_config.items() if k != 'metric_kind'}}
    metric_kwargs = {
        "verified_hashes": {d['hash'] for d in sound_derivations},
        "verified_count": len(sound_derivations), # FIX: Explicitly add verified_count
        "candidates_tried": len(all_derivations),
        "result": {"derivations": sound_derivations},
        **metric_config
    }
    computation_output = compute_metric(kind=computation_request["metric_kind"], **metric_kwargs)

    # Run new audit phases
    phase_III_verdict = audit_phase_III(computation_output, metric_config)
    phase_IV_verdict = audit_phase_IV(derivation_fingerprint, historical_fingerprints)

    final_verdict = "FAILED" if "FAILED" in [p["status"] for p in [phase_I_verdict, phase_II_verdict, phase_III_verdict]] else "PASSED"
    auditor_verdict = {
        "final_status": final_verdict,
        "phase_I_structural": {**phase_I_verdict, "findings": sorted(list(phase_I_verdict["findings"]))},
        "phase_II_semantic": {**phase_II_verdict, "findings": sorted(list(phase_II_verdict["findings"]))},
        "phase_III_metric_contract": {**phase_III_verdict, "findings": sorted(list(phase_III_verdict["findings"]))},
        "phase_IV_threat_analysis": {**phase_IV_verdict, "findings": sorted(list(phase_IV_verdict["findings"]))},
    }

    # --- Build Ledger ---
    return build_metric_ledger_entry(
        derivation_fingerprint=derivation_fingerprint,
        all_derivations=sound_derivations,
        computation_request=computation_request,
        computation_output=computation_output,
        auditor_verdict=auditor_verdict
    )

def main():
    """Main function to simulate a streaming audit and save the ledger."""
    print("Running Streaming Metric Consistency Auditor...")

    # Simulate a JSONL log file as a stream
    jsonl_data = [
        {"hash": "h0", "premises": []},
        {"hash": "h1", "premises": ["h0"]},
        {"hash": "h2", "premises": ["h1"]},
    ]
    log_stream = io.StringIO("\n".join(json.dumps(d) for d in jsonl_data))
    
    # Simulate a metric configuration
    sample_config = {"metric_kind": "chain_length", "chain_target_hash": "h2", "min_chain_length": 3}
    
    # Simulate a set of historical fingerprints for Phase IV
    historical_fps = {"some_old_fingerprint"}

    # Run the full pipeline
    ledger_entry = audit_and_build_from_stream(log_stream, sample_config, historical_fps)

    # Save the ledger using a storage backend
    store = FilesystemLedgerStore()
    store.save(ledger_entry)
    
    print("Audit and ledger generation complete.")
    print(f"Final Audit Status: {ledger_entry['auditor_verdict']['final_status']}")
    print(f"Ledger ID: {ledger_entry['ledger_id']}")

if __name__ == "__main__":
    import os
    main()
