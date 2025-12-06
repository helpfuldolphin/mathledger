"""
Evidence Chain Ledger for Phase III
====================================

Provides evidence chain ledger construction, CI gate evaluation, and
Markdown rendering for multi-experiment attestation audits.

The ledger serves as a first-class evidence pack that ties together
multiple experiments with a single cryptographic fingerprint.
"""

import json
from typing import Any, Dict, List, Sequence

from attestation.manifest_verifier import compute_sha256_string


def build_evidence_chain_ledger(audit_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build an evidence chain ledger from audit results.
    
    The ledger provides a canonical, cryptographically bound record of all
    audited experiments with a single hash fingerprint.
    
    Args:
        audit_results: Sequence of audit result dictionaries, each containing:
            - experiment_id: str
            - status: "PASS" | "FAIL" | "SKIP"
            - manifest_path: str
            - manifest_hash: str or None
            - artifacts: list of {path: str, hash: str or None}
            
    Returns:
        Dictionary with ledger structure:
        - schema_version: str
        - experiment_count: int
        - experiments: list of experiment records
        - global_status: "PASS" | "PARTIAL" | "FAIL"
        - ledger_hash: str (SHA-256 over canonical ledger body)
    """
    # Build experiment records
    experiments = []
    for result in audit_results:
        # Collect artifact hashes
        artifact_hashes = {}
        for artifact in result.get("artifacts", []):
            path = artifact.get("path")
            hash_value = artifact.get("hash")
            if path and hash_value:
                artifact_hashes[path] = hash_value
        
        # Build experiment record
        experiment = {
            "id": result["experiment_id"],
            "status": result["status"],
            "artifact_hashes": artifact_hashes,
            "report_path": result.get("manifest_path", "")
        }
        experiments.append(experiment)
    
    # Determine global status
    statuses = [exp["status"] for exp in experiments]
    if all(s == "PASS" for s in statuses):
        global_status = "PASS"
    elif any(s == "FAIL" for s in statuses):
        global_status = "FAIL"
    else:
        global_status = "PARTIAL"
    
    # Build ledger body (without hash)
    ledger_body = {
        "schema_version": "1.0",
        "experiment_count": len(experiments),
        "experiments": experiments,
        "global_status": global_status
    }
    
    # Compute ledger hash over canonical JSON
    canonical_json = json.dumps(ledger_body, sort_keys=True, separators=(',', ':'))
    ledger_hash = compute_sha256_string(canonical_json)
    
    # Build final ledger with hash
    ledger = {
        **ledger_body,
        "ledger_hash": ledger_hash
    }
    
    return ledger


def evaluate_evidence_chain_for_ci(ledger: Dict[str, Any]) -> int:
    """
    Evaluate evidence chain ledger for CI hard gate.
    
    This function provides deterministic exit codes for CI/CD pipelines
    based on the global status of the evidence chain.
    
    Args:
        ledger: Evidence chain ledger dictionary
        
    Returns:
        Exit code:
        - 0 if global_status == "PASS"
        - 1 if global_status == "PARTIAL"
        - 2 if global_status == "FAIL"
    """
    global_status = ledger.get("global_status", "FAIL")
    
    if global_status == "PASS":
        return 0
    elif global_status == "PARTIAL":
        return 1
    else:  # "FAIL" or any unexpected value
        return 2


def render_evidence_chain_section(ledger: Dict[str, Any]) -> str:
    """
    Render evidence chain section as Markdown.
    
    Generates a neutral, factual table showing experiment status and hashes,
    along with a description of the ledger structure.
    
    Args:
        ledger: Evidence chain ledger dictionary
        
    Returns:
        Markdown string with evidence chain table and description
    """
    lines = [
        "## Evidence Chain",
        "",
        "The following table lists all experiments in the evidence chain.",
        "All hashes are SHA-256. The ledger_hash can be used as a single",
        "attestation fingerprint for this entire evidence pack.",
        "",
        "| Experiment ID | Status | Manifest Hash | Evidence Hash |",
        "|---------------|--------|---------------|---------------|",
    ]
    
    for experiment in ledger.get("experiments", []):
        exp_id = experiment.get("id", "")
        status = experiment.get("status", "UNKNOWN")
        
        # Get manifest hash (first 8 chars)
        manifest_path = experiment.get("report_path", "")
        artifact_hashes = experiment.get("artifact_hashes", {})
        
        # Find manifest hash from artifact_hashes or use placeholder
        manifest_hash = ""
        if manifest_path in artifact_hashes:
            manifest_hash = artifact_hashes[manifest_path][:8] + "..."
        else:
            manifest_hash = "N/A"
        
        # Compute evidence hash (hash of all artifact hashes for this experiment)
        if artifact_hashes:
            # Sort paths for determinism
            sorted_hashes = [artifact_hashes[k] for k in sorted(artifact_hashes.keys())]
            combined = "".join(sorted_hashes)
            evidence_hash = compute_sha256_string(combined)[:8] + "..."
        else:
            evidence_hash = "N/A"
        
        # Status icon
        status_icon = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "—")
        
        lines.append(
            f"| `{exp_id}` | {status_icon} {status} | `{manifest_hash}` | `{evidence_hash}` |"
        )
    
    lines.extend([
        "",
        f"**Ledger Hash:** `{ledger.get('ledger_hash', 'N/A')}`",
        "",
        "This ledger hash serves as a cryptographic fingerprint of the entire evidence chain.",
        "It is computed as SHA-256 over the canonical JSON representation of the ledger body",
        "(experiments, statuses, and artifact hashes in sorted order).",
    ])
    
    return "\n".join(lines)
