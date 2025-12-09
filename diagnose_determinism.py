"""
This is a diagnostic script to isolate the non-determinism issue.
"""
import json
import sys
import copy

sys.path.insert(0, '.')
from experiments.metric_consistency_auditor import run_full_audit_and_build_ledger

def diagnose():
    print("--- Running Diagnostic ---")
    
    sample_derivations = [
        {"hash": "h0", "premises": []},
        {"hash": "h2", "premises": ["h0", "h1"]},
        {"hash": "h1", "premises": ["dangling_premise"]}, 
    ]
    
    sample_config = {
        "metric_kind": "chain_length",
        "chain_target_hash": "h2",
        "min_chain_length": 3,
    }

    # Run 1
    ledger1 = run_full_audit_and_build_ledger(
        copy.deepcopy(sample_derivations), 
        copy.deepcopy(sample_config)
    )
    
    # Run 2
    ledger2 = run_full_audit_and_build_ledger(
        copy.deepcopy(sample_derivations), 
        copy.deepcopy(sample_config)
    )

    # Remove timestamps for comparison
    ledger1.pop("timestamp_utc")
    ledger2.pop("timestamp_utc")

    # Get IDs
    id1 = ledger1.pop("ledger_id")
    id2 = ledger2.pop("ledger_id")

    # Serialize to compare
    j1 = json.dumps(ledger1, sort_keys=True, indent=2)
    j2 = json.dumps(ledger2, sort_keys=True, indent=2)
    
    if j1 == j2:
        print("JSON objects are identical.")
    else:
        print("JSON objects are DIFFERENT.")
        # Find the difference
        import difflib
        diff = difflib.unified_diff(j1.splitlines(keepends=True), j2.splitlines(keepends=True), fromfile='run1.json', tofile='run2.json')
        print("".join(diff))

    if id1 == id2:
        print(f"Ledger IDs are identical: {id1}")
    else:
        print(f"Ledger IDs are DIFFERENT: {id1} vs {id2}")

if __name__ == "__main__":
    diagnose()
