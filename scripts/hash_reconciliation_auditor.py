#!/usr/bin/env python3
# scripts/hash_reconciliation_auditor.py
import argparse
import hashlib
import json
import sys
from collections import OrderedDict, Counter
from pathlib import Path
from datetime import datetime, timezone
import yaml

# --- Inlined Helpers ---
# ... (hashing and loading functions remain the same) ...
def get_canonical_slice_hash(slice_data: dict) -> str:
    canonical_data = slice_data.copy()
    canonical_data.pop("name", None)
    canonical_str = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

def get_file_hash(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()

def hash_ledger(ledger_obj: dict) -> str:
    canonical_data = ledger_obj.copy()
    if 'audit_metadata' in canonical_data:
        canonical_data['audit_metadata'].pop('H_ledger', None)
    canonical_bytes = json.dumps(canonical_data, sort_keys=True, separators=(",", ":")).encode('utf-8')
    return hashlib.sha256(b"ledger:" + canonical_bytes).hexdigest()
    
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    # ...
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(stream, OrderedLoader)

def audit_check(check_id: str, check_type: str, status: bool, details: dict) -> dict:
    if not status and "error_code" not in details: details["error_code"] = "UNKNOWN_ERROR"
    return {"check_id": check_id, "check_type": check_type, "status": "PASSED" if status else "FAILED", "details": details}

def summarize_for_governance(results: list) -> dict:
    # ...
    any_fail = any(r['status'] == 'FAILED' for r in results)
    status = "FAIL" if any_fail else "OK"
    drift_slices = sorted(list({r['details'].get('slice_name', 'global') for r in results if r['status'] == 'FAILED'}))
    return OrderedDict([("status", status), ("any_fail", any_fail), ("drift_slices", drift_slices)])

def hash_formula(formula_string: str) -> str:
    # This is a simplified normalize for testing.
    normalized = "".join(formula_string.split())
    encoded = normalized.encode("ascii")
    return hashlib.sha256(b"formula:" + encoded).hexdigest()


def create_parser() -> argparse.ArgumentParser:
    # ... (same as before)
    parser = argparse.ArgumentParser(description="Stage 2: Hash Reconciliation Auditor.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--output", type=Path, help="Path for the full JSON ledger file.")
    mode_group.add_argument("--integrity-only", action="store_true", help="Emit a small governance summary JSON to stdout.")
    return parser

def main() -> None:
    args = create_parser().parse_args()
    
    try:
        with open(args.config, "r") as f: curriculum_data = ordered_load(f)
        with open(args.prereg, "r") as f: prereg_data = yaml.safe_load(f)
        with open(args.manifest, "r") as f: manifest_data = json.load(f)
    except Exception as e:
        print(f"TOOL_ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    results = []
    config_hash = get_file_hash(args.config)
    manifest_meta = manifest_data.get("metadata", {})
    status = config_hash == manifest_meta.get("curriculum_version_hash")
    results.append(audit_check("MANIFEST_VS_CURRICULUM", "MANIFEST_INTEGRITY", status, {"slice_name": "global", "error_code": "HASH-DRIFT-9" if not status else None}))

    experiments_by_id = {e["experiment_id"]: e for e in prereg_data}
    slices_by_name = {s["name"]: s for s in curriculum_data.get("systems", [{}])[0].get("slices", [])}

    for binding in manifest_data.get("experiment_bindings", []):
        exp_id, slice_name = binding["experiment_id"], binding["slice_name"]
        
        # ... (existing checks for HASH-DRIFT-3, 4, 5, 6) ...

        # Harden with internal slice checks (Level 1 Audit)
        slice_data = slices_by_name.get(slice_name)
        if slice_data:
            pool_hashes = set()
            # Check for HASH-DRIFT-1
            for entry in slice_data.get("formula_pool_entries", []):
                formula, expected_hash = entry.get("formula"), entry.get("hash")
                if expected_hash: pool_hashes.add(expected_hash)
                actual_hash = hash_formula(formula)
                status = actual_hash == expected_hash
                results.append(audit_check(f"FORMULA_HASH_{slice_name}", "SLICE_INTEGRITY", status, {"slice_name": slice_name, "formula": formula, "error_code": "HASH-DRIFT-1" if not status else None}))
            
            # Check for HASH-DRIFT-2
            metric = slice_data.get("success_metric", {})
            for h in metric.get("target_hashes", []):
                status = h in pool_hashes
                results.append(audit_check(f"BINDING_{slice_name}", "SLICE_INTEGRITY", status, {"slice_name": slice_name, "hash": h, "error_code": "HASH-DRIFT-2" if not status else None}))
    
    # Check for HASH-DRIFT-11
    slice_names = [s.get("name") for s in slices_by_name.values()]
    if len(slice_names) != len(set(slice_names)):
        results.append(audit_check("SLICE_NAME_COLLISION", "GLOBAL_INTEGRITY", False, {"slice_name": "global", "error_code": "HASH-DRIFT-11"}))


    # --- Output based on mode ---
    summary = summarize_for_governance(results)
    final_status = summary["status"]
    
    if args.integrity_only:
        print(json.dumps(summary, indent=2))
        sys.exit(1 if final_status == "FAIL" else 0)
    else:
        # ... (full ledger generation) ...
        print(f"Audit complete. Status: {final_status}")
        if final_status == "FAIL": sys.exit(1)

if __name__ == "__main__":
    main()
