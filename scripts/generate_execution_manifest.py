#!/usr/bin/env python3
# scripts/generate_execution_manifest.py
import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from datetime import datetime, timezone
import yaml

# Inlined helpers
def get_canonical_slice_hash(slice_data: dict) -> str:
    canonical_data = slice_data.copy()
    canonical_data.pop("name", None)
    canonical_str = json.dumps(canonical_data, sort_keys=True, separators=( ",", ":"))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

def get_file_hash(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()

def get_deterministic_timestamp(content_hash: str) -> str:
    seed = int(content_hash[:8], 16)
    base_ts = 1704067200 # 2024-01-01
    return datetime.fromtimestamp(base_ts + (seed % 31536000), tz=timezone.utc).isoformat()
    
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(stream, OrderedLoader)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1: Execution Manifest Generator.")
    parser.add_argument("experiment_ids", nargs="+", help="One or more experiment_ids to include.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="execution_manifest.json")
    return parser

def main() -> None:
    args = create_parser().parse_args()
    try:
        with open(args.config, "r") as f: curriculum_data = ordered_load(f)
        with open(args.prereg, "r") as f: prereg_data = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to load input files: {e}", file=sys.stderr)
        sys.exit(1)

    slices_by_name = {s["name"]: s for s in curriculum_data.get("systems", [{}])[0].get("slices", [])}
    experiments_by_id = {e["experiment_id"]: e for e in prereg_data}
    manifest = OrderedDict([("experiment_bindings", [])])
    errors = 0

    for exp_id in args.experiment_ids:
        if exp_id not in experiments_by_id:
            print(f"[FAIL] HASH-DRIFT-5: '{exp_id}' not in {args.prereg}.", file=sys.stderr)
            errors += 1
            continue

        prereg_entry = experiments_by_id[exp_id]
        slice_name = prereg_entry.get("slice_name")
        
        if not slice_name or slice_name not in slices_by_name:
            print(f"[FAIL] HASH-DRIFT-4: Slice '{slice_name}' not in {args.config}.", file=sys.stderr)
            errors += 1
            continue

        expected_hash = prereg_entry.get("slice_config_hash")
        actual_hash = get_canonical_slice_hash(slices_by_name[slice_name])

        if expected_hash != actual_hash:
            print(f"[FAIL] HASH-DRIFT-3: Hash mismatch for '{slice_name}'.", file=sys.stderr)
            errors += 1
            continue
        
        manifest["experiment_bindings"].append(OrderedDict([
            ("experiment_id", exp_id), ("slice_name", slice_name), ("prereg_slice_hash", expected_hash)
        ]))

    if errors > 0:
        print(f"\nFound {errors} errors. Manifest generation aborted.", file=sys.stderr)
        sys.exit(1)

    ts_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["metadata"] = OrderedDict([
        ("manifest_version", "1.1.0"),
        ("curriculum_version_hash", get_file_hash(args.config)),
        ("prereg_version_hash", get_file_hash(args.prereg)),
        ("timestamp_utc", get_deterministic_timestamp(ts_hash)),
    ])
    
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Successfully generated '{args.output}'")

if __name__ == "__main__":
    main()
