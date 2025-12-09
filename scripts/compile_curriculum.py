#!/usr/bin/env python3
# scripts/compile_curriculum.py
import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
import yaml
from yaml.constructor import ConstructorError
import re

# Inlined helpers
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        def represent_dict_order(self, data):
            return self.represent_mapping('tag:yaml.org,2002:map', data.items())
    OrderedDumper.add_representer(OrderedDict, OrderedDumper.represent_dict_order)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

DOMAIN_STMT = b"formula:"
def hash_formula(formula_string: str) -> str:
    # A real implementation would import the canonical normalizer
    normalized = "".join(formula_string.split())
    encoded = normalized.encode("ascii")
    return hashlib.sha256(DOMAIN_STMT + encoded).hexdigest()

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 0: Canonical Curriculum Compiler.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser

def main() -> None:
    args = create_parser().parse_args()
    try:
        source_text = args.source.read_text()
        data = ordered_load(source_text)
    except Exception as e:
        print(f"ERROR: Loading source curriculum failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Check for duplicate slice names
    slice_names = list(data.get("slices", {}).keys())
    if len(slice_names) != len(set(slice_names)):
        print("FATAL: HASH-DRIFT-11 - Duplicate slice names detected in source.", file=sys.stderr)
        sys.exit(1)

    new_data = OrderedDict()
    new_data["version"] = str(data.get("version", "unknown")) + "-hashed"
    new_data["reproducibility_signature"] = hashlib.sha256(source_text.encode()).hexdigest()
    new_data["systems"] = [OrderedDict([("name", "pl"), ("slices", [])])]
    slices_list = new_data["systems"][0]["slices"]
    
    for slice_name, slice_content in data.get("slices", {}).items():
        new_slice = OrderedDict([("name", slice_name)])
        new_slice.update(slice_content)
        
        formula_pool = new_slice.get("formula_pool_entries", [])
        if len(formula_pool) != len(set(formula_pool)):
            print(f"FATAL: HASH-DRIFT-10 in slice '{slice_name}'.", file=sys.stderr)
            sys.exit(1)

        hashed_pool = [OrderedDict([("formula", f), ("hash", hash_formula(f))]) for f in formula_pool]
        new_slice["formula_pool_entries"] = hashed_pool
        slices_list.append(new_slice)

    try:
        with open(args.output, "w") as f:
            ordered_dump(new_data, f)
        print(f"Successfully compiled curriculum to '{args.output}'")
    except IOError as e:
        print(f"ERROR: Writing output failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
