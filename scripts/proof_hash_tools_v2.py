"""Stand-alone proof hash tooling for Phase X.

Examples:
    python -m scripts.proof_hash_tools_v2 --backfill --input raw.jsonl --output hashed.jsonl
    python -m scripts.proof_hash_tools_v2 --diff --old run_a.jsonl --new run_b.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, TextIO, Union, Any


JsonDict = Dict[str, Any]
PathOrFile = Union[str, Path, TextIO]


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    return value


def compute_proof_hash(proof_obj: dict) -> str:
    """Return the deterministic hash for a proof JSON object."""
    if not isinstance(proof_obj, dict):
        raise TypeError("compute_proof_hash expects a JSON object/dict")

    filtered: JsonDict = {k: proof_obj[k] for k in proof_obj if k != "proof_hash"}
    canonical_obj = _canonicalize(filtered)
    payload = json.dumps(
        canonical_obj,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _open_reader(source: PathOrFile) -> Iterator[TextIO]:
    if hasattr(source, "read"):
        yield source  # type: ignore[misc]
    else:
        with open(source, "r", encoding="utf-8") as handle:
            yield handle


@contextmanager
def _open_writer(target: PathOrFile) -> Iterator[TextIO]:
    if hasattr(target, "write"):
        yield target  # type: ignore[misc]
    else:
        with open(target, "w", encoding="utf-8") as handle:
            yield handle


def _iter_jsonl(reader: TextIO) -> Iterator[JsonDict]:
    for line_number, raw_line in enumerate(reader, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if not isinstance(obj, dict):
            raise ValueError(f"Line {line_number} was not a JSON object")
        yield obj


def backfill_hashes(input_path: str, output_path: str) -> None:
    """Add proof_hash values for every JSONL line in input_path."""
    with _open_reader(input_path) as reader, _open_writer(output_path) as writer:
        for obj in _iter_jsonl(reader):
            obj["proof_hash"] = compute_proof_hash(obj)
            writer.write(
                json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"
            )


def _read_hash_set(source: PathOrFile) -> set[str]:
    hashes: set[str] = set()
    with _open_reader(source) as reader:
        for obj in _iter_jsonl(reader):
            proof_hash = obj.get("proof_hash") or compute_proof_hash(obj)
            hashes.add(proof_hash)
    return hashes


def compute_canonical_diff(old_path: str, new_path: str) -> Dict[str, List[str]]:
    """Return proof hash diff summary between two JSONL sources."""
    old_hashes = _read_hash_set(old_path)
    new_hashes = _read_hash_set(new_path)

    removed = sorted(old_hashes - new_hashes)
    added = sorted(new_hashes - old_hashes)
    unchanged = sorted(old_hashes & new_hashes)

    return {"added": added, "removed": removed, "unchanged": unchanged}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Proof hash tooling for JSONL logs.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill proof_hash fields into a JSONL file.",
    )
    action.add_argument(
        "--diff",
        action="store_true",
        help="Compare proof_hash sets between two JSONL files.",
    )
    parser.add_argument("--input", help="Input JSONL path for --backfill.")
    parser.add_argument("--output", help="Output JSONL path for --backfill.")
    parser.add_argument("--old", help="Old run JSONL path for --diff.")
    parser.add_argument("--new", help="New run JSONL path for --diff.")
    return parser


def _main(args: argparse.Namespace) -> int:
    if args.backfill:
        if not args.input or not args.output:
            raise SystemExit("--backfill requires --input and --output paths")
        backfill_hashes(args.input, args.output)
        return 0

    if args.diff:
        if not args.old or not args.new:
            raise SystemExit("--diff requires --old and --new paths")
        diff_result = compute_canonical_diff(args.old, args.new)
        print(json.dumps(diff_result, indent=2, sort_keys=True))
        return 0

    raise SystemExit("Select either --backfill or --diff")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
