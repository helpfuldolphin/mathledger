"""
Audit Plane v0 — Reference Implementation

SHADOW-OBSERVE: This module produces evidence only. It has no authority,
no gating capability, and does not influence verifiers or learning.

Functions:
- validate_event: Validate an audit event against the schema
- compute_event_id: Compute deterministic event_id from canonical JSON
- compute_audit_root: Compute A_t (Merkle root) over event_ids
- generate_audit_root_artifact: Generate audit_root.json artifact
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Schema version this implementation targets
SCHEMA_VERSION = "1.0.0"

# Path to the authoritative schema
SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "audit" / "audit_event.schema.json"

# Fields excluded from canonicalization (per spec)
EXCLUDED_FROM_CANONICAL = {"event_id", "timestamp"}

# Fields excluded from digest computation in meta
META_EXCLUDED = {"note"}


def load_schema() -> dict[str, Any]:
    """Load the audit event schema from disk."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_event(event: dict[str, Any], schema: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """
    Validate an audit event against the schema.

    Returns:
        (is_valid, list of error messages)

    Note: This is a minimal validator for schema conformance checking.
    Validation is informational only; it does not block event recording
    or influence any MathLedger operation.
    """
    errors: list[str] = []

    if schema is None:
        schema = load_schema()

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in event:
            errors.append(f"Missing required field: {field}")

    # Check schema_version
    if event.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Invalid schema_version: expected {SCHEMA_VERSION}, got {event.get('schema_version')}")

    # Check event_type enum
    valid_types = schema.get("properties", {}).get("event_type", {}).get("enum", [])
    if valid_types and event.get("event_type") not in valid_types:
        errors.append(f"Invalid event_type: {event.get('event_type')}")

    # Check severity enum
    valid_severities = schema.get("properties", {}).get("severity", {}).get("enum", [])
    if valid_severities and event.get("severity") not in valid_severities:
        errors.append(f"Invalid severity: {event.get('severity')}")

    # Check subject structure
    subject = event.get("subject", {})
    if not isinstance(subject, dict):
        errors.append("subject must be an object")
    elif "kind" not in subject or "ref" not in subject:
        errors.append("subject must have 'kind' and 'ref' fields")

    # Check digest structure
    digest = event.get("digest", {})
    if not isinstance(digest, dict):
        errors.append("digest must be an object")
    elif digest.get("alg") != "sha256":
        errors.append("digest.alg must be 'sha256'")
    elif not isinstance(digest.get("hex"), str) or len(digest.get("hex", "")) != 64:
        errors.append("digest.hex must be 64 lowercase hex characters")

    # Check event_id format if present
    event_id = event.get("event_id")
    if event_id is not None:
        if not isinstance(event_id, str) or len(event_id) != 64:
            errors.append("event_id must be 64 lowercase hex characters")

    return len(errors) == 0, errors


def canonicalize_event(event: dict[str, Any]) -> bytes:
    """
    Canonicalize an audit event for hashing.

    Rules:
    1. Exclude event_id and timestamp fields
    2. Sort keys alphabetically (recursive)
    3. Use compact JSON representation (no whitespace)
    4. Encode as UTF-8 bytes

    Returns:
        Canonical bytes representation
    """
    def sort_recursive(obj: Any) -> Any:
        if isinstance(obj, dict):
            # Filter excluded fields at top level only
            filtered = {k: v for k, v in obj.items() if k not in EXCLUDED_FROM_CANONICAL}
            # Handle meta.note exclusion
            if "meta" in filtered and isinstance(filtered["meta"], dict):
                filtered["meta"] = {k: v for k, v in filtered["meta"].items() if k not in META_EXCLUDED}
            return {k: sort_recursive(v) for k, v in sorted(filtered.items())}
        elif isinstance(obj, list):
            return [sort_recursive(item) for item in obj]
        return obj

    canonical = sort_recursive(event)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_event_id(event: dict[str, Any]) -> str:
    """
    Compute deterministic event_id from canonical event bytes.

    Returns:
        64-character lowercase hex SHA-256 digest
    """
    canonical_bytes = canonicalize_event(event)
    return hashlib.sha256(canonical_bytes).hexdigest()


def compute_audit_root(event_ids: list[str]) -> str:
    """
    Compute A_t (audit root) as Merkle root over sorted event_ids.

    Algorithm:
    1. Sort event_ids lexicographically (ascending)
    2. If empty, return SHA-256 of empty string
    3. Build binary Merkle tree with SHA-256
    4. Return root as 64-character hex

    Returns:
        64-character lowercase hex SHA-256 Merkle root
    """
    if not event_ids:
        # Empty tree: hash of empty bytes
        return hashlib.sha256(b"").hexdigest()

    # Sort lexicographically for determinism
    sorted_ids = sorted(event_ids)

    # Convert to bytes (each event_id is already a hex digest)
    leaves = [bytes.fromhex(eid) for eid in sorted_ids]

    # Build Merkle tree
    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            # If odd number, duplicate last
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            combined = hashlib.sha256(left + right).digest()
            next_level.append(combined)
        leaves = next_level

    return leaves[0].hex()


def generate_audit_root_artifact(
    events: list[dict[str, Any]],
    output_path: Path | None = None
) -> dict[str, Any]:
    """
    Generate audit_root.json artifact from a list of events.

    Returns:
        Artifact dict with schema_version, event_count, audit_root, and inputs
    """
    # Compute event_ids
    event_ids = []
    for event in events:
        # Use existing event_id if valid, otherwise compute
        if "event_id" in event and len(event.get("event_id", "")) == 64:
            event_ids.append(event["event_id"])
        else:
            event_ids.append(compute_event_id(event))

    # Compute A_t
    audit_root = compute_audit_root(event_ids)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "event_count": len(events),
        "audit_root": audit_root,
        "inputs": sorted(event_ids),  # Sorted for determinism
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True),
            encoding="utf-8"
        )

    return artifact


def load_events_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load audit events from a JSONL file (one JSON object per line)."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Audit Plane v0 — Compute audit root (A_t) from events"
    )
    parser.add_argument(
        "--events",
        type=Path,
        help="Path to audit_events.jsonl file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for audit_root.json (optional)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate events, don't compute root"
    )

    args = parser.parse_args()

    if args.events:
        events = load_events_from_jsonl(args.events)
        print(f"Loaded {len(events)} events from {args.events}")

        # Validate
        schema = load_schema()
        all_valid = True
        for i, event in enumerate(events):
            valid, errors = validate_event(event, schema)
            if not valid:
                print(f"Event {i}: INVALID - {errors}")
                all_valid = False

        if args.validate_only:
            sys.exit(0 if all_valid else 1)

        # Compute root
        artifact = generate_audit_root_artifact(events, args.output)
        print(f"A_t = {artifact['audit_root']}")
        print(f"Event count: {artifact['event_count']}")

        if args.output:
            print(f"Wrote: {args.output}")
    else:
        # Demo mode with sample events
        sample_events = [
            {
                "schema_version": "1.0.0",
                "event_type": "TEST_RESULT",
                "subject": {"kind": "TEST", "ref": "tests/audit/test_audit_root.py"},
                "digest": {"alg": "sha256", "hex": "a" * 64},
                "timestamp": "2025-12-18T12:00:00Z",
                "severity": "INFO",
                "source": "audit_plane_v0"
            },
            {
                "schema_version": "1.0.0",
                "event_type": "HASH_EMITTED",
                "subject": {"kind": "HASH", "ref": "H_t"},
                "digest": {"alg": "sha256", "hex": "b" * 64},
                "timestamp": "2025-12-18T12:00:01Z",
                "severity": "INFO",
                "source": "audit_plane_v0"
            }
        ]

        print("Audit Plane v0 — Demo")
        print("=" * 50)

        for i, event in enumerate(sample_events):
            event_id = compute_event_id(event)
            print(f"Event {i}: event_id = {event_id[:16]}...")

        artifact = generate_audit_root_artifact(sample_events)
        print(f"\nA_t = {artifact['audit_root']}")
        print(f"Event count: {artifact['event_count']}")
        print("\nDemo complete. SHADOW-OBSERVE: No authority; evidence only.")
