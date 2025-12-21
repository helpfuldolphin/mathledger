#!/usr/bin/env python3
"""
MathLedger Drop-In Governance Demo
===================================

Single-command demo proving MathLedger as a drop-in governance substrate.
An external engineer can run this in <10 minutes and independently verify:

1. Deterministic execution (same inputs → same outputs)
2. Dual attestation (R_t reasoning + U_t UI → H_t composite)
3. Governance verdict (F5.x predicates, claim level assignment)
4. Replayability (manifest + hashes for external audit)

Usage:
    uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/

Exit codes:
    0 - Demo completed successfully (governance pass OR fail-close are both success)
    1 - Infrastructure/environment error

Note: Governance triggering fail-close (claim level L0) is EXPECTED BEHAVIOR
for this demo seed. It demonstrates the fail-safe mechanism working correctly.

ACQUISITION-FACING: This script is designed for external due diligence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Imports from existing MathLedger primitives
# ---------------------------------------------------------------------------

try:
    from attestation.dual_root import (
        build_reasoning_attestation,
        build_ui_attestation,
        compute_composite_root,
        verify_composite_integrity,
    )
    from substrate.repro.toolchain import (
        capture_toolchain_snapshot,
        ToolchainSnapshot,
    )
    from backend.repro.determinism import (
        deterministic_hash,
        deterministic_timestamp,
        deterministic_merkle_root,
        SeededRNG,
    )
except ImportError as e:
    print(f"ERROR: Missing MathLedger primitive: {e}", file=sys.stderr)
    print("Ensure you are running from the repo root with: uv run python scripts/run_dropin_demo.py", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Demo Configuration
# ---------------------------------------------------------------------------

DEMO_SCHEMA_VERSION = "1.0.0"
DEMO_MODE = "SHADOW"  # Observational only, non-gating


@dataclass
class DemoConfig:
    """Demo execution configuration."""
    seed: int
    output_dir: Path
    num_reasoning_events: int = 10
    num_ui_events: int = 5
    variance_ratio_threshold: float = 2.0
    drift_threshold: float = 0.15


@dataclass
class GovernanceVerdict:
    """Governance verdict with F5.x codes."""
    claim_level: str  # L0-L5
    f5_codes: List[str]
    passed: bool
    rationale: str


# ---------------------------------------------------------------------------
# Synthetic Event Generation (deterministic)
# ---------------------------------------------------------------------------

def generate_synthetic_reasoning_events(seed: int, count: int) -> List[Dict[str, Any]]:
    """Generate deterministic synthetic reasoning/proof events."""
    rng = SeededRNG(seed)
    events = []

    for i in range(count):
        # Deterministic event content
        event = {
            "event_id": f"proof_{seed}_{i:04d}",
            "type": "proof_verification",
            "statement_hash": deterministic_hash(f"statement_{seed}_{i}"),
            "verification_result": "PASS" if rng.random()[0] > 0.1 else "FAIL",
            "depth": int(rng.randint(1, 10)[0]),
            "axiom_count": int(rng.randint(1, 5)[0]),
        }
        events.append(event)

    return events


def generate_synthetic_ui_events(seed: int, count: int) -> List[Dict[str, Any]]:
    """Generate deterministic synthetic UI/human interaction events."""
    rng = SeededRNG(seed + 1000)  # Offset seed for UI stream
    events = []

    for i in range(count):
        event = {
            "event_id": f"ui_{seed}_{i:04d}",
            "type": "user_interaction",
            "action": ["view", "expand", "verify", "flag"][int(rng.randint(0, 4)[0])],
            "target_hash": deterministic_hash(f"target_{seed}_{i}"),
            "session_id": deterministic_hash(f"session_{seed}")[:16],
        }
        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Governance Predicates (F5.x)
# ---------------------------------------------------------------------------

def evaluate_f5_predicates(
    reasoning_events: List[Dict[str, Any]],
    ui_events: List[Dict[str, Any]],
    config: DemoConfig,
) -> GovernanceVerdict:
    """
    Evaluate F5.x governance predicates.

    This is a simplified demonstration of fail-close governance.
    In production, these predicates operate on actual Δp measurements.
    """
    f5_codes = []

    # F5.1: Toolchain drift check (simulated)
    # In production, this compares toolchain fingerprints between arms
    toolchain_aligned = True  # Demo assumes aligned
    if not toolchain_aligned:
        f5_codes.append("F5.1")

    # F5.2: Variance ratio check (simulated)
    # Simulate variance check on reasoning event depths
    rng = SeededRNG(config.seed + 2000)
    simulated_variance_ratio = 1.0 + rng.random()[0] * 3.0  # 1.0-4.0
    if simulated_variance_ratio > config.variance_ratio_threshold:
        f5_codes.append("F5.2")

    # F5.3: Windowed drift check (simulated)
    simulated_drift = rng.random()[0] * 0.3  # 0.0-0.3
    if simulated_drift > config.drift_threshold:
        f5_codes.append("F5.3")

    # F5.4: Event count sanity
    if len(reasoning_events) < 5:
        f5_codes.append("F5.4")

    # F5.5: UI stream presence
    if len(ui_events) < 1:
        f5_codes.append("F5.5")

    # Claim level assignment
    if f5_codes:
        claim_level = "L0"
        passed = False
        rationale = f"Fail-close triggered by: {', '.join(f5_codes)}"
    else:
        claim_level = "L4"  # Single-run maximum
        passed = True
        rationale = "All F5.x predicates passed"

    return GovernanceVerdict(
        claim_level=claim_level,
        f5_codes=f5_codes,
        passed=passed,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Attestation Construction
# ---------------------------------------------------------------------------

def build_attestations(
    reasoning_events: List[Dict[str, Any]],
    ui_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build dual-root attestation from event streams."""

    # Build attestation trees
    reasoning_tree = build_reasoning_attestation(reasoning_events)
    ui_tree = build_ui_attestation(ui_events)

    # Extract roots
    r_t = reasoning_tree.root
    u_t = ui_tree.root

    # Compute composite
    h_t = compute_composite_root(r_t, u_t)

    # Verify integrity
    integrity_valid = verify_composite_integrity(r_t, u_t, h_t)

    return {
        "reasoning_root": r_t,
        "ui_root": u_t,
        "composite_root": h_t,
        "integrity_valid": integrity_valid,
        "reasoning_event_count": len(reasoning_events),
        "ui_event_count": len(ui_events),
        "reasoning_leaf_hashes": [leaf.leaf_hash for leaf in reasoning_tree.leaves],
        "ui_leaf_hashes": [leaf.leaf_hash for leaf in ui_tree.leaves],
    }


# ---------------------------------------------------------------------------
# Manifest Generation
# ---------------------------------------------------------------------------

def generate_manifest(
    config: DemoConfig,
    attestation: Dict[str, Any],
    verdict: GovernanceVerdict,
    toolchain: Optional[ToolchainSnapshot],
) -> Dict[str, Any]:
    """Generate the complete demo manifest for external verification."""

    # Deterministic timestamp from seed
    ts = deterministic_timestamp(config.seed)

    manifest = {
        "schema_version": DEMO_SCHEMA_VERSION,
        "demo_mode": DEMO_MODE,
        "generated_at": ts.isoformat(),
        "seed": config.seed,

        # Attestation
        "attestation": {
            "reasoning_merkle_root": attestation["reasoning_root"],
            "ui_merkle_root": attestation["ui_root"],
            "composite_attestation_root": attestation["composite_root"],
            "integrity_verified": attestation["integrity_valid"],
            "formula": "H_t = SHA256(R_t || U_t)",
            "reasoning_event_count": attestation["reasoning_event_count"],
            "ui_event_count": attestation["ui_event_count"],
        },

        # Governance
        "governance": {
            "claim_level": verdict.claim_level,
            "f5_codes": verdict.f5_codes,
            "passed": verdict.passed,
            "rationale": verdict.rationale,
        },

        # Reproducibility
        "reproducibility": {
            "deterministic": True,
            "same_seed_same_output": True,
            "verification_command": f"uv run python scripts/run_dropin_demo.py --seed {config.seed} --output demo_output_verify/",
        },
    }

    # Toolchain fingerprint (if available)
    if toolchain:
        manifest["toolchain"] = {
            "fingerprint": toolchain.fingerprint,
            "python_version": toolchain.python.version,
            "uv_lock_hash": toolchain.python.uv_lock_hash[:16] + "...",
        }

    return manifest


# ---------------------------------------------------------------------------
# Output Generation
# ---------------------------------------------------------------------------

def write_outputs(
    config: DemoConfig,
    manifest: Dict[str, Any],
    attestation: Dict[str, Any],
    reasoning_events: List[Dict[str, Any]],
    ui_events: List[Dict[str, Any]],
) -> None:
    """Write all demo outputs to the output directory."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Manifest (main verification artifact)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # 2. Individual root files (for easy diffing)
    (output_dir / "reasoning_root.txt").write_text(
        attestation["reasoning_root"] + "\n",
        encoding="utf-8",
    )
    (output_dir / "ui_root.txt").write_text(
        attestation["ui_root"] + "\n",
        encoding="utf-8",
    )
    (output_dir / "epoch_root.txt").write_text(
        attestation["composite_root"] + "\n",
        encoding="utf-8",
    )

    # 3. Events (for full replay)
    events_dir = output_dir / "events"
    events_dir.mkdir(exist_ok=True)

    (events_dir / "reasoning_events.jsonl").write_text(
        "\n".join(json.dumps(e, sort_keys=True) for e in reasoning_events) + "\n",
        encoding="utf-8",
    )
    (events_dir / "ui_events.jsonl").write_text(
        "\n".join(json.dumps(e, sort_keys=True) for e in ui_events) + "\n",
        encoding="utf-8",
    )

    # 4. Verification script
    verify_script = output_dir / "verify.py"
    verify_script.write_text(
        generate_verification_script(config.seed),
        encoding="utf-8",
    )


def generate_verification_script(seed: int) -> str:
    """Generate a standalone verification script for the demo output."""
    return f'''#!/usr/bin/env python3
"""
Standalone verification script for MathLedger drop-in demo.

This script can be run independently to verify:
1. Composite root integrity: H_t == SHA256(R_t || U_t)
2. Reproducibility: re-running demo produces identical outputs

Usage:
    python verify.py

Expected output:
    [PASS] Composite root verified
    [INFO] To verify reproducibility, run the demo again with seed {seed}
"""

import hashlib
import json
from pathlib import Path

def main():
    # Load roots
    r_t = Path("reasoning_root.txt").read_text().strip()
    u_t = Path("ui_root.txt").read_text().strip()
    h_t = Path("epoch_root.txt").read_text().strip()

    # Verify composite
    computed = hashlib.sha256((r_t + u_t).encode("ascii")).hexdigest()

    if computed == h_t:
        print("[PASS] Composite root verified: H_t == SHA256(R_t || U_t)")
    else:
        print("[FAIL] Composite root mismatch!")
        print(f"  Expected: {{h_t}}")
        print(f"  Computed: {{computed}}")
        exit(1)

    # Load manifest
    manifest = json.loads(Path("manifest.json").read_text())
    print(f"[INFO] Seed: {{manifest['seed']}}")
    print(f"[INFO] Claim level: {{manifest['governance']['claim_level']}}")
    print(f"[INFO] F5 codes: {{manifest['governance']['f5_codes']}}")
    print(f"[INFO] To verify reproducibility, run: uv run python scripts/run_dropin_demo.py --seed {seed} --output demo_output_verify/")

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def run_demo(config: DemoConfig) -> int:
    """Run the complete drop-in demo."""

    print("=" * 60)
    print("MathLedger Drop-In Governance Demo")
    print("=" * 60)
    print(f"Seed: {config.seed}")
    print(f"Output: {config.output_dir}")
    print(f"Mode: {DEMO_MODE} (observational only)")
    print()

    # Step 1: Capture toolchain snapshot (optional - Lean not required for demo)
    print("[1/5] Capturing toolchain snapshot...")
    try:
        toolchain = capture_toolchain_snapshot()
        print(f"      Fingerprint: {toolchain.fingerprint[:16]}...")
    except Exception as e:
        print(f"      Skipped (Lean toolchain not present - not required for demo)")
        toolchain = None

    # Step 2: Generate synthetic events
    print("[2/5] Generating synthetic events...")
    reasoning_events = generate_synthetic_reasoning_events(
        config.seed,
        config.num_reasoning_events,
    )
    ui_events = generate_synthetic_ui_events(
        config.seed,
        config.num_ui_events,
    )
    print(f"      Reasoning events: {len(reasoning_events)}")
    print(f"      UI events: {len(ui_events)}")

    # Step 3: Build attestations
    print("[3/5] Building dual-root attestation...")
    attestation = build_attestations(reasoning_events, ui_events)
    print(f"      R_t: {attestation['reasoning_root'][:16]}...")
    print(f"      U_t: {attestation['ui_root'][:16]}...")
    print(f"      H_t: {attestation['composite_root'][:16]}...")
    print(f"      Integrity: {'VALID' if attestation['integrity_valid'] else 'INVALID'}")

    # Step 4: Evaluate governance predicates
    print("[4/5] Evaluating governance predicates...")
    verdict = evaluate_f5_predicates(reasoning_events, ui_events, config)
    print(f"      Claim level: {verdict.claim_level}")
    print(f"      F5 codes: {verdict.f5_codes if verdict.f5_codes else '(none)'}")
    print(f"      Passed: {verdict.passed}")

    # Step 5: Generate outputs
    print("[5/5] Writing outputs...")
    manifest = generate_manifest(config, attestation, verdict, toolchain)
    write_outputs(config, manifest, attestation, reasoning_events, ui_events)
    print(f"      Manifest: {config.output_dir / 'manifest.json'}")
    print(f"      Roots: reasoning_root.txt, ui_root.txt, epoch_root.txt")
    print(f"      Events: events/reasoning_events.jsonl, events/ui_events.jsonl")
    print(f"      Verifier: verify.py")

    # Summary
    print()
    print("=" * 60)
    print("Demo Complete")
    print("=" * 60)
    print()
    print("Verification steps:")
    print(f"  1. cd {config.output_dir}")
    print("  2. python verify.py")
    print(f"  3. Compare outputs with: uv run python scripts/run_dropin_demo.py --seed {config.seed} --output demo_output_verify/")
    print()
    print("What this demonstrates:")
    print("  - Deterministic execution: Same seed -> same outputs")
    print("  - Dual attestation: R_t (reasoning) + U_t (UI) -> H_t (composite)")
    print("  - Governance: F5.x predicates evaluate fail-close behavior")
    print("  - Replayability: All inputs/outputs captured for audit")
    print()

    # Exit code based on demo success (not governance verdict)
    if not attestation["integrity_valid"]:
        print("[ERROR] Attestation integrity check failed")
        return 1

    if verdict.passed:
        print(f"[RESULT] Governance PASSED (claim level: {verdict.claim_level})")
    else:
        print(f"[RESULT] Governance triggered fail-close (claim level: {verdict.claim_level})")
        print(f"         This is EXPECTED BEHAVIOR demonstrating fail-safe governance")

    # Exit 0 = demo succeeded (regardless of governance outcome)
    return 0


def parse_args() -> DemoConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MathLedger Drop-In Governance Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/run_dropin_demo.py --seed 42
    uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/

Exit codes:
    0 - Demo completed successfully
    1 - Infrastructure/environment error

Note: Governance triggering fail-close (L0) is expected behavior for seed 42.
""",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic execution (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_output"),
        help="Output directory (default: demo_output/)",
    )
    parser.add_argument(
        "--reasoning-events",
        type=int,
        default=10,
        help="Number of synthetic reasoning events (default: 10)",
    )
    parser.add_argument(
        "--ui-events",
        type=int,
        default=5,
        help="Number of synthetic UI events (default: 5)",
    )

    args = parser.parse_args()

    return DemoConfig(
        seed=args.seed,
        output_dir=args.output,
        num_reasoning_events=args.reasoning_events,
        num_ui_events=args.ui_events,
    )


def main() -> None:
    """Main entry point."""
    config = parse_args()
    exit_code = run_demo(config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
