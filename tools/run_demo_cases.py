#!/usr/bin/env python3
"""
UVIL v0 Demo Regression Harness

Runs predefined demo cases against the local UVIL API and captures outputs
to fixtures/ for regression testing.

Usage:
    # Start backend first:
    uv run python demo/app.py

    # Then run this script:
    uv run python tools/run_demo_cases.py

    # Or run specific cases:
    uv run python tools/run_demo_cases.py --case mv_only
    uv run python tools/run_demo_cases.py --case adv_only --case mixed_mv_adv

Outputs:
    fixtures/<case_name>/input.json   - Input parameters
    fixtures/<case_name>/output.json  - API responses and attestation roots
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Configuration
API_BASE_URL = "http://localhost:8000"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TIMEOUT = 30.0


@dataclass
class DemoClaim:
    """A claim to submit in a demo case."""
    claim_text: str
    trust_class: str  # FV, MV, PA, ADV
    rationale: str = ""


@dataclass
class DemoCase:
    """A complete demo case definition."""
    name: str
    description: str
    task_text: str
    claims: List[DemoClaim]
    actor_id: str = "demo_harness"
    seed: int = 42
    expected_authority_count: Optional[int] = None
    expected_adv_excluded: bool = True
    expected_outcome: Optional[str] = None  # VERIFIED, REFUTED, or ABSTAINED


# ---------------------------------------------------------------------------
# Demo Cases
# ---------------------------------------------------------------------------

DEMO_CASES: List[DemoCase] = [
    DemoCase(
        name="mv_only",
        description="Pure MV claim - should commit and be authority-bearing",
        task_text="Prove that addition is commutative for natural numbers",
        claims=[
            DemoClaim(
                claim_text="forall a b : Nat, a + b = b + a",
                trust_class="MV",
                rationale="Commutativity of addition - mechanically checkable",
            ),
        ],
        expected_authority_count=1,
    ),
    DemoCase(
        name="mixed_mv_adv",
        description="Mixed MV + ADV - ADV must never enter R_t",
        task_text="Prove commutativity and consider generalizations",
        claims=[
            DemoClaim(
                claim_text="forall a b : Nat, a + b = b + a",
                trust_class="MV",
                rationale="Commutativity - mechanically checkable",
            ),
            DemoClaim(
                claim_text="This likely generalizes to arbitrary rings",
                trust_class="ADV",
                rationale="Speculation about generalization",
            ),
        ],
        expected_authority_count=1,  # Only MV, not ADV
    ),
    DemoCase(
        name="pa_only",
        description="PA only - user attestation, authority-bearing but no mechanical proof",
        task_text="Attest that requirement REQ-001 is satisfied",
        claims=[
            DemoClaim(
                claim_text="Requirement REQ-001 is satisfied by implementation in module X",
                trust_class="PA",
                rationale="User attestation based on manual review",
            ),
        ],
        expected_authority_count=1,
    ),
    DemoCase(
        name="adv_only",
        description="ADV only - must be blocked from authority stream, R_t empty",
        task_text="Speculate about Navier-Stokes",
        claims=[
            DemoClaim(
                claim_text="The Navier-Stokes equations probably have smooth solutions",
                trust_class="ADV",
                rationale="Pure speculation - no proof",
            ),
            DemoClaim(
                claim_text="Turbulence might be related to strange attractors",
                trust_class="ADV",
                rationale="Another guess",
            ),
        ],
        expected_authority_count=0,  # No authority-bearing claims
    ),
    DemoCase(
        name="underdetermined_navier_stokes",
        description="Underdetermined prompt - partition defaults to ADV, demonstrates stopping",
        task_text="Prove existence and smoothness of Navier-Stokes solutions in 3D",
        claims=[
            DemoClaim(
                claim_text="Existence of weak solutions follows from energy estimates",
                trust_class="ADV",
                rationale="Standard but unverified claim",
            ),
            DemoClaim(
                claim_text="Smoothness in 3D remains open (Millennium Prize problem)",
                trust_class="ADV",
                rationale="This is an open problem - cannot be verified",
            ),
            DemoClaim(
                claim_text="Partial regularity results exist (Caffarelli-Kohn-Nirenberg)",
                trust_class="ADV",
                rationale="Reference to literature, not mechanically checked",
            ),
        ],
        expected_authority_count=0,  # All ADV - system should abstain
        expected_outcome="ABSTAINED",
    ),
    # ---------------------------------------------------------------------------
    # MV Arithmetic Validator Cases (v0.1)
    # ---------------------------------------------------------------------------
    DemoCase(
        name="mv_arithmetic_verified",
        description="MV with valid arithmetic - demonstrates real verification",
        task_text="Verify that 2 + 2 = 4",
        claims=[
            DemoClaim(
                claim_text="2 + 2 = 4",
                trust_class="MV",
                rationale="Simple arithmetic - mechanically checkable",
            ),
        ],
        expected_authority_count=1,
        expected_outcome="VERIFIED",  # Arithmetic validator confirms
    ),
    DemoCase(
        name="mv_arithmetic_refuted",
        description="MV with invalid arithmetic - demonstrates refutation",
        task_text="Verify that 2 + 2 = 5",
        claims=[
            DemoClaim(
                claim_text="2 + 2 = 5",
                trust_class="MV",
                rationale="Incorrect arithmetic",
            ),
        ],
        expected_authority_count=1,
        expected_outcome="REFUTED",  # Arithmetic validator refutes
    ),
    DemoCase(
        name="same_claim_as_pa",
        description="Same arithmetic claim as PA - demonstrates trust class matters",
        task_text="Attest that 2 + 2 = 4",
        claims=[
            DemoClaim(
                claim_text="2 + 2 = 4",
                trust_class="PA",  # Same text, but PA not MV
                rationale="User attestation - not mechanically verified",
            ),
        ],
        expected_authority_count=1,
        expected_outcome="ABSTAINED",  # PA is not verified, even if claim is true
    ),
    DemoCase(
        name="same_claim_as_adv",
        description="Same arithmetic claim as ADV - excluded from authority entirely",
        task_text="Consider whether 2 + 2 = 4",
        claims=[
            DemoClaim(
                claim_text="2 + 2 = 4",
                trust_class="ADV",  # Same text, but ADV
                rationale="Advisory only - exploring the claim",
            ),
        ],
        expected_authority_count=0,  # ADV excluded
        expected_outcome="ABSTAINED",  # Nothing in authority stream
    ),
]


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

class UVILClient:
    """Client for UVIL API endpoints."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=TIMEOUT)

    def health_check(self) -> bool:
        """Check if the API is running."""
        try:
            resp = self.client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    def propose_partition(self, task_text: str) -> Dict[str, Any]:
        """POST /uvil/propose_partition"""
        resp = self.client.post(
            f"{self.base_url}/uvil/propose_partition",
            json={"problem_statement": task_text},
        )
        resp.raise_for_status()
        return resp.json()

    def commit_uvil(
        self,
        proposal_id: str,
        claims: List[DemoClaim],
        user_fingerprint: str = "demo_harness",
    ) -> Dict[str, Any]:
        """POST /uvil/commit_uvil"""
        resp = self.client.post(
            f"{self.base_url}/uvil/commit_uvil",
            json={
                "proposal_id": proposal_id,
                "edited_claims": [
                    {
                        "claim_text": c.claim_text,
                        "trust_class": c.trust_class,
                        "rationale": c.rationale,
                    }
                    for c in claims
                ],
                "user_fingerprint": user_fingerprint,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def run_verification(self, committed_partition_id: str) -> Dict[str, Any]:
        """POST /uvil/run_verification"""
        resp = self.client.post(
            f"{self.base_url}/uvil/run_verification",
            json={"committed_partition_id": committed_partition_id},
        )
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP client."""
        self.client.close()


# ---------------------------------------------------------------------------
# Case Runner
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Result of running a demo case."""
    case_name: str
    success: bool
    error: Optional[str] = None

    # Captured data
    draft_proposal: Optional[Dict[str, Any]] = None
    commit_response: Optional[Dict[str, Any]] = None
    verification_response: Optional[Dict[str, Any]] = None

    # Derived values
    proposal_id: Optional[str] = None
    committed_partition_id: Optional[str] = None
    u_t: Optional[str] = None
    r_t: Optional[str] = None
    h_t: Optional[str] = None
    authority_claim_count: Optional[int] = None
    total_claim_count: Optional[int] = None
    adv_excluded_from_rt: Optional[bool] = None
    authority_basis: Optional[Dict[str, Any]] = None
    outcome_explanation: Optional[str] = None
    outcome: Optional[str] = None  # VERIFIED, REFUTED, or ABSTAINED


def run_case(client: UVILClient, case: DemoCase) -> CaseResult:
    """Run a single demo case and capture all outputs."""
    result = CaseResult(case_name=case.name, success=False)

    try:
        # Step 1: Propose partition (exploration only)
        print(f"  [1/3] POST /uvil/propose_partition")
        draft = client.propose_partition(case.task_text)
        result.draft_proposal = draft
        result.proposal_id = draft["proposal_id"]
        print(f"        proposal_id: {result.proposal_id[:16]}...")

        # Step 2: Commit with our predefined claims (not the draft's suggestions)
        print(f"  [2/3] POST /uvil/commit_uvil")
        commit = client.commit_uvil(
            proposal_id=result.proposal_id,
            claims=case.claims,
            user_fingerprint=case.actor_id,
        )
        result.commit_response = commit
        result.committed_partition_id = commit["committed_partition_id"]
        result.u_t = commit["u_t"]
        result.r_t = commit["r_t"]
        result.h_t = commit["h_t"]
        print(f"        committed_id: {result.committed_partition_id[:16]}...")

        # Step 3: Run verification
        print(f"  [3/3] POST /uvil/run_verification")
        verify = client.run_verification(result.committed_partition_id)
        result.verification_response = verify
        result.authority_claim_count = verify["attestation"]["authority_claim_count"]
        result.total_claim_count = verify["attestation"]["total_claim_count"]

        # Update final attestation values
        result.u_t = verify["attestation"]["u_t"]
        result.r_t = verify["attestation"]["r_t"]
        result.h_t = verify["attestation"]["h_t"]

        # Capture authority basis (new in v0 fix)
        result.authority_basis = verify.get("authority_basis", {})
        result.outcome_explanation = result.authority_basis.get("explanation", "")

        # Check ADV exclusion
        adv_count = sum(1 for c in case.claims if c.trust_class == "ADV")
        result.adv_excluded_from_rt = (
            result.authority_claim_count == result.total_claim_count - adv_count
        )

        result.outcome = verify["outcome"]
        result.success = True
        print(f"        outcome: {result.outcome}")

    except httpx.HTTPStatusError as e:
        result.error = f"HTTP {e.response.status_code}: {e.response.text}"
        print(f"        ERROR: {result.error}")
    except httpx.RequestError as e:
        result.error = f"Request failed: {e}"
        print(f"        ERROR: {result.error}")
    except Exception as e:
        result.error = f"Unexpected error: {e}"
        print(f"        ERROR: {result.error}")

    return result


def save_fixtures(case: DemoCase, result: CaseResult) -> Path:
    """Save case inputs and outputs to fixtures directory."""
    case_dir = FIXTURES_DIR / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    # Input file
    input_data = {
        "case_name": case.name,
        "description": case.description,
        "task_text": case.task_text,
        "claims": [asdict(c) for c in case.claims],
        "actor_id": case.actor_id,
        "seed": case.seed,
        "expected_authority_count": case.expected_authority_count,
        "expected_adv_excluded": case.expected_adv_excluded,
        "expected_outcome": case.expected_outcome,
    }
    input_path = case_dir / "input.json"
    input_path.write_text(json.dumps(input_data, indent=2))

    # Output file
    output_data = {
        "success": result.success,
        "error": result.error,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "exploration": {
            "proposal_id": result.proposal_id,
            "draft_proposal": result.draft_proposal,
            "note": "proposal_id is exploration-only and MUST NOT appear in attestation",
        },
        "authority": {
            "committed_partition_id": result.committed_partition_id,
            "commit_response": result.commit_response,
        },
        "verification": {
            "response": result.verification_response,
        },
        "attestation": {
            "u_t": result.u_t,
            "r_t": result.r_t,
            "h_t": result.h_t,
        },
        "authority_basis": result.authority_basis,
        "outcome": result.outcome,
        "analysis": {
            "authority_claim_count": result.authority_claim_count,
            "total_claim_count": result.total_claim_count,
            "adv_excluded_from_rt": result.adv_excluded_from_rt,
            "outcome_explanation": result.outcome_explanation,
        },
    }
    output_path = case_dir / "output.json"
    output_path.write_text(json.dumps(output_data, indent=2))

    return case_dir


def print_summary(case: DemoCase, result: CaseResult):
    """Print a concise summary for a case."""
    print()
    print(f"  SUMMARY: {case.name}")
    print(f"  ---------{'-' * len(case.name)}")

    if not result.success:
        print(f"  Status: FAILED - {result.error}")
        return

    # Count claims by trust class
    tc_counts = {}
    for c in case.claims:
        tc_counts[c.trust_class] = tc_counts.get(c.trust_class, 0) + 1
    tc_str = ", ".join(f"{k}={v}" for k, v in sorted(tc_counts.items()))

    print(f"  Status: SUCCESS")
    print(f"  Claims: {tc_str}")
    print(f"  Authority-bearing: {result.authority_claim_count}/{result.total_claim_count}")
    print(f"  ADV excluded from R_t: {result.adv_excluded_from_rt}")
    print(f"  U_t: {result.u_t[:32]}...")
    print(f"  R_t: {result.r_t[:32]}...")
    print(f"  H_t: {result.h_t[:32]}...")

    # Check invariants
    if case.expected_authority_count is not None:
        if result.authority_claim_count == case.expected_authority_count:
            print(f"  [OK] Authority count matches expected ({case.expected_authority_count})")
        else:
            print(f"  [!!] Authority count {result.authority_claim_count} != expected {case.expected_authority_count}")

    if case.expected_adv_excluded and not result.adv_excluded_from_rt:
        print(f"  [!!] ADV was NOT excluded from R_t (invariant violation!)")

    if case.expected_outcome is not None:
        if result.outcome == case.expected_outcome:
            print(f"  [OK] Outcome matches expected ({case.expected_outcome})")
        else:
            print(f"  [!!] Outcome {result.outcome} != expected {case.expected_outcome}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run UVIL v0 demo cases")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run specific case(s) by name. Can be specified multiple times.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cases and exit.",
    )
    parser.add_argument(
        "--url",
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})",
    )
    args = parser.parse_args()

    # List cases
    if args.list:
        print("Available demo cases:")
        for case in DEMO_CASES:
            print(f"  {case.name}: {case.description}")
        return 0

    # Select cases to run
    if args.cases:
        case_names = set(args.cases)
        cases_to_run = [c for c in DEMO_CASES if c.name in case_names]
        unknown = case_names - {c.name for c in cases_to_run}
        if unknown:
            print(f"Unknown case(s): {', '.join(unknown)}")
            return 1
    else:
        cases_to_run = DEMO_CASES

    # Check API health
    client = UVILClient(base_url=args.url)
    print(f"Checking API at {args.url}...")

    if not client.health_check():
        print("ERROR: API is not responding. Start the backend first:")
        print("  uv run python demo/app.py")
        return 1

    print("API is healthy.\n")
    print("=" * 60)
    print("UVIL v0 Demo Regression Harness")
    print("=" * 60)

    # Run cases
    results = []
    for case in cases_to_run:
        print(f"\n[{case.name}] {case.description}")
        result = run_case(client, case)
        results.append((case, result))

        if result.success:
            fixture_dir = save_fixtures(case, result)
            print(f"        Saved to: {fixture_dir.relative_to(Path.cwd())}/")

        print_summary(case, result)

    client.close()

    # Final summary
    print("\n" + "=" * 60)
    success_count = sum(1 for _, r in results if r.success)
    print(f"Completed: {success_count}/{len(results)} cases succeeded")
    print(f"Fixtures written to: {FIXTURES_DIR.relative_to(Path.cwd())}/")
    print("=" * 60)

    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
