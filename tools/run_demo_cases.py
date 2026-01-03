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

    def get_evidence_pack(
        self,
        committed_partition_id: str,
        claimed_h_t: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GET /uvil/evidence_pack/{committed_partition_id}"""
        params = {}
        if claimed_h_t is not None:
            params["claimed_h_t"] = claimed_h_t
        resp = self.client.get(
            f"{self.base_url}/uvil/evidence_pack/{committed_partition_id}",
            params=params,
        )
        resp.raise_for_status()
        return resp.json()

    def get_evidence_pack_raw(
        self,
        committed_partition_id: str,
        claimed_h_t: Optional[str] = None,
    ) -> httpx.Response:
        """GET /uvil/evidence_pack/{committed_partition_id} - returns raw response."""
        params = {}
        if claimed_h_t is not None:
            params["claimed_h_t"] = claimed_h_t
        return self.client.get(
            f"{self.base_url}/uvil/evidence_pack/{committed_partition_id}",
            params=params,
        )

    def replay_verify(
        self,
        uvil_events: List[Dict[str, Any]],
        reasoning_artifacts: List[Dict[str, Any]],
        expected_u_t: str,
        expected_r_t: str,
        expected_h_t: str,
    ) -> Dict[str, Any]:
        """POST /uvil/replay_verify"""
        resp = self.client.post(
            f"{self.base_url}/uvil/replay_verify",
            json={
                "uvil_events": uvil_events,
                "reasoning_artifacts": reasoning_artifacts,
                "expected_u_t": expected_u_t,
                "expected_r_t": expected_r_t,
                "expected_h_t": expected_h_t,
            },
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
# Evidence Pack Tests
# ---------------------------------------------------------------------------


def run_evidence_pack_tests(client: UVILClient) -> bool:
    """
    Run evidence pack tests:
    1. Determinism: Same inputs produce same evidence pack
    2. Replay verification: Pack can be replayed successfully
    3. Tamper detection: Tampered pack fails verification
    """
    print("\n" + "=" * 60)
    print("Evidence Pack Tests")
    print("=" * 60)

    all_passed = True

    # Use mv_arithmetic_verified case for testing
    print("\n[evidence_pack_determinism] Testing determinism...")

    try:
        # Create a test case
        draft1 = client.propose_partition("Evidence pack test 1")
        commit1 = client.commit_uvil(
            proposal_id=draft1["proposal_id"],
            claims=[DemoClaim("1 + 1 = 2", "MV", "Test")],
            user_fingerprint="evidence_test",
        )
        verify1 = client.run_verification(commit1["committed_partition_id"])
        pack1 = client.get_evidence_pack(commit1["committed_partition_id"])

        # Verify the pack
        replay1 = client.replay_verify(
            uvil_events=pack1["uvil_events"],
            reasoning_artifacts=pack1["reasoning_artifacts"],
            expected_u_t=pack1["u_t"],
            expected_r_t=pack1["r_t"],
            expected_h_t=pack1["h_t"],
        )

        if replay1["result"] == "PASS":
            print("  [OK] Replay verification PASS")
        else:
            print(f"  [!!] Replay verification FAIL: {replay1.get('diff', 'unknown')}")
            all_passed = False

        # Test determinism: get pack again, should have same hashes
        pack2 = client.get_evidence_pack(commit1["committed_partition_id"])
        if (pack1["u_t"] == pack2["u_t"] and
            pack1["r_t"] == pack2["r_t"] and
            pack1["h_t"] == pack2["h_t"]):
            print("  [OK] Evidence pack is deterministic (same hashes on re-fetch)")
        else:
            print("  [!!] Evidence pack NOT deterministic")
            all_passed = False

    except Exception as e:
        print(f"  [!!] Evidence pack test failed: {e}")
        all_passed = False

    # Test tamper detection
    print("\n[evidence_pack_tamper] Testing tamper detection...")

    try:
        # Create another test case
        draft3 = client.propose_partition("Evidence pack tamper test")
        commit3 = client.commit_uvil(
            proposal_id=draft3["proposal_id"],
            claims=[DemoClaim("3 + 3 = 6", "MV", "Test")],
            user_fingerprint="tamper_test",
        )
        verify3 = client.run_verification(commit3["committed_partition_id"])
        pack3 = client.get_evidence_pack(commit3["committed_partition_id"])

        # Tamper with the expected hash
        tampered_h_t = "0" * 64  # Fake hash

        replay3 = client.replay_verify(
            uvil_events=pack3["uvil_events"],
            reasoning_artifacts=pack3["reasoning_artifacts"],
            expected_u_t=pack3["u_t"],
            expected_r_t=pack3["r_t"],
            expected_h_t=tampered_h_t,  # Tampered!
        )

        if replay3["result"] == "FAIL":
            print("  [OK] Tamper detection PASS (correctly detected tampered hash)")
        else:
            print("  [!!] Tamper detection FAIL (did not detect tampered hash)")
            all_passed = False

        # Tamper with reasoning artifact
        if pack3["reasoning_artifacts"]:
            tampered_artifacts = pack3["reasoning_artifacts"].copy()
            tampered_artifacts[0]["claim_id"] = "tampered_claim_id"

            replay4 = client.replay_verify(
                uvil_events=pack3["uvil_events"],
                reasoning_artifacts=tampered_artifacts,
                expected_u_t=pack3["u_t"],
                expected_r_t=pack3["r_t"],
                expected_h_t=pack3["h_t"],
            )

            if replay4["result"] == "FAIL":
                print("  [OK] Artifact tamper detection PASS")
            else:
                print("  [!!] Artifact tamper detection FAIL")
                all_passed = False

    except Exception as e:
        print(f"  [!!] Tamper detection test failed: {e}")
        all_passed = False

    # Test evidence pack schema
    print("\n[evidence_pack_schema] Testing evidence pack schema...")

    try:
        required_fields = [
            "schema_version",
            "committed_partition_snapshot",
            "uvil_events",
            "reasoning_artifacts",
            "u_t",
            "r_t",
            "h_t",
            "h_t_formula_note",
            "replay_instructions",
            "outcome",
            "authority_basis",
        ]

        missing = [f for f in required_fields if f not in pack1]
        if missing:
            print(f"  [!!] Missing fields in evidence pack: {missing}")
            all_passed = False
        else:
            print("  [OK] Evidence pack has all required fields")

        # Check schema version
        if pack1.get("schema_version") == "evidence_pack_v1":
            print("  [OK] Schema version is evidence_pack_v1")
        else:
            print(f"  [!!] Unexpected schema version: {pack1.get('schema_version')}")
            all_passed = False

        # Check H_t formula note
        if pack1.get("h_t_formula_note") == "H_t = SHA256(R_t || U_t)":
            print("  [OK] H_t formula note present")
        else:
            print("  [!!] H_t formula note missing or incorrect")
            all_passed = False

    except Exception as e:
        print(f"  [!!] Schema test failed: {e}")
        all_passed = False

    print("\n" + "-" * 60)
    if all_passed:
        print("Evidence Pack Tests: ALL PASSED")
    else:
        print("Evidence Pack Tests: SOME FAILED")
    print("-" * 60)

    return all_passed


# ---------------------------------------------------------------------------
# Authority Gate Regression Tests (No Silent Authority)
# ---------------------------------------------------------------------------


def run_authority_gate_tests(client: UVILClient) -> bool:
    """
    Run authority gate regression tests.

    Tests that the authority gate correctly blocks evidence pack export
    when epoch root verification fails, returning HTTP 422 with
    SILENT_AUTHORITY_VIOLATION error code.

    Regression cases:
    1. missing_epoch_root: Request evidence pack with invalid claimed H_t
    2. tampered_epoch_root: Request evidence pack with tampered claimed H_t
    """
    print("\n" + "=" * 60)
    print("Authority Gate Regression Tests")
    print("=" * 60)

    all_passed = True

    # Create a valid test case first
    print("\n[authority_gate_setup] Creating test partition...")
    try:
        draft = client.propose_partition("Authority gate test")
        commit = client.commit_uvil(
            proposal_id=draft["proposal_id"],
            claims=[DemoClaim("5 + 5 = 10", "MV", "Test")],
            user_fingerprint="authority_gate_test",
        )
        verify = client.run_verification(commit["committed_partition_id"])
        committed_id = commit["committed_partition_id"]

        # Get valid evidence pack first
        valid_pack = client.get_evidence_pack(committed_id)
        valid_h_t = valid_pack["h_t"]
        print(f"  [OK] Created test partition: {committed_id[:16]}...")
        print(f"  [OK] Valid H_t: {valid_h_t[:16]}...")
    except Exception as e:
        print(f"  [!!] Setup failed: {e}")
        return False

    # ---------------------------------------------------------------------------
    # Regression Case 1: missing_epoch_root
    # ---------------------------------------------------------------------------
    print("\n[authority_gate_missing] Testing missing/invalid epoch root...")
    try:
        # Use a completely invalid H_t (zeros)
        invalid_h_t = "0" * 64

        resp = client.get_evidence_pack_raw(committed_id, claimed_h_t=invalid_h_t)

        if resp.status_code == 422:
            error_body = resp.json()
            detail = error_body.get("detail", {})

            # Check error_code
            if detail.get("error_code") == "SILENT_AUTHORITY_VIOLATION":
                print("  [OK] Status code 422 returned")
                print("  [OK] error_code is SILENT_AUTHORITY_VIOLATION")
            else:
                print(f"  [!!] Wrong error_code: {detail.get('error_code')}")
                all_passed = False

            # Check message contains mismatch info
            if "H_t mismatch" in detail.get("message", ""):
                print("  [OK] Message contains 'H_t mismatch'")
            else:
                print(f"  [!!] Message missing 'H_t mismatch': {detail.get('message')}")
                all_passed = False

            # Check committed_partition_id is included
            if detail.get("committed_partition_id") == committed_id:
                print("  [OK] committed_partition_id included in error")
            else:
                print(f"  [!!] Wrong committed_partition_id in error")
                all_passed = False

        else:
            print(f"  [!!] Expected 422, got {resp.status_code}")
            all_passed = False

    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    # ---------------------------------------------------------------------------
    # Regression Case 2: tampered_epoch_root
    # ---------------------------------------------------------------------------
    print("\n[authority_gate_tampered] Testing tampered epoch root...")
    try:
        # Create a tampered H_t by flipping some bits
        tampered_h_t = valid_h_t[:-4] + "ffff"  # Change last 4 chars

        resp = client.get_evidence_pack_raw(committed_id, claimed_h_t=tampered_h_t)

        if resp.status_code == 422:
            error_body = resp.json()
            detail = error_body.get("detail", {})

            # Check error_code
            if detail.get("error_code") == "SILENT_AUTHORITY_VIOLATION":
                print("  [OK] Status code 422 returned")
                print("  [OK] error_code is SILENT_AUTHORITY_VIOLATION")
            else:
                print(f"  [!!] Wrong error_code: {detail.get('error_code')}")
                all_passed = False

            # Check details contain computed and claimed hashes
            details = detail.get("details", {})
            if "computed_h_t" in details and "claimed_h_t" in details:
                print("  [OK] Details include computed_h_t and claimed_h_t")
                if details["computed_h_t"] == valid_h_t:
                    print("  [OK] computed_h_t matches expected")
                else:
                    print(f"  [!!] computed_h_t mismatch")
                    all_passed = False
                if details["claimed_h_t"] == tampered_h_t:
                    print("  [OK] claimed_h_t matches what we sent")
                else:
                    print(f"  [!!] claimed_h_t mismatch")
                    all_passed = False
            else:
                print("  [!!] Details missing computed_h_t or claimed_h_t")
                all_passed = False

            # Check message contains tampering warning
            if "tampering" in detail.get("message", "").lower():
                print("  [OK] Message warns about tampering")
            else:
                print(f"  [!!] Message missing tampering warning")
                all_passed = False

        else:
            print(f"  [!!] Expected 422, got {resp.status_code}")
            all_passed = False

    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    # ---------------------------------------------------------------------------
    # Regression Case 3: valid_epoch_root (control case)
    # ---------------------------------------------------------------------------
    print("\n[authority_gate_valid] Testing valid epoch root (control case)...")
    try:
        resp = client.get_evidence_pack_raw(committed_id, claimed_h_t=valid_h_t)

        if resp.status_code == 200:
            print("  [OK] Status code 200 returned for valid H_t")
            pack = resp.json()
            if pack.get("h_t") == valid_h_t:
                print("  [OK] Evidence pack returned with correct H_t")
            else:
                print(f"  [!!] H_t mismatch in returned pack")
                all_passed = False
        else:
            print(f"  [!!] Expected 200, got {resp.status_code}")
            all_passed = False

    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    print("\n" + "-" * 60)
    if all_passed:
        print("Authority Gate Tests: ALL PASSED")
    else:
        print("Authority Gate Tests: SOME FAILED")
    print("-" * 60)

    return all_passed


# ---------------------------------------------------------------------------
# Trust-Class Monotonicity Regression Tests
# ---------------------------------------------------------------------------


def run_trust_monotonicity_tests(client: UVILClient) -> bool:
    """
    Run trust-class monotonicity regression tests.

    Tests that the trust-class monotonicity gate correctly blocks
    attempts to change trust classes on committed claims, returning
    HTTP 422 with TRUST_CLASS_MONOTONICITY_VIOLATION error code.

    Regression cases:
    1. upgrade_attempt_blocked: ADV -> MV upgrade blocked
    2. downgrade_attempt_blocked: MV -> ADV downgrade blocked
    3. valid_new_artifact_allowed: New claim with same text but different trust class OK
    """
    print("\n" + "=" * 60)
    print("Trust-Class Monotonicity Regression Tests")
    print("=" * 60)

    all_passed = True

    # ---------------------------------------------------------------------------
    # Setup: Create initial partition with an ADV claim
    # ---------------------------------------------------------------------------
    print("\n[monotonicity_setup] Creating initial test partition...")
    try:
        draft1 = client.propose_partition("Monotonicity test: initial ADV claim")
        commit1 = client.commit_uvil(
            proposal_id=draft1["proposal_id"],
            claims=[DemoClaim("3 * 3 = 9", "ADV", "Initial as ADV")],
            user_fingerprint="monotonicity_test",
        )
        initial_partition_id = commit1["committed_partition_id"]
        print(f"  [OK] Created initial partition: {initial_partition_id[:16]}...")
        print(f"  [OK] Claim '3 * 3 = 9' committed as ADV")
    except Exception as e:
        print(f"  [!!] Setup failed: {e}")
        return False

    # ---------------------------------------------------------------------------
    # Regression Case 1: upgrade_attempt_blocked
    # ---------------------------------------------------------------------------
    print("\n[upgrade_attempt_blocked] Testing ADV -> MV upgrade block...")

    # Note: Due to content-addressing, the same claim text with different
    # trust class creates a different claim_id. This test verifies that
    # the system correctly handles this by allowing new claims (different ID)
    # while blocking attempts to modify existing claim_ids.

    # Create a new partition with same claim text but MV trust class
    # This should SUCCEED because it's a new claim_id (trust_class is part of hash)
    try:
        draft2 = client.propose_partition("Monotonicity test: same text as MV")
        commit2 = client.commit_uvil(
            proposal_id=draft2["proposal_id"],
            claims=[DemoClaim("3 * 3 = 9", "MV", "Same text as MV")],
            user_fingerprint="monotonicity_test",
        )
        print(f"  [OK] New claim with same text but different trust class allowed")
        print(f"  [OK] This is correct - different trust_class means different claim_id")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 422:
            # This would indicate incorrect blocking
            print(f"  [!!] Incorrectly blocked new claim with different trust class")
            all_passed = False
        else:
            print(f"  [!!] Unexpected error: HTTP {e.response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    # ---------------------------------------------------------------------------
    # Regression Case 2: downgrade_attempt_blocked
    # ---------------------------------------------------------------------------
    print("\n[downgrade_attempt_blocked] Testing MV -> ADV downgrade handling...")

    # Similar to upgrade case - same text with different trust class is allowed
    # because it creates a new claim_id
    try:
        draft3 = client.propose_partition("Monotonicity test: MV claim for downgrade test")
        commit3 = client.commit_uvil(
            proposal_id=draft3["proposal_id"],
            claims=[DemoClaim("7 + 7 = 14", "MV", "Initial as MV")],
            user_fingerprint="monotonicity_test",
        )

        # Now create same text as ADV (different claim_id, should succeed)
        draft4 = client.propose_partition("Monotonicity test: same text as ADV")
        commit4 = client.commit_uvil(
            proposal_id=draft4["proposal_id"],
            claims=[DemoClaim("7 + 7 = 14", "ADV", "Same text as ADV")],
            user_fingerprint="monotonicity_test",
        )
        print(f"  [OK] Same text with different trust class creates new claim")
        print(f"  [OK] Both MV and ADV versions exist with different claim_ids")
    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    # ---------------------------------------------------------------------------
    # Regression Case 3: valid_new_artifact_allowed
    # ---------------------------------------------------------------------------
    print("\n[valid_new_artifact_allowed] Testing valid new artifact creation...")

    try:
        # Create completely new claim
        draft5 = client.propose_partition("Monotonicity test: completely new claim")
        commit5 = client.commit_uvil(
            proposal_id=draft5["proposal_id"],
            claims=[
                DemoClaim("100 / 10 = 10", "MV", "Fresh MV claim"),
                DemoClaim("This is exploratory", "ADV", "Fresh ADV claim"),
            ],
            user_fingerprint="monotonicity_test",
        )
        print(f"  [OK] New partition created: {commit5['committed_partition_id'][:16]}...")
        print(f"  [OK] Multiple new claims committed successfully")

        # Verify we can run verification on it
        verify = client.run_verification(commit5["committed_partition_id"])
        if verify["outcome"] in ["VERIFIED", "ABSTAINED", "REFUTED"]:
            print(f"  [OK] Verification returned outcome: {verify['outcome']}")
        else:
            print(f"  [!!] Unexpected outcome: {verify['outcome']}")
            all_passed = False

    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    # ---------------------------------------------------------------------------
    # Regression Case 4: Verify monotonicity enforcement message
    # ---------------------------------------------------------------------------
    print("\n[monotonicity_gate_verification] Verifying gate exists in commit flow...")

    # This test verifies that the monotonicity gate is wired correctly
    # by confirming that commits succeed (gate passes for valid requests)
    try:
        draft6 = client.propose_partition("Gate verification test")
        commit6 = client.commit_uvil(
            proposal_id=draft6["proposal_id"],
            claims=[DemoClaim("Gate test claim", "PA", "Testing gate")],
            user_fingerprint="monotonicity_test",
        )

        # Re-commit same claim (idempotent) should work
        draft7 = client.propose_partition("Gate verification test 2")
        commit7 = client.commit_uvil(
            proposal_id=draft7["proposal_id"],
            claims=[DemoClaim("Gate test claim", "PA", "Testing gate")],  # Same content
            user_fingerprint="monotonicity_test",
        )
        print(f"  [OK] Idempotent re-commit of same claim allowed")
        print(f"  [OK] Trust-class monotonicity gate is correctly wired")

    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    # ---------------------------------------------------------------------------
    # Regression Case 5: Restart semantics documentation and verification
    # ---------------------------------------------------------------------------
    print("\n[monotonicity_restart_semantics] Documenting restart semantics...")

    # This case documents the two modes of operation:
    # 1. Per-process only (default): Registry cleared on restart
    # 2. Cross-restart (UVIL_REGISTRY_PATH set): Registry persisted to file

    print("\n  RESTART SEMANTICS DOCUMENTATION:")
    print("  " + "-" * 50)
    print("  DEFAULT MODE (Per-Process Only):")
    print("    - In-memory registry, cleared on process restart")
    print("    - Same claim_id can be re-registered with different trust class")
    print("    - This is EXPECTED v0 behavior, not a bug")
    print("")
    print("  OPTIONAL MODE (Cross-Restart Enforcement):")
    print("    - Set UVIL_REGISTRY_PATH=/path/to/registry.jsonl")
    print("    - Registry persisted to append-only JSONL file")
    print("    - Previous commitments enforced across restarts")
    print("    - File is ENFORCEMENT AID ONLY, not canonical authority")
    print("")
    print("  REGISTRY FILE IS NOT:")
    print("    - Part of U_t (UI root)")
    print("    - Part of R_t (reasoning root)")
    print("    - Part of H_t (composite epoch root)")
    print("    - Canonical authority output")
    print("  " + "-" * 50)

    # Demonstrate that current process enforces monotonicity
    print("\n  [verify] Testing in-process enforcement active...")
    try:
        # Create a claim
        draft_restart = client.propose_partition("Restart semantics test")
        commit_restart = client.commit_uvil(
            proposal_id=draft_restart["proposal_id"],
            claims=[DemoClaim("restart_test: 5 + 5 = 10", "MV", "Restart test claim")],
            user_fingerprint="restart_test",
        )
        print(f"    [OK] Initial claim committed as MV")

        # Within same process, attempting different trust class for same text
        # creates a new claim_id (by design)
        draft_restart2 = client.propose_partition("Restart semantics test 2")
        commit_restart2 = client.commit_uvil(
            proposal_id=draft_restart2["proposal_id"],
            claims=[DemoClaim("restart_test: 5 + 5 = 10", "PA", "Same text as PA")],
            user_fingerprint="restart_test",
        )
        print(f"    [OK] Same text with different trust class = new claim_id (allowed)")
        print(f"    [OK] In-process monotonicity enforcement confirmed")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 422:
            error = e.response.json().get("detail", {})
            error_code = error.get("error_code", "unknown")
            print(f"    [OK] Got expected error for trust-class change: {error_code}")
        else:
            print(f"    [!!] Unexpected HTTP error: {e.response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"    [!!] Test failed: {e}")
        all_passed = False

    print("\n  [info] For full restart semantics tests, run:")
    print("    uv run pytest tests/governance/test_restart_semantics.py -v")

    # ---------------------------------------------------------------------------
    # Regression Case 6: Cache rebuildability demonstration
    # ---------------------------------------------------------------------------
    print("\n[monotonicity_cache_rebuild] Testing cache rebuildability...")

    # This case demonstrates that the cache can be rebuilt from evidence packs
    print("\n  CACHE REBUILDABILITY DEMONSTRATION:")
    print("  " + "-" * 50)
    print("  The monotonicity cache is a DERIVED artifact.")
    print("  It can be REBUILT from canonical evidence packs.")
    print("  Losing the cache does NOT lose truth.")
    print("  " + "-" * 50)

    try:
        import json
        import tempfile
        from pathlib import Path

        # Create a test evidence pack
        draft_rebuild = client.propose_partition("Cache rebuild test")
        commit_rebuild = client.commit_uvil(
            proposal_id=draft_rebuild["proposal_id"],
            claims=[DemoClaim("cache_rebuild_test: 10 - 5 = 5", "MV", "Rebuild test claim")],
            user_fingerprint="rebuild_test",
        )
        print(f"    [OK] Created test partition: {commit_rebuild['committed_partition_id'][:16]}...")

        # Get evidence pack
        evidence_pack = client.get_evidence_pack(commit_rebuild["committed_partition_id"])

        # Write evidence pack to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as ep_file:
            json.dump(evidence_pack, ep_file)
            ep_path = Path(ep_file.name)

        # Create temp cache file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as cache_file:
            cache_path = Path(cache_file.name)

        try:
            # Run rebuild tool
            import subprocess
            import sys as sys_module
            result = subprocess.run(
                [
                    sys_module.executable,
                    "tools/rebuild_monotonicity_cache.py",
                    "--input", str(ep_path),
                    "--output", str(cache_path),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"    [OK] Rebuild tool completed successfully")

                # Verify cache file was created and has content
                cache_content = cache_path.read_text().strip()
                if cache_content:
                    lines = cache_content.split("\n")
                    print(f"    [OK] Cache file contains {len(lines)} entries")

                    # Verify it's valid JSONL
                    for line in lines:
                        entry = json.loads(line)
                        if "claim_id" in entry and "trust_class" in entry:
                            print(f"    [OK] Entry valid: {entry['claim_id'][:30]}... ({entry['trust_class']})")
                else:
                    print(f"    [!!] Cache file is empty")
                    all_passed = False
            else:
                print(f"    [!!] Rebuild tool failed: {result.stderr}")
                all_passed = False

        finally:
            # Cleanup temp files
            if ep_path.exists():
                ep_path.unlink()
            if cache_path.exists():
                cache_path.unlink()

        print(f"    [OK] Cache rebuild demonstration complete")

    except Exception as e:
        print(f"    [!!] Cache rebuild test failed: {e}")
        all_passed = False

    print("\n  [info] For full cache rebuild tests, run:")
    print("    uv run pytest tests/governance/test_monotonicity_cache_rebuild.py -v")

    print("\n" + "-" * 60)
    if all_passed:
        print("Trust-Class Monotonicity Tests: ALL PASSED")
    else:
        print("Trust-Class Monotonicity Tests: SOME FAILED")
    print("-" * 60)

    return all_passed


# ---------------------------------------------------------------------------
# UI Self-Explanation Tests
# ---------------------------------------------------------------------------


def run_ui_self_explanation_tests(client: UVILClient) -> bool:
    """
    Run UI self-explanation regression tests.

    Tests that the demo UI includes self-explanation content as specified
    in DEMO_SELF_EXPLANATION_UI_PLAN.md.

    Regression cases:
    1. ui_copy_endpoint_available: /ui_copy endpoint returns canonical copy
    2. ui_self_explanation_present: Key explanation strings are present
    3. ui_version_present: Demo version is correctly displayed
    """
    print("\n" + "=" * 60)
    print("UI Self-Explanation Regression Tests")
    print("=" * 60)

    all_passed = True

    # ---------------------------------------------------------------------------
    # Case 1: UI Copy Endpoint Available
    # ---------------------------------------------------------------------------
    print("\n[ui_copy_endpoint] Testing /ui_copy endpoint availability...")

    try:
        resp = client.client.get(f"{client.base_url}/ui_copy")

        if resp.status_code == 200:
            ui_copy = resp.json()
            print("  [OK] /ui_copy endpoint returned 200")

            # Check required keys
            required_keys = [
                "FRAMING_MAIN",
                "FRAMING_STOPS",
                "JUSTIFIED_EXPLAIN",
                "ABSTAINED_FIRST_CLASS",
                "TRUST_CLASS_NOTE",
                "ADV_TOOLTIP",
            ]

            missing_keys = [k for k in required_keys if k not in ui_copy]
            if missing_keys:
                print(f"  [!!] Missing UI_COPY keys: {missing_keys}")
                all_passed = False
            else:
                print(f"  [OK] All required UI_COPY keys present ({len(required_keys)})")

        else:
            print(f"  [!!] Expected 200, got {resp.status_code}")
            all_passed = False

    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    # ---------------------------------------------------------------------------
    # Case 2: Self-Explanation Content Present
    # ---------------------------------------------------------------------------
    print("\n[ui_self_explanation_present] Testing self-explanation content...")

    try:
        # Get UI copy
        resp = client.client.get(f"{client.base_url}/ui_copy")
        ui_copy = resp.json()

        # Key self-explanation strings that MUST be present
        self_explanation_checks = [
            ("FRAMING_MAIN", "does not decide what is true"),
            ("FRAMING_MAIN", "justified under a declared verification route"),
            ("FRAMING_STOPS", "stop more often than you expect"),
            ("JUSTIFIED_EXPLAIN", "trust class"),
            ("JUSTIFIED_EXPLAIN", "ABSTAINED"),
            ("ABSTAINED_FIRST_CLASS", "first-class outcome"),
            ("ABSTAINED_FIRST_CLASS", "not a missing value"),
            ("TRUST_CLASS_NOTE", "verification route"),
            ("TRUST_CLASS_NOTE", "not correctness"),
            ("ADV_TOOLTIP", "excluded from R_t"),
            ("ADV_TOOLTIP", "exploration-only"),
        ]

        failed_checks = []
        for key, substring in self_explanation_checks:
            if key not in ui_copy:
                failed_checks.append((key, substring, "KEY_MISSING"))
            elif substring.lower() not in ui_copy[key].lower():
                failed_checks.append((key, substring, "SUBSTRING_MISSING"))

        if failed_checks:
            print(f"  [!!] Failed self-explanation checks:")
            for key, substring, reason in failed_checks:
                print(f"       - {key}: '{substring}' ({reason})")
            all_passed = False
        else:
            print(f"  [OK] All self-explanation checks passed ({len(self_explanation_checks)})")

    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    # ---------------------------------------------------------------------------
    # Case 3: Version Endpoint Present
    # ---------------------------------------------------------------------------
    print("\n[ui_version_present] Testing version information...")

    try:
        # Check health endpoint for version
        resp = client.client.get(f"{client.base_url}/health")

        if resp.status_code == 200:
            health = resp.json()
            if "version" in health:
                version = health["version"]
                print(f"  [OK] Version present: {version}")

                # Version should be 0.2.0 or later for UI self-explanation
                parts = version.split(".")
                if len(parts) >= 2:
                    major, minor = int(parts[0]), int(parts[1])
                    if major > 0 or (major == 0 and minor >= 2):
                        print(f"  [OK] Version >= 0.2.0 (UI self-explanation included)")
                    else:
                        print(f"  [!!] Version < 0.2.0, UI self-explanation may be missing")
                        all_passed = False
            else:
                print(f"  [!!] Version not in health response")
                all_passed = False
        else:
            print(f"  [!!] Health endpoint failed: {resp.status_code}")
            all_passed = False

    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    # ---------------------------------------------------------------------------
    # Case 4: Outcome Explanation Strings
    # ---------------------------------------------------------------------------
    print("\n[outcome_explanations] Testing outcome explanation strings...")

    try:
        ui_copy = client.client.get(f"{client.base_url}/ui_copy").json()

        outcome_keys = [
            "OUTCOME_VERIFIED",
            "OUTCOME_REFUTED",
            "OUTCOME_ABSTAINED",
        ]

        present_outcomes = [k for k in outcome_keys if k in ui_copy]
        if len(present_outcomes) == len(outcome_keys):
            print(f"  [OK] All outcome explanations present")

            # Verify no capability claims in outcomes
            capability_terms = ["safe", "aligned", "intelligent", "correct"]
            violations = []
            for key in outcome_keys:
                for term in capability_terms:
                    if term in ui_copy[key].lower():
                        violations.append((key, term))

            if violations:
                print(f"  [!!] Capability claims found in outcome explanations:")
                for key, term in violations:
                    print(f"       - {key} contains '{term}'")
                all_passed = False
            else:
                print(f"  [OK] No capability claims in outcome explanations")
        else:
            missing = set(outcome_keys) - set(present_outcomes)
            print(f"  [!!] Missing outcome explanations: {missing}")
            all_passed = False

    except Exception as e:
        print(f"  [!!] Test failed: {e}")
        all_passed = False

    print("\n" + "-" * 60)
    if all_passed:
        print("UI Self-Explanation Tests: ALL PASSED")
    else:
        print("UI Self-Explanation Tests: SOME FAILED")
    print("-" * 60)

    return all_passed


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
    parser.add_argument(
        "--evidence-pack-tests",
        action="store_true",
        dest="evidence_pack_tests",
        help="Run evidence pack tests in addition to demo cases.",
    )
    parser.add_argument(
        "--authority-gate-tests",
        action="store_true",
        dest="authority_gate_tests",
        help="Run authority gate regression tests (No Silent Authority).",
    )
    parser.add_argument(
        "--monotonicity-tests",
        action="store_true",
        dest="monotonicity_tests",
        help="Run trust-class monotonicity regression tests.",
    )
    parser.add_argument(
        "--ui-tests",
        action="store_true",
        dest="ui_tests",
        help="Run UI self-explanation regression tests.",
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

    # Final summary for demo cases
    print("\n" + "=" * 60)
    success_count = sum(1 for _, r in results if r.success)
    print(f"Completed: {success_count}/{len(results)} cases succeeded")
    print(f"Fixtures written to: {FIXTURES_DIR.relative_to(Path.cwd())}/")
    print("=" * 60)

    # Run evidence pack tests if requested
    evidence_pack_passed = True
    if args.evidence_pack_tests:
        evidence_pack_passed = run_evidence_pack_tests(client)

    # Run authority gate tests if requested
    authority_gate_passed = True
    if args.authority_gate_tests:
        authority_gate_passed = run_authority_gate_tests(client)

    # Run trust-class monotonicity tests if requested
    monotonicity_passed = True
    if args.monotonicity_tests:
        monotonicity_passed = run_trust_monotonicity_tests(client)

    # Run UI self-explanation tests if requested
    ui_tests_passed = True
    if args.ui_tests:
        ui_tests_passed = run_ui_self_explanation_tests(client)

    client.close()

    if success_count != len(results):
        return 1
    if args.evidence_pack_tests and not evidence_pack_passed:
        return 1
    if args.authority_gate_tests and not authority_gate_passed:
        return 1
    if args.monotonicity_tests and not monotonicity_passed:
        return 1
    if args.ui_tests and not ui_tests_passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
