import pytest
import os
import json
import yaml
import hashlib
import datetime
from pathlib import Path
from typing import Callable, Optional

# Add project root to sys.path to allow imports from backend
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.promotion.u2_evidence import assemble_dossier
from backend.promotion.admissibility_engine import check_admissibility, compute_dossier_hash, GOVV_REQUIRED_GATES, REQUIRED_ENVIRONMENTS

@pytest.fixture
def dossier_factory(tmp_path: Path) -> Callable:
    """A factory that creates a dossier in a temporary directory, with optional tampering."""
    
    def _create_dossier(run_id: str = "run_20251206_gold", tamper_func: Optional[Callable] = None):
        root = tmp_path / run_id
        
        # Create a full, valid set of artifacts
        (root / "docs" / "prereg").mkdir(parents=True, exist_ok=True)
        (root / "artifacts" / "u2" / "analysis").mkdir(parents=True, exist_ok=True)
        (root / "ops" / "logs").mkdir(parents=True, exist_ok=True)
        gov_dir = root / "artifacts" / "u2"; gov_dir.mkdir(exist_ok=True)
        
        env_dirs = {}
        for env in REQUIRED_ENVIRONMENTS:
            (root / "artifacts" / "u2" / env).mkdir(exist_ok=True)
            env_dirs[env] = root / "artifacts" / "u2" / env

        # A1 & A2
        a1_path = root / "docs/prereg/PREREG_UPLIFT_U2.yaml"
        a1_path.write_text(yaml.dump({"experiments": [{"experiment_id": "U2_EXP_GOLD"}]}))
        with open(a1_path, "rb") as f: a1_hash = hashlib.sha256(f.read()).hexdigest()
        (root / "docs/prereg/PREREG_UPLIFT_U2.yaml.sig").write_text(f"sha256:{a1_hash}\nsealed_at:2025-12-01T00:00:00Z")

        # A3 & A4
        for env, env_dir in env_dirs.items():
            (env_dir / f"telemetry_{run_id}.jsonl").write_text(f'{{"run_id":"{run_id}"}}\n')
            manifest_base = {"run_id": run_id, "environment_id": env, "runs": [{"started_at": "2025-12-02T00:00:00Z", "artifacts": []}]}
            canonical_json = json.dumps(manifest_base, sort_keys=True, separators=( "," , ":" )).encode('utf-8')
            manifest_hash = hashlib.sha256(canonical_json).hexdigest()
            manifest_final = manifest_base | {"manifest_hash": manifest_hash}
            (env_dir / "manifest.json").write_text(json.dumps(manifest_final))

        # A5, A6, A7
        (root / f"artifacts/u2/analysis/statistical_summary_{run_id}.json").write_text(json.dumps({"run_id": run_id}))
        for env in REQUIRED_ENVIRONMENTS: (root / f"artifacts/u2/analysis/uplift_curve_{env}_run_{run_id}.png").touch()
        (root / f"ops/logs/u2_compliance.jsonl").write_text(json.dumps({"run_id": run_id, "gate_evaluation": {}}))
        
        # Tamper if requested for a specific test case
        if tamper_func:
            tamper_func(root)

        # A8 (must be created last, based on the state of other artifacts)
        temp_dossier = assemble_dossier(run_id, root_path=str(root))
        dossier_hash_for_a8 = compute_dossier_hash(temp_dossier, exclude_a8=True)
        a8_content = {"report_id": "gv-report-gold", "verified_at": "2025-12-05T00:00:00Z", "verifier_version": "1.0.0", "dossier_hash": dossier_hash_for_a8, "gates": {g: {"status": "PASS"} for g in GOVV_REQUIRED_GATES}, "overall_status": "PASS", "artifact_inventory": {"found": [a.id for a in temp_dossier.artifacts.values() if a.found], "missing": [a.id for a in temp_dossier.artifacts.values() if not a.found]}, "signature": "placeholder-sig"}
        (gov_dir / "governance_verifier_report.json").write_text(json.dumps(a8_content))
        
        # Final assembly for the test
        return assemble_dossier(run_id, root_path=str(root))
    
    return _create_dossier

def test_admissible_dossier(dossier_factory: Callable):
    dossier = dossier_factory()
    report = check_admissibility(dossier)
    assert report.verdict.status == "ADMISSIBLE"

def test_maas_failure(dossier_factory: Callable):
    def tamper(root: Path):
        os.remove(root / "docs" / "prereg" / "PREREG_UPLIFT_U2.yaml")
    dossier = dossier_factory(tamper_func=tamper)
    report = check_admissibility(dossier)
    assert report.verdict.status == "NOT_ADMISSIBLE"
    assert report.verdict.code == "HE-S1"

def test_dossier_hash_mismatch_failure(dossier_factory: Callable):
    """GIVEN a file is tampered with after the A8 report is generated, WHEN adjudicated, THEN it fails with HE-GV4."""
    # ARRANGE
    # The factory generates a consistent dossier first
    dossier = dossier_factory() 
    
    # ACT
    # Now, tamper with a file on disk *after* the valid A8 was created.
    a5_path_str = dossier.artifacts['A5'].path
    Path(a5_path_str).write_text('{"run_id":"tampered"}')

    # Re-run the check with the same dossier object. 
    # The engine's internal `compute_dossier_hash` will now produce a different hash
    # than the one stored in the A8 file, because it re-scans the files.
    # To make this work, we need to ensure compute_dossier_hash re-reads from disk.
    # The current implementation of compute_dossier_hash uses the paths from the dossier object, not re-scans.
    # This means we need to re-assemble the dossier to get a new object with the new paths.
    
    run_id = dossier.run_id
    root_path = Path(a5_path_str).parents[3] # get the root path from the artifact path
    
    tampered_dossier = assemble_dossier(run_id, root_path=str(root_path))
    
    report = check_admissibility(tampered_dossier)

    # ASSERT
    assert report.verdict.status == "NOT_ADMISSIBLE"
    assert report.verdict.code == "HE-GV4"
    
    # =============================================================================
    # TESTS FOR DASHBOARD / HISTORY FUNCTIONS
    # =============================================================================
    
    def test_snapshot_creation():
        """Tests the build_admissibility_snapshot function for both verdicts."""
        from backend.promotion.admissibility_engine import build_admissibility_snapshot, AdmissibilityReport, AdmissibilityVerdict, AdmissibilityError
    
        # Case 1: ADMISSIBLE
        admissible_report = AdmissibilityReport(
            decision_id="uuid-1", executed_at="2025-01-01T00:00:00Z",
            verdict=AdmissibilityVerdict(status="ADMISSIBLE"),
            phases_executed={"phase_1_governance": {"executed": True, "passed": True}}
        )
        snapshot_admissible = build_admissibility_snapshot(admissible_report)
        assert snapshot_admissible["verdict"] == "ADMISSIBLE"
        assert snapshot_admissible["error_codes"] == []
    
        # Case 2: NOT_ADMISSIBLE
        inadmissible_report = AdmissibilityReport(
            decision_id="uuid-2", executed_at="2025-01-02T00:00:00Z",
            verdict=AdmissibilityVerdict(status="NOT_ADMISSIBLE", code="HE-S1"),
            errors=[AdmissibilityError("HE-S1", "", "", 2), AdmissibilityError("DOSSIER-9", "", "", 5)],
            phases_executed={"phase_1_governance": {"executed": True, "passed": True}, "phase_2_artifacts": {"executed": True, "passed": False}}
        )
        snapshot_inadmissible = build_admissibility_snapshot(inadmissible_report)
        assert snapshot_inadmissible["verdict"] == "NOT_ADMISSIBLE"
        assert snapshot_inadmissible["error_codes"] == ["DOSSIER-9", "HE-S1"]
    
    def test_timeline_builder():
        """Tests the build_admissibility_history function."""
        from backend.promotion.admissibility_engine import build_admissibility_history
        
        snapshots = [
            {"verdict": "ADMISSIBLE", "phase_failures": []},
            {"verdict": "NOT_ADMISSIBLE", "phase_failures": ["HE-S1"]},
            {"verdict": "ADMISSIBLE", "phase_failures": []},
            {"verdict": "NOT_ADMISSIBLE", "phase_failures": ["DOSSIER-11"]},
            {"verdict": "NOT_ADMISSIBLE", "phase_failures": ["HE-S1", "DOSSIER-9"]},
        ]
        history = build_admissibility_history(snapshots)
        assert history["total_cases"] == 5
        assert history["admissibility_rate"] == 0.4
        assert history["top_recurrent_blockers"] == {"HE-S1": 2, "DOSSIER-11": 1, "DOSSIER-9": 1}
    
    def test_global_health_summary():
        """Tests the summarize_admissibility_for_global_health function."""
        from backend.promotion.admissibility_engine import summarize_admissibility_for_global_health
        admissible_snapshot = {"verdict": "ADMISSIBLE", "phase_failures": []}
        summary_ok = summarize_admissibility_for_global_health(admissible_snapshot)
        assert summary_ok["status"] == "OK"
        inadmissible_snapshot = {"verdict": "NOT_ADMISSIBLE", "phase_failures": ["HE-GV4"]}
        summary_blocked = summarize_admissibility_for_global_health(inadmissible_snapshot)
        assert summary_blocked["status"] == "BLOCKED"
        assert summary_blocked["blocking_reasons"] == ["HE-GV4"]