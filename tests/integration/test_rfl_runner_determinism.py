import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from backend.repro.determinism import deterministic_uuid
from rfl.config import RFLConfig, CurriculumSlice
from rfl.runner import RFLRunner
from rfl.experiment import ExperimentResult

class MockExperiment:
    def __init__(self, db_url, system_id):
        self.db_url = db_url
        self.system_id = system_id

    def run(self, run_id, seed=0, **kwargs):
        # Return a result that is purely a function of the seed and run_id
        # This mimics a deterministic experiment
        from backend.repro.determinism import deterministic_timestamp
        
        ts = deterministic_timestamp(seed).isoformat() + "Z"
        
        return ExperimentResult(
            run_id=run_id,
            system_id=self.system_id,
            start_time=ts,
            end_time=ts,
            duration_seconds=1.0,
            total_statements=10,
            successful_proofs=8,
            failed_proofs=1,
            abstentions=1,
            throughput_proofs_per_hour=100.0,
            mean_depth=2.0,
            max_depth=3,
            statement_hashes=[deterministic_uuid(f"{seed}-{i}") for i in range(10)],
            distinct_statements=10,
            derive_steps=kwargs.get("derive_steps", 10),
            max_breadth=kwargs.get("max_breadth", 10),
            max_total=kwargs.get("max_total", 10),
            status="success",
            policy_context=kwargs.get("policy_context", {}),
            abstention_breakdown={"mock_abstain": 1}
        )

@pytest.fixture
def rfl_config():
    temp_dir = tempfile.mkdtemp()
    
    slice_cfg = CurriculumSlice(
        name="test-slice",
        start_run=1,
        end_run=4,
        derive_steps=10,
        max_breadth=10,
        max_total=10,
        depth_max=2
    )
    
    config = RFLConfig(
        experiment_id="test_determinism",
        system_id=1,
        database_url="sqlite:///:memory:", # Not used by mock
        artifacts_dir=temp_dir,
        num_runs=4,
        curriculum=[slice_cfg],
        random_seed=42,
        coverage_threshold=0.9,
        uplift_threshold=1.0
    )
    yield config
    shutil.rmtree(temp_dir)

def test_rfl_runner_determinism(rfl_config):
    """
    Verify that RFLRunner produces byte-for-byte identical artifacts 
    when run twice with the same config and seed, assuming the experiment 
    executor is deterministic (which we verified by passing the seed).
    """
    
    # Run 1
    with patch("rfl.runner.RFLExperiment", side_effect=MockExperiment):
        runner1 = RFLRunner(rfl_config)
        results1 = runner1.run_all()
    
    # Read artifacts from Run 1
    results_path = Path(rfl_config.artifacts_dir) / rfl_config.results_file
    with open(results_path, "rb") as f:
        bytes1 = f.read()
        
    # Clear artifacts (optional, but good for sanity)
    # results_path.unlink() 
    
    # Run 2
    with patch("rfl.runner.RFLExperiment", side_effect=MockExperiment):
        runner2 = RFLRunner(rfl_config)
        results2 = runner2.run_all()
        
    # Read artifacts from Run 2
    with open(results_path, "rb") as f:
        bytes2 = f.read()
        
    # Assert identical results structure
    # (We can't assert exact dict equality of results1/2 easily because of memory addresses, 
    # but the JSON output IS the contract).
    
    # 1. Byte-for-byte identity of the output JSON
    assert bytes1 == bytes2, "RFL results.json is not byte-for-byte identical across runs"
    
    # 2. Verify that the "seed" was actually used effectively (timestamps should be deterministic)
    data = json.loads(bytes1)
    assert len(data["runs"]) == 4
    run0 = data["runs"][0]
    # Check that start_time looks like our deterministic timestamp (2025-01-01 epoch + offset)
    assert "2025-01-01" in run0["start_time"]
    
    print(f"\n[PASS] RFLRunner produced identical {len(bytes1)} bytes across 2 runs.")

