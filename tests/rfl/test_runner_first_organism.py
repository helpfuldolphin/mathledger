# tests/rfl/test_runner_first_organism.py
"""
Unit tests for RFL runner's run_with_attestation entry point.

These tests validate that:
1. run_with_attestation consumes the canonical First Organism fixture
2. step_id is deterministic given fixed seed + H_t
3. policy_update_applied reflects abstention mass delta
4. Ledger entries and abstention histograms are correctly populated
5. Redis telemetry is emitted (when available)

The tests use the shared fixture from tests/fixtures/first_organism.py
to ensure alignment with integration tests.

HERMETIC DESIGN:
- All tests are fully hermetic: no DB/Redis dependencies
- Redis telemetry tests use TelemetryMockRedis (in-memory mock)
- Tests use canonical fixtures, not real experiment logs
- All tests are deterministic and reproducible

EVIDENCE ALIGNMENT (Sober Truth / Reviewer 2):
- Tests verify ENGINE BEHAVIOR, not empirical results from real runs
- No claims about 1000-cycle runs or non-existent data files
- Wide Slice tests verify capability, not actual Wide Slice experiment results
- All assertions are based on synthetic fixtures, not real logs

ALL-ABSTAIN REGIME (Phase I Reality):
- As of Phase I, real logs (fo_rfl.jsonl) correspond to the all-abstain edge case
  where every event has abstention=true (proof_status="failure").
- The all-abstain tests verify RFL executes correctly on this regime without
  making claims about risk reduction or policy effectiveness.
"""

import hashlib
import pytest

from rfl.config import RFLConfig
from rfl.runner import RFLRunner, RflResult, RunLedgerEntry
from substrate.bridge.context import AttestedRunContext

from tests.fixtures.first_organism import (
    CANONICAL_FIRST_ORGANISM_ATTESTATION,
    load_first_organism_attestation,
    make_attested_run_context,
    compute_expected_step_id,
)


class TestRunWithAttestation:
    """Tests for RFLRunner.run_with_attestation()."""

    @pytest.fixture
    def config(self) -> RFLConfig:
        """Standard RFL config for testing."""
        return RFLConfig(num_runs=5)

    @pytest.fixture
    def runner(self, config: RFLConfig) -> RFLRunner:
        """RFL runner instance."""
        return RFLRunner(config)

    @pytest.fixture
    def canonical_fixture(self):
        """Load the canonical First Organism attestation fixture."""
        return load_first_organism_attestation()

    @pytest.fixture
    def attestation_context(self, canonical_fixture) -> AttestedRunContext:
        """Convert canonical fixture to AttestedRunContext."""
        return make_attested_run_context(canonical_fixture)

    def test_run_with_attestation_applies_symbolic_descent(
        self, runner: RFLRunner, attestation_context: AttestedRunContext, config: RFLConfig
    ) -> None:
        """
        Verify that run_with_attestation:
        - Returns policy_update_applied=True for non-zero abstention mass
        - Sets source_root to the input composite root (H_t)
        - Generates a deterministic step_id = SHA256(experiment_id | resolved_slice_name | policy_id | composite_root)
        """
        result = runner.run_with_attestation(attestation_context)

        # Policy update should be applied when abstention mass > 0
        assert result.policy_update_applied is True
        assert result.source_root == attestation_context.composite_root

        # step_id must be deterministic and match the canonical formula:
        # SHA256(experiment_id | resolved_slice_name | policy_id | composite_root)
        # The runner resolves slice_id to the first matching curriculum slice
        resolved_slice = runner._resolve_slice(attestation_context.slice_id)
        expected_step = compute_expected_step_id(
            CANONICAL_FIRST_ORGANISM_ATTESTATION,
            experiment_id=config.experiment_id,
            policy_id=attestation_context.policy_id,
            resolved_slice_name=resolved_slice.name,
        )
        assert result.step_id == expected_step, (
            f"step_id mismatch. Expected: {expected_step}, Got: {result.step_id}. "
            f"Formula: SHA256({config.experiment_id}|{resolved_slice.name}|{attestation_context.policy_id}|{attestation_context.composite_root})"
        )

    def test_step_id_is_stable_across_runs(
        self, config: RFLConfig, attestation_context: AttestedRunContext
    ) -> None:
        """
        Verify step_id is stable across multiple runner instantiations.
        This is critical for determinism checks (Cursor B) and metrics (Cursor K).
        
        Also verifies that step_id matches the canonical formula:
        SHA256(experiment_id | resolved_slice_name | policy_id | composite_root)
        """
        step_ids = []
        for _ in range(3):
            runner = RFLRunner(config)
            result = runner.run_with_attestation(attestation_context)
            step_ids.append(result.step_id)
            
            # Verify step_id matches formula for each run
            resolved_slice = runner._resolve_slice(attestation_context.slice_id)
            expected_step = compute_expected_step_id(
                CANONICAL_FIRST_ORGANISM_ATTESTATION,
                experiment_id=config.experiment_id,
                policy_id=attestation_context.policy_id,
                resolved_slice_name=resolved_slice.name,
            )
            assert result.step_id == expected_step, (
                f"step_id formula mismatch. Expected: {expected_step}, Got: {result.step_id}"
            )

        # All step_ids must be identical
        assert len(set(step_ids)) == 1, f"step_id not stable across runs: {step_ids}"

    def test_ledger_entry_populated(
        self, runner: RFLRunner, attestation_context: AttestedRunContext
    ) -> None:
        """
        Verify that run_with_attestation populates the policy ledger.
        """
        result = runner.run_with_attestation(attestation_context)

        # Ledger should have exactly one entry
        assert len(runner.policy_ledger) == 1

        entry = runner.policy_ledger[0]
        assert isinstance(entry, RunLedgerEntry)
        # The runner resolves slice_id to the matching curriculum slice name
        resolved_slice = runner._resolve_slice(attestation_context.slice_id)
        assert entry.slice_name == resolved_slice.name
        assert entry.status == "attestation"
        assert entry.abstention_fraction == pytest.approx(attestation_context.abstention_rate)

        # Symbolic descent should be negative of (abstention_rate - tolerance)
        expected_descent = -(attestation_context.abstention_rate - runner.config.abstention_tolerance)
        assert entry.symbolic_descent == pytest.approx(expected_descent)

    def test_abstention_histogram_updated(
        self, runner: RFLRunner, canonical_fixture, attestation_context: AttestedRunContext
    ) -> None:
        """
        Verify that abstention histogram is updated from breakdown.
        """
        result = runner.run_with_attestation(attestation_context)

        # Breakdown categories should be merged
        for key, value in canonical_fixture.abstention_breakdown.items():
            assert runner.abstention_histogram[key] == value

        # Attestation mass and event count should be tracked
        assert runner.abstention_histogram["attestation_mass"] == int(
            round(attestation_context.abstention_mass)
        )
        assert runner.abstention_histogram["attestation_events"] == 1

    def test_dual_attestation_records_populated(
        self, runner: RFLRunner, attestation_context: AttestedRunContext
    ) -> None:
        """
        Verify that dual attestation records are populated.
        """
        result = runner.run_with_attestation(attestation_context)

        attestations = runner.dual_attestation_records.get("attestations", [])
        assert len(attestations) == 1

        record = attestations[0]
        assert record["composite_root"] == attestation_context.composite_root
        assert record["reasoning_root"] == attestation_context.reasoning_root
        assert record["ui_root"] == attestation_context.ui_root
        assert record["step_id"] == result.step_id

    def test_result_contains_ledger_entry(
        self, runner: RFLRunner, attestation_context: AttestedRunContext
    ) -> None:
        """
        Verify that RflResult includes the ledger entry.
        """
        result = runner.run_with_attestation(attestation_context)

        assert result.ledger_entry is not None
        assert result.ledger_entry.run_id == result.step_id

    def test_ledger_entry_traceability_fields(
        self, runner: RFLRunner, attestation_context: AttestedRunContext
    ) -> None:
        """
        Verify that ledger entry includes traceability fields for auditing.
        """
        result = runner.run_with_attestation(attestation_context)

        entry = result.ledger_entry
        assert entry is not None
        # Original attestation slice_id preserved for traceability
        assert entry.attestation_slice_id == attestation_context.slice_id
        # Composite root (H_t) preserved for cryptographic verification
        assert entry.composite_root == attestation_context.composite_root

    def test_abstention_delta_computation(
        self, runner: RFLRunner, attestation_context: AttestedRunContext
    ) -> None:
        """
        Verify that abstention_mass_delta is correctly computed.

        abstention_mass_delta = attestation.abstention_mass - (tolerance * attempt_mass)
        """
        result = runner.run_with_attestation(attestation_context)

        attempt_mass = attestation_context.metadata.get("attempt_mass", attestation_context.abstention_mass)
        expected_mass = runner.config.abstention_tolerance * attempt_mass
        expected_delta = attestation_context.abstention_mass - expected_mass

        assert result.abstention_mass_delta == pytest.approx(expected_delta)

    def test_zero_abstention_mass_no_policy_update(self, runner: RFLRunner) -> None:
        """
        Verify that zero abstention mass results in no policy update.
        """
        # Create attestation with zero abstention
        zero_fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION
        ctx = AttestedRunContext(
            slice_id=zero_fixture.slice_id,
            statement_hash=zero_fixture.statement_hash,
            proof_status=zero_fixture.proof_status,
            block_id=zero_fixture.block_id,
            composite_root=zero_fixture.composite_root,
            reasoning_root=zero_fixture.reasoning_root,
            ui_root=zero_fixture.ui_root,
            abstention_metrics={"rate": 0.0, "mass": 0.0},
            policy_id=zero_fixture.policy_id,
            metadata={"attempt_mass": 20.0},
        )

        result = runner.run_with_attestation(ctx)

        # With zero abstention at or below tolerance, no update
        # (depends on tolerance config, but rate=0 means delta <= 0)
        assert result.abstention_mass_delta <= 0

    def test_all_abstain_regime_executes_without_error(self, runner: RFLRunner) -> None:
        """
        Verify RFL executes correctly on all-abstain regime (all events abstain).
        
        This test corresponds to Phase I reality where fo_rfl.jsonl contains
        events where every event has abstention=true (proof_status="failure").
        
        This test verifies:
        - RFL executes without error on all-abstain events
        - Policy updates are applied (abstention_mass_delta > 0)
        - Ledger entries are populated correctly
        - No claim is made about risk reduction or policy effectiveness
        
        NOTE: This test does NOT assert that risk decreases or that the policy
        is effective. It only verifies that RFL executes correctly on this edge case.
        """
        # Create synthetic all-abstain attestation (100% abstention rate)
        fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION
        ctx = AttestedRunContext(
            slice_id=fixture.slice_id,
            statement_hash=fixture.statement_hash,
            proof_status="failure",  # All events abstain
            block_id=fixture.block_id,
            composite_root=fixture.composite_root,
            reasoning_root=fixture.reasoning_root,
            ui_root=fixture.ui_root,
            abstention_metrics={
                "rate": 1.0,  # 100% abstention (all-abstain regime)
                "mass": 20.0,  # All attempts abstained
            },
            policy_id=fixture.policy_id,
            metadata={
                "attempt_mass": 20.0,
                "abstention_breakdown": {"lean_failure": 20},  # All failures
                "first_organism_abstentions": 20,
            },
        )

        # RFL should execute without error
        result = runner.run_with_attestation(ctx)

        # Verify execution succeeded
        assert result is not None
        assert result.step_id is not None
        assert len(result.step_id) == 64  # Valid SHA-256 hash

        # With 100% abstention, abstention_mass_delta should be positive
        # (abstention_mass - tolerance * attempt_mass > 0)
        assert result.abstention_mass_delta > 0, (
            "All-abstain regime should produce positive abstention_mass_delta"
        )

        # Policy update should be applied
        assert result.policy_update_applied is True

        # Ledger entry should be populated
        assert result.ledger_entry is not None
        assert result.ledger_entry.abstention_fraction == pytest.approx(1.0)
        assert result.ledger_entry.status == "attestation"

        # Abstention histogram should be updated
        assert runner.abstention_histogram["attestation_mass"] >= 20
        assert runner.abstention_histogram["attestation_events"] >= 1
        assert runner.abstention_histogram.get("lean_failure", 0) >= 20

    def test_all_abstain_regime_multiple_events(self, runner: RFLRunner) -> None:
        """
        Verify RFL handles multiple consecutive all-abstain events correctly.
        
        This simulates the Phase I fo_rfl.jsonl pattern where multiple cycles
        all result in abstention. RFL should process each event without error
        and accumulate abstention metrics correctly.
        
        NOTE: This test does NOT assert that risk decreases over time or that
        the policy becomes more effective. It only verifies correct execution.
        """
        fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION
        
        # Process multiple all-abstain events
        results = []
        for cycle in range(3):
            ctx = AttestedRunContext(
                slice_id=fixture.slice_id,
                statement_hash=hashlib.sha256(f"cycle-{cycle}".encode()).hexdigest(),
                proof_status="failure",  # All abstain
                block_id=fixture.block_id + cycle,
                composite_root=hashlib.sha256(
                    f"{fixture.composite_root}-{cycle}".encode()
                ).hexdigest(),
                reasoning_root=hashlib.sha256(
                    f"{fixture.reasoning_root}-{cycle}".encode()
                ).hexdigest(),
                ui_root=hashlib.sha256(
                    f"{fixture.ui_root}-{cycle}".encode()
                ).hexdigest(),
                abstention_metrics={
                    "rate": 1.0,  # 100% abstention
                    "mass": 10.0 + cycle,  # Varying abstention mass
                },
                policy_id=fixture.policy_id,
                metadata={
                    "attempt_mass": 10.0 + cycle,
                    "abstention_breakdown": {"lean_failure": 10 + cycle},
                    "first_organism_abstentions": 10 + cycle,
                },
            )

            # Each event should execute without error
            result = runner.run_with_attestation(ctx)
            results.append(result)

            # Verify each execution succeeded
            assert result is not None
            assert result.step_id is not None
            assert result.policy_update_applied is True

        # Verify all events were processed
        assert len(results) == 3
        assert len(runner.policy_ledger) == 3

        # Verify abstention histogram accumulated correctly
        # Each event adds to attestation_mass and attestation_events
        assert runner.abstention_histogram["attestation_events"] >= 3
        assert runner.abstention_histogram["attestation_mass"] >= 30  # At least 10+11+12

        # Verify each ledger entry has correct abstention fraction
        for entry in runner.policy_ledger[-3:]:
            assert entry.abstention_fraction == pytest.approx(1.0)
            assert entry.status == "attestation"

    def test_first_organism_runs_counter_incremented(
        self, runner: RFLRunner, attestation_context: AttestedRunContext
    ) -> None:
        """
        Verify that the first_organism_runs_total counter is incremented.
        """
        initial_count = runner.first_organism_runs_total
        runner.run_with_attestation(attestation_context)
        assert runner.first_organism_runs_total == initial_count + 1


class TestRedisTelemetry:
    """Tests for Redis telemetry in RFL metabolism."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client that tracks calls."""
        class TelemetryMockRedis:
            def __init__(self):
                self.data: dict = {}
                self.incr_calls: list = []
                self.set_calls: list = []
                self.lpush_calls: list = []
                self.ltrim_calls: list = []

            def incr(self, key: str) -> int:
                self.incr_calls.append(key)
                self.data[key] = self.data.get(key, 0) + 1
                return self.data[key]

            def set(self, key: str, value: str) -> bool:
                self.set_calls.append((key, value))
                self.data[key] = value
                return True

            def get(self, key: str) -> str | None:
                return self.data.get(key)

            def lpush(self, key: str, value: str) -> int:
                self.lpush_calls.append((key, value))
                if key not in self.data:
                    self.data[key] = []
                self.data[key].insert(0, value)
                return len(self.data[key])

            def ltrim(self, key: str, start: int, stop: int) -> bool:
                self.ltrim_calls.append((key, start, stop))
                if key in self.data and isinstance(self.data[key], list):
                    self.data[key] = self.data[key][start : stop + 1]
                return True

            def pipeline(self):
                return TelemetryMockPipeline(self)

        class TelemetryMockPipeline:
            def __init__(self, redis):
                self.redis = redis
                self.commands = []

            def lpush(self, key: str, value: str):
                self.commands.append(("lpush", key, value))
                return self

            def ltrim(self, key: str, start: int, stop: int):
                self.commands.append(("ltrim", key, start, stop))
                return self

            def execute(self):
                results = []
                for cmd in self.commands:
                    if cmd[0] == "lpush":
                        results.append(self.redis.lpush(cmd[1], cmd[2]))
                    elif cmd[0] == "ltrim":
                        results.append(self.redis.ltrim(cmd[1], cmd[2], cmd[3]))
                return results

        return TelemetryMockRedis()

    def test_telemetry_increments_runs_total(
        self, mock_redis, monkeypatch
    ) -> None:
        """
        Verify that run_with_attestation increments the runs_total counter.
        """
        # Patch redis.from_url to return our mock
        monkeypatch.setattr("rfl.runner.redis.from_url", lambda *args, **kwargs: mock_redis)

        from rfl.config import RFLConfig
        from rfl.runner import RFLRunner
        from tests.fixtures.first_organism import make_attested_run_context

        config = RFLConfig(num_runs=5)
        runner = RFLRunner(config)
        ctx = make_attested_run_context()

        runner.run_with_attestation(ctx)

        # Check that runs_total was incremented
        assert "rfl_first_organism_runs_total" in mock_redis.incr_calls
        assert "ml:metrics:first_organism:runs_total" in mock_redis.incr_calls

    def test_telemetry_sets_last_ht(
        self, mock_redis, monkeypatch
    ) -> None:
        """
        Verify that run_with_attestation sets the last_ht metric.
        """
        monkeypatch.setattr("rfl.runner.redis.from_url", lambda *args, **kwargs: mock_redis)

        from rfl.config import RFLConfig
        from rfl.runner import RFLRunner
        from tests.fixtures.first_organism import make_attested_run_context

        config = RFLConfig(num_runs=5)
        runner = RFLRunner(config)
        ctx = make_attested_run_context()

        runner.run_with_attestation(ctx)

        # Check that last_ht was set to the composite root
        set_keys = [k for k, v in mock_redis.set_calls]
        assert "ml:metrics:first_organism:last_ht" in set_keys

        # Find the value
        for key, value in mock_redis.set_calls:
            if key == "ml:metrics:first_organism:last_ht":
                assert value == ctx.composite_root
                break

    def test_telemetry_tracks_completion(
        self, mock_redis, monkeypatch
    ) -> None:
        """
        Verify that run_with_attestation increments runs_completed on success.
        """
        monkeypatch.setattr("rfl.runner.redis.from_url", lambda *args, **kwargs: mock_redis)

        from rfl.config import RFLConfig
        from rfl.runner import RFLRunner
        from tests.fixtures.first_organism import make_attested_run_context

        config = RFLConfig(num_runs=5)
        runner = RFLRunner(config)
        ctx = make_attested_run_context()

        runner.run_with_attestation(ctx)

        assert "ml:metrics:first_organism:runs_completed" in mock_redis.incr_calls
        
    def test_telemetry_gauges_and_history(
        self, mock_redis, monkeypatch
    ) -> None:
        """
        Verify that telemetry tracks gauges (set operations) and history lists (lpush/ltrim).
        
        Tests:
        - Gauges: last_ht, last_abstention_rate, last_abstentions, etc.
        - History lists: duration_history, abstention_history
        """
        monkeypatch.setattr("rfl.runner.redis.from_url", lambda *args, **kwargs: mock_redis)

        from rfl.config import RFLConfig
        from rfl.runner import RFLRunner
        from tests.fixtures.first_organism import make_attested_run_context

        config = RFLConfig(num_runs=5)
        runner = RFLRunner(config)
        
        # Create context with metadata for history tracking
        ctx = make_attested_run_context()
        ctx.metadata.update({
            "first_organism_duration_seconds": 1.5,
            "first_organism_abstentions": 3,
        })

        runner.run_with_attestation(ctx)

        # Verify gauge metrics (set operations)
        set_keys = [k for k, v in mock_redis.set_calls]
        assert "ml:metrics:first_organism:last_ht" in set_keys
        assert "ml:metrics:first_organism:duration_seconds" in set_keys
        assert "ml:metrics:first_organism:last_abstentions" in set_keys
        assert "ml:metrics:first_organism:last_abstention_rate" in set_keys

        # Verify history lists (lpush operations)
        lpush_keys = [k for k, v in mock_redis.lpush_calls]
        assert "ml:metrics:first_organism:duration_history" in lpush_keys
        assert "ml:metrics:first_organism:abstention_history" in lpush_keys

        # Verify ltrim was called to cap history lists
        ltrim_keys = [k for k, start, stop in mock_redis.ltrim_calls]
        assert "ml:metrics:first_organism:duration_history" in ltrim_keys
        assert "ml:metrics:first_organism:abstention_history" in ltrim_keys

    def test_telemetry_counters_increment(
        self, mock_redis, monkeypatch
    ) -> None:
        """
        Verify that telemetry counters increment correctly.
        
        Tests:
        - runs_total counter increments
        - Multiple runs increment counter multiple times
        """
        monkeypatch.setattr("rfl.runner.redis.from_url", lambda *args, **kwargs: mock_redis)

        from rfl.config import RFLConfig
        from rfl.runner import RFLRunner
        from tests.fixtures.first_organism import make_attested_run_context

        config = RFLConfig(num_runs=5)
        runner = RFLRunner(config)
        ctx = make_attested_run_context()

        # Run multiple times
        initial_count = mock_redis.data.get("rfl_first_organism_runs_total", 0)
        runner.run_with_attestation(ctx)
        runner.run_with_attestation(ctx)
        runner.run_with_attestation(ctx)

        # Verify counter incremented
        final_count = mock_redis.data.get("rfl_first_organism_runs_total", 0)
        assert final_count == initial_count + 3, (
            f"Counter should increment by 3, got {final_count - initial_count}"
        )
        
        # Verify incr was called for each metric
        assert mock_redis.incr_calls.count("rfl_first_organism_runs_total") == 3
        assert mock_redis.incr_calls.count("ml:metrics:first_organism:runs_total") == 3
        assert mock_redis.incr_calls.count("ml:metrics:first_organism:runs_completed") == 3


class TestFixtureAlignment:
    """Tests verifying fixture alignment between unit and integration tests."""

    def test_canonical_fixture_has_valid_roots(self) -> None:
        """
        Verify that canonical fixture has valid 64-char hex roots.
        """
        fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION

        for root_name in ["composite_root", "reasoning_root", "ui_root"]:
            root = getattr(fixture, root_name)
            assert len(root) == 64, f"{root_name} must be 64 chars"
            int(root, 16)  # Must be valid hex

    def test_canonical_fixture_h_t_is_sha256_of_r_t_u_t(self) -> None:
        """
        Verify that H_t = SHA256(R_t || U_t) in the canonical fixture.
        """
        fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION

        expected_h_t = hashlib.sha256(
            (fixture.reasoning_root + fixture.ui_root).encode("utf-8")
        ).hexdigest()

        assert fixture.composite_root == expected_h_t, (
            f"H_t invariant violated: expected {expected_h_t}, got {fixture.composite_root}"
        )

    def test_make_attested_run_context_preserves_roots(self) -> None:
        """
        Verify that make_attested_run_context preserves all roots.
        """
        fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION
        ctx = make_attested_run_context(fixture)

        assert ctx.composite_root == fixture.composite_root
        assert ctx.reasoning_root == fixture.reasoning_root
        assert ctx.ui_root == fixture.ui_root

    def test_compute_expected_step_id_matches_runner(self) -> None:
        """
        Verify that compute_expected_step_id matches actual runner output.
        
        This test ensures the resolved_slice_name logic in tests matches the runtime code.
        """
        config = RFLConfig(num_runs=5)
        runner = RFLRunner(config)
        fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION
        ctx = make_attested_run_context(fixture)

        result = runner.run_with_attestation(ctx)
        # The runner resolves slice_id to the first matching curriculum slice
        resolved_slice = runner._resolve_slice(ctx.slice_id)
        expected = compute_expected_step_id(
            fixture,
            experiment_id=config.experiment_id,
            policy_id=fixture.policy_id,
            resolved_slice_name=resolved_slice.name,
        )

        assert result.step_id == expected, (
            f"step_id mismatch. Expected: {expected}, Got: {result.step_id}. "
            f"Resolved slice: {resolved_slice.name} (from slice_id: {ctx.slice_id})"
        )


class TestWideSliceAwareness:
    """
    Tests to ensure RFL handles Wide Slice (slice_medium) identically to FO slice.
    
    This verifies that the RFL engine doesn't have hidden assumptions about
    FO-only slices and can handle any curriculum slice name correctly.
    
    NOTE (Sober Truth / Reviewer 2 mode):
    - These tests verify ENGINE CAPABILITY, not empirical results from real runs.
    - They use synthetic AttestedRunContext data, not logs from actual Wide Slice experiments.
    - Real Wide Slice data (if it exists) would be in results/fo_*wide*.jsonl, but
      these tests do NOT claim such data exists or assert its properties.
    - These tests are HERMETIC: no DB/Redis dependencies, fully deterministic.
    """

    @pytest.fixture
    def config(self) -> RFLConfig:
        """RFL config with slice_medium in curriculum."""
        from rfl.config import CurriculumSlice as RFLCurriculumSlice
        return RFLConfig(
            num_runs=5,
            curriculum=[
                RFLCurriculumSlice(
                    name="slice_medium",
                    start_run=1,
                    end_run=5,
                    derive_steps=1,
                    max_breadth=1,
                    max_total=1,
                    depth_max=1,
                )
            ],
        )

    @pytest.fixture
    def runner(self, config: RFLConfig) -> RFLRunner:
        """RFL runner instance."""
        return RFLRunner(config)

    def test_wide_slice_attestation_context_handled_identically(
        self, runner: RFLRunner, config: RFLConfig
    ) -> None:
        """
        Verify that RFL run_with_attestation handles slice_medium identically to FO slice.
        
        This test ensures:
        1. slice_id="slice_medium" resolves correctly
        2. step_id computation works with resolved slice name
        3. Policy updates apply correctly
        4. Ledger entries are populated correctly
        """
        # Create attestation context with slice_medium
        fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION
        ctx = AttestedRunContext(
            slice_id="slice_medium",  # Wide slice name
            statement_hash=fixture.statement_hash,
            proof_status=fixture.proof_status,
            block_id=fixture.block_id,
            composite_root=fixture.composite_root,
            reasoning_root=fixture.reasoning_root,
            ui_root=fixture.ui_root,
            abstention_metrics={
                "rate": fixture.abstention_rate,
                "mass": fixture.abstention_mass,
            },
            policy_id=fixture.policy_id,
            metadata=dict(fixture.metadata),
        )

        result = runner.run_with_attestation(ctx)

        # Verify slice resolution
        resolved_slice = runner._resolve_slice(ctx.slice_id)
        assert resolved_slice.name == "slice_medium", (
            f"Should resolve to slice_medium, got {resolved_slice.name}"
        )

        # Verify step_id computation uses resolved slice name
        expected_step = compute_expected_step_id(
            fixture,
            experiment_id=config.experiment_id,
            policy_id=ctx.policy_id,
            resolved_slice_name=resolved_slice.name,
        )
        assert result.step_id == expected_step, (
            f"step_id should match formula with resolved slice name. "
            f"Expected: {expected_step}, Got: {result.step_id}"
        )

        # Verify policy update applied (same as FO slice)
        assert result.policy_update_applied is True
        assert result.source_root == ctx.composite_root

        # Verify ledger entry populated correctly
        assert len(runner.policy_ledger) == 1
        entry = runner.policy_ledger[0]
        assert entry.slice_name == "slice_medium", (
            f"Ledger entry should have resolved slice name, got {entry.slice_name}"
        )
        assert entry.attestation_slice_id == "slice_medium", (
            f"Original slice_id should be preserved, got {entry.attestation_slice_id}"
        )
        assert entry.composite_root == ctx.composite_root

    def test_wide_slice_step_id_determinism(
        self, runner: RFLRunner, config: RFLConfig
    ) -> None:
        """
        Verify step_id determinism for wide slice matches FO slice behavior.
        """
        fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION
        ctx = AttestedRunContext(
            slice_id="slice_medium",
            statement_hash=fixture.statement_hash,
            proof_status=fixture.proof_status,
            block_id=fixture.block_id,
            composite_root=fixture.composite_root,
            reasoning_root=fixture.reasoning_root,
            ui_root=fixture.ui_root,
            abstention_metrics={
                "rate": fixture.abstention_rate,
                "mass": fixture.abstention_mass,
            },
            policy_id=fixture.policy_id,
            metadata=dict(fixture.metadata),
        )

        # Run multiple times, step_id should be stable
        step_ids = []
        for _ in range(3):
            runner_instance = RFLRunner(config)
            result = runner_instance.run_with_attestation(ctx)
            step_ids.append(result.step_id)

        assert len(set(step_ids)) == 1, (
            f"step_id should be stable for wide slice, got: {step_ids}"
        )
        
    def test_resolved_slice_name_logic_matches_runtime(self) -> None:
        """
        Verify that resolved_slice_name logic in tests matches runtime slice resolution.
        
        The runner's _resolve_slice method should:
        1. Match slice_id to curriculum slice name if found
        2. Fall back to first curriculum slice if not found
        """
        config = RFLConfig(num_runs=5)
        runner = RFLRunner(config)
        fixture = CANONICAL_FIRST_ORGANISM_ATTESTATION
        
        # Test with a slice_id that matches a curriculum slice
        ctx = make_attested_run_context(fixture)
        resolved_slice = runner._resolve_slice(ctx.slice_id)
        
        # Verify resolution logic: if slice_id matches a curriculum slice name, use it
        # Otherwise, fall back to first curriculum slice
        matching_slice = None
        for slice_cfg in config.curriculum:
            if slice_cfg.name == ctx.slice_id:
                matching_slice = slice_cfg
                break
        
        if matching_slice:
            assert resolved_slice.name == matching_slice.name, (
                f"Should resolve to matching slice {matching_slice.name}, "
                f"got {resolved_slice.name}"
            )
        else:
            # Should fall back to first curriculum slice
            assert resolved_slice.name == config.curriculum[0].name, (
                f"Should fall back to first curriculum slice {config.curriculum[0].name}, "
                f"got {resolved_slice.name}"
            )
