# tests/backend/verification/test_lean_adapter_scaffold.py
"""
Test suite for Phase IIb Lean Adapter Scaffold.

These tests verify that the LeanAdapter scaffold:
1. Returns deterministic results based on input
2. Never invokes actual Lean
3. Properly sets abstention reasons
4. Is ready for Phase IIb activation
5. Simulation mode produces correct deterministic outcomes
6. Validation catches invalid requests

Markers:
    - unit: Fast, no external dependencies
    - phase2b: Phase IIb scaffolding tests
"""

import pytest

from backend.verification import (
    LeanAdapter,
    LeanAdapterMode,
    LeanAbstentionReason,
    LeanResourceBudget,
    LeanVerificationRequest,
    LeanVerificationResult,
    LeanAdapterConfig,
    LeanAdapterValidationError,
    LEAN_VERSION_REQUIRED,
    validate_verification_request,
    is_valid_canonical,
    simulate_lean_result,
    compute_formula_complexity,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def scaffold_adapter() -> LeanAdapter:
    """Create a LeanAdapter in PHASE_IIB_SCAFFOLD mode."""
    return LeanAdapter(mode=LeanAdapterMode.PHASE_IIB_SCAFFOLD)


@pytest.fixture
def disabled_adapter() -> LeanAdapter:
    """Create a LeanAdapter in DISABLED mode."""
    return LeanAdapter(mode=LeanAdapterMode.DISABLED)


@pytest.fixture
def simulate_adapter() -> LeanAdapter:
    """Create a LeanAdapter in SIMULATE mode."""
    return LeanAdapter(mode=LeanAdapterMode.SIMULATE)


@pytest.fixture
def sample_request() -> LeanVerificationRequest:
    """Create a sample verification request."""
    return LeanVerificationRequest(
        canonical="p->p",
        job_id="test_abc123",
        resource_budget=LeanResourceBudget(
            timeout_seconds=30,
            memory_mb=2048,
            disk_mb=100,
        ),
    )


@pytest.fixture
def another_request() -> LeanVerificationRequest:
    """Create a different verification request for comparison."""
    return LeanVerificationRequest(
        canonical="p->q->p",
        job_id="test_def456",
    )


@pytest.fixture
def complex_request() -> LeanVerificationRequest:
    """Create a request with a complex formula (will trigger timeout in simulation)."""
    # Create a formula > 50 chars to trigger timeout simulation
    long_formula = "p->" * 20 + "p"  # 63 chars
    return LeanVerificationRequest(
        canonical=long_formula,
        job_id="test_complex",
    )


@pytest.fixture
def very_complex_request() -> LeanVerificationRequest:
    """Create a request with a very complex formula (will trigger resource exceeded)."""
    # Create a formula > 80 chars to trigger resource exceeded simulation
    very_long_formula = "p->" * 30 + "p"  # 93 chars
    return LeanVerificationRequest(
        canonical=very_long_formula,
        job_id="test_very_complex",
    )


# =============================================================================
# CORE SCAFFOLD TESTS
# =============================================================================

@pytest.mark.unit
class TestLeanAdapterScaffoldDeterminism:
    """Test that scaffold mode produces deterministic results."""

    def test_scaffold_returns_deterministic_result(
        self,
        scaffold_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify that the same request produces the same result hash."""
        result1 = scaffold_adapter.verify(sample_request)
        result2 = scaffold_adapter.verify(sample_request)

        # Core determinism check
        assert result1.deterministic_hash == result2.deterministic_hash, (
            "Deterministic hash should be stable across calls"
        )
        
        # Full result should be consistent (except possibly duration_ms)
        assert result1.verified == result2.verified
        assert result1.abstention_reason == result2.abstention_reason
        assert result1.method == result2.method
        assert result1.lean_version_checked == result2.lean_version_checked

    def test_different_requests_produce_different_hashes(
        self,
        scaffold_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
        another_request: LeanVerificationRequest,
    ) -> None:
        """Verify that different requests produce different hashes."""
        result1 = scaffold_adapter.verify(sample_request)
        result2 = scaffold_adapter.verify(another_request)

        assert result1.deterministic_hash != result2.deterministic_hash, (
            "Different requests should produce different hashes"
        )

    def test_request_deterministic_hash_method(
        self,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify LeanVerificationRequest.deterministic_hash() is stable."""
        hash1 = sample_request.deterministic_hash()
        hash2 = sample_request.deterministic_hash()

        assert hash1 == hash2, "Request hash should be stable"
        assert len(hash1) == 64, "Hash should be SHA256 hex (64 chars)"


@pytest.mark.unit
class TestLeanAdapterScaffoldAbstention:
    """Test that scaffold mode sets correct abstention reasons."""

    def test_scaffold_sets_lean_disabled_reason(
        self,
        scaffold_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify scaffold mode returns LEAN_DISABLED abstention."""
        result = scaffold_adapter.verify(sample_request)

        assert result.verified is False, "Scaffold should not verify"
        assert result.abstention_reason == LeanAbstentionReason.LEAN_DISABLED, (
            "Scaffold should return LEAN_DISABLED abstention reason"
        )

    def test_disabled_mode_sets_lean_disabled_reason(
        self,
        disabled_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify disabled mode returns LEAN_DISABLED abstention."""
        result = disabled_adapter.verify(sample_request)

        assert result.verified is False, "Disabled should not verify"
        assert result.abstention_reason == LeanAbstentionReason.LEAN_DISABLED, (
            "Disabled should return LEAN_DISABLED abstention reason"
        )

    def test_scaffold_method_identifier(
        self,
        scaffold_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify scaffold mode uses correct method identifier."""
        result = scaffold_adapter.verify(sample_request)

        assert result.method == "lean_adapter_scaffold", (
            "Scaffold should use 'lean_adapter_scaffold' method identifier"
        )

    def test_disabled_method_identifier(
        self,
        disabled_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify disabled mode uses correct method identifier."""
        result = disabled_adapter.verify(sample_request)

        assert result.method == "lean_adapter_disabled", (
            "Disabled should use 'lean_adapter_disabled' method identifier"
        )


@pytest.mark.unit
class TestLeanAdapterScaffoldVersionPinning:
    """Test version pinning in scaffold mode."""

    def test_scaffold_reports_version_checked(
        self,
        scaffold_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify scaffold reports the required Lean version."""
        result = scaffold_adapter.verify(sample_request)

        assert result.lean_version_checked == LEAN_VERSION_REQUIRED, (
            f"Scaffold should report version {LEAN_VERSION_REQUIRED}"
        )

    def test_disabled_does_not_report_version(
        self,
        disabled_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify disabled mode does not report version."""
        result = disabled_adapter.verify(sample_request)

        assert result.lean_version_checked is None, (
            "Disabled mode should not report version"
        )

    def test_adapter_class_constant_matches_module(
        self,
        scaffold_adapter: LeanAdapter,
    ) -> None:
        """Verify class constant matches module constant."""
        assert scaffold_adapter.LEAN_VERSION_REQUIRED == LEAN_VERSION_REQUIRED


@pytest.mark.unit
class TestLeanAdapterScaffoldNoLeanInvocation:
    """Test that scaffold mode never invokes Lean."""

    def test_check_lean_availability_returns_false(
        self,
        scaffold_adapter: LeanAdapter,
    ) -> None:
        """Verify check_lean_availability() returns False in scaffold."""
        assert scaffold_adapter.check_lean_availability() is False, (
            "Scaffold should report Lean as unavailable"
        )

    def test_validate_lean_version_returns_false(
        self,
        scaffold_adapter: LeanAdapter,
    ) -> None:
        """Verify validate_lean_version() returns (False, None) in scaffold."""
        is_valid, version = scaffold_adapter.validate_lean_version()

        assert is_valid is False, "Scaffold should not validate version"
        assert version is None, "Scaffold should not detect version"

    def test_active_mode_raises_not_implemented(
        self,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify ACTIVE mode raises NotImplementedError."""
        adapter = LeanAdapter(mode=LeanAdapterMode.ACTIVE)

        with pytest.raises(NotImplementedError) as exc_info:
            adapter.verify(sample_request)

        assert "Phase IIb" in str(exc_info.value), (
            "NotImplementedError should mention Phase IIb"
        )


# =============================================================================
# SIMULATION MODE TESTS
# =============================================================================

@pytest.mark.unit
class TestLeanAdapterSimulationMode:
    """Test simulation mode produces correct deterministic outcomes."""

    def test_simulate_mode_returns_deterministic_result(
        self,
        simulate_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify simulation produces deterministic results."""
        result1 = simulate_adapter.verify(sample_request)
        result2 = simulate_adapter.verify(sample_request)

        assert result1.deterministic_hash == result2.deterministic_hash
        assert result1.verified == result2.verified
        assert result1.abstention_reason == result2.abstention_reason
        assert result1.method == result2.method

    def test_simulate_mode_method_identifier(
        self,
        simulate_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify simulation uses correct method identifier."""
        result = simulate_adapter.verify(sample_request)

        assert result.method == "lean_adapter_simulate"

    def test_simulate_mode_reports_version(
        self,
        simulate_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify simulation reports Lean version."""
        result = simulate_adapter.verify(sample_request)

        assert result.lean_version_checked == LEAN_VERSION_REQUIRED

    def test_simulate_mode_includes_complexity(
        self,
        simulate_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify simulation includes complexity score."""
        result = simulate_adapter.verify(sample_request)

        assert result.simulated_complexity is not None
        assert result.simulated_complexity == len(sample_request.canonical)

    def test_simulate_success_path(
        self,
        simulate_adapter: LeanAdapter,
    ) -> None:
        """Test simulation success path (simple formula, low hash)."""
        # "p->p" with job_id "success_test" should produce verified=True
        # based on hash branching (we test multiple to find one that succeeds)
        found_success = False
        for i in range(20):
            request = LeanVerificationRequest(
                canonical="p->p",
                job_id=f"success_test_{i}",
            )
            result = simulate_adapter.verify(request)
            if result.verified:
                found_success = True
                assert result.abstention_reason is None
                break
        
        assert found_success, "Should find at least one success in 20 tries (70% rate)"

    def test_simulate_timeout_path(
        self,
        simulate_adapter: LeanAdapter,
        complex_request: LeanVerificationRequest,
    ) -> None:
        """Test simulation timeout path for complex formulas."""
        result = simulate_adapter.verify(complex_request)

        assert result.verified is False
        assert result.abstention_reason == LeanAbstentionReason.LEAN_TIMEOUT
        assert result.simulated_complexity > 50

    def test_simulate_resource_exceeded_path(
        self,
        simulate_adapter: LeanAdapter,
        very_complex_request: LeanVerificationRequest,
    ) -> None:
        """Test simulation resource exceeded path for very complex formulas."""
        result = simulate_adapter.verify(very_complex_request)

        assert result.verified is False
        assert result.abstention_reason == LeanAbstentionReason.LEAN_RESOURCE_EXCEEDED
        assert result.simulated_complexity > 80


@pytest.mark.unit
class TestSimulateLeanResultHelper:
    """Test the simulate_lean_result() helper function directly."""

    def test_simulate_lean_result_deterministic(self) -> None:
        """Verify simulate_lean_result is deterministic."""
        result1 = simulate_lean_result("p->p", "job1", LeanResourceBudget())
        result2 = simulate_lean_result("p->p", "job1", LeanResourceBudget())

        assert result1.deterministic_hash == result2.deterministic_hash
        assert result1.verified == result2.verified

    def test_simulate_lean_result_different_inputs(self) -> None:
        """Verify different inputs produce different results."""
        result1 = simulate_lean_result("p->p", "job1", LeanResourceBudget())
        result2 = simulate_lean_result("q->q", "job1", LeanResourceBudget())

        assert result1.deterministic_hash != result2.deterministic_hash

    def test_simulate_lean_result_timeout_threshold(self) -> None:
        """Verify timeout threshold triggers correctly."""
        # 51 chars should trigger timeout
        long_formula = "a" * 51
        result = simulate_lean_result(long_formula, "job1", LeanResourceBudget())

        assert result.abstention_reason == LeanAbstentionReason.LEAN_TIMEOUT

    def test_simulate_lean_result_resource_threshold(self) -> None:
        """Verify resource threshold triggers correctly."""
        # 81 chars should trigger resource exceeded
        very_long_formula = "a" * 81
        result = simulate_lean_result(very_long_formula, "job1", LeanResourceBudget())

        assert result.abstention_reason == LeanAbstentionReason.LEAN_RESOURCE_EXCEEDED


@pytest.mark.unit
class TestComputeFormulaComplexity:
    """Test the compute_formula_complexity() helper function."""

    def test_complexity_equals_length(self) -> None:
        """Verify complexity is currently just length."""
        assert compute_formula_complexity("p->p") == 4
        assert compute_formula_complexity("p->q->p") == 7
        assert compute_formula_complexity("") == 0


# =============================================================================
# VALIDATION TESTS
# =============================================================================

@pytest.mark.unit
class TestLeanVerificationRequestValidation:
    """Test validation of LeanVerificationRequest."""

    def test_valid_request_passes(self) -> None:
        """Verify valid requests pass validation."""
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="abc123",
        )
        errors = validate_verification_request(request)
        assert errors == []

    def test_empty_canonical_fails(self) -> None:
        """Verify empty canonical fails validation."""
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanVerificationRequest(
                canonical="",
                job_id="abc123",
            )
        assert "canonical must be non-empty" in str(exc_info.value)

    def test_whitespace_canonical_fails(self) -> None:
        """Verify whitespace-only canonical fails validation."""
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanVerificationRequest(
                canonical="   ",
                job_id="abc123",
            )
        assert "whitespace-only" in str(exc_info.value)

    def test_empty_job_id_fails(self) -> None:
        """Verify empty job_id fails validation."""
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanVerificationRequest(
                canonical="p->p",
                job_id="",
            )
        assert "job_id must be non-empty" in str(exc_info.value)

    def test_whitespace_job_id_fails(self) -> None:
        """Verify whitespace-only job_id fails validation."""
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanVerificationRequest(
                canonical="p->p",
                job_id="   ",
            )
        assert "whitespace-only" in str(exc_info.value)

    def test_unsupported_characters_fail(self) -> None:
        """Verify unsupported characters fail validation."""
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanVerificationRequest(
                canonical="p→q",  # Unicode arrow not supported
                job_id="abc123",
            )
        assert "unsupported characters" in str(exc_info.value)

    def test_too_long_canonical_fails(self) -> None:
        """Verify overly long canonical fails validation."""
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanVerificationRequest(
                canonical="p" * 10001,  # Exceeds 10000 limit
                job_id="abc123",
            )
        assert "exceeds maximum length" in str(exc_info.value)

    def test_special_chars_in_propositional_logic(self) -> None:
        """Verify common propositional logic characters are valid."""
        # These should all pass validation
        valid_canonicals = [
            "p->q",
            "p<->q",
            "~p",
            "p/\\q",
            "p\\/q",
            "(p->q)",
            "p & q",
            "p | q",
            "!p",
            "p ^ q",
        ]
        for canonical in valid_canonicals:
            request = LeanVerificationRequest(
                canonical=canonical,
                job_id="test",
            )
            assert request.canonical == canonical


@pytest.mark.unit
class TestIsValidCanonical:
    """Test the is_valid_canonical() helper function."""

    def test_valid_canonicals(self) -> None:
        """Verify valid canonicals return True."""
        assert is_valid_canonical("p->p") is True
        assert is_valid_canonical("p->q->p") is True
        assert is_valid_canonical("(p/\\q)->p") is True

    def test_empty_canonical(self) -> None:
        """Verify empty canonical returns False."""
        assert is_valid_canonical("") is False

    def test_whitespace_only(self) -> None:
        """Verify whitespace-only returns False."""
        assert is_valid_canonical("   ") is False

    def test_too_long(self) -> None:
        """Verify too long canonical returns False."""
        assert is_valid_canonical("p" * 10001) is False

    def test_unicode_characters(self) -> None:
        """Verify unicode characters return False."""
        assert is_valid_canonical("p→q") is False
        assert is_valid_canonical("p∧q") is False
        assert is_valid_canonical("¬p") is False


@pytest.mark.unit
class TestValidateVerificationRequest:
    """Test the validate_verification_request() function directly."""

    def test_returns_list_of_errors(self) -> None:
        """Verify function returns list of error strings."""
        # Create a request that bypasses __post_init__ validation
        # by using object.__new__ and then setting attributes
        request = object.__new__(LeanVerificationRequest)
        object.__setattr__(request, 'canonical', '')
        object.__setattr__(request, 'job_id', '')
        object.__setattr__(request, 'resource_budget', LeanResourceBudget())
        
        errors = validate_verification_request(request)
        
        assert isinstance(errors, list)
        assert len(errors) >= 2  # At least canonical and job_id errors

    def test_empty_list_for_valid_request(self) -> None:
        """Verify function returns empty list for valid request."""
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="valid_id",
        )
        errors = validate_verification_request(request)
        assert errors == []


# =============================================================================
# DATACLASS TESTS
# =============================================================================

@pytest.mark.unit
class TestLeanResourceBudget:
    """Test LeanResourceBudget dataclass."""

    def test_default_values(self) -> None:
        """Verify default budget values."""
        budget = LeanResourceBudget()

        assert budget.timeout_seconds == 30
        assert budget.memory_mb == 2048
        assert budget.disk_mb == 100

    def test_custom_values(self) -> None:
        """Verify custom budget values."""
        budget = LeanResourceBudget(
            timeout_seconds=60,
            memory_mb=4096,
            disk_mb=200,
        )

        assert budget.timeout_seconds == 60
        assert budget.memory_mb == 4096
        assert budget.disk_mb == 200

    def test_to_dict(self) -> None:
        """Verify to_dict serialization."""
        budget = LeanResourceBudget()
        d = budget.to_dict()

        assert d["timeout_seconds"] == 30
        assert d["memory_mb"] == 2048
        assert d["disk_mb"] == 100


@pytest.mark.unit
class TestLeanVerificationRequest:
    """Test LeanVerificationRequest dataclass."""

    def test_required_fields(self) -> None:
        """Verify required fields."""
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="test123",
        )

        assert request.canonical == "p->p"
        assert request.job_id == "test123"
        assert isinstance(request.resource_budget, LeanResourceBudget)

    def test_to_dict(self) -> None:
        """Verify to_dict serialization."""
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="test123",
        )
        d = request.to_dict()

        assert d["canonical"] == "p->p"
        assert d["job_id"] == "test123"
        assert "resource_budget" in d


@pytest.mark.unit
class TestLeanVerificationResult:
    """Test LeanVerificationResult dataclass."""

    def test_is_abstention_property(self) -> None:
        """Verify is_abstention property."""
        abstention_result = LeanVerificationResult(
            verified=False,
            abstention_reason=LeanAbstentionReason.LEAN_DISABLED,
            method="test",
            lean_version_checked=None,
            duration_ms=0,
            deterministic_hash="abc123",
        )

        assert abstention_result.is_abstention is True

        success_result = LeanVerificationResult(
            verified=True,
            abstention_reason=None,
            method="test",
            lean_version_checked="v4.23.0",
            duration_ms=100,
            deterministic_hash="def456",
        )

        assert success_result.is_abstention is False

    def test_to_dict(self) -> None:
        """Verify to_dict serialization."""
        result = LeanVerificationResult(
            verified=False,
            abstention_reason=LeanAbstentionReason.LEAN_TIMEOUT,
            method="lean_adapter_scaffold",
            lean_version_checked="v4.23.0-rc2",
            duration_ms=5000,
            deterministic_hash="abc123def456",
        )
        d = result.to_dict()

        assert d["verified"] is False
        assert d["abstention_reason"] == "lean_timeout"
        assert d["method"] == "lean_adapter_scaffold"
        assert d["lean_version_checked"] == "v4.23.0-rc2"
        assert d["duration_ms"] == 5000

    def test_to_dict_with_simulated_complexity(self) -> None:
        """Verify to_dict includes simulated_complexity when present."""
        result = LeanVerificationResult(
            verified=True,
            abstention_reason=None,
            method="lean_adapter_simulate",
            lean_version_checked="v4.23.0-rc2",
            duration_ms=1,
            deterministic_hash="abc123",
            simulated_complexity=42,
        )
        d = result.to_dict()

        assert d["simulated_complexity"] == 42


# =============================================================================
# STATISTICS TESTS
# =============================================================================

@pytest.mark.unit
class TestLeanAdapterStatistics:
    """Test adapter statistics tracking."""

    def test_statistics_after_verify(
        self,
        scaffold_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify statistics are tracked after verify calls."""
        # Initial state
        stats_before = scaffold_adapter.get_statistics()
        assert stats_before["verification_count"] == 0

        # After one verify
        scaffold_adapter.verify(sample_request)
        stats_after = scaffold_adapter.get_statistics()
        
        assert stats_after["verification_count"] == 1
        assert stats_after["mode"] == "phase2b_scaffold"
        assert stats_after["lean_version_required"] == LEAN_VERSION_REQUIRED

    def test_simulation_statistics(
        self,
        simulate_adapter: LeanAdapter,
        sample_request: LeanVerificationRequest,
    ) -> None:
        """Verify simulation-specific statistics are tracked."""
        # Run multiple verifications
        for i in range(10):
            request = LeanVerificationRequest(
                canonical=f"p{i}->p{i}",
                job_id=f"stat_test_{i}",
            )
            simulate_adapter.verify(request)

        stats = simulate_adapter.get_statistics()
        
        assert stats["simulation_count"] == 10
        assert "simulation_success_count" in stats
        assert "simulation_success_rate" in stats
        assert 0 <= stats["simulation_success_rate"] <= 1


# =============================================================================
# ENUM TESTS
# =============================================================================

@pytest.mark.unit
class TestLeanAdapterEnums:
    """Test enum definitions."""

    def test_adapter_mode_values(self) -> None:
        """Verify LeanAdapterMode enum values."""
        assert LeanAdapterMode.DISABLED.value == "disabled"
        assert LeanAdapterMode.PHASE_IIB_SCAFFOLD.value == "phase2b_scaffold"
        assert LeanAdapterMode.SIMULATE.value == "simulate"
        assert LeanAdapterMode.ACTIVE.value == "active"

    def test_abstention_reason_values(self) -> None:
        """Verify LeanAbstentionReason enum values."""
        assert LeanAbstentionReason.LEAN_DISABLED.value == "lean_disabled"
        assert LeanAbstentionReason.LEAN_TIMEOUT.value == "lean_timeout"
        assert LeanAbstentionReason.LEAN_ERROR.value == "lean_error"
        assert LeanAbstentionReason.LEAN_UNAVAILABLE.value == "lean_unavailable"
        assert LeanAbstentionReason.LEAN_RESOURCE_EXCEEDED.value == "lean_resource_exceeded"
        assert LeanAbstentionReason.LEAN_VERSION_MISMATCH.value == "lean_version_mismatch"


# =============================================================================
# CONTRACT-GRADE DETERMINISM TESTS
# =============================================================================

@pytest.mark.unit
class TestContractGradeDeterminism:
    """
    Contract-grade tests ensuring simulation is fully deterministic.
    
    These tests verify the core invariant: identical inputs produce
    identical outputs across all calls, with no hidden I/O or randomness.
    """

    def test_simulation_identical_inputs_identical_outputs(self) -> None:
        """
        CONTRACT: Identical inputs MUST produce identical outputs.
        
        Verify that running the same request 100 times produces
        the exact same result every time.
        """
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        request = LeanVerificationRequest(
            canonical="p->q->p",
            job_id="contract_test_001",
        )
        
        # Run 100 times
        results = [adapter.verify(request) for _ in range(100)]
        
        # All should have identical deterministic fields
        first = results[0]
        for i, result in enumerate(results[1:], start=2):
            assert result.verified == first.verified, f"Run {i}: verified differs"
            assert result.abstention_reason == first.abstention_reason, f"Run {i}: abstention_reason differs"
            assert result.method == first.method, f"Run {i}: method differs"
            assert result.deterministic_hash == first.deterministic_hash, f"Run {i}: hash differs"
            assert result.lean_version_checked == first.lean_version_checked, f"Run {i}: version differs"
            assert result.simulated_complexity == first.simulated_complexity, f"Run {i}: complexity differs"

    def test_simulation_different_job_id_different_hash(self) -> None:
        """
        CONTRACT: Different job_id MUST produce different deterministic hashes.
        """
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        
        results = []
        for i in range(50):
            request = LeanVerificationRequest(
                canonical="p->p",  # Same canonical
                job_id=f"unique_job_{i}",  # Different job_id
            )
            results.append(adapter.verify(request))
        
        # All hashes must be unique
        hashes = [r.deterministic_hash for r in results]
        assert len(set(hashes)) == 50, "All 50 job_ids should produce unique hashes"

    def test_simulation_different_canonical_different_hash(self) -> None:
        """
        CONTRACT: Different canonical MUST produce different deterministic hashes.
        """
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        
        results = []
        for i in range(50):
            request = LeanVerificationRequest(
                canonical=f"p{i}->q{i}",  # Different canonical
                job_id="same_job",  # Same job_id
            )
            results.append(adapter.verify(request))
        
        # All hashes must be unique
        hashes = [r.deterministic_hash for r in results]
        assert len(set(hashes)) == 50, "All 50 canonicals should produce unique hashes"

    def test_version_pin_in_all_simulation_results(self) -> None:
        """
        CONTRACT: LEAN_VERSION_REQUIRED must be surfaced in all simulation results.
        """
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        
        # Test across different complexity thresholds
        test_cases = [
            ("p->p", "simple"),  # Success path
            ("a" * 55, "timeout"),  # Timeout path
            ("b" * 85, "resource"),  # Resource exceeded path
        ]
        
        for canonical, desc in test_cases:
            request = LeanVerificationRequest(
                canonical=canonical,
                job_id=f"version_test_{desc}",
            )
            result = adapter.verify(request)
            assert result.lean_version_checked == LEAN_VERSION_REQUIRED, (
                f"Version not surfaced for {desc} path"
            )


# =============================================================================
# CONTRACT-GRADE ACTIVE MODE GUARDRAILS
# =============================================================================

@pytest.mark.unit
class TestContractGradeActiveGuardrails:
    """
    Contract-grade tests ensuring ACTIVE mode is refused in Phase II.
    """

    def test_active_mode_constructor_allowed(self) -> None:
        """
        ACTIVE mode can be constructed, but cannot verify.
        """
        # Construction should work
        adapter = LeanAdapter(mode=LeanAdapterMode.ACTIVE)
        assert adapter.mode == LeanAdapterMode.ACTIVE

    def test_active_mode_verify_raises_not_implemented(self) -> None:
        """
        CONTRACT: ACTIVE mode verify() MUST raise NotImplementedError.
        """
        adapter = LeanAdapter(mode=LeanAdapterMode.ACTIVE)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="active_test",
        )
        
        with pytest.raises(NotImplementedError) as exc_info:
            adapter.verify(request)
        
        # Error message must mention Phase IIb
        assert "Phase IIb" in str(exc_info.value)
        assert "not implemented" in str(exc_info.value).lower()

    def test_active_mode_error_is_deterministic(self) -> None:
        """
        CONTRACT: ACTIVE mode error must be deterministic (same message).
        """
        adapter = LeanAdapter(mode=LeanAdapterMode.ACTIVE)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="determinism_test",
        )
        
        errors = []
        for _ in range(10):
            try:
                adapter.verify(request)
            except NotImplementedError as e:
                errors.append(str(e))
        
        # All error messages must be identical
        assert len(set(errors)) == 1, "ACTIVE mode error must be deterministic"


# =============================================================================
# CONTRACT-GRADE VALIDATION ERROR MESSAGES
# =============================================================================

@pytest.mark.unit
class TestContractGradeValidationErrors:
    """
    Contract-grade tests for validation error messages.
    """

    def test_validation_error_message_contains_field_name(self) -> None:
        """
        CONTRACT: Validation errors MUST mention the invalid field.
        """
        # Test empty canonical
        try:
            LeanVerificationRequest(canonical="", job_id="test")
        except LeanAdapterValidationError as e:
            assert "canonical" in str(e).lower()
        
        # Test empty job_id
        try:
            LeanVerificationRequest(canonical="p->p", job_id="")
        except LeanAdapterValidationError as e:
            assert "job_id" in str(e).lower()

    def test_validation_error_message_contains_reason(self) -> None:
        """
        CONTRACT: Validation errors MUST explain the reason.
        """
        # Test length exceeded
        try:
            LeanVerificationRequest(canonical="p" * 10001, job_id="test")
        except LeanAdapterValidationError as e:
            error_msg = str(e).lower()
            assert "length" in error_msg or "exceeds" in error_msg
        
        # Test unsupported characters
        try:
            LeanVerificationRequest(canonical="p→q", job_id="test")
        except LeanAdapterValidationError as e:
            assert "unsupported" in str(e).lower() or "character" in str(e).lower()

    def test_validate_function_returns_all_errors(self) -> None:
        """
        CONTRACT: validate_verification_request() returns ALL errors, not just first.
        """
        # Create a request with multiple validation failures
        request = object.__new__(LeanVerificationRequest)
        object.__setattr__(request, 'canonical', '')
        object.__setattr__(request, 'job_id', '')
        object.__setattr__(request, 'resource_budget', LeanResourceBudget())
        
        errors = validate_verification_request(request)
        
        # Should have at least 2 errors (empty canonical and empty job_id)
        assert len(errors) >= 2, "Should report multiple validation errors"
        assert any("canonical" in e.lower() for e in errors)
        assert any("job_id" in e.lower() for e in errors)


# =============================================================================
# CONTRACT-GRADE PURITY TESTS
# =============================================================================

@pytest.mark.unit
class TestContractGradePurity:
    """
    Contract-grade tests ensuring the adapter has no I/O side effects.
    """

    def test_simulation_does_not_use_filesystem(self) -> None:
        """
        CONTRACT: Simulation MUST NOT touch filesystem.
        
        Verify by running simulation and checking no files are created.
        """
        import tempfile
        import os
        
        # Get initial state of temp directory
        temp_dir = tempfile.gettempdir()
        initial_files = set(os.listdir(temp_dir)) if os.path.exists(temp_dir) else set()
        
        # Run many simulations
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        for i in range(100):
            request = LeanVerificationRequest(
                canonical=f"p{i}->p{i}",
                job_id=f"purity_test_{i}",
            )
            adapter.verify(request)
        
        # Check no new files created in temp
        # (This is a heuristic check, not perfect, but catches obvious violations)
        final_files = set(os.listdir(temp_dir)) if os.path.exists(temp_dir) else set()
        new_lean_files = [f for f in (final_files - initial_files) if 'lean' in f.lower()]
        assert len(new_lean_files) == 0, f"Simulation created files: {new_lean_files}"

    def test_simulation_result_independent_of_time(self) -> None:
        """
        CONTRACT: Simulation outcome MUST NOT depend on wall-clock time.
        
        Note: duration_ms is observational metadata, not outcome.
        """
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        request = LeanVerificationRequest(
            canonical="p->q->r",
            job_id="time_independence_test",
        )
        
        # Run at different "times" (simulated by running in sequence)
        results = [adapter.verify(request) for _ in range(10)]
        
        # All deterministic outcomes must match (duration_ms may vary)
        for result in results:
            assert result.verified == results[0].verified
            assert result.abstention_reason == results[0].abstention_reason
            assert result.deterministic_hash == results[0].deterministic_hash

    def test_no_subprocess_import_in_module(self) -> None:
        """
        CONTRACT: lean_adapter.py MUST NOT import subprocess.
        """
        import backend.verification.lean_adapter as module
        import sys
        
        # Check module's imports
        module_imports = [
            name for name in dir(module)
            if not name.startswith('_')
        ]
        
        assert 'subprocess' not in module_imports, "subprocess should not be imported"
        assert 'socket' not in module_imports, "socket should not be imported"
        assert 'random' not in module_imports, "random should not be imported"


# =============================================================================
# TASK 1: RESOURCE BUDGET VALIDATION TESTS
# =============================================================================

@pytest.mark.unit
class TestResourceBudgetValidation:
    """Tests for LeanResourceBudget validation (Task 1)."""

    def test_valid_budget_passes(self) -> None:
        """Valid budget should not raise."""
        budget = LeanResourceBudget(
            timeout_seconds=30,
            memory_mb=2048,
            disk_mb=100,
            max_proofs=10,
        )
        assert budget.timeout_seconds == 30

    def test_negative_timeout_fails(self) -> None:
        """Negative timeout should fail validation."""
        from backend.verification import LeanAdapterValidationError
        
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanResourceBudget(timeout_seconds=0)
        assert "timeout_seconds" in str(exc_info.value)

    def test_timeout_exceeds_max_fails(self) -> None:
        """Timeout exceeding max should fail validation."""
        from backend.verification import LeanAdapterValidationError
        
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanResourceBudget(timeout_seconds=100)  # Max is 90
        assert "timeout_seconds" in str(exc_info.value)
        assert "exceeds maximum" in str(exc_info.value)

    def test_negative_memory_fails(self) -> None:
        """Negative memory should fail validation."""
        from backend.verification import LeanAdapterValidationError
        
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanResourceBudget(memory_mb=0)
        assert "memory_mb" in str(exc_info.value)

    def test_memory_exceeds_max_fails(self) -> None:
        """Memory exceeding max should fail validation."""
        from backend.verification import LeanAdapterValidationError
        
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanResourceBudget(memory_mb=10000)  # Max is 8192
        assert "memory_mb" in str(exc_info.value)

    def test_negative_disk_fails(self) -> None:
        """Negative disk should fail validation."""
        from backend.verification import LeanAdapterValidationError
        
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanResourceBudget(disk_mb=0)
        assert "disk_mb" in str(exc_info.value)

    def test_disk_exceeds_max_fails(self) -> None:
        """Disk exceeding max should fail validation."""
        from backend.verification import LeanAdapterValidationError
        
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanResourceBudget(disk_mb=2000)  # Max is 1024
        assert "disk_mb" in str(exc_info.value)

    def test_negative_max_proofs_fails(self) -> None:
        """Negative max_proofs should fail validation."""
        from backend.verification import LeanAdapterValidationError
        
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanResourceBudget(max_proofs=0)
        assert "max_proofs" in str(exc_info.value)

    def test_max_proofs_exceeds_max_fails(self) -> None:
        """max_proofs exceeding max should fail validation."""
        from backend.verification import LeanAdapterValidationError
        
        with pytest.raises(LeanAdapterValidationError) as exc_info:
            LeanResourceBudget(max_proofs=200)  # Max is 100
        assert "max_proofs" in str(exc_info.value)

    def test_budget_factory_methods(self) -> None:
        """Test factory methods for convenience budgets."""
        default = LeanResourceBudget.default()
        assert default.timeout_seconds == 30
        
        minimal = LeanResourceBudget.minimal()
        assert minimal.timeout_seconds == 5
        assert minimal.max_proofs == 1
        
        generous = LeanResourceBudget.generous()
        assert generous.timeout_seconds == 60
        assert generous.max_proofs == 50

    def test_budget_to_dict_includes_max_proofs(self) -> None:
        """Budget serialization should include max_proofs."""
        budget = LeanResourceBudget()
        d = budget.to_dict()
        
        assert "max_proofs" in d
        assert d["max_proofs"] == 10

    def test_validate_resource_budget_function(self) -> None:
        """Test validate_resource_budget returns list of errors."""
        from backend.verification import validate_resource_budget
        
        # Create invalid budget by bypassing constructor
        budget = object.__new__(LeanResourceBudget)
        object.__setattr__(budget, 'timeout_seconds', 0)
        object.__setattr__(budget, 'memory_mb', 0)
        object.__setattr__(budget, 'disk_mb', 0)
        object.__setattr__(budget, 'max_proofs', 0)
        
        errors = validate_resource_budget(budget)
        
        assert len(errors) == 4
        assert any("timeout" in e for e in errors)
        assert any("memory" in e for e in errors)
        assert any("disk" in e for e in errors)
        assert any("proofs" in e for e in errors)


@pytest.mark.unit
class TestResourceBudgetInMetadata:
    """Tests for resource budget propagation in result metadata (Task 1)."""

    def test_simulation_echoes_budget_in_metadata(self) -> None:
        """Simulation result should include resource_budget_applied."""
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        budget = LeanResourceBudget(timeout_seconds=45, memory_mb=1024)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="budget_test",
            resource_budget=budget,
        )
        
        result = adapter.verify(request)
        
        assert result.resource_budget_applied is not None
        assert result.resource_budget_applied["timeout_seconds"] == 45
        assert result.resource_budget_applied["memory_mb"] == 1024

    def test_scaffold_echoes_budget_in_metadata(self) -> None:
        """Scaffold result should include resource_budget_applied."""
        adapter = LeanAdapter(mode=LeanAdapterMode.PHASE_IIB_SCAFFOLD)
        budget = LeanResourceBudget(timeout_seconds=20)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="scaffold_budget_test",
            resource_budget=budget,
        )
        
        result = adapter.verify(request)
        
        assert result.resource_budget_applied is not None
        assert result.resource_budget_applied["timeout_seconds"] == 20

    def test_disabled_echoes_budget_in_metadata(self) -> None:
        """Disabled result should include resource_budget_applied."""
        adapter = LeanAdapter(mode=LeanAdapterMode.DISABLED)
        budget = LeanResourceBudget(disk_mb=50)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="disabled_budget_test",
            resource_budget=budget,
        )
        
        result = adapter.verify(request)
        
        assert result.resource_budget_applied is not None
        assert result.resource_budget_applied["disk_mb"] == 50

    def test_to_dict_includes_budget(self) -> None:
        """Result to_dict should include resource_budget_applied."""
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="dict_test",
        )
        
        result = adapter.verify(request)
        d = result.to_dict()
        
        assert "resource_budget_applied" in d


# =============================================================================
# TASK 2: ERROR TAXONOMY TESTS
# =============================================================================

@pytest.mark.unit
class TestVerificationErrorKind:
    """Tests for VerificationErrorKind enum (Task 2)."""

    def test_all_error_kinds_have_values(self) -> None:
        """All error kinds should have string values."""
        from backend.verification import VerificationErrorKind
        
        assert VerificationErrorKind.NONE.value == "none"
        assert VerificationErrorKind.INVALID_REQUEST.value == "invalid_request"
        assert VerificationErrorKind.SIMULATION_ONLY.value == "simulation_only"
        assert VerificationErrorKind.RESOURCE_LIMIT.value == "resource_limit"
        assert VerificationErrorKind.INTERNAL_ERROR.value == "internal_error"
        assert VerificationErrorKind.LEAN_UNAVAILABLE.value == "lean_unavailable"
        assert VerificationErrorKind.NOT_IMPLEMENTED.value == "not_implemented"


@pytest.mark.unit
class TestErrorKindInResults:
    """Tests for error_kind in verification results (Task 2)."""

    def test_simulation_success_has_none_error_kind(self) -> None:
        """Successful simulation should have NONE error_kind."""
        from backend.verification import VerificationErrorKind
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        # Find a request that produces success
        for i in range(50):
            request = LeanVerificationRequest(
                canonical=f"p{i}->p{i}",
                job_id=f"success_search_{i}",
            )
            result = adapter.verify(request)
            if result.verified:
                assert result.error_kind == VerificationErrorKind.NONE
                return
        
        pytest.skip("Could not find a successful simulation in 50 tries")

    def test_simulation_failure_has_simulation_only_kind(self) -> None:
        """Failed simulation should have SIMULATION_ONLY error_kind."""
        from backend.verification import VerificationErrorKind
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        # Find a request that produces failure (not timeout/resource)
        for i in range(50):
            request = LeanVerificationRequest(
                canonical=f"q{i}->q{i}",
                job_id=f"failure_search_{i}",
            )
            result = adapter.verify(request)
            if not result.verified and result.abstention_reason == LeanAbstentionReason.LEAN_ERROR:
                assert result.error_kind == VerificationErrorKind.SIMULATION_ONLY
                return
        
        pytest.skip("Could not find a failed simulation in 50 tries")

    def test_timeout_has_resource_limit_kind(self) -> None:
        """Timeout simulation should have RESOURCE_LIMIT error_kind."""
        from backend.verification import VerificationErrorKind
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        # Use a formula > 50 chars to trigger timeout
        request = LeanVerificationRequest(
            canonical="p->" * 20 + "p",  # 63 chars
            job_id="timeout_test",
        )
        
        result = adapter.verify(request)
        
        assert result.error_kind == VerificationErrorKind.RESOURCE_LIMIT
        assert result.abstention_reason == LeanAbstentionReason.LEAN_TIMEOUT

    def test_resource_exceeded_has_resource_limit_kind(self) -> None:
        """Resource exceeded simulation should have RESOURCE_LIMIT error_kind."""
        from backend.verification import VerificationErrorKind
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        # Use a formula > 80 chars to trigger resource exceeded
        request = LeanVerificationRequest(
            canonical="p->" * 30 + "p",  # 93 chars
            job_id="resource_test",
        )
        
        result = adapter.verify(request)
        
        assert result.error_kind == VerificationErrorKind.RESOURCE_LIMIT
        assert result.abstention_reason == LeanAbstentionReason.LEAN_RESOURCE_EXCEEDED

    def test_scaffold_has_simulation_only_kind(self) -> None:
        """Scaffold mode should have SIMULATION_ONLY error_kind."""
        from backend.verification import VerificationErrorKind
        
        adapter = LeanAdapter(mode=LeanAdapterMode.PHASE_IIB_SCAFFOLD)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="scaffold_kind_test",
        )
        
        result = adapter.verify(request)
        
        assert result.error_kind == VerificationErrorKind.SIMULATION_ONLY

    def test_disabled_has_simulation_only_kind(self) -> None:
        """Disabled mode should have SIMULATION_ONLY error_kind."""
        from backend.verification import VerificationErrorKind
        
        adapter = LeanAdapter(mode=LeanAdapterMode.DISABLED)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="disabled_kind_test",
        )
        
        result = adapter.verify(request)
        
        assert result.error_kind == VerificationErrorKind.SIMULATION_ONLY

    def test_to_dict_includes_error_kind(self) -> None:
        """Result to_dict should include error_kind."""
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="error_kind_dict_test",
        )
        
        result = adapter.verify(request)
        d = result.to_dict()
        
        assert "error_kind" in d
        assert d["error_kind"] in ["none", "simulation_only", "resource_limit"]

    def test_is_success_property(self) -> None:
        """Test is_success property on LeanVerificationResult."""
        from backend.verification import VerificationErrorKind
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        for i in range(50):
            request = LeanVerificationRequest(
                canonical=f"p{i}->p{i}",
                job_id=f"is_success_{i}",
            )
            result = adapter.verify(request)
            if result.verified:
                assert result.is_success is True
                return
        
        pytest.skip("Could not find a success")

    def test_is_error_property(self) -> None:
        """Test is_error property on LeanVerificationResult."""
        from backend.verification import VerificationErrorKind
        
        adapter = LeanAdapter(mode=LeanAdapterMode.PHASE_IIB_SCAFFOLD)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="is_error_test",
        )
        
        result = adapter.verify(request)
        
        assert result.is_error is True  # SIMULATION_ONLY is not NONE


# =============================================================================
# TASK 3: EVIDENCE PACK SUMMARY TESTS
# =============================================================================

@pytest.mark.unit
class TestSummarizeLeanActivity:
    """Tests for summarize_lean_activity() helper (Task 3)."""

    def test_empty_results_summary(self) -> None:
        """Empty results should produce zero counts."""
        from backend.verification import summarize_lean_activity
        
        summary = summarize_lean_activity([])
        
        assert summary["total_requests"] == 0
        assert summary["success_count"] == 0
        assert summary["abstention_count"] == 0
        assert summary["error_kinds_histogram"] == {}
        assert summary["methods_histogram"] == {}
        assert summary["version_pin"] == LEAN_VERSION_REQUIRED

    def test_summary_counts_successes(self) -> None:
        """Summary should count successful verifications."""
        from backend.verification import summarize_lean_activity
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        for i in range(20):
            request = LeanVerificationRequest(
                canonical=f"p{i}->p{i}",
                job_id=f"summary_test_{i}",
            )
            results.append(adapter.verify(request))
        
        summary = summarize_lean_activity(results)
        
        assert summary["total_requests"] == 20
        assert summary["success_count"] > 0  # ~70% success rate
        assert summary["version_pin"] == LEAN_VERSION_REQUIRED

    def test_summary_counts_abstentions(self) -> None:
        """Summary should count abstentions."""
        from backend.verification import summarize_lean_activity
        
        adapter = LeanAdapter(mode=LeanAdapterMode.PHASE_IIB_SCAFFOLD)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id=f"abstain_{i}"))
            for i in range(5)
        ]
        
        summary = summarize_lean_activity(results)
        
        assert summary["abstention_count"] == 5  # All scaffold results are abstentions

    def test_summary_error_kinds_histogram(self) -> None:
        """Summary should include error_kinds_histogram."""
        from backend.verification import summarize_lean_activity
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Add some simple formulas (may succeed or fail with SIMULATION_ONLY)
        for i in range(5):
            request = LeanVerificationRequest(
                canonical=f"p{i}->p{i}",
                job_id=f"hist_simple_{i}",
            )
            results.append(adapter.verify(request))
        
        # Add timeout formulas (RESOURCE_LIMIT)
        for i in range(3):
            request = LeanVerificationRequest(
                canonical="p->" * 20 + "p",
                job_id=f"hist_timeout_{i}",
            )
            results.append(adapter.verify(request))
        
        summary = summarize_lean_activity(results)
        
        assert "error_kinds_histogram" in summary
        assert "resource_limit" in summary["error_kinds_histogram"]
        assert summary["error_kinds_histogram"]["resource_limit"] == 3

    def test_summary_methods_histogram(self) -> None:
        """Summary should include methods_histogram."""
        from backend.verification import summarize_lean_activity
        
        # Mix of simulation and scaffold results
        simulate_adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        scaffold_adapter = LeanAdapter(mode=LeanAdapterMode.PHASE_IIB_SCAFFOLD)
        
        results = []
        for i in range(3):
            results.append(simulate_adapter.verify(
                LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"sim_{i}")
            ))
        for i in range(2):
            results.append(scaffold_adapter.verify(
                LeanVerificationRequest(canonical=f"q{i}->q{i}", job_id=f"scaf_{i}")
            ))
        
        summary = summarize_lean_activity(results)
        
        assert "methods_histogram" in summary
        assert "lean_adapter_simulate" in summary["methods_histogram"]
        assert "lean_adapter_scaffold" in summary["methods_histogram"]
        assert summary["methods_histogram"]["lean_adapter_simulate"] == 3
        assert summary["methods_histogram"]["lean_adapter_scaffold"] == 2

    def test_summary_is_deterministic(self) -> None:
        """Summary should be deterministic for same inputs."""
        from backend.verification import summarize_lean_activity
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"det_{i}"))
            for i in range(10)
        ]
        
        summary1 = summarize_lean_activity(results)
        summary2 = summarize_lean_activity(results)
        
        assert summary1 == summary2

    def test_summary_json_serializable(self) -> None:
        """Summary should be JSON serializable."""
        import json
        from backend.verification import summarize_lean_activity
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"json_{i}"))
            for i in range(5)
        ]
        
        summary = summarize_lean_activity(results)
        
        # Should not raise
        json_str = json.dumps(summary)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed == summary


# =============================================================================
# PHASE III: ACTIVITY LEDGER TESTS
# =============================================================================

@pytest.mark.unit
class TestBuildLeanActivityLedger:
    """Tests for build_lean_activity_ledger (Phase III Task 1)."""

    def test_ledger_empty_results(self) -> None:
        """Ledger should handle empty results gracefully."""
        from backend.verification import build_lean_activity_ledger
        
        ledger = build_lean_activity_ledger([])
        
        assert ledger["schema_version"] == "1.0.0"
        assert ledger["total_requests"] == 0
        assert ledger["success_count"] == 0
        assert ledger["abstention_count"] == 0
        assert ledger["error_kind_histogram"] == {}
        assert ledger["resource_budget_histogram"] == {}
        assert ledger["max_resource_budget_observed"]["timeout_seconds"] == 0
        assert ledger["max_resource_budget_observed"]["memory_mb"] == 0
        assert ledger["max_resource_budget_observed"]["disk_mb"] == 0
        assert ledger["max_resource_budget_observed"]["max_proofs"] == 0
        assert "timestamp_utc" in ledger
        assert "version_pin" in ledger

    def test_ledger_with_results(self) -> None:
        """Ledger should aggregate results correctly."""
        from backend.verification import build_lean_activity_ledger
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Add some successful/failing results
        for i in range(10):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"ledger_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        
        assert ledger["schema_version"] == "1.0.0"
        assert ledger["total_requests"] == 10
        assert "error_kind_histogram" in ledger
        assert "resource_budget_histogram" in ledger
        assert "max_resource_budget_observed" in ledger
        assert "methods_histogram" in ledger

    def test_ledger_schema_version(self) -> None:
        """Ledger should have stable schema_version."""
        from backend.verification import build_lean_activity_ledger
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="schema_test"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        
        assert ledger["schema_version"] == "1.0.0"

    def test_ledger_resource_budget_histogram(self) -> None:
        """Ledger should track resource budget archetypes."""
        from backend.verification import (
            build_lean_activity_ledger,
            LeanResourceBudget,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Add results with different budget archetypes
        # Minimal budget
        results.append(adapter.verify(
            LeanVerificationRequest(
                canonical="p->p",
                job_id="minimal_budget",
                resource_budget=LeanResourceBudget.minimal(),
            )
        ))
        
        # Generous budget
        results.append(adapter.verify(
            LeanVerificationRequest(
                canonical="q->q",
                job_id="generous_budget",
                resource_budget=LeanResourceBudget.generous(),
            )
        ))
        
        # Default budget
        results.append(adapter.verify(
            LeanVerificationRequest(
                canonical="r->r",
                job_id="default_budget",
                resource_budget=LeanResourceBudget.default(),
            )
        ))
        
        ledger = build_lean_activity_ledger(results)
        
        assert "resource_budget_histogram" in ledger
        histogram = ledger["resource_budget_histogram"]
        assert "minimal" in histogram
        assert "generous" in histogram
        assert "default" in histogram

    def test_ledger_max_resource_observed(self) -> None:
        """Ledger should track maximum resource values observed."""
        from backend.verification import (
            build_lean_activity_ledger,
            LeanResourceBudget,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        
        # Create results with varying budgets
        results = [
            adapter.verify(LeanVerificationRequest(
                canonical="p->p",
                job_id="max_test_1",
                resource_budget=LeanResourceBudget(
                    timeout_seconds=10, memory_mb=1024, disk_mb=50, max_proofs=5
                ),
            )),
            adapter.verify(LeanVerificationRequest(
                canonical="q->q",
                job_id="max_test_2",
                resource_budget=LeanResourceBudget(
                    timeout_seconds=60, memory_mb=4096, disk_mb=200, max_proofs=20
                ),
            )),
        ]
        
        ledger = build_lean_activity_ledger(results)
        max_observed = ledger["max_resource_budget_observed"]
        
        assert max_observed["timeout_seconds"] == 60
        assert max_observed["memory_mb"] == 4096
        assert max_observed["disk_mb"] == 200
        assert max_observed["max_proofs"] == 20

    def test_ledger_json_serializable(self) -> None:
        """Ledger should be JSON serializable."""
        import json
        from backend.verification import build_lean_activity_ledger
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"json_ledger_{i}"))
            for i in range(5)
        ]
        
        ledger = build_lean_activity_ledger(results)
        
        # Should not raise
        json_str = json.dumps(ledger)
        assert isinstance(json_str, str)
        
        # Should round-trip (except timestamp which may differ)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == ledger["schema_version"]
        assert parsed["total_requests"] == ledger["total_requests"]

    def test_ledger_error_kind_histogram(self) -> None:
        """Ledger should include error_kind histogram."""
        from backend.verification import build_lean_activity_ledger
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Success/no error results
        for i in range(5):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"hist_success_{i}")
            ))
        
        # Timeout formulas (resource_limit)
        for i in range(3):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical="p->" * 20 + "p", job_id=f"hist_timeout_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        
        assert "error_kind_histogram" in ledger
        assert "resource_limit" in ledger["error_kind_histogram"]

    def test_ledger_has_timestamp(self) -> None:
        """Ledger should include a UTC timestamp."""
        from backend.verification import build_lean_activity_ledger
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="timestamp_test"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        
        assert "timestamp_utc" in ledger
        # Should be ISO format ending in Z
        assert ledger["timestamp_utc"].endswith("Z")


# =============================================================================
# PHASE III: SAFETY ENVELOPE TESTS
# =============================================================================

@pytest.mark.unit
class TestEvaluateLeanAdapterSafety:
    """Tests for evaluate_lean_adapter_safety (Phase III Task 2)."""

    def test_safety_ok_on_clean_results(self) -> None:
        """Safety should be OK when no errors present."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        # Simple formulas that succeed
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"safe_{i}"))
            for i in range(10)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        
        assert safety["has_internal_errors"] is False
        assert safety["has_resource_issues"] is False
        assert safety["safety_status"] == "OK"
        assert "no safety concerns" in safety["reasons"]

    def test_safety_warn_on_resource_limit(self) -> None:
        """Safety should WARN when resource limits hit."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Add timeouts (resource_limit)
        for i in range(3):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical="p->" * 20 + "p", job_id=f"warn_timeout_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        
        assert safety["has_resource_issues"] is True
        assert safety["safety_status"] == "WARN"
        assert safety["resource_limit_count"] > 0

    def test_safety_block_on_internal_error(self) -> None:
        """Safety should BLOCK when internal errors present."""
        from backend.verification import evaluate_lean_adapter_safety
        
        # Manually create a ledger with internal_error
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 5,
            "success_count": 3,
            "abstention_count": 2,
            "error_kind_histogram": {
                "none": 3,
                "internal_error": 2,
            },
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = evaluate_lean_adapter_safety(ledger)
        
        assert safety["has_internal_errors"] is True
        assert safety["safety_status"] == "BLOCK"
        assert safety["internal_error_count"] == 2
        assert "internal_error" in str(safety["reasons"])

    def test_safety_block_on_lean_unavailable(self) -> None:
        """Safety should BLOCK when Lean is unavailable."""
        from backend.verification import evaluate_lean_adapter_safety
        
        # Manually create a ledger with lean_unavailable
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 5,
            "success_count": 2,
            "abstention_count": 3,
            "error_kind_histogram": {
                "none": 2,
                "lean_unavailable": 3,
            },
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = evaluate_lean_adapter_safety(ledger)
        
        assert safety["has_internal_errors"] is True
        assert safety["safety_status"] == "BLOCK"
        assert safety["internal_error_count"] == 3
        assert "lean_unavailable" in str(safety["reasons"])

    def test_safety_block_trumps_warn(self) -> None:
        """BLOCK should override WARN when both conditions present."""
        from backend.verification import evaluate_lean_adapter_safety
        
        # Ledger with both internal_error and resource_limit
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 10,
            "success_count": 5,
            "abstention_count": 5,
            "error_kind_histogram": {
                "none": 5,
                "internal_error": 2,
                "resource_limit": 3,
            },
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = evaluate_lean_adapter_safety(ledger)
        
        # BLOCK should win
        assert safety["safety_status"] == "BLOCK"
        assert safety["has_internal_errors"] is True
        assert safety["has_resource_issues"] is True

    def test_safety_reasons_populated(self) -> None:
        """Safety evaluation should always have reasons."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="reasons_test"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        
        assert "reasons" in safety
        assert len(safety["reasons"]) > 0

    def test_safety_counts_match_histogram(self) -> None:
        """Safety counts should match histogram values."""
        from backend.verification import evaluate_lean_adapter_safety
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 10,
            "success_count": 5,
            "abstention_count": 5,
            "error_kind_histogram": {
                "none": 5,
                "internal_error": 1,
                "lean_unavailable": 2,
                "resource_limit": 2,
            },
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = evaluate_lean_adapter_safety(ledger)
        
        # internal_error + lean_unavailable = 3
        assert safety["internal_error_count"] == 3
        assert safety["resource_limit_count"] == 2

    def test_safety_empty_ledger(self) -> None:
        """Safety should handle empty ledger."""
        from backend.verification import evaluate_lean_adapter_safety
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 0,
            "success_count": 0,
            "abstention_count": 0,
            "error_kind_histogram": {},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = evaluate_lean_adapter_safety(ledger)
        
        assert safety["safety_status"] == "OK"
        assert safety["has_internal_errors"] is False
        assert safety["has_resource_issues"] is False


# =============================================================================
# PHASE III: GLOBAL HEALTH TESTS
# =============================================================================

@pytest.mark.unit
class TestSummarizeLeanForGlobalHealth:
    """Tests for summarize_lean_for_global_health (Phase III Task 3)."""

    def test_global_health_ok_status(self) -> None:
        """Global health should reflect OK status."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            summarize_lean_for_global_health,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"health_ok_{i}"))
            for i in range(10)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        health = summarize_lean_for_global_health(ledger, safety)
        
        assert health["lean_surface_ok"] is True
        assert health["status"] == "OK"
        assert health["total_requests"] == 10

    def test_global_health_warn_status(self) -> None:
        """Global health should reflect WARN status."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            summarize_lean_for_global_health,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Add timeouts
        for i in range(5):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical="p->" * 20 + "p", job_id=f"health_warn_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        health = summarize_lean_for_global_health(ledger, safety)
        
        assert health["lean_surface_ok"] is False
        assert health["status"] == "WARN"

    def test_global_health_block_status(self) -> None:
        """Global health should reflect BLOCK status."""
        from backend.verification import summarize_lean_for_global_health
        
        # Manual ledger with internal error
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 5,
            "success_count": 2,
            "abstention_count": 3,
            "error_kind_histogram": {"internal_error": 3},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = {
            "has_internal_errors": True,
            "has_resource_issues": False,
            "safety_status": "BLOCK",
            "reasons": ["internal_error: 3 occurrences"],
            "internal_error_count": 3,
            "resource_limit_count": 0,
        }
        
        health = summarize_lean_for_global_health(ledger, safety)
        
        assert health["lean_surface_ok"] is False
        assert health["status"] == "BLOCK"
        assert health["internal_error_count"] == 3

    def test_global_health_success_rate(self) -> None:
        """Global health should compute correct success rate."""
        from backend.verification import summarize_lean_for_global_health
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 100,
            "success_count": 75,
            "abstention_count": 25,
            "error_kind_histogram": {},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": ["no safety concerns"],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        health = summarize_lean_for_global_health(ledger, safety)
        
        assert health["success_rate"] == 0.75

    def test_global_health_zero_requests(self) -> None:
        """Global health should handle zero requests."""
        from backend.verification import summarize_lean_for_global_health
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 0,
            "success_count": 0,
            "abstention_count": 0,
            "error_kind_histogram": {},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": ["no safety concerns"],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        health = summarize_lean_for_global_health(ledger, safety)
        
        assert health["total_requests"] == 0
        assert health["success_rate"] == 0.0
        assert health["lean_surface_ok"] is True

    def test_global_health_includes_version_pin(self) -> None:
        """Global health should include version_pin."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            summarize_lean_for_global_health,
            LEAN_VERSION_REQUIRED,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="version_health"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        health = summarize_lean_for_global_health(ledger, safety)
        
        assert "version_pin" in health
        assert health["version_pin"] == LEAN_VERSION_REQUIRED

    def test_global_health_json_serializable(self) -> None:
        """Global health should be JSON serializable."""
        import json
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            summarize_lean_for_global_health,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"json_health_{i}"))
            for i in range(5)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        health = summarize_lean_for_global_health(ledger, safety)
        
        # Should not raise
        json_str = json.dumps(health)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed == health

    def test_global_health_counts_from_safety(self) -> None:
        """Global health should propagate counts from safety eval."""
        from backend.verification import summarize_lean_for_global_health
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 20,
            "success_count": 10,
            "abstention_count": 10,
            "error_kind_histogram": {},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = {
            "has_internal_errors": False,
            "has_resource_issues": True,
            "safety_status": "WARN",
            "reasons": ["resource_limit: 5 occurrences"],
            "internal_error_count": 0,
            "resource_limit_count": 5,
        }
        
        health = summarize_lean_for_global_health(ledger, safety)
        
        assert health["internal_error_count"] == 0
        assert health["resource_limit_count"] == 5


# =============================================================================
# PHASE III: INTEGRATION TESTS
# =============================================================================

@pytest.mark.unit
class TestPhaseIIIIntegration:
    """Integration tests for Phase III features together."""

    def test_full_pipeline_ok(self) -> None:
        """Full pipeline: results -> ledger -> safety -> health (OK path)."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            summarize_lean_for_global_health,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"pipeline_ok_{i}"))
            for i in range(20)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        health = summarize_lean_for_global_health(ledger, safety)
        
        # Should be OK
        assert safety["safety_status"] == "OK"
        assert health["lean_surface_ok"] is True
        assert health["total_requests"] == 20

    def test_full_pipeline_warn(self) -> None:
        """Full pipeline: results -> ledger -> safety -> health (WARN path)."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            summarize_lean_for_global_health,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Mix of success and timeout
        for i in range(10):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"pipeline_warn_ok_{i}")
            ))
        for i in range(5):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical="p->" * 20 + "p", job_id=f"pipeline_warn_timeout_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        health = summarize_lean_for_global_health(ledger, safety)
        
        # Should be WARN due to resource limits
        assert safety["safety_status"] == "WARN"
        assert health["lean_surface_ok"] is False
        assert health["status"] == "WARN"
        assert health["resource_limit_count"] > 0

    def test_full_pipeline_deterministic(self) -> None:
        """Full pipeline should be deterministic."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            summarize_lean_for_global_health,
        )
        
        def run_pipeline():
            adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
            results = [
                adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"det_pipeline_{i}"))
                for i in range(10)
            ]
            ledger = build_lean_activity_ledger(results)
            safety = evaluate_lean_adapter_safety(ledger)
            health = summarize_lean_for_global_health(ledger, safety)
            return (ledger, safety, health)
        
        # Run twice
        ledger1, safety1, health1 = run_pipeline()
        ledger2, safety2, health2 = run_pipeline()
        
        # Ledgers differ in timestamp, but other fields should match
        assert ledger1["total_requests"] == ledger2["total_requests"]
        assert ledger1["success_count"] == ledger2["success_count"]
        assert ledger1["error_kind_histogram"] == ledger2["error_kind_histogram"]
        
        # Safety should be identical
        assert safety1 == safety2
        
        # Health should be identical
        assert health1 == health2


# =============================================================================
# PHASE IV: CAPABILITY CLASSIFICATION TESTS
# =============================================================================

@pytest.mark.unit
class TestClassifyLeanCapabilities:
    """Tests for classify_lean_capabilities (Phase IV Task 1)."""

    def test_capability_basic_band(self) -> None:
        """Capability should be BASIC for low success rate."""
        from backend.verification import (
            build_lean_activity_ledger,
            classify_lean_capabilities,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Add mostly failing results (timeouts)
        for i in range(10):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical="p->" * 20 + "p", job_id=f"basic_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        capability = classify_lean_capabilities(ledger)
        
        assert capability["capability_band"] == "BASIC"
        assert capability["simulation_only"] is True

    def test_capability_intermediate_band(self) -> None:
        """Capability should be INTERMEDIATE for moderate success rate."""
        from backend.verification import (
            build_lean_activity_ledger,
            classify_lean_capabilities,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Mix of success and some timeouts
        for i in range(7):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"intermediate_ok_{i}")
            ))
        for i in range(3):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical="p->" * 20 + "p", job_id=f"intermediate_timeout_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        capability = classify_lean_capabilities(ledger)
        
        assert capability["capability_band"] == "INTERMEDIATE"

    def test_capability_advanced_band(self) -> None:
        """Capability should be ADVANCED for high success rate with no resource limits."""
        from backend.verification import (
            build_lean_activity_ledger,
            classify_lean_capabilities,
        )
        
        # Create a manual ledger with high success rate and no resource limits
        # (simulation has ~70% success rate, so we need to manually construct
        # a ledger that represents ADVANCED capability)
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 20,
            "success_count": 18,  # 90% success rate
            "abstention_count": 2,
            "error_kind_histogram": {
                "none": 18,
                "simulation_only": 2,
            },
            "resource_budget_histogram": {"default": 20},
            "max_resource_budget_observed": {
                "timeout_seconds": 30,
                "memory_mb": 2048,
                "disk_mb": 100,
                "max_proofs": 10,
            },
            "methods_histogram": {"lean_adapter_simulate": 20},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        capability = classify_lean_capabilities(ledger)
        
        assert capability["capability_band"] == "ADVANCED"

    def test_capability_empty_ledger(self) -> None:
        """Capability should handle empty ledger."""
        from backend.verification import classify_lean_capabilities
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 0,
            "success_count": 0,
            "abstention_count": 0,
            "error_kind_histogram": {},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        capability = classify_lean_capabilities(ledger)
        
        assert capability["capability_band"] == "BASIC"
        assert capability["simulation_only"] is True

    def test_capability_schema_version(self) -> None:
        """Capability should have schema_version."""
        from backend.verification import (
            build_lean_activity_ledger,
            classify_lean_capabilities,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="schema_test"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        capability = classify_lean_capabilities(ledger)
        
        assert capability["schema_version"] == "1.0.0"

    def test_capability_max_budget_used(self) -> None:
        """Capability should include max_budget_used."""
        from backend.verification import (
            build_lean_activity_ledger,
            classify_lean_capabilities,
            LeanResourceBudget,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(
                canonical="p->p",
                job_id="budget_test",
                resource_budget=LeanResourceBudget(
                    timeout_seconds=60, memory_mb=4096, disk_mb=200, max_proofs=20
                ),
            ))
        ]
        
        ledger = build_lean_activity_ledger(results)
        capability = classify_lean_capabilities(ledger)
        
        assert "max_budget_used" in capability
        assert capability["max_budget_used"]["timeout_seconds"] == 60
        assert capability["max_budget_used"]["memory_mb"] == 4096

    def test_capability_resource_profile(self) -> None:
        """Capability should include resource_profile description."""
        from backend.verification import (
            build_lean_activity_ledger,
            classify_lean_capabilities,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="profile_test"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        capability = classify_lean_capabilities(ledger)
        
        assert "resource_profile" in capability
        assert isinstance(capability["resource_profile"], str)
        assert len(capability["resource_profile"]) > 0

    def test_capability_simulation_only(self) -> None:
        """Capability should detect simulation-only mode."""
        from backend.verification import (
            build_lean_activity_ledger,
            classify_lean_capabilities,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="sim_only_test"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        capability = classify_lean_capabilities(ledger)
        
        assert capability["simulation_only"] is True

    def test_capability_json_serializable(self) -> None:
        """Capability should be JSON serializable."""
        import json
        from backend.verification import (
            build_lean_activity_ledger,
            classify_lean_capabilities,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="json_cap_test"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        capability = classify_lean_capabilities(ledger)
        
        # Should not raise
        json_str = json.dumps(capability)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["capability_band"] == capability["capability_band"]


# =============================================================================
# PHASE IV: MIGRATION CHECKLIST TESTS
# =============================================================================

@pytest.mark.unit
class TestBuildLeanMigrationChecklist:
    """Tests for build_lean_migration_checklist (Phase IV Task 2)."""

    def test_checklist_all_pass(self) -> None:
        """Checklist should pass when all conditions met."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            build_lean_migration_checklist,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"checklist_ok_{i}"))
            for i in range(20)  # Enough for minimum volume
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        checklist = build_lean_migration_checklist(ledger, safety)
        
        assert checklist["can_enable_live_mode"] is True
        assert len(checklist["blocking_reasons"]) == 0
        assert len(checklist["checklist_items"]) > 0
        
        # All items should be PASS
        for item in checklist["checklist_items"]:
            assert item["status"] == "PASS"

    def test_checklist_blocks_on_safety_status(self) -> None:
        """Checklist should block when safety status is not OK."""
        from backend.verification import build_lean_migration_checklist
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 20,
            "success_count": 10,
            "abstention_count": 10,
            "error_kind_histogram": {"internal_error": 5},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = {
            "has_internal_errors": True,
            "has_resource_issues": False,
            "safety_status": "BLOCK",
            "reasons": ["internal_error: 5 occurrences"],
            "internal_error_count": 5,
            "resource_limit_count": 0,
        }
        
        checklist = build_lean_migration_checklist(ledger, safety)
        
        assert checklist["can_enable_live_mode"] is False
        assert len(checklist["blocking_reasons"]) > 0
        assert "BLOCK" in str(checklist["blocking_reasons"])

    def test_checklist_blocks_on_low_volume(self) -> None:
        """Checklist should block when request volume is too low."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            build_lean_migration_checklist,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="low_vol"))
        ]  # Only 1 request
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        checklist = build_lean_migration_checklist(ledger, safety)
        
        assert checklist["can_enable_live_mode"] is False
        assert any("volume" in reason.lower() for reason in checklist["blocking_reasons"])

    def test_checklist_blocks_on_low_success_rate(self) -> None:
        """Checklist should block when success rate is too low."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            build_lean_migration_checklist,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # All timeouts (low success rate)
        for i in range(15):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical="p->" * 20 + "p", job_id=f"low_success_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        checklist = build_lean_migration_checklist(ledger, safety)
        
        assert checklist["can_enable_live_mode"] is False
        assert any("success rate" in reason.lower() for reason in checklist["blocking_reasons"])

    def test_checklist_blocks_on_high_resource_limits(self) -> None:
        """Checklist should block when resource limit rate is too high."""
        from backend.verification import build_lean_migration_checklist
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 20,
            "success_count": 5,
            "abstention_count": 15,
            "error_kind_histogram": {"resource_limit": 5},  # 25% > 10%
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = {
            "has_internal_errors": False,
            "has_resource_issues": True,
            "safety_status": "WARN",
            "reasons": ["resource_limit: 5 occurrences"],
            "internal_error_count": 0,
            "resource_limit_count": 5,
        }
        
        checklist = build_lean_migration_checklist(ledger, safety)
        
        assert checklist["can_enable_live_mode"] is False
        assert any("resource limit" in reason.lower() for reason in checklist["blocking_reasons"])

    def test_checklist_items_structure(self) -> None:
        """Checklist items should have correct structure."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            build_lean_migration_checklist,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"struct_{i}"))
            for i in range(15)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        checklist = build_lean_migration_checklist(ledger, safety)
        
        for item in checklist["checklist_items"]:
            assert "id" in item
            assert "description" in item
            assert "status" in item
            assert item["status"] in ("PASS", "FAIL")

    def test_checklist_version_pin_check(self) -> None:
        """Checklist should verify version pin."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            build_lean_migration_checklist,
            LEAN_VERSION_REQUIRED,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="version_check"))
            for _ in range(15)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        checklist = build_lean_migration_checklist(ledger, safety)
        
        # Find version pin check
        version_item = next(
            (item for item in checklist["checklist_items"] if item["id"] == "version_pin_correct"),
            None
        )
        assert version_item is not None
        assert LEAN_VERSION_REQUIRED in version_item["description"]

    def test_checklist_json_serializable(self) -> None:
        """Checklist should be JSON serializable."""
        import json
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            build_lean_migration_checklist,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="json_checklist"))
            for _ in range(15)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        checklist = build_lean_migration_checklist(ledger, safety)
        
        # Should not raise
        json_str = json.dumps(checklist)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["can_enable_live_mode"] == checklist["can_enable_live_mode"]


# =============================================================================
# PHASE IV: DIRECTOR PANEL TESTS
# =============================================================================

@pytest.mark.unit
class TestBuildLeanDirectorPanel:
    """Tests for build_lean_director_panel (Phase IV Task 3)."""

    def test_panel_green_status(self) -> None:
        """Panel should show GREEN for OK status."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            classify_lean_capabilities,
            build_lean_director_panel,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"green_{i}"))
            for i in range(20)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        panel = build_lean_director_panel(ledger, safety, capability)
        
        assert panel["status_light"] == "GREEN"
        assert panel["lean_surface_ok"] is True
        assert panel["safety_status"] == "OK"

    def test_panel_yellow_status(self) -> None:
        """Panel should show YELLOW for WARN status."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            classify_lean_capabilities,
            build_lean_director_panel,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = []
        
        # Mix with some timeouts
        for i in range(10):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"yellow_ok_{i}")
            ))
        for i in range(5):
            results.append(adapter.verify(
                LeanVerificationRequest(canonical="p->" * 20 + "p", job_id=f"yellow_timeout_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        panel = build_lean_director_panel(ledger, safety, capability)
        
        assert panel["status_light"] == "YELLOW"
        assert panel["safety_status"] == "WARN"

    def test_panel_red_status(self) -> None:
        """Panel should show RED for BLOCK status."""
        from backend.verification import build_lean_director_panel
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 10,
            "success_count": 5,
            "abstention_count": 5,
            "error_kind_histogram": {"internal_error": 3},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = {
            "has_internal_errors": True,
            "has_resource_issues": False,
            "safety_status": "BLOCK",
            "reasons": ["internal_error: 3 occurrences"],
            "internal_error_count": 3,
            "resource_limit_count": 0,
        }
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "BASIC",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        panel = build_lean_director_panel(ledger, safety, capability)
        
        assert panel["status_light"] == "RED"
        assert panel["lean_surface_ok"] is False
        assert panel["safety_status"] == "BLOCK"

    def test_panel_headline_includes_stats(self) -> None:
        """Panel headline should include request statistics."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            classify_lean_capabilities,
            build_lean_director_panel,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"headline_{i}"))
            for i in range(15)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        panel = build_lean_director_panel(ledger, safety, capability)
        
        assert "headline" in panel
        assert "15" in panel["headline"]  # total_requests
        assert isinstance(panel["headline"], str)

    def test_panel_headline_simulation_mode(self) -> None:
        """Panel headline should indicate simulation mode."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            classify_lean_capabilities,
            build_lean_director_panel,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="sim_headline"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        panel = build_lean_director_panel(ledger, safety, capability)
        
        assert "Simulation mode" in panel["headline"]

    def test_panel_includes_capability_band(self) -> None:
        """Panel should include capability_band."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            classify_lean_capabilities,
            build_lean_director_panel,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"cap_{i}"))
            for i in range(20)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        panel = build_lean_director_panel(ledger, safety, capability)
        
        assert "capability_band" in panel
        assert panel["capability_band"] in ("BASIC", "INTERMEDIATE", "ADVANCED")

    def test_panel_includes_error_counts(self) -> None:
        """Panel should include internal_error_count and resource_limit_count."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            classify_lean_capabilities,
            build_lean_director_panel,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="error_counts"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        panel = build_lean_director_panel(ledger, safety, capability)
        
        assert "internal_error_count" in panel
        assert "resource_limit_count" in panel
        assert isinstance(panel["internal_error_count"], int)
        assert isinstance(panel["resource_limit_count"], int)

    def test_panel_empty_ledger(self) -> None:
        """Panel should handle empty ledger."""
        from backend.verification import build_lean_director_panel
        
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 0,
            "success_count": 0,
            "abstention_count": 0,
            "error_kind_histogram": {},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": ["no safety concerns"],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "BASIC",
            "max_budget_used": {},
            "resource_profile": "No data",
            "simulation_only": True,
        }
        
        panel = build_lean_director_panel(ledger, safety, capability)
        
        assert panel["status_light"] == "GREEN"
        assert "No Lean verification activity" in panel["headline"]

    def test_panel_json_serializable(self) -> None:
        """Panel should be JSON serializable."""
        import json
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            classify_lean_capabilities,
            build_lean_director_panel,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical="p->p", job_id="json_panel"))
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        panel = build_lean_director_panel(ledger, safety, capability)
        
        # Should not raise
        json_str = json.dumps(panel)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["status_light"] == panel["status_light"]


# =============================================================================
# PHASE IV: INTEGRATION TESTS
# =============================================================================

@pytest.mark.unit
class TestPhaseIVIntegration:
    """Integration tests for Phase IV features together."""

    def test_full_phase_iv_pipeline(self) -> None:
        """Full pipeline: results -> ledger -> safety -> capability -> checklist -> panel."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            classify_lean_capabilities,
            build_lean_migration_checklist,
            build_lean_director_panel,
        )
        
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"phase4_{i}"))
            for i in range(20)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        checklist = build_lean_migration_checklist(ledger, safety)
        panel = build_lean_director_panel(ledger, safety, capability)
        
        # Verify all outputs are valid
        assert ledger["total_requests"] == 20
        assert safety["safety_status"] == "OK"
        assert capability["capability_band"] in ("BASIC", "INTERMEDIATE", "ADVANCED")
        assert "can_enable_live_mode" in checklist
        assert panel["status_light"] in ("GREEN", "YELLOW", "RED")

    def test_phase_iv_deterministic(self) -> None:
        """Phase IV pipeline should be deterministic."""
        from backend.verification import (
            build_lean_activity_ledger,
            evaluate_lean_adapter_safety,
            classify_lean_capabilities,
            build_lean_migration_checklist,
            build_lean_director_panel,
        )
        
        def run_pipeline():
            adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
            results = [
                adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"det_phase4_{i}"))
                for i in range(15)
            ]
            ledger = build_lean_activity_ledger(results)
            safety = evaluate_lean_adapter_safety(ledger)
            capability = classify_lean_capabilities(ledger)
            checklist = build_lean_migration_checklist(ledger, safety)
            panel = build_lean_director_panel(ledger, safety, capability)
            return (safety, capability, checklist, panel)
        
        # Run twice
        safety1, capability1, checklist1, panel1 = run_pipeline()
        safety2, capability2, checklist2, panel2 = run_pipeline()
        
        # All should be identical (except ledger timestamp)
        assert safety1 == safety2
        assert capability1 == capability2
        assert checklist1 == checklist2
        assert panel1 == panel2


# =============================================================================
# NEXT MISSION: LEAN MODE PLAYBOOK TESTS
# =============================================================================

@pytest.mark.unit
class TestBuildLeanModePlaybook:
    """Tests for build_lean_mode_playbook (Next Mission)."""

    def test_playbook_basic_band_simulation_only(self) -> None:
        """Playbook should recommend SIMULATION_ONLY for BASIC capability."""
        from backend.verification import (
            build_lean_mode_playbook,
            build_lean_migration_checklist,
        )
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "BASIC",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        checklist = {
            "can_enable_live_mode": False,
            "blocking_reasons": ["Safety status is WARN, must be OK"],
            "checklist_items": [],
        }
        
        playbook = build_lean_mode_playbook(capability, checklist)
        
        assert playbook["recommended_next_mode"] == "SIMULATION_ONLY"
        assert len(playbook["prerequisites"]) > 0
        assert "BASIC" in str(playbook["advisory_notes"])

    def test_playbook_intermediate_band_with_pass(self) -> None:
        """Playbook should recommend SHADOW for INTERMEDIATE with passing checklist."""
        from backend.verification import build_lean_mode_playbook
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "INTERMEDIATE",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        checklist = {
            "can_enable_live_mode": True,
            "blocking_reasons": [],
            "checklist_items": [],
        }
        
        playbook = build_lean_mode_playbook(capability, checklist)
        
        assert playbook["recommended_next_mode"] == "SHADOW"
        assert "INTERMEDIATE" in str(playbook["advisory_notes"])

    def test_playbook_intermediate_band_with_fail(self) -> None:
        """Playbook should recommend SIMULATION_ONLY for INTERMEDIATE with failing checklist."""
        from backend.verification import build_lean_mode_playbook
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "INTERMEDIATE",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        checklist = {
            "can_enable_live_mode": False,
            "blocking_reasons": ["Safety status is WARN, must be OK"],
            "checklist_items": [],
        }
        
        playbook = build_lean_mode_playbook(capability, checklist)
        
        assert playbook["recommended_next_mode"] == "SIMULATION_ONLY"
        assert len(playbook["prerequisites"]) > 0

    def test_playbook_advanced_band_with_pass(self) -> None:
        """Playbook should recommend SHADOW for ADVANCED with passing checklist."""
        from backend.verification import build_lean_mode_playbook
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "ADVANCED",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        checklist = {
            "can_enable_live_mode": True,
            "blocking_reasons": [],
            "checklist_items": [],
        }
        
        playbook = build_lean_mode_playbook(capability, checklist)
        
        assert playbook["recommended_next_mode"] == "SHADOW"
        assert "ADVANCED" in str(playbook["advisory_notes"])
        assert "MIXED" in str(playbook["advisory_notes"])

    def test_playbook_advanced_band_with_fail(self) -> None:
        """Playbook should recommend SIMULATION_ONLY for ADVANCED with failing checklist."""
        from backend.verification import build_lean_mode_playbook
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "ADVANCED",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        checklist = {
            "can_enable_live_mode": False,
            "blocking_reasons": ["Internal error count: 5"],
            "checklist_items": [],
        }
        
        playbook = build_lean_mode_playbook(capability, checklist)
        
        assert playbook["recommended_next_mode"] == "SIMULATION_ONLY"
        assert len(playbook["prerequisites"]) > 0

    def test_playbook_includes_advisory_never_auto_enable(self) -> None:
        """Playbook should include advisory note about never auto-enabling."""
        from backend.verification import build_lean_mode_playbook
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "INTERMEDIATE",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        checklist = {
            "can_enable_live_mode": True,
            "blocking_reasons": [],
            "checklist_items": [],
        }
        
        playbook = build_lean_mode_playbook(capability, checklist)
        
        assert any("auto-enable" in note.lower() for note in playbook["advisory_notes"])

    def test_playbook_schema_version(self) -> None:
        """Playbook should have schema_version."""
        from backend.verification import build_lean_mode_playbook
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "BASIC",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        checklist = {
            "can_enable_live_mode": False,
            "blocking_reasons": [],
            "checklist_items": [],
        }
        
        playbook = build_lean_mode_playbook(capability, checklist)
        
        assert playbook["schema_version"] == "1.0.0"

    def test_playbook_prerequisites_for_shadow_mode(self) -> None:
        """Playbook should include prerequisites for SHADOW mode."""
        from backend.verification import build_lean_mode_playbook
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "ADVANCED",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        checklist = {
            "can_enable_live_mode": True,
            "blocking_reasons": [],
            "checklist_items": [],
        }
        
        playbook = build_lean_mode_playbook(capability, checklist)
        
        assert playbook["recommended_next_mode"] == "SHADOW"
        assert len(playbook["prerequisites"]) > 0
        assert any("version" in prereq.lower() for prereq in playbook["prerequisites"])

    def test_playbook_json_serializable(self) -> None:
        """Playbook should be JSON serializable."""
        import json
        from backend.verification import build_lean_mode_playbook
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "INTERMEDIATE",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        checklist = {
            "can_enable_live_mode": True,
            "blocking_reasons": [],
            "checklist_items": [],
        }
        
        playbook = build_lean_mode_playbook(capability, checklist)
        
        # Should not raise
        json_str = json.dumps(playbook)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["recommended_next_mode"] == playbook["recommended_next_mode"]


# =============================================================================
# NEXT MISSION: EVIDENCE PACK ADAPTER TESTS
# =============================================================================

@pytest.mark.unit
class TestSummarizeLeanCapabilitiesForEvidence:
    """Tests for summarize_lean_capabilities_for_evidence (Next Mission)."""

    def test_evidence_tile_basic_band(self) -> None:
        """Evidence tile should include BASIC capability band."""
        from backend.verification import summarize_lean_capabilities_for_evidence
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "BASIC",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        safety_eval = {
            "has_internal_errors": False,
            "has_resource_issues": True,
            "safety_status": "WARN",
            "reasons": [],
            "internal_error_count": 0,
            "resource_limit_count": 5,
        }
        
        tile = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        
        assert tile["capability_band"] == "BASIC"
        assert tile["success_rate_proxy"] == 0.3
        assert tile["simulation_only"] is True

    def test_evidence_tile_intermediate_band(self) -> None:
        """Evidence tile should include INTERMEDIATE capability band."""
        from backend.verification import summarize_lean_capabilities_for_evidence
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "INTERMEDIATE",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        safety_eval = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": [],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        tile = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        
        assert tile["capability_band"] == "INTERMEDIATE"
        assert tile["success_rate_proxy"] == 0.65
        assert tile["safety_status"] == "OK"

    def test_evidence_tile_advanced_band(self) -> None:
        """Evidence tile should include ADVANCED capability band."""
        from backend.verification import summarize_lean_capabilities_for_evidence
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "ADVANCED",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": False,
        }
        
        safety_eval = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": [],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        tile = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        
        assert tile["capability_band"] == "ADVANCED"
        assert tile["success_rate_proxy"] == 0.85
        assert tile["simulation_only"] is False

    def test_evidence_tile_includes_resource_limit_count(self) -> None:
        """Evidence tile should include resource_limit_count."""
        from backend.verification import summarize_lean_capabilities_for_evidence
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "INTERMEDIATE",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        safety_eval = {
            "has_internal_errors": False,
            "has_resource_issues": True,
            "safety_status": "WARN",
            "reasons": [],
            "internal_error_count": 0,
            "resource_limit_count": 10,
        }
        
        tile = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        
        assert tile["resource_limit_count"] == 10

    def test_evidence_tile_includes_safety_status(self) -> None:
        """Evidence tile should include safety_status."""
        from backend.verification import summarize_lean_capabilities_for_evidence
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "BASIC",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        safety_eval = {
            "has_internal_errors": True,
            "has_resource_issues": False,
            "safety_status": "BLOCK",
            "reasons": [],
            "internal_error_count": 5,
            "resource_limit_count": 0,
        }
        
        tile = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        
        assert tile["safety_status"] == "BLOCK"

    def test_evidence_tile_deterministic(self) -> None:
        """Evidence tile should be deterministic for same inputs."""
        from backend.verification import summarize_lean_capabilities_for_evidence
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "INTERMEDIATE",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        safety_eval = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": [],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        tile1 = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        tile2 = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        
        assert tile1 == tile2

    def test_evidence_tile_no_normative_language(self) -> None:
        """Evidence tile should not contain normative language (should, must, etc.)."""
        from backend.verification import summarize_lean_capabilities_for_evidence
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "ADVANCED",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        safety_eval = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": [],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        tile = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        
        # Convert all values to strings and check for normative language
        tile_str = str(tile).lower()
        normative_words = ["should", "must", "need", "require", "ought"]
        for word in normative_words:
            assert word not in tile_str, f"Found normative language: {word}"

    def test_evidence_tile_json_serializable(self) -> None:
        """Evidence tile should be JSON serializable."""
        import json
        from backend.verification import summarize_lean_capabilities_for_evidence
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "INTERMEDIATE",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        safety_eval = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": [],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        tile = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        
        # Should not raise
        json_str = json.dumps(tile)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed == tile

    def test_evidence_tile_compact_shape(self) -> None:
        """Evidence tile should have compact shape with only required fields."""
        from backend.verification import summarize_lean_capabilities_for_evidence
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "BASIC",
            "max_budget_used": {},
            "resource_profile": "Test profile",
            "simulation_only": True,
        }
        
        safety_eval = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": [],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        tile = summarize_lean_capabilities_for_evidence(capability, safety_eval)
        
        # Should have exactly 5 fields
        assert len(tile) == 5
        assert "capability_band" in tile
        assert "success_rate_proxy" in tile
        assert "resource_limit_count" in tile
        assert "simulation_only" in tile
        assert "safety_status" in tile
