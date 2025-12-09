"""
TDA Telemetry Schema Extension for U2 Trace Events

Operation CORTEX: Phase I & II Activation
==========================================

This module extends the U2 trace schema (experiments/u2/schema.py) with
TDA Mind Scanner telemetry events for Phase I Shadow Mode and Phase II
Soft Gating integration.

Phase I (Shadow Mode): Observation only, no gating
Phase II (Soft Gating): HSS modulates learning rate and planner reweighting

Usage:
    from experiments.u2.tda_schema_extension import TDAEvaluationEvent
    from experiments.u2.tda_schema_extension import TDASoftGateEvent  # Phase II

Schema Version: tda-u2-trace-2.0.0
Parent Schema: u2-trace-1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal, List


TDA_SCHEMA_VERSION = "tda-u2-trace-2.0.0"


@dataclass(frozen=True)
class TDAEvaluationEvent:
    """
    TDA Mind Scanner evaluation result per cycle.

    This event is emitted after each cycle when TDAMonitor is active.
    In Phase I (Shadow Mode), this is purely telemetry - no gating.

    Attributes:
        cycle: Cycle index (0-based).
        slice_name: Curriculum slice name.
        mode: Execution mode ("baseline" or "rfl").

        hss: Hallucination Stability Score [0, 1].
        sns: Structural Non-Triviality Score [0, 1].
        pcs: Persistence Coherence Score [0, 1].
        drs: Deviation-from-Reference Score [0, 1].

        signal: Gating signal ("OK", "WARN", "BLOCK").
        block: Whether block condition was met.
        warn: Whether warn condition was met.

        betti_0: Betti number β₀ (connected components).
        betti_1: Betti number β₁ (loops/cycles).

        computation_ms: Time to compute TDA result in milliseconds.
        error: Error message if evaluation failed, None otherwise.
    """
    cycle: int
    slice_name: str
    mode: Literal["baseline", "rfl"]

    # Core scores
    hss: float
    sns: float
    pcs: float
    drs: float

    # Gating signals
    signal: Literal["OK", "WARN", "BLOCK"]
    block: bool
    warn: bool

    # Topological features
    betti_0: int
    betti_1: int

    # Performance
    computation_ms: float

    # Error handling
    error: Optional[str] = None


@dataclass(frozen=True)
class TDASessionSummaryEvent:
    """
    TDA Mind Scanner session summary.

    Emitted once at the end of an experiment session with aggregate
    TDA statistics for Phase I analysis.

    Attributes:
        run_id: Experiment run identifier.
        slice_name: Curriculum slice name.
        mode: Execution mode.
        schema_version: TDA schema version.

        total_evaluations: Total TDA evaluations performed.
        successful_evaluations: Evaluations without errors.
        error_count: Number of evaluation errors.

        hss_mean: Mean HSS across all evaluations.
        hss_std: Standard deviation of HSS.
        hss_min: Minimum HSS observed.
        hss_max: Maximum HSS observed.

        block_count: Number of BLOCK signals (Phase I: not enforced).
        warn_count: Number of WARN signals.
        ok_count: Number of OK signals.

        mean_computation_ms: Mean computation time per evaluation.
        p95_computation_ms: 95th percentile computation time.
    """
    run_id: str
    slice_name: str
    mode: Literal["baseline", "rfl"]
    schema_version: str

    total_evaluations: int
    successful_evaluations: int
    error_count: int

    hss_mean: float
    hss_std: float
    hss_min: float
    hss_max: float

    block_count: int
    warn_count: int
    ok_count: int

    mean_computation_ms: float
    p95_computation_ms: float


@dataclass(frozen=True)
class TDAThresholdCalibrationEvent:
    """
    TDA threshold calibration snapshot.

    Emitted when reference profiles are loaded or thresholds are updated.
    Used for Phase I analysis and Phase II threshold tuning.
    """
    slice_name: str
    timestamp: str

    # Threshold configuration
    hss_block_threshold: float
    hss_warn_threshold: float

    # Reference profile parameters
    n_ref: int
    lifetime_threshold: float
    deviation_max: float

    # Score weights
    alpha: float  # SNS weight
    beta: float   # PCS weight
    gamma: float  # DRS weight

    # Calibration source
    profile_source: str  # "file", "computed", "default"


# ============================================================================
# Phase II: Soft Gating Events
# ============================================================================

@dataclass(frozen=True)
class TDASoftGateEvent:
    """
    Phase II Soft Gating telemetry event.

    Emitted when HSS-based modulation is applied to learning rate or
    planner reweighting. Captures the full soft gating decision chain.

    Attributes:
        cycle: Cycle index (0-based).
        slice_name: Curriculum slice name.
        mode: Execution mode ("baseline" or "rfl").

        # Core TDA scores
        hss: Hallucination Stability Score [0, 1].
        sns: Structural Non-Triviality Score [0, 1].
        pcs: Persistence Coherence Score [0, 1].
        drs: Deviation-from-Reference Score [0, 1].

        # HSS classification
        hss_class: Classification ("OK", "WARN", "SOFT_BLOCK", "NO_TDA").

        # Learning rate modulation (RFLRunner)
        eta_base: Base learning rate before modulation.
        eta_eff: Effective learning rate after HSS modulation.
        learning_allowed: Whether learning was permitted.

        # Planner reweighting (U2Runner)
        reweighting_applied: Whether planner reweighting was applied.
        score_delta_mean: Mean score change from reweighting.
        candidates_reweighted: Number of candidates reweighted.

        # Modulation configuration
        theta_warn: HSS warning threshold used.
        theta_block: HSS soft-block threshold used.
        lambda_soft: Modulation factor used in WARN zone.

        # Performance
        computation_ms: Time to compute TDA + modulation in ms.
    """
    cycle: int
    slice_name: str
    mode: Literal["baseline", "rfl"]

    # Core TDA scores
    hss: float
    sns: float
    pcs: float
    drs: float

    # HSS classification
    hss_class: Literal["OK", "WARN", "SOFT_BLOCK", "NO_TDA"]

    # Learning rate modulation
    eta_base: float
    eta_eff: float
    learning_allowed: bool

    # Planner reweighting
    reweighting_applied: bool
    score_delta_mean: float
    candidates_reweighted: int

    # Modulation configuration
    theta_warn: float
    theta_block: float
    lambda_soft: float

    # Performance
    computation_ms: float


@dataclass(frozen=True)
class TDASoftGateSummaryEvent:
    """
    Phase II Soft Gating session summary.

    Aggregate statistics for HSS modulation across an experiment session.
    """
    run_id: str
    slice_name: str
    mode: Literal["baseline", "rfl"]
    schema_version: str

    # Cycle counts
    total_cycles: int
    cycles_with_tda: int

    # HSS class distribution
    ok_count: int
    warn_count: int
    soft_block_count: int
    no_tda_count: int

    # Learning rate statistics
    learning_skipped_count: int
    eta_eff_mean: float
    eta_eff_std: float
    eta_modulation_ratio: float  # mean(eta_eff) / eta_base

    # Planner reweighting statistics
    reweighting_applied_count: int
    mean_score_delta: float
    total_candidates_reweighted: int

    # HSS distribution
    hss_mean: float
    hss_std: float
    hss_min: float
    hss_max: float
    hss_p25: float
    hss_p50: float
    hss_p75: float

    # Phase II health metrics
    modulation_effectiveness: float  # Correlation between HSS and success
    planner_divergence: float  # KL divergence from baseline ordering


@dataclass(frozen=True)
class TDAModulationConfigEvent:
    """
    Phase II modulation configuration snapshot.

    Emitted at session start to record soft gating configuration.
    """
    slice_name: str
    timestamp: str
    schema_version: str

    # Learning rate modulation config
    modulation_enabled: bool
    theta_warn: float
    theta_block: float
    lambda_soft: float
    skip_on_block: bool

    # Planner reweighting config
    planner_enabled: bool
    base_weight: float  # α
    hss_weight: float   # β
    min_score: float
    max_score: float

    # Phase identifier
    cortex_phase: Literal["I", "II", "III"]


# ============================================================================
# Helper Functions
# ============================================================================

def tda_event_to_dict(event: TDAEvaluationEvent) -> Dict[str, Any]:
    """Convert TDA evaluation event to dictionary for JSONL serialization."""
    return {
        "event_type": "tda_evaluation",
        "schema_version": TDA_SCHEMA_VERSION,
        "cycle": event.cycle,
        "slice_name": event.slice_name,
        "mode": event.mode,
        "hss": event.hss,
        "sns": event.sns,
        "pcs": event.pcs,
        "drs": event.drs,
        "signal": event.signal,
        "block": event.block,
        "warn": event.warn,
        "betti_0": event.betti_0,
        "betti_1": event.betti_1,
        "computation_ms": event.computation_ms,
        "error": event.error,
    }


def tda_summary_to_dict(event: TDASessionSummaryEvent) -> Dict[str, Any]:
    """Convert TDA session summary to dictionary for JSONL serialization."""
    return {
        "event_type": "tda_session_summary",
        "schema_version": TDA_SCHEMA_VERSION,
        "run_id": event.run_id,
        "slice_name": event.slice_name,
        "mode": event.mode,
        "total_evaluations": event.total_evaluations,
        "successful_evaluations": event.successful_evaluations,
        "error_count": event.error_count,
        "hss_stats": {
            "mean": event.hss_mean,
            "std": event.hss_std,
            "min": event.hss_min,
            "max": event.hss_max,
        },
        "signal_counts": {
            "block": event.block_count,
            "warn": event.warn_count,
            "ok": event.ok_count,
        },
        "performance": {
            "mean_computation_ms": event.mean_computation_ms,
            "p95_computation_ms": event.p95_computation_ms,
        },
    }


def create_tda_summary(
    run_id: str,
    slice_name: str,
    mode: str,
    evaluations: list,
) -> TDASessionSummaryEvent:
    """
    Create a TDA session summary from a list of evaluation events.

    Args:
        run_id: Experiment run identifier.
        slice_name: Curriculum slice name.
        mode: Execution mode ("baseline" or "rfl").
        evaluations: List of TDAEvaluationEvent objects or dicts.

    Returns:
        TDASessionSummaryEvent with aggregate statistics.
    """
    import numpy as np

    if not evaluations:
        return TDASessionSummaryEvent(
            run_id=run_id,
            slice_name=slice_name,
            mode=mode,
            schema_version=TDA_SCHEMA_VERSION,
            total_evaluations=0,
            successful_evaluations=0,
            error_count=0,
            hss_mean=0.0,
            hss_std=0.0,
            hss_min=0.0,
            hss_max=0.0,
            block_count=0,
            warn_count=0,
            ok_count=0,
            mean_computation_ms=0.0,
            p95_computation_ms=0.0,
        )

    # Extract values
    hss_values = []
    computation_times = []
    signals = {"BLOCK": 0, "WARN": 0, "OK": 0}
    errors = 0

    for e in evaluations:
        if isinstance(e, dict):
            hss = e.get("hss")
            signal = e.get("signal", "OK")
            comp_ms = e.get("computation_ms", 0.0)
            error = e.get("error")
        else:
            hss = e.hss
            signal = e.signal
            comp_ms = e.computation_ms
            error = e.error

        if error:
            errors += 1
        else:
            hss_values.append(hss)
            computation_times.append(comp_ms)
            signals[signal] = signals.get(signal, 0) + 1

    hss_arr = np.array(hss_values) if hss_values else np.array([0.0])
    comp_arr = np.array(computation_times) if computation_times else np.array([0.0])

    return TDASessionSummaryEvent(
        run_id=run_id,
        slice_name=slice_name,
        mode=mode,
        schema_version=TDA_SCHEMA_VERSION,
        total_evaluations=len(evaluations),
        successful_evaluations=len(hss_values),
        error_count=errors,
        hss_mean=float(np.mean(hss_arr)),
        hss_std=float(np.std(hss_arr)),
        hss_min=float(np.min(hss_arr)),
        hss_max=float(np.max(hss_arr)),
        block_count=signals["BLOCK"],
        warn_count=signals["WARN"],
        ok_count=signals["OK"],
        mean_computation_ms=float(np.mean(comp_arr)),
        p95_computation_ms=float(np.percentile(comp_arr, 95)),
    )


# ============================================================================
# Phase II Helper Functions
# ============================================================================

def softgate_event_to_dict(event: TDASoftGateEvent) -> Dict[str, Any]:
    """Convert Phase II soft gate event to dictionary for JSONL serialization."""
    return {
        "event_type": "tda_softgate",
        "schema_version": TDA_SCHEMA_VERSION,
        "cycle": event.cycle,
        "slice_name": event.slice_name,
        "mode": event.mode,
        # Core TDA scores
        "hss": event.hss,
        "sns": event.sns,
        "pcs": event.pcs,
        "drs": event.drs,
        # HSS classification
        "hss_class": event.hss_class,
        # Learning rate modulation
        "eta_base": event.eta_base,
        "eta_eff": event.eta_eff,
        "learning_allowed": event.learning_allowed,
        # Planner reweighting
        "reweighting_applied": event.reweighting_applied,
        "score_delta_mean": event.score_delta_mean,
        "candidates_reweighted": event.candidates_reweighted,
        # Modulation config
        "theta_warn": event.theta_warn,
        "theta_block": event.theta_block,
        "lambda_soft": event.lambda_soft,
        # Performance
        "computation_ms": event.computation_ms,
    }


def softgate_summary_to_dict(event: TDASoftGateSummaryEvent) -> Dict[str, Any]:
    """Convert Phase II soft gate summary to dictionary for JSONL serialization."""
    return {
        "event_type": "tda_softgate_summary",
        "schema_version": TDA_SCHEMA_VERSION,
        "run_id": event.run_id,
        "slice_name": event.slice_name,
        "mode": event.mode,
        "cycle_counts": {
            "total": event.total_cycles,
            "with_tda": event.cycles_with_tda,
        },
        "hss_class_distribution": {
            "OK": event.ok_count,
            "WARN": event.warn_count,
            "SOFT_BLOCK": event.soft_block_count,
            "NO_TDA": event.no_tda_count,
        },
        "learning_rate_stats": {
            "skipped_count": event.learning_skipped_count,
            "eta_eff_mean": event.eta_eff_mean,
            "eta_eff_std": event.eta_eff_std,
            "modulation_ratio": event.eta_modulation_ratio,
        },
        "planner_reweighting": {
            "applied_count": event.reweighting_applied_count,
            "mean_score_delta": event.mean_score_delta,
            "total_candidates": event.total_candidates_reweighted,
        },
        "hss_distribution": {
            "mean": event.hss_mean,
            "std": event.hss_std,
            "min": event.hss_min,
            "max": event.hss_max,
            "p25": event.hss_p25,
            "p50": event.hss_p50,
            "p75": event.hss_p75,
        },
        "health_metrics": {
            "modulation_effectiveness": event.modulation_effectiveness,
            "planner_divergence": event.planner_divergence,
        },
    }


def modulation_config_to_dict(event: TDAModulationConfigEvent) -> Dict[str, Any]:
    """Convert modulation config event to dictionary for JSONL serialization."""
    return {
        "event_type": "tda_modulation_config",
        "schema_version": TDA_SCHEMA_VERSION,
        "slice_name": event.slice_name,
        "timestamp": event.timestamp,
        "cortex_phase": event.cortex_phase,
        "learning_rate_modulation": {
            "enabled": event.modulation_enabled,
            "theta_warn": event.theta_warn,
            "theta_block": event.theta_block,
            "lambda_soft": event.lambda_soft,
            "skip_on_block": event.skip_on_block,
        },
        "planner_reweighting": {
            "enabled": event.planner_enabled,
            "base_weight": event.base_weight,
            "hss_weight": event.hss_weight,
            "min_score": event.min_score,
            "max_score": event.max_score,
        },
    }


def create_softgate_summary(
    run_id: str,
    slice_name: str,
    mode: str,
    softgate_events: List[TDASoftGateEvent],
    eta_base: float = 0.1,
) -> TDASoftGateSummaryEvent:
    """
    Create a Phase II soft gate summary from a list of soft gate events.

    Args:
        run_id: Experiment run identifier.
        slice_name: Curriculum slice name.
        mode: Execution mode ("baseline" or "rfl").
        softgate_events: List of TDASoftGateEvent objects.
        eta_base: Base learning rate for modulation ratio calculation.

    Returns:
        TDASoftGateSummaryEvent with aggregate statistics.
    """
    import numpy as np

    if not softgate_events:
        return TDASoftGateSummaryEvent(
            run_id=run_id,
            slice_name=slice_name,
            mode=mode,
            schema_version=TDA_SCHEMA_VERSION,
            total_cycles=0,
            cycles_with_tda=0,
            ok_count=0,
            warn_count=0,
            soft_block_count=0,
            no_tda_count=0,
            learning_skipped_count=0,
            eta_eff_mean=0.0,
            eta_eff_std=0.0,
            eta_modulation_ratio=0.0,
            reweighting_applied_count=0,
            mean_score_delta=0.0,
            total_candidates_reweighted=0,
            hss_mean=0.0,
            hss_std=0.0,
            hss_min=0.0,
            hss_max=0.0,
            hss_p25=0.0,
            hss_p50=0.0,
            hss_p75=0.0,
            modulation_effectiveness=0.0,
            planner_divergence=0.0,
        )

    # Aggregate statistics
    hss_class_counts = {"OK": 0, "WARN": 0, "SOFT_BLOCK": 0, "NO_TDA": 0}
    hss_values = []
    eta_eff_values = []
    score_deltas = []
    learning_skipped = 0
    reweighting_applied = 0
    total_candidates = 0

    for e in softgate_events:
        hss_class_counts[e.hss_class] = hss_class_counts.get(e.hss_class, 0) + 1
        if e.hss_class != "NO_TDA":
            hss_values.append(e.hss)
        eta_eff_values.append(e.eta_eff)
        if not e.learning_allowed:
            learning_skipped += 1
        if e.reweighting_applied:
            reweighting_applied += 1
            score_deltas.append(e.score_delta_mean)
            total_candidates += e.candidates_reweighted

    hss_arr = np.array(hss_values) if hss_values else np.array([0.0])
    eta_arr = np.array(eta_eff_values) if eta_eff_values else np.array([0.0])

    return TDASoftGateSummaryEvent(
        run_id=run_id,
        slice_name=slice_name,
        mode=mode,
        schema_version=TDA_SCHEMA_VERSION,
        total_cycles=len(softgate_events),
        cycles_with_tda=len(hss_values),
        ok_count=hss_class_counts["OK"],
        warn_count=hss_class_counts["WARN"],
        soft_block_count=hss_class_counts["SOFT_BLOCK"],
        no_tda_count=hss_class_counts["NO_TDA"],
        learning_skipped_count=learning_skipped,
        eta_eff_mean=float(np.mean(eta_arr)),
        eta_eff_std=float(np.std(eta_arr)),
        eta_modulation_ratio=float(np.mean(eta_arr)) / eta_base if eta_base > 0 else 0.0,
        reweighting_applied_count=reweighting_applied,
        mean_score_delta=float(np.mean(score_deltas)) if score_deltas else 0.0,
        total_candidates_reweighted=total_candidates,
        hss_mean=float(np.mean(hss_arr)),
        hss_std=float(np.std(hss_arr)),
        hss_min=float(np.min(hss_arr)),
        hss_max=float(np.max(hss_arr)),
        hss_p25=float(np.percentile(hss_arr, 25)),
        hss_p50=float(np.percentile(hss_arr, 50)),
        hss_p75=float(np.percentile(hss_arr, 75)),
        modulation_effectiveness=0.0,  # Computed externally with success data
        planner_divergence=0.0,  # Computed externally with baseline data
    )
