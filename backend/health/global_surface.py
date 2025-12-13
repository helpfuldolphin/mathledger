"""Global health surface builder with dynamics integration.

Provides tile attachment for global health surface including:
- Dynamics tile (u2_dynamics)
- USLA tile (Phase X P2, SHADOW mode only)
- Semantic-TDA tile (Phase V, SHADOW mode only)
- Lean shadow tile (Phase 1, SHADOW mode only)
- Chronicle governance tile (Phase X, SHADOW mode only)
- Convergence pressure tile (Phase X, SHADOW mode only)
- Replay governance tile (Phase X, SHADOW mode only) [TODO: PHASE-X-REPLAY]

# TODO(PHASE-X-REPLAY): Implement replay_governance tile attachment
# Integration: Add replay_governance_envelope, replay_promotion_eval, replay_governance_view
#              parameters to build_global_health_surface() and attach via
#              attach_replay_governance_tile() following the same pattern as other tiles.
# Schema: docs/system_law/schemas/replay/replay_global_console_tile.schema.json
# Adapter: backend/health/replay_governance_adapter.py (to be created)
# Binding: docs/system_law/Replay_Governance_PhaseX_Binding.md

SHADOW MODE CONTRACT (USLA tile):
- The USLA tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- Tile is only attached when SHADOW mode is enabled

SHADOW MODE CONTRACT (semantic_tda tile):
- The semantic_tda tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- Tile is only attached when both semantic_panel and tda_panel are available

SHADOW MODE CONTRACT (lean_shadow tile):
- The lean_shadow tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- Compatible with both P3 (synthetic Lean evaluation) and P4 (real-runner + Lean shadow observations)
- No abort logic, no control feedback, no modification of pipeline/runner decisions
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from analysis.u2_dynamics import (
    DynamicsDebugSnapshot,
    summarize_dynamics_for_global_console,
)
from backend.health.u2_dynamics_tile import (
    attach_u2_dynamics_tile,
    build_u2_dynamics_tile,
)

GLOBAL_HEALTH_SURFACE_SCHEMA_VERSION = "global-health-surface/1.1.0"

# Module-level USLA producer (set via set_usla_producer)
_usla_producer: Optional[Any] = None
_usla_integration: Optional[Any] = None


def set_usla_producer(
    producer: Any,
    integration: Optional[Any] = None,
) -> None:
    """
    Install a USLAHealthTileProducer for SHADOW mode observability.

    SHADOW MODE CONTRACT:
    - The producer is read-only and side-effect free
    - The tile it produces does NOT influence any governance decisions
    - This is purely for observability and logging

    Args:
        producer: USLAHealthTileProducer instance
        integration: Optional USLAIntegration instance for produce_from_integration
    """
    global _usla_producer, _usla_integration
    _usla_producer = producer
    _usla_integration = integration


def clear_usla_producer() -> None:
    """Clear the USLA producer (for testing)."""
    global _usla_producer, _usla_integration
    _usla_producer = None
    _usla_integration = None


def _is_usla_shadow_enabled() -> bool:
    """Check if USLA shadow mode is enabled via environment."""
    return os.getenv("USLA_SHADOW_ENABLED", "").lower() in ("1", "true", "yes")


def _build_usla_tile() -> Optional[Dict[str, Any]]:
    """
    Build USLA health tile if producer is configured and SHADOW mode enabled.

    SHADOW MODE CONTRACT:
    - This function is read-only
    - The returned tile is purely observational
    - No control flow depends on the tile contents

    Returns:
        USLA health tile dict, or None if not available
    """
    global _usla_producer, _usla_integration

    if _usla_producer is None:
        return None

    if not _is_usla_shadow_enabled():
        return None

    try:
        # Prefer produce_from_integration if integration is available
        if _usla_integration is not None:
            if hasattr(_usla_producer, 'produce_from_integration'):
                tile = _usla_producer.produce_from_integration(_usla_integration)
                if tile is not None:
                    return tile

        # Fallback: produce with default state if producer has produce method
        if hasattr(_usla_producer, 'produce'):
            # Import here to avoid circular dependency
            from backend.topology.usla_simulator import USLAState
            default_state = USLAState.initial()
            return _usla_producer.produce(state=default_state, hard_ok=True)

    except Exception:
        # SHADOW MODE: Never fail the build due to USLA tile issues
        # Silently return None and continue
        return None

    return None


def build_global_health_surface(
    base_payload: Mapping[str, Any] | None = None,
    dynamics_snapshots: Sequence[DynamicsDebugSnapshot] | None = None,
    u2_dynamics_summary: Optional[Dict[str, Any]] = None,
    semantic_panel: Optional[Dict[str, Any]] = None,
    tda_panel: Optional[Dict[str, Any]] = None,
    semantic_timeline: Optional[Dict[str, Any]] = None,
    shadow_radar: Optional[Dict[str, Any]] = None,
    topology_pressure_field: Optional[Dict[str, Any]] = None,
    topology_promotion_gate: Optional[Dict[str, Any]] = None,
    topology_console_tile: Optional[Dict[str, Any]] = None,
    coherence_map: Optional[Dict[str, Any]] = None,
    drift_horizon: Optional[Dict[str, Any]] = None,
    coherence_console_tile: Optional[Dict[str, Any]] = None,
    epistemic_profile: Optional[Dict[str, Any]] = None,
    epistemic_storyline: Optional[Dict[str, Any]] = None,
    epistemic_drift_timeline: Optional[Dict[str, Any]] = None,
    semantic_integrity_data: Optional[Dict[str, Any]] = None,
    telemetry_governance_data: Optional[Dict[str, Any]] = None,
    drift_tensor: Optional[Dict[str, Any]] = None,
    poly_cause_view: Optional[Dict[str, Any]] = None,
    director_tile_v2: Optional[Dict[str, Any]] = None,
    readiness_tensor: Optional[Dict[str, Any]] = None,
    readiness_polygraph: Optional[Dict[str, Any]] = None,
    readiness_phase_transition: Optional[Dict[str, Any]] = None,
    atlas_lattice: Optional[Dict[str, Any]] = None,
    atlas_phase_gate: Optional[Dict[str, Any]] = None,
    atlas_director_tile_v2: Optional[Dict[str, Any]] = None,
    adversarial_pressure_model: Optional[Dict[str, Any]] = None,
    adversarial_scenario_plan: Optional[Dict[str, Any]] = None,
    adversarial_failover_plan_v2: Optional[Dict[str, Any]] = None,
    recurrence_projection: Optional[Dict[str, Any]] = None,
    invariant_check: Optional[Dict[str, Any]] = None,
    stability_scores: Optional[Dict[str, Any]] = None,
    alignment_tensor: Optional[Dict[str, Any]] = None,
    misalignment_forecast: Optional[Dict[str, Any]] = None,
    director_panel: Optional[Dict[str, Any]] = None,
    consensus_polygraph_result: Optional[Dict[str, Any]] = None,
    consensus_predictive_result: Optional[Dict[str, Any]] = None,
    consensus_director_panel: Optional[Dict[str, Any]] = None,
    evidence_phase_portrait: Optional[Dict[str, Any]] = None,
    evidence_forecast: Optional[Dict[str, Any]] = None,
    evidence_director_panel_v2: Optional[Dict[str, Any]] = None,
    semantic_drift_tensor: Optional[Dict[str, Any]] = None,
    semantic_drift_counterfactual: Optional[Dict[str, Any]] = None,
    semantic_drift_director_panel: Optional[Dict[str, Any]] = None,
    uplift_safety_tensor: Optional[Dict[str, Any]] = None,
    uplift_stability_forecaster: Optional[Dict[str, Any]] = None,
    uplift_gate_decision: Optional[Dict[str, Any]] = None,
    prng_governance_history: Optional[Dict[str, Any]] = None,
    runtime_profile_health: Optional[Dict[str, Any]] = None,
    realism_director_tile: Optional[Dict[str, Any]] = None,
    snapshot_root: Optional[str] = None,
    # Phase X: Replay governance tile inputs
    replay_radar: Optional[Dict[str, Any]] = None,
    replay_promotion_eval: Optional[Dict[str, Any]] = None,
    replay_director_panel: Optional[Dict[str, Any]] = None,
    # Phase X: Metrics governance tile inputs (CLAUDE D)
    metrics_drift_compass: Optional[Dict[str, Any]] = None,
    metrics_budget_view: Optional[Dict[str, Any]] = None,
    metrics_governance_signal: Optional[Dict[str, Any]] = None,
    ledger_guard_result: Optional[Dict[str, Any]] = None,
    policy_drift_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a global health payload and attach tiles.

    Attaches:
    - dynamics: U2 dynamics tile
    - u2_dynamics: U2 dynamics observational tile (SHADOW mode only, no control effect)
    - usla: USLA health tile (SHADOW mode only, purely observational)
    - semantic_tda: Semantic-TDA governance tile (SHADOW mode only, purely observational)
    - lean_shadow: Lean shadow capability tile (SHADOW mode only, purely observational)
    - chronicle_governance: Chronicle governance tile (Phase X, SHADOW mode only, purely observational)
    - topology_pressure: Topology pressure governance tile (SHADOW mode only, purely observational)
    - semantic_integrity: Semantic integrity grid tile (SHADOW mode only, purely observational)
    - coherence_health: Coherence governance tile (Phase X, purely observational)
    - drift_governance: Drift governance tile (Phase X, SHADOW mode only, purely observational)
    - metric_readiness: Metric readiness tensor tile (Phase X, SHADOW mode only, purely observational)
    - atlas_governance: Atlas governance tile (Phase X, SHADOW mode only, purely observational)
    - adversarial_governance: Adversarial pressure governance tile (Phase X, SHADOW mode only, purely observational)
    - consensus_governance: Consensus polygraph governance tile (Phase X, SHADOW mode only, purely observational)
    - epistemic_alignment: Epistemic alignment governance tile (Phase X, SHADOW mode only, purely observational)
    - semantic_curriculum_harmonic: Harmonic alignment governance tile (Phase X, SHADOW mode only, purely observational)
    - semantic_drift: Semantic drift governance tile (Phase X, SHADOW mode only, purely observational)
    - uplift_safety: Uplift safety governance tile (Phase X, SHADOW mode only, purely observational)
    - evidence_quality: Evidence quality governance tile (Phase X, SHADOW mode only, purely observational)
    - prng: PRNG governance tile (Phase X, SHADOW mode only, purely observational)
    - taxonomy: Taxonomy integrity governance tile (Phase V, SHADOW mode only, purely observational)
    - u2_snapshot: U2 snapshot continuity tile (Phase II, advisory only, purely observational)
    - runtime_profile: Runtime profile health tile (SHADOW mode only, purely observational)
    - ledger_guard: Ledger monotonicity advisory tile (SHADOW mode only, purely observational)
    - mock_oracle: Mock oracle drift tile (Phase X, SHADOW mode only, purely observational, negative control)

    SHADOW MODE CONTRACT (usla tile):
    - The USLA tile does NOT influence any other tiles
    - No control flow depends on the USLA tile
    - The USLA tile is purely for observability

    SHADOW MODE CONTRACT (semantic_tda tile):
    - The semantic_tda tile does NOT influence any other tiles
    - No control flow depends on the semantic_tda tile
    - The semantic_tda tile is purely for observability

    SHADOW MODE CONTRACT (lean_shadow tile):
    - The lean_shadow tile does NOT influence any other tiles
    - No control flow depends on the lean_shadow tile
    - The lean_shadow tile is purely for observability
    - Compatible with both P3 (synthetic Lean evaluation) and P4 (real-runner + Lean shadow observations)
    - No abort logic, no control feedback, no modification of pipeline/runner decisions

    SHADOW MODE CONTRACT (epistemic_abstention tile):
    - The epistemic_abstention tile does NOT influence any other tiles
    - No control flow depends on the epistemic_abstention tile
    - The epistemic_abstention tile is purely for observability

    SHADOW MODE CONTRACT (convergence_pressure tile):
    - The convergence_pressure tile does NOT influence any other tiles
    - No control flow depends on the convergence_pressure tile
    - The convergence_pressure tile is purely for observability

    SHADOW MODE CONTRACT (adversarial_governance tile):
    - The adversarial_governance tile does NOT influence any other tiles
    - No control flow depends on the adversarial_governance tile
    - The adversarial_governance tile is purely for observability
    - No modification of adversarial test state or metric promotion decisions

    SHADOW MODE CONTRACT (metric_readiness tile):
    - The metric_readiness tile does NOT influence any other tiles
    - No control flow depends on the metric_readiness tile
    - The metric_readiness tile is purely for observability
    - Tile is only attached when tensor + polygraph + phase_transition_eval are available

    SHADOW MODE CONTRACT (atlas_governance tile):
    - The atlas_governance tile does NOT influence any other tiles
    - No control flow depends on the atlas_governance tile
    - The atlas_governance tile is purely for observability
    - Tile is only attached when lattice + phase_gate + director_tile_v2 are available

    SHADOW MODE CONTRACT (semantic_curriculum_harmonic tile):
    - The semantic_curriculum_harmonic tile does NOT influence any other tiles
    - No control flow depends on the semantic_curriculum_harmonic tile
    - The semantic_curriculum_harmonic tile is purely for observability
    - Tile is only attached when harmonic_map + evolution_forecaster + harmonic_director_panel are available

    SHADOW MODE CONTRACT (epistemic_alignment tile):
    - The epistemic_alignment tile does NOT influence any other tiles
    - No control flow depends on the epistemic_alignment tile
    - The epistemic_alignment tile is purely for observability
    - Tile is only attached when alignment_tensor + misalignment_forecast + director_panel are available

    SHADOW MODE CONTRACT (uplift_safety tile):
    - The uplift_safety tile does NOT influence any other tiles
    - No control flow depends on the uplift_safety tile
    - The uplift_safety tile is purely for observability
    - This tile does NOT control deployments; it summarizes risk
    - Tile is only attached when safety_tensor + stability_forecaster + gate_decision are available

    SHADOW MODE CONTRACT (semantic_drift tile):
    - The semantic_drift tile does NOT influence any other tiles
    - No control flow depends on the semantic_drift tile
    - The semantic_drift tile is purely for observability
    - Tile is only attached when semantic_drift_tensor + semantic_drift_counterfactual + semantic_drift_director_panel are available

    SHADOW MODE CONTRACT (runtime_profile tile):
    - The runtime_profile tile does NOT influence any other tiles
    - No control flow depends on the runtime_profile tile
    - The runtime_profile tile is purely for observability
    - Tile is only attached when runtime_profile_health (chaos summary or manual snapshot) is provided

    Args:
        base_payload: Existing global health fields (e.g. metrics, telemetry).
        dynamics_snapshots: Snapshots from baseline/RFL/NC runs.
        u2_dynamics_summary: Optional precomputed U2 dynamics summary (observational only).
        semantic_panel: Optional semantic director panel for semantic-TDA integration.
        tda_panel: Optional TDA health panel for semantic-TDA integration.
        semantic_timeline: Optional full semantic timeline (preferred over panel-only).
        shadow_radar: Optional shadow capability radar from build_lean_shadow_capability_radar().
        topology_pressure_field: Optional topology pressure field from build_topological_pressure_field().
        topology_promotion_gate: Optional promotion gate from topology_curriculum_promotion_gate().
        topology_console_tile: Optional console tile from build_topology_console_tile().
        coherence_map: Optional coherence map from build_confusability_topology_coherence_map.
        drift_horizon: Optional drift horizon from build_confusability_drift_horizon_predictor.
        coherence_console_tile: Optional console tile from build_global_coherence_console_tile.
        epistemic_profile: Optional epistemic abstention profile (Phase V).
        epistemic_storyline: Optional abstention storyline (Phase V).
        epistemic_drift_timeline: Optional epistemic drift timeline (Phase V).
        semantic_integrity_data: Optional dict with keys: invariant_check, uplift_preview, director_tile.
        telemetry_governance_data: Optional dict with keys: fusion_tile, uplift_gate, director_tile_v2, telemetry_health.
    drift_tensor: Optional drift tensor from build_drift_tensor (Phase X).
    poly_cause_view: Optional poly-cause view from build_drift_poly_cause_analyzer (Phase X).
    director_tile_v2: Optional director tile v2 from build_drift_director_tile_v2 (Phase X).
    readiness_tensor: Optional readiness tensor from build_metric_readiness_tensor().
    readiness_polygraph: Optional drift polygraph from build_metric_drift_polygraph().
    readiness_phase_transition: Optional phase transition eval from evaluate_phase_transition_safety_v2().
    atlas_lattice: Optional atlas convergence lattice from build_atlas_convergence_lattice().
    atlas_phase_gate: Optional phase transition gate from derive_atlas_phase_transition_gate().
    atlas_director_tile_v2: Optional director tile v2 from build_atlas_director_tile_v2().
    adversarial_pressure_model: Optional adversarial pressure model from build_adversarial_pressure_model() (Phase X).
    adversarial_scenario_plan: Optional scenario plan from build_evolving_adversarial_scenario_plan() (Phase X).
    adversarial_failover_plan_v2: Optional failover plan v2 from build_adversarial_failover_plan_v2() (Phase X).
    alignment_tensor: Optional alignment tensor from build_epistemic_alignment_tensor() (Phase X).
    misalignment_forecast: Optional misalignment forecast from forecast_epistemic_misalignment() (Phase X).
    director_panel: Optional director panel from build_epistemic_director_panel() (Phase X).
    semantic_drift_tensor: Optional semantic drift tensor from build_semantic_drift_tensor() (Phase X).
    semantic_drift_counterfactual: Optional counterfactual analysis from analyze_semantic_drift_counterfactual() (Phase X).
    semantic_drift_director_panel: Optional director panel from build_semantic_drift_director_panel_v3() (Phase X).
    uplift_safety_tensor: Optional safety tensor from build_global_uplift_safety_tensor() (Phase X).
    uplift_stability_forecaster: Optional stability forecaster from build_uplift_stability_forecaster() (Phase X).
    uplift_gate_decision: Optional gate decision from compute_maas_uplift_gate_v3() (Phase X).
    prng_governance_history: Optional PRNG governance history from build_prng_governance_history() (Phase X).
    runtime_profile_health: Optional chaos summary from experiments/u2_runtime_chaos.py or manual snapshot for testing.
    policy_drift_tile: Optional policy drift health tile (SHADOW mode observability only).
    """
    payload: Dict[str, Any] = dict(base_payload or {})
    payload["schema_version"] = GLOBAL_HEALTH_SURFACE_SCHEMA_VERSION
    payload["dynamics"] = summarize_dynamics_for_global_console(
        dynamics_snapshots or []
    )
    if u2_dynamics_summary is not None:
        try:
            u2_tile = build_u2_dynamics_tile(u2_dynamics_summary)
            payload = attach_u2_dynamics_tile(payload, u2_tile)
        except Exception:
            # SHADOW MODE: Never fail or gate on the observational U2 dynamics tile
            pass

    if policy_drift_tile is not None:
        try:
            from backend.health.policy_drift_tile import attach_policy_drift_tile as _attach_policy_drift_tile

            payload = _attach_policy_drift_tile(payload, policy_drift_tile)
        except Exception:
            # SHADOW MODE: Never fail build due to policy drift tile issues
            pass

    # Phase X P2: Attach USLA tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    usla_tile = _build_usla_tile()
    if usla_tile is not None:
        payload["usla"] = usla_tile

    # Phase V: Attach semantic-TDA tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if semantic_panel is not None and tda_panel is not None:
        try:
            from backend.health.semantic_tda_adapter import (
                build_semantic_tda_tile_for_global_health,
            )
            semantic_tda_tile = build_semantic_tda_tile_for_global_health(
                semantic_panel=semantic_panel,
                tda_panel=tda_panel,
                semantic_timeline=semantic_timeline,
            )
            payload["semantic_tda"] = semantic_tda_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to semantic-TDA tile issues
            # Silently continue without the tile
            pass

    # Phase 1: Attach Lean shadow tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if shadow_radar is not None:
        try:
            from backend.health.lean_shadow_adapter import (
                build_lean_shadow_tile_for_global_health,
            )
            lean_shadow_tile = build_lean_shadow_tile_for_global_health(shadow_radar)
            payload["lean_shadow"] = lean_shadow_tile
            
            # Phase X: Structural cohesion link
            # Add lean_shadow_status to structure tile (observational only, does not affect status aggregation)
            if "structure" not in payload:
                payload["structure"] = {}
            # Use "status" from tile (not "status_light" - tile uses "status")
            payload["structure"]["lean_shadow_status"] = lean_shadow_tile.get("status", "OK")
        except Exception:
            # SHADOW MODE: Never fail the build due to Lean shadow tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach chronicle governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if (recurrence_projection is not None and 
        invariant_check is not None and 
        stability_scores is not None):
        try:
            from backend.health.chronicle_governance_adapter import (
                build_chronicle_governance_tile,
            )
            chronicle_tile = build_chronicle_governance_tile(
                recurrence_projection=recurrence_projection,
                invariant_check=invariant_check,
                stability_scores=stability_scores,
            )
            payload["chronicle_governance"] = chronicle_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to chronicle governance tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach topology pressure tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if (
        topology_pressure_field is not None
        and topology_promotion_gate is not None
        and topology_console_tile is not None
    ):
        try:
            from backend.health.topology_pressure_adapter import (
                build_topology_pressure_governance_tile,
            )
            topology_pressure_tile = build_topology_pressure_governance_tile(
                pressure_field=topology_pressure_field,
                promotion_gate=topology_promotion_gate,
                console_tile=topology_console_tile,
            )
            payload["topology_pressure"] = topology_pressure_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to topology pressure tile issues
            # Silently continue without the tile
            pass

    # Phase V: Attach epistemic abstention tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if epistemic_profile is not None and epistemic_storyline is not None and epistemic_drift_timeline is not None:
        try:
            from rfl.verification import summarize_abstention_for_global_console
            
            epistemic_tile = summarize_abstention_for_global_console(
                profile=epistemic_profile,
                storyline=epistemic_storyline,
                drift_timeline=epistemic_drift_timeline,
            )
            payload["epistemic_abstention"] = epistemic_tile
        except ImportError:
            # Gracefully degrade if abstention module not available
            pass
        except Exception:
            # SHADOW MODE: Never fail the build due to epistemic abstention tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach drift governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if drift_tensor is not None and poly_cause_view is not None and director_tile_v2 is not None:
        try:
            from backend.health.drift_tensor_adapter import (
                build_drift_governance_tile,
            )
            drift_governance_tile = build_drift_governance_tile(
                drift_tensor=drift_tensor,
                poly_cause_view=poly_cause_view,
                director_tile_v2=director_tile_v2,
            )
            payload["drift_governance"] = drift_governance_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to drift governance tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach atlas governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if atlas_lattice is not None and atlas_phase_gate is not None and atlas_director_tile_v2 is not None:
        try:
            from backend.health.atlas_governance_adapter import (
                build_atlas_governance_tile,
            )
            atlas_governance_tile = build_atlas_governance_tile(
                lattice=atlas_lattice,
                phase_gate=atlas_phase_gate,
                director_tile_v2=atlas_director_tile_v2,
            )
            payload["atlas_governance"] = atlas_governance_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to atlas governance tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach harmonic alignment tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    # NOTE: harmonic_map, evolution_forecaster, harmonic_director_panel are pre-existing parameters
    # that are not yet in the function signature. This defensive guard prevents NameError
    # when these parameters are not provided. This is a safe guard, not a semantic change.
    try:
        # Defensive check: use locals().get() to safely check for parameters
        # This allows the code to work whether or not these parameters are in the signature
        harmonic_map_val = locals().get('harmonic_map', None)
        evolution_forecaster_val = locals().get('evolution_forecaster', None)
        harmonic_director_panel_val = locals().get('harmonic_director_panel', None)
        
        if harmonic_map_val is not None and evolution_forecaster_val is not None and harmonic_director_panel_val is not None:
            from backend.health.harmonic_alignment_adapter import (
                build_harmonic_governance_tile,
            )
            harmonic_tile = build_harmonic_governance_tile(
                harmonic_map=harmonic_map_val,
                evolution_forecaster=evolution_forecaster_val,
                harmonic_director_panel=harmonic_director_panel_val,
            )
            payload["semantic_curriculum_harmonic"] = harmonic_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to harmonic alignment tile issues
        # Silently continue without the tile
        pass

    # Phase X: Attach epistemic alignment tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if alignment_tensor is not None and misalignment_forecast is not None and director_panel is not None:
        try:
            from backend.health.epistemic_alignment_adapter import (
                build_epistemic_alignment_tile_for_global_health,
            )
            epistemic_alignment_tile = build_epistemic_alignment_tile_for_global_health(
                alignment_tensor=alignment_tensor,
                misalignment_forecast=misalignment_forecast,
                director_panel=director_panel,
            )
            payload["epistemic_alignment"] = epistemic_alignment_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to epistemic alignment tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach consensus governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if (
        consensus_polygraph_result is not None
        and consensus_director_panel is not None
    ):
        try:
            from backend.health.consensus_polygraph_adapter import (
                build_consensus_governance_tile,
            )
            consensus_governance_tile = build_consensus_governance_tile(
                polygraph_result=consensus_polygraph_result,
                predictive_result=consensus_predictive_result,
                director_panel=consensus_director_panel,
            )
            payload["consensus_governance"] = consensus_governance_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to consensus governance tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach semantic drift governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if (
        semantic_drift_tensor is not None
        and semantic_drift_counterfactual is not None
        and semantic_drift_director_panel is not None
    ):
        try:
            from backend.health.semantic_drift_adapter import (
                build_semantic_drift_governance_tile,
            )
            semantic_drift_tile = build_semantic_drift_governance_tile(
                drift_tensor=semantic_drift_tensor,
                counterfactual=semantic_drift_counterfactual,
                drift_director_panel=semantic_drift_director_panel,
            )
            payload["semantic_drift"] = semantic_drift_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to semantic drift tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach uplift safety governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    # This tile does NOT control deployments; it summarizes risk
    if (
        uplift_safety_tensor is not None
        and uplift_stability_forecaster is not None
        and uplift_gate_decision is not None
    ):
        try:
            from backend.health.uplift_safety_adapter import (
                build_uplift_safety_governance_tile,
            )
            uplift_safety_tile = build_uplift_safety_governance_tile(
                safety_tensor=uplift_safety_tensor,
                stability_forecaster=uplift_stability_forecaster,
                gate_decision=uplift_gate_decision,
            )
            payload["uplift_safety"] = uplift_safety_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to uplift safety tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach runtime profile health tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if runtime_profile_health is not None:
        try:
            from backend.health.runtime_profile_adapter import (
                build_runtime_profile_tile_for_global_health,
            )
            runtime_profile_tile = build_runtime_profile_tile_for_global_health(
                chaos_summary=runtime_profile_health,
            )
            if runtime_profile_tile is not None:
                payload["runtime_profile"] = runtime_profile_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to runtime profile tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach ledger guard advisory tile (SHADOW mode only)
    # This tile is purely observational and carries no enforcement logic
    if ledger_guard_result is not None:
        try:
            from backend.health.ledger_guard_tile import build_ledger_guard_tile

            ledger_guard_tile = build_ledger_guard_tile(ledger_guard_result)
            payload["ledger_guard"] = ledger_guard_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to ledger guard tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach PRNG governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    # PRNG drift is a long-horizon governance signal, not a single-run gate
    # Note: prng_governance_tile is not in function signature
    try:
        if 'prng_governance_tile' in locals() and prng_governance_tile is not None:
            from backend.health.prng_governance_adapter import (
                build_prng_governance_tile_for_global_health,
            )
            prng_tile = build_prng_governance_tile_for_global_health(
                prng_governance_tile=prng_governance_tile,
            )
            payload["prng"] = prng_tile
    except (NameError, Exception):
        # SHADOW MODE: Never fail the build due to PRNG governance tile issues
        # Silently continue without the tile
        # NameError occurs when prng_governance_tile is not in function signature
        pass

    # Phase X: Attach budget invariants tile (SHADOW mode only)
    # Budget Invariants = "Energy Law" of First-Light runs
    # Storyline + BNH-Φ = temporal coherence evidence
    # These appear in P3 stability reports and P4 calibration bundles
    # Note: budget_invariants_timeline is not in function signature
    try:
        if 'budget_invariants_timeline' in locals() and budget_invariants_timeline is not None:
            from backend.health.budget_invariants_adapter import (
                build_budget_invariants_tile_for_global_health,
            )
            budget_invariants_tile = build_budget_invariants_tile_for_global_health(
                invariant_timeline=budget_invariants_timeline,
            )
            payload["budget_invariants"] = budget_invariants_tile
    except (NameError, Exception):
            # SHADOW MODE: Never fail the build due to budget invariants tile issues
            # Silently continue without the tile
            # NameError occurs when budget_invariants_timeline is not in function signature
            pass

    # Phase X: Attach evidence quality governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    if (
        evidence_phase_portrait is not None
        or evidence_forecast is not None
        or evidence_director_panel_v2 is not None
    ):
        try:
            from backend.health.evidence_quality_adapter import (
                build_evidence_governance_tile,
            )
            evidence_quality_tile = build_evidence_governance_tile(
                phase_portrait=evidence_phase_portrait,
                forecast=evidence_forecast,
                director_panel_v2=evidence_director_panel_v2,
            )
            payload["evidence_quality"] = evidence_quality_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to evidence quality tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach replay governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    # SHADOW MODE CONTRACT:
    # - The replay_governance tile does NOT influence any other tiles
    # - No control flow depends on the replay_governance tile
    # - The replay_governance tile is purely for observability
    if replay_radar is not None and replay_promotion_eval is not None:
        try:
            from backend.health.replay_governance_adapter import (
                build_replay_governance_tile_for_global_health,
            )
            replay_governance_tile = build_replay_governance_tile_for_global_health(
                radar=replay_radar,
                promotion_eval=replay_promotion_eval,
                director_panel=replay_director_panel,
            )
            payload["replay_governance"] = replay_governance_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to replay governance tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach metrics governance tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    # SHADOW MODE CONTRACT (CLAUDE D - Metrics Conformance Layer):
    # - The metrics_governance tile does NOT influence safety or replay layers
    # - No control flow depends on the metrics_governance tile
    # - The metrics_governance tile is purely for observability
    # - Tile is only attached when at least one of drift_compass/budget_view/governance_signal is present
    if (
        metrics_drift_compass is not None
        or metrics_budget_view is not None
        or metrics_governance_signal is not None
    ):
        try:
            from backend.health.metrics_governance_adapter import (
                build_metrics_governance_tile_for_global_health,
            )
            metrics_governance_tile = build_metrics_governance_tile_for_global_health(
                drift_compass=metrics_drift_compass,
                budget_view=metrics_budget_view,
                governance_signal=metrics_governance_signal,
            )
            payload["metrics_governance"] = metrics_governance_tile
        except Exception:
            # SHADOW MODE: Never fail the build due to metrics governance tile issues
            # Silently continue without the tile
            pass

    # Phase X: Attach topology_bundle tile (SHADOW mode only)
    # This tile is purely observational and does NOT influence other tiles
    # SHADOW MODE CONTRACT (CLAUDE B - Topology/Bundle Layer):
    # - The topology_bundle tile does NOT influence safety or replay layers
    # - No control flow depends on the topology_bundle tile
    # - The topology_bundle tile is purely for observability
    # - Zero gating logic — SHADOW MODE only
    # - Tile is only attached when topology_bundle_joint_view and topology_bundle_consistency are present
    # Note: topology_bundle_joint_view, topology_bundle_consistency, topology_bundle_director_panel
    # are not in function signature yet; defensive check via locals()
    try:
        topology_bundle_joint_view_val = locals().get('topology_bundle_joint_view', None)
        topology_bundle_consistency_val = locals().get('topology_bundle_consistency', None)
        topology_bundle_director_panel_val = locals().get('topology_bundle_director_panel', None)

        if topology_bundle_joint_view_val is not None and topology_bundle_consistency_val is not None:
            from backend.health.topology_bundle_adapter import (
                build_topology_bundle_tile_for_global_health,
            )
            topology_bundle_tile = build_topology_bundle_tile_for_global_health(
                joint_view=topology_bundle_joint_view_val,
                consistency_result=topology_bundle_consistency_val,
                director_panel=topology_bundle_director_panel_val,
            )
            payload["topology_bundle"] = topology_bundle_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to topology_bundle tile issues
        # Silently continue without the tile
        pass

    return payload


def attach_dynamics_tile(
    payload: MutableMapping[str, Any],
    snapshots: Sequence[DynamicsDebugSnapshot],
) -> Dict[str, Any]:
    """Attach/overwrite the dynamics tile on an existing payload."""
    payload = dict(payload)
    payload["schema_version"] = payload.get(
        "schema_version", GLOBAL_HEALTH_SURFACE_SCHEMA_VERSION
    )
    payload["dynamics"] = summarize_dynamics_for_global_console(snapshots)
    return payload


def attach_usla_tile(
    payload: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """
    Attach USLA tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The USLA tile does NOT influence any other tiles
    - No control flow depends on the USLA tile contents

    Args:
        payload: Existing global health payload

    Returns:
        Updated payload with USLA tile (if SHADOW mode enabled)
    """
    payload = dict(payload)
    usla_tile = _build_usla_tile()
    if usla_tile is not None:
        payload["usla"] = usla_tile
    return payload


def attach_semantic_tda_tile(
    payload: MutableMapping[str, Any],
    semantic_panel: Dict[str, Any],
    tda_panel: Dict[str, Any],
    semantic_timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach semantic-TDA tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The semantic_tda tile does NOT influence any other tiles
    - No control flow depends on the semantic_tda tile contents

    Args:
        payload: Existing global health payload
        semantic_panel: Semantic director panel
        tda_panel: TDA health panel
        semantic_timeline: Optional full semantic timeline

    Returns:
        Updated payload with semantic_tda tile (if both panels available)
    """
    payload = dict(payload)
    try:
        from backend.health.semantic_tda_adapter import (
            build_semantic_tda_tile_for_global_health,
        )
        semantic_tda_tile = build_semantic_tda_tile_for_global_health(
            semantic_panel=semantic_panel,
            tda_panel=tda_panel,
            semantic_timeline=semantic_timeline,
        )
        payload["semantic_tda"] = semantic_tda_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to semantic-TDA tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_lean_shadow_tile(
    payload: MutableMapping[str, Any],
    shadow_radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach Lean shadow tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The lean_shadow tile does NOT influence any other tiles
    - No control flow depends on the lean_shadow tile contents
    - Compatible with both P3 (synthetic Lean evaluation) and P4 (real-runner + Lean shadow observations)
    - No abort logic, no control feedback, no modification of pipeline/runner decisions

    Args:
        payload: Existing global health payload
        shadow_radar: Shadow capability radar from build_lean_shadow_capability_radar()

    Returns:
        Updated payload with lean_shadow tile (if shadow_radar provided)
    """
    payload = dict(payload)
    try:
        from backend.health.lean_shadow_adapter import (
            build_lean_shadow_tile_for_global_health,
        )
        lean_shadow_tile = build_lean_shadow_tile_for_global_health(shadow_radar)
        payload["lean_shadow"] = lean_shadow_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to Lean shadow tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_semantic_integrity_tile(
    payload: MutableMapping[str, Any],
    invariant_check: Dict[str, Any],
    uplift_preview: Dict[str, Any],
    director_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach semantic integrity tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The semantic_integrity tile does NOT influence any other tiles
    - No control flow depends on the semantic_integrity tile contents

    Args:
        payload: Existing global health payload
        invariant_check: From check_semantic_invariants()
        uplift_preview: From preview_semantic_uplift_gate()
        director_tile: From build_semantic_uplift_director_tile()

    Returns:
        Updated payload with semantic_integrity tile (if all inputs provided)
    """
    payload = dict(payload)
    try:
        from backend.health.semantic_integrity_adapter import (
            build_semantic_integrity_tile,
        )
        semantic_integrity_tile = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        payload["semantic_integrity"] = semantic_integrity_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to semantic integrity tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_atlas_governance_tile(
    payload: MutableMapping[str, Any],
    lattice: Dict[str, Any],
    phase_gate: Dict[str, Any],
    director_tile_v2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach atlas governance tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The atlas_governance tile does NOT influence any other tiles
    - No control flow depends on the atlas_governance tile contents

    Args:
        payload: Existing global health payload
        lattice: Atlas convergence lattice from build_atlas_convergence_lattice()
        phase_gate: Phase transition gate from derive_atlas_phase_transition_gate()
        director_tile_v2: Director tile v2 from build_atlas_director_tile_v2()

    Returns:
        Updated payload with atlas_governance tile (if all inputs provided)
    """
    payload = dict(payload)
    try:
        from backend.health.atlas_governance_adapter import (
            build_atlas_governance_tile,
        )
        atlas_governance_tile = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2,
        )
        payload["atlas_governance"] = atlas_governance_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to atlas governance tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_telemetry_governance_tile(
    payload: MutableMapping[str, Any],
    fusion_tile: Dict[str, Any],
    uplift_gate: Dict[str, Any],
    director_tile_v2: Dict[str, Any],
    telemetry_health: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach telemetry governance tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The telemetry_governance tile does NOT influence any other tiles
    - No control flow depends on the telemetry_governance tile contents

    Args:
        payload: Existing global health payload
        fusion_tile: Fusion tile from build_telemetry_topology_semantic_fusion()
        uplift_gate: Uplift phase gate from build_telemetry_driven_uplift_phase_gate()
        director_tile_v2: Director tile v2 from build_telemetry_director_tile_v2()
        telemetry_health: Optional telemetry health summary

    Returns:
        Updated payload with telemetry_governance tile (if all required inputs provided)
    """
    payload = dict(payload)
    try:
        from backend.health.telemetry_fusion_adapter import (
            build_telemetry_governance_tile,
        )
        telemetry_governance_tile = build_telemetry_governance_tile(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
            director_tile_v2=director_tile_v2,
            telemetry_health=telemetry_health,
        )
        payload["telemetry_governance"] = telemetry_governance_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to telemetry governance tile issues
        # Silently continue without the tile
        pass
    return payload


__all__ = [
    "GLOBAL_HEALTH_SURFACE_SCHEMA_VERSION",
    "attach_dynamics_tile",
    "attach_lean_shadow_tile",
    "attach_semantic_integrity_tile",
    "attach_semantic_tda_tile",
    "attach_telemetry_governance_tile",
    "attach_usla_tile",
    "build_global_health_surface",
    "clear_usla_producer",
    "set_usla_producer",
]


def attach_dynamics_tile(
    payload: MutableMapping[str, Any],
    snapshots: Sequence[DynamicsDebugSnapshot],
) -> Dict[str, Any]:
    """Attach/overwrite the dynamics tile on an existing payload."""
    payload = dict(payload)
    payload["schema_version"] = payload.get(
        "schema_version", GLOBAL_HEALTH_SURFACE_SCHEMA_VERSION
    )
    payload["dynamics"] = summarize_dynamics_for_global_console(snapshots)
    return payload


def attach_usla_tile(
    payload: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """
    Attach USLA tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The USLA tile does NOT influence any other tiles
    - No control flow depends on the USLA tile contents

    Args:
        payload: Existing global health payload

    Returns:
        Updated payload with USLA tile (if SHADOW mode enabled)
    """
    payload = dict(payload)
    usla_tile = _build_usla_tile()
    if usla_tile is not None:
        payload["usla"] = usla_tile
    return payload


def attach_semantic_tda_tile(
    payload: MutableMapping[str, Any],
    semantic_panel: Dict[str, Any],
    tda_panel: Dict[str, Any],
    semantic_timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach semantic-TDA tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The semantic_tda tile does NOT influence any other tiles
    - No control flow depends on the semantic_tda tile contents

    Args:
        payload: Existing global health payload
        semantic_panel: Semantic director panel
        tda_panel: TDA health panel
        semantic_timeline: Optional full semantic timeline

    Returns:
        Updated payload with semantic_tda tile (if both panels available)
    """
    payload = dict(payload)
    try:
        from backend.health.semantic_tda_adapter import (
            build_semantic_tda_tile_for_global_health,
        )
        semantic_tda_tile = build_semantic_tda_tile_for_global_health(
            semantic_panel=semantic_panel,
            tda_panel=tda_panel,
            semantic_timeline=semantic_timeline,
        )
        payload["semantic_tda"] = semantic_tda_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to semantic-TDA tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_lean_shadow_tile(
    payload: MutableMapping[str, Any],
    shadow_radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach Lean shadow tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The lean_shadow tile does NOT influence any other tiles
    - No control flow depends on the lean_shadow tile contents
    - Compatible with both P3 (synthetic Lean evaluation) and P4 (real-runner + Lean shadow observations)
    - No abort logic, no control feedback, no modification of pipeline/runner decisions

    Args:
        payload: Existing global health payload
        shadow_radar: Shadow capability radar from build_lean_shadow_capability_radar()

    Returns:
        Updated payload with lean_shadow tile (if shadow_radar provided)
    """
    payload = dict(payload)
    try:
        from backend.health.lean_shadow_adapter import (
            build_lean_shadow_tile_for_global_health,
        )
        lean_shadow_tile = build_lean_shadow_tile_for_global_health(shadow_radar)
        payload["lean_shadow"] = lean_shadow_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to Lean shadow tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_semantic_integrity_tile(
    payload: MutableMapping[str, Any],
    invariant_check: Dict[str, Any],
    uplift_preview: Dict[str, Any],
    director_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach semantic integrity tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The semantic_integrity tile does NOT influence any other tiles
    - No control flow depends on the semantic_integrity tile contents

    Args:
        payload: Existing global health payload
        invariant_check: From check_semantic_invariants()
        uplift_preview: From preview_semantic_uplift_gate()
        director_tile: From build_semantic_uplift_director_tile()

    Returns:
        Updated payload with semantic_integrity tile (if all inputs provided)
    """
    payload = dict(payload)
    try:
        from backend.health.semantic_integrity_adapter import (
            build_semantic_integrity_tile,
        )
        semantic_integrity_tile = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        payload["semantic_integrity"] = semantic_integrity_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to semantic integrity tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_topology_pressure_tile(
    payload: MutableMapping[str, Any],
    pressure_field: Dict[str, Any],
    promotion_gate: Dict[str, Any],
    console_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach topology pressure tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The topology_pressure tile does NOT influence any other tiles
    - No control flow depends on the topology_pressure tile contents
    - No modification of topology state or curriculum decisions

    Args:
        payload: Existing global health payload
        pressure_field: Pressure field from build_topological_pressure_field()
        promotion_gate: Promotion gate from topology_curriculum_promotion_gate()
        console_tile: Console tile from build_topology_console_tile()

    Returns:
        Updated payload with topology_pressure tile (if all components provided)
    """
    payload = dict(payload)
    try:
        from backend.health.topology_pressure_adapter import (
            build_topology_pressure_governance_tile,
        )
        topology_pressure_tile = build_topology_pressure_governance_tile(
            pressure_field=pressure_field,
            promotion_gate=promotion_gate,
            console_tile=console_tile,
        )
        payload["topology_pressure"] = topology_pressure_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to topology pressure tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_atlas_governance_tile(
    payload: MutableMapping[str, Any],
    lattice: Dict[str, Any],
    phase_gate: Dict[str, Any],
    director_tile_v2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach atlas governance tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The atlas_governance tile does NOT influence any other tiles
    - No control flow depends on the atlas_governance tile contents

    Args:
        payload: Existing global health payload
        lattice: Atlas convergence lattice from build_atlas_convergence_lattice()
        phase_gate: Phase transition gate from derive_atlas_phase_transition_gate()
        director_tile_v2: Director tile v2 from build_atlas_director_tile_v2()

    Returns:
        Updated payload with atlas_governance tile (if all inputs provided)
    """
    payload = dict(payload)
    try:
        from backend.health.atlas_governance_adapter import (
            build_atlas_governance_tile,
        )
        atlas_governance_tile = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2,
        )
        payload["atlas_governance"] = atlas_governance_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to atlas governance tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_replay_governance_tile(
    payload: MutableMapping[str, Any],
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach replay governance tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The replay_governance tile does NOT influence any other tiles
    - No control flow depends on the replay_governance tile contents
    - The tile is purely for observability and logging

    Args:
        payload: Existing global health payload
        radar: Replay governance radar view (from build_replay_safety_governance_view)
        promotion_eval: Promotion evaluation (from evaluate_replay_safety_for_promotion)
        director_panel: Optional director panel (from build_replay_safety_director_panel)

    Returns:
        Updated payload with replay_governance tile (if radar and promotion_eval provided)
    """
    payload = dict(payload)
    try:
        from backend.health.replay_governance_adapter import (
            build_replay_governance_tile_for_global_health,
        )
        replay_governance_tile = build_replay_governance_tile_for_global_health(
            radar=radar,
            promotion_eval=promotion_eval,
            director_panel=director_panel,
        )
        payload["replay_governance"] = replay_governance_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to replay governance tile issues
        # Silently continue without the tile
        pass
    return payload


__all__ = [
    "GLOBAL_HEALTH_SURFACE_SCHEMA_VERSION",
    "attach_atlas_governance_tile",
    "attach_budget_invariants_tile",
    "attach_dynamics_tile",
    "attach_harmonic_alignment_tile",
    "attach_lean_shadow_tile",
    "attach_replay_governance_tile",
    "attach_semantic_integrity_tile",
    "attach_semantic_tda_tile",
    "attach_topology_pressure_tile",
    "attach_uplift_council_tile",
    "attach_usla_tile",
    "build_global_health_surface",
    "clear_usla_producer",
    "set_usla_producer",
]


def attach_topology_pressure_tile(
    payload: MutableMapping[str, Any],
    pressure_field: Dict[str, Any],
    promotion_gate: Dict[str, Any],
    console_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach topology pressure tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The topology_pressure tile does NOT influence any other tiles
    - No control flow depends on the topology_pressure tile contents
    - No modification of topology state or curriculum decisions

    Args:
        payload: Existing global health payload
        pressure_field: Pressure field from build_topological_pressure_field()
        promotion_gate: Promotion gate from topology_curriculum_promotion_gate()
        console_tile: Console tile from build_topology_console_tile()

    Returns:
        Updated payload with topology_pressure tile (if all components provided)
    """
    payload = dict(payload)
    try:
        from backend.health.topology_pressure_adapter import (
            build_topology_pressure_governance_tile,
        )
        topology_pressure_tile = build_topology_pressure_governance_tile(
            pressure_field=pressure_field,
            promotion_gate=promotion_gate,
            console_tile=console_tile,
        )
        payload["topology_pressure"] = topology_pressure_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to topology pressure tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_atlas_governance_tile(
    payload: MutableMapping[str, Any],
    lattice: Dict[str, Any],
    phase_gate: Dict[str, Any],
    director_tile_v2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach atlas governance tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The atlas_governance tile does NOT influence any other tiles
    - No control flow depends on the atlas_governance tile contents

    Args:
        payload: Existing global health payload
        lattice: Atlas convergence lattice from build_atlas_convergence_lattice()
        phase_gate: Phase transition gate from derive_atlas_phase_transition_gate()
        director_tile_v2: Director tile v2 from build_atlas_director_tile_v2()

    Returns:
        Updated payload with atlas_governance tile (if all inputs provided)
    """
    payload = dict(payload)
    try:
        from backend.health.atlas_governance_adapter import (
            build_atlas_governance_tile,
        )
        atlas_governance_tile = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2,
        )
        payload["atlas_governance"] = atlas_governance_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to atlas governance tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_metrics_governance_tile(
    payload: MutableMapping[str, Any],
    drift_compass: Optional[Dict[str, Any]] = None,
    budget_view: Optional[Dict[str, Any]] = None,
    governance_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach metrics governance tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT (CLAUDE D - Metrics Conformance Layer):
    - This function is read-only (aside from dict mutation)
    - The metrics_governance tile does NOT influence safety or replay layers
    - No control flow depends on the metrics_governance tile contents
    - The metrics_governance tile is purely for observability

    Args:
        payload: Existing global health payload
        drift_compass: Drift compass from metric_drift_compass.schema.json
        budget_view: Budget view from metric_budget_joint_view.schema.json
        governance_signal: Governance signal from metric_governance_signal.schema.json

    Returns:
        Updated payload with metrics_governance tile (if any inputs provided)
    """
    payload = dict(payload)
    if drift_compass is None and budget_view is None and governance_signal is None:
        return payload

    try:
        from backend.health.metrics_governance_adapter import (
            build_metrics_governance_tile_for_global_health,
        )
        metrics_governance_tile = build_metrics_governance_tile_for_global_health(
            drift_compass=drift_compass,
            budget_view=budget_view,
            governance_signal=governance_signal,
        )
        payload["metrics_governance"] = metrics_governance_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to metrics governance tile issues
        # Silently continue without the tile
        pass
    return payload


def attach_topology_bundle_tile(
    payload: MutableMapping[str, Any],
    joint_view: Dict[str, Any],
    consistency_result: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach topology bundle tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT (CLAUDE B - Topology/Bundle Layer):
    - This function is read-only (aside from dict mutation)
    - The topology_bundle tile does NOT influence any other tiles
    - No control flow depends on the topology_bundle tile contents
    - Zero gating logic — SHADOW MODE only
    - The tile is purely for observability and logging

    Args:
        payload: Existing global health payload
        joint_view: Topology bundle joint view per topology_bundle_joint_view.schema.json
        consistency_result: Cross-system consistency evaluation result
        director_panel: Optional director panel per topology_bundle_director_panel.schema.json

    Returns:
        Updated payload with topology_bundle tile (if joint_view and consistency_result provided)
    """
    payload = dict(payload)
    try:
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_tile_for_global_health,
        )
        topology_bundle_tile = build_topology_bundle_tile_for_global_health(
            joint_view=joint_view,
            consistency_result=consistency_result,
            director_panel=director_panel,
        )
        payload["topology_bundle"] = topology_bundle_tile
    except Exception:
        # SHADOW MODE: Never fail the build due to topology_bundle tile issues
        # Silently continue without the tile
        pass
    return payload


__all__ = [
    "GLOBAL_HEALTH_SURFACE_SCHEMA_VERSION",
    "attach_atlas_governance_tile",
    "attach_budget_invariants_tile",
    "attach_dynamics_tile",
    "attach_harmonic_alignment_tile",
    "attach_lean_shadow_tile",
    "attach_metrics_governance_tile",
    "attach_replay_governance_tile",
    "attach_semantic_integrity_tile",
    "attach_semantic_tda_tile",
    "attach_topology_bundle_tile",
    "attach_topology_pressure_tile",
    "attach_uplift_council_tile",
    "attach_usla_tile",
    "build_global_health_surface",
    "clear_usla_producer",
    "set_usla_producer",
]
