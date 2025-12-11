"""
Curriculum Stability Integration Helpers

Provides convenience functions for integrating curriculum stability envelope
into runners (U2Runner, RFLRunner), First Light reports, and Evidence Packs.
"""

from typing import Dict, List, Any, Optional
from .stability import (
    build_stability_envelope,
    add_stability_to_first_light,
    add_stability_to_p4_calibration,
    attach_curriculum_stability_to_evidence,
    summarize_curriculum_stability_for_council,
    CurriculumStabilityEnvelope,
)


def extract_slice_metrics_from_rfl_runner(runner: Any) -> List[Dict[str, Any]]:
    """
    Extract slice metrics from RFLRunner for stability analysis.
    
    Args:
        runner: RFLRunner instance
        
    Returns:
        List of slice metrics dicts suitable for stability envelope
    """
    slice_metrics_list = []
    
    # Extract from policy ledger
    if hasattr(runner, 'policy_ledger') and runner.policy_ledger:
        # Group by slice
        slice_groups: Dict[str, List[Any]] = {}
        for entry in runner.policy_ledger:
            slice_name = entry.slice_name
            if slice_name not in slice_groups:
                slice_groups[slice_name] = []
            slice_groups[slice_name].append(entry)
        
        # Compute aggregate metrics per slice
        for slice_name, entries in slice_groups.items():
            # Average metrics across runs for this slice
            coverage_rates = [e.coverage_rate for e in entries if e.coverage_rate > 0]
            abstention_rates = [e.abstention_fraction for e in entries]
            
            avg_coverage = sum(coverage_rates) / len(coverage_rates) if coverage_rates else 0.0
            avg_abstention = sum(abstention_rates) / len(abstention_rates) if abstention_rates else 0.0
            
            # Get slice params from config
            slice_cfg = None
            if hasattr(runner, 'config') and hasattr(runner.config, 'curriculum'):
                for s in runner.config.curriculum:
                    if s.name == slice_name:
                        slice_cfg = s
                        break
            
            params = {}
            if slice_cfg:
                params = {
                    "atoms": slice_cfg.atoms if hasattr(slice_cfg, 'atoms') else 0,
                    "depth_max": slice_cfg.depth_max if hasattr(slice_cfg, 'depth_max') else 0,
                    "breadth_max": slice_cfg.max_breadth if hasattr(slice_cfg, 'max_breadth') else 0,
                }
            
            slice_metrics_list.append({
                "slice_name": slice_name,
                "params": params,
                "coverage_rate": avg_coverage,
                "abstention_rate": avg_abstention,
            })
    
    return slice_metrics_list


def extract_slice_metrics_from_u2_runner(runner: Any) -> List[Dict[str, Any]]:
    """
    Extract slice metrics from U2Runner for stability analysis.
    
    Args:
        runner: U2Runner instance
        
    Returns:
        List of slice metrics dicts suitable for stability envelope
    """
    slice_metrics_list = []
    
    # U2Runner has a single slice, extract from config
    if hasattr(runner, 'config'):
        config = runner.config
        slice_name = getattr(config, 'slice_name', 'default')
        
        # Extract slice params
        params = {}
        if hasattr(config, 'slice_config') and config.slice_config:
            params = config.slice_config
        else:
            params = {
                "max_depth": getattr(config, 'max_depth', 0),
                "max_beam_width": getattr(config, 'max_beam_width', 0),
            }
        
        # Compute metrics from results
        coverage_rate = 0.0
        abstention_rate = 0.0
        
        if hasattr(runner, 'results') and runner.results:
            # Average across cycles
            successful = [r for r in runner.results if r.success]
            if successful:
                # Heuristic: use candidates_generated as proxy for coverage
                total_gen = sum(r.candidates_generated for r in successful)
                total_proc = sum(r.candidates_processed for r in successful)
                if total_proc > 0:
                    coverage_rate = min(1.0, total_gen / (total_proc * 2.0))
                
                # Heuristic: failures as abstentions
                failed = len(runner.results) - len(successful)
                abstention_rate = failed / len(runner.results) if runner.results else 0.0
        
        slice_metrics_list.append({
            "slice_name": slice_name,
            "params": params,
            "coverage_rate": coverage_rate,
            "abstention_rate": abstention_rate,
        })
    
    return slice_metrics_list


def add_stability_to_rfl_results(
    results: Dict[str, Any],
    runner: Any,
    include_council: bool = False
) -> Dict[str, Any]:
    """
    Add curriculum stability envelope to RFL results dictionary.
    
    This is the primary integration point for RFLRunner._export_results.
    
    Args:
        results: Results dictionary from RFLRunner
        runner: RFLRunner instance
        include_council: Whether to include council advisory
        
    Returns:
        Updated results dictionary with curriculum_stability_envelope
    """
    # Extract slice metrics
    slice_metrics_list = extract_slice_metrics_from_rfl_runner(runner)
    
    # Build historical data if available
    historical_data: Dict[str, List[Dict[str, Any]]] = {}
    if hasattr(runner, 'policy_ledger') and runner.policy_ledger:
        for entry in runner.policy_ledger:
            slice_name = entry.slice_name
            if slice_name not in historical_data:
                historical_data[slice_name] = []
            historical_data[slice_name].append({
                "coverage_rate": entry.coverage_rate,
                "abstention_rate": entry.abstention_fraction,
            })
    
    # Build envelope
    if slice_metrics_list:
        envelope = build_stability_envelope(slice_metrics_list, historical_data)
        
        # Add to results as First Light block
        results = add_stability_to_first_light(results, envelope)
        
        # Optionally add council advisory
        if include_council:
            council_advisory = summarize_curriculum_stability_for_council(envelope)
            results["uplift_council_advisory"] = council_advisory
    
    return results


def add_stability_to_u2_results(
    results: Dict[str, Any],
    runner: Any,
    include_council: bool = False
) -> Dict[str, Any]:
    """
    Add curriculum stability envelope to U2 results dictionary.
    
    Args:
        results: Results dictionary from U2Runner
        runner: U2Runner instance
        include_council: Whether to include council advisory
        
    Returns:
        Updated results dictionary with curriculum_stability_envelope
    """
    # Extract slice metrics
    slice_metrics_list = extract_slice_metrics_from_u2_runner(runner)
    
    # U2 doesn't have long historical data, so no history
    historical_data = None
    
    # Build envelope
    if slice_metrics_list:
        envelope = build_stability_envelope(slice_metrics_list, historical_data)
        
        # Add to results as First Light block
        results = add_stability_to_first_light(results, envelope)
        
        # Optionally add council advisory
        if include_council:
            council_advisory = summarize_curriculum_stability_for_council(envelope)
            results["uplift_council_advisory"] = council_advisory
    
    return results


def create_p4_calibration_report_with_stability(
    calibration_data: Dict[str, Any],
    slice_metrics_list: List[Dict[str, Any]],
    historical_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Dict[str, Any]:
    """
    Create a P4 calibration report with curriculum stability section.
    
    Args:
        calibration_data: Base calibration report data
        slice_metrics_list: Current slice metrics
        historical_data: Optional historical metrics per slice
        
    Returns:
        P4 calibration report with curriculum_stability section
    """
    # Build envelope
    envelope = build_stability_envelope(slice_metrics_list, historical_data)
    
    # Add stability to P4 report
    report = add_stability_to_p4_calibration(calibration_data, envelope)
    
    return report


def create_evidence_pack_with_stability(
    evidence: Dict[str, Any],
    slice_metrics_list: List[Dict[str, Any]],
    historical_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Dict[str, Any]:
    """
    Create an evidence pack with curriculum stability attachment.
    
    Args:
        evidence: Base evidence pack
        slice_metrics_list: Current slice metrics
        historical_data: Optional historical metrics per slice
        
    Returns:
        Evidence pack with curriculum_stability under governance
    """
    # Build envelope
    envelope = build_stability_envelope(slice_metrics_list, historical_data)
    
    # Attach to evidence (non-mutating)
    new_evidence = attach_curriculum_stability_to_evidence(evidence, envelope)
    
    return new_evidence
