# PHASE II — NOT USED IN PHASE I
"""
Curriculum Loader Extension for Decoy Scoring

Provides lightweight loading of Phase II uplift slices with formula pools
for decoy scoring. This is a companion to CurriculumLoaderV2 that relaxes
schema validation to support the new decoy framework.

Usage:
    from experiments.decoys.loader import CurriculumDecoyLoader

    loader = CurriculumDecoyLoader()
    difficulty = loader.get_decoy_difficulty("slice_uplift_goal")
    report = loader.get_slice_report("slice_uplift_goal")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml

from .scoring import (
    DecoyScorer,
    DecoyScore,
    SliceScoreReport,
    score_slice_decoys,
    compute_confusability_index,
    score_all_slices,
    generate_markdown_report,
    generate_json_report,
)


class CurriculumDecoyLoader:
    """
    Loader for Phase II curriculum with decoy scoring integration.
    
    This loader provides access to decoy difficulty scores and confusability
    metrics for uplift slices with formula_pool_entries.
    
    Attributes:
        filepath: Path to the curriculum YAML file
        _data: Parsed YAML content
        _reports: Cached score reports
    """
    
    def __init__(self, filepath: str = "config/curriculum_uplift_phase2.yaml"):
        """
        Initialize the loader.
        
        Args:
            filepath: Path to the Phase II curriculum YAML file.
        """
        self.filepath = filepath
        with open(filepath, 'r', encoding='utf-8') as f:
            self._data: Dict[str, Any] = yaml.safe_load(f)
        self._reports: Dict[str, SliceScoreReport] = {}
    
    def list_uplift_slices(self) -> List[str]:
        """
        Return list of slice names that have formula_pool_entries.
        
        Returns:
            List of slice names with decoy-enabled formula pools.
        """
        slices = self._data.get('slices', {})
        return [
            name for name, data in slices.items()
            if isinstance(data, dict) and 'formula_pool_entries' in data
        ]
    
    def get_slice_report(self, slice_name: str) -> SliceScoreReport:
        """
        Get the full score report for a slice.
        
        Args:
            slice_name: Name of the slice to score.
            
        Returns:
            SliceScoreReport with all formula scores.
            
        Raises:
            KeyError: If slice not found.
            ValueError: If slice has no formula pool.
        """
        if slice_name not in self._reports:
            self._reports[slice_name] = score_slice_decoys(slice_name, self.filepath)
        return self._reports[slice_name]
    
    def get_decoy_difficulty(self, slice_name: str) -> Dict[str, Any]:
        """
        Get decoy difficulty summary for a slice.
        
        This is the primary integration point for CurriculumLoaderV2.
        
        Args:
            slice_name: Name of the slice.
            
        Returns:
            Dictionary with:
            - confusability_index: Overall confusability (0-1)
            - avg_near_difficulty: Average difficulty of near-decoys
            - avg_far_difficulty: Average difficulty of far-decoys
            - decoy_count: Total number of decoys
            - details: Per-decoy scores
        """
        report = self.get_slice_report(slice_name)
        
        return {
            "confusability_index": report.confusability_index,
            "avg_near_difficulty": report.avg_near_difficulty,
            "avg_far_difficulty": report.avg_far_difficulty,
            "decoy_count": len(report.decoys_near) + len(report.decoys_far),
            "target_count": len(report.targets),
            "details": {
                "near": [d.to_dict() for d in report.decoys_near],
                "far": [d.to_dict() for d in report.decoys_far],
            },
        }
    
    def get_all_reports(self) -> Dict[str, SliceScoreReport]:
        """
        Get score reports for all uplift slices.
        
        Returns:
            Dictionary mapping slice names to their reports.
        """
        return score_all_slices(self.filepath)
    
    def get_markdown_report(self, slice_names: Optional[List[str]] = None) -> str:
        """
        Generate Markdown report for specified slices.
        
        Args:
            slice_names: List of slices to include (None = all).
            
        Returns:
            Markdown-formatted report string.
        """
        if slice_names is None:
            reports = self.get_all_reports()
        else:
            reports = {name: self.get_slice_report(name) for name in slice_names}
        
        return generate_markdown_report(reports)
    
    def get_json_report(self, slice_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate JSON report for specified slices.
        
        Args:
            slice_names: List of slices to include (None = all).
            
        Returns:
            JSON-serializable dictionary.
        """
        if slice_names is None:
            reports = self.get_all_reports()
        else:
            reports = {name: self.get_slice_report(name) for name in slice_names}
        
        return generate_json_report(reports)
    
    def check_monotonicity(self) -> List[str]:
        """
        Check that near-decoys are harder than far-decoys on average.
        
        This is a design invariant: near-decoys should have higher
        difficulty scores than far-decoys.
        
        Returns:
            List of warning messages for violations.
        """
        warnings = []
        
        for slice_name in self.list_uplift_slices():
            report = self.get_slice_report(slice_name)
            
            if report.decoys_near and report.decoys_far:
                if report.avg_near_difficulty < report.avg_far_difficulty:
                    warnings.append(
                        f"Monotonicity violation in '{slice_name}': "
                        f"avg_near_difficulty ({report.avg_near_difficulty:.3f}) < "
                        f"avg_far_difficulty ({report.avg_far_difficulty:.3f})"
                    )
        
        return warnings
    
    def check_target_collisions(self) -> List[str]:
        """
        Check that no decoy hashes collide with target hashes.
        
        This is a critical invariant: decoys must never match targets.
        
        Returns:
            List of error messages for any collisions found.
        """
        errors = []
        
        for slice_name in self.list_uplift_slices():
            report = self.get_slice_report(slice_name)
            
            target_hashes = {t.hash for t in report.targets}
            
            for decoy in report.decoys_near + report.decoys_far:
                if decoy.hash in target_hashes:
                    errors.append(
                        f"CRITICAL: Decoy '{decoy.name}' in '{slice_name}' "
                        f"has same hash as a target!"
                    )
        
        return errors

