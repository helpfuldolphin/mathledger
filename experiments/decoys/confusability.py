# PHASE II — NOT USED IN PHASE I
"""
Decoy Confusability Map Engine

Computes a "confusability" metric for decoy formulas that measures how likely
a decoy is to be mistaken for a target formula. This is distinct from "difficulty"
which measures how hard a decoy is to distinguish in general.

Confusability Components:
1. Normalized Syntactic Similarity: Token-level and length similarity
2. Connective Signature Similarity: Distribution of logical operators
3. Atom Substitution Distance: How many atom renames to reach target
4. Implication Chain Alignment Penalty: Structural mismatch in chain depth

All computations are deterministic and suitable for CI integration.

Usage:
    from experiments.decoys.confusability import compute_confusability, ConfusabilityMap

    conf = compute_confusability("q -> (p -> q)", ["p -> (q -> p)"])
    # Returns float in [0, 1] - higher = more confusable

    cmap = ConfusabilityMap("slice_uplift_goal")
    report = cmap.generate_report()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from normalization.canon import normalize, get_atomic_propositions
from derivation.structure import (
    formula_depth,
    atom_frozenset,
    is_implication,
    implication_parts,
    strip_outer_parens,
)
from backend.crypto.hashing import hash_statement


# =============================================================================
# CONFUSABILITY PRIMITIVES
# =============================================================================

@lru_cache(maxsize=4096)
def _tokenize(normalized: str) -> Tuple[str, ...]:
    """
    Tokenize a normalized formula into logical tokens.
    
    Returns tuple of: atoms, operators (->  /\  \/  ~), parentheses
    """
    tokens = []
    i = 0
    while i < len(normalized):
        # Check for multi-char operators
        if normalized[i:i+2] == '->':
            tokens.append('->')
            i += 2
        elif normalized[i:i+2] == '/\\':
            tokens.append('/\\')
            i += 2
        elif normalized[i:i+2] == '\\/':
            tokens.append('\\/')
            i += 2
        elif normalized[i] == '~':
            tokens.append('~')
            i += 1
        elif normalized[i] in '()':
            tokens.append(normalized[i])
            i += 1
        elif normalized[i].isalpha():
            # Atom - collect full identifier
            j = i
            while j < len(normalized) and (normalized[j].isalnum() or normalized[j] == '_'):
                j += 1
            tokens.append(normalized[i:j])
            i = j
        else:
            i += 1  # Skip whitespace or unknown
    
    return tuple(tokens)


@lru_cache(maxsize=4096)
def _get_connective_counts(normalized: str) -> Tuple[int, int, int, int]:
    """
    Count each connective type in a normalized formula.
    
    Returns:
        (implications, conjunctions, disjunctions, negations)
    """
    return (
        normalized.count('->'),
        normalized.count('/\\'),
        normalized.count('\\/'),
        normalized.count('~'),
    )


@lru_cache(maxsize=4096)
def _get_implication_chain_structure(normalized: str) -> Tuple[int, int, int]:
    """
    Analyze implication chain structure.
    
    Returns:
        (total_depth, left_depth, right_depth)
    """
    if not is_implication(normalized):
        return (0, 0, 0)
    
    ante, cons = implication_parts(normalized)
    if ante is None or cons is None:
        return (0, 0, 0)
    
    left_depth = formula_depth(ante)
    right_depth = formula_depth(cons)
    total_depth = formula_depth(normalized)
    
    return (total_depth, left_depth, right_depth)


def _compute_atom_substitution_distance(
    decoy_atoms: frozenset,
    target_atoms: frozenset,
) -> float:
    """
    Compute normalized distance based on atom set differences.
    
    Measures how many atom renames would be needed to align sets.
    
    Returns:
        Float in [0, 1] where 0 = identical sets, 1 = disjoint sets
    """
    if not decoy_atoms and not target_atoms:
        return 0.0
    
    union = decoy_atoms | target_atoms
    if not union:
        return 0.0
    
    intersection = decoy_atoms & target_atoms
    
    # Jaccard distance = 1 - Jaccard similarity
    jaccard_sim = len(intersection) / len(union)
    return 1.0 - jaccard_sim


def _compute_token_similarity(
    decoy_tokens: Tuple[str, ...],
    target_tokens: Tuple[str, ...],
) -> float:
    """
    Compute token-level similarity using longest common subsequence ratio.
    
    Returns:
        Float in [0, 1] where 1 = identical token sequences
    """
    if not decoy_tokens and not target_tokens:
        return 1.0
    if not decoy_tokens or not target_tokens:
        return 0.0
    
    # LCS length computation (dynamic programming)
    m, n = len(decoy_tokens), len(target_tokens)
    
    # Optimize for very long sequences
    if m * n > 10000:
        # Fall back to simpler metric for large formulas
        common = set(decoy_tokens) & set(target_tokens)
        total = set(decoy_tokens) | set(target_tokens)
        return len(common) / len(total) if total else 1.0
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if decoy_tokens[i-1] == target_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    max_len = max(m, n)
    
    return lcs_len / max_len


def _compute_connective_similarity(
    decoy_counts: Tuple[int, int, int, int],
    target_counts: Tuple[int, int, int, int],
) -> float:
    """
    Compute connective signature similarity.
    
    Measures how similar the distribution of logical operators is.
    
    Returns:
        Float in [0, 1] where 1 = identical connective counts
    """
    total_diff = sum(abs(d - t) for d, t in zip(decoy_counts, target_counts))
    max_total = max(sum(decoy_counts), sum(target_counts), 1)
    
    # Normalize by maximum possible difference
    return max(0.0, 1.0 - total_diff / (2 * max_total))


def _compute_chain_alignment_penalty(
    decoy_structure: Tuple[int, int, int],
    target_structure: Tuple[int, int, int],
) -> float:
    """
    Compute implication chain alignment penalty.
    
    Measures structural mismatch in how implications are nested.
    
    Returns:
        Float in [0, 1] where 0 = perfectly aligned, 1 = maximally misaligned
    """
    if decoy_structure == (0, 0, 0) and target_structure == (0, 0, 0):
        return 0.0  # Both non-implications, no penalty
    
    if decoy_structure == (0, 0, 0) or target_structure == (0, 0, 0):
        return 0.5  # One is implication, other isn't - moderate penalty
    
    # Compare total depth
    depth_diff = abs(decoy_structure[0] - target_structure[0])
    
    # Compare left/right balance
    decoy_balance = decoy_structure[1] - decoy_structure[2]
    target_balance = target_structure[1] - target_structure[2]
    balance_diff = abs(decoy_balance - target_balance)
    
    # Normalize penalties
    max_depth = max(decoy_structure[0], target_structure[0], 1)
    depth_penalty = depth_diff / (max_depth + 1)
    balance_penalty = balance_diff / (max_depth + 2)
    
    return min(1.0, 0.6 * depth_penalty + 0.4 * balance_penalty)


# =============================================================================
# MAIN CONFUSABILITY FUNCTION
# =============================================================================

def compute_confusability(
    formula: str,
    target_set: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute confusability score for a formula against a target set.
    
    Confusability measures how likely a formula is to be mistaken for
    one of the target formulas based on structural similarity.
    
    Args:
        formula: The formula to score
        target_set: List of target formulas to compare against
        weights: Optional custom weights for components (defaults provided)
        
    Returns:
        Float in [0, 1] where higher = more confusable with targets
        
    Components:
        - syntactic: Normalized token similarity (default weight: 0.30)
        - connective: Connective signature similarity (default weight: 0.25)
        - atom_distance: Atom substitution distance penalty (default weight: 0.25)
        - chain_penalty: Implication chain alignment penalty (default weight: 0.20)
    """
    if not target_set:
        return 0.0
    
    # Default weights
    if weights is None:
        weights = {
            "syntactic": 0.30,
            "connective": 0.25,
            "atom_distance": 0.25,
            "chain_penalty": 0.20,
        }
    
    # Normalize the formula
    decoy_norm = normalize(formula)
    decoy_tokens = _tokenize(decoy_norm)
    decoy_atoms = atom_frozenset(decoy_norm)
    decoy_connectives = _get_connective_counts(decoy_norm)
    decoy_structure = _get_implication_chain_structure(decoy_norm)
    
    # Compute maximum similarity across all targets
    max_syntactic = 0.0
    max_connective = 0.0
    min_atom_distance = 1.0
    min_chain_penalty = 1.0
    
    for target in target_set:
        target_norm = normalize(target)
        target_tokens = _tokenize(target_norm)
        target_atoms = atom_frozenset(target_norm)
        target_connectives = _get_connective_counts(target_norm)
        target_structure = _get_implication_chain_structure(target_norm)
        
        # Compute component similarities
        syntactic = _compute_token_similarity(decoy_tokens, target_tokens)
        connective = _compute_connective_similarity(decoy_connectives, target_connectives)
        atom_dist = _compute_atom_substitution_distance(decoy_atoms, target_atoms)
        chain_pen = _compute_chain_alignment_penalty(decoy_structure, target_structure)
        
        max_syntactic = max(max_syntactic, syntactic)
        max_connective = max(max_connective, connective)
        min_atom_distance = min(min_atom_distance, atom_dist)
        min_chain_penalty = min(min_chain_penalty, chain_pen)
    
    # Combine scores (higher = more confusable)
    # Note: atom_distance and chain_penalty are penalties, so we invert them
    confusability = (
        weights["syntactic"] * max_syntactic +
        weights["connective"] * max_connective +
        weights["atom_distance"] * (1.0 - min_atom_distance) +
        weights["chain_penalty"] * (1.0 - min_chain_penalty)
    )
    
    return min(1.0, max(0.0, confusability))


def compute_confusability_components(
    formula: str,
    target_set: List[str],
) -> Dict[str, float]:
    """
    Compute individual confusability components for detailed analysis.
    
    Returns:
        Dictionary with component scores and final confusability
    """
    if not target_set:
        return {
            "syntactic": 0.0,
            "connective": 0.0,
            "atom_similarity": 0.0,
            "chain_alignment": 0.0,
            "confusability": 0.0,
        }
    
    decoy_norm = normalize(formula)
    decoy_tokens = _tokenize(decoy_norm)
    decoy_atoms = atom_frozenset(decoy_norm)
    decoy_connectives = _get_connective_counts(decoy_norm)
    decoy_structure = _get_implication_chain_structure(decoy_norm)
    
    max_syntactic = 0.0
    max_connective = 0.0
    min_atom_distance = 1.0
    min_chain_penalty = 1.0
    
    for target in target_set:
        target_norm = normalize(target)
        target_tokens = _tokenize(target_norm)
        target_atoms = atom_frozenset(target_norm)
        target_connectives = _get_connective_counts(target_norm)
        target_structure = _get_implication_chain_structure(target_norm)
        
        syntactic = _compute_token_similarity(decoy_tokens, target_tokens)
        connective = _compute_connective_similarity(decoy_connectives, target_connectives)
        atom_dist = _compute_atom_substitution_distance(decoy_atoms, target_atoms)
        chain_pen = _compute_chain_alignment_penalty(decoy_structure, target_structure)
        
        max_syntactic = max(max_syntactic, syntactic)
        max_connective = max(max_connective, connective)
        min_atom_distance = min(min_atom_distance, atom_dist)
        min_chain_penalty = min(min_chain_penalty, chain_pen)
    
    confusability = compute_confusability(formula, target_set)
    
    return {
        "syntactic": round(max_syntactic, 4),
        "connective": round(max_connective, 4),
        "atom_similarity": round(1.0 - min_atom_distance, 4),
        "chain_alignment": round(1.0 - min_chain_penalty, 4),
        "confusability": round(confusability, 4),
    }


# =============================================================================
# CONFUSABILITY MAP
# =============================================================================

@dataclass
class FormulaConfusability:
    """Confusability data for a single formula."""
    name: str
    formula: str
    normalized: str
    hash: str
    role: str
    difficulty: float
    confusability: float
    components: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "formula": self.formula,
            "normalized": self.normalized,
            "hash": self.hash,
            "role": self.role,
            "difficulty": round(self.difficulty, 4),
            "confusability": round(self.confusability, 4),
            "components": self.components,
        }


@dataclass
class ConfusabilityMapReport:
    """Complete confusability map for a slice."""
    slice_name: str
    formulas: List[FormulaConfusability] = field(default_factory=list)
    
    # Statistics
    avg_near_confusability: float = 0.0
    avg_far_confusability: float = 0.0
    near_far_gap: float = 0.0
    coverage_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "formulas": [f.to_dict() for f in self.formulas],
            "statistics": {
                "avg_near_confusability": round(self.avg_near_confusability, 4),
                "avg_far_confusability": round(self.avg_far_confusability, 4),
                "near_far_gap": round(self.near_far_gap, 4),
                "coverage_score": round(self.coverage_score, 4),
            },
        }


class ConfusabilityMap:
    """
    Generates a confusability map for a slice.
    
    The map shows how each formula relates to targets in terms of
    both difficulty and confusability, enabling analysis of decoy quality.
    """
    
    def __init__(
        self,
        slice_name: str,
        config_path: str = "config/curriculum_uplift_phase2.yaml",
    ):
        self.slice_name = slice_name
        self.config_path = config_path
        self._load_slice()
    
    def _load_slice(self) -> None:
        """Load slice configuration."""
        import yaml
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        slices = config.get('slices', {})
        if self.slice_name not in slices:
            raise KeyError(f"Slice '{self.slice_name}' not found")
        
        slice_data = slices[self.slice_name]
        if not isinstance(slice_data, dict) or 'formula_pool_entries' not in slice_data:
            raise ValueError(f"Slice '{self.slice_name}' has no formula_pool_entries")
        
        raw_entries = slice_data['formula_pool_entries']
        
        # Normalize entries to dict format
        if not raw_entries:
            raise ValueError(f"Slice '{self.slice_name}' has empty formula_pool_entries")
        
        if isinstance(raw_entries[0], str):
            # Legacy format: treat all as targets
            self.entries = [
                {
                    "name": f"formula_{i}",
                    "formula": formula,
                    "role": "target",
                }
                for i, formula in enumerate(raw_entries)
            ]
        else:
            self.entries = raw_entries
        
        # Extract targets
        self.target_entries = [e for e in self.entries if e.get('role') == 'target']
        self.target_formulas = [e['formula'] for e in self.target_entries]
    
    def generate_report(self) -> ConfusabilityMapReport:
        """Generate a complete confusability map report."""
        from .scoring import DecoyScorer
        
        report = ConfusabilityMapReport(slice_name=self.slice_name)
        scorer = DecoyScorer()
        
        near_confusabilities = []
        far_confusabilities = []
        
        for entry in self.entries:
            name = entry.get('name', 'unknown')
            formula = entry.get('formula', '')
            role = entry.get('role', 'unknown')
            
            # Get difficulty score
            score = scorer.score_formula(name, formula, role, self.target_formulas)
            
            # Compute confusability
            if role == 'target':
                confusability = 1.0
                components = {
                    "syntactic": 1.0,
                    "connective": 1.0,
                    "atom_similarity": 1.0,
                    "chain_alignment": 1.0,
                    "confusability": 1.0,
                }
            else:
                components = compute_confusability_components(formula, self.target_formulas)
                confusability = components["confusability"]
            
            fc = FormulaConfusability(
                name=name,
                formula=formula,
                normalized=score.normalized,
                hash=score.hash,
                role=role,
                difficulty=score.difficulty,
                confusability=confusability,
                components=components,
            )
            report.formulas.append(fc)
            
            if role == 'decoy_near':
                near_confusabilities.append(confusability)
            elif role == 'decoy_far':
                far_confusabilities.append(confusability)
        
        # Compute statistics
        if near_confusabilities:
            report.avg_near_confusability = sum(near_confusabilities) / len(near_confusabilities)
        if far_confusabilities:
            report.avg_far_confusability = sum(far_confusabilities) / len(far_confusabilities)
        
        report.near_far_gap = report.avg_near_confusability - report.avg_far_confusability
        
        # Coverage score: how well distributed are decoys in difficulty/confusability space
        all_difficulties = [f.difficulty for f in report.formulas if f.role != 'target']
        all_confusabilities = [f.confusability for f in report.formulas if f.role != 'target']
        
        if all_difficulties and all_confusabilities:
            # Measure spread (std dev proxy)
            diff_range = max(all_difficulties) - min(all_difficulties)
            conf_range = max(all_confusabilities) - min(all_confusabilities)
            report.coverage_score = (diff_range + conf_range) / 2
        
        return report
    
    def get_decoy_stats(self) -> Dict[str, Any]:
        """Get statistical summary of decoy distribution."""
        report = self.generate_report()
        
        near_decoys = [f for f in report.formulas if f.role == 'decoy_near']
        far_decoys = [f for f in report.formulas if f.role == 'decoy_far']
        bridges = [f for f in report.formulas if f.role == 'bridge']
        
        def stats_for(formulas: List[FormulaConfusability]) -> Dict[str, Any]:
            if not formulas:
                return {"count": 0, "avg_difficulty": 0.0, "avg_confusability": 0.0}
            return {
                "count": len(formulas),
                "avg_difficulty": sum(f.difficulty for f in formulas) / len(formulas),
                "avg_confusability": sum(f.confusability for f in formulas) / len(formulas),
                "min_difficulty": min(f.difficulty for f in formulas),
                "max_difficulty": max(f.difficulty for f in formulas),
                "min_confusability": min(f.confusability for f in formulas),
                "max_confusability": max(f.confusability for f in formulas),
            }
        
        return {
            "slice_name": self.slice_name,
            "near_decoys": stats_for(near_decoys),
            "far_decoys": stats_for(far_decoys),
            "bridges": stats_for(bridges),
            "targets": {"count": len(self.target_formulas)},
            "summary": {
                "near_far_gap": report.near_far_gap,
                "coverage_score": report.coverage_score,
            },
        }
    
    def get_quality_assessment(self) -> Dict[str, Any]:
        """Assess decoy quality against design invariants."""
        report = self.generate_report()
        
        issues = []
        warnings = []
        
        # Check if we have decoys (legacy format has no decoys)
        has_near = any(f.role == 'decoy_near' for f in report.formulas)
        has_far = any(f.role == 'decoy_far' for f in report.formulas)
        
        # Check 1: Near confusability > Far confusability
        # Only check if we have both near and far decoys
        if has_near and has_far:
            if report.avg_near_confusability <= report.avg_far_confusability:
                issues.append(
                    f"Near-decoys ({report.avg_near_confusability:.3f}) not more confusable "
                    f"than far-decoys ({report.avg_far_confusability:.3f})"
                )
        
        # Check 2: No decoy hash == target hash
        target_hashes = {f.hash for f in report.formulas if f.role == 'target'}
        for f in report.formulas:
            if f.role in ('decoy_near', 'decoy_far') and f.hash in target_hashes:
                issues.append(f"Decoy '{f.name}' has same hash as a target!")
        
        # Check 3: No structurally identical formulas across categories
        by_normalized: Dict[str, List[str]] = {}
        for f in report.formulas:
            key = f.normalized
            if key not in by_normalized:
                by_normalized[key] = []
            by_normalized[key].append(f"{f.role}:{f.name}")
        
        for norm, formulas in by_normalized.items():
            roles = set(f.split(':')[0] for f in formulas)
            if len(roles) > 1 and 'target' in roles:
                # Allow bridge duplicates but warn about decoy/target conflicts
                non_bridge_roles = roles - {'bridge'}
                if len(non_bridge_roles) > 1:
                    issues.append(
                        f"Normalized form '{norm[:30]}...' appears in multiple categories: {formulas}"
                    )
        
        # Check 4: Near-decoy not identical normalized form to target
        target_normalized = {f.normalized for f in report.formulas if f.role == 'target'}
        for f in report.formulas:
            if f.role == 'decoy_near' and f.normalized in target_normalized:
                issues.append(f"Near-decoy '{f.name}' has identical normalized form to a target!")
        
        # Quality score: 1.0 - (issues penalty)
        quality_score = max(0.0, 1.0 - len(issues) * 0.2 - len(warnings) * 0.05)
        
        return {
            "slice_name": self.slice_name,
            "quality_score": round(quality_score, 3),
            "issues": issues,
            "warnings": warnings,
            "metrics": {
                "near_far_gap": round(report.near_far_gap, 4),
                "coverage_score": round(report.coverage_score, 4),
                "avg_near_confusability": round(report.avg_near_confusability, 4),
                "avg_far_confusability": round(report.avg_far_confusability, 4),
            },
            "passed": len(issues) == 0,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_confusability_map(
    slice_name: str,
    config_path: str = "config/curriculum_uplift_phase2.yaml",
) -> ConfusabilityMapReport:
    """
    Get confusability map for a slice.
    
    Convenience function that creates a ConfusabilityMap and generates report.
    """
    cmap = ConfusabilityMap(slice_name, config_path)
    return cmap.generate_report()


def get_all_confusability_maps(
    config_path: str = "config/curriculum_uplift_phase2.yaml",
) -> Dict[str, ConfusabilityMapReport]:
    """
    Get confusability maps for all uplift slices.
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    slices = config.get('slices', {})
    reports = {}
    
    for name, data in slices.items():
        if isinstance(data, dict) and 'formula_pool_entries' in data:
            entries = data['formula_pool_entries']
            if entries and isinstance(entries[0], dict):
                try:
                    reports[name] = get_confusability_map(name, config_path)
                except (KeyError, ValueError):
                    continue
    
    return reports

