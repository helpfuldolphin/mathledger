# PHASE II — NOT USED IN PHASE I
"""
Decoy Difficulty Scoring Engine

Automated scoring of decoy formulas across four dimensions:
1. Syntactic Proximity (0-1): Token count and depth difference similarity
2. Atom Set Overlap (0-1): Jaccard similarity of atomic propositions
3. Structural Similarity (0-1): Connective patterns and implication chains
4. Semantic Confusability (0-1): Hash prefix distance clustering

The composite "difficulty" score indicates how hard a decoy is to distinguish
from its corresponding target(s). Higher = more confusable = harder decoy.

Usage:
    from experiments.decoys.scoring import DecoyScorer, score_slice_decoys

    scorer = DecoyScorer()
    scores = score_slice_decoys("slice_uplift_goal")
    for s in scores:
        print(f"{s.name}: difficulty={s.difficulty:.2f}")

All computations are deterministic and suitable for CI integration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from normalization.canon import normalize, get_atomic_propositions
from derivation.structure import formula_depth, atom_frozenset, is_implication, implication_parts
from backend.crypto.hashing import hash_statement


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class DecoyScore:
    """
    Immutable score result for a single formula.
    
    Attributes:
        name: Formula identifier from YAML
        formula: Original formula string
        normalized: Canonicalized formula
        hash: SHA256 hash via canonical pipeline
        role: Formula role (target, decoy_near, decoy_far, bridge)
        scores: Dictionary of individual dimension scores
        difficulty: Composite difficulty score (0-1)
    """
    name: str
    formula: str
    normalized: str
    hash: str
    role: str
    scores: Dict[str, float] = field(default_factory=dict)
    difficulty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "formula": self.formula,
            "normalized": self.normalized,
            "hash": self.hash,
            "role": self.role,
            "scores": dict(self.scores),
            "difficulty": round(self.difficulty, 4),
        }


@dataclass
class SliceScoreReport:
    """
    Complete scoring report for a slice.
    
    Attributes:
        slice_name: Name of the slice
        targets: List of target formula scores
        decoys_near: List of near-decoy scores
        decoys_far: List of far-decoy scores
        bridges: List of bridge formula scores
        avg_near_difficulty: Average difficulty of near-decoys
        avg_far_difficulty: Average difficulty of far-decoys
        confusability_index: Overall slice confusability metric
    """
    slice_name: str
    targets: List[DecoyScore] = field(default_factory=list)
    decoys_near: List[DecoyScore] = field(default_factory=list)
    decoys_far: List[DecoyScore] = field(default_factory=list)
    bridges: List[DecoyScore] = field(default_factory=list)
    avg_near_difficulty: float = 0.0
    avg_far_difficulty: float = 0.0
    confusability_index: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "slice_name": self.slice_name,
            "targets": [t.to_dict() for t in self.targets],
            "decoys_near": [d.to_dict() for d in self.decoys_near],
            "decoys_far": [d.to_dict() for d in self.decoys_far],
            "bridges": [b.to_dict() for b in self.bridges],
            "avg_near_difficulty": round(self.avg_near_difficulty, 4),
            "avg_far_difficulty": round(self.avg_far_difficulty, 4),
            "confusability_index": round(self.confusability_index, 4),
        }


# =============================================================================
# SCORING PRIMITIVES
# =============================================================================

@lru_cache(maxsize=4096)
def _count_tokens(normalized: str) -> int:
    """
    Count logical tokens in a normalized formula.
    
    Tokens are: atoms, operators (->  /\  \/  ~), parentheses.
    """
    # Atoms: sequences of letters
    atoms = re.findall(r'[A-Za-z]+', normalized)
    # Operators: ->, /\, \/, ~
    implications = normalized.count('->')
    conjunctions = normalized.count('/\\')
    disjunctions = normalized.count('\\/')
    negations = normalized.count('~')
    # Parentheses
    parens = normalized.count('(') + normalized.count(')')
    
    return len(atoms) + implications + conjunctions + disjunctions + negations + parens


@lru_cache(maxsize=4096)
def _get_connective_signature(normalized: str) -> Tuple[int, int, int, int]:
    """
    Extract connective counts as a signature tuple.
    
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
def _count_implication_chain_depth(normalized: str) -> int:
    """
    Count the maximum implication nesting depth.
    
    For "a->b->c" this is 2. For "(a->b)->c" this is also 2.
    """
    if not is_implication(normalized):
        return 0
    
    ante, cons = implication_parts(normalized)
    if ante is None or cons is None:
        return 0
    
    return 1 + max(
        _count_implication_chain_depth(ante),
        _count_implication_chain_depth(cons)
    )


def _jaccard_similarity(set_a: frozenset, set_b: frozenset) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0  # Both empty = identical
    union = set_a | set_b
    if not union:
        return 1.0
    intersection = set_a & set_b
    return len(intersection) / len(union)


def _normalized_difference(a: float, b: float, max_diff: float = 10.0) -> float:
    """
    Compute similarity based on absolute difference.
    
    Returns 1.0 for identical values, approaching 0.0 as difference grows.
    """
    diff = abs(a - b)
    return max(0.0, 1.0 - diff / max_diff)


def _hash_prefix_similarity(hash_a: str, hash_b: str, prefix_len: int = 8) -> float:
    """
    Compute similarity based on common hash prefix length.
    
    This is a proxy for semantic clustering - formulas with similar
    structure tend to have hashes that share longer prefixes.
    
    Note: This is NOT cryptographically meaningful, but serves as
    a heuristic for structural similarity in the hash space.
    """
    if not hash_a or not hash_b:
        return 0.0
    
    # Count matching hex characters from start
    common = 0
    for i in range(min(len(hash_a), len(hash_b), prefix_len)):
        if hash_a[i] == hash_b[i]:
            common += 1
        else:
            break
    
    return common / prefix_len


# =============================================================================
# SCORING DIMENSIONS
# =============================================================================

def compute_syntactic_proximity(
    decoy_normalized: str,
    target_normalized: str,
) -> float:
    """
    Compute syntactic proximity score (0-1).
    
    Dimensions:
    - Token count similarity (40%)
    - Depth similarity (40%)
    - Length similarity (20%)
    
    Higher = more similar.
    """
    decoy_tokens = _count_tokens(decoy_normalized)
    target_tokens = _count_tokens(target_normalized)
    token_sim = _normalized_difference(decoy_tokens, target_tokens, max_diff=20)
    
    decoy_depth = formula_depth(decoy_normalized)
    target_depth = formula_depth(target_normalized)
    depth_sim = _normalized_difference(decoy_depth, target_depth, max_diff=5)
    
    len_sim = _normalized_difference(len(decoy_normalized), len(target_normalized), max_diff=50)
    
    return 0.4 * token_sim + 0.4 * depth_sim + 0.2 * len_sim


def compute_atom_overlap(
    decoy_normalized: str,
    target_normalized: str,
) -> float:
    """
    Compute atom set overlap score (0-1).
    
    Uses Jaccard similarity between the sets of atomic propositions.
    """
    decoy_atoms = atom_frozenset(decoy_normalized)
    target_atoms = atom_frozenset(target_normalized)
    return _jaccard_similarity(decoy_atoms, target_atoms)


def compute_structural_similarity(
    decoy_normalized: str,
    target_normalized: str,
) -> float:
    """
    Compute structural similarity score (0-1).
    
    Dimensions:
    - Connective signature similarity (50%)
    - Implication chain depth similarity (30%)
    - Top-level operator match (20%)
    """
    decoy_sig = _get_connective_signature(decoy_normalized)
    target_sig = _get_connective_signature(target_normalized)
    
    # Connective signature similarity (normalized vector distance)
    sig_diffs = [abs(d - t) for d, t in zip(decoy_sig, target_sig)]
    max_total_diff = 20  # Reasonable upper bound
    total_diff = min(sum(sig_diffs), max_total_diff)
    sig_sim = 1.0 - total_diff / max_total_diff
    
    # Implication chain depth
    decoy_chain = _count_implication_chain_depth(decoy_normalized)
    target_chain = _count_implication_chain_depth(target_normalized)
    chain_sim = _normalized_difference(decoy_chain, target_chain, max_diff=5)
    
    # Top-level operator match
    decoy_is_imp = is_implication(decoy_normalized)
    target_is_imp = is_implication(target_normalized)
    top_match = 1.0 if decoy_is_imp == target_is_imp else 0.5
    
    return 0.5 * sig_sim + 0.3 * chain_sim + 0.2 * top_match


def compute_semantic_confusability(
    decoy_hash: str,
    target_hashes: List[str],
) -> float:
    """
    Compute semantic confusability score (0-1).
    
    Based on hash prefix similarity to nearest target.
    Higher = more "accidentally close" to a target in hash space.
    
    Note: This is a heuristic proxy, not a true semantic measure.
    Real semantic confusability would require theorem proving.
    """
    if not target_hashes:
        return 0.0
    
    # Find maximum similarity to any target
    max_sim = 0.0
    for target_hash in target_hashes:
        sim = _hash_prefix_similarity(decoy_hash, target_hash, prefix_len=8)
        max_sim = max(max_sim, sim)
    
    # Also factor in hash character distribution similarity
    # (formulas with similar structure have similar hash distributions)
    decoy_chars = set(decoy_hash[:16])
    avg_char_overlap = 0.0
    for target_hash in target_hashes:
        target_chars = set(target_hash[:16])
        overlap = len(decoy_chars & target_chars) / 16
        avg_char_overlap = max(avg_char_overlap, overlap)
    
    return 0.6 * max_sim + 0.4 * avg_char_overlap


# =============================================================================
# COMPOSITE SCORING
# =============================================================================

def compute_difficulty(scores: Dict[str, float], role: str) -> float:
    """
    Compute composite difficulty score from dimension scores.
    
    Weights are role-dependent:
    - decoy_near: Structure and syntax matter most
    - decoy_far: Semantic confusability matters more
    - bridge: All dimensions weighted equally
    
    Returns:
        Composite score in [0, 1]
    """
    if role == "decoy_near":
        # Near-decoys should be structurally similar
        weights = {
            "syntactic": 0.30,
            "atom_overlap": 0.25,
            "structure": 0.30,
            "semantic": 0.15,
        }
    elif role == "decoy_far":
        # Far-decoys should be semantically distinct but confusing
        weights = {
            "syntactic": 0.20,
            "atom_overlap": 0.15,
            "structure": 0.25,
            "semantic": 0.40,
        }
    else:
        # Bridges and others: equal weights
        weights = {
            "syntactic": 0.25,
            "atom_overlap": 0.25,
            "structure": 0.25,
            "semantic": 0.25,
        }
    
    total = sum(weights[k] * scores.get(k, 0.0) for k in weights)
    return min(1.0, max(0.0, total))


# =============================================================================
# PUBLIC API
# =============================================================================

class DecoyScorer:
    """
    Stateless scorer for decoy formulas.
    
    Computes difficulty scores relative to a set of target formulas.
    All methods are deterministic and cacheable.
    """
    
    def __init__(self):
        """Initialize the scorer."""
        self._cache: Dict[Tuple[str, frozenset], DecoyScore] = {}
    
    def score_formula(
        self,
        name: str,
        formula: str,
        role: str,
        target_formulas: List[str],
    ) -> DecoyScore:
        """
        Score a single formula against target formulas.
        
        Args:
            name: Formula identifier
            formula: Original formula string
            role: Formula role (target, decoy_near, decoy_far, bridge)
            target_formulas: List of target formula strings for comparison
            
        Returns:
            DecoyScore with all dimension scores and composite difficulty
        """
        normalized = normalize(formula)
        formula_hash = hash_statement(formula)
        
        # Normalize targets
        target_normalized = [normalize(t) for t in target_formulas]
        target_hashes = [hash_statement(t) for t in target_formulas]
        
        # For targets, score against themselves (baseline)
        if role == "target":
            scores = {
                "syntactic": 1.0,
                "atom_overlap": 1.0,
                "structure": 1.0,
                "semantic": 1.0,
            }
            difficulty = 1.0
        else:
            # Score against best-matching target
            best_syntactic = 0.0
            best_atom = 0.0
            best_structure = 0.0
            
            for t_norm in target_normalized:
                best_syntactic = max(best_syntactic, compute_syntactic_proximity(normalized, t_norm))
                best_atom = max(best_atom, compute_atom_overlap(normalized, t_norm))
                best_structure = max(best_structure, compute_structural_similarity(normalized, t_norm))
            
            semantic = compute_semantic_confusability(formula_hash, target_hashes)
            
            scores = {
                "syntactic": round(best_syntactic, 4),
                "atom_overlap": round(best_atom, 4),
                "structure": round(best_structure, 4),
                "semantic": round(semantic, 4),
            }
            difficulty = compute_difficulty(scores, role)
        
        return DecoyScore(
            name=name,
            formula=formula,
            normalized=normalized,
            hash=formula_hash,
            role=role,
            scores=scores,
            difficulty=round(difficulty, 4),
        )


def score_formula(
    name: str,
    formula: str,
    role: str,
    target_formulas: List[str],
) -> DecoyScore:
    """
    Convenience function to score a single formula.
    
    See DecoyScorer.score_formula for details.
    """
    scorer = DecoyScorer()
    return scorer.score_formula(name, formula, role, target_formulas)


def _normalize_entries(entries: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize formula pool entries to dictionary format.
    
    Handles both formats:
    - Legacy: List of formula strings ["p->q", "q->r"]
    - Decoy-enabled: List of dicts [{"name": "K", "role": "target", "formula": "p->q"}]
    
    For legacy format, all entries are treated as targets.
    """
    if not entries:
        return []
    
    if isinstance(entries[0], str):
        # Legacy format: treat all as targets
        return [
            {
                "name": f"formula_{i}",
                "formula": formula,
                "role": "target",
            }
            for i, formula in enumerate(entries)
        ]
    else:
        # Decoy-enabled format
        return entries


def score_slice_decoys(
    slice_name: str,
    config_path: str = "config/curriculum_uplift_phase2.yaml",
) -> SliceScoreReport:
    """
    Score all formulas in a slice and generate a report.
    
    Supports both legacy (string list) and decoy-enabled (dict list) formats.
    For legacy format, all formulas are treated as targets with no decoys.
    
    Args:
        slice_name: Name of the slice to score
        config_path: Path to the curriculum YAML file
        
    Returns:
        SliceScoreReport with all formula scores and summary metrics
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    slices = config.get('slices', {})
    if slice_name not in slices:
        raise KeyError(f"Slice '{slice_name}' not found in curriculum")
    
    slice_data = slices[slice_name]
    if not isinstance(slice_data, dict) or 'formula_pool_entries' not in slice_data:
        raise ValueError(f"Slice '{slice_name}' has no formula_pool_entries")
    
    raw_entries = slice_data['formula_pool_entries']
    entries = _normalize_entries(raw_entries)
    
    if not entries:
        raise ValueError(f"Slice '{slice_name}' has empty formula_pool_entries")
    
    # Extract targets
    target_entries = [e for e in entries if e.get('role') == 'target']
    target_formulas = [e['formula'] for e in target_entries]
    
    if not target_formulas:
        raise ValueError(f"Slice '{slice_name}' has no target formulas")
    
    # Score all formulas
    scorer = DecoyScorer()
    report = SliceScoreReport(slice_name=slice_name)
    
    for entry in entries:
        name = entry.get('name', 'unknown')
        formula = entry.get('formula', '')
        role = entry.get('role', 'unknown')
        
        score = scorer.score_formula(name, formula, role, target_formulas)
        
        if role == 'target':
            report.targets.append(score)
        elif role == 'decoy_near':
            report.decoys_near.append(score)
        elif role == 'decoy_far':
            report.decoys_far.append(score)
        elif role == 'bridge':
            report.bridges.append(score)
    
    # Compute summary metrics
    if report.decoys_near:
        report.avg_near_difficulty = sum(d.difficulty for d in report.decoys_near) / len(report.decoys_near)
    
    if report.decoys_far:
        report.avg_far_difficulty = sum(d.difficulty for d in report.decoys_far) / len(report.decoys_far)
    
    # Confusability index: weighted average of decoy difficulties
    # Near-decoys contribute more to confusability
    all_decoys = report.decoys_near + report.decoys_far
    if all_decoys:
        total_weight = 0.0
        weighted_sum = 0.0
        for d in report.decoys_near:
            weighted_sum += d.difficulty * 2.0  # Near-decoys weighted 2x
            total_weight += 2.0
        for d in report.decoys_far:
            weighted_sum += d.difficulty * 1.0
            total_weight += 1.0
        report.confusability_index = weighted_sum / total_weight
    
    return report


def compute_confusability_index(
    slice_name: str,
    config_path: str = "config/curriculum_uplift_phase2.yaml",
) -> float:
    """
    Compute the overall confusability index for a slice.
    
    This is a single scalar metric suitable for CI thresholds.
    
    Returns:
        Confusability index in [0, 1]. Higher = more confusing decoys.
    """
    report = score_slice_decoys(slice_name, config_path)
    return report.confusability_index


def score_all_slices(
    config_path: str = "config/curriculum_uplift_phase2.yaml",
) -> Dict[str, SliceScoreReport]:
    """
    Score all uplift slices in the curriculum.
    
    Returns:
        Dictionary mapping slice names to their score reports.
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    slices = config.get('slices', {})
    
    reports = {}
    for name, data in slices.items():
        if isinstance(data, dict) and 'formula_pool_entries' in data:
            try:
                reports[name] = score_slice_decoys(name, config_path)
            except (KeyError, ValueError):
                continue
    
    return reports


# =============================================================================
# REPORTING UTILITIES
# =============================================================================

def generate_markdown_report(reports: Dict[str, SliceScoreReport]) -> str:
    """
    Generate a Markdown report from slice score reports.
    
    Args:
        reports: Dictionary of slice name -> SliceScoreReport
        
    Returns:
        Markdown-formatted report string
    """
    lines = [
        "# Decoy Difficulty Scoring Report",
        "",
        "## Summary",
        "",
        "| Slice | Targets | Near Decoys | Far Decoys | Avg Near Diff | Avg Far Diff | Confusability |",
        "|-------|---------|-------------|------------|---------------|--------------|---------------|",
    ]
    
    for name, report in sorted(reports.items()):
        lines.append(
            f"| {name} | {len(report.targets)} | {len(report.decoys_near)} | "
            f"{len(report.decoys_far)} | {report.avg_near_difficulty:.3f} | "
            f"{report.avg_far_difficulty:.3f} | {report.confusability_index:.3f} |"
        )
    
    lines.extend([
        "",
        "## Detailed Scores",
        "",
    ])
    
    for name, report in sorted(reports.items()):
        lines.extend([
            f"### {name}",
            "",
            "#### Near Decoys",
            "",
            "| Name | Syntactic | Atom Overlap | Structure | Semantic | Difficulty |",
            "|------|-----------|--------------|-----------|----------|------------|",
        ])
        
        for d in sorted(report.decoys_near, key=lambda x: -x.difficulty):
            lines.append(
                f"| {d.name} | {d.scores.get('syntactic', 0):.3f} | "
                f"{d.scores.get('atom_overlap', 0):.3f} | {d.scores.get('structure', 0):.3f} | "
                f"{d.scores.get('semantic', 0):.3f} | **{d.difficulty:.3f}** |"
            )
        
        lines.extend([
            "",
            "#### Far Decoys",
            "",
            "| Name | Syntactic | Atom Overlap | Structure | Semantic | Difficulty |",
            "|------|-----------|--------------|-----------|----------|------------|",
        ])
        
        for d in sorted(report.decoys_far, key=lambda x: -x.difficulty):
            lines.append(
                f"| {d.name} | {d.scores.get('syntactic', 0):.3f} | "
                f"{d.scores.get('atom_overlap', 0):.3f} | {d.scores.get('structure', 0):.3f} | "
                f"{d.scores.get('semantic', 0):.3f} | **{d.difficulty:.3f}** |"
            )
        
        lines.append("")
    
    return "\n".join(lines)


def generate_json_report(reports: Dict[str, SliceScoreReport]) -> Dict[str, Any]:
    """
    Generate a JSON-serializable report from slice score reports.
    
    Args:
        reports: Dictionary of slice name -> SliceScoreReport
        
    Returns:
        JSON-serializable dictionary
    """
    return {
        "version": "1.0",
        "phase": "II",
        "slices": {name: report.to_dict() for name, report in reports.items()},
        "summary": {
            "total_slices": len(reports),
            "total_decoys": sum(
                len(r.decoys_near) + len(r.decoys_far) for r in reports.values()
            ),
            "avg_confusability": (
                sum(r.confusability_index for r in reports.values()) / len(reports)
                if reports else 0.0
            ),
        },
    }

