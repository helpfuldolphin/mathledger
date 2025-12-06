# PHASE II — NOT USED IN PHASE I
"""
RFL Feature Extraction
======================

Feature extraction for policy scoring.

Determinism Contract:
    - All feature extraction is deterministic
    - No external entropy or wall-clock time
    - Fixed key ordering for all dictionaries
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FeatureVector:
    """
    Immutable feature vector for policy scoring.
    
    Attributes:
        length: Formula text length
        depth: AST depth (if available)
        atom_count: Number of atomic propositions
        connective_count: Number of logical connectives
        success_rate: Historical success rate for this formula hash
        raw: Raw feature dictionary for extensibility
    """
    length: int = 0
    depth: int = 0
    atom_count: int = 0
    connective_count: int = 0
    success_rate: float = 0.0
    raw: Dict[str, float] = field(default_factory=dict)

    def to_array(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert to numpy array for scoring.
        
        Args:
            feature_names: Optional list of feature names to include.
                          If None, uses default features.
        
        Returns:
            1D numpy array of feature values.
        """
        if feature_names is None:
            feature_names = ["length", "depth", "atom_count", "connective_count", "success_rate"]
        
        values = []
        for name in feature_names:
            if hasattr(self, name):
                values.append(float(getattr(self, name)))
            elif name in self.raw:
                values.append(float(self.raw[name]))
            else:
                values.append(0.0)
        
        return np.array(values, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "length": self.length,
            "depth": self.depth,
            "atom_count": self.atom_count,
            "connective_count": self.connective_count,
            "success_rate": self.success_rate,
            "raw": dict(sorted(self.raw.items())),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureVector":
        """Create from dictionary."""
        return cls(
            length=d.get("length", 0),
            depth=d.get("depth", 0),
            atom_count=d.get("atom_count", 0),
            connective_count=d.get("connective_count", 0),
            success_rate=d.get("success_rate", 0.0),
            raw=d.get("raw", {}),
        )


# Slice-specific feature masks define which features are active per slice
SLICE_FEATURE_MASKS: Dict[str, List[str]] = {
    "arithmetic_simple": ["length", "depth"],
    "algebra_expansion": ["length", "depth", "connective_count"],
    "slice_uplift_goal": ["length", "depth", "atom_count", "connective_count", "success_rate"],
    "slice_uplift_proto": ["length", "depth", "success_rate"],
    "default": ["length", "depth", "atom_count", "connective_count", "success_rate"],
}


def _count_atoms(text: str) -> int:
    """
    Count atomic propositions in formula text.
    
    Simple heuristic: count unique lowercase letters that appear in the text.
    This is a basic approximation suitable for simple propositional formulas
    with single-letter atom names (p, q, r, etc.).
    
    Note: This does not handle multi-character atom names like 'p1' or 'atom_x'.
    For more complex formulas, a proper parser should be used.
    """
    atoms = set()
    for char in text:
        if char.isalpha() and char.islower():
            atoms.add(char)
    return len(atoms)


def _count_connectives(text: str) -> int:
    """
    Count logical connectives in formula text.
    
    Connectives: ->, <->, &, |, ~, !
    """
    connectives = ["->", "<->", "&", "|", "~", "!"]
    count = 0
    for conn in connectives:
        count += text.count(conn)
    return count


def _estimate_depth(text: str) -> int:
    """
    Estimate AST depth from parenthesis nesting.
    
    Simple heuristic: max nesting depth of parentheses.
    """
    depth = 0
    max_depth = 0
    for char in text:
        if char == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ')':
            depth = max(0, depth - 1)
    return max_depth if max_depth > 0 else 1


def extract_features(
    text: str,
    success_history: Optional[Dict[str, float]] = None,
    formula_hash: Optional[str] = None,
) -> FeatureVector:
    """
    Extract features from formula text.
    
    Args:
        text: Formula text
        success_history: Optional dict mapping formula_hash to success rate
        formula_hash: Optional hash of the formula for history lookup
    
    Returns:
        FeatureVector with extracted features
    
    Determinism:
        This function is fully deterministic given the same inputs.
    """
    length = len(text)
    depth = _estimate_depth(text)
    atom_count = _count_atoms(text)
    connective_count = _count_connectives(text)
    
    success_rate = 0.0
    if success_history and formula_hash:
        success_rate = success_history.get(formula_hash, 0.0)
    
    return FeatureVector(
        length=length,
        depth=depth,
        atom_count=atom_count,
        connective_count=connective_count,
        success_rate=success_rate,
        raw={
            "length": float(length),
            "depth": float(depth),
            "atom_count": float(atom_count),
            "connective_count": float(connective_count),
            "success_rate": success_rate,
        },
    )
