"""
RFL Feature Extraction

Extract structural features from formulas for policy-guided derivation.
Features are deterministic and based only on formula structure.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class FeatureVector:
    """
    Feature vector for a derivation candidate.
    
    Attributes:
        features: Dictionary mapping feature names to values
    """
    features: Dict[str, float]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return self.features


def extract_features(formula: str, context: Dict[str, Any] = None) -> FeatureVector:
    """
    Extract structural features from a formula.
    
    Features extracted:
    - tree_depth: Nesting depth of the formula
    - num_atoms: Number of atomic propositions
    - num_connectives: Number of logical connectives
    - num_implications: Number of implication operators
    - num_conjunctions: Number of AND operators
    - num_disjunctions: Number of OR operators
    - num_negations: Number of NOT operators
    - formula_length: Character length of formula
    
    Args:
        formula: String representation of formula
        context: Optional context information (unused in baseline)
        
    Returns:
        FeatureVector with extracted features
    """
    features = {}
    
    # Basic length feature
    features["formula_length"] = float(len(formula))
    
    # Count atoms (lowercase letters, simplified heuristic)
    num_atoms = sum(1 for c in formula if c.islower())
    features["num_atoms"] = float(num_atoms)
    
    # Count connectives
    features["num_implications"] = float(formula.count("->"))
    features["num_conjunctions"] = float(formula.count("&") + formula.count("∧"))
    features["num_disjunctions"] = float(formula.count("|") + formula.count("∨"))
    features["num_negations"] = float(formula.count("~") + formula.count("¬"))
    
    # Total connectives
    features["num_connectives"] = (
        features["num_implications"] +
        features["num_conjunctions"] +
        features["num_disjunctions"] +
        features["num_negations"]
    )
    
    # Tree depth (estimate from parentheses nesting)
    max_depth = 0
    current_depth = 0
    for c in formula:
        if c == "(":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif c == ")":
            current_depth -= 1
    features["tree_depth"] = float(max_depth)
    
    # Bias feature (always 1.0)
    features["bias"] = 1.0
    
    return FeatureVector(features=features)


def get_feature_names() -> list[str]:
    """
    Get list of all feature names.
    
    Returns:
        List of feature names
    """
    return [
        "formula_length",
        "num_atoms",
        "num_implications",
        "num_conjunctions",
        "num_disjunctions",
        "num_negations",
        "num_connectives",
        "tree_depth",
        "bias",
    ]
