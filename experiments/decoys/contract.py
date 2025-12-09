# PHASE II — NOT USED IN PHASE I
"""
Confusability Contract Schema & Deterministic Serialization

This module defines the canonical schema for confusability contracts and provides
deterministic, byte-stable serialization for CI reproducibility.

=============================================================================
CONFUSABILITY CONTRACT SCHEMA (CANONICAL DEFINITION) v1.1.0
=============================================================================

A Confusability Contract is a JSON document with the following structure:

```json
{
    "schema_version": "1.1.0",
    "slice_name": "<string>",
    "config_path": "<string>",
    "formulas": [
        {
            "name": "<string>",
            "role": "target" | "decoy_near" | "decoy_far" | "bridge",
            "formula": "<string>",
            "normalized": "<string>",
            "hash": "<64-char hex string>",
            "difficulty": <float 0.0-1.0>,
            "confusability": <float 0.0-1.0>,
            "components": {
                "syntactic": <float 0.0-1.0>,
                "connective": <float 0.0-1.0>,
                "atom_similarity": <float 0.0-1.0>,
                "chain_alignment": <float 0.0-1.0>
            },
            "family": "<string>"  # v1.1.0: Family assignment
        },
        ...
    ],
    "families": {  # v1.1.0: Family profiles
        "<family_name>": {
            "members": ["<formula_name>", ...],
            "avg_confusability": <float 0.0-1.0>,
            "difficulty_band": "easy" | "medium" | "hard"
        },
        ...
    },
    "summary": {
        "target_count": <int>,
        "decoy_near_count": <int>,
        "decoy_far_count": <int>,
        "bridge_count": <int>,
        "avg_confusability_near": <float 0.0-1.0>,
        "avg_confusability_far": <float 0.0-1.0>,
        "family_count": <int>  # v1.1.0
    }
}
```

SCHEMA INVARIANTS:
- All keys are sorted alphabetically at every nesting level
- Floats are rounded to exactly 6 decimal places
- No timestamp field (ensures byte-stable exports)
- Hashes are lowercase hex
- Formulas are ordered by (role_order, name) where role_order is:
  target=0, decoy_near=1, decoy_far=2, bridge=3

STABILITY GUARANTEES:
- Given the same slice input and config, export produces identical bytes
- Export is deterministic across Python versions (3.9+)
- No external state influences serialization

=============================================================================
CONFUSABILITY COMPONENTS (DOCUMENTATION)
=============================================================================

Confusability is a diagnostic metric measuring how likely a decoy formula
is to be mistaken for a target. It is computed from four components:

1. SYNTACTIC SIMILARITY (weight: 0.30)
   - Longest Common Subsequence ratio on tokenized formulas
   - Influenced by: token count, operator sequence, parenthesis structure
   - NOT influenced by: whitespace, atom naming

2. CONNECTIVE SIGNATURE (weight: 0.25)
   - Normalized Manhattan distance on operator count vectors
   - Influenced by: count of ->, /\, \/, ~ operators
   - NOT influenced by: operator ordering, nesting depth

3. ATOM SUBSTITUTION DISTANCE (weight: 0.25)
   - Jaccard similarity of atomic proposition sets
   - Influenced by: atom names used
   - NOT influenced by: atom positions, formula structure

4. IMPLICATION CHAIN ALIGNMENT (weight: 0.20)
   - Penalty for depth and balance mismatch in implication chains
   - Influenced by: formula_depth(), left/right implication balance
   - NOT influenced by: non-implication connectives

IMPORTANT ADVISORY NOTES:
- Confusability is DIAGNOSTIC ONLY - it does not influence uplift behavior
- Upward/downward drift in confusability scores is purely advisory
- Confusability is NOT a promotion rule or reward signal
- Scores should be interpreted as "structural similarity to targets"

=============================================================================
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# Schema version for contract format (v1.1.0 adds families)
SCHEMA_VERSION = "1.1.0"

# Role ordering for deterministic sorting
ROLE_ORDER = {
    "target": 0,
    "decoy_near": 1,
    "decoy_far": 2,
    "bridge": 3,
}

# Difficulty band thresholds
DIFFICULTY_BAND_THRESHOLDS = {
    "easy": 0.33,
    "medium": 0.67,
    # "hard" is anything above 0.67
}


@lru_cache(maxsize=1024)
def compute_structure_fingerprint(normalized: str) -> str:
    """
    Compute a deterministic structural fingerprint for family grouping.
    
    The fingerprint captures:
    1. Connective signature (counts of ->, /\, \/, ~)
    2. Depth signature (nesting level)
    3. Atom count (not specific atoms)
    
    This groups formulas with similar structural complexity together,
    regardless of specific atom names.
    
    Algorithm is DETERMINISTIC and STABLE across runs.
    """
    # Count connectives
    impl_count = normalized.count('->')
    conj_count = normalized.count('/\\')
    disj_count = normalized.count('\\/')
    neg_count = normalized.count('~')
    
    # Count parenthesis depth (max nesting)
    max_depth = 0
    current_depth = 0
    for c in normalized:
        if c == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif c == ')':
            current_depth -= 1
    
    # Count atoms (unique letters)
    atoms = set()
    for c in normalized:
        if c.isalpha():
            atoms.add(c)
    atom_count = len(atoms)
    
    # Create deterministic fingerprint string
    fingerprint_str = f"impl:{impl_count},conj:{conj_count},disj:{disj_count},neg:{neg_count},depth:{max_depth},atoms:{atom_count}"
    
    # Hash to short identifier
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:8]


def get_difficulty_band(avg_difficulty: float) -> str:
    """
    Classify average difficulty into a band.
    
    Returns:
        "easy" if avg < 0.33
        "medium" if 0.33 <= avg < 0.67
        "hard" if avg >= 0.67
    """
    if avg_difficulty < DIFFICULTY_BAND_THRESHOLDS["easy"]:
        return "easy"
    elif avg_difficulty < DIFFICULTY_BAND_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "hard"


@dataclass(frozen=True)
class FormulaEntry:
    """
    Single formula entry in a confusability contract.
    
    All fields are required and must be serializable.
    """
    name: str
    role: str  # "target" | "decoy_near" | "decoy_far" | "bridge"
    formula: str
    normalized: str
    hash: str  # 64-char lowercase hex
    difficulty: float  # [0.0, 1.0]
    confusability: float  # [0.0, 1.0]
    components: Dict[str, float]  # Component scores
    family: str = ""  # v1.1.0: Family fingerprint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sorted keys and rounded floats."""
        return {
            "components": {
                "atom_similarity": round(self.components.get("atom_similarity", 0.0), 6),
                "chain_alignment": round(self.components.get("chain_alignment", 0.0), 6),
                "connective": round(self.components.get("connective", 0.0), 6),
                "syntactic": round(self.components.get("syntactic", 0.0), 6),
            },
            "confusability": round(self.confusability, 6),
            "difficulty": round(self.difficulty, 6),
            "family": self.family,
            "formula": self.formula,
            "hash": self.hash.lower(),
            "name": self.name,
            "normalized": self.normalized,
            "role": self.role,
        }


@dataclass
class FamilyProfile:
    """
    Profile for a family of structurally similar formulas.
    
    v1.1.0 addition for hierarchical confusability analysis.
    """
    name: str  # Family fingerprint (8-char hex)
    members: List[str] = field(default_factory=list)  # Formula names
    avg_confusability: float = 0.0
    avg_difficulty: float = 0.0
    difficulty_band: str = "easy"  # "easy" | "medium" | "hard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sorted keys."""
        return {
            "avg_confusability": round(self.avg_confusability, 6),
            "difficulty_band": self.difficulty_band,
            "members": sorted(self.members),  # Sorted for determinism
        }


@dataclass
class ContractSummary:
    """
    Summary statistics for a confusability contract.
    """
    target_count: int = 0
    decoy_near_count: int = 0
    decoy_far_count: int = 0
    bridge_count: int = 0
    avg_confusability_near: float = 0.0
    avg_confusability_far: float = 0.0
    family_count: int = 0  # v1.1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sorted keys and rounded floats."""
        return {
            "avg_confusability_far": round(self.avg_confusability_far, 6),
            "avg_confusability_near": round(self.avg_confusability_near, 6),
            "bridge_count": self.bridge_count,
            "decoy_far_count": self.decoy_far_count,
            "decoy_near_count": self.decoy_near_count,
            "family_count": self.family_count,
            "target_count": self.target_count,
        }


@dataclass
class ConfusabilityContract:
    """
    Complete confusability contract for a slice.
    
    This is the canonical representation that can be serialized
    to JSON in a deterministic, byte-stable manner.
    
    v1.1.0 adds family profiles for hierarchical analysis.
    """
    slice_name: str
    config_path: str
    formulas: List[FormulaEntry] = field(default_factory=list)
    families: Dict[str, FamilyProfile] = field(default_factory=dict)
    summary: ContractSummary = field(default_factory=ContractSummary)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with sorted keys for deterministic serialization.
        
        Keys are alphabetically sorted at every level.
        Floats are rounded to 6 decimal places.
        No timestamp is included (ensures byte stability).
        """
        # Sort formulas by (role_order, name) for deterministic ordering
        sorted_formulas = sorted(
            self.formulas,
            key=lambda f: (ROLE_ORDER.get(f.role, 99), f.name)
        )
        
        # Sort families by name for determinism
        sorted_families = {
            k: self.families[k].to_dict()
            for k in sorted(self.families.keys())
        }
        
        return {
            "config_path": self.config_path,
            "families": sorted_families,
            "formulas": [f.to_dict() for f in sorted_formulas],
            "schema_version": SCHEMA_VERSION,
            "slice_name": self.slice_name,
            "summary": self.summary.to_dict(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serialize to deterministic JSON string.
        
        This method guarantees byte-stable output:
        - Keys sorted at all levels
        - Consistent float formatting
        - No variable elements (timestamps, etc.)
        - ASCII-safe output (no unicode escapes)
        """
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,  # Belt and suspenders: dict already sorted
            ensure_ascii=True,
            separators=(",", ": ") if indent else (",", ":"),
        )
    
    def to_bytes(self) -> bytes:
        """
        Serialize to deterministic bytes for hash comparison.
        """
        return self.to_json(indent=None).encode("utf-8")
    
    @classmethod
    def from_confusability_map(
        cls,
        cmap: "ConfusabilityMap",  # Forward reference
        config_path: str,
    ) -> "ConfusabilityContract":
        """
        Build a contract from a ConfusabilityMap report.
        
        Computes family profiles based on structural fingerprints.
        """
        report = cmap.generate_report()
        
        # First pass: compute fingerprints and build formulas
        formulas = []
        near_confs = []
        far_confs = []
        family_data: Dict[str, List[Tuple[str, float, float]]] = {}  # fingerprint -> [(name, conf, diff)]
        
        for f in report.formulas:
            # Compute family fingerprint
            fingerprint = compute_structure_fingerprint(f.normalized)
            
            entry = FormulaEntry(
                name=f.name,
                role=f.role,
                formula=f.formula,
                normalized=f.normalized,
                hash=f.hash,
                difficulty=f.difficulty,
                confusability=f.confusability,
                components=f.components,
                family=fingerprint,
            )
            formulas.append(entry)
            
            if f.role == "decoy_near":
                near_confs.append(f.confusability)
            elif f.role == "decoy_far":
                far_confs.append(f.confusability)
            
            # Track for family aggregation
            if fingerprint not in family_data:
                family_data[fingerprint] = []
            family_data[fingerprint].append((f.name, f.confusability, f.difficulty))
        
        # Build family profiles
        families: Dict[str, FamilyProfile] = {}
        for fingerprint, members in family_data.items():
            member_names = [m[0] for m in members]
            avg_conf = sum(m[1] for m in members) / len(members)
            avg_diff = sum(m[2] for m in members) / len(members)
            
            families[fingerprint] = FamilyProfile(
                name=fingerprint,
                members=member_names,
                avg_confusability=avg_conf,
                avg_difficulty=avg_diff,
                difficulty_band=get_difficulty_band(avg_diff),
            )
        
        summary = ContractSummary(
            target_count=sum(1 for f in formulas if f.role == "target"),
            decoy_near_count=sum(1 for f in formulas if f.role == "decoy_near"),
            decoy_far_count=sum(1 for f in formulas if f.role == "decoy_far"),
            bridge_count=sum(1 for f in formulas if f.role == "bridge"),
            avg_confusability_near=sum(near_confs) / len(near_confs) if near_confs else 0.0,
            avg_confusability_far=sum(far_confs) / len(far_confs) if far_confs else 0.0,
            family_count=len(families),
        )
        
        return cls(
            slice_name=report.slice_name,
            config_path=config_path,
            formulas=formulas,
            families=families,
            summary=summary,
        )


def export_contract(
    slice_name: str,
    config_path: str = "config/curriculum_uplift_phase2.yaml",
) -> ConfusabilityContract:
    """
    Export a confusability contract for a slice.
    
    This is the canonical entry point for contract generation.
    The returned contract can be serialized deterministically.
    
    Args:
        slice_name: Name of the slice to export
        config_path: Path to the curriculum YAML
        
    Returns:
        ConfusabilityContract instance
    """
    from .confusability import ConfusabilityMap
    
    cmap = ConfusabilityMap(slice_name, config_path)
    return ConfusabilityContract.from_confusability_map(cmap, config_path)


def validate_contract_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a contract dictionary against the canonical schema.
    
    Supports both v1.0.0 and v1.1.0 schemas.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Required top-level keys (v1.0.0 base)
    required_keys = {"schema_version", "slice_name", "config_path", "formulas", "summary"}
    missing = required_keys - set(data.keys())
    if missing:
        errors.append(f"Missing top-level keys: {missing}")
    
    # Validate schema version (accept 1.0.0 and 1.1.0)
    version = data.get("schema_version", "")
    if version not in ("1.0.0", "1.1.0"):
        errors.append(f"Unsupported schema version: {version}")
    
    # Validate formulas
    formulas = data.get("formulas", [])
    if not isinstance(formulas, list):
        errors.append("'formulas' must be a list")
    else:
        required_formula_keys = {
            "name", "role", "formula", "normalized", "hash",
            "difficulty", "confusability", "components"
        }
        # v1.1.0 adds "family" field
        if version == "1.1.0":
            required_formula_keys.add("family")
        
        valid_roles = {"target", "decoy_near", "decoy_far", "bridge"}
        
        for i, f in enumerate(formulas):
            if not isinstance(f, dict):
                errors.append(f"Formula {i} is not a dict")
                continue
            
            f_missing = required_formula_keys - set(f.keys())
            if f_missing:
                errors.append(f"Formula {i} missing keys: {f_missing}")
            
            if f.get("role") not in valid_roles:
                errors.append(f"Formula {i} has invalid role: {f.get('role')}")
            
            if not isinstance(f.get("difficulty"), (int, float)):
                errors.append(f"Formula {i} difficulty must be numeric")
            elif not (0.0 <= f.get("difficulty", -1) <= 1.0):
                errors.append(f"Formula {i} difficulty out of range [0, 1]")
            
            if not isinstance(f.get("confusability"), (int, float)):
                errors.append(f"Formula {i} confusability must be numeric")
            elif not (0.0 <= f.get("confusability", -1) <= 1.0):
                errors.append(f"Formula {i} confusability out of range [0, 1]")
    
    # Validate families (v1.1.0)
    if version == "1.1.0":
        families = data.get("families", {})
        if not isinstance(families, dict):
            errors.append("'families' must be a dict")
        else:
            valid_bands = {"easy", "medium", "hard"}
            for fam_name, fam_data in families.items():
                if not isinstance(fam_data, dict):
                    errors.append(f"Family '{fam_name}' is not a dict")
                    continue
                if "members" not in fam_data:
                    errors.append(f"Family '{fam_name}' missing 'members'")
                if "difficulty_band" in fam_data and fam_data["difficulty_band"] not in valid_bands:
                    errors.append(f"Family '{fam_name}' has invalid difficulty_band")
    
    # Validate summary
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        errors.append("'summary' must be a dict")
    else:
        required_summary_keys = {
            "target_count", "decoy_near_count", "decoy_far_count",
            "bridge_count", "avg_confusability_near", "avg_confusability_far"
        }
        # v1.1.0 adds family_count
        if version == "1.1.0":
            required_summary_keys.add("family_count")
        
        s_missing = required_summary_keys - set(summary.keys())
        if s_missing:
            errors.append(f"Summary missing keys: {s_missing}")
    
    return (len(errors) == 0, errors)


def contracts_are_equal(a: ConfusabilityContract, b: ConfusabilityContract) -> bool:
    """
    Check if two contracts are semantically equal.
    
    Uses byte-stable serialization for comparison.
    """
    return a.to_bytes() == b.to_bytes()

