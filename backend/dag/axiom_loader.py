# backend/dag/axiom_loader.py
"""
PHASE II - Axiom Registry Loader.

This module provides a function to load the canonical set of axiom hashes
from the global_axiom_registry.yaml file.
"""
from pathlib import Path
from typing import Set
import yaml

# Announce compliance on import
print("PHASE II — NOT USED IN PHASE I: Loading Axiom Registry Loader.", file=__import__("sys").stderr)

class AxiomLoaderError(Exception):
    """Custom exception for axiom loading failures."""
    pass

def load_axiom_registry(registry_path: Path) -> Set[str]:
    """
    Loads the axiom registry from the specified YAML file.

    Args:
        registry_path: The path to the global_axiom_registry.yaml file.

    Returns:
        A set of canonical axiom hashes.
    
    Raises:
        AxiomLoaderError: If the file is not found or is malformed.
    """
    if not registry_path.exists():
        raise AxiomLoaderError(f"Axiom registry file not found at: {registry_path}")

    try:
        with open(registry_path, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise AxiomLoaderError(f"Error parsing YAML file: {registry_path}") from e

    if not isinstance(data, dict) or "axioms" not in data:
        raise AxiomLoaderError(f"Malformed registry file: Missing top-level 'axioms' key in {registry_path}")

    axiom_list = data["axioms"]
    if not isinstance(axiom_list, list):
        raise AxiomLoaderError(f"Malformed registry file: 'axioms' key must contain a list in {registry_path}")

    axiom_hashes: Set[str] = set()
    for i, item in enumerate(axiom_list):
        if not isinstance(item, dict) or "hash" not in item:
            raise AxiomLoaderError(f"Malformed axiom entry at index {i} in {registry_path}: missing 'hash' key.")
        axiom_hashes.add(item["hash"])
        
    return axiom_hashes

