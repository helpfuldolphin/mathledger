"""
PHASE II — NOT USED IN PHASE I

Curriculum Loader V2
====================

Deterministic curriculum loading for Phase II U2 uplift experiments.
Provides stable, reproducible item ordering for baseline and RFL modes.

Key Features:
- Loads curriculum from YAML/JSON/JSONL sources
- Deterministic ordering per slice (seed-independent ordering)
- Clear error handling for missing/malformed files
- Simple interface: load_for_slice(slice_name) -> List[CurriculumItem]

Reference:
- config/curriculum_uplift_phase2.yaml — canonical slice definitions
- experiments/run_uplift_u2.py — integration point
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CurriculumItem:
    """
    A single item in the curriculum.
    
    Attributes:
        formula: The formula string (e.g., "p->q")
        hash: Optional hash identifier for the formula
        metadata: Additional metadata (role, complexity, etc.)
    """
    formula: str
    hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __repr__(self) -> str:
        return f"CurriculumItem(formula='{self.formula}', hash={self.hash})"


class CurriculumLoaderError(Exception):
    """Base exception for curriculum loading errors."""
    pass


class CurriculumNotFoundError(CurriculumLoaderError):
    """Raised when curriculum file or slice is not found."""
    pass


class CurriculumFormatError(CurriculumLoaderError):
    """Raised when curriculum file is malformed."""
    pass


class CurriculumLoader:
    """
    Deterministic curriculum loader for Phase II experiments.
    
    Loads curriculum from a well-specified source (YAML/JSON/JSONL)
    and provides deterministic item ordering per slice.
    
    Example:
        >>> loader = CurriculumLoader("config/curriculum_uplift_phase2.yaml")
        >>> items = loader.load_for_slice("slice_uplift_goal")
        >>> print(f"Loaded {len(items)} items")
    """
    
    def __init__(self, config_path: str | Path):
        """
        Initialize the curriculum loader.
        
        Args:
            config_path: Path to curriculum config file (YAML/JSON/JSONL)
            
        Raises:
            CurriculumNotFoundError: If config file doesn't exist
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise CurriculumNotFoundError(
                f"Curriculum config file not found: {self.config_path}"
            )
        
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and parse the curriculum config file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                suffix = self.config_path.suffix.lower()
                
                if suffix in ['.yaml', '.yml']:
                    self._config = yaml.safe_load(f)
                elif suffix == '.json':
                    self._config = json.load(f)
                elif suffix == '.jsonl':
                    # JSONL: one JSON object per line
                    lines = [json.loads(line) for line in f if line.strip()]
                    # Assume first line is config if multiple, or single object
                    self._config = lines[0] if len(lines) == 1 else {"slices": lines}
                else:
                    raise CurriculumFormatError(
                        f"Unsupported config format: {suffix}. "
                        "Supported: .yaml, .yml, .json, .jsonl"
                    )
                    
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise CurriculumFormatError(
                f"Failed to parse curriculum config: {e}"
            )
        except Exception as e:
            raise CurriculumLoaderError(
                f"Unexpected error loading curriculum: {e}"
            )
        
        # Validate basic structure
        if not isinstance(self._config, dict):
            raise CurriculumFormatError(
                "Curriculum config must be a dictionary at top level"
            )
        
        if "slices" not in self._config:
            raise CurriculumFormatError(
                "Curriculum config must have 'slices' key"
            )
    
    def load_for_slice(self, slice_name: str) -> List[CurriculumItem]:
        """
        Load curriculum items for a specific slice with deterministic ordering.
        
        Args:
            slice_name: Name of the slice (e.g., "slice_uplift_goal")
            
        Returns:
            List of CurriculumItem objects in deterministic order
            
        Raises:
            CurriculumNotFoundError: If slice not found in config
            CurriculumFormatError: If slice config is malformed
        """
        if self._config is None:
            raise CurriculumLoaderError("Config not loaded")
        
        slices = self._config.get("slices", {})
        
        # Handle both dict and list formats
        slice_config = None
        if isinstance(slices, dict):
            slice_config = slices.get(slice_name)
        elif isinstance(slices, list):
            for item in slices:
                if isinstance(item, dict) and item.get("name") == slice_name:
                    slice_config = item
                    break
        
        if slice_config is None:
            available = self._get_available_slices()
            raise CurriculumNotFoundError(
                f"Slice '{slice_name}' not found in curriculum config. "
                f"Available slices: {', '.join(available)}"
            )
        
        # Extract formula pool entries
        formula_entries = slice_config.get("formula_pool_entries", [])
        if not formula_entries:
            # Fallback to parameters.items if available (alternative structure)
            params = slice_config.get("parameters", {})
            formula_entries = params.get("items", [])
        
        if not formula_entries:
            raise CurriculumFormatError(
                f"Slice '{slice_name}' has no formula_pool_entries or items"
            )
        
        # Convert to CurriculumItem objects
        # Maintain deterministic ordering from config file
        items: List[CurriculumItem] = []
        for idx, entry in enumerate(formula_entries):
            if isinstance(entry, str):
                # Simple string format
                items.append(CurriculumItem(formula=entry, metadata={"index": idx}))
            elif isinstance(entry, dict):
                # Dictionary format with metadata
                formula = entry.get("formula") or entry.get("item")
                if not formula:
                    raise CurriculumFormatError(
                        f"Formula entry at index {idx} missing 'formula' or 'item' key"
                    )
                items.append(CurriculumItem(
                    formula=formula,
                    hash=entry.get("hash"),
                    metadata={**entry, "index": idx}
                ))
            else:
                raise CurriculumFormatError(
                    f"Invalid formula entry type at index {idx}: {type(entry)}"
                )
        
        return items
    
    def _get_available_slices(self) -> List[str]:
        """Get list of available slice names in the config."""
        if self._config is None:
            return []
        
        slices = self._config.get("slices", {})
        if isinstance(slices, dict):
            return sorted(slices.keys())
        elif isinstance(slices, list):
            return sorted([
                item.get("name", f"unnamed_{i}")
                for i, item in enumerate(slices)
                if isinstance(item, dict)
            ])
        return []
    
    def get_slice_config(self, slice_name: str) -> Dict[str, Any]:
        """
        Get the full configuration for a slice.
        
        Args:
            slice_name: Name of the slice
            
        Returns:
            Dictionary containing the slice configuration
            
        Raises:
            CurriculumNotFoundError: If slice not found
        """
        if self._config is None:
            raise CurriculumLoaderError("Config not loaded")
        
        slices = self._config.get("slices", {})
        
        if isinstance(slices, dict):
            config = slices.get(slice_name)
        elif isinstance(slices, list):
            config = None
            for item in slices:
                if isinstance(item, dict) and item.get("name") == slice_name:
                    config = item
                    break
        else:
            config = None
        
        if config is None:
            available = self._get_available_slices()
            raise CurriculumNotFoundError(
                f"Slice '{slice_name}' not found. Available: {', '.join(available)}"
            )
        
        return config
    
    def list_slices(self) -> List[str]:
        """
        List all available slice names in the curriculum.
        
        Returns:
            Sorted list of slice names
        """
        return self._get_available_slices()


def load_curriculum_for_slice(
    config_path: str | Path,
    slice_name: str
) -> List[CurriculumItem]:
    """
    Convenience function to load curriculum items for a slice.
    
    Args:
        config_path: Path to curriculum config file
        slice_name: Name of the slice to load
        
    Returns:
        List of CurriculumItem objects
        
    Raises:
        CurriculumLoaderError: If loading fails
    """
    loader = CurriculumLoader(config_path)
    return loader.load_for_slice(slice_name)
