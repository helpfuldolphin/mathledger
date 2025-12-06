"""
PHASE II — NOT USED IN PHASE I

Verbose Formatter for U2 Experiments
=====================================

Provides configurable verbose cycle logging with field selection.

Key Features:
- Environment variable configuration (U2_VERBOSE_FIELDS)
- Machine-parseable key=value format
- Default and custom field sets
- Type-safe formatting

Reference:
- experiments/run_uplift_u2.py — integration point
"""

import os
from typing import Any, Dict, List, Optional


def format_verbose_cycle(
    fields: List[str],
    data: Dict[str, Any]
) -> str:
    """
    Format cycle data for verbose output with configurable fields.
    
    Args:
        fields: List of field names to include
        data: Dictionary of available cycle data
        
    Returns:
        Formatted string with key=value pairs
        
    Example:
        >>> format_verbose_cycle(["cycle", "success"], {"cycle": 1, "success": True})
        'cycle=1 success=true'
    """
    parts = []
    for field in fields:
        if field in data:
            value = data[field]
            # Format value appropriately
            if isinstance(value, str):
                parts.append(f"{field}={value}")
            elif isinstance(value, bool):
                parts.append(f"{field}={str(value).lower()}")
            elif isinstance(value, (int, float)):
                parts.append(f"{field}={value}")
            else:
                parts.append(f"{field}={str(value)}")
        else:
            # Field not available - mark as N/A
            parts.append(f"{field}=N/A")
    
    return " ".join(parts)


def parse_verbose_fields(env_var: str = "U2_VERBOSE_FIELDS") -> Optional[List[str]]:
    """
    Parse verbose fields from environment variable.
    
    Args:
        env_var: Name of environment variable (default: U2_VERBOSE_FIELDS)
        
    Returns:
        List of field names, or None if not set
        
    Example:
        >>> os.environ["U2_VERBOSE_FIELDS"] = "cycle,mode,success"
        >>> parse_verbose_fields()
        ['cycle', 'mode', 'success']
    """
    value = os.environ.get(env_var)
    if not value:
        return None
    
    # Split by comma and strip whitespace
    fields = [f.strip() for f in value.split(",") if f.strip()]
    return fields if fields else None


# Default field sets for common use cases
DEFAULT_VERBOSE_FIELDS = ["cycle", "mode", "success", "item"]
EXTENDED_VERBOSE_FIELDS = ["cycle", "mode", "success", "item", "label", "slice", "seed"]
MINIMAL_VERBOSE_FIELDS = ["cycle", "success"]
