"""
Runtime Profile Health Tile Adapter for Global Health Surface

SHADOW MODE: This adapter is purely observational and does NOT influence
any other tiles or system health classification.

Provides integration between U2 runtime profile system and global health dashboard.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Import runtime profile functions
try:
    from experiments.u2.runtime import (
        summarize_runtime_profile_health_for_global_console,
    )
except ImportError:
    # Graceful degradation if runtime module not available
    summarize_runtime_profile_health_for_global_console = None


def build_runtime_profile_tile_for_global_health(
    chaos_summary: Optional[Dict[str, Any]] = None,
    manual_snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build runtime profile health tile for global health surface.

    SHADOW MODE CONTRACT:
    - This function is read-only
    - The runtime_profile tile does NOT influence any other tiles
    - No control flow depends on the runtime_profile tile contents
    - This tile is purely for observability

    Args:
        chaos_summary: Optional chaos harness summary from experiments/u2_runtime_chaos.py
        manual_snapshot: Optional manual synthetic snapshot for testing

    Returns:
        Runtime profile health tile dictionary, or None if adapter unavailable
    """
    if summarize_runtime_profile_health_for_global_console is None:
        # Graceful degradation: return None if runtime module not available
        return None

    # Prefer chaos_summary if provided
    if chaos_summary is not None:
        try:
            return summarize_runtime_profile_health_for_global_console(chaos_summary)
        except Exception:
            # SHADOW MODE: Never fail the build due to runtime profile tile issues
            # Silently return None
            return None

    # Fall back to manual_snapshot if provided
    if manual_snapshot is not None:
        try:
            # Manual snapshot should have same structure as chaos_summary
            return summarize_runtime_profile_health_for_global_console(manual_snapshot)
        except Exception:
            # SHADOW MODE: Never fail the build due to runtime profile tile issues
            # Silently return None
            return None

    # No input provided
    return None


__all__ = [
    "build_runtime_profile_tile_for_global_health",
]

