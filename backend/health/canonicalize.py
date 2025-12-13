"""Health canonicalization module.

Provides health data canonicalization for consistent output.
"""

import json
from typing import Any, Dict, List, Optional


def canonicalize_health(data: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize health data for consistent output."""
    return {
        "status": data.get("status", "unknown"),
        "timestamp": data.get("timestamp"),
        "components": data.get("components", []),
        "version": data.get("version", "1.0"),
    }


def canonicalize_component(component: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize a single health component."""
    return {
        "name": component.get("name", "unknown"),
        "status": component.get("status", "unknown"),
        "message": component.get("message", ""),
    }


def canonicalize_global_health(payload: Dict[str, Any]) -> str:
    """Canonicalize global health payload to deterministic JSON string.

    - Sorts keys alphabetically
    - Removes whitespace
    - Uses lowercase booleans (true/false)
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def normalize_health_status(status: str) -> str:
    """Normalize health status string."""
    status_lower = status.lower()
    if status_lower in ("ok", "healthy", "green", "pass", "good"):
        return "OK"
    elif status_lower in ("warn", "warning", "yellow", "degraded"):
        return "WARN"
    elif status_lower in ("error", "fail", "red", "unhealthy", "critical"):
        return "ERROR"
    return status.upper()


def compute_health_hash(payload: Dict[str, Any]) -> str:
    """Compute deterministic hash of health payload."""
    import hashlib
    canonical = canonicalize_global_health(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "canonicalize_health",
    "canonicalize_component",
    "canonicalize_global_health",
    "normalize_health_status",
    "compute_health_hash",
]
