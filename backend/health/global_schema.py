"""Global health schema module.

Provides schema definitions for global health data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


GLOBAL_HEALTH_SCHEMA_VERSION = "1.0.0"


@dataclass
class GlobalHealthSchema:
    """Schema for global health data."""
    version: str = GLOBAL_HEALTH_SCHEMA_VERSION
    required_fields: List[str] = field(default_factory=lambda: ["status", "timestamp"])
    optional_fields: List[str] = field(default_factory=lambda: ["components", "metadata"])


@dataclass
class ComponentSchema:
    """Schema for health component."""
    name: str
    status_values: List[str] = field(default_factory=lambda: ["ok", "warn", "error"])


def validate_global_health_schema(data: Dict[str, Any]) -> bool:
    """Validate data against global health schema."""
    schema = GlobalHealthSchema()
    for fld in schema.required_fields:
        if fld not in data:
            return False
    return True


def validate_global_health(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate global health data and return validation result."""
    schema = GlobalHealthSchema()
    errors = []
    warnings = []

    for fld in schema.required_fields:
        if fld not in data:
            errors.append(f"Missing required field: {fld}")

    valid = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "schema_version": schema.version,
    }


def get_global_health_schema() -> GlobalHealthSchema:
    """Get the global health schema."""
    return GlobalHealthSchema()


class SchemaValidationError(Exception):
    """Exception raised for schema validation errors."""
    pass


__all__ = [
    "GLOBAL_HEALTH_SCHEMA_VERSION",
    "GlobalHealthSchema",
    "ComponentSchema",
    "SchemaValidationError",
    "validate_global_health_schema",
    "validate_global_health",
    "get_global_health_schema",
]
