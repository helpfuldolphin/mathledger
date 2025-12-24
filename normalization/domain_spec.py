"""
Domain specification parsing for FOL_FIN_EQ_v1.

This module provides the DomainSpec dataclass and parse_domain_spec() function
for loading and validating finite domain specifications from JSON files.

NORMATIVE INVARIANTS:
- parse_domain_spec() accepts ONLY pathlike objects (fail-closed on dict input)
- All constants must resolve to elements in the domain
- All function tables must be total and closed
- Elements order is preserved exactly as specified (determinism requirement)
- DomainSpec.constants is immutable (MappingProxyType)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Union

# Type alias for path-like inputs
PathLike = Union[str, Path]


@dataclass(frozen=True)
class DomainSpec:
    """
    Immutable specification of a finite domain for FOL_FIN_EQ_v1 verification.

    Attributes:
        domain_id: Unique identifier for this domain (e.g., "Z2", "Z3")
        elements: Ordered tuple of domain elements. Order is semantic for enumeration.
        _constants: Private dict mapping constant names to element values.
        mul_table: Cayley table for the 'mul' function (and potentially others).

    INVARIANTS:
        - All constant values must be in elements
        - All function table entries must be in elements
        - elements order determines enumeration order (determinism)
    """

    domain_id: str
    elements: tuple[str, ...]
    _constants: dict[str, str]
    mul_table: dict[str, dict[str, str]]

    @property
    def constants(self) -> Mapping[str, str]:
        """Return immutable view of constants dict.

        Prevents accidental mutation that could break determinism guarantees.
        """
        return MappingProxyType(self._constants)

    @property
    def identity(self) -> str:
        """Convenience accessor for identity constant."""
        return self.resolve_constant("identity")

    def __len__(self) -> int:
        """Return number of elements in the domain."""
        return len(self.elements)

    def resolve_constant(self, name: str) -> str:
        """Resolve constant key to element value.

        FAIL-CLOSED: Raises ValueError if:
        1. name is not a key in constants, OR
        2. resolved value is not in elements (belt+suspenders)

        Args:
            name: Constant key (e.g., "identity", "two")

        Returns:
            Element value that the constant resolves to

        Raises:
            ValueError: If constant key is unknown or value not in elements
        """
        if name not in self._constants:
            raise ValueError(f"unknown constant: '{name}' not in constants")

        value = self._constants[name]

        # Belt+suspenders: verify closure even if parse-time should catch it
        if value not in self.elements:
            raise ValueError(
                f"constant '{name}' resolves to '{value}' which is not in elements"
            )

        return value

    def exceeds_enumeration_limit(self, max_size: int = 50) -> bool:
        """Check if domain size exceeds enumeration limit.

        Args:
            max_size: Maximum allowed domain size (default: 50 per spec)

        Returns:
            True if domain size exceeds limit
        """
        return len(self) > max_size


def parse_domain_spec(path: PathLike) -> DomainSpec:
    """Parse a domain specification from a JSON file.

    FAIL-CLOSED validation:
    - elements must exist and be non-empty
    - identity constant must exist and be in elements
    - all constant values must be in elements
    - all function tables must be total and closed

    Args:
        path: Path to JSON file containing domain specification.
              MUST be a path-like object, NOT a dict.

    Returns:
        Validated DomainSpec instance

    Raises:
        TypeError: If path is not a path-like object (e.g., if dict is passed)
        FileNotFoundError: If file does not exist
        ValueError: If validation fails
    """
    # Fail-closed: reject dict input
    if isinstance(path, dict):
        raise TypeError(
            "parse_domain_spec() requires a path-like object, not a dict. "
            "This enforces file-based governance and audit trail."
        )

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Domain spec file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return _validate_and_build(data, source_path=path)


def _validate_and_build(data: dict, source_path: Path) -> DomainSpec:
    """Validate domain spec data and build DomainSpec instance.

    Args:
        data: Parsed JSON data
        source_path: Path to source file (for error messages)

    Returns:
        Validated DomainSpec

    Raises:
        ValueError: If validation fails
    """
    # Required field: domain_id
    if "domain_id" not in data:
        raise ValueError(f"Missing 'domain_id' in {source_path}")
    domain_id = data["domain_id"]

    # Required field: elements (must be non-empty list)
    if "elements" not in data:
        raise ValueError(f"Missing 'elements' in {source_path}")
    elements_list = data["elements"]
    if not isinstance(elements_list, list) or len(elements_list) == 0:
        raise ValueError(f"'elements' must be a non-empty list in {source_path}")

    # Convert to tuple (preserves order, immutable)
    elements = tuple(elements_list)
    elements_set = set(elements)

    # Validate no duplicate elements
    if len(elements_set) != len(elements):
        raise ValueError(f"Duplicate elements in {source_path}")

    # Required field: constants with identity
    constants_data = data.get("constants", {})
    if "identity" not in constants_data:
        raise ValueError(f"Missing 'identity' in constants in {source_path}")

    # Validate all constant values are in elements
    for const_name, const_value in constants_data.items():
        if const_value not in elements_set:
            raise ValueError(
                f"Constant '{const_name}' has value '{const_value}' which is "
                f"not in elements in {source_path}"
            )

    # Validate identity specifically
    identity_value = constants_data["identity"]
    if identity_value not in elements_set:
        raise ValueError(
            f"identity constant '{identity_value}' not in elements in {source_path}"
        )

    # Get functions dict
    functions_data = data.get("functions", {})

    # Validate mul table if present
    mul_table: dict[str, dict[str, str]] = {}
    if "mul" in functions_data:
        mul_table = _validate_function_table(
            functions_data["mul"], "mul", elements, elements_set, source_path
        )

    # Validate any other function tables
    for func_name, func_table in functions_data.items():
        if func_name != "mul":
            _validate_function_table(
                func_table, func_name, elements, elements_set, source_path
            )

    return DomainSpec(
        domain_id=domain_id,
        elements=elements,
        _constants=dict(constants_data),  # Copy to prevent external mutation
        mul_table=mul_table,
    )


def _validate_function_table(
    table: dict,
    func_name: str,
    elements: tuple[str, ...],
    elements_set: set[str],
    source_path: Path,
) -> dict[str, dict[str, str]]:
    """Validate a function table is total and closed.

    TOTAL: For every pair (e1, e2) in elements × elements, table[e1][e2] exists
    CLOSED: For every entry, the output value is in elements

    Args:
        table: Function table to validate
        func_name: Name of function (for error messages)
        elements: Ordered tuple of elements
        elements_set: Set of elements (for fast lookup)
        source_path: Source file path (for error messages)

    Returns:
        Validated table (as dict)

    Raises:
        ValueError: If table is incomplete or not closed
    """
    if not isinstance(table, dict):
        raise ValueError(f"Function '{func_name}' table must be a dict in {source_path}")

    # Check all rows exist
    for e1 in elements:
        if e1 not in table:
            raise ValueError(
                f"Function '{func_name}' table incomplete: missing row for '{e1}' "
                f"in {source_path}"
            )

        row = table[e1]
        if not isinstance(row, dict):
            raise ValueError(
                f"Function '{func_name}' table row for '{e1}' must be a dict "
                f"in {source_path}"
            )

        # Check all columns exist and values are in elements
        for e2 in elements:
            if e2 not in row:
                raise ValueError(
                    f"Function '{func_name}' table incomplete: missing entry "
                    f"['{e1}']['{e2}'] in {source_path}"
                )

            output = row[e2]
            if output not in elements_set:
                raise ValueError(
                    f"Function '{func_name}' table entry [{e1}][{e2}] = '{output}' "
                    f"is not in elements in {source_path}"
                )

    # Check for unexpected keys in table
    for key in table:
        if key not in elements_set:
            raise ValueError(
                f"Function '{func_name}' table has unexpected key '{key}' "
                f"not in elements in {source_path}"
            )

    return dict(table)  # Return a copy
