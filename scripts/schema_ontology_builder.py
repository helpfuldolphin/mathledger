#!/usr/bin/env python3
"""
Schema Ontology Builder — Schema Evolution Intelligence System

Agent: doc-ops-2 (E2) — Curriculum–Semantic Mapper & Evolution Planner
Mission: Build ontology graphs, detect schema relationships, classify fields, detect regressions,
         score change impact, provide consumer-centric views, and map semantic-curriculum alignment.

This module:
    1. Infers relationships between schemas (inheritance, embedding, equivalence)
    2. Classifies fields into conceptual clusters (seed, metric, success, governance)
    3. Generates ontology graphs (schema_ontology.dot)
    4. Exports schema_ontology.json with SHA-256 hash commitments
    5. Detects schema regressions (narrowing, widening, field disappearance)
    6. Computes schema blast radius (downstream file impact)
    7. Generates schema change briefs for PR review
    8. Provides schema explanation queries
    9. Scores change impact severity (INFO/WARN/BLOCK)
    10. Provides consumer-centric change views
    11. Builds per-field impact matrices
    12. Generates reviewer checklists
    13. Exports impact manifests for agent interoperability
    14. Builds cross-schema impact indices
    15. Generates reviewer playbooks
    16. Provides director-facing status panels
    17. Evaluates schema change SLOs
    18. Routes schema changes to code owners
    19. Maps semantic-curriculum harmonic alignment
    20. Forecasts curriculum evolution needs

Usage:
    python scripts/schema_ontology_builder.py                           # Build ontology
    python scripts/schema_ontology_builder.py --check                   # CI mode
    python scripts/schema_ontology_builder.py --blast-radius            # Compute blast radius
    python scripts/schema_ontology_builder.py --regression-check        # Check for regressions
    python scripts/schema_ontology_builder.py --change-brief            # Generate change brief
    python scripts/schema_ontology_builder.py --explain NAME            # Explain a specific schema
    python scripts/schema_ontology_builder.py --explain-consumers FILE  # Consumer-centric view
    python scripts/schema_ontology_builder.py --review-checklist        # Generate reviewer checklist
    python scripts/schema_ontology_builder.py --field-matrix            # Per-field impact JSON
    python scripts/schema_ontology_builder.py --impact-manifest         # Compact manifest for agents
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# =============================================================================
# CONSTANTS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ONTOLOGY_JSON = PROJECT_ROOT / "docs" / "schema_ontology.json"
OUTPUT_ONTOLOGY_DOT = PROJECT_ROOT / "docs" / "schema_ontology.dot"
OUTPUT_REGRESSION_REPORT = PROJECT_ROOT / "docs" / "SCHEMA_REGRESSION_REPORT.md"
OUTPUT_CHANGE_BRIEF = PROJECT_ROOT / "docs" / "SCHEMA_CHANGE_BRIEF.md"
BASELINE_ONTOLOGY = PROJECT_ROOT / "docs" / ".schema_ontology_baseline.json"

# Schema sources (must match generate_schema_docs.py)
SCHEMA_SOURCES = {
    "yaml": [
        "config/curriculum_uplift_phase2.yaml",
        "config/verifier_budget_phase2.yaml",
        "config/curriculum.yaml",
    ],
    "json_config": [
        "config/allblue_lanes.json",
        "config/causal/default.json",
        "config/rfl/production.json",
        "config/rfl/quick.json",
        "config/rfl/quick_test.json",
    ],
    "json_schema": [
        "docs/evidence/experiment_manifest_schema.json",
    ],
    "python_definitions": [
        "experiments/manifest.py",
    ],
}

# Files that depend on schemas (for blast radius computation)
DOWNSTREAM_DEPENDENCIES = {
    "config/curriculum.yaml": [
        "backend/orchestrator/curriculum_loader.py",
        "experiments/curriculum_loader_v2.py",
        "experiments/run_fo_cycles.py",
        "scripts/run_first_organism.py",
    ],
    "config/curriculum_uplift_phase2.yaml": [
        "experiments/run_uplift_u2.py",
        "experiments/analyze_uplift_success.py",
    ],
    "config/rfl/production.json": [
        "experiments/rfl/run_experiment.py",
        "scripts/rfl/rfl_gate.py",
    ],
    "config/rfl/quick.json": [
        "experiments/rfl/run_experiment.py",
    ],
    "config/rfl/quick_test.json": [
        "experiments/rfl/run_experiment.py",
    ],
    "config/allblue_lanes.json": [
        "scripts/generate_allblue_fleet_state.py",
        "scripts/generate_allblue_epoch_seal.py",
    ],
    "config/causal/default.json": [
        "experiments/benchmark_chain_analysis.py",
    ],
    "docs/evidence/experiment_manifest_schema.json": [
        "experiments/manifest.py",
        "scripts/verify_manifest_integrity.py",
    ],
    "experiments/manifest.py": [
        "experiments/create_manifest_direct.py",
        "experiments/rfl/run_experiment.py",
    ],
}

# Schemas considered "widely used" — field removal triggers BLOCK severity
WIDELY_USED_SCHEMAS = {
    "curriculum",
    "curriculum_uplift_phase2",
    "production",
    "experiment_manifest_schema",
}

# Critical fields — type changes trigger BLOCK severity
# Format: {schema_name: [field_path_patterns]}
CRITICAL_FIELDS = {
    "curriculum": [
        "systems.pl.slices",
        "systems.pl.active",
        "version",
    ],
    "curriculum_uplift_phase2": [
        "slices",
        "version",
        "governance",
    ],
    "production": [
        "pass_criteria",
        "bootstrap_config",
        "version",
    ],
    "experiment_manifest_schema": [
        "manifest_version",
        "provenance",
        "results",
    ],
}

# Consumer field mappings — which fields each consumer reads
# This is declarative knowledge for the ontology (no static analysis)
CONSUMER_FIELD_USAGE = {
    "backend/orchestrator/curriculum_loader.py": {
        "schema": "curriculum",
        "fields_read": ["systems", "systems.pl", "systems.pl.slices", "systems.pl.active", "version"],
    },
    "experiments/curriculum_loader_v2.py": {
        "schema": "curriculum",
        "fields_read": ["systems", "systems.pl", "systems.pl.slices", "version"],
    },
    "experiments/run_fo_cycles.py": {
        "schema": "curriculum",
        "fields_read": ["systems.pl.slices", "systems.pl.active"],
    },
    "scripts/run_first_organism.py": {
        "schema": "curriculum",
        "fields_read": ["systems.pl"],
    },
    "experiments/run_uplift_u2.py": {
        "schema": "curriculum_uplift_phase2",
        "fields_read": ["slices", "version", "governance"],
    },
    "experiments/analyze_uplift_success.py": {
        "schema": "curriculum_uplift_phase2",
        "fields_read": ["slices"],
    },
    "experiments/rfl/run_experiment.py": {
        "schema": "production",
        "fields_read": ["bootstrap_config", "pass_criteria", "metrics", "output"],
    },
    "scripts/rfl/rfl_gate.py": {
        "schema": "production",
        "fields_read": ["pass_criteria", "commit_template"],
    },
    "scripts/generate_allblue_fleet_state.py": {
        "schema": "allblue_lanes",
        "fields_read": ["lanes", "epoch_seal", "archive"],
    },
    "scripts/generate_allblue_epoch_seal.py": {
        "schema": "allblue_lanes",
        "fields_read": ["epoch_seal", "lanes"],
    },
    "experiments/manifest.py": {
        "schema": "experiment_manifest_schema",
        "fields_read": ["manifest_version", "provenance", "configuration", "artifacts", "results"],
    },
    "scripts/verify_manifest_integrity.py": {
        "schema": "experiment_manifest_schema",
        "fields_read": ["manifest_version", "provenance", "artifacts"],
    },
}


# =============================================================================
# ENUMS
# =============================================================================


class FieldCluster(str, Enum):
    """Conceptual clusters for field classification."""

    SEED = "seed"  # Random seeds, initialization params
    METRIC = "metric"  # Measurement, statistics, counts
    SUCCESS = "success"  # Pass/fail criteria, thresholds
    GOVERNANCE = "governance"  # Permissions, restrictions, locks
    IDENTITY = "identity"  # IDs, hashes, versions
    TEMPORAL = "temporal"  # Timestamps, durations, windows
    STRUCTURAL = "structural"  # Paths, nesting, arrays
    CONFIGURATION = "configuration"  # General config params
    UNKNOWN = "unknown"


class RelationType(str, Enum):
    """Types of relationships between schemas."""

    EMBEDS = "embeds"  # Schema A contains fields from Schema B
    EXTENDS = "extends"  # Schema A is a superset of Schema B
    EQUIVALENT = "equivalent"  # Schemas share >80% fields
    REFERENCES = "references"  # Schema A references Schema B by name/path
    SIBLING = "sibling"  # Schemas share common parent structure
    NONE = "none"


class DriftType(str, Enum):
    """Types of schema drift/regression."""

    NARROWING = "narrowing"  # Type became more restrictive
    WIDENING = "widening"  # Type became less restrictive
    FIELD_ADDED = "field_added"  # New field appeared
    FIELD_REMOVED = "field_removed"  # Required field disappeared
    FIELD_RENAMED = "field_renamed"  # Field name changed (heuristic)
    TYPE_CHANGED = "type_changed"  # Field type changed
    CONSTRAINT_ADDED = "constraint_added"  # New constraint
    CONSTRAINT_REMOVED = "constraint_removed"  # Constraint removed
    STRUCTURAL = "structural"  # Nesting structure changed


class Severity(str, Enum):
    """Impact severity levels for schema changes."""

    INFO = "INFO"  # Cosmetic changes only, safe to ignore
    WARN = "WARN"  # New fields added, non-critical type widening
    BLOCK = "BLOCK"  # Field removed from widely-used schema, critical type change


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SchemaField:
    """Represents a field extracted from a schema."""

    name: str
    path: str
    field_type: str
    cluster: FieldCluster = FieldCluster.UNKNOWN
    required: bool = True
    constraints: list[str] = field(default_factory=list)
    example: Any = None


@dataclass
class SchemaRelationship:
    """Represents a relationship between two schemas."""

    source_schema: str
    target_schema: str
    relation_type: RelationType
    confidence: float  # 0.0 to 1.0
    shared_fields: list[str] = field(default_factory=list)
    evidence: str = ""


@dataclass
class SchemaDrift:
    """Represents a detected schema drift/regression."""

    schema_name: str
    drift_type: DriftType
    field_path: str
    old_value: str
    new_value: str
    severity: str  # "info", "warning", "error"
    description: str


@dataclass
class SchemaNode:
    """Represents a schema in the ontology graph."""

    name: str
    source_file: str
    schema_type: str
    version: str
    field_count: int
    fields: list[SchemaField] = field(default_factory=list)
    cluster_distribution: dict[str, int] = field(default_factory=dict)
    raw_hash: str = ""


@dataclass
class SchemaOntology:
    """Complete ontology model."""

    version: str = "1.0.0"
    generated_at: str = ""
    commit_hash: str = ""
    nodes: list[SchemaNode] = field(default_factory=list)
    relationships: list[SchemaRelationship] = field(default_factory=list)
    drifts: list[SchemaDrift] = field(default_factory=list)
    blast_radius: dict[str, list[str]] = field(default_factory=dict)


# =============================================================================
# FIELD CLUSTERING
# =============================================================================


# Classification patterns for field clustering
CLUSTER_PATTERNS: dict[FieldCluster, list[re.Pattern[str]]] = {
    FieldCluster.SEED: [
        re.compile(r"seed", re.I),
        re.compile(r"random", re.I),
        re.compile(r"salt", re.I),
        re.compile(r"nonce", re.I),
    ],
    FieldCluster.METRIC: [
        re.compile(r"count", re.I),
        re.compile(r"total", re.I),
        re.compile(r"mean", re.I),
        re.compile(r"median", re.I),
        re.compile(r"std", re.I),
        re.compile(r"variance", re.I),
        re.compile(r"rate", re.I),
        re.compile(r"throughput", re.I),
        re.compile(r"velocity", re.I),
        re.compile(r"_pct$", re.I),
        re.compile(r"percent", re.I),
        re.compile(r"ratio", re.I),
        re.compile(r"score", re.I),
        re.compile(r"metric", re.I),
        re.compile(r"stat", re.I),
    ],
    FieldCluster.SUCCESS: [
        re.compile(r"pass", re.I),
        re.compile(r"fail", re.I),
        re.compile(r"success", re.I),
        re.compile(r"threshold", re.I),
        re.compile(r"criteria", re.I),
        re.compile(r"gate", re.I),
        re.compile(r"_min$", re.I),
        re.compile(r"_max$", re.I),
        re.compile(r"^min_", re.I),
        re.compile(r"^max_", re.I),
        re.compile(r"bound", re.I),
        re.compile(r"limit", re.I),
        re.compile(r"target", re.I),
        re.compile(r"goal", re.I),
    ],
    FieldCluster.GOVERNANCE: [
        re.compile(r"required", re.I),
        re.compile(r"blocked", re.I),
        re.compile(r"allowed", re.I),
        re.compile(r"permission", re.I),
        re.compile(r"restrict", re.I),
        re.compile(r"lock", re.I),
        re.compile(r"seal", re.I),
        re.compile(r"governance", re.I),
        re.compile(r"policy", re.I),
        re.compile(r"attestation", re.I),
        re.compile(r"verify", re.I),
        re.compile(r"hermetic", re.I),
        re.compile(r"no_modify", re.I),
        re.compile(r"read_only", re.I),
    ],
    FieldCluster.IDENTITY: [
        re.compile(r"_id$", re.I),
        re.compile(r"^id$", re.I),
        re.compile(r"uuid", re.I),
        re.compile(r"hash", re.I),
        re.compile(r"sha", re.I),
        re.compile(r"version", re.I),
        re.compile(r"name$", re.I),
        re.compile(r"^name$", re.I),
        re.compile(r"slug", re.I),
        re.compile(r"key$", re.I),
    ],
    FieldCluster.TEMPORAL: [
        re.compile(r"time", re.I),
        re.compile(r"date", re.I),
        re.compile(r"duration", re.I),
        re.compile(r"timeout", re.I),
        re.compile(r"_s$"),  # seconds suffix
        re.compile(r"_ms$"),  # milliseconds suffix
        re.compile(r"window", re.I),
        re.compile(r"interval", re.I),
        re.compile(r"period", re.I),
        re.compile(r"minutes", re.I),
        re.compile(r"hours", re.I),
        re.compile(r"days", re.I),
    ],
    FieldCluster.STRUCTURAL: [
        re.compile(r"path", re.I),
        re.compile(r"file", re.I),
        re.compile(r"dir", re.I),
        re.compile(r"url", re.I),
        re.compile(r"endpoint", re.I),
        re.compile(r"host", re.I),
        re.compile(r"port", re.I),
        re.compile(r"database", re.I),
        re.compile(r"schema", re.I),
        re.compile(r"format", re.I),
    ],
}


def classify_field(field_name: str, field_type: str, constraints: list[str]) -> FieldCluster:
    """Classify a field into a conceptual cluster based on naming patterns."""
    # Check each cluster's patterns
    for cluster, patterns in CLUSTER_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(field_name):
                return cluster

    # Fallback based on type
    if field_type in ("object", "array"):
        return FieldCluster.STRUCTURAL
    if field_type in ("integer", "number") and any(
        c in constraints for c in ["minimum bound", "maximum bound"]
    ):
        return FieldCluster.SUCCESS

    return FieldCluster.CONFIGURATION


# =============================================================================
# TYPE INFERENCE
# =============================================================================


def infer_type(value: Any) -> str:
    """Infer the type string for a given value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        if not value:
            return "array"
        inner_types = {infer_type(v) for v in value[:5]}
        if len(inner_types) == 1:
            return f"array<{inner_types.pop()}>"
        return f"array<{' | '.join(sorted(inner_types))}>"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def extract_constraints(key: str, value: Any) -> list[str]:
    """Extract constraints from key naming conventions and value patterns."""
    constraints = []

    if "_min" in key or key.startswith("min_"):
        constraints.append("minimum bound")
    if "_max" in key or key.startswith("max_"):
        constraints.append("maximum bound")
    if "_pct" in key or "percent" in key.lower():
        constraints.append("0-100 percentage")
    if "_s" in key and isinstance(value, (int, float)):
        constraints.append("seconds")
    if "_ms" in key and isinstance(value, (int, float)):
        constraints.append("milliseconds")
    if "timeout" in key.lower():
        constraints.append("timeout value")
    if "hash" in key.lower() and isinstance(value, str):
        if len(value) == 64:
            constraints.append("SHA-256 hex string")
        elif len(value) == 32:
            constraints.append("MD5 hex string")
    if "url" in key.lower():
        constraints.append("URL string")
    if "path" in key.lower():
        constraints.append("file path")

    if isinstance(value, str):
        if re.match(r"^\d{4}-\d{2}-\d{2}", value):
            constraints.append("ISO date format")
        if re.match(r"^[a-f0-9]{64}$", value):
            constraints.append("SHA-256 hash")

    return constraints


# =============================================================================
# SCHEMA PARSING
# =============================================================================


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    content = filepath.read_bytes()
    return hashlib.sha256(content).hexdigest()


def parse_schema_file(filepath: Path) -> SchemaNode | None:
    """Parse a schema file into a SchemaNode."""
    if not filepath.exists():
        return None

    suffix = filepath.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return _parse_yaml_schema(filepath)
    elif suffix == ".json":
        return _parse_json_schema(filepath)
    elif suffix == ".py":
        return _parse_python_schema(filepath)

    return None


def _parse_yaml_schema(filepath: Path) -> SchemaNode | None:
    """Parse YAML schema file."""
    if yaml is None:
        return None

    try:
        content = filepath.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    fields: list[SchemaField] = []
    _extract_fields_recursive(data, "", fields)

    # Compute cluster distribution
    cluster_dist: dict[str, int] = defaultdict(int)
    for f in fields:
        cluster_dist[f.cluster.value] += 1

    return SchemaNode(
        name=filepath.stem,
        source_file=str(filepath.relative_to(PROJECT_ROOT)),
        schema_type="yaml",
        version=str(data.get("version", "")),
        field_count=len(fields),
        fields=fields,
        cluster_distribution=dict(cluster_dist),
        raw_hash=compute_file_hash(filepath),
    )


def _parse_json_schema(filepath: Path) -> SchemaNode | None:
    """Parse JSON schema file."""
    try:
        content = filepath.read_text(encoding="utf-8")
        data = json.loads(content)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    fields: list[SchemaField] = []
    _extract_fields_recursive(data, "", fields)

    cluster_dist: dict[str, int] = defaultdict(int)
    for f in fields:
        cluster_dist[f.cluster.value] += 1

    return SchemaNode(
        name=filepath.stem,
        source_file=str(filepath.relative_to(PROJECT_ROOT)),
        schema_type="json",
        version=str(data.get("version", data.get("manifest_version", ""))),
        field_count=len(fields),
        fields=fields,
        cluster_distribution=dict(cluster_dist),
        raw_hash=compute_file_hash(filepath),
    )


def _parse_python_schema(filepath: Path) -> SchemaNode | None:
    """Parse Python schema file (simplified - focuses on dataclasses/classes)."""
    import ast

    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except Exception:
        return None

    fields: list[SchemaField] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for arg in item.args.args:
                        if arg.arg == "self":
                            continue
                        type_str = "Any"
                        if arg.annotation:
                            type_str = _ast_type_to_string(arg.annotation)
                        constraints = extract_constraints(arg.arg, None)
                        sf = SchemaField(
                            name=arg.arg,
                            path=f"{node.name}.{arg.arg}",
                            field_type=type_str,
                            cluster=classify_field(arg.arg, type_str, constraints),
                            constraints=constraints,
                        )
                        fields.append(sf)

    cluster_dist: dict[str, int] = defaultdict(int)
    for f in fields:
        cluster_dist[f.cluster.value] += 1

    return SchemaNode(
        name=filepath.stem,
        source_file=str(filepath.relative_to(PROJECT_ROOT)),
        schema_type="python",
        version="",
        field_count=len(fields),
        fields=fields,
        cluster_distribution=dict(cluster_dist),
        raw_hash=compute_file_hash(filepath),
    )


def _ast_type_to_string(node: Any) -> str:
    """Convert AST type annotation to string."""
    import ast

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Subscript):
        base = _ast_type_to_string(node.value)
        if isinstance(node.slice, ast.Tuple):
            args = ", ".join(_ast_type_to_string(e) for e in node.slice.elts)
        else:
            args = _ast_type_to_string(node.slice)
        return f"{base}[{args}]"
    if isinstance(node, ast.Attribute):
        return f"{_ast_type_to_string(node.value)}.{node.attr}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return f"{_ast_type_to_string(node.left)} | {_ast_type_to_string(node.right)}"
    return "Any"


def _extract_fields_recursive(
    data: dict[str, Any],
    prefix: str,
    fields: list[SchemaField],
    max_depth: int = 5,
) -> None:
    """Recursively extract fields from nested dictionaries."""
    if max_depth <= 0:
        return

    for key, value in sorted(data.items()):
        path = f"{prefix}.{key}" if prefix else key
        field_type = infer_type(value)
        constraints = extract_constraints(key, value)

        sf = SchemaField(
            name=key,
            path=path,
            field_type=field_type,
            cluster=classify_field(key, field_type, constraints),
            constraints=constraints,
        )

        if not isinstance(value, (dict, list)):
            sf.example = value

        fields.append(sf)

        if isinstance(value, dict) and value:
            _extract_fields_recursive(value, path, fields, max_depth - 1)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            _extract_fields_recursive(value[0], f"{path}[]", fields, max_depth - 1)


# =============================================================================
# RELATIONSHIP INFERENCE
# =============================================================================


def infer_relationships(nodes: list[SchemaNode]) -> list[SchemaRelationship]:
    """Infer relationships between schema nodes."""
    relationships: list[SchemaRelationship] = []

    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1 :]:
            rel = _compute_relationship(node_a, node_b)
            if rel and rel.relation_type != RelationType.NONE:
                relationships.append(rel)

    return relationships


def _compute_relationship(
    node_a: SchemaNode, node_b: SchemaNode
) -> SchemaRelationship | None:
    """Compute the relationship between two schema nodes."""
    fields_a = {f.name for f in node_a.fields}
    fields_b = {f.name for f in node_b.fields}

    shared = fields_a & fields_b
    only_a = fields_a - fields_b
    only_b = fields_b - fields_a

    if not shared:
        return SchemaRelationship(
            source_schema=node_a.name,
            target_schema=node_b.name,
            relation_type=RelationType.NONE,
            confidence=0.0,
        )

    # Compute Jaccard similarity
    jaccard = len(shared) / len(fields_a | fields_b)

    # Determine relationship type
    if jaccard >= 0.8:
        rel_type = RelationType.EQUIVALENT
        evidence = f"Schemas share {len(shared)} fields ({jaccard:.1%} Jaccard similarity)"
    elif len(only_b) == 0 and len(only_a) > 0:
        rel_type = RelationType.EXTENDS
        evidence = f"{node_a.name} extends {node_b.name} with {len(only_a)} additional fields"
    elif len(only_a) == 0 and len(only_b) > 0:
        rel_type = RelationType.EXTENDS
        evidence = f"{node_b.name} extends {node_a.name} with {len(only_b)} additional fields"
    elif jaccard >= 0.3:
        rel_type = RelationType.SIBLING
        evidence = f"Schemas share {len(shared)} common fields, likely sibling schemas"
    else:
        rel_type = RelationType.REFERENCES
        evidence = f"Schemas share {len(shared)} fields with low overlap"

    return SchemaRelationship(
        source_schema=node_a.name,
        target_schema=node_b.name,
        relation_type=rel_type,
        confidence=jaccard,
        shared_fields=sorted(shared),
        evidence=evidence,
    )


# =============================================================================
# DRIFT DETECTION
# =============================================================================


def detect_drifts(
    current: list[SchemaNode], baseline: list[SchemaNode] | None
) -> list[SchemaDrift]:
    """Detect schema drifts between current and baseline ontology."""
    if baseline is None:
        return []

    drifts: list[SchemaDrift] = []
    baseline_map = {n.name: n for n in baseline}
    current_map = {n.name: n for n in current}

    for name, current_node in current_map.items():
        if name not in baseline_map:
            # New schema added
            drifts.append(
                SchemaDrift(
                    schema_name=name,
                    drift_type=DriftType.FIELD_ADDED,
                    field_path="<schema>",
                    old_value="",
                    new_value=current_node.source_file,
                    severity="info",
                    description=f"New schema '{name}' added",
                )
            )
            continue

        baseline_node = baseline_map[name]
        drifts.extend(_compare_schema_nodes(baseline_node, current_node))

    # Check for removed schemas
    for name in baseline_map:
        if name not in current_map:
            drifts.append(
                SchemaDrift(
                    schema_name=name,
                    drift_type=DriftType.FIELD_REMOVED,
                    field_path="<schema>",
                    old_value=baseline_map[name].source_file,
                    new_value="",
                    severity="error",
                    description=f"Schema '{name}' was removed",
                )
            )

    return drifts


def _compare_schema_nodes(
    baseline: SchemaNode, current: SchemaNode
) -> list[SchemaDrift]:
    """Compare two schema nodes and detect field-level drifts."""
    drifts: list[SchemaDrift] = []

    baseline_fields = {f.path: f for f in baseline.fields}
    current_fields = {f.path: f for f in current.fields}

    # Check for removed fields
    for path, bf in baseline_fields.items():
        if path not in current_fields:
            # Check if it might be renamed (heuristic: same type, similar path)
            possible_renames = [
                cf
                for cp, cf in current_fields.items()
                if cf.field_type == bf.field_type and cp not in baseline_fields
            ]
            if possible_renames:
                # Likely a rename
                drifts.append(
                    SchemaDrift(
                        schema_name=baseline.name,
                        drift_type=DriftType.FIELD_RENAMED,
                        field_path=path,
                        old_value=bf.name,
                        new_value=possible_renames[0].name,
                        severity="warning",
                        description=f"Field '{bf.name}' may have been renamed to '{possible_renames[0].name}'",
                    )
                )
            else:
                drifts.append(
                    SchemaDrift(
                        schema_name=baseline.name,
                        drift_type=DriftType.FIELD_REMOVED,
                        field_path=path,
                        old_value=bf.field_type,
                        new_value="",
                        severity="error",
                        description=f"Required field '{path}' was removed",
                    )
                )

    # Check for added fields
    for path, cf in current_fields.items():
        if path not in baseline_fields:
            drifts.append(
                SchemaDrift(
                    schema_name=current.name,
                    drift_type=DriftType.FIELD_ADDED,
                    field_path=path,
                    old_value="",
                    new_value=cf.field_type,
                    severity="info",
                    description=f"New field '{path}' added with type '{cf.field_type}'",
                )
            )

    # Check for type changes
    for path in baseline_fields.keys() & current_fields.keys():
        bf = baseline_fields[path]
        cf = current_fields[path]

        if bf.field_type != cf.field_type:
            drift_type = _classify_type_change(bf.field_type, cf.field_type)
            severity = "warning" if drift_type == DriftType.NARROWING else "info"
            drifts.append(
                SchemaDrift(
                    schema_name=current.name,
                    drift_type=drift_type,
                    field_path=path,
                    old_value=bf.field_type,
                    new_value=cf.field_type,
                    severity=severity,
                    description=f"Field '{path}' type changed from '{bf.field_type}' to '{cf.field_type}'",
                )
            )

        # Check constraint changes
        old_constraints = set(bf.constraints)
        new_constraints = set(cf.constraints)

        for added in new_constraints - old_constraints:
            drifts.append(
                SchemaDrift(
                    schema_name=current.name,
                    drift_type=DriftType.CONSTRAINT_ADDED,
                    field_path=path,
                    old_value="",
                    new_value=added,
                    severity="info",
                    description=f"Constraint '{added}' added to field '{path}'",
                )
            )

        for removed in old_constraints - new_constraints:
            drifts.append(
                SchemaDrift(
                    schema_name=current.name,
                    drift_type=DriftType.CONSTRAINT_REMOVED,
                    field_path=path,
                    old_value=removed,
                    new_value="",
                    severity="warning",
                    description=f"Constraint '{removed}' removed from field '{path}'",
                )
            )

    return drifts


def _classify_type_change(old_type: str, new_type: str) -> DriftType:
    """Classify a type change as narrowing, widening, or general change."""
    # Type hierarchy (more general -> more specific)
    type_hierarchy = {
        "Any": 0,
        "object": 1,
        "array": 2,
        "string": 3,
        "number": 4,
        "integer": 5,
        "boolean": 6,
        "null": 7,
    }

    old_base = old_type.split("<")[0].split("[")[0]
    new_base = new_type.split("<")[0].split("[")[0]

    old_rank = type_hierarchy.get(old_base, 3)
    new_rank = type_hierarchy.get(new_base, 3)

    if new_rank > old_rank:
        return DriftType.NARROWING  # More specific = narrowing
    elif new_rank < old_rank:
        return DriftType.WIDENING  # Less specific = widening
    else:
        return DriftType.TYPE_CHANGED


# =============================================================================
# BLAST RADIUS COMPUTATION
# =============================================================================


def compute_blast_radius(nodes: list[SchemaNode]) -> dict[str, list[str]]:
    """Compute downstream files affected by each schema."""
    blast_radius: dict[str, list[str]] = {}

    for node in nodes:
        source_file = node.source_file
        affected: list[str] = []

        # Direct dependencies
        if source_file in DOWNSTREAM_DEPENDENCIES:
            affected.extend(DOWNSTREAM_DEPENDENCIES[source_file])

        # Check if files actually exist
        affected = [f for f in affected if (PROJECT_ROOT / f).exists()]

        # Add transitive dependencies (files that import the direct deps)
        # This is a simplified version - full analysis would require import graph
        blast_radius[node.name] = sorted(set(affected))

    return blast_radius


# =============================================================================
# SCHEMA CHANGE BRIEF
# =============================================================================


@dataclass
class SchemaChange:
    """Represents changes to a single schema."""

    schema_name: str
    source_file: str
    fields_added: list[tuple[str, str]]  # (path, type)
    fields_removed: list[tuple[str, str]]  # (path, type)
    fields_renamed: list[tuple[str, str, str]]  # (old_path, old_name, new_name)
    fields_type_changed: list[tuple[str, str, str]]  # (path, old_type, new_type)
    downstream_consumers: list[str]


@dataclass
class SchemaChangeBrief:
    """Complete change brief for PR review."""

    generated_at: str
    base_hash: str
    current_hash: str
    schemas_changed: list[str]
    schema_changes: list[SchemaChange]
    total_fields_added: int
    total_fields_removed: int
    total_fields_modified: int
    notes: list[str] = field(default_factory=list)


@dataclass
class ConsumerInfo:
    """Information about a schema consumer."""

    file_path: str
    field_access_count: int
    fields_read: list[str] = field(default_factory=list)


@dataclass
class SchemaImpact:
    """Impact assessment for a single schema's changes."""

    schema_name: str
    severity: Severity
    reasons: list[str] = field(default_factory=list)
    blocked_fields: list[str] = field(default_factory=list)
    warned_fields: list[str] = field(default_factory=list)
    consumers: list[ConsumerInfo] = field(default_factory=list)


@dataclass
class ImpactSummary:
    """Overall impact summary for all schema changes."""

    overall_severity: Severity
    schema_impacts: list[SchemaImpact] = field(default_factory=list)
    block_count: int = 0
    warn_count: int = 0
    info_count: int = 0
    safe_to_merge: bool = True
    advisory_message: str = ""


# =============================================================================
# IMPACT SEVERITY SCORING
# =============================================================================


def _get_consumers_for_schema(schema_name: str, source_file: str) -> list[ConsumerInfo]:
    """Get consumer information for a schema.

    Args:
        schema_name: The schema name
        source_file: The schema source file path

    Returns:
        List of ConsumerInfo objects for downstream consumers
    """
    consumers: list[ConsumerInfo] = []

    # Normalize path
    source_normalized = source_file.replace("\\", "/")

    # Find matching consumers from DOWNSTREAM_DEPENDENCIES
    consumer_files = DOWNSTREAM_DEPENDENCIES.get(source_file, [])
    if not consumer_files:
        # Try normalized path lookup
        for dep_path, deps in DOWNSTREAM_DEPENDENCIES.items():
            if dep_path.replace("\\", "/") == source_normalized:
                consumer_files = deps
                break

    for consumer_file in sorted(consumer_files):
        usage_info = CONSUMER_FIELD_USAGE.get(consumer_file)
        if usage_info and usage_info.get("schema") == schema_name:
            fields_read = usage_info.get("fields_read", [])
            consumers.append(
                ConsumerInfo(
                    file_path=consumer_file,
                    field_access_count=len(fields_read),
                    fields_read=fields_read,
                )
            )
        else:
            # Consumer exists but field mapping unknown
            consumers.append(
                ConsumerInfo(
                    file_path=consumer_file,
                    field_access_count=-1,  # Unknown
                    fields_read=[],
                )
            )

    return consumers


def score_change_impact(change_brief: dict[str, Any]) -> dict[str, Any]:
    """
    Assigns a severity level (INFO/WARN/BLOCK) to each schema and overall.

    Rules:
    - BLOCK when:
        - Field removed from widely-used schema
        - Type change on critical fields (configured allowlist)
    - WARN when:
        - New fields added
        - Non-critical type widening
    - INFO for:
        - Cosmetic changes only (constraint additions, etc.)

    Args:
        change_brief: Dictionary from generate_schema_change_brief()

    Returns:
        Dictionary with formal impact contract:
        {
            "overall_severity": "BLOCK|WARN|INFO|NONE",
            "safe_to_merge": true|false,
            "per_schema": {
                "schema_name": {
                    "severity": "BLOCK|WARN|INFO",
                    "reasons": [...],
                    "consumers": [...]
                }
            }
        }
    """
    schema_impacts: list[SchemaImpact] = []

    for change in change_brief.get("schema_changes", []):
        schema_name = change["schema_name"]
        source_file = change.get("source_file", "")
        impact = _score_single_schema_impact(schema_name, change, source_file)
        schema_impacts.append(impact)

    # Compute overall severity
    block_count = sum(1 for i in schema_impacts if i.severity == Severity.BLOCK)
    warn_count = sum(1 for i in schema_impacts if i.severity == Severity.WARN)
    info_count = sum(1 for i in schema_impacts if i.severity == Severity.INFO)

    # Determine overall severity (NONE if no schemas changed)
    if not schema_impacts:
        overall_severity = Severity.INFO  # Use INFO for "no changes" case
        safe_to_merge = True
        advisory_message = "✅ NONE: No schema changes detected"
    elif block_count > 0:
        overall_severity = Severity.BLOCK
        safe_to_merge = False
        advisory_message = f"⛔ BLOCK: {block_count} schema(s) have breaking changes"
    elif warn_count > 0:
        overall_severity = Severity.WARN
        safe_to_merge = False
        advisory_message = f"⚠️ WARN: {warn_count} schema(s) have notable changes"
    else:
        overall_severity = Severity.INFO
        safe_to_merge = True
        advisory_message = "✅ INFO: Only cosmetic changes detected"

    summary = ImpactSummary(
        overall_severity=overall_severity,
        schema_impacts=schema_impacts,
        block_count=block_count,
        warn_count=warn_count,
        info_count=info_count,
        safe_to_merge=safe_to_merge,
        advisory_message=advisory_message,
    )

    return _impact_summary_to_dict(summary)


def _score_single_schema_impact(
    schema_name: str, change: dict[str, Any], source_file: str = ""
) -> SchemaImpact:
    """Score the impact of changes to a single schema.

    Args:
        schema_name: Name of the schema
        change: Change dictionary from change brief
        source_file: Schema source file path for consumer lookup

    Returns:
        SchemaImpact with severity, reasons, and consumers
    """
    reasons: list[str] = []
    blocked_fields: list[str] = []
    warned_fields: list[str] = []
    severity = Severity.INFO

    is_widely_used = schema_name in WIDELY_USED_SCHEMAS
    critical_fields = CRITICAL_FIELDS.get(schema_name, [])

    # Check field removals
    fields_removed = change.get("fields_removed", [])
    if fields_removed:
        if is_widely_used:
            severity = Severity.BLOCK
            for path, ftype in fields_removed:
                blocked_fields.append(path)
                reasons.append(f"Removed required field '{path}' from widely-used schema")
        else:
            if severity != Severity.BLOCK:
                severity = Severity.WARN
            for path, ftype in fields_removed:
                warned_fields.append(path)
                reasons.append(f"Field '{path}' removed")

    # Check type changes on critical fields
    fields_type_changed = change.get("fields_type_changed", [])
    for path, old_type, new_type in fields_type_changed:
        is_critical = any(
            path.startswith(cf) or cf.startswith(path) for cf in critical_fields
        )
        if is_critical:
            severity = Severity.BLOCK
            blocked_fields.append(path)
            reasons.append(
                f"Critical field '{path}' type changed: {old_type} → {new_type}"
            )
        else:
            # Type widening is less severe
            if severity == Severity.INFO:
                severity = Severity.WARN
            warned_fields.append(path)
            reasons.append(f"Field '{path}' type changed: {old_type} → {new_type}")

    # Check field additions (WARN level)
    fields_added = change.get("fields_added", [])
    if fields_added:
        if severity == Severity.INFO:
            severity = Severity.WARN
        for path, ftype in fields_added[:5]:  # Limit reasons
            warned_fields.append(path)
            reasons.append(f"New field '{path}' added ({ftype})")
        if len(fields_added) > 5:
            reasons.append(f"... and {len(fields_added) - 5} more fields added")

    # Check field renames (WARN level)
    fields_renamed = change.get("fields_renamed", [])
    for old_path, old_name, new_name in fields_renamed:
        if severity == Severity.INFO:
            severity = Severity.WARN
        warned_fields.append(old_path)
        reasons.append(f"Field renamed: '{old_name}' → '{new_name}'")

    # If no changes at all, mark as INFO
    if not reasons:
        reasons.append("No impactful changes detected")

    # Get consumers for this schema
    consumers = _get_consumers_for_schema(schema_name, source_file)

    return SchemaImpact(
        schema_name=schema_name,
        severity=severity,
        reasons=reasons,
        blocked_fields=blocked_fields,
        warned_fields=warned_fields,
        consumers=consumers,
    )


def _impact_summary_to_dict(summary: ImpactSummary) -> dict[str, Any]:
    """Convert ImpactSummary to formal contract dictionary.

    Returns format:
    {
        "overall_severity": "BLOCK|WARN|INFO|NONE",
        "safe_to_merge": true|false,
        "advisory_message": "...",
        "per_schema": {
            "schema_name": {
                "severity": "BLOCK|WARN|INFO",
                "reasons": [...],
                "consumers": [{"file_path": "...", "field_access_count": N}, ...]
            }
        }
    }
    """
    per_schema: dict[str, dict[str, Any]] = {}

    for impact in summary.schema_impacts:
        per_schema[impact.schema_name] = {
            "severity": impact.severity.value,
            "reasons": impact.reasons,
            "consumers": [
                {
                    "file_path": c.file_path,
                    "field_access_count": c.field_access_count,
                }
                for c in impact.consumers
            ],
        }

    return {
        "overall_severity": summary.overall_severity.value,
        "safe_to_merge": summary.safe_to_merge,
        "advisory_message": summary.advisory_message,
        "block_count": summary.block_count,
        "warn_count": summary.warn_count,
        "info_count": summary.info_count,
        "per_schema": per_schema,
    }


def is_schema_change_safe(change_brief: dict[str, Any]) -> bool:
    """
    Advisory heuristic: True if only INFO-level changes.

    Used only in logs; must not fail CI.
    Good for local pre-flight check.

    Args:
        change_brief: Dictionary from generate_schema_change_brief()

    Returns:
        True if all changes are INFO level (safe to merge), False otherwise
    """
    impact = score_change_impact(change_brief)
    return impact.get("safe_to_merge", False)


# =============================================================================
# FIELD IMPACT MATRIX
# =============================================================================


@dataclass
class FieldImpact:
    """Per-field impact classification."""

    schema: str
    field: str
    change_kind: str  # added, removed, type_changed, renamed
    severity: Severity
    consumers: list[ConsumerInfo] = field(default_factory=list)


def build_field_impact_matrix(
    change_brief: dict[str, Any], impact_result: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """
    Build a per-field impact matrix from change brief and impact result.

    Each entry contains:
    - schema: Schema name
    - field: Field path
    - change_kind: "added" | "removed" | "type_changed" | "renamed"
    - severity: "BLOCK" | "WARN" | "INFO"
    - consumers: List of consumers that read this field

    Args:
        change_brief: Dictionary from generate_schema_change_brief()
        impact_result: Optional pre-computed impact result (computed if None)

    Returns:
        List of field impact entries, sorted by (schema, field)
    """
    if impact_result is None:
        impact_result = score_change_impact(change_brief)

    field_impacts: list[FieldImpact] = []
    per_schema = impact_result.get("per_schema", {})

    for change in change_brief.get("schema_changes", []):
        schema_name = change["schema_name"]
        source_file = change.get("source_file", "")

        # Get schema-level info
        schema_info = per_schema.get(schema_name, {})
        schema_consumers = [
            ConsumerInfo(
                file_path=c.get("file_path", ""),
                field_access_count=c.get("field_access_count", -1),
                fields_read=c.get("fields_read", []),
            )
            for c in schema_info.get("consumers", [])
        ]

        # Determine which consumers read each field
        def get_field_consumers(field_path: str) -> list[ConsumerInfo]:
            """Get consumers that access a specific field."""
            matched: list[ConsumerInfo] = []
            for consumer in schema_consumers:
                # Check if consumer reads this field or a parent path
                usage_info = CONSUMER_FIELD_USAGE.get(consumer.file_path, {})
                fields_read = usage_info.get("fields_read", [])
                for read_path in fields_read:
                    if (
                        field_path.startswith(read_path)
                        or read_path.startswith(field_path)
                    ):
                        matched.append(consumer)
                        break
            return matched

        # Process field additions
        for path, ftype in change.get("fields_added", []):
            consumers = get_field_consumers(path)
            severity = Severity.WARN if consumers else Severity.INFO
            field_impacts.append(
                FieldImpact(
                    schema=schema_name,
                    field=path,
                    change_kind="added",
                    severity=severity,
                    consumers=consumers,
                )
            )

        # Process field removals
        is_widely_used = schema_name in WIDELY_USED_SCHEMAS
        for path, ftype in change.get("fields_removed", []):
            consumers = get_field_consumers(path)
            severity = Severity.BLOCK if is_widely_used else Severity.WARN
            field_impacts.append(
                FieldImpact(
                    schema=schema_name,
                    field=path,
                    change_kind="removed",
                    severity=severity,
                    consumers=consumers,
                )
            )

        # Process type changes
        critical_fields = CRITICAL_FIELDS.get(schema_name, [])
        for path, old_type, new_type in change.get("fields_type_changed", []):
            consumers = get_field_consumers(path)
            is_critical = any(
                path.startswith(cf) or cf.startswith(path) for cf in critical_fields
            )
            severity = Severity.BLOCK if is_critical else Severity.WARN
            field_impacts.append(
                FieldImpact(
                    schema=schema_name,
                    field=path,
                    change_kind="type_changed",
                    severity=severity,
                    consumers=consumers,
                )
            )

        # Process renames
        for old_path, old_name, new_name in change.get("fields_renamed", []):
            consumers = get_field_consumers(old_path)
            field_impacts.append(
                FieldImpact(
                    schema=schema_name,
                    field=old_path,
                    change_kind="renamed",
                    severity=Severity.WARN,
                    consumers=consumers,
                )
            )

    # Sort by (schema, field) for deterministic ordering
    field_impacts.sort(key=lambda fi: (fi.schema, fi.field))

    # Convert to dict format
    return [
        {
            "schema": fi.schema,
            "field": fi.field,
            "change_kind": fi.change_kind,
            "severity": fi.severity.value,
            "consumers": [
                {
                    "file_path": c.file_path,
                    "field_access_count": c.field_access_count,
                }
                for c in fi.consumers
            ],
        }
        for fi in field_impacts
    ]


# =============================================================================
# IMPACT MANIFEST
# =============================================================================


def build_impact_manifest(
    change_brief: dict[str, Any], impact_result: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Build a compact, stable impact manifest for use by other agents.

    The manifest provides a summary view of schema changes for interoperability
    with other agents (D1, D5, doc ops).

    Args:
        change_brief: Dictionary from generate_schema_change_brief()
        impact_result: Optional pre-computed impact result (computed if None)

    Returns:
        Stable, JSON-serializable manifest:
        {
            "schema_version": "1.0.0",
            "overall_severity": "BLOCK|WARN|INFO",
            "safe_to_merge": true|false,
            "schemas_touched": ["curriculum", ...],
            "fields_changed": ["curriculum.systems.pl.active", ...]
        }
    """
    if impact_result is None:
        impact_result = score_change_impact(change_brief)

    # Collect all changed fields with qualified paths
    fields_changed: list[str] = []
    for change in change_brief.get("schema_changes", []):
        schema_name = change["schema_name"]

        for path, _ in change.get("fields_added", []):
            fields_changed.append(f"{schema_name}.{path}")
        for path, _ in change.get("fields_removed", []):
            fields_changed.append(f"{schema_name}.{path}")
        for path, _, _ in change.get("fields_type_changed", []):
            fields_changed.append(f"{schema_name}.{path}")
        for path, _, _ in change.get("fields_renamed", []):
            fields_changed.append(f"{schema_name}.{path}")

    # Sort for deterministic ordering
    schemas_touched = sorted(change_brief.get("schemas_changed", []))
    fields_changed = sorted(set(fields_changed))

    return {
        "schema_version": "1.0.0",
        "overall_severity": impact_result.get("overall_severity", "INFO"),
        "safe_to_merge": impact_result.get("safe_to_merge", True),
        "schemas_touched": schemas_touched,
        "fields_changed": fields_changed,
    }


# =============================================================================
# REVIEWER CHECKLIST GENERATOR
# =============================================================================


def generate_review_checklist(
    change_brief: dict[str, Any],
    impact_result: dict[str, Any] | None = None,
    schema_filter: str | None = None,
) -> str:
    """
    Generate a Markdown reviewer checklist for schema changes.

    The checklist helps reviewers quickly identify what to inspect by listing
    fields that changed and which consumers use them.

    Args:
        change_brief: Dictionary from generate_schema_change_brief()
        impact_result: Optional pre-computed impact result
        schema_filter: Optional schema name to filter by

    Returns:
        Markdown-formatted checklist string
    """
    field_matrix = build_field_impact_matrix(change_brief, impact_result)

    # Apply schema filter if provided
    if schema_filter:
        field_matrix = [fi for fi in field_matrix if fi["schema"] == schema_filter]

    if not field_matrix:
        return "# Reviewer Checklist\n\nNo schema changes to review.\n"

    lines = [
        "# Reviewer Checklist",
        "",
        "Use this checklist to verify schema changes don't break downstream consumers.",
        "",
    ]

    # Group by schema for organization
    schemas: dict[str, list[dict[str, Any]]] = {}
    for fi in field_matrix:
        schema = fi["schema"]
        if schema not in schemas:
            schemas[schema] = []
        schemas[schema].append(fi)

    for schema_name in sorted(schemas.keys()):
        lines.append(f"## {schema_name}")
        lines.append("")

        for fi in schemas[schema_name]:
            field = fi["field"]
            change_kind = fi["change_kind"]
            severity = fi["severity"]
            consumers = fi["consumers"]

            # Format the change description
            severity_icon = {"BLOCK": "⛔", "WARN": "⚠️", "INFO": "ℹ️"}.get(
                severity, "❓"
            )
            kind_desc = {
                "added": "added",
                "removed": "removed",
                "type_changed": "type changed",
                "renamed": "renamed",
            }.get(change_kind, change_kind)

            lines.append(f"### `{field}` ({kind_desc}) {severity_icon}")
            lines.append("")

            if consumers:
                lines.append("**Consumers to check:**")
                lines.append("")
                for c in consumers:
                    fac = c.get("field_access_count", -1)
                    ref_count = f"{fac} references" if fac >= 0 else "references"
                    lines.append(
                        f"- [ ] Check uses of `{field}` in `{c['file_path']}` ({ref_count})"
                    )
            else:
                lines.append("- [ ] No known consumers (verify manually if needed)")
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# CROSS-SCHEMA IMPACT INDEX (Phase IV)
# =============================================================================


def build_cross_schema_impact_index(
    impact_manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a cross-schema impact index aggregating multiple schema impact manifests.

    PHASE IV — CROSS-SCHEMA IMPACT MAP
    Deterministic execution guaranteed.

    Args:
        impact_manifests: Sequence of impact manifest dictionaries from build_impact_manifest()

    Returns:
        Cross-schema index dictionary:
        {
            "schema_version": "1.0.0",
            "schemas_touched": ["curriculum", "metrics", ...],
            "blocking_schemas": ["curriculum", ...],
            "warn_schemas": ["metrics", ...],
            "change_density_by_schema": {
                "curriculum": 5,
                "metrics": 3
            }
        }
    """
    all_schemas: set[str] = set()
    blocking_schemas: set[str] = set()
    warn_schemas: set[str] = set()
    change_density: dict[str, int] = {}

    for manifest in impact_manifests:
        schemas_touched = manifest.get("schemas_touched", [])
        overall_severity = manifest.get("overall_severity", "INFO")
        fields_changed = manifest.get("fields_changed", [])

        for schema in schemas_touched:
            all_schemas.add(schema)

            # Count fields changed for this schema
            schema_field_count = sum(
                1 for field in fields_changed if field.startswith(f"{schema}.")
            )
            change_density[schema] = change_density.get(schema, 0) + schema_field_count

            # Categorize by severity
            if overall_severity == "BLOCK":
                blocking_schemas.add(schema)
            elif overall_severity == "WARN":
                warn_schemas.add(schema)

    # Remove schemas from warn_schemas if they're also blocking
    warn_schemas = warn_schemas - blocking_schemas

    return {
        "schema_version": "1.0.0",
        "schemas_touched": sorted(all_schemas),
        "blocking_schemas": sorted(blocking_schemas),
        "warn_schemas": sorted(warn_schemas),
        "change_density_by_schema": dict(sorted(change_density.items())),
    }


# =============================================================================
# REVIEWER PLAYBOOK SYNTHESIS (Phase IV)
# =============================================================================


def render_cross_schema_reviewer_playbook(
    impact_index: dict[str, Any],
    field_impact_matrices: list[list[dict[str, Any]]],
) -> str:
    """
    Render a cross-schema reviewer playbook in Markdown format.

    PHASE IV — REVIEWER PLAYBOOK
    Deterministic execution guaranteed.

    Args:
        impact_index: Cross-schema impact index from build_cross_schema_impact_index()
        field_impact_matrices: List of field impact matrices (one per schema manifest)

    Returns:
        Markdown-formatted playbook string
    """
    lines = [
        "# Cross-Schema Impact Review Playbook",
        "",
        "This playbook provides a structured view of schema changes across all affected schemas.",
        "",
    ]

    # Flatten all field matrices into a single list
    all_fields: list[dict[str, Any]] = []
    for matrix in field_impact_matrices:
        all_fields.extend(matrix)

    # Group by schema
    schemas_touched = impact_index.get("schemas_touched", [])
    blocking_schemas = set(impact_index.get("blocking_schemas", []))
    warn_schemas = set(impact_index.get("warn_schemas", []))

    for schema_name in schemas_touched:
        schema_fields = [f for f in all_fields if f["schema"] == schema_name]

        if not schema_fields:
            continue

        # Determine severity badge
        if schema_name in blocking_schemas:
            severity_badge = "⛔ BLOCK"
        elif schema_name in warn_schemas:
            severity_badge = "⚠️ WARN"
        else:
            severity_badge = "ℹ️ INFO"

        lines.append(f"## {schema_name} {severity_badge}")
        lines.append("")

        # Brief impact summary
        change_density = impact_index.get("change_density_by_schema", {}).get(
            schema_name, 0
        )
        lines.append(f"**Fields Changed:** {change_density}")
        lines.append("")

        # High-severity fields table
        high_severity_fields = [
            f for f in schema_fields if f["severity"] in ("BLOCK", "WARN")
        ]

        if high_severity_fields:
            lines.append("### High-Severity Fields")
            lines.append("")
            lines.append("| Schema.Field | Change Kind | Severity |")
            lines.append("|--------------|-------------|----------|")

            for field in sorted(high_severity_fields, key=lambda x: (x["severity"], x["field"])):
                qualified_field = f"{field['schema']}.{field['field']}"
                change_kind = field["change_kind"]
                severity = field["severity"]
                lines.append(f"| `{qualified_field}` | {change_kind} | {severity} |")

            lines.append("")

        # Checklist items for all fields
        lines.append("### Review Checklist")
        lines.append("")

        for field in sorted(schema_fields, key=lambda x: x["field"]):
            qualified_field = f"{field['schema']}.{field['field']}"
            change_kind = field["change_kind"]
            severity = field["severity"]
            consumers = field.get("consumers", [])

            severity_icon = {"BLOCK": "⛔", "WARN": "⚠️", "INFO": "ℹ️"}.get(
                severity, "❓"
            )

            if consumers:
                consumer_list = ", ".join(c["file_path"] for c in consumers[:3])
                if len(consumers) > 3:
                    consumer_list += f" (+{len(consumers) - 3} more)"
                lines.append(
                    f"- [ ] Review `{qualified_field}` ({change_kind}) {severity_icon} — affects: {consumer_list}"
                )
            else:
                lines.append(
                    f"- [ ] Review `{qualified_field}` ({change_kind}) {severity_icon}"
                )

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# DIRECTOR SCHEMA IMPACT PANEL (Phase IV)
# =============================================================================


def build_schema_impact_director_panel(
    impact_index: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a compact Director-facing schema impact panel.

    PHASE IV — DIRECTOR PANEL
    Deterministic execution guaranteed.

    Provides a high-level status view suitable for executive dashboards.

    Args:
        impact_index: Cross-schema impact index from build_cross_schema_impact_index()

    Returns:
        Director panel dictionary:
        {
            "status_light": "GREEN" | "YELLOW" | "RED",
            "blocking_schemas": ["curriculum", ...],
            "warn_schemas": ["metrics", ...],
            "headline": "short neutral sentence"
        }
    """
    blocking_schemas = impact_index.get("blocking_schemas", [])
    warn_schemas = impact_index.get("warn_schemas", [])
    schemas_touched = impact_index.get("schemas_touched", [])

    # Determine status light
    if blocking_schemas:
        status_light = "RED"
    elif warn_schemas:
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Build headline
    if blocking_schemas:
        headline = f"Schema changes detected: {len(blocking_schemas)} schema(s) with blocking changes"
    elif warn_schemas:
        headline = f"Schema changes detected: {len(warn_schemas)} schema(s) with notable changes"
    elif schemas_touched:
        headline = f"Schema changes detected: {len(schemas_touched)} schema(s) modified"
    else:
        headline = "No schema changes detected"

    return {
        "status_light": status_light,
        "blocking_schemas": sorted(blocking_schemas),
        "warn_schemas": sorted(warn_schemas),
        "headline": headline,
    }


# =============================================================================
# IMPACT SLO EVALUATOR (Phase V)
# =============================================================================


def evaluate_schema_change_slo(
    impact_index: dict[str, Any],
    *,
    max_blocking_schemas: int = 1,
    max_total_changes: int = 20,
) -> dict[str, Any]:
    """
    Evaluate schema changes against Service Level Objectives (SLOs).

    PHASE V — IMPACT SLOs
    Deterministic execution guaranteed.

    Args:
        impact_index: Cross-schema impact index from build_cross_schema_impact_index()
        max_blocking_schemas: Maximum number of blocking schemas allowed (default: 1)
        max_total_changes: Maximum total field changes allowed (default: 20)

    Returns:
        SLO evaluation dictionary:
        {
            "slo_ok": true|false,
            "status": "OK" | "ATTENTION" | "BLOCK",
            "neutral_reasons": ["reason1", "reason2", ...]
        }
    """
    blocking_schemas = impact_index.get("blocking_schemas", [])
    change_density = impact_index.get("change_density_by_schema", {})

    # Calculate total changes
    total_changes = sum(change_density.values())

    # Evaluate SLOs
    reasons: list[str] = []
    slo_ok = True
    status = "OK"

    # Check blocking schemas threshold
    blocking_count = len(blocking_schemas)
    if blocking_count > max_blocking_schemas:
        slo_ok = False
        status = "BLOCK"
        reasons.append(
            f"Blocking schemas count ({blocking_count}) exceeds threshold ({max_blocking_schemas})"
        )

    # Check total changes threshold
    if total_changes > max_total_changes:
        if status == "OK":
            status = "ATTENTION"
        slo_ok = False
        reasons.append(
            f"Total field changes ({total_changes}) exceeds threshold ({max_total_changes})"
        )

    # Additional attention indicators
    if blocking_count > 0 and status == "OK":
        status = "ATTENTION"
        reasons.append(f"{blocking_count} blocking schema(s) detected")

    if not reasons:
        reasons.append("All SLO thresholds met")

    return {
        "slo_ok": slo_ok,
        "status": status,
        "neutral_reasons": reasons,
    }


# =============================================================================
# OWNERSHIP ROUTING MAP (Phase V)
# =============================================================================


# Default ownership map for schema domains
DEFAULT_SCHEMA_OWNERSHIP_MAP: dict[str, list[str]] = {
    "curriculum": ["backend-team", "curriculum-owners"],
    "curriculum_uplift_phase2": ["backend-team", "curriculum-owners"],
    "metrics": ["metrics-team", "analytics-owners"],
    "evidence": ["docs-team", "evidence-owners"],
    "experiment_manifest_schema": ["experiments-team", "manifest-owners"],
    "production": ["rfl-team", "production-owners"],
    "allblue_lanes": ["allblue-team", "fleet-owners"],
    "manifest": ["experiments-team", "manifest-owners"],
}


def route_schema_changes_to_owners(
    field_matrix: list[dict[str, Any]],
    owner_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Route schema changes to appropriate code owners based on ownership map.

    PHASE V — OWNERSHIP ROUTING
    Deterministic execution guaranteed.

    Args:
        field_matrix: List of field impact entries from build_field_impact_matrix()
        owner_map: Optional ownership mapping (uses DEFAULT_SCHEMA_OWNERSHIP_MAP if None)
            Format: {schema_name: [owner1, owner2, ...]}

    Returns:
        Routing dictionary:
        {
            "owners_to_notify": {
                "owner1": ["schema1.field1", "schema2.field2", ...],
                "owner2": [...]
            },
            "status": "OK" | "ATTENTION",
            "neutral_notes": ["note1", "note2", ...]
        }
    """
    if owner_map is None:
        owner_map = DEFAULT_SCHEMA_OWNERSHIP_MAP

    # Build owner -> fields mapping
    owners_to_notify: dict[str, list[str]] = {}
    notes: list[str] = []

    for field_entry in field_matrix:
        schema = field_entry.get("schema", "")
        field = field_entry.get("field", "")
        qualified_field = f"{schema}.{field}"

        # Find owners for this schema
        owners = owner_map.get(schema, [])

        if not owners:
            # Schema not in ownership map
            notes.append(f"Schema '{schema}' has no assigned owners")
            # Assign to default owner
            default_owner = "unassigned"
            if default_owner not in owners_to_notify:
                owners_to_notify[default_owner] = []
            owners_to_notify[default_owner].append(qualified_field)
        else:
            # Add to all owners for this schema
            for owner in owners:
                if owner not in owners_to_notify:
                    owners_to_notify[owner] = []
                owners_to_notify[owner].append(qualified_field)

    # Sort fields for deterministic ordering
    for owner in owners_to_notify:
        owners_to_notify[owner] = sorted(owners_to_notify[owner])

    # Determine status
    unassigned_count = len(owners_to_notify.get("unassigned", []))
    if unassigned_count > 0:
        status = "ATTENTION"
        notes.append(f"{unassigned_count} field(s) have no assigned owners")
    else:
        status = "OK"
        if not notes:
            notes.append("All fields routed to assigned owners")

    return {
        "owners_to_notify": dict(sorted(owners_to_notify.items())),
        "status": status,
        "neutral_notes": notes,
    }


# =============================================================================
# SEMANTIC-CURRICULUM HARMONIC MAP (Phase VI)
# =============================================================================


def build_semantic_curriculum_harmonic_map(
    semantic_alignment: dict[str, Any],
    curriculum_alignment: dict[str, Any],
    atlas_coupling_view: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a harmonic map combining semantic, curriculum, and atlas alignment views.

    PHASE VI — SEMANTIC-CURRICULUM HARMONIC MAP
    Deterministic execution guaranteed.

    The harmonic map shows how well semantic alignment, curriculum alignment, and
    atlas coupling align across slices, identifying convergence bands and misaligned concepts.

    Args:
        semantic_alignment: Semantic alignment data (may have slice-level semantic_ok flags)
        curriculum_alignment: Curriculum alignment data (may have slice_alignment dict)
        atlas_coupling_view: Atlas coupling view from build_atlas_curriculum_coupling_view()

    Returns:
        Harmonic map dictionary:
        {
            "harmonic_scores": {"slice_name": float, ...},
            "convergence_band": "COHERENT" | "PARTIAL" | "MISMATCHED",
            "misaligned_concepts": ["slice1", ...],
            "neutral_notes": ["note1", ...]
        }
    """
    # Extract slice information from each view
    # Semantic alignment: may have per-slice semantic_ok flags
    semantic_slices: dict[str, bool] = {}
    if "slice_alignment" in semantic_alignment:
        for slice_name, status in semantic_alignment["slice_alignment"].items():
            semantic_slices[slice_name] = status in (True, "OK", "aligned", "semantic_ok")
    elif "per_slice_alignment" in semantic_alignment:
        for slice_name, slice_data in semantic_alignment["per_slice_alignment"].items():
            semantic_slices[slice_name] = slice_data.get("semantic_ok", False)

    # Curriculum alignment: may have slice_alignment dict
    curriculum_slices: dict[str, bool] = {}
    if "slice_alignment" in curriculum_alignment:
        for slice_name, status in curriculum_alignment["slice_alignment"].items():
            curriculum_slices[slice_name] = status in (True, "OK", "aligned", "curriculum_ok")
    elif "per_slice_alignment" in curriculum_alignment:
        for slice_name, slice_data in curriculum_alignment["per_slice_alignment"].items():
            curriculum_slices[slice_name] = slice_data.get("curriculum_ok", False)

    # Atlas coupling: has slices_with_atlas_support list
    atlas_supported = set(atlas_coupling_view.get("slices_with_atlas_support", []))

    # Collect all slices
    all_slices = set(semantic_slices.keys()) | set(curriculum_slices.keys()) | atlas_supported

    # Calculate harmonic scores per slice
    # Harmonic score = weighted average: semantic (0.4) + curriculum (0.4) + atlas (0.2)
    harmonic_scores: dict[str, float] = {}
    misaligned_concepts: list[str] = []
    notes: list[str] = []

    for slice_name in sorted(all_slices):
        semantic_ok = semantic_slices.get(slice_name, False)
        curriculum_ok = curriculum_slices.get(slice_name, False)
        atlas_ok = slice_name in atlas_supported

        # Calculate weighted harmonic score
        semantic_score = 1.0 if semantic_ok else 0.0
        curriculum_score = 1.0 if curriculum_ok else 0.0
        atlas_score = 1.0 if atlas_ok else 0.0

        harmonic_score = (0.4 * semantic_score + 0.4 * curriculum_score + 0.2 * atlas_score)
        harmonic_scores[slice_name] = round(harmonic_score, 3)

        # Identify misaligned concepts: semantic_ok=True but curriculum_ok=False
        if semantic_ok and not curriculum_ok:
            misaligned_concepts.append(slice_name)

    # Determine convergence band
    if not harmonic_scores:
        convergence_band = "MISMATCHED"
        notes.append("No slices found in alignment data")
    else:
        avg_score = sum(harmonic_scores.values()) / len(harmonic_scores)
        if avg_score >= 0.8:
            convergence_band = "COHERENT"
        elif avg_score >= 0.5:
            convergence_band = "PARTIAL"
        else:
            convergence_band = "MISMATCHED"

        notes.append(f"Average harmonic score: {avg_score:.3f} across {len(harmonic_scores)} slice(s)")

    if misaligned_concepts:
        notes.append(f"{len(misaligned_concepts)} concept(s) have semantic alignment but curriculum misalignment")

    return {
        "harmonic_scores": harmonic_scores,
        "convergence_band": convergence_band,
        "misaligned_concepts": sorted(misaligned_concepts),
        "neutral_notes": notes,
    }


# =============================================================================
# CURRICULUM EVOLUTION FORECASTER (Phase VI)
# =============================================================================


def build_curriculum_evolution_forecaster(
    harmonic_map: dict[str, Any],
    current_curriculum: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Forecast curriculum evolution needs based on harmonic map analysis.

    PHASE VI — CURRICULUM EVOLUTION FORECASTER
    Deterministic execution guaranteed.

    Predicts slice-level or concept-level adjustments needed to maintain consistency
    between semantic alignment, curriculum alignment, and atlas coupling.

    Args:
        harmonic_map: Harmonic map from build_semantic_curriculum_harmonic_map()
        current_curriculum: Optional current curriculum manifest for context

    Returns:
        Evolution forecast dictionary:
        {
            "forecasted_adjustments": [
                {
                    "slice": "slice_name",
                    "adjustment_kind": "semantic_sync" | "curriculum_sync" | "atlas_sync",
                    "priority": "HIGH" | "MEDIUM" | "LOW",
                    "neutral_reason": "..."
                }
            ],
            "forecast_status": "STABLE" | "EVOLVING" | "DIVERGING",
            "neutral_notes": [...]
        }
    """
    harmonic_scores = harmonic_map.get("harmonic_scores", {})
    misaligned_concepts = harmonic_map.get("misaligned_concepts", [])
    convergence_band = harmonic_map.get("convergence_band", "MISMATCHED")

    forecasted_adjustments: list[dict[str, Any]] = []
    notes: list[str] = []

    # Generate adjustments for misaligned concepts
    for slice_name in misaligned_concepts:
        score = harmonic_scores.get(slice_name, 0.0)

        # Determine adjustment kind based on score pattern
        if score < 0.4:
            adjustment_kind = "curriculum_sync"  # Low score suggests curriculum needs work
            priority = "HIGH"
            reason = f"Slice '{slice_name}' has low harmonic score ({score:.3f}), curriculum alignment needed"
        elif score < 0.7:
            adjustment_kind = "semantic_sync"
            priority = "MEDIUM"
            reason = f"Slice '{slice_name}' has moderate harmonic score ({score:.3f}), semantic alignment review recommended"
        else:
            adjustment_kind = "atlas_sync"
            priority = "LOW"
            reason = f"Slice '{slice_name}' has good alignment ({score:.3f}), atlas coupling could be improved"

        forecasted_adjustments.append({
            "slice": slice_name,
            "adjustment_kind": adjustment_kind,
            "priority": priority,
            "neutral_reason": reason,
        })

    # Determine forecast status
    if convergence_band == "COHERENT" and not misaligned_concepts:
        forecast_status = "STABLE"
        notes.append("Harmonic map shows coherent alignment across all slices")
    elif convergence_band == "PARTIAL" or misaligned_concepts:
        forecast_status = "EVOLVING"
        notes.append(f"Alignment is partial: {len(misaligned_concepts)} concept(s) need adjustment")
    else:
        forecast_status = "DIVERGING"
        notes.append("Significant misalignment detected across semantic, curriculum, and atlas views")

    # Sort adjustments by priority (HIGH first), then by slice name
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    forecasted_adjustments.sort(key=lambda a: (priority_order.get(a["priority"], 99), a["slice"]))

    return {
        "forecasted_adjustments": forecasted_adjustments,
        "forecast_status": forecast_status,
        "neutral_notes": notes,
    }


# =============================================================================
# HARMONIC DIRECTOR PANEL (Phase VI)
# =============================================================================


def build_harmonic_director_panel(
    harmonic_map: dict[str, Any],
    evolution_forecast: dict[str, Any] | None = None,
    d6_lattice_status: dict[str, Any] | None = None,
    c2_drift_risk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a Director-facing panel integrating harmonic map with D6 lattice and C2 drift risk.

    PHASE VI — HARMONIC DIRECTOR PANEL
    Deterministic execution guaranteed.

    Provides a unified view of semantic-curriculum alignment, evolution forecast,
    and integration with D6 lattice status and C2 drift risk signals.

    Args:
        harmonic_map: Harmonic map from build_semantic_curriculum_harmonic_map()
        evolution_forecast: Optional forecast from build_curriculum_evolution_forecaster()
        d6_lattice_status: Optional D6 lattice status (expected to have status_light or similar)
        c2_drift_risk: Optional C2 drift risk (expected to have risk_level or similar)

    Returns:
        Director panel dictionary:
        {
            "status_light": "GREEN" | "YELLOW" | "RED",
            "convergence_band": "COHERENT" | "PARTIAL" | "MISMATCHED",
            "forecast_status": "STABLE" | "EVOLVING" | "DIVERGING",
            "misaligned_count": N,
            "headline": "short neutral sentence",
            "integrated_risks": [...]
        }
    """
    convergence_band = harmonic_map.get("convergence_band", "MISMATCHED")
    misaligned_concepts = harmonic_map.get("misaligned_concepts", [])
    misaligned_count = len(misaligned_concepts)

    forecast_status = "UNKNOWN"
    if evolution_forecast:
        forecast_status = evolution_forecast.get("forecast_status", "UNKNOWN")

    # Determine status light
    # RED: MISMATCHED convergence OR DIVERGING forecast OR high drift risk
    # YELLOW: PARTIAL convergence OR EVOLVING forecast OR medium drift risk
    # GREEN: COHERENT convergence AND STABLE forecast AND low/no drift risk

    status_light = "GREEN"
    integrated_risks: list[str] = []

    if convergence_band == "MISMATCHED" or forecast_status == "DIVERGING":
        status_light = "RED"
        if convergence_band == "MISMATCHED":
            integrated_risks.append("Harmonic map shows mismatched alignment")
        if forecast_status == "DIVERGING":
            integrated_risks.append("Evolution forecast indicates diverging alignment")
    elif convergence_band == "PARTIAL" or forecast_status == "EVOLVING":
        status_light = "YELLOW"
        if convergence_band == "PARTIAL":
            integrated_risks.append("Harmonic map shows partial alignment")
        if forecast_status == "EVOLVING":
            integrated_risks.append("Evolution forecast indicates ongoing adjustments needed")

    # Integrate D6 lattice status if provided
    if d6_lattice_status:
        d6_status = d6_lattice_status.get("status_light") or d6_lattice_status.get("status", "")
        if d6_status == "RED":
            status_light = "RED"
            integrated_risks.append("D6 lattice status indicates issues")
        elif d6_status == "YELLOW" and status_light == "GREEN":
            status_light = "YELLOW"
            integrated_risks.append("D6 lattice status requires attention")

    # Integrate C2 drift risk if provided
    if c2_drift_risk:
        c2_risk = c2_drift_risk.get("risk_level") or c2_drift_risk.get("status", "")
        if c2_risk in ("HIGH", "BLOCK", "RED"):
            status_light = "RED"
            integrated_risks.append("C2 drift risk is elevated")
        elif c2_risk in ("MEDIUM", "WARN", "YELLOW") and status_light == "GREEN":
            status_light = "YELLOW"
            integrated_risks.append("C2 drift risk requires monitoring")

    # Build headline
    if status_light == "RED":
        headline = f"Semantic-curriculum alignment issues detected: {misaligned_count} misaligned concept(s)"
    elif status_light == "YELLOW":
        headline = f"Semantic-curriculum alignment is partial: {misaligned_count} concept(s) need attention"
    else:
        headline = "Semantic-curriculum alignment is coherent across all slices"

    return {
        "status_light": status_light,
        "convergence_band": convergence_band,
        "forecast_status": forecast_status,
        "misaligned_count": misaligned_count,
        "headline": headline,
        "integrated_risks": integrated_risks,
    }


# =============================================================================
# CONSUMER-CENTRIC VIEW
# =============================================================================


def explain_consumers(schema_file: str, ontology: SchemaOntology | None = None) -> str:
    """
    Generate a consumer-centric view for a schema file.

    Shows which runtime consumers use the schema and what fields they read.

    Args:
        schema_file: Path to schema file (e.g., "config/curriculum_uplift_phase2.yaml")
        ontology: Optional ontology to use (built if None)

    Returns:
        Plain text explanation of consumers
    """
    if ontology is None:
        ontology = build_ontology()

    # Normalize path separators
    schema_file_normalized = schema_file.replace("\\", "/")

    # Find the schema node by source file
    schema_node = None
    for node in ontology.nodes:
        node_source = node.source_file.replace("\\", "/")
        if node_source == schema_file_normalized or node_source.endswith(schema_file_normalized):
            schema_node = node
            break

    if schema_node is None:
        # Try matching by schema name
        schema_name = Path(schema_file).stem
        for node in ontology.nodes:
            if node.name == schema_name:
                schema_node = node
                break

    if schema_node is None:
        return f"Schema file '{schema_file}' not found.\n\nAvailable schema files:\n" + "\n".join(
            f"  - {n.source_file}" for n in ontology.nodes
        )

    lines = [
        f"Consumer Analysis: {schema_node.name}",
        f"{'=' * (len(schema_node.name) + 20)}",
        "",
        f"Source File: {schema_node.source_file}",
        f"Total Fields: {schema_node.field_count}",
        "",
    ]

    # Get all field paths for this schema
    all_field_paths = {f.path for f in schema_node.fields}

    # Find consumers
    consumers = DOWNSTREAM_DEPENDENCIES.get(schema_node.source_file, [])

    if not consumers:
        # Try with normalized path
        for dep_path in DOWNSTREAM_DEPENDENCIES:
            if dep_path.replace("\\", "/") == schema_node.source_file.replace("\\", "/"):
                consumers = DOWNSTREAM_DEPENDENCIES[dep_path]
                break

    if not consumers:
        lines.extend([
            "Consumers:",
            "-" * 10,
            "  (no known consumers)",
            "",
        ])
        return "\n".join(lines)

    lines.extend([
        "Consumers:",
        "-" * 10,
        "",
    ])

    for consumer in consumers:
        lines.append(f"📄 {consumer}")

        # Get field usage info
        usage_info = CONSUMER_FIELD_USAGE.get(consumer)
        if usage_info and usage_info.get("schema") == schema_node.name:
            fields_read = usage_info.get("fields_read", [])
            lines.append(f"   Fields Read: {len(fields_read)}")

            # Show which fields are read
            if fields_read:
                lines.append("   ├── Known Field Access:")
                for i, field_path in enumerate(fields_read):
                    prefix = "│   └──" if i == len(fields_read) - 1 else "│   ├──"
                    exists = any(
                        fp.startswith(field_path) or field_path.startswith(fp)
                        for fp in all_field_paths
                    )
                    status = "✓" if exists else "?"
                    lines.append(f"   {prefix} {status} {field_path}")
        else:
            lines.append("   Fields Read: (unknown - not mapped)")

        lines.append("")

    return "\n".join(lines)


def explain_consumers_with_changes(
    schema_file: str,
    change_brief: dict[str, Any] | None = None,
    ontology: SchemaOntology | None = None,
) -> str:
    """
    Show consumer view with change impact overlay.

    For each consumer, shows whether changed fields are read or ignored.

    Args:
        schema_file: Path to schema file
        change_brief: Change brief dict (generated if None)
        ontology: Optional ontology

    Returns:
        Plain text explanation with change impact
    """
    if ontology is None:
        ontology = build_ontology()

    base_output = explain_consumers(schema_file, ontology)

    if change_brief is None:
        return base_output

    # Find changes for this schema
    schema_name = Path(schema_file).stem
    schema_changes = None
    for change in change_brief.get("schema_changes", []):
        if change["schema_name"] == schema_name:
            schema_changes = change
            break

    if schema_changes is None:
        return base_output + "\n\n(No changes detected for this schema)"

    # Collect changed fields
    changed_fields: set[str] = set()
    for path, _ in schema_changes.get("fields_added", []):
        changed_fields.add(path)
    for path, _ in schema_changes.get("fields_removed", []):
        changed_fields.add(path)
    for path, _, _ in schema_changes.get("fields_type_changed", []):
        changed_fields.add(path)
    for path, _, _ in schema_changes.get("fields_renamed", []):
        changed_fields.add(path)

    if not changed_fields:
        return base_output + "\n\n(No field-level changes)"

    lines = [
        "",
        "Change Impact on Consumers:",
        "-" * 27,
        "",
    ]

    consumers = DOWNSTREAM_DEPENDENCIES.get(
        next(
            (n.source_file for n in ontology.nodes if n.name == schema_name),
            "",
        ),
        [],
    )

    for consumer in consumers:
        usage_info = CONSUMER_FIELD_USAGE.get(consumer)
        if not usage_info:
            lines.append(f"📄 {consumer}: (field usage unknown)")
            continue

        fields_read = set(usage_info.get("fields_read", []))

        # Check overlap
        affected_fields = []
        for changed in changed_fields:
            for read_field in fields_read:
                if changed.startswith(read_field) or read_field.startswith(changed):
                    affected_fields.append((changed, read_field))

        if affected_fields:
            lines.append(f"📄 {consumer}: ⚠️ AFFECTED")
            for changed, read in affected_fields[:3]:
                lines.append(f"   └── Changed '{changed}' impacts read of '{read}'")
            if len(affected_fields) > 3:
                lines.append(f"   └── ... and {len(affected_fields) - 3} more")
        else:
            lines.append(f"📄 {consumer}: ✅ Not affected by these changes")

        lines.append("")

    return base_output + "\n".join(lines)


def generate_schema_change_brief(
    base_ontology_path: Path | str | None,
    current_ontology: SchemaOntology | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Generate a schema change brief comparing baseline to current ontology.

    Args:
        base_ontology_path: Path to baseline ontology JSON, or None for default
        current_ontology: Current ontology (built if None)
        out_path: Output path for markdown brief (uses default if None)

    Returns:
        Dictionary representation of the change brief
    """
    # Load baseline
    if base_ontology_path is None:
        base_ontology_path = BASELINE_ONTOLOGY
    else:
        base_ontology_path = Path(base_ontology_path)

    baseline_nodes = _load_ontology_nodes(base_ontology_path)

    # Build current if not provided
    if current_ontology is None:
        current_ontology = build_ontology()

    # Compute blast radius for current
    blast_radius = compute_blast_radius(current_ontology.nodes)

    # Build maps
    baseline_map = {n.name: n for n in baseline_nodes} if baseline_nodes else {}
    current_map = {n.name: n for n in current_ontology.nodes}

    # Find changed schemas
    all_schemas = set(baseline_map.keys()) | set(current_map.keys())
    schema_changes: list[SchemaChange] = []
    schemas_changed: list[str] = []

    total_added = 0
    total_removed = 0
    total_modified = 0

    for schema_name in sorted(all_schemas):
        baseline_node = baseline_map.get(schema_name)
        current_node = current_map.get(schema_name)

        if baseline_node is None and current_node is not None:
            # New schema
            schemas_changed.append(schema_name)
            fields_added = [(f.path, f.field_type) for f in current_node.fields]
            total_added += len(fields_added)
            schema_changes.append(
                SchemaChange(
                    schema_name=schema_name,
                    source_file=current_node.source_file,
                    fields_added=fields_added,
                    fields_removed=[],
                    fields_renamed=[],
                    fields_type_changed=[],
                    downstream_consumers=blast_radius.get(schema_name, []),
                )
            )
        elif current_node is None and baseline_node is not None:
            # Removed schema
            schemas_changed.append(schema_name)
            fields_removed = [(f.path, f.field_type) for f in baseline_node.fields]
            total_removed += len(fields_removed)
            schema_changes.append(
                SchemaChange(
                    schema_name=schema_name,
                    source_file=baseline_node.source_file,
                    fields_added=[],
                    fields_removed=fields_removed,
                    fields_renamed=[],
                    fields_type_changed=[],
                    downstream_consumers=blast_radius.get(schema_name, []),
                )
            )
        elif baseline_node and current_node:
            # Compare fields
            change = _compute_schema_change(
                baseline_node, current_node, blast_radius.get(schema_name, [])
            )
            if (
                change.fields_added
                or change.fields_removed
                or change.fields_renamed
                or change.fields_type_changed
            ):
                schemas_changed.append(schema_name)
                schema_changes.append(change)
                total_added += len(change.fields_added)
                total_removed += len(change.fields_removed)
                total_modified += len(change.fields_renamed) + len(
                    change.fields_type_changed
                )

    # Build brief
    brief = SchemaChangeBrief(
        generated_at=datetime.now(timezone.utc).isoformat(),
        base_hash=_compute_ontology_hash(baseline_nodes) if baseline_nodes else "N/A",
        current_hash=_compute_ontology_hash(current_ontology.nodes),
        schemas_changed=schemas_changed,
        schema_changes=schema_changes,
        total_fields_added=total_added,
        total_fields_removed=total_removed,
        total_fields_modified=total_modified,
    )

    # Compute impact scoring
    brief_dict = _change_brief_to_dict(brief)
    impact = score_change_impact(brief_dict) if schema_changes else None

    # Generate markdown output
    if out_path is None:
        out_path = OUTPUT_CHANGE_BRIEF
    else:
        out_path = Path(out_path)

    md_content = _generate_change_brief_markdown(brief, impact)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")

    # Return dict representation with impact
    result = brief_dict
    if impact:
        result["impact"] = impact
    return result


def _compute_schema_change(
    baseline: SchemaNode, current: SchemaNode, consumers: list[str]
) -> SchemaChange:
    """Compute detailed changes between two schema versions."""
    baseline_fields = {f.path: f for f in baseline.fields}
    current_fields = {f.path: f for f in current.fields}

    fields_added: list[tuple[str, str]] = []
    fields_removed: list[tuple[str, str]] = []
    fields_renamed: list[tuple[str, str, str]] = []
    fields_type_changed: list[tuple[str, str, str]] = []

    # Track unmatched fields for rename detection
    removed_paths = set(baseline_fields.keys()) - set(current_fields.keys())
    added_paths = set(current_fields.keys()) - set(baseline_fields.keys())

    # Detect potential renames (same type, different path)
    rename_matches: set[str] = set()
    for old_path in list(removed_paths):
        old_field = baseline_fields[old_path]
        for new_path in list(added_paths):
            new_field = current_fields[new_path]
            if old_field.field_type == new_field.field_type:
                # Potential rename - check if field names are similar-ish
                old_name = old_path.split(".")[-1]
                new_name = new_path.split(".")[-1]
                # Consider it a rename if parent paths match
                old_parent = ".".join(old_path.split(".")[:-1])
                new_parent = ".".join(new_path.split(".")[:-1])
                if old_parent == new_parent:
                    fields_renamed.append((old_path, old_name, new_name))
                    removed_paths.discard(old_path)
                    added_paths.discard(new_path)
                    rename_matches.add(old_path)
                    rename_matches.add(new_path)
                    break

    # Remaining are true adds/removes
    for path in sorted(removed_paths):
        fields_removed.append((path, baseline_fields[path].field_type))
    for path in sorted(added_paths):
        fields_added.append((path, current_fields[path].field_type))

    # Check type changes on common paths
    for path in sorted(set(baseline_fields.keys()) & set(current_fields.keys())):
        old_type = baseline_fields[path].field_type
        new_type = current_fields[path].field_type
        if old_type != new_type:
            fields_type_changed.append((path, old_type, new_type))

    return SchemaChange(
        schema_name=current.name,
        source_file=current.source_file,
        fields_added=fields_added,
        fields_removed=fields_removed,
        fields_renamed=fields_renamed,
        fields_type_changed=fields_type_changed,
        downstream_consumers=consumers,
    )


def _load_ontology_nodes(path: Path) -> list[SchemaNode] | None:
    """Load schema nodes from an ontology JSON file."""
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes: list[SchemaNode] = []

        for node_data in data.get("nodes", []):
            fields = [
                SchemaField(
                    name=f["name"],
                    path=f["path"],
                    field_type=f["field_type"],
                    cluster=FieldCluster(f.get("cluster", "unknown")),
                    constraints=f.get("constraints", []),
                )
                for f in node_data.get("fields", [])
            ]

            nodes.append(
                SchemaNode(
                    name=node_data["name"],
                    source_file=node_data["source_file"],
                    schema_type=node_data["schema_type"],
                    version=node_data.get("version", ""),
                    field_count=node_data.get("field_count", len(fields)),
                    fields=fields,
                    cluster_distribution=node_data.get("cluster_distribution", {}),
                    raw_hash=node_data.get("raw_hash", ""),
                )
            )

        return nodes
    except Exception:
        return None


def _compute_ontology_hash(nodes: list[SchemaNode] | None) -> str:
    """Compute a hash representing the ontology state."""
    if not nodes:
        return "empty"
    content = json.dumps(
        sorted([n.raw_hash for n in nodes]), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _generate_change_brief_markdown(
    brief: SchemaChangeBrief, impact: dict[str, Any] | None = None
) -> str:
    """Generate markdown content for the change brief."""
    lines = [
        "<!-- AUTO-GENERATED — DO NOT EDIT BY HAND -->",
        "<!-- Generated by scripts/schema_ontology_builder.py -->",
        f"<!-- Timestamp: {brief.generated_at} -->",
        "",
        "# Schema Change Brief",
        "",
        "This document summarizes structural changes to schema files for PR review.",
        "It describes **what** changed and estimates potential impact.",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Schemas Changed | {len(brief.schemas_changed)} |",
        f"| Fields Added | {brief.total_fields_added} |",
        f"| Fields Removed | {brief.total_fields_removed} |",
        f"| Fields Modified | {brief.total_fields_modified} |",
        f"| Base Hash | `{brief.base_hash}` |",
        f"| Current Hash | `{brief.current_hash}` |",
        "",
    ]

    if not brief.schemas_changed:
        lines.extend([
            "✅ **No schema changes detected.**",
            "",
        ])
        return "\n".join(lines)

    # Impact Summary section (if impact scoring available)
    if impact:
        overall_severity = impact.get("overall_severity", "INFO")
        safe_to_merge = impact.get("safe_to_merge", True)
        advisory = impact.get("advisory_message", "")

        severity_icons = {"INFO": "ℹ️", "WARN": "⚠️", "BLOCK": "⛔"}
        icon = severity_icons.get(overall_severity, "❓")

        lines.extend([
            "## Impact Summary",
            "",
            f"**Overall Severity:** {icon} **{overall_severity}**",
            "",
            f"**Safe to Merge (Advisory):** {'✅ Yes' if safe_to_merge else '❌ No'}",
            "",
            f"> {advisory}",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| ⛔ BLOCK | {impact.get('block_count', 0)} |",
            f"| ⚠️ WARN | {impact.get('warn_count', 0)} |",
            f"| ℹ️ INFO | {impact.get('info_count', 0)} |",
            "",
        ])

        # Per-schema impact details
        per_schema = impact.get("per_schema", {})
        if per_schema:
            lines.extend([
                "### Per-Schema Impact",
                "",
            ])
            for schema_name, si in sorted(per_schema.items()):
                sev = si.get("severity", "INFO")
                icon = severity_icons.get(sev, "❓")
                lines.append(f"#### {icon} {schema_name} — {sev}")
                lines.append("")

                # Reasons
                reasons = si.get("reasons", [])
                if reasons:
                    lines.append("**Reasons:**")
                    for reason in reasons[:10]:
                        lines.append(f"- {reason}")
                    if len(reasons) > 10:
                        lines.append(f"- ... and {len(reasons) - 10} more")
                    lines.append("")

                # Consumers
                consumers = si.get("consumers", [])
                if consumers:
                    lines.append("**Consumers:**")
                    for c in consumers[:10]:
                        fac = c.get("field_access_count", -1)
                        fac_str = f"{fac} fields" if fac >= 0 else "unknown"
                        lines.append(f"- `{c.get('file_path', 'unknown')}` ({fac_str})")
                    if len(consumers) > 10:
                        lines.append(f"- ... and {len(consumers) - 10} more consumers")
                    lines.append("")

    # Schemas Changed section
    lines.extend([
        "## Schemas Changed",
        "",
    ])
    for name in brief.schemas_changed:
        lines.append(f"- `{name}`")
    lines.append("")

    # Field-Level Changes section
    lines.extend([
        "## Field-Level Changes",
        "",
    ])

    for change in brief.schema_changes:
        lines.append(f"### {change.schema_name}")
        lines.append("")
        lines.append(f"**Source:** `{change.source_file}`")
        lines.append("")

        if change.fields_added:
            lines.append("**Fields Added:**")
            lines.append("")
            lines.append("| Path | Type |")
            lines.append("|------|------|")
            for path, ftype in change.fields_added[:20]:  # Limit display
                lines.append(f"| `{path}` | `{ftype}` |")
            if len(change.fields_added) > 20:
                lines.append(f"| ... | (+{len(change.fields_added) - 20} more) |")
            lines.append("")

        if change.fields_removed:
            lines.append("**Fields Removed:**")
            lines.append("")
            lines.append("| Path | Type |")
            lines.append("|------|------|")
            for path, ftype in change.fields_removed[:20]:
                lines.append(f"| `{path}` | `{ftype}` |")
            if len(change.fields_removed) > 20:
                lines.append(f"| ... | (+{len(change.fields_removed) - 20} more) |")
            lines.append("")

        if change.fields_renamed:
            lines.append("**Fields Renamed:**")
            lines.append("")
            lines.append("| Path | Old Name | New Name |")
            lines.append("|------|----------|----------|")
            for path, old_name, new_name in change.fields_renamed:
                lines.append(f"| `{path}` | `{old_name}` | `{new_name}` |")
            lines.append("")

        if change.fields_type_changed:
            lines.append("**Type Changes:**")
            lines.append("")
            lines.append("| Path | Old Type | New Type |")
            lines.append("|------|----------|----------|")
            for path, old_type, new_type in change.fields_type_changed:
                lines.append(f"| `{path}` | `{old_type}` | `{new_type}` |")
            lines.append("")

    # Affected Components section
    lines.extend([
        "## Affected Components",
        "",
        "Files that may be impacted by these schema changes:",
        "",
    ])

    has_consumers = False
    for change in brief.schema_changes:
        if change.downstream_consumers:
            has_consumers = True
            lines.append(f"### {change.schema_name}")
            lines.append("")
            for consumer in change.downstream_consumers:
                lines.append(f"- `{consumer}`")
            lines.append("")

    if not has_consumers:
        lines.append("*No downstream consumers identified.*")
        lines.append("")

    # Notes section
    lines.extend([
        "## Notes",
        "",
    ])
    if brief.notes:
        for note in brief.notes:
            lines.append(f"- {note}")
    else:
        lines.append("*No additional notes.*")
    lines.append("")

    return "\n".join(lines)


def _change_brief_to_dict(brief: SchemaChangeBrief) -> dict[str, Any]:
    """Convert change brief to dictionary."""
    return {
        "generated_at": brief.generated_at,
        "base_hash": brief.base_hash,
        "current_hash": brief.current_hash,
        "schemas_changed": brief.schemas_changed,
        "total_fields_added": brief.total_fields_added,
        "total_fields_removed": brief.total_fields_removed,
        "total_fields_modified": brief.total_fields_modified,
        "schema_changes": [
            {
                "schema_name": c.schema_name,
                "source_file": c.source_file,
                "fields_added": c.fields_added,
                "fields_removed": c.fields_removed,
                "fields_renamed": c.fields_renamed,
                "fields_type_changed": c.fields_type_changed,
                "downstream_consumers": c.downstream_consumers,
            }
            for c in brief.schema_changes
        ],
        "notes": brief.notes,
    }


# =============================================================================
# SCHEMA EXPLAIN
# =============================================================================


def explain_schema(schema_name: str, ontology: SchemaOntology | None = None) -> str:
    """
    Generate a plain-text explanation of a schema.

    Args:
        schema_name: Name of the schema to explain
        ontology: Ontology to use (built if None)

    Returns:
        Plain text explanation string
    """
    if ontology is None:
        ontology = build_ontology()

    # Find the schema
    node = None
    for n in ontology.nodes:
        if n.name == schema_name or n.name.lower() == schema_name.lower():
            node = n
            break

    if node is None:
        return f"Schema '{schema_name}' not found.\n\nAvailable schemas:\n" + "\n".join(
            f"  - {n.name}" for n in ontology.nodes
        )

    lines = [
        f"Schema: {node.name}",
        f"{'=' * (len(node.name) + 8)}",
        "",
        f"Source File: {node.source_file}",
        f"Type: {node.schema_type}",
        f"Version: {node.version or 'N/A'}",
        f"Field Count: {node.field_count}",
        f"Hash: {node.raw_hash[:16]}",
        "",
    ]

    # Relationships
    lines.append("Relationships:")
    lines.append("-" * 14)

    rels_found = False
    for rel in ontology.relationships:
        if rel.source_schema == node.name:
            rels_found = True
            lines.append(
                f"  → {rel.relation_type.value} {rel.target_schema} ({rel.confidence:.0%})"
            )
        elif rel.target_schema == node.name:
            rels_found = True
            lines.append(
                f"  ← {rel.relation_type.value} from {rel.source_schema} ({rel.confidence:.0%})"
            )

    if not rels_found:
        lines.append("  (none)")

    lines.append("")

    # Field clusters
    lines.append("Field Clusters:")
    lines.append("-" * 15)
    if node.cluster_distribution:
        for cluster, count in sorted(
            node.cluster_distribution.items(), key=lambda x: -x[1]
        ):
            bar = "█" * min(count, 20)
            lines.append(f"  {cluster:15} {count:3} {bar}")
    else:
        lines.append("  (no fields)")

    lines.append("")

    # Known consumers
    lines.append("Known Consumers:")
    lines.append("-" * 16)
    consumers = ontology.blast_radius.get(node.name, [])
    if consumers:
        for consumer in consumers:
            lines.append(f"  - {consumer}")
    else:
        lines.append("  (none identified)")

    lines.append("")

    return "\n".join(lines)


# =============================================================================
# OUTPUT GENERATION
# =============================================================================


def generate_dot_graph(ontology: SchemaOntology) -> str:
    """Generate Graphviz DOT representation of the ontology."""
    lines = [
        "// AUTO-GENERATED — DO NOT EDIT BY HAND",
        "// Generated by scripts/schema_ontology_builder.py",
        f"// Timestamp: {ontology.generated_at}",
        "",
        "digraph SchemaOntology {",
        '    rankdir=LR;',
        '    node [shape=record, fontname="Helvetica", fontsize=10];',
        '    edge [fontname="Helvetica", fontsize=8];',
        "",
    ]

    # Define cluster colors
    cluster_colors = {
        "seed": "#E8F5E9",
        "metric": "#E3F2FD",
        "success": "#FFF3E0",
        "governance": "#FCE4EC",
        "identity": "#F3E5F5",
        "temporal": "#E0F7FA",
        "structural": "#FFF8E1",
        "configuration": "#ECEFF1",
        "unknown": "#FAFAFA",
    }

    # Generate nodes
    lines.append("    // Schema nodes")
    for node in ontology.nodes:
        # Compute dominant cluster for coloring
        dominant_cluster = max(
            node.cluster_distribution.items(),
            key=lambda x: x[1],
            default=("unknown", 0),
        )[0]
        color = cluster_colors.get(dominant_cluster, "#FAFAFA")

        label_parts = [
            f"{node.name}|",
            f"type: {node.schema_type}\\l",
            f"fields: {node.field_count}\\l",
        ]
        if node.version:
            label_parts.append(f"version: {node.version}\\l")

        label = "".join(label_parts)
        lines.append(
            f'    {_safe_node_id(node.name)} [label="{{{label}}}", '
            f'style=filled, fillcolor="{color}"];'
        )

    lines.append("")
    lines.append("    // Relationships")

    # Edge styles for different relationships
    edge_styles = {
        RelationType.EQUIVALENT: 'style=bold, color="#4CAF50"',
        RelationType.EXTENDS: 'style=solid, color="#2196F3"',
        RelationType.EMBEDS: 'style=dashed, color="#9C27B0"',
        RelationType.REFERENCES: 'style=dotted, color="#FF9800"',
        RelationType.SIBLING: 'style=dotted, color="#607D8B"',
    }

    for rel in ontology.relationships:
        if rel.relation_type == RelationType.NONE:
            continue
        style = edge_styles.get(rel.relation_type, "")
        label = f"{rel.relation_type.value} ({rel.confidence:.0%})"
        lines.append(
            f'    {_safe_node_id(rel.source_schema)} -> '
            f'{_safe_node_id(rel.target_schema)} [{style}, label="{label}"];'
        )

    lines.extend(["", "}"])
    return "\n".join(lines)


def _safe_node_id(name: str) -> str:
    """Convert schema name to safe DOT node ID."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def generate_ontology_json(ontology: SchemaOntology) -> str:
    """Generate JSON representation with hash commitment."""
    # Convert to dict, handling enums
    data = _ontology_to_dict(ontology)

    # Compute content hash (excluding the hash field itself)
    content_for_hash = json.dumps(data, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()

    # Add hash commitment
    data["content_hash"] = content_hash

    return json.dumps(data, indent=2, sort_keys=True)


def _ontology_to_dict(ontology: SchemaOntology) -> dict[str, Any]:
    """Convert ontology to dictionary, handling enums."""
    result: dict[str, Any] = {
        "version": ontology.version,
        "generated_at": ontology.generated_at,
        "commit_hash": ontology.commit_hash,
        "nodes": [],
        "relationships": [],
        "drifts": [],
        "blast_radius": ontology.blast_radius,
    }

    for node in ontology.nodes:
        node_dict = {
            "name": node.name,
            "source_file": node.source_file,
            "schema_type": node.schema_type,
            "version": node.version,
            "field_count": node.field_count,
            "cluster_distribution": node.cluster_distribution,
            "raw_hash": node.raw_hash,
            "fields": [
                {
                    "name": f.name,
                    "path": f.path,
                    "field_type": f.field_type,
                    "cluster": f.cluster.value,
                    "constraints": f.constraints,
                }
                for f in node.fields
            ],
        }
        result["nodes"].append(node_dict)

    for rel in ontology.relationships:
        result["relationships"].append(
            {
                "source_schema": rel.source_schema,
                "target_schema": rel.target_schema,
                "relation_type": rel.relation_type.value,
                "confidence": rel.confidence,
                "shared_fields": rel.shared_fields,
                "evidence": rel.evidence,
            }
        )

    for drift in ontology.drifts:
        result["drifts"].append(
            {
                "schema_name": drift.schema_name,
                "drift_type": drift.drift_type.value,
                "field_path": drift.field_path,
                "old_value": drift.old_value,
                "new_value": drift.new_value,
                "severity": drift.severity,
                "description": drift.description,
            }
        )

    return result


def generate_regression_report(ontology: SchemaOntology) -> str:
    """Generate markdown regression report."""
    lines = [
        "<!-- AUTO-GENERATED — DO NOT EDIT BY HAND -->",
        "<!-- Generated by scripts/schema_ontology_builder.py -->",
        f"<!-- Timestamp: {ontology.generated_at} -->",
        "",
        "# Schema Regression Report",
        "",
    ]

    if not ontology.drifts:
        lines.extend([
            "✅ **No schema regressions detected.**",
            "",
            "All schemas match baseline structure.",
            "",
        ])
    else:
        # Group by severity
        errors = [d for d in ontology.drifts if d.severity == "error"]
        warnings = [d for d in ontology.drifts if d.severity == "warning"]
        infos = [d for d in ontology.drifts if d.severity == "info"]

        lines.extend([
            f"## Summary",
            "",
            f"| Severity | Count |",
            f"|----------|-------|",
            f"| 🔴 Error | {len(errors)} |",
            f"| 🟡 Warning | {len(warnings)} |",
            f"| 🔵 Info | {len(infos)} |",
            "",
        ])

        if errors:
            lines.extend([
                "## 🔴 Errors (Breaking Changes)",
                "",
                "| Schema | Drift Type | Field | Description |",
                "|--------|------------|-------|-------------|",
            ])
            for d in errors:
                lines.append(
                    f"| {d.schema_name} | {d.drift_type.value} | `{d.field_path}` | {d.description} |"
                )
            lines.append("")

        if warnings:
            lines.extend([
                "## 🟡 Warnings (Potential Issues)",
                "",
                "| Schema | Drift Type | Field | Old | New |",
                "|--------|------------|-------|-----|-----|",
            ])
            for d in warnings:
                lines.append(
                    f"| {d.schema_name} | {d.drift_type.value} | `{d.field_path}` | `{d.old_value}` | `{d.new_value}` |"
                )
            lines.append("")

        if infos:
            lines.extend([
                "## 🔵 Info (Non-Breaking Changes)",
                "",
                "| Schema | Drift Type | Field | Description |",
                "|--------|------------|-------|-------------|",
            ])
            for d in infos:
                lines.append(
                    f"| {d.schema_name} | {d.drift_type.value} | `{d.field_path}` | {d.description} |"
                )
            lines.append("")

    # Blast radius section
    lines.extend([
        "## Schema Blast Radius",
        "",
        "Files potentially affected by schema changes:",
        "",
    ])

    for schema_name, affected in sorted(ontology.blast_radius.items()):
        if affected:
            lines.append(f"### {schema_name}")
            lines.append("")
            for f in affected:
                lines.append(f"- `{f}`")
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# BASELINE MANAGEMENT
# =============================================================================


def load_baseline() -> list[SchemaNode] | None:
    """Load baseline ontology for regression detection."""
    if not BASELINE_ONTOLOGY.exists():
        return None

    try:
        data = json.loads(BASELINE_ONTOLOGY.read_text(encoding="utf-8"))
        nodes: list[SchemaNode] = []

        for node_data in data.get("nodes", []):
            fields = [
                SchemaField(
                    name=f["name"],
                    path=f["path"],
                    field_type=f["field_type"],
                    cluster=FieldCluster(f.get("cluster", "unknown")),
                    constraints=f.get("constraints", []),
                )
                for f in node_data.get("fields", [])
            ]

            nodes.append(
                SchemaNode(
                    name=node_data["name"],
                    source_file=node_data["source_file"],
                    schema_type=node_data["schema_type"],
                    version=node_data.get("version", ""),
                    field_count=node_data.get("field_count", len(fields)),
                    fields=fields,
                    cluster_distribution=node_data.get("cluster_distribution", {}),
                    raw_hash=node_data.get("raw_hash", ""),
                )
            )

        return nodes
    except Exception:
        return None


def save_baseline(ontology: SchemaOntology) -> None:
    """Save current ontology as baseline."""
    BASELINE_ONTOLOGY.parent.mkdir(parents=True, exist_ok=True)
    content = generate_ontology_json(ontology)
    BASELINE_ONTOLOGY.write_text(content, encoding="utf-8")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def collect_schema_nodes() -> list[SchemaNode]:
    """Collect all schema nodes from configured sources."""
    nodes: list[SchemaNode] = []

    for category in SCHEMA_SOURCES.values():
        for rel_path in category:
            filepath = PROJECT_ROOT / rel_path
            node = parse_schema_file(filepath)
            if node:
                nodes.append(node)

    return nodes


def build_ontology() -> SchemaOntology:
    """Build complete schema ontology."""
    nodes = collect_schema_nodes()
    relationships = infer_relationships(nodes)
    baseline = load_baseline()
    drifts = detect_drifts(nodes, baseline)
    blast_radius = compute_blast_radius(nodes)

    return SchemaOntology(
        version="1.0.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        commit_hash="",  # Could be populated from git
        nodes=nodes,
        relationships=relationships,
        drifts=drifts,
        blast_radius=blast_radius,
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build schema ontology with relationship inference and drift detection."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI mode: fail if regressions detected",
    )
    parser.add_argument(
        "--blast-radius",
        action="store_true",
        help="Only compute and display blast radius",
    )
    parser.add_argument(
        "--regression-check",
        action="store_true",
        help="Only check for regressions against baseline",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current ontology as baseline",
    )
    parser.add_argument(
        "--change-brief",
        action="store_true",
        help="Generate schema change brief comparing to baseline",
    )
    parser.add_argument(
        "--change-brief-output",
        type=str,
        default=None,
        help="Output path for change brief (default: docs/SCHEMA_CHANGE_BRIEF.md)",
    )
    parser.add_argument(
        "--explain",
        type=str,
        metavar="SCHEMA_NAME",
        help="Explain a specific schema (relationships, clusters, consumers)",
    )
    parser.add_argument(
        "--explain-consumers",
        type=str,
        metavar="SCHEMA_FILE",
        help="Consumer-centric view for a schema file (e.g., config/curriculum.yaml)",
    )
    parser.add_argument(
        "--safe-to-merge",
        action="store_true",
        help="Advisory check: returns 0 if only INFO-level changes (does not fail CI)",
    )
    parser.add_argument(
        "--review-checklist",
        action="store_true",
        help="Generate Markdown reviewer checklist for schema changes",
    )
    parser.add_argument(
        "--field-matrix",
        action="store_true",
        help="Output per-field impact matrix as JSON",
    )
    parser.add_argument(
        "--impact-manifest",
        action="store_true",
        help="Output compact impact manifest for other agents",
    )
    parser.add_argument(
        "--schema",
        type=str,
        metavar="SCHEMA_NAME",
        help="Filter --review-checklist to a specific schema",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    args = parser.parse_args()

    if yaml is None:
        print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        return 1

    # Handle --explain mode (early exit)
    if args.explain:
        print(explain_schema(args.explain))
        return 0

    # Handle --explain-consumers mode (early exit)
    if args.explain_consumers:
        print(explain_consumers(args.explain_consumers))
        return 0

    print("Building schema ontology...")
    ontology = build_ontology()
    print(f"Found {len(ontology.nodes)} schemas")
    print(f"Inferred {len(ontology.relationships)} relationships")
    print(f"Detected {len(ontology.drifts)} drifts")

    # Handle --change-brief mode
    if args.change_brief or args.safe_to_merge:
        print("\nGenerating schema change brief...")
        out_path = args.change_brief_output if args.change_brief else None
        brief_data = generate_schema_change_brief(
            base_ontology_path=None,
            current_ontology=ontology,
            out_path=out_path if args.change_brief else None,
        )

        if args.change_brief:
            output_file = out_path or OUTPUT_CHANGE_BRIEF
            print(f"Wrote {output_file}")

        print(f"\nChange Summary:")
        print(f"  Schemas changed: {len(brief_data['schemas_changed'])}")
        print(f"  Fields added: {brief_data['total_fields_added']}")
        print(f"  Fields removed: {brief_data['total_fields_removed']}")
        print(f"  Fields modified: {brief_data['total_fields_modified']}")

        # Show impact summary
        impact = brief_data.get("impact")
        if impact:
            print(f"\nImpact Summary:")
            print(f"  Overall Severity: {impact['overall_severity']}")
            print(f"  {impact['advisory_message']}")
            print(f"  BLOCK: {impact['block_count']}, WARN: {impact['warn_count']}, INFO: {impact['info_count']}")

        # Handle --safe-to-merge advisory check
        if args.safe_to_merge:
            safe = is_schema_change_safe(brief_data)
            if safe:
                print("\n✅ Safe to merge (advisory): Only INFO-level changes detected")
            else:
                print("\n⚠️ Review recommended: WARN or BLOCK level changes detected")
            # Note: does NOT fail CI, always returns 0
            return 0

        return 0

    # Handle --review-checklist
    if args.review_checklist:
        print("\nGenerating reviewer checklist...")
        brief_data = generate_schema_change_brief(
            base_ontology_path=None,
            current_ontology=ontology,
            out_path=None,  # Don't write file
        )
        checklist = generate_review_checklist(
            brief_data, schema_filter=args.schema
        )
        print(checklist)
        return 0

    # Handle --field-matrix
    if args.field_matrix:
        print("\nGenerating field impact matrix...")
        brief_data = generate_schema_change_brief(
            base_ontology_path=None,
            current_ontology=ontology,
            out_path=None,
        )
        matrix = build_field_impact_matrix(brief_data)
        print(json.dumps(matrix, indent=2))
        return 0

    # Handle --impact-manifest
    if args.impact_manifest:
        print("\nGenerating impact manifest...")
        brief_data = generate_schema_change_brief(
            base_ontology_path=None,
            current_ontology=ontology,
            out_path=None,
        )
        manifest = build_impact_manifest(brief_data)
        print(json.dumps(manifest, indent=2))
        return 0

    if args.verbose:
        print("\nSchema Nodes:")
        for node in ontology.nodes:
            print(f"  - {node.name} ({node.schema_type}): {node.field_count} fields")
            print(f"    Clusters: {node.cluster_distribution}")

        print("\nRelationships:")
        for rel in ontology.relationships:
            if rel.relation_type != RelationType.NONE:
                print(
                    f"  - {rel.source_schema} --[{rel.relation_type.value}]--> "
                    f"{rel.target_schema} ({rel.confidence:.1%})"
                )

    if args.blast_radius:
        print("\nBlast Radius:")
        for schema, affected in sorted(ontology.blast_radius.items()):
            print(f"  {schema}:")
            for f in affected:
                print(f"    - {f}")
        return 0

    if args.regression_check or args.check:
        errors = [d for d in ontology.drifts if d.severity == "error"]
        warnings = [d for d in ontology.drifts if d.severity == "warning"]

        print(f"\nRegression Summary:")
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")

        if errors:
            print("\n❌ REGRESSION CHECK FAILED")
            for d in errors:
                print(f"  - [{d.drift_type.value}] {d.schema_name}: {d.description}")
            return 1

        if args.check and warnings:
            print("\n⚠️  Warnings detected (non-blocking):")
            for d in warnings:
                print(f"  - [{d.drift_type.value}] {d.schema_name}: {d.description}")

        print("\n✅ Regression check passed")
        return 0

    # Generate outputs
    print("\nGenerating outputs...")

    # JSON ontology
    OUTPUT_ONTOLOGY_JSON.parent.mkdir(parents=True, exist_ok=True)
    json_content = generate_ontology_json(ontology)
    OUTPUT_ONTOLOGY_JSON.write_text(json_content, encoding="utf-8")
    print(f"Wrote {OUTPUT_ONTOLOGY_JSON}")

    # DOT graph
    dot_content = generate_dot_graph(ontology)
    OUTPUT_ONTOLOGY_DOT.write_text(dot_content, encoding="utf-8")
    print(f"Wrote {OUTPUT_ONTOLOGY_DOT}")

    # Regression report
    report_content = generate_regression_report(ontology)
    OUTPUT_REGRESSION_REPORT.write_text(report_content, encoding="utf-8")
    print(f"Wrote {OUTPUT_REGRESSION_REPORT}")

    if args.save_baseline:
        save_baseline(ontology)
        print(f"Saved baseline to {BASELINE_ONTOLOGY}")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

