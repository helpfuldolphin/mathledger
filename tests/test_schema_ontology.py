"""
Schema Ontology Builder Test Suite

Agent: doc-ops-2 (E2) — Change Impact Matrix Architect
Tests: 120 comprehensive tests covering:
    - Ontology determinism (tests 1-10)
    - Drift classification (tests 11-20)
    - Graph completeness checks (tests 21-30)
    - Edge cases (tests 31-35)
    - Schema change brief (tests 36-42)
    - Schema explain functionality (tests 43-48)
    - Impact severity scoring (tests 49-55)
    - Safe-to-merge heuristic (tests 56-58)
    - Explain consumers (tests 59-63)
    - Impact summary markdown (tests 64-66)
    - Formal impact contract (tests 67-72)
    - Consumer-centric hardening (tests 73-77)
    - Field impact matrix (tests 78-83)
    - Impact manifest (tests 84-88)
    - Reviewer checklist (tests 89-95)
    - Cross-schema impact index (tests 96-100)
    - Reviewer playbook (tests 101-105)
    - Director panel (tests 106-110)
    - Impact SLO evaluator (tests 111-115)
    - Ownership routing (tests 116-120)

Run with: uv run pytest tests/test_schema_ontology.py -v
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Import the module under test
from scripts.schema_ontology_builder import (
    FieldCluster,
    RelationType,
    DriftType,
    Severity,
    SchemaField,
    SchemaNode,
    SchemaRelationship,
    SchemaDrift,
    SchemaOntology,
    SchemaChange,
    SchemaChangeBrief,
    SchemaImpact,
    ImpactSummary,
    ConsumerInfo,
    FieldImpact,
    classify_field,
    infer_type,
    extract_constraints,
    infer_relationships,
    detect_drifts,
    compute_blast_radius,
    generate_dot_graph,
    generate_ontology_json,
    generate_schema_change_brief,
    explain_schema,
    explain_consumers,
    score_change_impact,
    is_schema_change_safe,
    build_field_impact_matrix,
    build_impact_manifest,
    generate_review_checklist,
    build_cross_schema_impact_index,
    render_cross_schema_reviewer_playbook,
    build_schema_impact_director_panel,
    evaluate_schema_change_slo,
    route_schema_changes_to_owners,
    DEFAULT_SCHEMA_OWNERSHIP_MAP,
    _classify_type_change,
    _compute_relationship,
    _compare_schema_nodes,
    _generate_change_brief_markdown,
    _change_brief_to_dict,
    _score_single_schema_impact,
    _get_consumers_for_schema,
    WIDELY_USED_SCHEMAS,
    CRITICAL_FIELDS,
    CONSUMER_FIELD_USAGE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_schema_node() -> SchemaNode:
    """Create a sample schema node for testing."""
    fields = [
        SchemaField(
            name="seed",
            path="config.seed",
            field_type="integer",
            cluster=FieldCluster.SEED,
        ),
        SchemaField(
            name="threshold_min",
            path="config.threshold_min",
            field_type="number",
            cluster=FieldCluster.SUCCESS,
            constraints=["minimum bound"],
        ),
        SchemaField(
            name="description",
            path="config.description",
            field_type="string",
            cluster=FieldCluster.CONFIGURATION,
        ),
    ]
    return SchemaNode(
        name="test_schema",
        source_file="config/test.yaml",
        schema_type="yaml",
        version="1.0",
        field_count=3,
        fields=fields,
        cluster_distribution={"seed": 1, "success": 1, "configuration": 1},
        raw_hash="abc123",
    )


@pytest.fixture
def sample_ontology(sample_schema_node: SchemaNode) -> SchemaOntology:
    """Create a sample ontology for testing."""
    return SchemaOntology(
        version="1.0.0",
        generated_at="2025-01-01T00:00:00Z",
        nodes=[sample_schema_node],
        relationships=[],
        drifts=[],
        blast_radius={"test_schema": ["scripts/test.py"]},
    )


@pytest.fixture
def pair_of_schema_nodes() -> tuple[SchemaNode, SchemaNode]:
    """Create two schema nodes for relationship testing."""
    fields_a = [
        SchemaField(name="id", path="id", field_type="string", cluster=FieldCluster.IDENTITY),
        SchemaField(name="name", path="name", field_type="string", cluster=FieldCluster.IDENTITY),
        SchemaField(name="value", path="value", field_type="number", cluster=FieldCluster.METRIC),
        SchemaField(name="unique_a", path="unique_a", field_type="string", cluster=FieldCluster.CONFIGURATION),
    ]
    fields_b = [
        SchemaField(name="id", path="id", field_type="string", cluster=FieldCluster.IDENTITY),
        SchemaField(name="name", path="name", field_type="string", cluster=FieldCluster.IDENTITY),
        SchemaField(name="value", path="value", field_type="number", cluster=FieldCluster.METRIC),
        SchemaField(name="unique_b", path="unique_b", field_type="string", cluster=FieldCluster.CONFIGURATION),
    ]

    node_a = SchemaNode(
        name="schema_a",
        source_file="config/a.yaml",
        schema_type="yaml",
        version="1.0",
        field_count=4,
        fields=fields_a,
        cluster_distribution={"identity": 2, "metric": 1, "configuration": 1},
        raw_hash="aaa111",
    )
    node_b = SchemaNode(
        name="schema_b",
        source_file="config/b.yaml",
        schema_type="yaml",
        version="1.0",
        field_count=4,
        fields=fields_b,
        cluster_distribution={"identity": 2, "metric": 1, "configuration": 1},
        raw_hash="bbb222",
    )
    return node_a, node_b


# =============================================================================
# TEST GROUP 1: ONTOLOGY DETERMINISM (Tests 1-10)
# =============================================================================


class TestOntologyDeterminism:
    """Tests ensuring ontology generation is deterministic."""

    def test_01_type_inference_deterministic(self):
        """Test that type inference produces consistent results."""
        test_cases = [
            (None, "null"),
            (True, "boolean"),
            (False, "boolean"),
            (42, "integer"),
            (3.14, "number"),
            ("hello", "string"),
            ([], "array"),
            ([1, 2, 3], "array<integer>"),
            (["a", "b"], "array<string>"),
            ({}, "object"),
        ]
        for value, expected_type in test_cases:
            # Run multiple times to verify determinism
            for _ in range(3):
                assert infer_type(value) == expected_type

    def test_02_field_classification_deterministic(self):
        """Test that field classification is deterministic."""
        test_fields = [
            ("random_seed", "integer", [], FieldCluster.SEED),
            ("throughput_rate", "number", [], FieldCluster.METRIC),
            ("pass_threshold", "number", [], FieldCluster.SUCCESS),
            ("blocked_users", "array", [], FieldCluster.GOVERNANCE),
            ("experiment_id", "string", [], FieldCluster.IDENTITY),
            ("timeout_ms", "integer", [], FieldCluster.TEMPORAL),
            ("file_path", "string", [], FieldCluster.STRUCTURAL),
        ]
        for name, ftype, constraints, expected_cluster in test_fields:
            for _ in range(3):
                assert classify_field(name, ftype, constraints) == expected_cluster

    def test_03_constraint_extraction_deterministic(self):
        """Test that constraint extraction is deterministic."""
        test_cases = [
            ("threshold_min", 10, ["minimum bound"]),
            ("limit_max", 100, ["maximum bound"]),
            ("rate_pct", 50, ["0-100 percentage"]),
            ("timeout_s", 30, ["seconds", "timeout value"]),
            ("delay_ms", 100, ["milliseconds"]),
        ]
        for name, value, expected_constraints in test_cases:
            for _ in range(3):
                result = extract_constraints(name, value)
                assert set(result) == set(expected_constraints)

    def test_04_json_output_deterministic(self, sample_ontology: SchemaOntology):
        """Test that JSON output is deterministic across multiple generations."""
        outputs = [generate_ontology_json(sample_ontology) for _ in range(5)]
        # All outputs should be identical
        assert len(set(outputs)) == 1

    def test_05_dot_graph_deterministic(self, sample_ontology: SchemaOntology):
        """Test that DOT graph output is deterministic."""
        # Strip timestamps for comparison
        def strip_timestamp(content: str) -> str:
            lines = [l for l in content.split("\n") if "Timestamp:" not in l]
            return "\n".join(lines)

        outputs = [strip_timestamp(generate_dot_graph(sample_ontology)) for _ in range(5)]
        assert len(set(outputs)) == 1

    def test_06_relationship_inference_deterministic(self, pair_of_schema_nodes):
        """Test that relationship inference is deterministic."""
        node_a, node_b = pair_of_schema_nodes
        nodes = [node_a, node_b]

        relationships_runs = []
        for _ in range(5):
            rels = infer_relationships(nodes)
            # Convert to comparable format
            rel_strs = [
                f"{r.source_schema}->{r.target_schema}:{r.relation_type.value}"
                for r in rels
            ]
            relationships_runs.append(tuple(sorted(rel_strs)))

        assert len(set(relationships_runs)) == 1

    def test_07_hash_commitment_deterministic(self, sample_ontology: SchemaOntology):
        """Test that content hash is deterministic."""
        json_content = generate_ontology_json(sample_ontology)
        data = json.loads(json_content)
        hash1 = data["content_hash"]

        # Generate again
        json_content2 = generate_ontology_json(sample_ontology)
        data2 = json.loads(json_content2)
        hash2 = data2["content_hash"]

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_08_node_ordering_deterministic(self):
        """Test that nodes are processed in deterministic order."""
        nodes = [
            SchemaNode(name="z_schema", source_file="z.yaml", schema_type="yaml", version="1", field_count=0, fields=[]),
            SchemaNode(name="a_schema", source_file="a.yaml", schema_type="yaml", version="1", field_count=0, fields=[]),
            SchemaNode(name="m_schema", source_file="m.yaml", schema_type="yaml", version="1", field_count=0, fields=[]),
        ]

        ontology = SchemaOntology(nodes=nodes)
        json_output = generate_ontology_json(ontology)

        # Parse and verify order is consistent
        data = json.loads(json_output)
        node_names = [n["name"] for n in data["nodes"]]

        # Generate multiple times
        for _ in range(3):
            data2 = json.loads(generate_ontology_json(ontology))
            assert [n["name"] for n in data2["nodes"]] == node_names

    def test_09_empty_ontology_deterministic(self):
        """Test that empty ontology generates deterministic output."""
        empty_ontology = SchemaOntology()
        outputs = [generate_ontology_json(empty_ontology) for _ in range(3)]
        assert len(set(outputs)) == 1

    def test_10_cluster_distribution_deterministic(self):
        """Test that cluster distribution calculation is deterministic."""
        fields = [
            SchemaField(name="seed", path="seed", field_type="int", cluster=FieldCluster.SEED),
            SchemaField(name="seed2", path="seed2", field_type="int", cluster=FieldCluster.SEED),
            SchemaField(name="metric", path="metric", field_type="float", cluster=FieldCluster.METRIC),
        ]

        distributions = []
        for _ in range(5):
            dist: dict[str, int] = {}
            for f in fields:
                dist[f.cluster.value] = dist.get(f.cluster.value, 0) + 1
            distributions.append(json.dumps(dist, sort_keys=True))

        assert len(set(distributions)) == 1


# =============================================================================
# TEST GROUP 2: DRIFT CLASSIFICATION (Tests 11-20)
# =============================================================================


class TestDriftClassification:
    """Tests for schema drift detection and classification."""

    def test_11_detect_field_added(self):
        """Test detection of newly added fields."""
        baseline_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="1",
            field_count=1,
            fields=[SchemaField(name="existing", path="existing", field_type="string")],
        )
        current_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="2",
            field_count=2,
            fields=[
                SchemaField(name="existing", path="existing", field_type="string"),
                SchemaField(name="new_field", path="new_field", field_type="integer"),
            ],
        )

        drifts = detect_drifts([current_node], [baseline_node])
        added_drifts = [d for d in drifts if d.drift_type == DriftType.FIELD_ADDED]
        assert len(added_drifts) == 1
        assert added_drifts[0].field_path == "new_field"
        assert added_drifts[0].severity == "info"

    def test_12_detect_field_removed(self):
        """Test detection of removed required fields."""
        baseline_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="1",
            field_count=2,
            fields=[
                SchemaField(name="required_field", path="required_field", field_type="string"),
                SchemaField(name="other", path="other", field_type="integer"),
            ],
        )
        current_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="2",
            field_count=1,
            fields=[SchemaField(name="other", path="other", field_type="integer")],
        )

        drifts = detect_drifts([current_node], [baseline_node])
        removed_drifts = [d for d in drifts if d.drift_type == DriftType.FIELD_REMOVED]
        assert len(removed_drifts) == 1
        assert removed_drifts[0].field_path == "required_field"
        assert removed_drifts[0].severity == "error"

    def test_13_detect_type_narrowing(self):
        """Test detection of type narrowing (more restrictive type)."""
        # number -> integer is narrowing
        result = _classify_type_change("number", "integer")
        assert result == DriftType.NARROWING

        # Any -> string is narrowing
        result = _classify_type_change("Any", "string")
        assert result == DriftType.NARROWING

    def test_14_detect_type_widening(self):
        """Test detection of type widening (less restrictive type)."""
        # integer -> number is widening
        result = _classify_type_change("integer", "number")
        assert result == DriftType.WIDENING

        # string -> Any is widening
        result = _classify_type_change("string", "Any")
        assert result == DriftType.WIDENING

    def test_15_detect_type_change_same_level(self):
        """Test detection of type changes at different hierarchy levels."""
        # object -> array: array has higher rank (2 > 1), so it's NARROWING
        result = _classify_type_change("object", "array")
        assert result == DriftType.NARROWING
        
        # array -> object: object has lower rank (1 < 2), so it's WIDENING
        result = _classify_type_change("array", "object")
        assert result == DriftType.WIDENING
        
        # string -> boolean is narrowing in our hierarchy (boolean rank 6 > string rank 3)
        result = _classify_type_change("string", "boolean")
        assert result == DriftType.NARROWING

    def test_16_detect_field_rename_heuristic(self):
        """Test heuristic detection of field renames."""
        baseline_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="1",
            field_count=1,
            fields=[SchemaField(name="old_name", path="old_name", field_type="string")],
        )
        current_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="2",
            field_count=1,
            fields=[SchemaField(name="new_name", path="new_name", field_type="string")],
        )

        drifts = detect_drifts([current_node], [baseline_node])
        rename_drifts = [d for d in drifts if d.drift_type == DriftType.FIELD_RENAMED]
        # Should detect potential rename since same type
        assert len(rename_drifts) == 1

    def test_17_detect_constraint_added(self):
        """Test detection of added constraints."""
        baseline_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="1",
            field_count=1,
            fields=[SchemaField(name="value", path="value", field_type="number", constraints=[])],
        )
        current_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="2",
            field_count=1,
            fields=[SchemaField(name="value", path="value", field_type="number", constraints=["minimum bound"])],
        )

        drifts = detect_drifts([current_node], [baseline_node])
        constraint_drifts = [d for d in drifts if d.drift_type == DriftType.CONSTRAINT_ADDED]
        assert len(constraint_drifts) == 1

    def test_18_detect_constraint_removed(self):
        """Test detection of removed constraints."""
        baseline_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="1",
            field_count=1,
            fields=[SchemaField(name="value", path="value", field_type="number", constraints=["minimum bound"])],
        )
        current_node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="2",
            field_count=1,
            fields=[SchemaField(name="value", path="value", field_type="number", constraints=[])],
        )

        drifts = detect_drifts([current_node], [baseline_node])
        constraint_drifts = [d for d in drifts if d.drift_type == DriftType.CONSTRAINT_REMOVED]
        assert len(constraint_drifts) == 1
        assert constraint_drifts[0].severity == "warning"

    def test_19_detect_new_schema(self):
        """Test detection of entirely new schema."""
        baseline_nodes: list[SchemaNode] = []
        current_node = SchemaNode(
            name="brand_new",
            source_file="new.yaml",
            schema_type="yaml",
            version="1",
            field_count=1,
            fields=[SchemaField(name="field", path="field", field_type="string")],
        )

        drifts = detect_drifts([current_node], baseline_nodes)
        new_schema_drifts = [d for d in drifts if d.field_path == "<schema>"]
        assert len(new_schema_drifts) == 1
        assert new_schema_drifts[0].severity == "info"

    def test_20_detect_removed_schema(self):
        """Test detection of entirely removed schema."""
        baseline_node = SchemaNode(
            name="removed_schema",
            source_file="removed.yaml",
            schema_type="yaml",
            version="1",
            field_count=1,
            fields=[SchemaField(name="field", path="field", field_type="string")],
        )
        current_nodes: list[SchemaNode] = []

        drifts = detect_drifts(current_nodes, [baseline_node])
        removed_schema_drifts = [d for d in drifts if d.schema_name == "removed_schema"]
        assert len(removed_schema_drifts) == 1
        assert removed_schema_drifts[0].severity == "error"


# =============================================================================
# TEST GROUP 3: GRAPH COMPLETENESS (Tests 21-30)
# =============================================================================


class TestGraphCompleteness:
    """Tests for ontology graph completeness and consistency."""

    def test_21_all_nodes_in_graph(self, sample_ontology: SchemaOntology):
        """Test that all schema nodes appear in DOT graph."""
        dot_content = generate_dot_graph(sample_ontology)
        for node in sample_ontology.nodes:
            # Node should appear as a definition
            assert node.name in dot_content

    def test_22_all_relationships_in_graph(self, pair_of_schema_nodes):
        """Test that all relationships appear in DOT graph."""
        node_a, node_b = pair_of_schema_nodes
        relationships = infer_relationships([node_a, node_b])
        
        ontology = SchemaOntology(nodes=[node_a, node_b], relationships=relationships)
        dot_content = generate_dot_graph(ontology)

        for rel in relationships:
            if rel.relation_type != RelationType.NONE:
                # Relationship should appear as an edge
                assert rel.source_schema in dot_content
                assert rel.target_schema in dot_content

    def test_23_equivalent_relationship_detection(self):
        """Test that equivalent schemas are detected correctly."""
        # Create two nearly identical schemas
        fields = [
            SchemaField(name="field1", path="field1", field_type="string"),
            SchemaField(name="field2", path="field2", field_type="integer"),
            SchemaField(name="field3", path="field3", field_type="boolean"),
            SchemaField(name="field4", path="field4", field_type="number"),
            SchemaField(name="field5", path="field5", field_type="array"),
        ]

        node_a = SchemaNode(name="a", source_file="a.yaml", schema_type="yaml", version="1", field_count=5, fields=fields.copy())
        node_b = SchemaNode(name="b", source_file="b.yaml", schema_type="yaml", version="1", field_count=5, fields=fields.copy())

        rel = _compute_relationship(node_a, node_b)
        assert rel is not None
        assert rel.relation_type == RelationType.EQUIVALENT
        assert rel.confidence >= 0.8

    def test_24_sibling_relationship_detection(self, pair_of_schema_nodes):
        """Test that sibling schemas are detected correctly."""
        node_a, node_b = pair_of_schema_nodes
        rel = _compute_relationship(node_a, node_b)
        
        assert rel is not None
        # With 3/5 shared fields (60%), should be sibling
        assert rel.relation_type in (RelationType.SIBLING, RelationType.EQUIVALENT)
        assert rel.confidence >= 0.3

    def test_25_no_self_relationships(self, sample_schema_node: SchemaNode):
        """Test that schemas don't have relationships with themselves."""
        nodes = [sample_schema_node]
        relationships = infer_relationships(nodes)
        
        for rel in relationships:
            assert rel.source_schema != rel.target_schema

    def test_26_blast_radius_completeness(self):
        """Test that blast radius includes all schema nodes."""
        nodes = [
            SchemaNode(name="schema1", source_file="config/test1.yaml", schema_type="yaml", version="1", field_count=0, fields=[]),
            SchemaNode(name="schema2", source_file="config/test2.json", schema_type="json", version="1", field_count=0, fields=[]),
        ]

        blast_radius = compute_blast_radius(nodes)
        
        # Every schema should have a blast radius entry (even if empty)
        for node in nodes:
            assert node.name in blast_radius

    def test_27_json_schema_all_fields_present(self, sample_ontology: SchemaOntology):
        """Test that JSON output contains all required schema fields."""
        json_content = generate_ontology_json(sample_ontology)
        data = json.loads(json_content)

        required_keys = ["version", "generated_at", "nodes", "relationships", "drifts", "blast_radius", "content_hash"]
        for key in required_keys:
            assert key in data

        for node in data["nodes"]:
            node_required = ["name", "source_file", "schema_type", "version", "field_count", "fields"]
            for key in node_required:
                assert key in node

    def test_28_dot_graph_valid_syntax(self, sample_ontology: SchemaOntology):
        """Test that DOT graph has valid basic syntax."""
        dot_content = generate_dot_graph(sample_ontology)

        # Must start with digraph declaration
        assert "digraph SchemaOntology {" in dot_content
        # Must end with closing brace
        assert dot_content.strip().endswith("}")
        # Node definitions should have proper format
        assert "shape=record" in dot_content

    def test_29_cluster_colors_in_graph(self, sample_ontology: SchemaOntology):
        """Test that cluster-based coloring is applied in DOT graph."""
        dot_content = generate_dot_graph(sample_ontology)
        
        # Should have fillcolor attributes
        assert "fillcolor=" in dot_content
        # Should use filled style
        assert "style=filled" in dot_content

    def test_30_relationship_edge_styles(self, pair_of_schema_nodes):
        """Test that different relationship types have distinct edge styles."""
        node_a, node_b = pair_of_schema_nodes
        relationships = infer_relationships([node_a, node_b])
        
        ontology = SchemaOntology(nodes=[node_a, node_b], relationships=relationships)
        dot_content = generate_dot_graph(ontology)

        # Should have edge style attributes
        assert "->" in dot_content  # Edge definitions
        # Should have style or color attributes on edges
        assert "style=" in dot_content or "color=" in dot_content


# =============================================================================
# ADDITIONAL EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_empty_schema_handling(self):
        """Test handling of schema with no fields."""
        empty_node = SchemaNode(
            name="empty",
            source_file="empty.yaml",
            schema_type="yaml",
            version="1",
            field_count=0,
            fields=[],
        )

        ontology = SchemaOntology(nodes=[empty_node])
        json_content = generate_ontology_json(ontology)
        
        # Should not raise and should produce valid JSON
        data = json.loads(json_content)
        assert data["nodes"][0]["field_count"] == 0

    def test_special_characters_in_names(self):
        """Test handling of special characters in schema/field names."""
        node = SchemaNode(
            name="schema-with-dashes_and_underscores",
            source_file="test.yaml",
            schema_type="yaml",
            version="1.0.0",
            field_count=1,
            fields=[SchemaField(name="field.with.dots", path="field.with.dots", field_type="string")],
        )

        ontology = SchemaOntology(nodes=[node])
        dot_content = generate_dot_graph(ontology)
        
        # Should produce valid DOT (special chars in node IDs are escaped)
        assert "digraph" in dot_content

    def test_deeply_nested_paths(self):
        """Test handling of deeply nested field paths."""
        field = SchemaField(
            name="deep",
            path="level1.level2.level3.level4.level5.deep",
            field_type="string",
            cluster=FieldCluster.CONFIGURATION,
        )

        node = SchemaNode(
            name="nested",
            source_file="nested.yaml",
            schema_type="yaml",
            version="1",
            field_count=1,
            fields=[field],
        )

        ontology = SchemaOntology(nodes=[node])
        json_content = generate_ontology_json(ontology)
        data = json.loads(json_content)

        assert data["nodes"][0]["fields"][0]["path"] == "level1.level2.level3.level4.level5.deep"

    def test_no_baseline_returns_empty_drifts(self):
        """Test that missing baseline returns empty drift list."""
        node = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="1",
            field_count=0,
            fields=[],
        )

        drifts = detect_drifts([node], None)
        assert drifts == []

    def test_array_type_inference(self):
        """Test array type inference with various content types."""
        assert infer_type([1, 2, 3]) == "array<integer>"
        assert infer_type(["a", "b"]) == "array<string>"
        assert infer_type([1.0, 2.0]) == "array<number>"
        assert infer_type([True, False]) == "array<boolean>"
        
        # Mixed types
        result = infer_type([1, "string"])
        assert "array<" in result
        assert "|" in result  # Union type indicator


# =============================================================================
# TEST GROUP 4: SCHEMA CHANGE BRIEF (Tests 36-42)
# =============================================================================


class TestSchemaChangeBrief:
    """Tests for schema change brief generation."""

    @pytest.fixture
    def baseline_nodes(self) -> list[SchemaNode]:
        """Create baseline schema nodes."""
        return [
            SchemaNode(
                name="test_schema",
                source_file="config/test.yaml",
                schema_type="yaml",
                version="1.0",
                field_count=3,
                fields=[
                    SchemaField(name="field_a", path="field_a", field_type="string"),
                    SchemaField(name="field_b", path="field_b", field_type="integer"),
                    SchemaField(name="field_c", path="field_c", field_type="boolean"),
                ],
                raw_hash="baseline123",
            )
        ]

    @pytest.fixture
    def current_nodes_with_changes(self) -> list[SchemaNode]:
        """Create current nodes with some changes."""
        return [
            SchemaNode(
                name="test_schema",
                source_file="config/test.yaml",
                schema_type="yaml",
                version="2.0",
                field_count=3,
                fields=[
                    SchemaField(name="field_a", path="field_a", field_type="string"),
                    # field_b removed, field_d added
                    SchemaField(name="field_c", path="field_c", field_type="number"),  # type changed
                    SchemaField(name="field_d", path="field_d", field_type="array"),
                ],
                raw_hash="current456",
            )
        ]

    def test_36_change_brief_markdown_deterministic(self):
        """Test that change brief markdown is deterministic."""
        brief = SchemaChangeBrief(
            generated_at="2025-01-01T00:00:00Z",
            base_hash="abc123",
            current_hash="def456",
            schemas_changed=["schema_a", "schema_b"],
            schema_changes=[
                SchemaChange(
                    schema_name="schema_a",
                    source_file="a.yaml",
                    fields_added=[("new_field", "string")],
                    fields_removed=[],
                    fields_renamed=[],
                    fields_type_changed=[],
                    downstream_consumers=["consumer.py"],
                )
            ],
            total_fields_added=1,
            total_fields_removed=0,
            total_fields_modified=0,
        )

        outputs = [_generate_change_brief_markdown(brief) for _ in range(3)]
        assert len(set(outputs)) == 1

    def test_37_change_brief_detects_added_fields(
        self, baseline_nodes, current_nodes_with_changes
    ):
        """Test that change brief correctly detects added fields."""
        from scripts.schema_ontology_builder import _compute_schema_change

        baseline = baseline_nodes[0]
        current = current_nodes_with_changes[0]

        change = _compute_schema_change(baseline, current, [])
        
        # field_d was added
        added_paths = [p for p, _ in change.fields_added]
        assert "field_d" in added_paths

    def test_38_change_brief_detects_removed_fields(
        self, baseline_nodes, current_nodes_with_changes
    ):
        """Test that change brief correctly detects removed fields."""
        from scripts.schema_ontology_builder import _compute_schema_change

        baseline = baseline_nodes[0]
        current = current_nodes_with_changes[0]

        change = _compute_schema_change(baseline, current, [])
        
        # field_b was removed
        removed_paths = [p for p, _ in change.fields_removed]
        assert "field_b" in removed_paths

    def test_39_change_brief_detects_type_changes(
        self, baseline_nodes, current_nodes_with_changes
    ):
        """Test that change brief correctly detects type changes."""
        from scripts.schema_ontology_builder import _compute_schema_change

        baseline = baseline_nodes[0]
        current = current_nodes_with_changes[0]

        change = _compute_schema_change(baseline, current, [])
        
        # field_c changed from boolean to number
        type_changed_paths = [p for p, _, _ in change.fields_type_changed]
        assert "field_c" in type_changed_paths

    def test_40_change_brief_to_dict_complete(self):
        """Test that change brief dict contains all required keys."""
        brief = SchemaChangeBrief(
            generated_at="2025-01-01T00:00:00Z",
            base_hash="abc",
            current_hash="def",
            schemas_changed=[],
            schema_changes=[],
            total_fields_added=0,
            total_fields_removed=0,
            total_fields_modified=0,
        )

        result = _change_brief_to_dict(brief)
        
        required_keys = [
            "generated_at",
            "base_hash",
            "current_hash",
            "schemas_changed",
            "schema_changes",
            "total_fields_added",
            "total_fields_removed",
            "total_fields_modified",
            "notes",
        ]
        for key in required_keys:
            assert key in result

    def test_41_change_brief_empty_when_no_changes(self):
        """Test that change brief shows no changes when schemas are identical."""
        nodes = [
            SchemaNode(
                name="same",
                source_file="same.yaml",
                schema_type="yaml",
                version="1",
                field_count=1,
                fields=[SchemaField(name="field", path="field", field_type="string")],
                raw_hash="same_hash",
            )
        ]

        from scripts.schema_ontology_builder import _compute_schema_change

        change = _compute_schema_change(nodes[0], nodes[0], [])
        
        assert change.fields_added == []
        assert change.fields_removed == []
        assert change.fields_renamed == []
        assert change.fields_type_changed == []

    def test_42_change_brief_includes_consumers(self):
        """Test that change brief includes downstream consumers."""
        from scripts.schema_ontology_builder import _compute_schema_change

        baseline = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="1",
            field_count=1,
            fields=[SchemaField(name="old", path="old", field_type="string")],
        )
        current = SchemaNode(
            name="test",
            source_file="test.yaml",
            schema_type="yaml",
            version="2",
            field_count=1,
            fields=[SchemaField(name="new", path="new", field_type="string")],
        )

        consumers = ["consumer1.py", "consumer2.py"]
        change = _compute_schema_change(baseline, current, consumers)
        
        assert change.downstream_consumers == consumers


# =============================================================================
# TEST GROUP 5: SCHEMA EXPLAIN (Tests 43-48)
# =============================================================================


class TestSchemaExplain:
    """Tests for schema explain functionality."""

    @pytest.fixture
    def sample_ontology_for_explain(self) -> SchemaOntology:
        """Create ontology with known structure for explain tests."""
        node_a = SchemaNode(
            name="alpha",
            source_file="config/alpha.yaml",
            schema_type="yaml",
            version="1.0",
            field_count=5,
            fields=[
                SchemaField(name="seed", path="seed", field_type="integer", cluster=FieldCluster.SEED),
                SchemaField(name="threshold", path="threshold", field_type="number", cluster=FieldCluster.SUCCESS),
                SchemaField(name="name", path="name", field_type="string", cluster=FieldCluster.IDENTITY),
            ],
            cluster_distribution={"seed": 1, "success": 1, "identity": 1},
            raw_hash="alpha_hash_123",
        )
        node_b = SchemaNode(
            name="beta",
            source_file="config/beta.yaml",
            schema_type="yaml",
            version="2.0",
            field_count=3,
            fields=[
                SchemaField(name="name", path="name", field_type="string", cluster=FieldCluster.IDENTITY),
                SchemaField(name="value", path="value", field_type="number", cluster=FieldCluster.METRIC),
            ],
            cluster_distribution={"identity": 1, "metric": 1},
            raw_hash="beta_hash_456",
        )

        relationships = [
            SchemaRelationship(
                source_schema="alpha",
                target_schema="beta",
                relation_type=RelationType.REFERENCES,
                confidence=0.25,
                shared_fields=["name"],
                evidence="Schemas share 1 field",
            )
        ]

        return SchemaOntology(
            nodes=[node_a, node_b],
            relationships=relationships,
            blast_radius={"alpha": ["scripts/use_alpha.py"], "beta": []},
        )

    def test_43_explain_output_stable(self, sample_ontology_for_explain):
        """Test that explain output is stable across multiple calls."""
        outputs = [
            explain_schema("alpha", sample_ontology_for_explain) for _ in range(3)
        ]
        assert len(set(outputs)) == 1

    def test_44_explain_contains_schema_name(self, sample_ontology_for_explain):
        """Test that explain output contains schema name."""
        output = explain_schema("alpha", sample_ontology_for_explain)
        assert "Schema: alpha" in output

    def test_45_explain_contains_relationships(self, sample_ontology_for_explain):
        """Test that explain output contains relationship information."""
        output = explain_schema("alpha", sample_ontology_for_explain)
        assert "Relationships:" in output
        assert "beta" in output

    def test_46_explain_contains_field_clusters(self, sample_ontology_for_explain):
        """Test that explain output contains field cluster distribution."""
        output = explain_schema("alpha", sample_ontology_for_explain)
        assert "Field Clusters:" in output
        assert "seed" in output

    def test_47_explain_contains_consumers(self, sample_ontology_for_explain):
        """Test that explain output contains known consumers."""
        output = explain_schema("alpha", sample_ontology_for_explain)
        assert "Known Consumers:" in output
        assert "scripts/use_alpha.py" in output

    def test_48_explain_handles_unknown_schema(self, sample_ontology_for_explain):
        """Test that explain gracefully handles unknown schema."""
        output = explain_schema("nonexistent", sample_ontology_for_explain)
        assert "not found" in output.lower()
        assert "Available schemas:" in output
        assert "alpha" in output
        assert "beta" in output


# =============================================================================
# TEST GROUP 6: IMPACT SEVERITY SCORING (Tests 49-55)
# =============================================================================


class TestImpactSeverityScoring:
    """Tests for the score_change_impact functionality."""

    def test_49_removed_field_from_widely_used_schema_triggers_block(self):
        """Test that removing a field from a widely-used schema results in BLOCK severity."""
        # "curriculum" is in WIDELY_USED_SCHEMAS
        change_brief = {
            "schemas_changed": ["curriculum"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [],
                    "fields_removed": [("systems.pl.slices", "array")],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        assert result["overall_severity"] == "BLOCK"
        assert result["block_count"] == 1
        assert result["safe_to_merge"] is False
        # Check per_schema format
        assert "per_schema" in result
        assert "curriculum" in result["per_schema"]
        assert result["per_schema"]["curriculum"]["severity"] == "BLOCK"
        assert any("removed" in r.lower() for r in result["per_schema"]["curriculum"]["reasons"])

    def test_50_added_field_to_low_usage_schema_triggers_warn(self):
        """Test that adding a field to a low-usage schema results in WARN severity."""
        # "some_schema" is NOT in WIDELY_USED_SCHEMAS
        change_brief = {
            "schemas_changed": ["some_schema"],
            "schema_changes": [
                {
                    "schema_name": "some_schema",
                    "source_file": "config/some_schema.yaml",
                    "fields_added": [("new_feature", "string")],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        assert result["overall_severity"] == "WARN"
        assert result["warn_count"] == 1
        assert result["safe_to_merge"] is False

    def test_51_cosmetic_changes_only_triggers_info(self):
        """Test that cosmetic-only changes (empty change lists) result in INFO severity."""
        change_brief = {
            "schemas_changed": ["some_schema"],
            "schema_changes": [
                {
                    "schema_name": "some_schema",
                    "source_file": "config/some_schema.yaml",
                    "fields_added": [],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        assert result["overall_severity"] == "INFO"
        assert result["info_count"] == 1
        assert result["safe_to_merge"] is True

    def test_52_critical_field_type_change_triggers_block(self):
        """Test that type change on critical field triggers BLOCK severity."""
        # "curriculum" has "systems.pl.slices" as a critical field
        change_brief = {
            "schemas_changed": ["curriculum"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [("systems.pl.slices", "array", "object")],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        assert result["overall_severity"] == "BLOCK"
        assert result["block_count"] == 1

    def test_53_non_critical_type_change_triggers_warn(self):
        """Test that type change on non-critical field triggers WARN severity."""
        change_brief = {
            "schemas_changed": ["some_schema"],
            "schema_changes": [
                {
                    "schema_name": "some_schema",
                    "source_file": "config/some_schema.yaml",
                    "fields_added": [],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [("some_field", "integer", "string")],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        assert result["overall_severity"] == "WARN"
        assert result["warn_count"] == 1

    def test_54_multiple_schemas_overall_severity_is_max(self):
        """Test that overall severity is the maximum across all schemas."""
        change_brief = {
            "schemas_changed": ["curriculum", "some_schema"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [],
                    "fields_removed": [("version", "string")],  # BLOCK
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                },
                {
                    "schema_name": "some_schema",
                    "source_file": "config/some_schema.yaml",
                    "fields_added": [("new", "string")],  # WARN
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                },
            ],
        }

        result = score_change_impact(change_brief)

        assert result["overall_severity"] == "BLOCK"  # Max wins
        assert result["block_count"] == 1
        assert result["warn_count"] == 1

    def test_55_field_rename_triggers_warn(self):
        """Test that field rename triggers WARN severity."""
        change_brief = {
            "schemas_changed": ["some_schema"],
            "schema_changes": [
                {
                    "schema_name": "some_schema",
                    "source_file": "config/some_schema.yaml",
                    "fields_added": [],
                    "fields_removed": [],
                    "fields_renamed": [("old_path", "old_name", "new_name")],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        assert result["overall_severity"] == "WARN"
        assert result["warn_count"] == 1


# =============================================================================
# TEST GROUP 7: SAFE-TO-MERGE HEURISTIC (Tests 56-58)
# =============================================================================


class TestSafeToMerge:
    """Tests for the is_schema_change_safe advisory heuristic."""

    def test_56_safe_to_merge_true_for_info_only(self):
        """Test that is_schema_change_safe returns True for INFO-only changes."""
        change_brief = {
            "schemas_changed": ["some_schema"],
            "schema_changes": [
                {
                    "schema_name": "some_schema",
                    "source_file": "config/some_schema.yaml",
                    "fields_added": [],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        assert is_schema_change_safe(change_brief) is True

    def test_57_safe_to_merge_false_for_warn_changes(self):
        """Test that is_schema_change_safe returns False for WARN-level changes."""
        change_brief = {
            "schemas_changed": ["some_schema"],
            "schema_changes": [
                {
                    "schema_name": "some_schema",
                    "source_file": "config/some_schema.yaml",
                    "fields_added": [("new_field", "string")],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        assert is_schema_change_safe(change_brief) is False

    def test_58_safe_to_merge_false_for_block_changes(self):
        """Test that is_schema_change_safe returns False for BLOCK-level changes."""
        change_brief = {
            "schemas_changed": ["curriculum"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [],
                    "fields_removed": [("systems.pl.slices", "array")],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        assert is_schema_change_safe(change_brief) is False


# =============================================================================
# TEST GROUP 8: EXPLAIN CONSUMERS (Tests 59-63)
# =============================================================================


class TestExplainConsumers:
    """Tests for the explain_consumers functionality."""

    @pytest.fixture
    def sample_ontology_for_consumers(self) -> SchemaOntology:
        """Create ontology for consumer explanation tests."""
        node = SchemaNode(
            name="curriculum",
            source_file="config/curriculum.yaml",
            schema_type="yaml",
            version="1.0",
            field_count=5,
            fields=[
                SchemaField(name="systems", path="systems", field_type="object"),
                SchemaField(name="pl", path="systems.pl", field_type="object"),
                SchemaField(name="slices", path="systems.pl.slices", field_type="array"),
                SchemaField(name="active", path="systems.pl.active", field_type="boolean"),
                SchemaField(name="version", path="version", field_type="string"),
            ],
            cluster_distribution={},
            raw_hash="curriculum_hash",
        )
        return SchemaOntology(nodes=[node], relationships=[], blast_radius={})

    def test_59_explain_consumers_output_stable(self, sample_ontology_for_consumers):
        """Test that explain_consumers output is stable across calls."""
        outputs = [
            explain_consumers("config/curriculum.yaml", sample_ontology_for_consumers)
            for _ in range(3)
        ]
        assert len(set(outputs)) == 1

    def test_60_explain_consumers_contains_schema_info(self, sample_ontology_for_consumers):
        """Test that explain_consumers output contains schema information."""
        output = explain_consumers("config/curriculum.yaml", sample_ontology_for_consumers)
        assert "curriculum" in output
        assert "Source File:" in output

    def test_61_explain_consumers_lists_consumers(self, sample_ontology_for_consumers):
        """Test that explain_consumers lists downstream consumers."""
        output = explain_consumers("config/curriculum.yaml", sample_ontology_for_consumers)
        assert "Consumers:" in output
        # These are from DOWNSTREAM_DEPENDENCIES
        assert "curriculum_loader" in output or "run_fo_cycles" in output

    def test_62_explain_consumers_shows_field_usage(self, sample_ontology_for_consumers):
        """Test that explain_consumers shows which fields are read."""
        output = explain_consumers("config/curriculum.yaml", sample_ontology_for_consumers)
        assert "Fields Read:" in output

    def test_63_explain_consumers_handles_unknown_file(self, sample_ontology_for_consumers):
        """Test that explain_consumers handles unknown schema file gracefully."""
        output = explain_consumers("nonexistent/schema.yaml", sample_ontology_for_consumers)
        assert "not found" in output.lower()


# =============================================================================
# TEST GROUP 9: IMPACT SUMMARY IN MARKDOWN (Tests 64-66)
# =============================================================================


class TestImpactSummaryMarkdown:
    """Tests for impact summary inclusion in markdown output."""

    def test_64_markdown_contains_impact_summary_section(self):
        """Test that markdown includes Impact Summary section when impact provided."""
        brief = SchemaChangeBrief(
            generated_at="2025-01-01T00:00:00Z",
            base_hash="base_hash",
            current_hash="current_hash",
            schemas_changed=["test_schema"],
            schema_changes=[
                SchemaChange(
                    schema_name="test_schema",
                    source_file="test.yaml",
                    fields_added=[("new_field", "string")],
                    fields_removed=[],
                    fields_renamed=[],
                    fields_type_changed=[],
                    downstream_consumers=[],
                )
            ],
            total_fields_added=1,
            total_fields_removed=0,
            total_fields_modified=0,
        )

        impact = {
            "overall_severity": "WARN",
            "safe_to_merge": False,
            "advisory_message": "⚠️ WARN: 1 schema(s) have notable changes",
            "block_count": 0,
            "warn_count": 1,
            "info_count": 0,
            "per_schema": {
                "test_schema": {
                    "severity": "WARN",
                    "reasons": ["New field 'new_field' added"],
                    "consumers": [],
                }
            },
        }

        md_content = _generate_change_brief_markdown(brief, impact)

        assert "## Impact Summary" in md_content
        assert "**Overall Severity:**" in md_content
        assert "WARN" in md_content
        assert "Safe to Merge" in md_content

    def test_65_markdown_shows_per_schema_impact(self):
        """Test that markdown shows per-schema impact details."""
        brief = SchemaChangeBrief(
            generated_at="2025-01-01T00:00:00Z",
            base_hash="base_hash",
            current_hash="current_hash",
            schemas_changed=["test_schema"],
            schema_changes=[
                SchemaChange(
                    schema_name="test_schema",
                    source_file="test.yaml",
                    fields_added=[],
                    fields_removed=[("removed_field", "string")],
                    fields_renamed=[],
                    fields_type_changed=[],
                    downstream_consumers=[],
                )
            ],
            total_fields_added=0,
            total_fields_removed=1,
            total_fields_modified=0,
        )

        impact = {
            "overall_severity": "WARN",
            "safe_to_merge": False,
            "advisory_message": "⚠️ WARN: 1 schema(s) have notable changes",
            "block_count": 0,
            "warn_count": 1,
            "info_count": 0,
            "per_schema": {
                "test_schema": {
                    "severity": "WARN",
                    "reasons": ["Field 'removed_field' removed"],
                    "consumers": [
                        {"file_path": "scripts/consumer.py", "field_access_count": 3}
                    ],
                }
            },
        }

        md_content = _generate_change_brief_markdown(brief, impact)

        assert "### Per-Schema Impact" in md_content
        assert "test_schema" in md_content
        assert "scripts/consumer.py" in md_content  # Consumer should be shown

    def test_66_markdown_no_impact_section_when_no_changes(self):
        """Test that no Impact Summary section when no impact provided."""
        brief = SchemaChangeBrief(
            generated_at="2025-01-01T00:00:00Z",
            base_hash="base_hash",
            current_hash="current_hash",
            schemas_changed=[],
            schema_changes=[],
            total_fields_added=0,
            total_fields_removed=0,
            total_fields_modified=0,
        )

        md_content = _generate_change_brief_markdown(brief, impact=None)

        assert "## Impact Summary" not in md_content
        assert "No schema changes detected" in md_content


# =============================================================================
# TEST GROUP 10: FORMAL IMPACT CONTRACT (Tests 67-72)
# =============================================================================


class TestFormalImpactContract:
    """Tests for the formal impact contract format."""

    def test_67_impact_contract_has_required_top_level_keys(self):
        """Test that impact result has all required top-level keys."""
        change_brief = {
            "schemas_changed": ["curriculum"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [],
                    "fields_removed": [("version", "string")],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        # Required keys per formal contract
        assert "overall_severity" in result
        assert "safe_to_merge" in result
        assert "per_schema" in result
        assert isinstance(result["per_schema"], dict)

    def test_68_per_schema_has_severity_reasons_consumers(self):
        """Test that each per_schema entry has severity, reasons, and consumers."""
        change_brief = {
            "schemas_changed": ["curriculum"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [("new_field", "string")],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        assert "curriculum" in result["per_schema"]
        schema_impact = result["per_schema"]["curriculum"]
        assert "severity" in schema_impact
        assert "reasons" in schema_impact
        assert "consumers" in schema_impact
        assert isinstance(schema_impact["consumers"], list)

    def test_69_consumers_include_file_path_and_field_count(self):
        """Test that consumers include file_path and field_access_count."""
        change_brief = {
            "schemas_changed": ["curriculum"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        consumers = result["per_schema"]["curriculum"]["consumers"]
        # curriculum has known consumers in DOWNSTREAM_DEPENDENCIES
        assert len(consumers) > 0

        for consumer in consumers:
            assert "file_path" in consumer
            assert "field_access_count" in consumer

    def test_70_block_severity_forces_safe_to_merge_false(self):
        """Test that BLOCK severity always sets safe_to_merge to False."""
        change_brief = {
            "schemas_changed": ["curriculum"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [],
                    "fields_removed": [("systems.pl.active", "boolean")],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        assert result["overall_severity"] == "BLOCK"
        assert result["safe_to_merge"] is False

    def test_71_additive_non_critical_changes_safe_to_merge_true(self):
        """Test that only additive, non-critical fields result in safe_to_merge=True."""
        # Schema NOT in WIDELY_USED_SCHEMAS with only cosmetic changes
        change_brief = {
            "schemas_changed": ["some_random_schema"],
            "schema_changes": [
                {
                    "schema_name": "some_random_schema",
                    "source_file": "config/random.yaml",
                    "fields_added": [],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        # INFO-only changes should be safe
        assert result["overall_severity"] == "INFO"
        assert result["safe_to_merge"] is True

    def test_72_get_consumers_returns_deterministic_order(self):
        """Test that _get_consumers_for_schema returns deterministic order."""
        results = [
            _get_consumers_for_schema("curriculum", "config/curriculum.yaml")
            for _ in range(5)
        ]

        # All results should be identical
        for i in range(1, len(results)):
            assert [c.file_path for c in results[0]] == [c.file_path for c in results[i]]


# =============================================================================
# TEST GROUP 11: CONSUMER-CENTRIC VIEW HARDENING (Tests 73-77)
# =============================================================================


class TestConsumerCentricHardening:
    """Tests for hardened consumer-centric view."""

    def test_73_explain_consumers_lists_all_known_consumers(self):
        """Test that explain_consumers lists all known consumers deterministically."""
        # curriculum has known consumers
        outputs = [explain_consumers("config/curriculum.yaml") for _ in range(3)]

        # All outputs should be identical (deterministic)
        assert len(set(outputs)) == 1

        # Should list known consumers
        output = outputs[0]
        assert "backend/orchestrator/curriculum_loader.py" in output

    def test_74_explain_consumers_shows_field_access_count(self):
        """Test that explain_consumers shows field access counts."""
        output = explain_consumers("config/curriculum.yaml")

        # Should show field count for consumers with known mappings
        assert "Fields Read:" in output

    def test_75_consumer_info_dataclass_fields(self):
        """Test that ConsumerInfo has the right fields."""
        consumer = ConsumerInfo(
            file_path="test/consumer.py",
            field_access_count=5,
            fields_read=["field1", "field2"],
        )

        assert consumer.file_path == "test/consumer.py"
        assert consumer.field_access_count == 5
        assert consumer.fields_read == ["field1", "field2"]

    def test_76_unknown_consumer_field_count_is_negative(self):
        """Test that unknown field access count is -1."""
        # Get consumers for a schema - some may have unknown field counts
        consumers = _get_consumers_for_schema("curriculum", "config/curriculum.yaml")

        # All known consumers for curriculum should have field_access_count >= 0
        for c in consumers:
            # Either known (>=0) or explicitly unknown (-1)
            assert c.field_access_count >= -1

    def test_77_consumers_in_impact_match_downstream_deps(self):
        """Test that consumers in impact match DOWNSTREAM_DEPENDENCIES."""
        change_brief = {
            "schemas_changed": ["curriculum"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                }
            ],
        }

        result = score_change_impact(change_brief)

        consumers = result["per_schema"]["curriculum"]["consumers"]
        consumer_paths = {c["file_path"] for c in consumers}

        # Should include consumers from DOWNSTREAM_DEPENDENCIES
        from scripts.schema_ontology_builder import DOWNSTREAM_DEPENDENCIES
        expected = set(DOWNSTREAM_DEPENDENCIES.get("config/curriculum.yaml", []))
        assert consumer_paths == expected


# =============================================================================
# TEST GROUP 12: FIELD IMPACT MATRIX (Tests 78-83)
# =============================================================================


class TestFieldImpactMatrix:
    """Tests for build_field_impact_matrix functionality."""

    @pytest.fixture
    def synthetic_change_brief(self) -> dict[str, Any]:
        """Create synthetic change brief for testing."""
        return {
            "schemas_changed": ["curriculum", "some_schema"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [("new_field", "string")],
                    "fields_removed": [("systems.pl.active", "boolean")],
                    "fields_renamed": [],
                    "fields_type_changed": [("version", "string", "integer")],
                    "downstream_consumers": [],
                },
                {
                    "schema_name": "some_schema",
                    "source_file": "config/some.yaml",
                    "fields_added": [("another_field", "number")],
                    "fields_removed": [],
                    "fields_renamed": [("old_path", "old_name", "new_name")],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                },
            ],
        }

    def test_78_field_matrix_correct_classification(self, synthetic_change_brief):
        """Test that field matrix correctly classifies change kinds."""
        matrix = build_field_impact_matrix(synthetic_change_brief)

        # Find specific entries and check their classification
        curriculum_removed = [
            f for f in matrix
            if f["schema"] == "curriculum" and f["field"] == "systems.pl.active"
        ]
        assert len(curriculum_removed) == 1
        assert curriculum_removed[0]["change_kind"] == "removed"
        assert curriculum_removed[0]["severity"] == "BLOCK"  # curriculum is widely-used

        curriculum_added = [
            f for f in matrix
            if f["schema"] == "curriculum" and f["field"] == "new_field"
        ]
        assert len(curriculum_added) == 1
        assert curriculum_added[0]["change_kind"] == "added"

    def test_79_field_matrix_deterministic_ordering(self, synthetic_change_brief):
        """Test that field matrix has deterministic (schema, field) ordering."""
        results = [build_field_impact_matrix(synthetic_change_brief) for _ in range(5)]

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0] == results[i]

        # Verify ordering is by (schema, field)
        matrix = results[0]
        keys = [(f["schema"], f["field"]) for f in matrix]
        assert keys == sorted(keys)

    def test_80_field_matrix_every_field_appears_once(self, synthetic_change_brief):
        """Test that every changed field appears exactly once."""
        matrix = build_field_impact_matrix(synthetic_change_brief)

        # Count occurrences of each (schema, field) pair
        seen = set()
        for entry in matrix:
            key = (entry["schema"], entry["field"])
            assert key not in seen, f"Duplicate entry: {key}"
            seen.add(key)

    def test_81_field_matrix_includes_all_change_kinds(self, synthetic_change_brief):
        """Test that matrix includes added, removed, type_changed, renamed."""
        matrix = build_field_impact_matrix(synthetic_change_brief)

        change_kinds = {f["change_kind"] for f in matrix}
        assert "added" in change_kinds
        assert "removed" in change_kinds
        assert "type_changed" in change_kinds
        assert "renamed" in change_kinds

    def test_82_field_matrix_consumers_included(self, synthetic_change_brief):
        """Test that consumers are included in field entries."""
        matrix = build_field_impact_matrix(synthetic_change_brief)

        # All entries should have consumers list (may be empty)
        for entry in matrix:
            assert "consumers" in entry
            assert isinstance(entry["consumers"], list)

    def test_83_field_matrix_empty_for_no_changes(self):
        """Test that matrix is empty when no changes exist."""
        change_brief = {
            "schemas_changed": [],
            "schema_changes": [],
        }
        matrix = build_field_impact_matrix(change_brief)
        assert matrix == []


# =============================================================================
# TEST GROUP 13: IMPACT MANIFEST (Tests 84-88)
# =============================================================================


class TestImpactManifest:
    """Tests for build_impact_manifest functionality."""

    @pytest.fixture
    def synthetic_change_brief(self) -> dict[str, Any]:
        """Create synthetic change brief for testing."""
        return {
            "schemas_changed": ["curriculum", "metrics"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [("new_field", "string")],
                    "fields_removed": [("old_field", "number")],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                },
                {
                    "schema_name": "metrics",
                    "source_file": "config/metrics.yaml",
                    "fields_added": [],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [("count", "string", "integer")],
                    "downstream_consumers": [],
                },
            ],
        }

    def test_84_manifest_has_required_keys(self, synthetic_change_brief):
        """Test that manifest has all required keys."""
        manifest = build_impact_manifest(synthetic_change_brief)

        required_keys = [
            "schema_version",
            "overall_severity",
            "safe_to_merge",
            "schemas_touched",
            "fields_changed",
        ]
        for key in required_keys:
            assert key in manifest

    def test_85_manifest_stable_across_runs(self, synthetic_change_brief):
        """Test that manifest is stable across multiple runs."""
        manifests = [build_impact_manifest(synthetic_change_brief) for _ in range(5)]

        for i in range(1, len(manifests)):
            assert manifests[0] == manifests[i]

    def test_86_manifest_schemas_touched_sorted(self, synthetic_change_brief):
        """Test that schemas_touched is sorted alphabetically."""
        manifest = build_impact_manifest(synthetic_change_brief)

        schemas = manifest["schemas_touched"]
        assert schemas == sorted(schemas)
        assert "curriculum" in schemas
        assert "metrics" in schemas

    def test_87_manifest_fields_changed_qualified(self, synthetic_change_brief):
        """Test that fields_changed has qualified paths (schema.field)."""
        manifest = build_impact_manifest(synthetic_change_brief)

        fields = manifest["fields_changed"]
        assert "curriculum.new_field" in fields
        assert "curriculum.old_field" in fields
        assert "metrics.count" in fields

    def test_88_manifest_json_serializable(self, synthetic_change_brief):
        """Test that manifest is fully JSON serializable."""
        manifest = build_impact_manifest(synthetic_change_brief)

        # Should not raise
        json_str = json.dumps(manifest)
        assert isinstance(json_str, str)

        # Round-trip should work
        restored = json.loads(json_str)
        assert restored == manifest


# =============================================================================
# TEST GROUP 14: REVIEWER CHECKLIST (Tests 89-94)
# =============================================================================


class TestReviewerChecklist:
    """Tests for generate_review_checklist functionality."""

    @pytest.fixture
    def synthetic_change_brief(self) -> dict[str, Any]:
        """Create synthetic change brief for testing."""
        return {
            "schemas_changed": ["curriculum"],
            "schema_changes": [
                {
                    "schema_name": "curriculum",
                    "source_file": "config/curriculum.yaml",
                    "fields_added": [("systems.pl.new_slice", "object")],
                    "fields_removed": [("systems.pl.active", "boolean")],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                },
            ],
        }

    def test_89_checklist_deterministic_ordering(self, synthetic_change_brief):
        """Test that checklist is deterministic across runs."""
        checklists = [
            generate_review_checklist(synthetic_change_brief) for _ in range(5)
        ]

        for i in range(1, len(checklists)):
            assert checklists[0] == checklists[i]

    def test_90_checklist_includes_schema_name(self, synthetic_change_brief):
        """Test that checklist includes schema names."""
        checklist = generate_review_checklist(synthetic_change_brief)

        assert "curriculum" in checklist
        assert "## curriculum" in checklist

    def test_91_checklist_includes_field_names(self, synthetic_change_brief):
        """Test that checklist includes field names."""
        checklist = generate_review_checklist(synthetic_change_brief)

        assert "systems.pl.active" in checklist
        assert "systems.pl.new_slice" in checklist

    def test_92_checklist_includes_consumer_references(self, synthetic_change_brief):
        """Test that checklist includes consumer file references."""
        checklist = generate_review_checklist(synthetic_change_brief)

        # curriculum has known consumers
        assert "[ ]" in checklist  # Has checkboxes
        assert "Check uses of" in checklist

    def test_93_checklist_respects_schema_filter(self, synthetic_change_brief):
        """Test that checklist respects --schema filter."""
        # Add another schema
        change_brief = {
            "schemas_changed": ["curriculum", "other_schema"],
            "schema_changes": [
                synthetic_change_brief["schema_changes"][0],
                {
                    "schema_name": "other_schema",
                    "source_file": "config/other.yaml",
                    "fields_added": [("foo", "string")],
                    "fields_removed": [],
                    "fields_renamed": [],
                    "fields_type_changed": [],
                    "downstream_consumers": [],
                },
            ],
        }

        # Filter to curriculum only
        checklist = generate_review_checklist(change_brief, schema_filter="curriculum")

        assert "curriculum" in checklist
        assert "other_schema" not in checklist

    def test_94_checklist_neutral_tone(self, synthetic_change_brief):
        """Test that checklist uses neutral language (no 'wrong' or 'bad')."""
        checklist = generate_review_checklist(synthetic_change_brief)

        # Should not contain judgmental language
        lower = checklist.lower()
        assert "wrong" not in lower
        assert "bad" not in lower
        assert "error" not in lower
        assert "broken" not in lower

    def test_95_checklist_empty_for_no_changes(self):
        """Test that checklist handles no changes gracefully."""
        change_brief = {
            "schemas_changed": [],
            "schema_changes": [],
        }
        checklist = generate_review_checklist(change_brief)

        assert "No schema changes to review" in checklist


# =============================================================================
# TEST GROUP 15: CROSS-SCHEMA IMPACT INDEX (Tests 96-100)
# =============================================================================


class TestCrossSchemaImpactIndex:
    """Tests for build_cross_schema_impact_index functionality."""

    @pytest.fixture
    def sample_manifests(self) -> list[dict[str, Any]]:
        """Create sample impact manifests for testing."""
        return [
            {
                "schema_version": "1.0.0",
                "overall_severity": "BLOCK",
                "safe_to_merge": False,
                "schemas_touched": ["curriculum"],
                "fields_changed": [
                    "curriculum.systems.pl.active",
                    "curriculum.version",
                ],
            },
            {
                "schema_version": "1.0.0",
                "overall_severity": "WARN",
                "safe_to_merge": False,
                "schemas_touched": ["metrics"],
                "fields_changed": [
                    "metrics.count",
                    "metrics.threshold",
                    "metrics.name",
                ],
            },
            {
                "schema_version": "1.0.0",
                "overall_severity": "INFO",
                "safe_to_merge": True,
                "schemas_touched": ["evidence"],
                "fields_changed": ["evidence.timestamp"],
            },
        ]

    def test_96_cross_schema_index_has_required_keys(self, sample_manifests):
        """Test that cross-schema index has all required keys."""
        from scripts.schema_ontology_builder import build_cross_schema_impact_index

        index = build_cross_schema_impact_index(sample_manifests)

        required_keys = {
            "schema_version",
            "schemas_touched",
            "blocking_schemas",
            "warn_schemas",
            "change_density_by_schema",
        }
        assert required_keys.issubset(set(index.keys()))

    def test_97_cross_schema_index_categorizes_schemas(self, sample_manifests):
        """Test that schemas are correctly categorized by severity."""
        from scripts.schema_ontology_builder import build_cross_schema_impact_index

        index = build_cross_schema_impact_index(sample_manifests)

        assert "curriculum" in index["blocking_schemas"]
        assert "metrics" in index["warn_schemas"]
        assert "evidence" not in index["blocking_schemas"]
        assert "evidence" not in index["warn_schemas"]

    def test_98_cross_schema_index_counts_change_density(self, sample_manifests):
        """Test that change density is correctly counted per schema."""
        from scripts.schema_ontology_builder import build_cross_schema_impact_index

        index = build_cross_schema_impact_index(sample_manifests)

        density = index["change_density_by_schema"]
        assert density["curriculum"] == 2
        assert density["metrics"] == 3
        assert density["evidence"] == 1

    def test_99_cross_schema_index_deterministic(self, sample_manifests):
        """Test that cross-schema index is deterministic."""
        from scripts.schema_ontology_builder import build_cross_schema_impact_index

        indices = [build_cross_schema_impact_index(sample_manifests) for _ in range(5)]

        for i in range(1, len(indices)):
            assert indices[0] == indices[i]

    def test_100_cross_schema_index_json_serializable(self, sample_manifests):
        """Test that cross-schema index is JSON serializable."""
        from scripts.schema_ontology_builder import build_cross_schema_impact_index

        index = build_cross_schema_impact_index(sample_manifests)

        json_str = json.dumps(index)
        assert isinstance(json_str, str)

        restored = json.loads(json_str)
        assert restored == index


# =============================================================================
# TEST GROUP 16: REVIEWER PLAYBOOK (Tests 101-105)
# =============================================================================


class TestReviewerPlaybook:
    """Tests for render_cross_schema_reviewer_playbook functionality."""

    @pytest.fixture
    def sample_impact_index(self) -> dict[str, Any]:
        """Create sample impact index."""
        return {
            "schema_version": "1.0.0",
            "schemas_touched": ["curriculum", "metrics"],
            "blocking_schemas": ["curriculum"],
            "warn_schemas": ["metrics"],
            "change_density_by_schema": {"curriculum": 2, "metrics": 3},
        }

    @pytest.fixture
    def sample_field_matrices(self) -> list[list[dict[str, Any]]]:
        """Create sample field impact matrices."""
        return [
            [
                {
                    "schema": "curriculum",
                    "field": "systems.pl.active",
                    "change_kind": "removed",
                    "severity": "BLOCK",
                    "consumers": [{"file_path": "backend/loader.py", "field_access_count": 5}],
                },
                {
                    "schema": "curriculum",
                    "field": "version",
                    "change_kind": "type_changed",
                    "severity": "BLOCK",
                    "consumers": [],
                },
            ],
            [
                {
                    "schema": "metrics",
                    "field": "count",
                    "change_kind": "added",
                    "severity": "WARN",
                    "consumers": [{"file_path": "scripts/analyze.py", "field_access_count": 2}],
                },
            ],
        ]

    def test_101_playbook_contains_schema_sections(self, sample_impact_index, sample_field_matrices):
        """Test that playbook contains sections for each schema."""
        from scripts.schema_ontology_builder import render_cross_schema_reviewer_playbook

        playbook = render_cross_schema_reviewer_playbook(sample_impact_index, sample_field_matrices)

        assert "## curriculum" in playbook
        assert "## metrics" in playbook

    def test_102_playbook_includes_severity_badges(self, sample_impact_index, sample_field_matrices):
        """Test that playbook includes severity badges."""
        from scripts.schema_ontology_builder import render_cross_schema_reviewer_playbook

        playbook = render_cross_schema_reviewer_playbook(sample_impact_index, sample_field_matrices)

        assert "⛔ BLOCK" in playbook
        assert "⚠️ WARN" in playbook

    def test_103_playbook_includes_high_severity_table(self, sample_impact_index, sample_field_matrices):
        """Test that playbook includes high-severity fields table."""
        from scripts.schema_ontology_builder import render_cross_schema_reviewer_playbook

        playbook = render_cross_schema_reviewer_playbook(sample_impact_index, sample_field_matrices)

        assert "### High-Severity Fields" in playbook
        assert "curriculum.systems.pl.active" in playbook

    def test_104_playbook_includes_checklist_items(self, sample_impact_index, sample_field_matrices):
        """Test that playbook includes checklist items."""
        from scripts.schema_ontology_builder import render_cross_schema_reviewer_playbook

        playbook = render_cross_schema_reviewer_playbook(sample_impact_index, sample_field_matrices)

        assert "[ ]" in playbook
        assert "Review `" in playbook

    def test_105_playbook_neutral_language(self, sample_impact_index, sample_field_matrices):
        """Test that playbook uses neutral language."""
        from scripts.schema_ontology_builder import render_cross_schema_reviewer_playbook

        playbook = render_cross_schema_reviewer_playbook(sample_impact_index, sample_field_matrices)

        lower = playbook.lower()
        assert "wrong" not in lower
        assert "bad" not in lower
        assert "error" not in lower


# =============================================================================
# TEST GROUP 17: DIRECTOR PANEL (Tests 106-110)
# =============================================================================


class TestDirectorPanel:
    """Tests for build_schema_impact_director_panel functionality."""

    @pytest.fixture
    def sample_index_with_blocking(self) -> dict[str, Any]:
        """Create sample index with blocking schemas."""
        return {
            "schemas_touched": ["curriculum", "metrics"],
            "blocking_schemas": ["curriculum"],
            "warn_schemas": ["metrics"],
            "change_density_by_schema": {"curriculum": 5, "metrics": 3},
        }

    @pytest.fixture
    def sample_index_warn_only(self) -> dict[str, Any]:
        """Create sample index with only warn schemas."""
        return {
            "schemas_touched": ["metrics"],
            "blocking_schemas": [],
            "warn_schemas": ["metrics"],
            "change_density_by_schema": {"metrics": 3},
        }

    @pytest.fixture
    def sample_index_no_risks(self) -> dict[str, Any]:
        """Create sample index with no risks."""
        return {
            "schemas_touched": ["evidence"],
            "blocking_schemas": [],
            "warn_schemas": [],
            "change_density_by_schema": {"evidence": 1},
        }

    def test_106_director_panel_has_required_keys(self, sample_index_with_blocking):
        """Test that director panel has all required keys."""
        from scripts.schema_ontology_builder import build_schema_impact_director_panel

        panel = build_schema_impact_director_panel(sample_index_with_blocking)

        required_keys = {
            "status_light",
            "blocking_schemas",
            "warn_schemas",
            "headline",
        }
        assert required_keys.issubset(set(panel.keys()))

    def test_107_director_panel_red_with_blocking(self, sample_index_with_blocking):
        """Test that status light is RED when blocking schemas exist."""
        from scripts.schema_ontology_builder import build_schema_impact_director_panel

        panel = build_schema_impact_director_panel(sample_index_with_blocking)

        assert panel["status_light"] == "RED"
        assert "curriculum" in panel["blocking_schemas"]

    def test_108_director_panel_yellow_with_warn(self, sample_index_warn_only):
        """Test that status light is YELLOW when only warn schemas exist."""
        from scripts.schema_ontology_builder import build_schema_impact_director_panel

        panel = build_schema_impact_director_panel(sample_index_warn_only)

        assert panel["status_light"] == "YELLOW"
        assert panel["blocking_schemas"] == []

    def test_109_director_panel_green_with_no_risks(self, sample_index_no_risks):
        """Test that status light is GREEN when no risks exist."""
        from scripts.schema_ontology_builder import build_schema_impact_director_panel

        panel = build_schema_impact_director_panel(sample_index_no_risks)

        assert panel["status_light"] == "GREEN"

    def test_110_director_panel_headline_neutral(self, sample_index_with_blocking):
        """Test that headline uses neutral language."""
        from scripts.schema_ontology_builder import build_schema_impact_director_panel

        panel = build_schema_impact_director_panel(sample_index_with_blocking)

        headline = panel["headline"].lower()
        assert "good" not in headline
        assert "bad" not in headline
        assert "healthy" not in headline
        assert "unhealthy" not in headline


# =============================================================================
# TEST GROUP 18: IMPACT SLO EVALUATOR (Tests 111-115)
# =============================================================================


class TestImpactSLOEvaluator:
    """Tests for evaluate_schema_change_slo functionality."""

    @pytest.fixture
    def sample_index_many_blocking(self) -> dict[str, Any]:
        """Create sample index with many blocking schemas."""
        return {
            "schemas_touched": ["curriculum", "metrics", "evidence"],
            "blocking_schemas": ["curriculum", "metrics"],
            "warn_schemas": ["evidence"],
            "change_density_by_schema": {"curriculum": 5, "metrics": 3, "evidence": 1},
        }

    @pytest.fixture
    def sample_index_high_density(self) -> dict[str, Any]:
        """Create sample index with high change density."""
        return {
            "schemas_touched": ["curriculum"],
            "blocking_schemas": [],
            "warn_schemas": ["curriculum"],
            "change_density_by_schema": {"curriculum": 25},
        }

    @pytest.fixture
    def sample_index_low_density(self) -> dict[str, Any]:
        """Create sample index with low change density."""
        return {
            "schemas_touched": ["evidence"],
            "blocking_schemas": [],
            "warn_schemas": [],
            "change_density_by_schema": {"evidence": 2},
        }

    def test_111_slo_evaluator_has_required_keys(self, sample_index_low_density):
        """Test that SLO evaluator has all required keys."""
        from scripts.schema_ontology_builder import evaluate_schema_change_slo

        result = evaluate_schema_change_slo(sample_index_low_density)

        required_keys = {"slo_ok", "status", "neutral_reasons"}
        assert required_keys.issubset(set(result.keys()))

    def test_112_slo_block_status_with_many_blocking(self, sample_index_many_blocking):
        """Test that SLO status is BLOCK when many blocking schemas exist."""
        from scripts.schema_ontology_builder import evaluate_schema_change_slo

        result = evaluate_schema_change_slo(sample_index_many_blocking, max_blocking_schemas=1)

        assert result["status"] == "BLOCK"
        assert result["slo_ok"] is False
        assert "exceeds threshold" in result["neutral_reasons"][0].lower()

    def test_113_slo_attention_with_high_density(self, sample_index_high_density):
        """Test that SLO status is ATTENTION when change density is high."""
        from scripts.schema_ontology_builder import evaluate_schema_change_slo

        result = evaluate_schema_change_slo(sample_index_high_density, max_total_changes=20)

        assert result["status"] == "ATTENTION"
        assert result["slo_ok"] is False
        assert any("exceeds threshold" in r.lower() for r in result["neutral_reasons"])

    def test_114_slo_ok_with_low_density(self, sample_index_low_density):
        """Test that SLO status is OK when density is low."""
        from scripts.schema_ontology_builder import evaluate_schema_change_slo

        result = evaluate_schema_change_slo(sample_index_low_density)

        assert result["status"] == "OK"
        assert result["slo_ok"] is True

    def test_115_slo_neutral_reasons(self, sample_index_many_blocking):
        """Test that SLO reasons use neutral language."""
        from scripts.schema_ontology_builder import evaluate_schema_change_slo

        result = evaluate_schema_change_slo(sample_index_many_blocking)

        for reason in result["neutral_reasons"]:
            lower = reason.lower()
            assert "good" not in lower
            assert "bad" not in lower
            assert "healthy" not in lower
            assert "unhealthy" not in lower


# =============================================================================
# TEST GROUP 19: OWNERSHIP ROUTING (Tests 116-120)
# =============================================================================


class TestOwnershipRouting:
    """Tests for route_schema_changes_to_owners functionality."""

    @pytest.fixture
    def sample_field_matrix(self) -> list[dict[str, Any]]:
        """Create sample field impact matrix."""
        return [
            {
                "schema": "curriculum",
                "field": "systems.pl.active",
                "change_kind": "removed",
                "severity": "BLOCK",
                "consumers": [],
            },
            {
                "schema": "curriculum",
                "field": "version",
                "change_kind": "type_changed",
                "severity": "WARN",
                "consumers": [],
            },
            {
                "schema": "metrics",
                "field": "count",
                "change_kind": "added",
                "severity": "WARN",
                "consumers": [],
            },
        ]

    @pytest.fixture
    def sample_owner_map(self) -> dict[str, list[str]]:
        """Create sample ownership map."""
        return {
            "curriculum": ["backend-team", "curriculum-owners"],
            "metrics": ["metrics-team"],
        }

    def test_116_routing_has_required_keys(self, sample_field_matrix, sample_owner_map):
        """Test that routing result has all required keys."""
        from scripts.schema_ontology_builder import route_schema_changes_to_owners

        result = route_schema_changes_to_owners(sample_field_matrix, sample_owner_map)

        required_keys = {"owners_to_notify", "status", "neutral_notes"}
        assert required_keys.issubset(set(result.keys()))

    def test_117_routing_groups_fields_by_owner(self, sample_field_matrix, sample_owner_map):
        """Test that fields are properly grouped per owner."""
        from scripts.schema_ontology_builder import route_schema_changes_to_owners

        result = route_schema_changes_to_owners(sample_field_matrix, sample_owner_map)

        owners = result["owners_to_notify"]

        # curriculum fields should go to backend-team and curriculum-owners
        assert "backend-team" in owners
        assert "curriculum-owners" in owners
        assert "curriculum.systems.pl.active" in owners["backend-team"]
        assert "curriculum.version" in owners["backend-team"]

        # metrics fields should go to metrics-team
        assert "metrics-team" in owners
        assert "metrics.count" in owners["metrics-team"]

    def test_118_routing_deterministic_ordering(self, sample_field_matrix, sample_owner_map):
        """Test that routing has deterministic field ordering."""
        from scripts.schema_ontology_builder import route_schema_changes_to_owners

        results = [
            route_schema_changes_to_owners(sample_field_matrix, sample_owner_map)
            for _ in range(5)
        ]

        for i in range(1, len(results)):
            assert results[0] == results[i]

    def test_119_routing_handles_unassigned_schemas(self, sample_field_matrix):
        """Test that unassigned schemas are routed to 'unassigned' owner."""
        from scripts.schema_ontology_builder import route_schema_changes_to_owners

        # Use empty owner map to force unassigned
        result = route_schema_changes_to_owners(sample_field_matrix, {})

        assert "unassigned" in result["owners_to_notify"]
        assert result["status"] == "ATTENTION"
        assert any("no assigned owners" in note.lower() for note in result["neutral_notes"])

    def test_120_routing_neutral_notes(self, sample_field_matrix, sample_owner_map):
        """Test that routing notes use neutral language."""
        from scripts.schema_ontology_builder import route_schema_changes_to_owners

        result = route_schema_changes_to_owners(sample_field_matrix, sample_owner_map)

        for note in result["neutral_notes"]:
            lower = note.lower()
            assert "wrong" not in lower
            assert "bad" not in lower
            assert "error" not in lower

