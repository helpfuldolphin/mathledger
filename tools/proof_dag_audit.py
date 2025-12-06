"""
Proof DAG Auditor — Standalone audit tool for proof dependency graph integrity.

STATUS: INTERNAL DIAGNOSTIC SPEC / TOOLING FOR PHASE II.
        Not part of Evidence Pack v1 claims.

        This module is a SKELETON with NotImplementedError stubs. It has NOT
        been used to audit any production DAG state. The First Organism test
        validates lineage for a single derivation; this tool is intended for
        future comprehensive audits but is not yet operational.

        RFL LOGS OUT OF SCOPE: The RFL (Reflective Feedback Loop) logs from
        50-cycle and 330-cycle runs do not create any DAG entries. RFL operates
        as a standalone refinement harness that emits JSONL logs but does not
        write to proof_parents or any ledger tables. Therefore, RFL logs are
        not subject to DAG invariants and would not be audited by this tool.

This module provides the ProofDagAuditor class for comprehensive DAG auditing,
including consistency checks, cycle detection, duplicate detection, statistics
computation, and report generation.

Usage (FUTURE — not yet implemented):
    from tools.proof_dag_audit import ProofDagAuditor

    auditor = ProofDagAuditor(connection_string="postgresql://...")
    report = auditor.audit_consistency()
    auditor.export_report(report, "audit_report.json")

See docs/PROOF_DAG_INVARIANTS.md for invariant specifications.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class AuditSeverity(Enum):
    """Severity levels for audit findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditFinding:
    """Individual audit finding with context."""

    invariant: str
    severity: AuditSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_count: int = 0


@dataclass
class DagStatistics:
    """Computed statistics for the proof DAG."""

    node_count: int = 0
    edge_count: int = 0
    root_count: int = 0
    leaf_count: int = 0
    interior_count: int = 0
    max_depth: int = 0
    avg_in_degree: float = 0.0
    avg_out_degree: float = 0.0
    proof_count: int = 0
    statement_count: int = 0


@dataclass
class AuditReport:
    """Complete audit report for the proof DAG."""

    timestamp: datetime
    ok: bool
    summary: str
    findings: List[AuditFinding] = field(default_factory=list)
    statistics: Optional[DagStatistics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "ok": self.ok,
            "summary": self.summary,
            "findings": [
                {
                    "invariant": f.invariant,
                    "severity": f.severity.value,
                    "message": f.message,
                    "details": f.details,
                    "affected_count": f.affected_count,
                }
                for f in self.findings
            ],
            "statistics": (
                {
                    "node_count": self.statistics.node_count,
                    "edge_count": self.statistics.edge_count,
                    "root_count": self.statistics.root_count,
                    "leaf_count": self.statistics.leaf_count,
                    "interior_count": self.statistics.interior_count,
                    "max_depth": self.statistics.max_depth,
                    "avg_in_degree": self.statistics.avg_in_degree,
                    "avg_out_degree": self.statistics.avg_out_degree,
                    "proof_count": self.statistics.proof_count,
                    "statement_count": self.statistics.statement_count,
                }
                if self.statistics
                else None
            ),
            "metadata": self.metadata,
        }


class ProofDagAuditor:
    """
    Comprehensive auditor for proof DAG integrity and consistency.

    **PHASE II — NOT YET IMPLEMENTED**

    This class is a skeleton. All primary methods raise NotImplementedError.
    No audit has been run against production data. This tooling exists for
    future work; it is not part of Evidence Pack v1 claims.

    Intended to implement invariant checks specified in PROOF_DAG_INVARIANTS.md:
    - INV-1: Acyclicity (no directed cycles)
    - INV-2: No self-loops
    - INV-3: No duplicate edges
    - INV-4: Hash/ID consistency
    - INV-5: Complete edges
    - INV-6: Edge index ordering
    - INV-7: Referential integrity (statements)
    - INV-8: Referential integrity (proofs)
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        connection: Optional[Any] = None,
    ) -> None:
        """
        Initialize auditor with database connection.

        Args:
            connection_string: PostgreSQL connection string (optional)
            connection: Existing database connection (optional)

        Either connection_string or connection must be provided.
        """
        self._connection_string = connection_string
        self._connection = connection
        self._cursor: Optional[Any] = None

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------

    def _get_connection(self) -> Any:
        """Get or create database connection."""
        # TODO: Implement connection management
        # - If self._connection is set, use it
        # - Otherwise, create new connection from connection_string
        # - Use psycopg2 or psycopg for PostgreSQL
        raise NotImplementedError("Connection management not yet implemented")

    def _get_cursor(self) -> Any:
        """Get database cursor."""
        # TODO: Implement cursor management
        raise NotImplementedError("Cursor management not yet implemented")

    # ------------------------------------------------------------------
    # Primary Audit Methods
    # ------------------------------------------------------------------

    def audit_consistency(self) -> AuditReport:
        """
        Run full consistency audit on the proof DAG.

        Checks all invariants (INV-1 through INV-8) and computes statistics.

        Returns:
            AuditReport with findings and statistics
        """
        # TODO: Implement full audit
        # 1. Load DAG from database
        # 2. Run in-memory checks (cycles, duplicates, self-loops, etc.)
        # 3. Run database checks (referential integrity)
        # 4. Compute statistics
        # 5. Compile findings into report
        raise NotImplementedError("audit_consistency not yet implemented")

    def detect_cycles(self) -> List[AuditFinding]:
        """
        Detect cycles in the proof DAG (INV-1).

        Uses Kahn's algorithm (topological sort) to identify nodes
        participating in cycles.

        Returns:
            List of findings for cycle violations
        """
        # TODO: Implement cycle detection
        # 1. Build adjacency list from edges
        # 2. Compute in-degrees for all nodes
        # 3. Run Kahn's algorithm
        # 4. Nodes with non-zero in-degree after processing are in cycles
        raise NotImplementedError("detect_cycles not yet implemented")

    def detect_duplicate_edges(self) -> List[AuditFinding]:
        """
        Detect duplicate edges in the proof DAG (INV-3).

        An edge is considered duplicate if the same (proof_id, child, parent)
        triple appears more than once.

        Returns:
            List of findings for duplicate edge violations
        """
        # TODO: Implement duplicate edge detection
        # 1. Group edges by (proof_id, child_key, parent_key)
        # 2. Flag groups with count > 1
        # 3. Create findings with details
        raise NotImplementedError("detect_duplicate_edges not yet implemented")

    def compute_statistics(self) -> DagStatistics:
        """
        Compute comprehensive statistics for the proof DAG.

        Statistics include:
        - Node/edge counts
        - Root/leaf/interior node counts
        - Maximum depth (longest path)
        - Average in/out degree
        - Proof and statement counts

        Returns:
            DagStatistics with computed values
        """
        # TODO: Implement statistics computation
        # 1. Count nodes and edges
        # 2. Identify roots (in-degree = 0) and leaves (out-degree = 0)
        # 3. Compute max depth via BFS from roots
        # 4. Calculate average degrees
        # 5. Query proof and statement counts
        raise NotImplementedError("compute_statistics not yet implemented")

    def export_report(
        self,
        report: AuditReport,
        output_path: str,
        format: str = "json",
    ) -> None:
        """
        Export audit report to file.

        Args:
            report: AuditReport to export
            output_path: Destination file path
            format: Output format ("json" or "markdown")
        """
        # TODO: Implement report export
        # 1. Convert report to appropriate format
        # 2. Write to file with proper encoding
        if format == "json":
            self._export_json(report, output_path)
        elif format == "markdown":
            self._export_markdown(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    # ------------------------------------------------------------------
    # Additional Audit Checks
    # ------------------------------------------------------------------

    def detect_self_loops(self) -> List[AuditFinding]:
        """
        Detect self-referential edges (INV-2).

        Returns:
            List of findings for self-loop violations
        """
        # TODO: Implement self-loop detection
        # Check for edges where child_key == parent_key
        raise NotImplementedError("detect_self_loops not yet implemented")

    def check_hash_consistency(self) -> List[AuditFinding]:
        """
        Check hash/ID consistency (INV-4).

        Verifies that each statement_id maps to exactly one hash.

        Returns:
            List of findings for consistency violations
        """
        # TODO: Implement hash consistency check
        # Build map of statement_id -> set of hashes
        # Flag IDs that map to multiple hashes
        raise NotImplementedError("check_hash_consistency not yet implemented")

    def check_edge_completeness(self) -> List[AuditFinding]:
        """
        Check that all edges have complete identifiers (INV-5).

        Returns:
            List of findings for incomplete edges
        """
        # TODO: Implement edge completeness check
        # Flag edges missing both child identifiers or both parent identifiers
        raise NotImplementedError("check_edge_completeness not yet implemented")

    def check_edge_index_ordering(self) -> List[AuditFinding]:
        """
        Check edge index ordering per proof (INV-6).

        Returns:
            List of findings for ordering violations
        """
        # TODO: Implement edge index ordering check
        # For each proof, verify edge_index forms [0, 1, 2, ...]
        raise NotImplementedError("check_edge_index_ordering not yet implemented")

    def check_referential_integrity(self) -> List[AuditFinding]:
        """
        Check referential integrity for statements and proofs (INV-7, INV-8).

        Returns:
            List of findings for referential integrity violations
        """
        # TODO: Implement referential integrity checks
        # 1. Check parent_statement_id references
        # 2. Check child_statement_id references
        # 3. Check proof_id references
        raise NotImplementedError("check_referential_integrity not yet implemented")

    # ------------------------------------------------------------------
    # RFL Audit Methods (Phase II Design Only — NOT IMPLEMENTED)
    # ------------------------------------------------------------------
    #
    # The following methods sketch future support for auditing RFL-originated
    # DAG entries. As of Phase I, RFL logs do not create DAG entries, so these
    # methods have no data to audit.
    #
    # When RFL integration is implemented (Phase II), these methods will:
    # - Filter for edges where origin LIKE 'rfl_%'
    # - Apply INV-RFL-1 through INV-RFL-4 (see PROOF_DAG_INVARIANTS.md)
    # - Cross-reference experiment manifests
    # - Validate verification receipts
    #
    # CLI flag design: --include-rfl
    #   When passed, the auditor will:
    #   1. Include RFL-originated edges in the audit scope
    #   2. Apply stricter provenance checks
    #   3. Require experiment manifest references
    #   4. Validate verification receipts exist
    #
    # CURRENT STATUS: NOT IMPLEMENTED. No RFL run writes to proof_parents.

    def audit_rfl_edges(self) -> List[AuditFinding]:
        """
        Audit RFL-originated DAG edges (Phase II — NOT IMPLEMENTED).

        This method will apply INV-RFL-1 through INV-RFL-4 to edges where
        origin LIKE 'rfl_%'. Currently raises NotImplementedError because
        no RFL run creates DAG entries.

        Future checks:
        - INV-RFL-1: Provenance tag present
        - INV-RFL-2: No cycles introduced
        - INV-RFL-3: Verification receipt linked
        - INV-RFL-4: Experiment manifest referenced

        Returns:
            List of findings for RFL-specific invariant violations
        """
        # PHASE II: Not implemented. No RFL edges exist in current DAG.
        # When implemented:
        # 1. Query: SELECT * FROM proof_parents WHERE origin LIKE 'rfl_%'
        # 2. For each edge, verify INV-RFL-1 through INV-RFL-4
        # 3. Cross-reference rfl_experiments and rfl_verification_receipts
        raise NotImplementedError(
            "audit_rfl_edges not implemented. "
            "No RFL run currently writes to proof_parents. "
            "This is Phase II design only."
        )

    def validate_rfl_experiment_manifest(
        self, experiment_id: str
    ) -> List[AuditFinding]:
        """
        Validate an RFL experiment manifest (Phase II — NOT IMPLEMENTED).

        Checks that the manifest exists, is sealed, and all referenced
        edges have valid verification receipts.

        Args:
            experiment_id: The RFL experiment identifier

        Returns:
            List of findings for manifest validation issues
        """
        # PHASE II: Not implemented.
        # When implemented:
        # 1. Load manifest from rfl_experiments table
        # 2. Verify manifest_hash matches content
        # 3. Count edges with origin = experiment_id
        # 4. Verify proofs_verified count matches
        raise NotImplementedError(
            "validate_rfl_experiment_manifest not implemented. "
            "Phase II design only."
        )

    # ------------------------------------------------------------------
    # Classification Methods
    # ------------------------------------------------------------------

    def find_root_nodes(self) -> Set[str]:
        """
        Find all root nodes (axioms) in the DAG.

        Root nodes have no incoming edges (in-degree = 0).

        Returns:
            Set of root node hashes
        """
        # TODO: Implement root node detection
        raise NotImplementedError("find_root_nodes not yet implemented")

    def find_leaf_nodes(self) -> Set[str]:
        """
        Find all leaf nodes (frontier theorems) in the DAG.

        Leaf nodes have no outgoing edges (out-degree = 0).

        Returns:
            Set of leaf node hashes
        """
        # TODO: Implement leaf node detection
        raise NotImplementedError("find_leaf_nodes not yet implemented")

    def compute_node_depths(self) -> Dict[str, int]:
        """
        Compute depth (distance from nearest root) for all nodes.

        Returns:
            Dict mapping node hash to depth
        """
        # TODO: Implement depth computation via BFS from roots
        raise NotImplementedError("compute_node_depths not yet implemented")

    # ------------------------------------------------------------------
    # Export Helpers
    # ------------------------------------------------------------------

    def _export_json(self, report: AuditReport, output_path: str) -> None:
        """Export report as JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    def _export_markdown(self, report: AuditReport, output_path: str) -> None:
        """Export report as Markdown."""
        # TODO: Implement Markdown export
        # Format report as readable Markdown with sections for:
        # - Summary
        # - Findings (grouped by severity)
        # - Statistics
        # - Metadata
        raise NotImplementedError("Markdown export not yet implemented")


# ---------------------------------------------------------------------------
# CLI Entry Point (optional)
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for DAG auditing."""
    # TODO: Implement CLI
    # - Parse arguments (connection string, output path, format)
    # - Create auditor
    # - Run audit
    # - Export report
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit proof DAG for integrity and consistency"
    )
    parser.add_argument(
        "--database-url",
        default="postgresql://ml:mlpass@localhost:5432/mathledger",
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="dag_audit_report.json",
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "markdown"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--include-rfl",
        action="store_true",
        help=(
            "Include RFL-originated edges in audit (Phase II — NOT IMPLEMENTED). "
            "When implemented, this will apply INV-RFL-1 through INV-RFL-4 to "
            "edges where origin LIKE 'rfl_%%'. Currently has no effect because "
            "no RFL run writes to proof_parents."
        ),
    )

    args = parser.parse_args()

    print(f"Connecting to: {args.database_url}")
    print("Running DAG audit...")

    if args.include_rfl:
        print(
            "WARNING: --include-rfl flag passed but RFL audit is NOT IMPLEMENTED.\n"
            "         No RFL run currently writes to proof_parents.\n"
            "         This flag is Phase II design only and has no effect."
        )

    # TODO: Instantiate auditor and run audit
    # auditor = ProofDagAuditor(connection_string=args.database_url)
    # report = auditor.audit_consistency()
    # if args.include_rfl:
    #     rfl_findings = auditor.audit_rfl_edges()  # Phase II
    #     report.findings.extend(rfl_findings)
    # auditor.export_report(report, args.output, format=args.format)
    # print(f"Report written to: {args.output}")

    print("NOTE: Audit implementation pending. See proof_dag_audit.py for skeleton.")


if __name__ == "__main__":
    main()
