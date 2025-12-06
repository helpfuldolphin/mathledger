"""
# NOTE: Canonical substrate module; backend.* imports are forbidden here.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ProofEdge:
    """Directed parent→child relationship recorded for a proof."""

    proof_id: Optional[int]
    child_statement_id: Optional[int]
    child_hash: Optional[str]
    parent_statement_id: Optional[int]
    parent_hash: Optional[str]
    edge_index: int = 0


@dataclass
class ProofDagValidationReport:
    """Validation outcome for the proof DAG."""

    ok: bool
    summary: str
    issues: Dict[str, Any]
    metrics: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "issues": self.issues,
            "metrics": self.metrics,
        }


def _node_key(statement_id: Optional[int], statement_hash: Optional[str]) -> Optional[str]:
    if statement_id is not None:
        return f"id:{statement_id}"
    if statement_hash:
        return f"hash:{statement_hash}"
    return None


def _edge_sort_key(edge: ProofEdge) -> Tuple[Any, ...]:
    child_id = edge.child_statement_id if edge.child_statement_id is not None else -1
    parent_id = edge.parent_statement_id if edge.parent_statement_id is not None else -1
    child_hash = edge.child_hash or ""
    parent_hash = edge.parent_hash or ""
    proof_id = edge.proof_id if edge.proof_id is not None else -1
    return (proof_id, child_id, child_hash, parent_id, parent_hash, edge.edge_index)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return str(value)


class ProofDag:
    """In-memory proof dependency graph with lineage and integrity helpers."""

    def __init__(self, edges: Iterable[ProofEdge]):
        self._edges: List[ProofEdge] = sorted(list(edges), key=_edge_sort_key)
        self._parents_by_child_id: Dict[int, List[ProofEdge]] = defaultdict(list)
        self._parents_by_child_hash: Dict[str, List[ProofEdge]] = defaultdict(list)
        self._children_by_parent_id: Dict[int, List[ProofEdge]] = defaultdict(list)
        self._children_by_parent_hash: Dict[str, List[ProofEdge]] = defaultdict(list)
        self._hash_by_statement_id: Dict[int, str] = {}
        self._node_keys: Set[str] = set()
        self._issues: Dict[str, Any] = {}

        for edge in self._edges:
            child_id = edge.child_statement_id
            child_hash = edge.child_hash
            parent_id = edge.parent_statement_id
            parent_hash = edge.parent_hash

            if child_id is not None:
                self._parents_by_child_id[child_id].append(edge)
            if child_hash is not None:
                self._parents_by_child_hash[child_hash].append(edge)
            if parent_id is not None:
                self._children_by_parent_id[parent_id].append(edge)
            if parent_hash is not None:
                self._children_by_parent_hash[parent_hash].append(edge)

            child_key = _node_key(child_id, child_hash)
            parent_key = _node_key(parent_id, parent_hash)
            if child_key:
                self._node_keys.add(child_key)
            if parent_key:
                self._node_keys.add(parent_key)

            if child_id is None and child_hash is None:
                self._issues.setdefault("incomplete_edge", []).append(
                    {
                        "proof_id": edge.proof_id,
                        "edge_index": edge.edge_index,
                        "missing": "child",
                    }
                )
            if parent_id is None and parent_hash is None:
                self._issues.setdefault("incomplete_edge", []).append(
                    {
                        "proof_id": edge.proof_id,
                        "edge_index": edge.edge_index,
                        "missing": "parent",
                    }
                )

            if child_id is not None and child_hash is not None:
                prev_child = self._hash_by_statement_id.setdefault(child_id, child_hash)
                if prev_child != child_hash:
                    self._issues.setdefault("child_hash_mismatch", []).append(
                        {
                            "statement_id": child_id,
                            "hash_a": prev_child,
                            "hash_b": child_hash,
                        }
                    )

            if parent_id is not None and parent_hash is not None:
                prev_parent = self._hash_by_statement_id.setdefault(parent_id, parent_hash)
                if prev_parent != parent_hash:
                    self._issues.setdefault("parent_hash_mismatch", []).append(
                        {
                            "statement_id": parent_id,
                            "hash_a": prev_parent,
                            "hash_b": parent_hash,
                        }
                    )

    @property
    def edges(self) -> Sequence[ProofEdge]:
        return self._edges

    def parents_of(self, child_statement_id: int) -> List[ProofEdge]:
        return list(self._parents_by_child_id.get(child_statement_id, ()))

    def parents_of_hash(self, child_hash: str) -> List[ProofEdge]:
        return list(self._parents_by_child_hash.get(child_hash, ()))

    def children_of(self, parent_statement_id: int) -> List[ProofEdge]:
        return list(self._children_by_parent_id.get(parent_statement_id, ()))

    def children_of_hash(self, parent_hash: str) -> List[ProofEdge]:
        return list(self._children_by_parent_hash.get(parent_hash, ()))

    def ancestors(
        self,
        child_hash: str,
        *,
        max_depth: Optional[int] = None,
        include_self: bool = False,
    ) -> List[str]:
        """Return ancestor hashes (BFS order) optionally bounded by depth."""
        seen: Dict[str, int] = {}
        queue: deque[Tuple[str, int]] = deque([(child_hash, 0)])
        result: List[str] = []

        if include_self:
            seen[child_hash] = 0

        while queue:
            node_hash, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue

            for edge in self._parents_by_child_hash.get(node_hash, ()):
                parent_hash = edge.parent_hash
                if parent_hash is None:
                    continue
                next_depth = depth + 1
                if parent_hash not in seen or next_depth < seen[parent_hash]:
                    seen[parent_hash] = next_depth
                    result.append(parent_hash)
                    queue.append((parent_hash, next_depth))
        return result

    def descendants(
        self,
        parent_hash: str,
        *,
        max_depth: Optional[int] = None,
        include_self: bool = False,
    ) -> List[str]:
        """Return descendant hashes (BFS order) optionally bounded by depth."""
        seen: Dict[str, int] = {}
        queue: deque[Tuple[str, int]] = deque([(parent_hash, 0)])
        result: List[str] = []

        if include_self:
            seen[parent_hash] = 0

        while queue:
            node_hash, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue

            for edge in self._children_by_parent_hash.get(node_hash, ()):
                child_hash = edge.child_hash
                if child_hash is None:
                    continue
                next_depth = depth + 1
                if child_hash not in seen or next_depth < seen[child_hash]:
                    seen[child_hash] = next_depth
                    result.append(child_hash)
                    queue.append((child_hash, next_depth))
        return result

    def _detect_self_loops(self) -> List[Tuple[int, str]]:
        loops: List[Tuple[int, str]] = []
        for edge in self._edges:
            child_id = edge.child_statement_id
            parent_id = edge.parent_statement_id
            child_hash = edge.child_hash
            parent_hash = edge.parent_hash
            if (
                child_id is not None
                and parent_id is not None
                and child_id == parent_id
            ):
                loops.append((child_id, child_hash or ""))
            elif (
                child_hash is not None
                and parent_hash is not None
                and child_hash == parent_hash
            ):
                loops.append((-1, child_hash))
        return loops

    def _detect_duplicate_edges(self) -> List[Dict[str, Any]]:
        counts: Counter[Tuple[Any, Any, Any]] = Counter()
        for edge in self._edges:
            child_key = _node_key(edge.child_statement_id, edge.child_hash)
            parent_key = _node_key(edge.parent_statement_id, edge.parent_hash)
            if child_key is None or parent_key is None:
                continue
            counts[(edge.proof_id, child_key, parent_key)] += 1
        return [
            {
                "proof_id": proof_id,
                "child": child_key,
                "parent": parent_key,
                "count": count,
            }
            for (proof_id, child_key, parent_key), count in counts.items()
            if count > 1
        ]

    def _detect_cycles(self) -> List[str]:
        children: Dict[str, Set[str]] = defaultdict(set)
        indegree: Dict[str, int] = defaultdict(int)
        nodes: Set[str] = set()

        unresolved_edges: List[ProofEdge] = []
        for edge in self._edges:
            parent_key = _node_key(edge.parent_statement_id, edge.parent_hash)
            child_key = _node_key(edge.child_statement_id, edge.child_hash)
            if parent_key is None or child_key is None:
                unresolved_edges.append(edge)
                continue
            nodes.update((parent_key, child_key))
            if child_key not in children[parent_key]:
                children[parent_key].add(child_key)
                indegree[child_key] += 1
            indegree.setdefault(parent_key, indegree.get(parent_key, 0))

        if unresolved_edges:
            self._issues.setdefault("cycle_check_skipped_edges", []).extend(
                [
                    {
                        "proof_id": edge.proof_id,
                        "child_statement_id": edge.child_statement_id,
                        "child_hash": edge.child_hash,
                        "parent_statement_id": edge.parent_statement_id,
                        "parent_hash": edge.parent_hash,
                    }
                    for edge in unresolved_edges
                ]
            )

        queue: deque[str] = deque(node for node in nodes if indegree[node] == 0)
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if visited == len(nodes):
            return []
        return sorted(node for node in nodes if indegree[node] > 0)

    def validate(self) -> ProofDagValidationReport:
        """Run structural validation checks on the DAG."""
        issues: Dict[str, Any] = dict(self._issues)

        duplicates = self._detect_duplicate_edges()
        if duplicates:
            issues["duplicate_edges"] = duplicates

        self_loops = self._detect_self_loops()
        if self_loops:
            issues["self_loops"] = self_loops

        cycles = self._detect_cycles()
        if cycles:
            issues["cycle_nodes"] = cycles

        metrics = {
            "edge_count": len(self._edges),
            "node_count": len(self._node_keys),
            "children_with_ids": len(self._parents_by_child_id),
            "children_with_hashes": len(self._parents_by_child_hash),
            "parents_with_ids": len(self._children_by_parent_id),
            "parents_with_hashes": len(self._children_by_parent_hash),
        }

        ok = not issues
        summary = "Proof DAG valid" if ok else "Proof DAG validation failed"
        return ProofDagValidationReport(ok=ok, summary=summary, issues=issues, metrics=metrics)


class ProofDagRepository:
    """Database access wrapper for proof DAG operations."""

    def __init__(self, cur) -> None:
        self._cur = cur
        self._columns = _get_table_columns(cur, "proof_parents")
        self._statement_columns = _get_table_columns(cur, "statements")
        self._proof_columns = _get_table_columns(cur, "proofs")
        self._has_proof_id = "proof_id" in self._columns
        self._has_child_statement_id = "child_statement_id" in self._columns
        self._has_parent_statement_id = "parent_statement_id" in self._columns
        self._has_child_hash = "child_hash" in self._columns
        self._has_parent_hash = "parent_hash" in self._columns
        self._has_edge_index = "edge_index" in self._columns

        if not (
            self._has_child_statement_id
            or self._has_child_hash
            or self._has_proof_id
        ):
            raise RuntimeError(
                "proof_parents must expose child identifiers (statement_id, child_hash, or proof_id)"
            )
        if not (self._has_parent_statement_id or self._has_parent_hash):
            raise RuntimeError(
                "proof_parents must expose parent identifiers (statement_id or hash)"
            )
        if self._has_proof_id and "statement_id" not in self._proof_columns:
            raise RuntimeError(
                "proofs table must provide statement_id column when proof_id is stored in proof_parents"
            )
        self._hash_column = _resolve_hash_column(self._statement_columns)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def insert_edge(
        self,
        *,
        proof_id: Optional[int],
        child_statement_id: Optional[int],
        child_hash: Optional[str],
        parent_statement_id: Optional[int],
        parent_hash: Optional[str],
        edge_index: int = 0,
    ) -> None:
        columns: List[str] = []
        params: List[Any] = []

        if self._has_proof_id:
            if proof_id is None:
                raise ValueError("proof_id is required by proof_parents schema")
            columns.append("proof_id")
            params.append(proof_id)

        child_identifiers_available = self._has_child_statement_id or self._has_child_hash
        if not (child_identifiers_available or self._has_proof_id):
            raise RuntimeError("proof_parents schema lacks child identifiers")

        if self._has_child_statement_id:
            if child_statement_id is None:
                raise ValueError("child_statement_id is required by proof_parents schema")
            columns.append("child_statement_id")
            params.append(child_statement_id)

        if self._has_child_hash:
            if child_hash is None:
                raise ValueError("child_hash is required by proof_parents schema")
            columns.append("child_hash")
            params.append(child_hash)

        if self._has_parent_statement_id:
            if parent_statement_id is None:
                raise ValueError("parent_statement_id is required by proof_parents schema")
            columns.append("parent_statement_id")
            params.append(parent_statement_id)

        if self._has_parent_hash:
            if parent_hash is None:
                raise ValueError("parent_hash is required by proof_parents schema")
            columns.append("parent_hash")
            params.append(parent_hash)

        if not self._has_parent_statement_id and not self._has_parent_hash:
            raise RuntimeError("proof_parents schema lacks parent identifiers")

        if self._has_edge_index:
            columns.append("edge_index")
            params.append(edge_index)

        placeholders = ",".join(["%s"] * len(columns))
        sql = (
            f"INSERT INTO proof_parents ({', '.join(columns)}) "
            f"VALUES ({placeholders}) ON CONFLICT DO NOTHING"
        )
        self._cur.execute(sql, params)

    # ------------------------------------------------------------------
    # Read / validation path
    # ------------------------------------------------------------------

    def load_dag(self) -> ProofDag:
        """Load the full DAG from the database."""
        selects: List[Tuple[str, str]] = []
        proof_join_required = False
        child_statement_join_required = False
        parent_statement_join_required = False

        # proof_id
        if self._has_proof_id:
            selects.append(("proof_id", "pp.proof_id"))
        else:
            selects.append(("proof_id", "NULL::BIGINT"))

        # child statement id expression
        child_id_expr: Optional[str] = None
        if self._has_child_statement_id and self._has_proof_id:
            child_id_expr = "COALESCE(pp.child_statement_id, p.statement_id)"
            proof_join_required = True
        elif self._has_child_statement_id:
            child_id_expr = "pp.child_statement_id"
        elif self._has_proof_id:
            child_id_expr = "p.statement_id"
            proof_join_required = True

        if child_id_expr is not None:
            selects.append(("child_statement_id", child_id_expr))
        else:
            selects.append(("child_statement_id", "NULL::BIGINT"))

        # child hash expression
        if self._has_child_hash:
            selects.append(("child_hash", "pp.child_hash"))
        elif child_id_expr is not None:
            child_statement_join_required = True
            selects.append(("child_hash", f"cs.{self._hash_column}"))
        else:
            selects.append(("child_hash", "NULL::TEXT"))

        # parent statement id
        if self._has_parent_statement_id:
            selects.append(("parent_statement_id", "pp.parent_statement_id"))
        else:
            selects.append(("parent_statement_id", "NULL::BIGINT"))

        # parent hash
        if self._has_parent_hash:
            selects.append(("parent_hash", "pp.parent_hash"))
        elif self._has_parent_statement_id:
            parent_statement_join_required = True
            selects.append(("parent_hash", f"ps.{self._hash_column}"))
        else:
            selects.append(("parent_hash", "NULL::TEXT"))

        # edge index
        if self._has_edge_index:
            selects.append(("edge_index", "COALESCE(pp.edge_index, 0)"))
        else:
            selects.append(("edge_index", "0"))

        select_sql = ",\n                ".join(
            f"{expr} AS {alias}" for alias, expr in selects
        )

        joins: List[str] = []
        if proof_join_required:
            joins.append("LEFT JOIN proofs p ON p.id = pp.proof_id")
        if child_statement_join_required and child_id_expr is not None:
            joins.append(f"LEFT JOIN statements cs ON cs.id = {child_id_expr}")
        if parent_statement_join_required and self._has_parent_statement_id:
            joins.append("LEFT JOIN statements ps ON ps.id = pp.parent_statement_id")

        join_sql = ""
        if joins:
            join_sql = "\n            " + "\n            ".join(joins)

        order_by: List[str] = []
        if self._has_proof_id:
            order_by.append("pp.proof_id")
        if self._has_child_statement_id:
            order_by.append("pp.child_statement_id")
        if self._has_parent_statement_id:
            order_by.append("pp.parent_statement_id")
        if self._has_parent_hash and not self._has_parent_statement_id:
            order_by.append("pp.parent_hash")
        if self._has_edge_index:
            order_by.append("pp.edge_index")

        order_sql = ""
        if order_by:
            order_sql = "\n            ORDER BY " + ", ".join(order_by)

        sql = f"""
            SELECT
                {select_sql}
            FROM proof_parents pp{join_sql}{order_sql}
        """
        self._cur.execute(sql)
        rows = self._cur.fetchall()

        edges: List[ProofEdge] = []
        for row in rows:
            record = {alias: row[idx] for idx, (alias, _) in enumerate(selects)}
            edges.append(
                ProofEdge(
                    proof_id=_coerce_int(record["proof_id"]),
                    child_statement_id=_coerce_int(record["child_statement_id"]),
                    child_hash=_coerce_str(record["child_hash"]),
                    parent_statement_id=_coerce_int(record["parent_statement_id"]),
                    parent_hash=_coerce_str(record["parent_hash"]),
                    edge_index=_coerce_int(record["edge_index"]) or 0,
                )
            )
        return ProofDag(edges)

    def validate(self) -> ProofDagValidationReport:
        """Validate DAG contents against structural and DB-level invariants."""
        dag = self.load_dag()
        dag_report = dag.validate()
        db_issues = self._run_db_checks()

        issues = dict(dag_report.issues)
        issues.update(db_issues)

        metrics = dict(dag_report.metrics)
        metrics.update(
            {
                "db_missing_parents": db_issues.get("missing_parents", 0),
                "db_missing_children": db_issues.get("missing_children", 0),
                "db_missing_proofs": db_issues.get("missing_proofs", 0),
                "db_duplicate_edges": len(db_issues.get("duplicate_edges_db", [])),
            }
        )

        blocking_issue_keys = {
            "missing_parents",
            "missing_children",
            "missing_proofs",
            "duplicate_edges_db",
            "missing_parents_unverified",
            "missing_children_unverified",
            "duplicate_edges_unverified",
        }
        ok = dag_report.ok and not any(key in issues for key in blocking_issue_keys)
        summary = "Proof DAG valid" if ok else "Proof DAG validation failed"
        return ProofDagValidationReport(ok=ok, summary=summary, issues=issues, metrics=metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_db_checks(self) -> Dict[str, Any]:
        issues: Dict[str, Any] = {}

        missing_parents = self._count_missing_parents()
        if missing_parents is None:
            issues["missing_parents_unverified"] = (
                "proof_parents schema lacks resolvable parent identifiers"
            )
        elif missing_parents:
            issues["missing_parents"] = missing_parents

        missing_children = self._count_missing_children()
        if missing_children is None:
            issues["missing_children_unverified"] = (
                "proof_parents schema lacks resolvable child identifiers"
            )
        elif missing_children:
            issues["missing_children"] = missing_children

        missing_proofs = self._count_missing_proofs()
        if missing_proofs is None:
            if self._has_proof_id:
                # unreachable, but defensive
                issues["missing_proofs_unverified"] = "unable to verify proof references"
        elif missing_proofs:
            issues["missing_proofs"] = missing_proofs

        duplicates = self._find_duplicate_rows()
        if duplicates is None:
            issues["duplicate_edges_unverified"] = (
                "insufficient columns to detect duplicate proof edges"
            )
        elif duplicates:
            issues["duplicate_edges_db"] = duplicates

        return issues

    def _count_missing_parents(self) -> Optional[int]:
        if self._has_parent_statement_id:
            self._cur.execute(
                """
                SELECT COUNT(*)
                FROM proof_parents pp
                LEFT JOIN statements ps ON ps.id = pp.parent_statement_id
                WHERE ps.id IS NULL AND pp.parent_statement_id IS NOT NULL
                """
            )
            return int(self._cur.fetchone()[0] or 0)
        if self._has_parent_hash:
            self._cur.execute(
                f"""
                SELECT COUNT(*)
                FROM proof_parents pp
                LEFT JOIN statements ps
                       ON ps.{self._hash_column} = pp.parent_hash
                WHERE ps.id IS NULL AND pp.parent_hash IS NOT NULL
                """
            )
            return int(self._cur.fetchone()[0] or 0)
        return None

    def _count_missing_children(self) -> Optional[int]:
        if self._has_child_statement_id:
            self._cur.execute(
                """
                SELECT COUNT(*)
                FROM proof_parents pp
                LEFT JOIN statements cs ON cs.id = pp.child_statement_id
                WHERE cs.id IS NULL AND pp.child_statement_id IS NOT NULL
                """
            )
            return int(self._cur.fetchone()[0] or 0)
        if self._has_child_hash:
            self._cur.execute(
                f"""
                SELECT COUNT(*)
                FROM proof_parents pp
                LEFT JOIN statements cs
                       ON cs.{self._hash_column} = pp.child_hash
                WHERE cs.id IS NULL AND pp.child_hash IS NOT NULL
                """
            )
            return int(self._cur.fetchone()[0] or 0)
        if self._has_proof_id:
            self._cur.execute(
                """
                SELECT COUNT(*)
                FROM proof_parents pp
                LEFT JOIN proofs p ON p.id = pp.proof_id
                LEFT JOIN statements cs ON cs.id = p.statement_id
                WHERE p.id IS NOT NULL AND cs.id IS NULL
                """
            )
            return int(self._cur.fetchone()[0] or 0)
        return None

    def _count_missing_proofs(self) -> Optional[int]:
        if not self._has_proof_id:
            return None
        self._cur.execute(
            """
            SELECT COUNT(*)
            FROM proof_parents pp
            LEFT JOIN proofs p ON p.id = pp.proof_id
            WHERE pp.proof_id IS NOT NULL AND p.id IS NULL
            """
        )
        return int(self._cur.fetchone()[0] or 0)

    def _find_duplicate_rows(self) -> Optional[List[Dict[str, Any]]]:
        group_exprs: List[Tuple[str, str]] = []
        joins: List[str] = []

        if self._has_proof_id:
            group_exprs.append(("proof_id", "pp.proof_id"))
            joins.append("LEFT JOIN proofs p ON p.id = pp.proof_id")

        child_group_expr: Optional[Tuple[str, str]] = None
        if self._has_child_statement_id:
            child_group_expr = ("child", "pp.child_statement_id")
        elif self._has_child_hash:
            child_group_expr = ("child", "pp.child_hash")
        elif self._has_proof_id:
            child_group_expr = ("child", "p.statement_id")
        if child_group_expr:
            group_exprs.append(child_group_expr)

        if self._has_parent_statement_id:
            group_exprs.append(("parent", "pp.parent_statement_id"))
        elif self._has_parent_hash:
            group_exprs.append(("parent", "pp.parent_hash"))
        else:
            return None

        if len(group_exprs) < 2:
            return None

        select_clauses = ", ".join(expr for _, expr in group_exprs)
        group_by_clause = ", ".join(str(idx + 1) for idx in range(len(group_exprs)))

        join_sql = ""
        if joins:
            join_sql = "\n            " + "\n            ".join(
                j for idx, j in enumerate(joins) if idx == 0 or joins[idx] != joins[idx - 1]
            )

        sql = f"""
            SELECT {select_clauses}, COUNT(*) AS dup_count
            FROM proof_parents pp{join_sql}
            GROUP BY {group_by_clause}
            HAVING COUNT(*) > 1
        """
        self._cur.execute(sql)
        rows = self._cur.fetchall()
        if not rows:
            return []

        def _row_value(value: Any) -> Any:
            if isinstance(value, memoryview):
                return bytes(value)
            return value

        results: List[Dict[str, Any]] = []
        for row in rows:
            payload: Dict[str, Any] = {}
            for idx, (alias, _) in enumerate(group_exprs):
                if alias == "child":
                    payload["child"] = _row_value(row[idx])
                elif alias == "parent":
                    payload["parent"] = _row_value(row[idx])
                elif alias == "proof_id":
                    payload["proof_id"] = _row_value(row[idx])
            payload["count"] = int(row[len(group_exprs)])
            results.append(payload)
        return results


def _get_table_columns(cur, table: str) -> Set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return {row[0].lower() for row in cur.fetchall()}


def _resolve_hash_column(columns: Set[str]) -> str:
    for candidate in ("hash", "canonical_hash"):
        if candidate in columns:
            return candidate
    raise RuntimeError("statements table must expose hash or canonical_hash column")

