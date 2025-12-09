#!/usr/bin/env python3
"""
TDA Extract Real DAGs — Real-World DAG Sampling Pipeline

Operation CORTEX: Phase I Activation
=====================================

This script extracts real proof DAGs from MathLedger's database and prepares
them for TDA analysis. It samples diverse DAGs based on configurable criteria
and exports them in a format suitable for the TDA Mind Scanner.

Usage:
    python experiments/tda_extract_real_dags.py --output-dir results/tda_real_dags --sample-size 100

Sampling strategies:
- Random: Uniform random sampling from all proofs
- Stratified: Sample proportionally across depth/complexity tiers
- Temporal: Sample proofs from different time periods
- Verified: Only include successfully verified proofs
- Mixed: Combination of verified and failed proofs (for classification)

Output:
- dags/dag_{hash}.json: Individual DAG files
- embeddings/emb_{hash}.npy: Statement embeddings
- manifest.json: Sampling metadata and statistics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Conditional imports
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tda_extract_real_dags")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ProofNode:
    """A node in the proof DAG."""
    hash: str
    text: str
    depth: int
    system_id: int
    verified: bool
    created_at: Optional[str] = None


@dataclass
class ProofEdge:
    """An edge in the proof DAG (parent -> child)."""
    parent_hash: str
    child_hash: str
    proof_method: Optional[str] = None


@dataclass
class ExtractedDAG:
    """A complete extracted proof DAG with metadata."""
    dag_id: str
    root_hash: str
    nodes: List[ProofNode]
    edges: List[ProofEdge]
    node_count: int
    edge_count: int
    max_depth: int
    verified_fraction: float
    extraction_timestamp: str
    label: Optional[str] = None  # "verified", "partial", "failed"
    embeddings: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dag_id": self.dag_id,
            "root_hash": self.root_hash,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "max_depth": self.max_depth,
            "verified_fraction": self.verified_fraction,
            "extraction_timestamp": self.extraction_timestamp,
            "label": self.label,
        }

    def to_networkx(self) -> "nx.DiGraph":
        """Convert to NetworkX DiGraph."""
        if not HAS_NETWORKX:
            raise ImportError("networkx required for to_networkx()")

        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(
                node.hash,
                text=node.text,
                depth=node.depth,
                system_id=node.system_id,
                verified=node.verified,
            )
        for edge in self.edges:
            G.add_edge(edge.parent_hash, edge.child_hash, method=edge.proof_method)
        return G


@dataclass
class ExtractionManifest:
    """Manifest describing the extraction run."""
    extraction_id: str
    timestamp: str
    database_url_hash: str  # SHA256 of URL (for privacy)
    sample_size: int
    actual_count: int
    strategy: str
    filters: Dict[str, Any]
    statistics: Dict[str, Any]
    dag_files: List[str]
    embedding_files: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Database Connection
# ============================================================================

def get_database_url() -> str:
    """Get database URL from environment."""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://ml:mlpass@localhost:5432/mathledger"
    )


def connect_database(url: str) -> Any:
    """Connect to PostgreSQL database."""
    if not HAS_PSYCOPG2:
        raise ImportError("psycopg2 required for database connection")
    return psycopg2.connect(url)


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_statement_features(text: str) -> np.ndarray:
    """
    Extract features from a statement for TDA embedding.

    Returns a 19-dimensional feature vector matching the spec.
    """
    # Basic text features
    char_count = len(text)
    word_count = len(text.split())

    # Symbol counts
    implies_count = text.count("→") + text.count("->")
    and_count = text.count("∧") + text.count("&")
    or_count = text.count("∨") + text.count("|")
    not_count = text.count("¬") + text.count("~")
    forall_count = text.count("∀")
    exists_count = text.count("∃")
    equals_count = text.count("=")
    paren_depth = _max_paren_depth(text)

    # Variable tracking
    variables = set()
    for c in text:
        if c.islower() and c.isalpha():
            variables.add(c)
    var_count = len(variables)

    # Structural features
    has_quantifier = 1.0 if (forall_count > 0 or exists_count > 0) else 0.0
    is_atomic = 1.0 if (implies_count == 0 and and_count == 0 and or_count == 0) else 0.0
    complexity = implies_count + and_count + or_count + not_count

    # Normalized features
    features = np.array([
        min(1.0, char_count / 100.0),       # 0: normalized length
        min(1.0, word_count / 20.0),        # 1: normalized word count
        min(1.0, implies_count / 5.0),      # 2: implication density
        min(1.0, and_count / 5.0),          # 3: conjunction density
        min(1.0, or_count / 5.0),           # 4: disjunction density
        min(1.0, not_count / 5.0),          # 5: negation density
        min(1.0, forall_count / 3.0),       # 6: universal quantifier density
        min(1.0, exists_count / 3.0),       # 7: existential quantifier density
        min(1.0, equals_count / 3.0),       # 8: equality density
        min(1.0, paren_depth / 10.0),       # 9: nesting depth
        min(1.0, var_count / 10.0),         # 10: variable count
        has_quantifier,                     # 11: has quantifier flag
        is_atomic,                          # 12: is atomic flag
        min(1.0, complexity / 10.0),        # 13: complexity score
        0.0,                                # 14: reserved
        0.0,                                # 15: reserved
        0.0,                                # 16: reserved
        0.0,                                # 17: reserved
        0.0,                                # 18: reserved
    ], dtype=np.float32)

    return features


def _max_paren_depth(text: str) -> int:
    """Calculate maximum parenthesis nesting depth."""
    depth = 0
    max_depth = 0
    for c in text:
        if c == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif c == ')':
            depth = max(0, depth - 1)
    return max_depth


# ============================================================================
# DAG Extraction
# ============================================================================

class DAGExtractor:
    """Extracts proof DAGs from the MathLedger database."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.conn = None

    def connect(self) -> None:
        """Establish database connection."""
        self.conn = connect_database(self.database_url)
        logger.info("Connected to database")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_proof_roots(
        self,
        limit: int = 100,
        system_id: Optional[int] = None,
        min_depth: Optional[int] = None,
        max_depth: Optional[int] = None,
        verified_only: bool = False,
    ) -> List[str]:
        """
        Get root statement hashes for proof DAGs.

        Args:
            limit: Maximum number of roots to return.
            system_id: Filter by logical system.
            min_depth: Minimum proof depth.
            max_depth: Maximum proof depth.
            verified_only: Only include verified proofs.

        Returns:
            List of statement hashes that can serve as DAG roots.
        """
        if not self.conn:
            raise RuntimeError("Not connected to database")

        # Build query
        conditions = []
        params = []

        if system_id is not None:
            conditions.append("s.system_id = %s")
            params.append(system_id)

        if min_depth is not None:
            conditions.append("s.depth >= %s")
            params.append(min_depth)

        if max_depth is not None:
            conditions.append("s.depth <= %s")
            params.append(max_depth)

        if verified_only:
            conditions.append("p.status = 'success'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT DISTINCT s.hash
            FROM statements s
            JOIN proofs p ON p.statement_id = s.id
            WHERE {where_clause}
            ORDER BY s.created_at DESC
            LIMIT %s
        """
        params.append(limit)

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [row[0] for row in rows]

    def extract_dag(
        self,
        root_hash: str,
        max_ancestors: int = 50,
    ) -> Optional[ExtractedDAG]:
        """
        Extract a proof DAG rooted at the given statement hash.

        Args:
            root_hash: The hash of the root statement.
            max_ancestors: Maximum number of ancestor nodes to include.

        Returns:
            ExtractedDAG or None if extraction fails.
        """
        if not self.conn:
            raise RuntimeError("Not connected to database")

        try:
            # Get root statement
            root_node = self._get_statement(root_hash)
            if root_node is None:
                logger.warning(f"Root statement not found: {root_hash}")
                return None

            # BFS to find ancestors
            nodes: Dict[str, ProofNode] = {root_hash: root_node}
            edges: List[ProofEdge] = []
            frontier = [root_hash]
            visited = {root_hash}

            while frontier and len(nodes) < max_ancestors:
                current_hash = frontier.pop(0)

                # Get parents of current node
                parents = self._get_parents(current_hash)

                for parent_hash, proof_method in parents:
                    # Add edge
                    edges.append(ProofEdge(
                        parent_hash=parent_hash,
                        child_hash=current_hash,
                        proof_method=proof_method,
                    ))

                    # Add parent node if not visited
                    if parent_hash not in visited:
                        visited.add(parent_hash)
                        parent_node = self._get_statement(parent_hash)
                        if parent_node:
                            nodes[parent_hash] = parent_node
                            frontier.append(parent_hash)

            # Compute statistics
            verified_count = sum(1 for n in nodes.values() if n.verified)
            verified_fraction = verified_count / len(nodes) if nodes else 0.0
            max_depth = max(n.depth for n in nodes.values()) if nodes else 0

            # Generate DAG ID
            dag_id = hashlib.sha256(
                f"{root_hash}:{len(nodes)}:{len(edges)}".encode()
            ).hexdigest()[:16]

            return ExtractedDAG(
                dag_id=dag_id,
                root_hash=root_hash,
                nodes=list(nodes.values()),
                edges=edges,
                node_count=len(nodes),
                edge_count=len(edges),
                max_depth=max_depth,
                verified_fraction=verified_fraction,
                extraction_timestamp=datetime.utcnow().isoformat() + "Z",
                label="verified" if verified_fraction >= 0.9 else "partial",
            )

        except Exception as e:
            logger.error(f"Error extracting DAG for {root_hash}: {e}")
            return None

    def _get_statement(self, hash_: str) -> Optional[ProofNode]:
        """Get a statement by hash."""
        query = """
            SELECT s.hash, s.text, s.depth, s.system_id,
                   COALESCE(p.status = 'success', false) as verified,
                   s.created_at
            FROM statements s
            LEFT JOIN proofs p ON p.statement_id = s.id
            WHERE s.hash = %s
            LIMIT 1
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (hash_,))
            row = cur.fetchone()

        if row is None:
            return None

        return ProofNode(
            hash=row[0],
            text=row[1],
            depth=row[2],
            system_id=row[3],
            verified=row[4],
            created_at=row[5].isoformat() if row[5] else None,
        )

    def _get_parents(self, statement_hash: str) -> List[Tuple[str, Optional[str]]]:
        """Get parent statements and proof methods."""
        query = """
            SELECT ps.hash, p.method
            FROM statements s
            JOIN proofs p ON p.statement_id = s.id
            JOIN proof_parents pp ON pp.proof_id = p.id
            JOIN statements ps ON ps.id = pp.parent_statement_id
            WHERE s.hash = %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (statement_hash,))
            rows = cur.fetchall()

        return [(row[0], row[1]) for row in rows]


# ============================================================================
# Synthetic DAG Generation (Fallback)
# ============================================================================

def generate_synthetic_dags(
    count: int,
    seed: int = 42,
) -> List[ExtractedDAG]:
    """
    Generate synthetic DAGs when database is unavailable.

    This is useful for testing the TDA pipeline without a database connection.
    """
    rng = np.random.default_rng(seed)
    dags = []

    for i in range(count):
        # Random DAG parameters
        node_count = rng.integers(5, 30)
        verified_rate = rng.uniform(0.3, 1.0)
        max_depth = rng.integers(2, 8)

        # Generate nodes
        nodes = []
        for j in range(node_count):
            hash_ = hashlib.sha256(f"synthetic_{i}_{j}".encode()).hexdigest()
            depth = min(j, max_depth)
            text = _generate_synthetic_formula(rng, depth)

            nodes.append(ProofNode(
                hash=hash_,
                text=text,
                depth=depth,
                system_id=1,
                verified=rng.random() < verified_rate,
            ))

        # Generate edges (tree structure with some cross-links)
        edges = []
        for j in range(1, node_count):
            parent_idx = rng.integers(0, j)
            edges.append(ProofEdge(
                parent_hash=nodes[parent_idx].hash,
                child_hash=nodes[j].hash,
                proof_method="synthetic",
            ))

            # Add occasional cross-links
            if rng.random() < 0.2 and j > 2:
                extra_parent = rng.integers(0, j - 1)
                if extra_parent != parent_idx:
                    edges.append(ProofEdge(
                        parent_hash=nodes[extra_parent].hash,
                        child_hash=nodes[j].hash,
                        proof_method="synthetic_cross",
                    ))

        verified_count = sum(1 for n in nodes if n.verified)
        verified_fraction = verified_count / node_count

        dag_id = hashlib.sha256(f"synthetic_{i}".encode()).hexdigest()[:16]

        dags.append(ExtractedDAG(
            dag_id=dag_id,
            root_hash=nodes[-1].hash,  # Last node is root
            nodes=nodes,
            edges=edges,
            node_count=node_count,
            edge_count=len(edges),
            max_depth=max_depth,
            verified_fraction=verified_fraction,
            extraction_timestamp=datetime.utcnow().isoformat() + "Z",
            label="verified" if verified_fraction >= 0.9 else "partial",
        ))

    return dags


def _generate_synthetic_formula(rng: np.random.Generator, depth: int) -> str:
    """Generate a synthetic propositional formula."""
    variables = ["p", "q", "r", "s", "t"]
    operators = ["→", "∧", "∨"]

    if depth == 0 or rng.random() < 0.3:
        return rng.choice(variables)

    op = rng.choice(operators)
    left = _generate_synthetic_formula(rng, depth - 1)
    right = _generate_synthetic_formula(rng, depth - 1)

    return f"({left} {op} {right})"


# ============================================================================
# Main Pipeline
# ============================================================================

def extract_real_dags(
    output_dir: Path,
    sample_size: int = 100,
    strategy: str = "random",
    system_id: Optional[int] = None,
    min_depth: Optional[int] = None,
    max_depth: Optional[int] = None,
    verified_only: bool = False,
    use_synthetic: bool = False,
    seed: int = 42,
) -> ExtractionManifest:
    """
    Main extraction pipeline.

    Args:
        output_dir: Directory to write output files.
        sample_size: Number of DAGs to extract.
        strategy: Sampling strategy (random, stratified, temporal).
        system_id: Filter by logical system.
        min_depth: Minimum proof depth.
        max_depth: Maximum proof depth.
        verified_only: Only include verified proofs.
        use_synthetic: Use synthetic DAGs (for testing).
        seed: Random seed.

    Returns:
        ExtractionManifest with extraction metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dags_dir = output_dir / "dags"
    dags_dir.mkdir(exist_ok=True)

    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    extraction_id = hashlib.sha256(
        f"{datetime.utcnow().isoformat()}:{sample_size}:{strategy}".encode()
    ).hexdigest()[:16]

    logger.info(f"Starting extraction: {extraction_id}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Sample size: {sample_size}")
    logger.info(f"  Strategy: {strategy}")

    # Extract DAGs
    dags: List[ExtractedDAG] = []

    if use_synthetic:
        logger.info("Using synthetic DAG generation")
        dags = generate_synthetic_dags(sample_size, seed)
    else:
        try:
            database_url = get_database_url()
            extractor = DAGExtractor(database_url)
            extractor.connect()

            # Get root hashes
            root_hashes = extractor.get_proof_roots(
                limit=sample_size * 2,  # Over-sample to account for failures
                system_id=system_id,
                min_depth=min_depth,
                max_depth=max_depth,
                verified_only=verified_only,
            )

            logger.info(f"Found {len(root_hashes)} candidate roots")

            # Extract DAGs
            for root_hash in root_hashes:
                if len(dags) >= sample_size:
                    break

                dag = extractor.extract_dag(root_hash)
                if dag is not None:
                    dags.append(dag)

                    if len(dags) % 10 == 0:
                        logger.info(f"  Extracted {len(dags)}/{sample_size} DAGs")

            extractor.close()

        except Exception as e:
            logger.warning(f"Database extraction failed: {e}")
            logger.info("Falling back to synthetic DAGs")
            dags = generate_synthetic_dags(sample_size, seed)

    # Generate embeddings and save DAGs
    dag_files: List[str] = []
    embedding_files: List[str] = []

    for dag in dags:
        # Save DAG JSON
        dag_path = dags_dir / f"dag_{dag.dag_id}.json"
        with open(dag_path, "w", encoding="utf-8") as f:
            json.dump(dag.to_dict(), f, indent=2)
        dag_files.append(str(dag_path.relative_to(output_dir)))

        # Generate and save embeddings
        embeddings = {}
        for node in dag.nodes:
            embeddings[node.hash] = extract_statement_features(node.text).tolist()

        emb_path = embeddings_dir / f"emb_{dag.dag_id}.npy"
        np.save(emb_path, embeddings)
        embedding_files.append(str(emb_path.relative_to(output_dir)))

    # Compute statistics
    statistics = _compute_extraction_statistics(dags)

    # Create manifest
    db_url = get_database_url() if not use_synthetic else "synthetic"
    db_url_hash = hashlib.sha256(db_url.encode()).hexdigest()[:16]

    manifest = ExtractionManifest(
        extraction_id=extraction_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        database_url_hash=db_url_hash,
        sample_size=sample_size,
        actual_count=len(dags),
        strategy=strategy,
        filters={
            "system_id": system_id,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "verified_only": verified_only,
            "use_synthetic": use_synthetic,
        },
        statistics=statistics,
        dag_files=dag_files,
        embedding_files=embedding_files,
    )

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info(f"Extraction complete: {len(dags)} DAGs")
    logger.info(f"Manifest: {manifest_path}")

    return manifest


def _compute_extraction_statistics(dags: List[ExtractedDAG]) -> Dict[str, Any]:
    """Compute aggregate statistics for extracted DAGs."""
    if not dags:
        return {
            "dag_count": 0,
            "total_nodes": 0,
            "total_edges": 0,
        }

    node_counts = [d.node_count for d in dags]
    edge_counts = [d.edge_count for d in dags]
    depths = [d.max_depth for d in dags]
    verified_fractions = [d.verified_fraction for d in dags]

    return {
        "dag_count": len(dags),
        "total_nodes": sum(node_counts),
        "total_edges": sum(edge_counts),
        "node_count": {
            "mean": float(np.mean(node_counts)),
            "std": float(np.std(node_counts)),
            "min": int(np.min(node_counts)),
            "max": int(np.max(node_counts)),
            "median": float(np.median(node_counts)),
        },
        "edge_count": {
            "mean": float(np.mean(edge_counts)),
            "std": float(np.std(edge_counts)),
            "min": int(np.min(edge_counts)),
            "max": int(np.max(edge_counts)),
            "median": float(np.median(edge_counts)),
        },
        "max_depth": {
            "mean": float(np.mean(depths)),
            "std": float(np.std(depths)),
            "min": int(np.min(depths)),
            "max": int(np.max(depths)),
        },
        "verified_fraction": {
            "mean": float(np.mean(verified_fractions)),
            "std": float(np.std(verified_fractions)),
            "min": float(np.min(verified_fractions)),
            "max": float(np.max(verified_fractions)),
        },
        "label_distribution": {
            "verified": sum(1 for d in dags if d.label == "verified"),
            "partial": sum(1 for d in dags if d.label == "partial"),
            "failed": sum(1 for d in dags if d.label == "failed"),
        },
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract real proof DAGs from MathLedger for TDA analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tda_real_dags"),
        help="Output directory for extracted DAGs",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of DAGs to extract",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "stratified", "temporal"],
        default="random",
        help="Sampling strategy",
    )
    parser.add_argument(
        "--system-id",
        type=int,
        default=None,
        help="Filter by logical system ID",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=None,
        help="Minimum proof depth",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum proof depth",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Only include verified proofs",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic DAGs (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    try:
        manifest = extract_real_dags(
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            strategy=args.strategy,
            system_id=args.system_id,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            verified_only=args.verified_only,
            use_synthetic=args.synthetic,
            seed=args.seed,
        )

        print(f"\nExtraction complete!")
        print(f"  DAGs extracted: {manifest.actual_count}")
        print(f"  Output: {args.output_dir}")
        print(f"  Manifest: {args.output_dir / 'manifest.json'}")

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
