# backend/db/clients.py
"""
PHASE II - Graph Database Client Interface and Implementations.

This module defines the abstract interface for a graph database client and
provides concrete implementations for different backends. This allows the
system to be tested without a live database and to switch between database
technologies.
"""
import abc
from typing import Dict, List, Set, Tuple

# Announce compliance on import
print("PHASE II — NOT USED IN PHASE I: Loading GraphDB Clients.", file=__import__("sys").stderr)

class GraphDBClient(abc.ABC):
    """Abstract Base Class for a graph database client."""

    @abc.abstractmethod
    def connect(self):
        """Establish a connection to the database."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """Close the connection to the database."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_derivation(self, conclusion: str, premises: Tuple[str, ...]):
        """
        Adds a single derivation to the database atomically.
        This includes creating nodes for the conclusion and all premises if
        they do not already exist.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_ancestors(self, node: str) -> Set[str]:
        """
        Fetches the set of all ancestors for a given node.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def clear_all_data(self):
        """Utility method to clear the entire database, for testing."""
        raise NotImplementedError

class InMemoryGraphDB(GraphDBClient):
    """
    An in-memory implementation of the GraphDBClient for testing.
    It mimics the behavior of a graph database using dictionaries.
    """
    def __init__(self):
        # conclusion -> set of premise-tuples
        self._dag: Dict[str, Set[Tuple[str, ...]]] = {}
        self._nodes: Set[str] = set()
        self.connected = False

    def connect(self):
        print("[INFO] In-memory DB connected.", file=__import__("sys").stderr)
        self.connected = True

    def close(self):
        print("[INFO] In-memory DB disconnected.", file=__import__("sys").stderr)
        self.connected = False
    
    def add_derivation(self, conclusion: str, premises: Tuple[str, ...]):
        if not self.connected:
            raise ConnectionError("Database is not connected.")
        
        # Add all nodes involved
        self._nodes.add(conclusion)
        self._nodes.update(premises)
        
        # Add the derivation edge
        if conclusion not in self._dag:
            self._dag[conclusion] = set()
        self._dag[conclusion].add(premises)

    def get_ancestors(self, node: str) -> Set[str]:
        if not self.connected:
            raise ConnectionError("Database is not connected.")
        if node not in self._nodes:
            return set()
            
        ancestors = set()
        q = [node]
        visited = {node}
        
        while q:
            current = q.pop(0)
            
            # Find parents (premises) of the current node
            if current in self._dag:
                 for proof in self._dag[current]:
                    for premise in proof:
                        if premise not in visited:
                            ancestors.add(premise)
                            visited.add(premise)
                            q.append(premise)
        return ancestors

    def clear_all_data(self):
        if not self.connected:
            raise ConnectionError("Database is not connected.")
        self._dag = {}
        self._nodes = set()
        print("[INFO] In-memory DB cleared.", file=__import__("sys").stderr)

class Neo4jClient(GraphDBClient):
    """
    A graph database client for a Neo4j backend.
    
    NOTE: Requires the `neo4j` Python package to be installed (`pip install neo4j`).
    NOTE: Expects a running Neo4j instance.
    """
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        # The neo4j driver is an optional dependency.
        try:
            from neo4j import GraphDatabase
            self._driver_module = GraphDatabase
        except ImportError:
            raise ImportError("Neo4j client requires the 'neo4j' package. Please install it using 'pip install neo4j'.")
        
        self._uri = uri
        self._auth = (user, password)
        self._driver = None

    def connect(self):
        """Establishes a connection to the Neo4j database."""
        if self._driver:
            self.close()
        try:
            self._driver = self._driver_module.driver(self._uri, auth=self._auth)
            self._driver.verify_connectivity()
            self._ensure_constraints()
            print(f"[INFO] Neo4j client connected to {self._uri}.", file=__import__("sys").stderr)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j at {self._uri}: {e}")

    def close(self):
        """Closes the connection to the database."""
        if self._driver:
            self._driver.close()
            self._driver = None
            print("[INFO] Neo4j client disconnected.", file=__import__("sys").stderr)
    
    def _ensure_constraints(self):
        """Ensures the unique constraint on Formula hash is set."""
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT formula_hash_unique IF NOT EXISTS FOR (f:Formula) REQUIRE f.hash IS UNIQUE")

    def add_derivation(self, conclusion: str, premises: Tuple[str, ...]):
        """
        Adds a single derivation to the database atomically using a single
        transaction with MERGE operations.
        """
        if not self._driver:
            raise ConnectionError("Database is not connected.")
            
        with self._driver.session() as session:
            session.execute_write(self._add_derivation_tx, conclusion, premises)

    @staticmethod
    def _add_derivation_tx(tx, conclusion, premises):
        # 1. Ensure the conclusion node exists
        tx.run("MERGE (c:Formula {hash: $conclusion})", conclusion=conclusion)
        
        # 2. For each premise, ensure it exists and create the relationship
        # This uses a more efficient UNWIND approach
        if premises:
            tx.run("""
                UNWIND $premises as p_hash
                MERGE (p:Formula {hash: p_hash})
                WITH p
                MATCH (c:Formula {hash: $conclusion})
                MERGE (p)-[:IS_PREMISE_FOR]->(c)
            """, premises=list(premises), conclusion=conclusion)

    def get_ancestors(self, node: str) -> Set[str]:
        """
        Fetches all ancestors (recursive premises) for a given node hash.
        Uses a Cypher query with variable-length path matching.
        """
        if not self._driver:
            raise ConnectionError("Database is not connected.")
            
        with self._driver.session() as session:
            result = session.run("""
                MATCH (start_node:Formula {hash: $node})<-[:IS_PREMISE_FOR*]-(ancestor:Formula)
                RETURN COLLECT(DISTINCT ancestor.hash) as ancestors
            """, node=node)
            record = result.single()
            return set(record["ancestors"]) if record and record["ancestors"] else set()

    def clear_all_data(self):
        """Deletes all nodes and relationships from the database."""
        if not self._driver:
            raise ConnectionError("Database is not connected.")
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[INFO] Neo4j DB cleared.", file=__import__("sys").stderr)