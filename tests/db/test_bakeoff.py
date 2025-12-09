# tests/db/test_bakeoff.py
"""
PHASE II - Bake-off test suite for GraphDBClient implementations.

This test suite runs a common set of validation and performance tests against
any class that implements the GraphDBClient interface. It uses pytest's
parametrization to run the same tests on both the InMemoryGraphDB and the
live Neo4jClient.
"""
import pytest
import time
from typing import Type

# Add project root for local imports
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.db.clients import GraphDBClient, InMemoryGraphDB, Neo4jClient

# --- Fixture Configuration ---

# List of all client classes to be tested
CLIENT_CLASSES = [InMemoryGraphDB]

# Check if Neo4j is available and add it to the test list if so
try:
    # Attempt to instantiate and connect the Neo4j client
    neo4j_client = Neo4jClient()
    neo4j_client.connect()
    neo4j_client.close()
    CLIENT_CLASSES.append(Neo4jClient)
    NEO4J_AVAILABLE = True
except (ImportError, ConnectionError) as e:
    print(f"\n[WARN] Neo4j client not available. Skipping Neo4j tests. Reason: {e}", file=sys.stderr)
    NEO4J_AVAILABLE = False

@pytest.fixture(params=CLIENT_CLASSES)
def db_client(request) -> GraphDBClient:
    """Pytest fixture to provide instances of each DB client."""
    client_class: Type[GraphDBClient] = request.param
    
    if client_class == Neo4jClient and not NEO4J_AVAILABLE:
        pytest.skip("Neo4j client is not available or could not connect.")
        
    client = client_class()
    client.connect()
    client.clear_all_data() # Ensure clean state for each test
    
    yield client # Provide the client to the test
    
    client.clear_all_data()
    client.close()

# --- Test Cases ---

def test_connection(db_client: GraphDBClient):
    """Test that connect and close methods work."""
    # The fixture already handles connect/close, so we just need to assert it's there
    assert db_client is not None

def test_add_and_get_ancestors(db_client: GraphDBClient):
    """Test adding derivations and fetching ancestors for correctness."""
    # A -> B -> C
    db_client.add_derivation("C", ("B",))
    db_client.add_derivation("B", ("A",))
    db_client.add_derivation("A", tuple())

    # Test ancestors of C
    ancestors_c = db_client.get_ancestors("C")
    assert ancestors_c == {"A", "B"}
    
    # Test ancestors of B
    ancestors_b = db_client.get_ancestors("B")
    assert ancestors_b == {"A"}

    # Test ancestors of A (an axiom)
    ancestors_a = db_client.get_ancestors("A")
    assert ancestors_a == set()
    
    # Test a non-existent node
    ancestors_z = db_client.get_ancestors("Z")
    assert ancestors_z == set()

def test_multi_proof_and_diamond(db_client: GraphDBClient):
    """Test a diamond dependency graph (D -> B, D -> C, B -> A, C -> A)."""
    db_client.add_derivation("A", ("B",))
    db_client.add_derivation("A", ("C",)) # second proof for A
    db_client.add_derivation("B", ("D",))
    db_client.add_derivation("C", ("D",))

    ancestors_a = db_client.get_ancestors("A")
    assert ancestors_a == {"B", "C", "D"}

    ancestors_b = db_client.get_ancestors("B")
    assert ancestors_b == {"D"}

@pytest.mark.performance
def test_performance_bake_off(db_client: GraphDBClient):
    """Measures and reports performance of bulk additions."""
    client_name = db_client.__class__.__name__
    num_derivations = 1000
    
    print(f"\n--- Performance Bake-off: {client_name} ---")

    # Generate a long chain of derivations
    derivations = [("h_0", tuple())]
    for i in range(1, num_derivations):
        derivations.append((f"h_{i}", (f"h_{i-1}",)))

    # --- Test Bulk Add Performance ---
    start_time = time.perf_counter()
    for conclusion, premises in derivations:
        db_client.add_derivation(conclusion, premises)
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"  -> add_derivation ({num_derivations} ops): {duration:.6f} seconds")
    
    # --- Test Read Performance ---
    start_time = time.perf_counter()
    ancestors = db_client.get_ancestors(f"h_{num_derivations-1}")
    end_time = time.perf_counter()
    read_duration = end_time - start_time
    
    assert len(ancestors) == num_derivations - 1
    print(f"  -> get_ancestors (deep query): {read_duration:.6f} seconds")
    print("------------------------------------------")
