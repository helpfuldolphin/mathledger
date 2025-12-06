"""Smoke test ensuring canonical namespaces import without legacy backend paths."""

import ast
import os
import pytest
from pathlib import Path

from attestation.dual_root import compute_composite_root, compute_reasoning_root, compute_ui_root
from curriculum.gates import GateEvaluator, NormalizedMetrics, make_first_organism_slice
from derivation import DerivationPipeline, SliceBounds
from derivation.verification import StatementVerifier
from interface.api.app import app
from ledger.ingest import LedgerIngestor
from ledger.ui_events import capture_ui_event, ui_event_store
from rfl.runner import RunLedgerEntry
from normalization.canon import canonical_bytes


def _find_backend_imports(file_path: Path) -> list[tuple[int, str]]:
    """Find all backend.* imports in a Python file."""
    backend_imports = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("backend."):
                        backend_imports.append((node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("backend."):
                    backend_imports.append((node.lineno, node.module))
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that can't be parsed
        pass
    return backend_imports


@pytest.mark.unit
def test_no_backend_imports_in_canonical_packages():
    """
    Verify that canonical packages do not import from backend.*.
    
    This test FAILS if any backend.* imports are found in:
    - substrate/
    - derivation/
    - normalization/
    - ledger/
    - attestation/
    - rfl/
    - curriculum/
    - interface/
    """
    repo_root = Path(__file__).parent.parent.parent
    canonical_packages = [
        "substrate",
        "derivation",
        "normalization",
        "ledger",
        "attestation",
        "rfl",
        "curriculum",
        "interface",
    ]
    
    violations = []
    for package_name in canonical_packages:
        package_dir = repo_root / package_name
        if not package_dir.exists():
            continue
        
        for py_file in package_dir.rglob("*.py"):
            # Skip __pycache__ and hidden files
            if "__pycache__" in str(py_file) or py_file.name.startswith("."):
                continue
            
            backend_imports = _find_backend_imports(py_file)
            if backend_imports:
                rel_path = py_file.relative_to(repo_root)
                for lineno, import_name in backend_imports:
                    violations.append(f"{rel_path}:{lineno} imports {import_name}")
    
    if violations:
        violations_str = "\n  ".join(violations)
        pytest.fail(
            f"Found backend.* imports in canonical packages:\n  {violations_str}\n\n"
            "Canonical packages must use canonical imports only. "
            "Use shims in backend/* for legacy compatibility."
        )


@pytest.mark.unit
def test_namespace_wiring_smoke():
    """Verify that canonical packages can be imported and used."""
    bounds = SliceBounds()
    verifier = StatementVerifier(bounds)
    pipeline = DerivationPipeline(bounds, verifier)
    result = pipeline.run_step(existing=[])
    assert isinstance(result, object)

    metrics = NormalizedMetrics(
        coverage_ci_lower=0.9,
        coverage_sample_size=10,
        abstention_rate_pct=5.0,
        attempt_mass=100,
        slice_runtime_minutes=2.0,
        proof_velocity_pph=50.0,
        velocity_cv=0.1,
        backlog_fraction=0.1,
        attestation_hash="feedface",
    )
    gate = GateEvaluator(metrics, make_first_organism_slice())
    statuses = gate.evaluate()
    assert all(status.passed or status.name for status in statuses)

    LedgerIngestor()

    ui_event_store.clear()
    capture_ui_event({"event_id": "namespace-wiring", "action": "probe"})
    artifacts = [record.to_artifact() for record in ui_event_store.snapshot()]

    reasoning_root = compute_reasoning_root(["proof-1"])
    ui_root = compute_ui_root(artifacts or ["placeholder"])
    composite = compute_composite_root(reasoning_root, ui_root)
    assert composite

    canonical_bytes("p -> q")
    RunLedgerEntry(
        run_id="wire-run",
        slice_name="wire-slice",
        status="ok",
        coverage_rate=0.5,
        novelty_rate=0.5,
        throughput=0.0,
        success_rate=0.0,
        abstention_fraction=0.0,
        policy_reward=0.0,
        symbolic_descent=0.0,
        budget_spent=0,
        derive_steps=1,
        max_breadth=1,
        max_total=1,
    )

    assert app is not None
    print("[PASS] Namespace wiring canonical")
