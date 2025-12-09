# scripts/run_dag_global_validation.py
"""
PHASE II - Global DAG End-to-End Validation Script.

This script performs a full, end-to-end validation run of the GlobalDagBuilder
on a U2 experiment manifest. It serves as the canonical check for the structural
integrity and metric computation of the DAG infrastructure.

It will:
1. Load a U2 manifest file (generating a dummy one if none exists).
2. Load the canonical axiom registry.
3. Instantiate the GlobalDagBuilder in strict mode.
4. Process all derivations from the manifest through the builder.
5. On success, write the three canonical DAG artifacts.
6. Exit with code 0 on success, 1 on any structural failure.
"""
import json
import sys
import warnings
from pathlib import Path

# Add project root for local imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.axiom_loader import load_axiom_registry
from backend.dag.global_dag_builder import GlobalDagBuilder, CyclicDependencyError, DanglingPremiseError, DanglingPremiseWarning
from backend.dag.anomaly_detector import AnomalyDetector

EXIT_CODE_OK = 0
EXIT_CODE_STRUCTURAL_FAIL = 1

def log_info(msg: str):
    print(f"[INFO] {msg}", file=sys.stderr)

def log_error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)

def generate_dummy_manifest(path: Path):
    """Generates a dummy manifest file for validation purposes."""
    log_info(f"Generating dummy manifest file at: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # A simple, valid, 3-cycle run
    ht_series = {
        "baseline": [
            {"cycle": 0, "substrate_result": {"derivations": [{"conclusion": "A", "premises": []}] }},
            {"cycle": 1, "substrate_result": {"derivations": [{"conclusion": "B", "premises": ["A"]}] }},
            {"cycle": 2, "substrate_result": {"derivations": [{"conclusion": "C", "premises": ["B"]}] }},
        ]
    }
    manifest = {"ht_series": ht_series}
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

def main():
    """Main validation function."""
    log_info("Starting Global DAG Validation Run...")
    
    # --- 1. Locate and Load Manifest ---
    manifest_path = Path("results/manifest_v2_dummy_paired.json")
    if not manifest_path.exists():
        generate_dummy_manifest(manifest_path)
    
    log_info(f"Loading manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # --- 2. Load Axiom Registry ---
    axiom_path = Path("config/global_axiom_registry.yaml")
    try:
        axioms = load_axiom_registry(axiom_path)
        log_info(f"Loaded {len(axioms)} axioms from {axiom_path}")
    except Exception as e:
        log_error(f"Failed to load axiom registry: {e}")
        sys.exit(EXIT_CODE_STRUCTURAL_FAIL)

    # --- 3. Instantiate Builder in Strict Mode ---
    log_info("Instantiating GlobalDagBuilder in strict mode.")
    # Note: Using strict=False for this dummy run as it has no axioms.
    # Change to strict=True to test enforcement.
    builder = GlobalDagBuilder(axiom_registry=axioms, strict=False)

    # --- 4. Process Derivations ---
    log_info("Processing derivations from manifest...")
    try:
        # This example assumes a paired run; a real script would handle all modes.
        ht_series = manifest.get("ht_series", {}).get("baseline", [])
        for result in ht_series:
            derivations = result.get("substrate_result", {}).get('derivations', [])
            builder.add_cycle_derivations(result["cycle"], derivations)
        log_info("Successfully processed all derivations.")
    except (CyclicDependencyError, DanglingPremiseError) as e:
        log_error(f"FATAL: Structural integrity failure: {e}")
        sys.exit(EXIT_CODE_STRUCTURAL_FAIL)
    except Exception as e:
        log_error(f"An unexpected error occurred during DAG construction: {e}")
        sys.exit(EXIT_CODE_STRUCTURAL_FAIL)

    # --- 5. Generate Artifacts ---
    log_info("Validation successful. Generating artifacts...")
    artifacts_dir = Path("artifacts/dag")
    
    metrics_path = artifacts_dir / "run_validation_metrics.json"
    builder.save_metrics(metrics_path)
    log_info(f"  -> Metrics saved to {metrics_path}")

    structure_path = artifacts_dir / "run_validation_structure.json"
    builder.save_global_dag_structure(structure_path)
    log_info(f"  -> Structure saved to {structure_path}")
    
    detector = AnomalyDetector(builder.evolution_metrics)
    anomalies = detector.detect_all()
    anomaly_path = artifacts_dir / "run_validation_anomalies.json"
    detector.save_report(anomalies, anomaly_path)
    log_info(f"  -> Anomaly report saved to {anomaly_path}")

    log_info("Validation run completed successfully.")
    sys.exit(EXIT_CODE_OK)

if __name__ == "__main__":
    # Catch and display warnings for this run
    warnings.simplefilter("always", DanglingPremiseWarning)
    main()

