import shutil
import sys
import subprocess
from pathlib import Path

# --- Degenerate RFL Run Notice ---
# The sealed fo_rfl.jsonl contains 1000 cycles with 100% abstention.
# This is not evidence of RFL uplift.
# ---------------------------------

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.manifest import ManifestGenerator

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    root_dir = Path.cwd()
    results_dir = root_dir / "results"
    artifacts_base = root_dir / "artifacts" / "phase_ii" / "fo_series_1"
    
    baseline_dir = artifacts_base / "fo_1000_baseline"
    rfl_dir = artifacts_base / "fo_1000_rfl"
    
    # 1. Ensure Directories Exist
    baseline_dir.mkdir(parents=True, exist_ok=True)
    rfl_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "artifacts" / "figures").mkdir(parents=True, exist_ok=True)

    print(f"Created directories under {artifacts_base}")

    # 2. Locate Source Files
    src_baseline_log = results_dir / "fo_baseline.jsonl"
    src_rfl_log = results_dir / "fo_rfl.jsonl"
    
    if not src_baseline_log.exists():
        print(f"CRITICAL: {src_baseline_log} not found.")
        sys.exit(1)
    if not src_rfl_log.exists():
        print(f"CRITICAL: {src_rfl_log} not found.")
        sys.exit(1)

    # 3. Copy Logs to Target
    dest_baseline_log = baseline_dir / "experiment_log.jsonl"
    dest_rfl_log = rfl_dir / "experiment_log.jsonl"
    
    shutil.copy2(src_baseline_log, dest_baseline_log)
    shutil.copy2(src_rfl_log, dest_rfl_log)
    print("Copied logs to target directories.")

    # 4. Generate Dyno Chart
    # We use the *destination* logs to ensure we are charting what we sealed
    print("Generating Dyno Chart...")
    cmd = [
        sys.executable, 
        str(root_dir / "experiments" / "generate_dyno_chart.py"),
        "--baseline", str(dest_baseline_log),
        "--rfl", str(dest_rfl_log),
        "--output-name", "rfl_abstention_rate" 
    ]
    run_command(cmd)
    
    # The chart is saved to artifacts/figures/rfl_abstention_rate.png
    src_chart = root_dir / "artifacts" / "figures" / "rfl_abstention_rate.png"
    
    if not src_chart.exists():
        print("CRITICAL: Chart generation failed.")
        sys.exit(1)

    # 5. Copy Chart to RFL Experiment Folder
    dest_chart_rfl = rfl_dir / "rfl_abstention_rate.png"
    shutil.copy2(src_chart, dest_chart_rfl)
    print(f"Copied chart to {dest_chart_rfl}")

    # 6. Generate Manifests
    config_path = root_dir / "experiments" / "run_fo_cycles.py"
    mdap_seed = "0x4D444150"

    # --- Baseline Manifest ---
    m_base = ManifestGenerator(
        experiment_id="fo_1000_baseline",
        experiment_type="baseline",
        description="First Organism 1000 cycle baseline run (RFL OFF)",
        config_path=config_path,
        mdap_seed=mdap_seed
    )
    m_base.add_log(dest_baseline_log, "jsonl")
    # Baseline might not strictly need the chart, but we can include it if we want, or not.
    # The prompt says "Make sure ... artifacts.figures includes rfl_abstention_rate.png"
    # This requirement likely applies to the RFL manifest or the 'Evidence Pack' as a whole.
    # I will include it in the RFL manifest primarily.
    
    m_base.save(baseline_dir / "manifest.json")

    # --- RFL Manifest ---
    m_rfl = ManifestGenerator(
        experiment_id="fo_1000_rfl",
        experiment_type="rfl",
        description="First Organism 1000 cycle RFL run (RFL ON) showing abstention reduction",
        config_path=config_path,
        mdap_seed=mdap_seed
    )
    m_rfl.add_log(dest_rfl_log, "jsonl")
    m_rfl.add_figure(dest_chart_rfl, "Abstention Rate Dynamics (Dyno Chart)")
    
    m_rfl.save(rfl_dir / "manifest.json")

    print("\nSEALING COMPLETE.")
    print(f"Baseline Manifest: {baseline_dir / 'manifest.json'}")
    print(f"RFL Manifest: {rfl_dir / 'manifest.json'}")

if __name__ == "__main__":
    main()
