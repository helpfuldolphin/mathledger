import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from rfl.config import RFLConfig
from rfl.runner import RFLRunner

def main():
    parser = argparse.ArgumentParser(description="RFL Experiment Executor")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--output-dir", type=str, help="Override artifacts directory")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    try:
        config = RFLConfig.from_json(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    if args.output_dir:
        print(f"Overriding artifacts dir: {args.output_dir}")
        config.artifacts_dir = args.output_dir

    print(f"Initializing RFL Runner for {config.experiment_id}...")
    runner = RFLRunner(config)

    print("Starting experiment execution...")
    results = runner.run_all()

    print("\nExperiment Complete.")
    print(f"Total Runs: {len(results['runs'])}")
    print(f"Successful: {results['execution_summary']['successful_runs']}")
    print(f"Metabolism Verification: {'PASS' if results['metabolism_verification']['passed'] else 'FAIL'}")
    
    if not results['metabolism_verification']['passed']:
        print(f"Reason: {results['metabolism_verification']['message']}")

if __name__ == "__main__":
    main()
