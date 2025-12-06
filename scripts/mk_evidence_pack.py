#!/usr/bin/env python3
"""
Evidence Pack Generator for WPV5 Analysis
Creates artifacts/wpv5/EVIDENCE.md with validator results, metrics snapshot,
policy hash, and last ablation plot path.
"""
import os
import sys
import json
import subprocess
import csv
import statistics as stats
from datetime import datetime
from typing import Dict, Any, Optional

def run_command(cmd: list, capture_output: bool = True) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)

def get_validator_results() -> Dict[str, Any]:
    """Run validator and capture results."""
    results = {}

    # Run guidance gate validation
    rc, stdout, stderr = run_command([
        "python", "scripts/validate_metrics.py",
        "--system", "evidence_pack",
        "--guidance-gate",
        "--target-pps", "0.005"
    ])

    results["guidance_gate"] = {
        "return_code": rc,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "passed": rc == 0
    }

    # Run ScaleA validation
    rc, stdout, stderr = run_command([
        "python", "scripts/validate_metrics.py",
        "--system", "evidence_pack",
        "--target-pps", "0.005"
    ])

    results["scaleA"] = {
        "return_code": rc,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "passed": rc == 0
    }

    return results

def get_metrics_snapshot() -> Dict[str, Any]:
    """Get metrics snapshot from various sources."""
    metrics = {}

    # Try to get database metrics
    try:
        from backend.tools.progress import get_latest_run_data

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL not set")
        data = get_latest_run_data(db_url, offline=True)  # Use offline mode for safety

        if data:
            metrics["proofs_success"] = data.get("proofs_success", 0)
            metrics["blocks_height"] = data.get("latest_block", {}).get("block_height", 0)
            metrics["latest_merkle"] = data.get("latest_block", {}).get("merkle_root", "N/A")
        else:
            metrics["proofs_success"] = 0
            metrics["blocks_height"] = 0
            metrics["latest_merkle"] = "N/A"
    except Exception as e:
        metrics["proofs_success"] = f"Error: {e}"
        metrics["blocks_height"] = f"Error: {e}"
        metrics["latest_merkle"] = f"Error: {e}"

    # Try to get Redis length
    try:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
        r = redis.from_url(redis_url)
        # Try to get queue length (common queue name)
        queue_length = r.llen("derivation_queue")
        metrics["redis_len"] = queue_length
    except Exception as e:
        metrics["redis_len"] = f"Error: {e}"

    return metrics

def get_policy_hash() -> str:
    """Get policy hash from policy.json or policy_inference."""
    # Try policy.json first
    policy_paths = [
        "artifacts/policy/policy.json",
        "artifacts/policy/policy_inference.json",
        "artifacts/policy.json"
    ]

    for path in policy_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    policy_data = json.load(f)
                    return policy_data.get("hash", "N/A")
            except Exception:
                continue

    # Try to get from database
    try:
        from backend.tools.progress import get_latest_run_data
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return "N/A (DATABASE_URL not set)"
        # This would need to be implemented in the progress module
        return "N/A (DB query not implemented)"
    except Exception:
        return "N/A (No policy found)"

def get_last_ablation_plot_path() -> str:
    """Get the path to the last ablation plot."""
    plot_path = "artifacts/wpv5/throughput_vs_depth.png"
    if os.path.exists(plot_path):
        return plot_path
    return "N/A (Plot not found)"

def get_csv_summary() -> Dict[str, Any]:
    """Get summary statistics from CSV files."""
    summary = {}

    csv_files = {
        "baseline": "artifacts/wpv5/baseline_runs.csv",
        "guided": "artifacts/wpv5/guided_runs.csv"
    }

    for name, path in csv_files.items():
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                pph_values = []
                for row in rows:
                    try:
                        pph = float(row.get("proofs_per_hour", 0))
                        if pph > 0:
                            pph_values.append(pph)
                    except (ValueError, TypeError):
                        continue

                if pph_values:
                    summary[name] = {
                        "count": len(pph_values),
                        "mean": stats.mean(pph_values),
                        "min": min(pph_values),
                        "max": max(pph_values)
                    }
                else:
                    summary[name] = {"count": 0, "mean": 0, "min": 0, "max": 0}
            except Exception as e:
                summary[name] = {"error": str(e)}
        else:
            summary[name] = {"error": "File not found"}

    return summary

def create_evidence_markdown(validator_results: Dict[str, Any],
                           metrics: Dict[str, Any],
                           policy_hash: str,
                           plot_path: str,
                           csv_summary: Dict[str, Any]) -> str:
    """Create the evidence markdown content."""

    timestamp = datetime.now().isoformat()

    # Helper function to format mean values safely
    def format_mean(data, key):
        mean_val = data.get(key, {}).get('mean', 'N/A')
        if isinstance(mean_val, (int, float)):
            return f"{mean_val:.1f}"
        return 'N/A'

    content = f"""# WPV5 Evidence Pack
Generated: {timestamp}

## Validator Results

### Guidance Gate
- **Status**: {'PASS' if validator_results['guidance_gate']['passed'] else 'FAIL'}
- **Output**: {validator_results['guidance_gate']['stdout']}
- **Return Code**: {validator_results['guidance_gate']['return_code']}

### ScaleA Target (0.005 pps)
- **Status**: {'PASS' if validator_results['scaleA']['passed'] else 'FAIL'}
- **Output**: {validator_results['scaleA']['stdout']}
- **Return Code**: {validator_results['scaleA']['return_code']}

## Metrics Snapshot

- **proofs.success**: {metrics['proofs_success']}
- **blocks.height**: {metrics['blocks_height']}
- **latest.merkle**: {metrics['latest_merkle']}
- **redis len**: {metrics['redis_len']}

## Policy Information

- **Policy Hash**: {policy_hash}

## Data Summary

### Baseline Runs
- **Count**: {csv_summary.get('baseline', {}).get('count', 'N/A')}
- **Mean PPH**: {format_mean(csv_summary, 'baseline')}
- **Range**: {csv_summary.get('baseline', {}).get('min', 'N/A')} - {csv_summary.get('baseline', {}).get('max', 'N/A')}

### Guided Runs
- **Count**: {csv_summary.get('guided', {}).get('count', 'N/A')}
- **Mean PPH**: {format_mean(csv_summary, 'guided')}
- **Range**: {csv_summary.get('guided', {}).get('min', 'N/A')} - {csv_summary.get('guided', {}).get('max', 'N/A')}

## Last Ablation Plot

- **Path**: {plot_path}
- **Exists**: {'Yes' if os.path.exists(plot_path) else 'No'}

## Summary for Strategist

**Guidance Gate**: {'PASS' if validator_results['guidance_gate']['passed'] else 'FAIL'} - {validator_results['guidance_gate']['stdout']}
**ScaleA**: {'PASS' if validator_results['scaleA']['passed'] else 'FAIL'} - {validator_results['scaleA']['stdout']}
**Metrics**: proofs.success={metrics['proofs_success']}, blocks.height={metrics['blocks_height']}, latest.merkle={metrics['latest_merkle']}, redis.len={metrics['redis_len']}
**Policy**: {policy_hash}
**Plot**: {plot_path}
"""

    return content

def main():
    """Main function to generate evidence pack."""
    print("Generating WPV5 Evidence Pack...")

    # Ensure output directory exists
    output_dir = "artifacts/wpv5"
    os.makedirs(output_dir, exist_ok=True)

    # Collect all data
    print("Running validator...")
    validator_results = get_validator_results()

    print("Collecting metrics...")
    metrics = get_metrics_snapshot()

    print("Getting policy hash...")
    policy_hash = get_policy_hash()

    print("Getting plot path...")
    plot_path = get_last_ablation_plot_path()

    print("Analyzing CSV data...")
    csv_summary = get_csv_summary()

    # Generate markdown
    print("Creating evidence markdown...")
    evidence_content = create_evidence_markdown(
        validator_results, metrics, policy_hash, plot_path, csv_summary
    )

    # Write to file
    output_path = os.path.join(output_dir, "EVIDENCE.md")
    with open(output_path, 'w') as f:
        f.write(evidence_content)

    print(f"Evidence pack created: {output_path}")

    # Print summary for quick reference
    print("\n" + "="*50)
    print("QUICK SUMMARY FOR STRATEGIST:")
    print("="*50)
    print(f"Guidance Gate: {'PASS' if validator_results['guidance_gate']['passed'] else 'FAIL'}")
    print(f"ScaleA: {'PASS' if validator_results['scaleA']['passed'] else 'FAIL'}")
    print(f"Metrics: proofs.success={metrics['proofs_success']}, blocks.height={metrics['blocks_height']}")
    print(f"Policy: {policy_hash}")
    print(f"Plot: {plot_path}")
    print("="*50)

if __name__ == "__main__":
    main()
