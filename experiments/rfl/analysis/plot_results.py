import json
import sys
import argparse
from pathlib import Path
import statistics

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_markdown_summary(results, output_path):
    experiment_id = results.get("experiment_id", "Unknown")
    runs = results.get("runs", [])
    policy_ledger = results.get("policy", {}).get("ledger", [])

    lines = []
    lines.append(f"# Experiment Summary: {experiment_id}")
    lines.append(f"**Total Runs:** {len(runs)}")
    
    successful_runs = [r for r in runs if r["status"] == "success"]
    success_rate = len(successful_runs) / len(runs) if runs else 0
    lines.append(f"**Success Rate:** {success_rate:.2%}")

    # Table of Runs
    lines.append("\n## Run Metrics")
    lines.append("| Run | Slice | Steps | Breadth | Total | Proofs | Abstain % | Throughput | Descent |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for i, run in enumerate(runs):
        run_id = run.get("run_id")
        # Try to find corresponding ledger entry
        ledger_entry = next((l for l in policy_ledger if l["run_id"] == run_id), None)
        
        descent = f"{ledger_entry['symbolic_descent']:.4f}" if ledger_entry else "N/A"
        slice_name = ledger_entry['slice_name'] if ledger_entry else "N/A"
        
        proofs = run.get("successful_proofs", 0)
        total_stmts = run.get("total_statements", 0)
        abstain_pct = (run.get("abstentions", 0) / total_stmts) * 100 if total_stmts > 0 else 0.0
        throughput = run.get("throughput_proofs_per_hour", 0.0)

        lines.append(f"| {i+1} | {slice_name} | {run.get('derive_steps')} | {run.get('max_breadth')} | {run.get('max_total')} | {proofs} | {abstain_pct:.2f}% | {throughput:.2f} | {descent} |")

    # Aggregate Stats
    if successful_runs:
        throughputs = [r.get("throughput_proofs_per_hour", 0) for r in successful_runs]
        avg_throughput = statistics.mean(throughputs)
        lines.append(f"\n**Average Throughput:** {avg_throughput:.2f} proofs/hr")
    
    if policy_ledger:
        descents = [l["symbolic_descent"] for l in policy_ledger]
        avg_descent = statistics.mean(descents)
        lines.append(f"**Average Symbolic Descent:** {avg_descent:.4f}")

    summary_text = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(summary_text)
        print(f"Summary written to {output_path}")
    else:
        print(summary_text)

def main():
    parser = argparse.ArgumentParser(description="RFL Results Analyzer")
    parser.add_argument("results_file", type=str, help="Path to rfl_results.json")
    parser.add_argument("--output", type=str, help="Path to output summary markdown")
    args = parser.parse_args()

    try:
        results = load_results(args.results_file)
        generate_markdown_summary(results, args.output)
    except Exception as e:
        print(f"Error analyzing results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
