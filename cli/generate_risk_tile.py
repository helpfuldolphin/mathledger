"""CLI tool to generate a Risk Assessment Tile and log it."""
import json
import argparse
import logging
from datetime import datetime
from backend.risk.risk_tile import normalize_metric, compute_overall_risk, map_to_band

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_file(file_path: str):
    """Loads a JSON file and returns its content."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error: The file '{file_path}' is not a valid JSON file.")
        exit(1)

def main():
    """Main function to generate and log the risk tile."""
    parser = argparse.ArgumentParser(description="Generate a Risk Assessment Tile from stability and P4 reports.")
    parser.add_argument(
        "--stability-report",
        default="stability_report.json",
        help="Path to the stability report JSON file."
    )
    parser.add_argument(
        "--p4-summary",
        default="p4_summary.json",
        help="Path to the P4 summary JSON file."
    )
    args = parser.parse_args()

    # Load data
    stability_data = load_json_file(args.stability_report)
    p4_data = load_json_file(args.p4_summary)

    # Metric definitions (thresholds, operators, etc.)
    metric_definitions = {
        'delta_p': {'name': 'Performance Delta', 'threshold': 0.05, 'operator': '<', 'gate': 'P3'},
        'rsi': {'name': 'Robustness Stress Index', 'threshold': 0.80, 'operator': '>', 'gate': 'P3'},
        'omega': {'name': 'Generalization Score (Omega)', 'threshold': 0.95, 'operator': '>', 'gate': 'P3'},
        'tda': {'name': 'Domain Awareness', 'threshold': 0.99, 'operator': '>', 'gate': 'P3'},
        'divergence': {'name': 'Mission Divergence', 'threshold': 0.02, 'operator': '<', 'gate': 'P4'},
    }
    
    all_metrics = {**stability_data, **p4_data}
    normalized_scores = {}
    risk_tiles = []

    for metric_id, value in all_metrics.items():
        if metric_id in metric_definitions:
            defn = metric_definitions[metric_id]
            normalized_scores[metric_id] = normalize_metric(value, defn['threshold'], defn['operator'])
    
    overall_risk = compute_overall_risk(normalized_scores)
    risk_band = map_to_band(overall_risk)
    
    # In a real system, you would generate a tile for each metric.
    # For this exercise, we'll create a single summary tile.
    summary_tile = {
        "metric_id": "overall_risk_summary",
        "metric_name": "Overall Risk Summary",
        "value": overall_risk,
        "threshold": 0.4, # Example threshold for 'HIGH' risk
        "operator": "<",
        "gate": "P4",
        "risk_band": risk_band,
        "timestamp": datetime.utcnow().isoformat(),
        "justification_ref": "docs/risk/FORMAL_SPEC_RISK_STRATEGY.md"
    }

    # Log the output
    logging.info("Generated Risk Tile (Shadow Mode):")
    logging.info(json.dumps(summary_tile, indent=2))


if __name__ == "__main__":
    main()
