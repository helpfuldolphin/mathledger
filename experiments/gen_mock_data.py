
import json
import random
import os

def generate_mock_log(filename, mode, cycles=300):
    print(f"Generating {filename} for {mode}...")
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(cycles):
            # Baseline: Constant abstention ~20%
            # RFL: High (~20%) then drops to ~5% after cycle 100
            p_abstain = 0.20
            if mode == 'rfl' and i > 100:
                p_abstain = 0.05
            
            entry = {
                "cycle": i,
                "mode": mode,
            }
            
            if random.random() < p_abstain:
                # Simulate abstention
                entry["status"] = "abstain"
                entry["method"] = "lean-disabled" # As per contract
            else:
                entry["status"] = "verified"
                entry["method"] = "auto"
            
            f.write(json.dumps(entry) + "\n")

def main():
    os.makedirs("results", exist_ok=True)
    generate_mock_log("results/fo_baseline.jsonl", "baseline")
    generate_mock_log("results/fo_rfl.jsonl", "rfl")

if __name__ == "__main__":
    main()

