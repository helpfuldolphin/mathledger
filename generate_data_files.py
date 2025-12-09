import json
from analysis.conjecture_engine import generate_mock_data

def create_files():
    baseline_records, rfl_records = generate_mock_data("positive_logistic")
    
    with open('artifacts/dynamics/baseline.jsonl', 'w') as f:
        for record in baseline_records:
            f.write(json.dumps(record) + '\n')
            
    with open('artifacts/dynamics/rfl.jsonl', 'w') as f:
        for record in rfl_records:
            f.write(json.dumps(record) + '\n')

if __name__ == '__main__':
    create_files()
