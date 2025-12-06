import json
import hashlib
import sys
from pathlib import Path

def validate_and_report(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"FAIL: File not found: {file_path}")
        sys.exit(1)

    if path.stat().st_size == 0:
         print(f"FAIL: File is empty: {file_path}")
         sys.exit(1)

    lines = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"FAIL: Could not read file: {e}")
        sys.exit(1)

    line_count = len(lines)
    if line_count != 1000:
        print(f"FAIL: Expected 1000 lines, got {line_count}")
        # We continue to inspect what we have, but exit code will be non-zero eventually if strictly enforced.
        # For now, let's gather metrics.

    first_abstention = None
    last_abstention = None
    
    valid_lines = 0
    
    # Calculate SHA256 of the file content
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            
            # Check Schema
            if "roots" not in data or "h_t" not in data["roots"] or "r_t" not in data["roots"] or "u_t" not in data["roots"]:
                print(f"FAIL: Missing roots on line {i+1}")
                sys.exit(1)
            
            if data.get("mode") != "rfl":
                print(f"FAIL: Incorrect mode '{data.get('mode')}' on line {i+1}")
                sys.exit(1)

            # Get Abstention
            abstention = data.get("abstention")
            if i == 0:
                first_abstention = abstention
            if i == line_count - 1:
                last_abstention = abstention
                
            valid_lines += 1
            
        except json.JSONDecodeError:
            print(f"FAIL: Invalid JSON on line {i+1}")
            sys.exit(1)

    print(f"SUCCESS: Validated {valid_lines} lines.")
    print(f"SIZE: {path.stat().st_size} bytes")
    print(f"LINES: {line_count}")
    print(f"FIRST_ABSTENTION: {first_abstention}")
    print(f"LAST_ABSTENTION: {last_abstention}")
    print(f"SHA256: {file_hash}")

if __name__ == "__main__":
    validate_and_report("results/fo_rfl.jsonl")
