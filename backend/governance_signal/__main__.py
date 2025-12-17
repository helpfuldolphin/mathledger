# backend/governance_signal/__main__.py
import json
import sys
import argparse
from .validator import validate_signal

def main():
    """CLI entrypoint for validating a Governance Signal JSON file."""
    parser = argparse.ArgumentParser(
        description="Validate a Governance Signal JSON file against the official schema.",
        prog="python -m backend.governance_signal.validate"
    )
    parser.add_argument(
        "signal_file",
        type=argparse.FileType("r"),
        help="The path to the JSON file containing the Governance Signal.",
    )
    args = parser.parse_args()

    try:
        signal_data = json.load(args.signal_file)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.signal_file.name}. {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        args.signal_file.close()

    is_valid, message = validate_signal(signal_data)

    if is_valid:
        print(f"✅ Validation successful: {args.signal_file.name}")
        print(message)
        sys.exit(0)
    else:
        print(f"❌ Validation failed: {args.signal_file.name}", file=sys.stderr)
        print(message, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
