#!/usr/bin/env python3
"""
Performance Passport Upload Script for CI/CD Integration

This script uploads the performance passport to a designated location
for Manus A's Trust Demo and performance monitoring.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path


def upload_performance_passport(passport_file: str, upload_destination: str = None):
    """
    Upload performance passport to designated location.

    Args:
        passport_file: Path to the performance passport JSON file
        upload_destination: Destination for upload (default: artifacts/performance/)
    """
    if not os.path.exists(passport_file):
        print(f"[x] Performance passport not found: {passport_file}")
        return False

    # Default upload destination
    if upload_destination is None:
        upload_destination = "artifacts/performance/"

    # Ensure upload directory exists
    os.makedirs(upload_destination, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_passport_{timestamp}.json"
    destination_path = os.path.join(upload_destination, filename)

    try:
        # Copy passport to destination
        import shutil
        shutil.copy2(passport_file, destination_path)

        # Also create a latest symlink/copy
        latest_path = os.path.join(upload_destination, "latest_performance_passport.json")
        shutil.copy2(passport_file, latest_path)

        print(f"Performance passport uploaded successfully!")
        print(f"Destination: {destination_path}")
        print(f"Latest: {latest_path}")

        # Print summary for CI/CD logs
        with open(passport_file, 'r') as f:
            passport_data = json.load(f)

        summary = passport_data.get('summary', {})
        print(f"Summary: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)} tests passed")
        print(f"Max Latency: {summary.get('max_latency_ms', 0):.3f}ms")
        print(f"Max Memory: {summary.get('max_memory_mb', 0):.2f}MB")
        print(f"Status: {summary.get('overall_status', 'UNKNOWN')}")

        return True

    except Exception as e:
        print(f"[x] Failed to upload performance passport: {e}")
        return False


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Upload Performance Passport")
    parser.add_argument("passport_file", help="Path to performance passport JSON file")
    parser.add_argument("--destination", "-d", help="Upload destination directory")
    parser.add_argument("--ci", action="store_true", help="CI/CD mode with enhanced logging")

    args = parser.parse_args()

    if args.ci:
        print("CI/CD Mode: Uploading Performance Passport")
        print(f"Passport: {args.passport_file}")
        print(f"Destination: {args.destination or 'artifacts/performance/'}")

    success = upload_performance_passport(args.passport_file, args.destination)

    if success:
        print("Performance Passport upload completed successfully!")
        if args.ci:
            print("CI/CD integration successful - Manus A can access the passport!")
        sys.exit(0)
    else:
        print("Performance Passport upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
