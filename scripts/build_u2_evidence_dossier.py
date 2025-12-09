"""
CLI tool to build and validate a U2 Evidence Dossier for a given run ID.
"""
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.promotion import u2_evidence
from backend.promotion import admissibility_engine

def main():
    parser = argparse.ArgumentParser(description="Build and validate a U2 Evidence Dossier.")
    parser.add_argument("run_id", help="The unique identifier for the U2 run.")
    parser.add_argument("--validate", action="store_true", help="Run the Admissibility Law Engine.")
    parser.add_argument("--report-path", default="evidence_admissibility_report.json", help="Output path for the admissibility report.")
    args = parser.parse_args()

    dossier = u2_evidence.assemble_dossier(args.run_id, root_path=".")
    if dossier.dossier_status == "INCOMPLETE" and args.validate:
        print("\nFATAL: Dossier assembly failed. Cannot proceed to admissibility checks.")
        sys.exit(1)

    if args.validate:
        report = admissibility_engine.check_admissibility(dossier)
        try:
            with open(args.report_path, "w", encoding="utf-8") as f:
                json.dump(report.to_json(), f, indent=2)
            print(f"\nSuccessfully wrote admissibility report to: {args.report_path}")
        except IOError as e:
            print(f"\nError writing report file: {e}", file=sys.stderr)
            sys.exit(1)

        if report.verdict.status != "ADMISSIBLE":
            print(f"\nVERDICT: Dossier is NOT ADMISSIBLE.")
            if report.verdict.reason:
                print(f"  REASON: {report.verdict.reason} (Code: {report.verdict.code})")
            sys.exit(1)
        else:
            print("\nVERDICT: Dossier is ADMISSIBLE.")
    
    print("\nDossier build process complete.")

if __name__ == "__main__":
    main()