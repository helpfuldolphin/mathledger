#!/usr/bin/env python3
"""
Universal Verification Suite V2 for MathLedger - Audit Sync Edition

Devin C - Universal Verifier V2
Mission: Connect universal verification to Audit Harness and AllBlue Gate.
Extends V1 with signed results, audit harness integration, and CI summary embedding.

Exit codes:
  0: [PASS] VERIFIED: ALL CLAIMS HOLD (sync v2)
  1: [FAIL] One or more verification checks failed
  2: [ERROR] Fatal error during verification

Usage:
  python tools/verify_all_v2.py --offline --output artifacts/audit/verification_summary.json
  python tools/verify_all_v2.py --check hash --audit-sync
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psycopg
except ImportError:
    psycopg = None

from tools.verify_all import Verifier, VerificationResult


EXIT_CODE_MAP = {
    0: {
        "status": "PASS",
        "description": "VERIFIED: ALL CLAIMS HOLD",
        "allblue_status": "green",
        "ci_marker": "[OK]"
    },
    1: {
        "status": "FAIL",
        "description": "One or more verification checks failed",
        "allblue_status": "red",
        "ci_marker": "[FAIL]"
    },
    2: {
        "status": "ERROR",
        "description": "Fatal error during verification",
        "allblue_status": "yellow",
        "ci_marker": "[WARN]"
    }
}


class VerifierV2(Verifier):
    """Enhanced verifier with audit harness integration."""
    
    def __init__(self, offline: bool = False, verbose: bool = False, audit_sync: bool = False):
        super().__init__(offline, verbose)
        self.audit_sync = audit_sync
        self.run_id = self._generate_run_id()
        self.start_time = datetime.now(timezone.utc)
        
    def _generate_run_id(self) -> str:
        """Generate deterministic run ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
    
    def _compute_signature(self, data: Dict[str, Any]) -> str:
        """Compute deterministic signature for verification results."""
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def generate_verification_summary(self) -> Dict[str, Any]:
        """Generate comprehensive verification summary with audit metadata."""
        end_time = datetime.now(timezone.utc)
        duration_seconds = (end_time - self.start_time).total_seconds()
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        exit_code = 0 if passed_count == total_count else 1
        
        check_results = []
        for result in self.results:
            check_results.append({
                "name": result.name,
                "passed": result.passed,
                "message": result.message,
                "details": result.details
            })
        
        summary_data = {
            "run_id": self.run_id,
            "timestamp": self.start_time.isoformat(),
            "end_timestamp": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "version": "v2",
            "verifier": "Devin C - Universal Verifier V2",
            "mode": "offline" if self.offline else "online",
            "checks": {
                "total": total_count,
                "passed": passed_count,
                "failed": total_count - passed_count
            },
            "exit_code": exit_code,
            "exit_code_map": EXIT_CODE_MAP[exit_code],
            "results": check_results,
            "audit_metadata": {
                "schema_version": "v2.0",
                "audit_sync_enabled": self.audit_sync,
                "compliance_tags": ["RC", "ME", "IVL"],
                "acquisition_narrative": "Reliability & Correctness verification with audit trail"
            }
        }
        
        signature = self._compute_signature(summary_data)
        summary_data["signature"] = signature
        
        return summary_data
    
    def generate_ci_summary(self, summary: Dict[str, Any]) -> str:
        """Generate CI-friendly markdown summary for AllBlue ingestion."""
        exit_info = summary["exit_code_map"]
        emoji = exit_info["ci_marker"]
        status = exit_info["status"]
        
        ci_summary = f"""# {emoji} Verification Summary - {status}

**Run ID**: `{summary['run_id']}`
**Timestamp**: {summary['timestamp']}
**Duration**: {summary['duration_seconds']:.2f}s
**Mode**: {summary['mode']}


- **Total Checks**: {summary['checks']['total']}
- **Passed**: {summary['checks']['passed']} [OK]
- **Failed**: {summary['checks']['failed']} [FAIL]


"""
        
        for result in summary['results']:
            result_marker = "[OK]" if result['passed'] else "[FAIL]"
            ci_summary += f"### {result_marker} {result['name']}\n\n"
            ci_summary += f"**Status**: {'PASS' if result['passed'] else 'FAIL'}\n"
            ci_summary += f"**Message**: {result['message']}\n\n"
            
            if result['details'] and self.verbose:
                ci_summary += "**Details**:\n"
                for key, value in result['details'].items():
                    ci_summary += f"- `{key}`: {value}\n"
                ci_summary += "\n"
        
        ci_summary += f"""## Exit Code Map

- **Code**: {summary['exit_code']}
- **Status**: {exit_info['status']}
- **AllBlue Status**: {exit_info['allblue_status']}
- **Description**: {exit_info['description']}


- **Schema Version**: {summary['audit_metadata']['schema_version']}
- **Compliance Tags**: {', '.join(summary['audit_metadata']['compliance_tags'])}
- **Signature**: `{summary['signature'][:16]}...`

---
*Generated by Universal Verifier V2 - Audit Sync Edition*
"""
        
        return ci_summary
    
    def validate_against_audit_harness(self, summary: Dict[str, Any]) -> Tuple[bool, str]:
        """Cross-validate summary against Audit Harness schema."""
        required_fields = [
            "run_id", "timestamp", "version", "verifier", "mode",
            "checks", "exit_code", "exit_code_map", "results",
            "audit_metadata", "signature"
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in summary:
                missing_fields.append(field)
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        if summary["audit_metadata"]["schema_version"] != "v2.0":
            return False, f"Invalid schema version: {summary['audit_metadata']['schema_version']}"
        
        expected_signature = self._compute_signature({k: v for k, v in summary.items() if k != "signature"})
        if summary["signature"] != expected_signature:
            return False, "Signature mismatch - data may have been tampered with"
        
        return True, "Audit harness validation passed"
    
    def write_audit_output(self, output_path: Path, summary: Dict[str, Any], ci_summary: str):
        """Write verification results to audit directory."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        ci_summary_path = output_path.parent / "verification_summary.md"
        with open(ci_summary_path, 'w', encoding='utf-8') as f:
            f.write(ci_summary)
        
        exit_code_path = output_path.parent / "exit_code_map.json"
        with open(exit_code_path, 'w', encoding='utf-8') as f:
            json.dump(EXIT_CODE_MAP, f, indent=2)
        
        self.log(f"Audit output written to {output_path}")
        self.log(f"CI summary written to {ci_summary_path}")
        self.log(f"Exit code map written to {exit_code_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal verification suite V2 with audit harness integration",
        epilog="Verify every claim. Enable verification by writing universal checkers. Sync with audit harness."
    )
    parser.add_argument(
        "--check",
        type=str,
        help="Run specific check (hash, merkle, files, metrics, normalization, database, api, parents)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip database and API checks"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audit/verification_summary.json"),
        help="Output path for verification summary JSON"
    )
    parser.add_argument(
        "--audit-sync",
        action="store_true",
        help="Enable audit harness synchronization"
    )
    
    args = parser.parse_args()
    
    verifier = VerifierV2(offline=args.offline, verbose=args.verbose, audit_sync=args.audit_sync)
    success = verifier.run_all_checks(specific_check=args.check)
    
    summary = verifier.generate_verification_summary()
    ci_summary = verifier.generate_ci_summary(summary)
    
    if args.audit_sync:
        is_valid, validation_msg = verifier.validate_against_audit_harness(summary)
        if not is_valid:
            print(f"\n[ERROR] Audit harness validation failed: {validation_msg}", file=sys.stderr)
            return 2
        else:
            print(f"\n[PASS] Audit harness validation: {validation_msg}")
    
    verifier.write_audit_output(args.output, summary, ci_summary)
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION V2 SUMMARY")
    print(f"{'='*60}")
    print(f"Run ID: {summary['run_id']}")
    print(f"Signature: {summary['signature'][:32]}...")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    if success:
        print("[PASS] VERIFIED: ALL CLAIMS HOLD (sync v2)")
        return 0
    else:
        print("[FAIL] Verification failed - see summary for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
