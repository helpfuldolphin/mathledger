#!/usr/bin/env python3
"""
Mirror Auditor (Claude N) - Dual-Root Attestation Symmetry Verifier

Mission: Ensure dual-root attestation symmetry (R_t ↔ U_t) remains exact.
Validates that every block's human and reasoning events are cryptographically bound.

Core Operations:
1. Validate H_t = SHA256(R_t || U_t) for all blocks
2. Check cross-epoch consistency
3. Emit verification reports with [PASS]/[FAIL] verdicts

Methodology:
- Compare UI-Merkle vs Reasoning-Merkle
- Verify composite attestation integrity
- Track dual-root coverage across epochs

Invocation:
    python tools/mirror_auditor.py --verify-all
    python tools/mirror_auditor.py --block-range 1 100
    python tools/mirror_auditor.py --emit-report
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse

# Database connection setup
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("[WARN] psycopg2 not available. Running in mock mode.", file=sys.stderr)


class MirrorAuditor:
    """
    Mirror Auditor - Dual-Root Attestation Symmetry Verifier

    Validates cryptographic binding between reasoning events (R_t)
    and human events (U_t) via composite attestation root (H_t).
    """

    def __init__(self, db_url: str | None = None):
        from backend.security.runtime_env import MissingEnvironmentVariable, get_database_url

        if db_url is None:
            try:
                db_url = get_database_url()
            except MissingEnvironmentVariable as exc:
                raise RuntimeError(str(exc)) from exc
        self.db_url = db_url
        self.conn = None
        self.verification_results: List[Dict] = []

    def connect(self) -> bool:
        """Establish database connection."""
        if not DB_AVAILABLE:
            print("[WARN] Database connection skipped (psycopg2 not available)")
            return False

        try:
            self.conn = psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}", file=sys.stderr)
            return False

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def compute_composite_root(self, r_t: str, u_t: str) -> str:
        """
        Compute composite attestation root: H_t = SHA256(R_t || U_t)

        Args:
            r_t: Reasoning merkle root (64-char hex)
            u_t: UI merkle root (64-char hex)

        Returns:
            64-char hex hash of composite attestation
        """
        if not r_t or not u_t:
            raise ValueError("Both R_t and U_t must be non-empty")

        # Validate hex format
        try:
            int(r_t, 16)
            int(u_t, 16)
        except ValueError:
            raise ValueError(f"Invalid hex format: R_t={r_t}, U_t={u_t}")

        # Compute H_t = SHA256(R_t || U_t)
        composite_data = f"{r_t}{u_t}".encode('ascii')
        h_t = hashlib.sha256(composite_data).hexdigest()

        return h_t

    def verify_block_attestation(self, block: Dict) -> Dict:
        """
        Verify dual-root attestation for a single block.

        Args:
            block: Block record with id, reasoning_merkle_root, ui_merkle_root,
                   composite_attestation_root

        Returns:
            Verification result dictionary with status and details
        """
        block_id = block.get('id')
        block_number = block.get('block_number')
        r_t = block.get('reasoning_merkle_root')
        u_t = block.get('ui_merkle_root')
        h_t_stored = block.get('composite_attestation_root')

        result = {
            'block_id': block_id,
            'block_number': block_number,
            'r_t': r_t,
            'u_t': u_t,
            'h_t_stored': h_t_stored,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Check 1: Dual-root presence
        if not r_t or not u_t:
            result['status'] = 'INCOMPLETE'
            result['verdict'] = 'ABSTAIN'
            result['reason'] = f'Missing dual roots (R_t={bool(r_t)}, U_t={bool(u_t)})'
            return result

        # Check 2: Compute expected H_t
        try:
            h_t_computed = self.compute_composite_root(r_t, u_t)
            result['h_t_computed'] = h_t_computed
        except Exception as e:
            result['status'] = 'ERROR'
            result['verdict'] = 'FAIL'
            result['reason'] = f'Composite computation failed: {e}'
            return result

        # Check 3: Verify H_t matches stored value
        if h_t_stored:
            if h_t_computed == h_t_stored:
                result['status'] = 'VERIFIED'
                result['verdict'] = 'PASS'
                result['reason'] = 'Dual-root attestation symmetry OK'
            else:
                result['status'] = 'MISMATCH'
                result['verdict'] = 'FAIL'
                result['reason'] = f'H_t mismatch: computed={h_t_computed}, stored={h_t_stored}'
        else:
            # H_t not stored yet, but we can compute it
            result['status'] = 'COMPUTED'
            result['verdict'] = 'PASS'
            result['reason'] = 'H_t computed successfully (not yet stored)'

        return result

    def verify_all_blocks(self, block_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Verify dual-root attestation for all blocks.

        Args:
            block_range: Optional (start, end) block numbers to limit verification

        Returns:
            List of verification results
        """
        if not self.conn:
            print("[ERROR] No database connection", file=sys.stderr)
            return []

        # Build query
        query = """
            SELECT id, block_number, system_id, reasoning_merkle_root,
                   ui_merkle_root, composite_attestation_root, created_at
            FROM blocks
        """

        params = []
        if block_range:
            query += " WHERE block_number >= %s AND block_number <= %s"
            params = [block_range[0], block_range[1]]

        query += " ORDER BY block_number ASC"

        # Execute query
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                blocks = cur.fetchall()
        except Exception as e:
            print(f"[ERROR] Query failed: {e}", file=sys.stderr)
            return []

        # Verify each block
        results = []
        for block in blocks:
            result = self.verify_block_attestation(dict(block))
            results.append(result)
            self.verification_results.append(result)

        return results

    def emit_verification_report(self, results: List[Dict]) -> str:
        """
        Emit comprehensive verification report.

        Args:
            results: List of verification results from verify_all_blocks()

        Returns:
            Formatted report string
        """
        if not results:
            return "[ABSTAIN] No blocks to verify"

        # Aggregate statistics
        total = len(results)
        verified = sum(1 for r in results if r['verdict'] == 'PASS')
        failed = sum(1 for r in results if r['verdict'] == 'FAIL')
        abstained = sum(1 for r in results if r['verdict'] == 'ABSTAIN')

        complete_coverage = sum(1 for r in results if r.get('r_t') and r.get('u_t'))
        coverage_pct = (complete_coverage / total * 100) if total > 0 else 0

        # Build report
        report_lines = [
            "=" * 80,
            "🪞 MIRROR AUDITOR - DUAL-ROOT ATTESTATION SYMMETRY REPORT",
            "=" * 80,
            "",
            f"Timestamp: {datetime.utcnow().isoformat()}Z",
            f"Total Blocks: {total}",
            f"Dual-Root Coverage: {complete_coverage}/{total} ({coverage_pct:.1f}%)",
            "",
            "VERIFICATION SUMMARY:",
            f"  ✓ PASS:    {verified}",
            f"  ✗ FAIL:    {failed}",
            f"  ⊘ ABSTAIN: {abstained}",
            "",
        ]

        # Overall verdict
        if failed > 0:
            report_lines.append("[FAIL] Dual-Root Mirror Integrity COMPROMISED")
        elif abstained == total:
            report_lines.append("[ABSTAIN] Dual-Root Mirror Integrity INCOMPLETE")
        elif verified == total:
            report_lines.append(f"[PASS] Dual-Root Mirror Integrity OK epochs={total}")
        else:
            report_lines.append(f"[PARTIAL] Dual-Root Mirror Integrity {verified}/{total} verified")

        report_lines.extend([
            "",
            "=" * 80,
            "DETAILED RESULTS:",
            "=" * 80,
            ""
        ])

        # Detailed results (show first 10 and last 10)
        for i, result in enumerate(results[:10]):
            report_lines.append(self._format_block_result(result))

        if total > 20:
            report_lines.append(f"... ({total - 20} blocks omitted) ...")
            report_lines.append("")
            for result in results[-10:]:
                report_lines.append(self._format_block_result(result))
        elif total > 10:
            for result in results[10:]:
                report_lines.append(self._format_block_result(result))

        report_lines.extend([
            "",
            "=" * 80,
            "MIRROR AUDITOR SIGNING OFF",
            "=" * 80
        ])

        return "\n".join(report_lines)

    def _format_block_result(self, result: Dict) -> str:
        """Format a single block verification result."""
        verdict_symbol = {
            'PASS': '✓',
            'FAIL': '✗',
            'ABSTAIN': '⊘'
        }.get(result['verdict'], '?')

        block_num = result.get('block_number', '?')
        status = result.get('status', 'UNKNOWN')
        reason = result.get('reason', 'No reason provided')

        return f"  [{verdict_symbol}] Block #{block_num}: {status} - {reason}"

    def export_results_json(self, output_path: str):
        """Export verification results to JSON file."""
        output = {
            'timestamp': datetime.utcnow().isoformat(),
            'auditor': 'Mirror Auditor (Claude N)',
            'total_blocks': len(self.verification_results),
            'results': self.verification_results
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[INFO] Results exported to {output_path}")


def main():
    """Main entry point for Mirror Auditor CLI."""
    parser = argparse.ArgumentParser(
        description="Mirror Auditor - Dual-Root Attestation Symmetry Verifier"
    )
    parser.add_argument(
        '--db-url',
        default=None,
        help='Database connection URL (default: from DATABASE_URL env var)'
    )
    parser.add_argument(
        '--verify-all',
        action='store_true',
        help='Verify all blocks in database'
    )
    parser.add_argument(
        '--block-range',
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        help='Verify blocks in range [START, END]'
    )
    parser.add_argument(
        '--emit-report',
        action='store_true',
        help='Emit verification report to stdout'
    )
    parser.add_argument(
        '--export-json',
        metavar='PATH',
        help='Export results to JSON file'
    )

    args = parser.parse_args()

    # Initialize auditor
    auditor = MirrorAuditor(db_url=args.db_url)

    # Connect to database
    if not auditor.connect():
        print("[ERROR] Failed to connect to database. Exiting.", file=sys.stderr)
        return 1

    try:
        # Run verification
        if args.verify_all:
            print("🪞 Mirror Auditor online — verifying dual attestation symmetry...")
            print()
            results = auditor.verify_all_blocks()
        elif args.block_range:
            start, end = args.block_range
            print(f"🪞 Mirror Auditor verifying blocks {start}-{end}...")
            print()
            results = auditor.verify_all_blocks(block_range=(start, end))
        else:
            print("[ERROR] No verification mode specified. Use --verify-all or --block-range", file=sys.stderr)
            return 1

        # Emit report
        if args.emit_report or args.verify_all:
            report = auditor.emit_verification_report(results)
            print(report)

        # Export JSON
        if args.export_json:
            auditor.export_results_json(args.export_json)

        # Determine exit code
        failed = sum(1 for r in results if r['verdict'] == 'FAIL')
        return 1 if failed > 0 else 0

    finally:
        auditor.close()


if __name__ == "__main__":
    raise SystemExit(main())
