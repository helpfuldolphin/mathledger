#!/usr/bin/env python3
"""
Replay Verification CLI

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Date: 2025-12-06

Purpose:
    Command-line tool for replay verification of ledger blocks.
    
Usage:
    # Replay single block
    python scripts/replay_verify.py --block-id 123
    
    # Replay block range
    python scripts/replay_verify.py --block-range 100-200
    
    # Replay all blocks
    python scripts/replay_verify.py --all
    
    # Replay recent N blocks (sliding window)
    python scripts/replay_verify.py --sliding-window 1000
    
    # Replay specific system
    python scripts/replay_verify.py --system-id <uuid> --all
    
    # Strict mode (fail on any warning)
    python scripts/replay_verify.py --all --strict
    
    # Output results to JSON
    python scripts/replay_verify.py --all --output results.json
    
    # Performance report
    python scripts/replay_verify.py --all --performance-report perf.json

Exit Codes:
    0: All blocks passed replay verification
    1: One or more blocks failed replay verification
    2: Command-line argument error
    3: Database connection error
"""

import argparse
import sys
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, '/home/ubuntu/mathledger')

from backend.ledger.replay import (
    replay_block,
    replay_chain,
    replay_block_range,
)


class ReplayVerifyCLI:
    """Command-line interface for replay verification."""
    
    def __init__(self, db_connection, strict: bool = False):
        self.db = db_connection
        self.strict = strict
        self.results = {
            "total_blocks": 0,
            "valid_blocks": 0,
            "invalid_blocks": 0,
            "success_rate": 0.0,
            "failures": [],
            "performance": {
                "start_time": None,
                "end_time": None,
                "total_time_s": 0.0,
                "avg_replay_time_ms": 0.0,
            },
        }
    
    def fetch_block(self, block_id: int) -> Optional[Dict[str, Any]]:
        """Fetch single block by ID."""
        cursor = self.db.execute("""
            SELECT 
                id, block_number, system_id,
                reasoning_attestation_root,
                ui_attestation_root,
                composite_attestation_root,
                canonical_proofs,
                canonical_statements,
                attestation_metadata
            FROM blocks
            WHERE id = %s
        """, (block_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    
    def fetch_blocks_range(self, start: int, end: int) -> List[Dict[str, Any]]:
        """Fetch blocks by block_number range."""
        cursor = self.db.execute("""
            SELECT 
                id, block_number, system_id,
                reasoning_attestation_root,
                ui_attestation_root,
                composite_attestation_root,
                canonical_proofs,
                canonical_statements,
                attestation_metadata
            FROM blocks
            WHERE block_number >= %s AND block_number < %s
            ORDER BY block_number
        """, (start, end))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def fetch_all_blocks(self, system_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all blocks, optionally filtered by system_id."""
        if system_id:
            cursor = self.db.execute("""
                SELECT 
                    id, block_number, system_id,
                    reasoning_attestation_root,
                    ui_attestation_root,
                    composite_attestation_root,
                    canonical_proofs,
                    canonical_statements,
                    attestation_metadata
                FROM blocks
                WHERE system_id = %s
                ORDER BY block_number
            """, (system_id,))
        else:
            cursor = self.db.execute("""
                SELECT 
                    id, block_number, system_id,
                    reasoning_attestation_root,
                    ui_attestation_root,
                    composite_attestation_root,
                    canonical_proofs,
                    canonical_statements,
                    attestation_metadata
                FROM blocks
                ORDER BY block_number
            """)
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def fetch_sliding_window(self, window_size: int, system_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch most recent N blocks."""
        if system_id:
            cursor = self.db.execute("""
                SELECT 
                    id, block_number, system_id,
                    reasoning_attestation_root,
                    ui_attestation_root,
                    composite_attestation_root,
                    canonical_proofs,
                    canonical_statements,
                    attestation_metadata
                FROM blocks
                WHERE system_id = %s
                ORDER BY block_number DESC
                LIMIT %s
            """, (system_id, window_size))
        else:
            cursor = self.db.execute("""
                SELECT 
                    id, block_number, system_id,
                    reasoning_attestation_root,
                    ui_attestation_root,
                    composite_attestation_root,
                    canonical_proofs,
                    canonical_statements,
                    attestation_metadata
                FROM blocks
                ORDER BY block_number DESC
                LIMIT %s
            """, (window_size,))
        
        columns = [desc[0] for desc in cursor.description]
        blocks = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Reverse to get chronological order
        return list(reversed(blocks))
    
    def verify_single_block(self, block_id: int) -> bool:
        """Verify single block."""
        print(f"Fetching block {block_id}...")
        block = self.fetch_block(block_id)
        
        if not block:
            print(f"ERROR: Block {block_id} not found")
            return False
        
        print(f"Replaying block {block['block_number']}...")
        
        start_time = time.time()
        result = replay_block(block)
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.results["total_blocks"] = 1
        
        if result.is_valid:
            self.results["valid_blocks"] = 1
            print(f"✓ Block {block['block_number']} VALID ({elapsed_ms:.2f}ms)")
            return True
        else:
            self.results["invalid_blocks"] = 1
            self.results["failures"].append(result.to_dict())
            print(f"✗ Block {block['block_number']} INVALID: {result.error}")
            return False
    
    def verify_blocks(self, blocks: List[Dict[str, Any]]) -> bool:
        """Verify multiple blocks."""
        if not blocks:
            print("No blocks to verify")
            return True
        
        print(f"Replaying {len(blocks)} blocks...")
        
        start_time = time.time()
        chain_result = replay_chain(blocks, stop_on_failure=not self.strict)
        elapsed_s = time.time() - start_time
        
        self.results["total_blocks"] = chain_result["total_blocks"]
        self.results["valid_blocks"] = chain_result["valid_blocks"]
        self.results["invalid_blocks"] = chain_result["invalid_blocks"]
        self.results["success_rate"] = chain_result["success_rate"]
        self.results["failures"] = chain_result.get("failures", [])
        
        self.results["performance"]["total_time_s"] = elapsed_s
        self.results["performance"]["avg_replay_time_ms"] = (elapsed_s / len(blocks)) * 1000
        
        print(f"\nReplay Summary:")
        print(f"  Total blocks: {self.results['total_blocks']}")
        print(f"  Valid blocks: {self.results['valid_blocks']}")
        print(f"  Invalid blocks: {self.results['invalid_blocks']}")
        print(f"  Success rate: {self.results['success_rate']:.2%}")
        print(f"  Total time: {elapsed_s:.2f}s")
        print(f"  Avg time per block: {self.results['performance']['avg_replay_time_ms']:.2f}ms")
        
        if self.results["invalid_blocks"] > 0:
            print(f"\nFailures:")
            for failure in self.results["failures"]:
                print(f"  - Block {failure['block_number']}: {failure['error']}")
        
        return self.results["success_rate"] == 1.0
    
    def run(
        self,
        block_id: Optional[int] = None,
        block_range: Optional[str] = None,
        all_blocks: bool = False,
        sliding_window: Optional[int] = None,
        system_id: Optional[str] = None,
        output: Optional[str] = None,
        performance_report: Optional[str] = None,
    ) -> bool:
        """Run replay verification."""
        self.results["performance"]["start_time"] = datetime.utcnow().isoformat()
        
        try:
            # Determine which blocks to verify
            if block_id is not None:
                success = self.verify_single_block(block_id)
            
            elif block_range is not None:
                # Parse range (e.g., "100-200")
                try:
                    start, end = map(int, block_range.split('-'))
                    blocks = self.fetch_blocks_range(start, end)
                    success = self.verify_blocks(blocks)
                except ValueError:
                    print(f"ERROR: Invalid block range format: {block_range}")
                    print("Expected format: START-END (e.g., 100-200)")
                    return False
            
            elif all_blocks:
                blocks = self.fetch_all_blocks(system_id=system_id)
                success = self.verify_blocks(blocks)
            
            elif sliding_window is not None:
                blocks = self.fetch_sliding_window(sliding_window, system_id=system_id)
                success = self.verify_blocks(blocks)
            
            else:
                print("ERROR: Must specify one of: --block-id, --block-range, --all, --sliding-window")
                return False
            
            self.results["performance"]["end_time"] = datetime.utcnow().isoformat()
            
            # Write output files
            if output:
                with open(output, 'w') as f:
                    json.dump(self.results, f, indent=2)
                print(f"\nResults written to: {output}")
            
            if performance_report:
                with open(performance_report, 'w') as f:
                    json.dump(self.results["performance"], f, indent=2)
                print(f"Performance report written to: {performance_report}")
            
            return success
        
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Replay verification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Block selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--block-id", type=int, help="Verify single block by ID")
    group.add_argument("--block-range", help="Verify block range (e.g., 100-200)")
    group.add_argument("--all", action="store_true", dest="all_blocks", help="Verify all blocks")
    group.add_argument("--sliding-window", type=int, help="Verify recent N blocks")
    
    # Filters
    parser.add_argument("--system-id", help="Filter by system ID (UUID)")
    
    # Options
    parser.add_argument("--strict", action="store_true", help="Fail on any warning")
    parser.add_argument("--output", help="Output results to JSON file")
    parser.add_argument("--performance-report", help="Output performance report to JSON file")
    
    # Database
    parser.add_argument("--db-url", default="postgresql://localhost/mathledger", help="Database URL")
    
    args = parser.parse_args()
    
    # Connect to database
    print(f"Connecting to database: {args.db_url}")
    # db = connect_to_database(args.db_url)
    db = None  # Placeholder
    
    if db is None:
        print("ERROR: Database connection not implemented")
        print("This script requires a database connection to run.")
        sys.exit(3)
    
    # Run verification
    cli = ReplayVerifyCLI(db, strict=args.strict)
    success = cli.run(
        block_id=args.block_id,
        block_range=args.block_range,
        all_blocks=args.all_blocks,
        sliding_window=args.sliding_window,
        system_id=args.system_id,
        output=args.output,
        performance_report=args.performance_report,
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
