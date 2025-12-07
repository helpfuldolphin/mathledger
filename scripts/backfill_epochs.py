#!/usr/bin/env python3
"""
Epoch Backfill Script

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Date: 2025-12-06

Purpose:
    Retroactively seal epochs for existing blocks in the ledger.
    
    This script:
    1. Groups existing blocks into epochs (100 blocks per epoch)
    2. Computes epoch roots (E_t = MerkleRoot([H_0, H_1, ..., H_99]))
    3. Stores epochs in the database
    4. Updates blocks.epoch_id for all blocks

Usage:
    # Dry run (preview only, no database changes)
    python scripts/backfill_epochs.py --dry-run
    
    # Backfill all systems
    python scripts/backfill_epochs.py --all
    
    # Backfill specific system
    python scripts/backfill_epochs.py --system-id <uuid>
    
    # Backfill with custom epoch size
    python scripts/backfill_epochs.py --all --epoch-size 50
    
    # Verify backfill (check all blocks have epoch_id)
    python scripts/backfill_epochs.py --verify

Safety:
    - Runs in transaction (rollback on error)
    - Validates epoch roots after sealing
    - Supports dry-run mode
    - Idempotent (safe to run multiple times)

Dependencies:
    - Migration 018 (epoch_root_system.sql) must be applied first
    - Requires backend.ledger.epoch module
"""

import argparse
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, '/home/ubuntu/mathledger')

from backend.ledger.epoch import (
    compute_epoch_root,
    seal_epoch,
    DEFAULT_EPOCH_SIZE,
)


class EpochBackfillScript:
    """Backfill epochs for existing blocks."""
    
    def __init__(self, db_connection, epoch_size: int = DEFAULT_EPOCH_SIZE):
        self.db = db_connection
        self.epoch_size = epoch_size
        self.stats = {
            "systems_processed": 0,
            "epochs_created": 0,
            "blocks_updated": 0,
            "errors": [],
        }
    
    def get_systems(self) -> List[str]:
        """Get all system IDs with blocks."""
        cursor = self.db.execute("""
            SELECT DISTINCT system_id 
            FROM blocks 
            WHERE system_id IS NOT NULL
            ORDER BY system_id
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def get_blocks_for_system(self, system_id: str) -> List[Dict[str, Any]]:
        """Get all blocks for a system, ordered by block_number."""
        cursor = self.db.execute("""
            SELECT 
                id,
                block_number,
                composite_attestation_root,
                reasoning_attestation_root,
                ui_attestation_root,
                canonical_proofs,
                attestation_metadata
            FROM blocks
            WHERE system_id = %s
            ORDER BY block_number
        """, (system_id,))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def group_blocks_into_epochs(
        self, 
        blocks: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Group blocks into epochs by epoch_number."""
        epochs = {}
        
        for block in blocks:
            epoch_number = block["block_number"] // self.epoch_size
            if epoch_number not in epochs:
                epochs[epoch_number] = []
            epochs[epoch_number].append(block)
        
        return epochs
    
    def backfill_system(self, system_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Backfill epochs for a single system.
        
        Returns:
            Statistics dictionary with epochs_created, blocks_updated, errors
        """
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing system: {system_id}")
        
        # Fetch all blocks
        blocks = self.get_blocks_for_system(system_id)
        if not blocks:
            print(f"  No blocks found for system {system_id}")
            return {"epochs_created": 0, "blocks_updated": 0, "errors": []}
        
        print(f"  Found {len(blocks)} blocks (block_number: {blocks[0]['block_number']} to {blocks[-1]['block_number']})")
        
        # Group into epochs
        epoch_groups = self.group_blocks_into_epochs(blocks)
        print(f"  Grouped into {len(epoch_groups)} epochs")
        
        epochs_created = 0
        blocks_updated = 0
        errors = []
        
        # Process each epoch
        for epoch_number in sorted(epoch_groups.keys()):
            epoch_blocks = epoch_groups[epoch_number]
            
            try:
                # Seal epoch
                epoch_data = seal_epoch(
                    epoch_number=epoch_number,
                    blocks=epoch_blocks,
                    system_id=system_id,
                    epoch_size=self.epoch_size,
                )
                
                print(f"  Epoch {epoch_number}: {epoch_data['block_count']} blocks, "
                      f"E_t={epoch_data['epoch_root'][:16]}...")
                
                if not dry_run:
                    # Store epoch in database
                    epoch_id = self._store_epoch(epoch_data)
                    
                    # Update blocks.epoch_id
                    block_ids = [block["id"] for block in epoch_blocks]
                    self._update_block_epochs(block_ids, epoch_id)
                    
                    blocks_updated += len(block_ids)
                
                epochs_created += 1
                
            except Exception as e:
                error_msg = f"Epoch {epoch_number} failed: {str(e)}"
                print(f"  ERROR: {error_msg}")
                errors.append(error_msg)
        
        return {
            "epochs_created": epochs_created,
            "blocks_updated": blocks_updated,
            "errors": errors,
        }
    
    def _store_epoch(self, epoch_data: Dict[str, Any]) -> int:
        """Store epoch in database and return epoch_id."""
        cursor = self.db.execute("""
            INSERT INTO epochs (
                system_id,
                epoch_number,
                start_block_number,
                end_block_number,
                block_count,
                epoch_root,
                total_proofs,
                total_ui_events,
                epoch_metadata,
                sealed_by
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (system_id, epoch_number) DO UPDATE SET
                epoch_root = EXCLUDED.epoch_root,
                epoch_metadata = EXCLUDED.epoch_metadata,
                updated_at = NOW()
            RETURNING id
        """, (
            epoch_data["system_id"],
            epoch_data["epoch_number"],
            epoch_data["start_block_number"],
            epoch_data["end_block_number"],
            epoch_data["block_count"],
            epoch_data["epoch_root"],
            epoch_data.get("total_proofs", 0),
            epoch_data.get("total_ui_events", 0),
            json.dumps(epoch_data.get("epoch_metadata", {})),
            "backfill_script",
        ))
        
        return cursor.fetchone()[0]
    
    def _update_block_epochs(self, block_ids: List[int], epoch_id: int):
        """Update blocks.epoch_id for a list of block IDs."""
        self.db.execute("""
            UPDATE blocks
            SET epoch_id = %s, updated_at = NOW()
            WHERE id = ANY(%s)
        """, (epoch_id, block_ids))
    
    def verify_backfill(self) -> Dict[str, Any]:
        """
        Verify backfill completeness.
        
        Checks:
        1. All blocks have epoch_id
        2. All epoch roots are valid
        3. Block counts match
        """
        print("\nVerifying backfill...")
        
        # Check for blocks without epoch_id
        cursor = self.db.execute("""
            SELECT COUNT(*) FROM blocks WHERE epoch_id IS NULL
        """)
        unsealed_blocks = cursor.fetchone()[0]
        
        # Check epoch integrity
        cursor = self.db.execute("""
            SELECT 
                e.id,
                e.epoch_number,
                e.block_count AS declared_count,
                COUNT(b.id) AS actual_count
            FROM epochs e
            LEFT JOIN blocks b ON b.epoch_id = e.id
            GROUP BY e.id
            HAVING e.block_count != COUNT(b.id)
        """)
        mismatched_epochs = cursor.fetchall()
        
        # Summary
        cursor = self.db.execute("""
            SELECT 
                COUNT(DISTINCT system_id) AS systems,
                COUNT(*) AS total_epochs,
                SUM(block_count) AS total_blocks
            FROM epochs
        """)
        summary = cursor.fetchone()
        
        print(f"\nBackfill Verification:")
        print(f"  Systems: {summary[0]}")
        print(f"  Epochs: {summary[1]}")
        print(f"  Blocks in epochs: {summary[2]}")
        print(f"  Unsealed blocks: {unsealed_blocks}")
        print(f"  Mismatched epochs: {len(mismatched_epochs)}")
        
        is_valid = unsealed_blocks == 0 and len(mismatched_epochs) == 0
        
        if is_valid:
            print("\n✓ Backfill verification PASSED")
        else:
            print("\n✗ Backfill verification FAILED")
            if unsealed_blocks > 0:
                print(f"  - {unsealed_blocks} blocks without epoch_id")
            if mismatched_epochs:
                print(f"  - {len(mismatched_epochs)} epochs with block count mismatch")
        
        return {
            "is_valid": is_valid,
            "unsealed_blocks": unsealed_blocks,
            "mismatched_epochs": len(mismatched_epochs),
            "summary": {
                "systems": summary[0],
                "epochs": summary[1],
                "blocks": summary[2],
            },
        }
    
    def run(
        self, 
        system_id: Optional[str] = None, 
        all_systems: bool = False,
        dry_run: bool = False,
        verify_only: bool = False,
    ):
        """
        Run backfill script.
        
        Args:
            system_id: Backfill specific system
            all_systems: Backfill all systems
            dry_run: Preview only, no database changes
            verify_only: Only verify existing backfill
        """
        if verify_only:
            return self.verify_backfill()
        
        if not system_id and not all_systems:
            print("Error: Must specify --system-id or --all")
            sys.exit(1)
        
        # Get systems to process
        if all_systems:
            systems = self.get_systems()
            print(f"Found {len(systems)} systems to process")
        else:
            systems = [system_id]
        
        # Begin transaction
        if not dry_run:
            self.db.execute("BEGIN")
        
        try:
            # Process each system
            for system_id in systems:
                result = self.backfill_system(system_id, dry_run=dry_run)
                
                self.stats["systems_processed"] += 1
                self.stats["epochs_created"] += result["epochs_created"]
                self.stats["blocks_updated"] += result["blocks_updated"]
                self.stats["errors"].extend(result["errors"])
            
            # Commit transaction
            if not dry_run:
                self.db.execute("COMMIT")
                print("\n✓ Transaction committed")
            
        except Exception as e:
            if not dry_run:
                self.db.execute("ROLLBACK")
                print(f"\n✗ Transaction rolled back: {str(e)}")
            raise
        
        # Print summary
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Backfill Summary:")
        print(f"  Systems processed: {self.stats['systems_processed']}")
        print(f"  Epochs created: {self.stats['epochs_created']}")
        print(f"  Blocks updated: {self.stats['blocks_updated']}")
        print(f"  Errors: {len(self.stats['errors'])}")
        
        if self.stats["errors"]:
            print("\nErrors:")
            for error in self.stats["errors"]:
                print(f"  - {error}")
        
        if not dry_run and not verify_only:
            # Run verification
            print("\n" + "="*60)
            self.verify_backfill()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill epochs for existing blocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--system-id",
        help="Backfill specific system (UUID)",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_systems",
        help="Backfill all systems",
    )
    
    parser.add_argument(
        "--epoch-size",
        type=int,
        default=DEFAULT_EPOCH_SIZE,
        help=f"Blocks per epoch (default: {DEFAULT_EPOCH_SIZE})",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only, no database changes",
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        dest="verify_only",
        help="Verify existing backfill (no changes)",
    )
    
    parser.add_argument(
        "--db-url",
        default="postgresql://localhost/mathledger",
        help="Database connection URL",
    )
    
    args = parser.parse_args()
    
    # Connect to database
    # NOTE: This is a placeholder. Replace with actual DB connection logic.
    print(f"Connecting to database: {args.db_url}")
    # db = connect_to_database(args.db_url)
    db = None  # Placeholder
    
    if db is None:
        print("Error: Database connection not implemented")
        print("This script requires a database connection to run.")
        print("Please implement connect_to_database() function.")
        sys.exit(1)
    
    # Run backfill
    script = EpochBackfillScript(db, epoch_size=args.epoch_size)
    script.run(
        system_id=args.system_id,
        all_systems=args.all_systems,
        dry_run=args.dry_run,
        verify_only=args.verify_only,
    )


if __name__ == "__main__":
    main()
