#!/usr/bin/env python3
"""
Hermetic Matrix Resharding Tool

Partitions matrix history into time-window shards for distributed verification.

Usage:
    python tools/hermetic/reshard_history.py --num-shards 64
    python tools/hermetic/reshard_history.py --num-shards 16 --input custom_history.jsonl

Features:
- Time-window partitioning of matrix history
- Configurable shard count (16, 64, 256, etc.)
- Preserves chronological ordering within shards
- Generates shard manifest for tracking

Pass-Lines:
    [PASS] Resharding Complete (64 shards)
    [ABSTAIN] Insufficient history for resharding
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from backend.repro.determinism import deterministic_isoformat


class HistoryResharder:
    """Reshards matrix history into time-window partitions."""

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize resharder with repository root."""
        self.repo_root = repo_root or Path(__file__).parent.parent.parent
        self.history_file = self.repo_root / "artifacts" / "no_network" / "matrix_history.jsonl"
        self.shard_dir = self.repo_root / "artifacts" / "hermetic"
        self.shard_dir.mkdir(parents=True, exist_ok=True)

    def load_history(self) -> List[Dict]:
        """Load all history entries from JSONL file."""
        if not self.history_file.exists():
            return []
        
        history = []
        try:
            with open(self.history_file, 'r', encoding='ascii') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[ERROR] Failed to load history: {e}", file=sys.stderr)
        
        return history

    def partition_by_time_window(self, history: List[Dict], num_shards: int) -> List[List[Dict]]:
        """
        Partition history into equal-sized time windows.
        
        Args:
            history: List of history entries
            num_shards: Number of shards to create
        
        Returns:
            List of shard partitions (each partition is a list of entries)
        """
        if not history:
            return []
        
        entries_per_shard = max(1, len(history) // num_shards)
        remainder = len(history) % num_shards
        
        shards = []
        start_idx = 0
        
        for i in range(num_shards):
            extra = 1 if i < remainder else 0
            end_idx = start_idx + entries_per_shard + extra
            
            shard_entries = history[start_idx:end_idx]
            if shard_entries:
                shards.append(shard_entries)
            else:
                shards.append([])
            
            start_idx = end_idx
        
        return shards

    def write_shards(self, shards: List[List[Dict]], num_shards: int) -> int:
        """
        Write shard partitions to JSONL files.
        
        Args:
            shards: List of shard partitions
            num_shards: Total number of shards (for zero-padding)
        
        Returns:
            Number of shards written
        """
        written_count = 0
        
        for shard_id, shard_entries in enumerate(shards):
            if not shard_entries:
                continue
            
            shard_file = self.shard_dir / f"shard_{shard_id:02d}.jsonl"
            
            try:
                with open(shard_file, 'w', encoding='ascii') as f:
                    for entry in shard_entries:
                        canonical_entry = json.dumps(
                            entry,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(',', ':'),
                        )
                        f.write(canonical_entry + '\n')
                
                written_count += 1
                print(f"[INFO] Wrote shard_{shard_id:02d}.jsonl ({len(shard_entries)} entries)")
            
            except Exception as e:
                print(f"[ERROR] Failed to write shard {shard_id}: {e}", file=sys.stderr)
        
        return written_count

    def generate_manifest(self, shards: List[List[Dict]], num_shards: int) -> Dict:
        """
        Generate shard manifest with metadata.
        
        Args:
            shards: List of shard partitions
            num_shards: Total number of shards
        
        Returns:
            Manifest dictionary
        """
        shard_metadata = []
        total_entries = 0
        
        for shard_id, shard_entries in enumerate(shards):
            if shard_entries:
                first_timestamp = shard_entries[0].get("timestamp", "")
                last_timestamp = shard_entries[-1].get("timestamp", "")
                entry_count = len(shard_entries)
                total_entries += entry_count
                
                shard_metadata.append({
                    "shard_id": shard_id,
                    "entries": entry_count,
                    "first_timestamp": first_timestamp,
                    "last_timestamp": last_timestamp,
                    "status": "present"
                })
            else:
                shard_metadata.append({
                    "shard_id": shard_id,
                    "entries": 0,
                    "first_timestamp": "",
                    "last_timestamp": "",
                    "status": "empty"
                })
        
        manifest = {
            "timestamp": deterministic_isoformat("reshard_manifest", num_shards, total_entries),
            "num_shards": num_shards,
            "total_entries": total_entries,
            "shards": shard_metadata
        }
        
        return manifest

    def save_manifest(self, manifest: Dict) -> Path:
        """
        Save shard manifest to JSON file.
        
        Args:
            manifest: Manifest dictionary
        
        Returns:
            Path to saved manifest file
        """
        manifest_file = self.shard_dir / "shard_manifest.json"
        
        with open(manifest_file, 'w', encoding='ascii') as f:
            canonical_manifest = json.dumps(
                manifest,
                ensure_ascii=True,
                sort_keys=True,
                separators=(',', ':'),
            )
            f.write(canonical_manifest)
        
        print(f"[INFO] Manifest saved: {manifest_file}")
        return manifest_file

    def reshard(self, num_shards: int = 64) -> bool:
        """
        Reshard history into specified number of shards.
        
        Args:
            num_shards: Number of shards to create
        
        Returns:
            True if successful, False otherwise
        """
        history = self.load_history()
        
        if not history:
            print(f"[ABSTAIN] Insufficient history for resharding", file=sys.stderr)
            return False
        
        if len(history) < num_shards:
            print(f"[WARNING] History has {len(history)} entries but {num_shards} shards requested")
            print(f"[WARNING] Some shards will be empty")
        
        shards = self.partition_by_time_window(history, num_shards)
        
        written_count = self.write_shards(shards, num_shards)
        
        manifest = self.generate_manifest(shards, num_shards)
        self.save_manifest(manifest)
        
        print(f"[PASS] Resharding Complete ({written_count} shards)")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hermetic Matrix Resharding Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/hermetic/reshard_history.py --num-shards 64
    
    python tools/hermetic/reshard_history.py --num-shards 16
    
    python tools/hermetic/reshard_history.py --num-shards 256
        """
    )
    
    parser.add_argument(
        '--num-shards',
        type=int,
        default=64,
        help='Number of shards to create (default: 64)'
    )
    
    args = parser.parse_args()
    
    if args.num_shards < 1:
        print("[ERROR] Number of shards must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    resharder = HistoryResharder()
    success = resharder.reshard(num_shards=args.num_shards)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
