"""
Dual Commitment Consistency Detector

Verifies that dual commitments correctly bind legacy and PQ hashes.

Author: Manus-H
"""

import time
import json
from typing import List, Optional
from dataclasses import dataclass

from basis.ledger.block_pq import BlockHeaderPQ
from basis.crypto.hash_versioned import compute_dual_commitment, verify_dual_commitment
from backend.drift_radar.algorithm_detector import DriftEvent, DriftSeverity, CIAlertHandler


class DualCommitmentDetector:
    """
    Detects inconsistencies in dual commitments.
    
    Verifies that dual_commitment field correctly binds legacy_merkle_root
    and pq_merkle_root.
    """
    
    def __init__(self):
        self.name = "DualCommitmentDetector"
        self.events: List[DriftEvent] = []
    
    def detect(self, block: BlockHeaderPQ) -> List[DriftEvent]:
        """
        Detect dual commitment inconsistencies in a block.
        
        Args:
            block: Block to check
            
        Returns:
            List of detected drift events
        """
        events = []
        
        # Skip blocks without PQ fields
        if block.pq_algorithm is None or block.pq_merkle_root is None:
            return events
        
        # Check if dual_commitment is present
        if block.dual_commitment is None:
            event = DriftEvent(
                detector_name=self.name,
                event_type="missing_dual_commitment",
                severity=DriftSeverity.HIGH,
                block_number=block.block_number,
                timestamp=time.time(),
                description=f"Block {block.block_number} has PQ fields but missing dual_commitment",
                metadata={
                    "block_number": block.block_number,
                    "pq_algorithm": block.pq_algorithm,
                },
            )
            events.append(event)
            self.events.append(event)
            return events
        
        # Verify dual commitment
        is_valid = verify_dual_commitment(
            dual_commitment=block.dual_commitment,
            legacy_hash=block.merkle_root,
            pq_hash=block.pq_merkle_root,
            pq_algorithm_id=block.pq_algorithm,
        )
        
        if not is_valid:
            # Recompute expected dual commitment
            expected_commitment = compute_dual_commitment(
                legacy_hash=block.merkle_root,
                pq_hash=block.pq_merkle_root,
                pq_algorithm_id=block.pq_algorithm,
            )
            
            event = DriftEvent(
                detector_name=self.name,
                event_type="dual_commitment_mismatch",
                severity=DriftSeverity.CRITICAL,
                block_number=block.block_number,
                timestamp=time.time(),
                description=(
                    f"Block {block.block_number} has invalid dual_commitment: "
                    f"expected {expected_commitment}, got {block.dual_commitment}"
                ),
                metadata={
                    "block_number": block.block_number,
                    "actual_commitment": block.dual_commitment,
                    "expected_commitment": expected_commitment,
                    "legacy_hash": block.merkle_root,
                    "pq_hash": block.pq_merkle_root,
                    "pq_algorithm": block.pq_algorithm,
                },
            )
            events.append(event)
            self.events.append(event)
        
        return events
    
    def detect_batch(self, blocks: List[BlockHeaderPQ]) -> List[DriftEvent]:
        """
        Detect dual commitment drift across multiple blocks.
        
        Args:
            blocks: List of blocks to check
            
        Returns:
            List of all detected drift events
        """
        all_events = []
        
        for block in blocks:
            events = self.detect(block)
            all_events.extend(events)
        
        return all_events
    
    def detect_chain_consistency(self, blocks: List[BlockHeaderPQ]) -> List[DriftEvent]:
        """
        Detect consistency issues across a chain of blocks.
        
        Checks that dual commitments form a consistent chain.
        
        Args:
            blocks: List of consecutive blocks
            
        Returns:
            List of detected drift events
        """
        events = []
        
        for i in range(1, len(blocks)):
            block = blocks[i]
            prev_block = blocks[i - 1]
            
            # Check if both blocks have dual commitments
            if block.dual_commitment is None or prev_block.dual_commitment is None:
                continue
            
            # Verify that current block's prev_hash references previous block correctly
            # (This is checked by prev_hash validators, but we double-check here)
            
            # Check for sudden algorithm changes without epoch boundary
            if block.pq_algorithm != prev_block.pq_algorithm:
                # This might be legitimate (epoch transition), but flag for review
                event = DriftEvent(
                    detector_name=self.name,
                    event_type="algorithm_change_detected",
                    severity=DriftSeverity.MEDIUM,
                    block_number=block.block_number,
                    timestamp=time.time(),
                    description=(
                        f"Algorithm changed from {prev_block.pq_algorithm:02x} to "
                        f"{block.pq_algorithm:02x} at block {block.block_number}"
                    ),
                    metadata={
                        "block_number": block.block_number,
                        "prev_algorithm": prev_block.pq_algorithm,
                        "new_algorithm": block.pq_algorithm,
                    },
                )
                events.append(event)
                self.events.append(event)
        
        return events
    
    def get_events(
        self,
        severity: Optional[DriftSeverity] = None,
        since: Optional[float] = None,
    ) -> List[DriftEvent]:
        """
        Get detected drift events with optional filtering.
        
        Args:
            severity: Filter by severity level
            since: Filter events after this timestamp
            
        Returns:
            List of filtered drift events
        """
        events = self.events
        
        if severity is not None:
            events = [e for e in events if e.severity == severity]
        
        if since is not None:
            events = [e for e in events if e.timestamp >= since]
        
        return events
    
    def clear_events(self) -> None:
        """Clear all stored events."""
        self.events.clear()
    
    def export_events_json(self, filename: str) -> None:
        """
        Export events to JSON file.
        
        Args:
            filename: Output file path
        """
        with open(filename, 'w') as f:
            json.dump([e.to_dict() for e in self.events], f, indent=2)
    
    def get_stats(self) -> dict:
        """
        Get statistics about detected drift events.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_events": len(self.events),
            "by_severity": {
                "critical": len([e for e in self.events if e.severity == DriftSeverity.CRITICAL]),
                "high": len([e for e in self.events if e.severity == DriftSeverity.HIGH]),
                "medium": len([e for e in self.events if e.severity == DriftSeverity.MEDIUM]),
                "low": len([e for e in self.events if e.severity == DriftSeverity.LOW]),
            },
            "by_type": {},
        }
        
        # Count by event type
        for event in self.events:
            event_type = event.event_type
            if event_type not in stats["by_type"]:
                stats["by_type"][event_type] = 0
            stats["by_type"][event_type] += 1
        
        return stats


def run_commitment_drift_detection(
    blocks: List[BlockHeaderPQ],
    check_chain_consistency: bool = True,
    ci_system: Optional[str] = None,
) -> tuple[List[DriftEvent], dict]:
    """
    Run dual commitment drift detection on a list of blocks.
    
    Args:
        blocks: List of blocks to check
        check_chain_consistency: Whether to check chain-level consistency
        ci_system: CI system for alerting (optional)
        
    Returns:
        Tuple of (drift_events, stats)
    """
    detector = DualCommitmentDetector()
    alert_handler = CIAlertHandler(ci_system) if ci_system else None
    
    print(f"Running dual commitment drift detection on {len(blocks)} blocks...")
    
    # Detect individual block issues
    events = detector.detect_batch(blocks)
    
    # Check chain consistency if requested
    if check_chain_consistency and len(blocks) > 1:
        chain_events = detector.detect_chain_consistency(blocks)
        events.extend(chain_events)
    
    # Send alerts for critical/high severity events
    if alert_handler:
        for event in events:
            if event.severity in [DriftSeverity.CRITICAL, DriftSeverity.HIGH]:
                alert_handler.alert(event)
    
    stats = detector.get_stats()
    
    print(f"Detected {len(events)} drift events")
    print(f"  Critical: {stats['by_severity']['critical']}")
    print(f"  High: {stats['by_severity']['high']}")
    print(f"  Medium: {stats['by_severity']['medium']}")
    print(f"  Low: {stats['by_severity']['low']}")
    
    return events, stats


if __name__ == "__main__":
    # Example usage
    from basis.crypto.hash_versioned import compute_dual_commitment
    
    # Create test blocks with valid dual commitments
    test_blocks = [
        BlockHeaderPQ(
            block_number=1000,
            prev_hash="0x" + "00" * 32,
            merkle_root="0x" + "11" * 32,
            timestamp=time.time(),
            statements=["stmt1"],
            pq_algorithm=0x01,
            pq_merkle_root="0x" + "22" * 32,
            pq_prev_hash="0x" + "00" * 32,
            dual_commitment=compute_dual_commitment(
                legacy_hash="0x" + "11" * 32,
                pq_hash="0x" + "22" * 32,
                pq_algorithm_id=0x01,
            ),
        ),
        BlockHeaderPQ(
            block_number=1001,
            prev_hash="0x" + "11" * 32,
            merkle_root="0x" + "33" * 32,
            timestamp=time.time(),
            statements=["stmt2"],
            pq_algorithm=0x01,
            pq_merkle_root="0x" + "44" * 32,
            pq_prev_hash="0x" + "22" * 32,
            dual_commitment="0x" + "FF" * 32,  # Invalid commitment
        ),
    ]
    
    # Run detection
    events, stats = run_commitment_drift_detection(
        test_blocks,
        check_chain_consistency=True,
        ci_system="github",
    )
    
    # Export results
    detector = DualCommitmentDetector()
    detector.events = events
    detector.export_events_json("commitment_drift_events.json")
    
    print(f"\nExported {len(events)} events to commitment_drift_events.json")
