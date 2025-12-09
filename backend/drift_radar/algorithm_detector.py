"""
Algorithm Drift Detector

Detects mismatches between expected and actual hash algorithms in blocks.

Author: Manus-H
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, List
from enum import Enum

from backend.consensus_pq.epoch import get_epoch_for_block
from basis.ledger.block_pq import BlockHeaderPQ


class DriftSeverity(Enum):
    """Severity levels for drift events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftEvent:
    """
    Represents a detected drift event.
    
    Attributes:
        detector_name: Name of the detector that found the drift
        event_type: Type of drift event
        severity: Severity level
        block_number: Block number where drift was detected
        timestamp: Unix timestamp when drift was detected
        description: Human-readable description
        metadata: Additional metadata about the drift
    """
    
    detector_name: str
    event_type: str
    severity: DriftSeverity
    block_number: int
    timestamp: float
    description: str
    metadata: dict
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["severity"] = self.severity.value
        return d
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AlgorithmDriftDetector:
    """
    Detects algorithm mismatches in blocks.
    
    Monitors blocks to ensure they use the correct hash algorithm
    for their epoch.
    """
    
    def __init__(self):
        self.name = "AlgorithmDriftDetector"
        self.events: List[DriftEvent] = []
    
    def detect(self, block: BlockHeaderPQ) -> List[DriftEvent]:
        """
        Detect algorithm drift in a block.
        
        Args:
            block: Block to check
            
        Returns:
            List of detected drift events
        """
        events = []
        
        # Get expected algorithm for this block's epoch
        epoch = get_epoch_for_block(block.block_number)
        
        if epoch is None:
            # No epoch registered - this is a drift event
            event = DriftEvent(
                detector_name=self.name,
                event_type="missing_epoch",
                severity=DriftSeverity.CRITICAL,
                block_number=block.block_number,
                timestamp=time.time(),
                description=f"No epoch registered for block {block.block_number}",
                metadata={
                    "block_number": block.block_number,
                },
            )
            events.append(event)
            self.events.append(event)
            return events
        
        # Check if block has PQ fields when it should
        if epoch.algorithm_id != 0x00:  # Non-SHA256 epoch
            if block.pq_algorithm is None:
                event = DriftEvent(
                    detector_name=self.name,
                    event_type="missing_pq_fields",
                    severity=DriftSeverity.HIGH,
                    block_number=block.block_number,
                    timestamp=time.time(),
                    description=f"Block {block.block_number} missing PQ fields in epoch {epoch.algorithm_name}",
                    metadata={
                        "block_number": block.block_number,
                        "expected_algorithm": epoch.algorithm_name,
                        "expected_algorithm_id": epoch.algorithm_id,
                    },
                )
                events.append(event)
                self.events.append(event)
        
        # Check if PQ algorithm matches epoch
        if block.pq_algorithm is not None:
            if block.pq_algorithm != epoch.algorithm_id:
                event = DriftEvent(
                    detector_name=self.name,
                    event_type="algorithm_mismatch",
                    severity=DriftSeverity.CRITICAL,
                    block_number=block.block_number,
                    timestamp=time.time(),
                    description=(
                        f"Block {block.block_number} uses algorithm {block.pq_algorithm:02x} "
                        f"but epoch expects {epoch.algorithm_id:02x}"
                    ),
                    metadata={
                        "block_number": block.block_number,
                        "actual_algorithm_id": block.pq_algorithm,
                        "expected_algorithm_id": epoch.algorithm_id,
                        "expected_algorithm": epoch.algorithm_name,
                    },
                )
                events.append(event)
                self.events.append(event)
        
        return events
    
    def detect_batch(self, blocks: List[BlockHeaderPQ]) -> List[DriftEvent]:
        """
        Detect algorithm drift across multiple blocks.
        
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


# CI-compatible alerting
class CIAlertHandler:
    """
    Handles alerts in CI/CD environments.
    
    Outputs alerts in formats compatible with GitHub Actions, GitLab CI, etc.
    """
    
    def __init__(self, ci_system: str = "github"):
        """
        Initialize alert handler.
        
        Args:
            ci_system: CI system type ("github", "gitlab", "jenkins", "generic")
        """
        self.ci_system = ci_system
    
    def alert(self, event: DriftEvent) -> None:
        """
        Send alert for a drift event.
        
        Args:
            event: Drift event to alert on
        """
        if self.ci_system == "github":
            self._alert_github(event)
        elif self.ci_system == "gitlab":
            self._alert_gitlab(event)
        else:
            self._alert_generic(event)
    
    def _alert_github(self, event: DriftEvent) -> None:
        """Format alert for GitHub Actions."""
        # GitHub Actions annotation format
        if event.severity == DriftSeverity.CRITICAL:
            level = "error"
        elif event.severity == DriftSeverity.HIGH:
            level = "error"
        elif event.severity == DriftSeverity.MEDIUM:
            level = "warning"
        else:
            level = "notice"
        
        print(f"::{level}::PQ Drift Detected - {event.description}")
        print(f"::set-output name=drift_detected::true")
        print(f"::set-output name=drift_severity::{event.severity.value}")
    
    def _alert_gitlab(self, event: DriftEvent) -> None:
        """Format alert for GitLab CI."""
        # GitLab CI format (JSON to stdout)
        alert_data = {
            "severity": event.severity.value,
            "message": event.description,
            "detector": event.detector_name,
            "block_number": event.block_number,
        }
        print(f"GITLAB_ALERT: {json.dumps(alert_data)}")
    
    def _alert_generic(self, event: DriftEvent) -> None:
        """Generic alert format."""
        print(f"[{event.severity.value.upper()}] PQ Drift Detected")
        print(f"  Detector: {event.detector_name}")
        print(f"  Type: {event.event_type}")
        print(f"  Block: {event.block_number}")
        print(f"  Description: {event.description}")


def run_algorithm_drift_detection(
    blocks: List[BlockHeaderPQ],
    ci_system: Optional[str] = None,
) -> tuple[List[DriftEvent], dict]:
    """
    Run algorithm drift detection on a list of blocks.
    
    Args:
        blocks: List of blocks to check
        ci_system: CI system for alerting (optional)
        
    Returns:
        Tuple of (drift_events, stats)
    """
    detector = AlgorithmDriftDetector()
    alert_handler = CIAlertHandler(ci_system) if ci_system else None
    
    print(f"Running algorithm drift detection on {len(blocks)} blocks...")
    
    events = detector.detect_batch(blocks)
    
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
    from backend.consensus_pq.epoch import initialize_genesis_epoch
    
    # Initialize genesis epoch
    initialize_genesis_epoch()
    
    # Create test blocks
    test_blocks = [
        BlockHeaderPQ(
            block_number=1,
            prev_hash="0x" + "00" * 32,
            merkle_root="0x" + "11" * 32,
            timestamp=time.time(),
            statements=["stmt1"],
        ),
        BlockHeaderPQ(
            block_number=2,
            prev_hash="0x" + "11" * 32,
            merkle_root="0x" + "22" * 32,
            timestamp=time.time(),
            statements=["stmt2"],
            pq_algorithm=0x01,  # Algorithm mismatch (epoch expects 0x00)
            pq_merkle_root="0x" + "33" * 32,
        ),
    ]
    
    # Run detection
    events, stats = run_algorithm_drift_detection(test_blocks, ci_system="github")
    
    # Export results
    detector = AlgorithmDriftDetector()
    detector.events = events
    detector.export_events_json("algorithm_drift_events.json")
