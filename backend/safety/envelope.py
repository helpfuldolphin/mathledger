"""
Safety Envelope

Wraps TDA decisions with safety guarantees and SLO tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime


class SLOStatus(str, Enum):
    """Safety SLO status outcomes."""
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    ADVISORY = "advisory"  # DRY_RUN mode


@dataclass
class SafetySLO:
    """
    Safety Service Level Objective.
    
    Tracks safety metrics and status for a given operation.
    """
    status: SLOStatus
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetySLO":
        """Create from dictionary."""
        return cls(
            status=SLOStatus(data["status"]),
            message=data["message"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp"),
        )


@dataclass
class SafetyEnvelope:
    """
    Safety Envelope wrapping TDA decisions.
    
    Integrates TDA (Topological Decision Analysis) hard gate decisions
    with safety SLO outcomes.
    """
    slo: SafetySLO
    tda_should_block: bool
    tda_mode: str  # "BLOCK", "DRY_RUN", "SHADOW"
    decision: str  # Final decision: "proceed", "block", "advisory"
    audit_trail: Dict[str, Any] = field(default_factory=dict)
    
    def is_blocking(self) -> bool:
        """Check if this envelope blocks execution."""
        return self.decision == "block"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "slo": self.slo.to_dict(),
            "tda_should_block": self.tda_should_block,
            "tda_mode": self.tda_mode,
            "decision": self.decision,
            "audit_trail": self.audit_trail,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyEnvelope":
        """Create from dictionary."""
        return cls(
            slo=SafetySLO.from_dict(data["slo"]),
            tda_should_block=data["tda_should_block"],
            tda_mode=data["tda_mode"],
            decision=data["decision"],
            audit_trail=data.get("audit_trail", {}),
        )
