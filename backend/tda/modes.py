"""
TDA Modes

Defines TDA operational modes and decision structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TDAMode(str, Enum):
    """TDA operational modes."""
    BLOCK = "BLOCK"  # Actually blocks execution
    DRY_RUN = "DRY_RUN"  # Advisory only, doesn't block
    SHADOW = "SHADOW"  # Records hypothetical status


@dataclass
class TDADecision:
    """
    TDA decision output.
    
    Represents the result of topological analysis for a hard gate.
    """
    should_block: bool
    mode: TDAMode
    reason: str
    confidence: float = 1.0  # Confidence in decision (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "should_block": self.should_block,
            "mode": self.mode.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TDADecision":
        """Create from dictionary."""
        return cls(
            should_block=data["should_block"],
            mode=TDAMode(data["mode"]),
            reason=data["reason"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )
