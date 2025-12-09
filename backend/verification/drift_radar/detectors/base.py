"""
Base Detector Interface

All drift detectors inherit from this base class.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from backend.verification.telemetry.schema import LeanVerificationTelemetry


@dataclass
class DetectorConfig:
    """Base configuration for detectors."""
    
    enabled: bool = True
    name: str = "base_detector"


class DriftDetector(ABC):
    """Base class for all drift detectors."""
    
    def __init__(self, config: DetectorConfig):
        """Initialize detector.
        
        Args:
            config: Detector configuration
        """
        self.config = config
        self.alarm_count = 0
    
    @abstractmethod
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update detector with new telemetry.
        
        Args:
            telemetry: Lean verification telemetry
        
        Returns:
            Alarm dict if drift detected, None otherwise
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current detector state for logging.
        
        Returns:
            Dict with detector state
        """
        pass
