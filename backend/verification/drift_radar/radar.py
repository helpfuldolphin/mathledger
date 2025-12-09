"""
Unified Verifier Noise Drift Radar

Combines multiple detectors for comprehensive drift monitoring.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from backend.verification.telemetry.schema import LeanVerificationTelemetry
from .detectors.cusum_detector import CUSUMDetector, CUSUMConfig
from .detectors.tier_skew_detector import TierSkewDetector, TierSkewConfig
from .detectors.scan_statistics_detector import ScanStatisticsDetector, ScanStatisticsConfig


class AlertLevel(Enum):
    """Alert severity levels."""
    NORMAL = "normal"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class RadarConfig:
    """Configuration for unified drift radar."""
    
    # Detector configs
    cusum_timeout: CUSUMConfig = field(default_factory=lambda: CUSUMConfig(
        name="cusum_timeout",
        metric="timeout_rate",
        mu_0=0.1,
        k=0.05,
        h=5.0,
    ))
    
    cusum_spurious_fail: CUSUMConfig = field(default_factory=lambda: CUSUMConfig(
        name="cusum_spurious_fail",
        metric="spurious_fail_rate",
        mu_0=0.05,
        k=0.025,
        h=5.0,
    ))
    
    tier_skew: TierSkewConfig = field(default_factory=TierSkewConfig)
    
    scan_statistics: ScanStatisticsConfig = field(default_factory=ScanStatisticsConfig)
    
    # Alert thresholds
    warning_threshold: int = 5   # Alarms before WARNING
    alert_threshold: int = 10    # Alarms before ALERT
    critical_threshold: int = 20  # Alarms before CRITICAL


class VerifierNoiseDriftRadar:
    """Unified drift radar with multiple detectors.
    
    Usage:
        config = RadarConfig()
        radar = VerifierNoiseDriftRadar(config)
        
        # For each telemetry
        alarms = radar.update(telemetry)
        
        # Check alert level
        level = radar.get_alert_level()
    """
    
    def __init__(self, config: RadarConfig):
        """Initialize drift radar.
        
        Args:
            config: Radar configuration
        """
        self.config = config
        
        # Initialize detectors
        self.detectors = {
            "cusum_timeout": CUSUMDetector(config.cusum_timeout),
            "cusum_spurious_fail": CUSUMDetector(config.cusum_spurious_fail),
            "tier_skew": TierSkewDetector(config.tier_skew),
            "scan_statistics": ScanStatisticsDetector(config.scan_statistics),
        }
        
        # Alarm history
        self.alarm_history: List[Dict[str, Any]] = []
        
        # Total alarm count
        self.total_alarms = 0
        
        # Current alert level
        self.alert_level = AlertLevel.NORMAL
    
    def update(self, telemetry: LeanVerificationTelemetry) -> List[Dict[str, Any]]:
        """Update all detectors with new telemetry.
        
        Args:
            telemetry: Lean verification telemetry
        
        Returns:
            List of alarms (empty if no drift detected)
        """
        
        alarms = []
        
        # Update each detector
        for name, detector in self.detectors.items():
            alarm = detector.update(telemetry)
            
            if alarm:
                # Add timestamp and detector name
                alarm["timestamp"] = telemetry.timestamp
                alarm["detector_name"] = name
                
                # Record alarm
                alarms.append(alarm)
                self.alarm_history.append(alarm)
                self.total_alarms += 1
        
        # Update alert level
        self._update_alert_level()
        
        return alarms
    
    def _update_alert_level(self) -> None:
        """Update alert level based on total alarms."""
        
        if self.total_alarms >= self.config.critical_threshold:
            self.alert_level = AlertLevel.CRITICAL
        elif self.total_alarms >= self.config.alert_threshold:
            self.alert_level = AlertLevel.ALERT
        elif self.total_alarms >= self.config.warning_threshold:
            self.alert_level = AlertLevel.WARNING
        else:
            self.alert_level = AlertLevel.NORMAL
    
    def get_alert_level(self) -> AlertLevel:
        """Get current alert level.
        
        Returns:
            AlertLevel
        """
        return self.alert_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get current radar state.
        
        Returns:
            Dict with radar state
        """
        detector_states = {}
        for name, detector in self.detectors.items():
            detector_states[name] = detector.get_state()
        
        return {
            "alert_level": self.alert_level.value,
            "total_alarms": self.total_alarms,
            "detector_states": detector_states,
            "recent_alarms": self.alarm_history[-10:],  # Last 10 alarms
        }
    
    def reset(self) -> None:
        """Reset all detectors and alarm history."""
        for detector in self.detectors.values():
            detector.reset()
        
        self.alarm_history.clear()
        self.total_alarms = 0
        self.alert_level = AlertLevel.NORMAL
    
    def export_dashboard_json(self) -> Dict[str, Any]:
        """Export dashboard configuration for Grafana.
        
        Returns:
            Grafana dashboard JSON
        """
        
        dashboard = {
            "title": "Verifier Noise Drift Radar",
            "panels": [
                {
                    "title": "Alert Level",
                    "type": "stat",
                    "targets": [{"expr": "drift_radar_alert_level"}],
                },
                {
                    "title": "Total Alarms",
                    "type": "graph",
                    "targets": [{"expr": "drift_radar_total_alarms"}],
                },
                {
                    "title": "CUSUM Timeout",
                    "type": "graph",
                    "targets": [
                        {"expr": "drift_radar_cusum_timeout_S_plus"},
                        {"expr": "drift_radar_cusum_timeout_S_minus"},
                    ],
                },
                {
                    "title": "Tier Skew",
                    "type": "table",
                    "targets": [{"expr": "drift_radar_tier_timeout_rates"}],
                },
                {
                    "title": "Scan Statistics",
                    "type": "graph",
                    "targets": [{"expr": "drift_radar_scan_statistics_S"}],
                },
                {
                    "title": "Recent Alarms",
                    "type": "logs",
                    "targets": [{"expr": "drift_radar_alarms"}],
                },
            ],
        }
        
        return dashboard
