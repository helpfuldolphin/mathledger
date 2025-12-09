"""
Lean Verification Telemetry Schema

Dataclass for structured telemetry from Lean verification runs.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import time
from backend.verification.error_codes import VerifierErrorCode, VerifierTier


@dataclass
class LeanVerificationTelemetry:
    """Complete telemetry record for a single Lean verification run.
    
    This dataclass captures all relevant information about a verification attempt,
    including identity, configuration, outcome, resource usage, Lean-specific metrics,
    failure diagnostics, and noise injection metadata.
    """
    
    # === Identity ===
    verification_id: str
    timestamp: float = field(default_factory=time.time)
    module_name: str = ""
    context: str = ""  # Context string for PRNG seeding
    
    # === Configuration ===
    tier: VerifierTier = VerifierTier.BALANCED
    timeout_s: float = 60.0
    lean_version: str = "unknown"
    
    # === Outcome ===
    outcome: VerifierErrorCode = VerifierErrorCode.VERIFIER_INTERNAL_ERROR
    success: bool = False
    duration_ms: float = 0.0
    
    # === Resource Usage ===
    cpu_time_ms: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    memory_final_mb: Optional[float] = None
    
    # === Lean-Specific Metrics ===
    tactic_count: Optional[int] = None
    tactic_depth: Optional[int] = None
    proof_size_bytes: Optional[int] = None
    search_nodes: Optional[int] = None
    
    # === Failure Diagnostics ===
    stderr: str = ""
    returncode: Optional[int] = None
    signal: Optional[int] = None
    
    # === Noise Injection Metadata ===
    noise_injected: bool = False
    noise_type: Optional[str] = None  # "timeout", "spurious_fail", "spurious_pass"
    ground_truth: Optional[str] = None  # "VERIFIED", "INVALID", "UNKNOWN"
    
    # === Arbitrary Metadata ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert enums to strings
        d["outcome"] = self.outcome.value
        d["tier"] = self.tier.value
        return d
    
    def to_json_line(self) -> str:
        """Convert to JSON line for JSONL logging."""
        import json
        return json.dumps(self.to_dict())
