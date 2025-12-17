# REAL-READY
"""
Telemetry Runtime for Lean Verification

Provides comprehensive telemetry collection for Lean verification runs,
including CPU/memory/time monitoring, error code mapping, and tactic extraction.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: REAL-READY
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class LeanVerificationTelemetry:
    """
    Complete telemetry record for a single Lean verification.
    
    This dataclass captures all relevant information for calibration,
    analysis, and debugging of Lean verification runs.
    """
    
    # Identity
    verification_id: str
    timestamp: float
    module_name: str
    context: str
    
    # Configuration
    tier: str = "balanced"
    timeout_s: float = 60.0
    lean_version: str = "unknown"
    
    # Outcome
    outcome: str = "unknown"  # VerifierErrorCode value
    success: bool = False
    duration_ms: float = 0.0
    
    # Resource Usage
    cpu_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    memory_final_mb: float = 0.0
    
    # Lean-specific Metrics
    tactic_count: int = 0
    tactic_depth: int = 0
    proof_size_bytes: int = 0
    search_nodes: int = 0
    
    # Failure Diagnostics
    stderr: str = ""
    returncode: int = 0
    signal: Optional[int] = None
    
    # Noise Injection Metadata
    noise_injected: bool = False
    noise_type: Optional[str] = None
    ground_truth: Optional[str] = None
    noise_seed: Optional[int] = None
    
    # Policy Metadata
    policy_confidence: float = 0.0
    policy_version: str = "unknown"
    
    # Arbitrary Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format (single line)."""
        return json.dumps(self.to_dict(), ensure_ascii=True)


def run_lean_with_monitoring(
    module_name: str,
    lean_command: List[str],
    timeout_s: float,
    verification_id: str,
    context: str = "default",
    tier: str = "balanced",
) -> LeanVerificationTelemetry:
    """
    Run Lean verification with comprehensive monitoring.
    
    Args:
        module_name: Name of the Lean module being verified
        lean_command: Complete Lean command as list of strings
        timeout_s: Timeout in seconds
        verification_id: Unique identifier for this verification
        context: Context string (e.g., "calibration", "u2_experiment")
        tier: Verifier tier (e.g., "fast", "balanced", "slow")
    
    Returns:
        LeanVerificationTelemetry with complete telemetry data
    """
    
    from backend.verification.error_mapper import map_lean_outcome_to_error_code
    from backend.verification.tactic_extractor import extract_tactics_from_output
    from backend.verification.lean_executor import get_lean_version
    
    # Initialize telemetry
    telemetry = LeanVerificationTelemetry(
        verification_id=verification_id,
        timestamp=time.time(),
        module_name=module_name,
        context=context,
        tier=tier,
        timeout_s=timeout_s,
        lean_version=get_lean_version(),
    )
    
    # Start timer
    start_time = time.time()
    
    # Initialize resource tracking
    cpu_samples = []
    memory_samples = []
    
    try:
        # Start Lean process
        process = subprocess.Popen(
            lean_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Monitor process if psutil available
        if PSUTIL_AVAILABLE:
            try:
                ps_process = psutil.Process(process.pid)
                
                # Sample every 100ms until timeout or completion
                sample_interval = 0.1  # 100ms
                elapsed = 0.0
                
                while elapsed < timeout_s:
                    # Check if process completed
                    if process.poll() is not None:
                        break
                    
                    # Sample CPU and memory
                    try:
                        cpu_times = ps_process.cpu_times()
                        memory_info = ps_process.memory_info()
                        
                        cpu_ms = (cpu_times.user + cpu_times.system) * 1000
                        memory_mb = memory_info.rss / (1024 * 1024)
                        
                        cpu_samples.append(cpu_ms)
                        memory_samples.append(memory_mb)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                    
                    # Sleep for sample interval
                    time.sleep(sample_interval)
                    elapsed += sample_interval
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass  # Process may have completed before we could attach
        
        # Wait for process to complete (with timeout)
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
            returncode = process.returncode
            signal_num = None
        except subprocess.TimeoutExpired:
            # Timeout - terminate process
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            
            returncode = -1
            signal_num = 15  # SIGTERM
        
        # Stop timer
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Calculate resource usage
        cpu_time_ms = max(cpu_samples) if cpu_samples else 0.0
        memory_peak_mb = max(memory_samples) if memory_samples else 0.0
        memory_final_mb = memory_samples[-1] if memory_samples else 0.0
        
        # Map outcome to error code
        outcome = map_lean_outcome_to_error_code(
            returncode=returncode,
            stderr=stderr,
            duration_ms=duration_ms,
            timeout_ms=timeout_s * 1000,
        )
        
        # Extract tactics
        tactic_info = extract_tactics_from_output(stdout, stderr)
        
        # Update telemetry
        telemetry.outcome = outcome
        telemetry.success = (outcome == "verified")
        telemetry.duration_ms = duration_ms
        telemetry.cpu_time_ms = cpu_time_ms
        telemetry.memory_peak_mb = memory_peak_mb
        telemetry.memory_final_mb = memory_final_mb
        telemetry.tactic_count = len(tactic_info.get("tactics", []))
        telemetry.tactic_depth = tactic_info.get("tactic_depth", 0)
        telemetry.stderr = stderr[:1000]  # Truncate to 1000 chars
        telemetry.returncode = returncode
        telemetry.signal = signal_num
        
        return telemetry
    
    except Exception as e:
        # Handle unexpected errors
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        telemetry.outcome = "verifier_internal_error"
        telemetry.success = False
        telemetry.duration_ms = duration_ms
        telemetry.stderr = str(e)[:1000]
        telemetry.returncode = -1
        
        return telemetry
