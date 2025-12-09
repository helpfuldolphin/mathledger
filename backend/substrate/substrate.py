# PHASE II — NOT USED IN PHASE I
#
# This module defines the Substrate Abstraction Layer for the U2 harness.
# It includes the Substrate base class with the Operation Substrate-SEAL
# template method pattern for Identity Envelope generation.

import abc
import builtins
import hashlib
import inspect
import json
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from backend.substrate.identity import SubstrateIdentityEnvelope, get_source_file_hash

print("PHASE II — NOT USED IN PHASE I: Loading Substrate Abstraction Layer with Identity-SEAL.", file=sys.stderr)

SPEC_VERSION = "1.0" # From docs/U2_SUBSTRATE_SPECIFICATION.md

@dataclass
class SubstrateResult:
    """Standardized result from a substrate execution's internal logic."""
    success: bool
    result_data: Dict[str, Any]
    verified_hashes: List[str] = field(default_factory=list)
    error: str | None = None

class Substrate(abc.ABC):
    """
    Abstract Base Class for all substrates. Implements the template method
    pattern to enforce Identity Envelope generation on every execution.
    """
    
    def execute(self, item: str, cycle_seed: int) -> Dict[str, Any]:
        """
        Template Method: Executes an item and wraps the result in a signed
        Identity Envelope. Do not override. Subclasses MUST implement
        _execute_internal instead.
        """
        # 1. Prepare for audit
        audit_report = {
            "global_rng_check": "PASSED",
            "file_io_check": "PASSED",
            "network_check": "PASSED",
            "time_query_check": "PASSED",
        }
        
        # 2. Activate behavior audit patches
        with patch('random.random', side_effect=lambda: audit_report.update({"global_rng_check": "FAILED"})) as p_rng, \
             patch('builtins.open', side_effect=lambda *a,**kw: audit_report.update({"file_io_check": "FAILED"})) as p_open, \
             patch('socket.socket', side_effect=lambda *a,**kw: audit_report.update({"network_check": "FAILED"})) as p_sock, \
             patch('time.time', side_effect=lambda: audit_report.update({"time_query_check": "FAILED"})) as p_time:
            
            # 3. Call the subclass's actual implementation
            internal_result = self._execute_internal(item, cycle_seed)

        # 4. Assemble the envelope fields
        source_file = inspect.getfile(self.__class__)
        
        envelope = SubstrateIdentityEnvelope(
            substrate_name=self.__class__.__name__,
            version_hash=get_source_file_hash(source_file),
            spec_version=SPEC_VERSION,
            execution_input={"item": item, "cycle_seed": cycle_seed},
            execution_output={
                "success": internal_result.success,
                "result_data": internal_result.result_data,
                "verified_hashes": internal_result.verified_hashes,
                "error": internal_result.error,
            },
            forbidden_behavior_audit=audit_report
        )
        
        # 5. Sign the envelope
        envelope.sign()
        
        # 6. Return the sealed envelope as a dictionary
        return envelope.to_dict()

    @abc.abstractmethod
    def _execute_internal(self, item: str, cycle_seed: int) -> SubstrateResult:
        """
        Subclass implementation for executing a given item. This method
        is called by the 'execute' template method.
        """
        raise NotImplementedError


class MockSubstrate(Substrate):
    """
    A mock substrate that simulates execution using Python's eval().
    """
    def _execute_internal(self, item: str, cycle_seed: int) -> SubstrateResult:
        try:
            # Use a deterministically seeded RNG, not the global one
            rng = random.Random(cycle_seed)
            
            if "algebra" in item or "Expanded" in item:
                 result = f"Expanded({item})"
            else:
                 result = eval(item)
            
            success = True
            # Use the seeded RNG for deterministic mock hashes
            verified_hashes = [f"mock_verified_{rng.randint(0, 2**32):x}"]

            return SubstrateResult(
                success=success,
                result_data={"mock_result": result},
                verified_hashes=verified_hashes
            )
        except Exception as e:
            return SubstrateResult(success=False, result_data={}, error=f"MockSubstrate eval failed: {e}")


class FoSubstrate(Substrate):
    """
    A substrate that calls the First-Organism (FO) runner script.
    """
    def __init__(self, fo_script_path: str = "backend/substrate/run_fo_cycles.py"):
        # The path is now relative to project root
        self.fo_script_path = str(Path(project_root) / fo_script_path)
        self.python_executable = sys.executable

    def _execute_internal(self, item: str, cycle_seed: int) -> SubstrateResult:
        command = [self.python_executable, self.fo_script_path, "--item", item, "--seed", str(cycle_seed)]
        
        if not Path(self.fo_script_path).exists():
            return SubstrateResult(success=False, result_data={}, error=f"SUB-1: FoSubstrate script not found at {self.fo_script_path}")

        try:
            process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
            result_data = json.loads(process.stdout)
            success = result_data.get("outcome") == "VERIFIED"
            
            return SubstrateResult(
                success=success,
                result_data=result_data,
                verified_hashes=result_data.get("verified_hashes", [])
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"SUB-12: FoSubstrate script failed with exit code {e.returncode}. Stderr: {e.stderr}"
            return SubstrateResult(success=False, result_data={}, error=error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"SUB-22: FoSubstrate failed to decode JSON from script output: {e.msg}. Output: {process.stdout}"
            return SubstrateResult(success=False, result_data={}, error=error_msg)
        except Exception as e:
            return SubstrateResult(success=False, result_data={}, error=f"An unexpected error occurred in FoSubstrate: {e}")