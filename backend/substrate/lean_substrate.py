"""
This module contains the implementation of the LeanSubstrate, which manages
Inter-Process Communication (IPC) with a Lean theorem prover process. It is
designed to enforce the determinism contract and the SUB-Error taxonomy.
"""
import json
import os
import subprocess
import logging
from typing import Dict, Any

from backend.substrate.lean_substrate_interface import (
    SubstrateRequest,
    SubstrateResponse,
    compute_request_hash,
)

# Setup logger for this module
log = logging.getLogger(__name__)

# ==============================================================================
# Custom Exceptions for Substrate Failures
# ==============================================================================

class SubstrateError(Exception):
    """Base exception for all substrate-related errors."""
    def __init__(self, message: str, code: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message

# ==============================================================================
# LeanSubstrate Implementation
# ==============================================================================

class LeanSubstrate:
    """
    A substrate that executes a Lean process, communicating via a typed
    JSON protocol over stdin/stdout.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lean_executable_path = self.config.get("lean_executable_path")

        # SUB-1: Configuration Error
        if not self.lean_executable_path:
            raise SubstrateError(
                "Path to Lean executable ('lean_executable_path') is not configured.",
                "SUB-1"
            )
        # SUB-2: Binary Not Found
        if not os.path.exists(self.lean_executable_path):
            raise SubstrateError(
                f"Lean executable not found at: {self.lean_executable_path}",
                "SUB-2"
            )

    def execute(self, request: SubstrateRequest) -> SubstrateResponse:
        """
        Executes the full FSM for a request against the Lean substrate.
        VALIDATE -> SERIALIZE -> SPAWN -> AWAIT -> DESERIALIZE -> VALIDATE -> END
        """
        # 1. VALIDATE_REQUEST (Partially done by type hints)
        if request['budget']['taut_timeout_s'] <= 0:
            raise SubstrateError("taut_timeout_s must be positive.", "SUB-15")

        # 2. SERIALIZE_REQUEST
        request_hash = compute_request_hash(request)
        try:
            # Note: The canonical serializer is in the interface module, but here
            # we just need a standard dump for the subprocess.
            request_json = json.dumps(request)
        except TypeError as e:
            raise SubstrateError(f"Request is not JSON serializable: {e}", "SUB-15")

        # 3. SPAWN_PROCESS
        proc = None
        try:
            proc = subprocess.Popen(
                [self.lean_executable_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
        except FileNotFoundError:
            raise SubstrateError(f"Lean executable not found at: {self.lean_executable_path}", "SUB-2")
        except Exception as e:
            raise SubstrateError(f"Failed to spawn Lean process: {e}", "SUB-2")

        # 4. AWAIT_RESPONSE
        try:
            stdout, stderr = proc.communicate(
                input=request_json + "\n",
                timeout=request['budget']['taut_timeout_s']
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            raise SubstrateError(f"Substrate timed out after {request['budget']['taut_timeout_s']}s.", "SUB-11")
        finally:
            # Ensure process is cleaned up
            if proc and proc.poll() is None:
                proc.kill()


        if stderr:
            raise SubstrateError(f"Substrate wrote to stderr, violating IPC contract. Stderr: {stderr}", "SUB-22")

        if not stdout:
            raise SubstrateError("Substrate closed stdout without sending a response.", "SUB-24")

        # 5. DESERIALIZE_RESPONSE
        try:
            response_data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise SubstrateError(f"Failed to decode JSON response from substrate: {e}. Response: {stdout}", "SUB-21")

        # 6. VALIDATE_RESPONSE
        if not isinstance(response_data, dict) or "status" not in response_data:
            raise SubstrateError(f"Response is malformed or missing 'status' field. Response: {response_data}", "SUB-23")
        
        if response_data.get("request_hash") != request_hash:
            raise SubstrateError(
                f"Response request_hash mismatch. Expected {request_hash}, got {response_data.get('request_hash')}",
                "SUB-32"
            )

        if response_data["status"] == "error":
            # Propagate the error from the substrate.
            raise SubstrateError(
                response_data.get("message", "Unknown error from substrate."),
                response_data.get("error_code", "SUB-UNKNOWN")
            )
        
        # TODO: Add full schema validation for SubstrateSuccessResponse

        # 7. SUCCESS_END
        return response_data

    # ==========================================================================
    # Phase III Stubs
    # ==========================================================================

    def execute_batch(self, requests: list[SubstrateRequest]) -> list[SubstrateResponse]:
        # TODO: Implement multi-goal batching (Protocol v2.0).
        # This will require a long-lived process and a more complex IPC framing
        # protocol than simple newline-delimited JSON.
        raise NotImplementedError("Phase III: Batch execution is not yet implemented.")

    def execute_sandboxed(self, request: SubstrateRequest) -> SubstrateResponse:
        # TODO: Integrate with a deterministic VM execution sandbox (e.g., gVisor).
        # The sandbox configuration will become a hashed part of the request.
        raise NotImplementedError("Phase III: Sandboxed execution is not yet implemented.")

    def get_merkle_receipt(self, request: SubstrateRequest) -> Dict:
        # TODO: Evolve the protocol to return a Merkle root of the proof tree.
        raise NotImplementedError("Phase III: Merkle proof receipts are not yet implemented.")

    def get_zk_proof(self, request: SubstrateRequest) -> Dict:
        # TODO: Implement ZK-proving extensions for trustless verification.
        raise NotImplementedError("Phase III: ZK-proving extensions are not yet implemented.")