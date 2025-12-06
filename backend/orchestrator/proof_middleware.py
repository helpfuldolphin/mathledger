"""
Proof-of-Execution Ledger Middleware for FastAPI orchestrator.

Wraps API calls with cryptographic proof generation:
- Computes Merkle root of request inputs
- Canonicalizes payload via RFC 8785
- Signs with Ed25519
- Logs to append-only execution log

Reference: MathLedger Whitepaper §6.3 (Proof-of-Execution API).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.crypto.core import (
    rfc8785_canonicalize,
    sha256_hex_concat,
    merkle_root,
    ed25519_sign_b64,
    ed25519_generate_keypair,
)

logger = logging.getLogger(__name__)


# Global keypair for signing (in production, load from secure storage)
_SIGNING_KEYPAIR: Optional[tuple[bytes, bytes]] = None


def _ensure_keypair() -> tuple[bytes, bytes]:
    """Ensure Ed25519 keypair exists, generate if needed."""
    global _SIGNING_KEYPAIR
    if _SIGNING_KEYPAIR is None:
        _SIGNING_KEYPAIR = ed25519_generate_keypair()
    return _SIGNING_KEYPAIR


class ProofOfExecutionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that attaches proof-of-execution to all API requests.
    
    Features:
    - Merkle root computation from request inputs
    - RFC 8785 canonical JSON snapshot
    - Ed25519 signature
    - Append-only execution log
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_path: str = "artifacts/proof/execution_log.jsonl",
        enabled: bool = True,
    ):
        """
        Initialize proof middleware.
        
        Args:
            app: ASGI application
            log_path: Path to execution log file
            enabled: Whether to enable proof generation
        """
        super().__init__(app)
        self.log_path = log_path
        self.enabled = enabled
        self.request_count = 0
        
        # Ensure log directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Proof-of-Execution Ledger Middleware initialized (enabled=%s)", enabled)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with proof-of-execution tracking.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response with proof headers
        """
        if not self.enabled:
            return await call_next(request)
        
        # Increment request counter
        self.request_count += 1
        request_id = f"req_{self.request_count}_{datetime.utcnow().timestamp()}"
        
        # Extract request data
        request_data = await self._extract_request_data(request)
        
        # Compute Merkle root of inputs
        input_fields = [
            str(request.method),
            str(request.url.path),
            json.dumps(request_data.get("query_params", {}), sort_keys=True),
            json.dumps(request_data.get("body", {}), sort_keys=True),
        ]
        merkle = merkle_root(input_fields)
        
        # Create canonical payload snapshot
        payload_snapshot = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "query_params": request_data.get("query_params", {}),
            "body": request_data.get("body", {}),
            "merkle_root": merkle,
        }
        
        canonical_payload = rfc8785_canonicalize(payload_snapshot)
        
        # Sign with Ed25519
        private_key, public_key = _ensure_keypair()
        signature = ed25519_sign_b64(canonical_payload, private_key)
        
        # Execute request
        response = await call_next(request)
        
        # Add proof headers
        response.headers["X-Proof-Merkle-Root"] = merkle
        response.headers["X-Proof-Signature"] = signature
        response.headers["X-Proof-Request-ID"] = request_id
        
        # Log execution record
        execution_record = {
            **payload_snapshot,
            "signature": signature,
            "status_code": response.status_code,
        }
        
        self._append_to_log(execution_record)
        
        return response
    
    async def _extract_request_data(self, request: Request) -> Dict[str, Any]:
        """Extract relevant data from request."""
        data = {
            "query_params": dict(request.query_params),
        }
        
        # Try to read body if present
        try:
            body = await request.body()
            if body:
                try:
                    data["body"] = json.loads(body.decode())
                except Exception:
                    data["body"] = {"raw": body.decode()[:200]}
        except Exception:
            data["body"] = {}
        
        return data
    
    def _append_to_log(self, record: Dict[str, Any]) -> None:
        """Append execution record to log file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("Failed to write execution log: %s", e)


def get_execution_log_count(log_path: str = "artifacts/proof/execution_log.jsonl") -> int:
    """
    Get count of entries in execution log.
    
    Args:
        log_path: Path to execution log
        
    Returns:
        Number of log entries
    """
    try:
        with open(log_path, "r") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def emit_proof_middleware_passline(log_path: str = "artifacts/proof/execution_log.jsonl") -> None:
    """
    Emit pass-line for CI verification.
    
    Args:
        log_path: Path to execution log
    """
    count = get_execution_log_count(log_path)
    logger.info("Proof-of-Execution Ledger Active (logs=%s)", count)
