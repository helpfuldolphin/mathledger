"""
from backend.repro.determinism import deterministic_unix_timestamp

_GLOBAL_SEED = 0

FastAPI middleware for integration performance monitoring.
"""

import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.integration.metrics import LatencyTracker


class IntegrationMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track latency and performance of FastAPI endpoints.
    
    Tracks:
    - Request/response latency
    - Endpoint-specific metrics
    - Success/failure rates
    - Integration point performance
    """

    def __init__(self, app: ASGIApp, tracker: LatencyTracker = None):
        super().__init__(app)
        self.tracker = tracker or LatencyTracker()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request latency and metadata."""
        start_time = deterministic_unix_timestamp(_GLOBAL_SEED)
        error = None
        success = True
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            success = 200 <= status_code < 400
            return response
        except Exception as e:
            error = str(e)
            success = False
            raise
        finally:
            end_time = deterministic_unix_timestamp(_GLOBAL_SEED)
            duration_ms = (end_time - start_time) * 1000

            from backend.integration.metrics import LatencyMeasurement
            measurement = LatencyMeasurement(
                operation=f"{request.method} {request.url.path}",
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error=error,
                metadata={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "client": request.client.host if request.client else None
                }
            )
            self.tracker.measurements.append(measurement)

    def get_stats(self):
        """Get statistics for all tracked requests."""
        return self.tracker.get_stats()

    def get_endpoint_stats(self, path: str):
        """Get statistics for a specific endpoint."""
        return self.tracker.get_stats(operation=path)


class PerformanceHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add performance timing headers to responses.
    
    Adds:
    - X-Response-Time: Response time in milliseconds
    - X-Integration-Version: Integration layer version
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add performance headers to response."""
        start_time = deterministic_unix_timestamp(_GLOBAL_SEED)

        response = await call_next(request)

        duration_ms = (deterministic_unix_timestamp(_GLOBAL_SEED) - start_time) * 1000

        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        response.headers["X-Integration-Version"] = "1.0.0"

        return response


def setup_integration_middleware(app, tracker: LatencyTracker = None):
    """
    Setup integration middleware on FastAPI app.
    
    Args:
        app: FastAPI application instance
        tracker: Optional LatencyTracker instance
        
    Returns:
        Configured middleware instance
    """
    monitoring = IntegrationMonitoringMiddleware(app, tracker)
    app.add_middleware(IntegrationMonitoringMiddleware, tracker=tracker)
    app.add_middleware(PerformanceHeadersMiddleware)

    return monitoring
