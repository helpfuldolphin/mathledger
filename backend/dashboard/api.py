# backend/dashboard/api.py
from fastapi import FastAPI, Query, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from . import storage

app = FastAPI(
    title="Governance History API",
    description="API for querying historical governance data.",
    version="0.1.0",
)

class Metrics(BaseModel):
    delta_p: float
    rsi: float
    omega: float
    divergence: float
    quarantine_ratio: float
    budget_invalid_percent: float

class GovernanceEvent(BaseModel):
    timestamp: str
    run_id: str
    layer: str
    governance_status: str
    metrics: Metrics

class PaginatedResponse(BaseModel):
    data: List[GovernanceEvent]
    pagination: Dict[str, Any]


@app.on_event("startup")
def on_startup():
    """Initialize the database when the application starts."""
    storage.initialize_db()

@app.get("/api/v1/governance/history", response_model=PaginatedResponse)
def get_governance_history(
    limit: int = Query(100, ge=1, le=1000, description="The maximum number of records to return."),
    start_time: Optional[str] = Query(None, description="ISO 8601 timestamp for the start of the time window."),
    end_time: Optional[str] = Query(None, description="ISO 8601 timestamp for the end of the time window."),
    layers: Optional[str] = Query(None, description="Comma-separated list of layer names to include."),
):
    """
    Retrieves a list of governance log entries, ordered from newest to oldest.
    """
    layer_list = layers.split(',') if layers else None

    try:
        events = storage.query_events(
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            layers=layer_list,
        )
        # In a real app, 'has_more' would require a more complex query (e.g., fetching limit+1 items)
        has_more = len(events) == limit
        return {
            "data": events,
            "pagination": {"limit": limit, "has_more": has_more}
        }
    except Exception as e:
        # Basic error handling
        raise HTTPException(status_code=500, detail=str(e))
