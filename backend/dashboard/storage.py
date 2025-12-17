# backend/dashboard/storage.py
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

DB_PATH = "governance_history.db"

def _get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """
    Initializes the database and creates the events table and indexes if they don't exist.
    """
    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS governance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            run_id TEXT NOT NULL,
            layer TEXT NOT NULL,
            governance_status TEXT NOT NULL,
            metrics TEXT NOT NULL,
            insertion_time TEXT NOT NULL
        )
    """)
    # Add indexes as defined in the Query Performance Plan
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_governance_events_layer_timestamp ON governance_events (layer, timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_governance_events_timestamp ON governance_events (timestamp DESC)")
    conn.commit()
    conn.close()

def insert_event(event: Dict[str, Any]):
    """
    Inserts a single governance event into the database.
    The 'metrics' dict is stored as a JSON string.
    """
    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO governance_events (timestamp, run_id, layer, governance_status, metrics, insertion_time)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            event["timestamp"],
            event["run_id"],
            event["layer"],
            event["governance_status"],
            json.dumps(event["metrics"]),
            datetime.utcnow().isoformat()
        )
    )
    conn.commit()
    conn.close()

def query_events(
    limit: int = 100,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    layers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Queries governance events with optional filtering and limits.
    """
    conn = _get_db_connection()
    cursor = conn.cursor()

    query = "SELECT timestamp, run_id, layer, governance_status, metrics FROM governance_events"
    conditions = []
    params = []

    if start_time:
        conditions.append("timestamp >= ?")
        params.append(start_time)

    if end_time:
        conditions.append("timestamp <= ?")
        params.append(end_time)

    if layers:
        placeholders = ','.join('?' for _ in layers)
        conditions.append(f"layer IN ({placeholders})")
        params.extend(layers)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    # Convert rows to dictionaries and parse the metrics JSON
    results = []
    for row in rows:
        row_dict = dict(row)
        row_dict["metrics"] = json.loads(row_dict["metrics"])
        results.append(row_dict)

    return results

def clear_db_for_testing():
    """Utility function to clear the database during tests."""
    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM governance_events")
    conn.commit()
    conn.close()
