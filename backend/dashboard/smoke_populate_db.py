# backend/dashboard/smoke_populate_db.py
import random
import os
from datetime import datetime, timedelta

# This allows the script to be run from the root directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.dashboard import storage

def generate_sample_events(count=10):
    """Generates a list of specified count of random governance events."""
    events = []
    layers = ["P3", "P4", "Substrate"]
    statuses = ["STABLE", "WARNING", "CRITICAL"]
    base_time = datetime.utcnow()

    for i in range(count):
        event = {
            "timestamp": (base_time - timedelta(minutes=i * 5)).isoformat() + "Z",
            "run_id": f"smoke-run-{i}",
            "layer": random.choice(layers),
            "governance_status": random.choice(statuses),
            "metrics": {
                "delta_p": round(random.uniform(0, 0.2), 4),
                "rsi": round(random.uniform(30, 90), 2),
                "omega": round(random.uniform(0.9, 1.0), 4),
                "divergence": round(random.uniform(0, 0.1), 4),
                "quarantine_ratio": round(random.uniform(0, 0.1), 4),
                "budget_invalid_percent": round(random.uniform(0, 1.5), 2),
            }
        }
        events.append(event)
    return events

def main():
    """
    Main function to initialize the DB, clear it, and insert 10 sample events.
    """
    print("Initializing database...")
    storage.initialize_db()
    print("Clearing existing data...")
    storage.clear_db_for_testing()
    print("Generating and inserting 10 sample events...")
    sample_events = generate_sample_events(10)
    for event in sample_events:
        storage.insert_event(event)
    print("Database populated successfully for smoke test.")

if __name__ == "__main__":
    main()
