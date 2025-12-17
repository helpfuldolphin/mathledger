# backend/dashboard/load_test.py
from locust import HttpUser, task, between

class GovernanceDashboardUser(HttpUser):
    """
    Simulates a user browsing the governance history dashboard.
    """
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks

    @task(3)
    def get_last_100_runs_for_p4(self):
        """Common task: User looks at the most recent P4 data."""
        self.client.get("/api/v1/governance/history?limit=100&layers=P4")

    @task(2)
    def get_last_50_runs_all_layers(self):
        """User looks at the most recent data for all layers."""
        self.client.get("/api/v1/governance/history?limit=50")

    @task(1)
    def get_24_hour_window(self):
        """Less common: User queries a specific 24-hour time window for P3."""
        # In a real test, these times would be dynamic
        start_time = "2025-12-09T12:00:00Z"
        end_time = "2025-12-10T12:00:00Z"
        self.client.get(f"/api/v1/governance/history?layers=P3&start_time={start_time}&end_time={end_time}")

# To run this test:
# 1. Make sure the FastAPI app is running:
#    uvicorn backend.dashboard.api:app --host 0.0.0.0 --port 8000
# 2. Run locust:
#    locust -f backend/dashboard/load_test.py --host http://localhost:8000
# 3. Open your browser to http://localhost:8089 and start the test.
