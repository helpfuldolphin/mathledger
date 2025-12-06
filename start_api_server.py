#!/usr/bin/env python3
"""
Start the MathLedger API server.
This script sets up the environment and starts the FastAPI server.
"""

import os
import sys
import subprocess

from backend.security.runtime_env import MissingEnvironmentVariable, get_required_env


def _require_env(var_name: str) -> str:
    try:
        return get_required_env(var_name)
    except MissingEnvironmentVariable as exc:
        print(f"[fatal] {exc}")
        raise SystemExit(1) from exc


def main():
    # Validate mandatory security configuration before boot.
    required_vars = [
        "LEDGER_API_KEY",
        "CORS_ALLOWED_ORIGINS",
        "DATABASE_URL",
        "REDIS_URL",
    ]
    for var in required_vars:
        _require_env(var)

    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    print("Starting MathLedger API server...")
    print("Server will be available at: http://localhost:8010")
    print("API Documentation: http://localhost:8010/docs")
    print("UI Dashboard: http://localhost:8010/ui")
    print("\nPress Ctrl+C to stop the server")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "interface.api.app:app",
                "--port",
                "8010",
                "--reload",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
