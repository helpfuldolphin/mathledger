# tests/plugins/__init__.py
"""
Pytest plugins for MathLedger test infrastructure.

Available plugins:
- first_organism_telemetry_hook: Captures First Organism test metrics for Redis telemetry
"""

# Register plugins for auto-discovery
pytest_plugins = [
    "tests.plugins.first_organism_telemetry_hook",
]
