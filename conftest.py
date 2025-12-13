# Root conftest.py - MUST be at project root to load .env before test collection
# This file is loaded by pytest before any test modules are imported.

# Load environment variables FIRST, before any other imports
# This ensures DATABASE_URL and REDIS_URL are available when modules
# like backend.worker and backend.rfl.config are imported during collection.
from dotenv import load_dotenv
load_dotenv()

# Note: Fixtures from tests/conftest.py are automatically discovered by pytest
# since tests/ is a subdirectory. Do NOT use pytest_plugins here as it causes
# double-registration errors.
