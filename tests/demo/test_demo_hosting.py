"""
Demo Hosting Tests

Tests for version pinning, headers, and BASE_PATH support.

Run with:
    uv run pytest tests/demo/test_demo_hosting.py -v
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from demo.app import (
    app,
    DEMO_VERSION,
    DEMO_TAG,
    DEMO_COMMIT,
    BASE_PATH,
    get_html_content,
)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: Version Constants
# ---------------------------------------------------------------------------


class TestVersionConstants:
    """Verify version constants are correctly set."""

    def test_demo_version_format(self):
        """Version must be semver format."""
        assert re.match(r"^\d+\.\d+\.\d+$", DEMO_VERSION)

    def test_demo_version_is_0_2_0(self):
        """Version must be 0.2.0 (pinned)."""
        assert DEMO_VERSION == "0.2.0"

    def test_demo_tag_format(self):
        """Tag must start with 'v' and include version."""
        assert DEMO_TAG.startswith("v")
        assert "0.2.0" in DEMO_TAG

    def test_demo_tag_is_v0_2_0_demo_lock(self):
        """Tag must be v0.2.0-demo-lock (pinned)."""
        assert DEMO_TAG == "v0.2.0-demo-lock"

    def test_demo_commit_format(self):
        """Commit must be 40-char hex string."""
        assert len(DEMO_COMMIT) == 40
        assert re.match(r"^[0-9a-f]{40}$", DEMO_COMMIT)

    def test_demo_commit_is_pinned(self):
        """Commit must match pinned value."""
        assert DEMO_COMMIT == "27a94c8a58139cb10349f6418336c618f528cbab"


# ---------------------------------------------------------------------------
# Tests: Version Headers
# ---------------------------------------------------------------------------


class TestVersionHeaders:
    """Verify version headers are present on all responses."""

    def test_root_has_version_header(self, client):
        """Root endpoint includes X-MathLedger-Version header."""
        response = client.get("/")
        assert "X-MathLedger-Version" in response.headers
        assert response.headers["X-MathLedger-Version"] == f"v{DEMO_VERSION}"

    def test_root_has_commit_header(self, client):
        """Root endpoint includes X-MathLedger-Commit header."""
        response = client.get("/")
        assert "X-MathLedger-Commit" in response.headers
        assert response.headers["X-MathLedger-Commit"] == DEMO_COMMIT

    def test_root_has_base_path_header(self, client):
        """Root endpoint includes X-MathLedger-Base-Path header."""
        response = client.get("/")
        assert "X-MathLedger-Base-Path" in response.headers
        # Default is "/" or empty
        assert response.headers["X-MathLedger-Base-Path"] in ["/", ""]

    def test_health_has_version_headers(self, client):
        """Health endpoint includes version headers."""
        response = client.get("/health")
        assert "X-MathLedger-Version" in response.headers
        assert "X-MathLedger-Commit" in response.headers

    def test_healthz_has_version_headers(self, client):
        """Healthz endpoint includes version headers."""
        response = client.get("/healthz")
        assert "X-MathLedger-Version" in response.headers


# ---------------------------------------------------------------------------
# Tests: Health Endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    """Verify health endpoints work correctly."""

    def test_healthz_returns_ok(self, client):
        """Healthz endpoint returns 200 OK."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.text == "ok"

    def test_health_returns_json(self, client):
        """Health endpoint returns JSON with version info."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == DEMO_VERSION
        assert data["tag"] == DEMO_TAG
        assert data["commit"] == DEMO_COMMIT

    def test_health_includes_base_path(self, client):
        """Health endpoint includes base_path in response."""
        response = client.get("/health")
        data = response.json()
        assert "base_path" in data


# ---------------------------------------------------------------------------
# Tests: Version Banner
# ---------------------------------------------------------------------------


class TestVersionBanner:
    """Verify version banner is present in HTML."""

    def test_html_contains_version_banner(self):
        """HTML content includes version banner."""
        html = get_html_content()
        assert "GOVERNANCE DEMO (not capability)" in html

    def test_html_contains_version(self):
        """HTML content includes version number."""
        html = get_html_content()
        assert DEMO_VERSION in html

    def test_html_contains_tag(self):
        """HTML content includes tag."""
        html = get_html_content()
        assert DEMO_TAG in html

    def test_html_contains_commit_prefix(self):
        """HTML content includes commit prefix (first 12 chars)."""
        html = get_html_content()
        commit_prefix = DEMO_COMMIT[:12]
        assert commit_prefix in html

    def test_html_title_includes_version(self):
        """HTML title includes version."""
        html = get_html_content()
        assert f"<title>MathLedger Demo v{DEMO_VERSION}</title>" in html


# ---------------------------------------------------------------------------
# Tests: API Endpoints
# ---------------------------------------------------------------------------


class TestAPIEndpoints:
    """Verify API endpoints work correctly."""

    def test_scenarios_endpoint(self, client):
        """Scenarios endpoint returns scenario list."""
        response = client.get("/scenarios")
        assert response.status_code == 200
        data = response.json()
        assert "mv_only" in data
        assert "mixed_mv_adv" in data

    def test_ui_copy_endpoint(self, client):
        """UI copy endpoint returns canonical strings."""
        response = client.get("/ui_copy")
        assert response.status_code == 200
        data = response.json()
        assert "FRAMING_MAIN" in data
        assert "ABSTAINED_NOT_FAILURE" in data


# ---------------------------------------------------------------------------
# Tests: BASE_PATH in HTML
# ---------------------------------------------------------------------------


class TestBasePathInHTML:
    """Verify BASE_PATH is used correctly in HTML."""

    def test_html_contains_api_base(self):
        """HTML content defines API_BASE in JavaScript."""
        html = get_html_content()
        assert "const API_BASE = '" in html

    def test_fetch_calls_use_api_base(self):
        """JavaScript fetch calls use API_BASE."""
        html = get_html_content()
        # Should use template literal with API_BASE
        assert "${API_BASE}/uvil/" in html

    def test_doc_links_use_base_path(self):
        """Documentation links include base path."""
        html = get_html_content()
        # Doc links should include the base path variable
        assert "/docs/view/" in html


# ---------------------------------------------------------------------------
# Tests: Docker Labels (file-based)
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Verify Dockerfile contains correct labels."""

    def test_dockerfile_exists(self):
        """Dockerfile exists."""
        from pathlib import Path
        dockerfile = Path(__file__).parent.parent.parent / "Dockerfile"
        assert dockerfile.exists()

    def test_dockerfile_contains_version_label(self):
        """Dockerfile contains version label."""
        from pathlib import Path
        dockerfile = Path(__file__).parent.parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert 'LABEL version="0.2.0"' in content

    def test_dockerfile_contains_tag_label(self):
        """Dockerfile contains tag label."""
        from pathlib import Path
        dockerfile = Path(__file__).parent.parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert 'LABEL tag="v0.2.0-demo-lock"' in content

    def test_dockerfile_contains_commit_label(self):
        """Dockerfile contains commit label."""
        from pathlib import Path
        dockerfile = Path(__file__).parent.parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert 'LABEL commit="27a94c8a58139cb10349f6418336c618f528cbab"' in content

    def test_dockerfile_healthcheck(self):
        """Dockerfile contains healthcheck."""
        from pathlib import Path
        dockerfile = Path(__file__).parent.parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "HEALTHCHECK" in content
        assert "/healthz" in content
