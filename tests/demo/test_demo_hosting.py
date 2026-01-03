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

    def test_base_path_header_pinned_to_root(self, client):
        """
        ROOT MOUNT ARCHITECTURE: X-MathLedger-Base-Path MUST be "/" (pinned).

        The Fly app serves at root. Cloudflare Worker rewrites /demo/* -> /*.
        If this test fails, the architecture contract is broken.
        """
        response = client.get("/health")
        base_path = response.headers.get("X-MathLedger-Base-Path", "")
        # Must be "/" or "" (both mean root mount)
        assert base_path in ["/", ""], (
            f"BASE_PATH drift detected! Expected '/' or '', got '{base_path}'. "
            f"ROOT MOUNT architecture requires app to serve at root."
        )

    def test_root_has_cache_control_header(self, client):
        """Root endpoint includes Cache-Control: no-store header."""
        response = client.get("/")
        assert "Cache-Control" in response.headers
        assert "no-store" in response.headers["Cache-Control"]

    def test_health_has_cache_control_header(self, client):
        """Health endpoint includes Cache-Control: no-store header."""
        response = client.get("/health")
        assert "Cache-Control" in response.headers
        assert "no-store" in response.headers["Cache-Control"]

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

    def test_html_contains_dynamic_api_base(self):
        """
        HTML content defines API_BASE dynamically using window.location.pathname.

        This is CRITICAL for Cloudflare Worker routing:
        - When served at /demo/, API_BASE must be '/demo'
        - When served at / (local dev), API_BASE must be ''
        """
        html = get_html_content()
        # Must use dynamic detection, not server-side injection
        assert "window.location.pathname.startsWith('/demo')" in html, (
            "API_BASE must be computed dynamically from window.location.pathname. "
            "Server-side injection breaks Cloudflare Worker routing."
        )

    def test_api_base_ternary_correct(self):
        """API_BASE ternary produces /demo when path starts with /demo."""
        html = get_html_content()
        # The exact pattern we expect
        expected = "const API_BASE = window.location.pathname.startsWith('/demo') ? '/demo' : '';"
        assert expected in html, (
            f"Expected API_BASE detection: {expected}"
        )

    def test_fetch_calls_use_api_base(self):
        """JavaScript fetch calls use API_BASE template literal."""
        html = get_html_content()
        # Should use template literal with API_BASE
        assert "${API_BASE}/uvil/" in html

    def test_doc_links_rewritten_by_js(self):
        """Documentation links are rewritten by JavaScript on load."""
        html = get_html_content()
        # Doc links should be plain /docs/view/* (JS rewrites them)
        assert 'href="/docs/view/' in html
        # JS rewriter should exist
        assert "querySelectorAll('a[href^=\"/docs/view/\"]')" in html

    def test_boundary_demo_uses_api_base(self):
        """
        Boundary demo fetch calls use API_BASE.

        This is the specific failure mode: boundary demo shows ERROR
        because fetch('/uvil/...') bypasses Cloudflare Worker.
        """
        html = get_html_content()
        # All three boundary demo API calls must use API_BASE
        assert "${API_BASE}/uvil/propose_partition" in html
        assert "${API_BASE}/uvil/commit_uvil" in html
        assert "${API_BASE}/uvil/run_verification" in html


# ---------------------------------------------------------------------------
# Tests: Server-side endpoint availability
# ---------------------------------------------------------------------------


class TestEndpointAvailability:
    """Verify UVIL endpoints respond correctly."""

    def test_propose_partition_endpoint(self, client):
        """POST /uvil/propose_partition responds 200."""
        response = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "test problem"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "proposal_id" in data

    def test_commit_uvil_endpoint(self, client):
        """POST /uvil/commit_uvil responds after propose."""
        # First propose
        propose_resp = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "test"}
        )
        proposal_id = propose_resp.json()["proposal_id"]

        # Then commit
        response = client.post(
            "/uvil/commit_uvil",
            json={
                "proposal_id": proposal_id,
                "edited_claims": [
                    {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "test"}
                ],
                "user_fingerprint": "test_user"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "committed_partition_id" in data

    def test_run_verification_endpoint(self, client):
        """POST /uvil/run_verification responds after commit."""
        # Propose
        propose_resp = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "test"}
        )
        proposal_id = propose_resp.json()["proposal_id"]

        # Commit
        commit_resp = client.post(
            "/uvil/commit_uvil",
            json={
                "proposal_id": proposal_id,
                "edited_claims": [
                    {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "test"}
                ],
                "user_fingerprint": "test_user"
            }
        )
        committed_id = commit_resp.json()["committed_partition_id"]

        # Verify
        response = client.post(
            "/uvil/run_verification",
            json={"committed_partition_id": committed_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert "outcome" in data
        assert data["outcome"] in ["VERIFIED", "REFUTED", "ABSTAINED"]


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


# ---------------------------------------------------------------------------
# Tests: fly.toml Configuration (ROOT MOUNT Architecture)
# ---------------------------------------------------------------------------


class TestFlyToml:
    """Verify fly.toml has correct ROOT MOUNT configuration."""

    def test_flytoml_exists(self):
        """fly.toml exists."""
        from pathlib import Path
        flytoml = Path(__file__).parent.parent.parent / "fly.toml"
        assert flytoml.exists()

    def test_flytoml_app_name(self):
        """fly.toml has authoritative app name."""
        from pathlib import Path
        flytoml = Path(__file__).parent.parent.parent / "fly.toml"
        content = flytoml.read_text()
        # Authoritative app name for v0.2.0
        assert 'app = "mathledger-demo-v0-2-0-helpfuldolphin"' in content, (
            "fly.toml app name mismatch. "
            "Authoritative name: mathledger-demo-v0-2-0-helpfuldolphin"
        )

    def test_flytoml_base_path_empty(self):
        """
        fly.toml BASE_PATH MUST be empty (ROOT MOUNT architecture).

        If this test fails, the architecture contract is broken.
        """
        from pathlib import Path
        flytoml = Path(__file__).parent.parent.parent / "fly.toml"
        content = flytoml.read_text()
        # BASE_PATH must be empty for ROOT MOUNT
        assert 'BASE_PATH = ""' in content, (
            "fly.toml BASE_PATH must be empty for ROOT MOUNT architecture. "
            "Cloudflare Worker handles /demo/* -> /* rewriting."
        )

    def test_flytoml_healthcheck_at_root(self):
        """fly.toml health checks must be at root paths (not /demo/*)."""
        from pathlib import Path
        flytoml = Path(__file__).parent.parent.parent / "fly.toml"
        content = flytoml.read_text()
        # Health check paths must NOT include /demo
        assert 'path = "/healthz"' in content
        assert 'path = "/health"' in content
        assert '/demo/healthz' not in content, (
            "fly.toml must NOT have /demo/* paths. ROOT MOUNT serves at /."
        )
