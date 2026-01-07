"""
MathLedger UVIL v0.2 Demo

Front-facing demo with scenario selector, split panels, and self-explanation UI.
Serves the UVIL API and interactive HTML interface.

v0.2 Changes:
- UI self-explanation integrated (9 integration points)
- Abstention Preservation enforcement active
- Docs sidebar with inline documentation

Run: uv run python demo/app.py
Open: http://localhost:8000

Docker: docker build -t mathledger-demo . && docker run -p 8000:8000 mathledger-demo
Mounted: BASE_PATH=/demo docker run -p 8000:8000 -e BASE_PATH=/demo mathledger-demo
"""

import os
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pathlib import Path
import uvicorn

from backend.api.uvil import router as uvil_router

# ---------------------------------------------------------------------------
# VERSION PINNING (from releases.json - single source of truth)
# ---------------------------------------------------------------------------

def _load_version_from_releases() -> tuple[str, str, str]:
    """
    Load version info from releases.json at startup.

    This ensures the demo always reports the correct version from the
    canonical source (releases.json), eliminating version drift.

    Returns: (version, tag, commit)
    """
    import json
    releases_path = Path(__file__).parent.parent / "releases" / "releases.json"
    try:
        with open(releases_path, encoding="utf-8") as f:
            data = json.load(f)
        current = data.get("current_version", "v0.2.2")
        version_data = data.get("versions", {}).get(current, {})

        # Strip leading 'v' from version for consistency (v0.2.2 -> 0.2.2)
        version = current.lstrip("v")
        tag = version_data.get("tag", f"{current}-unknown")
        commit = version_data.get("commit", "unknown")

        return version, tag, commit
    except Exception as e:
        # Fallback if releases.json is missing or malformed
        # This should never happen in a proper deployment
        print(f"WARNING: Failed to load releases.json: {e}")
        return "0.2.2", "v0.2.2-link-integrity", "unknown"


# Load version at module startup from releases.json
DEMO_VERSION, DEMO_TAG, DEMO_COMMIT = _load_version_from_releases()

# Repository URL (single source of truth)
REPO_URL = "https://github.com/helpfuldolphin/mathledger"


# ---------------------------------------------------------------------------
# RELEASE PIN VALIDATION
# ---------------------------------------------------------------------------

def _load_releases_json() -> dict:
    """Load releases.json for version validation."""
    releases_path = Path(__file__).parent.parent / "releases" / "releases.json"
    try:
        import json
        with open(releases_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _validate_release_pin() -> dict:
    """
    Compare running version against releases.json current_version.

    Returns dict with:
      - is_stale: bool
      - expected: dict with version/tag/commit from releases.json
      - actual: dict with version/tag/commit from hardcoded constants
      - mismatch_fields: list of fields that don't match
    """
    releases = _load_releases_json()
    current_version = releases.get("current_version", "unknown")
    version_data = releases.get("versions", {}).get(current_version, {})

    # Expected values from releases.json
    expected = {
        "version": current_version.lstrip("v"),  # "v0.2.1" -> "0.2.1"
        "tag": version_data.get("tag", "unknown"),
        "commit": version_data.get("commit", "unknown"),
    }

    # Actual values from hardcoded constants
    actual = {
        "version": DEMO_VERSION,
        "tag": DEMO_TAG,
        "commit": DEMO_COMMIT,
    }

    # Check for mismatches
    mismatch_fields = []
    if expected["version"] != actual["version"]:
        mismatch_fields.append("version")
    if expected["tag"] != actual["tag"]:
        mismatch_fields.append("tag")
    if expected["commit"] != actual["commit"]:
        mismatch_fields.append("commit")

    return {
        "is_stale": len(mismatch_fields) > 0,
        "expected": expected,
        "actual": actual,
        "mismatch_fields": mismatch_fields,
    }


# Compute release pin status at module load (startup)
_RELEASE_PIN_STATUS = _validate_release_pin()

# ---------------------------------------------------------------------------
# BASE_PATH Configuration (for reverse proxy mounting)
# ---------------------------------------------------------------------------

BASE_PATH = os.environ.get("BASE_PATH", "").rstrip("/")
if BASE_PATH and not BASE_PATH.startswith("/"):
    BASE_PATH = "/" + BASE_PATH

# ---------------------------------------------------------------------------
# Version Header Middleware
# ---------------------------------------------------------------------------


class VersionHeaderMiddleware(BaseHTTPMiddleware):
    """Add version and cache headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Version headers
        response.headers["X-MathLedger-Version"] = f"v{DEMO_VERSION}"
        response.headers["X-MathLedger-Commit"] = DEMO_COMMIT
        response.headers["X-MathLedger-Base-Path"] = BASE_PATH or "/"
        # Cache control - no caching for demo (state changes frequently)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MathLedger Demo",
    description="UVIL v0 + Trust Classes v0: Epistemic Governance Demo",
    version=DEMO_VERSION,
    root_path=BASE_PATH,
)

# Add version header middleware
app.add_middleware(VersionHeaderMiddleware)

# Mount UVIL API (respects root_path)
app.include_router(uvil_router)

# Predefined scenarios (mirrors fixtures from harness)
SCENARIOS = {
    "mv_only": {
        "name": "MV Only",
        "description": "Single mechanically-validated claim",
        "task_text": "Prove that addition is commutative for natural numbers",
        "claims": [
            {"claim_text": "forall a b : Nat, a + b = b + a", "trust_class": "MV", "rationale": "Commutativity - mechanically checkable"}
        ]
    },
    "mixed_mv_adv": {
        "name": "Mixed MV + ADV",
        "description": "One authority-bearing, one exploration-only",
        "task_text": "Prove commutativity and consider generalizations",
        "claims": [
            {"claim_text": "forall a b : Nat, a + b = b + a", "trust_class": "MV", "rationale": "Commutativity - mechanically checkable"},
            {"claim_text": "This likely generalizes to arbitrary rings", "trust_class": "ADV", "rationale": "Speculation about generalization"}
        ]
    },
    "pa_only": {
        "name": "PA Only (User Attestation)",
        "description": "User-attested claim, not mechanically verified",
        "task_text": "Attest that requirement REQ-001 is satisfied",
        "claims": [
            {"claim_text": "Requirement REQ-001 is satisfied by implementation in module X", "trust_class": "PA", "rationale": "User attestation based on manual review"}
        ]
    },
    "adv_only": {
        "name": "ADV Only (Exploration)",
        "description": "All claims are advisory - nothing enters authority stream",
        "task_text": "Speculate about Navier-Stokes",
        "claims": [
            {"claim_text": "The Navier-Stokes equations probably have smooth solutions", "trust_class": "ADV", "rationale": "Pure speculation"},
            {"claim_text": "Turbulence might be related to strange attractors", "trust_class": "ADV", "rationale": "Another guess"}
        ]
    },
    "underdetermined": {
        "name": "Underdetermined (Open Problem)",
        "description": "System correctly stops - cannot verify open problems",
        "task_text": "Prove existence and smoothness of Navier-Stokes solutions in 3D",
        "claims": [
            {"claim_text": "Existence of weak solutions follows from energy estimates", "trust_class": "ADV", "rationale": "Standard but unverified"},
            {"claim_text": "Smoothness in 3D remains open (Millennium Prize)", "trust_class": "ADV", "rationale": "Open problem"},
            {"claim_text": "Partial regularity results exist (Caffarelli-Kohn-Nirenberg)", "trust_class": "ADV", "rationale": "Reference to literature"}
        ]
    }
}

# Canonical UI copy strings (for regression testing)
UI_COPY = {
    "FRAMING_MAIN": "The system does not decide what is true. It decides what is justified under a declared verification route.",
    "FRAMING_STOPS": "This demo will stop more often than you expect. It reports what it cannot verify.",
    "FRAMING_NOT_ALWAYS": "If you are looking for a system that always has an answer, this demo is not it.",
    "JUSTIFIED_EXPLAIN": "A claim is justified when it passes through attestation with a declared trust class. VERIFIED means a validator confirmed the claim. ABSTAINED means no validator could confirm or refute it. Neither outcome implies truth — only what the system could verify.",
    "ABSTAINED_NOT_FAILURE": "ABSTAINED is not failure.",
    "ABSTAINED_DETAIL": "It means: no verifier exists for this claim in v0, so the system refuses to assert correctness. The authority stream shows commitments and boundaries, not truth.",
    "ABSTAINED_FIRST_CLASS": "ABSTAINED is recorded in R_t. It is a first-class outcome, not a missing value. Downstream systems cannot ignore it.",
    "TRUST_CLASS_NOTE": "Trust class determines verification route, not correctness. Selecting MV does not make a claim verified — it declares that mechanical verification should be attempted. If no validator exists, the outcome is ABSTAINED.",
    "TRANSITION_TITLE": "Transition: Exploration → Authority",
    "TRANSITION_DETAIL": "The random proposal_id is discarded. The committed_id is derived from claim content. This ID is immutable — changing the claims would produce a different ID.",
    "ADV_TOOLTIP": "ADV (Advisory) claims are exploration-only. They do not enter R_t and cannot influence authority attestation. Use ADV for speculation, notes, or hypotheses.",
    "BOUNDARY_CONCLUSION": "Same claim text, different trust class → different outcome. Same trust class, different truth → VERIFIED vs REFUTED.",
    "OUTCOME_VERIFIED": "The validator confirmed the claim. VERIFIED means the arithmetic validator computed that the equation holds.",
    "OUTCOME_REFUTED": "The validator disproved the claim. REFUTED means the arithmetic validator computed that the equation does not hold.",
    "OUTCOME_ABSTAINED": "No verifier could confirm or refute this claim. ABSTAINED means no validator exists for this claim type in v0.",
}

# ---------------------------------------------------------------------------
# HTML Frontend Generator (BASE_PATH-aware)
# ---------------------------------------------------------------------------


def get_html_content() -> str:
    """Generate HTML content with BASE_PATH-aware URLs."""
    # Compute API base for JavaScript
    api_base = BASE_PATH if BASE_PATH else ""

    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MathLedger Demo v""" + DEMO_VERSION + """</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background: #f5f5f5;
            color: #1a1a1a;
            line-height: 1.5;
            font-size: 14px;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 1.5rem; display: flex; gap: 1.5rem; }
        .main-content { flex: 1; min-width: 0; }
        h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem; }
        .version-badge {
            font-size: 0.7rem;
            background: #e0e0e0;
            padding: 0.2rem 0.5rem;
            border-radius: 2px;
            vertical-align: middle;
            margin-left: 0.5rem;
        }

        /* Docs Sidebar */
        .docs-sidebar {
            width: 280px;
            flex-shrink: 0;
            background: #fff;
            border: 1px solid #ddd;
            padding: 1rem;
            max-height: calc(100vh - 3rem);
            overflow-y: auto;
            position: sticky;
            top: 1.5rem;
        }
        .docs-sidebar h3 {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #666;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #eee;
        }
        .docs-sidebar ul { list-style: none; }
        .docs-sidebar li { margin: 0.5rem 0; }
        .docs-sidebar a {
            color: #1565c0;
            text-decoration: none;
            font-size: 0.85rem;
        }
        .docs-sidebar a:hover { text-decoration: underline; }
        .docs-sidebar .doc-desc {
            font-size: 0.75rem;
            color: #888;
            margin-top: 0.25rem;
        }
        @media (max-width: 1100px) {
            .docs-sidebar { display: none; }
        }

        /* Framing box */
        .framing {
            background: #fff;
            border: 1px solid #ddd;
            padding: 1rem;
            margin-bottom: 1.5rem;
            font-size: 0.85rem;
        }
        .framing p { margin: 0.5rem 0; }
        .framing p:last-of-type { font-style: italic; margin-top: 0.75rem; }
        .framing details { margin-top: 0.75rem; font-size: 0.8rem; }
        .framing summary { cursor: pointer; color: #1565c0; }
        .framing details p { margin-top: 0.5rem; padding-left: 1rem; color: #666; font-style: normal; }

        /* Scenario selector */
        .scenario-bar {
            background: #fff;
            border: 1px solid #ddd;
            padding: 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }
        .scenario-bar label { font-weight: 600; }
        .scenario-bar select {
            padding: 0.5rem;
            font-size: 0.9rem;
            min-width: 200px;
        }
        .scenario-bar button {
            background: #2a2a2a;
            color: #fff;
            border: none;
            padding: 0.5rem 1rem;
            cursor: pointer;
        }
        .scenario-bar button:hover { background: #444; }
        .scenario-desc { font-size: 0.85rem; color: #666; }

        /* Split panel layout */
        .panels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 900px) {
            .panels { grid-template-columns: 1fr; }
        }

        .panel {
            background: #fff;
            border: 1px solid #ddd;
            padding: 1rem;
        }
        .panel-header {
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding-bottom: 0.75rem;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid #eee;
        }

        /* Exploration panel */
        .exploration-panel { border-left: 4px solid #888; }
        .exploration-panel .panel-header { color: #666; }

        /* Authority panel */
        .authority-panel { border-left: 4px solid #2a2a2a; }
        .authority-panel .panel-header { color: #1a1a1a; }

        /* Claims */
        .claim-item {
            background: #fafafa;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border: 1px solid #eee;
            font-size: 0.85rem;
        }
        .claim-text {
            font-family: monospace;
            margin-bottom: 0.5rem;
            word-break: break-word;
        }
        .claim-text input {
            width: 100%;
            padding: 0.4rem;
            font-family: monospace;
            font-size: 0.85rem;
            border: 1px solid #ccc;
        }
        .claim-meta { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
        .trust-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 2px;
        }
        .trust-fv { background: #e3f2fd; color: #1565c0; }
        .trust-mv { background: #e8f5e9; color: #2e7d32; }
        .trust-pa { background: #fff3e0; color: #ef6c00; }
        .trust-adv { background: #f5f5f5; color: #757575; }

        .trust-select {
            padding: 0.3rem;
            font-size: 0.8rem;
            border: 1px solid #ccc;
        }

        .excluded-badge {
            background: #ffebee;
            color: #c62828;
            padding: 0.2rem 0.5rem;
            font-size: 0.7rem;
            font-weight: 600;
            cursor: help;
        }
        .included-badge {
            background: #e8f5e9;
            color: #2e7d32;
            padding: 0.2rem 0.5rem;
            font-size: 0.7rem;
            font-weight: 600;
        }

        /* Trust class note */
        .trust-class-note {
            background: #f5f5f5;
            padding: 0.75rem;
            margin-top: 0.75rem;
            font-size: 0.8rem;
            border-left: 3px solid #888;
        }

        /* Transition note */
        .transition-note {
            background: #fff3e0;
            padding: 0.75rem;
            margin: 0.5rem 0;
            font-size: 0.85rem;
            border-left: 3px solid #ff9800;
        }

        /* Hash display */
        .hash-section {
            background: #f8f8f8;
            padding: 0.75rem;
            margin-top: 1rem;
            font-family: monospace;
            font-size: 0.7rem;
            word-break: break-all;
        }
        .hash-row { margin: 0.25rem 0; }
        .hash-label {
            font-weight: 600;
            color: #444;
            cursor: help;
            border-bottom: 1px dotted #888;
        }

        /* Outcome */
        .outcome-section {
            margin-top: 1rem;
            padding: 1rem;
            background: #fff3e0;
        }
        .outcome-section.verified { background: #e8f5e9; }
        .outcome-section.refuted { background: #ffebee; }
        .outcome-header {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        .outcome-explanation {
            font-size: 0.85rem;
            color: #666;
        }

        /* Outcome explanation detail boxes */
        .outcome-explanation-detail {
            margin-top: 1rem;
            padding: 1rem;
            font-size: 0.85rem;
        }
        .outcome-explanation-detail.abstained {
            background: #fffde7;
            border: 1px solid #ffc107;
        }
        .outcome-explanation-detail.verified {
            background: #e8f5e9;
            border: 1px solid #4caf50;
        }
        .outcome-explanation-detail.refuted {
            background: #ffebee;
            border: 1px solid #f44336;
        }
        .outcome-explanation-detail ul {
            margin: 0.5rem 0 0 1.5rem;
            padding: 0;
        }
        .outcome-explanation-detail li {
            margin: 0.25rem 0;
        }

        /* Actions */
        .actions {
            margin-top: 1rem;
            display: flex;
            gap: 0.5rem;
        }
        .actions button {
            background: #2a2a2a;
            color: #fff;
            border: none;
            padding: 0.6rem 1.2rem;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .actions button:hover { background: #444; }
        .actions button:disabled { background: #999; cursor: not-allowed; }
        .actions button.secondary {
            background: #fff;
            color: #2a2a2a;
            border: 1px solid #2a2a2a;
        }

        /* Status */
        .status { font-size: 0.8rem; color: #666; margin-top: 0.5rem; }
        .error { color: #c00; background: #fee; padding: 0.5rem; margin: 0.5rem 0; font-size: 0.85rem; border-left: 3px solid #c00; }
        .hidden { display: none; }

        /* Notes */
        .note { font-size: 0.8rem; color: #888; margin-top: 0.5rem; }
        .warning { font-size: 0.8rem; color: #c00; margin-top: 0.5rem; font-weight: 600; }

        /* Footer */
        .footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
            font-size: 0.8rem;
            color: #888;
        }

        /* Boundary Demo */
        .boundary-demo {
            background: #1a1a1a;
            color: #fff;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .boundary-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .boundary-title {
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .boundary-header button {
            background: #fff;
            color: #1a1a1a;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .boundary-header button:hover { background: #e0e0e0; }
        .boundary-header button:disabled { background: #666; color: #999; cursor: not-allowed; }
        .boundary-results { margin-top: 1rem; }
        .boundary-step {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid #333;
            opacity: 0.4;
            transition: opacity 0.3s;
        }
        .boundary-step.active { opacity: 1; }
        .boundary-step.done { opacity: 1; }
        .step-label {
            min-width: 140px;
            font-size: 0.85rem;
            color: #aaa;
        }
        .step-claim {
            font-family: monospace;
            background: #333;
            padding: 0.3rem 0.6rem;
            font-size: 0.85rem;
            color: #fff;
        }
        .step-arrow { color: #666; }
        .step-outcome {
            font-weight: 700;
            min-width: 100px;
            font-size: 0.9rem;
        }
        .step-outcome.verified { color: #4caf50; }
        .step-outcome.refuted { color: #f44336; }
        .step-outcome.abstained { color: #ff9800; }
        .step-reason {
            font-size: 0.75rem;
            color: #888;
            flex: 1;
        }
        .boundary-conclusion {
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid #444;
            font-size: 0.9rem;
            opacity: 0;
            transition: opacity 0.5s;
        }
        .boundary-conclusion.visible { opacity: 1; }
        .boundary-conclusion p { margin: 0; color: #ccc; }
        .boundary-breakdown {
            margin-top: 1rem;
            font-size: 0.8rem;
            color: #aaa;
        }
        .boundary-breakdown summary { cursor: pointer; }
        .boundary-breakdown ul { margin: 0.5rem 0 0 1rem; padding: 0; }
        .boundary-breakdown li { margin: 0.25rem 0; }

        /* What Gets Rejected Section */
        .rejection-demo {
            background: #fff;
            border: 1px solid #f44336;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        .rejection-demo summary {
            cursor: pointer;
            font-weight: 600;
            color: #c62828;
            font-size: 0.95rem;
        }
        .rejection-demo-content {
            margin-top: 1rem;
        }
        .rejection-demo-content p {
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 1rem;
        }
        .rejection-buttons {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }
        .rejection-buttons button {
            background: #fff;
            color: #c62828;
            border: 1px solid #c62828;
            padding: 0.5rem 1rem;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .rejection-buttons button:hover {
            background: #ffebee;
        }
        .rejection-buttons button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .rejection-result {
            margin-top: 0.75rem;
            padding: 0.75rem;
            font-family: monospace;
            font-size: 0.8rem;
            white-space: pre-wrap;
            word-break: break-all;
            background: #ffebee;
            border: 1px solid #f44336;
            color: #c62828;
        }
        .rejection-result .error-code {
            font-weight: 700;
            color: #b71c1c;
        }

        /* Evidence Pack */
        .evidence-pack-section {
            margin-top: 1.5rem;
            padding: 1rem;
            background: #f0f8ff;
            border: 1px solid #b3d9ff;
        }
        .evidence-pack-header {
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
            color: #1565c0;
        }
        .evidence-pack-section details {
            margin-bottom: 0.75rem;
            font-size: 0.8rem;
        }
        .evidence-pack-section summary {
            cursor: pointer;
            color: #1565c0;
        }
        .evidence-pack-section details > div {
            margin-top: 0.5rem;
            padding-left: 1rem;
            color: #666;
        }
        .evidence-pack-section details ul {
            margin: 0.5rem 0 0.5rem 1rem;
            padding: 0;
        }
        .evidence-pack-actions {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        .evidence-pack-actions button {
            background: #1565c0;
            color: #fff;
            border: none;
            padding: 0.6rem 1.2rem;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .evidence-pack-actions button:hover { background: #0d47a1; }
        .evidence-pack-actions button:disabled { background: #90caf9; cursor: not-allowed; }
        .evidence-pack-actions button.secondary {
            background: #fff;
            color: #1565c0;
            border: 1px solid #1565c0;
        }
        .evidence-pack-actions button.secondary:hover { background: #e3f2fd; }
        .replay-result {
            margin-top: 1rem;
            padding: 0.75rem;
            font-family: monospace;
            font-size: 0.85rem;
        }
        .replay-result.pass {
            background: #e8f5e9;
            border: 1px solid #4caf50;
            color: #2e7d32;
        }
        .replay-result.fail {
            background: #ffebee;
            border: 1px solid #f44336;
            color: #c62828;
        }
        .replay-diff {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: #fff;
            font-size: 0.75rem;
            word-break: break-all;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-content">
            <!-- Version Banner -->
            <div style="background:#1a1a1a; color:#fff; padding:0.5rem 1rem; margin-bottom:1rem; font-size:0.75rem; font-family:monospace; display:flex; justify-content:space-between; align-items:center;">
                <span>GOVERNANCE DEMO (not capability)</span>
                <span>
                    <a href="/v""" + DEMO_VERSION + """/" style="color:#90caf9; text-decoration:none; margin-right:1rem;">View v""" + DEMO_VERSION + """ Archive</a>
                    <span class="live-indicator" style="background:#4caf50; color:#fff; padding:0.1rem 0.4rem; font-size:0.65rem; margin-right:0.5rem;">LIVE</span>
                    v""" + DEMO_VERSION + """ | """ + DEMO_TAG + """ | """ + DEMO_COMMIT[:12] + """
                </span>
            </div>

            <h1>MathLedger Demo <span class="version-badge">v""" + DEMO_VERSION + """</span></h1>

            <!-- Integration Point 1: Framing Box with expandable detail -->
            <div class="framing" id="framing-box">
                <p><strong>""" + UI_COPY["FRAMING_MAIN"] + """</strong></p>
                <p>""" + UI_COPY["FRAMING_STOPS"] + """</p>
                <p>""" + UI_COPY["FRAMING_NOT_ALWAYS"] + """</p>
                <details>
                    <summary>What does "justified" mean?</summary>
                    <p>""" + UI_COPY["JUSTIFIED_EXPLAIN"] + """</p>
                </details>
            </div>

            <!-- Governance Explanation Link -->
            <div class="governance-link" style="margin: 1rem 0; padding: 0.75rem 1rem; background: #f5f5f5; border: 1px solid #ddd; border-left: 4px solid #1565c0; font-size: 0.9rem;">
                <strong>Governance Model:</strong>
                <a href="/v""" + DEMO_VERSION + """/docs/explanation/" style="color: #1565c0; margin-left: 0.5rem;">How This Demo Explains Itself</a>
                <span style="color: #666; margin-left: 0.5rem;">|</span>
                <span style="color: #666; margin-left: 0.5rem; font-size: 0.8rem;">Two-Lane Architecture + ACE Methodology + Epistemic Invariants</span>
                <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #666;">
                    <strong>Status: LOCKED (v0.2.13)</strong> |
                    <a href="https://arxiv.org/abs/2601.00816" target="_blank" style="color: #666;">arXiv:2601.00816</a>
                </div>
            </div>

            <!-- Boundary Demo with Integration Point 7: expandable breakdown -->
            <div class="boundary-demo" id="boundary-demo-section">
                <div class="boundary-header">
                    <div>
                        <span class="boundary-title">Same Claim, Different Authority</span>
                        <div style="font-size:0.75rem; color:#888; margin-top:0.25rem;">This is not a proof. It is an authority-routing demonstration.</div>
                    </div>
                    <button id="btn-boundary-demo" onclick="runBoundaryDemo()">Run Boundary Demo (≈8s)</button>
                </div>
                <div id="boundary-results" class="boundary-results hidden">
                    <div class="boundary-step" id="step-1">
                        <span class="step-label">1. ADV (Advisory)</span>
                        <code class="step-claim">"2 + 2 = 4"</code>
                        <span class="step-arrow">→</span>
                        <span class="step-outcome" id="outcome-adv">...</span>
                        <span class="step-reason" id="reason-adv"></span>
                    </div>
                    <div class="boundary-step" id="step-2">
                        <span class="step-label">2. PA (Attested)</span>
                        <code class="step-claim">"2 + 2 = 4"</code>
                        <span class="step-arrow">→</span>
                        <span class="step-outcome" id="outcome-pa">...</span>
                        <span class="step-reason" id="reason-pa"></span>
                    </div>
                    <div class="boundary-step" id="step-3">
                        <span class="step-label">3. MV (Validated)</span>
                        <code class="step-claim">"2 + 2 = 4"</code>
                        <span class="step-arrow">→</span>
                        <span class="step-outcome" id="outcome-mv">...</span>
                        <span class="step-reason" id="reason-mv"></span>
                    </div>
                    <div class="boundary-step" id="step-4">
                        <span class="step-label">4. MV (False)</span>
                        <code class="step-claim">"3 * 3 = 8"</code>
                        <span class="step-arrow">→</span>
                        <span class="step-outcome" id="outcome-refuted">...</span>
                        <span class="step-reason" id="reason-refuted"></span>
                    </div>
                    <div class="boundary-conclusion" id="boundary-conclusion">
                        <p><strong>""" + UI_COPY["BOUNDARY_CONCLUSION"] + """</strong></p>
                        <details class="boundary-breakdown">
                            <summary>What does this prove?</summary>
                            <div>
                                <p><strong>Outcome is determined by:</strong> trust class + validator + claim truth</p>
                                <ul>
                                    <li>ADV: Always excluded from R_t (no verification attempted)</li>
                                    <li>PA: Enters R_t but no validator exists → ABSTAINED</li>
                                    <li>MV + true arithmetic: Validator confirms → VERIFIED</li>
                                    <li>MV + false arithmetic: Validator refutes → REFUTED</li>
                                </ul>
                                <p>The system does not infer correctness from content. It applies declared verification routes.</p>
                            </div>
                        </details>
                    </div>
                </div>
            </div>

            <!-- What Gets Rejected - Collapsible -->
            <details class="rejection-demo">
                <summary>What Gets Rejected (3 live tests)</summary>
                <div class="rejection-demo-content">
                    <p>These buttons demonstrate governance violations that the system rejects with structured errors.
                       Each is a live API call — not mocked.</p>
                    <div class="rejection-buttons">
                        <button id="btn-double-commit" onclick="testDoubleCommit()">Double Commit Attempt</button>
                        <button id="btn-monotonicity" onclick="testMonotonicity()">Trust-Class Monotonicity Violation</button>
                        <button id="btn-silent-authority" onclick="testSilentAuthority()">Silent Authority Violation</button>
                    </div>
                    <div id="rejection-result" class="rejection-result hidden"></div>
                </div>
            </details>

            <!-- Scenario selector -->
            <div class="scenario-bar">
                <label>Scenario:</label>
                <select id="scenario-select" onchange="loadScenario()">
                    <option value="">-- Select a scenario --</option>
                    <option value="mv_only">MV Only (Mechanically Validated)</option>
                    <option value="mixed_mv_adv">Mixed MV + ADV</option>
                    <option value="pa_only">PA Only (User Attestation)</option>
                    <option value="adv_only">ADV Only (Exploration)</option>
                    <option value="underdetermined">Underdetermined (Open Problem)</option>
                    <option value="custom">Custom Input</option>
                </select>
                <span id="scenario-desc" class="scenario-desc"></span>
            </div>

            <!-- Split panels -->
            <div class="panels">
                <!-- Left: Exploration Stream -->
                <div class="panel exploration-panel">
                    <div class="panel-header">Exploration Stream (Not Authority)</div>

                    <div id="exploration-content">
                        <p class="note">Select a scenario or enter custom input.</p>
                    </div>

                    <div id="custom-input" class="hidden">
                        <label>Problem statement:</label>
                        <textarea id="problem-input" style="width:100%; min-height:60px; margin:0.5rem 0; padding:0.5rem; font-size:0.85rem;" placeholder="Enter a problem or claim..."></textarea>
                        <button onclick="proposePartition()">Generate Draft</button>
                    </div>

                    <div id="draft-section" class="hidden">
                        <p class="note" style="margin-bottom:0.5rem;">
                            <strong>Proposal ID:</strong> <code id="proposal-id"></code>
                        </p>
                        <p class="warning">This ID is exploration-only. It will NOT appear in any committed data.</p>

                        <div id="draft-claims"></div>

                        <!-- Integration Point 2: Trust class note -->
                        <div class="trust-class-note" id="trust-class-note">
                            <strong>""" + UI_COPY["TRUST_CLASS_NOTE"].split('.')[0] + """.</strong>
                            """ + '. '.join(UI_COPY["TRUST_CLASS_NOTE"].split('.')[1:]) + """
                        </div>

                        <div class="actions">
                            <button id="btn-commit" onclick="commitUVIL()">Commit to Authority</button>
                        </div>
                    </div>
                </div>

                <!-- Right: Authority Stream -->
                <div class="panel authority-panel">
                    <div class="panel-header">Authority Stream (Bound)</div>

                    <div id="authority-empty">
                        <p class="note">Nothing committed yet. Authority stream is empty.</p>
                    </div>

                    <div id="authority-content" class="hidden">
                        <!-- Integration Point 3: Transition note -->
                        <div class="transition-note" id="transition-note">
                            <strong>""" + UI_COPY["TRANSITION_TITLE"] + """</strong><br>
                            """ + UI_COPY["TRANSITION_DETAIL"] + """
                        </div>

                        <p class="note" style="margin-bottom:0.5rem;">
                            <strong>Committed ID:</strong> <code id="committed-id"></code>
                        </p>
                        <p class="note">This ID is derived from content. It is immutable.</p>

                        <div id="authority-claims"></div>

                        <!-- Integration Point 6: Hash tooltips -->
                        <div class="hash-section">
                            <div class="hash-row">
                                <span class="hash-label" title="U_t is the Merkle root of all user interaction events. It commits to what the user saw and did.">U_t (UI):</span>
                                <span id="hash-ut">-</span>
                            </div>
                            <div class="hash-row">
                                <span class="hash-label" title="R_t is the Merkle root of reasoning artifacts. ADV claims are excluded. Only MV, PA, FV claims enter this root.">R_t (Reasoning):</span>
                                <span id="hash-rt">-</span>
                            </div>
                            <div class="hash-row">
                                <span class="hash-label" title="H_t = SHA256(R_t || U_t). This is the composite epoch root that binds UI and reasoning together.">H_t (Composite):</span>
                                <span id="hash-ht">-</span>
                            </div>
                            <p class="note" style="margin-top:0.5rem; font-size:0.7rem;">
                                R_t commits to authority-bearing artifacts only (no ADV). In v0, these are demo placeholders marked <code>v0_mock:true</code> — not proofs.
                            </p>
                        </div>

                        <div class="actions">
                            <button id="btn-verify" onclick="runVerification()">Run Verification</button>
                        </div>
                    </div>

                    <div id="result-section" class="hidden">
                        <!-- Integration Point 4: Outcome-specific explanations -->
                        <div class="outcome-section" id="outcome-section">
                            <div class="outcome-header" id="outcome-text">ABSTAINED</div>
                            <div class="outcome-explanation" id="outcome-explanation"></div>
                        </div>

                        <!-- Dynamic outcome explanation (populated by JS) -->
                        <div id="outcome-detail-container"></div>

                        <div id="authority-breakdown" style="margin-top:1rem;"></div>

                        <div class="hash-section">
                            <div class="hash-row">
                                <span class="hash-label" title="U_t is the Merkle root of all user interaction events.">Final U_t:</span>
                                <span id="final-ut">-</span>
                            </div>
                            <div class="hash-row">
                                <span class="hash-label" title="R_t is the Merkle root of reasoning artifacts (excludes ADV).">Final R_t:</span>
                                <span id="final-rt">-</span>
                            </div>
                            <div class="hash-row">
                                <span class="hash-label" title="H_t = SHA256(R_t || U_t).">Final H_t:</span>
                                <span id="final-ht">-</span>
                            </div>
                        </div>

                        <!-- Integration Point 5: Evidence Pack with expandable explanation -->
                        <div class="evidence-pack-section">
                            <div class="evidence-pack-header">Audit Verification</div>
                            <details>
                                <summary>What is an evidence pack?</summary>
                                <div>
                                    <p>The evidence pack is a self-contained JSON file containing:</p>
                                    <ul>
                                        <li><code>uvil_events</code> — User interaction events (hashed into U_t)</li>
                                        <li><code>reasoning_artifacts</code> — Claims with outcomes (hashed into R_t)</li>
                                        <li><code>u_t</code>, <code>r_t</code>, <code>h_t</code> — Attestation roots</li>
                                        <li><code>replay_instructions</code> — Commands to reproduce hashes</li>
                                    </ul>
                                    <p>Anyone with the evidence pack can recompute the hashes independently. If replay produces different hashes, the pack is invalid.</p>
                                </div>
                            </details>
                            <p class="note" style="margin-bottom:0.75rem;">
                                Download the evidence pack to independently verify attestation hashes.
                                Replay verification recomputes U_t, R_t, H_t locally with no external calls.
                            </p>
                            <div class="evidence-pack-actions">
                                <button id="btn-download-evidence" onclick="downloadEvidencePack()">Download Evidence Pack</button>
                                <button id="btn-replay-verify" class="secondary" onclick="replayVerify()">Replay & Verify</button>
                                <a href="/v""" + DEMO_VERSION + """/evidence-pack/verify/" target="_blank" class="secondary" style="display:inline-block; padding:0.6rem 1.2rem; background:#fff; color:#1565c0; border:1px solid #1565c0; text-decoration:none; font-size:0.85rem;">Open Auditor Tool</a>
                            </div>
                            <div id="replay-result-display" class="hidden"></div>
                        </div>

                        <div class="actions">
                            <button class="secondary" onclick="reset()">Start Over</button>
                        </div>
                    </div>
                </div>
            </div>

            <div id="error-display" class="error hidden"></div>
            <div id="status-display" class="status"></div>

            <div class="footer">
                v""" + DEMO_VERSION + """ (""" + DEMO_TAG + """) | Governance substrate only | MV arithmetic validator only | <a href="/docs/view/V0_LOCK.md">Scope Lock</a>
            </div>
        </div>

        <!-- Docs Sidebar - links rewritten by JS to include API_BASE -->
        <div class="docs-sidebar">
            <h3>Documentation</h3>
            <ul>
                <li>
                    <a href="/docs/view/HOW_THE_DEMO_EXPLAINS_ITSELF.md" target="_blank">How the Demo Explains Itself</a>
                    <div class="doc-desc">UI behavior and outcomes explained</div>
                </li>
                <li>
                    <a href="/docs/view/HOW_TO_APPROACH_THIS_DEMO.md" target="_blank">How to Approach This Demo</a>
                    <div class="doc-desc">Framing and expectations</div>
                </li>
                <li>
                    <a href="/docs/view/V0_LOCK.md" target="_blank">v0 Scope Lock</a>
                    <div class="doc-desc">What is and isn't in scope</div>
                </li>
                <li>
                    <a href="/docs/view/V0_SYSTEM_BOUNDARY_MEMO.md" target="_blank">System Boundary Memo</a>
                    <div class="doc-desc">Formal claims and non-claims</div>
                </li>
                <li>
                    <a href="/docs/view/invariants_status.md" target="_blank">Invariants Status</a>
                    <div class="doc-desc">Tier A/B/C classification</div>
                </li>
            </ul>
            <h3 style="margin-top:1.5rem;">Trust Classes</h3>
            <ul class="trust-class-tooltips">
                <li style="font-size:0.8rem; color:#666; margin-bottom:0.5rem;" title="Formally Verified: Requires machine-checkable proof. Not implemented in v0 — returns ABSTAINED.">
                    <strong style="color:#1565c0;">FV</strong>: Formal proof<br>
                    <span style="font-size:0.7rem; color:#999;">→ ABSTAINED in v0 (no prover)</span>
                </li>
                <li style="font-size:0.8rem; color:#666; margin-bottom:0.5rem;" title="Mechanically Validated: Checked by arithmetic validator. Only handles 'a op b = c' patterns.">
                    <strong style="color:#2e7d32;">MV</strong>: Mechanical validation<br>
                    <span style="font-size:0.7rem; color:#999;">→ Arithmetic only in v0</span>
                </li>
                <li style="font-size:0.8rem; color:#666; margin-bottom:0.5rem;" title="Procedurally Attested: User attests correctness. Authority-bearing but not mechanically verified.">
                    <strong style="color:#f57c00;">PA</strong>: User attestation<br>
                    <span style="font-size:0.7rem; color:#999;">→ ABSTAINED (no verifier)</span>
                </li>
                <li style="font-size:0.8rem; color:#c62828; margin-bottom:0.5rem;" title="Advisory: Exploration-only claims. EXCLUDED FROM R_t. Cannot influence authority attestation.">
                    <strong style="color:#c62828;">ADV</strong>: Advisory<br>
                    <span style="font-size:0.7rem; color:#c62828; font-weight:600;">→ EXCLUDED FROM R_t</span>
                </li>
            </ul>
        </div>
    </div>

    <script>
        // Dynamic API base path detection for Cloudflare Worker routing
        // When served at /demo/, API calls must go to /demo/uvil/*
        // When served at / (local dev), API calls go to /uvil/*
        const API_BASE = window.location.pathname.startsWith('/demo') ? '/demo' : '';

        // Rewrite doc links on page load to match detected base path
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('a[href^="/docs/view/"]').forEach(function(link) {
                link.href = API_BASE + link.getAttribute('href');
            });
            // Also fix the footer scope lock link
            document.querySelectorAll('.footer a[href*="/docs/view/"]').forEach(function(link) {
                if (!link.href.startsWith(window.location.origin + API_BASE)) {
                    link.href = API_BASE + '/docs/view/V0_LOCK.md';
                }
            });
            // Fix archive links (dynamic version from server)
            var archiveVersion = 'v""" + DEMO_VERSION + """';
            document.querySelectorAll('a[href^="/' + archiveVersion + '/"]').forEach(function(link) {
                link.href = API_BASE + link.getAttribute('href');
            });
        });

        const SCENARIOS = """ + str(SCENARIOS).replace("'", '"') + """;

        // Integration Point 9: Governance error templates
        const ERROR_TEMPLATES = {
            "TRUST_CLASS_MONOTONICITY_VIOLATION": "Cannot change trust class of committed claim. To use a different trust class, create a new claim (which produces a new claim_id).",
            "SILENT_AUTHORITY_VIOLATION": "Cannot produce evidence pack: attestation verification failed. Claimed H_t does not match computed H_t. This indicates tampering or corruption.",
            "DOUBLE_COMMIT": "This partition has already been committed. Duplicate commit rejected.",
            "ABSTENTION_PRESERVATION_VIOLATION": "Invalid or missing validation_outcome. ABSTAINED must be explicit, not null or missing."
        };

        // Integration Point 4: Outcome explanation templates
        const OUTCOME_EXPLANATIONS = {
            "ABSTAINED": `
                <div class="outcome-explanation-detail abstained">
                    <strong>Why ABSTAINED?</strong>
                    <ul>
                        <li>v0 has no formal verifier (FV claims always abstain)</li>
                        <li>MV validator handles arithmetic only (<code>a op b = c</code>)</li>
                        <li>PA claims require human attestation, not mechanical verification</li>
                    </ul>
                    <p style="margin-top:0.75rem; margin-bottom:0;">
                        <strong>""" + UI_COPY["ABSTAINED_FIRST_CLASS"] + """</strong>
                    </p>
                </div>
            `,
            "VERIFIED": `
                <div class="outcome-explanation-detail verified">
                    <strong>VERIFIED by arithmetic validator</strong>
                    <p style="margin:0.5rem 0 0 0;">
                        The MV validator parsed this claim as <code>a op b = c</code> and computed
                        that the equation holds. This is the only validator in v0.
                    </p>
                    <p style="margin-top:0.5rem; margin-bottom:0; color:#666;">
                        VERIFIED means the validator returned true. It does not mean the claim is universally correct.
                    </p>
                </div>
            `,
            "REFUTED": `
                <div class="outcome-explanation-detail refuted">
                    <strong>REFUTED by arithmetic validator</strong>
                    <p style="margin:0.5rem 0 0 0;">
                        The MV validator parsed this claim as <code>a op b = c</code> and computed
                        that the equation does not hold.
                    </p>
                    <p style="margin-top:0.5rem; margin-bottom:0; color:#666;">
                        REFUTED means the validator returned false. The claim is recorded with this outcome.
                    </p>
                </div>
            `
        };

        let currentProposalId = null;
        let currentDraftPayload = null;  // v0.2.10: Self-sufficient payload
        let currentCommittedId = null;
        let editedClaims = [];
        let currentScenario = null;

        function loadScenario() {
            const select = document.getElementById('scenario-select');
            const scenarioKey = select.value;
            const descEl = document.getElementById('scenario-desc');

            reset();

            if (!scenarioKey) {
                descEl.textContent = '';
                return;
            }

            if (scenarioKey === 'custom') {
                descEl.textContent = 'Enter your own problem and claims';
                document.getElementById('custom-input').classList.remove('hidden');
                document.getElementById('exploration-content').classList.add('hidden');
                return;
            }

            const scenario = SCENARIOS[scenarioKey];
            if (!scenario) return;

            currentScenario = scenario;
            descEl.textContent = scenario.description;

            // Auto-propose with scenario
            proposeWithScenario(scenario);
        }

        async function proposeWithScenario(scenario) {
            setStatus('Generating proposal...');

            try {
                const response = await fetch(`${API_BASE}/uvil/propose_partition`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ problem_statement: scenario.task_text })
                });

                if (!response.ok) throw new Error('Request failed');

                const data = await response.json();
                currentProposalId = data.proposal_id;
                // v0.2.10: Store draft_payload for self-sufficient commit
                currentDraftPayload = data.draft_payload;

                // Use scenario claims instead of generated ones
                editedClaims = scenario.claims.map(c => ({...c}));

                renderDraftClaims();
                document.getElementById('proposal-id').textContent = currentProposalId;
                document.getElementById('exploration-content').classList.add('hidden');
                document.getElementById('draft-section').classList.remove('hidden');
                setStatus('');

            } catch (e) {
                showError(e.message);
            }
        }

        async function proposePartition() {
            const input = document.getElementById('problem-input').value.trim();
            if (!input) {
                showError('Please enter a problem statement.');
                return;
            }

            setStatus('Generating proposal...');

            try {
                const response = await fetch(`${API_BASE}/uvil/propose_partition`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ problem_statement: input })
                });

                if (!response.ok) throw new Error('Request failed');

                const data = await response.json();
                currentProposalId = data.proposal_id;
                // v0.2.10: Store draft_payload for self-sufficient commit
                currentDraftPayload = data.draft_payload;
                editedClaims = data.claims.map(c => ({
                    claim_text: c.claim_text,
                    trust_class: c.suggested_trust_class,
                    rationale: c.rationale || ''
                }));

                renderDraftClaims();
                document.getElementById('proposal-id').textContent = currentProposalId;
                document.getElementById('custom-input').classList.add('hidden');
                document.getElementById('draft-section').classList.remove('hidden');
                setStatus('');

            } catch (e) {
                showError(e.message);
            }
        }

        // Integration Point 2 & 8: Trust class tooltips and ADV badge tooltip
        function renderDraftClaims() {
            const container = document.getElementById('draft-claims');
            container.innerHTML = '';

            const TRUST_TOOLTIP = "FV: Claim requires formal proof (returns ABSTAINED in v0)\\nMV: Claim is mechanically checkable (arithmetic only in v0)\\nPA: User attests correctness (returns ABSTAINED)\\nADV: Exploration only, excluded from R_t";
            const ADV_TOOLTIP = '""" + UI_COPY["ADV_TOOLTIP"].replace("'", "\\'") + """';

            editedClaims.forEach((claim, idx) => {
                const div = document.createElement('div');
                div.className = 'claim-item';
                div.innerHTML = `
                    <div class="claim-text">
                        <input type="text" value="${escapeHtml(claim.claim_text)}"
                               onchange="editedClaims[${idx}].claim_text = this.value">
                    </div>
                    <div class="claim-meta">
                        <select class="trust-select" title="${TRUST_TOOLTIP}" onchange="editedClaims[${idx}].trust_class = this.value; renderDraftClaims();">
                            <option value="ADV" ${claim.trust_class === 'ADV' ? 'selected' : ''}>ADV</option>
                            <option value="PA" ${claim.trust_class === 'PA' ? 'selected' : ''}>PA</option>
                            <option value="MV" ${claim.trust_class === 'MV' ? 'selected' : ''}>MV</option>
                            <option value="FV" ${claim.trust_class === 'FV' ? 'selected' : ''}>FV</option>
                        </select>
                        <span class="trust-badge trust-${claim.trust_class.toLowerCase()}">${claim.trust_class}</span>
                        ${claim.trust_class === 'ADV'
                            ? `<span class="excluded-badge" title="${ADV_TOOLTIP}">EXCLUDED FROM R_t</span>`
                            : '<span class="included-badge">ENTERS R_t</span>'}
                    </div>
                `;
                container.appendChild(div);
            });
        }

        async function commitUVIL() {
            if ((!currentProposalId && !currentDraftPayload) || editedClaims.length === 0) return;

            document.getElementById('btn-commit').disabled = true;
            setStatus('Committing...');

            try {
                // v0.2.10: Send draft_payload for self-sufficient commit
                const requestBody = {
                    edited_claims: editedClaims,
                    user_fingerprint: 'demo_user'
                };
                // Prefer draft_payload (self-sufficient), fall back to proposal_id
                if (currentDraftPayload) {
                    requestBody.draft_payload = currentDraftPayload;
                } else {
                    requestBody.proposal_id = currentProposalId;
                }

                const response = await fetch(`${API_BASE}/uvil/commit_uvil`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    const err = await response.json();
                    // Integration Point 9: Use error templates
                    const errorCode = err.detail?.error_code || err.error_code || '';
                    const template = ERROR_TEMPLATES[errorCode];
                    // v0.2.10: System-responsible error handling
                    if (errorCode === 'PROPOSAL_STATE_LOST') {
                        throw new Error('Server lost proposal state. This is a demo reliability bug. Refresh and retry.');
                    }
                    throw new Error(template || err.detail?.message || err.detail || 'Commit failed');
                }

                const data = await response.json();
                currentCommittedId = data.committed_partition_id;

                // Update authority panel
                document.getElementById('committed-id').textContent = currentCommittedId;
                document.getElementById('hash-ut').textContent = data.u_t;
                document.getElementById('hash-rt').textContent = data.r_t;
                document.getElementById('hash-ht').textContent = data.h_t;

                renderAuthorityClaims();

                document.getElementById('authority-empty').classList.add('hidden');
                document.getElementById('authority-content').classList.remove('hidden');
                setStatus('Committed. Ready to verify.');

            } catch (e) {
                showError(e.message);
                document.getElementById('btn-commit').disabled = false;
            }
        }

        // Integration Point 8: ADV badge tooltip in authority claims
        function renderAuthorityClaims() {
            const container = document.getElementById('authority-claims');
            container.innerHTML = '';

            const ADV_TOOLTIP = '""" + UI_COPY["ADV_TOOLTIP"].replace("'", "\\'") + """';

            editedClaims.forEach(claim => {
                const isAuthority = claim.trust_class !== 'ADV';
                const div = document.createElement('div');
                div.className = 'claim-item';
                div.style.opacity = isAuthority ? '1' : '0.5';
                div.innerHTML = `
                    <div class="claim-text">${escapeHtml(claim.claim_text)}</div>
                    <div class="claim-meta">
                        <span class="trust-badge trust-${claim.trust_class.toLowerCase()}">${claim.trust_class}</span>
                        ${isAuthority
                            ? '<span class="included-badge">IN AUTHORITY STREAM</span>'
                            : `<span class="excluded-badge" title="${ADV_TOOLTIP}">EXCLUDED (ADV)</span>`}
                    </div>
                `;
                container.appendChild(div);
            });
        }

        async function runVerification() {
            if (!currentCommittedId) return;

            document.getElementById('btn-verify').disabled = true;
            setStatus('Running verification...');

            try {
                const response = await fetch(`${API_BASE}/uvil/run_verification`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ committed_partition_id: currentCommittedId })
                });

                if (!response.ok) throw new Error('Verification failed');

                const data = await response.json();

                // Show result with appropriate styling
                const outcomeSection = document.getElementById('outcome-section');
                outcomeSection.className = 'outcome-section ' + data.outcome.toLowerCase();

                document.getElementById('outcome-text').textContent = data.outcome;
                document.getElementById('outcome-explanation').textContent =
                    data.authority_basis.explanation;

                // Integration Point 4: Show outcome-specific explanation
                const detailContainer = document.getElementById('outcome-detail-container');
                detailContainer.innerHTML = OUTCOME_EXPLANATIONS[data.outcome] || '';

                // Breakdown
                const breakdown = document.getElementById('authority-breakdown');
                breakdown.innerHTML = `
                    <p><strong>Authority Basis:</strong></p>
                    <ul style="margin:0.5rem 0; padding-left:1.5rem; font-size:0.85rem;">
                        <li>FV claims: ${data.authority_basis.fv_count}</li>
                        <li>MV claims: ${data.authority_basis.mv_count}</li>
                        <li>PA claims: ${data.authority_basis.pa_count}</li>
                        <li>ADV claims (excluded): ${data.authority_basis.adv_count}</li>
                    </ul>
                    <p class="note">Mechanically verified: ${data.authority_basis.mechanically_verified ? 'Yes' : 'No (v0 has no verifier)'}</p>
                `;

                // Final hashes
                document.getElementById('final-ut').textContent = data.attestation.u_t;
                document.getElementById('final-rt').textContent = data.attestation.r_t;
                document.getElementById('final-ht').textContent = data.attestation.h_t;

                document.getElementById('authority-content').classList.add('hidden');
                document.getElementById('result-section').classList.remove('hidden');
                setStatus('');

            } catch (e) {
                showError(e.message);
                document.getElementById('btn-verify').disabled = false;
            }
        }

        function reset() {
            currentProposalId = null;
            currentDraftPayload = null;  // v0.2.10
            currentCommittedId = null;
            editedClaims = [];
            currentScenario = null;
            currentEvidencePack = null;

            document.getElementById('exploration-content').classList.remove('hidden');
            document.getElementById('custom-input').classList.add('hidden');
            document.getElementById('draft-section').classList.add('hidden');
            document.getElementById('draft-claims').innerHTML = '';

            document.getElementById('authority-empty').classList.remove('hidden');
            document.getElementById('authority-content').classList.add('hidden');
            document.getElementById('result-section').classList.add('hidden');
            document.getElementById('outcome-detail-container').innerHTML = '';

            document.getElementById('btn-commit').disabled = false;
            document.getElementById('btn-verify').disabled = false;

            document.getElementById('problem-input').value = '';
            document.getElementById('replay-result-display').classList.add('hidden');
            hideError();
            setStatus('');
        }

        function setStatus(msg) {
            document.getElementById('status-display').textContent = msg;
        }

        function showError(msg) {
            const el = document.getElementById('error-display');
            el.textContent = msg;
            el.classList.remove('hidden');
        }

        function hideError() {
            document.getElementById('error-display').classList.add('hidden');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Evidence Pack Download
        let currentEvidencePack = null;

        async function downloadEvidencePack() {
            if (!currentCommittedId) {
                showError('No committed partition to download.');
                return;
            }

            const btn = document.getElementById('btn-download-evidence');
            btn.disabled = true;
            btn.textContent = 'Downloading...';

            try {
                const response = await fetch(`${API_BASE}/uvil/evidence_pack/${currentCommittedId}`);
                if (!response.ok) {
                    const err = await response.json();
                    const errorCode = err.error_code || '';
                    const template = ERROR_TEMPLATES[errorCode];
                    throw new Error(template || err.detail || 'Failed to fetch evidence pack');
                }

                const data = await response.json();
                currentEvidencePack = data;

                // Create download
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `evidence_pack_${currentCommittedId.substring(0, 8)}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                btn.textContent = 'Downloaded!';
                setTimeout(() => {
                    btn.disabled = false;
                    btn.textContent = 'Download Evidence Pack';
                }, 2000);

            } catch (e) {
                showError(e.message);
                btn.disabled = false;
                btn.textContent = 'Download Evidence Pack';
            }
        }

        async function replayVerify() {
            if (!currentCommittedId) {
                showError('No committed partition to verify.');
                return;
            }

            const btn = document.getElementById('btn-replay-verify');
            btn.disabled = true;
            btn.textContent = 'Verifying...';

            const resultDisplay = document.getElementById('replay-result-display');
            resultDisplay.classList.add('hidden');
            resultDisplay.className = 'hidden';

            try {
                // First get the evidence pack if we don't have it
                if (!currentEvidencePack || currentEvidencePack.committed_partition_snapshot.committed_partition_id !== currentCommittedId) {
                    const packResponse = await fetch(`${API_BASE}/uvil/evidence_pack/${currentCommittedId}`);
                    if (!packResponse.ok) throw new Error('Failed to fetch evidence pack');
                    currentEvidencePack = await packResponse.json();
                }

                // Now replay verify
                const response = await fetch(`${API_BASE}/uvil/replay_verify`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        uvil_events: currentEvidencePack.uvil_events,
                        reasoning_artifacts: currentEvidencePack.reasoning_artifacts,
                        expected_u_t: currentEvidencePack.u_t,
                        expected_r_t: currentEvidencePack.r_t,
                        expected_h_t: currentEvidencePack.h_t
                    })
                });

                if (!response.ok) throw new Error('Replay verification request failed');

                const data = await response.json();

                // Display result
                resultDisplay.classList.remove('hidden');
                if (data.result === 'PASS') {
                    resultDisplay.className = 'replay-result pass';
                    resultDisplay.innerHTML = `
                        <strong>REPLAY VERIFICATION: PASS</strong><br>
                        All hashes match. Attestation is deterministic and reproducible.<br>
                        <span style="font-size:0.75rem;">
                            U_t: ${data.computed_u_t.substring(0, 16)}... ✓<br>
                            R_t: ${data.computed_r_t.substring(0, 16)}... ✓<br>
                            H_t: ${data.computed_h_t.substring(0, 16)}... ✓
                        </span>
                    `;
                } else {
                    resultDisplay.className = 'replay-result fail';
                    let diffHtml = '';
                    if (data.diff) {
                        diffHtml = '<div class="replay-diff"><strong>Diff:</strong><br>';
                        if (data.diff.u_t_diff) {
                            diffHtml += `U_t: expected ${data.diff.u_t_diff.expected.substring(0,16)}..., got ${data.diff.u_t_diff.computed.substring(0,16)}...<br>`;
                        }
                        if (data.diff.r_t_diff) {
                            diffHtml += `R_t: expected ${data.diff.r_t_diff.expected.substring(0,16)}..., got ${data.diff.r_t_diff.computed.substring(0,16)}...<br>`;
                        }
                        if (data.diff.h_t_diff) {
                            diffHtml += `H_t: expected ${data.diff.h_t_diff.expected.substring(0,16)}..., got ${data.diff.h_t_diff.computed.substring(0,16)}...<br>`;
                        }
                        diffHtml += '</div>';
                    }
                    resultDisplay.innerHTML = `
                        <strong>REPLAY VERIFICATION: FAIL</strong><br>
                        Hash mismatch detected. Attestation integrity compromised.
                        ${diffHtml}
                    `;
                }

                btn.disabled = false;
                btn.textContent = 'Replay & Verify';

            } catch (e) {
                showError(e.message);
                btn.disabled = false;
                btn.textContent = 'Replay & Verify';
            }
        }

        // Boundary Demo - orchestrated sequence
        // v0.2.10: Uses draft_payload for self-sufficient commits, adds retry on error
        let boundaryDemoHadError = false;

        async function runBoundaryDemo() {
            const btn = document.getElementById('btn-boundary-demo');
            btn.disabled = true;
            btn.textContent = 'Running...';
            boundaryDemoHadError = false;

            const results = document.getElementById('boundary-results');
            results.classList.remove('hidden');

            // Reset all steps
            document.querySelectorAll('.boundary-step').forEach(el => {
                el.classList.remove('active', 'done');
            });
            document.querySelectorAll('.step-outcome').forEach(el => {
                el.textContent = '...';
                el.className = 'step-outcome';
            });
            document.querySelectorAll('.step-reason').forEach(el => {
                el.textContent = '';
            });
            document.getElementById('boundary-conclusion').classList.remove('visible');

            const steps = [
                { id: 'step-1', outcomeId: 'outcome-adv', reasonId: 'reason-adv',
                  claim: '2 + 2 = 4', trustClass: 'ADV', task: 'Boundary demo: ADV' },
                { id: 'step-2', outcomeId: 'outcome-pa', reasonId: 'reason-pa',
                  claim: '2 + 2 = 4', trustClass: 'PA', task: 'Boundary demo: PA' },
                { id: 'step-3', outcomeId: 'outcome-mv', reasonId: 'reason-mv',
                  claim: '2 + 2 = 4', trustClass: 'MV', task: 'Boundary demo: MV verified' },
                { id: 'step-4', outcomeId: 'outcome-refuted', reasonId: 'reason-refuted',
                  claim: '3 * 3 = 8', trustClass: 'MV', task: 'Boundary demo: MV refuted' }
            ];

            for (const step of steps) {
                const stepEl = document.getElementById(step.id);
                stepEl.classList.add('active');

                try {
                    // Propose
                    const proposeRes = await fetch(`${API_BASE}/uvil/propose_partition`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ problem_statement: step.task })
                    });
                    if (!proposeRes.ok) throw new Error('Propose failed');
                    const proposeData = await proposeRes.json();

                    // v0.2.10: Commit using draft_payload (self-sufficient)
                    const commitRes = await fetch(`${API_BASE}/uvil/commit_uvil`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            draft_payload: proposeData.draft_payload,
                            edited_claims: [{
                                claim_text: step.claim,
                                trust_class: step.trustClass,
                                rationale: 'Boundary demo'
                            }],
                            user_fingerprint: 'boundary_demo'
                        })
                    });
                    if (!commitRes.ok) throw new Error('Commit failed');
                    const commitData = await commitRes.json();

                    // Verify
                    const verifyRes = await fetch(`${API_BASE}/uvil/run_verification`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ committed_partition_id: commitData.committed_partition_id })
                    });
                    if (!verifyRes.ok) throw new Error('Verify failed');
                    const verifyData = await verifyRes.json();

                    // Display result
                    const outcomeEl = document.getElementById(step.outcomeId);
                    outcomeEl.textContent = verifyData.outcome;
                    outcomeEl.className = 'step-outcome ' + verifyData.outcome.toLowerCase();

                    const reasonEl = document.getElementById(step.reasonId);
                    if (step.trustClass === 'ADV') {
                        reasonEl.textContent = 'Excluded from authority stream';
                    } else if (step.trustClass === 'PA') {
                        reasonEl.textContent = 'Authority-bearing but no validator';
                    } else if (verifyData.outcome === 'VERIFIED') {
                        reasonEl.textContent = 'Arithmetic validator confirmed';
                    } else if (verifyData.outcome === 'REFUTED') {
                        reasonEl.textContent = 'Arithmetic validator disproved (3*3=9)';
                    } else if (verifyData.outcome === 'ABSTAINED') {
                        reasonEl.textContent = 'Cannot parse as arithmetic';
                    }

                    stepEl.classList.remove('active');
                    stepEl.classList.add('done');

                } catch (e) {
                    // v0.2.10: System-responsible error display
                    const outcomeEl = document.getElementById(step.outcomeId);
                    outcomeEl.textContent = 'ERROR';
                    outcomeEl.className = 'step-outcome';
                    outcomeEl.style.color = '#c62828';
                    const reasonEl = document.getElementById(step.reasonId);
                    reasonEl.textContent = 'Server error. Click Retry below.';
                    reasonEl.style.color = '#c62828';
                    stepEl.classList.remove('active');
                    boundaryDemoHadError = true;
                }

                // Pause for dramatic effect
                await new Promise(r => setTimeout(r, 800));
            }

            // Show conclusion or error message
            await new Promise(r => setTimeout(r, 500));
            const conclusion = document.getElementById('boundary-conclusion');
            if (boundaryDemoHadError) {
                conclusion.innerHTML = `
                    <p style="color:#c62828;"><strong>Demo encountered an error.</strong></p>
                    <p style="color:#888; font-size:0.85rem;">This is a demo reliability issue, not a user error. The server may have lost state.</p>
                    <button onclick="runBoundaryDemo()" style="margin-top:0.75rem; background:#fff; color:#c62828; border:1px solid #c62828; padding:0.5rem 1rem; cursor:pointer;">Retry Boundary Demo</button>
                `;
            }
            conclusion.classList.add('visible');

            btn.disabled = false;
            btn.textContent = boundaryDemoHadError ? 'Retry (≈8s)' : 'Run Again (≈8s)';
        }

        // =====================================================================
        // REJECTION DEMOS - What Gets Rejected
        // =====================================================================

        function showRejectionResult(data) {
            const el = document.getElementById('rejection-result');
            el.classList.remove('hidden');
            if (data.error_code) {
                el.innerHTML = `<span class="error-code">ERROR: ${data.error_code}</span>\\n${JSON.stringify(data, null, 2)}`;
            } else {
                el.textContent = JSON.stringify(data, null, 2);
            }
        }

        // Test 1: Double Commit Attempt
        // v0.2.10: Uses draft_payload for commits
        async function testDoubleCommit() {
            const btn = document.getElementById('btn-double-commit');
            btn.disabled = true;
            btn.textContent = 'Testing...';

            try {
                // First: propose
                const proposeRes = await fetch(`${API_BASE}/uvil/propose_partition`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ problem_statement: 'Double commit test' })
                });
                const proposeData = await proposeRes.json();

                // First commit (should succeed)
                const firstCommit = await fetch(`${API_BASE}/uvil/commit_uvil`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        draft_payload: proposeData.draft_payload,
                        edited_claims: [{ claim_text: '1 + 1 = 2', trust_class: 'MV', rationale: 'test' }],
                        user_fingerprint: 'rejection_test'
                    })
                });
                await firstCommit.json();

                // Second commit with SAME draft_payload (should fail)
                const secondCommit = await fetch(`${API_BASE}/uvil/commit_uvil`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        draft_payload: proposeData.draft_payload,
                        edited_claims: [{ claim_text: '2 + 2 = 4', trust_class: 'MV', rationale: 'test' }],
                        user_fingerprint: 'rejection_test'
                    })
                });

                const result = await secondCommit.json();
                if (!secondCommit.ok) {
                    showRejectionResult(result);
                } else {
                    showRejectionResult({ unexpected: 'Commit succeeded when it should have failed', data: result });
                }

            } catch (e) {
                showRejectionResult({ error: e.message });
            }

            btn.disabled = false;
            btn.textContent = 'Double Commit Attempt';
        }

        // Test 2: Trust-Class Monotonicity Violation
        // v0.2.10: Uses draft_payload for commits
        async function testMonotonicity() {
            const btn = document.getElementById('btn-monotonicity');
            btn.disabled = true;
            btn.textContent = 'Testing...';

            try {
                // Propose
                const proposeRes = await fetch(`${API_BASE}/uvil/propose_partition`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ problem_statement: 'Monotonicity test' })
                });
                const proposeData = await proposeRes.json();

                // Commit with MV
                const commitRes = await fetch(`${API_BASE}/uvil/commit_uvil`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        draft_payload: proposeData.draft_payload,
                        edited_claims: [{ claim_text: '5 + 5 = 10', trust_class: 'MV', rationale: 'test' }],
                        user_fingerprint: 'rejection_test'
                    })
                });
                const commitData = await commitRes.json();

                // Try to change trust class (should fail with TRUST_CLASS_MONOTONICITY_VIOLATION)
                const changeRes = await fetch(`${API_BASE}/uvil/change_trust_class`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        committed_partition_id: commitData.committed_partition_id,
                        claim_index: 0,
                        new_trust_class: 'ADV'
                    })
                });

                const result = await changeRes.json();
                if (!changeRes.ok) {
                    showRejectionResult(result);
                } else {
                    showRejectionResult({ unexpected: 'Change succeeded when it should have failed', data: result });
                }

            } catch (e) {
                showRejectionResult({ error: e.message });
            }

            btn.disabled = false;
            btn.textContent = 'Trust-Class Monotonicity Violation';
        }

        // Test 3: Silent Authority Violation (evidence pack with tampered hash)
        // v0.2.10: Uses draft_payload for commits
        async function testSilentAuthority() {
            const btn = document.getElementById('btn-silent-authority');
            btn.disabled = true;
            btn.textContent = 'Testing...';

            try {
                // Propose
                const proposeRes = await fetch(`${API_BASE}/uvil/propose_partition`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ problem_statement: 'Silent authority test' })
                });
                const proposeData = await proposeRes.json();

                // Commit
                const commitRes = await fetch(`${API_BASE}/uvil/commit_uvil`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        draft_payload: proposeData.draft_payload,
                        edited_claims: [{ claim_text: '7 + 7 = 14', trust_class: 'MV', rationale: 'test' }],
                        user_fingerprint: 'rejection_test'
                    })
                });
                const commitData = await commitRes.json();

                // Run verification
                await fetch(`${API_BASE}/uvil/run_verification`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ committed_partition_id: commitData.committed_partition_id })
                });

                // Try to verify with tampered hash (should fail with SILENT_AUTHORITY_VIOLATION)
                const tamperedRes = await fetch(`${API_BASE}/uvil/verify_attestation`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        committed_partition_id: commitData.committed_partition_id,
                        claimed_h_t: '0000000000000000000000000000000000000000000000000000000000000000'
                    })
                });

                const result = await tamperedRes.json();
                if (!tamperedRes.ok) {
                    showRejectionResult(result);
                } else {
                    showRejectionResult({ unexpected: 'Verification succeeded with tampered hash', data: result });
                }

            } catch (e) {
                showRejectionResult({ error: e.message });
            }

            btn.disabled = false;
            btn.textContent = 'Silent Authority Violation';
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the demo frontend."""
    return get_html_content()


def get_releases_json_version() -> dict:
    """Load version info from releases.json for deployment verification."""
    releases_path = Path(__file__).parent.parent / "releases" / "releases.json"
    try:
        import json
        with open(releases_path, encoding="utf-8") as f:
            data = json.load(f)
        current = data.get("current_version", "unknown")
        version_data = data.get("versions", {}).get(current, {})
        return {
            "releases_json_version": current,
            "releases_json_commit": version_data.get("commit", "unknown"),
            "releases_json_tag": version_data.get("tag", "unknown"),
        }
    except Exception:
        return {
            "releases_json_version": "error",
            "releases_json_commit": "error",
            "releases_json_tag": "error",
        }


@app.get("/health")
async def health():
    """Health check endpoint with version info and release pin validation.

    Includes build metadata for deployment verification:
    - version/tag/commit: Hardcoded in this file (what's running)
    - releases_json_*: From releases.json (canonical source)
    - release_pin: Validation result comparing running vs. expected

    If is_stale=true, status becomes "FAIL_STALE_DEPLOY".
    """
    releases_info = get_releases_json_version()

    # Determine status based on release pin
    if _RELEASE_PIN_STATUS["is_stale"]:
        status = "FAIL_STALE_DEPLOY"
    else:
        status = "ok"

    return {
        "status": status,
        "version": DEMO_VERSION,
        "tag": DEMO_TAG,
        "commit": DEMO_COMMIT,
        "base_path": BASE_PATH or "/",
        # Build metadata for deployment verification
        "build_commit": DEMO_COMMIT,
        "build_tag": DEMO_TAG,
        **releases_info,
        # Release pin validation result
        "release_pin": _RELEASE_PIN_STATUS,
    }


@app.get("/healthz")
async def healthz():
    """Kubernetes-style health check endpoint.

    Always returns 200 (container is running), but includes version headers
    for hostile auditor verification:
    - X-MathLedger-Version: vX.Y.Z
    - X-MathLedger-Commit: full commit hash
    - X-MathLedger-Tag: version tag
    - X-MathLedger-Stale-Deploy: true/false
    """
    is_stale = _RELEASE_PIN_STATUS["is_stale"]
    return PlainTextResponse(
        "ok",
        status_code=200,
        headers={
            "X-MathLedger-Version": f"v{DEMO_VERSION}",
            "X-MathLedger-Commit": DEMO_COMMIT,
            "X-MathLedger-Tag": DEMO_TAG,
            "X-MathLedger-Stale-Deploy": str(is_stale).lower(),
        }
    )


@app.get("/scenarios")
async def get_scenarios():
    """Return available scenarios for the demo."""
    return SCENARIOS


@app.get("/ui_copy")
async def get_ui_copy():
    """Return canonical UI copy strings for regression testing."""
    return UI_COPY


# Docs viewer endpoint
@app.get("/docs/view/{doc_name}", response_class=HTMLResponse)
async def view_doc(doc_name: str):
    """Serve documentation files as HTML with markdown rendering."""
    docs_path = Path(__file__).parent.parent / "docs" / doc_name

    if not docs_path.exists() or not docs_path.is_file():
        return HTMLResponse(
            content=f"<h1>404</h1><p>Document not found: {doc_name}</p>",
            status_code=404
        )

    content = docs_path.read_text(encoding="utf-8")

    # Simple HTML wrapper with basic styling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{doc_name} - MathLedger Docs</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
                max-width: 900px;
                margin: 0 auto;
                padding: 2rem;
                line-height: 1.6;
                color: #1a1a1a;
                background: #f9f9f9;
            }}
            pre {{
                background: #f0f0f0;
                padding: 1rem;
                overflow-x: auto;
                border-radius: 4px;
            }}
            code {{
                background: #f0f0f0;
                padding: 0.2rem 0.4rem;
                border-radius: 2px;
                font-size: 0.9em;
            }}
            pre code {{
                background: none;
                padding: 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1rem 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 0.5rem;
                text-align: left;
            }}
            th {{
                background: #f0f0f0;
            }}
            h1, h2, h3 {{
                border-bottom: 1px solid #ddd;
                padding-bottom: 0.5rem;
            }}
            a {{
                color: #1565c0;
            }}
            .back-link {{
                display: inline-block;
                margin-bottom: 1rem;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <a href="{BASE_PATH or '/'}" class="back-link">← Back to Demo</a>
        <pre style="background:#fff; border:1px solid #ddd; white-space:pre-wrap;">{content}</pre>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    print("=" * 60)
    print(f"MathLedger UVIL v{DEMO_VERSION} Demo")
    print("=" * 60)
    print()
    print("Open in browser: http://localhost:8000")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
