"""
Build static site for mathledger.ai deployment.

This script generates site/ as an EPISTEMIC ARCHIVE, not a product site.
Each version is a complete, immutable snapshot with:
- Rendered documentation (scope lock, explanation, invariants)
- Downloadable fixtures with SHA256 checksums
- Evidence pack for replay verification
- Manifest with file hashes for integrity verification

VERSION METADATA SOURCE: releases/releases.json (CANONICAL)
This script reads ALL version metadata from that file. No hardcoded defaults.

The interactive demo is NOT hosted statically. The archive clearly states:
- "Archive mode": docs, fixtures, evidence — available now
- "Interactive demo": run locally or (future) demo.mathledger.ai

Usage:
    uv run python scripts/build_static_site.py --clean --all
    uv run python scripts/build_static_site.py --version v0.2.0
    uv run python scripts/build_static_site.py --verify  # Verify build matches releases.json
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
SITE_DIR = REPO_ROOT / "site"
RELEASES_FILE = REPO_ROOT / "releases" / "releases.json"


class BuildError(Exception):
    """Raised when build verification fails."""
    pass


def load_releases() -> dict[str, Any]:
    """Load release metadata from canonical releases.json file."""
    if not RELEASES_FILE.exists():
        raise BuildError(
            f"FATAL: Canonical release file not found: {RELEASES_FILE}\n"
            "This file is the ONLY source of version metadata.\n"
            "Create it manually or copy from a known-good state."
        )

    with open(RELEASES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Validate required fields
    if "current_version" not in data:
        raise BuildError("releases.json missing 'current_version' field")
    if "versions" not in data:
        raise BuildError("releases.json missing 'versions' field")

    current = data["current_version"]
    if current not in data["versions"]:
        raise BuildError(
            f"current_version '{current}' not found in versions dict"
        )

    # Validate each version has required fields
    required_fields = ["tag", "commit", "date_locked", "status", "invariants"]
    for v, config in data["versions"].items():
        for field in required_fields:
            if field not in config:
                raise BuildError(f"Version '{v}' missing required field: {field}")

        # Validate invariants structure
        inv = config["invariants"]
        for tier_field in ["tier_a", "tier_b", "tier_c"]:
            if tier_field not in inv:
                raise BuildError(f"Version '{v}' invariants missing: {tier_field}")

    print(f"Loaded releases.json: {len(data['versions'])} versions, current={current}")
    return data


def get_version_config(releases: dict, version: str) -> dict:
    """Get config for a specific version, converting docs format."""
    if version not in releases["versions"]:
        available = list(releases["versions"].keys())
        raise BuildError(f"Unknown version: {version}. Available: {available}")

    config = releases["versions"][version].copy()

    # Convert docs format from releases.json to internal format
    # releases.json: [{"source": "...", "slug": "...", "title": "..."}]
    # internal: [("source", "slug", "title")]
    if "docs" in config and isinstance(config["docs"], list):
        if config["docs"] and isinstance(config["docs"][0], dict):
            config["docs"] = [
                (d["source"], d["slug"], d["title"])
                for d in config["docs"]
            ]

    return config


# CSS styles (shared across all pages)
STYLES = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: #f5f5f5;
    color: #1a1a1a;
    line-height: 1.6;
    font-size: 15px;
}
.container { max-width: 900px; margin: 0 auto; padding: 2rem; }

/* Version banner — appears on every page */
.version-banner {
    background: #fff;
    border: 1px solid #ddd;
    border-left: 4px solid var(--status-color, #757575);
    padding: 1rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
}
.version-banner .title { font-weight: 600; font-size: 1rem; }
.version-banner .status { font-weight: 600; color: var(--status-color, #757575); }
.version-banner code { background: #f0f0f0; padding: 0.15em 0.35em; font-size: 0.85em; }
.version-banner .meta { margin-top: 0.5rem; font-size: 0.8rem; color: #555; }

/* Invariant snapshot — appears on every page */
.invariant-snapshot {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    padding: 0.75rem 1rem;
    margin-bottom: 1.5rem;
    font-size: 0.8rem;
    font-family: monospace;
}
.invariant-snapshot .counts {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 0.5rem;
}
.invariant-snapshot .tier-a { color: #2e7d32; }
.invariant-snapshot .tier-b { color: #f57c00; }
.invariant-snapshot .tier-c { color: #757575; }
.invariant-snapshot .cannot-enforce {
    border-top: 1px solid #e0e0e0;
    padding-top: 0.5rem;
    margin-top: 0.5rem;
    color: #666;
}
.invariant-snapshot .cannot-enforce strong { color: #c62828; }

/* Navigation */
.nav { margin-bottom: 1.5rem; font-size: 0.9rem; }
.nav a { margin-right: 1rem; color: #0066cc; text-decoration: none; }
.nav a:hover { text-decoration: underline; }
.nav .sep { color: #999; margin: 0 0.25rem; }

/* Mode indicator — archive vs demo */
.mode-indicator {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    border-radius: 3px;
    margin-left: 0.5rem;
}
.mode-archive { background: #e3f2fd; color: #1565c0; }
.mode-demo { background: #fff3e0; color: #e65100; }
.mode-local { background: #fce4ec; color: #c62828; }
.mode-hosted { background: #e8f5e9; color: #2e7d32; }

/* Demo button */
.demo-button {
    display: inline-block;
    padding: 0.75rem 1.5rem;
    background: #2e7d32;
    color: #fff;
    text-decoration: none;
    font-weight: 600;
    border-radius: 4px;
    margin: 0.5rem 0;
}
.demo-button:hover { background: #1b5e20; }
.demo-available { border-left-color: #2e7d32; }

/* Content styles */
h1 { font-size: 1.4rem; margin-bottom: 1rem; }
h2 { font-size: 1.2rem; margin: 1.5rem 0 0.75rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }
h3 { font-size: 1.05rem; margin: 1.25rem 0 0.5rem; }
p { margin: 0.75rem 0; }
ul, ol { margin: 0.75rem 0 0.75rem 1.5rem; }
li { margin: 0.25rem 0; }
code { background: #f0f0f0; padding: 0.15em 0.35em; font-size: 0.9em; }
pre { background: #1a1a1a; color: #f0f0f0; padding: 1rem; overflow-x: auto; margin: 1rem 0; border-radius: 4px; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
th { background: #f5f5f5; }
blockquote { border-left: 3px solid #ddd; padding-left: 1rem; margin: 1rem 0; color: #555; }
hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
a { color: #0066cc; }

/* Info boxes */
.info-box {
    background: #fff;
    border: 1px solid #ddd;
    padding: 1rem;
    margin: 1rem 0;
}
.info-box.warning { border-left: 4px solid #f57c00; }
.info-box.local-only { border-left: 4px solid #c62828; background: #fff8f8; }

/* Footer */
footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #ddd;
    font-size: 0.75rem;
    color: #666;
}
footer code { font-size: 0.7rem; }
"""


def get_current_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    """Compute SHA256 of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def render_markdown_to_html(md_content: str) -> str:
    """Convert markdown to HTML (basic regex-based, no dependencies)."""
    html = md_content

    # Fenced code blocks
    html = re.sub(
        r"```(\w*)\n(.*?)```",
        lambda m: f'<pre><code class="language-{m.group(1)}">{m.group(2)}</code></pre>',
        html,
        flags=re.DOTALL,
    )

    # Inline code
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

    # Headers
    for i in range(6, 0, -1):
        html = re.sub(
            rf"^{'#' * i}\s+(.+)$",
            rf"<h{i}>\1</h{i}>",
            html,
            flags=re.MULTILINE,
        )

    # Bold and italic
    html = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", html)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

    # Horizontal rules
    html = re.sub(r"^---+$", r"<hr>", html, flags=re.MULTILINE)

    # Blockquotes
    html = re.sub(r"^>\s+(.+)$", r"<blockquote>\1</blockquote>", html, flags=re.MULTILINE)

    # Links
    html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)

    # Tables
    def convert_table(match: re.Match) -> str:
        lines = match.group(0).strip().split("\n")
        rows = []
        for i, line in enumerate(lines):
            if "|" not in line or re.match(r"^\|?[\s\-:|]+\|?$", line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            rows.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        return "<table>" + "\n".join(rows) + "</table>" if rows else match.group(0)

    html = re.sub(r"(\|.+\|\n)+", convert_table, html)

    # Unordered lists
    def convert_ul(match: re.Match) -> str:
        items = re.findall(r"^[-*]\s+(.+)$", match.group(0), flags=re.MULTILINE)
        return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

    html = re.sub(r"(^[-*]\s+.+$\n?)+", convert_ul, html, flags=re.MULTILINE)

    # Ordered lists
    def convert_ol(match: re.Match) -> str:
        items = re.findall(r"^\d+\.\s+(.+)$", match.group(0), flags=re.MULTILINE)
        return "<ol>" + "".join(f"<li>{item}</li>" for item in items) + "</ol>"

    html = re.sub(r"(^\d+\.\s+.+$\n?)+", convert_ol, html, flags=re.MULTILINE)

    # Paragraphs
    paragraphs = []
    for block in re.split(r"\n\n+", html):
        block = block.strip()
        if not block:
            continue
        if block.startswith("<"):
            paragraphs.append(block)
        else:
            paragraphs.append(f"<p>{block}</p>")

    return "\n".join(paragraphs)


def build_version_banner(config: dict, version: str) -> str:
    """Build the version banner HTML that appears on every page."""
    status = config["status"]
    if status == "current":
        status_label = "CURRENT"
        status_color = "#2e7d32"
    elif status.startswith("superseded"):
        status_label = status.upper().replace("-", " ")
        status_color = "#f57c00"
    elif status == "internal":
        status_label = "INTERNAL"
        status_color = "#9e9e9e"
    else:
        status_label = status.upper()
        status_color = "#757575"

    return f"""
    <div class="version-banner" style="--status-color: {status_color};">
        <div class="title">MathLedger — Version {version}</div>
        <div><span class="status">Status: {status_label}</span></div>
        <div class="meta">
            Tag: <code>{config['tag']}</code> |
            Commit: <code>{config['commit'][:12]}</code> |
            Locked: {config['date_locked']}
        </div>
    </div>
    """


def build_invariant_snapshot(config: dict) -> str:
    """Build the invariant snapshot HTML that appears on every page."""
    inv = config["invariants"]
    cannot_items = "".join(f"<li>{item}</li>" for item in inv.get("cannot_enforce", []))

    return f"""
    <div class="invariant-snapshot">
        <div class="counts">
            <span class="tier-a">Tier A (enforced): {inv['tier_a']}</span>
            <span class="tier-b">Tier B (logged): {inv['tier_b']}</span>
            <span class="tier-c">Tier C (aspirational): {inv['tier_c']}</span>
        </div>
        <div class="cannot-enforce">
            <strong>What this version cannot enforce:</strong>
            <ul>{cannot_items}</ul>
        </div>
    </div>
    """


def build_nav(version: str, current_section: str = "") -> str:
    """Build navigation bar."""
    sections = [
        ("", "Archive"),
        ("docs/scope-lock/", "Scope"),
        ("docs/explanation/", "Explanation"),
        ("docs/invariants/", "Invariants"),
        ("fixtures/", "Fixtures"),
        ("evidence-pack/", "Evidence"),
    ]
    links = []
    for path, label in sections:
        href = f"/{version}/{path}" if path else f"/{version}/"
        if path.rstrip("/") == current_section.rstrip("/"):
            links.append(f"<strong>{label}</strong>")
        else:
            links.append(f'<a href="{href}">{label}</a>')

    return f'<nav class="nav">{" ".join(links)} | <a href="/versions/">All Versions</a></nav>'


def build_page(
    title: str,
    content: str,
    config: dict,
    version: str,
    current_section: str = "",
    mode: str = "archive",
    build_time: str = "",
) -> str:
    """Build a complete HTML page with banner, invariant snapshot, and content."""
    mode_class = f"mode-{mode}"
    mode_label = {"archive": "ARCHIVE", "demo": "DEMO", "local": "LOCAL ONLY"}.get(mode, mode.upper())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — MathLedger {version}</title>
    <style>{STYLES}</style>
</head>
<body>
    <div class="container">
        {build_version_banner(config, version)}
        {build_invariant_snapshot(config)}
        {build_nav(version, current_section)}

        <article>
            <h1>{title} <span class="mode-indicator {mode_class}">{mode_label}</span></h1>
            {content}
        </article>

        <footer>
            Site built from commit <code>{config['commit']}</code> at <code>{build_time}</code><br>
            This is an epistemic archive. Content is immutable once published.
        </footer>
    </div>
</body>
</html>
"""


def build_version_landing(config: dict, version: str, build_time: str) -> str:
    """Build the version landing page that distinguishes archive from demo."""
    is_current = config.get("status") == "current"
    demo_url = config.get("demo_url")
    hosted_demo = config.get("hosted_demo", False)

    # Hosted demo is available at /demo/ for the CURRENT version only
    if is_current and hosted_demo:
        # Current version with hosted demo: show prominent hosted demo link
        demo_section = f"""
        <div class="info-box demo-available">
            <strong>Interactive Demo</strong> <span class="mode-indicator mode-hosted">HOSTED</span><br>
            <p>The interactive demo for this version is hosted and ready to use.</p>
            <a href="/demo/" class="demo-button">Open Hosted Demo</a>
            <p style="margin-top: 1rem; font-size: 0.85rem; color: #666;">
                The hosted demo runs the same code as this archived version ({config['tag']}).
            </p>
            <details style="margin-top: 1rem;">
                <summary style="cursor: pointer; color: #666;">Run locally instead</summary>
                <div style="margin-top: 0.5rem;">
                    <code>git clone https://github.com/your-org/mathledger</code><br>
                    <code>git checkout {config['tag']}</code><br>
                    <code>uv run python demo/app.py</code><br>
                    <code>Open http://localhost:8000</code>
                </div>
            </details>
        </div>

        <div class="info-box" style="border-left: 4px solid #1565c0; margin-top: 1rem;">
            <strong>Post-Deploy Smoke Check</strong>
            <p style="font-size: 0.85rem; margin: 0.5rem 0;">
                After deployment, verify the hosted demo is working:
            </p>
            <ol style="font-size: 0.85rem; margin: 0.5rem 0 0 1.5rem; padding: 0;">
                <li>Visit <a href="/demo/healthz">/demo/healthz</a> → should return <code>ok</code></li>
                <li>Run the "Same Claim 90-second proof" → all 4 steps should complete without ERROR</li>
                <li>Download evidence pack and verify with <a href="/{version}/evidence-pack/verify/">the auditor tool</a></li>
            </ol>
        </div>
        """
    elif demo_url:
        # Explicit demo_url override (future use)
        demo_section = f"""
        <div class="info-box">
            <strong>Interactive Demo</strong><br>
            Available at: <a href="{demo_url}">{demo_url}</a>
        </div>
        """
    else:
        # Superseded version: local only, but mention current demo exists
        demo_section = f"""
        <div class="info-box local-only">
            <strong>Interactive Demo</strong> <span class="mode-indicator mode-local">LOCAL ONLY</span><br>
            This archived version's demo requires local Python execution.<br><br>
            <strong>To run locally:</strong><br>
            <code>git clone https://github.com/your-org/mathledger</code><br>
            <code>git checkout {config['tag']}</code><br>
            <code>uv run python demo/app.py</code><br>
            <code>Open http://localhost:8000</code>
            <p style="margin-top: 1rem; font-size: 0.85rem; color: #666;">
                The <a href="/demo/">hosted demo</a> runs the current version, not this archived version.
            </p>
        </div>
        """

    inv = config["invariants"]
    tier_a_items = "".join(f"<li>{item}</li>" for item in inv.get("tier_a_list", []))
    tier_b_items = "".join(f"<li>{item}</li>" for item in inv.get("tier_b_list", []))
    tier_c_items = "".join(f"<li>{item}</li>" for item in inv.get("tier_c_list", []))

    content = f"""
        <p>This is the archive for MathLedger version <code>{version}</code>.
        All artifacts below are static, verifiable, and immutable.</p>

        <h2>Archive Contents</h2>
        <table>
            <tr><th>Artifact</th><th>Description</th></tr>
            <tr><td><a href="docs/scope-lock/">Scope Lock</a></td><td>What this version does and does not demonstrate</td></tr>
            <tr><td><a href="docs/explanation/">Explanation</a></td><td>How the demo explains its own behavior</td></tr>
            <tr><td><a href="docs/invariants/">Invariants</a></td><td>Tier A/B/C enforcement status</td></tr>
            <tr><td><a href="fixtures/">Fixtures</a></td><td>Regression test cases with golden outputs</td></tr>
            <tr><td><a href="evidence-pack/">Evidence Pack</a></td><td>Replay verification artifacts</td></tr>
            <tr><td><a href="manifest.json">Manifest</a></td><td>Version metadata + file checksums</td></tr>
        </table>

        <h2>Interactive Demo</h2>
        {demo_section}

        <h2>Invariant Details</h2>
        <h3 class="tier-a" style="color: #2e7d32;">Tier A: Cryptographically/Structurally Enforced ({inv['tier_a']})</h3>
        <ul>{tier_a_items}</ul>

        <h3 class="tier-b" style="color: #f57c00;">Tier B: Logged, Not Hard-Gated ({inv['tier_b']})</h3>
        <ul>{tier_b_items}</ul>

        <h3 class="tier-c" style="color: #757575;">Tier C: Documented, Not Enforced ({inv['tier_c']})</h3>
        <ul>{tier_c_items}</ul>

        <h2>Verification</h2>
        <p>To verify this archive matches the source:</p>
        <ol>
            <li>Clone the repository</li>
            <li>Checkout commit <code>{config['commit']}</code></li>
            <li>Run <code>uv run python scripts/build_static_site.py --version {version}</code></li>
            <li>Compare generated files to this archive</li>
        </ol>
        <p>File checksums are in <a href="manifest.json">manifest.json</a>.</p>
    """

    return build_page(
        title=f"Version {version} Archive",
        content=content,
        config=config,
        version=version,
        current_section="",
        mode="archive",
        build_time=build_time,
    )


def build_fixtures_index(config: dict, version: str, fixtures: list, build_time: str) -> str:
    """Build fixtures index page."""
    rows = []
    for fixture in fixtures:
        name = fixture["name"]
        file_count = len(fixture["files"])
        rows.append(f'<tr><td><a href="{name}/">{name}</a></td><td>{file_count} files</td></tr>')

    content = f"""
        <p>These are the regression test fixtures for version <code>{version}</code>.
        Each fixture contains input and expected output JSON files.</p>

        <table>
            <tr><th>Fixture</th><th>Files</th></tr>
            {"".join(rows)}
        </table>

        <h2>Checksum Verification</h2>
        <p>Download <a href="index.json">index.json</a> for SHA256 checksums of all files.</p>

        <h2>Usage</h2>
        <pre><code># Run regression harness locally
git checkout {config['tag']}
uv run python tools/run_demo_cases.py</code></pre>
    """

    return build_page(
        title="Test Fixtures",
        content=content,
        config=config,
        version=version,
        current_section="fixtures",
        mode="archive",
        build_time=build_time,
    )


def build_evidence_pack_page(config: dict, version: str, files: list, build_time: str) -> str:
    """Build evidence pack page."""
    file_links = "".join(f'<li><a href="{f}">{f}</a></li>' for f in files)

    content = f"""
        <p>The evidence pack enables independent replay verification.
        An auditor can recompute attestation hashes without running the demo.</p>

        <h2>Files</h2>
        <ul>{file_links}</ul>

        <h2>What Replay Verification Proves</h2>
        <ul>
            <li>The recorded hashes match what the inputs produce</li>
            <li>The attestation trail has not been tampered with</li>
            <li>Determinism: same inputs produce same outputs</li>
        </ul>

        <h2>What Replay Verification Does NOT Prove</h2>
        <ul>
            <li>That the claims are true</li>
            <li>That the verification was sound</li>
            <li>That the system behaved safely</li>
        </ul>

        <h2>Replay Instructions</h2>
        <ol>
            <li>Download the evidence pack JSON file(s) above</li>
            <li>Clone the repo and checkout <code>{config['tag']}</code></li>
            <li>Run replay verification:
                <pre><code>uv run python -c "
from backend.api.uvil import replay_verify
import json
with open('evidence_pack.json') as f:
    pack = json.load(f)
result = replay_verify(pack)
print('PASS' if result['verified'] else 'FAIL')
"</code></pre>
            </li>
            <li>PASS = hashes match; FAIL = tampering detected</li>
        </ol>

        <p>See <a href="../docs/explanation/">How the Demo Explains Itself</a> for full details.</p>
    """

    return build_page(
        title="Evidence Pack",
        content=content,
        config=config,
        version=version,
        current_section="evidence-pack",
        mode="archive",
        build_time=build_time,
    )


def build_verifier_page(config: dict, version: str, has_vectors: bool) -> str:
    """Build the evidence pack verifier page with optional self-test."""
    tag = config["tag"]
    commit = config["commit"][:12]

    selftest_section = ""
    selftest_js = ""
    if has_vectors:
        selftest_section = '''
<div class="vbox">
<h2>Self-Test (Built-in Vectors)</h2>
<p>Run verification against canonical test vectors.</p>
<button class="btn-p" id="selftest-btn" onclick="runSelfTest()">Run self-test (built-in vectors)</button>
<div id="selftest-status" style="margin:0.5rem 0;font-weight:600;display:none"></div>
<table id="selftest-table" style="display:none">
<thead><tr><th>Name</th><th>Expected</th><th>Actual</th><th>Result</th><th>Reason</th></tr></thead>
<tbody id="selftest-body"></tbody>
</table>
</div>'''
        selftest_js = '''
async function runSelfTest(){const btn=document.getElementById("selftest-btn");const status=document.getElementById("selftest-status");const table=document.getElementById("selftest-table");const tbody=document.getElementById("selftest-body");btn.disabled=true;status.style.display="block";status.textContent="Loading...";status.className="";tbody.innerHTML="";table.style.display="none";try{const resp=await fetch("vectors.json");if(!resp.ok)throw new Error("vectors.json not found");const vectors=await resp.json();status.textContent="Running tests...";const results=[];for(const tc of(vectors.valid_packs||[])){const r=await testPack(tc.pack,tc.expected_result,tc.expected_failure_reason);results.push({name:tc.name,expected:tc.expected_result,actual:r.actual,pass:r.pass,reason:r.reason});}for(const tc of(vectors.invalid_packs||[])){const r=await testPack(tc.pack,tc.expected_result,tc.expected_failure_reason);results.push({name:tc.name,expected:tc.expected_result,actual:r.actual,pass:r.pass,reason:r.reason});}let allPass=true;for(const r of results){const tr=document.createElement("tr");tr.className=r.pass?"row-pass":"row-fail";const cls=r.pass?"match":"mismatch";const txt=r.pass?"PASS":"FAIL";tr.innerHTML="<td>"+esc(r.name)+"</td><td>"+esc(r.expected)+"</td><td>"+esc(r.actual)+"</td><td class=\\""+cls+"\\">"+txt+"</td><td>"+esc(r.reason||"-")+"</td>";tbody.appendChild(tr);if(!r.pass)allPass=false;}table.style.display="table";status.className=allPass?"match":"mismatch";status.textContent=allPass?"SELF-TEST PASSED ("+results.length+")":"SELF-TEST FAILED";}catch(e){status.className="mismatch";status.textContent="Error: "+e.message;}btn.disabled=false;}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
async function testPack(pack,expectedResult,expectedReason){try{const uvil=pack.uvil_events||[];const arts=pack.reasoning_artifacts||[];const declaredU=pack.u_t||"";const declaredR=pack.r_t||"";const declaredH=pack.h_t||"";for(const a of arts){if(!("validation_outcome"in a))return{actual:"FAIL",pass:expectedResult==="FAIL"&&expectedReason==="missing_required_field",reason:"missing_required_field"};}const computedU=await sha(can(uvil));const computedR=await sha(can(arts));const computedH=await sha(computedR+computedU);if(computedU!==declaredU)return{actual:"FAIL",pass:expectedResult==="FAIL"&&expectedReason==="u_t_mismatch",reason:"u_t_mismatch"};if(computedR!==declaredR)return{actual:"FAIL",pass:expectedResult==="FAIL"&&expectedReason==="r_t_mismatch",reason:"r_t_mismatch"};if(computedH!==declaredH)return{actual:"FAIL",pass:expectedResult==="FAIL"&&expectedReason==="h_t_mismatch",reason:"h_t_mismatch"};return{actual:"PASS",pass:expectedResult==="PASS",reason:null};}catch(e){return{actual:"FAIL",pass:expectedResult==="FAIL",reason:e.message};}}'''

    css = "*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,monospace;background:#f5f5f5;line-height:1.6}.container{max-width:900px;margin:0 auto;padding:2rem}.banner{background:#fff;border:1px solid #ddd;border-left:4px solid #2e7d32;padding:1rem;margin-bottom:1.5rem}.status{font-weight:600;color:#2e7d32}code{background:#f0f0f0;padding:0.15em 0.35em}h1{font-size:1.4rem;margin-bottom:1rem}h2{font-size:1.2rem;margin:1.5rem 0 0.75rem;border-bottom:1px solid #ddd}p{margin:0.75rem 0}.info{background:#fff;border:1px solid #ddd;padding:1rem;margin:1rem 0;border-left:4px solid #f57c00}.vbox{background:#fff;border:1px solid #ddd;padding:1.5rem;margin:1rem 0}.result{padding:1rem;margin:1rem 0;font-family:monospace}.pass{background:#e8f5e9;border-left:4px solid #2e7d32}.fail{background:#ffebee;border-left:4px solid #c62828}.pending{background:#fff3e0;border-left:4px solid #f57c00}.row{margin:0.5rem 0}.row label{font-weight:600;display:inline-block;width:100px}.match{color:#2e7d32!important;font-weight:600}.mismatch{color:#c62828!important;font-weight:600}textarea{width:100%;height:200px;font-family:monospace;font-size:0.85rem}button{padding:0.5rem 1rem;margin:0.5rem 0.5rem 0.5rem 0;cursor:pointer}button:disabled{opacity:0.5;cursor:not-allowed}.btn-p{background:#0066cc;color:#fff;border:none}.btn-s{background:#f5f5f5;border:1px solid #ddd}footer{margin-top:2rem;padding-top:1rem;border-top:1px solid #ddd;font-size:0.75rem;color:#666}a{color:#0066cc}.nav{margin-bottom:1.5rem}.nav a{margin-right:1rem}table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem}th,td{border:1px solid #ddd;padding:0.5rem;text-align:left}th{background:#f5f5f5}.row-pass{background:#f1f8e9}.row-fail{background:#ffebee}"

    core_js = r'''function can(o){if(o===null)return'null';if(typeof o==='boolean')return o?'true':'false';if(typeof o==='number')return Object.is(o,-0)?'0':String(o);if(typeof o==='string'){let r='"';for(let i=0;i<o.length;i++){const c=o.charCodeAt(i);if(c===8)r+='\b';else if(c===9)r+='\t';else if(c===10)r+='\n';else if(c===12)r+='\f';else if(c===13)r+='\r';else if(c===34)r+='\"';else if(c===92)r+='\\';else if(c<32)r+='\u'+c.toString(16).padStart(4,'0');else r+=o[i];}return r+'"';}if(Array.isArray(o))return'['+o.map(can).join(',')+']';if(typeof o==='object'){const k=Object.keys(o).sort();return'{'+k.map(x=>can(x)+':'+can(o[x])).join(',')+'}';}throw Error('bad');}
async function sha(s){const d=new TextEncoder().encode(s);const h=await crypto.subtle.digest('SHA-256',d);return Array.from(new Uint8Array(h)).map(b=>b.toString(16).padStart(2,'0')).join('');}
document.getElementById('fi').onchange=e=>{if(e.target.files[0]){const r=new FileReader();r.onload=x=>document.getElementById('inp').value=x.target.result;r.readAsText(e.target.files[0]);}};
async function verify(){const R=document.getElementById('res'),D=document.getElementById('det');try{const v=document.getElementById('inp').value.trim();if(!v){R.className='result pending';R.innerHTML='<strong>Status:</strong> No input';D.style.display='none';return;}const p=JSON.parse(v);const uvil=p.uvil_events||[];const arts=p.reasoning_artifacts||[];const eu=p.u_t||'';const er=p.r_t||'';const eh=p.h_t||'';const cu=await sha(can(uvil));const cr=await sha(can(arts));const ch=await sha(cr+cu);document.getElementById('eu').textContent=eu||'-';document.getElementById('cu').textContent=cu;document.getElementById('er').textContent=er||'-';document.getElementById('cr').textContent=cr;document.getElementById('eh').textContent=eh||'-';document.getElementById('ch').textContent=ch;const uok=!eu||cu===eu,rok=!er||cr===er,hok=!eh||ch===eh;document.getElementById('cu').className=uok?'match':'mismatch';document.getElementById('cr').className=rok?'match':'mismatch';document.getElementById('ch').className=hok?'match':'mismatch';D.style.display='block';if(!eu&&!er&&!eh){R.className='result pending';R.innerHTML='<strong>Status:</strong> COMPUTED';}else if(uok&&rok&&hok){R.className='result pass';R.innerHTML='<strong>Status:</strong> PASS';}else{R.className='result fail';R.innerHTML='<strong>Status:</strong> FAIL';}}catch(e){R.className='result fail';R.innerHTML='<strong>Status:</strong> '+e.message;D.style.display='none';}}'''

    vectors_link = '<a href="vectors.json">Test Vectors</a>' if has_vectors else ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evidence Pack Verifier - MathLedger {version}</title>
<style>{css}</style>
</head>
<body>
<div class="container">
<div class="banner">
<div style="font-weight:600">MathLedger - {version}</div>
<div><span class="status">CURRENT</span></div>
<div style="font-size:0.8rem;color:#555;margin-top:0.5rem">Tag: <code>{tag}</code> | Commit: <code>{commit}</code></div>
</div>
<nav class="nav"><a href="/{version}/">Archive</a> <a href="/{version}/evidence-pack/">Evidence</a> <strong>Verifier</strong> | <a href="/versions/">All</a></nav>
<h1>Evidence Pack Verifier</h1>
<div class="info"><strong>Pure JS</strong> - Runs in browser, no server. Uses RFC 8785 canonicalization.</div>
{selftest_section}
<div class="vbox">
<h2>Manual Verification</h2>
<textarea id="inp" placeholder="Paste evidence_pack.json..."></textarea>
<div>
<input type="file" id="fi" accept=".json" style="display:none">
<button class="btn-s" onclick="document.getElementById('fi').click()">Upload</button>
<button class="btn-p" onclick="verify()">Verify</button>
</div>
</div>
<div id="res" class="result pending"><strong>Status:</strong> Waiting...</div>
<div id="det" style="display:none">
<h2>Hashes</h2>
<div class="row"><label>U_t:</label> Exp: <code id="eu"></code> Got: <code id="cu"></code></div>
<div class="row"><label>R_t:</label> Exp: <code id="er"></code> Got: <code id="cr"></code></div>
<div class="row"><label>H_t:</label> Exp: <code id="eh"></code> Got: <code id="ch"></code></div>
</div>
<footer>MathLedger {version} Verifier | {vectors_link}</footer>
</div>
<script>
{core_js}
{selftest_js}
</script>
</body>
</html>"""


def build_versions_index(versions: list[dict], build_time: str) -> str:
    """Build /versions/ index page. NOT a changelog."""
    rows = []
    for v in versions:
        status = v["status"]
        if status == "current":
            status_class = "current"
            status_label = "CURRENT"
        elif status.startswith("superseded"):
            status_class = "superseded"
            status_label = status.upper().replace("-", " ")
        elif status == "internal":
            status_class = "internal"
            status_label = "INTERNAL"
        else:
            status_class = ""
            status_label = status.upper()

        demo_link = ""
        if v.get("demo_url"):
            demo_link = f' | <a href="{v["demo_url"]}">Demo</a>'

        rows.append(f"""
            <tr>
                <td><a href="/{v['version']}/">{v['version']}</a>{demo_link}</td>
                <td class="{status_class}">{status_label}</td>
                <td>{v['date_locked']}</td>
                <td><code>{v['commit'][:7]}</code></td>
            </tr>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Version Archive — MathLedger</title>
    <style>
        {STYLES}
        .current {{ color: #2e7d32; font-weight: bold; }}
        .superseded {{ color: #f57c00; }}
        .internal {{ color: #9e9e9e; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MathLedger Version Archive</h1>

        <p>Each version is a complete, immutable snapshot.
        Navigate to a version to inspect its full state.</p>

        <p><strong>This is not a changelog.</strong>
        Versions are not summarized. Each version page contains all artifacts and documentation.</p>

        <table>
            <tr>
                <th>Version</th>
                <th>Status</th>
                <th>Locked</th>
                <th>Commit</th>
            </tr>
            {"".join(rows)}
        </table>

        <hr>

        <h2>Archive Integrity</h2>
        <p>Each version directory is immutable once deployed.
        Superseded versions remain fully navigable.
        Prior versions are never modified; only their status label changes.</p>

        <footer>
            Site built at <code>{build_time}</code><br>
            This is an epistemic archive, not a product site.
        </footer>
    </div>
</body>
</html>
"""


def collect_file_checksums(directory: Path, base_path: Path) -> list[dict]:
    """Recursively collect file paths and SHA256 checksums."""
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel_path = str(path.relative_to(base_path)).replace("\\", "/")
            files.append({
                "path": rel_path,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            })
    return files


def build_version(releases: dict, version: str, build_time: str) -> dict:
    """Build static site for a specific version. Returns manifest data."""
    config = get_version_config(releases, version)

    print(f"Building {version}...")
    print(f"  Tag: {config['tag']}")
    print(f"  Commit: {config['commit']}")
    print(f"  Status: {config['status']}")

    version_dir = SITE_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Build version landing page
    landing_html = build_version_landing(config, version, build_time)
    (version_dir / "index.html").write_text(landing_html, encoding="utf-8")
    print(f"  Created {version}/index.html (landing)")

    # Render docs
    docs_dir = version_dir / "docs"
    for md_path, slug, doc_title in config.get("docs", []):
        src = REPO_ROOT / md_path
        if not src.exists():
            print(f"  Warning: {md_path} not found, skipping")
            continue

        dest_dir = docs_dir / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        md_content = src.read_text(encoding="utf-8")
        html_content = render_markdown_to_html(md_content)

        full_html = build_page(
            title=doc_title,
            content=html_content,
            config=config,
            version=version,
            current_section=f"docs/{slug}",
            mode="archive",
            build_time=build_time,
        )

        (dest_dir / "index.html").write_text(full_html, encoding="utf-8")
        print(f"  Rendered {md_path} -> docs/{slug}/index.html")

    # Copy fixtures
    fixture_data = []
    fixtures_dir = config.get("fixtures_dir", "fixtures")
    fixtures_src = REPO_ROOT / fixtures_dir
    if fixtures_src.exists():
        fixtures_dest = version_dir / "fixtures"
        if fixtures_dest.exists():
            shutil.rmtree(fixtures_dest)

        # Copy fixture directories (skip golden_evidence_pack, handled separately)
        for fixture_dir in sorted(fixtures_src.iterdir()):
            if fixture_dir.is_dir() and fixture_dir.name != "golden_evidence_pack":
                dest = fixtures_dest / fixture_dir.name
                shutil.copytree(fixture_dir, dest)

                fixture_entry = {"name": fixture_dir.name, "files": []}
                for f in sorted(dest.glob("*.json")):
                    fixture_entry["files"].append({
                        "name": f.name,
                        "sha256": sha256_file(f),
                    })
                fixture_data.append(fixture_entry)

        # Write fixtures index.json
        (fixtures_dest / "index.json").write_text(
            json.dumps({"version": version, "fixtures": fixture_data}, indent=2),
            encoding="utf-8",
        )

        # Write fixtures index.html
        fixtures_html = build_fixtures_index(config, version, fixture_data, build_time)
        (fixtures_dest / "index.html").write_text(fixtures_html, encoding="utf-8")
        print(f"  Copied fixtures ({len(fixture_data)} cases)")

    # Copy evidence pack
    evidence_files = []
    golden_src = REPO_ROOT / "fixtures" / "golden_evidence_pack"
    if golden_src.exists():
        evidence_dest = version_dir / "evidence-pack"
        evidence_dest.mkdir(parents=True, exist_ok=True)

        for f in golden_src.glob("*.json"):
            shutil.copy(f, evidence_dest / f.name)
            evidence_files.append(f.name)

        evidence_html = build_evidence_pack_page(config, version, evidence_files, build_time)
        (evidence_dest / "index.html").write_text(evidence_html, encoding="utf-8")
        print(f"  Copied evidence pack ({len(evidence_files)} files)")

    # Generate verifier page with self-test
    verify_dir = version_dir / "evidence-pack" / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    vectors_file = REPO_ROOT / "releases" / f"evidence_pack_verifier_vectors.{version}.json"
    has_vectors = vectors_file.exists()
    vectors_sha256 = None
    if has_vectors:
        shutil.copy(vectors_file, verify_dir / "vectors.json")
        vectors_sha256 = sha256_file(vectors_file)
        print(f"  Copied vectors.json (sha256: {vectors_sha256[:16]}...)")
    verifier_html = build_verifier_page(config, version, has_vectors)
    (verify_dir / "index.html").write_text(verifier_html, encoding="utf-8")
    print(f"  Generated verify/index.html (self-test: {has_vectors})")

    # Generate manifest.json with file checksums
    all_files = collect_file_checksums(version_dir, version_dir)

    manifest = {
        "version": version,
        "tag": config["tag"],
        "commit": config["commit"],
        "date_locked": config["date_locked"],
        "status": config["status"],
        "demo_url": config.get("demo_url"),
        "invariants": {
            "tier_a": config["invariants"]["tier_a"],
            "tier_b": config["invariants"]["tier_b"],
            "tier_c": config["invariants"]["tier_c"],
        },
        "build_time": build_time,
        "build_commit": get_current_commit(),
        "source": "releases/releases.json",
        "files": all_files,
    }

    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  Generated manifest.json ({len(all_files)} files)")

    return manifest


def build_root_files(releases: dict, build_time: str) -> None:
    """Build root-level site files."""
    print("Building root files...")

    current_version = releases["current_version"]
    versions = releases["versions"]

    # _redirects (Cloudflare Pages format)
    redirects_lines = [
        f"/  /{current_version}/  302",  # Root -> current version
    ]
    for v in versions:
        # Trailing slash normalization
        redirects_lines.append(f"/{v}  /{v}/  301")

    (SITE_DIR / "_redirects").write_text("\n".join(redirects_lines) + "\n", encoding="utf-8")
    print(f"  Created _redirects (root -> /{current_version}/)")

    # _headers (Cloudflare Pages format)
    headers_lines = [
        "# Versioned content: immutable (1 year cache)",
    ]
    for v in versions:
        headers_lines.extend([
            f"/{v}/*",
            "  Cache-Control: public, max-age=31536000, immutable",
            "",
        ])
    headers_lines.extend([
        "# Root: short cache (redirect target may change)",
        "/",
        "  Cache-Control: public, max-age=300",
        "",
        "# Version index: medium cache",
        "/versions/*",
        "  Cache-Control: public, max-age=3600",
    ])

    (SITE_DIR / "_headers").write_text("\n".join(headers_lines) + "\n", encoding="utf-8")
    print("  Created _headers")

    # /versions/index.html
    versions_dir = SITE_DIR / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    version_list = []
    for v, config in sorted(versions.items()):
        version_list.append({
            "version": v,
            "status": config["status"],
            "date_locked": config["date_locked"],
            "commit": config["commit"],
            "demo_url": config.get("demo_url"),
        })

    versions_html = build_versions_index(version_list, build_time)
    (versions_dir / "index.html").write_text(versions_html, encoding="utf-8")
    print("  Created versions/index.html")


def verify_build(releases: dict) -> bool:
    """Verify built site matches releases.json. Returns True if valid."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Checking build against releases/releases.json")
    print("=" * 60)

    errors = []
    current_version = releases["current_version"]

    # 1. Check _redirects targets current version
    redirects_file = SITE_DIR / "_redirects"
    if not redirects_file.exists():
        errors.append("_redirects file missing")
    else:
        redirects_content = redirects_file.read_text()
        expected_redirect = f"/  /{current_version}/  302"
        if expected_redirect not in redirects_content:
            errors.append(
                f"_redirects root redirect mismatch:\n"
                f"  Expected: {expected_redirect}\n"
                f"  Got: {redirects_content.split(chr(10))[0]}"
            )
        else:
            print(f"[OK] _redirects: / -> /{current_version}/")

    # 2. Check each version manifest matches releases.json
    for version, expected in releases["versions"].items():
        manifest_file = SITE_DIR / version / "manifest.json"
        if not manifest_file.exists():
            errors.append(f"{version}/manifest.json missing")
            continue

        manifest = json.loads(manifest_file.read_text())

        # Check tag
        if manifest.get("tag") != expected["tag"]:
            errors.append(
                f"{version}: tag mismatch\n"
                f"  Expected: {expected['tag']}\n"
                f"  Got: {manifest.get('tag')}"
            )
        else:
            print(f"[OK] {version}: tag = {expected['tag']}")

        # Check commit
        if manifest.get("commit") != expected["commit"]:
            errors.append(
                f"{version}: commit mismatch\n"
                f"  Expected: {expected['commit']}\n"
                f"  Got: {manifest.get('commit')}"
            )
        else:
            print(f"[OK] {version}: commit = {expected['commit'][:12]}")

        # Check date_locked
        if manifest.get("date_locked") != expected["date_locked"]:
            errors.append(
                f"{version}: date_locked mismatch\n"
                f"  Expected: {expected['date_locked']}\n"
                f"  Got: {manifest.get('date_locked')}"
            )
        else:
            print(f"[OK] {version}: date_locked = {expected['date_locked']}")

        # Check status
        if manifest.get("status") != expected["status"]:
            errors.append(
                f"{version}: status mismatch\n"
                f"  Expected: {expected['status']}\n"
                f"  Got: {manifest.get('status')}"
            )
        else:
            print(f"[OK] {version}: status = {expected['status']}")

        # Check tier counts
        manifest_inv = manifest.get("invariants", {})
        expected_inv = expected["invariants"]
        for tier in ["tier_a", "tier_b", "tier_c"]:
            if manifest_inv.get(tier) != expected_inv[tier]:
                errors.append(
                    f"{version}: {tier} mismatch\n"
                    f"  Expected: {expected_inv[tier]}\n"
                    f"  Got: {manifest_inv.get(tier)}"
                )
            else:
                print(f"[OK] {version}: {tier} = {expected_inv[tier]}")

        # Check source field
        if manifest.get("source") != "releases/releases.json":
            errors.append(
                f"{version}: manifest missing 'source: releases/releases.json' field"
            )
        else:
            print(f"[OK] {version}: source = releases/releases.json")

        # Check files have sha256
        files = manifest.get("files", [])
        if not files:
            errors.append(f"{version}: manifest has no files")
        else:
            missing_sha = [f for f in files if "sha256" not in f]
            if missing_sha:
                errors.append(f"{version}: {len(missing_sha)} files missing sha256")
            else:
                print(f"[OK] {version}: {len(files)} files with sha256 checksums")

    # 3. Check _headers includes all versions
    headers_file = SITE_DIR / "_headers"
    if headers_file.exists():
        headers_content = headers_file.read_text()
        for version in releases["versions"]:
            if f"/{version}/*" not in headers_content:
                errors.append(f"_headers missing cache rule for /{version}/*")
            else:
                print(f"[OK] _headers: /{version}/* has immutable cache")

    # 4. Check verifier page exists for ALL versions (required)
    for version in releases["versions"]:
        verifier_path = SITE_DIR / version / "evidence-pack" / "verify" / "index.html"
        if not verifier_path.exists():
            errors.append(f"{version}: evidence-pack/verify/index.html MISSING (required)")
        else:
            # Check it's in manifest
            manifest_file = SITE_DIR / version / "manifest.json"
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                files = manifest.get("files", [])
                verifier_in_manifest = [
                    f for f in files
                    if f["path"] == "evidence-pack/verify/index.html"
                ]
                if not verifier_in_manifest:
                    errors.append(f"{version}: verify/index.html not in manifest")
                elif "sha256" not in verifier_in_manifest[0]:
                    errors.append(f"{version}: verify/index.html missing sha256")
                else:
                    print(f"[OK] {version}: verify/index.html exists (sha256 in manifest)")

    # 5. Check verifier self-test (vectors.json exists, sha256 in manifest, selftest-btn present)
    for version in releases["versions"]:
        vectors_src = REPO_ROOT / "releases" / f"evidence_pack_verifier_vectors.{version}.json"
        if vectors_src.exists():
            # vectors.json should be in verify dir
            vectors_dest = SITE_DIR / version / "evidence-pack" / "verify" / "vectors.json"
            if not vectors_dest.exists():
                errors.append(f"{version}: vectors.json missing from verify/")
            else:
                print(f"[OK] {version}: vectors.json present in verify/")

            # Check sha256 of vectors.json is in manifest
            manifest_file = SITE_DIR / version / "manifest.json"
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                files = manifest.get("files", [])
                vectors_in_manifest = [
                    f for f in files
                    if f["path"] == "evidence-pack/verify/vectors.json"
                ]
                if not vectors_in_manifest:
                    errors.append(f"{version}: vectors.json not in manifest files list")
                elif "sha256" not in vectors_in_manifest[0]:
                    errors.append(f"{version}: vectors.json missing sha256 in manifest")
                else:
                    print(f"[OK] {version}: vectors.json sha256 in manifest")

            # Check selftest-btn present in verify/index.html
            verifier_html = SITE_DIR / version / "evidence-pack" / "verify" / "index.html"
            if verifier_html.exists():
                html_content = verifier_html.read_text()
                if 'id="selftest-btn"' not in html_content:
                    errors.append(f"{version}: selftest-btn missing from verify/index.html")
                else:
                    print(f"[OK] {version}: selftest-btn present in verify/index.html")
            else:
                errors.append(f"{version}: verify/index.html missing")

    # 6. Check CURRENT version has /demo/ link
    current_version = releases.get("current_version")
    if current_version:
        current_landing = SITE_DIR / current_version / "index.html"
        if current_landing.exists():
            landing_content = current_landing.read_text()
            if 'href="/demo/"' in landing_content:
                print(f"[OK] {current_version}: /demo/ link present in landing page")
            else:
                errors.append(f"{current_version}: /demo/ link MISSING from landing page (required for current version)")
        else:
            errors.append(f"{current_version}: index.html missing")

    # 7. Check runbook has /demo routing documentation
    runbook_path = REPO_ROOT / "docs" / "CLOUDFLARE_DEPLOYMENT_RUNBOOK.md"
    if runbook_path.exists():
        runbook_content = runbook_path.read_text(encoding="utf-8")
        # Check /demo/healthz verification
        if "/demo/healthz" in runbook_content:
            print("[OK] Runbook: /demo/healthz verification documented")
        else:
            errors.append("Runbook: /demo/healthz verification commands MISSING")
        # Check DEMO_STRIP_PREFIX truth table
        if "DEMO_STRIP_PREFIX" in runbook_content and "Fly BASE_PATH" in runbook_content:
            print("[OK] Runbook: DEMO_STRIP_PREFIX truth table documented")
        else:
            errors.append("Runbook: DEMO_STRIP_PREFIX truth table MISSING")
        # Check X-Proxied-By verification (proves worker not intercepting archive)
        if "X-Proxied-By" in runbook_content:
            print("[OK] Runbook: X-Proxied-By header check documented")
        else:
            errors.append("Runbook: X-Proxied-By header verification MISSING")
    else:
        errors.append("Runbook: docs/CLOUDFLARE_DEPLOYMENT_RUNBOOK.md not found")

    print("=" * 60)
    if errors:
        print(f"VERIFICATION FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return False
    else:
        print("VERIFICATION PASSED: Build matches releases/releases.json")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Build static site for mathledger.ai (epistemic archive)"
    )
    parser.add_argument(
        "--version",
        help="Version to build (from releases/releases.json)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all versions from releases/releases.json",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean site/ directory before building",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing build matches releases/releases.json",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after build",
    )
    args = parser.parse_args()

    # Load canonical release metadata
    releases = load_releases()

    # Verify-only mode
    if args.verify:
        if not SITE_DIR.exists():
            print("ERROR: site/ directory does not exist. Run build first.")
            sys.exit(1)
        success = verify_build(releases)
        sys.exit(0 if success else 1)

    # Build mode
    if args.clean and SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
        print(f"Cleaned {SITE_DIR}")

    SITE_DIR.mkdir(parents=True, exist_ok=True)

    build_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.all:
        for v in releases["versions"]:
            build_version(releases, v, build_time)
    else:
        version = args.version or releases["current_version"]
        build_version(releases, version, build_time)

    build_root_files(releases, build_time)

    print(f"\nBuild complete: {SITE_DIR}")
    print(f"Current version: {releases['current_version']}")

    # Run verification unless skipped
    if not args.no_verify:
        if not verify_build(releases):
            print("\nBuild verification FAILED. See errors above.")
            sys.exit(1)

    print("\nTo deploy:")
    print("  wrangler pages deploy ./site --project-name mathledger-ai")
    print("  # or: git add site/ && git commit && git push origin site-deploy")


if __name__ == "__main__":
    main()
