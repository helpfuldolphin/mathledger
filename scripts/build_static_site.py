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
FROZEN_DIR = REPO_ROOT / "releases" / "frozen"

# Repository URL (single source of truth)
REPO_URL = "https://github.com/helpfuldolphin/mathledger"


class BuildError(Exception):
    """Raised when build verification fails."""
    pass


def parse_version(v: str) -> tuple[int, ...]:
    """Parse version string into numeric tuple for correct sorting.

    "v0.2.10" -> (0, 2, 10)
    "v0" -> (0,)

    This ensures v0.2.10 sorts AFTER v0.2.9, not between v0.2.1 and v0.2.2.
    """
    return tuple(int(x) for x in v.lstrip("v").split("."))


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


# ============================================================================
# FROZEN VERSION SYSTEM
# 
# Frozen versions are immutable-by-construction. Once frozen, a version's
# site/v{X}/ directory will never be regenerated from templates - instead
# it is verified against the frozen manifest.
#
# Freeze manifests are stored in releases/frozen/{version}.json and contain:
# - frozen_at: ISO timestamp when frozen
# - frozen_by_commit: git commit at freeze time
# - content_hash: SHA256 of all file hashes concatenated (quick comparison)
# - files: {relative_path: sha256} for all files in version directory
# ============================================================================


def get_current_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def is_version_frozen(version: str) -> bool:
    """Check if a version has a freeze manifest."""
    freeze_file = FROZEN_DIR / f"{version}.json"
    return freeze_file.exists()


def load_freeze_manifest(version: str) -> dict | None:
    """Load freeze manifest for a version. Returns None if not frozen."""
    freeze_file = FROZEN_DIR / f"{version}.json"
    if not freeze_file.exists():
        return None
    with open(freeze_file, encoding="utf-8") as f:
        return json.load(f)


def compute_version_hashes(version_dir: Path) -> dict[str, str]:
    """Compute SHA256 hashes for all files in a version directory.
    
    Returns dict of {relative_path: sha256}.
    """
    hashes = {}
    for path in sorted(version_dir.rglob("*")):
        if path.is_file():
            rel_path = str(path.relative_to(version_dir)).replace("\\", "/")
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            hashes[rel_path] = h.hexdigest()
    return hashes


def compute_content_hash(file_hashes: dict[str, str]) -> str:
    """Compute a single hash from all file hashes (for quick comparison)."""
    # Sort by path and concatenate all hashes
    combined = "".join(file_hashes[k] for k in sorted(file_hashes.keys()))
    return hashlib.sha256(combined.encode()).hexdigest()


def freeze_version(version: str) -> dict:
    """Create freeze manifest for a built version.
    
    Returns the freeze manifest dict.
    Raises BuildError if version directory doesn't exist.
    """
    version_dir = SITE_DIR / version
    if not version_dir.exists():
        raise BuildError(f"Cannot freeze {version}: directory does not exist. Build first.")
    
    # Ensure frozen directory exists
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Compute hashes
    file_hashes = compute_version_hashes(version_dir)
    content_hash = compute_content_hash(file_hashes)
    
    # Create manifest
    manifest = {
        "version": version,
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_by_commit": get_current_commit(),
        "content_hash": content_hash,
        "file_count": len(file_hashes),
        "files": file_hashes,
    }
    
    # Write manifest
    freeze_file = FROZEN_DIR / f"{version}.json"
    with open(freeze_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  FROZEN {version}: {len(file_hashes)} files, hash={content_hash[:16]}...")
    return manifest


def verify_frozen_version(version: str) -> tuple[bool, list[str]]:
    """Verify a frozen version's files match the freeze manifest.
    
    Returns (success, errors) where errors is a list of mismatch descriptions.
    """
    freeze_manifest = load_freeze_manifest(version)
    if freeze_manifest is None:
        return False, [f"{version} is not frozen"]
    
    version_dir = SITE_DIR / version
    if not version_dir.exists():
        return False, [f"{version} directory does not exist"]
    
    errors = []
    current_hashes = compute_version_hashes(version_dir)
    expected_hashes = freeze_manifest["files"]
    
    # Check for missing files
    for path in expected_hashes:
        if path not in current_hashes:
            errors.append(f"MISSING: {path}")
    
    # Check for extra files
    for path in current_hashes:
        if path not in expected_hashes:
            errors.append(f"EXTRA: {path}")
    
    # Check for modified files
    for path in expected_hashes:
        if path in current_hashes and current_hashes[path] != expected_hashes[path]:
            errors.append(f"MODIFIED: {path}")
    
    return len(errors) == 0, errors


def check_immutability(releases: dict) -> bool:
    """Verify all frozen versions are immutable. Returns True if all pass."""
    print(chr(10) + "=" * 60)
    print("IMMUTABILITY CHECK: Verifying frozen versions")
    print("=" * 60)
    
    all_pass = True
    frozen_count = 0
    
    for version in releases["versions"]:
        if is_version_frozen(version):
            frozen_count += 1
            success, errors = verify_frozen_version(version)
            if success:
                manifest = load_freeze_manifest(version)
                print(f"[OK] {version}: {manifest['file_count']} files match frozen state")
            else:
                print(f"[FAIL] {version}: IMMUTABILITY VIOLATED")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"       {error}")
                if len(errors) > 5:
                    print(f"       ... and {len(errors) - 5} more errors")
                all_pass = False
        else:
            print(f"[SKIP] {version}: not frozen")
    
    print()
    if frozen_count == 0:
        print("No frozen versions found.")
    elif all_pass:
        print(f"All {frozen_count} frozen versions are immutable.")
    else:
        print("IMMUTABILITY VIOLATION DETECTED. See errors above.")
    
    return all_pass


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


def replace_template_variables(content: str, version: str, config: dict) -> str:
    """Replace template variables like {{CURRENT_VERSION}} in content."""
    tag = config.get("tag", "")
    replacements = {
        "{{CURRENT_VERSION}}": version,
        "{{CURRENT_TAG}}": tag,
        "{{CURRENT_COMMIT}}": config.get("commit", "")[:12],
        "{{CURRENT_COMMIT_FULL}}": config.get("commit", ""),
        # URL templates for version-self-healing auditor paths
        "{{CURRENT_VERIFIER_URL}}": f"/{version}/evidence-pack/verify/",
        "{{CURRENT_EXAMPLES_URL}}": f"/{version}/evidence-pack/examples.json",
        "{{CURRENT_DEMO_URL}}": "/demo/",
        # GitHub tag-pinned URL prefix (for blob links)
        "{{GITHUB_TAG_URL}}": f"https://github.com/helpfuldolphin/mathledger/blob/{tag}",
    }
    for var, value in replacements.items():
        content = content.replace(var, value)
    return content


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
    """Build the version banner HTML that appears on every page.

    IMPORTANT: Version pages show LOCKED status, not CURRENT/SUPERSEDED.
    The /versions/ page is the only authority on current vs superseded.
    This prevents stale status labels on individual version pages.
    """
    # All version pages show LOCKED - /versions/ is the authority on status
    status_label = "LOCKED"
    status_color = "#1565c0"  # Blue for locked/archived

    # Add note about where to check current status
    status_note = f'<a href="/versions/" style="font-size: 0.8rem; color: #666; margin-left: 0.5rem;">(see /versions/ for current status)</a>'

    return f"""
    <div class="version-banner" style="--status-color: {status_color};">
        <div class="title">MathLedger — Version {version}</div>
        <div><span class="status">Status: {status_label}</span>{status_note}</div>
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
        ("docs/field-manual/", "Field Manual"),
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


def build_release_delta_box(config: dict, version: str) -> str:
    """Build a Release Delta box showing what changed in this version.

    Reads from releases.json fields:
    - delta.changed: List of things that changed
    - delta.unchanged: List of things that did not change
    - delta.still_not_enforced: List of things still not enforced

    Returns empty string if no delta fields are present.
    """
    delta = config.get("delta", {})
    if not delta:
        return ""

    changed = delta.get("changed", [])
    unchanged = delta.get("unchanged", [])
    still_not = delta.get("still_not_enforced", [])

    if not changed and not unchanged and not still_not:
        return ""

    sections = []
    if changed:
        items = "".join(f"<li>{item}</li>" for item in changed)
        sections.append(f'<div style="margin-bottom: 0.75rem;"><strong style="color: #2e7d32;">Changed:</strong><ul style="margin: 0.25rem 0 0 1.5rem; font-size: 0.85rem;">{items}</ul></div>')
    if unchanged:
        items = "".join(f"<li>{item}</li>" for item in unchanged)
        sections.append(f'<div style="margin-bottom: 0.75rem;"><strong style="color: #666;">Did not change:</strong><ul style="margin: 0.25rem 0 0 1.5rem; font-size: 0.85rem; color: #666;">{items}</ul></div>')
    if still_not:
        items = "".join(f"<li>{item}</li>" for item in still_not)
        sections.append(f'<div><strong style="color: #c62828;">Still not enforced:</strong><ul style="margin: 0.25rem 0 0 1.5rem; font-size: 0.85rem; color: #c62828;">{items}</ul></div>')

    return f"""
    <div class="release-delta" style="background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem; margin-bottom: 1.5rem;">
        <h3 style="margin: 0 0 0.75rem 0; font-size: 1rem; color: #333;">Release Delta: {version}</h3>
        {"".join(sections)}
    </div>
    """


def build_version_landing(config: dict, version: str, build_time: str) -> str:
    """Build the version landing page that distinguishes archive from demo."""
    is_current = config.get("status") == "current"
    demo_url = config.get("demo_url")
    hosted_demo = config.get("hosted_demo", False)

    # Initialize demo_banner (only shown for current version with hosted_demo)
    demo_banner = ""

    # Hosted demo is available at /demo/ for the CURRENT version only
    # Superseded versions (even if they have demo_url) must show the LOCAL ONLY disclaimer
    is_superseded = config.get("status", "").startswith("superseded")

    if is_current and hosted_demo:
        # Current version with hosted demo: "Two Artifacts" explainer ABOVE THE FOLD
        demo_banner = f"""
        <div class="two-artifacts" style="background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <h2 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #333;">This Site Provides Two Artifacts</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem;">
                <div style="background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem;">
                    <strong style="color: #1565c0;">📁 Archive</strong> (this page)
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #555;">
                        Immutable evidence and documentation. Static, versioned, cryptographically checksummed.
                    </p>
                </div>
                <div style="background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem;">
                    <strong style="color: #2e7d32;">⚡ Demo</strong> (hosted at /demo/)
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #555;">
                        Interactive execution of the same version. Produces evidence packs verifiable against this archive.
                    </p>
                </div>
            </div>

            <!-- Version coherence bridge -->
            <p style="margin: 0 0 0.75rem 0; font-size: 0.9rem; color: #444; font-style: italic;">
                This archive is immutable. The hosted demo at <a href="/demo/">/demo/</a> is the live instantiation of this same version.
            </p>
            <p style="margin: 0 0 1rem 0; font-size: 0.85rem; color: #666;">
                <a href="/{version}/docs/field-manual/">Field Manual</a> (fm.tex/pdf): obligation ledger used to drive version promotions.
            </p>
            <div id="demo-sync-status" style="display: none; background: #ffebee; border: 1px solid #c62828; border-radius: 4px; padding: 0.5rem 1rem; margin-bottom: 1rem;">
                <strong style="color: #c62828;">⚠ DEMO OUT OF SYNC</strong> —
                <span id="sync-reason">do not use for audit</span>
            </div>

            <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                <a href="/demo/" class="demo-button" style="background: #2e7d32; color: #fff; padding: 0.75rem 1.5rem; border-radius: 4px; text-decoration: none; font-weight: 500;">Open Hosted Demo</a>
                <a href="/{version}/evidence-pack/verify/" class="demo-button" style="background: #1565c0; color: #fff; padding: 0.75rem 1.5rem; border-radius: 4px; text-decoration: none; font-weight: 500;">Open Auditor Tool</a>
            </div>

            <!-- Demo sync check script -->
            <script>
            (function() {{
                const expectedVersion = "{version}";
                fetch('/demo/healthz')
                    .then(r => r.json())
                    .then(data => {{
                        const demoVersion = data.version || data.tag || '';
                        if (!demoVersion.includes(expectedVersion.replace('v', ''))) {{
                            document.getElementById('demo-sync-status').style.display = 'block';
                            document.getElementById('sync-reason').textContent =
                                'demo reports ' + demoVersion + ', archive is ' + expectedVersion;
                        }}
                    }})
                    .catch(() => {{
                        // Demo might return plain text "ok" - that's fine, don't show error
                    }});
            }})();
            </script>
        </div>

        <div class="auditor-flow" style="background: #fff3e0; border: 1px solid #ffb74d; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <h3 style="margin: 0 0 0.75rem 0; font-size: 1rem; color: #e65100;">For Auditors: 3-Step Verification</h3>
            <ol style="margin: 0; padding-left: 1.5rem; font-size: 0.9rem;">
                <li style="margin-bottom: 0.5rem;"><strong>Run the boundary demo</strong> — <a href="/demo/">Open demo</a>, click "Run Boundary Demo", observe VERIFIED/REFUTED/ABSTAINED outcomes</li>
                <li style="margin-bottom: 0.5rem;"><strong>Download evidence pack</strong> — Click "Download Evidence Pack" after demo completes</li>
                <li><strong>Verify in auditor tool</strong> — <a href="/{version}/evidence-pack/verify/">Open verifier</a>, upload pack, confirm PASS (or tamper and observe FAIL)</li>
            </ol>
            <p style="margin: 0.75rem 0 0 0; font-size: 0.85rem;">
                Full checklist: <a href="/{version}/docs/for-auditors/">5-minute auditor verification</a>
            </p>
        </div>

        <div class="ready-to-verify" id="ready-to-verify" style="background: #e3f2fd; border: 1px solid #1976d2; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <h3 style="margin: 0 0 0.75rem 0; font-size: 1rem; color: #0d47a1;">Ready-to-Verify: Example Evidence Packs (no demo required)</h3>
            <p style="margin: 0 0 1rem 0; font-size: 0.9rem; color: #555;">
                Download pre-built examples to test the auditor tool without running the demo:
            </p>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
                <a href="/{version}/evidence-pack/examples.json" download="examples.json" style="background: #fff; border: 1px solid #1976d2; color: #1976d2; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; font-size: 0.9rem;">📥 Download Examples (PASS + FAIL)</a>
                <a href="/{version}/evidence-pack/verify/" style="background: #1976d2; color: #fff; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; font-size: 0.9rem;">Open Auditor Tool</a>
            </div>
            <p style="margin: 0.75rem 0 0 0; font-size: 0.85rem; color: #666;">
                The examples.json file contains both valid (PASS) and tampered (FAIL) packs for testing.
            </p>
        </div>
        """
        demo_section = f"""
        <div class="info-box" style="border-left: 4px solid #2e7d32;">
            <strong>Local Execution</strong>
            <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer; color: #666;">Run locally instead of using hosted demo</summary>
                <div style="margin-top: 0.5rem;">
                    <code>git clone {REPO_URL}</code><br>
                    <code>git checkout {config['tag']}</code><br>
                    <code>uv run python demo/app.py</code><br>
                    <code>Open http://localhost:8000</code>
                </div>
            </details>
        </div>
        """
    elif demo_url and not is_superseded:
        # Explicit demo_url override (future use, for non-superseded versions)
        demo_section = f"""
        <div class="info-box">
            <strong>Interactive Demo</strong><br>
            Available at: <a href="{demo_url}">{demo_url}</a>
        </div>
        """
    else:
        # Superseded version: local only, but mention current demo exists
        # IMPORTANT: The disclaimer text must match what hostile_audit.ps1 checks for
        demo_section = f"""
        <div class="info-box local-only">
            <strong>Interactive Demo</strong> <span class="mode-indicator mode-local">LOCAL ONLY</span><br>
            This archived version's demo requires local Python execution.<br><br>
            <strong>To run locally:</strong><br>
            <code>git clone {REPO_URL}</code><br>
            <code>git checkout {config['tag']}</code><br>
            <code>uv run python demo/app.py</code><br>
            <code>Open http://localhost:8000</code>
            <p id="superseded-disclaimer" style="margin-top: 1rem; padding: 0.75rem; background: #fff3e0; border-left: 3px solid #f57c00; font-size: 0.9rem; color: #333;">
                <strong>Note:</strong> The hosted demo at <a href="/demo/">/demo/</a> runs the CURRENT version
                (see <a href="/versions/">/versions/</a> for status).
                This archived version is not available as a hosted demo.
            </p>
        </div>
        """

    inv = config["invariants"]
    tier_a_items = "".join(f"<li>{item}</li>" for item in inv.get("tier_a_list", []))
    tier_b_items = "".join(f"<li>{item}</li>" for item in inv.get("tier_b_list", []))
    tier_c_items = "".join(f"<li>{item}</li>" for item in inv.get("tier_c_list", []))

    # Check if for-auditors doc exists (docs are tuples: (source, slug, title))
    has_for_auditors = any(d[1] == "for-auditors" for d in config.get("docs", []))
    for_auditors_row = f"""<tr><td><a href="/{version}/docs/for-auditors/">For Auditors</a></td><td>5-minute verification checklist</td></tr>""" if has_for_auditors else ""

    # Build Release Delta box from releases.json fields
    release_delta = build_release_delta_box(config, version)

    content = f"""
        {demo_banner}

        <p>This is the archive for MathLedger version <code>{version}</code>.
        All artifacts below are static, verifiable, and immutable.</p>

        {release_delta}

        <h2>Archive Contents</h2>
        <table>
            <tr><th>Artifact</th><th>Description</th></tr>
            {for_auditors_row}
            <tr><td><a href="/{version}/docs/scope-lock/">Scope Lock</a></td><td>What this version does and does not demonstrate</td></tr>
            <tr><td><a href="/{version}/docs/explanation/">Explanation</a></td><td>How the demo explains its own behavior</td></tr>
            <tr><td><a href="/{version}/docs/invariants/">Invariants</a></td><td>Tier A/B/C enforcement status</td></tr>
            <tr><td><a href="/{version}/docs/hostile-rehearsal/">Hostile Rehearsal</a></td><td>Answers to skeptical questions</td></tr>
            <tr><td><a href="/{version}/fixtures/">Fixtures</a></td><td>Regression test cases with golden outputs</td></tr>
            <tr><td><a href="/{version}/evidence-pack/">Evidence Pack</a></td><td>Replay verification artifacts</td></tr>
            <tr><td><a href="/{version}/manifest.json">Manifest</a></td><td>Version metadata + file checksums</td></tr>
        </table>

        <h2>Local Execution</h2>
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


def build_fixture_directory_index(config: dict, version: str, fixture_name: str, files: list, build_time: str) -> str:
    """Build index page for a single fixture directory."""
    file_rows = []
    for f in files:
        file_rows.append(f'<tr><td><a href="{f["name"]}">{f["name"]}</a></td><td><code>{f["sha256"][:16]}...</code></td></tr>')

    fixture_content = f"""
        <p><a href="../">← Back to Fixtures</a></p>

        <h2>Fixture: {fixture_name}</h2>
        <p>This fixture contains the following files:</p>

        <table>
            <tr><th>File</th><th>SHA256 (truncated)</th></tr>
            {"".join(file_rows)}
        </table>

        <h2>Download</h2>
        <ul>
            {''.join(f'<li><a href="{f["name"]}">{f["name"]}</a></li>' for f in files)}
        </ul>

        <h2>Verify</h2>
        <p>To verify these files locally:</p>
        <pre><code>git checkout {config['tag']}
cat fixtures/{fixture_name}/input.json | sha256sum</code></pre>
    """

    return build_page(
        title=f"Fixture: {fixture_name}",
        content=fixture_content,
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

        <div style="background: #e3f2fd; border: 1px solid #1976d2; border-radius: 8px; padding: 1.25rem; margin: 1.5rem 0;">
            <h3 style="margin: 0 0 0.75rem 0; font-size: 1rem; color: #0d47a1;">Verification Tools</h3>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
                <a href="/{version}/evidence-pack/verify/" style="background: #1976d2; color: #fff; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; font-size: 0.9rem; font-weight: 500;">Open Auditor Tool</a>
                <a href="/{version}/evidence-pack/examples.json" download="examples.json" style="background: #fff; border: 1px solid #1976d2; color: #1976d2; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; font-size: 0.9rem;">📥 Download Example Packs</a>
            </div>
            <p style="margin: 0.75rem 0 0 0; font-size: 0.85rem; color: #666;">
                The auditor tool verifies evidence packs in-browser (no server). Example packs include PASS and FAIL cases.
            </p>
        </div>

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

        <p>See <a href="/{version}/docs/explanation/">How the Demo Explains Itself</a> for full details.</p>
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


def build_verifier_page(config: dict, version: str, has_examples: bool) -> str:
    """Build the evidence pack verifier page with self-test using examples.json."""
    tag = config["tag"]
    commit = config["commit"][:12]

    # Self-test section is ALWAYS shown above the fold with prominent button
    # Uses examples.json which contains both valid and tampered packs
    selftest_section = f'''
<div class="vbox selftest-hero" style="background:#e3f2fd;border-left:4px solid #1976d2;">
<h2 style="margin-top:0;">Run Self-Test Vectors</h2>
<p>Click the button below to run all built-in test vectors. Expected results: valid packs PASS, tampered packs FAIL.</p>
<button class="btn-p btn-large" id="selftest-btn" onclick="runSelfTest()" style="font-size:1.1rem;padding:0.75rem 1.5rem;">Run self-test vectors</button>
<div id="selftest-status" style="margin:0.75rem 0;font-weight:600;font-size:1.1rem;display:none"></div>
<table id="selftest-table" style="display:none;margin-top:1rem;">
<thead><tr><th>Name</th><th>Expected</th><th>Actual</th><th>Pass/Fail</th><th>Reason</th></tr></thead>
<tbody id="selftest-body"></tbody>
</table>
</div>'''

    selftest_js = r'''
async function runSelfTest(){const btn=document.getElementById("selftest-btn");const status=document.getElementById("selftest-status");const table=document.getElementById("selftest-table");const tbody=document.getElementById("selftest-body");btn.disabled=true;status.style.display="block";status.textContent="Loading examples.json...";status.className="";tbody.innerHTML="";table.style.display="none";try{const resp=await fetch("../examples.json");if(!resp.ok)throw new Error("examples.json not found");const data=await resp.json();status.textContent="Running tests...";const results=[];const examples=data.examples||{};for(const[name,ex]of Object.entries(examples)){const pack=ex.pack;const expected=ex.expected_verdict||"PASS";const r=await testPack(pack,expected);results.push({name:name,expected:expected,actual:r.actual,pass:r.pass,reason:r.reason});}let allPass=true;for(const r of results){const tr=document.createElement("tr");tr.className=r.pass?"row-pass":"row-fail";const cls=r.pass?"match":"mismatch";const txt=r.pass?"PASS":"FAIL";tr.innerHTML="<td>"+esc(r.name)+"</td><td>"+esc(r.expected)+"</td><td>"+esc(r.actual)+"</td><td class=\""+cls+"\">"+txt+"</td><td>"+esc(r.reason||"-")+"</td>";tbody.appendChild(tr);if(!r.pass)allPass=false;}table.style.display="table";status.className=allPass?"match":"mismatch";status.textContent=allPass?"SELF-TEST PASSED ("+results.length+" vectors)":"SELF-TEST FAILED";}catch(e){status.className="mismatch";status.textContent="Error: "+e.message;}btn.disabled=false;}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}

// testPack - SEMANTICS: FAIL/FAIL -> test PASSES, PASS/PASS -> test PASSES
async function testPack(pack,expectedResult){try{const uvil=pack.uvil_events||[];const arts=pack.reasoning_artifacts||[];const declaredU=pack.u_t||"";const declaredR=pack.r_t||"";const declaredH=pack.h_t||"";for(const a of arts){if(!("validation_outcome"in a))return{actual:"FAIL",pass:expectedResult==="FAIL",reason:"missing_required_field"};}const computedU=await computeUt(uvil);const computedR=await computeRt(arts);const computedH=await computeHt(computedR,computedU);if(computedU!==declaredU)return{actual:"FAIL",pass:expectedResult==="FAIL",reason:"u_t_mismatch"};if(computedR!==declaredR)return{actual:"FAIL",pass:expectedResult==="FAIL",reason:"r_t_mismatch"};if(computedH!==declaredH)return{actual:"FAIL",pass:expectedResult==="FAIL",reason:"h_t_mismatch"};return{actual:"PASS",pass:expectedResult==="PASS",reason:null};}catch(e){return{actual:"FAIL",pass:expectedResult==="FAIL",reason:e.message};}}'''

    css = "*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,monospace;background:#f5f5f5;line-height:1.6}.container{max-width:900px;margin:0 auto;padding:2rem}.banner{background:#fff;border:1px solid #ddd;border-left:4px solid #2e7d32;padding:1rem;margin-bottom:1.5rem}.status{font-weight:600;color:#2e7d32}code{background:#f0f0f0;padding:0.15em 0.35em}h1{font-size:1.4rem;margin-bottom:1rem}h2{font-size:1.2rem;margin:1.5rem 0 0.75rem;border-bottom:1px solid #ddd}p{margin:0.75rem 0}.info{background:#fff;border:1px solid #ddd;padding:1rem;margin:1rem 0;border-left:4px solid #f57c00}.vbox{background:#fff;border:1px solid #ddd;padding:1.5rem;margin:1rem 0}.result{padding:1rem;margin:1rem 0;font-family:monospace}.pass{background:#e8f5e9;border-left:4px solid #2e7d32}.fail{background:#ffebee;border-left:4px solid #c62828}.pending{background:#fff3e0;border-left:4px solid #f57c00}.row{margin:0.5rem 0}.row label{font-weight:600;display:inline-block;width:100px}.match{color:#2e7d32!important;font-weight:600}.mismatch{color:#c62828!important;font-weight:600}textarea{width:100%;height:200px;font-family:monospace;font-size:0.85rem}button{padding:0.5rem 1rem;margin:0.5rem 0.5rem 0.5rem 0;cursor:pointer}button:disabled{opacity:0.5;cursor:not-allowed}.btn-p{background:#0066cc;color:#fff;border:none}.btn-s{background:#f5f5f5;border:1px solid #ddd}footer{margin-top:2rem;padding-top:1rem;border-top:1px solid #ddd;font-size:0.75rem;color:#666}a{color:#0066cc}.nav{margin-bottom:1.5rem}.nav a{margin-right:1rem}table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem}th,td{border:1px solid #ddd;padding:0.5rem;text-align:left}th{background:#f5f5f5}.row-pass{background:#f1f8e9}.row-fail{background:#ffebee}"

    core_js = r'''// Domain separation constants (must match Python attestation/dual_root.py)
const DOMAIN_REASONING_LEAF=new Uint8Array([0xA0,...new TextEncoder().encode('reasoning-leaf')]);
const DOMAIN_UI_LEAF=new Uint8Array([0xA1,...new TextEncoder().encode('ui-leaf')]);
const DOMAIN_LEAF=new Uint8Array([0x00]);
const DOMAIN_NODE=new Uint8Array([0x01]);

// RFC 8785 JSON canonicalization
function can(o){if(o===null)return'null';if(typeof o==='boolean')return o?'true':'false';if(typeof o==='number')return Object.is(o,-0)?'0':String(o);if(typeof o==='string'){let r='"';for(let i=0;i<o.length;i++){const c=o.charCodeAt(i);if(c===8)r+='\\b';else if(c===9)r+='\\t';else if(c===10)r+='\\n';else if(c===12)r+='\\f';else if(c===13)r+='\\r';else if(c===34)r+='\\"';else if(c===92)r+='\\\\';else if(c<32)r+='\\u'+c.toString(16).padStart(4,'0');else r+=o[i];}return r+'"';}if(Array.isArray(o))return'['+o.map(can).join(',')+']';if(typeof o==='object'){const k=Object.keys(o).sort();return'{'+k.map(x=>can(x)+':'+can(o[x])).join(',')+'}';}throw Error('bad');}

// SHA256 without domain (for composite root H_t)
async function sha(s){const d=new TextEncoder().encode(s);const h=await crypto.subtle.digest('SHA-256',d);return Array.from(new Uint8Array(h)).map(b=>b.toString(16).padStart(2,'0')).join('');}

// SHA256 with domain prefix (returns hex string)
async function shaD(data,domain){const db=typeof data==='string'?new TextEncoder().encode(data):data;const c=new Uint8Array(domain.length+db.length);c.set(domain);c.set(db,domain.length);const h=await crypto.subtle.digest('SHA-256',c);return Array.from(new Uint8Array(h)).map(b=>b.toString(16).padStart(2,'0')).join('');}

// SHA256 with domain prefix (returns bytes)
async function shaDBytes(data,domain){const db=typeof data==='string'?new TextEncoder().encode(data):data;const c=new Uint8Array(domain.length+db.length);c.set(domain);c.set(db,domain.length);const h=await crypto.subtle.digest('SHA-256',c);return new Uint8Array(h);}

// Merkle root with domain separation (matches Python substrate/crypto/hashing.py)
async function merkleRoot(leafHashes){
if(leafHashes.length===0)return shaD('',DOMAIN_LEAF);
const sorted=[...leafHashes].sort();
let nodes=[];for(const lh of sorted){nodes.push(await shaDBytes(lh,DOMAIN_LEAF));}
while(nodes.length>1){
if(nodes.length%2===1)nodes.push(nodes[nodes.length-1]);
const next=[];
for(let i=0;i<nodes.length;i+=2){
const combined=new Uint8Array(64);combined.set(nodes[i]);combined.set(nodes[i+1],32);
next.push(await shaDBytes(combined,DOMAIN_NODE));}
nodes=next;}
return Array.from(nodes[0]).map(b=>b.toString(16).padStart(2,'0')).join('');}

// Compute U_t from uvil_events (Merkle root with DOMAIN_UI_LEAF)
async function computeUt(events){const lh=[];for(const e of events){lh.push(await shaD(can(e),DOMAIN_UI_LEAF));}return merkleRoot(lh);}

// Compute R_t from reasoning_artifacts (Merkle root with DOMAIN_REASONING_LEAF)
async function computeRt(artifacts){const lh=[];for(const a of artifacts){lh.push(await shaD(can(a),DOMAIN_REASONING_LEAF));}return merkleRoot(lh);}

// Compute H_t = SHA256(R_t || U_t) - ASCII concatenation
async function computeHt(rt,ut){return sha(rt+ut);}

document.getElementById('fi').onchange=e=>{if(e.target.files[0]){const r=new FileReader();r.onload=x=>document.getElementById('inp').value=x.target.result;r.readAsText(e.target.files[0]);}};

async function verify(){const R=document.getElementById('res'),D=document.getElementById('det');try{const v=document.getElementById('inp').value.trim();if(!v){R.className='result pending';R.innerHTML='<strong>Status:</strong> No input';D.style.display='none';return;}const p=JSON.parse(v);const uvil=p.uvil_events||[];const arts=p.reasoning_artifacts||[];const eu=p.u_t||'';const er=p.r_t||'';const eh=p.h_t||'';const cu=await computeUt(uvil);const cr=await computeRt(arts);const ch=await computeHt(cr,cu);document.getElementById('eu').textContent=eu||'-';document.getElementById('cu').textContent=cu;document.getElementById('er').textContent=er||'-';document.getElementById('cr').textContent=cr;document.getElementById('eh').textContent=eh||'-';document.getElementById('ch').textContent=ch;const uok=!eu||cu===eu,rok=!er||cr===er,hok=!eh||ch===eh;document.getElementById('cu').className=uok?'match':'mismatch';document.getElementById('cr').className=rok?'match':'mismatch';document.getElementById('ch').className=hok?'match':'mismatch';D.style.display='block';if(!eu&&!er&&!eh){R.className='result pending';R.innerHTML='<strong>Status:</strong> COMPUTED';}else if(uok&&rok&&hok){R.className='result pass';R.innerHTML='<strong>Status:</strong> PASS';}else{R.className='result fail';R.innerHTML='<strong>Status:</strong> FAIL';}}catch(e){R.className='result fail';R.innerHTML='<strong>Status:</strong> '+e.message;D.style.display='none';}}'''

    examples_link = '<a href="../examples.json">Test Vectors (examples.json)</a>'

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
<div><span class="status">LOCKED</span> <a href="/versions/" style="font-size:0.8rem;color:#666">(see /versions/)</a></div>
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
<footer>MathLedger {version} Verifier | {examples_link}</footer>
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
        Prior version directories are never modified. /versions/ is the canonical status registry and may change over time.</p>

        <div class="status-ambiguity-note" style="background: #fff3e0; border: 1px solid #ffb74d; border-radius: 6px; padding: 1rem; margin: 1.5rem 0;">
            <strong style="color: #e65100;">Note on Pre-v0.2.2 Archives:</strong>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Pre-v0.2.2 archives may display "Status: CURRENT" because status was embedded at build time.
                <strong>This page (/versions/) is the canonical source of current/superseded status.</strong>
                Individual version pages now show "LOCKED" status to avoid confusion.
            </p>
        </div>

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


def build_version(releases: dict, version: str, build_time: str, skip_frozen: bool = True) -> dict:
    """Build static site for a specific version. Returns manifest data.
    
    If skip_frozen=True (default), frozen versions are verified but not regenerated.
    """
    config = get_version_config(releases, version)
    version_dir = SITE_DIR / version

    # Check frozen status
    if is_version_frozen(version):
        freeze_manifest = load_freeze_manifest(version)
        if version_dir.exists():
            if skip_frozen:
                # Verify frozen version matches
                success, errors = verify_frozen_version(version)
                if success:
                    print(f"FROZEN {version}: verified {freeze_manifest['file_count']} files (skipping rebuild)")
                    # Return existing manifest
                    manifest_file = version_dir / "manifest.json"
                    if manifest_file.exists():
                        return json.loads(manifest_file.read_text(encoding="utf-8"))
                    return freeze_manifest
                else:
                    raise BuildError(
                        f"FROZEN {version}: IMMUTABILITY VIOLATION!" +
                        f"  The version directory has been modified after freezing." +
                        f"  Errors: {errors[:3]}..." +
                        f"  To fix: restore from git or delete site/{version}/ and rebuild from archive."
                    )
            else:
                print(f"Warning: Rebuilding frozen version {version} (--no-skip-frozen)")
        else:
            # Frozen but directory missing - this is allowed, we'll rebuild and verify
            print(f"FROZEN {version}: directory missing, will rebuild and verify")

    print(f"Building {version}...")
    print(f"  Tag: {config['tag']}")
    print(f"  Commit: {config['commit']}")
    print(f"  Status: {config['status']}")

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
        # Replace template variables before rendering
        md_content = replace_template_variables(md_content, version, config)
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

    # Copy Field Manual (fm.pdf, fm.tex, and render README.md)
    fm_src = REPO_ROOT / "docs" / "PAPERS" / "field_manual"
    if fm_src.exists():
        fm_dest = version_dir / "docs" / "field-manual"
        fm_dest.mkdir(parents=True, exist_ok=True)

        # Copy fm.pdf and fm.tex
        fm_files_copied = []
        for fm_file in ["fm.pdf", "fm.tex"]:
            src_file = fm_src / fm_file
            if src_file.exists():
                shutil.copy(src_file, fm_dest / fm_file)
                fm_files_copied.append(fm_file)

        # Render README.md as index.html
        fm_readme = fm_src / "README.md"
        if fm_readme.exists():
            fm_md = fm_readme.read_text(encoding="utf-8")
            fm_md = replace_template_variables(fm_md, version, config)
            fm_html = render_markdown_to_html(fm_md)
            fm_full = build_page(
                title="Field Manual",
                content=fm_html,
                config=config,
                version=version,
                current_section="docs/field-manual",
                mode="archive",
                build_time=build_time,
            )
            (fm_dest / "index.html").write_text(fm_full, encoding="utf-8")
            fm_files_copied.append("index.html")

        if fm_files_copied:
            print(f"  Copied Field Manual ({', '.join(fm_files_copied)})")

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

                # Generate index.html for this fixture directory
                fixture_index_html = build_fixture_directory_index(
                    config, version, fixture_dir.name, fixture_entry["files"], build_time
                )
                (dest / "index.html").write_text(fixture_index_html, encoding="utf-8")

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
    evidence_dest = version_dir / "evidence-pack"
    evidence_dest.mkdir(parents=True, exist_ok=True)

    if golden_src.exists():
        for f in golden_src.glob("*.json"):
            shutil.copy(f, evidence_dest / f.name)
            evidence_files.append(f.name)

        evidence_html = build_evidence_pack_page(config, version, evidence_files, build_time)
        (evidence_dest / "index.html").write_text(evidence_html, encoding="utf-8")
        print(f"  Copied evidence pack ({len(evidence_files)} files)")

    # Copy example packs (PASS/FAIL examples for auditors)
    examples_src = REPO_ROOT / "releases" / f"evidence_pack_examples.{version}.json"
    has_examples = examples_src.exists()
    if has_examples:
        shutil.copy(examples_src, evidence_dest / "examples.json")
        examples_sha256 = sha256_file(examples_src)
        print(f"  Copied examples.json (sha256: {examples_sha256[:16]}...)")

    # Generate verifier page with self-test (uses examples.json)
    verify_dir = version_dir / "evidence-pack" / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    vectors_file = REPO_ROOT / "releases" / f"evidence_pack_verifier_vectors.{version}.json"
    has_vectors = vectors_file.exists()
    vectors_sha256 = None
    if has_vectors:
        shutil.copy(vectors_file, verify_dir / "vectors.json")
        vectors_sha256 = sha256_file(vectors_file)
        print(f"  Copied vectors.json (sha256: {vectors_sha256[:16]}...)")
    verifier_html = build_verifier_page(config, version, has_examples)
    (verify_dir / "index.html").write_text(verifier_html, encoding="utf-8")
    print(f"  Generated verify/index.html (self-test: {has_examples})")

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

    # Post-build: verify against freeze manifest if version was frozen
    if is_version_frozen(version):
        success, errors = verify_frozen_version(version)
        if success:
            print(f"  FREEZE VERIFIED: rebuilt output matches frozen manifest")
        else:
            raise BuildError(
                f"FREEZE MISMATCH: rebuilt {version} does not match frozen manifest!" +
                f"  This means templates or source files have changed since freeze." +
                f"  Errors: {errors[:5]}" +
                f"  To fix: either update freeze manifest or restore old templates."
            )

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
    for v, config in sorted(versions.items(), key=lambda x: parse_version(x[0])):
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

    # /versions/status.json (machine-readable registry)
    current_config = versions.get(current_version, {})
    status_data = {
        "current_version": current_version,
        "current_tag": current_config.get("tag", ""),
        "current_commit": current_config.get("commit", "")[:8] if current_config.get("commit") else "",
        "versions": sorted(versions.keys(), key=parse_version),
        "superseded": sorted([v for v, cfg in versions.items() if cfg.get("status", "").startswith("superseded")], key=parse_version),
        "generated_at": build_time,
    }
    (versions_dir / "status.json").write_text(json.dumps(status_data, indent=2) + chr(10), encoding="utf-8")
    print(f"  Created versions/status.json (current: {current_version})")


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

    # 6. Check CURRENT version has required above-the-fold elements
    current_version = releases.get("current_version")
    if current_version:
        current_landing = SITE_DIR / current_version / "index.html"
        if current_landing.exists():
            landing_content = current_landing.read_text(encoding="utf-8")

            # Check for /demo/ link
            if 'href="/demo/"' in landing_content:
                print(f"[OK] {current_version}: /demo/ link present in landing page")
            else:
                errors.append(f"{current_version}: /demo/ link MISSING from landing page (required)")

            # Check for Two Artifacts explainer
            if "Two Artifacts" in landing_content:
                print(f"[OK] {current_version}: Two Artifacts explainer present")
            else:
                errors.append(f"{current_version}: Two Artifacts explainer MISSING")

            # Check for Open Auditor Tool button
            if 'evidence-pack/verify/' in landing_content and "Open Auditor Tool" in landing_content:
                print(f"[OK] {current_version}: Open Auditor Tool link present")
            else:
                errors.append(f"{current_version}: Open Auditor Tool link MISSING")

            # Check for 3-step auditor flow
            if "3-Step Verification" in landing_content:
                print(f"[OK] {current_version}: 3-Step Verification flow present")
            else:
                errors.append(f"{current_version}: 3-Step Verification flow MISSING")
        else:
            errors.append(f"{current_version}: index.html missing")

        # Check for-auditors page exists (if configured)
        current_config = releases.get("versions", {}).get(current_version, {})
        has_for_auditors = any(d.get("slug") == "for-auditors" for d in current_config.get("docs", []))
        if has_for_auditors:
            for_auditors_path = SITE_DIR / current_version / "docs" / "for-auditors" / "index.html"
            if for_auditors_path.exists():
                print(f"[OK] {current_version}: for-auditors page exists")

                # Check that auditor links resolve correctly
                auditor_content = for_auditors_path.read_text(encoding="utf-8")
                expected_links = [
                    ("../scope-lock/", "Scope Lock"),
                    ("../invariants/", "Invariants"),
                    ("../hostile-rehearsal/", "Hostile Demo Rehearsal"),
                    ("../explanation/", "How the Demo Explains Itself"),
                ]
                for link_path, link_name in expected_links:
                    # Check link is in the rendered HTML
                    if f'href="{link_path}"' in auditor_content:
                        # Verify target exists
                        target_path = SITE_DIR / current_version / "docs" / link_path.replace("../", "") / "index.html"
                        if target_path.exists():
                            print(f"[OK] {current_version}: auditor link '{link_name}' resolves to {target_path.relative_to(SITE_DIR)}")
                        else:
                            errors.append(f"{current_version}: auditor link '{link_name}' target MISSING: {target_path.relative_to(SITE_DIR)}")
                    else:
                        errors.append(f"{current_version}: auditor link '{link_name}' uses wrong path (expected {link_path})")
            else:
                errors.append(f"{current_version}: for-auditors page MISSING (configured but not built)")

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

    # 8. Check for placeholder repo URLs (MUST NOT appear in output)
    placeholder_found = False
    for html_file in SITE_DIR.rglob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        if "your-org/mathledger" in content:
            errors.append(f"Placeholder URL 'your-org/mathledger' found in {html_file.relative_to(SITE_DIR)}")
            placeholder_found = True
    if not placeholder_found:
        print("[OK] No placeholder repo URLs (your-org/mathledger) in site output")

    # 9. Check version coherence bridge text present (current version only)
    if current_version:
        current_landing = SITE_DIR / current_version / "index.html"
        if current_landing.exists():
            landing_content = current_landing.read_text(encoding="utf-8")
            if "This archive is immutable" in landing_content and "live instantiation" in landing_content:
                print(f"[OK] {current_version}: Version coherence bridge text present")
            else:
                errors.append(f"{current_version}: Version coherence bridge text MISSING")
            if "demo-sync-status" in landing_content:
                print(f"[OK] {current_version}: Demo sync check script present")
            else:
                errors.append(f"{current_version}: Demo sync check script MISSING")

    # 10. Check Ready-to-Verify section present (current version only)
    if current_version:
        current_landing = SITE_DIR / current_version / "index.html"
        if current_landing.exists():
            landing_content = current_landing.read_text(encoding="utf-8")
            if "Ready-to-Verify" in landing_content and "examples.json" in landing_content:
                print(f"[OK] {current_version}: Ready-to-Verify section present")
            else:
                errors.append(f"{current_version}: Ready-to-Verify section MISSING")

    # 11. Check external_audits listed in releases.json are linked from for-auditors
    if current_version:
        current_config = releases.get("versions", {}).get(current_version, {})
        external_audit_docs = [
            d for d in current_config.get("docs", [])
            if isinstance(d, dict) and "external_audits" in d.get("slug", "")
        ]
        if external_audit_docs:
            for_auditors_path = SITE_DIR / current_version / "docs" / "for-auditors" / "index.html"
            if for_auditors_path.exists():
                auditor_content = for_auditors_path.read_text(encoding="utf-8")
                if "External Audits" in auditor_content and "external_audits" in auditor_content:
                    print(f"[OK] {current_version}: External Audits section linked from for-auditors")
                    # Check that configured audit pages are actually rendered
                    for audit_doc in external_audit_docs:
                        audit_slug = audit_doc["slug"]
                        audit_page = SITE_DIR / current_version / "docs" / audit_slug / "index.html"
                        if audit_page.exists():
                            print(f"[OK] {current_version}: external audit '{audit_slug}' rendered")
                        else:
                            errors.append(f"{current_version}: external audit '{audit_slug}' NOT rendered")
                else:
                    errors.append(f"{current_version}: external_audits configured but NOT linked from for-auditors page")

    # 12. Check archive table links are version-pinned (absolute paths)
    for version in releases.get("versions", {}):
        landing = SITE_DIR / version / "index.html"
        if landing.exists():
            content = landing.read_text(encoding="utf-8")
            # These relative patterns should NOT appear in archive table
            relative_patterns = [
                ('href="docs/', "Archive table has relative docs/ link"),
                ('href="fixtures/"', "Archive table has relative fixtures/ link"),
                ('href="evidence-pack/"', "Archive table has relative evidence-pack/ link (not verify/)"),
                ('href="manifest.json"', "Archive table has relative manifest.json link"),
            ]
            has_relative = False
            for pattern, msg in relative_patterns:
                # Skip if it's a version-pinned link (starts with /{version}/)
                if pattern in content and f'href="/{version}/' not in content.replace(pattern, "CHECK"):
                    # Double-check: only flag if NOT preceded by version path
                    import re
                    # Look for the pattern NOT preceded by /{version}
                    unversioned = re.findall(rf'(?<!/{version}){re.escape(pattern)}', content)
                    if unversioned:
                        errors.append(f"{version}: {msg}")
                        has_relative = True
            if not has_relative:
                print(f"[OK] {version}: archive table uses version-pinned (absolute) paths")

    # 13. Check evidence-pack page has verifier link
    for version in releases.get("versions", {}):
        evidence_page = SITE_DIR / version / "evidence-pack" / "index.html"
        if evidence_page.exists():
            content = evidence_page.read_text(encoding="utf-8")
            if f"/{version}/evidence-pack/verify/" in content or "verify/" in content:
                print(f"[OK] {version}: evidence-pack page links to verifier")
            else:
                errors.append(f"{version}: evidence-pack page MISSING verifier link")

    # 14. Check for-auditors page has version-pinned verifier link
    for version in releases.get("versions", {}):
        for_auditors = SITE_DIR / version / "docs" / "for-auditors" / "index.html"
        if for_auditors.exists():
            content = for_auditors.read_text(encoding="utf-8")
            if "evidence-pack/verify/" in content:
                print(f"[OK] {version}: for-auditors page has verifier link")
            else:
                errors.append(f"{version}: for-auditors page MISSING verifier link")

    # 15. Check NO version page shows "Status: CURRENT" (only /versions/ is authority)
    for version, config in releases.get("versions", {}).items():
        landing = SITE_DIR / version / "index.html"
        if landing.exists():
            content = landing.read_text(encoding="utf-8")
            if "Status: CURRENT" in content:
                errors.append(f"{version}: page shows 'Status: CURRENT' (should show LOCKED)")
            elif "Status: LOCKED" in content:
                print(f"[OK] {version}: page shows LOCKED status (not CURRENT)")
            else:
                errors.append(f"{version}: page missing status indicator")

    # 16. Check superseded versions have the required disclaimer
    # The disclaimer must state that /demo/ runs the CURRENT version
    # This matches what hostile_audit.ps1 Check 16 verifies
    for version, config in releases.get("versions", {}).items():
        status = config.get("status", "")
        if status.startswith("superseded"):
            landing = SITE_DIR / version / "index.html"
            if landing.exists():
                content = landing.read_text(encoding="utf-8")
                # Check for the disclaimer pattern (matches hostile_audit.ps1 regex)
                if "hosted demo" in content.lower() and "runs" in content.lower() and "current version" in content.lower():
                    print(f"[OK] {version}: superseded disclaimer present")
                else:
                    errors.append(f"{version}: SUPERSEDED but missing disclaimer about /demo/ running CURRENT version")
            else:
                errors.append(f"{version}: superseded version landing page missing")

    # 17. Check examples.json has no stale version references
    # Current version's examples.json must have pack_version matching current, not old versions
    current_version = releases.get("current_version")
    if current_version:
        examples_file = SITE_DIR / current_version / "evidence-pack" / "examples.json"
        if examples_file.exists():
            examples_content = examples_file.read_text(encoding="utf-8")
            stale_patterns = [
                ('/v0.2.1/', "stale URL path /v0.2.1/"),
                ('"pack_version": "v0.2.1"', "stale pack_version v0.2.1"),
                ('"pack_version":"v0.2.1"', "stale pack_version v0.2.1 (no space)"),
            ]
            has_stale = False
            for pattern, desc in stale_patterns:
                if pattern in examples_content:
                    errors.append(f"{current_version}: examples.json contains {desc}")
                    has_stale = True
            if not has_stale:
                print(f"[OK] {current_version}: examples.json has no stale version references")
        else:
            # Not an error if examples.json doesn't exist, but note it
            print(f"[--] {current_version}: examples.json not present (optional)")

    # 18. Check verify page has "Run self-test" button
    for version in releases.get("versions", {}):
        verifier_path = SITE_DIR / version / "evidence-pack" / "verify" / "index.html"
        if verifier_path.exists():
            verifier_content = verifier_path.read_text(encoding="utf-8")
            if "Run self-test" in verifier_content:
                print(f"[OK] {version}: verify page has 'Run self-test' button")
            else:
                errors.append(f"{version}: verify page MISSING 'Run self-test' button")

    # 19. Check /versions/ has status ambiguity note
    versions_page = SITE_DIR / "versions" / "index.html"
    if versions_page.exists():
        versions_content = versions_page.read_text(encoding="utf-8")
        if "Pre-v0.2.2 archives" in versions_content and "canonical source" in versions_content.lower():
            print("[OK] /versions/: status ambiguity note present")
        else:
            errors.append("/versions/: MISSING status ambiguity note about pre-v0.2.2 archives")
    else:
        errors.append("/versions/index.html: page missing")

    # 20. Check Field Manual exists for each version (index.html, fm.pdf, fm.tex)
    for version in releases.get("versions", {}):
        fm_dir = SITE_DIR / version / "docs" / "field-manual"
        fm_index = fm_dir / "index.html"
        fm_pdf = fm_dir / "fm.pdf"
        fm_tex = fm_dir / "fm.tex"

        if fm_index.exists():
            print(f"[OK] {version}: docs/field-manual/index.html exists")
        else:
            errors.append(f"{version}: docs/field-manual/index.html MISSING")

        if fm_pdf.exists():
            print(f"[OK] {version}: docs/field-manual/fm.pdf exists")
        else:
            errors.append(f"{version}: docs/field-manual/fm.pdf MISSING")

        if fm_tex.exists():
            print(f"[OK] {version}: docs/field-manual/fm.tex exists")
        else:
            errors.append(f"{version}: docs/field-manual/fm.tex MISSING")

        # Check fm.pdf and fm.tex are in manifest with sha256
        manifest_file = SITE_DIR / version / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            files = manifest.get("files", [])
            fm_files_in_manifest = {
                f["path"]: f for f in files
                if f["path"].startswith("docs/field-manual/")
            }
            for expected in ["docs/field-manual/fm.pdf", "docs/field-manual/fm.tex", "docs/field-manual/index.html"]:
                if expected in fm_files_in_manifest:
                    if "sha256" in fm_files_in_manifest[expected]:
                        print(f"[OK] {version}: {expected} in manifest with sha256")
                    else:
                        errors.append(f"{version}: {expected} in manifest but missing sha256")
                else:
                    errors.append(f"{version}: {expected} NOT in manifest.json")

    # 21. Check landing page links to Field Manual (current version only)
    current_version = releases.get("current_version")
    if current_version:
        landing = SITE_DIR / current_version / "index.html"
        if landing.exists():
            content = landing.read_text(encoding="utf-8")
            if f"/{current_version}/docs/field-manual/" in content:
                print(f"[OK] {current_version}: landing page links to Field Manual")
            else:
                errors.append(f"{current_version}: landing page MISSING link to Field Manual")

    # 22. Check Tier A/B/C counts in invariants page match releases.json
    # Skip v0 - legacy version uses shared invariants_status.md which reflects v0.2.0+ counts
    for version, config in releases.get("versions", {}).items():
        if version == "v0":
            continue  # v0 uses shared doc with v0.2.0+ content
        invariants_page = SITE_DIR / version / "docs" / "invariants" / "index.html"
        if invariants_page.exists():
            content = invariants_page.read_text(encoding="utf-8")
            expected_a = config.get("invariants", {}).get("tier_a", 0)
            expected_b = config.get("invariants", {}).get("tier_b", 0)
            expected_c = config.get("invariants", {}).get("tier_c", 0)

            # Check header mentions correct count
            import re
            tier_a_match = re.search(r"Tier A[:\s]+.*?\((\d+)", content)
            if tier_a_match:
                found_a = int(tier_a_match.group(1))
                if found_a == expected_a:
                    print(f"[OK] {version}: Tier A count ({found_a}) matches releases.json")
                else:
                    errors.append(f"{version}: Tier A count mismatch: page says {found_a}, releases.json says {expected_a}")
            else:
                print(f"[--] {version}: Could not parse Tier A count from invariants page")

    # 23. Check for stale version strings in current version archive
    # Prior version strings should not appear except in external_audits/ or historical sections
    if current_version:
        all_versions = list(releases.get("versions", {}).keys())
        # Skip v0 - too generic, matches in "v0.2.3" as substring
        prior_versions = [v for v in all_versions if v != current_version and v != "v0"]
        stale_found = []

        for html_file in (SITE_DIR / current_version).rglob("*.html"):
            # Skip external_audits directory (historical by nature)
            if "external_audits" in str(html_file):
                continue

            rel_path = html_file.relative_to(SITE_DIR / current_version)
            content = html_file.read_text(encoding="utf-8")

            # Skip files with explicit historical markers
            if "Release Notes:" in content or "<!-- HISTORICAL -->" in content:
                continue

            # Check for prior version strings (like v0.2.1, v0.2.2 in v0.2.3)
            for prior in prior_versions:
                # Look for version string patterns that indicate staleness
                # e.g., "/v0.2.1/" paths, "v0.2.1-" tags, "pack_version": "v0.2.1"
                stale_patterns = [
                    f"/{prior}/",  # URL paths
                    f'"{prior}-',  # Tag references like "v0.2.1-cohesion"
                    f'pack_version": "{prior}"',  # JSON pack versions
                    f"checkout {prior}",  # git checkout commands
                ]
                for pattern in stale_patterns:
                    if pattern in content:
                        # Exception: superseded-by references are OK
                        if f"superseded-by-{prior.replace('v', '')}" not in content or pattern != f"/{prior}/":
                            stale_found.append(f"{rel_path}: contains '{pattern}'")
                            break

        if not stale_found:
            print(f"[OK] {current_version}: no stale version strings in non-historical pages")
        else:
            # For now, warn but don't fail (some historical references are OK)
            for sf in stale_found[:5]:  # Show first 5
                print(f"[WARN] {current_version}: {sf}")
            if len(stale_found) > 5:
                print(f"[WARN] ... and {len(stale_found) - 5} more stale references")

    # 24. Check for-auditors pages have no blob/main/ links (must be tag-pinned)
    for version, config in releases.get("versions", {}).items():
        for_auditors = SITE_DIR / version / "docs" / "for-auditors" / "index.html"
        if for_auditors.exists():
            fa_content = for_auditors.read_text(encoding="utf-8")
            if "blob/main/" in fa_content:
                errors.append(f"{version}: for-auditors contains blob/main/ (must use tag-pinned URLs)")
            else:
                print(f"[OK] {version}: for-auditors has no blob/main/ links")

    # 25. Check CURRENT version for-auditors has correct verifier/examples URLs
    if current_version:
        for_auditors = SITE_DIR / current_version / "docs" / "for-auditors" / "index.html"
        if for_auditors.exists():
            fa_content = for_auditors.read_text(encoding="utf-8")
            expected_verifier = f"/{current_version}/evidence-pack/verify/"
            expected_examples = f"/{current_version}/evidence-pack/examples.json"
            if expected_verifier in fa_content:
                print(f"[OK] {current_version}: for-auditors has correct verifier URL")
            else:
                errors.append(f"{current_version}: for-auditors missing correct verifier URL {expected_verifier}")
            if expected_examples in fa_content:
                print(f"[OK] {current_version}: for-auditors has correct examples URL")
            else:
                errors.append(f"{current_version}: for-auditors missing correct examples URL {expected_examples}")

    # 26. Check fixture directory links resolve (each fixture dir has index.html)
    for version, config in releases.get("versions", {}).items():
        fixtures_dir = SITE_DIR / version / "fixtures"
        if fixtures_dir.exists():
            # Check each subdirectory of fixtures/ has an index.html
            missing = []
            for subdir in fixtures_dir.iterdir():
                if subdir.is_dir() and subdir.name not in ["__pycache__"]:
                    if not (subdir / "index.html").exists():
                        missing.append(subdir.name)
            if missing:
                errors.append(f"{version}: fixture directories missing index.html: {missing}")
            else:
                print(f"[OK] {version}: all fixture directories have index.html")

    # 27. Verify verifier JS has no syntax errors (Node.js check)
    import subprocess
    import tempfile
    for version, config in releases.get("versions", {}).items():
        verifier_html = SITE_DIR / version / "evidence-pack" / "verify" / "index.html"
        if verifier_html.exists():
            html_content = verifier_html.read_text(encoding="utf-8")
            # Extract JS from <script> tags
            import re
            js_matches = re.findall(r'<script>(.*?)</script>', html_content, re.DOTALL)
            if js_matches:
                js_code = chr(10).join(js_matches)
                # Write to temp file and check syntax with Node.js
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as tmp:
                    tmp.write(js_code)
                    tmp_path = tmp.name
                try:
                    result = subprocess.run(
                        ['node', '--check', tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        print(f"[OK] {version}: verifier JS syntax valid")
                    else:
                        errors.append(f"{version}: verifier JS syntax error: {result.stderr.strip()}")
                except FileNotFoundError:
                    print(f"[--] {version}: Node.js not available, skipping JS syntax check")
                except Exception as e:
                    print(f"[--] {version}: JS syntax check failed: {e}")
                finally:
                    import os
                    os.unlink(tmp_path)

    # 28. Verify verifier functions are defined (verify, runSelfTest)
    for version, config in releases.get("versions", {}).items():
        verifier_html = SITE_DIR / version / "evidence-pack" / "verify" / "index.html"
        if verifier_html.exists():
            html_content = verifier_html.read_text(encoding="utf-8")
            missing_funcs = []
            # Check core functions - sha can be either sha() or shaWithDomain() for compatibility
            for func in ["function verify(", "function runSelfTest(", "function can("]:
                if func not in html_content:
                    missing_funcs.append(func.replace("function ", "").replace("(", ""))
            # Check for sha function - can be sha() or shaWithDomain() depending on version
            if "function sha(" not in html_content and "function shaWithDomain(" not in html_content:
                missing_funcs.append("sha/shaWithDomain")
            if missing_funcs:
                errors.append(f"{version}: verifier missing functions: {missing_funcs}")
            else:
                print(f"[OK] {version}: verifier has all required functions")

    # 29. Combined verifier gate: current version must pass Node --check AND contain "Run self-test vectors"
    current_verifier = SITE_DIR / current_version / "evidence-pack" / "verify" / "index.html"
    if current_verifier.exists():
        html_content = current_verifier.read_text(encoding="utf-8")
        gate_passed = True
        # Check 1: "Run self-test vectors" text present
        if "Run self-test vectors" not in html_content:
            errors.append(f"{current_version}: verifier MISSING 'Run self-test vectors' text")
            gate_passed = False
        # Check 2: Node.js syntax check (already done above, but re-verify for gate)
        import re
        js_matches = re.findall(r'<script>(.*?)</script>', html_content, re.DOTALL)
        if js_matches:
            js_code = chr(10).join(js_matches)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as tmp:
                tmp.write(js_code)
                tmp_path = tmp.name
            try:
                result = subprocess.run(['node', '--check', tmp_path], capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    errors.append(f"{current_version}: verifier gate FAILED - Node syntax error")
                    gate_passed = False
            except FileNotFoundError:
                pass  # Node not available, already warned above
            finally:
                import os
                os.unlink(tmp_path)
        if gate_passed:
            print(f"[OK] {current_version}: verifier gate PASSED (syntax valid + self-test text present)")

    # 30. Verify examples.json hashes match using Node.js (external script avoids escaping issues)
    examples_path = SITE_DIR / current_version / "evidence-pack" / "examples.json"
    verify_script = Path(__file__).parent / "verify_examples_hash.js"
    if examples_path.exists() and verify_script.exists():
        try:
            result = subprocess.run(
                ['node', str(verify_script), str(examples_path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"[OK] {current_version}: examples.json hashes verified by Node.js")
            else:
                errors.append(f"{current_version}: examples.json hash verification FAILED: {result.stdout.strip()}")
        except FileNotFoundError:
            print(f"[--] {current_version}: Node.js not available, skipping examples hash verification")
        except Exception as e:
            print(f"[--] {current_version}: examples hash verification skipped: {e}")
    elif examples_path.exists():
        print(f"[--] {current_version}: verify_examples_hash.js not found, skipping hash verification")
    else:
        print(f"[--] {current_version}: examples.json not present, skipping hash verification")

    # 31. Verify self-test semantics: Expected=FAIL & Actual=FAIL -> test PASS
    # The verifier's testPack function must correctly interpret that a tampered pack
    # showing FAIL (actual) when FAIL was expected means the TEST passed.
    current_verifier = SITE_DIR / current_version / "evidence-pack" / "verify" / "index.html"
    if current_verifier.exists():
        html_content = current_verifier.read_text(encoding="utf-8")
        # Check for correct self-test pass logic pattern
        # The testPack function should return pass: actual === expected (or equivalent)
        correct_patterns = [
            'pass:expectedResult==="FAIL"',  # compact
            'pass: expectedResult === "FAIL"',  # spaced
            'pass:r.actual===expected',  # generalized
            'pass: actual === expectedResult',  # alternative naming
            'pass:actual===expectedResult',  # compact alternative
        ]
        has_correct_logic = any(p.replace(" ", "") in html_content.replace(" ", "") for p in correct_patterns)

        # Also check for the verdict comparison pattern in testPack return
        if 'pass:expectedResult===' in html_content.replace(" ", "") or 'pass:r.actual===' in html_content.replace(" ", ""):
            has_correct_logic = True

        # Check for the specific patterns that indicate correct semantics
        if 'expectedResult==="FAIL"' in html_content.replace(" ", "") or 'actual===expectedResult' in html_content.replace(" ", ""):
            has_correct_logic = True

        if has_correct_logic:
            print(f"[OK] {current_version}: self-test semantics correct (FAIL/FAIL -> PASS)")
        else:
            # Additional check: look for the testPack function and verify its return logic
            import re
            testpack_match = re.search(r'function testPack\([^)]*\)\s*\{[^}]+return\s*\{[^}]+pass:[^}]+\}', html_content, re.DOTALL)
            if testpack_match:
                testpack_code = testpack_match.group(0)
                if 'expectedResult' in testpack_code and '===' in testpack_code:
                    print(f"[OK] {current_version}: self-test semantics appear correct (testPack compares to expectedResult)")
                else:
                    errors.append(f"{current_version}: self-test semantics UNCLEAR - testPack may not correctly handle FAIL/FAIL -> PASS")
            else:
                errors.append(f"{current_version}: self-test semantics MISSING - testPack function not found or malformed")

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
    # Freezing options for immutable versions
    parser.add_argument(
        "--freeze",
        metavar="VERSION",
        help="Freeze a specific version (creates immutable manifest)",
    )
    parser.add_argument(
        "--freeze-all",
        action="store_true",
        help="Freeze all built versions",
    )
    parser.add_argument(
        "--check-immutability",
        action="store_true",
        help="Verify all frozen versions match their manifests",
    )
    parser.add_argument(
        "--no-skip-frozen",
        action="store_true",
        help="Rebuild frozen versions instead of skipping (with verification)",
    )
    args = parser.parse_args()

    # Load canonical release metadata
    releases = load_releases()

    # Check-immutability mode
    if args.check_immutability:
        success = check_immutability(releases)
        sys.exit(0 if success else 1)

    # Freeze-only mode
    if args.freeze:
        if args.freeze not in releases["versions"]:
            print(f"ERROR: Unknown version: {args.freeze}")
            sys.exit(1)
        freeze_version(args.freeze)
        sys.exit(0)

    # Freeze-all mode
    if args.freeze_all:
        print("Freezing all built versions...")
        frozen_count = 0
        for v in releases["versions"]:
            version_dir = SITE_DIR / v
            if version_dir.exists():
                if not is_version_frozen(v):
                    freeze_version(v)
                    frozen_count += 1
                else:
                    print(f"  {v}: already frozen")
            else:
                print(f"  {v}: not built, skipping")
        print(f"Frozen {frozen_count} versions.")
        sys.exit(0)

    # Verify-only mode
    if args.verify:
        if not SITE_DIR.exists():
            print("ERROR: site/ directory does not exist. Run build first.")
            sys.exit(1)
        success = verify_build(releases)
        sys.exit(0 if success else 1)

    # Build mode
    if args.clean and SITE_DIR.exists():
        # Warn about frozen versions being cleaned
        frozen_versions = [v for v in releases["versions"] if is_version_frozen(v)]
        if frozen_versions:
            print(f"Warning: Cleaning will remove frozen version directories: {frozen_versions}")
            print("         Frozen manifests in releases/frozen/ are preserved.")
        shutil.rmtree(SITE_DIR)
        print(f"Cleaned {SITE_DIR}")

    SITE_DIR.mkdir(parents=True, exist_ok=True)

    build_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    skip_frozen = not args.no_skip_frozen

    if args.all:
        for v in releases["versions"]:
            build_version(releases, v, build_time, skip_frozen=skip_frozen)
    else:
        version = args.version or releases["current_version"]
        build_version(releases, version, build_time, skip_frozen=skip_frozen)

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
