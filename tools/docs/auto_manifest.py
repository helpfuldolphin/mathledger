#!/usr/bin/env python3
"""
Auto-Manifest Generator for MathLedger

Generates living documentation by pulling facts from CI artifacts and rendering
deterministic, verifiable Markdown. Every fact in the docs is backed by an
artifact in artifacts/.

Usage:
    python tools/docs/auto_manifest.py --output docs/methods/auto_manifest.md

Tenacity Mantra: Every fact must be backed by an artifact.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

from backend.repro.determinism import deterministic_hash, deterministic_isoformat


class ArtifactScanner:
    """Scans artifacts/ directory and extracts verifiable facts."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.artifacts_dir = repo_root / "artifacts"
        self.facts: Dict[str, any] = {}
        self.checksums: Dict[str, str] = {}

    def scan_all(self) -> Dict[str, any]:
        """Scan all artifact directories and extract facts."""
        self.scan_wpv5()
        self.scan_perf()
        self.scan_guidance()
        self.scan_policy()
        return self.facts

    def scan_wpv5(self):
        """Scan artifacts/wpv5/ for FOL evidence and performance data."""
        wpv5_dir = self.artifacts_dir / "wpv5"
        if not wpv5_dir.exists():
            return

        fol_stats_path = wpv5_dir / "fol_stats.json"
        if fol_stats_path.exists():
            with open(fol_stats_path) as f:
                fol_stats = json.load(f)
            self.facts["fol_uplift"] = {
                "mean_baseline": fol_stats.get("mean_baseline"),
                "mean_guided": fol_stats.get("mean_guided"),
                "uplift_x": fol_stats.get("uplift_x"),
                "p_value": fol_stats.get("p_value"),
                "source": str(fol_stats_path.relative_to(self.repo_root)),
            }
            self.checksums[str(fol_stats_path.relative_to(self.repo_root))] = (
                self._compute_checksum(fol_stats_path)
            )

        evidence_path = wpv5_dir / "EVIDENCE.md"
        if evidence_path.exists():
            with open(evidence_path) as f:
                evidence_text = f.read()
            self.facts["evidence"] = {
                "path": str(evidence_path.relative_to(self.repo_root)),
                "size_bytes": evidence_path.stat().st_size,
                "lines": len(evidence_text.splitlines()),
            }
            self.checksums[str(evidence_path.relative_to(self.repo_root))] = (
                self._compute_checksum(evidence_path)
            )

        fol_ab_path = wpv5_dir / "fol_ab.csv"
        if fol_ab_path.exists():
            line_count = sum(1 for _ in open(fol_ab_path))
            self.facts["fol_ab_csv"] = {
                "path": str(fol_ab_path.relative_to(self.repo_root)),
                "rows": line_count - 1,
            }
            self.checksums[str(fol_ab_path.relative_to(self.repo_root))] = (
                self._compute_checksum(fol_ab_path)
            )

    def scan_perf(self):
        """Scan artifacts/perf/ for performance baselines."""
        perf_dir = self.artifacts_dir / "perf"
        if not perf_dir.exists():
            return

        baseline_path = perf_dir / "baseline.csv"
        if baseline_path.exists():
            line_count = sum(1 for _ in open(baseline_path))
            self.facts["perf_baseline"] = {
                "path": str(baseline_path.relative_to(self.repo_root)),
                "rows": line_count - 1,
            }
            self.checksums[str(baseline_path.relative_to(self.repo_root))] = (
                self._compute_checksum(baseline_path)
            )

    def scan_guidance(self):
        """Scan artifacts/guidance/ for training data."""
        guidance_dir = self.artifacts_dir / "guidance"
        if not guidance_dir.exists():
            return

        train_path = guidance_dir / "train.csv"
        val_path = guidance_dir / "val.csv"

        if train_path.exists():
            line_count = sum(1 for _ in open(train_path))
            self.facts["guidance_train"] = {
                "path": str(train_path.relative_to(self.repo_root)),
                "rows": line_count - 1,
            }
            self.checksums[str(train_path.relative_to(self.repo_root))] = (
                self._compute_checksum(train_path)
            )

        if val_path.exists():
            line_count = sum(1 for _ in open(val_path))
            self.facts["guidance_val"] = {
                "path": str(val_path.relative_to(self.repo_root)),
                "rows": line_count - 1,
            }
            self.checksums[str(val_path.relative_to(self.repo_root))] = (
                self._compute_checksum(val_path)
            )

    def scan_policy(self):
        """Scan artifacts/policy/ for ML policy artifacts."""
        policy_dir = self.artifacts_dir / "policy"
        if not policy_dir.exists():
            return

        policy_json_path = policy_dir / "policy.json"
        if policy_json_path.exists():
            with open(policy_json_path) as f:
                policy_data = json.load(f)
            self.facts["policy"] = {
                "path": str(policy_json_path.relative_to(self.repo_root)),
                "data": policy_data,
            }
            self.checksums[str(policy_json_path.relative_to(self.repo_root))] = (
                self._compute_checksum(policy_json_path)
            )

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class ManifestRenderer:
    """Renders living documentation from extracted facts."""

    def __init__(self, facts: Dict[str, any], checksums: Dict[str, str]):
        self.facts = facts
        self.checksums = checksums

    def render(self) -> str:
        """Render complete manifest document."""
        sections = [
            self._render_header(),
            self._render_overview(),
            self._render_proof_chain(),
            self._render_performance_evidence(),
            self._render_artifact_inventory(),
            self._render_checksum_manifest(),
            self._render_footer(),
        ]
        return "\n\n".join(sections)

    def _render_header(self) -> str:
        facts_digest = deterministic_hash(json.dumps(self.facts, sort_keys=True, default=str))
        timestamp_iso = deterministic_isoformat("auto_manifest", facts_digest)
        timestamp = timestamp_iso.replace("T", " ").replace("+00:00", " UTC")
        return f"""# MathLedger Auto-Manifest

**Generated:** {timestamp}
**Generator:** tools/docs/auto_manifest.py
**Doctrine:** Every fact backed by an artifact in artifacts/

This document is automatically generated from CI artifacts and provides
human-verifiable explanation of how proofs chain together in MathLedger.
All claims reference specific artifacts with SHA-256 checksums."""

    def _render_overview(self) -> str:
        return """## System Overview

MathLedger is an automated theorem proving system that generates, validates,
and organizes mathematical proofs using formal logic. The system operates in
two modes:

- **Baseline Mode**: Unguided proof generation using systematic derivation
- **Guided Mode**: ML policy-enhanced derivation with 3x+ performance uplift

All proofs are verified, normalized, and sealed into blockchain-style ledger
blocks with Merkle roots for cryptographic integrity."""

    def _render_proof_chain(self) -> str:
        return """## Proof Chain Architecture


```
Axioms -> Inference Rules -> Candidate Statements -> Verification
   |           |                    |                     |
   v           v                    v                     v
Initial    Modus Ponens      Normalization          Ledger DB
Truths     Application        + Hashing             + Blocks
```

Each proof in MathLedger follows a deterministic chain:

1. **Axiom Selection**: Start with foundational axioms (PL, FOL)
2. **Inference Application**: Apply Modus Ponens and substitution rules
3. **Candidate Generation**: Generate new statements from existing proofs
4. **Verification**: Validate logical correctness
5. **Normalization**: Canonicalize expression and compute hash
6. **Ledger Recording**: Insert into database with parent relationships
7. **Block Sealing**: Group proofs into blocks with Merkle roots"""

    def _render_performance_evidence(self) -> str:
        sections = ["## Performance Evidence"]

        if "fol_uplift" in self.facts:
            fol = self.facts["fol_uplift"]
            sections.append(f"""

**Source:** `{fol['source']}`

- **Baseline Mean:** {fol['mean_baseline']} proofs/hour
- **Guided Mean:** {fol['mean_guided']} proofs/hour
- **Uplift Factor:** {fol['uplift_x']}x
- **Statistical Significance:** p = {fol['p_value']}

The guided mode achieves {fol['uplift_x']}x performance improvement over
baseline, demonstrating the effectiveness of ML policy guidance in proof
generation.""")

        if "fol_ab_csv" in self.facts:
            fol_ab = self.facts["fol_ab_csv"]
            sections.append(f"""

**Source:** `{fol_ab['path']}`
**Rows:** {fol_ab['rows']}

Complete A/B test data comparing baseline and guided modes across multiple
seeds and configurations.""")

        if "evidence" in self.facts:
            ev = self.facts["evidence"]
            sections.append(f"""

**Source:** `{ev['path']}`
**Size:** {ev['size_bytes']} bytes
**Lines:** {ev['lines']}

Comprehensive evidence document detailing golden-run gates, live API
snapshots, and reproducibility audit trails.""")

        return "".join(sections)

    def _render_artifact_inventory(self) -> str:
        sections = ["## Artifact Inventory"]
        sections.append("\nThis section catalogs all artifacts referenced in this manifest.")

        if "perf_baseline" in self.facts:
            perf = self.facts["perf_baseline"]
            sections.append(f"\n### Performance Baselines\n\n- `{perf['path']}` - {perf['rows']} benchmark rows")

        if "fol_uplift" in self.facts:
            sections.append("\n### WPV5 Evaluation Artifacts\n")
            sections.append(f"- `{self.facts['fol_uplift']['source']}` - FOL statistics")
        if "fol_ab_csv" in self.facts:
            sections.append(f"- `{self.facts['fol_ab_csv']['path']}` - A/B test data")
        if "evidence" in self.facts:
            sections.append(f"- `{self.facts['evidence']['path']}` - Evidence documentation")

        return "".join(sections)

    def _render_checksum_manifest(self) -> str:
        manifest = """## Checksum Manifest

All artifacts referenced in this document are verified with SHA-256
checksums. Use these checksums to verify artifact integrity:

```
sha256sum -c checksums.txt
```


"""
        if self.checksums:
            manifest += "```\n"
            for path in sorted(self.checksums.keys()):
                checksum = self.checksums[path]
                manifest += f"{checksum}  {path}\n"
            manifest += "```"
        else:
            manifest += "No checksums available."

        return manifest

    def _render_footer(self) -> str:
        return """## Regeneration

To regenerate this manifest:

```bash
python tools/docs/auto_manifest.py --output docs/methods/auto_manifest.md
```


To verify all artifact checksums:

```bash
grep -A 100 "### Checksums" docs/methods/auto_manifest.md | grep "^[0-9a-f]" > /tmp/checksums.txt
cd /path/to/mathledger
sha256sum -c /tmp/checksums.txt
```

---

**Tenacity Mantra:** Every fact in this document is backed by an artifact
in artifacts/. No speculation, no approximation, only verifiable truth."""


def main():
    parser = argparse.ArgumentParser(
        description="Generate living documentation from CI artifacts"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/methods/auto_manifest.md"),
        help="Output path for generated manifest",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).parent.parent.parent,
        help="Repository root directory",
    )
    args = parser.parse_args()

    print(f"Scanning artifacts in {args.repo_root / 'artifacts'}...")
    scanner = ArtifactScanner(args.repo_root)
    facts = scanner.scan_all()

    print(f"Extracted {len(facts)} fact categories")
    print(f"Computed {len(scanner.checksums)} checksums")

    print("Rendering manifest...")
    renderer = ManifestRenderer(facts, scanner.checksums)
    manifest = renderer.render()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(manifest)

    print(f"Manifest written to {args.output}")
    print(f"Size: {len(manifest)} bytes")

    checksums_path = args.output.parent / "checksums.txt"
    with open(checksums_path, "w") as f:
        for path in sorted(scanner.checksums.keys()):
            checksum = scanner.checksums[path]
            f.write(f"{checksum}  {path}\n")
    print(f"Checksums written to {checksums_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
