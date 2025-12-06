# MathLedger Auto-Manifest

**Generated:** 2025-10-19 18:23:53 UTC
**Generator:** tools/docs/auto_manifest.py
**Doctrine:** Every fact backed by an artifact in artifacts/

This document is automatically generated from CI artifacts and provides
human-verifiable explanation of how proofs chain together in MathLedger.
All claims reference specific artifacts with SHA-256 checksums.

## System Overview

MathLedger is an automated theorem proving system that generates, validates,
and organizes mathematical proofs using formal logic. The system operates in
two modes:

- **Baseline Mode**: Unguided proof generation using systematic derivation
- **Guided Mode**: ML policy-enhanced derivation with 3x+ performance uplift

All proofs are verified, normalized, and sealed into blockchain-style ledger
blocks with Merkle roots for cryptographic integrity.

## Proof Chain Architecture


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
7. **Block Sealing**: Group proofs into blocks with Merkle roots

## Performance Evidence

**Source:** `artifacts/wpv5/fol_stats.json`

- **Baseline Mean:** 44.0 proofs/hour
- **Guided Mean:** 132.0 proofs/hour
- **Uplift Factor:** 3.0x
- **Statistical Significance:** p = 0.0

The guided mode achieves 3.0x performance improvement over
baseline, demonstrating the effectiveness of ML policy guidance in proof
generation.

**Source:** `artifacts/wpv5/fol_ab.csv`
**Rows:** 6

Complete A/B test data comparing baseline and guided modes across multiple
seeds and configurations.

**Source:** `artifacts/wpv5/EVIDENCE.md`
**Size:** 3321 bytes
**Lines:** 71

Comprehensive evidence document detailing golden-run gates, live API
snapshots, and reproducibility audit trails.

## Artifact Inventory
This section catalogs all artifacts referenced in this manifest.
### Performance Baselines

- `artifacts/perf/baseline.csv` - 3 benchmark rows
### WPV5 Evaluation Artifacts
- `artifacts/wpv5/fol_stats.json` - FOL statistics- `artifacts/wpv5/fol_ab.csv` - A/B test data- `artifacts/wpv5/EVIDENCE.md` - Evidence documentation

## Checksum Manifest

All artifacts referenced in this document are verified with SHA-256
checksums. Use these checksums to verify artifact integrity:

```
sha256sum -c checksums.txt
```


```
e7fe9d0f24495f47aa10bb518c1c7dc64aa9c11c4aa7b397e6daa09fc2899373  artifacts/guidance/train.csv
fbeb4a105aeff075c104371eea835365374fc986dd5c1e76bc0f50ad66d13b08  artifacts/guidance/val.csv
027936ffa71a891109c3c65786436272e7caa50453529630cfe95d7276b40949  artifacts/perf/baseline.csv
b94421a302a2ad722f30ea3225c6b97eafcc34e35d05b91aaf9ca11e5d1d39fc  artifacts/policy/policy.json
61508d5b868277e61ddec5835334fdfd2a73beb094c1287552a82ede3fa06d0d  artifacts/wpv5/EVIDENCE.md
c55f1fa48c841d18095d17371c7910a5888d2267a5796bca7483c9092999a317  artifacts/wpv5/fol_ab.csv
9d431398f5361fd042956117a02573d7010736b87c21b890bdc5307a0697017d  artifacts/wpv5/fol_stats.json
```

## Regeneration

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
in artifacts/. No speculation, no approximation, only verifiable truth.