# Packaging Strategy for the MathLedger Monorepo

**Author:** GEMINI-B
**Status:** Active
**Date:** 2025-12-10

## 1. Context: The Static vs. Dynamic Packaging Debate

This project uses a **static package list** in `pyproject.toml` under the `[tool.setuptools.packages]` key. This was a deliberate choice over using dynamic discovery mechanisms like `[tool.setuptools.packages.find]`.

This document explains the trade-offs of this decision and establishes the rules for maintaining our packaging configuration.

## 2. Rationale for a Static Package List

While dynamic discovery is convenient, it can introduce ambiguity and risk in a large, complex monorepo. The static list was chosen for two primary reasons:

1.  **Explicitness and Safety:** The static list acts as a manifest of the project's top-level namespaces. There is no ambiguity about what is considered a package. This prevents accidental inclusion of non-package directories (like `tests`, `docs`, `scripts` without an `__init__.py`) that might be picked up by broad-matching dynamic finders. Every package is declared intentionally.

2.  **Architectural Clarity:** The list provides an immediate, high-level overview of the monorepo's structure directly within the core configuration file. It encourages developers to be deliberate about creating new top-level packages.

## 3. The Trade-Off: Maintenance Overhead

The primary disadvantage of a static list is the manual overhead. A developer creating a new top-level package must remember to add it to `pyproject.toml`.

**This risk is mitigated by a CI guard test (`tests/test_packaging_integrity.py`).** This test automatically compares the packages on the filesystem with the declared list in `pyproject.toml` and will **fail the build** if they are out of sync. This provides the safety of a static list with the convenience of automated validation.

## 4. How CI Enforces This Policy

This policy is not just a guideline; it is an actively enforced contract, guaranteed by our CI pipeline.

-   **Enforcement Test:** `tests/test_packaging_integrity.py`
-   **Mechanism:** This test runs on every commit. It scans the repository for all top-level package directories (those containing an `__init__.py`) and compares this list against the static `[tool.setuptools.packages]` list in `pyproject.toml`.

If the lists do not match perfectly, **the test will fail, blocking the build**.

### Remediation Command

If this test fails, read the error message carefully. It will tell you exactly which packages to add or remove.

**Example Failure:**
```
FAILED tests/test_packaging_integrity.py::test_packaging_integrity_against_filesystem - AssertionError:
ERROR: The package list in `pyproject.toml` is out of sync with the filesystem.
...
  [+] PACKAGES TO ADD:
      The following packages exist but are NOT DECLARED: ['new_cool_feature']
      REMEDIATION: Add their names to the `packages` list in `pyproject.toml`.
```

To fix this, you would edit `pyproject.toml` and add `"new_cool_feature"` to the `packages` list.

## 5. When to Revert to Dynamic Discovery

We should only reconsider this strategy if the rate of creation/deletion of top-level packages becomes so high that the static list causes constant merge conflicts or becomes a significant developer annoyance. Until that point, the safety and explicitness provided by the current method, backed by CI enforcement, is the preferred approach.
