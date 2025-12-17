# RFC: Test Execution Model for Curriculum Drift Chronicle

**Author:** GEMINI-B
**Status:** Draft
**Date:** 2025-12-10

## 1. Summary

This document proposes a resolution to the `ModuleNotFoundError` failures encountered during the execution of the E2E test suite for the Curriculum Drift Chronicle. These failures stem from a mismatch between the project's directory layout and the Python interpreter's path resolution (`PYTHONPATH`). We present two primary options to create a robust and maintainable test environment.

## 2. The Failure Mode: `ModuleNotFoundError`

During CI execution of the test suite, pytest failed with `ModuleNotFoundError`. This occurred because the tests, located in the top-level `tests/` directory, attempted to import modules from the application package located in `backend/` (e.g., `from backend.chronicle.archiver import ...`).

The default Python import mechanism does not automatically search the project's root directory for packages. When pytest runs from `tests/`, it does not know that `backend/` is a source root containing installable packages. Consequently, the import statements fail. This is a common issue in Python projects that separates test code from application code without proper configuration.

## 3. Proposed Solutions

We propose two viable options to resolve this issue permanently.

### Option A: Co-locate Tests within the Application Package

This approach involves moving the test suite into the main application package, making the tests a submodule of the code they are testing.

**Proposed Structure:**

```
.
├── backend/
│   ├── __init__.py
│   ├── chronicle/
│   │   ├── __init__.py
│   │   └── archiver.py
│   └── tests/
│       ├── __init__.py
│       └── test_archiver.py
├── scripts/
│   └── curriculum_drift_chronicle.py
└── pyproject.toml
```

**Required Changes:**

1.  **Move Files:** `mv tests/ backend/tests/`
2.  **Create `__init__.py`:** Ensure `backend/tests/__init__.py` exists to mark it as a package.
3.  **Update Test Paths:** Modify `pytest.ini` or `pyproject.toml` to point the `testpaths` variable to `backend/tests`.
4.  **Adjust Imports:** Test imports would become relative, e.g., `from ..chronicle.archiver import ...`.

**Pros:**
*   **Simplicity:** Imports become clear and explicit, following standard Python package rules.
*   **Discoverability:** Tests are located directly alongside the code they validate, which can be intuitive for developers.
*   **PEP 420 Compliance:** Aligns well with modern Python's implicit namespace packages.

**Cons:**
*   **Package Bloat:** The installed production package would include the test suite, which is generally undesirable. This can be mitigated with build/packaging configuration (`find_packages(exclude=["tests"])`).
*   **Convention Shift:** May differ from the established convention if other packages in the monorepo keep tests separate.

**Risk to CI:**
*   **Medium.** The CI workflow must be updated to target the new test path. Existing scripts that assume a top-level `tests/` directory will fail until updated.

### Option B: Configure the Project as an Installable Package

This approach treats the project as a formal Python package and uses tooling to make it discoverable during test runs, without changing the directory structure.

**Proposed Structure:** (No changes)

```
.
├── backend/
│   ├── __init__.py
│   └── chronicle/
│       ├── __init__.py
│       └── archiver.py
├── tests/
│   └── test_archiver.py
└── pyproject.toml
```

**Required Changes:**

1.  **Configure `pyproject.toml`:** Ensure `pyproject.toml` is configured to define the project's source paths. Specifically, marking `backend` as a source root.
    ```toml
    [tool.setuptools.packages.find]
    where = ["backend"]
    ```
2.  **Install in CI:** The CI test step must run an "editable" install before executing pytest:
    ```bash
    uv pip install -e .
    pytest
    ```
    This command creates a link from the site-packages directory to the source code, making `backend` importable from anywhere, including the `tests/` directory.

**Pros:**
*   **Clean Separation:** Keeps application code (`backend/`) strictly separate from test code (`tests/`).
*   **Best Practice:** Using editable installs is the modern, standard way to handle package development and testing in Python. It mirrors how the package will behave once published.
*   **Minimal Intrusion:** No files need to be moved.

**Cons:**
*   **Tooling Dependency:** Relies on the build/install tooling (`uv`, `pip`) being configured correctly.
*   **Extra Step:** Requires an explicit `pip install -e .` step in the workflow, which is an added dependency if not already present.

**Risk to CI:**
*   **Low.** The change is isolated to the test setup command within the CI script. It is less likely to have unintended side effects on other parts of the system.

## 4. Recommendation

We recommend **Option B**. It represents the most robust, modern, and non-invasive solution. It aligns with Python community best practices for package development and testing, and it carries the lowest risk of disrupting existing CI workflows. By formalizing the package structure and using an editable install, we solve the `PYTHONPATH` problem reliably without restructuring the repository.
