import os
import tomllib  # Requires Python 3.11+
from pathlib import Path

# Directories to exclude from the package discovery check at the project root.
# These are directories that might exist at the top level but are never considered packages.
EXCLUDE_DIRS = {
    ".venv",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "docs",
    "tests",
    "artifacts",
    "reports",
    "logs",
    "mathledger.egg-info",
}


def test_packaging_integrity_against_filesystem():
    """
    Ensures the static package list in `pyproject.toml` matches the actual
    top-level Python packages on the filesystem.

    This test prevents "package drift" where a new package is created but not
    formally declared in the build configuration, which is critical for
    editable installs and distribution builds.
    """
    project_root = Path(__file__).parent.parent.resolve()

    # 1. Find all actual top-level packages on the filesystem.
    # A top-level package is a directory in the project root that contains an `__init__.py`.
    actual_packages = set()
    for item in project_root.iterdir():
        if item.is_dir() and item.name not in EXCLUDE_DIRS:
            if (item / "__init__.py").exists():
                actual_packages.add(item.name)

    # 2. Read the declared package list from pyproject.toml.
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    declared_packages = set(pyproject_data.get("tool", {}).get("setuptools", {}).get("packages", []))

    # 3. Assert that the sets are identical and provide a clear error if not.
    if not actual_packages == declared_packages:
        missing_from_pyproject = actual_packages - declared_packages
        extraneous_in_pyproject = declared_packages - actual_packages

        error_message = (
            "ERROR: The package list in `pyproject.toml` is out of sync with the filesystem.\n"
            "       This test (`tests/test_packaging_integrity.py`) enforces our packaging strategy.\n"
            "       Please manually update `[tool.setuptools.packages]`.\n"
        )
        if missing_from_pyproject:
            error_message += (
                f"\n  [+] PACKAGES TO ADD:\n"
                f"      The following packages exist but are NOT DECLARED: {sorted(list(missing_from_pyproject))}\n"
                f"      REMEDIATION: Add their names to the `packages` list in `pyproject.toml`.\n"
            )
        if extraneous_in_pyproject:
            error_message += (
                f"\n  [-] PACKAGES TO REMOVE:\n"
                f"      The following packages are DECLARED but DO NOT EXIST: {sorted(list(extraneous_in_pyproject))}\n"
                f"      REMEDIATION: Remove their names from the `packages` list in `pyproject.toml`.\n"
            )

        assert False, error_message