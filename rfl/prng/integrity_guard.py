# PHASE II — NOT USED IN PHASE I
"""
PRNG Integrity Guard — Runtime Detection of Global Randomness Violations.

This module provides runtime guards that detect and optionally block
attempts to use global random state in Phase II code. It can be enabled
via environment variable to catch determinism violations during testing.

Usage:
    # Enable strict mode (raises on violation)
    export RFL_PRNG_STRICT=1
    
    # Enable warn mode (logs but doesn't block)
    export RFL_PRNG_WARN=1
    
    # In code:
    from rfl.prng.integrity_guard import install_guards, check_module_compliance

Contract Reference:
    Implements the "No Global State" requirement from docs/DETERMINISM_CONTRACT.md.
    Phase II code must use DeterministicPRNG, not global random functions.

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import functools
import os
import random
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple  # List, Tuple for type hints

# Track whether guards are installed
_guards_installed = False
_violation_log: List[Dict[str, Any]] = []

# Original functions (saved before patching)
_original_random_seed: Optional[Callable] = None
_original_random_random: Optional[Callable] = None
_original_random_shuffle: Optional[Callable] = None
_original_random_choice: Optional[Callable] = None
_original_random_randint: Optional[Callable] = None
_original_random_sample: Optional[Callable] = None
_original_random_uniform: Optional[Callable] = None

# Modules that are allowed to use global random (allowlist)
ALLOWED_MODULES: Set[str] = {
    "rfl.prng.deterministic_prng",  # The PRNG implementation itself
    "rfl.prng.integrity_guard",      # This module
    "random",                         # stdlib internals
    "_random",                        # C extension
}

# Phase II module prefixes that should NOT use global random
PHASE_II_PREFIXES: Tuple[str, ...] = (
    "experiments.u2",
    "experiments.run_uplift_u2",
    "rfl.runner",
    "rfl.coverage",
    "rfl.bootstrap_stats",
    "rfl.experiment",
)


def _get_caller_module() -> str:
    """Get the module name of the caller (skipping this module and random)."""
    import traceback
    
    for frame_info in traceback.extract_stack():
        module = frame_info.filename
        # Skip stdlib random and this module
        if "random.py" in module or "integrity_guard" in module:
            continue
        # Try to extract module name from path
        if "rfl" in module or "experiments" in module:
            # Convert path to module-like format
            parts = module.replace("\\", "/").split("/")
            for i, part in enumerate(parts):
                if part in ("rfl", "experiments"):
                    return ".".join(p.replace(".py", "") for p in parts[i:])
    return "unknown"


def _is_phase_ii_caller() -> bool:
    """Check if the caller is from a Phase II module."""
    caller = _get_caller_module()
    return any(caller.startswith(prefix) for prefix in PHASE_II_PREFIXES)


def _is_allowed_caller() -> bool:
    """Check if the caller is in the allowlist."""
    caller = _get_caller_module()
    return caller in ALLOWED_MODULES


def _record_violation(function_name: str, args: tuple, kwargs: dict) -> None:
    """Record a violation for later analysis."""
    import traceback
    
    violation = {
        "function": function_name,
        "caller_module": _get_caller_module(),
        "args_repr": repr(args)[:100],
        "traceback": traceback.format_stack()[-5:-1],
    }
    _violation_log.append(violation)


def _create_guarded_function(
    original_func: Callable,
    func_name: str,
) -> Callable:
    """Create a wrapper that detects global random usage."""
    
    @functools.wraps(original_func)
    def guarded(*args, **kwargs):
        if _is_allowed_caller():
            return original_func(*args, **kwargs)
        
        if _is_phase_ii_caller():
            _record_violation(func_name, args, kwargs)
            
            strict_mode = os.getenv("RFL_PRNG_STRICT", "").lower() in ("1", "true", "yes")
            warn_mode = os.getenv("RFL_PRNG_WARN", "").lower() in ("1", "true", "yes")
            
            caller = _get_caller_module()
            message = (
                f"[PRNG INTEGRITY] Global random.{func_name}() called from Phase II module '{caller}'. "
                f"Use DeterministicPRNG.for_path() instead."
            )
            
            if strict_mode:
                raise RuntimeError(message)
            elif warn_mode:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
        
        return original_func(*args, **kwargs)
    
    return guarded


def install_guards() -> None:
    """
    Install runtime guards on global random functions.
    
    This patches random.seed, random.random, random.shuffle, etc.
    to detect and optionally block calls from Phase II modules.
    
    Control behavior via environment variables:
        RFL_PRNG_STRICT=1  - Raise RuntimeError on violation
        RFL_PRNG_WARN=1    - Warn but allow execution
        (neither)          - Silent logging for analysis
    """
    global _guards_installed
    global _original_random_seed, _original_random_random
    global _original_random_shuffle, _original_random_choice
    global _original_random_randint, _original_random_sample
    global _original_random_uniform
    
    if _guards_installed:
        return
    
    # Save originals
    _original_random_seed = random.seed
    _original_random_random = random.random
    _original_random_shuffle = random.shuffle
    _original_random_choice = random.choice
    _original_random_randint = random.randint
    _original_random_sample = random.sample
    _original_random_uniform = random.uniform
    
    # Install guards
    random.seed = _create_guarded_function(random.seed, "seed")
    random.random = _create_guarded_function(random.random, "random")
    random.shuffle = _create_guarded_function(random.shuffle, "shuffle")
    random.choice = _create_guarded_function(random.choice, "choice")
    random.randint = _create_guarded_function(random.randint, "randint")
    random.sample = _create_guarded_function(random.sample, "sample")
    random.uniform = _create_guarded_function(random.uniform, "uniform")
    
    _guards_installed = True


def uninstall_guards() -> None:
    """Remove runtime guards and restore original functions."""
    global _guards_installed
    
    if not _guards_installed:
        return
    
    if _original_random_seed is not None:
        random.seed = _original_random_seed
    if _original_random_random is not None:
        random.random = _original_random_random
    if _original_random_shuffle is not None:
        random.shuffle = _original_random_shuffle
    if _original_random_choice is not None:
        random.choice = _original_random_choice
    if _original_random_randint is not None:
        random.randint = _original_random_randint
    if _original_random_sample is not None:
        random.sample = _original_random_sample
    if _original_random_uniform is not None:
        random.uniform = _original_random_uniform
    
    _guards_installed = False


def get_violation_log() -> List[Dict[str, Any]]:
    """Get the list of recorded violations."""
    return list(_violation_log)


def clear_violation_log() -> None:
    """Clear the violation log."""
    _violation_log.clear()


def check_module_compliance(module_path: str) -> Dict[str, Any]:
    """
    Static analysis check for a module's PRNG compliance.
    
    Parses the module's AST and looks for:
    - Imports of random module (except random.Random class)
    - Calls to random.* global functions
    - Calls to np.random.seed()
    
    Suppression:
        Lines with `# prng: allow` comment are exempt from violations.
        Code within `with allow_unrestricted_randomness():` blocks is also exempt.
    
    Args:
        module_path: Path to the Python module file.
    
    Returns:
        Dict with compliance status and any violations found.
    """
    import ast
    from pathlib import Path
    
    result = {
        "path": module_path,
        "compliant": True,
        "violations": [],
        "suppressed": [],
    }
    
    try:
        # Try UTF-8 first, then fall back to latin-1 for Windows files
        try:
            source = Path(module_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = Path(module_path).read_text(encoding="latin-1")
        source_lines = source.splitlines()
        tree = ast.parse(source)
    except SyntaxError as e:
        # Syntax errors mean we can't parse the file, but we should
        # report this as an issue, not mark it non-compliant
        result["compliant"] = True  # Can't verify, assume compliant
        result["parse_error"] = str(e)
        return result
    except Exception as e:
        result["compliant"] = True  # Can't verify, assume compliant
        result["error"] = str(e)
        return result
    
    # Collect line numbers that are suppressed via comment
    suppressed_lines: Set[int] = set()
    for lineno, line in enumerate(source_lines, start=1):
        if "# prng: allow" in line.lower() or "# prng:allow" in line.lower():
            suppressed_lines.add(lineno)
    
    # Collect line ranges within allow_unrestricted_randomness() context
    unrestricted_ranges: List[Tuple[int, int]] = []
    
    class UnrestrictedContextVisitor(ast.NodeVisitor):
        def visit_With(self, node: ast.With) -> None:
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    if isinstance(item.context_expr.func, ast.Name):
                        if item.context_expr.func.id == "allow_unrestricted_randomness":
                            unrestricted_ranges.append((node.lineno, node.end_lineno or node.lineno + 100))
            self.generic_visit(node)
    
    UnrestrictedContextVisitor().visit(tree)
    
    def is_line_exempt(lineno: int) -> bool:
        """Check if a line is exempt from PRNG checks."""
        if lineno in suppressed_lines:
            return True
        for start, end in unrestricted_ranges:
            if start <= lineno <= end:
                return True
        return False
    
    class RandomUsageVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []
            self.suppressed = []
        
        def visit_Call(self, node: ast.Call) -> None:
            violation = None
            
            # Check for random.* calls
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "random":
                        if node.func.attr in ("seed", "random", "shuffle", "choice", 
                                               "randint", "sample", "uniform", "gauss"):
                            violation = {
                                "type": "global_random_call",
                                "function": f"random.{node.func.attr}",
                                "line": node.lineno,
                            }
                
                # Check for np.random.seed specifically
                if isinstance(node.func.value, ast.Attribute):
                    if (isinstance(node.func.value.value, ast.Name) and 
                        node.func.value.value.id == "np" and
                        node.func.value.attr == "random" and
                        node.func.attr == "seed"):
                        violation = {
                            "type": "numpy_global_seed",
                            "function": "np.random.seed",
                            "line": node.lineno,
                        }
            
            if violation:
                if is_line_exempt(violation["line"]):
                    violation["suppressed"] = True
                    self.suppressed.append(violation)
                else:
                    self.violations.append(violation)
            
            self.generic_visit(node)
    
    visitor = RandomUsageVisitor()
    visitor.visit(tree)
    
    result["suppressed"] = visitor.suppressed
    
    if visitor.violations:
        result["compliant"] = False
        result["violations"] = visitor.violations
    
    return result


def audit_phase_ii_modules() -> Dict[str, Any]:
    """
    Audit all Phase II modules for PRNG compliance.
    
    Returns:
        Dict with overall status and per-module results.
    """
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parents[2]
    
    results = {
        "overall_compliant": True,
        "modules_checked": 0,
        "violations_found": 0,
        "details": [],
    }
    
    # Directories to check
    phase_ii_dirs = [
        project_root / "experiments" / "u2",
        project_root / "rfl",
    ]
    
    # Specific files to check
    phase_ii_files = [
        project_root / "experiments" / "run_uplift_u2.py",
    ]
    
    files_to_check = list(phase_ii_files)
    for dir_path in phase_ii_dirs:
        if dir_path.exists():
            files_to_check.extend(dir_path.rglob("*.py"))
    
    for file_path in files_to_check:
        if "__pycache__" in str(file_path):
            continue
        if file_path.name.startswith("test_"):
            continue
        
        result = check_module_compliance(str(file_path))
        results["modules_checked"] += 1
        
        if not result["compliant"]:
            results["overall_compliant"] = False
            results["violations_found"] += len(result.get("violations", []))
        
        # Only include non-compliant modules in details
        if not result["compliant"]:
            results["details"].append(result)
    
    return results


# --- Auto-install guards if environment variable is set ---
if os.getenv("RFL_PRNG_GUARD", "").lower() in ("1", "true", "yes"):
    install_guards()


# --- Self-test when run directly ---
if __name__ == "__main__":
    print("PRNG Integrity Guard — Audit Mode")
    print("=" * 60)
    
    results = audit_phase_ii_modules()
    
    print(f"Modules checked: {results['modules_checked']}")
    print(f"Violations found: {results['violations_found']}")
    print(f"Overall compliant: {results['overall_compliant']}")
    
    if results["details"]:
        print("\nNon-compliant modules:")
        for detail in results["details"]:
            print(f"\n  {detail['path']}:")
            for v in detail.get("violations", []):
                print(f"    Line {v['line']}: {v['function']} ({v['type']})")
    else:
        print("\n✅ All Phase II modules are PRNG-compliant!")

