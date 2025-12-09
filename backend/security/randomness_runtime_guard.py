# backend/security/randomness_runtime_guard.py
"""
Phase III Runtime Guard for Determinism.

This module provides functionality to monkey-patch Python's randomness
modules at runtime, preventing any code from accessing uncontrolled
sources of entropy during a U2 run.

This is a powerful, high-level guard that complements the static linter.
"""

import random
import numpy.random
import os
import time
from contextlib import contextmanager

from .u2_security import SecurityException

# --- State ---

_original_functions = {}
_guard_is_active = False

# --- Forbidden Functions ---

def _forbidden_call_factory(name: str):
    """Creates a function that raises a SecurityException when called."""
    def forbidden_function(*args, **kwargs):
        raise SecurityException(
            f"Illegal call to '{name}' while randomness runtime guard is active. "
            "All randomness must originate from the manifest-bound DeterministicPRNG."
        )
    return forbidden_function

# --- Public API ---

@contextmanager
def allow_unrestricted_randomness():
    """
    A context manager to temporarily disable the runtime guard.
    
    This should ONLY be used for controlled, deterministic operations
    like the initial seeding of the PRNGs from the manifest hash.
    """
    global _guard_is_active
    if not _guard_is_active:
        yield
        return

    original_active_state = _guard_is_active
    _guard_is_active = False
    try:
        # Restore original functions temporarily
        _uninstall_patches()
        yield
    finally:
        # Re-install patches and restore guard state
        _install_patches()
        _guard_is_active = original_active_state


def activate_runtime_guard():
    """
    Monkey-patches standard library randomness functions to forbid their use.
    
    This function is NOT reversible. Once called, the guards are in place
    for the lifetime of the process, only bypassable via the
    `allow_unrestricted_randomness` context manager.
    """
    global _guard_is_active, _original_functions
    if _guard_is_active:
        return # Already activated

    print("--- Activating Randomness Runtime Guard ---")

    modules_to_patch = {
        "random": random,
        "numpy.random": numpy.random,
        "os": os,
        "time": time,
    }
    
    functions_to_patch = {
        "random": ["random", "randint", "randrange", "choice", "choices", "shuffle", "sample", "uniform"],
        "numpy.random": ["rand", "randn", "randint", "random_integers", "random_sample", "choice"],
        "os": ["urandom"],
        "time": ["time"],
    }
    
    # Store original functions and apply patches
    for module_name, module_obj in modules_to_patch.items():
        for func_name in functions_to_patch[module_name]:
            full_name = f"{module_name}.{func_name}"
            if hasattr(module_obj, func_name):
                _original_functions[full_name] = getattr(module_obj, func_name)
                setattr(module_obj, func_name, _forbidden_call_factory(full_name))
    
    _guard_is_active = True
    print("Runtime guard is ACTIVE. Uncontrolled randomness is now forbidden.")

def _install_patches():
    """Internal function to re-apply patches."""
    modules = {'random': random, 'numpy.random': numpy.random, 'os': os, 'time': time}
    for full_name, _ in _original_functions.items():
        module_name, func_name = full_name.rsplit('.', 1)
        if module_name in modules:
            setattr(modules[module_name], func_name, _forbidden_call_factory(full_name))

def _uninstall_patches():
    """Internal function to remove patches."""
    modules = {'random': random, 'numpy.random': numpy.random, 'os': os, 'time': time}
    for full_name, original_func in _original_functions.items():
        module_name, func_name = full_name.rsplit('.', 1)
        if module_name in modules:
            setattr(modules[module_name], func_name, original_func)
