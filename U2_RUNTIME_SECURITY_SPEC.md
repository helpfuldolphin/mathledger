# U2 Runtime Security Specification

**DOCUMENT ID**: `U2_RUNTIME_SECURITY_SPEC.md`
**PHASE**: II
**OWNER**: Gemini K, U2 Security Systems Engineer
**STATUS**: DRAFT

---

## 1. Overview

This document specifies the runtime security model for U2 (Asymmetric Uplift) experiments. The primary goal of this model is to guarantee **manifest-bound determinism**: the assurance that a given experiment's outcome is a pure function of its pre-registered manifest. This guarantee is essential for producing verifiable, reproducible, and tamper-evident scientific results.

The security model is enforced by a series of runtime checks and architectural patterns implemented in `backend/security/u2_security.py` and integrated into the `rfl/runner.py` entry point.

---

## 2. End-to-End Threat Model

The system is modeled as a pipeline: **Manifest -> Runner Process -> Results**. Threats are considered at each stage.

| Stage | Threat Actor | Threat Description | Consequence |
| :--- | :--- | :--- | :--- |
| **Manifest (Input)** | Malicious/Careless Operator | The experiment manifest (`--config` file) is altered after the run begins. | **Goalpost Shifting**: The experiment's parameters are changed post-hoc to fit results, invalidating the scientific claim. |
| **Runner (Process)** | Malicious Code / Side Channel | **Illicit Randomness**: Code calls an uncontrolled source of entropy (e.g., `random.random()`, `os.urandom()`, system time). | **Irreproducibility**: The experiment cannot be re-run to produce the same result, destroying the determinism guarantee. |
| **Runner (Process)** | Misconfiguration | The runner is executed in the wrong environment mode or with an incorrect manifest. | **Invalid Execution**: The run proceeds without the required security guarantees, producing an invalid result that may be mistaken for a valid one. |
| **Runner (Process)** | State Leakage | State from a previous run (e.g., files, database entries, cached objects) influences the current run. | **Confounded Results**: The experiment's outcome is not a pure function of its manifest, as it is influenced by hidden variables from prior executions. |
| **Results (Output)** | Malicious/Careless Operator | Log files or JSON results are tampered with after the run is complete. | **Result Falsification**: The evidence of the experiment is altered to support a false claim. |

---

## 3. Runtime Protections & Enforcement

The following protections are implemented and enforced at runtime for any execution where `RFL_ENV_MODE` is `PHASE-II-U2`.

### 3.1. Environment Gating

- **Control**: The entire suite of U2 security checks is gated by the presence of the environment variable `RFL_ENV_MODE="PHASE-II-U2"`.
- **Enforcement**: The `main()` function in `rfl/runner.py` checks for this value at startup. If it is not present, the security module is not activated.
- **Threat Mitigated**: Invalid Execution.

### 3.2. Fail-Fast Pre-Flight Checks

- **Control**: Before any substantive work begins, the runner executes a mandatory sequence of security checks.
- **Enforcement**: `run_pre_flight_checks()` is called at the top of the `main` function. Failure of any check raises a `SecurityException`, which is caught and causes the process to exit with a non-zero status code.
- **Threat Mitigated**: Invalid Execution, Goalpost Shifting.

### 3.3. Manifest-Bound Determinism

This is the core of the security model, enforced by two mechanisms:

1.  **Manifest Hashing**:
    - **Control**: The manifest file provided via `--config` is treated as the canonical source of truth for the run. Its contents are hashed using SHA256.
    - **Enforcement**: `calculate_manifest_hash()` is called during pre-flight checks. The resulting hash is the foundation for the run's identity.
    - **Threat Mitigated**: Goalpost Shifting.

2.  **Manifest-Derived Seed Injection**:
    - **Control**: All sources of randomness must be seeded from the manifest hash.
    - **Enforcement**: The hex digest of the manifest hash is converted to an integer. This integer is used to:
        a. Initialize the `DeterministicPRNG` singleton guard.
        b. Forcefully overwrite the `random_seed` attribute of the `RFLConfig` object loaded from the manifest. This ensures all legacy code paths that depend on the config seed are re-routed to use the secure, manifest-derived seed.
    - **Threat Mitigated**: Illicit Randomness.

---

## 4. PRNG Guard Invariants

The `DeterministicPRNG` class in `u2_security.py` provides a controlled source of randomness. All new code written for U2 experiments that requires randomness MUST use this guard. It upholds the following invariants:

1.  **Singleton Access**: The class guarantees only one instance of the guard can exist, preventing multiple sources of controlled randomness.
2.  **Initialize-Once**: The guard's `initialize()` method can only be called once. Any subsequent attempt will raise a `SecurityException`.
3.  **Immutable Seed**: Once initialized, the internal `random.Random` instance cannot be re-seeded or accessed directly. All random operations are conducted through the guarded `prng` property.
4.  **Manifest-Bound**: The initialization seed MUST be the integer derived from the manifest hash. This is enforced by the startup logic in `rfl/runner.py`.

---

## 5. Forbidden Patterns

The following patterns are explicitly forbidden in U2-related code. Introduction of such code is considered a security violation and should be caught in code review.

**BAD**: Direct use of the global `random` module.
```python
# FORBIDDEN
import random
# ...
x = random.random() 
y = random.randint(0, 10)
```

**GOOD**: Using the `DeterministicPRNG` guard.
```python
from backend.security.u2_security import DeterministicPRNG

# ...
prng_guard = DeterministicPRNG()
# prng_guard has been initialized at startup
x = prng_guard.prng.random()
y = prng_guard.prng.randint(0, 10)
```

**BAD**: Using uncontrolled sources of entropy.
```python
# FORBIDDEN
import time
import os

# Using timestamps for algorithmic decisions
seed = int(time.time()) 

# Using OS-level randomness
entropy = os.urandom(16)
```

**BAD**: Unseeded use of `numpy.random`.
```python
# FORBIDDEN (if numpy's seed has not been set by the runner)
import numpy as np

arr = np.random.rand(3, 3)
```

**GOOD**: Relying on the runner's seed injection.
```python
# rfl/runner.py already calls np.random.seed() using the manifest-derived master seed.
# Therefore, subsequent calls are deterministic within the context of that run.
import numpy as np

# This is safe because the global numpy PRNG is seeded at startup.
arr = np.random.rand(3, 3)
```

---

## 6. Roadmap for Global Randomness Auditing

The current system relies on convention and startup-time seed injection. The following roadmap outlines a path to a more robust, near-ironclad guarantee of determinism.

### Phase 1: Convention and Injection (Current)
- **Description**: Rely on developers to use the `DeterministicPRNG` guard for new code. For existing code, inject the manifest-derived seed into the `RFLConfig` object at startup.
- **Strength**: Ensures major systems like `numpy` and the core experiment loop are seeded correctly.
- **Weakness**: An unnoticed call to `random.random()` in a deep library could still introduce non-determinism.

### Phase 2: Static Analysis Enforcement
- **Description**: Implement a custom `ruff` or `pylint` rule that scans the codebase for forbidden patterns, such as direct imports of `random` or calls to `random.random`.
- **Enforcement**: This check would be integrated into the CI/CD pipeline, failing any build that introduces such a pattern.
- **Strength**: Proactively prevents developers from adding new sources of illicit randomness.
- **Weakness**: Cannot detect dynamic or obscure methods of accessing randomness (e.g., via `getattr`).

### Phase 3: Runtime Monkey-Patching
- **Description**: During U2 run startup, dynamically "monkey-patch" Python's built-in `random` module and the `numpy.random` module. The patched versions would raise a `SecurityException` if any function is called, preventing all use except for an explicit, one-time seeding operation if needed.
- **Enforcement**: This is a runtime guard that makes it impossible for code to accidentally use the global random instances.
- **Strength**: Provides a very strong guarantee against the most common sources of non-determinism.
- **Weakness**: A sophisticated attacker could potentially circumvent the patch (e.g., by re-importing the original module).

### Phase 4: System Call Auditing
- **Description**: Leverage container-level `seccomp-bpf` profiles to create a hardened kernel interface for the U2 runner process. This profile would explicitly deny access to system calls that provide entropy, such as `getrandom(2)`.
- **Enforcement**: Enforced by the container runtime (Docker/Podman) and the Linux kernel itself.
- **Strength**: The ultimate guarantee. The operating system itself prevents the process from accessing external entropy, regardless of what the Python code attempts to do.
- **Weakness**: Requires more complex operational setup and maintenance of `seccomp` profiles.
