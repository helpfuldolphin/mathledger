# RUNTIME_SAFETY_BARRIER_SPEC.md

## 1. Introduction

This document specifies the "Runtime Safety Barrier," a set of mandatory security controls and guardrails designed to contain and neutralize threats from untrusted worker code. The safety barrier is a core component of the Gemini F Verifier Runtime.

## 2. Forbidden Worker Behaviors

The following behaviors are strictly forbidden and will result in immediate termination of the worker.

*   **Arbitrary Filesystem Access:** Workers are prohibited from accessing any part of the filesystem outside of a designated, ephemeral scratch space. All input and output must be handled through standardized streams.
*   **Unauthorized Network Access:** Workers are not permitted to initiate network connections. All communication with external services must be proxied through the Verifier.
*   **Subprocess Spawning:** Workers are not allowed to spawn subprocesses, shells, or any other executable code.
*   **Use of Insecure Libraries/Calls:** The runtime will restrict the use of known insecure libraries and system calls. An allow-list of approved system calls will be enforced.

## 3. Mandatory Guardrails

The following guardrails are enforced by the Verifier Runtime to ensure the integrity and stability of the system.

*   **Strict Resource Limits:** The Verifier imposes hard limits on CPU time, memory allocation, and the number of open file handles. These limits are not negotiable by the worker.
*   **Sandboxing:** All worker code is executed in a sandboxed environment (e.g., a lightweight container or a seccomp-bpf filter) to isolate it from the host system and other workers.
*   **Input/Output Validation:** All data passed to and from a worker is rigorously validated and sanitized to prevent injection attacks and data corruption.
*   **Immutable Worker Images:** Workers are instantiated from immutable, cryptographically signed images. This ensures that the worker code cannot be tampered with at runtime.
