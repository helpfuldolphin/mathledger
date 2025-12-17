# Lean Integration Specification

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Version**: 1.0  
**Status**: Test-Ready

---

## Overview

This document specifies the **exact interface** between the telemetry runtime and the Lean theorem prover. It defines command-line invocation, return formats, stdout/stderr patterns, and tactic extraction contracts.

---

## 1. Lean Command-Line Interface

### 1.1 Expected Lean Executable

**Primary**: `lake env lean` (Lean 4 with Lake build system)  
**Fallback**: `lean` (standalone Lean 4 executable)  
**Legacy**: `lean3` (Lean 3, if needed for compatibility)

**Detection**:
```bash
# Check Lean 4 with Lake
which lake && lake env lean --version

# Check standalone Lean 4
which lean && lean --version

# Check Lean 3
which lean3 && lean3 --version
```

**Expected Output**:
```
Lean (version 4.3.0, commit 1234567890ab, Release)
```

### 1.2 Verification Command

**Format**:
```bash
lake env lean <module_path> --timeout <timeout_s>
```

**Parameters**:
- `<module_path>`: Absolute or relative path to `.lean` file
- `--timeout <timeout_s>`: Timeout in seconds (integer)

**Example**:
```bash
lake env lean Mathlib/Algebra/Ring/Basic.lean --timeout 60
```

**Alternative Format** (for standalone Lean):
```bash
lean --timeout 60000 Mathlib/Algebra/Ring/Basic.lean
```
(Note: Lean 4 standalone uses milliseconds)

### 1.3 Environment Variables

**Required**:
- `LEAN_PATH`: Path to Lean libraries (usually set by Lake)
- `LEAN_SRC_PATH`: Path to source files

**Optional**:
- `LEAN_MEMORY_LIMIT`: Memory limit in MB (e.g., `4096`)
- `LEAN_THREADS`: Number of threads (e.g., `4`)

**Setup**:
```bash
# Using Lake (recommended)
export LEAN_PATH=$(lake env printenv LEAN_PATH)
export LEAN_SRC_PATH=$(lake env printenv LEAN_SRC_PATH)

# Manual setup (if Lake unavailable)
export LEAN_PATH=/path/to/lean/lib
export LEAN_SRC_PATH=/path/to/mathlib
```

### 1.4 Working Directory

**Requirement**: Command must be executed from MathLib root directory

**Example**:
```bash
cd /path/to/mathlib
lake env lean Mathlib/Algebra/Ring/Basic.lean --timeout 60
```

**Rationale**: Lean resolves imports relative to working directory

---

## 2. Return Formats

### 2.1 Return Codes

| Return Code | Meaning | Error Code Mapping |
|-------------|---------|-------------------|
| **0** | Success (proof verified) | `VERIFIED` |
| **1** | Proof invalid (type error, sorry, etc.) | `PROOF_INVALID` |
| **137** | Killed by SIGKILL (timeout or OOM) | `VERIFIER_TIMEOUT` or `MEMORY_LIMIT_EXCEEDED` |
| **139** | Segmentation fault | `VERIFIER_INTERNAL_ERROR` |
| **-9** | Killed by signal 9 (SIGKILL) | `VERIFIER_TIMEOUT` |
| **-15** | Killed by signal 15 (SIGTERM) | `VERIFIER_TIMEOUT` |
| **Other** | Unexpected error | `VERIFIER_INTERNAL_ERROR` |

**Disambiguation Logic**:
```python
def map_returncode_to_error_code(
    returncode: int,
    signal: Optional[int],
    stderr: str,
    duration_ms: float,
    timeout_ms: float,
) -> VerifierErrorCode:
    # Success
    if returncode == 0:
        return VerifierErrorCode.VERIFIED
    
    # Timeout (killed by signal or duration exceeded)
    if signal in [9, 15] or returncode in [137, -9, -15]:
        if duration_ms >= timeout_ms * 0.95:
            return VerifierErrorCode.VERIFIER_TIMEOUT
        else:
            # Killed but not due to timeout (likely OOM)
            if "out of memory" in stderr.lower():
                return VerifierErrorCode.MEMORY_LIMIT_EXCEEDED
            else:
                return VerifierErrorCode.VERIFIER_INTERNAL_ERROR
    
    # Proof invalid (type error, sorry, etc.)
    if returncode == 1:
        if any(pattern in stderr for pattern in PROOF_INVALID_PATTERNS):
            return VerifierErrorCode.PROOF_INVALID
        else:
            return VerifierErrorCode.VERIFIER_INTERNAL_ERROR
    
    # Segmentation fault
    if returncode == 139 or signal == 11:
        return VerifierErrorCode.VERIFIER_INTERNAL_ERROR
    
    # Other errors
    return VerifierErrorCode.VERIFIER_INTERNAL_ERROR
```

### 2.2 Stdout Format

**Success Case** (returncode 0):
```
# No output (Lean 4 is silent on success)
```

**Alternative** (verbose mode):
```
Mathlib.Algebra.Ring.Basic:123:45: info: proof complete
```

**Tactic Trace** (if `set_option trace.tactic true`):
```
[tactic.apply] applied theorem foo
[tactic.rw] rewrote with bar
[tactic.simp] simplified to baz
```

### 2.3 Stderr Format

**Proof Invalid**:
```
Mathlib/Algebra/Ring/Basic.lean:123:45: error: type mismatch
  expected: Nat
  got: Int
```

**Timeout** (no stderr, killed by signal):
```
(empty)
```

**Out of Memory**:
```
out of memory
```

**Internal Error**:
```
PANIC at <location>: <error message>
```

**Sorry (incomplete proof)**:
```
Mathlib/Algebra/Ring/Basic.lean:123:45: warning: declaration uses 'sorry'
```

---

## 3. Stdout/Stderr Patterns

### 3.1 Proof Invalid Patterns

**Type Mismatch**:
```regex
error: type mismatch
```

**Unknown Identifier**:
```regex
error: unknown identifier '.*'
```

**Tactic Failed**:
```regex
error: tactic '.*' failed
```

**Unsolved Goals**:
```regex
error: unsolved goals
```

**Sorry**:
```regex
warning: declaration uses 'sorry'
```

**Pattern List**:
```python
PROOF_INVALID_PATTERNS = [
    "error: type mismatch",
    "error: unknown identifier",
    "error: tactic .* failed",
    "error: unsolved goals",
    "warning: declaration uses 'sorry'",
    "error: invalid",
    "error: expected",
]
```

### 3.2 Resource Constraint Patterns

**Out of Memory**:
```regex
out of memory
```

**Stack Overflow**:
```regex
stack overflow
```

**Timeout** (explicit):
```regex
timeout
```

**Pattern List**:
```python
RESOURCE_CONSTRAINT_PATTERNS = [
    "out of memory",
    "stack overflow",
    "timeout",
    "resource limit exceeded",
]
```

### 3.3 Internal Error Patterns

**Panic**:
```regex
PANIC at .*: .*
```

**Assertion Failed**:
```regex
assertion failed: .*
```

**Segmentation Fault** (from system, not Lean):
```regex
Segmentation fault
```

**Pattern List**:
```python
INTERNAL_ERROR_PATTERNS = [
    "PANIC at",
    "assertion failed",
    "Segmentation fault",
    "internal error",
    "unreachable code",
]
```

### 3.4 Pattern Matching Function

```python
def classify_stderr(stderr: str) -> VerifierErrorCode:
    stderr_lower = stderr.lower()
    
    # Check proof invalid patterns
    for pattern in PROOF_INVALID_PATTERNS:
        if pattern.lower() in stderr_lower:
            return VerifierErrorCode.PROOF_INVALID
    
    # Check resource constraint patterns
    for pattern in RESOURCE_CONSTRAINT_PATTERNS:
        if pattern.lower() in stderr_lower:
            if "memory" in pattern.lower():
                return VerifierErrorCode.MEMORY_LIMIT_EXCEEDED
            else:
                return VerifierErrorCode.VERIFIER_TIMEOUT
    
    # Check internal error patterns
    for pattern in INTERNAL_ERROR_PATTERNS:
        if pattern.lower() in stderr_lower:
            return VerifierErrorCode.VERIFIER_INTERNAL_ERROR
    
    # Default: internal error if stderr non-empty
    if stderr.strip():
        return VerifierErrorCode.VERIFIER_INTERNAL_ERROR
    
    # No stderr: should not happen if returncode != 0
    return VerifierErrorCode.VERIFIER_INTERNAL_ERROR
```

---

## 4. Tactic Extraction Contract

### 4.1 Tactic Trace Activation

**Method 1**: Command-line option (Lean 4)
```bash
lean --trace tactic Mathlib/Algebra/Ring/Basic.lean
```

**Method 2**: Set option in file
```lean
set_option trace.tactic true
```

**Method 3**: Environment variable
```bash
export LEAN_TRACE=tactic
lean Mathlib/Algebra/Ring/Basic.lean
```

**Recommendation**: Method 1 (command-line) for per-invocation control

### 4.2 Tactic Trace Format

**Example Output**:
```
[tactic.apply] applied theorem Nat.add_comm
[tactic.rw] rewrote with Nat.mul_comm
[tactic.simp] simplified to 0
[tactic.ring] ring normalization complete
```

**Pattern**:
```regex
\[tactic\.([a-z_]+)\] (.*)
```

**Capture Groups**:
1. Tactic name (e.g., `apply`, `rw`, `simp`)
2. Tactic message (e.g., `applied theorem Nat.add_comm`)

### 4.3 Tactic Extraction Function

```python
import re
from typing import List, Dict

TACTIC_PATTERN = re.compile(r'\[tactic\.([a-z_]+)\] (.*)')

def extract_tactics_from_trace(stdout: str) -> Dict[str, int]:
    """Extract tactic usage counts from Lean trace output.
    
    Args:
        stdout: Lean stdout with tactic trace
    
    Returns:
        Dict mapping tactic name to usage count
    """
    tactic_counts = {}
    
    for line in stdout.split('\n'):
        match = TACTIC_PATTERN.match(line)
        if match:
            tactic_name = match.group(1)
            tactic_counts[tactic_name] = tactic_counts.get(tactic_name, 0) + 1
    
    return tactic_counts
```

### 4.4 Fallback: Proof Text Parsing

**If tactic trace unavailable**, parse proof text directly:

**Method**: Read `.lean` file and extract tactics from proof blocks

**Proof Block Pattern**:
```regex
(theorem|lemma|def) .* := by\n(.*?)\n\n
```

**Tactic Pattern** (within proof block):
```regex
^\s*(apply|rw|simp|ring|exact|intro|cases|induction|refl|conv|norm_num|omega|tauto|decide)
```

**Extraction Function**:
```python
def extract_tactics_from_proof_text(lean_file: Path, theorem_name: str) -> List[str]:
    """Extract tactics from proof text in .lean file.
    
    Args:
        lean_file: Path to .lean file
        theorem_name: Name of theorem to extract
    
    Returns:
        List of tactic names used in proof
    """
    with open(lean_file) as f:
        content = f.read()
    
    # Find theorem
    theorem_pattern = re.compile(
        rf'(theorem|lemma|def) {theorem_name}.*:= by\n(.*?)\n\n',
        re.DOTALL
    )
    match = theorem_pattern.search(content)
    if not match:
        return []
    
    proof_text = match.group(2)
    
    # Extract tactics
    tactic_pattern = re.compile(
        r'^\s*(apply|rw|simp|ring|exact|intro|cases|induction|refl|conv|norm_num|omega|tauto|decide)',
        re.MULTILINE
    )
    tactics = tactic_pattern.findall(proof_text)
    
    return tactics
```

**Limitation**: This method only extracts top-level tactics, not nested tactics or tactic combinators.

### 4.5 Tactic Depth Estimation

**Heuristic 1**: Count `begin`/`end` or `by` nesting levels

```python
def estimate_tactic_depth(proof_text: str) -> int:
    """Estimate tactic nesting depth from proof text.
    
    Args:
        proof_text: Proof text (between `by` and end of proof)
    
    Returns:
        Maximum nesting depth
    """
    depth = 0
    max_depth = 0
    
    for line in proof_text.split('\n'):
        # Count opening delimiters
        depth += line.count('{') + line.count('begin') + line.count('by')
        
        # Count closing delimiters
        depth -= line.count('}') + line.count('end')
        
        # Track maximum
        max_depth = max(max_depth, depth)
    
    return max_depth
```

**Heuristic 2**: Count tactic combinator nesting (`<;>`, `<|>`, etc.)

```python
def count_tactic_combinators(proof_text: str) -> int:
    """Count tactic combinator usage.
    
    Args:
        proof_text: Proof text
    
    Returns:
        Number of tactic combinators
    """
    combinators = ['<;>', '<|>', '>>', '>>>', 'try', 'repeat', 'iterate']
    count = 0
    
    for combinator in combinators:
        count += proof_text.count(combinator)
    
    return count
```

### 4.6 Proof Size Estimation

**Method 1**: Count characters in proof text
```python
def estimate_proof_size(proof_text: str) -> int:
    return len(proof_text.encode('utf-8'))
```

**Method 2**: Count proof term size (if available in Lean output)
```regex
proof term size: (\d+) bytes
```

**Method 3**: Count AST nodes (requires Lean API access)

---

## 5. Integration Testing

### 5.1 Test Cases

**Test 1: Successful Verification**

**Input**:
```lean
-- test_success.lean
theorem test_success : 1 + 1 = 2 := by
  rfl
```

**Expected**:
- Return code: 0
- Stdout: (empty or success message)
- Stderr: (empty)
- Error code: `VERIFIED`
- Tactics: `["rfl"]`

**Test 2: Type Mismatch**

**Input**:
```lean
-- test_type_mismatch.lean
theorem test_type_mismatch : 1 + 1 = "2" := by
  rfl
```

**Expected**:
- Return code: 1
- Stdout: (empty)
- Stderr: `error: type mismatch`
- Error code: `PROOF_INVALID`

**Test 3: Timeout**

**Input**:
```lean
-- test_timeout.lean
theorem test_timeout : True := by
  sorry  -- Replace with infinite loop
```

**Command**: `lean test_timeout.lean --timeout 1`

**Expected**:
- Return code: 137 or -9
- Stdout: (empty)
- Stderr: (empty or timeout message)
- Error code: `VERIFIER_TIMEOUT`
- Duration: ≥ 1000ms

**Test 4: Out of Memory**

**Input**: Large proof that exceeds memory limit

**Expected**:
- Return code: 137
- Stdout: (empty)
- Stderr: `out of memory`
- Error code: `MEMORY_LIMIT_EXCEEDED`

**Test 5: Tactic Extraction**

**Input**:
```lean
-- test_tactics.lean
theorem test_tactics (n : Nat) : n + 0 = n := by
  apply Nat.add_zero
```

**Expected**:
- Tactics: `["apply"]`
- Tactic counts: `{"apply": 1}`

### 5.2 Test Execution

**Script**: `backend/verification/tests/test_lean_integration.py`

```python
import subprocess
from pathlib import Path

def test_lean_integration():
    # Test 1: Success
    result = subprocess.run(
        ["lake", "env", "lean", "test_success.lean", "--timeout", "10"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert map_returncode_to_error_code(result.returncode, None, result.stderr, 100, 10000) == VerifierErrorCode.VERIFIED
    
    # Test 2: Type mismatch
    result = subprocess.run(
        ["lake", "env", "lean", "test_type_mismatch.lean", "--timeout", "10"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "type mismatch" in result.stderr
    assert classify_stderr(result.stderr) == VerifierErrorCode.PROOF_INVALID
    
    # Test 3: Timeout
    result = subprocess.run(
        ["lake", "env", "lean", "test_timeout.lean", "--timeout", "1"],
        capture_output=True,
        text=True,
        timeout=2,
    )
    assert result.returncode in [137, -9]
    
    # Test 4: Tactic extraction
    result = subprocess.run(
        ["lake", "env", "lean", "test_tactics.lean", "--trace", "tactic"],
        capture_output=True,
        text=True,
    )
    tactics = extract_tactics_from_trace(result.stdout)
    assert "apply" in tactics
    assert tactics["apply"] >= 1
```

---

## 6. Error Handling

### 6.1 Lean Executable Not Found

**Detection**:
```python
try:
    subprocess.run(["lake", "env", "lean", "--version"], check=True)
except FileNotFoundError:
    raise LeanNotFoundError("Lean executable not found. Install Lean 4 and Lake.")
```

**Recovery**: Provide installation instructions

### 6.2 Module Not Found

**Detection**: Stderr contains `error: file not found`

**Recovery**: Verify module path and working directory

### 6.3 Import Error

**Detection**: Stderr contains `error: unknown package` or `error: import not found`

**Recovery**: Build MathLib dependencies (`lake build`)

### 6.4 Lean Version Mismatch

**Detection**: Stderr contains `error: version mismatch`

**Recovery**: Update Lean or MathLib to compatible versions

---

## 7. Performance Optimization

### 7.1 Caching

**Strategy**: Cache verification results by module hash

**Implementation**:
```python
import hashlib
from pathlib import Path

def get_module_hash(module_path: Path) -> str:
    """Compute hash of module file."""
    with open(module_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def check_cache(module_path: Path) -> Optional[VerifierOutcome]:
    """Check if verification result is cached."""
    module_hash = get_module_hash(module_path)
    cache_path = Path(f".cache/verification/{module_hash}.json")
    
    if cache_path.exists():
        with open(cache_path) as f:
            return VerifierOutcome.from_dict(json.load(f))
    
    return None

def save_cache(module_path: Path, outcome: VerifierOutcome):
    """Save verification result to cache."""
    module_hash = get_module_hash(module_path)
    cache_path = Path(f".cache/verification/{module_hash}.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, 'w') as f:
        json.dump(outcome.to_dict(), f)
```

### 7.2 Parallel Execution

**Strategy**: Run multiple Lean processes in parallel

**Implementation**: Use `multiprocessing.Pool` (already in calibration CLI)

**Constraint**: Limit workers to avoid memory exhaustion (16-32 workers on 64GB system)

### 7.3 Incremental Compilation

**Strategy**: Use Lake's incremental compilation

**Implementation**: Run `lake build` before verification to ensure dependencies are compiled

---

## 8. Summary

This Lean integration specification provides a **complete, test-ready interface**:

✅ **Command-Line Interface**: `lake env lean <module> --timeout <s>`  
✅ **Return Codes**: 11 return codes mapped to stable error codes  
✅ **Stdout/Stderr Patterns**: 15+ patterns for error classification  
✅ **Tactic Extraction**: Trace-based and text-based extraction  
✅ **Integration Tests**: 5 test cases covering all scenarios  
✅ **Error Handling**: Detection and recovery for common errors  
✅ **Performance Optimization**: Caching, parallelization, incremental compilation

**Status**: Test-ready. Ready for implementation and integration testing.

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*
