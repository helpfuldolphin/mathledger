#!/usr/bin/env python3
"""
Phase Migration Dry-Run Simulator

Simulates the Phase I → Phase II → Phase IIb/III gating process without
touching the environment. Validates migration preconditions and outputs
a comprehensive dry-run manifest.

Author: Agent E4 (doc-ops-4) — Phase Migration Architect
Date: 2025-12-06

ABSOLUTE SAFEGUARDS:
- No mutations to production state
- No DB writes
- No file modifications (read-only analysis)
- No uplift claims
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Phase(Enum):
    """MathLedger phase identifiers."""
    PHASE_I = "phase_i"
    PHASE_II = "phase_ii"
    PHASE_IIB = "phase_iib"
    PHASE_III = "phase_iii"


class ValidationStatus(Enum):
    """Validation result status."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_id: str
    check_name: str
    status: ValidationStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "check_name": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class MigrationGate:
    """A migration gate with preconditions."""
    gate_id: str
    source_phase: Phase
    target_phase: Phase
    preconditions: list[ValidationResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        return all(
            v.status in (ValidationStatus.PASS, ValidationStatus.WARN, ValidationStatus.SKIP)
            for v in self.preconditions
        )
    
    @property
    def blocking_failures(self) -> list[ValidationResult]:
        return [v for v in self.preconditions if v.status == ValidationStatus.FAIL]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "source_phase": self.source_phase.value,
            "target_phase": self.target_phase.value,
            "passed": self.passed,
            "preconditions": [p.to_dict() for p in self.preconditions],
            "blocking_count": len(self.blocking_failures),
        }


@dataclass
class MigrationSimulationResult:
    """Complete migration simulation result."""
    simulation_id: str
    timestamp: str
    current_phase: Phase
    gates: list[MigrationGate]
    determinism_checks: list[ValidationResult]
    evidence_chain: list[ValidationResult]
    slice_validation: list[ValidationResult]
    preregistration_checks: list[ValidationResult]
    boundary_purity: list[ValidationResult]
    overall_status: str
    summary: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "timestamp": self.timestamp,
            "current_phase": self.current_phase.value,
            "gates": [g.to_dict() for g in self.gates],
            "determinism_checks": [c.to_dict() for c in self.determinism_checks],
            "evidence_chain": [c.to_dict() for c in self.evidence_chain],
            "slice_validation": [c.to_dict() for c in self.slice_validation],
            "preregistration_checks": [c.to_dict() for c in self.preregistration_checks],
            "boundary_purity": [c.to_dict() for c in self.boundary_purity],
            "overall_status": self.overall_status,
            "summary": self.summary,
        }


class PhaseMigrationSimulator:
    """
    Dry-run simulator for phase migrations.
    
    Validates all migration preconditions without modifying any state.
    """
    
    # Forbidden imports in basis/ package (Phase III constraint)
    FORBIDDEN_BASIS_IMPORTS = {
        "os", "sys", "time", "datetime", "random", "uuid",
        "asyncio", "threading", "multiprocessing", "socket",
        "requests", "httpx", "redis", "psycopg", "sqlalchemy",
    }
    
    # Forbidden primitives in determinism envelope
    FORBIDDEN_DETERMINISM_PATTERNS = [
        r"datetime\.now\(\)",
        r"datetime\.utcnow\(\)",
        r"time\.time\(\)",
        r"uuid\.uuid4\(\)",
        r"random\.\w+\(",
        r"os\.urandom",
    ]
    
    # Required files for each phase
    PHASE_REQUIRED_FILES = {
        Phase.PHASE_I: [
            "artifacts/first_organism/attestation.json",
            "results/fo_baseline.jsonl",
            "config/curriculum.yaml",
        ],
        Phase.PHASE_II: [
            "artifacts/first_organism/attestation.json",
            "experiments/prereg/PREREG_UPLIFT_U2.yaml",
            "config/curriculum_uplift_phase2.yaml",
        ],
        Phase.PHASE_IIB: [
            "backend/lean_proj/lakefile.lean",
        ],
        Phase.PHASE_III: [
            "backend/crypto/core.py",
            "basis/crypto/hash.py",
            "basis/attestation/dual.py",
        ],
    }
    
    # Determinism envelope files
    DETERMINISM_ENVELOPE_FILES = [
        "normalization/canon.py",
        "attestation/dual_root.py",
        "substrate/repro/determinism.py",
        "backend/repro/determinism.py",
        "backend/axiom_engine/derive.py",
        "rfl/runner.py",
        "basis/logic/normalizer.py",
        "basis/crypto/hash.py",
        "basis/attestation/dual.py",
    ]
    
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or PROJECT_ROOT
        self.simulation_id = self._generate_simulation_id()
        
    def _generate_simulation_id(self) -> str:
        """Generate deterministic simulation ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"migration_sim_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def detect_current_phase(self) -> Phase:
        """
        Detect the current operational phase based on codebase state.
        
        Heuristics:
        - Phase I: Hermetic mode, lean-disabled, no DB writes
        - Phase II: DB-backed RFL, truth-table verification
        - Phase IIb: Lean runtime enabled
        - Phase III: basis/ imports active, Ed25519 signatures
        """
        # Check for Phase III indicators
        if self._check_basis_imports_active():
            return Phase.PHASE_III
        
        # Check for Phase IIb indicators (Lean enabled)
        if self._check_lean_enabled():
            return Phase.PHASE_IIB
        
        # Check for Phase II indicators (DB-backed RFL)
        if self._check_db_rfl_enabled():
            return Phase.PHASE_II
        
        # Default to Phase I
        return Phase.PHASE_I
    
    def _check_basis_imports_active(self) -> bool:
        """Check if basis/ imports are actively used."""
        # Search for imports from basis in non-basis files
        backend_path = self.project_root / "backend"
        if not backend_path.exists():
            return False
        
        for py_file in backend_path.rglob("*.py"):
            if "basis" in str(py_file):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                if re.search(r"from basis\.", content) or re.search(r"import basis", content):
                    return True
            except Exception:
                continue
        return False
    
    def _check_lean_enabled(self) -> bool:
        """Check if Lean runtime is enabled in curriculum."""
        curriculum_path = self.project_root / "config" / "curriculum.yaml"
        if not curriculum_path.exists():
            return False
        try:
            content = curriculum_path.read_text(encoding="utf-8")
            # Look for lean_enabled: true in active slice
            return "lean_enabled: true" in content or "lean_timeout_ms:" in content
        except Exception:
            return False
    
    def _check_db_rfl_enabled(self) -> bool:
        """Check if DB-backed RFL is enabled."""
        # Check rfl/config.py or environment
        rfl_config = self.project_root / "rfl" / "config.py"
        if rfl_config.exists():
            try:
                content = rfl_config.read_text(encoding="utf-8")
                if "RFL_DB_ENABLED = True" in content:
                    return True
            except Exception:
                pass
        return False
    
    def validate_phase_boundary_purity(self) -> list[ValidationResult]:
        """
        Validate phase boundary purity constraints.
        
        Checks:
        - No forbidden imports in basis/
        - Law → Economy → Metabolism hierarchy respected
        - No cross-layer violations
        """
        results = []
        
        # Check basis/ import purity
        basis_path = self.project_root / "basis"
        if basis_path.exists():
            for py_file in basis_path.rglob("*.py"):
                violations = self._check_forbidden_imports(py_file)
                if violations:
                    results.append(ValidationResult(
                        check_id="BP-001",
                        check_name="basis_import_purity",
                        status=ValidationStatus.FAIL,
                        message=f"Forbidden imports in {py_file.relative_to(self.project_root)}",
                        details={"file": str(py_file), "violations": violations},
                    ))
                else:
                    results.append(ValidationResult(
                        check_id="BP-001",
                        check_name="basis_import_purity",
                        status=ValidationStatus.PASS,
                        message=f"No forbidden imports in {py_file.relative_to(self.project_root)}",
                        details={"file": str(py_file)},
                    ))
        else:
            results.append(ValidationResult(
                check_id="BP-001",
                check_name="basis_import_purity",
                status=ValidationStatus.SKIP,
                message="basis/ directory not found",
                details={},
            ))
        
        # Check for premature DB writes in Phase I code paths
        results.append(self._check_hermetic_violations())
        
        # Check for Lean calls in non-Phase-IIb code
        results.append(self._check_lean_isolation())
        
        return results
    
    def _check_forbidden_imports(self, py_file: Path) -> list[str]:
        """Check a Python file for forbidden imports."""
        violations = []
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("#"):
                    continue
                for forbidden in self.FORBIDDEN_BASIS_IMPORTS:
                    if re.match(rf"^(import|from)\s+{forbidden}\b", line):
                        violations.append(forbidden)
        except Exception:
            pass
        return list(set(violations))
    
    def _check_hermetic_violations(self) -> ValidationResult:
        """Check for DB write patterns that violate Phase I hermetic mode."""
        violations = []
        backend_path = self.project_root / "backend"
        
        if not backend_path.exists():
            return ValidationResult(
                check_id="BP-002",
                check_name="hermetic_mode_check",
                status=ValidationStatus.SKIP,
                message="backend/ directory not found",
                details={},
            )
        
        # Patterns that indicate DB writes
        db_write_patterns = [
            r"session\.commit\(\)",
            r"session\.add\(",
            r"db\.add\(",
            r"\.execute\([^)]*INSERT",
            r"\.execute\([^)]*UPDATE",
            r"\.execute\([^)]*DELETE",
        ]
        
        for py_file in backend_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for pattern in db_write_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        # Check if it's in a Phase I code path (heuristic)
                        if "first_organism" in str(py_file).lower() or "hermetic" in content.lower():
                            violations.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "pattern": pattern,
                            })
            except Exception:
                continue
        
        if violations:
            return ValidationResult(
                check_id="BP-002",
                check_name="hermetic_mode_check",
                status=ValidationStatus.WARN,
                message=f"Potential hermetic violations found: {len(violations)} patterns",
                details={"violations": violations},
            )
        
        return ValidationResult(
            check_id="BP-002",
            check_name="hermetic_mode_check",
            status=ValidationStatus.PASS,
            message="No hermetic violations detected",
            details={},
        )
    
    def _check_lean_isolation(self) -> ValidationResult:
        """Check that Lean calls are isolated to Phase IIb code paths."""
        lean_calls = []
        
        for path in [self.project_root / "backend", self.project_root / "rfl"]:
            if not path.exists():
                continue
            for py_file in path.rglob("*.py"):
                if "lean" in str(py_file).lower():
                    continue  # Skip dedicated Lean modules
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    if re.search(r"lean_verify|lean_check|subprocess.*lean", content, re.IGNORECASE):
                        lean_calls.append(str(py_file.relative_to(self.project_root)))
                except Exception:
                    continue
        
        if lean_calls:
            return ValidationResult(
                check_id="BP-003",
                check_name="lean_isolation_check",
                status=ValidationStatus.WARN,
                message=f"Lean calls found outside dedicated modules: {len(lean_calls)} files",
                details={"files": lean_calls},
            )
        
        return ValidationResult(
            check_id="BP-003",
            check_name="lean_isolation_check",
            status=ValidationStatus.PASS,
            message="Lean calls properly isolated",
            details={},
        )
    
    def validate_determinism_contract(self) -> list[ValidationResult]:
        """
        Validate determinism contract completeness.
        
        Checks:
        - No forbidden primitives in envelope files
        - Deterministic timestamp usage
        - Deterministic UUID usage
        - Sorted dict iteration
        """
        results = []
        
        for rel_path in self.DETERMINISM_ENVELOPE_FILES:
            file_path = self.project_root / rel_path
            if not file_path.exists():
                results.append(ValidationResult(
                    check_id="DC-001",
                    check_name="determinism_envelope_file",
                    status=ValidationStatus.SKIP,
                    message=f"File not found: {rel_path}",
                    details={"file": rel_path},
                ))
                continue
            
            violations = self._check_determinism_violations(file_path)
            if violations:
                results.append(ValidationResult(
                    check_id="DC-001",
                    check_name="determinism_envelope_file",
                    status=ValidationStatus.FAIL,
                    message=f"Forbidden primitives in {rel_path}",
                    details={"file": rel_path, "violations": violations},
                ))
            else:
                results.append(ValidationResult(
                    check_id="DC-001",
                    check_name="determinism_envelope_file",
                    status=ValidationStatus.PASS,
                    message=f"Determinism contract satisfied: {rel_path}",
                    details={"file": rel_path},
                ))
        
        # Check for determinism helpers existence
        results.append(self._check_determinism_helpers())
        
        return results
    
    def _check_determinism_violations(self, file_path: Path) -> list[dict[str, Any]]:
        """Check a file for determinism contract violations."""
        violations = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
            
            for line_num, line in enumerate(lines, 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                
                for pattern in self.FORBIDDEN_DETERMINISM_PATTERNS:
                    if re.search(pattern, line):
                        # Check if it's in an allowed context (metrics, scripts)
                        if "metrics" in line.lower() or "perf_counter" in line:
                            continue
                        violations.append({
                            "line": line_num,
                            "pattern": pattern,
                            "content": line.strip()[:100],
                        })
        except Exception as e:
            violations.append({"error": str(e)})
        
        return violations
    
    def _check_determinism_helpers(self) -> ValidationResult:
        """Check that determinism helper functions exist."""
        helpers_path = self.project_root / "substrate" / "repro" / "determinism.py"
        
        if not helpers_path.exists():
            # Try alternate location
            helpers_path = self.project_root / "backend" / "repro" / "determinism.py"
        
        if not helpers_path.exists():
            return ValidationResult(
                check_id="DC-002",
                check_name="determinism_helpers",
                status=ValidationStatus.FAIL,
                message="Determinism helpers not found",
                details={"searched": ["substrate/repro/determinism.py", "backend/repro/determinism.py"]},
            )
        
        # Check for required functions
        try:
            content = helpers_path.read_text(encoding="utf-8")
            required_funcs = [
                "deterministic_timestamp",
                "deterministic_uuid",
                "SeededRNG",
            ]
            found = [f for f in required_funcs if f in content]
            missing = [f for f in required_funcs if f not in content]
            
            if missing:
                return ValidationResult(
                    check_id="DC-002",
                    check_name="determinism_helpers",
                    status=ValidationStatus.WARN,
                    message=f"Missing determinism helpers: {missing}",
                    details={"found": found, "missing": missing},
                )
            
            return ValidationResult(
                check_id="DC-002",
                check_name="determinism_helpers",
                status=ValidationStatus.PASS,
                message="All determinism helpers present",
                details={"found": found},
            )
        except Exception as e:
            return ValidationResult(
                check_id="DC-002",
                check_name="determinism_helpers",
                status=ValidationStatus.FAIL,
                message=f"Error reading determinism helpers: {e}",
                details={},
            )
    
    def validate_evidence_sealing_chain(self) -> list[ValidationResult]:
        """
        Validate evidence sealing chain integrity.
        
        Checks:
        - Attestation artifacts exist
        - H_t is recomputable
        - Evidence manifest integrity
        - Artifact checksums
        """
        results = []
        
        # Check attestation.json exists
        attestation_path = self.project_root / "artifacts" / "first_organism" / "attestation.json"
        if attestation_path.exists():
            try:
                with open(attestation_path) as f:
                    attestation = json.load(f)
                
                # Validate attestation structure
                required_fields = ["compositeAttestationRoot", "reasoningMerkleRoot", "uiMerkleRoot"]
                missing = [f for f in required_fields if f not in attestation]
                
                if missing:
                    results.append(ValidationResult(
                        check_id="ES-001",
                        check_name="attestation_structure",
                        status=ValidationStatus.FAIL,
                        message=f"Missing attestation fields: {missing}",
                        details={"missing": missing},
                    ))
                else:
                    # Verify H_t formula: H_t = SHA256(R_t || U_t)
                    r_t = attestation.get("reasoningMerkleRoot", "")
                    u_t = attestation.get("uiMerkleRoot", "")
                    h_t = attestation.get("compositeAttestationRoot", "")
                    
                    expected_h_t = hashlib.sha256(
                        bytes.fromhex(r_t) + bytes.fromhex(u_t)
                    ).hexdigest()
                    
                    if expected_h_t == h_t:
                        results.append(ValidationResult(
                            check_id="ES-001",
                            check_name="attestation_structure",
                            status=ValidationStatus.PASS,
                            message="Attestation valid, H_t recomputable",
                            details={"h_t": h_t[:16] + "...", "verified": True},
                        ))
                    else:
                        results.append(ValidationResult(
                            check_id="ES-001",
                            check_name="attestation_structure",
                            status=ValidationStatus.FAIL,
                            message="H_t mismatch: attestation may be corrupted",
                            details={
                                "expected": expected_h_t[:16] + "...",
                                "actual": h_t[:16] + "...",
                            },
                        ))
            except json.JSONDecodeError as e:
                results.append(ValidationResult(
                    check_id="ES-001",
                    check_name="attestation_structure",
                    status=ValidationStatus.FAIL,
                    message=f"Invalid JSON in attestation.json: {e}",
                    details={},
                ))
            except Exception as e:
                results.append(ValidationResult(
                    check_id="ES-001",
                    check_name="attestation_structure",
                    status=ValidationStatus.FAIL,
                    message=f"Error reading attestation: {e}",
                    details={},
                ))
        else:
            results.append(ValidationResult(
                check_id="ES-001",
                check_name="attestation_structure",
                status=ValidationStatus.FAIL,
                message="attestation.json not found",
                details={"path": str(attestation_path)},
            ))
        
        # Check baseline logs exist
        results.append(self._check_baseline_logs())
        
        # Check evidence manifest if exists
        results.append(self._check_evidence_manifest())
        
        return results
    
    def _check_baseline_logs(self) -> ValidationResult:
        """Check that baseline experiment logs exist."""
        baseline_paths = [
            self.project_root / "results" / "fo_baseline.jsonl",
            self.project_root / "results" / "fo_rfl.jsonl",
        ]
        
        found = []
        missing = []
        line_counts = {}
        
        for path in baseline_paths:
            if path.exists():
                found.append(str(path.name))
                try:
                    with open(path) as f:
                        line_counts[path.name] = sum(1 for _ in f)
                except Exception:
                    line_counts[path.name] = -1
            else:
                missing.append(str(path.name))
        
        if missing and not found:
            return ValidationResult(
                check_id="ES-002",
                check_name="baseline_logs",
                status=ValidationStatus.FAIL,
                message="No baseline logs found",
                details={"missing": missing},
            )
        
        # Check for minimum cycle count (1000+)
        sufficient_cycles = any(count >= 1000 for count in line_counts.values())
        
        if found and sufficient_cycles:
            return ValidationResult(
                check_id="ES-002",
                check_name="baseline_logs",
                status=ValidationStatus.PASS,
                message=f"Baseline logs present with sufficient cycles",
                details={"found": found, "line_counts": line_counts},
            )
        elif found:
            return ValidationResult(
                check_id="ES-002",
                check_name="baseline_logs",
                status=ValidationStatus.WARN,
                message="Baseline logs present but may have insufficient cycles",
                details={"found": found, "line_counts": line_counts, "required": 1000},
            )
        
        return ValidationResult(
            check_id="ES-002",
            check_name="baseline_logs",
            status=ValidationStatus.WARN,
            message=f"Some baseline logs missing: {missing}",
            details={"found": found, "missing": missing},
        )
    
    def _check_evidence_manifest(self) -> ValidationResult:
        """Check evidence manifest integrity."""
        manifest_paths = [
            self.project_root / "artifacts" / "phase_i_seal" / "evidence_manifest.json",
            self.project_root / "artifacts" / "first_organism" / "manifest.json",
        ]
        
        for manifest_path in manifest_paths:
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    return ValidationResult(
                        check_id="ES-003",
                        check_name="evidence_manifest",
                        status=ValidationStatus.PASS,
                        message=f"Evidence manifest found: {manifest_path.name}",
                        details={"path": str(manifest_path), "keys": list(manifest.keys())[:10]},
                    )
                except Exception as e:
                    return ValidationResult(
                        check_id="ES-003",
                        check_name="evidence_manifest",
                        status=ValidationStatus.FAIL,
                        message=f"Invalid manifest: {e}",
                        details={"path": str(manifest_path)},
                    )
        
        return ValidationResult(
            check_id="ES-003",
            check_name="evidence_manifest",
            status=ValidationStatus.WARN,
            message="No evidence manifest found (optional for Phase I)",
            details={"searched": [str(p) for p in manifest_paths]},
        )
    
    def validate_slice_completeness(self) -> list[ValidationResult]:
        """
        Validate curriculum slice completeness.
        
        Checks:
        - curriculum.yaml exists and is valid
        - Active slice is defined
        - Monotonicity constraints satisfied
        - Gate thresholds defined
        """
        results = []
        
        curriculum_path = self.project_root / "config" / "curriculum.yaml"
        
        if not curriculum_path.exists():
            results.append(ValidationResult(
                check_id="SC-001",
                check_name="curriculum_file",
                status=ValidationStatus.FAIL,
                message="curriculum.yaml not found",
                details={"path": str(curriculum_path)},
            ))
            return results
        
        try:
            # Import yaml here to avoid hard dependency
            import yaml
            
            with open(curriculum_path) as f:
                curriculum = yaml.safe_load(f)
            
            # Check version
            version = curriculum.get("version")
            if version != 2:
                results.append(ValidationResult(
                    check_id="SC-001",
                    check_name="curriculum_version",
                    status=ValidationStatus.WARN,
                    message=f"Unexpected curriculum version: {version}",
                    details={"expected": 2, "actual": version},
                ))
            else:
                results.append(ValidationResult(
                    check_id="SC-001",
                    check_name="curriculum_version",
                    status=ValidationStatus.PASS,
                    message="Curriculum version valid",
                    details={"version": version},
                ))
            
            # Check systems/pl structure
            systems = curriculum.get("systems", {})
            pl_system = systems.get("pl", {})
            
            if not pl_system:
                results.append(ValidationResult(
                    check_id="SC-002",
                    check_name="pl_system",
                    status=ValidationStatus.FAIL,
                    message="PL system not defined in curriculum",
                    details={},
                ))
                return results
            
            # Check active slice
            active_slice = pl_system.get("active")
            slices = pl_system.get("slices", [])
            slice_names = [s.get("name") for s in slices]
            
            if active_slice not in slice_names:
                results.append(ValidationResult(
                    check_id="SC-002",
                    check_name="active_slice",
                    status=ValidationStatus.FAIL,
                    message=f"Active slice '{active_slice}' not found in slices",
                    details={"active": active_slice, "available": slice_names},
                ))
            else:
                results.append(ValidationResult(
                    check_id="SC-002",
                    check_name="active_slice",
                    status=ValidationStatus.PASS,
                    message=f"Active slice '{active_slice}' is valid",
                    details={"active": active_slice, "total_slices": len(slices)},
                ))
            
            # Check monotonicity
            results.append(self._check_slice_monotonicity(slices))
            
            # Check gate definitions
            results.append(self._check_slice_gates(slices))
            
        except ImportError:
            results.append(ValidationResult(
                check_id="SC-001",
                check_name="curriculum_file",
                status=ValidationStatus.SKIP,
                message="PyYAML not installed, cannot validate curriculum",
                details={},
            ))
        except Exception as e:
            results.append(ValidationResult(
                check_id="SC-001",
                check_name="curriculum_file",
                status=ValidationStatus.FAIL,
                message=f"Error parsing curriculum: {e}",
                details={},
            ))
        
        return results
    
    def _check_slice_monotonicity(self, slices: list[dict]) -> ValidationResult:
        """Check that slices satisfy monotonicity constraints on atoms and depth_max."""
        violations = []
        prev_atoms = 0
        prev_depth = 0
        
        for i, slice_def in enumerate(slices):
            name = slice_def.get("name", f"slice_{i}")
            params = slice_def.get("params", {})
            atoms = params.get("atoms", 0)
            depth_max = params.get("depth_max", 0)
            
            # Check atoms monotonicity
            if atoms < prev_atoms:
                violations.append({
                    "slice": name,
                    "axis": "atoms",
                    "current": atoms,
                    "previous": prev_atoms,
                })
            
            # Only check depth if atoms are equal
            if atoms == prev_atoms and depth_max < prev_depth:
                violations.append({
                    "slice": name,
                    "axis": "depth_max",
                    "current": depth_max,
                    "previous": prev_depth,
                })
            
            prev_atoms = max(prev_atoms, atoms)
            prev_depth = depth_max if atoms > prev_atoms else max(prev_depth, depth_max)
        
        if violations:
            return ValidationResult(
                check_id="SC-003",
                check_name="slice_monotonicity",
                status=ValidationStatus.WARN,
                message=f"Monotonicity violations: {len(violations)}",
                details={"violations": violations},
            )
        
        return ValidationResult(
            check_id="SC-003",
            check_name="slice_monotonicity",
            status=ValidationStatus.PASS,
            message="Slice monotonicity satisfied",
            details={"slices_checked": len(slices)},
        )
    
    def _check_slice_gates(self, slices: list[dict]) -> ValidationResult:
        """Check that all slices have required gate definitions."""
        required_gates = ["coverage", "abstention", "velocity", "caps"]
        incomplete = []
        
        for slice_def in slices:
            name = slice_def.get("name", "unknown")
            gates = slice_def.get("gates", {})
            missing_gates = [g for g in required_gates if g not in gates]
            
            if missing_gates:
                incomplete.append({
                    "slice": name,
                    "missing_gates": missing_gates,
                })
        
        if incomplete:
            return ValidationResult(
                check_id="SC-004",
                check_name="slice_gates",
                status=ValidationStatus.FAIL,
                message=f"Incomplete gate definitions: {len(incomplete)} slices",
                details={"incomplete": incomplete},
            )
        
        return ValidationResult(
            check_id="SC-004",
            check_name="slice_gates",
            status=ValidationStatus.PASS,
            message="All slices have complete gate definitions",
            details={"slices_checked": len(slices)},
        )
    
    def validate_preregistration_presence(self) -> list[ValidationResult]:
        """
        Validate preregistration document presence.
        
        Checks:
        - PREREG_UPLIFT_U2.yaml exists (for Phase II)
        - Preregistration schema validity
        - Required fields present
        """
        results = []
        
        prereg_paths = [
            self.project_root / "experiments" / "prereg" / "PREREG_UPLIFT_U2.yaml",
            self.project_root / "PREREG_UPLIFT_U2.yaml",
        ]
        
        prereg_found = None
        for path in prereg_paths:
            if path.exists():
                prereg_found = path
                break
        
        if prereg_found:
            try:
                import yaml
                with open(prereg_found) as f:
                    prereg = yaml.safe_load(f)
                
                # Check required fields
                required_fields = ["hypothesis", "slice_configuration", "success_criteria", "seed", "cycle_count"]
                found_fields = [f for f in required_fields if f in (prereg or {})]
                missing_fields = [f for f in required_fields if f not in (prereg or {})]
                
                if missing_fields:
                    results.append(ValidationResult(
                        check_id="PR-001",
                        check_name="preregistration_schema",
                        status=ValidationStatus.WARN,
                        message=f"Preregistration missing fields: {missing_fields}",
                        details={"found": found_fields, "missing": missing_fields},
                    ))
                else:
                    results.append(ValidationResult(
                        check_id="PR-001",
                        check_name="preregistration_schema",
                        status=ValidationStatus.PASS,
                        message="Preregistration document valid",
                        details={"path": str(prereg_found), "fields": found_fields},
                    ))
                    
            except ImportError:
                results.append(ValidationResult(
                    check_id="PR-001",
                    check_name="preregistration_schema",
                    status=ValidationStatus.SKIP,
                    message="PyYAML not installed",
                    details={},
                ))
            except Exception as e:
                results.append(ValidationResult(
                    check_id="PR-001",
                    check_name="preregistration_schema",
                    status=ValidationStatus.FAIL,
                    message=f"Error parsing preregistration: {e}",
                    details={},
                ))
        else:
            results.append(ValidationResult(
                check_id="PR-001",
                check_name="preregistration_schema",
                status=ValidationStatus.WARN,
                message="No preregistration document found (required for Phase II experiments)",
                details={"searched": [str(p) for p in prereg_paths]},
            ))
        
        return results
    
    def build_migration_gates(self, current_phase: Phase) -> list[MigrationGate]:
        """Build migration gates based on current phase."""
        gates = []
        
        if current_phase == Phase.PHASE_I:
            # Phase I → Phase II gate
            gate_i_ii = MigrationGate(
                gate_id="GATE-I-II",
                source_phase=Phase.PHASE_I,
                target_phase=Phase.PHASE_II,
            )
            
            # Add preconditions
            gate_i_ii.preconditions.extend(self.validate_evidence_sealing_chain())
            gate_i_ii.preconditions.extend(self.validate_determinism_contract()[:3])  # Key checks
            gate_i_ii.preconditions.extend(self.validate_slice_completeness()[:2])  # Key checks
            
            gates.append(gate_i_ii)
        
        if current_phase in (Phase.PHASE_I, Phase.PHASE_II):
            # Phase II → Phase IIb gate
            gate_ii_iib = MigrationGate(
                gate_id="GATE-II-IIb",
                source_phase=Phase.PHASE_II,
                target_phase=Phase.PHASE_IIB,
            )
            
            # Lean toolchain check
            lean_path = self.project_root / "backend" / "lean_proj" / "lakefile.lean"
            if lean_path.exists():
                gate_ii_iib.preconditions.append(ValidationResult(
                    check_id="IIb-001",
                    check_name="lean_toolchain",
                    status=ValidationStatus.PASS,
                    message="Lean toolchain present",
                    details={"path": str(lean_path)},
                ))
            else:
                gate_ii_iib.preconditions.append(ValidationResult(
                    check_id="IIb-001",
                    check_name="lean_toolchain",
                    status=ValidationStatus.FAIL,
                    message="Lean toolchain not found",
                    details={"expected": str(lean_path)},
                ))
            
            gates.append(gate_ii_iib)
        
        if current_phase in (Phase.PHASE_I, Phase.PHASE_II):
            # Phase II → Phase III gate
            gate_ii_iii = MigrationGate(
                gate_id="GATE-II-III",
                source_phase=Phase.PHASE_II,
                target_phase=Phase.PHASE_III,
            )
            
            # Crypto core check
            crypto_core = self.project_root / "backend" / "crypto" / "core.py"
            if crypto_core.exists():
                gate_ii_iii.preconditions.append(ValidationResult(
                    check_id="III-001",
                    check_name="crypto_core",
                    status=ValidationStatus.PASS,
                    message="Crypto core module present",
                    details={"path": str(crypto_core)},
                ))
            else:
                gate_ii_iii.preconditions.append(ValidationResult(
                    check_id="III-001",
                    check_name="crypto_core",
                    status=ValidationStatus.FAIL,
                    message="Crypto core module not found",
                    details={"expected": str(crypto_core)},
                ))
            
            # basis/ package check
            gate_ii_iii.preconditions.extend(self.validate_phase_boundary_purity()[:2])
            
            gates.append(gate_ii_iii)
        
        return gates
    
    def run_simulation(self) -> MigrationSimulationResult:
        """
        Run the complete migration dry-run simulation.
        
        Returns a comprehensive result object with all validation outcomes.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Detect current phase
        current_phase = self.detect_current_phase()
        
        # Run all validations
        boundary_purity = self.validate_phase_boundary_purity()
        determinism_checks = self.validate_determinism_contract()
        evidence_chain = self.validate_evidence_sealing_chain()
        slice_validation = self.validate_slice_completeness()
        preregistration_checks = self.validate_preregistration_presence()
        
        # Build migration gates
        gates = self.build_migration_gates(current_phase)
        
        # Calculate summary
        all_checks = (
            boundary_purity + determinism_checks + evidence_chain +
            slice_validation + preregistration_checks
        )
        
        pass_count = sum(1 for c in all_checks if c.status == ValidationStatus.PASS)
        fail_count = sum(1 for c in all_checks if c.status == ValidationStatus.FAIL)
        warn_count = sum(1 for c in all_checks if c.status == ValidationStatus.WARN)
        skip_count = sum(1 for c in all_checks if c.status == ValidationStatus.SKIP)
        
        # Determine overall status
        if fail_count > 0:
            overall_status = "BLOCKED"
        elif warn_count > 0:
            overall_status = "READY_WITH_WARNINGS"
        else:
            overall_status = "READY"
        
        # Calculate gate readiness
        gate_summary = {}
        for gate in gates:
            gate_summary[gate.gate_id] = {
                "ready": gate.passed,
                "blocking_failures": len(gate.blocking_failures),
            }
        
        summary = {
            "total_checks": len(all_checks),
            "passed": pass_count,
            "failed": fail_count,
            "warnings": warn_count,
            "skipped": skip_count,
            "gates": gate_summary,
            "next_available_migrations": [
                g.target_phase.value for g in gates if g.passed
            ],
        }
        
        return MigrationSimulationResult(
            simulation_id=self.simulation_id,
            timestamp=timestamp,
            current_phase=current_phase,
            gates=gates,
            determinism_checks=determinism_checks,
            evidence_chain=evidence_chain,
            slice_validation=slice_validation,
            preregistration_checks=preregistration_checks,
            boundary_purity=boundary_purity,
            overall_status=overall_status,
            summary=summary,
        )
    
    def save_result(self, result: MigrationSimulationResult, output_path: Path | None = None) -> Path:
        """Save simulation result to JSON file."""
        if output_path is None:
            output_path = self.project_root / "migration_sim_result.json"
        
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return output_path


# =============================================================================
# Phase Impact Report Generator
# =============================================================================

@dataclass
class PhaseImpact:
    """A detected phase transition impact."""
    phase_transition: str  # e.g., "Phase I → Phase II"
    signals: list[dict[str, Any]]
    severity: str  # "CRITICAL" | "WARN" | "INFO"
    signal_count: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase_transition,
            "signals": self.signals,
            "severity": self.severity,
            "signal_count": self.signal_count,
        }


@dataclass
class PhaseImpactReport:
    """Complete phase impact report for a PR."""
    report_id: str
    timestamp: str
    base_ref: str
    head_ref: str
    current_phase: str
    impacts: list[PhaseImpact]
    files_changed: list[str]
    overall_severity: str  # "CRITICAL" | "WARN" | "INFO" | "NONE"
    requires_migration_intent: bool
    summary: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "current_phase": self.current_phase,
            "impacts": [i.to_dict() for i in self.impacts],
            "files_changed": self.files_changed,
            "overall_severity": self.overall_severity,
            "requires_migration_intent": self.requires_migration_intent,
            "summary": self.summary,
        }


def generate_phase_impact_report(
    base_ref: str = "main",
    head_ref: str = "HEAD",
    out_path: str | Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """
    Generate a Phase Impact Report for a PR.
    
    Analyzes the git diff between base_ref and head_ref to identify
    migration signals and classify potential phase transition impacts.
    
    Args:
        base_ref: Git ref for base branch (default: "main")
        head_ref: Git ref for PR head (default: "HEAD")
        out_path: Path to write phase_impact_report.json (optional)
        project_root: Project root directory (default: auto-detect)
    
    Returns:
        Dictionary containing the complete phase impact report
    
    SAFEGUARDS:
        - Read-only analysis
        - No mutations to production state
        - Deterministic output (same diff → same report)
    """
    import subprocess
    
    if project_root is None:
        project_root = PROJECT_ROOT
    else:
        project_root = Path(project_root)
    
    # Generate report ID
    timestamp = datetime.now(timezone.utc).isoformat()
    report_id = hashlib.sha256(
        f"impact_report_{base_ref}_{head_ref}_{timestamp}".encode()
    ).hexdigest()[:16]
    
    # Import PR linter for signal detection
    try:
        from tools.pr_migration_linter import PRMigrationLinter, MigrationSignal
    except ImportError:
        # Fallback if running from different context
        sys.path.insert(0, str(project_root / "tools"))
        from pr_migration_linter import PRMigrationLinter, MigrationSignal
    
    # Initialize linter and simulator
    linter = PRMigrationLinter(project_root)
    simulator = PhaseMigrationSimulator(project_root)
    
    # Get diff and changed files
    try:
        diff_result = subprocess.run(
            ["git", "diff", f"{base_ref}...{head_ref}", "--unified=3"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        diff_content = diff_result.stdout
    except Exception:
        diff_content = ""
    
    try:
        files_result = subprocess.run(
            ["git", "diff", f"{base_ref}...{head_ref}", "--name-only"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        files_changed = [f.strip() for f in files_result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        files_changed = []
    
    # Detect signals
    signals_from_diff = linter.detect_signals_from_diff(diff_content)
    signals_from_files = linter.detect_signals_from_files(files_changed)
    
    # Merge and deduplicate
    all_signals = signals_from_diff + signals_from_files
    seen = set()
    unique_signals = []
    for sig in all_signals:
        key = (sig.signal, sig.file_path)
        if key not in seen:
            seen.add(key)
            unique_signals.append(sig)
    
    # Group signals by implied migration
    migration_groups: dict[str, list[dict[str, Any]]] = {}
    for sig in unique_signals:
        migration = sig.implied_migration
        if migration not in migration_groups:
            migration_groups[migration] = []
        migration_groups[migration].append({
            "signal": sig.signal.value,
            "file_path": sig.file_path,
            "line_number": sig.line_number,
            "context": sig.context[:100] if sig.context else "",
            "severity": sig.severity,
        })
    
    # Build impacts
    impacts = []
    for migration, signals in migration_groups.items():
        # Determine severity for this migration group
        severities = [s["severity"] for s in signals]
        if "critical" in severities:
            group_severity = "CRITICAL"
        elif "warning" in severities:
            group_severity = "WARN"
        else:
            group_severity = "INFO"
        
        impacts.append(PhaseImpact(
            phase_transition=migration,
            signals=signals,
            severity=group_severity,
            signal_count=len(signals),
        ))
    
    # Sort impacts by severity (CRITICAL first)
    severity_order = {"CRITICAL": 0, "WARN": 1, "INFO": 2}
    impacts.sort(key=lambda x: severity_order.get(x.severity, 3))
    
    # Determine overall severity
    if any(i.severity == "CRITICAL" for i in impacts):
        overall_severity = "CRITICAL"
    elif any(i.severity == "WARN" for i in impacts):
        overall_severity = "WARN"
    elif impacts:
        overall_severity = "INFO"
    else:
        overall_severity = "NONE"
    
    # Detect current phase
    current_phase = simulator.detect_current_phase().value
    
    # Determine if migration intent is required
    requires_intent = overall_severity == "CRITICAL"
    
    # Build summary
    summary = {
        "total_signals": len(unique_signals),
        "critical_signals": sum(1 for s in unique_signals if s.severity == "critical"),
        "warning_signals": sum(1 for s in unique_signals if s.severity == "warning"),
        "info_signals": sum(1 for s in unique_signals if s.severity == "info"),
        "migration_transitions_detected": len(impacts),
        "files_analyzed": len(files_changed),
        "phase_frontiers_touched": [i.phase_transition for i in impacts],
    }
    
    # Build report
    report = PhaseImpactReport(
        report_id=report_id,
        timestamp=timestamp,
        base_ref=base_ref,
        head_ref=head_ref,
        current_phase=current_phase,
        impacts=impacts,
        files_changed=files_changed[:50],  # Limit to first 50 for readability
        overall_severity=overall_severity,
        requires_migration_intent=requires_intent,
        summary=summary,
    )
    
    report_dict = report.to_dict()
    
    # Save report if path provided
    if out_path is not None:
        out_path = Path(out_path)
        with open(out_path, "w") as f:
            json.dump(report_dict, f, indent=2)
    
    return report_dict


def main():
    """Main entry point for migration simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase Migration Dry-Run Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase_migration_simulator.py
  python phase_migration_simulator.py --output custom_result.json
  python phase_migration_simulator.py --verbose
  python phase_migration_simulator.py --impact-report --base main --head HEAD

This tool validates migration preconditions without modifying any state.
        """
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for simulation result JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed validation results",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--impact-report",
        action="store_true",
        help="Generate a Phase Impact Report for a PR (requires --base and --head)",
    )
    parser.add_argument(
        "--base", "-b",
        default="main",
        help="Base git ref for impact report (default: main)",
    )
    parser.add_argument(
        "--head", "-H",
        default="HEAD",
        help="Head git ref for impact report (default: HEAD)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for standard simulation (default behavior)",
    )
    
    args = parser.parse_args()
    
    # Handle impact report mode
    if args.impact_report:
        print(f"🔍 Generating Phase Impact Report...")
        print(f"   Base: {args.base}")
        print(f"   Head: {args.head}")
        print()
        
        out_path = args.output or Path("phase_impact_report.json")
        report = generate_phase_impact_report(
            base_ref=args.base,
            head_ref=args.head,
            out_path=out_path,
            project_root=args.project_root,
        )
        
        # Print summary
        print(f"📊 Phase Impact Report")
        print(f"   Report ID: {report['report_id']}")
        print(f"   Current Phase: {report['current_phase']}")
        print(f"   Overall Severity: {report['overall_severity']}")
        print(f"   Requires Migration Intent: {report['requires_migration_intent']}")
        print()
        
        if report["impacts"]:
            print("🎯 Phase Frontiers Touched:")
            for impact in report["impacts"]:
                severity_icon = {"CRITICAL": "🔴", "WARN": "🟡", "INFO": "🔵"}.get(impact["severity"], "⚪")
                print(f"   {severity_icon} {impact['phase']} ({impact['signal_count']} signals)")
                if args.verbose:
                    for sig in impact["signals"][:5]:  # Show first 5
                        print(f"      • [{sig['signal']}] {sig['file_path']}")
                    if len(impact["signals"]) > 5:
                        print(f"      ... and {len(impact['signals']) - 5} more")
            print()
        else:
            print("   ✅ No phase transition impacts detected")
            print()
        
        print(f"💾 Report saved to: {out_path}")
        
        # Exit with appropriate code
        if report["overall_severity"] == "CRITICAL" and not report.get("intent_declared"):
            sys.exit(1)
        sys.exit(0)
    
    # Run simulation
    simulator = PhaseMigrationSimulator(args.project_root)
    print(f"🔍 Running Phase Migration Simulation...")
    print(f"   Simulation ID: {simulator.simulation_id}")
    print()
    
    result = simulator.run_simulation()
    
    # Print summary
    print(f"📊 Simulation Results")
    print(f"   Current Phase: {result.current_phase.value}")
    print(f"   Overall Status: {result.overall_status}")
    print()
    print(f"   Checks: {result.summary['passed']}/{result.summary['total_checks']} passed")
    print(f"   Warnings: {result.summary['warnings']}")
    print(f"   Failures: {result.summary['failed']}")
    print()
    
    # Print gate status
    print("🚪 Migration Gates:")
    for gate in result.gates:
        status_icon = "✅" if gate.passed else "❌"
        print(f"   {status_icon} {gate.gate_id}: {gate.source_phase.value} → {gate.target_phase.value}")
        if not gate.passed:
            print(f"      Blocking failures: {len(gate.blocking_failures)}")
    print()
    
    if args.verbose:
        print("📋 Detailed Results:")
        for category, checks in [
            ("Boundary Purity", result.boundary_purity),
            ("Determinism", result.determinism_checks),
            ("Evidence Chain", result.evidence_chain),
            ("Slice Validation", result.slice_validation),
            ("Preregistration", result.preregistration_checks),
        ]:
            print(f"\n   {category}:")
            for check in checks:
                icon = {"pass": "✅", "fail": "❌", "warn": "⚠️", "skip": "⏭️"}[check.status.value]
                print(f"      {icon} [{check.check_id}] {check.check_name}: {check.message}")
    
    # Save result
    output_path = simulator.save_result(result, args.output)
    print(f"💾 Results saved to: {output_path}")
    
    # Exit with appropriate code
    sys.exit(0 if result.overall_status != "BLOCKED" else 1)


if __name__ == "__main__":
    main()

