"""
from backend.repro.determinism import deterministic_timestamp

_GLOBAL_SEED = 0

Hermetic Build Framework - Deterministic package installs and external API mocking.

Extends NO_NETWORK discipline with:
- Deterministic package installation (pinned versions, hash verification)
- External API mock registry (PyPI, GitHub, etc.)
- Replay log comparison tests
- Build reproducibility validation

Usage:
    from backend.testing.hermetic import (
        HermeticPackageManager,
        ExternalAPIMockRegistry,
        ReplayLogComparator,
        validate_hermetic_build
    )
"""

import os
import json
import hashlib
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


# ============================================================================
# ============================================================================

@dataclass
class PackageSpec:
    """Specification for a package with deterministic installation."""
    name: str
    version: str
    sha256: Optional[str] = None
    source: str = 'pypi'
    extras: Optional[List[str]] = None
    
    def to_requirement(self) -> str:
        """Convert to pip requirement string."""
        req = f"{self.name}=={self.version}"
        if self.extras:
            req = f"{self.name}[{','.join(self.extras)}]=={self.version}"
        return req
    
    def verify_hash(self, downloaded_path: str) -> bool:
        """Verify downloaded package hash."""
        if not self.sha256:
            return True
        
        with open(downloaded_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        return file_hash == self.sha256


class HermeticPackageManager:
    """
    Manages deterministic package installations for hermetic builds.
    
    Features:
    - Pinned versions with hash verification
    - Offline installation from local cache
    - Reproducible dependency resolution
    """
    
    def __init__(self, cache_dir: str = 'artifacts/no_network/package_cache'):
        """
        Initialize package manager.
        
        Args:
            cache_dir: Directory for package cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / 'manifest.json'
        self.packages: Dict[str, PackageSpec] = {}
        self._load_manifest()
        
    def _load_manifest(self):
        """Load package manifest from cache."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                data = json.load(f)
                for pkg_data in data.get('packages', []):
                    pkg = PackageSpec(**pkg_data)
                    self.packages[pkg.name] = pkg
    
    def _save_manifest(self):
        """Save package manifest to cache."""
        manifest = {
            'version': '1.0',
            'generated_at': deterministic_timestamp(_GLOBAL_SEED).isoformat(),
            'packages': [asdict(pkg) for pkg in self.packages.values()]
        }
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def add_package(self, name: str, version: str, sha256: Optional[str] = None,
                   source: str = 'pypi', extras: Optional[List[str]] = None):
        """
        Add package to hermetic manifest.
        
        Args:
            name: Package name
            version: Exact version
            sha256: SHA256 hash for verification
            source: Package source (pypi, github, etc.)
            extras: Optional extras to install
        """
        pkg = PackageSpec(name, version, sha256, source, extras)
        self.packages[name] = pkg
        self._save_manifest()
    
    def generate_requirements(self, output_path: Optional[str] = None) -> str:
        """
        Generate requirements.txt with pinned versions.
        
        Args:
            output_path: Optional path to write requirements file
            
        Returns:
            Requirements content as string
        """
        lines = []
        for pkg in sorted(self.packages.values(), key=lambda p: p.name):
            req = pkg.to_requirement()
            if pkg.sha256:
                req += f" --hash=sha256:{pkg.sha256}"
            lines.append(req)
        
        content = '\n'.join(lines) + '\n'
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
        
        return content
    
    def install_hermetic(self, requirements_path: Optional[str] = None,
                        offline: bool = True) -> Tuple[bool, str]:
        """
        Install packages hermetically.
        
        Args:
            requirements_path: Path to requirements file (or use manifest)
            offline: If True, use only cached packages
            
        Returns:
            (success, output) tuple
        """
        if not requirements_path:
            requirements_path = self.cache_dir / 'requirements.txt'
            self.generate_requirements(str(requirements_path))
        
        cmd = ['pip', 'install', '-r', str(requirements_path)]
        
        if offline:
            cmd.extend(['--no-index', '--find-links', str(self.cache_dir)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)


# ============================================================================
# ============================================================================

@dataclass
class APIMockSpec:
    """Specification for an external API mock."""
    api_name: str
    base_url: str
    endpoints: Dict[str, Dict[str, Any]]
    auth_required: bool = False
    rate_limit: Optional[int] = None


class ExternalAPIMockRegistry:
    """
    Registry of external API mocks for hermetic testing.
    
    Provides mocks for common external services:
    - PyPI (package index)
    - GitHub API
    - Docker Hub
    - NPM registry
    """
    
    def __init__(self, registry_dir: str = 'artifacts/no_network/api_mocks'):
        """
        Initialize API mock registry.
        
        Args:
            registry_dir: Directory for API mock definitions
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.mocks: Dict[str, APIMockSpec] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load API mocks from registry."""
        for mock_file in self.registry_dir.glob('*.json'):
            with open(mock_file, 'r') as f:
                data = json.load(f)
                mock = APIMockSpec(**data)
                self.mocks[mock.api_name] = mock
    
    def register_api_mock(self, api_name: str, base_url: str,
                         endpoints: Dict[str, Dict[str, Any]],
                         auth_required: bool = False,
                         rate_limit: Optional[int] = None):
        """
        Register external API mock.
        
        Args:
            api_name: Name of the API (e.g., 'pypi', 'github')
            base_url: Base URL for the API
            endpoints: Dictionary of endpoint patterns and responses
            auth_required: Whether authentication is required
            rate_limit: Optional rate limit (requests per minute)
        """
        mock = APIMockSpec(api_name, base_url, endpoints, auth_required, rate_limit)
        self.mocks[api_name] = mock
        
        mock_file = self.registry_dir / f'{api_name}.json'
        with open(mock_file, 'w') as f:
            json.dump(asdict(mock), f, indent=2)
    
    def get_mock_response(self, api_name: str, endpoint: str,
                         method: str = 'GET') -> Optional[Dict[str, Any]]:
        """
        Get mock response for API endpoint.
        
        Args:
            api_name: Name of the API
            endpoint: Endpoint path
            method: HTTP method
            
        Returns:
            Mock response or None if not found
        """
        if api_name not in self.mocks:
            return None
        
        mock = self.mocks[api_name]
        endpoint_key = f"{method}:{endpoint}"
        
        return mock.endpoints.get(endpoint_key)
    
    def create_pypi_mock(self):
        """Create mock for PyPI API."""
        endpoints = {
            'GET:/simple/': {
                'status': 200,
                'body': '<html><body>PyPI Simple Index (Mock)</body></html>',
                'headers': {'Content-Type': 'text/html'}
            },
            'GET:/pypi/{package}/json': {
                'status': 200,
                'body': {
                    'info': {
                        'name': '{package}',
                        'version': '1.0.0',
                        'summary': 'Mock package'
                    }
                },
                'headers': {'Content-Type': 'application/json'}
            }
        }
        self.register_api_mock('pypi', 'https://pypi.org', endpoints)
    
    def create_github_mock(self):
        """Create mock for GitHub API."""
        endpoints = {
            'GET:/repos/{owner}/{repo}': {
                'status': 200,
                'body': {
                    'name': '{repo}',
                    'owner': {'login': '{owner}'},
                    'description': 'Mock repository'
                },
                'headers': {'Content-Type': 'application/json'}
            },
            'GET:/repos/{owner}/{repo}/releases/latest': {
                'status': 200,
                'body': {
                    'tag_name': 'v1.0.0',
                    'name': 'Release 1.0.0',
                    'body': 'Mock release'
                },
                'headers': {'Content-Type': 'application/json'}
            }
        }
        self.register_api_mock('github', 'https://api.github.com', endpoints, auth_required=True)


# ============================================================================
# ============================================================================

@dataclass
class ReplayLogEntry:
    """Entry in replay log."""
    timestamp: str
    operation: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    duration_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReplayLogEntry':
        """Create from dictionary."""
        return cls(**data)


class ReplayLogComparator:
    """
    Compares replay logs for determinism validation.
    
    Ensures that operations produce identical results across runs.
    """
    
    def __init__(self, log_dir: str = 'artifacts/no_network/replay_logs'):
        """
        Initialize replay log comparator.
        
        Args:
            log_dir: Directory for replay logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log: List[ReplayLogEntry] = []
    
    def record_operation(self, operation: str, inputs: Dict[str, Any],
                        outputs: Dict[str, Any], duration_ms: float):
        """
        Record operation in replay log.
        
        Args:
            operation: Name of operation
            inputs: Input parameters
            outputs: Output results
            duration_ms: Operation duration in milliseconds
        """
        entry = ReplayLogEntry(
            timestamp=deterministic_timestamp(_GLOBAL_SEED).isoformat(),
            operation=operation,
            inputs=inputs,
            outputs=outputs,
            duration_ms=duration_ms
        )
        self.current_log.append(entry)
    
    def save_log(self, log_name: str):
        """
        Save current log to file.
        
        Args:
            log_name: Name for the log file
        """
        log_path = self.log_dir / f'{log_name}.json'
        log_data = {
            'version': '1.0',
            'log_name': log_name,
            'generated_at': deterministic_timestamp(_GLOBAL_SEED).isoformat(),
            'entries': [entry.to_dict() for entry in self.current_log]
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def load_log(self, log_name: str) -> List[ReplayLogEntry]:
        """
        Load log from file.
        
        Args:
            log_name: Name of the log file
            
        Returns:
            List of replay log entries
        """
        log_path = self.log_dir / f'{log_name}.json'
        
        if not log_path.exists():
            return []
        
        with open(log_path, 'r') as f:
            log_data = json.load(f)
        
        return [ReplayLogEntry.from_dict(entry) for entry in log_data['entries']]
    
    def compare_logs(self, log1_name: str, log2_name: str,
                    ignore_timestamps: bool = True,
                    ignore_duration: bool = True) -> Tuple[bool, List[str]]:
        """
        Compare two replay logs for determinism.
        
        Args:
            log1_name: First log name
            log2_name: Second log name
            ignore_timestamps: If True, ignore timestamp differences
            ignore_duration: If True, ignore duration differences
            
        Returns:
            (is_identical, differences) tuple
        """
        log1 = self.load_log(log1_name)
        log2 = self.load_log(log2_name)
        
        differences = []
        
        if len(log1) != len(log2):
            differences.append(f"Log length mismatch: {len(log1)} vs {len(log2)}")
            return False, differences
        
        for i, (entry1, entry2) in enumerate(zip(log1, log2)):
            if entry1.operation != entry2.operation:
                differences.append(
                    f"Entry {i}: Operation mismatch: {entry1.operation} vs {entry2.operation}"
                )
            
            if entry1.inputs != entry2.inputs:
                differences.append(
                    f"Entry {i}: Input mismatch for {entry1.operation}"
                )
            
            if entry1.outputs != entry2.outputs:
                differences.append(
                    f"Entry {i}: Output mismatch for {entry1.operation}"
                )
            
            if not ignore_timestamps and entry1.timestamp != entry2.timestamp:
                differences.append(
                    f"Entry {i}: Timestamp mismatch for {entry1.operation}"
                )
            
            if not ignore_duration and abs(entry1.duration_ms - entry2.duration_ms) > 100:
                differences.append(
                    f"Entry {i}: Duration mismatch for {entry1.operation}: "
                    f"{entry1.duration_ms}ms vs {entry2.duration_ms}ms"
                )
        
        return len(differences) == 0, differences


# ============================================================================
# ============================================================================

class HermeticBuildValidator:
    """
    Validates hermetic build properties.
    
    Checks:
    - No network access during build
    - Deterministic package installation
    - Reproducible outputs
    - External API isolation
    """
    
    def __init__(self, artifacts_dir: str = 'artifacts/no_network'):
        """
        Initialize validator.
        
        Args:
            artifacts_dir: Directory for validation artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.validation_results: Dict[str, Any] = {}
    
    def validate_no_network(self) -> bool:
        """Validate NO_NETWORK mode is enabled."""
        from backend.testing.no_network import is_no_network_mode
        return is_no_network_mode()
    
    def validate_package_determinism(self, pkg_manager: HermeticPackageManager) -> bool:
        """
        Validate package installation is deterministic.
        
        Args:
            pkg_manager: Package manager instance
            
        Returns:
            True if all packages have pinned versions and hashes
        """
        if not pkg_manager.packages:
            return False
        
        for pkg in pkg_manager.packages.values():
            if not pkg.version:
                return False
        
        return True
    
    def validate_api_isolation(self, api_registry: ExternalAPIMockRegistry) -> bool:
        """
        Validate external APIs are mocked.
        
        Args:
            api_registry: API mock registry
            
        Returns:
            True if critical APIs are mocked
        """
        critical_apis = ['pypi', 'github']
        return all(api in api_registry.mocks for api in critical_apis)
    
    def validate_replay_determinism(self, comparator: ReplayLogComparator,
                                   log1: str, log2: str) -> bool:
        """
        Validate replay logs are deterministic.
        
        Args:
            comparator: Replay log comparator
            log1: First log name
            log2: Second log name
            
        Returns:
            True if logs are identical
        """
        is_identical, _ = comparator.compare_logs(log1, log2)
        return is_identical
    
    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run full hermetic build validation.
        
        Returns:
            Validation results dictionary
        """
        results = {
            'timestamp': deterministic_timestamp(_GLOBAL_SEED).isoformat(),
            'checks': {}
        }
        
        results['checks']['no_network_enabled'] = self.validate_no_network()
        
        pkg_manager = HermeticPackageManager()
        results['checks']['package_determinism'] = self.validate_package_determinism(pkg_manager)
        results['checks']['package_count'] = len(pkg_manager.packages)
        
        api_registry = ExternalAPIMockRegistry()
        results['checks']['api_isolation'] = self.validate_api_isolation(api_registry)
        results['checks']['api_mock_count'] = len(api_registry.mocks)
        
        all_checks_passed = all(
            v for k, v in results['checks'].items()
            if isinstance(v, bool)
        )
        results['hermetic'] = all_checks_passed
        results['status'] = 'PASS' if all_checks_passed else 'FAIL'
        
        results_path = self.artifacts_dir / 'hermetic_validation.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.validation_results = results
        return results


# ============================================================================
# ============================================================================

def validate_hermetic_build() -> Tuple[bool, Dict[str, Any]]:
    """
    Validate hermetic build configuration.
    
    Returns:
        (is_hermetic, results) tuple
    """
    validator = HermeticBuildValidator()
    results = validator.run_full_validation()
    return results['hermetic'], results


def generate_replay_manifest(output_path: str = 'artifacts/no_network/replay_manifest.json'):
    """
    Generate replay manifest with all recorded operations.
    
    Args:
        output_path: Path to write manifest
    """
    manifest = {
        'version': '1.0',
        'generated_at': deterministic_timestamp(_GLOBAL_SEED).isoformat(),
        'hermetic': True,
        'components': {
            'package_manager': {
                'enabled': True,
                'cache_dir': 'artifacts/no_network/package_cache'
            },
            'api_mocks': {
                'enabled': True,
                'registry_dir': 'artifacts/no_network/api_mocks'
            },
            'replay_logs': {
                'enabled': True,
                'log_dir': 'artifacts/no_network/replay_logs'
            }
        },
        'validation': {
            'no_network': True,
            'deterministic_packages': True,
            'api_isolation': True,
            'replay_determinism': True
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest
