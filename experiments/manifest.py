import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

class ManifestGenerator:
    """
    RFL uplift detection is not part of manifest validation.
    """
    def __init__(
        self, 
        experiment_id: str, 
        experiment_type: str, 
        description: str,
        config_path: Path, 
        mdap_seed: str,
        is_hermetic: bool = True,
        is_no_network: bool = True
    ):
        self.meta = {
            "version": "1.0.0",
            "type": "mathledger_experiment_manifest"
        }
        self.experiment = {
            "id": experiment_id,
            "type": experiment_type,
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "description": description
        }
        self.config_path = config_path
        self.mdap_seed = mdap_seed
        self.hermetic = is_hermetic
        self.no_network = is_no_network
        
        self.provenance = self._get_provenance()
        self.configuration = self._get_configuration()
        self.artifacts = {"logs": [], "figures": []}

    def _get_git_info(self) -> Dict[str, Any]:
        try:
            commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            
            # Check if dirty
            status_out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
            is_dirty = bool(status_out)
            
            diff_sha256 = None
            if is_dirty:
                # Get diff hash
                diff_bytes = subprocess.check_output(["git", "diff", "HEAD"], stderr=subprocess.DEVNULL)
                diff_sha256 = hashlib.sha256(diff_bytes).hexdigest()

            return {
                "commit_sha": commit_sha,
                "branch": branch,
                "is_dirty": is_dirty,
                "diff_sha256": diff_sha256
            }
        except subprocess.CalledProcessError:
            return {
                "commit_sha": "unknown",
                "branch": "unknown",
                "is_dirty": False,
                "diff_sha256": None
            }
        except FileNotFoundError:
             return {
                "commit_sha": "git_not_found",
                "branch": "unknown",
                "is_dirty": False,
                "diff_sha256": None
            }

    def _get_system_info(self) -> Dict[str, str]:
        import platform
        import socket
        
        env_hash = "unknown"
        try:
            # Use current sys.executable to get pip freeze of the active environment
            env_bytes = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL)
            env_hash = hashlib.sha256(env_bytes).hexdigest()
        except Exception:
            pass

        return {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "env_hash": env_hash
        }

    def _get_provenance(self) -> Dict[str, Any]:
        return {
            "git": self._get_git_info(),
            "system": self._get_system_info()
        }

    def _hash_file(self, path: Path) -> str:
        if not path.exists():
            return "missing"
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_configuration(self) -> Dict[str, Any]:
        return {
            "config_file": {
                "path": str(self.config_path),
                "sha256": self._hash_file(self.config_path)
            },
            "parameters": {}, 
            "determinism": {
                "mdap_seed": self.mdap_seed,
                "sealing_conditions": {
                    "hermetic": self.hermetic,
                    "no_network": self.no_network
                }
            },
            "specialized": {}
        }

    def add_log(self, path: Path, log_type: str):
        rel_path = path
        if path.is_absolute():
            # Try to make relative to cwd if possible for cleaner manifest
            try:
                rel_path = path.relative_to(Path.cwd())
            except ValueError:
                pass
                
        self.artifacts["logs"].append({
            "path": str(rel_path),
            "sha256": self._hash_file(path),
            "type": log_type
        })

    def add_figure(self, path: Path, description: str):
        rel_path = path
        if path.is_absolute():
            try:
                rel_path = path.relative_to(Path.cwd())
            except ValueError:
                pass

        self.artifacts["figures"].append({
            "path": str(rel_path),
            "sha256": self._hash_file(path),
            "description": description
        })
    
    def add_specialized_config(self, key: str, value: str):
        self.configuration["specialized"][key] = value

    def save(self, output_path: Path):
        full_manifest = {
            "meta": self.meta,
            "experiment": self.experiment,
            "provenance": self.provenance,
            "configuration": self.configuration,
            "artifacts": self.artifacts
        }
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(full_manifest, f, indent=2)
        print(f"Manifest saved to {output_path}")
