import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any

class SpanningSetCartographer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.manifest: Dict[str, Any] = {
            "generated_at": time.time(),
            "root": str(self.root_dir),
            "entries": {}
        }
        self.ignore_dirs = {
            ".git", "__pycache__", "node_modules", "venv", ".cursor", ".pytest_cache", "dist", "build", "coverage"
        }
        self.ignore_files = {
            ".DS_Store", "Thumbs.db"
        }

    def calculate_hash(self, file_path: Path) -> str:
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            return f"error: {str(e)}"

    def classify_path(self, relative_path: str) -> Dict[str, str]:
        path_obj = Path(relative_path)
        parts = path_obj.parts
        
        classification = "unknown"
        justification = "Pending classification"

        # Root level files
        if len(parts) == 1:
            if parts[0] in ["pyproject.toml", "uv.lock", "Makefile", "pytest.ini", "docker-compose.yml"]:
                classification = "supporting"
                justification = "Project configuration and build tools"
            elif parts[0].startswith("README"):
                classification = "supporting"
                justification = "Documentation"
            elif parts[0].endswith(".md") or parts[0].endswith(".txt"):
                 # Loose reports are generally archive/experimental unless critical
                 if parts[0] in ["CLAUDE.md", "AGENTS.md"]:
                     classification = "supporting"
                     justification = "Agent context"
                 else:
                    classification = "archive-candidate"
                    justification = "Loose root report/doc"
            elif parts[0].endswith(".py") or parts[0].endswith(".ps1") or parts[0].endswith(".sh"):
                # Root scripts are usually supporting or experimental
                if parts[0] in ["worker.py", "monitor.py"]: # Example
                    classification = "supporting"
                    justification = "Root operational script"
                else:
                    classification = "experimental"
                    justification = "Root script, likely ad-hoc"
            elif parts[0].endswith(".patch") or parts[0].endswith(".diff"):
                 classification = "experimental"
                 justification = "Patch file"
            elif parts[0].startswith("_fail"):
                 classification = "experimental"
                 justification = "Failure log"

        # Directory based classification
        elif parts[0] == "backend":
            if len(parts) > 1:
                sub = parts[1]
                if sub in ["crypto", "ledger", "logic", "rfl", "orchestrator", "axiom_engine", "lean_proj", "consensus", "dag", "fol_eq", "frontier", "generator", "causal", "models"]:
                    classification = "core"
                    justification = f"Backend core module: {sub} (Logic/Ledger/Mechanism)"
                elif sub in ["api", "worker.py", "metrics"]: # API is interface, not core logic usually, but critical supporting
                    classification = "supporting"
                    justification = "Backend interface/operational layer"
                elif sub in ["repro", "testing", "phase_ix", "governance", "integration", "audit"]: # Testing/Validation stuff
                    classification = "supporting" # Or experimental?
                    justification = "Backend validation/testing infrastructure"
                    if sub == "phase_ix": # Specific phase work might be experimental
                         classification = "experimental"
                         justification = "Phase IX specific work"
                elif sub in ["ui"]:
                     classification = "supporting"
                     justification = "Backend served UI assets"
                else:
                     classification = "supporting"
                     justification = f"Backend module: {sub}"
        
        elif parts[0] == "basis":
            classification = "core"
            justification = "Target destination for minimal basis"
        
        elif parts[0] == "substrate":
            classification = "core"
            justification = "Whitepaper: Substrate (Lean/Formal Verification)"
            
        elif parts[0] == "attestation":
            classification = "core"
            justification = "Whitepaper: Dual Attestation Root"
            
        elif parts[0] == "curriculum":
            classification = "core"
            justification = "Whitepaper: Curriculum/Learning Schedule"
            
        elif parts[0] == "derivation":
             classification = "core"
             justification = "Whitepaper: Derivation Engine"
        
        elif parts[0] == "ledger": # Root ledger dir
             classification = "core"
             justification = "Ledger components (if distinct from backend)"

        elif parts[0] == "normalization":
             classification = "core"
             justification = "Logic normalization components"

        elif parts[0] == "rfl": # Root rfl dir
             classification = "core"
             justification = "Reinforced Feedback Loop (Root)"

        elif parts[0] in ["apps", "ui"]:
            classification = "supporting"
            justification = "Frontend/UI"
            
        elif parts[0] in ["infra", "config", "scripts", "tools", "migrations", "cli"]:
            classification = "supporting"
            justification = "Operational Infrastructure"
            
        elif parts[0] == "tests":
            classification = "supporting"
            justification = "Test Suite"
            
        elif parts[0] == "docs":
            classification = "supporting"
            justification = "Project Documentation"
            
        elif parts[0] in ["archive", "allblue_archive"]:
            classification = "archive-candidate"
            justification = "Explicit archive directory"
            
        elif parts[0] in ["tmp", "artifacts", "bootstrap_output", "exports", "logs"]:
             classification = "experimental"
             justification = "Ephemeral or output directory"
        
        elif parts[0] == "metrics": # Root metrics
             classification = "supporting"
             justification = "Metrics output/config"

        # Refine based on extensions or specific patterns
        if path_obj.name.endswith(".bak"):
            classification = "archive-candidate"
            justification = "Backup file"
            
        if classification == "unknown":
            classification = "supporting" # Default to supporting if not clearly core or trash
            justification = "Unclassified, defaulted to supporting"

        return {"class": classification, "justification": justification}

    def map_repo(self):
        for root, dirs, files in os.walk(self.root_dir):
            # Modify dirs in-place to skip ignored
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for name in files:
                if name in self.ignore_files:
                    continue
                    
                full_path = Path(root) / name
                rel_path = full_path.relative_to(self.root_dir)
                
                # Skip .git internal files if they leaked through
                if ".git" in rel_path.parts:
                    continue
                
                stats = full_path.stat()
                file_hash = self.calculate_hash(full_path)
                classification = self.classify_path(str(rel_path))
                
                loc = 0
                try:
                    # Simple LOC count for text-ish files
                    if name.endswith(('.py', '.js', '.ts', '.tsx', '.lean', '.html', '.css', '.md', '.json', '.sql', '.ps1', '.sh', '.yml', '.yaml')):
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            loc = sum(1 for _ in f)
                except:
                    pass

                self.manifest["entries"][str(rel_path)] = {
                    "type": "file",
                    "size": stats.st_size,
                    "lines": loc,
                    "last_modified": stats.st_mtime,
                    "hash": file_hash,
                    "classification": classification["class"],
                    "justification": classification["justification"]
                }
            
            # Record directory entries (optional, but good for structure)
            rel_root = Path(root).relative_to(self.root_dir)
            if str(rel_root) != ".":
                 classification = self.classify_path(str(rel_root))
                 self.manifest["entries"][str(rel_root) + "/"] = {
                    "type": "directory",
                    "classification": classification["class"],
                    "justification": classification["justification"]
                 }

    def save(self, output_path: str):
        with open(output_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

if __name__ == "__main__":
    print("Starting Spanning Set Census...")
    cartographer = SpanningSetCartographer(".")
    cartographer.map_repo()
    output = "ops/spanning_set_manifest.json"
    cartographer.save(output)
    print(f"Census complete. Manifest saved to {output}")

