import os
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path

# Configuration
ROOT_DIR = Path(".")
OUTPUT_FILE = Path("ops/spanning_set_manifest.json")

IGNORE_DIRS = {
    ".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", 
    ".quarantine", ".gemini", "tmp", ".grok", ".github", "bootstrap_output"
}

IGNORE_FILES = {
    ".DS_Store", "Thumbs.db"
}

# Classification Rules
def classify_path(path):
    parts = path.parts
    name = path.name
    
    # Root level files
    if len(parts) == 1:
        if name.endswith((".diff", ".patch")):
            return "experimental", "Patch file - likely residue"
        if name.startswith("_fail_"):
            return "obsolete", "Failure log"
        if name in ["pyproject.toml", "uv.lock", "Makefile", "docker-compose.yml", "pytest.ini", ".editorconfig", ".gitattributes", ".gitignore", ".pre-commit-config.yaml", ".coveragerc", ".coverage"]:
            return "supporting", "Project configuration"
        if name in ["ledgerctl.py", "run_tests.py", "run_migration.py", "run_migration_simple.py", "run_all_migrations.py", "sanity.ps1", "run-nightly.ps1", "rfl_gate.py", "monitor.py", "fix_test.py", "get_schema.py", "start_api_server.py", "verify_dual_root.py", "verify_local_schema.py", "phase_ix_attestation.py"]:
            return "supporting", "Operational script"
        if name.startswith("test_"):
            return "experimental", "Root level test script"
        if name.endswith((".md", ".txt", ".json")) and (name.isupper() or "README" in name or "LICENSE" in name or "report" in name.lower()):
             return "supporting", "Documentation/Meta"
            
    # Directories
    top_dir = parts[0]
    
    if top_dir == "basis":
        return "core", "Cursor O's basis - Primary Nucleus"
    
    if top_dir in ["ledger", "derivation", "rfl", "substrate", "attestation", "normalization", "metrics"]:
        return "core", "Legacy Core Component"
        
    if top_dir == "mathledger_basis_repo":
        return "core", "Basis Repository Mirror"

    if top_dir in ["infra", "ops", "config", "tools", "scripts", "migrations", "ci_verification", ".grok", ".github"]:
        return "supporting", "Infrastructure and Operations"
        
    if top_dir in ["backend", "services", "interface", "api", "cli"]: 
        return "core", "Application Layer"
        
    if top_dir in ["tests", "testing", ".quarantine"]:
        return "supporting", "Verification"
        
    if top_dir in ["archive", "allblue_archive"]:
        return "archive-candidate", "Explicit Archive"
        
    if top_dir in ["artifacts", "reports", "logs", "bootstrap_output"]:
        return "supporting", "Artifacts and Logs"
        
    if top_dir == "ui" or top_dir == "apps":
        return "core", "Frontend/UI"

    return "experimental", "Unclassified / Experimental"

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        return f"ERROR: {str(e)}"

def count_loc(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def generate_manifest():
    manifest = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "generated_by": "Spanning Set Cartographer",
            "root_dir": str(ROOT_DIR.absolute())
        },
        "files": []
    }
    
    print(f"Starting census of {ROOT_DIR.absolute()}...")
    
    for root, dirs, files in os.walk(ROOT_DIR):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file in IGNORE_FILES:
                continue
                
            file_path = Path(root) / file
            rel_path = file_path.relative_to(ROOT_DIR)
            
            # Skip hidden files at root if not explicitly handled? No, keep them for now.
            
            try:
                stats = file_path.stat()
                role, justification = classify_path(rel_path)
                file_hash = calculate_sha256(file_path)
                loc = count_loc(file_path)
                
                entry = {
                    "path": str(rel_path).replace("\\", "/"),
                    "type": "file",
                    "size": stats.st_size,
                    "loc": loc,
                    "last_modified": datetime.utcfromtimestamp(stats.st_mtime).isoformat() + "Z",
                    "hash": file_hash,
                    "role": role,
                    "justification": justification
                }
                manifest["files"].append(entry)
                
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Census complete. Manifest written to {OUTPUT_FILE}")
    
    # Generate summary statistics
    summary = {}
    loc_summary = {}
    for entry in manifest["files"]:
        role = entry["role"]
        summary[role] = summary.get(role, 0) + 1
        loc_summary[role] = loc_summary.get(role, 0) + entry["loc"]
        
    print("Summary:")
    for role, count in summary.items():
        print(f"  {role}: {count} files, {loc_summary[role]} LOC")

if __name__ == "__main__":
    generate_manifest()
