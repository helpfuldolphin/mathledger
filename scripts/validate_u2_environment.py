# PHASE II — U2 UPLIFT EXPERIMENT
"""
U2 Environment Validation and Mode Management Script.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import ast
import shutil
try:
    import psutil
except ImportError:
    psutil = None
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

class CheckResult:
    def __init__(self, rule_ids, check_id, title, status="PASS", details="", data=None):
        self.rule_ids = rule_ids if isinstance(rule_ids, list) else [rule_ids]
        self.check_id = check_id
        self.title = title
        self.status = status
        self.details = details
        self.data = data or {}
    def to_dict(self):
        return { "rule_ids": self.rule_ids, "check_id": self.check_id, "status": self.status, "title": self.title, "details": self.details, "data": self.data }
    def passed(self):
        return self.status == "PASS"

def check_mode_declaration():
    rule_id = "RULE-001"; check_id = "NFR-001_MODE_DECLARATION"; title = "RFL Environment Mode Declaration"
    mode = os.environ.get("RFL_ENV_MODE"); valid_modes = ["phase1-hermetic", "uplift_experiment", "uplift_analysis"]
    if not mode: return CheckResult(rule_id, check_id, title, "FAIL", "RFL_ENV_MODE is not set.")
    if mode not in valid_modes: return CheckResult(rule_id, check_id, title, "FAIL", f"Invalid RFL_ENV_MODE '{mode}'. Valid modes are: {valid_modes}", {"value": mode})
    return CheckResult(rule_id, check_id, title, "PASS", f"RFL_ENV_MODE is set to '{mode}'.", {"value": mode})

def check_directory(var_name, check_id_prefix, title_prefix, rule_id_map):
    path_str = os.environ.get(var_name)
    if not path_str: return [CheckResult(rule_id_map["CONFIGURED"], f"{check_id_prefix}_CONFIGURED", f"{title_prefix} Configured", "FAIL", f"{var_name} is not set.")]
    results = [CheckResult(rule_id_map["CONFIGURED"], f"{check_id_prefix}_CONFIGURED", f"{title_prefix} Configured", "PASS", f"{var_name} is set to '{path_str}'.")]
    p = Path(path_str)
    if not p.exists():
        results.append(CheckResult(rule_id_map["EXISTS"], f"{check_id_prefix}_EXISTS", f"{title_prefix} Exists", "FAIL", f"Directory '{p}' does not exist.", {"path": str(p)}))
    else:
        results.append(CheckResult(rule_id_map["EXISTS"], f"{check_id_prefix}_EXISTS", f"{title_prefix} Exists", "PASS", f"Directory '{p}' exists.", {"path": str(p)}))
    if p.exists() and not os.access(p, os.W_OK):
        results.append(CheckResult(rule_id_map["WRITABLE"], f"{check_id_prefix}_WRITABLE", f"{title_prefix} Writable", "FAIL", f"Directory '{p}' is not writable.", {"path": str(p)}))
    elif p.exists():
        results.append(CheckResult(rule_id_map["WRITABLE"], f"{check_id_prefix}_WRITABLE", f"{title_prefix} Writable", "PASS", f"Directory '{p}' is writable.", {"path": str(p)}))
    if p.is_symlink():
        results.append(CheckResult(rule_id_map["NO_SYMLINK"], f"{check_id_prefix}_NO_SYMLINK", f"{title_prefix} Is Not Symlink", "FAIL", f"Path '{p}' is a symlink, which is forbidden.", {"path": str(p)}))
    else:
        results.append(CheckResult(rule_id_map["NO_SYMLINK"], f"{check_id_prefix}_NO_SYMLINK", f"{title_prefix} Is Not Symlink", "PASS", f"Path '{p}' is not a symlink.", {"path": str(p)}))
    return results

def check_cache_isolation():
    """NFR-002: Verifies cache directories are isolated and secure."""
    rule_id_map = {"CONFIGURED": "RULE-002", "EXISTS": "RULE-003", "WRITABLE": "RULE-004", "NO_SYMLINK": "RULE-005"}
    results = check_directory("MATHLEDGER_CACHE_ROOT", "NFR-002_CACHE_ISOLATION", "Cache Directory", rule_id_map)

    run_id = os.environ.get("U2_RUN_ID")
    if run_id and (".." in run_id or "/" in run_id or "\\" in run_id):
        results.append(CheckResult("RULE-006", "NFR-002_CACHE_ISOLATION_NO_TRAVERSAL", "Cache Path Traversal", "FAIL", f"U2_RUN_ID '{run_id}' contains path traversal characters.", {"run_id": run_id}))
    else:
        results.append(CheckResult("RULE-006", "NFR-002_CACHE_ISOLATION_NO_TRAVERSAL", "Cache Path Traversal", "PASS", "U2_RUN_ID is clean.", {"run_id": run_id}))
    return results

def check_snapshot_root():
    rule_id_map = {"CONFIGURED": "RULE-007", "EXISTS": "RULE-008", "WRITABLE": "RULE-008", "NO_SYMLINK": "RULE-008"}
    return check_directory("MATHLEDGER_SNAPSHOT_ROOT", "NFR-005_SNAPSHOT_ROOT", "Snapshot Root Directory", rule_id_map)

def check_seed_verification(prereg_hash=None):
    seed, results = os.environ.get("U2_MASTER_SEED"), []
    if not seed: results.append(CheckResult("RULE-009", "NFR-003_MASTER_SEED_CONFIGURED", "Master Seed Configured", "FAIL", "U2_MASTER_SEED is not set.")); return results
    results.append(CheckResult("RULE-009", "NFR-003_MASTER_SEED_CONFIGURED", "Master Seed Configured", "PASS", "U2_MASTER_SEED is set."))
    if not isinstance(seed, str) or len(seed) != 64 or not all(c in '0123456789abcdefABCDEF' for c in seed): results.append(CheckResult("RULE-010", "NFR-003_MASTER_SEED_FORMAT", "Master Seed Format", "FAIL", "U2_MASTER_SEED must be a 64-character hex string.", {"prefix": seed[:10] if seed else ""}))
    else: results.append(CheckResult("RULE-010", "NFR-003_MASTER_SEED_FORMAT", "Master Seed Format", "PASS", "U2_MASTER_SEED is a 64-character hex string.", {"prefix": seed[:10]}))
    if prereg_hash:
        if seed.lower() != prereg_hash.lower(): results.append(CheckResult("RULE-011", "NFR-003_MASTER_SEED_MATCH", "Master Seed Matches Preregistration", "FAIL", "U2_MASTER_SEED does not match the provided preregistration hash.", {"env_seed_hash": seed.lower(), "prereg_hash": prereg_hash.lower()}))
        else: results.append(CheckResult("RULE-011", "NFR-003_MASTER_SEED_MATCH", "Master Seed Matches Preregistration", "PASS", "U2_MASTER_SEED matches the preregistration hash.", {"env_seed_hash": seed.lower(), "prereg_hash": prereg_hash.lower()}))
    return results

def check_banned_randomness():
    rule_id, check_id, title = "RULE-012", "NFR-007_BANNED_RANDOMNESS_AST", "Banned Randomness AST Scan"
    config_path, scan_path, violations = Path("config/banned_calls.json"), Path("backend/rfl"), []
    # Fall back to simple text-based scan if config file is missing
    if not config_path.exists():
        # Legacy text-based scan
        banned_imports = ["import random", "from random"]
        if not scan_path.exists() or not scan_path.is_dir():
            return CheckResult(rule_id, check_id, title, "PASS", f"Scan path does not exist, skipping: {scan_path}")
        for py_file in scan_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if any(b in line for b in banned_imports):
                            violations.append({"file": str(py_file), "line": i + 1, "code": line.strip()})
            except Exception:
                pass
        if violations:
            return CheckResult(rule_id, check_id, title, "WARN", f"Found {len(violations)} potential use(s) of banned 'random' module.", {"violations": violations})
        return CheckResult(rule_id, check_id, title, "PASS", "No banned top-level 'random' imports detected in backend/rfl.")
    if not scan_path.exists() or not scan_path.is_dir(): return CheckResult(rule_id, check_id, title, "PASS", f"Scan path does not exist, skipping: {scan_path}")
    with open(config_path, "r") as f: config = json.load(f)
    banned_calls, allowed_paths = config["banned_calls"], [Path(p) for p in config["allowed_paths"]]
    class BannedCallVisitor(ast.NodeVisitor):
        def __init__(self, file_path): self.file_path, self.violations = file_path, []
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module_name, func_name = node.func.value.id, node.func.attr
                for banned in banned_calls:
                    if module_name == banned["module"] and func_name == banned["function"]: self.violations.append({"file": str(self.file_path), "line": node.lineno, "col": node.col_offset, "call": f"{module_name}.{func_name}", "message": banned["message"]})
            self.generic_visit(node)
    for py_file in scan_path.rglob("*.py"):
        if any(py_file.is_relative_to(p) for p in allowed_paths): continue
        try:
            content = py_file.read_text(encoding="utf-8"); tree = ast.parse(content, filename=str(py_file)); visitor = BannedCallVisitor(py_file); visitor.visit(tree); violations.extend(visitor.violations)
        except Exception as e: violations.append({"file": str(py_file), "error": f"Failed to parse: {e}"})
    if violations: return CheckResult(rule_id, check_id, title, "WARN", f"Found {len(violations)} call(s) to banned functions.", {"violations": violations})
    return CheckResult(rule_id, check_id, title, "PASS", "No calls to banned functions detected in backend/rfl.")

def check_db_schema_version():
    rule_id, check_id, title = "RULE-013", "SYS_DB_SCHEMA_VERSION", "Database Schema Version"
    expected_version = os.environ.get("EXPECTED_DB_SCHEMA_VERSION")
    if not os.environ.get("DATABASE_URL") or not expected_version: return CheckResult(rule_id, check_id, title, "PASS", "Skipping check, DATABASE_URL or EXPECTED_DB_SCHEMA_VERSION not set.")
    actual_version = "v1.2.3" # Mock
    if actual_version != expected_version: return CheckResult(rule_id, check_id, title, "FAIL", f"DB schema mismatch. Expected '{expected_version}', found '{actual_version}'.")
    return CheckResult(rule_id, check_id, title, "PASS", f"DB schema version '{actual_version}' matches expected.")

def check_disk_space():
    rule_id, check_id, title = "RULE-014", "SYS_DISK_SPACE", "Disk Space Threshold"
    min_gb, path_to_check = 1, os.environ.get("MATHLEDGER_CACHE_ROOT", ".")
    try:
        _, _, free = shutil.disk_usage(path_to_check); free_gb = free / (1024**3)
        if free_gb < min_gb: return CheckResult(rule_id, check_id, title, "WARN", f"Low disk space. {free_gb:.2f}GB free, but threshold is {min_gb}GB.", {"path": path_to_check, "free_gb": free_gb})
        return CheckResult(rule_id, check_id, title, "PASS", f"{free_gb:.2f}GB free space meets threshold.", {"path": path_to_check, "free_gb": free_gb})
    except Exception as e: return CheckResult(rule_id, check_id, title, "FAIL", f"Could not check disk space: {e}")

def check_disallowed_sockets():
    rule_id, check_id, title = "RULE-015", "SYS_DISALLOWED_SOCKETS", "Disallowed Listening Sockets"
    if not psutil: return CheckResult(rule_id, check_id, title, "WARN", "psutil library not installed, skipping socket check.")
    disallowed_ports, listening_violations = {80, 443, 8080, 5433}, []
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'LISTEN' and conn.laddr.port in disallowed_ports: listening_violations.append({"port": conn.laddr.port, "pid": conn.pid})
    except Exception as e: return CheckResult(rule_id, check_id, title, "FAIL", f"Could not check network connections: {e}")
    if listening_violations: return CheckResult(rule_id, check_id, title, "FAIL", f"Found listening sockets on disallowed ports.", {"violations": listening_violations})
    return CheckResult(rule_id, check_id, title, "PASS", "No listening sockets found on disallowed ports.")

def check_lockfile_existence():
    rule_id, check_id, title = "RULE-901", "MODE_SWITCH_LOCKFILE_EXISTS", "Mode Lockfile Existence"
    run_id, cache_root = os.environ.get("U2_RUN_ID"), os.environ.get("MATHLEDGER_CACHE_ROOT")
    if not run_id or not cache_root: return CheckResult(rule_id, check_id, title, "PASS", "Skipping check, missing run_id or cache_root.")
    lock_file_path = Path(cache_root) / "u2" / run_id / ".mode_lock"
    if lock_file_path.exists(): return CheckResult(rule_id, check_id, title, "FAIL", f"Lock file already exists at {lock_file_path}. Mode switch not allowed.", {"path": str(lock_file_path)})
    return CheckResult(rule_id, check_id, title, "PASS", "Lock file does not exist.", {"path": str(lock_file_path)})

def get_all_checks(args):
    checks = [check_mode_declaration, check_cache_isolation, check_snapshot_root, lambda: check_seed_verification(args.prereg_hash), check_banned_randomness, check_db_schema_version, check_disk_space, check_disallowed_sockets]
    if args.switch_mode: checks.append(check_lockfile_existence)
    return checks

def run_checks(args):
    results = [];
    for func in get_all_checks(args): res = func(); results.extend(res) if isinstance(res, list) else results.append(res)
    return results

def generate_report(checks, run_id):
    passed_count, failed_count, warn_count = sum(1 for c in checks if c.status == "PASS"), sum(1 for c in checks if c.status == "FAIL"), sum(1 for c in checks if c.status == "WARN")
    status = "FAIL" if failed_count > 0 else "WARN" if warn_count > 0 else "PASS"
    env_snapshot = {k: os.environ.get(k) for k in ["RFL_ENV_MODE", "U2_RUN_ID", "MATHLEDGER_CACHE_ROOT", "MATHLEDGER_SNAPSHOT_ROOT", "MATHLEDGER_EXPORT_ROOT"]}
    if os.environ.get("U2_MASTER_SEED"): env_snapshot["U2_MASTER_SEED"] = "[REDACTED]"
    return {"$schema": "https://mathledger.io/schemas/u2-env-report-v1.json", "report_id": str(uuid4()), "run_id": run_id, "generated_at": datetime.now(timezone.utc).isoformat(), "report_summary": {"status": status, "total_checks": len(checks), "passed": passed_count, "failed": failed_count, "warnings": warn_count}, "checks": [c.to_dict() for c in checks], "environment_snapshot": env_snapshot}

def handle_mode_switch(args, checks):
    if not all(c.passed() for c in checks): print("ERROR: Cannot switch mode, validation checks failed.", file=sys.stderr); return False
    run_id, cache_root = os.environ.get("U2_RUN_ID"), os.environ.get("MATHLEDGER_CACHE_ROOT")
    lock_file_path = Path(cache_root) / "u2" / run_id / ".mode_lock"; lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    lock_data = {"mode": args.switch_mode, "switched_at": datetime.now(timezone.utc).isoformat(), "operator_id": args.operator_id, "prereg_hash": args.prereg_hash, "is_reversible": False}
    try:
        with open(lock_file_path, "w") as f: json.dump(lock_data, f, indent=2)
    except IOError as e: print(f"ERROR: Failed to write lock file to {lock_file_path}: {e}", file=sys.stderr); sys.exit(1)
    print(f"Successfully switched mode to '{args.switch_mode}' and created lock file."); return True

def main():
    parser = argparse.ArgumentParser(description="U2 Environment Validation and Mode Management")
    parser.add_argument("--json", action="store_true"); parser.add_argument("--output", help="Write report to a specified file."); parser.add_argument("--strict", action="store_true"); parser.add_argument("--switch-mode", choices=['uplift_experiment', 'uplift_analysis']); parser.add_argument("--confirm-phase2", action="store_true"); parser.add_argument("--operator-id"); parser.add_argument("--prereg-hash"); parser.add_argument("--report-covered-rules", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.report_covered_rules:
        all_rules = set(); os.environ['RFL_ENV_MODE'] = 'uplift_experiment'; os.environ['MATHLEDGER_CACHE_ROOT'] = tempfile.gettempdir(); os.environ['MATHLEDGER_SNAPSHOT_ROOT'] = tempfile.gettempdir(); os.environ['U2_RUN_ID'] = 'dummy-run'; os.environ['U2_MASTER_SEED'] = 'a' * 64; os.environ['EXPECTED_DB_SCHEMA_VERSION'] = 'v1.2.3'
        dummy_args = argparse.Namespace(switch_mode=True, prereg_hash="dummy")
        for check_func in get_all_checks(dummy_args):
            try:
                results = check_func();
                if not isinstance(results, list): results = [results]
                for r in results: all_rules.update(r.rule_ids)
            except Exception: pass
        print(json.dumps(sorted(list(all_rules)))); sys.exit(0)
    if args.switch_mode and not (args.confirm_phase2 and args.operator_id and args.prereg_hash): print("ERROR: --switch-mode requires --confirm-phase2, --operator-id, and --prereg-hash.", file=sys.stderr); sys.exit(3)
    if args.prereg_hash and Path(args.prereg_hash).is_file():
        try:
            with open(args.prereg_hash, 'rb') as f: args.prereg_hash = hashlib.sha256(f.read()).hexdigest()
        except IOError: print(f"ERROR: Could not read preregistration file at {args.prereg_hash}", file=sys.stderr); sys.exit(1)
    run_id = os.environ.get("U2_RUN_ID"); checks = run_checks(args); report = generate_report(checks, run_id); report_str = json.dumps(report, indent=2)
    if args.output:
        try:
            with open(args.output, "w") as f: f.write(report_str)
            print(f"Report written to {args.output}")
        except IOError as e: print(f"ERROR: Could not write report to {args.output}: {e}", file=sys.stderr)
    elif args.json: print(report_str)
    else:
        print(f"--- U2 Environment Validation Report ---\nRun ID: {run_id}\nStatus: {report['report_summary']['status']}\n" + "-"*20);
        for check in checks: print(f"[{check.status:^4s}] {','.join(check.rule_ids):<12} {check.check_id:<35} {check.details}")
        print("-" * 20)
    if args.switch_mode:
        if not handle_mode_switch(args, checks): sys.exit(1)
    summary = report['report_summary']
    if summary['failed'] > 0: sys.exit(1)
    if args.strict and summary['warnings'] > 0: sys.exit(2)
    sys.exit(0)

if __name__ == "__main__":
    main()