import ast
import sys
import re
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Set

# --- Configuration ---

FORBIDDEN_CALLS = {
    # module: {function1, function2, ...}
    "random": {"random", "randint", "randrange", "choice", "choices", "shuffle", "sample", "uniform"},
    "numpy.random": {"rand", "randn", "randint", "random_integers", "random_sample", "choice"},
    "os": {"urandom"},
    "time": {"time"},
}

WAIVER_PATTERN = re.compile(r'#\s*DETERMINISM-WAIVER:\s*JUSTIFIED\s*([A-Z0-9-]+)')
WHITELIST_FILE = "determinism_waivers.yml"

# --- AST Visitor ---

class RandomnessVisitor(ast.NodeVisitor):
    """
    An AST visitor that finds calls to forbidden functions.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.violations: List[Tuple[int, int, str]] = []
        # Track aliases and direct imports
        self._numpy_alias = "numpy"
        self._forbidden_imports: Dict[str, str] = {}  # e.g., {"random": "random", "urandom": "os"}

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name == "numpy":
                self._numpy_alias = alias.asname or "numpy"
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module in FORBIDDEN_CALLS:
            for alias in node.names:
                if alias.name in FORBIDDEN_CALLS[node.module]:
                    imported_as = alias.asname or alias.name
                    self._forbidden_imports[imported_as] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """
        Check if a function call is one of the forbidden patterns.
        """
        # Case 1: Direct call to a forbidden imported function (e.g., `random()`)
        if isinstance(node.func, ast.Name) and node.func.id in self._forbidden_imports:
            original_name = self._forbidden_imports[node.func.id]
            self.violations.append(
                (node.lineno, node.col_offset, f"Direct call to forbidden import '{original_name}'")
            )

        # Case 2: Attribute call (e.g., `random.random()`)
        if isinstance(node.func, ast.Attribute):
            func = node.func
            module_name = self._get_full_module_name(func.value)
            func_name = func.attr

            # Check for numpy alias
            if module_name == self._numpy_alias:
                module_name = "numpy"
            
            # Special handling for numpy.random
            if module_name and module_name.endswith(".random"):
                 if "numpy" in module_name: # e.g., from `np.random.rand()`
                     module_name = "numpy.random"

            if module_name in FORBIDDEN_CALLS and func_name in FORBIDDEN_CALLS[module_name]:
                self.violations.append(
                    (node.lineno, node.col_offset, f"{module_name}.{func_name}")
                )

            # Case 3: Unseeded numpy.random.RandomState
            if module_name == "numpy.random" and func_name == "RandomState":
                if not node.args and not any(kw.arg == 'seed' for kw in node.keywords):
                    self.violations.append(
                        (node.lineno, node.col_offset, "Unseeded numpy.random.RandomState instantiation")
                    )
        
        self.generic_visit(node)

    def _get_full_module_name(self, node: ast.AST) -> str:
        """Recursively resolve the full name of a module (e.g., `numpy.random`)."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._get_full_module_name(node.value)}.{node.attr}"
        return ""

# --- Main Linter Logic ---

def load_waiver_whitelist() -> Set[str]:
    """Loads the set of authorized waiver ticket IDs from the YAML file."""
    whitelist_path = Path(WHITELIST_FILE)
    if not whitelist_path.exists():
        print(f"Warning: Whitelist file not found at '{WHITELIST_FILE}'. No waivers will be honored.", file=sys.stderr)
        return set()
    
    with open(whitelist_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
            if data and 'authorized_waivers' in data and isinstance(data['authorized_waivers'], list):
                return set(data['authorized_waivers'])
            return set()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML from {WHITELIST_FILE}: {e}", file=sys.stderr)
            return set()

def lint_file(file_path: Path, authorized_waivers: Set[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Lints a single Python file, separating hard violations from waived ones.
    """
    hard_violations = []
    waived_violations = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_lines = f.readlines()
        
        source_text = "".join(source_lines)
        tree = ast.parse(source_text, filename=str(file_path))
        visitor = RandomnessVisitor(str(file_path))
        visitor.visit(tree)

        for lineno, col, pattern in visitor.violations:
            violation_details = {
                "file": str(file_path),
                "line": lineno,
                "col": col,
                "pattern": pattern
            }
            line_content = source_lines[lineno - 1].strip()
            match = WAIVER_PATTERN.search(line_content)
            
            if match:
                ticket_id = match.group(1)
                if ticket_id in authorized_waivers:
                    violation_details["waiver_ticket"] = ticket_id
                    waived_violations.append(violation_details)
                else:
                    violation_details["error"] = f"Unapproved waiver ticket '{ticket_id}'"
                    hard_violations.append(violation_details)
            else:
                hard_violations.append(violation_details)

        return hard_violations, waived_violations

    except (SyntaxError, UnicodeDecodeError, IndexError) as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return [{"file": str(file_path), "error": str(e)}], []


def main():
    """
    Main entry point for the static linter.
    """
    if len(sys.argv) < 2:
        print("Usage: python backend/security/randomness_static_linter.py <file1.py> <file2.py> ...", file=sys.stderr)
        sys.exit(1)

    authorized_waivers = load_waiver_whitelist()
    target_files = [Path(p) for p in sys.argv[1:] if p.endswith(".py")]
    
    total_hard_violations = 0
    total_waived_violations = 0
    
    print("--- GEMINI-K: Running Randomness Static Linter ---")
    print(f"Loaded {len(authorized_waivers)} authorized waiver ticket(s) from {WHITELIST_FILE}")

    for file_path in target_files:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue
        
        hard, waived = lint_file(file_path, authorized_waivers)
        
        if waived:
            print(f"\n[WAIVED] Found {len(waived)} waived violation(s) in {file_path}:")
            for v in waived:
                print(f"  -> {v['file']}:{v['line']}:{v['col']}: Waived '{v['pattern']}' (Ticket: {v['waiver_ticket']})")
            total_waived_violations += len(waived)

        if hard:
            print(f"\n[VIOLATION] Found {len(hard)} hard violation(s) in {file_path}:")
            for v in hard:
                error_msg = f" -> {v.get('file', 'N/A')}:{v.get('line', 'N/A')}:{v.get('col', 'N/A')}: Forbidden call to '{v.get('pattern', 'N/A')}'"
                if 'error' in v:
                    error_msg += f" - ERROR: {v['error']}"
                print(error_msg, file=sys.stderr)
            total_hard_violations += len(hard)

    print("-" * 50)
    if total_hard_violations > 0:
        print(f"Linter Verdict: FAILED. Found {total_hard_violations} hard violation(s).")
        if total_waived_violations > 0:
            print(f"Note: Found and accepted {total_waived_violations} waived violation(s).")
        sys.exit(1)
    else:
        print("Linter Verdict: PASSED.")
        print(f"Summary: {total_hard_violations} hard violations, {total_waived_violations} waived violations.")
        sys.exit(0)

if __name__ == "__main__":
    # Add pyyaml to dependencies if not present
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is not installed. Please run 'pip install pyyaml'", file=sys.stderr)
        sys.exit(1)
    main()
