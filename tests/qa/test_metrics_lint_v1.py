import json, subprocess, tempfile, os, sys, pathlib, unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root for this worktree
LINTER = ROOT / "tools" / "metrics_lint_v1.py"

def run_linter(path):
    env = os.environ.copy()
    env.setdefault("NO_NETWORK","true")
    p = subprocess.run([sys.executable, str(LINTER), path], capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")

class TestMetricsLintV1(unittest.TestCase):
    """Test cases for metrics v1 linter."""

    def test_v1_only_exit_0(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
            rec = {
                "system":"fol",
                "mode":"baseline",
                "method":"fol-baseline",
                "seed":"1",                 # string
                "inserted_proofs":1,        # > 0
                "wall_minutes":0.1,         # > 0
                "block_no":1,
                "merkle":"0"*64
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            code, out = run_linter(f.name)
            self.assertEqual(code, 0, out)

    def test_mixed_schema_nonzero_and_has_message(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
            f.write('{"legacy": true}\n')
            f.write(json.dumps({
                "system":"fol","mode":"baseline","method":"fol-baseline",
                "seed":"1","inserted_proofs":1,"wall_minutes":0.1,"block_no":1,"merkle":"0"*64
            }) + "\n")
            f.flush()
            code, out = run_linter(f.name)
            self.assertIn(code, [1, 2], out)
            self.assertTrue(("mixed" in out.lower()) or ("schema" in out.lower()), out)

if __name__ == '__main__':
    unittest.main()
