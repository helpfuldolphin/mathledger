import json
import sys
import types
from pathlib import Path

import pytest

# Provide a lightweight jsonschema stub for tests (schema validation is optional here)
class _DummyValidator:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def iter_errors(self, *_: object, **__: object):
        return []


if "jsonschema" not in sys.modules:
    dummy = types.SimpleNamespace(
        Draft7Validator=_DummyValidator,
        exceptions=types.SimpleNamespace(ValidationError=Exception),
        ValidationError=Exception,
        SchemaError=Exception,
        validate=lambda *args, **kwargs: None,
    )
    sys.modules["jsonschema"] = dummy
    sys.modules["jsonschema.exceptions"] = dummy.exceptions

from backend.topology.first_light.evidence_pack import build_evidence_pack
from scripts.generate_first_light_status import generate_status


pytestmark = pytest.mark.unit


def _run_spike_harness(tmp_dir: Path) -> Path:
    """Run a short spike pathology harness and return the run directory."""
    root = Path(__file__).resolve().parents[2]
    output_dir = tmp_dir / "p3_runs"
    cmd = [
        sys.executable,
        "scripts/usla_first_light_harness.py",
        "--cycles",
        "15",
        "--seed",
        "321",
        "--output-dir",
        str(output_dir),
        "--pathology",
        "spike",
    ]
    # Harness prints progress; no need to capture output
    import subprocess

    subprocess.check_call(cmd, cwd=root)

    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    return run_dirs[0]


def _write_minimal_p4_stub(p4_base: Path) -> None:
    """Write minimal P4 stub artifacts to satisfy status generator expectations."""
    run_dir = p4_base / "p4_stub"
    run_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        "real_cycles.jsonl",
        "twin_predictions.jsonl",
        "divergence_log.jsonl",
        "p4_summary.json",
        "twin_accuracy.json",
        "run_config.json",
    ]
    for name in expected:
        path = run_dir / name
        path.touch()
        if name == "p4_summary.json":
            path.write_text(json.dumps({"mode": "SHADOW"}), encoding="utf-8")


def test_pathology_annotated_in_manifest_and_status(tmp_path: Path) -> None:
    run_dir = _run_spike_harness(tmp_path)

    # Build evidence pack (manifest written into run_dir)
    result = build_evidence_pack(run_dir)
    assert result.success
    manifest_path = Path(result.manifest_path)
    manifest = json.loads(manifest_path.read_text())

    pathology_evidence = manifest["evidence"]["data"]["p3_pathology"]
    assert pathology_evidence["pathology"] == "spike"
    assert pathology_evidence["pathology_params"]["magnitude"] == pytest.approx(0.75)

    governance = manifest["governance"]["p3_pathology"]
    assert governance["type"] == "spike"
    assert governance["expected_effects"]
    assert governance["magnitude"] == pytest.approx(0.75)

    # Validate pathology blocks against schema when jsonschema is available
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "evidence" / "p3_pathology.schema.json"
    jsonschema_mod = sys.modules.get("jsonschema")
    if jsonschema_mod and hasattr(jsonschema_mod, "Draft7Validator"):
        schema = json.loads(schema_path.read_text())
        validator_cls = getattr(jsonschema_mod, "Draft7Validator")
        assert not list(validator_cls(schema).iter_errors(pathology_evidence))
        assert not list(validator_cls(schema).iter_errors(governance))

    # Generate status JSON and ensure pathology marker is present
    p4_dir = tmp_path / "p4_runs"
    _write_minimal_p4_stub(p4_dir)
    status = generate_status(
        p3_dir=run_dir.parent,
        p4_dir=p4_dir,
        evidence_pack_dir=run_dir,
    )

    assert status["pathology_used"] is True
    assert status["pathology_type"] == "spike"
