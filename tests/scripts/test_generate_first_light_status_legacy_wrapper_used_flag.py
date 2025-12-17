from __future__ import annotations

from pathlib import Path


def test_generate_status_emits_legacy_wrapper_used_flag(tmp_path: Path) -> None:
    from scripts import generate_first_light_status

    p3_dir = tmp_path / "p3"
    pack_dir = tmp_path / "pack"
    p3_dir.mkdir()
    pack_dir.mkdir()

    legacy_path = p3_dir / generate_first_light_status.LEGACY_P3_WRAPPER_PRIMARY_FILENAME
    legacy_path.write_text('{"cycle": 1}\n', encoding="utf-8")

    status = generate_first_light_status.generate_status(p3_dir=p3_dir, evidence_pack_dir=pack_dir)
    assert status["legacy_wrapper_used"] is True

    legacy_path.unlink()
    status = generate_first_light_status.generate_status(p3_dir=p3_dir, evidence_pack_dir=pack_dir)
    assert status["legacy_wrapper_used"] is False


def test_legacy_p3_wrapper_ci_check_dry_run_reports_callsite(tmp_path: Path, capsys) -> None:
    from scripts import legacy_p3_wrapper_ci_check

    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()

    (tmp_path / "scripts" / "first_light_p3_harness.py").write_text("# legacy wrapper\n", encoding="utf-8")
    (tmp_path / "tests" / "test_allowed.py").write_text(
        "print('python scripts/first_light_p3_harness.py')\n",
        encoding="utf-8",
    )

    violating = tmp_path / "scripts" / "calls_legacy_wrapper.py"
    violating.write_text(
        "cmd = 'uv run python scripts/first_light_p3_harness.py --cycles 1'\n",
        encoding="utf-8",
    )

    exit_code = legacy_p3_wrapper_ci_check.main(["--root", str(tmp_path), "--dry-run"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "FOUND" in captured.out
    assert "scripts/calls_legacy_wrapper.py" in captured.out
    assert "tests/test_allowed.py" not in captured.out
