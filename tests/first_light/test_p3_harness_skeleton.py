from __future__ import annotations

import sys
from pathlib import Path
import warnings

from scripts import first_light_p3_harness


def test_harness_wrapper_delegates_and_preserves_artifact(monkeypatch, tmp_path: Path) -> None:
    forwarded: dict[str, object] = {}

    def fake_main() -> int:
        forwarded["argv"] = tuple(sys.argv)
        argv = forwarded["argv"]
        assert argv is not None
        assert argv[0].endswith("usla_first_light_harness.py")
        output_idx = argv.index("--output-dir")
        harness_output = Path(argv[output_idx + 1])
        run_dir = harness_output / "fl_fake_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "synthetic_raw.jsonl").write_text('{"cycle": 1}\n', encoding="utf-8")
        return 17

    monkeypatch.setattr(
        "scripts.first_light_p3_harness.usla_first_light_harness.main",
        fake_main,
    )

    args = [
        "--cycles",
        "25",
        "--output-dir",
        str(tmp_path),
        "--seed",
        "123",
        "--slice",
        "propositional_tautology",
        "--runner-type",
        "rfl",
        "--tau-0",
        "0.33",
        "--window-size",
        "20",
        "--dry-run",
        "--verbose",
        "--extra-flag",
        "value",
    ]

    exit_code = first_light_p3_harness.main(args)
    assert exit_code == 17

    argv = forwarded["argv"]
    assert argv is not None
    assert "--extra-flag" in argv
    assert "--dry-run" in argv
    assert "--verbose" in argv

    legacy_path = tmp_path / first_light_p3_harness.OUTPUT_FILENAME
    assert legacy_path.exists()
    assert legacy_path.read_text(encoding="utf-8").strip() == '{"cycle": 1}'


def test_wrapper_mirror_overwrites_existing_placeholder(monkeypatch, tmp_path: Path) -> None:
    def fake_main() -> int:
        argv = tuple(sys.argv)
        output_idx = argv.index("--output-dir")
        harness_output = Path(argv[output_idx + 1])
        run_dir = harness_output / "run_latest"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / first_light_p3_harness.REAL_PRIMARY_FILENAME).write_text(
            '{"cycle": 5}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "scripts.first_light_p3_harness.usla_first_light_harness.main",
        fake_main,
    )

    legacy_path = tmp_path / first_light_p3_harness.OUTPUT_FILENAME
    legacy_path.write_text("stale", encoding="utf-8")

    exit_code = first_light_p3_harness.main(
        [
            "--cycles",
            "10",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert legacy_path.read_text(encoding="utf-8").strip() == '{"cycle": 5}'


def test_wrapper_emits_warning_and_banner(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_main() -> int:
        argv = tuple(sys.argv)
        output_idx = argv.index("--output-dir")
        harness_output = Path(argv[output_idx + 1])
        run_dir = harness_output / "run_warning"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / first_light_p3_harness.REAL_PRIMARY_FILENAME).write_text(
            '{"cycle": 9}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "scripts.first_light_p3_harness.usla_first_light_harness.main",
        fake_main,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first_light_p3_harness.main(
            [
                "--cycles",
                "12",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert any(
        "scripts/usla_first_light_harness.py" in str(item.message)
        for item in caught
    )

    captured = capsys.readouterr()
    assert "LEGACY ONLY — DO NOT USE FOR NEW RUNS" in captured.out
    assert "uv run python scripts/usla_first_light_harness.py" in captured.out


def test_wrapper_does_not_support_quiet_legacy_warning_flag() -> None:
    help_text = first_light_p3_harness._build_parser().format_help()
    assert "--quiet-legacy-warning" not in help_text


def test_wrapper_attempted_suppression_still_prints_banner(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_main() -> int:
        argv = tuple(sys.argv)
        output_idx = argv.index("--output-dir")
        harness_output = Path(argv[output_idx + 1])
        run_dir = harness_output / "run_quiet"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / first_light_p3_harness.REAL_PRIMARY_FILENAME).write_text(
            '{"cycle": 11}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "scripts.first_light_p3_harness.usla_first_light_harness.main",
        fake_main,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first_light_p3_harness.main(
            [
                "--quiet-legacy-warning",
                "--cycles",
                "12",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert any(
        "scripts/usla_first_light_harness.py" in str(item.message)
        for item in caught
    )

    captured = capsys.readouterr()
    assert "LEGACY ONLY — DO NOT USE FOR NEW RUNS" in captured.out
