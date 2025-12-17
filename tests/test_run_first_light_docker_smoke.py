from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List

from scripts import run_first_light_docker_smoke as smoke


def test_run_smoke_emits_summary(monkeypatch, capsys) -> None:
    """Ensure the smoke runner emits the expected JSON shape when docker is available."""
    # Pretend docker is available
    monkeypatch.setattr(smoke, "docker_available", lambda: True)

    calls: List[List[str]] = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="warn")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    summary, ok = smoke.run_smoke()
    assert ok is True
    assert summary["docker_found"] is True
    assert summary["p3_ok"] is True
    assert summary["p4_ok"] is True
    assert summary["p3_host_dir"] is None
    assert summary["reason_codes"] == []
    assert summary["p3_artifact_dir"] == smoke.P3_ARTIFACT_DIR
    assert summary["p4_artifact_dir"] == smoke.P4_ARTIFACT_DIR
    assert summary["host_extraction_instructions"] == smoke.HOST_EXTRACTION_INSTRUCTIONS
    assert "p3-harness" in " ".join(calls[0])
    assert "p4-harness" in " ".join(calls[1])

    # Validate JSON is serializable and contains required keys
    serialized = json.dumps(summary)
    decoded = json.loads(serialized)
    assert set(decoded.keys()) >= {
        "p3_ok",
        "p4_ok",
        "p3_artifact_dir",
        "p4_artifact_dir",
        "p3_host_dir",
        "output_paths",
        "container_image_tags",
        "stderr_excerpts",
        "host_extraction_instructions",
        "reason_codes",
        "docker_found",
    }


def test_run_smoke_dry_run_stable_keys(monkeypatch) -> None:
    """Smoke summary should include stable artifact pointers even when docker is missing."""
    monkeypatch.setattr(smoke, "docker_available", lambda: False)

    summary, ok = smoke.run_smoke()
    assert ok is True
    assert summary["docker_found"] is False
    assert summary["p3_ok"] is None
    assert summary["p4_ok"] is None
    assert summary["p3_host_dir"] is None
    assert summary["reason_codes"] == [smoke.REASON_DOCKER_NOT_FOUND]
    assert summary["p3_artifact_dir"] == smoke.P3_ARTIFACT_DIR
    assert summary["p4_artifact_dir"] == smoke.P4_ARTIFACT_DIR
    assert summary["host_extraction_instructions"] == smoke.HOST_EXTRACTION_INSTRUCTIONS


def test_run_smoke_with_p3_bind_mount(monkeypatch, tmp_path: Path) -> None:
    """Passing --p3-bind-mount should add volume mount and set p3_host_dir."""
    monkeypatch.setattr(smoke, "docker_available", lambda: True)
    monkeypatch.chdir(tmp_path)

    calls: List[List[str]] = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        if "p3-harness" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="P3_COPY_OK=1\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    host_dir = "p3_artifacts"
    summary, ok = smoke.run_smoke(p3_bind_mount=host_dir)
    assert ok is True
    expected_host_dir = (tmp_path / host_dir).resolve().as_posix()
    assert summary["p3_host_dir"] == expected_host_dir
    assert summary["reason_codes"] == []

    p3_cmd = calls[0]
    assert "-v" in p3_cmd
    vol_idx = p3_cmd.index("-v")
    assert p3_cmd[vol_idx + 1] == f"{expected_host_dir}:{smoke.P3_MOUNT_DIR}"
    assert "sh" in p3_cmd


def test_reason_codes_copy_failed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(smoke, "docker_available", lambda: True)
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, capture_output, text):
        stdout = "P3_COPY_OK=0\n" if "p3-harness" in cmd else "ok"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    summary, ok = smoke.run_smoke(p3_bind_mount="p3_artifacts")
    assert ok is True
    assert summary["reason_codes"] == [smoke.REASON_COPY_FAILED]


def test_reason_codes_ordering(monkeypatch) -> None:
    monkeypatch.setattr(smoke, "docker_available", lambda: False)

    summary, ok = smoke.run_smoke(p3_bind_mount="bad\0path")
    assert ok is True
    assert summary["reason_codes"] == [
        smoke.REASON_DOCKER_NOT_FOUND,
        smoke.REASON_BIND_MOUNT_INVALID,
        smoke.REASON_COPY_FAILED,
    ]
