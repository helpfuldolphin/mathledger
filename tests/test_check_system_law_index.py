import json

from scripts import check_system_law_index


def test_wrapper_reports_up_to_date(monkeypatch, capsys):
    def fake_main(args=None):
        print("indexer noise")
        return 0

    monkeypatch.setattr(check_system_law_index.indexer, "main", fake_main)

    exit_code = check_system_law_index.main([])
    captured = capsys.readouterr().out.strip()

    assert exit_code == 0
    assert captured == "system-law-index: up to date (no action needed)"


def test_wrapper_reports_out_of_date(monkeypatch, capsys):
    def fake_main(args=None):
        print("indexer noise")
        return 1

    monkeypatch.setattr(check_system_law_index.indexer, "main", fake_main)

    exit_code = check_system_law_index.main([])
    captured = capsys.readouterr().out.strip()

    assert exit_code == 0
    assert captured == (
        "system-law-index: out of date; run `python tools/generate_system_law_index.py`"
    )


def test_wrapper_reports_up_to_date_json(monkeypatch, capsys):
    def fake_main(args=None):
        print("indexer noise")
        return 0

    monkeypatch.setattr(check_system_law_index.indexer, "main", fake_main)

    exit_code = check_system_law_index.main(["--json"])
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)

    assert exit_code == 0
    assert payload == {
        "schema_version": 1,
        "mode": "SHADOW",
        "up_to_date": True,
        "reason_code": "UP_TO_DATE",
        "remediation": "no action needed",
    }


def test_wrapper_reports_out_of_date_json(monkeypatch, capsys):
    def fake_main(args=None):
        print("indexer noise")
        return 1

    monkeypatch.setattr(check_system_law_index.indexer, "main", fake_main)

    exit_code = check_system_law_index.main(["--json"])
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)

    assert exit_code == 0
    assert payload == {
        "schema_version": 1,
        "mode": "SHADOW",
        "up_to_date": False,
        "reason_code": "OUT_OF_DATE",
        "remediation": "run `python tools/generate_system_law_index.py`",
    }


def test_wrapper_json_stdout_matches_file_output(monkeypatch, capsys, tmp_path):
    def fake_main(args=None):
        print("indexer noise")
        return 0

    monkeypatch.setattr(check_system_law_index.indexer, "main", fake_main)

    output_path = tmp_path / "system_law_index.json"
    exit_code = check_system_law_index.main(["--json", "--output", str(output_path)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert output_path.read_bytes() == captured.encode("utf-8")


def test_wrapper_json_stdout_matches_file_output_when_out_of_date(
    monkeypatch, capsys, tmp_path
):
    def fake_main(args=None):
        print("indexer noise")
        return 1

    monkeypatch.setattr(check_system_law_index.indexer, "main", fake_main)

    output_path = tmp_path / "system_law_index.json"
    exit_code = check_system_law_index.main(["--json", "--output", str(output_path)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert output_path.read_bytes() == captured.encode("utf-8")


def test_wrapper_reports_write_failed_json(monkeypatch, capsys, tmp_path):
    def fake_main(args=None):
        print("indexer noise")
        return 0

    def fake_write_text(self, data, encoding=None, errors=None, newline=None):
        raise OSError("boom")

    monkeypatch.setattr(check_system_law_index.indexer, "main", fake_main)
    monkeypatch.setattr(check_system_law_index.Path, "write_text", fake_write_text)

    output_path = tmp_path / "system_law_index.json"
    exit_code = check_system_law_index.main(["--json", "--output", str(output_path)])
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)

    assert exit_code == 0
    assert not output_path.exists()
    assert payload == {
        "schema_version": 1,
        "mode": "SHADOW",
        "up_to_date": True,
        "reason_code": "WRITE_FAILED",
        "remediation": "no action needed",
    }
