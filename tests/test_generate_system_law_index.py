from pathlib import Path

from tools import generate_system_law_index as indexer


def test_check_mode_detects_divergence(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs" / "system_law"
    docs_dir.mkdir(parents=True)
    (docs_dir / "alpha.md").write_text("alpha", encoding="utf-8")
    (docs_dir / "beta.md").write_text("beta", encoding="utf-8")

    index_path = docs_dir / "index.md"
    monkeypatch.setattr(indexer, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)

    assert indexer.main([]) == 0
    baseline = index_path.read_text(encoding="utf-8")

    index_path.write_text(baseline + "\nextra\n", encoding="utf-8")
    exit_code = indexer.main(["--check"])

    assert exit_code == 1
    assert index_path.read_text(encoding="utf-8").endswith("extra\n")
