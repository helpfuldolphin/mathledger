from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import pytest

from backend.logging.jsonl_writer import JsonlWriter
from backend.logging.jsonl_writer import JsonlRotator


class DemoRotator:
    """Test-only rotator that mimics Phase Y behavior.

    Rotates after a write pushes the active segment to >= max_bytes.
    Uses JsonlWriter for serialization so invariants match production.
    """

    def __init__(
        self,
        path_factory: Callable[[], str],
        max_bytes: int,
        *,
        json_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._path_factory = path_factory
        self._max_bytes = max_bytes
        self._json_kwargs = json_kwargs
        self._writer: Optional[JsonlWriter] = None
        self._active_path: Optional[Path] = None

    def _ensure_writer(self) -> None:
        if self._writer is None:
            path_str = self._path_factory()
            self._active_path = Path(path_str)
            self._writer = JsonlWriter(path_str, json_kwargs=self._json_kwargs)

    def write(self, record: Dict[str, Any]) -> None:
        self._ensure_writer()
        assert self._writer is not None
        self._writer.write(record)

        if self._active_path is not None and self._active_path.exists():
            if self._active_path.stat().st_size >= self._max_bytes:
                self._writer.close()
                self._writer = None
                self._active_path = None

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._active_path = None


def _run_rotator_contract(rotator_cls: Type[Any], tmp_path: Path) -> None:
    """Exercise a rotator implementation and assert Jsonl invariants.

    Contract:
      * Never reorder records across rotations.
      * Never mutate record content (including in-place dict mutation).
      * Preserve UTF-8 literal Unicode (ensure_ascii=False by default).
      * Preserve compact JsonlWriter separators (",", ":") by default.
    """
    created_paths: List[Path] = []
    counter = itertools.count()

    def path_factory() -> str:
        idx = next(counter)
        path = tmp_path / f"segment.{idx}.jsonl"
        created_paths.append(path)
        return str(path)

    rotator = rotator_cls(path_factory, max_bytes=1)

    records: List[Dict[str, Any]] = [
        {"alpha": "A", "pi": "π", "count": 1},
        {"beta": 2, "label": "漢字", "nested": {"x": 1, "y": "Ω"}},
        {"gamma": [1, 2, 3], "note": "plain-ascii"},
    ]
    originals = [copy.deepcopy(r) for r in records]

    write_exc: Optional[BaseException] = None
    try:
        for record in records:
            rotator.write(record)
    except BaseException as exc:  # pragma: no cover
        write_exc = exc
        raise
    finally:
        try:
            rotator.close()
        except BaseException:
            if write_exc is None:
                raise

    assert records == originals, "Rotator must not mutate input records"

    raw_lines: List[str] = []
    for path in created_paths:
        if path.exists():
            raw_lines.extend(path.read_text(encoding="utf-8").splitlines())

    assert len(raw_lines) == len(originals)

    parsed = [json.loads(line) for line in raw_lines]
    assert parsed == originals, "Rotator must preserve order and content"

    expected_lines = [
        json.dumps(r, separators=(",", ":"), ensure_ascii=False) for r in originals
    ]
    assert raw_lines == expected_lines, "Rotator must preserve separators and literal Unicode"


def test_demo_rotator_satisfies_contract(tmp_path):
    _run_rotator_contract(DemoRotator, tmp_path)


def _skip_if_jsonl_rotator_stub(tmp_path: Path) -> None:
    stub_message = "JsonlRotator is a stub; implement in Phase Y."

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    def probe_path_factory() -> str:
        return str(probe_dir / "probe.jsonl")

    probe = JsonlRotator(probe_path_factory, max_bytes=1)
    try:
        probe.write({"probe": True})
    except NotImplementedError as exc:
        if str(exc) == stub_message:
            with pytest.raises(NotImplementedError):
                probe.close()
            pytest.skip("JsonlRotator is still a Phase Y stub.")
        raise
    else:
        probe.close()


def test_jsonl_rotator_satisfies_contract_when_implemented(tmp_path):
    _skip_if_jsonl_rotator_stub(tmp_path)

    contract_dir = tmp_path / "contract"
    contract_dir.mkdir(parents=True, exist_ok=True)
    _run_rotator_contract(JsonlRotator, contract_dir)


def test_jsonl_rotator_does_not_mutate_input_record_when_implemented(tmp_path):
    _skip_if_jsonl_rotator_stub(tmp_path)

    counter = itertools.count()

    def path_factory() -> str:
        idx = next(counter)
        return str(tmp_path / f"mutation.{idx}.jsonl")

    rotator = JsonlRotator(path_factory, max_bytes=1)
    record = {"alpha": "A", "pi": "π", "nested": {"x": 1, "y": "Ω"}}
    expected = copy.deepcopy(record)

    rotator.write(record)
    rotator.close()

    assert record == expected, "JsonlRotator must not mutate caller-owned records"
