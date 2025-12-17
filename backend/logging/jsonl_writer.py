from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO


class JsonlWriter:
    """Append-only JSONL writer with simple rotation-friendly semantics.

    **Invariants (must not change without a version bump):**
      * **Record ordering:** each call to ``write(record)`` appends exactly one
        JSON object line, in the same order as calls were made.
      * **Key ordering:** object keys are emitted in the insertion order of the
        input dict (no sorting is performed).
      * **Separators:** by default JSON is rendered in compact form using
        ``separators=(",", ":")`` and no trailing whitespace. A single logical
        newline is appended after each record (platform text-mode translation
        may persist this as CRLF on Windows).
      * **Encoding:** files are opened in text append mode with UTF-8 encoding.
        By default ``ensure_ascii=False`` so non‑ASCII characters are written
        literally (not escaped). Callers may override ``ensure_ascii`` or
        ``separators`` via ``json_kwargs``.
      * **Flush:** every successful ``write`` flushes the file handle before
        returning, so records are durable even if the process exits abruptly.

    Rotation hook (spec only):
      * Harnesses monitor file size or age externally (e.g., os.path.getsize).
      * Once a threshold is exceeded they call ``close()`` and create a new
        JsonlWriter pointing at the next rollover path (timestamp suffix, etc.).
      * A future `JsonlRotator` helper will encapsulate this by accepting a
        ``path_factory`` and ``max_bytes`` so harnesses only provide records and
        receive callbacks when archives finalize.
    """

    def __init__(self, path: str, *, json_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[TextIO] = self._path.open("a", encoding="utf-8")
        self._closed = False
        base_kwargs: Dict[str, Any] = {"separators": (",", ":"), "ensure_ascii": False}
        if json_kwargs:
            base_kwargs.update(json_kwargs)
        self._json_kwargs = base_kwargs

    def write(self, obj: Dict[str, Any]) -> None:
        if self._closed or self._file is None:
            raise ValueError("Cannot write to a closed JsonlWriter.")
        line = json.dumps(obj, **self._json_kwargs)
        self._file.write(f"{line}\n")
        self._file.flush()

    def close(self) -> None:
        if not self._closed and self._file is not None:
            self._file.close()
            self._file = None
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class JsonlRotator:
    """Placeholder helper for size-based JSONL rollover.

    **Stable interface (Phase Y will replace this stub):**
      * ``__init__(path_factory, max_bytes, *, json_kwargs=None)``
      * ``write(record)``
      * ``close()``

    **Output invariants:** a concrete implementation MUST preserve the
    ``JsonlWriter`` invariants above: record order, key insertion order, default
    compact separators, UTF-8 encoding (with literal Unicode unless overridden),
    and flush-after-write semantics. Rotation must be transparent to callers:
    they provide records; the rotator decides when to roll files, but never
    reorders or drops records.

    Concrete implementations are guarded by the contract harness in
    ``tests/logging/test_jsonl_rotator_contract.py``.

    System-law contract: ``docs/system_law/logging/JSONL_ROTATION_CONTRACT.md``.

    TODO (Phase Y):
      * Accept a ``path_factory`` that yields the next log path (e.g., timestamp
        suffix or numbered segments like ``synthetic_raw.0003.jsonl``).
      * Track ``max_bytes``; when the active file exceeds the threshold close
        the current writer, rename/archive it (if needed), then open a fresh
        JsonlWriter instance.
      * Surface callbacks so harnesses can publish rotation events.
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

    def write(self, obj: Dict[str, Any]) -> None:
        raise NotImplementedError("JsonlRotator is a stub; implement in Phase Y.")

    def close(self) -> None:
        raise NotImplementedError("JsonlRotator is a stub; implement in Phase Y.")
