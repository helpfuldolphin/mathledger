import types
from collections import deque
import sys

import pytest

from backend.axiom_engine import derive as derive_module
from backend.axiom_engine.derive import DerivationEngine


class FakeCursor:
    def __init__(self):
        self.fetchall_queue = deque()
        self.fetchone_queue = deque()
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.executed.append((normalized, params))
        if "information_schema.columns" in sql:
            table = None
            if params:
                table = params[0]
            elif "table_name='systems'" in sql:
                table = "systems"
            elif "table_name='statements'" in sql:
                table = "statements"
            elif "table_name='proofs'" in sql:
                table = "proofs"
            if table == "systems":
                self.fetchall_queue.append([("id",), ("name",), ("slug",), ("created_at",), ("updated_at",)])
            elif table == "statements":
                self.fetchall_queue.append([(col,) for col in ("id", "system_id", "text", "normalized_text", "hash", "created_at", "updated_at")])
            elif table == "proofs":
                self.fetchall_queue.append([(col,) for col in ("id", "statement_id", "method", "status", "success", "created_at")])
            else:
                raise AssertionError(f"Unexpected table lookup: {sql} {params}")
        elif "SELECT id FROM systems" in sql:
            self.fetchone_queue.append(None)
        elif normalized.startswith("INSERT INTO systems"):
            self.fetchone_queue.append((1,))
        elif "SELECT text, normalized_text, hash FROM statements" in normalized:
            self.fetchall_queue.append([
                ("p -> q", "p -> q", derive_module._sha("p -> q")),
                ("p", "p", derive_module._sha("p")),
            ])
        elif "SELECT id FROM statements WHERE hash" in normalized:
            self.fetchone_queue.append(None)
        elif normalized.startswith("INSERT INTO statements"):
            self.fetchone_queue.append((2,))
        elif normalized.startswith("INSERT INTO proofs"):
            self.fetchone_queue.append((3,))
        elif "INSERT INTO proof_parents" in normalized:
            # no result required
            return
        else:
            raise AssertionError(f"Unexpected SQL: {normalized}")

    def fetchall(self):
        if not self.fetchall_queue:
            raise AssertionError("fetchall queue exhausted")
        return self.fetchall_queue.popleft()

    def fetchone(self):
        if not self.fetchone_queue:
            return None
        return self.fetchone_queue.popleft()


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.commits = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_derivation_engine_nominal(monkeypatch):
    fake_conn = FakeConnection()
    fake_psycopg = types.SimpleNamespace(connect=lambda *a, **k: fake_conn)
    monkeypatch.setattr(derive_module, "psycopg", fake_psycopg)

    engine = DerivationEngine("postgresql://test", "redis://test", max_breadth=5)
    summary = engine.derive_statements(steps=1)

    assert summary["n_new"] == 1
    assert summary["pct_success"] == pytest.approx(100.0)
    assert fake_conn.commits == 1


def test_derivation_engine_error_path(monkeypatch):
    monkeypatch.setattr(derive_module, "psycopg", types.SimpleNamespace(connect=lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))))

    engine = DerivationEngine("postgresql://test", "redis://test")
    summary = engine.derive_statements(steps=2)

    assert summary == {"n_new": 0, "max_depth": 0, "n_jobs": 0, "pct_success": 0.0}


def test_upsert_statement_fallback(monkeypatch):
    calls = {
        "select": 0,
        "insert": 0,
    }

    class MinimalCursor:
        def __init__(self):
            self._next_fetchone = deque([None, (123,)])

        def execute(self, sql, params=None):
            if "SELECT id FROM statements" in sql:
                calls["select"] += 1
            elif sql.strip().startswith("INSERT INTO statements"):
                calls["insert"] += 1
            else:
                raise AssertionError(sql)

        def fetchone(self):
            return self._next_fetchone.popleft()

    cur = MinimalCursor()
    monkeypatch.setattr(derive_module, "_get_table_columns", lambda _cur, _table: {"normalized_text"})

    engine = DerivationEngine("postgresql://test", "redis://test")
    sid = engine._upsert_statement(cur, system_id=7, text="p", normalized="p")

    assert sid == 123
    assert calls == {"select": 1, "insert": 1}


def test_insert_proof_minimal_columns(monkeypatch):
    monkeypatch.setattr(derive_module, "_get_table_columns", lambda _cur, _table: set())
    engine = DerivationEngine("postgresql://test", "redis://test")

    captured = []

    class DummyCursor:
        def execute(self, sql, params=None):
            captured.append(sql.strip())

        def fetchone(self):
            return (88,)

    proof_id = engine._insert_proof(DummyCursor(), statement_id=5, method="mp")

    assert proof_id == 88
    assert captured and captured[0].startswith("INSERT INTO proofs")


def test_normalize_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "backend.logic.canon", types.SimpleNamespace(normalize=lambda _: (_ for _ in ()).throw(RuntimeError("nope"))))
    engine = DerivationEngine("postgresql://test", "redis://test")

    normalized = engine._normalize("p -> q")
    assert normalized.replace(" ", "") == "p->q"


def test_get_or_create_system_id_existing(monkeypatch):
    monkeypatch.setattr(derive_module, "_get_table_columns", lambda *a, **k: {"id", "name"})

    class Cursor:
        def __init__(self):
            self._calls = deque([(1,)])

        def execute(self, sql, params=None):
            pass

        def fetchone(self):
            return self._calls.popleft()

    cur = Cursor()
    assert derive_module._get_or_create_system_id(cur, "pl") == 1
