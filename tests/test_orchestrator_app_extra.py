import asyncio
import types

import pytest

from backend.orchestrator import app as app_module


class PatternCursor:
    def __init__(self, rules):
        self.rules = list(rules)
        self.pending = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        for predicate, responder in self.rules:
            if predicate(normalized, params):
                self.pending = responder()
                break
        else:
            raise AssertionError(f"Unexpected SQL: {normalized} params={params}")

    def fetchone(self):
        if not self.pending:
            raise AssertionError("fetchone without pending result")
        kind, value = self.pending
        assert kind == "one", f"Expected one, got {kind}"
        self.pending = None
        return value

    def fetchall(self):
        if not self.pending:
            raise AssertionError("fetchall without pending result")
        kind, value = self.pending
        assert kind == "all", f"Expected all, got {kind}"
        self.pending = None
        return value


class PatternConnection:
    def __init__(self, rules):
        self.cursor_obj = PatternCursor(rules)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj


def _fake_template(name, context):
    return context


def test_ui_dashboard_populates_metrics(monkeypatch):
    rules = [
        (lambda s, p: "information_schema.columns" in s and "proofs" in s,
         lambda: ("all", [("success", "boolean"), ("created_at", "timestamp"), ("status", "text")])),
        (lambda s, p: "information_schema.columns" in s and "statements" in s,
         lambda: ("all", [("text",), ("normalized_text",), ("hash",), ("created_at",)])),
        (lambda s, p: "SELECT COUNT(*) FROM proofs WHERE" in s and "interval" not in s,
         lambda: ("one", (7,))),
        (lambda s, p: "interval '300 seconds'" in s, lambda: ("one", (210,))),
        (lambda s, p: "COALESCE(MAX(block_number)" in s, lambda: ("one", (4,))),
        (lambda s, p: "SELECT merkle_root FROM blocks" in s, lambda: ("one", ("deadbeef",))),
        (lambda s, p: "SELECT policy_hash FROM policy_settings" in s, lambda: ("one", ("policyhash",))),
        (lambda s, p: "SELECT value FROM policy_settings" in s, lambda: ("one", None)),
        (lambda s, p: "SELECT text, normalized_text, hash FROM statements" in s,
         lambda: ("all", [("Foo", "Foo", "hash1"), (None, "Bar", "hash2")])),
    ]

    fake_conn = PatternConnection(rules)
    monkeypatch.setattr(app_module, "psycopg", types.SimpleNamespace(connect=lambda *a, **k: fake_conn))
    monkeypatch.setattr(app_module.templates, "TemplateResponse", _fake_template)
    monkeypatch.setattr(app_module, "_get_redis", lambda: types.SimpleNamespace(llen=lambda _: 3))

    ctx = app_module.ui_dashboard(request=types.SimpleNamespace())

    assert ctx["metrics"]["proofs_success"] == 7
    assert ctx["metrics"]["proofs_per_sec"] == pytest.approx(210 / 300)
    assert ctx["metrics"]["blocks_height"] == 4
    assert ctx["metrics"]["merkle"] == "deadbeef"
    assert ctx["metrics"]["policy_hash"] == "policyhash"
    assert [item["display"] for item in ctx["recent"]] == ["Foo", "Bar"]


def test_proof_success_predicate_falls_back_to_status():
    class StatusCursor:
        def execute(self, sql, params=None):
            pass

        def fetchall(self):
            return [("status", "character varying")]

    where, cols = app_module._proof_success_predicate(StatusCursor())
    assert where == "LOWER(status) IN ('success','ok','passed')"
    assert "status" in cols


def test_recent_statements_handles_missing_hash():
    class HashlessCursor:
        calls = 0

        def execute(self, sql, params=None):
            HashlessCursor.calls += 1

        def fetchall(self):
            if HashlessCursor.calls == 1:
                return [("text",), ("normalized_text",)]
            return [("foo", "bar")]

    assert app_module._recent_statements(HashlessCursor(), limit=5) == []

    class StatementCursor:
        calls = 0

        def execute(self, sql, params=None):
            StatementCursor.calls += 1

        def fetchall(self):
            if StatementCursor.calls == 1:
                return [("statement",), ("normalized",), ("canonical_hash",), ("created_at",)]
            return [("S", "N", "h1"), (None, "N2", "h2")]

    rows = app_module._recent_statements(StatementCursor(), limit=2)
    assert rows[0]["display"] == "S"
    assert rows[1]["display"] == "N2"


def test_metrics_collects_counts(monkeypatch):
    rules = [
        (lambda s, p: "COALESCE(MAX(block_number)" in s, lambda: ("one", (5,))),
        (lambda s, p: "SELECT COUNT(*) FROM blocks" in s, lambda: ("one", (2,))),
        (lambda s, p: "SELECT COUNT(*) FROM statements" in s, lambda: ("one", (42,))),
        (lambda s, p: "COALESCE(MAX(derivation_depth)" in s, lambda: ("one", (3,))),
        (lambda s, p: "SELECT COUNT(*) FROM proofs" in s and "WHERE" not in s, lambda: ("one", (10,))),
        (lambda s, p: "information_schema.columns" in s and "proofs" in s,
         lambda: ("all", [("success", "boolean")])) ,
        (lambda s, p: "SELECT COUNT(*) FROM proofs WHERE success" in s, lambda: ("one", (6,))),
        (lambda s, p: "SELECT policy_hash FROM policy_settings" in s, lambda: ("one", ("phash",))),
        (lambda s, p: "SELECT value FROM policy_settings" in s, lambda: ("one", None)),
        (lambda s, p: "SELECT merkle_root FROM blocks" in s, lambda: ("one", ("dead",))),
    ]

    fake_conn = PatternConnection(rules)
    result = app_module.metrics(conn=fake_conn)

    assert result["proofs"]["success"] == 6
    assert result["proofs"]["failure"] == 4
    assert result["block_count"] == 2
    assert result["max_depth"] == 3
    assert result["success_rate"] == pytest.approx(0.6)


def test_lifespan_checks_db(monkeypatch):
    connect_calls = []

    def fake_connect(url):
        connect_calls.append(url)
        class _Conn:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def cursor(self):
                return types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda *a: False, execute=lambda *_: None)
        return _Conn()

    monkeypatch.setattr(app_module, "psycopg", types.SimpleNamespace(connect=fake_connect))
    monkeypatch.setenv("DISABLE_DB_STARTUP", "0")

    async def run_lifespan():
        async with app_module.lifespan(app_module.app):
            assert connect_calls

    asyncio.run(run_lifespan())


def test_get_db_connection_errors(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mock://testing")
    with pytest.raises(app_module.HTTPException):
        next(app_module.get_db_connection())


def test_health_returns_status():
    status = app_module.health()
    assert status["status"] == "healthy"
