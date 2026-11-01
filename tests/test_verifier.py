from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest
import requests

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from agent.spec_generator import build_api_contract
from agent.verifier import _read_error_logs, _read_port, verify_generated_app


def test_read_port_and_logs(tmp_path):
    port_file = tmp_path / "port.txt"
    port_file.write_text("5001", encoding="utf-8")
    (tmp_path / "stderr.log").write_text("Traceback ...\nValueError: bad", encoding="utf-8")

    assert _read_port(tmp_path) == 5001
    assert "ValueError: bad" in _read_error_logs(tmp_path)


def test_verify_generated_app_handles_timeout(tmp_path):
    (tmp_path / "port.txt").write_text("9999", encoding="utf-8")

    result = verify_generated_app(tmp_path, routes=["/"], timeout_seconds=1, request_timeout_seconds=0.1)

    assert result["status"] in {"timed_out", "unknown"}
    assert result["error"] is not None


@pytest.mark.parametrize("failure", [requests.ConnectionError("connection refused"), requests.Timeout("request timed out")])
def test_verifier_retries_transient_api_connection_failures(tmp_path, monkeypatch, failure):
    import agent.verifier as verifier

    (tmp_path / "port.txt").write_text("5001", encoding="utf-8")
    calls = []

    def get(url, timeout):
        calls.append(url)
        if url.endswith("/"):
            return SimpleNamespace(ok=True, status_code=200, text="ready")
        if sum("/api/items" in item for item in calls) == 1:
            raise failure
        return SimpleNamespace(ok=True, status_code=200, text="[]")

    monkeypatch.setattr(verifier.requests, "get", get)
    result = verify_generated_app(
        tmp_path, api_contract={"endpoints": [{"method": "GET", "path": "/api/items"}]}, timeout_seconds=1,
    )

    assert result["status"] == "ok"
    assert any(item.get("error_type") in {"ConnectionError", "Timeout"} for item in result["responses"])
    assert calls.count("http://127.0.0.1:5001/api/items") == 2


def test_verifier_does_not_accept_unsuccessful_route_response(tmp_path, monkeypatch):
    import agent.verifier as verifier

    (tmp_path / "port.txt").write_text("5001", encoding="utf-8")
    monkeypatch.setattr(verifier.requests, "get", lambda url, timeout: SimpleNamespace(
        ok=url.endswith("/"), status_code=200 if url.endswith("/") else 404, text="missing",
    ))

    result = verify_generated_app(
        tmp_path, api_contract={"endpoints": [{"method": "GET", "path": "/api/items"}]}, timeout_seconds=1,
    )

    assert result["status"] == "failed"
    assert result["error"] == "Route /api/items returned HTTP 404."
    assert result["responses"][-1]["status_code"] == 404


@pytest.mark.parametrize("exit_code", [0, 3])
def test_verifier_reports_early_process_exit(tmp_path, monkeypatch, exit_code):
    import agent.verifier as verifier

    (tmp_path / "process.json").write_text('{"pid": 321}', encoding="utf-8")
    monkeypatch.setitem(verifier.PROCESS_HANDLES, 321, SimpleNamespace(poll=lambda: exit_code))
    monkeypatch.setattr(verifier.requests, "get", lambda *_args, **_kwargs: pytest.fail("no request after process exit"))

    result = verify_generated_app(tmp_path, routes=["/"], timeout_seconds=1)

    assert result["status"] == "failed"
    assert result["process_exit_code"] == exit_code
    assert f"status {exit_code}" in result["error"]


def test_verifier_times_out_when_live_process_never_becomes_ready(tmp_path, monkeypatch):
    import agent.verifier as verifier

    (tmp_path / "port.txt").write_text("5001", encoding="utf-8")
    (tmp_path / "process.json").write_text('{"pid": 321}', encoding="utf-8")
    monkeypatch.setitem(verifier.PROCESS_HANDLES, 321, SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(verifier.requests, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        requests.ConnectionError("connection refused")))

    result = verify_generated_app(tmp_path, routes=["/"], timeout_seconds=0.3, request_timeout_seconds=0.05)

    assert result["status"] == "timed_out"
    assert "connection refused" in result["error"]
    assert result["responses"]


def test_finance_contract_verifies_create_update_delete_and_cleans_row(tmp_path):
    database = tmp_path / "finance.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE expenses (id INTEGER PRIMARY KEY, description TEXT NOT NULL, amount REAL NOT NULL, "
            "category TEXT NOT NULL, spent_on TEXT NOT NULL)"
        )

    service = Flask("finance_verifier_fixture")

    @service.get("/")
    def home():
        return "ready"

    @service.route("/api/expenses", methods=["GET", "POST"])
    def expenses():
        with sqlite3.connect(database) as connection:
            if request.method == "POST":
                item = request.get_json()
                cursor = connection.execute(
                    "INSERT INTO expenses(description, amount, category, spent_on) VALUES (?, ?, ?, ?)",
                    (item["description"], item["amount"], item["category"], item["spent_on"]),
                )
                return jsonify({"id": cursor.lastrowid, **item}), 201
            rows = connection.execute(
                "SELECT id, description, amount, category, spent_on FROM expenses ORDER BY id"
            ).fetchall()
        return jsonify({"items": [dict(zip(("id", "description", "amount", "category", "spent_on"), row)) for row in rows]})

    @service.route("/api/expenses/<int:item_id>", methods=["GET", "PUT", "DELETE"])
    def expense(item_id):
        with sqlite3.connect(database) as connection:
            if request.method == "DELETE":
                cursor = connection.execute("DELETE FROM expenses WHERE id = ?", (item_id,))
                return ("", 204) if cursor.rowcount else (jsonify({"error": "missing"}), 404)
            if request.method == "PUT":
                item = request.get_json()
                cursor = connection.execute(
                    "UPDATE expenses SET description = ?, amount = ?, category = ?, spent_on = ? WHERE id = ?",
                    (item["description"], item["amount"], item["category"], item["spent_on"], item_id),
                )
                if not cursor.rowcount:
                    return jsonify({"error": "missing"}), 404
            row = connection.execute(
                "SELECT id, description, amount, category, spent_on FROM expenses WHERE id = ?", (item_id,)
            ).fetchone()
        if row is None:
            return jsonify({"error": "missing"}), 404
        return jsonify(dict(zip(("id", "description", "amount", "category", "spent_on"), row)))

    server = make_server("127.0.0.1", 0, service)
    (tmp_path / "port.txt").write_text(str(server.server_port), encoding="utf-8")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    contract = build_api_contract({
        "entities": [{"name": "Expenses", "fields": [
            {"name": "description", "type": "string"},
            {"name": "amount", "type": "number"},
            {"name": "category", "type": "select", "options": ["Food", "Transport"]},
            {"name": "spent_on", "type": "date"},
        ]}],
        "routes_needed": ["GET /api/expenses"],
    })
    try:
        result = verify_generated_app(tmp_path, api_contract=contract)
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert result["status"] == "ok", result
    assert result["records_persisted"] is True
    assert {item["operation"] for item in result["workflow_checks"]} == {
        "create_and_persist", "update", "read_after_update", "delete", "read_after_delete",
    }
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM expenses").fetchone()[0] == 0


def test_verifier_does_not_treat_response_echo_as_sqlite_persistence(tmp_path):
    service = Flask("verifier_no_persistence_fixture")

    @service.get("/")
    def home():
        return "ready"

    @service.route("/api/items", methods=["GET", "POST"])
    def items():
        if request.method == "POST":
            return jsonify({"id": 1, "name": "sample"}), 201
        return jsonify({"items": [{"id": 1, "name": "sample"}]})

    server = make_server("127.0.0.1", 0, service)
    (tmp_path / "port.txt").write_text(str(server.server_port), encoding="utf-8")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = verify_generated_app(tmp_path, api_contract={"endpoints": [
            {"method": "GET", "path": "/api/items", "db_table": "items"},
            {"method": "POST", "path": "/api/items", "request_json": {"name": "sample"}, "db_table": "items"},
        ]})
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert result["status"] == "unknown"
    assert result["records_persisted"] is False
    assert "not found in the generated SQLite database" in result["error"]
