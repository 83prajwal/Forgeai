import io
import json
import sqlite3
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from agent.spec_generator import build_api_contract
from agent.verifier import verify_generated_app
from app import JOB_STORE, _prepare_generated_files, _select_relevant_file, app as forge_app, validate_generated_project_files


def test_api_contract_contains_endpoint_shapes_and_database_columns():
    contract = build_api_contract({
        "entities": [{"name": "Task", "fields": [{"name": "title", "type": "string"}, {"name": "done", "type": "boolean"}]}],
        "routes_needed": ["GET /tasks"],
    })

    post = next(item for item in contract["endpoints"] if item["method"] == "POST")
    assert post["path"] == "/tasks"
    assert post["request_json"] == {"title": "sample", "done": True}
    assert post["db_table"] == "task"
    assert post["db_columns"] == ["id", "title", "done"]
    assert {("GET", "/tasks/<int:item_id>"), ("PUT", "/tasks/<int:item_id>"), ("DELETE", "/tasks/<int:item_id>")} <= {
        (item["method"], item["path"]) for item in contract["endpoints"]
    }
    assert all({"method", "path", "request_json", "response_json", "db_table", "db_columns"} <= item.keys() for item in contract["endpoints"])
    assert not any(item["path"] == "/" for item in contract["endpoints"])


def test_generated_entrypoint_reserves_root_for_html_ui():
    from app import _deterministic_file

    source = _deterministic_file("app.py")
    assert source.index('def index():') < source.index('app.register_blueprint(bp)')


def test_generated_project_preflight_checks_required_assets_and_contract_routes():
    files = [
        {"filename": "requirements.txt", "content": "Flask"},
        {"filename": "models.py", "content": ""},
        {"filename": "app.py", "content": ""},
        {"filename": "routes.py", "content": 'from flask import Blueprint\nbp = Blueprint("api", __name__)\n@bp.route("/api/tasks", methods=["GET", "POST"])\ndef tasks(): pass\n'},
        {"filename": "index.html", "content": '<link rel="stylesheet" href="style.css"><script src="script.js"></script>'},
        {"filename": "style.css", "content": ""},
        {"filename": "script.js", "content": "fetch('/api/tasks')"},
    ]
    contract = {"endpoints": [
        {"method": "GET", "path": "/api/tasks"},
        {"method": "POST", "path": "/api/tasks"},
        {"method": "DELETE", "path": "/api/tasks/<int:item_id>"},
    ]}

    issues = validate_generated_project_files(files, contract)

    assert len(issues) == 1
    assert "DELETE /api/tasks/<int:item_id>" in issues[0]


def test_ast_gate_repairs_invalid_python_before_project_write(tmp_path, monkeypatch):
    def repair(project_dir, filename, _error):
        (Path(project_dir) / filename).write_text("value = 1\n", encoding="utf-8")
        return filename

    monkeypatch.setattr("app.repair_failed_file", repair)
    generated = _prepare_generated_files([{"filename": "models.py", "content": "def broken(:\n"}])

    assert generated[0]["content"] == "value = 1\n"
    assert not (tmp_path / "models.py").exists()


def test_repair_validates_filename_and_syntax_before_caching_or_writing(tmp_path, monkeypatch):
    import app as backend_app

    target = tmp_path / "routes.py"
    target.write_text("def broken(:\n", encoding="utf-8")
    cached = []
    responses = iter([
        SimpleNamespace(text=json.dumps({"filename": "models.py", "content": "value = 1\n"}), complete=True,
                       cache_validated=lambda: cached.append("wrong-file")),
        SimpleNamespace(text=json.dumps({"filename": "routes.py", "content": "def fixed():\n    return 1\n"}), complete=True,
                       cache_validated=lambda: cached.append("valid")),
    ])
    monkeypatch.setattr(backend_app, "generate_with_backoff", lambda *_args, **_kwargs: next(responses))

    backend_app.repair_failed_file(tmp_path, "routes.py", "invalid syntax")

    assert target.read_text(encoding="utf-8") == "def fixed():\n    return 1\n"
    assert cached == ["valid"]


def test_repair_selects_deepest_traceback_file(tmp_path):
    (tmp_path / "app.py").write_text("from routes import bp\n", encoding="utf-8")
    (tmp_path / "routes.py").write_text("raise RuntimeError('route')\n", encoding="utf-8")
    files = [
        {"filename": "app.py", "content": "from routes import bp\n"},
        {"filename": "routes.py", "content": "raise RuntimeError('route')\n"},
    ]
    traceback = (
        f'File "{tmp_path / "app.py"}", line 1, in <module>\n'
        f'File "{tmp_path / "routes.py"}", line 1, in endpoint\n'
        "RuntimeError: route"
    )

    assert _select_relevant_file(tmp_path, files, traceback) == "routes.py"


def test_verifier_posts_and_confirms_sqlite_record(tmp_path):
    database = tmp_path / "items.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")

    service = Flask("verifier_fixture")

    @service.get("/")
    def home():
        return "ready"

    @service.get("/api/items")
    def get_items():
        with sqlite3.connect(database) as connection:
            rows = connection.execute("SELECT id, name FROM items").fetchall()
        return jsonify([{"id": row[0], "name": row[1]} for row in rows])

    @service.post("/api/items")
    def post_item():
        name = request.get_json()["name"]
        with sqlite3.connect(database) as connection:
            cursor = connection.execute("INSERT INTO items(name) VALUES (?)", (name,))
            item_id = cursor.lastrowid
        return jsonify({"id": item_id, "name": name}), 201

    server = make_server("127.0.0.1", 0, service)
    (tmp_path / "port.txt").write_text(str(server.server_port), encoding="utf-8")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = verify_generated_app(tmp_path, api_contract={"endpoints": [
            {"method": "GET", "path": "/api/items"},
            {"method": "POST", "path": "/api/items", "request_json": {"name": "sample"}, "db_table": "items"},
        ]})
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert result["status"] == "ok"
    assert result["endpoints_verified"] == 2
    assert result["records_persisted"] is True


def test_download_route_creates_clean_runnable_zip(tmp_path, monkeypatch):
    project = tmp_path / "generated"
    project.mkdir()
    (project / "app.py").write_text("print('ok')", encoding="utf-8")
    (project / "requirements.txt").write_text("Flask\n", encoding="utf-8")
    (project / "server.log").write_text("log", encoding="utf-8")
    (project / "data.db").write_bytes(b"database")
    monkeypatch.setattr("app.GENERATED_PROJECTS_DIR", tmp_path)
    monkeypatch.setitem(JOB_STORE, "phase1-test", {"execution": {"project_dir": str(project)}})

    response = forge_app.test_client().get("/download/phase1-test")
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        names = set(archive.namelist())

    assert response.status_code == 200
    assert {"app.py", "requirements.txt", "README.md", "Dockerfile"} <= names
    assert not any(name.endswith((".log", ".db")) or name.endswith("process.json") for name in names)
