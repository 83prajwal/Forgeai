"""Representative live pipeline tests with provider output mocked locally."""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pytest
import requests

from agent import executor, llm_client
from agent.executor import stop_generated_app


def _application_files(spec: dict, prompt: str) -> dict[str, str]:
    entity = spec["entities"][0]
    table = re.sub(r"[^a-z0-9]+", "_", entity["name"].lower()).strip("_")
    fields = [field["name"] for field in entity["fields"]]
    types = {field["name"]: field.get("type", "string") for field in entity["fields"]}
    required = [field["name"] for field in entity["fields"] if field.get("required")]
    table_literal = repr(table)
    fields_literal = repr(fields)
    required_literal = repr(required)
    schema_sql = ", ".join(
        f'"{name}" {"REAL" if types[name] in {"number", "integer", "float"} else "TEXT"}'
        for name in fields
    )
    models = f'''import os
import sqlite3
from flask import current_app, g

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_app(app):
    app.config["DATABASE"] = os.path.join(app.root_path, "app.db")
    with sqlite3.connect(app.config["DATABASE"]) as db:
        db.execute('CREATE TABLE IF NOT EXISTS "{table}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {schema_sql})')
    app.teardown_appcontext(close_db)
'''
    routes = f'''from flask import Blueprint, jsonify, request
from models import get_db

bp = Blueprint("api", __name__)
TABLE = {table_literal}
FIELDS = {fields_literal}
REQUIRED = {required_literal}

@bp.route("/api/{table}", methods=["GET", "POST"])
def collection():
    db = get_db()
    if request.method == "GET":
        rows = db.execute(f'SELECT * FROM "{{TABLE}}" ORDER BY id DESC').fetchall()
        return jsonify({{"items": [dict(row) for row in rows]}})
    data = request.get_json(silent=True) or {{}}
    for name in REQUIRED:
        if not str(data.get(name, "")).strip():
            return jsonify({{"error": name + " is required"}}), 400
    names = [name for name in FIELDS if name in data]
    if not names:
        return jsonify({{"error": "At least one field is required"}}), 400
    cursor = db.execute(f'INSERT INTO "{{TABLE}}" ({{",".join(names)}}) VALUES ({{",".join("?" for _ in names)}})', [data[name] for name in names])
    db.commit()
    return jsonify({{"id": cursor.lastrowid}}), 201

@bp.route("/api/{table}/<int:item_id>", methods=["GET", "PUT", "DELETE"])
def item(item_id):
    db = get_db()
    if request.method == "GET":
        row = db.execute(f'SELECT * FROM "{{TABLE}}" WHERE id=?', (item_id,)).fetchone()
        return (jsonify(dict(row)), 200) if row else (jsonify({{"error": "Not found"}}), 404)
    if request.method == "DELETE":
        cursor = db.execute(f'DELETE FROM "{{TABLE}}" WHERE id=?', (item_id,))
        db.commit()
        return (jsonify({{"deleted": True}}), 200) if cursor.rowcount else (jsonify({{"error": "Not found"}}), 404)
    data = request.get_json(silent=True) or {{}}
    names = [name for name in FIELDS if name in data]
    if not names:
        return jsonify({{"error": "At least one field is required"}}), 400
    cursor = db.execute(f'UPDATE "{{TABLE}}" SET {{",".join(name + "=?" for name in names)}} WHERE id=?', [data[name] for name in names] + [item_id])
    db.commit()
    row = db.execute(f'SELECT * FROM "{{TABLE}}" WHERE id=?', (item_id,)).fetchone()
    return (jsonify(dict(row)), 200) if row else (jsonify({{"error": "Not found"}}), 404)
'''
    controls = []
    for field in entity["fields"]:
        kind = field.get("type", "text")
        label = field.get("label", field["name"])
        if kind == "select":
            options = "".join(f'<option>{option}</option>' for option in field.get("options", []))
            controls.append(f'<label>{label}<select name="{field["name"]}">{options}</select></label>')
        else:
            input_type = {"number": "number", "date": "date", "month": "month", "email": "email"}.get(kind, "text")
            required_attr = " required" if field.get("required") else ""
            controls.append(f'<label>{label}<input name="{field["name"]}" type="{input_type}"{required_attr}></label>')
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>' + spec["app_name"] + '</title><link rel="stylesheet" href="style.css"></head><body><main class="shell">'
        '<h1>' + spec["app_name"] + '</h1><p>' + prompt + '</p><form id="record-form">'
        + "".join(controls) + '<button type="submit">Save</button></form><p id="message" role="status"></p>'
        '<label>Search<input id="search" type="search"></label><ul id="items"></ul></main><script defer src="script.js"></script></body></html>'
    )
    css = '''*{box-sizing:border-box}body{margin:0;padding:32px;background:#f3f6f5;color:#192421;font:16px/1.5 system-ui,sans-serif}.shell{max-width:850px;margin:auto}form{display:flex;flex-wrap:wrap;gap:12px;padding:20px;background:white;border:1px solid #d9e1de;border-radius:14px}label{display:grid;gap:5px}input,select,button{font:inherit;padding:10px;border:1px solid #aab8b2;border-radius:8px}button{background:#087f68;color:white;cursor:pointer}li{padding:12px;margin:8px 0;background:white;border-radius:8px}@media(max-width:600px){body{padding:16px}form{display:grid}input,select,button{width:100%}}'''
    script = f'''const endpoint="/api/{table}";
const form=document.querySelector("#record-form"), list=document.querySelector("#items"), message=document.querySelector("#message"), search=document.querySelector("#search");
let rows=[];
async function refresh(){{const response=await fetch(endpoint);const body=await response.json();if(!response.ok)throw Error(body.error||"Could not load records");rows=body.items||[];render()}}
function render(){{const term=search.value.trim().toLowerCase();list.replaceChildren();const visible=rows.filter(row=>JSON.stringify(row).toLowerCase().includes(term));if(!visible.length){{const empty=document.createElement("li");empty.textContent="No records yet. Add the first one above.";list.append(empty);return}}for(const row of visible){{const item=document.createElement("li");item.textContent={json.dumps(" · ".join(fields))}.split(" · ").map(key=>row[key]??"").join(" · ");const edit=document.createElement("button");edit.type="button";edit.textContent="Edit";edit.onclick=()=>{{for(const [key,value] of Object.entries(row)){{const control=form.elements.namedItem(key);if(control)control.value=value}}form.dataset.editId=row.id}};const remove=document.createElement("button");remove.type="button";remove.textContent="Delete";remove.onclick=async()=>{{const response=await fetch(endpoint+"/"+row.id,{{method:"DELETE"}});if(!response.ok){{message.textContent="Could not delete this record.";return}}await refresh()}};item.append(" ",edit," ",remove);list.append(item)}}}}
form.onsubmit=async event=>{{event.preventDefault();const payload=Object.fromEntries(new FormData(form));const id=form.dataset.editId;const response=await fetch(endpoint+(id?"/"+id:""),{{method:id?"PUT":"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(payload)}});const body=await response.json();if(!response.ok){{message.textContent=body.error||"Please check the form.";return}}delete form.dataset.editId;form.reset();message.textContent=id?"Changes saved.":"Record added.";await refresh()}};
search.addEventListener("input",render);refresh().catch(error=>message.textContent=error.message);
'''
    return {
        "models.py": models,
        "routes.py": routes,
        "index.html": html,
        "style.css": css,
        "script.js": script,
    }


@pytest.mark.parametrize(
    ("prompt", "app_name", "entity", "fields"),
    [
        (
            "Build a task tracker with due dates and priority levels.",
            "Campus Task Tracker",
            "Tasks",
            [
                {"name": "title", "type": "string", "required": True},
                {"name": "priority", "type": "select", "options": ["High", "Normal", "Low"]},
                {"name": "due_date", "type": "date"},
            ],
        ),
        (
            "Build a personal finance manager to track expenses by amount, category, and date.",
            "Personal Finance Manager",
            "Expenses",
            [
                {"name": "description", "type": "string", "required": True},
                {"name": "amount", "type": "number", "required": True},
                {"name": "category", "type": "select", "options": ["Food", "Travel", "Books"]},
                {"name": "spent_on", "type": "date"},
            ],
        ),
    ],
    ids=["task-tracker", "personal-finance"],
)
def test_mocked_provider_generation_runs_and_verifies_domain_app(
    monkeypatch, tmp_path, prompt, app_name, entity, fields
):
    import app as backend_app

    spec = {
        "app_name": app_name,
        "purpose": prompt,
        "user_roles": [],
        "core_features": ["Create, edit, search, and delete records"],
        "entities": [{"name": entity, "fields": fields}],
        "relationships": [],
        "auth_required": False,
        "routes_needed": [],
        "pages_needed": ["Dashboard", entity],
        "pages": [{"name": entity, "purpose": "Manage the requested domain records", "widgets": ["Search", "Create", "Edit", "Delete"]}],
        "navigation": [],
        "dashboard_kpis": [],
        "database_requirements": ["Persist records in SQLite"],
        "validation_requirements": ["Validate required fields"],
        "error_handling_requirements": ["Show clear validation errors"],
        "responsive_requirements": ["Support narrow mobile screens"],
        "assumptions": ["Single local user"],
        "theme": {"accent_colour": "#16836e", "icon": "grid", "tone": "calm"},
    }
    contents = _application_files(spec, prompt)
    prompts = []
    job_id = "representative-" + entity.lower()
    backend_app.JOB_STORE[job_id] = {
        "status": "queued", "current_step": "queued", "logs": [], "result": None,
        "spec": None, "plan": None, "generated_files": [], "execution": None,
        "verification": None, "versions": [], "started_at": time.time(), "prompt": prompt,
        "force_demo": False,
    }
    monkeypatch.setattr(backend_app, "JOB_STORE", {job_id: backend_app.JOB_STORE[job_id]})
    monkeypatch.setattr(backend_app, "GENERATION_QUEUE", [])
    monkeypatch.setattr(backend_app, "MANAGED_PROJECT_DIRS", set())
    monkeypatch.setattr(backend_app, "_persist_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_app, "_make_room_for_app", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_app, "match_showcase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "GENERATED_PROJECTS_DIR", tmp_path / "generated")
    monkeypatch.setattr(executor, "install_requirements", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(llm_client, "LLM_CACHE_ENABLED", False)
    monkeypatch.setattr(llm_client, "_throttle", lambda *_args: 0)
    monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")

    def groq(response_prompt, _model, _max_tokens, json_mode):
        assert json_mode is True
        prompts.append(response_prompt)
        if "Create a concise application specification" in response_prompt:
            response = {"spec": spec, "steps": ["Build the domain workflow", "Verify persistence"]}
        else:
            match = re.search(r"Current component to generate:\s*([^\s]+)", response_prompt)
            assert match, "mocked codegen request should identify its output file"
            filename = match.group(1)
            if filename.startswith("page_"):
                content = "// Domain page component is merged into the app bundle.\n"
            else:
                content = contents[filename]
            response = {"filename": filename, "content": content}
        return json.dumps(response), object()

    monkeypatch.setattr(llm_client, "_groq", groq)
    try:
        backend_app.run_generation_job(job_id, prompt)
        job = backend_app.JOB_STORE[job_id]
        assert job["status"] == "success", job.get("result")
        assert not job.get("replay", False)
        assert job["verification"]["status"] == "ok"
        assert job["verification"]["verification"]["records_persisted"] is True
        assert {check["operation"] for check in job["verification"]["verification"]["workflow_checks"]} >= {
            "create_and_persist", "update", "read_after_update", "delete", "read_after_delete",
        }
        assert any(f"Domain profile: {'task management' if entity == 'Tasks' else 'personal finance'}" in p for p in prompts)
        assert len(prompts) == 7  # one spec/plan response plus existing file-generation responses
        assert sum("Create a concise application specification" in p for p in prompts) == 1
        assert (Path(job["execution"]["project_dir"]) / "app.db").is_file()
        base_url = f"http://127.0.0.1:{job['execution']['port']}"
        table = re.sub(r"[^a-z0-9]+", "_", entity.lower()).strip("_")
        invalid = requests.post(f"{base_url}/api/{table}", json={}, timeout=2)
        missing = requests.get(f"{base_url}/api/{table}/99999", timeout=2)
        assert invalid.status_code == 400
        assert "required" in invalid.json()["error"].lower()
        assert missing.status_code == 404
    finally:
        job = backend_app.JOB_STORE.get(job_id, {})
        execution = job.get("execution")
        if execution and execution.get("project_dir"):
            stop_generated_app(Path(execution["project_dir"]))
