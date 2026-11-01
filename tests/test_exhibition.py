import ast
import json
from pathlib import Path

from agent.executor import start_generated_app, stop_generated_app
import agent.code_generator as code_generator
import agent.gemini_client as gemini_client
import agent.llm_client as llm_client
import agent.showcase as showcase_module
import agent.spec_generator as spec_generator
import agent.task_planner as task_planner
from agent.showcase import GENERATED_FILENAMES, contract_for_showcase, load_showcases, match_showcase, materialize_showcase
from agent.verifier import verify_generated_app
import app as backend_app
from app import app


def test_showcase_catalog_has_six_locally_materializable_apps(tmp_path):
    catalog = load_showcases()

    assert len(catalog) == 6
    assert {item["id"] for item in catalog} == {
        "task-tracker", "fitness-log", "expense-splitter", "attendance", "inventory", "resume-builder",
    }
    for showcase in catalog:
        files = materialize_showcase(showcase, tmp_path / showcase["id"])
        assert [entry["filename"] for entry in files] == GENERATED_FILENAMES
        ast.parse((tmp_path / showcase["id"] / "app.py").read_text(encoding="utf-8"))
        assert "127.0.0.1" in (tmp_path / showcase["id"] / "app.py").read_text(encoding="utf-8")


def test_showcase_fuzzy_match_selects_local_replay():
    assert match_showcase("Please make me a workout tracker")["id"] == "fitness-log"
    assert match_showcase("totally unrelated application concept") is None


def test_normal_prompt_does_not_start_with_a_fixed_replay():
    assert backend_app._select_initial_showcase(force_demo=False) is None
    assert backend_app._select_initial_showcase(force_demo=True)["id"] == "task-tracker"
    assert backend_app._select_initial_showcase(True, "A student workout tracker")["id"] == "fitness-log"


def test_missing_or_invalid_showcase_catalog_entries_are_ignored(tmp_path):
    missing = tmp_path / "missing-catalog.json"
    assert load_showcases(missing) == []

    valid = load_showcases()[0]
    catalog = tmp_path / "mixed-catalog.json"
    catalog.write_text(json.dumps([None, {"id": "../outside", "name": "Invalid"}, valid]), encoding="utf-8")
    assert [item["id"] for item in load_showcases(catalog)] == [valid["id"]]


def test_materialize_missing_runtime_files_fails_with_clear_message(monkeypatch, tmp_path):
    showcase = load_showcases()[0]
    monkeypatch.setattr(showcase_module, "RUNTIME_DIR", tmp_path / "missing-runtime")

    try:
        materialize_showcase(showcase, tmp_path / "project")
    except FileNotFoundError as exc:
        assert "runtime is missing required files" in str(exc)
    else:
        raise AssertionError("A missing runtime should fail clearly instead of producing a partial project")


def test_demo_endpoint_rejects_unavailable_catalog_without_queueing(monkeypatch):
    monkeypatch.setattr(backend_app, "load_showcases", lambda: [])

    response = app.test_client().post("/generate", json={"prompt": "task tracker", "demo_mode": True})

    assert response.status_code == 503
    assert "No valid offline showcase" in response.get_json()["error"]


def test_task_tracker_replay_has_real_task_features_and_contract(tmp_path):
    showcase = next(item for item in load_showcases() if item["id"] == "task-tracker")
    files = materialize_showcase(showcase, tmp_path / "task-tracker")
    content = {item["filename"]: item["content"] for item in files}

    assert "priority" in content["profile.json"] and "due_date" in content["profile.json"]
    assert 'methods=["GET", "PUT", "DELETE"]' in content["app.py"]
    assert "id=\"navigation\"" in content["index.html"]
    assert "page-filter" in content["script.js"]
    assert any(endpoint["path"] == "/api/tasks" for endpoint in contract_for_showcase(showcase)["endpoints"])


def test_task_tracker_replay_passes_live_verification(tmp_path):
    showcase = next(item for item in load_showcases() if item["id"] == "task-tracker")
    project_dir = tmp_path / "task-tracker"
    materialize_showcase(showcase, project_dir)
    start_generated_app(project_dir)
    try:
        result = verify_generated_app(project_dir, api_contract=contract_for_showcase(showcase))
        assert result["status"] == "ok", result
        assert result["records_persisted"] is True
        assert result["endpoints_verified"] >= 5
        assert {check["operation"] for check in result["workflow_checks"]} >= {
            "create_and_persist", "update", "read_after_update", "delete", "read_after_delete",
        }
    finally:
        stop_generated_app(project_dir)


def test_showcase_server_starts_when_parent_was_started_with_flask_cli(monkeypatch, tmp_path):
    showcase = next(item for item in load_showcases() if item["id"] == "fitness-log")
    project_dir = tmp_path / "fitness-log"
    materialize_showcase(showcase, project_dir)
    monkeypatch.setenv("FLASK_RUN_FROM_CLI", "true")
    execution = start_generated_app(project_dir)
    try:
        result = verify_generated_app(project_dir, api_contract=contract_for_showcase(showcase))
        assert execution["process"].poll() is None
        assert result["status"] == "ok", result
    finally:
        stop_generated_app(project_dir)


def test_demo_mode_runs_task_tracker_without_any_llm_calls_and_serves_persistent_preview(monkeypatch, tmp_path):
    prompt = "A task tracker with due dates and priority levels"
    calls = []

    def forbidden_llm(*_args, **_kwargs):
        calls.append("llm")
        raise AssertionError("Demo replay must not call an LLM client")

    class InlineThread:
        def __init__(self, target, args=(), daemon=None):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(backend_app, "JOB_STORE", {})
    monkeypatch.setattr(backend_app, "GENERATED_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(backend_app, "MANAGED_PROJECT_DIRS", set())
    monkeypatch.setattr(backend_app, "_persist_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_app, "_make_room_for_app", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_app.threading, "Thread", InlineThread)
    monkeypatch.setattr(backend_app, "generate_spec_and_plan", forbidden_llm)
    monkeypatch.setattr(backend_app, "generate_spec", forbidden_llm)
    monkeypatch.setattr(backend_app, "generate_file_for_component", forbidden_llm)
    monkeypatch.setattr(backend_app, "generate_with_backoff", forbidden_llm)
    monkeypatch.setattr(llm_client, "generate_with_backoff", forbidden_llm)
    monkeypatch.setattr(spec_generator, "generate_with_backoff", forbidden_llm)
    monkeypatch.setattr(code_generator, "generate_with_backoff", forbidden_llm)
    monkeypatch.setattr(task_planner, "generate_with_backoff", forbidden_llm)
    monkeypatch.setattr(gemini_client, "generate_with_backoff", forbidden_llm)

    client = app.test_client()
    queued = client.post("/generate", json={"prompt": prompt, "demo_mode": True})
    assert queued.status_code == 200
    job_id = queued.get_json()["job_id"]
    job = backend_app.JOB_STORE[job_id]

    assert job["status"] == "success", job.get("result")
    assert job["replay"] is True
    assert job["spec"]["app_name"] == "Task & Project Manager"
    assert job["result"]["label"].startswith("Prebuilt showcase replay:")
    assert any("No AI generation" in line for line in job["logs"])
    assert calls == []

    job_status = client.get(f"/status/{job_id}").get_json()
    assert {item["filename"] for item in job_status["generated_files"]} == set(GENERATED_FILENAMES)
    frontend_script = client.get("/script.js").data
    assert b"function renderFiles" in frontend_script
    assert b"previewFrame.src = previewUrl" in frontend_script

    preview = client.get(f"/preview/{job_id}/")
    assert preview.status_code == 200
    assert b"id=\"content\"" in preview.data
    profile = client.get(f"/preview/{job_id}/api/profile")
    assert profile.status_code == 200
    assert profile.get_json()["name"] == "Task & Project Manager"

    created = client.post(f"/preview/{job_id}/api/tasks", json={
        "title": "Expo follow-up", "project": "College Expo", "status": "Planned",
        "priority": "High", "due_date": "2026-10-20",
    })
    assert created.status_code == 201
    task_id = created.get_json()["id"]
    updated = client.put(f"/preview/{job_id}/api/tasks/{task_id}", json={"status": "Complete"})
    assert updated.status_code == 200
    tasks = client.get(f"/preview/{job_id}/api/tasks").get_json()["items"]
    assert any(task["title"] == "Expo follow-up" and task["priority"] == "High" and task["status"] == "Complete" for task in tasks)
    assert (Path(job["execution"]["project_dir"]) / "showcase.db").is_file()

    stop_generated_app(Path(job["execution"]["project_dir"]))


def test_health_page_is_served_locally():
    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert b"System health" in response.data


def test_windows_process_check_uses_tasklist(monkeypatch):
    class Result:
        stdout = '"python.exe","4321","Console","1","12,000 K"'

    monkeypatch.setattr(backend_app.os, "name", "nt")
    monkeypatch.setattr(backend_app.subprocess, "run", lambda *args, **kwargs: Result())

    assert backend_app._pid_is_running(4321) is True


def test_owned_windows_process_uses_popen_handle(monkeypatch):
    class RunningProcess:
        def poll(self):
            return None

    monkeypatch.setattr(backend_app.os, "name", "nt")
    monkeypatch.setitem(backend_app.PROCESS_HANDLES, 4321, RunningProcess())
    monkeypatch.setattr(backend_app.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tasklist should not be needed")))

    assert backend_app._pid_is_running(4321) is True


def test_full_app_pool_closes_oldest_completed_preview(monkeypatch, tmp_path):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "process.json").write_text('{"pid": 111, "started_at": 100}', encoding="utf-8")
    (new_dir / "process.json").write_text('{"pid": 222, "started_at": 200}', encoding="utf-8")
    jobs = {
        "old-job": {"status": "success", "spec": {"app_name": "Old preview"}, "execution": {"project_dir": str(old_dir)}},
        "new-job": {"status": "success", "spec": {"app_name": "New preview"}, "execution": {"project_dir": str(new_dir)}},
    }
    stopped = []
    messages = []
    monkeypatch.setattr(backend_app, "JOB_STORE", jobs)
    monkeypatch.setattr(backend_app, "MAX_CONCURRENT_APPS", 1)
    monkeypatch.setattr(backend_app, "_active_generated_count", lambda: 1 if not stopped else 0)
    monkeypatch.setattr(backend_app, "_pid_is_running", lambda pid: True)
    monkeypatch.setattr(backend_app, "stop_generated_app", lambda path: stopped.append(Path(path)))
    monkeypatch.setattr(backend_app, "_persist_job", lambda job_id, job: None)

    backend_app._make_room_for_app(messages.append)

    assert stopped == [old_dir]
    assert jobs["old-job"]["preview_closed"] is True
    assert "reopen it from build history" in messages[0]
