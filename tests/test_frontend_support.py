from app import JOB_STORE, app


def test_history_counter_counts_successful_jobs(monkeypatch):
    before = app.test_client().get("/history").get_json()["apps_forged"]
    monkeypatch.setitem(JOB_STORE, "history-counter-test", {"status": "success"})

    response = app.test_client().get("/history")

    assert response.status_code == 200
    assert response.get_json()["apps_forged"] == before + 1


def test_stored_replay_remains_labeled_after_job_restore(monkeypatch, tmp_path):
    from pathlib import Path
    import app as backend_app

    jobs_dir = tmp_path / "jobs"
    projects_dir = tmp_path / "generated_projects"
    project_dir = projects_dir / "replay_task_tracker"
    project_dir.mkdir(parents=True)
    monkeypatch.setattr(backend_app, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(backend_app, "GENERATED_PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(backend_app, "JOB_STORE", {})
    monkeypatch.setattr(backend_app, "_pid_is_running", lambda _pid: False)
    replay_job = {
        "status": "success", "current_step": "8_complete", "replay": True,
        "spec": {"app_name": "Task Showcase"}, "plan": [], "prompt": "task planner",
        "execution": {"project_dir": str(project_dir), "port": 5100},
        "result": {"success": True, "replay": True, "label": "Prebuilt showcase replay: Task Tracker"},
        "verification": {"status": "ok"}, "generated_files": [], "versions": [],
    }
    backend_app._persist_job("persisted-replay", replay_job)
    backend_app._restore_jobs_and_cleanup()

    client = app.test_client()
    status = client.get("/status/persisted-replay").get_json()
    history = client.get("/history").get_json()

    assert status["replay"] is True
    assert status["generation_outcome"] == "showcase_replay"
    assert history["apps_forged"] == 0
    assert history["jobs"][0]["replay"] is True
    assert "Prebuilt showcase replay" in history["jobs"][0]["label"]


def test_frontend_and_offline_qr_assets_are_served_locally():
    client = app.test_client()

    assert client.get("/").status_code == 200
    assert client.get("/style.css").status_code == 200
    assert client.get("/script.js").status_code == 200
    assert b"Verification results" in client.get("/").data
    assert b"API checks passed" in client.get("/").data
    assert b"Try offline replay" in client.get("/").data
    script = client.get("/script.js").data
    assert b"offline-replay-btn" in script
    assert b"consecutivePollFailures >= 10" in script
    assert b"form.requestSubmit()" in script
    assert b"Stored showcase replay" in script
    assert b"Build request failed" in script
    qr = client.get("/qr.js")
    assert qr.status_code == 200
    assert b"offline QR encoder" in qr.data
