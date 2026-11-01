from app import JOB_STORE, _fallback_affected_files, app


def test_modification_file_fallback_is_scoped_to_related_files():
    assert _fallback_affected_files("add dark mode colors") == ["index.html", "style.css", "script.js"]
    assert _fallback_affected_files("add a priority field") == ["models.py", "routes.py", "index.html", "script.js"]


def test_modify_endpoint_requires_an_existing_successful_build():
    response = app.test_client().post("/modify/not-a-job", json={"change": "add search"})

    assert response.status_code == 404
    assert "successful generated app" in response.get_json()["error"]


def test_history_gallery_only_lists_successful_projects():
    JOB_STORE["history-modification-task"] = {
        "status": "success",
        "parent_job_id": "parent-build",
    }

    response = app.test_client().get("/history").get_json()

    assert all(item["job_id"] != "history-modification-task" for item in response["jobs"])


def test_modification_planner_provider_failure_is_not_hidden_by_file_fallback(monkeypatch):
    import app as backend_app

    calls = []

    def provider_failure(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("Groq rejected GROQ_API_KEY (HTTP 401)")

    monkeypatch.setattr(backend_app, "generate_with_backoff", provider_failure)

    try:
        backend_app._plan_modification(
            "add search", {"app_name": "Tasks"},
            [{"filename": name, "content": ""} for name in backend_app.MODIFIABLE_FILES],
        )
    except RuntimeError as exc:
        assert "401" in str(exc)
    else:
        raise AssertionError("provider authentication failures must stop modification planning")

    assert len(calls) == 1


def test_malformed_modification_plan_uses_visible_deterministic_fallback(monkeypatch):
    import app as backend_app

    messages = []
    monkeypatch.setattr(backend_app, "generate_with_backoff", lambda *_args, **_kwargs: type(
        "Response", (), {"text": '{"files":[', "complete": True}
    )())

    files, revised = backend_app._plan_modification(
        "add search", {"app_name": "Tasks"},
        [{"filename": name, "content": ""} for name in backend_app.MODIFIABLE_FILES],
        on_model_switch=messages.append,
    )

    assert files == ["routes.py", "index.html", "script.js"]
    assert revised["app_name"] == "Tasks"
    assert any("malformed JSON" in message for message in messages)
