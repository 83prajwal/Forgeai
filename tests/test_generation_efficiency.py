"""Efficiency tests mock all provider requests and localize cache files to tmp_path."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time

import pytest

from agent import llm_client


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_client, "LLM_CACHE_ENABLED", True)
    monkeypatch.setattr(llm_client, "_cache_read", llm_client._cache_read)
    monkeypatch.setattr(llm_client, "_cache_write", llm_client._cache_write)

    def cache_file(provider, model, prompt, **parameters):
        digest = hashlib.sha256(repr((provider, model, prompt, sorted(parameters.items()))).encode()).hexdigest()
        return tmp_path / f"{digest}.json"

    monkeypatch.setattr(llm_client, "_cache_file", cache_file)
    monkeypatch.setattr(llm_client, "_throttle", lambda *args: 0)
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_: None)
    llm_client._cooldown.clear()
    llm_client.reset_call_stats()
    monkeypatch.setenv("GROQ_API_KEY", "mock-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini")


def test_identical_valid_request_hits_cache(isolated_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (calls.append(args) or '{"ok":true}', object()))

    first = llm_client.generate_with_backoff("same prompt", json_mode=True, max_tokens=100, cache_context={"file": "routes.py"})
    first.cache_validated()
    second = llm_client.generate_with_backoff("same prompt", json_mode=True, max_tokens=100, cache_context={"file": "routes.py"})

    assert first.text == second.text == '{"ok":true}'
    assert len(calls) == 1
    assert llm_client.get_call_metrics()["cache_hits"] == 1


def test_cache_key_changes_with_model_parameters_and_context(isolated_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (calls.append(args) or '{"ok":true}', object()))
    for kwargs in (
        {"preferred_model": "model-a", "max_tokens": 10, "cache_context": {"file": "a.py"}},
        {"preferred_model": "model-a", "max_tokens": 11, "cache_context": {"file": "a.py"}},
        {"preferred_model": "model-a", "max_tokens": 10, "cache_context": {"file": "b.py"}},
        {"preferred_model": "model-b", "max_tokens": 10, "cache_context": {"file": "a.py"}},
    ):
        llm_client.generate_with_backoff("prompt", json_mode=True, **kwargs).cache_validated()
    llm_client.generate_with_backoff("prompt", preferred_model="model-a", json_mode=False, max_tokens=10, cache_context={"file": "a.py"})
    llm_client.generate_with_backoff("changed prompt", preferred_model="model-a", json_mode=True, max_tokens=10, cache_context={"file": "a.py"}).cache_validated()

    assert len(calls) == 6


def test_local_token_guard_accepts_request_just_below_configured_budget(monkeypatch):
    llm_client._usage["groq"].clear()
    assert llm_client._throttle("groq", 10_000, 9_999) == 0


def test_local_token_guard_rejects_request_just_above_configured_budget():
    llm_client._usage["groq"].clear()
    with pytest.raises(llm_client.ProviderError, match="local per-minute safety budget"):
        llm_client._throttle("groq", 10_000, 10_001)


def test_local_groq_budget_override_is_configurable(monkeypatch):
    monkeypatch.setattr(llm_client, "GROQ_TOKENS_PER_MINUTE", "9000")
    monkeypatch.setattr(llm_client, "GROQ_TPM_70B", "10000")

    assert llm_client._token_budget("groq", "openai/gpt-oss-120b") == 9000


def test_invalid_or_provider_truncated_output_is_not_cached(isolated_cache, monkeypatch):
    calls = []

    class TruncatedResponse:
        def json(self):
            return {"choices": [{"finish_reason": "length"}]}

    monkeypatch.setattr(llm_client, "_groq", lambda *args: (calls.append(1) or '{"partial":', TruncatedResponse()))
    for _ in range(2):
        llm_client.generate_with_backoff("prompt", json_mode=True)
    assert len(calls) == 2


def test_concurrent_identical_requests_share_one_provider_call(isolated_cache, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    calls = []

    def groq(*args):
        with count_lock:
            calls.append(1)
        entered.set()
        assert release.wait(2)
        return ('{"shared":true}', object())

    monkeypatch.setattr(llm_client, "_groq", groq)
    results = []
    first = threading.Thread(target=lambda: results.append(llm_client.generate_with_backoff("duplicate")))
    second = threading.Thread(target=lambda: results.append(llm_client.generate_with_backoff("duplicate")))
    first.start()
    assert entered.wait(1)
    second.start()
    release.set()
    first.join(3)
    second.join(3)

    assert not first.is_alive() and not second.is_alive()
    assert len(calls) == 1
    assert len(results) == 2


def test_retry_count_is_hard_bounded(isolated_cache, monkeypatch):
    # Isolate the hard ceiling from the deployment's lower configured retry cap.
    monkeypatch.setattr(llm_client, "GROQ_MAX_RETRIES", "3")
    groq_calls, gemini_calls = [], []

    def unavailable(calls):
        def call(*args):
            calls.append(1)
            raise llm_client.ProviderError("provider", "model", "temporary unavailable", 503)
        return call

    monkeypatch.setattr(llm_client, "_groq", unavailable(groq_calls))
    monkeypatch.setattr(llm_client, "_gemini", unavailable(gemini_calls))
    with pytest.raises(RuntimeError):
        llm_client.generate_with_backoff("prompt", max_retries=100)

    assert len(groq_calls) == 4  # initial attempt plus the hard limit of three retries
    assert len(gemini_calls) == 1
    assert llm_client.get_call_metrics()["provider_attempts"] == 5


def test_rate_limit_cooldown_uses_gemini_cache_without_repeating_groq(isolated_cache, monkeypatch):
    groq_calls, gemini_calls = [], []

    def rate_limited(*args):
        groq_calls.append(1)
        raise llm_client.ProviderError("groq", "model", "daily quota exhausted", 429)

    monkeypatch.setattr(llm_client, "_groq", rate_limited)
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: (gemini_calls.append(1) or '{"ok":true}', object()))
    first = llm_client.generate_with_backoff("same prompt", json_mode=True)
    first.cache_validated()
    llm_client._cooldown.clear()  # Simulate a server restart before the repeated request.
    second = llm_client.generate_with_backoff("same prompt", json_mode=True)

    assert first.provider == second.provider == "gemini"
    assert len(groq_calls) == len(gemini_calls) == 1


def test_job_reuses_saved_spec_plan_and_completed_files(monkeypatch, tmp_path):
    import app as backend_app

    job_id = "efficiency-reuse-job"
    spec = {"app_name": "Test App", "entities": [], "api_contract": {}}
    files = [
        {"filename": "requirements.txt", "content": "old requirements"},
        {"filename": "models.py", "content": "# saved models"},
        {"filename": "routes.py", "content": "from flask import Blueprint\nbp = Blueprint('api', __name__)"},
        {"filename": "app.py", "content": "old app"},
        {"filename": "index.html", "content": '<link rel="stylesheet" href="style.css"><script src="script.js"></script><main>Saved</main>'},
        {"filename": "style.css", "content": "body{}"},
        {"filename": "script.js", "content": "window.saved=true;"},
    ]
    job = {"status": "queued", "current_step": "queued", "logs": [], "spec": spec,
           "plan": ["saved plan"], "generated_files": files, "execution": None,
           "versions": [], "started_at": 1.0, "prompt": "same request", "force_demo": False}
    backend_app.JOB_STORE[job_id] = job

    def forbidden(*args, **kwargs):
        pytest.fail("saved spec and generated files should avoid LLM calls")

    monkeypatch.setattr(backend_app, "generate_spec_and_plan", forbidden)
    monkeypatch.setattr(backend_app, "generate_file_for_component", forbidden)
    monkeypatch.setattr(backend_app, "_persist_job", lambda *args: None)
    monkeypatch.setattr(backend_app, "_make_room_for_app", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_app, "_prepare_generated_files", lambda value: value)
    monkeypatch.setattr(backend_app, "execute_generated_app", lambda *args: {"project_dir": str(tmp_path), "port": 49999})
    monkeypatch.setattr(backend_app, "self_correct_generated_app", lambda *args, **kwargs: {"status": "ok", "attempts_used": 1})

    backend_app.run_generation_job(job_id, "same request")

    assert job["status"] == "success"
    assert "Reusing completed spec and plan from this job" in job["logs"]
    assert any("Reused completed script.js" in message for message in job["logs"])


def test_mocked_groq_success_runs_normal_generation_pipeline(monkeypatch, tmp_path):
    import app as backend_app

    spec = {"app_name": "Mocked Pipeline", "entities": [], "pages": [], "routes_needed": []}
    file_contents = {
        "models.py": "# models\n",
        "routes.py": "from flask import Blueprint\nbp = Blueprint('api', __name__)\n",
        "index.html": '<!doctype html><html><head><link rel="stylesheet" href="style.css"></head><body>Mocked app<script src="script.js"></script></body></html>',
        "style.css": "body { color: black; }",
        "script.js": "console.log('ready');",
    }
    groq_calls = []

    def groq(prompt, model, max_tokens, json_mode):
        assert json_mode is True
        groq_calls.append(model)
        if "Create a concise application specification" in prompt:
            result = {"spec": spec, "steps": ["Build and verify"]}
        else:
            match = re.search(r"Current component to generate:\s*([^\s]+)", prompt)
            assert match, "codegen prompt should name its output component"
            filename = match.group(1)
            result = {"filename": filename, "content": file_contents[filename]}
        return json.dumps(result), object()

    monkeypatch.setattr(llm_client, "LLM_CACHE_ENABLED", False)
    monkeypatch.setattr(llm_client, "_groq", groq)
    monkeypatch.setattr(llm_client, "_throttle", lambda *args: 0)
    monkeypatch.setattr(backend_app, "JOB_STORE", {})
    monkeypatch.setattr(backend_app, "GENERATION_QUEUE", [])
    monkeypatch.setattr(backend_app, "MANAGED_PROJECT_DIRS", set())
    monkeypatch.setattr(backend_app, "_persist_job", lambda *args: None)
    monkeypatch.setattr(backend_app, "_make_room_for_app", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_app, "execute_generated_app", lambda *args: {"project_dir": str(tmp_path), "port": 49001})
    monkeypatch.setattr(backend_app, "self_correct_generated_app", lambda *args, **kwargs: {"status": "ok", "attempts_used": 1})
    job = {"status": "queued", "current_step": "queued", "logs": [], "result": None,
           "spec": None, "plan": None, "generated_files": [], "execution": None,
           "verification": None, "versions": [], "started_at": time.time(),
           "prompt": "make a minimal demo", "force_demo": False}
    backend_app.JOB_STORE["mocked-groq-job"] = job

    backend_app.run_generation_job("mocked-groq-job", job["prompt"])

    assert job["status"] == "success"
    assert job["current_step"] == "8_complete"
    assert job["plan"] == ["Build and verify"]
    assert {item["filename"] for item in job["generated_files"]} == {
        "requirements.txt", "models.py", "routes.py", "app.py", "index.html", "style.css", "script.js",
    }
    assert len(groq_calls) == 6  # combined spec+plan and five model-generated components


def test_both_unavailable_providers_end_job_with_offline_replay_option(monkeypatch, tmp_path):
    import app as backend_app

    monkeypatch.setattr(backend_app, "JOB_STORE", {})
    monkeypatch.setattr(backend_app, "GENERATION_QUEUE", [])
    monkeypatch.setattr(backend_app, "_persist_job", lambda *args: None)
    monkeypatch.setattr(backend_app, "_make_room_for_app", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_app, "match_showcase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("groq", "model", "temporarily unavailable", 503)))
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("gemini", "model", "temporarily unavailable", 503)))
    job = {"status": "queued", "current_step": "queued", "logs": [], "result": None,
           "spec": None, "plan": None, "generated_files": [], "execution": None,
           "verification": None, "versions": [], "started_at": time.time(),
           "prompt": "an unrelated product idea", "force_demo": False}
    backend_app.JOB_STORE["providers-unavailable-job"] = job

    backend_app.run_generation_job("providers-unavailable-job", job["prompt"])

    assert job["status"] == "failed"
    assert job["current_step"] == "error"
    assert "Groq" in job["result"]["reason"] and "Gemini" in job["result"]["reason"]
    assert b"Try offline replay" in (backend_app.FRONTEND_DIR / "index.html").read_bytes()


def test_failed_verification_stops_generated_app_and_releases_managed_slot(monkeypatch, tmp_path):
    import app as backend_app
    from pathlib import Path

    job_id = "failed-verification-cleanup"
    project_dir = tmp_path / "generated"
    stopped = []
    job = {
        "status": "queued", "current_step": "queued", "logs": [], "result": None,
        "spec": {"app_name": "Cleanup", "api_contract": {}}, "plan": ["Build"],
        "generated_files": [{"filename": "app.py", "content": "print('ok')"}],
        "execution": None, "verification": None, "versions": [], "started_at": time.time(),
        "prompt": "cleanup test", "force_demo": False,
    }
    monkeypatch.setattr(backend_app, "JOB_STORE", {job_id: job})
    monkeypatch.setattr(backend_app, "GENERATION_QUEUE", [])
    monkeypatch.setattr(backend_app, "MANAGED_PROJECT_DIRS", {project_dir})
    monkeypatch.setattr(backend_app, "_persist_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_app, "_generate_project_files", lambda *args, **kwargs: job["generated_files"])
    monkeypatch.setattr(backend_app, "_prepare_generated_files", lambda files: files)
    monkeypatch.setattr(backend_app, "validate_generated_project_files", lambda *args: [])
    monkeypatch.setattr(backend_app, "_make_room_for_app", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_app, "execute_generated_app", lambda *args: {"project_dir": str(project_dir), "port": 49002})
    monkeypatch.setattr(backend_app, "self_correct_generated_app", lambda *args, **kwargs: {"status": "failed", "error": "app did not start"})
    monkeypatch.setattr(backend_app, "match_showcase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_app, "stop_generated_app", lambda path: stopped.append(Path(path)))

    backend_app.run_generation_job(job_id, job["prompt"])

    assert job["status"] == "failed"
    assert stopped == [project_dir]
    assert project_dir not in backend_app.MANAGED_PROJECT_DIRS


def test_ui_error_formatter_keeps_json_failure_stage_and_diagnostic():
    import app as backend_app

    error = "Combined spec+plan response remained invalid after one formatting retry: Unable to parse JSON at line 1, column 9"
    assert backend_app._format_error_message(error) == error


def test_ui_error_formatter_preserves_ollama_timeout_source():
    import app as backend_app

    error = "Combined spec+plan provider request failed: Ollama request timed out at http://localhost:11434 (ReadTimeout)."
    assert backend_app._format_error_message(error) == error


def test_verification_repair_iterations_are_capped(monkeypatch, tmp_path):
    import app as backend_app

    (tmp_path / "app.py").write_text("print('unchanged')", encoding="utf-8")
    verify_calls, repair_calls = [], []

    def failed_verification(*args, **kwargs):
        verify_calls.append(1)
        return {"status": "failed", "error": "app.py failed", "traceback": ""}

    def repair(project_dir, filename, error):
        repair_calls.append(filename)
        return filename

    monkeypatch.setattr(backend_app, "verify_generated_app", failed_verification)
    monkeypatch.setattr(backend_app, "repair_failed_file", repair)
    monkeypatch.setattr(backend_app, "restart_generated_app", lambda *args, **kwargs: None)

    result = backend_app.self_correct_generated_app(tmp_path, max_attempts=99)

    assert result["status"] == "failed_after_retries"
    assert result["attempts_used"] == backend_app.MAX_AUTO_REPAIR_ITERATIONS == 3
    assert len(repair_calls) == 3
    assert len(verify_calls) == 4
