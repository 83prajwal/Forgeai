"""Blueprint generation: model JSON -> normalized profile -> proven runtime."""
from __future__ import annotations

import json
import time

import pytest

from agent import blueprint as blueprint_module
from agent.blueprint import _title_from_prompt, normalize_blueprint

MODEL_REPLY = {
    "name": "Recipe Manager", "category": "Food", "tone": "Warm and tidy", "accent": "#e07a2f", "icon": "🍲",
    "entities": [
        {"name": "Recipes", "icon": "📖", "fields": [
            {"name": "Title", "type": "string", "required": True},
            {"name": "prep_time", "label": "Prep time (min)", "type": "integer"},
            {"name": "difficulty", "type": "enum", "options": ["Easy", "Medium", "Hard"]},
            {"name": "is_favorite", "type": "bool"},
            {"name": "id", "type": "integer"},
            {"name": "created_at", "type": "datetime"},
        ]},
        {"name": "Settings", "fields": [{"name": "note", "type": "text"}]},
        {"name": "Empty", "fields": []},
    ],
}


def test_normalize_maps_types_and_drops_reserved_names():
    blueprint = normalize_blueprint(MODEL_REPLY, "Build a recipe manager app")

    recipes = blueprint["entities"][0]
    assert blueprint["name"] == "Recipe Manager" and blueprint["id"] == "recipe-manager"
    assert [(f["name"], f["type"]) for f in recipes["fields"]] == [
        ("title", "text"), ("prep_time", "number"), ("difficulty", "select"), ("is_favorite", "boolean")]
    assert recipes["fields"][0]["required"] is True
    assert blueprint["entities"][1]["table"] == "settings_items"  # never collides with the runtime's settings table
    assert len(blueprint["entities"]) == 2  # entity without fields dropped


def test_normalize_falls_back_to_users_words_and_rejects_empty():
    blueprint = normalize_blueprint({"entities": [{"name": "Workouts", "fields": [{"name": "activity"}]}]},
                                    "Build a workout tracker app")
    assert blueprint["name"] == "Workout Tracker"
    assert blueprint["accent"].startswith("#")
    with pytest.raises(ValueError):
        normalize_blueprint({"name": "X", "entities": []}, "x")


@pytest.mark.parametrize("prompt,expected", [
    ("Build a workout tracker app", "Workout Tracker"),
    ("create an expense tracker for students", "Expense Tracker"),
    ("Recipe manager with ratings", "Recipe Manager"),
])
def test_title_from_prompt(prompt, expected):
    assert _title_from_prompt(prompt) == expected


def test_generate_blueprint_retries_once_on_bad_json(monkeypatch):
    replies = iter(["not json", json.dumps(MODEL_REPLY)])

    class Reply:
        def __init__(self, text):
            self.text = text

    monkeypatch.setattr(blueprint_module, "generate_with_backoff", lambda *a, **k: Reply(next(replies)))
    assert blueprint_module.generate_blueprint("Build a recipe manager app")["name"] == "Recipe Manager"


def test_blueprint_job_builds_and_verifies_on_real_runtime(monkeypatch, tmp_path):
    """Full path with a mocked model: blueprint -> runtime files -> running app -> CRUD verification."""
    import app as backend_app

    monkeypatch.setattr(backend_app, "GENERATION_MODE", "blueprint")
    monkeypatch.setattr(backend_app, "GENERATED_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(backend_app, "JOB_STORE", {})
    monkeypatch.setattr(backend_app, "GENERATION_QUEUE", [])
    monkeypatch.setattr(backend_app, "_persist_job", lambda *args: None)
    monkeypatch.setattr(backend_app, "generate_blueprint",
                        lambda prompt, **kwargs: normalize_blueprint(MODEL_REPLY, prompt))
    job = {"status": "queued", "current_step": "queued", "logs": [], "result": None, "spec": None, "plan": None,
           "generated_files": [], "execution": None, "verification": None, "versions": [],
           "started_at": time.time(), "prompt": "Build a recipe manager app", "force_demo": False}
    backend_app.JOB_STORE["blueprint-job"] = job
    try:
        backend_app.run_generation_job("blueprint-job", job["prompt"])
        assert job["status"] == "success", job["logs"][-3:]
        assert job["replay"] is False
        assert job["spec"]["app_name"] == "Recipe Manager"
        assert job["verification"]["status"] == "ok"
        assert {f["filename"] for f in job["generated_files"]} >= {"app.py", "profile.json", "script.js"}
    finally:
        if job.get("execution"):
            backend_app.stop_generated_app(backend_app.Path(job["execution"]["project_dir"]))
