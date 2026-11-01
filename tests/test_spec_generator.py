import json
from types import SimpleNamespace

import pytest

import agent.spec_generator as spec_generator
from agent.spec_generator import build_combined_spec_plan_prompt, build_spec_prompt, extract_json_payload, normalize_spec


def test_build_spec_prompt_includes_json_only_instructions():
    prompt = build_spec_prompt("Build a task tracker app with users and tasks")

    assert "Return ONLY valid JSON" in prompt
    assert "no markdown fences" in prompt.lower()
    assert "app_name" in prompt
    assert "routes_needed" in prompt


def test_combined_spec_plan_captures_requirements_without_an_extra_stage():
    prompt = build_combined_spec_plan_prompt("Build a budget planner for one person")

    assert '"user_roles":[]' in prompt
    assert '"database_requirements":[]' in prompt
    assert '"validation_requirements":[]' in prompt
    assert '"responsive_requirements":[]' in prompt
    assert "single local user" in prompt
    assert "at most 3 short implementation steps" in prompt


def test_extract_json_payload_handles_wrapped_json():
    wrapped = '''Here is the app spec:
```json
{
  "app_name": "Task Tracker",
  "entities": [
    {"name": "User", "fields": [{"name": "id", "type": "string"}]}
  ],
  "relationships": [{"from": "User", "to": "Task", "type": "one-to-many"}],
  "auth_required": true,
  "routes_needed": ["/login", "/tasks"],
  "pages_needed": ["Login", "Dashboard"]
}
```
'''

    payload = extract_json_payload(wrapped)

    assert payload["app_name"] == "Task Tracker"
    assert payload["auth_required"] is True
    assert payload["routes_needed"] == ["/login", "/tasks"]


def test_extract_json_payload_handles_nested_fenced_objects_and_trailing_prose():
    response = 'Result:\n```json\n' + json.dumps({
        "spec": {"entities": [{"name": "Task", "fields": [{"name": "metadata", "type": "object"}]}]},
        "plan": ["Build {task}", "Verify"],
    }) + '\n```\nDone.'

    payload = extract_json_payload(response)

    assert payload["spec"]["entities"][0]["fields"][0]["name"] == "metadata"
    assert payload["plan"] == ["Build {task}", "Verify"]


@pytest.mark.parametrize("response", ["", "```json\n{\"spec\": {\n```", "not JSON"])
def test_extract_json_payload_rejects_empty_malformed_and_truncated_output(response):
    with pytest.raises(ValueError):
        extract_json_payload(response)


def test_combined_generation_retries_invalid_json_once_then_validates(monkeypatch):
    spec = {"app_name": "Retry Test", "entities": [], "pages": []}
    responses = [
        SimpleNamespace(text='```json\n{"spec":{"entities":[', complete=True),
        SimpleNamespace(text=json.dumps({"spec": spec, "plan": ["Build it"]}), complete=True),
    ]
    prompts = []

    def generate(prompt, **_kwargs):
        prompts.append(prompt)
        return responses.pop(0)

    monkeypatch.setattr(spec_generator, "generate_with_backoff", generate)
    parsed_spec, plan = spec_generator.generate_spec_and_plan("make a test app")

    assert parsed_spec["app_name"] == "Retry Test"
    assert plan == ["Build it"]
    assert len(prompts) == 2
    assert "RETRY:" in prompts[1]


def test_combined_generation_does_not_retry_provider_errors(monkeypatch):
    calls = []

    def provider_failure(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("Groq API rate limit reached (HTTP 429)")

    monkeypatch.setattr(spec_generator, "generate_with_backoff", provider_failure)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        spec_generator.generate_spec_and_plan("make a test app")
    assert len(calls) == 1


def test_incomplete_ollama_spec_response_is_rejected_without_expensive_retry(monkeypatch):
    calls = []
    incomplete = SimpleNamespace(
        provider="ollama", model="qwen2.5-coder:1.5b", text='{"spec":', complete=False,
        raw=SimpleNamespace(json=lambda: {"done": True, "done_reason": "length"}),
    )

    def generate(*_args, **_kwargs):
        calls.append(1)
        return incomplete

    monkeypatch.setattr(spec_generator, "generate_with_backoff", generate)
    with pytest.raises(RuntimeError, match=r"Ollama returned incomplete spec\+plan output.*finish_reason=length"):
        spec_generator.generate_spec_and_plan("make a test app")
    assert len(calls) == 1


def test_normalize_spec_adds_safe_defaults_and_drops_malformed_optional_fields():
    spec = normalize_spec({
        "app_name": "Budget Log",
        "entities": [{"name": "Expense", "fields": [{"name": "amount", "type": "number"}, None]}],
        "seed_data": {"Expense": [{"amount": 12}]},
    })

    assert spec["purpose"] == "Budget Log"
    assert spec["core_features"] == []
    assert spec["entities"][0]["fields"] == [{"name": "amount", "type": "number"}]
    assert spec["theme"] == {}


def test_normalize_spec_rejects_malformed_core_collections():
    with pytest.raises(ValueError, match="entities.*must be a list"):
        normalize_spec({"entities": "Task"})


def test_legacy_spec_generation_rejects_provider_marked_truncation(monkeypatch):
    monkeypatch.setattr(spec_generator, "generate_with_backoff", lambda *_args, **_kwargs: SimpleNamespace(
        text='{"app_name":"Incomplete","entities":[]}', complete=False,
    ))

    with pytest.raises(RuntimeError, match="truncated"):
        spec_generator.generate_spec("make an app")


def test_plan_steps_accept_object_and_numbered_shapes():
    from agent.spec_generator import _coerce_plan_steps

    assert _coerce_plan_steps(["Build models", "Add UI"]) == ["Build models", "Add UI"]
    assert _coerce_plan_steps([{"step": 1, "description": "Build models"}, {"title": "Add UI"}]) == ["Build models", "Add UI"]
    assert _coerce_plan_steps("1. Build models\n2) Add UI") == ["Build models", "Add UI"]
    with pytest.raises(ValueError, match="list of strings"):
        _coerce_plan_steps([{"step": 1}, 5])


def test_api_contract_converts_express_and_openapi_path_params():
    from agent.spec_generator import build_api_contract

    spec = {"entities": [{"name": "Workouts", "fields": [{"name": "title", "type": "string"}]}],
            "routes_needed": ["GET /api/workouts", "GET /api/workouts/:id", "DELETE /api/workouts/{id}"]}
    paths = {(item["method"], item["path"]) for item in build_api_contract(spec)["endpoints"]}

    assert ("GET", "/api/workouts/<int:id>") in paths
    assert ("DELETE", "/api/workouts/<int:id>") in paths
    assert not any(":" in path.replace("<int:", "") or "{" in path for _method, path in paths)


def test_api_contract_ignores_new_and_edit_form_pages():
    from agent.spec_generator import build_api_contract

    spec = {"entities": [{"name": "Workouts", "fields": [{"name": "title", "type": "string"}]}],
            "routes_needed": ["GET /workouts/new", "GET /workouts/:id/edit", "GET /api/workouts"]}
    paths = [item["path"] for item in build_api_contract(spec)["endpoints"]]

    assert not any(path.endswith(("/new", "/edit")) for path in paths)
    assert "/api/workouts" in paths


def test_api_contract_keeps_flask_path_params_unchanged():
    from agent.spec_generator import build_api_contract

    spec = {"entities": [{"name": "Workouts", "fields": [{"name": "title", "type": "string"}]}],
            "routes_needed": ["GET /api/workouts/<int:id>", "PUT /api/workouts/<id>"]}
    paths = {item["path"] for item in build_api_contract(spec)["endpoints"]}

    assert "/api/workouts/<int:id>" in paths and "/api/workouts/<id>" in paths
    assert not any("<int<" in path for path in paths)


def test_api_contract_assigns_nested_routes_to_their_child_entity():
    from agent.spec_generator import build_api_contract

    spec = {"entities": [{"name": "Workout", "fields": [{"name": "title", "type": "string"}]},
                         {"name": "Exercise", "fields": [{"name": "name", "type": "string"}]}],
            "routes_needed": ["GET /api/workouts/<int:id>/exercises", "GET /api/workouts", "PUT /api/workouts/<int:id>"]}
    endpoints = build_api_contract(spec)["endpoints"]
    by_route = {(item["method"], item["path"]): item["db_table"] for item in endpoints}

    assert by_route[("GET", "/api/workouts/<int:id>/exercises")] == "exercise"
    assert by_route[("PUT", "/api/workouts/<int:id>")] == "workout"
    workout_details = {path for (method, path), table in by_route.items() if table == "workout" and "<" in path}
    assert workout_details == {"/api/workouts/<int:id>"}


def test_api_contract_strips_query_strings_from_routes():
    from agent.spec_generator import build_api_contract

    spec = {"entities": [{"name": "Recipes", "fields": [{"name": "title", "type": "string"}]}],
            "routes_needed": ["GET /api/recipes/search?query=", "GET /api/recipes?page=1"]}
    paths = {item["path"] for item in build_api_contract(spec)["endpoints"]}

    assert "/api/recipes/search" in paths and "/api/recipes" in paths
    assert not any("?" in path for path in paths)
