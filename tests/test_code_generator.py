import json

import pytest

from agent.code_generator import build_codegen_prompt, extract_codegen_payload


def test_build_codegen_prompt_includes_spec_and_component_context():
    prompt = build_codegen_prompt(
        {
            "app_name": "Task Tracker",
            "entities": [{"name": "Task", "fields": [{"name": "title", "type": "string"}]}],
        },
        "app.py",
        "Main Flask application entrypoint",
    )

    assert "Task Tracker" in prompt
    assert "app.py" in prompt
    assert "Main Flask application entrypoint" in prompt
    assert "Return valid JSON only" in prompt


def test_routes_prompt_trims_redundant_context_but_keeps_api_contract():
    spec = {
        "app_name": "Inventory Console",
        "purpose": "Track stock",
        "core_features": ["Record stock movements"],
        "api_contract": {"endpoints": [{
            "method": "POST", "path": "/api/movements", "request_json": {"quantity": "number"},
            "response_json": {"success": True}, "db_table": "movements", "db_columns": ["quantity"],
        }]},
    }
    prompt = build_codegen_prompt(
        spec, "routes.py", "Implement contract routes",
        context_files=[{"filename": "models.py", "content": "DO_NOT_INCLUDE_FULL_MODEL_SOURCE"}],
        user_prompt="DO_NOT_REPEAT_THE_RAW_USER_REQUEST",
    )

    assert '"/api/movements"' in prompt
    assert "get_db()" in prompt and "init_app(app)" in prompt
    assert "DO_NOT_INCLUDE_FULL_MODEL_SOURCE" not in prompt
    assert "DO_NOT_REPEAT_THE_RAW_USER_REQUEST" not in prompt


def test_extract_codegen_payload_handles_wrapped_json():
    wrapped = '''Here is the generated file:
```json
{
  "filename": "app.py",
  "content": "from flask import Flask\n\napp = Flask(__name__)\n" 
}
```
'''

    payload = extract_codegen_payload(wrapped)

    assert payload["filename"] == "app.py"
    assert "Flask" in payload["content"]


def test_extract_codegen_payload_handles_fenced_source_with_nested_braces():
    raw = '```json\n' + json.dumps({
        "filename": "script.js",
        "content": 'const item = { label: "ok" };\nif (item) { run(); }',
    }) + '\n```'

    assert extract_codegen_payload(raw)["content"] == 'const item = { label: "ok" };\nif (item) { run(); }'


def test_extract_codegen_payload_handles_valid_json():
    assert extract_codegen_payload('{"filename":"app.py","content":"print(1)"}') == {
        "filename": "app.py", "content": "print(1)"
    }


def test_extract_codegen_payload_handles_escaped_quotes_and_multiline_source():
    source = 'const message = "Ada said \\"hello\\"";\nfunction run() {\n  return message;\n}\n'
    assert extract_codegen_payload(json.dumps({"filename": "page_task.js", "content": source})) == {
        "filename": "page_task.js", "content": source,
    }


def test_extract_codegen_payload_preserves_python_nested_dictionaries_and_quotes():
    source = '''def create_app():
    settings = {"label": "Ada's \\\"workspace\\\"", "flags": {"ready": True}}
    return settings
'''
    raw = json.dumps({"filename": "routes.py", "content": source})

    assert extract_codegen_payload(raw) == {"filename": "routes.py", "content": source}


def test_extract_codegen_payload_rejects_malformed_json():
    with pytest.raises(ValueError, match="Unable to parse generated file JSON"):
        extract_codegen_payload('{"filename":"app.py","content":"print(1)"')


def test_extract_codegen_payload_rejects_missing_fields():
    with pytest.raises(ValueError, match="filename and content"):
        extract_codegen_payload('{"filename":"app.py"}')


def test_extract_codegen_payload_rejects_truncated_response():
    with pytest.raises(ValueError, match="Unable to parse generated file JSON"):
        extract_codegen_payload('{"filename":"app.py","content":"def run():\\n    return 1')


def test_extract_codegen_payload_rejects_empty_source():
    with pytest.raises(ValueError, match="must not be empty"):
        extract_codegen_payload('{"filename":"app.py","content":""}')


def test_extract_codegen_payload_rejects_empty_and_malformed_response():
    with pytest.raises(ValueError, match="empty response"):
        extract_codegen_payload("")
    with pytest.raises(ValueError, match="Unable to parse generated file JSON"):
        extract_codegen_payload('```json\n{"filename":"app.py","content":"print(1)"')


def test_provider_error_is_not_retried_as_a_json_parse_failure(monkeypatch):
    import agent.code_generator as code_generator

    calls = []

    def provider_failure(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("Groq quota exhausted (HTTP 429)")

    monkeypatch.setattr(code_generator, "generate_with_backoff", provider_failure)
    with pytest.raises(RuntimeError, match="429"):
        code_generator.generate_file_for_component({}, "app.py", "entry point")
    assert len(calls) == 1


def test_validation_diagnostics_include_metadata_and_redact_sensitive_preview(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    import agent.code_generator as code_generator

    raw = SimpleNamespace(
        status_code=200,
        json=lambda: {"choices": [{"finish_reason": "length"}]},
    )
    response = SimpleNamespace(
        text='{"filename":"page_task.js","content":"Bearer secret-token-1234567890 user@example.com',
        provider="groq", model="test-model", raw=raw, complete=False,
    )
    monkeypatch.setattr(code_generator, "generate_with_backoff", lambda *_args, **_kwargs: response)

    with caplog.at_level(logging.WARNING, logger="forgeai.codegen"):
        with pytest.raises(RuntimeError, match="Incomplete model output.*finish_reason=length"):
            code_generator.generate_file_for_component({}, "page_task.js", "page")

    message = caplog.text
    assert "provider=groq" in message and "model=test-model" in message
    assert "status=200" in message and "finish_reason=length" in message
    assert "response_chars=" in message and "preview=" in message
    assert "secret-token-1234567890" not in message
    assert "user@example.com" not in message
    assert "Ada" not in message


def test_missing_response_metadata_is_logged_as_unavailable(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    import agent.code_generator as code_generator

    response = SimpleNamespace(text='{"filename":"page_task.js","content":', complete=True)
    monkeypatch.setattr(code_generator, "generate_with_backoff", lambda *_args, **_kwargs: response)
    with caplog.at_level(logging.WARNING, logger="forgeai.codegen"):
        with pytest.raises(RuntimeError):
            code_generator.generate_file_for_component({}, "page_task.js", "page")

    assert "provider=None" in caplog.text and "model=None" in caplog.text
    assert "status=None" in caplog.text and "finish_reason=None" in caplog.text


def test_provider_marked_truncated_response_is_rejected(monkeypatch):
    from types import SimpleNamespace

    import agent.code_generator as code_generator

    validated = []
    responses = [
        SimpleNamespace(text='not-json-at-all', provider="gemini", model="gemini-test", complete=False,
                        raw=SimpleNamespace(candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")]),
                        cache_validated=lambda: validated.append(True)),
        SimpleNamespace(text='still-not-json', provider="gemini", model="gemini-test", complete=False,
                        raw=SimpleNamespace(candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")]),
                        cache_validated=lambda: validated.append(True)),
    ]
    calls = []
    monkeypatch.setattr(code_generator, "generate_with_backoff", lambda prompt, **_kwargs: calls.append(prompt) or responses.pop(0))
    monkeypatch.setattr(code_generator, "extract_codegen_payload", lambda *_args: pytest.fail("incomplete output must not be parsed"))

    with pytest.raises(RuntimeError, match="MAX_TOKENS.*Bounded recovery exhausted"):
        code_generator.generate_file_for_component({}, "routes.py", "route handlers")

    assert len(calls) == 2
    assert "BOUNDED RECOVERY" in calls[1]
    assert validated == []


def test_complete_response_is_accepted_and_cache_validated(monkeypatch):
    from types import SimpleNamespace

    import agent.code_generator as code_generator

    validated = []
    payload = {"filename": "routes.py", "content": "app = None\n"}
    response = SimpleNamespace(
        text=json.dumps(payload), provider="groq", model="groq-test", complete=True,
        cache_validated=lambda: validated.append(True),
    )
    monkeypatch.setattr(code_generator, "generate_with_backoff", lambda *_args, **_kwargs: response)

    assert code_generator.generate_file_for_component({}, "routes.py", "route handlers") == payload
    assert validated == [True]


def test_model_filename_typo_is_canonicalized_to_requested_component(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    import agent.code_generator as code_generator

    response = SimpleNamespace(
        text=json.dumps({"filename": "page_coffeeeshop.js", "content": "window.ForgeAI.registerPage('coffeeshop', () => {});"}),
        provider="ollama", model="qwen2.5-coder:1.5b", complete=True,
    )
    monkeypatch.setattr(code_generator, "generate_with_backoff", lambda *_args, **_kwargs: response)

    with caplog.at_level(logging.WARNING, logger="forgeai.codegen"):
        payload = code_generator.generate_file_for_component({}, "page_coffeeshop.js", "coffee shop page")

    assert payload["filename"] == "page_coffeeshop.js"
    assert "window.ForgeAI.registerPage" in payload["content"]
    assert "filename mismatch" in caplog.text
    assert "page_coffeeeshop.js" not in caplog.text


def test_incomplete_routes_file_is_not_published_to_generated_file_callback(monkeypatch):
    import app as backend_app

    published = []

    def generate(_spec, component_name, *_args, **_kwargs):
        if component_name == "routes.py":
            raise RuntimeError("Incomplete model output for routes.py (finish_reason=MAX_TOKENS).")
        return {"filename": component_name, "content": "# complete component\n"}

    monkeypatch.setattr(backend_app, "generate_file_for_component", generate)
    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        backend_app._generate_project_files({}, on_file=lambda files: published.append([f["filename"] for f in files]))

    assert published[-1] == ["requirements.txt", "models.py"]
    assert all("routes.py" not in filenames for filenames in published)


def test_provider_error_diagnostics_use_structured_metadata_without_error_text(monkeypatch, caplog):
    import logging

    import agent.code_generator as code_generator
    from agent.llm_client import ProviderError

    def provider_failure(*_args, **_kwargs):
        raise ProviderError("gemini", "model-x", "secret-bearing provider body", status=429)

    monkeypatch.setattr(code_generator, "generate_with_backoff", provider_failure)
    with caplog.at_level(logging.WARNING, logger="forgeai.codegen"):
        with pytest.raises(RuntimeError):
            code_generator.generate_file_for_component({}, "page_task.js", "page")

    assert "provider=gemini" in caplog.text and "model=model-x" in caplog.text
    assert "status=429" in caplog.text
    assert "secret-bearing provider body" not in caplog.text


def test_malformed_model_json_gets_one_bounded_repair_attempt(monkeypatch):
    import agent.code_generator as code_generator

    from types import SimpleNamespace

    responses = iter([
        SimpleNamespace(text='{"filename":"routes.py","content":from flask import Flask}', complete=True),
        SimpleNamespace(text=json.dumps({"filename": "routes.py", "content": "app = None\n"}), complete=True),
    ])
    calls = []

    def mocked_generation(prompt, **_kwargs):
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr(code_generator, "generate_with_backoff", mocked_generation)
    payload = code_generator.generate_file_for_component({}, "routes.py", "route handlers")

    assert payload == {"filename": "routes.py", "content": "app = None\n"}
    assert len(calls) == 2
    assert "RETRY:" in calls[1]


def test_routes_codegen_uses_smaller_output_reserve(monkeypatch):
    import agent.code_generator as code_generator

    from types import SimpleNamespace

    observed = {}

    def mocked_generation(_prompt, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(text=json.dumps({"filename": "routes.py", "content": "app = None\n"}), complete=True)

    monkeypatch.setattr(code_generator, "generate_with_backoff", mocked_generation)
    code_generator.generate_file_for_component({}, "routes.py", "route handlers")

    assert observed["max_tokens"] == 3500


def test_extract_codegen_payload_repairs_double_escaped_content():
    import json
    from agent.code_generator import extract_codegen_payload

    source = 'import sqlite3\n\ndef greet(name):\n    return f"hi {name}\\n"\n'
    double_escaped = json.dumps({"filename": "models.py", "content": json.dumps(source)[1:-1]})
    payload = extract_codegen_payload(double_escaped)

    assert payload["content"] == source
    compile(payload["content"], "models.py", "exec")
    normal = json.dumps({"filename": "models.py", "content": source})
    assert extract_codegen_payload(normal)["content"] == source
