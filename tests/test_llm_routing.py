"""Provider routing tests use mocks only and never contact model APIs."""
from __future__ import annotations

import pytest

from agent import llm_client


@pytest.fixture(autouse=True)
def isolated_router(monkeypatch):
    llm_client._cooldown.clear()
    llm_client._usage["groq"].clear()
    llm_client._usage["gemini"].clear()
    llm_client._request_usage["groq"].clear()
    llm_client.reset_call_stats()
    monkeypatch.setattr(llm_client, "_cache_read", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_client, "_cache_write", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_client, "_throttle", lambda *args: 0)
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_: None)


def _keys(monkeypatch, groq="groq-test-key", gemini="gemini-test-key"):
    monkeypatch.setenv("GROQ_API_KEY", groq)
    monkeypatch.setenv("GEMINI_API_KEY", gemini)


def test_groq_success_does_not_call_gemini(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(llm_client, "_groq", lambda *args: ('{"ok":true}', object()))
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: pytest.fail("Gemini should not be called"))

    response = llm_client.generate_with_backoff("prompt", json_mode=True)

    assert (response.text, response.provider) == ('{"ok":true}', "groq")


def test_configured_primary_and_fallback_order_is_respected(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "groq")
    monkeypatch.setattr(llm_client, "_gemini", lambda *_args: ('{"from":"gemini"}', object()))
    monkeypatch.setattr(llm_client, "_groq", lambda *_args: pytest.fail("primary provider succeeded"))

    response = llm_client.generate_with_backoff("prompt", json_mode=True)

    assert response.provider == "gemini"


def test_invalid_provider_configuration_fails_before_any_request(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "unknown")
    monkeypatch.setattr(llm_client, "_groq", lambda *_args: pytest.fail("invalid config must not call a provider"))
    monkeypatch.setattr(llm_client, "_gemini", lambda *_args: pytest.fail("invalid config must not call a provider"))

    with pytest.raises(RuntimeError, match="LLM_PROVIDER must be"):
        llm_client.generate_with_backoff("prompt")


def test_groq_quota_exhaustion_falls_back_with_json_format(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("groq", "model", "daily quota exhausted", 429)))
    gemini_calls = []

    def gemini(*args):
        gemini_calls.append(args)
        return ('{"steps":["build"]}', object())

    monkeypatch.setattr(llm_client, "_gemini", gemini)
    response = llm_client.generate_with_backoff("prompt", json_mode=True, max_tokens=321, step="planner")

    assert response.provider == "gemini"
    assert response.model == llm_client.GEMINI_MODEL_SPEC
    assert gemini_calls[0][2:] == (321, True)
    metrics = llm_client.get_call_metrics()
    assert metrics["provider_attempts"] == 2
    assert metrics["provider_failures"] == 1
    assert metrics["fallback_activations"] == 1


def test_local_groq_budget_skip_uses_configured_gemini_without_groq_request(monkeypatch):
    _keys(monkeypatch)
    groq_calls = []
    monkeypatch.setattr(llm_client, "_token_budget", lambda *_args: 512)

    def throttle(provider, budget, estimate):
        if estimate > budget:
            raise llm_client.LocalBudgetError(
                provider, "model", "Estimated request exceeds the local per-minute safety budget."
            )
        return 0

    monkeypatch.setattr(llm_client, "_throttle", throttle)
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (groq_calls.append(args) or pytest.fail("oversized call must not reach Groq")))
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: ('{"ok":true}', object()))

    response = llm_client.generate_with_backoff("prompt", json_mode=True, max_tokens=600)

    assert response.provider == "gemini"
    assert groq_calls == []


def test_malformed_groq_http_envelope_falls_back_without_retries(monkeypatch):
    _keys(monkeypatch)
    calls = []

    def malformed_envelope(*_args):
        calls.append("groq")
        raise llm_client.ProviderError("groq", "model", "malformed JSON response envelope", 200)

    monkeypatch.setattr(llm_client, "_groq", malformed_envelope)
    monkeypatch.setattr(llm_client, "_gemini", lambda *_args: ('{"ok":true}', object()))

    response = llm_client.generate_with_backoff("prompt", json_mode=True, max_retries=3)

    assert response.provider == "gemini"
    assert calls == ["groq"]


def test_groq_request_per_minute_guard_waits_within_a_rolling_window(monkeypatch):
    llm_client._request_usage["groq"].clear()
    monkeypatch.setattr(llm_client, "GROQ_LOCAL_REQUESTS_PER_MINUTE", "2")
    clock = [100.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    assert llm_client._throttle_requests("groq") == 0
    assert llm_client._throttle_requests("groq") == 0
    assert llm_client._throttle_requests("groq") == 60


def test_both_providers_rate_limited_returns_safe_error(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("groq", "model", "quota exhausted", 429)))
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("gemini", "model", "rate limit", 429)))

    with pytest.raises(RuntimeError, match="Groq quota exhausted.*Gemini rate limit") as caught:
        llm_client.generate_with_backoff("prompt")
    assert "test-key" not in str(caught.value)


@pytest.mark.parametrize("status", [401, 403])
def test_invalid_credentials_without_fallback_raise(monkeypatch, status):
    _keys(monkeypatch, gemini="")
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("groq", "model", "invalid key", status)))

    expected = "Create or rotate the key" if status == 401 else "denied access"
    with pytest.raises(RuntimeError, match=expected):
        llm_client.generate_with_backoff("prompt")


def test_groq_unauthorized_falls_back_to_configured_gemini(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("groq", "model", "invalid key", 401)))
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: ('{"ok":true}', object()))

    response = llm_client.generate_with_backoff("prompt", json_mode=True)

    assert (response.text, response.provider) == ('{"ok":true}', "gemini")


@pytest.mark.parametrize("status", [400, 404])
def test_unsupported_model_without_fallback_raises(monkeypatch, status):
    _keys(monkeypatch, gemini="")
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("groq", "model", "unsupported model", status)))

    with pytest.raises(RuntimeError, match="rejected the configured model"):
        llm_client.generate_with_backoff("prompt")


def _chain(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "mistral,nvidia")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test-key")
    monkeypatch.setattr(llm_client, "_gemini", lambda *_args: pytest.fail("Gemini is not in the chain"))


def _fail(provider, status):
    return lambda *args: (_ for _ in ()).throw(llm_client.ProviderError(provider, "model", "boom", status))


def test_chain_order_is_groq_then_mistral_then_nvidia(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "mistral,nvidia")
    assert llm_client._provider_order() == ["groq", "mistral", "nvidia"]


def test_groq_success_does_not_call_fallbacks(monkeypatch):
    _chain(monkeypatch)
    monkeypatch.setattr(llm_client, "_groq", lambda *args: ('{"from":"groq"}', object()))
    monkeypatch.setattr(llm_client, "_mistral", lambda *args: pytest.fail("Groq succeeded"))
    monkeypatch.setattr(llm_client, "_nvidia", lambda *args: pytest.fail("Groq succeeded"))

    assert llm_client.generate_with_backoff("prompt").provider == "groq"


@pytest.mark.parametrize("status", [429, 401, 400, 503])
def test_groq_failure_falls_back_to_mistral(monkeypatch, status):
    _chain(monkeypatch)
    monkeypatch.setattr(llm_client, "_groq", _fail("groq", status))
    calls = []
    monkeypatch.setattr(llm_client, "_mistral", lambda *args: (calls.append(args[1]) or ('{"from":"mistral"}', object())))
    monkeypatch.setattr(llm_client, "_nvidia", lambda *args: pytest.fail("Mistral succeeded"))

    response = llm_client.generate_with_backoff("prompt", preferred_model="openai/gpt-oss-120b", json_mode=True)

    assert (response.text, response.provider) == ('{"from":"mistral"}', "mistral")
    assert calls == [llm_client.MISTRAL_MODEL_CODEGEN]


def test_groq_and_mistral_failures_fall_back_to_nvidia(monkeypatch):
    _chain(monkeypatch)
    monkeypatch.setattr(llm_client, "_groq", _fail("groq", 429))
    monkeypatch.setattr(llm_client, "_mistral", _fail("mistral", 401))
    monkeypatch.setattr(llm_client, "_nvidia", lambda *args: ('{"from":"nvidia"}', object()))
    switches = []

    response = llm_client.generate_with_backoff("prompt", on_model_switch=switches.append)

    assert response.provider == "nvidia"
    assert any("falling back to Mistral" in line for line in switches)
    assert any("falling back to Nvidia" in line for line in switches)


def test_all_chain_providers_failing_reports_each(monkeypatch):
    _chain(monkeypatch)
    monkeypatch.setattr(llm_client, "_groq", _fail("groq", 429))
    monkeypatch.setattr(llm_client, "_mistral", _fail("mistral", 429))
    monkeypatch.setattr(llm_client, "_nvidia", _fail("nvidia", 403))

    with pytest.raises(RuntimeError, match="Groq .*Mistral .*Nvidia") as caught:
        llm_client.generate_with_backoff("prompt")
    assert "test-key" not in str(caught.value)


def test_missing_mistral_key_skips_to_nvidia(monkeypatch):
    _chain(monkeypatch)
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    monkeypatch.setattr(llm_client, "_groq", _fail("groq", 429))
    monkeypatch.setattr(llm_client, "_mistral", lambda *args: pytest.fail("Mistral has no key"))
    monkeypatch.setattr(llm_client, "_nvidia", lambda *args: ('{"ok":true}', object()))

    assert llm_client.generate_with_backoff("prompt").provider == "nvidia"


def test_nvidia_request_omits_json_response_format(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test-key")
    sent = {}

    class FakeResponse:
        status_code = 200
        headers = {}

        def json(self):
            return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}

    def post(url, headers, json, timeout):
        sent.update(url=url, body=json, auth=headers["Authorization"])
        return FakeResponse()

    monkeypatch.setattr(llm_client.requests, "post", post)
    text, _raw = llm_client._nvidia("prompt", "meta/llama-3.3-70b-instruct", 100, True)

    assert text == "{}"
    assert sent["url"].endswith("/chat/completions") and "nvidia.com" in sent["url"]
    assert "response_format" not in sent["body"]
    assert sent["auth"] == "Bearer nvidia-test-key"


def test_missing_keys_is_clear_and_no_clients_called(monkeypatch):
    _keys(monkeypatch, groq="", gemini="")
    monkeypatch.setattr(llm_client, "_groq", lambda *args: pytest.fail("no Groq key"))
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: pytest.fail("no Gemini key"))

    with pytest.raises(RuntimeError, match="No LLM API keys are configured"):
        llm_client.generate_with_backoff("prompt")


def test_gemini_only_configuration_is_supported(monkeypatch):
    _keys(monkeypatch, groq="", gemini="gemini-test-key")
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: ('{"ok":true}', object()))
    monkeypatch.setattr(llm_client, "_groq", lambda *args: pytest.fail("Groq has no key"))

    response = llm_client.generate_with_backoff("prompt", json_mode=True)
    assert response.provider == "gemini"


def test_ollama_selection_never_calls_hosted_providers(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(llm_client, "_ollama", lambda *_args: ('{"ok":true}', object()))
    monkeypatch.setattr(llm_client, "_groq", lambda *_args: pytest.fail("local-only mode called Groq"))
    monkeypatch.setattr(llm_client, "_gemini", lambda *_args: pytest.fail("local-only mode called Gemini"))

    response = llm_client.generate_with_backoff("prompt", preferred_model="hosted-model", json_mode=True)

    assert response.provider == "ollama"
    assert response.model == llm_client.OLLAMA_MODEL


def test_ollama_connection_error_is_preserved_without_fallback_or_retries(monkeypatch):
    import requests

    _keys(monkeypatch)
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "gemini")
    calls = []

    def unavailable(*_args, **_kwargs):
        calls.append(1)
        raise requests.ConnectionError("connection refused; Authorization: Bearer secret-token-1234567890")

    monkeypatch.setattr(llm_client.requests, "post", unavailable)
    monkeypatch.setattr(llm_client, "_groq", lambda *_args: pytest.fail("local-only mode called Groq"))
    monkeypatch.setattr(llm_client, "_gemini", lambda *_args: pytest.fail("local-only mode called Gemini"))

    with pytest.raises(RuntimeError) as error:
        llm_client.generate_with_backoff("prompt", max_retries=3)
    assert "ConnectionError" in str(error.value)
    assert "connection refused" in str(error.value)
    assert "secret-token-1234567890" not in str(error.value)
    assert "Could not connect to Ollama at http://localhost:11434" in str(error.value)
    assert len(calls) == 1


def test_ollama_timeout_is_reported_as_timeout_not_unavailable(monkeypatch):
    import requests

    _keys(monkeypatch, groq="", gemini="")
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client.requests, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.ReadTimeout("read timed out")))

    with pytest.raises(RuntimeError, match="Ollama request timed out.*ReadTimeout"):
        llm_client.generate_with_backoff("prompt", max_retries=3)


def test_ollama_chat_response_parses_json_and_caps_local_output(monkeypatch):
    from types import SimpleNamespace

    _keys(monkeypatch)
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "gemini")
    observed = {}
    raw = SimpleNamespace(
        status_code=200,
        json=lambda: {"message": {"content": '{"ok":true}'}, "done": True, "done_reason": "stop"},
    )

    def post(url, **kwargs):
        observed.update(url=url, **kwargs)
        return raw

    monkeypatch.setattr(llm_client.requests, "post", post)
    response = llm_client.generate_with_backoff("prompt", json_mode=True, max_tokens=3500)

    assert response.text == '{"ok":true}'
    assert response.complete is True
    assert observed["url"] == "http://localhost:11434/api/chat"
    assert "headers" not in observed
    assert observed["json"]["format"] == "json"
    assert observed["json"]["options"]["num_predict"] == 1024
    assert observed["json"]["options"]["num_ctx"] == 4096
    assert observed["timeout"] == 300


def test_ollama_timeout_is_bounded_and_logged_with_stage(monkeypatch, caplog):
    import logging
    import requests

    _keys(monkeypatch, groq="", gemini="")
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.setattr(llm_client.requests, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.ReadTimeout("read timed out")))

    with caplog.at_level(logging.INFO, logger="forgeai.llm"):
        with pytest.raises(RuntimeError, match="Ollama request timed out.*ReadTimeout"):
            llm_client.generate_with_backoff("private prompt", step="spec", json_mode=True)

    assert "provider=ollama" in caplog.text
    assert "stage=spec" in caplog.text
    assert "elapsed_seconds=" in caplog.text
    assert "private prompt" not in caplog.text
    assert "provider=gemini" not in caplog.text


def test_ollama_max_tokens_finish_reason_is_incomplete(monkeypatch):
    from types import SimpleNamespace

    _keys(monkeypatch, groq="", gemini="")
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "_ollama", lambda *_args: (
        '{"filename":"routes.py"',
        SimpleNamespace(json=lambda: {"message": {"content": "partial"}, "done": True, "done_reason": "length"}),
    ))

    response = llm_client.generate_with_backoff("prompt", json_mode=True)

    assert response.complete is False


def test_ollama_health_probe_stays_on_local_service(monkeypatch):
    import app as backend_app
    from types import SimpleNamespace

    monkeypatch.setattr(backend_app, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(backend_app, "OLLAMA_BASE_URL", "http://localhost:11434")
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(backend_app.requests, "get", get)
    result = backend_app.app.test_client().get("/health/data")

    assert result.status_code == 200
    assert calls == ["http://localhost:11434/api/tags"]
    assert result.get_json()["providers"]["ollama"]["reachable"] is True
    assert result.get_json()["active_provider"] == "ollama"


def test_groq_bad_request_falls_back_to_configured_gemini(monkeypatch):
    _keys(monkeypatch)
    groq_calls = []

    def groq(*args):
        groq_calls.append(args)
        message = "failed to generate JSON" if args[3] else "plain request rejected"
        raise llm_client.ProviderError("groq", "model", message, 400)

    monkeypatch.setattr(llm_client, "_groq", groq)
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: ('{"ok":true}', object()))

    response = llm_client.generate_with_backoff("prompt", json_mode=True)

    assert [args[3] for args in groq_calls] == [True, False]
    assert response.provider == "gemini"


def test_groq_structured_error_is_preserved_and_sanitized_in_logs(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    _keys(monkeypatch, gemini="")
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "")
    response = SimpleNamespace(
        status_code=400,
        headers={"x-ratelimit-remaining-tokens": "132"},
        json=lambda: {"error": {
            "type": "invalid_request_error",
            "code": "json_validation_failed",
            "message": "Rejected Bearer gsk_superSecretCredential123 api_key=anotherSecretValue contact@example.com",
        }},
    )
    monkeypatch.setattr(llm_client.requests, "post", lambda *_args, **_kwargs: response)

    with caplog.at_level(logging.ERROR, logger="forgeai.llm"):
        with pytest.raises(RuntimeError, match="rejected the configured model or request"):
            llm_client.generate_with_backoff("private prompt content", json_mode=True, max_tokens=3500)

    record = next(record for record in caplog.records if record.name == "forgeai.llm")
    assert "status=400" in record.message
    assert "error_type=invalid_request_error" in record.message
    assert "error_code=json_validation_failed" in record.message
    assert "Rejected" in record.message
    assert "superSecretCredential123" not in record.message
    assert "anotherSecretValue" not in record.message
    assert "contact@example.com" not in record.message
    assert "private prompt content" not in record.message


def test_gemini_max_tokens_response_is_marked_incomplete_and_not_cached(monkeypatch):
    from types import SimpleNamespace

    _keys(monkeypatch, groq="", gemini="gemini-test-key")
    cache_writes = []
    monkeypatch.setattr(llm_client, "_cache_write", lambda *args, **kwargs: cache_writes.append(args))
    raw = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")])
    monkeypatch.setattr(llm_client, "_gemini", lambda *_args: ('{"filename":"routes.py"', raw))

    response = llm_client.generate_with_backoff("prompt", json_mode=True, max_tokens=3500)

    assert response.complete is False
    response.cache_validated()
    assert cache_writes == []


def test_missing_gemini_sdk_fails_without_retries(monkeypatch):
    _keys(monkeypatch, groq="", gemini="gemini-test-key")
    calls = []

    def unavailable(*args):
        calls.append(1)
        raise llm_client.ProviderError("gemini", "model", "The google-genai SDK is not installed.")

    monkeypatch.setattr(llm_client, "_gemini", unavailable)
    with pytest.raises(RuntimeError, match="Gemini request failed"):
        llm_client.generate_with_backoff("prompt", max_retries=3)
    assert len(calls) == 1


def test_missing_gemini_key_reports_fallback_unavailable(monkeypatch):
    _keys(monkeypatch, gemini="")
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("groq", "model", "daily quota exhausted", 429)))
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: pytest.fail("Gemini key absent"))

    with pytest.raises(RuntimeError, match="Groq quota exhausted.*set GEMINI_API_KEY in .env"):
        llm_client.generate_with_backoff("prompt")


def test_short_rate_limit_observes_retry_after_before_fallback(monkeypatch):
    _keys(monkeypatch)
    calls = []

    def groq(*args):
        calls.append("groq")
        if len(calls) == 1:
            raise llm_client.ProviderError("groq", "model", "request rate limit", 429, retry_after=2)
        return ("retried", object())

    monkeypatch.setattr(llm_client, "_groq", groq)
    monkeypatch.setattr(llm_client, "_gemini", lambda *args: pytest.fail("retry succeeded"))
    response = llm_client.generate_with_backoff("prompt", max_retries=1, max_delay=3)

    assert response.text == "retried"
    assert calls == ["groq", "groq"]


def test_provider_error_logs_and_stats_do_not_include_keys(monkeypatch, caplog):
    _keys(monkeypatch, groq="SECRET_GROQ_KEY", gemini="SECRET_GEMINI_KEY")
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "")
    monkeypatch.setattr(llm_client, "_groq", lambda *args: (_ for _ in ()).throw(
        llm_client.ProviderError("groq", "model", "secret SECRET_GROQ_KEY invalid key", 401)))

    with pytest.raises(RuntimeError):
        llm_client.generate_with_backoff("prompt")
    assert "SECRET_GROQ_KEY" not in caplog.text
    assert "SECRET_GEMINI_KEY" not in caplog.text
    assert "SECRET_GROQ_KEY" not in repr(llm_client.get_call_stats())


def test_gemini_sdk_receives_structured_output_config(monkeypatch):
    from google import genai

    _keys(monkeypatch, groq="", gemini="gemini-test-key")
    observed = {}

    class Models:
        def generate_content(self, **kwargs):
            observed.update(kwargs)
            return type("Response", (), {"text": '{"ok":true}'})()

    class Client:
        def __init__(self, **kwargs):
            observed["client"] = kwargs
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    response = llm_client.generate_with_backoff("prompt", json_mode=True, max_tokens=81)

    assert response.provider == "gemini"
    assert observed["config"].response_mime_type == "application/json"
    assert observed["config"].max_output_tokens == 81
    assert observed["client"]["api_key"] == "gemini-test-key"

