"""Sequential, observable LLM routing with token-aware throttling and caching."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import requests

from config import (
    BASE_DIR, GEMINI_MODEL_CODEGEN, GEMINI_MODEL_FIX, GEMINI_MODEL_SPEC,
    GROQ_MODEL_CODEGEN, GROQ_MODEL_FIX, GROQ_MODEL_SPEC, GROQ_TOKENS_PER_MINUTE,
    GROQ_TPM_70B, GROQ_TPM_8B, GROQ_LOCAL_REQUESTS_PER_MINUTE, GROQ_MAX_RETRIES,
    LLM_PROVIDER, LLM_FALLBACK_PROVIDER, LLM_CACHE_ENABLED,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_MAX_OUTPUT_TOKENS,
    OLLAMA_REQUEST_TIMEOUT_SECONDS,
)
from config import get_groq_api_key

logger = logging.getLogger("forgeai.llm")
_call_lock = threading.RLock()
_usage: dict[str, deque[tuple[float, int]]] = {"groq": deque(), "gemini": deque(), "ollama": deque()}
_request_usage: dict[str, deque[float]] = {"groq": deque()}
_cooldown: dict[str, float] = {}
_call_stats: list[dict[str, Any]] = []
_stats_lock = threading.Lock()
_call_metrics = {"cache_hits": 0, "provider_attempts": 0, "fallback_activations": 0,
                 "provider_failures": 0, "generation_failures": 0}


class ProviderError(RuntimeError):
    def __init__(self, provider: str, model: str, message: str, status: int | None = None,
                 retry_after: float | None = None, headers: Any = None,
                 error_type: Any = None, error_code: Any = None):
        super().__init__(message)
        self.provider, self.model, self.status, self.retry_after, self.headers = provider, model, status, retry_after, headers
        self.error_type, self.error_code = error_type, error_code


class LocalBudgetError(ProviderError):
    """A local Groq preflight limit, distinct from a provider response."""


class LLMResponse:
    def __init__(self, text: str, provider: str, model: str, retries: int = 0, raw: Any = None,
                 cache_callback: Callable[[], None] | None = None) -> None:
        self.text, self.provider, self.model = text, provider, model
        self.retries, self.raw = retries, raw
        self.complete = _response_complete(raw)
        self._cache_callback = cache_callback

    def cache_validated(self) -> None:
        """Commit structured output only after its caller validates the schema."""
        if self._cache_callback is not None:
            callback, self._cache_callback = self._cache_callback, None
            callback()


def get_call_stats() -> list[dict[str, Any]]:
    with _stats_lock:
        return [dict(row) for row in _call_stats]


def reset_call_stats() -> None:
    with _stats_lock:
        _call_stats.clear()
        for name in _call_metrics:
            _call_metrics[name] = 0


def get_call_metrics() -> dict[str, int]:
    """Return aggregate counters without prompts, credentials, or provider text."""
    with _stats_lock:
        return dict(_call_metrics)


def _headers(headers: Any) -> str:
    if not headers:
        return "{}"
    return json.dumps({str(k): str(v) for k, v in headers.items() if str(k).lower() == "retry-after" or str(k).lower().startswith("x-ratelimit-")}, sort_keys=True)


_LOG_SECRET_PATTERNS = (
    re.compile(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|secret|password)\b\s*[:=]\s*[\"']?)[^\s,;\"'}]+"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gsk_[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,})\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
)


def _safe_error_detail(value: Any, limit: int = 240) -> str | None:
    if value is None:
        return None
    detail = " ".join(str(value).split())
    for variable in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        secret = os.getenv(variable, "").strip()
        if len(secret) >= 8:
            detail = detail.replace(secret, "[REDACTED]")
    for pattern in _LOG_SECRET_PATTERNS:
        if pattern.groups == 1:
            detail = pattern.sub(lambda match: match.group(1) + "[REDACTED]" if match.group(1).lower().startswith("http") else "[REDACTED]", detail)
        else:
            detail = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", detail)
    return detail[:limit]


def _status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) or getattr(exc, "status", None) or getattr(exc, "code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        match = re.search(r"\b(400|401|402|403|404|429|5\d\d)\b", str(exc))
        return int(match.group(1)) if match else None


def _retry_seconds(headers: Any, message: str) -> float | None:
    try:
        val = (headers or {}).get("retry-after")
        if val is not None:
            return max(0.0, float(val))
    except (TypeError, ValueError):
        pass
    match = re.search(r"(?:retry|try again) in\s*([\d.]+)\s*s", message, re.I)
    if match:
        return float(match.group(1))
    match = re.search(r"retry in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+)s", message, re.I)
    if match:
        return int(match.group(1) or 0) * 3600 + int(match.group(2) or 0) * 60 + float(match.group(3))
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", message, re.I)
    return float(match.group(1)) if match else None


def _provider_model(provider: str, step: str, preferred_model: str | None) -> str:
    if provider == "ollama":
        return OLLAMA_MODEL
    if provider == "gemini":
        return {"spec": GEMINI_MODEL_SPEC, "planner": GEMINI_MODEL_SPEC, "fix": GEMINI_MODEL_FIX}.get(step, GEMINI_MODEL_CODEGEN)
    if preferred_model and not preferred_model.lower().startswith("gemini-"):
        return preferred_model
    return {"spec": GROQ_MODEL_SPEC, "planner": GROQ_MODEL_SPEC, "fix": GROQ_MODEL_FIX}.get(step, GROQ_MODEL_CODEGEN)


def _provider_order() -> list[str]:
    """Resolve configured provider order while allowing a deliberate no-fallback setting."""
    supported = {"groq", "gemini", "ollama"}
    primary = LLM_PROVIDER
    fallback = LLM_FALLBACK_PROVIDER
    # Ollama is intentionally isolated: a local-only selection can never fall
    # through to a hosted provider, even if an older .env still names Gemini.
    if primary == "ollama":
        return ["ollama"]
    if primary not in supported:
        raise RuntimeError("LLM_PROVIDER must be 'groq', 'gemini', or 'ollama'.")
    if fallback and fallback not in supported:
        raise RuntimeError("LLM_FALLBACK_PROVIDER must be 'groq', 'gemini', or empty to disable fallback.")
    return [primary] + ([fallback] if fallback and fallback != primary else [])


def _provider_output_limit(provider: str, requested: int) -> int:
    if provider == "ollama":
        return min(max(1, int(requested)), OLLAMA_MAX_OUTPUT_TOKENS)
    return int(requested)


def _token_budget(provider: str, model: str) -> int:
    """Return the local estimated TPM guard, separate from provider limits."""
    if GROQ_TOKENS_PER_MINUTE:
        return max(512, int(GROQ_TOKENS_PER_MINUTE))
    # Match an 8B model explicitly. A substring check for "20b" accidentally
    # classified "120b" as the smaller tier and rejected valid codegen calls.
    is_8b = re.search(r"(?<!\d)8b(?!\d)", model.lower()) is not None
    configured = GROQ_TPM_8B if is_8b else GROQ_TPM_70B
    return max(512, int(os.getenv("GROQ_TOKENS_PER_MINUTE", configured)))


def _throttle(provider: str, budget: int, estimate: int) -> float:
    """Reserve prompt+completion tokens across concurrent app jobs."""
    wait_total = 0.0
    with _call_lock:
        queue = _usage[provider]
        while True:
            now = time.monotonic()
            while queue and now - queue[0][0] >= 60:
                queue.popleft()
            used = sum(tokens for _, tokens in queue)
            if estimate > budget:
                # A request larger than the configured window cannot be made safe by waiting.
                raise LocalBudgetError(
                    provider, "",
                    f"Estimated request ({estimate} tokens) exceeds the local per-minute safety budget ({budget}). "
                    "Reduce prompt context or the output-token reserve; actual provider limits are enforced separately.",
                )
            if used + estimate <= budget:
                queue.append((now, estimate))
                return wait_total
            delay = max(0.05, 60 - (now - queue[0][0]))
            time.sleep(delay)
            wait_total += delay


def _throttle_requests(provider: str) -> float:
    """Apply a local Groq requests-per-minute guard independently of TPM."""
    if provider != "groq":
        return 0.0
    limit = max(1, int(GROQ_LOCAL_REQUESTS_PER_MINUTE))
    wait_total = 0.0
    with _call_lock:
        queue = _request_usage[provider]
        while True:
            now = time.monotonic()
            while queue and now - queue[0] >= 60:
                queue.popleft()
            if len(queue) < limit:
                queue.append(now)
                return wait_total
            delay = max(0.05, 60 - (now - queue[0]))
            time.sleep(delay)
            wait_total += delay


def _cache_file(provider: str, model: str, prompt: str, *, max_tokens: int = 2000,
                json_mode: bool = False, step: str = "codegen", context: Any = None) -> Path:
    request = {"version": 2, "provider": provider, "model": model, "prompt": prompt,
               "max_tokens": max_tokens, "json_mode": json_mode, "step": step, "context": context}
    serialized = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return BASE_DIR / "backend" / "llm_cache" / f"{digest}.json"


def _cache_read(provider: str, model: str, prompt: str, *, max_tokens: int = 2000,
                json_mode: bool = False, step: str = "codegen", context: Any = None) -> str | None:
    if not LLM_CACHE_ENABLED:
        return None
    path = _cache_file(provider, model, prompt, max_tokens=max_tokens, json_mode=json_mode, step=step, context=context)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = payload.get("text") if isinstance(payload, dict) else None
        return text if isinstance(text, str) and text.strip() else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _cache_write(provider: str, model: str, prompt: str, text: str, *, max_tokens: int = 2000,
                 json_mode: bool = False, step: str = "codegen", context: Any = None) -> None:
    if not LLM_CACHE_ENABLED or not isinstance(text, str) or not text.strip():
        return
    if json_mode:
        try:
            json.loads(text)
        except (TypeError, ValueError):
            # Keep permissive parsing behavior for callers, but don't preserve
            # malformed or truncated structured output as a successful result.
            return
    path = _cache_file(provider, model, prompt, max_tokens=max_tokens, json_mode=json_mode, step=step, context=context)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text, "complete": True}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.exception("Could not write LLM cache")


def _log_failure(provider: str, model: str, prompt: str, max_tokens: int, exc: Exception,
                 response: Any = None, *, stage: str = "codegen", elapsed: float | None = None) -> None:
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None) or getattr(getattr(exc, "response", None), "headers", None)
    status = _status(exc)
    # Keep the provider's useful error fields while bounding and redacting its message.
    logger.error(
        "LLM failure provider=%s model=%s status=%s error_type=%s error_code=%s "
        "stage=%s elapsed_seconds=%s message=%r prompt_tokens~%d prompt_chars=%d max_tokens=%d headers=%s",
        provider, model, status,
        _safe_error_detail(getattr(exc, "error_type", None), 80),
        _safe_error_detail(getattr(exc, "error_code", None), 80),
        stage, round(elapsed, 2) if elapsed is not None else None,
        _safe_error_detail(exc, 240), len(prompt) // 4, len(prompt), max_tokens, _headers(headers),
    )


def _groq(prompt: str, model: str, max_tokens: int, json_mode: bool) -> tuple[str, Any]:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise ProviderError("groq", model, "GROQ_API_KEY is missing or empty.")
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body,
                             timeout=float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "90")))
    if response.status_code >= 400:
        error_type = error_code = None
        try:
            payload = response.json()
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                message = error.get("message") or f"Groq returned HTTP {response.status_code}."
                error_type, error_code = error.get("type"), error.get("code")
            else:
                message = f"Groq returned HTTP {response.status_code}."
        except ValueError:
            message = f"Groq returned HTTP {response.status_code} with a non-JSON error body."
        retry = _retry_seconds(response.headers, str(message))
        raise ProviderError("groq", model, str(message), response.status_code, retry, response.headers,
                            error_type=error_type, error_code=error_code)
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError("groq", model, "Groq returned a malformed JSON response envelope (HTTP 200).", 200) from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("groq", model, "Groq returned an incomplete response envelope (HTTP 200).", 200) from exc
    if not isinstance(content, str):
        raise ProviderError("groq", model, "Groq returned a non-text response envelope (HTTP 200).", 200)
    return content, response


def _gemini(prompt: str, model: str, max_tokens: int, json_mode: bool) -> tuple[str, Any]:
    """Call the supported google-genai SDK. Import lazily for offline/live isolation."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ProviderError("gemini", model, "GEMINI_API_KEY is missing or empty.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ProviderError("gemini", model, "The google-genai SDK is not installed.") from exc
    timeout_ms = int(float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "90")) * 1000)
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else None,
    )
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout_ms))
    try:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
    except Exception as exc:
        response_obj = getattr(exc, "response", None)
        status = _status(exc)
        headers = getattr(response_obj, "headers", None)
        message = str(exc)
        raise ProviderError("gemini", model, message, status, _retry_seconds(headers, message), headers) from exc
    text = getattr(response, "text", None)
    if not text:
        raise ProviderError("gemini", model, "Gemini returned an empty response.")
    return text, response


def _ollama(prompt: str, model: str, max_tokens: int, json_mode: bool) -> tuple[str, Any]:
    """Call the local Ollama chat API without credentials or hosted fallback."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_predict": min(max_tokens, OLLAMA_MAX_OUTPUT_TOKENS),
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    if json_mode:
        body["format"] = "json"
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", json=body,
            timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # Keep the transport failure visible: a timeout is materially different
        # from a refused connection, and neither proves Ollama is stopped.
        detail = _safe_error_detail(exc)
        message = (f"{type(exc).__name__} after {OLLAMA_REQUEST_TIMEOUT_SECONDS:g}s: {detail}"
                   if detail else f"{type(exc).__name__} after {OLLAMA_REQUEST_TIMEOUT_SECONDS:g}s")
        raise ProviderError("ollama", model, message, error_type=type(exc).__name__) from exc
    if response.status_code >= 400:
        error_type = error_code = None
        message = f"Ollama returned HTTP {response.status_code}."
        try:
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("error"), str):
                message = payload["error"]
        except ValueError:
            pass
        raise ProviderError("ollama", model, message, response.status_code, headers=response.headers,
                            error_type=error_type, error_code=error_code)
    try:
        payload = response.json()
        content = payload["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError("ollama", model, "Ollama returned an incomplete chat response.", response.status_code) from exc
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("ollama", model, "Ollama returned an empty response.", response.status_code)
    return content, response


def _quota_exhausted(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(term in message for term in (
        "per day", "perday", "daily quota", "quota exhausted", "quota exceeded", "quota_exhausted",
        "insufficient_quota", "tokens per day", "requests per day",
    ))


def _safe_failure(exc: Exception, provider: str) -> str:
    status = _status(exc)
    if provider == "ollama":
        if status == 404:
            return f"Ollama model '{getattr(exc, 'model', OLLAMA_MODEL)}' is unavailable locally. Confirm Ollama is running and the model is installed."
        if isinstance(exc, ProviderError) and exc.error_type in {"Timeout", "ConnectTimeout", "ReadTimeout"}:
            return f"Ollama request timed out at {OLLAMA_BASE_URL} ({exc.error_type}). The local model may need more time to answer."
        if isinstance(exc, ProviderError) and exc.error_type in {"ConnectionError", "ConnectionRefusedError"}:
            detail = _safe_error_detail(exc, 180)
            return f"Could not connect to Ollama at {OLLAMA_BASE_URL}: {detail or exc.error_type}. Confirm the local service is listening."
        if status:
            return f"Ollama local request failed (HTTP {status})."
        if isinstance(exc, ProviderError) and exc.error_type:
            detail = _safe_error_detail(exc, 180)
            return f"Ollama local request failed ({exc.error_type}): {detail or 'check the local service and request.'}"
        return "Ollama local request failed. Check the local service and response format."
    if status == 200 and any(term in str(exc).lower() for term in ("malformed json response envelope", "incomplete response envelope", "non-text response envelope")):
        return f"{provider.title()} returned an invalid success response envelope (HTTP 200)."
    if status == 429:
        return f"{provider.title()} rate limit or quota exhausted (HTTP 429)."
    if status == 401 and provider == "groq":
        return (
            "Groq rejected GROQ_API_KEY (HTTP 401). Create or rotate the key in the Groq Console, "
            "replace GROQ_API_KEY in the project-root .env file, then restart ForgeAI."
        )
    if status == 401 and provider == "gemini":
        return (
            "Gemini rejected GEMINI_API_KEY (HTTP 401). Check or replace the key in the project-root .env file, "
            "then restart ForgeAI."
        )
    if status == 403 and provider == "groq":
        return "Groq denied access (HTTP 403). Check the account, key permissions, and configured model."
    if status in {401, 403}:
        return f"{provider.title()} credentials or account access rejected (HTTP {status})."
    if status == 402:
        return f"{provider.title()} account billing rejected (HTTP 402)."
    if status in {400, 404}:
        return f"{provider.title()} rejected the configured model or request (HTTP {status})."
    if status and status >= 500:
        return f"{provider.title()} is temporarily unavailable (HTTP {status})."
    if isinstance(exc, ImportError):
        return f"{provider.title()} SDK is unavailable."
    if isinstance(exc, ProviderError) and "missing or empty" in str(exc).lower():
        return f"{provider.title()} API key is not configured."
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return f"{provider.title()} network request failed or timed out."
    return f"{provider.title()} request failed ({type(exc).__name__})."


def _response_complete(raw: Any) -> bool:
    """Reject provider responses explicitly marked as truncated from the cache."""
    try:
        if hasattr(raw, "json"):
            payload = raw.json()
            if isinstance(payload, dict) and "done" in payload:
                if payload.get("done") is False:
                    return False
                reason = str(payload.get("done_reason", "")).lower()
                return reason not in {"length", "max_tokens"}
            finish = payload["choices"][0].get("finish_reason")
            return finish not in {"length", "max_tokens"}
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        pass
    candidates = getattr(raw, "candidates", None) or []
    if candidates:
        finish = str(getattr(candidates[0], "finish_reason", "")).upper()
        if "MAX_TOKENS" in finish:
            return False
    return True


def _finish_reason(raw: Any) -> str | None:
    """Return provider termination metadata when the response exposes it."""
    try:
        if hasattr(raw, "json"):
            payload = raw.json()
            if isinstance(payload, dict):
                reason = payload.get("done_reason")
                if reason:
                    return str(reason)
                choices = payload.get("choices") or []
                if choices and choices[0].get("finish_reason") is not None:
                    return str(choices[0]["finish_reason"])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        pass
    candidates = getattr(raw, "candidates", None) or []
    if candidates and getattr(candidates[0], "finish_reason", None) is not None:
        return str(candidates[0].finish_reason)
    return None


def generate_with_backoff(
    prompt: str,
    preferred_model: str | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    json_mode: bool = False,
    on_model_switch: Callable[[str], None] | None = None,
    max_tokens: int = 2000,
    step: str = "codegen",
    cache_context: Any = None,
) -> LLMResponse:
    """Call Groq first and fall back to Gemini for quota/temporary failures."""
    key_names = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
    providers = _provider_order()
    configured = [provider for provider in providers if provider == "ollama" or os.getenv(key_names[provider], "").strip()]
    errors: list[str] = []
    max_retries = min(3, max(0, int(max_retries)), max(0, int(GROQ_MAX_RETRIES)))
    attempts_left = max_retries
    fallback_counted = False

    with _call_lock:
        models = {provider: _provider_model(provider, step, preferred_model) for provider in providers}
        # Search successful provider-specific caches before making any request.
        # This allows a prior Gemini fallback result to survive process restarts
        # without first spending another Groq request.
        for provider in providers:
            model = models[provider]
            provider_max_tokens = _provider_output_limit(provider, max_tokens)
            cached = _cache_read(provider, model, prompt, max_tokens=provider_max_tokens, json_mode=json_mode,
                                 step=step, context=cache_context)
            if cached is not None:
                logger.info("LLM cache hit provider=%s model=%s prompt_chars=%d", provider, model, len(prompt))
                with _stats_lock:
                    _call_metrics["cache_hits"] += 1
                    if provider == "gemini" and "groq" in configured:
                        _call_metrics["fallback_activations"] += 1
                    _call_stats.append({"provider": provider, "model": model, "estimated_tokens": 0,
                                        "wait_seconds": 0, "cached": True})
                return LLMResponse(cached, provider, model)

        for provider in providers:
            model = models[provider]
            provider_max_tokens = _provider_output_limit(provider, max_tokens)
            if provider not in configured:
                continue
            cooldown_until = _cooldown.get(provider, 0.0)
            if cooldown_until > time.monotonic():
                errors.append(f"{provider.title()} is cooling down; skipping another request.")
                continue
            if provider != configured[0] and not fallback_counted:
                with _stats_lock:
                    _call_metrics["fallback_activations"] += 1
                fallback_counted = True

            # Conservative local estimate: prompt characters/4 plus the full
            # completion reservation. Provider token accounting remains authoritative.
            estimate = len(prompt) // 4 + provider_max_tokens
            wait_seconds = 0.0
            if provider == "groq":
                try:
                    wait_seconds = _throttle(provider, _token_budget(provider, model), estimate)
                except LocalBudgetError as exc:
                    # The local guard is not a Groq rejection. If Gemini is
                    # configured, let the established fallback policy handle
                    # this request without spending a Groq call.
                    errors.append(str(exc))
                    logger.info("Skipping Groq before request: local estimated token budget exceeded")
                    continue
            if on_model_switch:
                on_model_switch(f"Using {provider.title()} {model}")

            provider_call = {"groq": _groq, "gemini": _gemini, "ollama": _ollama}[provider]
            json_retry_used = False
            while True:
                started = time.monotonic()
                logger.info("LLM request start provider=%s model=%s stage=%s prompt_chars=%d max_tokens=%d",
                            provider, model, step, len(prompt), provider_max_tokens)
                try:
                    if provider == "groq":
                        wait_seconds += _throttle_requests(provider)
                    with _stats_lock:
                        _call_metrics["provider_attempts"] += 1
                    text, raw = provider_call(prompt, model, provider_max_tokens, json_mode)
                    elapsed = time.monotonic() - started
                    logger.info("LLM request complete provider=%s model=%s stage=%s elapsed_seconds=%.2f "
                                "response_chars=%d complete=%s finish_reason=%s",
                                provider, model, step, elapsed, len(text or ""), _response_complete(raw),
                                _safe_error_detail(_finish_reason(raw), 80))
                    cacheable = bool(text and text.strip() and _response_complete(raw))
                    if cacheable and not json_mode:
                        _cache_write(provider, model, prompt, text, max_tokens=provider_max_tokens,
                                     json_mode=json_mode, step=step, context=cache_context)
                    with _stats_lock:
                        _call_stats.append({"provider": provider, "model": model, "estimated_tokens": estimate,
                                            "wait_seconds": round(wait_seconds, 2), "elapsed_seconds": round(elapsed, 2), "cached": False})
                    cache_callback = None
                    if json_mode and cacheable:
                        cache_callback = lambda: _cache_write(
                            provider, model, prompt, text, max_tokens=provider_max_tokens,
                            json_mode=True, step=step, context=cache_context,
                        )
                    return LLMResponse(text, provider, model, retries=max_retries - attempts_left,
                                       raw=raw, cache_callback=cache_callback)
                except Exception as exc:
                    with _stats_lock:
                        _call_metrics["provider_failures"] += 1
                    elapsed = time.monotonic() - started
                    _log_failure(provider, model, prompt, provider_max_tokens, exc,
                                 stage=step, elapsed=elapsed)
                    with _stats_lock:
                        _call_stats.append({"provider": provider, "model": model, "estimated_tokens": estimate,
                                            "wait_seconds": round(wait_seconds, 2), "elapsed_seconds": round(elapsed, 2),
                                            "cached": False, "status": _status(exc)})

                    status = _status(exc)
                    validation_error = any(phrase in str(exc).lower() for phrase in ("failed to validate json", "failed to generate json"))
                    if provider == "groq" and status == 400 and json_mode and validation_error and not json_retry_used:
                        json_retry_used = True
                        if on_model_switch:
                            on_model_switch(f"Groq JSON validation failed for {step}; retrying once with prompt-only JSON formatting.")
                        try:
                            _throttle_requests("groq")
                            with _stats_lock:
                                _call_metrics["provider_attempts"] += 1
                            text, raw = _groq(prompt, model, max_tokens, False)
                            cacheable = bool(text and text.strip() and _response_complete(raw))
                            cache_callback = None
                            if json_mode and cacheable:
                                cache_callback = lambda: _cache_write(
                                    provider, model, prompt, text, max_tokens=max_tokens,
                                    json_mode=True, step=step, context=cache_context,
                                )
                            elif cacheable:
                                _cache_write(provider, model, prompt, text, max_tokens=max_tokens,
                                             json_mode=False, step=step, context=cache_context)
                            return LLMResponse(text, provider, model, retries=1, raw=raw,
                                               cache_callback=cache_callback)
                        except Exception as retry_exc:
                            with _stats_lock:
                                _call_metrics["provider_failures"] += 1
                            exc = retry_exc
                            status = _status(exc)

                    # A provider-specific request/model rejection should not
                    # prevent a configured secondary provider from serving the
                    # request. Authentication and billing failures remain
                    # visible, except that a rejected Groq key may use Gemini.
                    if status in {400, 401, 402, 403, 404}:
                        retryable_groq_json_failure = status == 400 and json_retry_used
                        rejected_groq_key = status == 401
                        if provider == "groq" and (retryable_groq_json_failure or rejected_groq_key) and "gemini" in configured:
                            errors.append(_safe_failure(exc, provider))
                            break
                        with _stats_lock:
                            _call_metrics["generation_failures"] += 1
                        raise RuntimeError(_safe_failure(exc, provider)) from exc

                    if status == 429:
                        quota = _quota_exhausted(exc)
                        retry_after = getattr(exc, "retry_after", None)
                        if not quota and retry_after is not None and attempts_left > 0 and 0 < retry_after <= max_delay:
                            attempts_left -= 1
                            time.sleep(max(base_delay, retry_after))
                            continue
                        reason = "quota exhausted" if quota else "rate limit"
                        errors.append(f"{provider.title()} {reason} (HTTP 429).")
                        if provider == "groq" and not os.getenv("GEMINI_API_KEY", "").strip():
                            errors.append("Gemini fallback is unavailable because GEMINI_API_KEY is not configured.")
                        cooldown_seconds = retry_after if retry_after is not None else (3600.0 if quota else 5.0)
                        _cooldown[provider] = time.monotonic() + max(1.0, cooldown_seconds)
                        break

                    # Service/network failures can be transient. Retry within the
                    # caller's explicit budget, then move to the secondary provider.
                    provider_message = str(exc).lower()
                    provider_unavailable = isinstance(exc, ProviderError) and any(
                        marker in provider_message for marker in ("sdk is not installed", "missing or empty", "empty response")
                    )
                    if provider == "ollama" and isinstance(exc, ProviderError):
                        provider_unavailable = True
                    transient = (not provider_unavailable) and (
                        status is None or status >= 500 or isinstance(exc, (requests.Timeout, requests.ConnectionError))
                    )
                    if transient and provider != "ollama" and attempts_left > 0:
                        attempts_left -= 1
                        time.sleep(min(max_delay, base_delay * (2 ** (max_retries - attempts_left - 1))))
                        continue
                    errors.append(_safe_failure(exc, provider))
                    break

        if not configured:
            errors.insert(0, "No LLM API keys are configured.")
        with _stats_lock:
            _call_metrics["generation_failures"] += 1
        raise RuntimeError(" ".join(errors) or "No configured LLM provider was available.")
