"""Simple Gemini client wrapper adding retries, backoff, and jitter.

This centralizes API key configuration and retry behavior so callers
don't accidentally flood the model and hit rate limits.
"""
import time
import random
import threading
import os
import re
from datetime import datetime
from typing import Any

import google.generativeai as genai

from config import get_gemini_api_key


DEFAULT_MODEL_CANDIDATES = ["gemini-3.6-flash", "gemini-3.5-flash"]


class TokenBucket:
    """A simple token-bucket rate limiter.

    capacity: maximum tokens stored
    refill_rate: tokens added per second
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0, timeout: float | None = None) -> bool:
        """Block until tokens are available or timeout occurs. Returns True if acquired."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.try_acquire(tokens):
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def peek(self) -> float:
        """Return current token count (approximate)."""
        with self._lock:
            self._refill()
            return float(self._tokens)


# Shared token bucket: allow up to 6 requests per minute by default
# (refill_rate tokens per second = tokens_per_minute / 60)
_DEFAULT_TOKENS_PER_MINUTE = max(1, int(os.getenv("LLM_REQUESTS_PER_MINUTE", "15")))
token_bucket = TokenBucket(capacity=_DEFAULT_TOKENS_PER_MINUTE, refill_rate=_DEFAULT_TOKENS_PER_MINUTE / 60.0)


_LOG_PATH = None


def _ensure_log_path() -> None:
    global _LOG_PATH
    if _LOG_PATH is not None:
        return
    try:
        from pathlib import Path

        base = Path(__file__).resolve().parents[2] / "backend" / "logs"
        base.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = base / "gemini_calls.log"
    except Exception:
        _LOG_PATH = None


def _log_call(model_name: str, prompt_summary: str, attempt: int, status: str, reason: str = "") -> None:
    _ensure_log_path()
    entry = f"{datetime.utcnow().isoformat()}Z\tmodel={model_name}\tattempt={attempt}\tstatus={status}\treason={reason}\tsummary={prompt_summary}\n"
    if _LOG_PATH is not None:
        try:
            with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(entry)
        except Exception:
            # Best-effort logging; do not raise
            pass


def _prompt_summary(prompt: str, max_len: int = 200) -> str:
    s = " ".join(prompt.split())
    return (s[:max_len] + "...") if len(s) > max_len else s


def _rate_limit_reason(message: str) -> str:
    lowered = message.lower()
    if "generate_content_free_tier_requests" in lowered and "perday" in lowered:
        limit = re.search(r"limit:\s*(\d+)", message, re.IGNORECASE)
        retry = re.search(r"retry in\s+([^\s.]+)", message, re.IGNORECASE)
        reason = f"Gemini free-tier daily request quota exhausted ({limit.group(1)} requests/day)" if limit else "Gemini free-tier daily request quota exhausted"
        if retry:
            reason += f"; provider retry window {retry.group(1)}"
        return reason
    if "perminute" in lowered or "requests_per_minute" in lowered:
        return "Gemini requests-per-minute quota reached (HTTP 429)"
    return "Gemini rate limit or request quota reached (HTTP 429)"


def get_debug_info(last_lines: int = 100) -> dict:
    """Return debug information: whether a GEMINI key exists, token-bucket state, and last log lines.

    This is a local-only diagnostic helper used by the developer to verify runtime state.
    """
    info: dict = {}
    try:
        from config import GEMINI_API_KEY
    except Exception:
        GEMINI_API_KEY = None

    info["gemini_key_present"] = bool(GEMINI_API_KEY)
    try:
        info["tokens_available"] = token_bucket.peek()
    except Exception:
        info["tokens_available"] = None

    _ensure_log_path()
    lines: list[str] = []
    if _LOG_PATH is not None and _LOG_PATH.exists():
        try:
            with open(_LOG_PATH, "r", encoding="utf-8") as fh:
                all_lines = fh.readlines()
            lines = [l.rstrip("\n") for l in all_lines[-last_lines:]]
        except Exception:
            lines = ["<unable to read log>"]

    info["last_log_lines"] = lines
    return info


def generate_with_backoff(
    prompt: str,
    preferred_model: str = "gemini-3.8-flash",
    max_retries: int = 0,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    json_mode: bool = False,
    on_model_switch: Any = None,
    max_output_tokens: int | None = None,
) -> Any:
    """Generate content with exponential backoff on rate-limit errors.

    - Respects an in-process token bucket to limit outgoing request rate.
    - Tries the preferred model first, then falls back to DEFAULT_MODEL_CANDIDATES.
    - On rate-limit-like errors (429, quota, resourceexhausted), retries with
      exponential backoff and jitter up to `max_retries` times per model.
    """
    genai.configure(api_key=get_gemini_api_key())

    first_model = preferred_model or "gemini-3.8-flash"
    candidates = [first_model] + [m for m in DEFAULT_MODEL_CANDIDATES if m != first_model]
    last_exc: Exception | None = None
    # Do not write prompt contents to disk; prompts may contain user supplied secrets.
    summary = "generation request"

    for model_name in candidates:
        if model_name != first_model and on_model_switch:
            on_model_switch(f"Switched model to {model_name}")
        attempt = 0
        while True:
            # Acquire token before making an outgoing request
            got = token_bucket.acquire(timeout=30.0)
            if not got:
                _log_call(model_name, summary, attempt, "throttle_timeout")
                raise RuntimeError("Timed out waiting for token bucket permission to call Gemini.")

            try:
                model = genai.GenerativeModel(model_name)
                generation_config = {}
                if json_mode:
                    generation_config["response_mime_type"] = "application/json"
                if max_output_tokens is not None:
                    generation_config["max_output_tokens"] = max_output_tokens
                generation_config = generation_config or None
                options = {"timeout": float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "90"))}
                resp = model.generate_content(prompt, generation_config=generation_config, request_options=options) if generation_config else model.generate_content(prompt, request_options=options)
                _log_call(model_name, summary, attempt, "ok")
                return resp
            except Exception as exc:
                last_exc = exc
                text = str(exc or "").lower()
                if any(tok in text for tok in ("rate limit", "429", "quota", "resourceexhausted")):
                    _log_call(model_name, summary, attempt, "rate_limited", _rate_limit_reason(str(exc)))
                    if attempt >= max_retries:
                        break
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    delay = delay * (0.8 + random.random() * 0.4)
                    time.sleep(delay)
                    attempt += 1
                    continue
                if any(token in text for token in ("timeout", "timed out")):
                    reason = "timeout"
                elif any(token in text for token in ("invalid api key", "unauthorized", "authentication", "401", "403")):
                    reason = "authentication / API key rejected"
                elif any(token in text for token in ("connection", "name resolution", "dns", "network")):
                    reason = "network unavailable"
                else:
                    reason = f"provider error ({type(exc).__name__})"
                _log_call(model_name, summary, attempt, "error", reason)
                break

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No Gemini model candidates were available.")
