"""TODO: Turn a spec into a sequence of build steps."""

import json
import re
from pathlib import Path
from typing import Any

from agent.llm_client import generate_with_backoff
from config import GEMINI_MODEL_SPEC

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "planner_prompt.txt"


def load_prompt_template() -> str:
    """Load the planning prompt template from disk."""
    if PROMPT_TEMPLATE_PATH.exists():
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    return """You are a senior full-stack engineer.

Turn the following application specification into an ordered list of build steps.

Spec JSON:
{spec_json}

Return ONLY valid JSON.

The JSON must be one of these shapes:
1. {"steps": ["step 1", "step 2", ...]}
2. ["step 1", "step 2", ...]

Rules:
- Output JSON only; no markdown fences, no prose, no comments.
- The list must be ordered in the best implementation sequence.
- Include steps such as: create project folders, install dependencies, create backend files, define models/data, add routes, create UI pages, run the app.
- Keep each step concise but actionable.
- Ensure the result is valid JSON parseable by json.loads().
"""


def build_plan_prompt(spec: dict[str, Any] | str) -> str:
    """Render the planner prompt for a spec object or JSON string."""
    payload = spec if isinstance(spec, str) else json.dumps(spec, indent=2)
    template = load_prompt_template()
    return template.replace("{spec_json}", payload)


def _extract_json_collection(raw_text: str) -> Any:
    """Extract JSON from Gemini output even if wrapped in markdown or prose."""
    cleaned = (raw_text or "").strip()
    if not cleaned:
        raise ValueError("Gemini returned an empty response.")

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if cleaned.lower().startswith("json") and "\n" in cleaned:
        cleaned = cleaned.split("\n", 1)[1].strip()

    starts = [position for position in (cleaned.find("{"), cleaned.find("[")) if position >= 0]
    if not starts:
        raise ValueError("Model response did not contain a JSON object or array.")
    start = min(starts)

    try:
        payload, _end = json.JSONDecoder().raw_decode(cleaned[start:])
        return payload
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Unable to parse JSON from model output at line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc


def parse_plan_steps(raw_text: str) -> list[str]:
    """Normalize Gemini output into a list of ordered build steps."""
    payload = _extract_json_collection(raw_text)

    if isinstance(payload, list):
        steps = payload
    elif isinstance(payload, dict):
        for key in ("steps", "ordered_steps", "build_steps", "plan"):
            if key in payload and isinstance(payload[key], list):
                steps = payload[key]
                break
        else:
            raise ValueError("Model output did not contain a recognized list of steps.")
    else:
        raise ValueError("Model output was not a JSON list or object with a steps array.")

    if not isinstance(steps, list) or not all(isinstance(item, str) for item in steps):
        raise ValueError("Parsed build steps must be a list of strings.")

    return steps


def _normalize_model_error(exc: Exception | str) -> str:
    """Turn Gemini or JSON failures into a clear message for the UI log."""
    message = str(exc or "Unknown error").strip() or "Unknown error"
    lowered = message.lower()
    if "rate limit" in lowered or "429" in lowered or "quota" in lowered or "resourceexhausted" in lowered:
        return "Gemini API rate limit reached. Please wait a moment and try again."
    if any(token in lowered for token in ["invalid api key", "api key", "authentication", "401", "403", "forbidden"]):
        return "Gemini API key is missing or invalid. Check GEMINI_API_KEY in your .env file."
    if "json" in lowered and "parse" in lowered:
        return f"Model returned invalid JSON: {message}"
    return message


DEFAULT_MODEL_CANDIDATES = [GEMINI_MODEL_SPEC]


def _call_gemini_with_fallback(prompt: str, preferred_model: str, on_model_switch: Any = None) -> Any:
    """Keep planner calls on the shared Groq-primary routing path."""
    return generate_with_backoff(prompt, preferred_model=preferred_model, json_mode=True,
                                 on_model_switch=on_model_switch, step="planner",
                                 cache_context={"stage": "build_plan"})


def generate_plan(spec: dict[str, Any] | str, model_name: str = GEMINI_MODEL_SPEC, on_model_switch: Any = None) -> list[str]:
    """Ask Gemini to return an ordered set of app-building steps from a spec."""
    prompt = build_plan_prompt(spec)

    try:
        response = _call_gemini_with_fallback(prompt, model_name, on_model_switch=on_model_switch)
        response_text = getattr(response, "text", None)
        if not response_text:
            raise ValueError("Gemini returned no content for the planning prompt.")
        if not getattr(response, "complete", True):
            raise ValueError("Provider marked the planning response as truncated.")
        steps = parse_plan_steps(response_text)
        if hasattr(response, "cache_validated"):
            response.cache_validated()
        return steps
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Plan generation failed: {_normalize_model_error(exc)}") from exc
    except Exception as exc:  # pragma: no cover - surfaced in runtime path
        error_msg = _normalize_model_error(exc)
        if "not found" in error_msg.lower() or "not supported" in error_msg.lower():
            error_msg += " The model is unavailable for your account. Please use the exact model name shown in Google AI Studio."
        raise RuntimeError(f"Gemini request failed: {error_msg}") from exc
