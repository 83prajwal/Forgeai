"""TODO: Generate app files for each planned component."""

import json
import re
import copy
import logging
from pathlib import Path
from typing import Any

from agent.design_profiles import build_design_guidance
from agent.llm_client import generate_with_backoff

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "codegen_prompt.txt"
logger = logging.getLogger("forgeai.codegen")
_PREVIEW_LIMIT = 180


class IncompleteGenerationError(ValueError):
    """The provider marked a file response incomplete, so its text is unusable."""


def _safe_response_preview(text: str) -> str:
    """Return a bounded, structure-only preview that cannot expose generated content."""
    preview = re.sub(r"(?u)\w+", "[TEXT]", text[:_PREVIEW_LIMIT])
    return preview.encode("unicode_escape", errors="replace").decode("ascii")[:_PREVIEW_LIMIT]


def _response_metadata(response: Any) -> tuple[str | None, str | None, int | None, str | None]:
    """Read only metadata explicitly exposed by the provider response object."""
    provider = getattr(response, "provider", None)
    model = getattr(response, "model", None)
    raw = getattr(response, "raw", None)
    status = getattr(raw, "status_code", None)
    finish = None
    try:
        if callable(getattr(raw, "json", None)):
            payload = raw.json()
            if isinstance(payload, dict) and "done_reason" in payload:
                finish = payload.get("done_reason")
            else:
                choice = payload.get("choices", [{}])[0]
                finish = choice.get("finish_reason") if isinstance(choice, dict) else None
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        pass
    if finish is None:
        candidates = getattr(raw, "candidates", None) or []
        if candidates:
            finish = getattr(candidates[0], "finish_reason", None)
    if finish is not None:
        finish = str(getattr(finish, "name", finish))
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    return provider, model, status, finish


def _provider_error_metadata(exc: BaseException) -> tuple[str | None, str | None, int | None]:
    """Find structured provider metadata in an exception chain without logging its text."""
    current: BaseException | None = exc
    while current is not None:
        provider = getattr(current, "provider", None)
        model = getattr(current, "model", None)
        status = getattr(current, "status", None)
        if provider or model or status is not None:
            try:
                status = int(status) if status is not None else None
            except (TypeError, ValueError):
                status = None
            return provider, model, status
        current = current.__cause__ or current.__context__
    return None, None, None


def load_prompt_template() -> str:
    """Load the code generation prompt template from disk."""
    if PROMPT_TEMPLATE_PATH.exists():
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    return """You are a senior Python and web app engineer.

Generate production-ready file contents for a web application.

Full app specification:
{spec_json}

Current component to generate:
{component_name}

Purpose:
{component_purpose}

Requirements:
- Return valid JSON only.
- The JSON must have exactly this structure:
  {
    "filename": "string",
    "content": "string"
  }
- The content must be the full file contents for the component.
- Use Flask for the backend, SQLite for persistence, and HTML/CSS/JS for frontend pages.
- When the current component is requirements.txt, return a valid pip requirements file listing Flask and every third-party Python package imported by the generated backend. Do not list Python standard-library modules.
- Keep dependencies minimal and use Python's built-in sqlite3 module for SQLite unless the spec explicitly requires an ORM.
- In the Flask entrypoint, bind using host="0.0.0.0" and port=int(os.environ.get("PORT", 5000)); do not hard-code a different port. Disable Flask debug mode and the reloader.
- Use Flask APIs supported by current Flask releases. Do not use `before_first_request` or `before_app_first_request`; these hooks are not available in current Flask.
- Return the exact filename named under Current component to generate; do not rename it or create additional files.
- Put all user interface markup in index.html. Serve it from the Flask root route with send_from_directory, and do not reference ungenerated Jinja templates such as login.html.
- Ensure the backend code matches the routes and entities described in the spec.
- Keep file names logical and consistent with the app.
- Do not include markdown fences or explanation outside the JSON object.
- The content must be parseable and ready to write to disk.
"""


def build_codegen_prompt(
    spec: dict[str, Any] | str,
    component_name: str,
    component_purpose: str,
    context_files: list[dict[str, str]] | None = None,
    user_prompt: str = "",
) -> str:
    """Render the prompt template for an individual file generation request."""
    if isinstance(spec, str):
        compact_spec: dict[str, Any] = json.loads(spec)
    else:
        compact_spec = copy.deepcopy(spec)
    # Keep each request scoped to the data needed for the requested file.
    keys = {"models.py": ("app_name", "purpose", "entities", "relationships", "database_requirements", "assumptions"),
            "routes.py": ("app_name", "purpose", "core_features", "entities", "relationships", "validation_requirements", "error_handling_requirements", "database_requirements", "assumptions"),
            "index.html": ("app_name", "purpose", "theme", "core_features", "user_roles", "entities", "pages", "navigation", "assumptions"),
            "style.css": ("app_name", "purpose", "theme", "pages", "responsive_requirements"),
            "script.js": ("app_name", "purpose", "theme", "core_features", "entities", "pages", "navigation", "dashboard_kpis", "validation_requirements", "error_handling_requirements", "assumptions")}
    relevant = {key: compact_spec[key] for key in keys.get(component_name, tuple(compact_spec)) if key in compact_spec}
    payload = json.dumps(relevant, separators=(",", ":"), ensure_ascii=False)
    template = load_prompt_template()
    # routes.py's API contract already carries the schema and response shapes;
    # appending an unrelated requirements file or a truncated source file adds
    # tokens without preserving a useful interface.
    prompt_context = [] if component_name == "routes.py" else (context_files or [])[:1]
    context = "\n".join(f"--- {item['filename']} ---\n{item['content'][:1200]}" for item in prompt_context) or "(None)"
    contract = spec.get("api_contract", {}) if isinstance(spec, dict) else {}
    route_names = [f"{item.get('method')} {item.get('path')}" for item in contract.get("endpoints", []) if isinstance(item, dict)]
    contract_for_file = contract if component_name == "routes.py" else {}
    design_guidance = build_design_guidance(compact_spec, user_prompt)
    return (
        template.replace("{spec_json}", payload)
        .replace("{component_name}", component_name)
        .replace("{component_purpose}", component_purpose)
        + "\n\nROUTES:\n"
        + json.dumps(route_names if not contract_for_file else [], separators=(",", ":"))
        + ("\nAPI CONTRACT (follow exactly):\n" + json.dumps(contract_for_file, separators=(",", ":")) if contract_for_file else "")
        + (
            "\n\nORIGINAL USER REQUEST (preserve its product intent and wording):\n"
            + user_prompt[:1200]
            + "\n\nAPPLICATION-SPECIFIC DESIGN AND FUNCTIONALITY BRIEF (deterministic guidance; do not add model calls):\n"
            + design_guidance
            if component_name != "routes.py" else ""
        )
        + "\n\nPreviously generated files (keep interfaces consistent):\n"
        + context
        + (
            "\n\nHard rules: use sqlite3 only; do not use an ORM unless explicitly present in the spec. "
            "Never use lambdas as route handlers or define an API route at `/` (the entrypoint reserves `/` for the UI). "
            "Build the app the prompt actually describes, with the requested domain-specific features; do not reduce it to a generic records list. "
            "Every API route returns JSON and uses correct HTTP status codes. "
            "Do not invent endpoint paths or request/response shapes outside the API contract."
            if component_name not in {"models.py", "routes.py"} else ""
        )
        + (
            "\n\nThis is a page module, not a whole application. Return only JavaScript that registers exactly one renderer using "
            "window.ForgeAI.registerPage('<page-slug>', async function(ctx) { ... }). The shared shell exposes ctx.root, "
            "ctx.profile, ctx.api(path, options), and ctx.search. Render this page inside ctx.root, use only exact API contract paths, "
            "and do not replace the sidebar, top bar, route handler, dashboard, or settings page. The module is merged into script.js "
            "after the shared router, so register synchronously at script evaluation time."
            if component_name.startswith("page_") and component_name.endswith(".js") else ""
        )
        + (
            "\n\nThe script.js file is the shared client shell and navigation controller. Define a window.ForgeAI.registerPage(slug, renderer) registry. "
            "On hash-route changes, call the registered page renderer with {root, profile, api, search}; retain the dashboard and settings routes. "
            "Entity page renderers are generated separately and merged after this shared file."
            if component_name == "script.js" else ""
        )
        + (
            "\n\nHTML FILE RULES: Return concise semantic HTML markup only. Do not include <style> blocks, CSS, inline style attributes, or JavaScript; "
            "style.css and script.js are generated as separate files. Link those files with <link rel=\"stylesheet\" href=\"style.css\"> "
            "and <script defer src=\"script.js\"></script>. Include the app shell, navigation targets, page container, and accessible labels. "
            "Do not implement visual styling or application behavior in this file."
            if component_name == "index.html" else ""
        )
    )


def extract_codegen_payload(raw_text: str) -> dict[str, str]:
    """Extract and parse the JSON object returned for a generated file."""
    cleaned = (raw_text or "").strip()
    if not cleaned:
        raise ValueError("Groq returned an empty response.")

    # Do not use a non-greedy fenced-object regex here: source strings commonly
    # contain braces (JS objects, Python dictionaries), which prematurely end it.
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()
    if cleaned.lower().startswith("json") and "\n" in cleaned:
        cleaned = cleaned.split("\n", 1)[1].strip()

    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        in_string = escaped = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    cleaned = cleaned[start : index + 1]
                    break

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Models sometimes put literal line breaks inside the JSON content value.
        # Escape control characters only while inside a quoted JSON string, then retry.
        repaired: list[str] = []
        in_string = False
        escaped = False
        for char in cleaned:
            if in_string and char in "\n\r\t":
                repaired.append({"\n": r"\n", "\r": r"\r", "\t": r"\t"}[char])
                continue
            repaired.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
        try:
            payload = json.loads("".join(repaired))
        except json.JSONDecodeError as repair_exc:
            # Report the final decoder location; the response body may contain
            # generated code or user supplied content and should not be logged.
            error = repair_exc if repair_exc.pos >= exc.pos else exc
            raise ValueError(
                "Unable to parse generated file JSON: "
                f"{error.msg} at line {error.lineno} column {error.colno} (char {error.pos})."
            ) from error

    if not isinstance(payload, dict):
        raise ValueError("Generated content must be a JSON object.")

    if "filename" not in payload or "content" not in payload:
        raise ValueError("Generated file payload must contain filename and content keys.")

    if not isinstance(payload["filename"], str) or not isinstance(payload["content"], str):
        raise ValueError("Generated file payload filename/content fields must be strings.")
    if not payload["filename"].strip() or not payload["content"].strip():
        raise ValueError("Generated file payload filename and content must not be empty.")

    return {"filename": payload["filename"], "content": _undo_double_escaping(payload["content"])}


def _undo_double_escaping(content: str) -> str:
    """Repair content a model JSON-escaped twice, which arrives as one line of literal \\n sequences."""
    if "\n" in content.strip() or "\\n" not in content:
        return content
    try:
        decoded = json.loads(f'"{content}"')
    except json.JSONDecodeError:
        return content
    return decoded if isinstance(decoded, str) and "\n" in decoded else content


def _normalize_model_error(exc: Exception | str) -> str:
    """Convert Gemini and parsing failure details into clear UI-friendly text."""
    message = str(exc or "Unknown error").strip() or "Unknown error"
    lowered = message.lower()
    if any(code in lowered for code in ("http 401", "http 402", "http 403")):
        return message
    if "groq http 429" in lowered or "groq rate limit (http 429)" in lowered:
        detail = message.split("Groq HTTP 429:", 1)[-1].strip() if "Groq HTTP 429:" in message else message
        return f"Groq API rate limit reached (HTTP 429): {detail}"
    if "generate_content_free_tier_requests" in lowered and "perday" in lowered:
        limit = re.search(r"limit:\s*(\d+)", message, re.IGNORECASE)
        retry = re.search(r"retry in\s+([^\s.]+)", message, re.IGNORECASE)
        reason = f"Gemini free-tier daily request quota exhausted ({limit.group(1)} requests/day)" if limit else "Gemini free-tier daily request quota exhausted"
        if retry:
            reason += f"; provider retry window {retry.group(1)}"
        if "groq failed" in lowered and "429" in lowered:
            reason += "; Groq also returned HTTP 429"
        return reason
    if "http 429" in lowered or "(429)" in lowered:
        return message
    if "ollama request timed out" in lowered:
        return message
    if any(token in lowered for token in ("timeout", "timed out", "deadlineexceeded", "deadline exceeded")):
        return "Live generation timed out while waiting for the model provider."
    if any(token in lowered for token in ["invalid api key", "api key", "authentication", "401", "403", "forbidden"]):
        return "Groq authentication failed (HTTP 401/403). Check GROQ_API_KEY in .env."
    if "json" in lowered and any(token in lowered for token in ("parse", "decode", "valid json")):
        return f"Invalid model JSON: {message}"
    if "413" in lowered or "payload too large" in lowered:
        return "Live model request rejected: prompt payload is too large (HTTP 413)."
    if any(token in lowered for token in ("connection", "name resolution", "dns", "network unavailable")):
        return "Live model request failed: provider network is unavailable."
    return message


# Use the shared gemini client wrapper (generate_with_backoff)


def generate_file_for_component(
    spec: dict[str, Any] | str,
    component_name: str,
    component_purpose: str,
    model_name: str = "openai/gpt-oss-120b",
    context_files: list[dict[str, str]] | None = None,
    on_model_switch: Any = None,
    user_prompt: str = "",
) -> dict[str, str]:
    """Generate the content for a single file component using the full spec as context."""
    prompt = build_codegen_prompt(spec, component_name, component_purpose, context_files=context_files, user_prompt=user_prompt)
    # The shared client tries the preferred model and the next models in its
    # configured chain. Avoid wrapping that chain in another three-call loop:
    # nested retries multiplied provider traffic and delayed replay for minutes.
    output_reserves = {
        "routes.py": 3500,
        "models.py": 3000,
        "index.html": 3000,
        "style.css": 3000,
        "script.js": 4500,
    }
    max_tokens = output_reserves.get(component_name, 2500 if component_name.startswith("page_") else 3500)
    incomplete_retry = False
    try:
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                if incomplete_retry:
                    request_prompt += (
                        "\n\nBOUNDED RECOVERY: The prior response reached its output limit. Return a more concise, "
                        "complete implementation without comments, docstrings, decorative markup, or repeated helpers. "
                        "Preserve all required behavior and every API contract endpoint. Do not omit functionality. "
                        'Return one JSON object with string keys "filename" and "content"; escape all newlines, quotes, '
                        "and backslashes in content."
                    )
                else:
                    request_prompt += (
                        "\n\nRETRY: Your previous response could not be parsed as JSON. Return one complete JSON object only, "
                        'with string keys "filename" and "content". JSON-escape every newline, quote, and backslash inside '
                        "content. Do not use Markdown fences or commentary, and do not truncate the file."
                    )
            response = generate_with_backoff(
                request_prompt, preferred_model=model_name, max_retries=3,
                json_mode=True, on_model_switch=on_model_switch,
                max_tokens=max_tokens, step="codegen",
                cache_context={"component": component_name, "purpose": component_purpose},
            )
            response_text = getattr(response, "text", None)
            try:
                if not getattr(response, "complete", True):
                    _provider, _model, _status, finish = _response_metadata(response)
                    termination = f"finish_reason={finish}" if finish else "finish_reason unavailable"
                    raise IncompleteGenerationError(
                        f"Incomplete model output for {component_name} ({termination})."
                    )
                if not response_text:
                    raise ValueError("Model returned an empty response for the generated file.")
                payload = extract_codegen_payload(response_text)
                if payload["filename"] != component_name:
                    # The caller requested one known component and always
                    # assembles its content under that name. Treat the model's
                    # redundant filename field as untrusted metadata, so a
                    # spelling error cannot abort an otherwise valid file or
                    # redirect output to a different path.
                    logger.warning(
                        "Codegen returned a filename mismatch provider=%s model=%s stage=codegen "
                        "component=%s; using requested filename",
                        getattr(response, "provider", None), getattr(response, "model", None), component_name,
                    )
                    payload["filename"] = component_name
                if component_name.endswith(".py"):
                    compile(payload["content"], component_name, "exec")
                if hasattr(response, "cache_validated"):
                    response.cache_validated()
                return payload
            except (ValueError, TypeError, SyntaxError, json.JSONDecodeError) as exc:
                pos = getattr(exc, "pos", None)
                provider, response_model, status, finish = _response_metadata(response)
                preview = _safe_response_preview(response_text) if isinstance(response_text, str) else ""
                logger.warning(
                    "Codegen response validation failed provider=%s model=%s status=%s finish_reason=%s "
                    "stage=codegen component=%s response_chars=%s error_type=%s position=%s preview=%r",
                    provider, response_model, status, finish,
                    component_name, len(response_text) if isinstance(response_text, str) else 0,
                    type(exc).__name__, pos, preview,
                )
                if attempt:
                    if isinstance(exc, IncompleteGenerationError):
                        raise IncompleteGenerationError(
                            f"{exc} Bounded recovery exhausted after 2 attempts; reduce the requested file scope."
                        ) from exc
                    raise
                incomplete_retry = isinstance(exc, IncompleteGenerationError)
        raise ValueError("Model returned invalid JSON twice.")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Code generation failed for {component_name}: {_normalize_model_error(exc)}") from exc
    except Exception as exc:  # surfaced in runtime path with its provider cause
        provider, response_model, status = _provider_error_metadata(exc)
        logger.warning(
            "Codegen provider request failed provider=%s model=%s status=%s stage=codegen component=%s",
            provider, response_model, status, component_name,
        )
        error_msg = _normalize_model_error(exc)
        if "not found" in error_msg.lower() or "not supported" in error_msg.lower():
            error_msg += " Check the model ID and permissions in Groq Console."
        raise RuntimeError(f"Model request failed for {component_name}: {error_msg}") from exc
