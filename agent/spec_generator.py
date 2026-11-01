"""TODO: Generate a machine-readable spec from a plain-English app description."""

import json
import re
from pathlib import Path
from typing import Any

from agent.llm_client import generate_with_backoff

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "spec_prompt.txt"
DEFAULT_MODEL_CANDIDATES = []


def load_prompt_template() -> str:
    """Load the spec generation prompt template from disk."""
    if PROMPT_TEMPLATE_PATH.exists():
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return """You are generating a JSON application specification.
User request: {user_prompt}

Return ONLY valid JSON with no markdown, no code fences, no commentary.
The JSON object must contain:
- app_name: string
- entities: array of objects with name and fields array of {name, type}
- relationships: array of relationship objects with from, to, type
- auth_required: boolean
- routes_needed: array of strings
- pages_needed: array of strings

Example format:
{
  "app_name": "Example App",
  "entities": [{"name": "User", "fields": [{"name": "id", "type": "string"}]}],
  "relationships": [{"from": "User", "to": "Post", "type": "one-to-many"}],
  "auth_required": true,
  "routes_needed": ["GET /login", "POST /posts"],
  "pages_needed": ["Home", "Login", "Dashboard"]
}

Return only the JSON object.
"""


def build_spec_prompt(user_prompt: str) -> str:
    """Render the prompt template for the user request."""
    template = load_prompt_template()
    return template.replace("{user_prompt}", user_prompt)


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract and parse JSON from model output when wrapped in prose or markdown."""
    cleaned = (raw_text or "").strip()

    if not cleaned:
        raise ValueError("Groq returned an empty response.")

    fenced_match = re.search(r"```[^\r\n]*\r?\n(.*?)```", cleaned, flags=re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    if cleaned.startswith("json") and "\n" in cleaned:
        cleaned = cleaned.split("\n", 1)[1].strip()

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Model response did not contain a JSON object.")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Unable to parse JSON from model output at line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from the model response.")

    return parsed


def _normalize_model_error(exc: Exception | str) -> str:
    """Convert low-level Gemini/JSON failures into a user-friendly error message."""
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
    if any(token in lowered for token in ["invalid api key", "api key", "authentication", "401", "403", "forbidden"]):
        return "Groq API key is missing or invalid. Check GROQ_API_KEY in your .env file."
    if "json" in lowered and "parse" in lowered:
        return f"Model returned invalid JSON: {message}"
    return message





def build_api_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize a generated spec into a concrete endpoint and SQLite contract."""
    entities = spec.get("entities") if isinstance(spec.get("entities"), list) else []
    tables: dict[str, dict[str, Any]] = {}
    for entity in entities:
        if not isinstance(entity, dict) or not entity.get("name"):
            continue
        name = str(entity["name"])
        table = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "items"
        columns = ["id"]
        for field in entity.get("fields", []):
            if isinstance(field, dict) and field.get("name"):
                column = re.sub(r"[^a-z0-9_]+", "_", str(field["name"]).lower()).strip("_")
                if column and column not in columns:
                    columns.append(column)
        tables[table] = {"entity": name, "columns": columns}

    def contract_value(field: dict[str, Any]) -> Any:
        kind = str(field.get("type", "string")).lower()
        if kind in {"number", "integer", "int", "float"}:
            return 1
        if kind in {"boolean", "bool"}:
            return True
        if kind in {"array", "list"}:
            return []
        if kind in {"select", "enum"} and field.get("options"):
            return field["options"][0]
        if kind in {"email", "e-mail"}:
            return "verification@example.test"
        if kind == "date":
            return "2026-01-15"
        if kind == "month":
            return "2026-01"
        return "sample"

    raw_routes = spec.get("routes_needed") if isinstance(spec.get("routes_needed"), list) else []
    routes: list[tuple[str, str]] = [("GET", "/")]
    for raw in raw_routes:
        if not isinstance(raw, str):
            continue
        match = re.match(r"^\s*(GET|POST|PUT|PATCH|DELETE)\s+(.+?)\s*$", raw, re.I)
        if match:
            pair = (match.group(1).upper(), "/" + match.group(2).lstrip("/"))
        else:
            pair = ("GET", "/" + raw.lstrip("/"))
        if pair not in routes:
            routes.append(pair)
    for table in tables:
        collection = next((path for _method, path in routes if table in path.lower() and "<" not in path), None)
        collection = collection or f"/api/{table}{'' if table.endswith('s') else 's'}"
        if not any(method == "GET" and path == collection for method, path in routes):
            routes.append(("GET", collection))
        if not any(method == "POST" and path == collection for method, path in routes):
            routes.append(("POST", collection))
        detail = next((path for method, path in routes
                       if method in {"GET", "PUT", "PATCH", "DELETE"}
                       and table in path.lower() and "<" in path), None)
        detail = detail or f"{collection}/<int:item_id>"
        for method in ("GET", "PUT", "DELETE"):
            if not any(existing_method == method and existing_path == detail for existing_method, existing_path in routes):
                routes.append((method, detail))
    if len(routes) == 1:
        for table in tables:
            collection = f"/api/{table}"
            routes.extend([("GET", collection), ("POST", collection), ("GET", f"{collection}/<int:item_id>")])

    endpoints: list[dict[str, Any]] = []
    for method, path in routes:
        # The generated entrypoint owns `/` and serves the user interface.
        # Do not let the model turn the page root into a JSON API endpoint.
        if method == "GET" and path == "/":
            continue
        table = next((name for name in tables if name in path.lower()), None)
        columns = tables[table]["columns"] if table else []
        entity = next((item for item in entities if isinstance(item, dict) and table and re.sub(r"[^a-z0-9]+", "_", str(item.get("name", "")).lower()).strip("_") == table), None)
        request_shape = {
            re.sub(r"[^a-z0-9_]+", "_", str(field.get("name", "")).lower()).strip("_"): contract_value(field)
            for field in (entity or {}).get("fields", []) if isinstance(field, dict) and field.get("name")
        }
        field_types = {
            re.sub(r"[^a-z0-9_]+", "_", str(field.get("name", "")).lower()).strip("_"): str(field.get("type", "string")).lower()
            for field in (entity or {}).get("fields", []) if isinstance(field, dict) and field.get("name")
        }
        request_json = request_shape if method in {"POST", "PUT", "PATCH"} else None
        response_json: Any = {"success": True, "data": [] if method == "GET" and "<" not in path else {"id": 1}} if table else {"success": True}
        endpoints.append({
            "method": method,
            "path": path,
            "request_json": request_json,
            "response_json": response_json,
            "request": request_json,
            "response": response_json,
            "db_table": table,
            "db_columns": columns,
            "field_types": field_types,
        })
    return {"endpoints": endpoints, "tables": tables}


def clear_seed_data(spec: dict[str, Any]) -> None:
    """Enforce the product requirement that generated apps start without fake records."""
    entities = spec.get("entities") if isinstance(spec.get("entities"), list) else []
    spec["seed_data"] = {
        str(entity.get("name", "Entity")): []
        for entity in entities
        if isinstance(entity, dict) and entity.get("name")
    }


def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Fill safe structural defaults before deriving the contract and prompts.

    This is deliberately deterministic: incomplete optional planning fields do
    not trigger another model call, while malformed core collections fail
    clearly instead of surfacing later as generation errors.
    """
    if not isinstance(spec, dict):
        raise ValueError("The application specification must be a JSON object.")
    normalized = dict(spec)
    normalized["app_name"] = str(normalized.get("app_name") or "ForgeAI App").strip() or "ForgeAI App"
    normalized["purpose"] = str(normalized.get("purpose") or normalized["app_name"]).strip()
    for key in ("user_roles", "core_features", "entities", "relationships", "routes_needed", "pages_needed", "pages",
                "navigation", "dashboard_kpis", "database_requirements", "validation_requirements",
                "error_handling_requirements", "responsive_requirements", "assumptions"):
        value = normalized.get(key)
        if value is None:
            normalized[key] = []
        elif not isinstance(value, list):
            raise ValueError(f"Specification field '{key}' must be a list.")
    for index, entity in enumerate(normalized["entities"]):
        if not isinstance(entity, dict) or not str(entity.get("name", "")).strip():
            raise ValueError(f"Specification entity {index + 1} must have a name.")
        fields = entity.get("fields", [])
        if not isinstance(fields, list):
            raise ValueError(f"Fields for entity '{entity['name']}' must be a list.")
        entity["fields"] = [field for field in fields if isinstance(field, dict) and str(field.get("name", "")).strip()]
    theme = normalized.get("theme")
    normalized["theme"] = theme if isinstance(theme, dict) else {}
    normalized["auth_required"] = bool(normalized.get("auth_required", False))
    return normalized


def generate_spec(user_prompt: str, model_name: str = "openai/gpt-oss-20b", on_model_switch: Any = None) -> dict[str, Any]:
    """Request a Groq-generated app specification JSON for a plain-English app description."""
    prompt = build_spec_prompt(user_prompt)

    try:
        response = generate_with_backoff(prompt, preferred_model=model_name, json_mode=True,
                                         on_model_switch=on_model_switch, step="spec",
                                         cache_context={"stage": "specification"})
        response_text = getattr(response, "text", None)
        if not response_text:
            raise ValueError("Groq returned no text content.")
        if not getattr(response, "complete", True):
            raise ValueError("Provider marked the specification response as truncated.")
        spec = extract_json_payload(response_text)
        spec.setdefault("app_name", "ForgeAI App")
        clear_seed_data(spec)
        spec["api_contract"] = build_api_contract(spec)
        if hasattr(response, "cache_validated"):
            response.cache_validated()
        return spec
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Groq JSON parsing failed: {_normalize_model_error(exc)}") from exc
    except Exception as exc:  # pragma: no cover - surfaced in runtime path
        error_msg = _normalize_model_error(exc)
        if "not found" in error_msg.lower() or "not supported" in error_msg.lower():
            error_msg += " Check the model ID and permissions in Groq Console."
        raise RuntimeError(f"Groq request failed: {error_msg}") from exc


def build_combined_spec_plan_prompt(user_prompt: str) -> str:
    """Build the single existing spec+plan request with a concise app brief."""
    return f'''Create a concise application specification and build plan for this request:
{user_prompt}

Return one valid JSON object only, using this structure:
{{"spec":{{"app_name":"","purpose":"","user_roles":[],"core_features":[],"entities":[{{"name":"","fields":[{{"name":"","type":"","required":false}}]}}],"relationships":[],"auth_required":false,"routes_needed":[],"pages_needed":[],"pages":[{{"name":"","purpose":"","widgets":[]}}],"navigation":[],"dashboard_kpis":[],"database_requirements":[],"validation_requirements":[],"error_handling_requirements":[],"responsive_requirements":[],"assumptions":[],"theme":{{}}}},"steps":[]}}

Rules:
- Use concise values. Include purpose, roles, requested features, domain workflow, entities/relationships, pages/navigation, database, validation/errors, and responsive needs.
- Require authentication only when explicitly requested; otherwise assume a single local user and record it. Do not invent external integrations. Do not include demo records or an API contract.
- Give at most 3 short implementation steps. Output JSON only, without markdown or prose.'''


def generate_spec_and_plan(user_prompt: str, model_name: str = "openai/gpt-oss-20b", on_model_switch: Any = None) -> tuple[dict[str, Any], list[str]]:
    """Ask Gemini for both a machine-readable spec and an ordered build plan in a single response.

    The model is instructed to return a single JSON object with keys `spec` and `plan`:
    {
      "spec": { ... },
      "plan": ["step 1", "step 2", ...]
    }

    This reduces outbound call count by combining two related steps.
    """
    # Keep the response schema compact and valid JSON. The API contract and
    # empty seed-data map are derived locally after parsing, so asking the model
    # to emit them wastes tokens and increases constrained-JSON failures.
    prompt = build_combined_spec_plan_prompt(user_prompt)

    for attempt in range(2):
        request_prompt = prompt
        if attempt:
            request_prompt += (
                "\n\nRETRY: The previous specification response was incomplete or invalid. Return one complete JSON object "
                'with "spec" and "plan" keys only. Do not use Markdown fences or prose.'
            )
        try:
            response = generate_with_backoff(request_prompt, preferred_model=model_name, json_mode=True,
                                             on_model_switch=on_model_switch, max_tokens=1600, step="spec",
                                             cache_context={"stage": "combined_spec_and_plan", "attempt": attempt})
            response_text = getattr(response, "text", None)
            if not response_text:
                raise ValueError("Provider returned an empty combined specification response.")
            if not getattr(response, "complete", True):
                reason = None
                raw = getattr(response, "raw", None)
                try:
                    raw_payload = raw.json() if raw is not None and hasattr(raw, "json") else {}
                    reason = raw_payload.get("done_reason") if isinstance(raw_payload, dict) else None
                except (TypeError, ValueError):
                    pass
                termination = f" finish_reason={reason}" if reason else ""
                error = f"Provider marked the combined specification response as incomplete.{termination}"
                if getattr(response, "provider", None) == "ollama":
                    raise RuntimeError(f"Ollama returned incomplete spec+plan output; reduce request size and retry.{termination}")
                raise ValueError(error)

            parsed = extract_json_payload(response_text)
            direct_spec = parsed.get("spec") or parsed.get("application_spec") or parsed.get("app_spec") or parsed.get("specification")
            if isinstance(direct_spec, dict):
                spec = direct_spec
                plan = parsed.get("steps") or parsed.get("plan") or parsed.get("build_plan") or []
            elif isinstance(parsed.get("application"), dict):
                spec = parsed["application"]
                plan = parsed.get("steps") or parsed.get("plan") or parsed.get("build_plan") or []
            elif isinstance(parsed.get("entities"), list):
                # Some providers return the inner specification directly.
                spec = parsed
                plan = [f"Implement {page.get('name', 'the page')} with its specified purpose and widgets"
                        for page in spec.get("pages", []) if isinstance(page, dict)]
                plan = plan or ["Implement the domain data model", "Build the API and app interface", "Verify routes and persistence"]
            else:
                raise ValueError("Response must contain a spec and plan, or a complete spec object.")

            if not isinstance(spec, dict):
                raise ValueError("The specification must be a JSON object.")
            if plan and (not isinstance(plan, list) or not all(isinstance(step, str) for step in plan)):
                raise ValueError("The specification plan must be a list of strings.")
            if not plan:
                plan = [f"Implement {page.get('name', 'the page')} with its specified purpose and widgets"
                        for page in spec.get("pages", []) if isinstance(page, dict)]
                plan = plan or ["Implement the domain data model", "Build the API and app interface", "Verify routes and persistence"]

            spec = normalize_spec(spec)
            plan = [step.strip() for step in plan if step.strip()][:4]
            if not plan:
                plan = ["Implement the domain data model", "Build the API and app interface", "Verify routes and persistence"]
            clear_seed_data(spec)
            spec["api_contract"] = build_api_contract(spec)
            if hasattr(response, "cache_validated"):
                response.cache_validated()
            return spec, plan
        except (ValueError, TypeError) as exc:
            if attempt == 0:
                continue
            raise RuntimeError(f"Combined spec+plan response remained invalid after one formatting retry: {exc}") from exc
        except Exception as exc:  # provider/auth/quota failures are handled by the LLM client
            error_msg = _normalize_model_error(exc)
            if "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                error_msg += " Check the model ID and permissions in Groq Console."
            raise RuntimeError(f"Combined spec+plan provider request failed: {error_msg}") from exc

    raise RuntimeError("Combined spec+plan generation failed without a usable response.")
