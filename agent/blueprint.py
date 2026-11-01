"""Prompt -> app blueprint -> proven runtime.

The model only describes the app (name, theme, entities, fields). A tested,
data-driven runtime (showcase/runtime) supplies the backend and UI, so a build
never depends on model-written server code.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from agent.llm_client import generate_with_backoff
from agent.showcase import _valid_showcase
from agent.spec_generator import extract_json_payload

FIELD_TYPES = {"text", "number", "select", "boolean", "date", "email", "month"}
TYPE_ALIASES = {
    "string": "text", "textarea": "text", "url": "text", "phone": "text", "time": "text",
    "integer": "number", "int": "number", "float": "number", "decimal": "number", "currency": "number",
    "money": "number", "price": "number", "rating": "number",
    "bool": "boolean", "checkbox": "boolean",
    "datetime": "date", "timestamp": "date",
    "enum": "select", "choice": "select", "dropdown": "select",
}
# Names the runtime already uses for its own routes, tables, or columns.
RESERVED_TABLES = {"settings", "profile", "dashboard", "sqlite_sequence"}
RESERVED_FIELDS = {"id", "created_at"}
DEFAULT_ACCENT = "#6d5efc"

BLUEPRINT_PROMPT = """You design small data apps. Turn the request into an app blueprint.

Request: {prompt}

Return ONE JSON object only (no markdown, no prose):
{{"name": "App name", "category": "one or two words", "tone": "3-5 word style",
 "accent": "#hex colour that suits the app", "icon": "one emoji",
 "entities": [{{"name": "Plural entity name", "icon": "one emoji",
   "fields": [{{"name": "snake_case", "label": "Field label", "type": "text|number|select|boolean|date|email|month",
               "required": true, "options": ["only", "for", "select"]}}]}}]}}

Rules:
- name: use the user's own words, title-cased ("build a workout tracker app" -> "Workout Tracker"). Never swap in a synonym.
- 1-4 entities that the request actually needs; 3-7 fields each. The first field is the record's title and is required.
- Use only the listed types. Use select (with 2-6 options) for fixed choices like status, priority, category.
- Do not add id, created_at, user, or owner fields; the app adds ids and timestamps itself."""


def _identifier(value: Any, fallback: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    if not name or not name[0].isalpha():
        name = f"{fallback}_{name}".strip("_") if name else fallback
    return name[:40]


def _title_from_prompt(prompt: str) -> str:
    """Fallback name taken from the user's wording: "build a recipe manager app" -> "Recipe Manager"."""
    text = re.sub(r"^\s*(please\s+)?(build|create|make|generate|design)\s+(me\s+)?(an?|the)?\s*", "", prompt, flags=re.I)
    text = re.split(r"\b(app|application|website|tool|system)\b|[.,:;!?\n]| for | with | that | to ", text, maxsplit=1, flags=re.I)[0]
    words = text.split()[:5]
    return " ".join(word.capitalize() for word in words) or "My App"


def normalize_blueprint(raw: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Coerce a model blueprint into the runtime profile format, or raise ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("Blueprint must be a JSON object.")
    name = str(raw.get("name") or "").strip()[:60] or _title_from_prompt(prompt)
    accent = str(raw.get("accent") or "")
    entities: list[dict[str, Any]] = []
    used_tables: set[str] = set()
    for index, entity in enumerate(raw.get("entities") or []):
        if not isinstance(entity, dict) or len(entities) >= 4:
            continue
        entity_name = str(entity.get("name") or "").strip()[:40] or f"Items {index + 1}"
        table = _identifier(entity.get("table") or entity_name, "items")
        if table in RESERVED_TABLES:
            table = f"{table}_items"
        while table in used_tables:
            table += "_2"
        fields: list[dict[str, Any]] = []
        used_fields: set[str] = set()
        for field in entity.get("fields") or []:
            if not isinstance(field, dict) or len(fields) >= 8:
                continue
            field_name = _identifier(field.get("name") or field.get("label"), "field")
            if field_name in RESERVED_FIELDS or field_name in used_fields:
                continue
            kind = str(field.get("type") or "text").lower().strip()
            kind = TYPE_ALIASES.get(kind, kind if kind in FIELD_TYPES else "text")
            options = [str(option)[:40] for option in field.get("options") or [] if str(option).strip()][:8]
            if kind == "select" and len(options) < 2:
                kind, options = "text", []
            item: dict[str, Any] = {"name": field_name, "label": str(field.get("label") or field_name.replace("_", " ").title())[:40],
                                    "type": kind, "required": bool(field.get("required"))}
            if kind == "select":
                item["options"] = options
            fields.append(item)
            used_fields.add(field_name)
        if not fields:
            continue
        fields[0]["required"] = True
        used_tables.add(table)
        entities.append({"name": entity_name, "table": table, "icon": str(entity.get("icon") or "◆")[:4],
                         "fields": fields, "seed": []})
    if not entities:
        raise ValueError("Blueprint has no usable entities with fields.")
    blueprint = {
        "id": _identifier(name, "app").replace("_", "-"),
        "name": name,
        "category": str(raw.get("category") or "Custom app").strip()[:30] or "Custom app",
        "prompt": prompt.strip()[:300] or name,
        "keywords": [],
        "accent": accent if re.fullmatch(r"#[0-9a-fA-F]{6}", accent) else DEFAULT_ACCENT,
        "icon": str(raw.get("icon") or "✦").strip()[:4] or "✦",
        "tone": str(raw.get("tone") or "Clear and focused").strip()[:60] or "Clear and focused",
        "entities": entities,
    }
    if not _valid_showcase(blueprint):
        raise ValueError("Blueprint did not pass runtime validation.")
    return blueprint


def generate_blueprint(prompt: str, on_model_switch: Callable[[str], None] | None = None,
                       current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ask the model for a blueprint (or a revision of `current`) and normalize it. One retry on bad JSON."""
    if current is None:
        request = BLUEPRINT_PROMPT.format(prompt=prompt)
        source_prompt = prompt
    else:
        request = (
            BLUEPRINT_PROMPT.format(prompt=current.get("prompt", ""))
            + f"\n\nCurrent blueprint:\n{json.dumps(current, ensure_ascii=False)}"
            + f"\n\nApply only this change and return the full updated blueprint. Keep the name and existing fields unless the change says otherwise:\n{prompt}"
        )
        source_prompt = current.get("prompt", "")
    last_error: Exception | None = None
    for attempt in range(2):
        response = generate_with_backoff(
            request + ("\n\nYour previous answer was not valid. Return the JSON object only." if attempt else ""),
            json_mode=True, max_tokens=3000, step="spec", on_model_switch=on_model_switch,
            cache_context={"stage": "blueprint", "attempt": attempt},
        )
        try:
            blueprint = normalize_blueprint(extract_json_payload(response.text), source_prompt)
        except (ValueError, TypeError) as exc:
            last_error = exc
            continue
        if current is not None:
            blueprint["id"], blueprint["prompt"] = current.get("id", blueprint["id"]), current.get("prompt", blueprint["prompt"])
        if hasattr(response, "cache_validated"):
            response.cache_validated()
        return blueprint
    raise RuntimeError(f"The model did not return a usable app blueprint: {last_error}")
