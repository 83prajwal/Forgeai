"""Offline, locally stored app replays for exhibition demos."""
from __future__ import annotations

import json
import logging
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import ast

SHOWCASE_DIR = Path(__file__).resolve().parents[1] / "showcase"
RUNTIME_DIR = SHOWCASE_DIR / "runtime"
CATALOG_PATH = SHOWCASE_DIR / "catalog.json"
GENERATED_FILENAMES = ["requirements.txt", "app.py", "index.html", "style.css", "script.js", "profile.json"]
logger = logging.getLogger("forgeai.showcase")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHOWCASE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _valid_showcase(showcase: Any) -> bool:
    if not isinstance(showcase, dict):
        return False
    if not all(isinstance(showcase.get(key), str) and showcase[key].strip() for key in ("id", "name", "prompt", "category", "accent", "icon", "tone")):
        return False
    if not _SHOWCASE_ID.fullmatch(showcase["id"]):
        return False
    entities = showcase.get("entities")
    if not isinstance(entities, list) or not entities:
        return False
    for entity in entities:
        if not isinstance(entity, dict) or not isinstance(entity.get("name"), str) or not entity["name"].strip():
            return False
        if not isinstance(entity.get("icon"), str) or not entity["icon"].strip():
            return False
        if not isinstance(entity.get("table"), str) or not _IDENTIFIER.fullmatch(entity["table"]):
            return False
        fields = entity.get("fields")
        if not isinstance(fields, list) or not fields:
            return False
        for field in fields:
            if not isinstance(field, dict) or not isinstance(field.get("name"), str) or not _IDENTIFIER.fullmatch(field["name"]):
                return False
            if not isinstance(field.get("label"), str) or not field["label"].strip():
                return False
            if field.get("type") not in {"text", "number", "select", "boolean", "date", "email", "month"}:
                return False
            if field.get("type") == "select" and not isinstance(field.get("options"), list):
                return False
    return True


def load_showcases(catalog_path: Path | None = None) -> list[dict[str, Any]]:
    """Load valid local showcase definitions, skipping broken entries safely."""
    path = catalog_path or CATALOG_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Showcase catalog unavailable at %s: %s", path, exc)
        return []
    if not isinstance(raw, list):
        logger.warning("Showcase catalog at %s must contain a list", path)
        return []

    valid: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw:
        if not _valid_showcase(item) or item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        valid.append(item)
    if len(valid) != len(raw):
        logger.warning("Skipped %d invalid showcase catalog entr%s", len(raw) - len(valid), "y" if len(raw) - len(valid) == 1 else "ies")
    return valid


def match_showcase(prompt: str, showcases: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    normalized = " ".join(prompt.lower().split())
    best: dict[str, Any] | None = None
    best_score = 0.0
    for showcase in showcases if showcases is not None else load_showcases():
        phrases = [showcase["name"].lower(), showcase["prompt"].lower(), *showcase.get("keywords", [])]
        score = max(SequenceMatcher(None, normalized, phrase).ratio() for phrase in phrases)
        score = max(score, max((0.8 for word in showcase.get("keywords", []) if word in normalized), default=0.0))
        if score > best_score:
            best, best_score = showcase, score
    return best if best_score >= 0.48 else None


def materialize_showcase(showcase: dict[str, Any], project_dir: Path) -> list[dict[str, str]]:
    if not _valid_showcase(showcase):
        raise ValueError("Cannot load showcase: its catalog entry is missing required app, entity, or field details.")
    runtime_files = {filename: RUNTIME_DIR / filename for filename in GENERATED_FILENAMES if filename != "profile.json"}
    missing = [str(path.name) for path in runtime_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Showcase runtime is missing required files: {', '.join(missing)}")
    project_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, str]] = []
    for filename in GENERATED_FILENAMES:
        if filename == "profile.json":
            content = json.dumps(showcase, ensure_ascii=False, indent=2)
        else:
            source = runtime_files[filename]
            content = source.read_text(encoding="utf-8")
        (project_dir / filename).write_text(content, encoding="utf-8")
        files.append({"filename": filename, "content": content})
    return files


def contract_for_showcase(showcase: dict[str, Any]) -> dict[str, Any]:
    endpoints = [{"method":"GET", "path":"/api/profile", "request_json":None, "response_json":{"name":showcase["name"]}},
                 {"method":"GET", "path":"/api/dashboard", "request_json":None, "response_json":{"totals":{}}},
                 {"method":"GET", "path":"/api/settings", "request_json":None, "response_json":{} }]
    for entity in showcase["entities"]:
        table = entity["table"]
        columns = [field["name"] for field in entity["fields"]]
        collection = f"/api/{table}"
        detail = f"{collection}/<int:item_id>"
        endpoints.append({"method":"GET", "path":collection, "request_json":None, "response_json":{"items":[]}, "db_table":table, "db_columns":columns})
        field_types = {field["name"]: field.get("type", "text") for field in entity["fields"]}
        request_json = {}
        for field in entity["fields"]:
            kind = field.get("type")
            if kind == "select":
                value = field.get("options", ["Verification"])[0]
            elif kind == "number":
                value = 1
            elif kind == "boolean":
                value = True
            elif kind == "email":
                value = "verification@example.test"
            elif kind == "date":
                value = "2026-01-15"
            elif kind == "month":
                value = "2026-01"
            else:
                value = "Verification"
            request_json[field["name"]] = value
        endpoints.append({"method":"POST", "path":collection, "request_json":request_json, "response_json":{"id":"integer"}, "db_table":table, "db_columns":columns, "field_types":field_types})
        endpoints.extend([
            {"method":"GET", "path":detail, "request_json":None, "response_json":{"id":"integer"}, "db_table":table, "db_columns":columns},
            {"method":"PUT", "path":detail, "request_json":request_json, "response_json":{"success":True}, "db_table":table, "db_columns":columns},
            {"method":"DELETE", "path":detail, "request_json":None, "response_json":{"success":True}, "db_table":table, "db_columns":columns},
        ])
    return {"endpoints": endpoints}


def prepare_showcases() -> list[dict[str, Any]]:
    """Prebuild all six app folders from the local runtime template."""
    catalog = load_showcases()
    # Rebuild seed databases against the current showcase schemas. Older
    # releases used one generic `records` table, so keep those columns from
    # leaking into the newly generated domain apps.
    remove_showcase_databases()
    for showcase in catalog:
        files = materialize_showcase(showcase, SHOWCASE_DIR / showcase["id"])
        for item in files:
            if item["filename"].endswith(".py"):
                ast.parse(item["content"])
    return catalog


def remove_showcase_databases() -> None:
    for path in SHOWCASE_DIR.glob("*/showcase.db"):
        path.unlink(missing_ok=True)


def clear_showcase_runtime() -> None:
    """Remove generated state if the demo catalog is rebuilt locally."""
    for path in SHOWCASE_DIR.glob("*/showcase.db"):
        path.unlink(missing_ok=True)
