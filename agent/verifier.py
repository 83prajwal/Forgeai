"""TODO: Check whether the generated app works and capture errors."""

import json
import re
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any

import os
from email.utils import parsedate_to_datetime

import requests
from agent.executor import PROCESS_HANDLES

# Generated Flask apps can take ~10s to boot on Windows (interpreter start plus
# antivirus scanning), so the default leaves room for a slow machine.
DEFAULT_VERIFY_TIMEOUT_SECONDS = float(os.getenv("VERIFY_TIMEOUT_SECONDS", "45"))


def _read_error_logs(project_dir: Path) -> str:
    """Return combined stdout/stderr logs from the generated app if they exist."""
    log_parts: list[str] = []
    for name in ("stdout.log", "stderr.log"):
        log_path = project_dir / name
        if log_path.exists():
            log_parts.append(f"--- {name} ---\n{log_path.read_text(encoding='utf-8', errors='replace')}")
    return "\n".join(log_parts)


def _read_port(project_dir: Path) -> int | None:
    """Read the port used by the generated app from the executor metadata file."""
    port_file = project_dir / "port.txt"
    if port_file.exists():
        try:
            return int(port_file.read_text(encoding="utf-8").strip())
        except ValueError:
            return None
    return None


def _process_exit_code(project_dir: Path) -> int | None:
    """Return the local Popen exit code when the launched process has exited."""
    try:
        process_data = json.loads((project_dir / "process.json").read_text(encoding="utf-8"))
        process = PROCESS_HANDLES.get(int(process_data.get("pid", 0)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return process.poll() if process is not None else None


def _resolve_table(connection: sqlite3.Connection, table: str) -> str:
    """Return the contract table, or its singular/plural spelling if the app used that instead."""
    existing = {row[0].lower(): row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    base = table.lower()
    candidates = [base, base + "s", base + "es", base[:-1] if base.endswith("s") else None,
                  base[:-2] if base.endswith("es") else None,
                  base[:-1] + "ies" if base.endswith("y") else None,
                  base[:-3] + "y" if base.endswith("ies") else None]
    return next((existing[name] for name in candidates if name and name in existing), table)


def _remove_verifier_record(project_dir: Path, endpoint: dict[str, Any] | None, body: dict[str, Any] | None,
                            record_id: Any = None) -> None:
    """Remove the temporary CRUD-check row so verification never pollutes the app."""
    if not endpoint or not body:
        return
    table = endpoint.get("db_table")
    if not isinstance(table, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        return
    for db_path in project_dir.rglob("*.db"):
        try:
            with sqlite3.connect(db_path) as connection:
                table = _resolve_table(connection, table)
                columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
                if record_id is not None and "id" in columns:
                    connection.execute(f'DELETE FROM "{table}" WHERE "id" = ?', (record_id,))
                    continue
                names = [name for name in body if name in columns]
                if names:
                    where = " AND ".join(f'"{name}" = ?' for name in names)
                    connection.execute(f'DELETE FROM "{table}" WHERE {where}', [body[name] for name in names])
        except sqlite3.Error:
            continue


def verify_generated_app(
    project_dir: str | Path,
    routes: list[str] | None = None,
    timeout_seconds: float | None = None,
    request_timeout_seconds: float = 5.0,
    api_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Boot the app, exercise CRUD contract routes, and check SQLite persistence."""
    if timeout_seconds is None:
        timeout_seconds = DEFAULT_VERIFY_TIMEOUT_SECONDS
    project_dir = Path(project_dir)
    endpoints = (api_contract or {}).get("endpoints", [])
    get_endpoints = [endpoint for endpoint in endpoints if endpoint.get("method", "").upper() == "GET"]
    post_endpoint = next((endpoint for endpoint in endpoints if endpoint.get("method", "").upper() == "POST" and endpoint.get("request_json")), None)
    if not get_endpoints:
        get_endpoints = [{"method": "GET", "path": path} for path in (routes or ["/"])]

    result: dict[str, Any] = {
        "project_dir": str(project_dir),
        "routes_checked": [],
        "endpoints_verified": 0,
        "records_persisted": False,
        "status": "unknown",
        "booted": False,
        "responses": [],
        "workflow_checks": [],
        "error": None,
        "traceback": None,
        "process_exit_code": None,
    }

    start = time.time()
    while time.time() - start < timeout_seconds:
        exit_code = _process_exit_code(project_dir)
        if exit_code is not None:
            result["process_exit_code"] = exit_code
            result["status"] = "failed"
            result["error"] = f"App process exited with status {exit_code} before completing verification."
            result["traceback"] = _read_error_logs(project_dir)[-8000:]
            return result
        port = _read_port(project_dir)
        if port is None:
            time.sleep(1)
            continue

        # A simple root request confirms the process is listening before exercising APIs.
        try:
            root = requests.get(f"http://127.0.0.1:{port}/", timeout=request_timeout_seconds)
            result["booted"] = True
            if not root.ok:
                result["responses"].append({"method": "GET", "route": "/", "url": f"http://127.0.0.1:{port}/", "status_code": root.status_code})
                result["error"] = f"Route / returned HTTP {root.status_code}."
                result["traceback"] = _read_error_logs(project_dir)[-8000:]
                if root.status_code >= 500:
                    time.sleep(min(0.25, max(0, timeout_seconds - (time.time() - start))))
                    continue
                result["status"] = "failed"
                return result
        except requests.RequestException as exc:
            result["error"] = str(exc)
            result["traceback"] = _read_error_logs(project_dir)[-8000:]
            result["responses"].append({"method": "GET", "route": "/", "url": f"http://127.0.0.1:{port}/", "error_type": type(exc).__name__, "error": str(exc)[:300]})
            time.sleep(min(0.25, max(0, timeout_seconds - (time.time() - start))))
            continue

        inserted_id: Any = None
        posted_body: dict[str, Any] | None = None
        if post_endpoint:
            post_url = f"http://127.0.0.1:{port}{post_endpoint['path']}"
            posted_body = dict(post_endpoint["request_json"])
            try:
                posted = requests.post(post_url, json=posted_body, timeout=request_timeout_seconds)
                result["responses"].append({"method": "POST", "route": post_endpoint["path"], "status_code": posted.status_code})
                if posted.status_code not in {200, 201, 202}:
                    _remove_verifier_record(project_dir, post_endpoint, posted_body, inserted_id)
                    result["error"] = f"POST {post_endpoint['path']} returned HTTP {posted.status_code}."
                    result["traceback"] = f"{posted.text[:1000]}\n\n{_read_error_logs(project_dir)[-8000:]}"
                    if posted.status_code >= 500:
                        time.sleep(min(0.25, max(0, timeout_seconds - (time.time() - start))))
                        continue
                    result["status"] = "failed"
                    return result
                try:
                    posted_data = posted.json()
                except ValueError:
                    posted_data = {}
                if isinstance(posted_data, dict):
                    inserted_id = posted_data.get("id") or posted_data.get("item_id")
                    nested = posted_data.get("data") or posted_data.get("item") or posted_data.get("record")
                    if isinstance(nested, dict):
                        inserted_id = inserted_id or nested.get("id")
                result["endpoints_verified"] += 1
            except requests.RequestException as exc:
                _remove_verifier_record(project_dir, post_endpoint, posted_body, inserted_id)
                result["error"] = f"POST {post_endpoint['path']} failed: {exc}"
                result["traceback"] = _read_error_logs(project_dir)[-8000:]
                result["responses"].append({"method": "POST", "route": post_endpoint["path"], "url": post_url, "error_type": type(exc).__name__, "error": str(exc)[:300]})
                time.sleep(min(0.25, max(0, timeout_seconds - (time.time() - start))))
                continue

        retry_verification = False
        for endpoint in get_endpoints:
            route = _route_for_record(endpoint.get("path", "/"), inserted_id or 1)
            if route.startswith("http://") or route.startswith("https://"):
                url = route
            else:
                url = f"http://127.0.0.1:{port}{route}"

            try:
                response = requests.get(url, timeout=request_timeout_seconds)
                result["responses"].append(
                    {
                        "route": route,
                        "url": url,
                        "status_code": response.status_code,
                        "text": response.text[:500],
                    }
                )
                result["routes_checked"].append(endpoint.get("path", route))
                is_detail_route = "<" in str(endpoint.get("path", ""))
                has_verification_record = bool(
                    post_endpoint and endpoint.get("db_table") == post_endpoint.get("db_table") and inserted_id is not None
                )
                if is_detail_route and not has_verification_record and response.status_code in {404, 410}:
                    result["endpoints_verified"] += 1
                    result["workflow_checks"].append({"operation": "missing_record", "status": "passed", "route": route})
                    continue
                if response.ok:
                    result["endpoints_verified"] += 1
                    continue
                _remove_verifier_record(project_dir, post_endpoint, posted_body, inserted_id)
                result["error"] = f"Route {route} returned HTTP {response.status_code}."
                result["traceback"] = response.text[:1000]
                app_logs = _read_error_logs(project_dir)
                if app_logs:
                    result["traceback"] += f"\n\n--- app logs ---\n{app_logs[-8000:]}"
                if response.status_code >= 500:
                    retry_verification = True
                    break
                result["status"] = "failed"
                return result
            except requests.RequestException as exc:
                _remove_verifier_record(project_dir, post_endpoint, posted_body, inserted_id)
                result["error"] = str(exc)
                result["traceback"] = f"{traceback.format_exc()}\n\n{_read_error_logs(project_dir)[-8000:]}"
                result["responses"].append({"method": "GET", "route": route, "url": url, "error_type": type(exc).__name__, "error": str(exc)[:300]})
                retry_verification = True
                break

        if retry_verification:
            time.sleep(min(0.25, max(0, timeout_seconds - (time.time() - start))))
            continue

        if post_endpoint and posted_body is not None:
            table = post_endpoint.get("db_table")
            if table and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
                for db_path in project_dir.rglob("*.db"):
                    try:
                        with sqlite3.connect(db_path) as connection:
                            table = _resolve_table(connection, table)
                            columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
                            if not columns:
                                continue
                            if inserted_id is not None and "id" in columns:
                                found = connection.execute(f'SELECT 1 FROM "{table}" WHERE "id" = ? LIMIT 1', (inserted_id,)).fetchone()
                                if found:
                                    result["records_persisted"] = True
                                    break
                            values = list(posted_body.values())
                            names = [name for name in posted_body if name in columns]
                            if not names:
                                continue
                            where = " AND ".join(f'"{name}" = ?' for name in names)
                            found = connection.execute(f'SELECT 1 FROM "{table}" WHERE {where} LIMIT 1', [posted_body[name] for name in names]).fetchone()
                            if found:
                                result["records_persisted"] = True
                                break
                    except sqlite3.Error:
                        continue
            if not result["records_persisted"]:
                _remove_verifier_record(project_dir, post_endpoint, posted_body, inserted_id)
                result["error"] = "The created record was not found in the generated SQLite database after POST."
                result["traceback"] = _read_error_logs(project_dir)[-8000:]
                return result
            result["workflow_checks"].append({"operation": "create_and_persist", "status": "passed"})

            workflow_error = _verify_update_delete(
                f"http://127.0.0.1:{port}", endpoints, post_endpoint, inserted_id,
                posted_body, result, request_timeout_seconds,
            )
            if workflow_error:
                _remove_verifier_record(project_dir, post_endpoint, posted_body, inserted_id)
                result["error"] = workflow_error
                result["traceback"] = _read_error_logs(project_dir)[-8000:]
                return result

        _remove_verifier_record(project_dir, post_endpoint, posted_body, inserted_id)
        result["status"] = "ok"
        result["error"] = None
        return result

    result["status"] = "timed_out"
    if not result.get("error"):
        result["error"] = "Timed out while waiting for the app to respond."
    else:
        result["error"] = f"Timed out while waiting for the app to respond. Last failure: {result['error']}"
    app_logs = _read_error_logs(project_dir)
    if app_logs:
        result["traceback"] = f"{result.get('traceback') or ''}\n\n--- app logs ---\n{app_logs}".strip()
    return result


def _route_for_record(path: str, record_id: Any) -> str:
    """Replace Flask-style contract variables with the id returned by create."""
    value = str(record_id)
    path = re.sub(r"<int:[^>]+>", value, path)
    return re.sub(r"<[^>]+>", value, path)


def _is_system_field(name: str) -> bool:
    name = name.casefold()
    return name in {"id", "timestamp"} or name.endswith(("_id", "_at")) or name.startswith(("created", "updated"))


def _value_matches(expected: Any, actual: Any) -> bool:
    """Compare an updated value with what the API returned, tolerating SQLite/JSON encodings."""
    if isinstance(expected, bool):
        return actual in (expected, int(expected), str(expected).lower(), str(int(expected)))
    if isinstance(expected, (int, float)) and not isinstance(actual, bool):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, dict)):
        return actual == expected or (isinstance(actual, str) and actual.replace(" ", "") == json.dumps(expected).replace(" ", ""))
    if actual == expected or (isinstance(actual, str) and str(expected) in actual):
        return True
    # Flask serializes date objects as HTTP dates ("Fri, 16 Jan 2026 00:00:00 GMT").
    if isinstance(expected, str) and isinstance(actual, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected):
        try:
            return parsedate_to_datetime(actual).date().isoformat() == expected
        except (TypeError, ValueError):
            return False
    return False


def _updated_values_missing(detail_response: requests.Response, update_payload: dict[str, Any],
                            posted_body: dict[str, Any]) -> list[str]:
    """Return changed fields the read-back record does not reflect."""
    changed = {key: value for key, value in update_payload.items() if value != posted_body.get(key)}
    try:
        body = detail_response.json()
    except ValueError:
        body = None
    record = body.get("data", body) if isinstance(body, dict) else None
    if isinstance(record, list) and len(record) == 1:
        record = record[0]
    missing = []
    for key, value in changed.items():
        if isinstance(record, dict) and key in record:
            if not _value_matches(value, record[key]):
                missing.append(key)
        elif json.dumps(value) not in detail_response.text and str(value) not in detail_response.text:
            missing.append(key)
    return missing


def _updated_verification_payload(body: dict[str, Any], field_types: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make valid, visibly changed values where the field shape is inferable."""
    updated = dict(body)
    field_types = field_types or {}
    for key, value in body.items():
        name = key.casefold()
        kind = str(field_types.get(key, "")).casefold()
        if _is_system_field(name):
            continue  # Owners, foreign keys, and timestamps are not user-editable.
        if isinstance(value, bool):
            updated[key] = not value
        elif isinstance(value, (int, float)):
            updated[key] = value + 1
        elif isinstance(value, str):
            if kind in {"select", "enum"}:
                continue
            if kind in {"email", "e-mail"} or name in {"email", "e_mail"}:
                updated[key] = "updated.verification@example.test"
            elif kind == "date" or name in {"date", "due_date", "spent_on", "workout_date", "target_date", "attended_on"}:
                updated[key] = "2026-01-16"
            elif kind == "month" or name in {"month", "budget_month"}:
                updated[key] = "2026-02"
            elif name in {"status", "priority", "category", "type", "intensity"}:
                continue  # May be an enum; preserve a valid declared option.
            else:
                updated[key] = f"{value}-updated"
    return updated


def _verify_update_delete(
    base_url: str,
    endpoints: list[dict[str, Any]],
    post_endpoint: dict[str, Any],
    record_id: Any,
    posted_body: dict[str, Any],
    result: dict[str, Any],
    timeout: float,
) -> str | None:
    """Exercise the declared update/delete routes for the created verification row."""
    table = post_endpoint.get("db_table")
    # Only single-record routes (/items/<id>) can update, read back, or delete
    # the verification row; nested routes like /items/<id>/children cannot.
    related = [item for item in endpoints if item.get("db_table") == table
               and (str(item.get("path", "")).endswith(">") or "<" not in str(item.get("path", "")))]
    update = next((item for item in related if item.get("method", "").upper() in {"PUT", "PATCH"}
                   and isinstance(item.get("request_json"), dict) and str(item.get("path", "")).endswith(">")), None)
    delete = next((item for item in related if item.get("method", "").upper() == "DELETE"
                   and str(item.get("path", "")).endswith(">")), None)
    if not update and not delete:
        return None  # Backward-compatible contracts may only describe read/create.
    if record_id is None:
        return "The create route did not return a record id, so update/delete could not be verified."

    update_payload = _updated_verification_payload(
        dict(update.get("request_json") or posted_body), post_endpoint.get("field_types")
    ) if update else None
    update_response = None
    if update:
        method = update.get("method", "PUT").upper()
        route = _route_for_record(update.get("path", ""), record_id)
        try:
            update_response = requests.request(method, f"{base_url}{route}", json=update_payload, timeout=timeout)
        except requests.RequestException as exc:
            return f"{method} {route} failed: {exc}"
        result["responses"].append({"method": method, "route": route, "status_code": update_response.status_code})
        if update_response.status_code not in {200, 201, 202, 204}:
            return f"{method} {route} returned HTTP {update_response.status_code}."
        result["workflow_checks"].append({"operation": "update", "status": "passed", "route": route})
        result["endpoints_verified"] += 1

        detail = next((item for item in related if item.get("method", "").upper() == "GET"
                       and str(item.get("path", "")).endswith(">")), None)
        if detail:
            detail_route = _route_for_record(detail["path"], record_id)
            try:
                detail_response = requests.get(f"{base_url}{detail_route}", timeout=timeout)
            except requests.RequestException as exc:
                return f"GET {detail_route} after update failed: {exc}"
            result["responses"].append({"method": "GET", "route": detail_route, "status_code": detail_response.status_code})
            if not detail_response.ok:
                return f"GET {detail_route} after update returned HTTP {detail_response.status_code}."
            missing = _updated_values_missing(detail_response, update_payload, posted_body) if update_payload else []
            if missing:
                return f"GET {detail_route} did not return the updated verification values for: {', '.join(missing)}."
            result["workflow_checks"].append({"operation": "read_after_update", "status": "passed", "route": detail_route})
            result["endpoints_verified"] += 1

    if delete:
        route = _route_for_record(delete.get("path", ""), record_id)
        try:
            deleted = requests.delete(f"{base_url}{route}", timeout=timeout)
        except requests.RequestException as exc:
            return f"DELETE {route} failed: {exc}"
        result["responses"].append({"method": "DELETE", "route": route, "status_code": deleted.status_code})
        if deleted.status_code not in {200, 202, 204}:
            return f"DELETE {route} returned HTTP {deleted.status_code}."
        result["workflow_checks"].append({"operation": "delete", "status": "passed", "route": route})
        result["endpoints_verified"] += 1

        detail = next((item for item in related if item.get("method", "").upper() == "GET"
                       and str(item.get("path", "")).endswith(">")), None)
        if detail:
            detail_route = _route_for_record(detail["path"], record_id)
            try:
                after_delete = requests.get(f"{base_url}{detail_route}", timeout=timeout)
            except requests.RequestException as exc:
                return f"GET {detail_route} after delete failed: {exc}"
            result["responses"].append({"method": "GET", "route": detail_route, "status_code": after_delete.status_code})
            if after_delete.status_code not in {404, 410}:
                return f"GET {detail_route} after delete should return HTTP 404 or 410; got {after_delete.status_code}."
            result["workflow_checks"].append({"operation": "read_after_delete", "status": "passed", "route": detail_route})
            result["endpoints_verified"] += 1
    return None
