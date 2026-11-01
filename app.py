import json
import ast
import atexit
import copy
import os
import re
import shutil
import socket
import subprocess
import threading
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from agent.llm_client import generate_with_backoff
from flask import Flask, Response, jsonify, request, send_from_directory

from agent.code_generator import extract_codegen_payload, generate_file_for_component
from agent.executor import PROCESS_HANDLES, execute_generated_app, restart_generated_app, start_generated_app, stop_generated_app, write_generated_files
from agent.showcase import contract_for_showcase, load_showcases, match_showcase, materialize_showcase, prepare_showcases
from agent.spec_generator import build_api_contract, extract_json_payload, generate_spec, generate_spec_and_plan
from agent.verifier import verify_generated_app
from agent.llm_client import get_call_metrics, get_call_stats
from config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL

# Serve frontend from the parent directory's frontend folder
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

FIX_ERROR_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "fix_error_prompt.txt"
JOB_STORE: dict[str, dict[str, Any]] = {}
MODIFICATION_LOCKS: dict[str, threading.Lock] = {}
GENERATION_SEMAPHORE = threading.Semaphore(max(1, int(os.getenv("MAX_CONCURRENT_GENERATIONS", "3"))))
APP_LIFECYCLE_LOCK = threading.RLock()
QUEUE_LOCK = threading.Lock()
GENERATION_QUEUE: list[str] = []
MAX_CONCURRENT_APPS = 8
MAX_APP_LIFETIME_SECONDS = 30 * 60
MAX_AUTO_REPAIR_ITERATIONS = 3
MAX_SYNTAX_REPAIR_ITERATIONS = 2
MANAGED_PROJECT_DIRS: set[Path] = set()
BACKEND_DIR = Path(__file__).resolve().parent
JOBS_DIR = BACKEND_DIR / "jobs"
GENERATED_PROJECTS_DIR = BACKEND_DIR / "generated_projects"


def _persist_job(job_id: str, job: dict[str, Any]) -> None:
    """Persist the small set of fields needed to restore status and previews."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    execution = job.get("execution") or {}
    metadata = {
        "status": job.get("status"),
        "current_step": job.get("current_step"),
        "spec": job.get("spec"),
        "plan": job.get("plan"),
        "project_dir": execution.get("project_dir"),
        "port": execution.get("port"),
        "result": job.get("result"),
        "prompt": job.get("prompt"),
        "created_at": job.get("created_at"),
        "duration_seconds": job.get("duration_seconds"),
        "app_name": (job.get("spec") or {}).get("app_name", job.get("app_name")),
        "versions": job.get("versions", []),
        "generated_files": job.get("generated_files", []),
        "preview_closed": job.get("preview_closed", False),
        "live_generation_failure": job.get("live_generation_failure"),
        "replay": bool(job.get("replay") or (job.get("result") or {}).get("replay")),
        "verification": job.get("verification"),
    }
    temporary = JOBS_DIR / f"{job_id}.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(JOBS_DIR / f"{job_id}.json")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    process_handle = PROCESS_HANDLES.get(pid)
    if process_handle is not None:
        return process_handle.poll() is None
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, check=False, timeout=3,
            )
            return f'"{pid}"' in result.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _restore_jobs_and_cleanup() -> None:
    """Restore persisted jobs, remove old projects, and stop untracked app processes."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    tracked_projects: set[Path] = set()
    for metadata_path in JOBS_DIR.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw_project_dir = metadata.get("project_dir")
            project_dir = Path(raw_project_dir).resolve() if raw_project_dir else None
            is_valid_project = bool(
                project_dir
                and project_dir.is_relative_to(GENERATED_PROJECTS_DIR.resolve())
                and project_dir.exists()
            )
            if is_valid_project:
                tracked_projects.add(project_dir)
                # Stop any child that survived an unclean server shutdown. It
                # will be relaunched on demand from the history gallery.
                stop_generated_app(project_dir)
            execution = {"project_dir": str(project_dir), "port": metadata.get("port")} if is_valid_project else None
            restored_status = metadata.get("status", "unknown")
            restored_result = metadata.get("result")
            if restored_status in {"queued", "running"}:
                restored_status = "failed"
                restored_result = {"success": False, "reason": "The server restarted before this build could finish. Please try again."}
            JOB_STORE[metadata_path.stem] = {
                "status": restored_status,
                "current_step": metadata.get("current_step", "restored"),
                "spec": metadata.get("spec"),
                "plan": metadata.get("plan"),
                "execution": execution,
                "result": restored_result,
                "replay": bool(metadata.get("replay") or (restored_result or {}).get("replay")),
                "live_generation_failure": metadata.get("live_generation_failure") or (restored_result or {}).get("live_generation_failure"),
                "verification": metadata.get("verification"),
                    "duration_seconds": metadata.get("duration_seconds"),
                    "prompt": metadata.get("prompt"),
                    "created_at": metadata.get("created_at"),
                    "versions": metadata.get("versions", []),
                    "generated_files": metadata.get("generated_files", []),
                    "preview_closed": metadata.get("preview_closed", False),
                "logs": ["Restored job metadata after server restart."],
            }
        except (OSError, ValueError, TypeError):
            continue

    cutoff = time.time() - 7 * 24 * 60 * 60
    if GENERATED_PROJECTS_DIR.exists():
        for project in GENERATED_PROJECTS_DIR.iterdir():
            if not project.is_dir() or project.name.startswith("preflight_"):
                continue
            process_file = project / "process.json"
            try:
                process_data = json.loads(process_file.read_text(encoding="utf-8"))
                pid = int(process_data.get("pid", 0))
            except (OSError, ValueError, TypeError):
                pid = 0
            if project.resolve() not in tracked_projects and _pid_is_running(pid):
                try:
                    if os.name == "nt":
                        import subprocess
                        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
                    else:
                        os.kill(pid, 15)
                except OSError:
                    pass
            try:
                if project.resolve() not in tracked_projects and project.stat().st_mtime < cutoff:
                    shutil.rmtree(project, ignore_errors=True)
            except OSError:
                pass


_restore_jobs_and_cleanup()


def _format_error_message(exc: Exception | str | None) -> str:
    """Normalize runtime exceptions into a human-readable message for logs and UI output."""
    message = str(exc or "Unknown error").strip() or "Unknown error"
    lowered = message.lower()

    if any(code in lowered for code in ("http 401", "http 402", "http 403", "(401)", "(402)", "(403)")):
        return f"LLM provider configuration or billing error: {message}"

    if "daily request quota exhausted" in lowered or ("generate_content_free_tier_requests" in lowered and "perday" in lowered):
        limit = re.search(r"\((\d+) requests/day\)", message, re.IGNORECASE)
        reason = f"Gemini free-tier daily request quota exhausted ({limit.group(1)} requests/day)" if limit else "Gemini free-tier daily request quota exhausted"
        retry = re.search(r"provider retry window\s+([^;\s]+)", message, re.IGNORECASE)
        if retry:
            reason += f"; retry window {retry.group(1)}"
        if "groq" in lowered and "429" in lowered:
            reason += "; Groq also returned HTTP 429"
        return reason
    if "http 429" in lowered or "(429)" in lowered or "429 resource_exhausted" in lowered:
        if "groq" in lowered and "429" in lowered:
            detail = message.split("Groq HTTP 429:", 1)[-1].strip() if "Groq HTTP 429:" in message else message
            return f"Groq API rate limit reached (HTTP 429): {detail}"
        reason = "Rate limit reached (HTTP 429) from the live model provider."
        return reason
    if "ollama request timed out" in lowered:
        return message
    if any(token in lowered for token in ("timeout", "timed out", "deadlineexceeded", "deadline exceeded")):
        return "Live generation timed out while waiting for the model provider."
    if any(token in lowered for token in ["invalid api key", "api key", "authentication", "401", "403", "forbidden"]):
        return "Live model authentication failed (HTTP 401/403). Check the provider API key in .env."
    if "repair request failed" in lowered:
        return message
    if "combined spec+plan response" in lowered or "code generation failed for" in lowered:
        return message
    if "json" in lowered and any(token in lowered for token in ("parse", "decode", "valid json")):
        return f"Model returned invalid JSON: {message}"
    if "413" in lowered or "payload too large" in lowered:
        return "Live model request rejected: prompt payload is too large (HTTP 413)."
    if any(token in lowered for token in ("connection", "name resolution", "dns", "network unavailable")):
        return "Live model request failed: provider network is unavailable."
    if "verifier failure" in lowered or "verification failed" in lowered or "failed verification" in lowered:
        detail = message.split(":", 1)[-1].strip() if ":" in message else "generated app did not pass checks"
        return f"Verifier failure: {detail}"
    return message


def _select_initial_showcase(force_demo: bool, prompt: str = "") -> dict[str, Any] | None:
    """Select a prompt-matched local replay only when the user enabled demo mode."""
    if not force_demo:
        return None
    showcases = load_showcases()
    if not showcases:
        return None
    return match_showcase(prompt, showcases) or showcases[0]


def _active_generated_count() -> int:
    count = 0
    for job in list(JOB_STORE.values()):
        if job.get("status") not in {"success", "running"}:
            continue
        execution = job.get("execution") or {}
        project_dir = Path(execution["project_dir"]) if execution.get("project_dir") else None
        if project_dir is None:
            continue
        process_file = project_dir / "process.json"
        try:
            process_data = json.loads(process_file.read_text(encoding="utf-8"))
            pid = int(process_data.get("pid", 0))
            started = float(process_data.get("started_at", process_file.stat().st_mtime))
        except (OSError, ValueError, TypeError):
            continue
        if time.time() - started >= MAX_APP_LIFETIME_SECONDS:
            continue
        # Metadata can outlive an app process (for example after a crash or a
        # server restart). Only live processes should consume a concurrency slot.
        if not _pid_is_running(pid):
            continue
        count += 1
    return count


def _make_room_for_app(log: Any, exclude_job_id: str | None = None) -> None:
    """Close the oldest completed preview when all local app slots are occupied.

    Callers hold APP_LIFECYCLE_LOCK through the subsequent launch so two preview
    requests cannot both claim the same final slot.
    """
    while _active_generated_count() >= MAX_CONCURRENT_APPS:
        candidates: list[tuple[float, str, dict[str, Any], Path]] = []
        for candidate_id, candidate in JOB_STORE.items():
            if candidate_id == exclude_job_id or candidate.get("status") != "success":
                continue
            execution = candidate.get("execution") or {}
            raw_dir = execution.get("project_dir")
            if not raw_dir:
                continue
            project_dir = Path(raw_dir)
            try:
                process_data = json.loads((project_dir / "process.json").read_text(encoding="utf-8"))
                pid = int(process_data.get("pid", 0))
                started = float(process_data.get("started_at", 0))
            except (OSError, ValueError, TypeError):
                continue
            if pid > 0 and _pid_is_running(pid):
                candidates.append((started, candidate_id, candidate, project_dir))

        if not candidates:
            raise RuntimeError("All preview slots are in use by builds that are still running. Try again in a moment.")

        _, old_job_id, old_job, old_project_dir = min(candidates, key=lambda item: item[0])
        stop_generated_app(old_project_dir)
        old_job["preview_closed"] = True
        old_job["preview_closed_at"] = datetime.now(timezone.utc).isoformat()
        _persist_job(old_job_id, old_job)
        app_name = (old_job.get("spec") or {}).get("app_name", "an older app")
        log(f"Closed the oldest preview ({app_name}) to make room. You can reopen it from build history.")

        if _active_generated_count() >= MAX_CONCURRENT_APPS:
            raise RuntimeError("ForgeAI could not free a preview slot. Try again in a moment.")


def _model_network_reachable() -> bool:
    if LLM_PROVIDER == "ollama":
        try:
            return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.2).status_code < 500
        except requests.RequestException:
            return False
    if not os.getenv("GROQ_API_KEY", "").strip():
        return False
    try:
        return requests.get("https://api.groq.com", timeout=1.2).status_code < 500
    except requests.RequestException:
        return False


def _run_showcase_job(job_id: str, showcase: dict[str, Any], log: Any) -> None:
    """Launch a locally stored, domain-specific pre-built showcase app."""
    job = JOB_STORE[job_id]
    previous_execution = job.get("execution") or {}
    if previous_execution.get("project_dir"):
        stop_generated_app(Path(previous_execution["project_dir"]))
    job["replay"] = True
    showcase_entities = [{"name": entity["name"], "fields": [{"name": field["name"], "type": field.get("type", "string")} for field in entity["fields"]]} for entity in showcase["entities"]]
    job["spec"] = {"app_name": showcase["name"], "entities": showcase_entities, "relationships": [],
        "pages_needed": ["Dashboard", *[entity["name"] for entity in showcase["entities"]], "Settings"],
        "routes_needed": ["GET /", *[f"GET /api/{entity['table']}" for entity in showcase["entities"]]],
        "api_contract": contract_for_showcase(showcase)}
    log(f"Prebuilt showcase replay selected: {showcase['name']}. No AI generation or external model calls are used.")
    job["current_step"] = "3_spec_generation"; time.sleep(0.18)
    log("Idea matched to a locally stored showcase blueprint")
    job["current_step"] = "5_code_generation"
    project_dir = GENERATED_PROJECTS_DIR / f"replay_{showcase['id']}_{uuid.uuid4().hex[:8]}"
    files = materialize_showcase(showcase, project_dir)
    for file in files:
        log(f"Loaded verified file: {file['filename']}")
    job["generated_files"] = files
    job["current_step"] = "6_execution"
    with APP_LIFECYCLE_LOCK:
        _make_room_for_app(log, exclude_job_id=job_id)
        execution = start_generated_app(project_dir)
    execution["process"].poll()
    job["execution"] = execution
    MANAGED_PROJECT_DIRS.add(project_dir)
    log("Showcase app started on its local port")
    job["current_step"] = "7_verification"
    verification = verify_generated_app(project_dir, api_contract=job["spec"]["api_contract"])
    job["verification"] = {"status": "ok" if verification.get("status") == "ok" else "failed", "verification": verification, "repairs": []}
    if verification.get("status") != "ok":
        reason = verification.get("error") or verification.get("status") or "unknown verifier result"
        raise RuntimeError(f"Stored showcase failed its live verification: {reason}")
    job["status"] = "success"; job["current_step"] = "8_complete"
    job["duration_seconds"] = round(time.time() - job.get("started_at", time.time()), 1)
    job["result"] = {"success": True, "path": str(project_dir), "port": execution["port"], "verification": job["verification"], "replay": True,
        "label": f"Prebuilt showcase replay: {showcase['name']}",
        "live_generation_failure": job.get("live_generation_failure")}
    log("Pre-built example passed runtime verification.")
    _persist_job(job_id, job)


def _stop_managed_apps() -> None:
    for project_dir in list(MANAGED_PROJECT_DIRS):
        stop_generated_app(project_dir)


def _expire_generated_apps() -> None:
    while True:
        time.sleep(30)
        for job_id, job in list(JOB_STORE.items()):
            execution = job.get("execution") or {}
            project_dir = Path(execution["project_dir"]) if execution.get("project_dir") else None
            if project_dir is None:
                continue
            try:
                process_file = project_dir / "process.json"
                process_data = json.loads(process_file.read_text(encoding="utf-8"))
                is_expired = time.time() - float(process_data.get("started_at", process_file.stat().st_mtime)) >= MAX_APP_LIFETIME_SECONDS
            except (OSError, ValueError, TypeError):
                is_expired = False
            if is_expired:
                stop_generated_app(project_dir)
                job["preview_closed"] = True
                _persist_job(job_id, job)


def _log_job_message(job: dict[str, Any], message: str) -> None:
    """Append a log line to a background job without crashing on missing state."""
    job.setdefault("logs", [])
    job["logs"].append(message)


def load_fix_error_prompt_template() -> str:
    """Load the prompt template used to repair generated app failures."""
    if FIX_ERROR_PROMPT_PATH.exists():
        return FIX_ERROR_PROMPT_PATH.read_text(encoding="utf-8")

    return """Fix the following app file.

Error reported:
{error_message}

Relevant file: {file_path}

Current file contents:
{file_content}

Return ONLY valid JSON with the structure:
{"filename": "string", "content": "string"}

Rules:
- Do not include markdown fences.
- Return corrected file contents only.
- Keep the app stable and consistent with the original spec.
"""


def build_fix_error_prompt(file_path: str, file_content: str, error_message: str) -> str:
    """Construct the repair prompt sent back to Groq after verification fails."""
    template = load_fix_error_prompt_template()
    error_tail = "\n".join(error_message.splitlines()[-30:])
    return (
        template.replace("{file_path}", file_path)
        .replace("{file_content}", file_content)
        .replace("{error_message}", error_tail)
    )


def _select_relevant_file(
    project_dir: Path,
    generated_files: list[dict[str, str]] | None = None,
    error_detail: str = "",
) -> str | None:
    """Pick the highest-probability failing file for a repair pass."""
    priority = ["app.py", "models.py", "routes.py", "main.py", "index.html", "style.css", "script.js"]

    if generated_files:
        name_map = {item["filename"].split("/")[-1]: item["filename"] for item in generated_files if "filename" in item}
        lowered_error = error_detail.lower()
        traceback_files = re.findall(r'File ["\']([^"\']+\.(?:py|html|css|js))["\']', error_detail)
        if traceback_files:
            # Python tracebacks list outer callers first and the failing frame
            # last, so repair the deepest named source file.
            traceback_path = Path(traceback_files[-1])
            candidates = [traceback_path.name]
            try:
                candidates.insert(0, str(traceback_path.resolve().relative_to(project_dir.resolve())))
            except (OSError, ValueError):
                pass
            for candidate in candidates:
                if candidate in name_map:
                    return name_map[candidate]
        if "no module named" in lowered_error or "modulenotfounderror" in lowered_error:
            requirements = next((name for name in name_map.values() if name.endswith("requirements.txt")), None)
            if requirements:
                return requirements
        for filename in name_map.values():
            if filename.lower() in lowered_error and (project_dir / filename).exists():
                return filename
        for candidate in priority:
            if candidate in name_map:
                return name_map[candidate]

    for candidate in priority:
        if (project_dir / candidate).exists():
            return candidate

    for path in sorted(project_dir.rglob("*.py")):
        if path.is_file():
            return str(path.relative_to(project_dir))

    for path in sorted(project_dir.rglob("*.html")):
        if path.is_file():
            return str(path.relative_to(project_dir))

    return None


def repair_failed_file(project_dir: str | Path, file_path: str, error_message: str, model_name: str = "openai/gpt-oss-120b") -> str:
    """Request a corrected version of a failing file from Groq and overwrite it on disk."""
    project_dir = Path(project_dir)
    target = project_dir / file_path
    if not target.exists():
        raise FileNotFoundError(f"Cannot repair missing file: {target}")

    current_content = target.read_text(encoding="utf-8")
    prompt = build_fix_error_prompt(file_path, current_content, error_message)

    try:
        payload = None
        for attempt in range(2):
            repair_prompt = prompt
            if attempt:
                repair_prompt += (
                    "\n\nRETRY: Return a complete JSON object with string keys filename and content. "
                    "Escape all newlines, quotes, and backslashes in content. Return no markdown or commentary."
                )
            response = generate_with_backoff(
                repair_prompt, preferred_model=model_name, max_retries=1,
                json_mode=True, max_tokens=4500, step="fix",
                cache_context={"stage": "repair", "file": file_path},
            )
            try:
                if not getattr(response, "complete", True):
                    raise ValueError("Provider marked the repair response as truncated.")
                payload = extract_codegen_payload(getattr(response, "text", ""))
                if payload["filename"] != file_path:
                    raise ValueError(f"Repair returned {payload['filename']!r}; expected {file_path!r}.")
                if file_path.endswith(".py"):
                    compile(payload["content"], file_path, "exec")
                if hasattr(response, "cache_validated"):
                    response.cache_validated()
                break
            except (ValueError, TypeError, SyntaxError, json.JSONDecodeError):
                if attempt:
                    raise
        if payload is None:
            raise ValueError("Repair model returned invalid JSON twice.")
    except Exception as exc:
        raise RuntimeError(
            f"Repair request failed for {file_path}; generated Python has a syntax error ({error_message}): "
            f"{_format_error_message(exc)}"
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload["content"], encoding="utf-8")
    return file_path


def self_correct_generated_app(
    project_dir: str | Path,
    generated_files: list[dict[str, str]] | None = None,
    routes: list[str] | None = None,
    max_attempts: int = 4,
    api_contract: dict[str, Any] | None = None,
    on_repair: Any = None,
    install_dependencies_on_repair: bool = True,
) -> dict[str, Any]:
    """Verify a generated app and repair failed files up to max_attempts times."""
    project_dir = Path(project_dir)
    max_attempts = min(MAX_AUTO_REPAIR_ITERATIONS, max(0, int(max_attempts)))
    routes = routes or ["/"]

    verification = verify_generated_app(project_dir, routes=routes, api_contract=api_contract)
    repairs: list[dict[str, str]] = []
    for attempt in range(1, max_attempts + 1):
        if verification.get("status") == "ok":
            return {
                "status": "ok",
                "attempts_used": attempt,
                "repairs": repairs,
                "verification": verification,
            }

        if verification.get("error") is None:
            return {
                "status": "unknown",
                "attempts_used": attempt,
                "repairs": repairs,
                "verification": verification,
            }

        error_detail = f"{verification.get('error') or 'Verification failed'}\n\n{verification.get('traceback') or ''}".strip()
        failed_file = _select_relevant_file(project_dir, generated_files, error_detail)
        if failed_file is None:
            return {
                "status": "failed_no_repair_target",
                "attempts_used": attempt,
                "repairs": repairs,
                "verification": verification,
            }

        try:
            before_content = (project_dir / failed_file).read_text(encoding="utf-8")
            repaired_name = repair_failed_file(project_dir, failed_file, error_detail)
            after_content = (project_dir / repaired_name).read_text(encoding="utf-8")
            if after_content != before_content and on_repair:
                on_repair(f"Self-repair applied to {failed_file} after verification found an error.")
            before_lines = before_content.splitlines()
            after_lines = after_content.splitlines()
            changed_before = next((line for line in before_lines if line not in after_lines), "No previous line")
            changed_after = next((line for line in after_lines if line not in before_lines), "Updated implementation")
            if after_content != before_content:
                repairs.append({"file": repaired_name, "before": changed_before, "after": changed_after})
            if generated_files:
                for generated in generated_files:
                    if generated.get("filename") == repaired_name:
                        generated["content"] = after_content
            restart_generated_app(project_dir, install=install_dependencies_on_repair)
        except Exception as repair_exc:
            return {
                "status": "failed_repair",
                "attempts_used": attempt,
                "error": _format_error_message(repair_exc),
                "traceback": str(repair_exc),
                "repairs": repairs,
                "verification": verification,
            }

        verification = verify_generated_app(project_dir, routes=routes, api_contract=api_contract)

        if verification.get("status") == "ok":
            return {
                "status": "ok",
                "attempts_used": attempt + 1,
                "repaired_file": repaired_name,
                "repairs": repairs,
                "verification": verification,
            }

    final_status = verification.get("status", "failed_after_retries")
    return {
        "status": "failed_after_retries",
        "attempts_used": max_attempts,
        "error": _format_error_message(verification.get("error") or "App is still failing after the retry limit."),
        "traceback": verification.get("traceback") or "No traceback captured.",
        "repairs": repairs,
        "verification": verification,
    }


def repair_generated_syntax_errors(
    project_dir: str | Path,
    max_attempts: int = 4,
) -> None:
    """Compile generated Python files and ask the model to fix syntax errors."""
    project_dir = Path(project_dir)
    max_attempts = min(MAX_SYNTAX_REPAIR_ITERATIONS, max(0, int(max_attempts)))
    for _ in range(max_attempts + 1):
        syntax_error: tuple[str, str] | None = None
        for path in sorted(project_dir.rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as exc:
                syntax_error = (str(path.relative_to(project_dir)), str(exc))
                break

        if syntax_error is None:
            return

        file_path, error_message = syntax_error
        if _ == max_attempts:
            raise RuntimeError(f"Generated Python syntax error in {file_path}: {error_message}")
        repair_failed_file(project_dir, file_path, error_message)


def _prepare_generated_files(
    generated_files: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Repair generated Python syntax errors in a temporary project before execution."""
    repaired_files = [dict(item) for item in generated_files]
    generated_root = Path(__file__).resolve().parent / "generated_projects"
    generated_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preflight_", dir=generated_root) as temp_dir:
        preflight_dir = Path(temp_dir)
        for item in repaired_files:
            if not item["filename"].endswith(".py"):
                continue
            # One repair attempt keeps preflight bounded. Repeated model repairs
            # were adding minutes while masking the original generated syntax error.
            for attempt in range(2):
                try:
                    ast.parse(item["content"], filename=item["filename"])
                    break
                except SyntaxError as exc:
                    if attempt == 1:
                        raise RuntimeError(f"Generated Python syntax error in {item['filename']}: {exc}") from exc
                    target = preflight_dir / item["filename"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(item["content"], encoding="utf-8")
                    repair_failed_file(preflight_dir, item["filename"], str(exc))
                    item["content"] = target.read_text(encoding="utf-8")
        return repaired_files


def validate_generated_project_files(
    generated_files: list[dict[str, str]], api_contract: dict[str, Any] | None = None,
) -> list[str]:
    """Check required assets and declared API routes before starting a project."""
    by_name = {item.get("filename"): item.get("content", "") for item in generated_files if isinstance(item, dict)}
    required = {"requirements.txt", "models.py", "routes.py", "app.py", "index.html", "style.css", "script.js"}
    issues = [f"Required generated file is missing: {name}" for name in sorted(required - by_name.keys())]
    html = by_name.get("index.html", "")
    if "style.css" not in html:
        issues.append("index.html does not link the generated style.css.")
    if "script.js" not in html:
        issues.append("index.html does not load the generated script.js.")
    if not by_name.get("script.js", "").strip():
        issues.append("script.js is empty; the generated interface has no client behavior.")

    try:
        tree = ast.parse(by_name.get("routes.py", ""), filename="routes.py")
        declared: set[tuple[str, str]] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                    continue
                if not isinstance(decorator.func.value, ast.Name) or decorator.func.value.id != "bp":
                    continue
                method = decorator.func.attr.upper()
                if method == "ROUTE":
                    method = "GET"
                try:
                    path = ast.literal_eval(decorator.args[0])
                except (IndexError, ValueError, TypeError):
                    continue
                if not isinstance(path, str):
                    continue
                methods = ["GET"]
                for keyword in decorator.keywords:
                    if keyword.arg == "methods":
                        try:
                            methods = [str(value).upper() for value in ast.literal_eval(keyword.value)]
                        except (ValueError, TypeError):
                            methods = []
                declared.update((candidate, path) for candidate in methods)
        contract_endpoints = (api_contract or {}).get("endpoints", [])
        missing_routes = [f"{item.get('method')} {item.get('path')}" for item in contract_endpoints
                          if (str(item.get("method", "GET")).upper(), item.get("path")) not in declared]
        if missing_routes:
            issues.append("routes.py is missing contract endpoint(s): " + ", ".join(missing_routes))
        script = by_name.get("script.js", "")
        missing_client_paths = sorted({
            str(item.get("path")) for item in contract_endpoints
            if item.get("db_table") and "<" not in str(item.get("path", ""))
            and str(item.get("path", "")) not in script
        })
        if missing_client_paths:
            issues.append("script.js does not reference contract API collection path(s): " + ", ".join(missing_client_paths))
    except SyntaxError as exc:
        issues.append(f"routes.py has invalid Python syntax: {exc}")
    return issues


def _deterministic_file(component_name: str) -> str | None:
    if component_name == "requirements.txt":
        return "Flask>=3.1,<4\n"
    if component_name == "app.py":
        return '''import os\nfrom flask import Flask, jsonify, send_from_directory\nfrom routes import bp\nimport models\n\napp = Flask(__name__)\n\n# Initialize the generated schema before serving any API request. The models\n# and routes modules are required to share this database connection/schema.\nif callable(getattr(models, "init_app", None)):\n    models.init_app(app)\nelif callable(getattr(models, "init_db", None)):\n    with app.app_context():\n        models.init_db()\n\n@app.get("/")\ndef index():\n    return send_from_directory(app.root_path, "index.html")\n\n# Keep the generated UI root ahead of API routes, even if a model emits a duplicate.\napp.register_blueprint(bp)\n\n@app.get("/style.css")\ndef stylesheet():\n    return send_from_directory(app.root_path, "style.css")\n\n@app.get("/script.js")\ndef javascript():\n    return send_from_directory(app.root_path, "script.js")\n\n@app.errorhandler(404)\ndef not_found(_error):\n    return jsonify({"error": "Not found"}), 404\n\nif __name__ == "__main__":\n    port = int(os.environ.get("PORT", 5000))\n    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)\n'''
    return None


def _generate_project_files(spec: dict[str, Any], log: Any = None, on_file: Any = None, user_prompt: str = "",
                            existing_files: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Generate in dependency order, using deterministic boilerplate and prior files as context."""
    component_names = ["requirements.txt", "models.py", "routes.py", "app.py", "index.html", "style.css", "script.js"]
    existing_by_name = {
        item.get("filename"): item.get("content")
        for item in (existing_files or [])
        if isinstance(item, dict) and isinstance(item.get("filename"), str)
        and isinstance(item.get("content"), str) and item.get("content", "").strip()
    }
    generated_files: list[dict[str, str]] = []
    purposes = {
        "requirements.txt": "Python dependencies imported by the generated backend.",
        "models.py": "SQLite schema, connection helpers, and persistence utilities for the contract.",
        "routes.py": "Flask blueprint implementing the API contract with JSON responses.",
        "index.html": "Accessible UI markup that calls only contract endpoints.",
        "style.css": "Responsive styling for index.html.",
        "script.js": "Shared client shell, hash router, dashboard, settings page, and page-module registry using the contract's request and response shapes.",
    }
    for component_name in component_names:
        content = _deterministic_file(component_name)
        if content is None:
            existing_content = existing_by_name.get(component_name)
            if existing_content:
                content = existing_content
                if log:
                    log(f"Reused completed {component_name} from this job")
            elif component_name == "index.html":
                file_context = []
            elif component_name == "style.css":
                file_context = [item for item in generated_files if item["filename"] == "index.html"]
            elif component_name == "script.js":
                # The contract and shell requirements below already describe
                # the client interfaces; CSS adds no useful context for JS.
                file_context = [item for item in generated_files if item["filename"] == "index.html"]
            else:
                file_context = generated_files[-2:]
            if not existing_content:
                payload = generate_file_for_component(
                    spec,
                    component_name,
                    purposes[component_name],
                    context_files=file_context,
                    on_model_switch=log,
                    user_prompt=user_prompt,
                    model_name=os.getenv("GROQ_MODEL_CODEGEN", "openai/gpt-oss-120b"),
                )
                content = payload["content"]
        generated_files.append({"filename": component_name, "content": content})
        if on_file:
            on_file(generated_files)
        if log:
            log(f"Generated {component_name}")

        # Generate each domain page as a small, focused JS module, then merge
        # those modules into script.js so the app still ships as one frontend
        # file. The shared shell/router is generated first and passed as context.
        if component_name == "script.js":
            page_specs = {
                str(page.get("name", "")).casefold(): page
                for page in spec.get("pages", []) if isinstance(page, dict)
            }
            entities = [entity for entity in spec.get("entities", []) if isinstance(entity, dict)]
            seen: set[str] = set()
            for entity in entities:
                page_name = str(entity.get("name", "")).strip()
                if not page_name or page_name.casefold() in {"dashboard", "settings"}:
                    continue
                page = page_specs.get(page_name.casefold(), {})
                slug = re.sub(r"[^a-z0-9]+", "_", str(entity.get("name", page_name)).lower()).strip("_")
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                module_name = f"page_{slug}.js"
                page_purpose = str(page.get("purpose", ""))
                page_widgets = ", ".join(str(widget) for widget in page.get("widgets", []))
                module_purpose = (
                    f"A focused browser JavaScript page module for the {page_name} page. "
                    f"{page_purpose} Widgets: {page_widgets}. Implement that page's domain workflow and CRUD interactions. Register it with "
                    f"window.ForgeAI.registerPage('{slug}', renderFunction), where renderFunction(ctx) "
                    "receives {root, profile, api, search}. Do not define a second router or shell."
                )
                script_file = next(item for item in generated_files if item["filename"] == "script.js")
                module_marker = f"/* ---- {page_name} page module ---- */"
                if module_marker in script_file["content"]:
                    continue
                module = generate_file_for_component(
                    spec, module_name, module_purpose,
                    # The module contract is stated in module_purpose and the
                    # API contract is included automatically. Avoid resending
                    # the full shared shell for every entity page.
                    context_files=[],
                    on_model_switch=log, user_prompt=user_prompt,
                    model_name=os.getenv("GROQ_MODEL_CODEGEN", "openai/gpt-oss-120b"),
                )
                script_file["content"] += f"\n\n/* ---- {page_name} page module ---- */\n" + module["content"]
                if on_file:
                    on_file(generated_files)
                if log:
                    log(f"Generated and merged {module_name} into script.js")
    return generated_files


def generate_and_verify_app(user_prompt: str, routes: list[str] | None = None) -> dict[str, Any]:
    """Generate a project, execute and verify it, with a bounded repair loop."""
    spec, plan = generate_spec_and_plan(user_prompt)

    generated_files = _generate_project_files(spec, user_prompt=user_prompt)

    generated_files = _prepare_generated_files(generated_files)
    issues = validate_generated_project_files(generated_files, spec.get("api_contract"))
    if issues:
        raise RuntimeError("Generated project preflight failed: " + "; ".join(issues))
    execution = execute_generated_app(spec.get("app_name", "generated_app"), generated_files)
    self_check = self_correct_generated_app(
        execution["project_dir"], generated_files=generated_files,
        routes=routes or ["/"], api_contract=spec.get("api_contract"),
    )

    return {
        "spec": spec,
        "plan": plan,
        "execution": execution,
        "verification": self_check,
    }


def run_generation_job(job_id: str, user_prompt: str) -> None:
    """Run the full app-generation pipeline for a given prompt in the background."""
    job = JOB_STORE.get(job_id)
    if job is None:
        return

    GENERATION_SEMAPHORE.acquire()
    with QUEUE_LOCK:
        if job_id in GENERATION_QUEUE:
            GENERATION_QUEUE.remove(job_id)

    def log(message: str) -> None:
        _log_job_message(job, str(message))
        _persist_job(job_id, job)


    try:
        job["status"] = "running"
        # A normal prompt always gets the personalized generator first. A
        # preset replay is selected only when the visitor explicitly forces demo mode.
        showcase = None
        if job.get("force_demo"):
            showcase = next(
                (item for item in load_showcases() if item.get("id") == job.get("showcase_id")),
                None,
            )
            if showcase is None:
                raise RuntimeError("The selected offline showcase is no longer available. Check backend/showcase/catalog.json.")
        if showcase:
            _run_showcase_job(job_id, showcase, log)
            return
        job["current_step"] = "3_spec_generation"
        log("Starting combined spec+plan generation")
        if isinstance(job.get("spec"), dict) and isinstance(job.get("plan"), list) and all(isinstance(step, str) for step in job["plan"]):
            spec, plan = job["spec"], job["plan"]
            log("Reusing completed spec and plan from this job")
        else:
            spec, plan = generate_spec_and_plan(user_prompt, on_model_switch=log)
        job["spec"] = spec
        job["plan"] = plan
        _persist_job(job_id, job)
        log("Spec and plan generated")

        job["current_step"] = "5_code_generation"
        log("Generating app files")
        generated_files = _generate_project_files(
            spec, log=log,
            on_file=lambda files: job.update({"generated_files": [dict(item) for item in files]}),
            user_prompt=user_prompt,
            existing_files=job.get("generated_files"),
        )
        job["generated_files"] = generated_files
        log(f"Generated {len(generated_files)} file(s)")

        # Catch malformed Python before pip install or background launch.
        generated_files = _prepare_generated_files(generated_files)
        job["generated_files"] = generated_files
        job["current_step"] = "6_preflight"
        log("Checking required files and contract routes before launch")
        preflight_issues = validate_generated_project_files(generated_files, spec.get("api_contract"))
        if preflight_issues:
            raise RuntimeError("Generated project preflight failed: " + "; ".join(preflight_issues))
        log("Generated files, frontend assets, and declared API routes passed preflight")

        job["current_step"] = "6_execution"
        log("Executing generated app")
        with APP_LIFECYCLE_LOCK:
            _make_room_for_app(log, exclude_job_id=job_id)
            execution = execute_generated_app(spec.get("app_name", "generated_app"), generated_files)
        job["execution"] = execution
        MANAGED_PROJECT_DIRS.add(Path(execution["project_dir"]))
        _persist_job(job_id, job)
        log(f"App started on port {execution.get('port')}")

        job["current_step"] = "7_verification"
        log("Verifying generated app")
        verification = self_correct_generated_app(
            execution["project_dir"], generated_files=generated_files,
            routes=["/"], api_contract=spec.get("api_contract"), on_repair=log,
        )
        job["verification"] = verification

        if verification.get("status") == "ok":
            job["current_step"] = "8_complete"
            job["status"] = "success"
            job["duration_seconds"] = round(time.time() - job.get("started_at", time.time()), 1)
            job["result"] = {
                "success": True,
                "path": execution.get("project_dir"),
                "port": execution.get("port"),
                "verification": verification,
            }
            log("Verification passed. App is working.")
            return

        job["current_step"] = "8_failed"
        job["status"] = "failed"
        job["result"] = {
            "success": False,
            "reason": _format_error_message(verification.get("error") or "App failed verification."),
            "details": verification,
        }
        log(f"Verification failed: {job['result']['reason']}")
        raise RuntimeError(f"Verifier failure: {job['result']['reason']}")

    except Exception as exc:
        failure_reason = _format_error_message(exc)
        if job.get("force_demo"):
            log(f"Demo replay failed: {failure_reason}")
            showcase = None
        else:
            job["live_generation_failure"] = failure_reason
            log(f"Live generation failed: {failure_reason}")
            showcase = match_showcase(user_prompt)
        if not job.get("replay") and showcase is not None:
            try:
                log(f"Live generation failed: {failure_reason}. Showing a prebuilt showcase replay instead.")
                _run_showcase_job(job_id, showcase, log)
                job["result"]["live_generation_failure"] = failure_reason
                job["result"]["label"] = f"Prebuilt showcase replay: {showcase['name']} (live generation unavailable)"
                _persist_job(job_id, job)
                return
            except Exception as replay_error:
                exc = replay_error
        job["status"] = "failed"
        job["current_step"] = "error"
        error_message = _format_error_message(exc)
        job["result"] = {
            "success": False,
            "reason": error_message,
        }
        _log_job_message(job, f"ERROR: {error_message}")
        _persist_job(job_id, job)
    finally:
        try:
            if job.get("status") != "success":
                execution = job.get("execution") or {}
                project_dir = execution.get("project_dir")
                if project_dir:
                    failed_project = Path(project_dir)
                    stop_generated_app(failed_project)
                    MANAGED_PROJECT_DIRS.discard(failed_project)
        finally:
            GENERATION_SEMAPHORE.release()

MODIFIABLE_FILES = ["models.py", "routes.py", "index.html", "style.css", "script.js"]


def _fallback_affected_files(change: str) -> list[str]:
    text = change.lower()
    if any(term in text for term in ("field", "column", "entity", "database", "save", "store", "record")):
        chosen = ["models.py", "routes.py", "index.html", "script.js"]
        if any(term in text for term in ("style", "color", "dark mode", "layout")):
            chosen.append("style.css")
    elif any(term in text for term in ("search", "filter", "sort", "validation", "button", "interaction")):
        chosen = ["routes.py", "index.html", "script.js"]
    elif any(term in text for term in ("style", "color", "dark mode", "layout", "font", "spacing")):
        chosen = ["index.html", "style.css", "script.js"]
    else:
        chosen = ["index.html", "script.js", "style.css"]
    return [filename for filename in MODIFIABLE_FILES if filename in chosen]


def _plan_modification(
    change: str,
    spec: dict[str, Any],
    generated_files: list[dict[str, str]],
    on_model_switch: Any = None,
) -> tuple[list[str], dict[str, Any]]:
    """Ask the model for the smallest affected-file set and a narrowly updated spec."""
    allowed = [item["filename"] for item in generated_files if item.get("filename") in MODIFIABLE_FILES]
    prompt = (
        "You are planning a small change to an existing Flask + SQLite app.\n"
        "Return JSON with exactly `files` (an array of filenames from the allowed list) and `updated_spec` "
        "(the full revised spec object). Keep the request narrowly scoped. Include every dependent file that must change, "
        "but never include app.py or requirements.txt. Preserve unrelated features and existing API routes.\n"
        f"Allowed files: {json.dumps(allowed)}\n\nCurrent spec:\n{json.dumps(spec, indent=2)}\n\n"
        f"Requested change:\n{change}\n"
    )
    # Provider/auth/quota failures propagate. Do not hide them behind a
    # deterministic plan and spend more calls on files.
    response = generate_with_backoff(prompt, preferred_model="openai/gpt-oss-120b", json_mode=True,
                                     on_model_switch=on_model_switch, max_tokens=800, step="planner",
                                     cache_context={"stage": "modification_plan"})
    try:
        if not getattr(response, "complete", True):
            raise ValueError("Provider marked the modification plan as truncated.")
        parsed = extract_json_payload(getattr(response, "text", ""))
        selected = parsed.get("files", [])
        revised = parsed.get("updated_spec")
        response_valid = (
            isinstance(selected, list) and bool(selected)
            and all(isinstance(name, str) and name in allowed for name in selected)
            and isinstance(revised, dict)
        )
        if not response_valid:
            if on_model_switch:
                on_model_switch("Modification planner response was invalid; using deterministic file selection and the existing spec.")
            return _fallback_affected_files(change), copy.deepcopy(spec)
        files = [filename for filename in MODIFIABLE_FILES if filename in selected and filename in allowed]
        revised = {**spec, **revised}
        revised.pop("api_contract", None)
        revised["api_contract"] = build_api_contract(revised)
        if hasattr(response, "cache_validated"):
            response.cache_validated()
        return files, revised
    except (ValueError, TypeError, json.JSONDecodeError):
        if on_model_switch:
            on_model_switch("Modification planner returned malformed JSON; using deterministic file selection and the existing spec.")
        return _fallback_affected_files(change), copy.deepcopy(spec)


def run_modification_job(modification_id: str, parent_id: str, change: str, lock: threading.Lock) -> None:
    """Regenerate only selected files, verify them, and publish a new reversible app version."""
    task = JOB_STORE.get(modification_id)
    parent = JOB_STORE.get(parent_id)
    if task is None or parent is None:
        lock.release()
        return

    original_files = copy.deepcopy(parent.get("generated_files", []))
    original_spec = copy.deepcopy(parent.get("spec") or {})
    project_dir = Path(parent["execution"]["project_dir"])
    changed_on_disk = False

    def log(message: str) -> None:
        _log_job_message(task, message)

    try:
        task["status"] = "running"
        task["current_step"] = "modify_file_selection"
        log("Planning the smallest safe file update.")
        files, revised_spec = _plan_modification(change, original_spec, original_files, on_model_switch=log)
        task["current_step"] = "modify_code_generation"
        task["spec"] = revised_spec
        log(f"Updating {', '.join(files)}")
        spec_for_generation = {**revised_spec, "modification_request": change}
        working_files = copy.deepcopy(original_files)
        updated_entries: list[dict[str, str]] = []
        for filename in files:
            payload = generate_file_for_component(
                spec_for_generation,
                filename,
                f"Apply only this requested change: {change}. Preserve existing behavior and keep all interfaces consistent.",
                context_files=working_files,
                on_model_switch=log,
                user_prompt=f"Original app request: {parent.get('prompt', '')}\nRequested modification: {change}",
            )
            content = payload["content"]
            updated = {"filename": filename, "content": content}
            working_files = [updated if item["filename"] == filename else item for item in working_files]
            updated_entries.append(updated)
            log(f"Updated {filename}")

        updated_entries = _prepare_generated_files(updated_entries)
        working_files = [
            next((entry for entry in updated_entries if entry["filename"] == item["filename"]), item)
            for item in working_files
        ]
        write_generated_files(project_dir, updated_entries)
        changed_on_disk = True
        task["current_step"] = "modify_verification"
        restarted = restart_generated_app(project_dir, install=False)
        parent["execution"]["port"] = restarted["port"]
        parent["execution"]["process"] = restarted["process"]
        log("Rechecking every API endpoint and the sample database record.")
        verification = self_correct_generated_app(
            project_dir,
            generated_files=working_files,
            routes=["/"],
            api_contract=revised_spec.get("api_contract"),
            on_repair=log,
            install_dependencies_on_repair=False,
        )
        if verification.get("status") != "ok":
            raise RuntimeError(verification.get("error") or "Updated app did not pass verification.")

        parent.setdefault("versions", []).append({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generated_files": original_files,
            "spec": original_spec,
        })
        parent["versions"] = parent["versions"][-12:]
        parent["generated_files"] = working_files
        parent["spec"] = revised_spec
        parent["verification"] = verification
        parent["result"] = {
            **(parent.get("result") or {}),
            "success": True,
            "verification": verification,
        }
        task["status"] = "success"
        task["current_step"] = "modify_complete"
        task["generated_files"] = working_files
        task["verification"] = verification
        task["result"] = {"success": True, "parent_job_id": parent_id, "modified_files": files}
        log("Change verified. The preview is updated and this version can be undone.")
        _persist_job(parent_id, parent)
    except Exception as exc:
        if changed_on_disk:
            try:
                write_generated_files(project_dir, original_files)
                restarted = restart_generated_app(project_dir, install=False)
                parent["execution"]["port"] = restarted["port"]
                parent["execution"]["process"] = restarted["process"]
            except Exception:
                pass
        task["status"] = "failed"
        task["current_step"] = "modify_failed"
        task["result"] = {"success": False, "reason": "ForgeAI couldn't apply that change. The previous version is still running."}
        task["verification"] = {"error": str(exc), "traceback": str(exc)}
        log("The change did not pass verification. The previous app version was restored.")
    finally:
        lock.release()


@app.post("/modify/<job_id>")
def modify_generated_app(job_id: str):
    """Queue a focused modification for a successfully generated app."""
    parent = JOB_STORE.get(job_id)
    if not parent or parent.get("status") != "success" or not parent.get("execution"):
        return jsonify({"error": "A successful generated app is required before modifying it."}), 404
    payload = request.get_json(silent=True) or {}
    change = str(payload.get("change") or "").strip()
    if not change:
        return jsonify({"error": "Describe the change you want to make."}), 400
    lock = MODIFICATION_LOCKS.setdefault(job_id, threading.Lock())
    if not lock.acquire(blocking=False):
        return jsonify({"error": "A change is already being verified for this app."}), 409
    modification_id = uuid.uuid4().hex
    JOB_STORE[modification_id] = {
        "status": "queued", "current_step": "queued", "logs": [], "result": None,
        "spec": None, "generated_files": [], "verification": None, "parent_job_id": job_id,
    }
    thread = threading.Thread(target=run_modification_job, args=(modification_id, job_id, change, lock), daemon=True)
    thread.start()
    return jsonify({"job_id": modification_id, "parent_job_id": job_id, "status": "queued"})


@app.post("/undo/<job_id>")
def undo_generated_app(job_id: str):
    """Restore and verify the previous generated app version."""
    job = JOB_STORE.get(job_id)
    if not job or not job.get("versions"):
        return jsonify({"error": "There is no earlier version to restore."}), 404
    lock = MODIFICATION_LOCKS.setdefault(job_id, threading.Lock())
    if not lock.acquire(blocking=False):
        return jsonify({"error": "A change is still being verified for this app."}), 409
    try:
        project_dir = Path(job["execution"]["project_dir"])
        current_files = copy.deepcopy(job.get("generated_files", []))
        current_spec = copy.deepcopy(job.get("spec") or {})
        previous = job["versions"][-1]
        previous_files = copy.deepcopy(previous["generated_files"])
        previous_spec = copy.deepcopy(previous["spec"])
        write_generated_files(project_dir, previous_files)
        restarted = restart_generated_app(project_dir, install=False)
        job["execution"]["port"] = restarted["port"]
        job["execution"]["process"] = restarted["process"]
        verification = self_correct_generated_app(
            project_dir, generated_files=previous_files, routes=["/"],
            api_contract=previous_spec.get("api_contract"), install_dependencies_on_repair=False,
        )
        if verification.get("status") != "ok":
            write_generated_files(project_dir, current_files)
            restarted = restart_generated_app(project_dir, install=False)
            job["execution"]["port"] = restarted["port"]
            job["execution"]["process"] = restarted["process"]
            return jsonify({"error": "The earlier version did not pass verification, so the current version was kept."}), 500
        job["versions"].pop()
        job["generated_files"] = previous_files
        job["spec"] = previous_spec
        job["verification"] = verification
        job["result"] = {**(job.get("result") or {}), "success": True, "verification": verification}
        _persist_job(job_id, job)
        return jsonify({"success": True, "spec": previous_spec, "generated_files": previous_files, "version_count": len(job["versions"]) + 1})
    except Exception as exc:
        return jsonify({"error": "Could not restore the earlier app version.", "details": str(exc)}), 500
    finally:
        lock.release()

@app.route("/preview/<job_id>/", defaults={"subpath": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@app.route("/preview/<job_id>/<path:subpath>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def preview_generated_app(job_id: str, subpath: str) -> Response:
    """Proxy preview traffic and restart its generated server if it has stopped."""
    job = JOB_STORE.get(job_id)
    execution = job.get("execution") if job else None
    project_dir = Path(execution["project_dir"]) if execution and execution.get("project_dir") else None
    if project_dir is None or not project_dir.exists():
        return Response("This generated preview is no longer available. Generate the app again.", status=404)

    expired = False
    try:
        process_data = json.loads((project_dir / "process.json").read_text(encoding="utf-8"))
        if time.time() - float(process_data.get("started_at", 0)) >= MAX_APP_LIFETIME_SECONDS:
            stop_generated_app(project_dir)
            expired = True
    except (OSError, ValueError, TypeError):
        pass

    try:
        port = int((project_dir / "port.txt").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        port = int(execution.get("port", 0))

    quoted_subpath = quote(subpath, safe="/@:+")
    request_headers = {
        key: value for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection"}
    }

    def forward(port_number: int) -> requests.Response:
        url = f"http://127.0.0.1:{port_number}/{quoted_subpath}" if subpath else f"http://127.0.0.1:{port_number}/"
        return requests.request(
            request.method,
            url,
            params=request.args,
            data=request.get_data(),
            headers=request_headers,
            allow_redirects=False,
            timeout=30,
        )

    try:
        if expired:
            raise requests.ConnectionError("Preview lifetime elapsed")
        upstream = forward(port)
    except requests.RequestException:
        try:
            with APP_LIFECYCLE_LOCK:
                _make_room_for_app(
                    lambda message: (_log_job_message(job, message), _persist_job(job_id, job)),
                    exclude_job_id=job_id,
                )
                restarted = restart_generated_app(project_dir)
            execution["port"] = restarted["port"]
            execution["process"] = restarted["process"]
            job["preview_closed"] = False
            _persist_job(job_id, job)
            port = restarted["port"]
            for attempt in range(20):
                try:
                    upstream = forward(port)
                    break
                except requests.RequestException:
                    if attempt == 19:
                        raise
                    time.sleep(0.5)
        except Exception as exc:
            return Response(f"Preview server could not be restarted: {exc}", status=503)

    body = upstream.content
    content_type = upstream.headers.get("Content-Type", "")
    prefix = f"/preview/{job_id}"
    if any(kind in content_type.lower() for kind in ("text/html", "text/css", "javascript")):
        text_body = body.decode(upstream.encoding or "utf-8", errors="replace")
        text_body = re.sub(r'((?:href|src|action)\s*=\s*["\'])/(?!/)', rf"\1{prefix}/", text_body, flags=re.IGNORECASE)
        text_body = re.sub(r'(url\(\s*["\']?)/(?!/)', rf"\1{prefix}/", text_body, flags=re.IGNORECASE)
        text_body = re.sub(r'((?:fetch|open)\(\s*["\'])/(?!/)', rf"\1{prefix}/", text_body, flags=re.IGNORECASE)
        if "text/html" in content_type.lower():
            # Generated frontends often hide fetch behind helpers, e.g.
            # api(path) { return fetch(path) }. The literal-path rewrite above
            # cannot see that the caller passes '/api/...', so those requests
            # escape the preview proxy and hit ForgeAI's HTML route. Bridge all
            # same-origin root requests through this app's preview prefix.
            fetch_bridge = f'''<script>(function(){{
  const previewBase = {json.dumps(prefix)};
  const originalFetch = window.fetch.bind(window);
  function scoped(input) {{
    if (typeof input === "string" && input.startsWith("/") && !input.startsWith("//") && input !== previewBase && !input.startsWith(previewBase + "/")) return previewBase + input;
    if (input instanceof URL && input.origin === window.location.origin && !input.pathname.startsWith(previewBase + "/")) return previewBase + input.pathname + input.search + input.hash;
    if (input instanceof Request) {{
      const url = new URL(input.url);
      if (url.origin === window.location.origin && !url.pathname.startsWith(previewBase + "/")) return new Request(previewBase + url.pathname + url.search + url.hash, input);
    }}
    return input;
  }}
  window.fetch = function(input, init) {{ return originalFetch(scoped(input), init); }};
}})();</script>'''
            if re.search(r"</head\s*>", text_body, flags=re.IGNORECASE):
                text_body = re.sub(r"</head\s*>", lambda match: fetch_bridge + match.group(0), text_body, count=1, flags=re.IGNORECASE)
            else:
                text_body = fetch_bridge + text_body
        body = text_body.encode(upstream.encoding or "utf-8")

    response_headers = {
        key: value for key, value in upstream.headers.items()
        if key.lower() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}
    }
    location = response_headers.get("Location")
    if location and location.startswith("/") and not location.startswith("//"):
        response_headers["Location"] = f"{prefix}{location}"
    return Response(body, status=upstream.status_code, headers=response_headers)


@app.get("/")
def home():
    """Return the ForgeAI frontend HTML."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/style.css")
def serve_css():
    """Serve the frontend stylesheet."""
    return send_from_directory(FRONTEND_DIR, "style.css")


@app.get("/script.js")
def serve_js():
    """Serve the frontend JavaScript."""
    return send_from_directory(FRONTEND_DIR, "script.js")


@app.post("/generate")
def generate_endpoint():
    """Generate, run, verify, and repair an app from a natural-language prompt."""
    payload = request.get_json(silent=True) or {}
    prompt = payload.get("prompt") or payload.get("user_prompt")
    if not prompt:
        return jsonify({"error": "A prompt is required."}), 400

    force_demo = bool(payload.get("demo_mode"))
    selected_showcase = _select_initial_showcase(force_demo, str(prompt))
    if force_demo and selected_showcase is None:
        return jsonify({
            "error": "No valid offline showcase is available. Check backend/showcase/catalog.json and backend/showcase/runtime/."
        }), 503

    job_id = uuid.uuid4().hex
    JOB_STORE[job_id] = {
        "status": "queued",
        "current_step": "queued",
        "logs": [],
        "result": None,
        "spec": None,
        "plan": None,
        "generated_files": [],
        "execution": None,
        "verification": None,
        "versions": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": time.time(),
        "prompt": str(prompt),
        "force_demo": force_demo,
        "showcase_id": selected_showcase["id"] if selected_showcase else None,
    }
    _persist_job(job_id, JOB_STORE[job_id])

    with QUEUE_LOCK:
        GENERATION_QUEUE.append(job_id)

    thread = threading.Thread(target=run_generation_job, args=(job_id, str(prompt)), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.get("/download/<job_id>")
def download_generated_app(job_id: str):
    """Package a completed generated project with portable run instructions."""
    job = JOB_STORE.get(job_id)
    execution = job.get("execution") if job else None
    project_dir = Path(execution["project_dir"]) if execution and execution.get("project_dir") else None
    if project_dir is None or not project_dir.is_dir():
        return jsonify({"error": "Generated project not found."}), 404
    allowed_root = GENERATED_PROJECTS_DIR.resolve()
    if not project_dir.resolve().is_relative_to(allowed_root):
        return jsonify({"error": "Generated project not found."}), 404

    archive = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
    excluded_names = {"__pycache__", "process.json", "port.txt"}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in project_dir.rglob("*"):
            if not path.is_file() or excluded_names.intersection(path.parts):
                continue
            if path.suffix.lower() in {".log", ".db", ".pyc"}:
                continue
            bundle.write(path, path.relative_to(project_dir).as_posix())
        bundle.writestr("README.md", "# Generated Flask app\n\nInstall dependencies and start the app:\n\n```bash\npip install -r requirements.txt\npython app.py\n```\n\nThe app runs locally. Set `PORT` to choose its port.\n")
        bundle.writestr("Dockerfile", "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nENV PORT=5000\nEXPOSE 5000\nCMD [\"python\", \"app.py\"]\n")
    archive.seek(0)
    return Response(archive.read(), mimetype="application/zip", headers={"Content-Disposition": f"attachment; filename=forgeai-{job_id}.zip"})


@app.get("/status/<job_id>")
def status_endpoint(job_id: str):
    """Return the current status, logs, and final result for a background job."""
    job = JOB_STORE.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404

    replay = bool(job.get("replay") or (job.get("result") or {}).get("replay"))
    response = {
        "job_id": job_id,
        "status": job.get("status"),
        "current_step": job.get("current_step"),
        "logs": job.get("logs", []),
        "result": job.get("result"),
        "spec": job.get("spec"),
        "generated_files": job.get("generated_files", []),
        "verification": job.get("verification"),
        "versions": [
            {"version": index + 1, "created_at": item.get("created_at")}
            for index, item in enumerate(job.get("versions", []))
        ],
        "version_count": len(job.get("versions", [])) + 1,
        "duration_seconds": job.get("duration_seconds"),
        "replay": replay,
        "live_generation_failure": job.get("live_generation_failure") or (job.get("result") or {}).get("live_generation_failure"),
        "generation_outcome": _generation_outcome(job, replay),
        "queue_position": GENERATION_QUEUE.index(job_id) + 1 if job_id in GENERATION_QUEUE else None,
    }
    return jsonify(response)


def _generation_outcome(job: dict[str, Any], replay: bool | None = None) -> str:
    """Describe whether a job is live generation, verification failure, or replay."""
    result = job.get("result") or {}
    if replay is None:
        replay = bool(job.get("replay") or result.get("replay"))
    if replay:
        return "showcase_replay"
    status = job.get("status")
    if status == "success":
        return "live_verified"
    if status in {"queued", "running"}:
        return "in_progress"
    verification = job.get("verification") or result.get("details")
    if isinstance(verification, dict) and verification and verification.get("status") != "ok":
        return "live_verification_failed"
    reason = str(result.get("reason") or job.get("live_generation_failure") or "").lower()
    if any(marker in reason for marker in ("provider", "api key", "authentication", "quota", "rate limit", "http 401", "http 429")):
        return "provider_configuration_failed"
    return "live_generation_failed"


@app.get("/history")
def history_summary_endpoint():
    """Return persisted build summaries for the counter and history gallery."""
    count = sum(
        1 for job in JOB_STORE.values()
        if job.get("status") == "success" and not job.get("parent_job_id")
        and not (job.get("replay") or (job.get("result") or {}).get("replay"))
    )
    history = [
        {
            "job_id": job_id,
            "app_name": (job.get("spec") or {}).get("app_name", "Generated app"),
            "prompt": job.get("prompt", ""),
            "created_at": job.get("created_at"),
            "status": job.get("status"),
            "replay": bool(job.get("replay") or (job.get("result") or {}).get("replay")),
            "label": (job.get("result") or {}).get("label"),
            "generation_outcome": _generation_outcome(job),
            "version_count": len(job.get("versions", [])) + 1,
        }
        for job_id, job in JOB_STORE.items()
        if job.get("status") == "success" and job.get("execution")
    ]
    history.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return jsonify({"apps_forged": count, "jobs": history})


@app.get("/network-address")
def network_address_endpoint():
    """Return a usable LAN origin for QR links when this page was opened via localhost."""
    address = "127.0.0.1"
    probe = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("192.0.2.1", 80))
        candidate = probe.getsockname()[0]
        if candidate and not candidate.startswith("127."):
            address = candidate
    except OSError:
        try:
            candidates = socket.gethostbyname_ex(socket.gethostname())[2]
            address = next((item for item in candidates if not item.startswith("127.")), address)
        except OSError:
            pass
    finally:
        if probe:
            probe.close()
    return jsonify({"origin": f"http://{address}:5000"})


@app.get("/health")
def health_page():
    return send_from_directory(FRONTEND_DIR, "health.html")


@app.get("/health/data")
def health_data():
    checks: dict[str, Any] = {}
    if LLM_PROVIDER == "ollama":
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5)
            reachable = response.status_code < 500
        except requests.RequestException:
            reachable = False
        checks["ollama"] = {"reachable": reachable, "configured": bool(OLLAMA_MODEL)}
        active_provider_label = "Ollama (local)"
        model = OLLAMA_MODEL
    else:
        try:
            response = requests.get("https://api.groq.com", timeout=1.5)
            checks["groq"] = {"reachable": response.status_code < 500, "configured": bool(os.getenv("GROQ_API_KEY", "").strip())}
        except requests.RequestException:
            checks["groq"] = {"reachable": False, "configured": bool(os.getenv("GROQ_API_KEY", "").strip())}
        active_provider_label = "Groq API"
        model = f"Groq: {os.getenv('GROQ_MODEL_SPEC', 'openai/gpt-oss-20b')} / {os.getenv('GROQ_MODEL_CODEGEN', 'openai/gpt-oss-120b')}"
    try:
        free_bytes = shutil.disk_usage(BACKEND_DIR).free
    except OSError:
        free_bytes = 0
    return jsonify({
        "providers": checks,
        "active_provider": LLM_PROVIDER,
        "active_provider_label": active_provider_label,
        "active_model": model,
        "free_disk_bytes": free_bytes,
        "free_disk_gb": round(free_bytes / (1024 ** 3), 2),
        "running_generated_apps": _active_generated_count(),
        "app_limit": MAX_CONCURRENT_APPS,
        "app_lifetime_minutes": MAX_APP_LIFETIME_SECONDS // 60,
    })


@app.get("/debug")
def debug_endpoint():
    """Return the configured provider and recent LLM call summaries."""
    return jsonify({"ok": True, "debug": {"provider": LLM_PROVIDER, "groq_key_present": bool(os.getenv("GROQ_API_KEY", "").strip()),
                                             "calls": get_call_stats(), "metrics": get_call_metrics()}})


if __name__ == "__main__":
    for showcase in prepare_showcases():
        showcase_dir = BACKEND_DIR / "showcase" / showcase["id"]
        try:
            start_generated_app(showcase_dir)
            verification = verify_generated_app(showcase_dir, api_contract=contract_for_showcase(showcase))
            if verification.get("status") != "ok":
                raise RuntimeError(f"Showcase {showcase['name']} failed verification")
        finally:
            stop_generated_app(showcase_dir)
    # Keep history on disk, but don't relaunch every historical preview at boot.
    # A preview is started on demand when its history card is opened.
    atexit.register(_stop_managed_apps)
    atexit.register(_stop_managed_apps)
    threading.Thread(target=_expire_generated_apps, daemon=True, name="forgeai-app-expiry").start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
