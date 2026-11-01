"""TODO: Write generated files, install dependencies, and run the app."""

import json
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


GENERATED_PROJECTS_DIR = Path(__file__).resolve().parents[1] / "generated_projects"
PORT_FILE_NAME = "port.txt"
PROCESS_FILE_NAME = "process.json"
PROCESS_HANDLES: dict[int, subprocess.Popen[str]] = {}
_SENSITIVE_ENV_MARKERS = (
    "API_KEY", "_TOKEN", "TOKEN_", "SECRET", "PASSWORD", "CREDENTIAL",
    "AUTH", "ACCESS_KEY", "_DSN", "DATABASE_URL", "CONNECTION_STRING",
)


def _generated_app_environment() -> dict[str, str]:
    """Copy runtime environment while withholding ForgeAI and other credentials."""
    return {
        name: value for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }


def _normalize_generated_files(generated_files: list[dict[str, str]] | dict[str, str] | None) -> list[dict[str, str]]:
    """Accept either a list of {filename, content} entries or a mapping of filename->content."""
    if not generated_files:
        return []

    if isinstance(generated_files, dict):
        return [{"filename": name, "content": content} for name, content in generated_files.items()]

    normalized: list[dict[str, str]] = []
    for item in generated_files:
        if isinstance(item, dict):
            if "filename" in item and "content" in item:
                normalized.append({"filename": str(item["filename"]), "content": str(item["content"])})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            normalized.append({"filename": str(item[0]), "content": str(item[1])})
    return normalized


def _resolve_free_port() -> int:
    """Return a free TCP port for the generated app."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _find_entrypoint(project_dir: Path) -> Path:
    """Locate the main Flask app file in a generated project directory."""
    candidates = [project_dir / "app.py", project_dir / "main.py", project_dir / "server.py"]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    py_files = sorted(project_dir.rglob("*.py"))
    if py_files:
        return py_files[0]

    raise FileNotFoundError(f"No Python entrypoint found in generated project: {project_dir}")


def write_generated_files(project_dir: Path, generated_files: list[dict[str, str]] | dict[str, str]) -> None:
    """Write generated file payloads to disk."""
    project_dir.mkdir(parents=True, exist_ok=True)

    for file_info in _normalize_generated_files(generated_files):
        file_path = project_dir / file_info["filename"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file_info["content"], encoding="utf-8")

    # Generated output can omit or misname this file; keep Flask apps runnable
    # with a minimal dependency manifest in that case.
    requirements_path = project_dir / "requirements.txt"
    if not requirements_path.exists():
        requirements_path.write_text("Flask\n", encoding="utf-8")


def install_requirements(project_dir: Path) -> subprocess.CompletedProcess[str]:
    """Install dependencies from the generated app requirements.txt using the current Python."""
    requirements_path = project_dir / "requirements.txt"
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements.txt in generated project: {project_dir}")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        check=False,
        env=_generated_app_environment(),
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "Unknown pip install error").strip()
        raise RuntimeError(f"Dependency installation failed: {details[:1000]}")
    return result


def start_generated_app(project_dir: Path, port: int | None = None) -> dict[str, Any]:
    """Launch the generated Flask app as a background subprocess and capture stdout/stderr to log files."""
    if port is None:
        port = _resolve_free_port()

    entrypoint = _find_entrypoint(project_dir)
    stdout_log = project_dir / "stdout.log"
    stderr_log = project_dir / "stderr.log"
    port_file = project_dir / PORT_FILE_NAME
    process_file = project_dir / PROCESS_FILE_NAME

    env = _generated_app_environment()
    # A parent started with ``flask run`` sets this flag. Flask then makes an
    # app.run() call return normally instead of starting a server, which looks
    # like a successful child exit (status 0) but leaves no listening process.
    # These markers belong to the parent CLI/reloader and must not leak into
    # the independently launched app process.
    env.pop("FLASK_RUN_FROM_CLI", None)
    env.pop("WERKZEUG_RUN_MAIN", None)
    env["PORT"] = str(port)
    env["FLASK_RUN_PORT"] = str(port)
    env["FLASK_APP"] = str(entrypoint)

    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(entrypoint.name)], cwd=str(project_dir),
            stdout=stdout, stderr=stderr, env=env,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    PROCESS_HANDLES[process.pid] = process

    port_file.write_text(str(port), encoding="utf-8")
    process_file.write_text(
        json.dumps({"pid": process.pid, "port": port, "entrypoint": str(entrypoint), "started_at": datetime.now().timestamp()}, indent=2),
        encoding="utf-8",
    )

    return {
        "process": process,
        "entrypoint": str(entrypoint),
        "project_dir": str(project_dir),
        "port": port,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def stop_generated_app(project_dir: Path) -> None:
    """Stop a generated app and all child processes it started."""
    process_file = Path(project_dir) / PROCESS_FILE_NAME
    try:
        pid = int(json.loads(process_file.read_text(encoding="utf-8")).get("pid", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return
    if pid <= 0:
        return
    process = PROCESS_HANDLES.pop(pid, None)
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False, timeout=5)
        else:
            os.killpg(pid, signal.SIGTERM)
    except (OSError, subprocess.TimeoutExpired):
        pass


def restart_generated_app(project_dir: Path, install: bool = True) -> dict[str, Any]:
    """Stop the current generated process and launch the edited project again."""
    project_dir = Path(project_dir)
    stop_generated_app(project_dir)

    if install:
        install_requirements(project_dir)
    return start_generated_app(project_dir)


def execute_generated_app(
    app_name: str,
    generated_files: list[dict[str, str]] | dict[str, str],
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """Create the generated project, install dependencies, and launch the Flask app on a free port."""
    if project_root is None:
        project_root = GENERATED_PROJECTS_DIR
    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = project_root / f"{app_name.lower().replace(' ', '_')}_{timestamp}"
    write_generated_files(project_dir, generated_files)

    try:
        install_result = install_requirements(project_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to prepare generated app: {exc}") from exc

    try:
        app_run = start_generated_app(project_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to start generated app: {exc}") from exc

    return {
        "project_dir": str(project_dir),
        "port": app_run["port"],
        "entrypoint": app_run["entrypoint"],
        "stdout_log": app_run["stdout_log"],
        "stderr_log": app_run["stderr_log"],
        "pip_install": {
            "returncode": install_result.returncode,
            "stdout": install_result.stdout,
            "stderr": install_result.stderr,
        },
        "process": app_run["process"],
    }
