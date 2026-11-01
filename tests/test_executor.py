from pathlib import Path

from agent.executor import _normalize_generated_files, write_generated_files


def test_normalize_generated_files_accepts_dict_and_list():
    mapping = {"app.py": "print('hello')"}
    entries = _normalize_generated_files(mapping)

    assert entries == [{"filename": "app.py", "content": "print('hello')"}]

    list_entries = [{"filename": "models.py", "content": "class User: pass"}]
    assert _normalize_generated_files(list_entries) == list_entries


def test_write_generated_files_creates_nested_structure(tmp_path):
    project_dir = tmp_path / "demo_project"
    generated_files = [{"filename": "app/templates/index.html", "content": "<h1>Hello</h1>"}]

    write_generated_files(project_dir, generated_files)

    assert (project_dir / "app" / "templates" / "index.html").exists()
    assert "Hello" in (project_dir / "app" / "templates" / "index.html").read_text(encoding="utf-8")


def test_generated_process_environment_withholds_credentials(monkeypatch):
    from agent.executor import _generated_app_environment

    monkeypatch.setenv("GROQ_API_KEY", "never-forward")
    monkeypatch.setenv("GEMINI_API_KEY", "never-forward")
    monkeypatch.setenv("CUSTOM_ACCESS_TOKEN", "never-forward")
    monkeypatch.setenv("APP_SECRET", "never-forward")
    monkeypatch.setenv("DATABASE_URL", "never-forward")
    monkeypatch.setenv("FORGEAI_SAFE_TEST_VALUE", "available")

    env = _generated_app_environment()

    assert "GROQ_API_KEY" not in env
    assert "GEMINI_API_KEY" not in env
    assert "CUSTOM_ACCESS_TOKEN" not in env
    assert "APP_SECRET" not in env
    assert "DATABASE_URL" not in env
    assert env["FORGEAI_SAFE_TEST_VALUE"] == "available"
