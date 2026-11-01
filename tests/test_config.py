import os
from pathlib import Path

from config import get_gemini_api_key, load_environment


def test_load_environment_reads_dotenv_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=test-key\n", encoding="utf-8")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    original_base_dir = Path(__file__).resolve().parents[1]

    import config as config_module
    config_module.BASE_DIR = tmp_path

    load_environment()

    assert os.getenv("GEMINI_API_KEY") == "test-key"
    assert get_gemini_api_key() == "test-key"

    config_module.BASE_DIR = original_base_dir
