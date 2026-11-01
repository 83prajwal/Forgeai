import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent


def load_environment() -> None:
    """Load variables from the project root .env file."""
    load_dotenv(BASE_DIR / ".env")


load_environment()


def get_gemini_api_key() -> str:
    """Return the Gemini API key from the environment or raise a clear error."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return api_key


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("GEMINI_BACKEND_MODEL") or "gemini-3.8-flash"
GEMINI_MODEL_SPEC = os.getenv("GEMINI_MODEL_SPEC") or GEMINI_MODEL
GEMINI_MODEL_CODEGEN = os.getenv("GEMINI_MODEL_CODEGEN") or GEMINI_MODEL
GEMINI_MODEL_FIX = os.getenv("GEMINI_MODEL_FIX") or GEMINI_MODEL
# GROQ (OpenAI-compatible) provider settings
# Use a currently available high-capability model by default. Override via
# GROQ_MODEL in .env when the account exposes a different model.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "gemini").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b").strip()
# Conservative local inference settings for machines with limited memory.
OLLAMA_NUM_CTX = min(4096, max(2048, int(os.getenv("OLLAMA_NUM_CTX", "4096"))))
OLLAMA_MAX_OUTPUT_TOKENS = min(1024, max(256, int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "1024"))))
OLLAMA_REQUEST_TIMEOUT_SECONDS = min(600.0, max(30.0, float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "300"))))
# Small model for planning; larger model for code and repairs. GROQ_MODEL remains
# supported as a global override for existing installations.
GROQ_MODEL_SPEC = os.getenv("GROQ_MODEL_SPEC") or "openai/gpt-oss-20b"
GROQ_MODEL_CODEGEN = os.getenv("GROQ_MODEL_CODEGEN") or "openai/gpt-oss-120b"
GROQ_MODEL_FIX = os.getenv("GROQ_MODEL_FIX") or "openai/gpt-oss-120b"
GROQ_MODEL = os.getenv("GROQ_MODEL") or GROQ_MODEL_CODEGEN  # legacy single-model setting
# Local rolling-window safeguards used for preflight token reservations. These
# are estimates, not Groq's advertised limits (provider 429s are handled from
# the HTTP response). Keep the older names as supported configuration aliases.
GROQ_TPM_70B = os.getenv("GROQ_LOCAL_TPM_70B") or os.getenv("GROQ_TPM_70B", "10000")
GROQ_TPM_8B = os.getenv("GROQ_LOCAL_TPM_8B") or os.getenv("GROQ_TPM_8B", "5000")
GROQ_TOKENS_PER_MINUTE = os.getenv("GROQ_LOCAL_TOKENS_PER_MINUTE") or os.getenv("GROQ_TOKENS_PER_MINUTE")
GROQ_LOCAL_REQUESTS_PER_MINUTE = os.getenv("GROQ_LOCAL_REQUESTS_PER_MINUTE") or os.getenv("GROQ_REQUESTS_PER_MINUTE", "15")
GROQ_MAX_RETRIES = os.getenv("GROQ_MAX_RETRIES", "2")
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}

def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    return api_key

