"""Keep provider routing independent of the developer's local .env."""
from __future__ import annotations

import pytest

from agent import llm_client


@pytest.fixture(autouse=True)
def default_provider_chain(monkeypatch):
    # Tests exercise the Groq -> Gemini pair unless they set their own chain.
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm_client, "LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    # Legacy pipeline tests target the codegen path; blueprint tests opt in.
    import app as backend_app
    monkeypatch.setattr(backend_app, "GENERATION_MODE", "codegen")
