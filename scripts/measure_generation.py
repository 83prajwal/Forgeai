"""Run one live end-to-end generation and report estimated LLM traffic."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from agent.llm_client import get_call_stats, reset_call_stats  # noqa: E402
from app import generate_and_verify_app  # noqa: E402


def main() -> int:
    reset_call_stats()
    prompt = "build a task tracker app with users and tasks"
    try:
        result = generate_and_verify_app(prompt)
        print(f"Generation status: {result.get('verification', {}).get('status', 'completed')}")
    except Exception as exc:
        print(f"Generation failed: {exc}")
    stats = get_call_stats()
    for index, call in enumerate(stats, 1):
        print(
            f"call={index} provider={call['provider']} model={call['model']} "
            f"tokens_estimated={call.get('estimated_tokens', 0)} "
            f"wait_seconds={call.get('wait_seconds', 0)} "
            f"elapsed_seconds={call.get('elapsed_seconds', 0)} "
            f"cached={call.get('cached', False)} status={call.get('status', 'ok')}"
        )
    print(f"Total LLM calls made: {sum(not c.get('cached', False) for c in stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
