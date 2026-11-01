from agent.llm_client import generate_with_backoff

if __name__ == "__main__":
    prompt = "Say hello in one sentence"
    try:
        resp = generate_with_backoff(prompt)
        provider = getattr(resp, "provider", "unknown")
        retries = getattr(resp, "retries", None)
        text = getattr(resp, "text", str(resp))
        print(f"Provider: {provider}, retries: {retries}\nResponse:\n{text}")
    except Exception as e:
        print(f"LLM client call failed: {e}")
