"""
Wrapper around the Groq SDK (`pip install groq`).

Groq hosts open-weight models (Llama, etc.) with very fast inference and
a generous free tier, so this is the piece that keeps the whole project
laptop-friendly. Only the embeddings (sentence-transformers) and the graph
traversal (networkx) run locally, and both are lightweight.

Get a free API key at https://console.groq.com/keys and put it in
engine/.env as GROQ_API_KEY=...
"""
import json
import re
import time
from groq import Groq
from app.config import settings

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to engine/.env "
                "(copy engine/.env.example first)."
            )
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


def _extract_suggested_wait(e: Exception) -> float | None:
    """
    Groq's 429 error message includes the exact wait time, e.g.
    'Please try again in 459.99999ms' or '...in 12.5s'. Parse it out so we
    wait exactly as long as needed instead of guessing with a fixed backoff.
    """
    match = re.search(r"try again in ([\d.]+)\s*(ms|s)\b", str(e))
    if not match:
        return None
    value, unit = match.groups()
    seconds = float(value) / 1000.0 if unit == "ms" else float(value)
    return seconds


def _chat_completion_with_retry(messages: list[dict], temperature: float, max_retries: int = 4):
    client = get_client()
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            wait_seconds = _extract_suggested_wait(e)
            if wait_seconds is None:
                # No parseable wait time in the error -> exponential backoff,
                # capped at 30s so a single stuck call can't hang forever.
                wait_seconds = min(2 ** attempt, 30)
            wait_seconds += 0.5  # small safety margin so we don't re-hit the limit instantly
            print(
                f"[llm_client] rate limited (attempt {attempt + 1}/{max_retries + 1}), "
                f"waiting {wait_seconds:.2f}s before retry..."
            )
            time.sleep(wait_seconds)
    raise RuntimeError("unreachable")  # loop always returns or raises


def generate_text(prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _chat_completion_with_retry(messages, temperature=temperature)
    return (resp.choices[0].message.content or "").strip()


def generate_json(prompt: str, system: str | None = None, retries: int = 2) -> dict | list:
    """
    Ask the model for JSON only, parse it, and self-repair once or twice
    if the output isn't valid JSON (models occasionally wrap it in prose
    or markdown fences despite instructions).

    Note: this retry loop is for JSON-parse failures specifically. Rate
    limit retries happen one layer down, inside generate_text's call to
    _chat_completion_with_retry, so a 429 here won't burn through these
    JSON-repair attempts.
    """
    last_error = None
    last_raw = ""
    for attempt in range(retries + 1):
        sys_prompt = (system or "") + (
            "\nRespond with ONLY valid JSON. No markdown fences, no preamble, "
            "no trailing commentary."
        )
        if attempt > 0:
            sys_prompt += (
                f"\nYour previous reply could not be parsed as JSON "
                f"(error: {last_error}). Raw reply was:\n{last_raw}\n"
                "Return corrected, valid JSON only."
            )
        raw = generate_text(prompt, system=sys_prompt, temperature=0.1)
        last_raw = raw
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = str(e)
            time.sleep(0.5)
            continue
    raise ValueError(f"Model did not return valid JSON after {retries + 1} attempts: {last_raw}")
