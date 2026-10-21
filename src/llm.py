"""Thin OpenRouter client: images in, JSON out, every call traced with its cost."""
import base64
import json
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Two models: VISION_MODEL reads the drawing (views, points, edges, parts, matching across views), where the
# result depends most on the model; MODEL, the cheap one, does everything else.
VISION_MODEL = os.getenv("D2C_VISION_MODEL") or "anthropic/claude-sonnet-5.5"
MODEL = os.getenv("D2C_MODEL") or "anthropic/claude-haiku-5.5"
MAX_ATTEMPTS = 3
_lock = threading.Lock()


def client() -> OpenAI:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Add OPENROUTER_API_KEY to .env or your environment")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                  timeout=300, max_retries=5)  # the SDK backs off on 429 / 5xx


def image_part(png: bytes) -> dict:
    data = base64.b64encode(png).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def parse_json(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    text = match.group(1) if match else text
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.JSONDecoder().raw_decode(text[start:])[0]  # ignores anything after the object


def ask_json(content: list, step: str, trace: Path, messages: list | None = None,
             model: str | None = None) -> tuple[dict, list]:
    """Send one user turn (after optional history) to `model` (default MODEL); return (parsed JSON, history)."""
    model = model or MODEL
    history = (messages or []) + [{"role": "user", "content": content}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # A reasoning model can spend the whole output budget thinking and reply with nothing;
        # after a failed attempt, ask for low reasoning effort.
        extra = {"reasoning": {"effort": "low"}} if attempt > 1 else {}
        start = time.time()
        reply = client().chat.completions.create(
            model=model, messages=history, max_tokens=16000,
            response_format={"type": "json_object"}, extra_body=extra)
        text = reply.choices[0].message.content or ""
        usage = reply.usage.model_dump() if reply.usage else {}
        record = {"step": step, "attempt": attempt, "id": reply.id, "model": model,
                  "seconds": round(time.time() - start, 1), "cost": usage.get("cost"),
                  "finish_reason": reply.choices[0].finish_reason, "usage": usage, "reply": text}
        try:
            data, error = parse_json(text), None
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            data, error = None, str(exc)
            record["parse_error"] = error
        with _lock, trace.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        cost = usage.get("cost")
        print(f"[{step}] {record['seconds']}s, cost "
              f"{'$%.5f' % cost if cost is not None else 'unavailable'}"
              + (f", unreadable reply ({error[:80]}), retrying" if error else ""), flush=True)
        if data is not None:
            history.append({"role": "assistant", "content": text})
            return data, history
    raise ValueError(f"{step}: no readable JSON after {MAX_ATTEMPTS} attempts")
