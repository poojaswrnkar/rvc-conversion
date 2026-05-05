from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import AppConfig
from .schema import Script


SYSTEM_PROMPT = """You are an expert content creator for "Dev-aemon", a tech-focused parody inspired-by Doraemon.
Write a ~60-second video script in Hinglish (Hindi + English) using 2026 dev slang.

CHARACTER GUIDELINES:
1) Nobita (Junior Dev): whiny, stressed, "aura minus 1000", "brain rot code", "crash out" vibes.
2) Doraemon (Senior AI): rational, slightly condescending but helpful. Introduces a real-world tool as a "Gadget".

SCRIPT STRUCTURE:
Scene 1: Nobita panics about the specific tech topic
Scene 2: Doraemon reveals gadget (include "Tadaaa!" cue)
Scene 3: Realistic technical explanation (actionable) of how the gadget solves the topic
Scene 4: Funny chaos ending where Nobita overuses it and breaks something

OUTPUT REQUIREMENTS:
- Output ONLY valid JSON (no markdown, no code fences).
- Use this schema exactly:
{
  "title": "...",
  "topic": "...",
  "segments": [
    {"char":"Nobita","text":"...","emotion":"crying","scene":1,"mood_prompt":"Angry Nobita, 90s anime aesthetic, legally-distinct"},
    {"char":"Doraemon","text":"...","emotion":"heroic","scene":2,"mood_prompt":"Happy robot mentor, 90s anime aesthetic, legally-distinct"}
  ]
}

Rules:
- 6-8 segments total; keep lines short for TTS.
- Alternate characters frequently.
- Include at least 1 concrete technical tip/command in Scene 3.
- Avoid real IP/copyright names; say "Doraemon-proxy" only in mood_prompt if needed.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    # If the model wrapped extra text, attempt to locate the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    return json.loads(text)


def generate_script(topic: str, cfg: AppConfig) -> Script:
    client = OpenAI(
        base_url=cfg.qwen.base_url,
        api_key=cfg.qwen.api_key,
        timeout=cfg.qwen.timeout_s,
    )

    user_prompt = f"Tech topic: {topic}\nWrite the script JSON now."
    resp = client.chat.completions.create(
        model=cfg.qwen.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        # Prevent truncation; many vLLM/Qwen setups are verbose.
        max_tokens=2500,
    )

    # Some OpenAI-compatible servers (notably certain Qwen/vLLM configs)
    # return text in `reasoning` with `content=None`.
    msg = resp.choices[0].message
    content = (getattr(msg, "content", None) or getattr(msg, "reasoning", None) or "").strip()
    data = _extract_json(content)
    data.setdefault("topic", topic)
    script = Script.model_validate(data)
    return script
