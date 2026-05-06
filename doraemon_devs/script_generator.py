from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from .config import AppConfig
from .schema import Script


OUTLINE_SYSTEM = """Write a Dev-aemon style script (Hinglish + 2026 dev slang).

Your ENTIRE reply must be 6 to 8 lines.
Each line MUST look exactly like:

N|1|panicking|Short line here
D|2|calm|Short line here

Rules per line:
- Start with N (Nobita) or D (Doraemon)
- scene is 1-4
- emotion is a short English label
- last field is the spoken line (short, TTS-friendly)

Content:
- Alternate N and D
- Scene 3 must include one real terminal/command snippet (e.g. git/docker) inside the line text
- Do not mention real anime/cartoon IP; keep it a generic parody vibe
"""


JSON_FORMAT_SYSTEM = """You convert an outline into STRICT JSON only.

Output rules (STRICT):
- Output ONLY one JSON object.
- First character MUST be `{`, last character MUST be `}`.
- No markdown, no code fences, no commentary, no thinking.

JSON schema:
{
  "title": "...",
  "topic": "...",
  "segments": [
    {"char":"Nobita","text":"...","emotion":"...","scene":1,"mood_prompt":"..."},
    {"char":"Doraemon","text":"...","emotion":"...","scene":2,"mood_prompt":"..."}
  ]
}

mood_prompt should be a short image prompt: character + mood + "90s anime aesthetic, legally-distinct".
"""

_OUTLINE_LINE_RE = re.compile(
    r"^\s*([ND])\s*\|\s*([1-4])\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$"
)
# Same pattern, but can appear anywhere in a noisy line (model adds bullets/backticks)
_OUTLINE_LINE_FIND_RE = re.compile(
    r"([ND])\s*\|\s*([1-4])\s*\|\s*([^|]+?)\s*\|\s*(.+?)(?:\s*`*)$"
)


def _extract_json(text: str) -> dict[str, Any]:
    """
    vLLM/Qwen setups sometimes return long preambles (or put text in `reasoning`).
    We decode the first complete JSON object in the string.
    """
    raw = text.strip()
    if not raw:
        raise ValueError("LLM returned empty text; cannot parse JSON.")

    # Strip common ```json fences if present
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1 :]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].rstrip()

    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(raw[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    snippet = raw[:4000]
    raise ValueError(f"Could not find a JSON object in LLM output. Snippet:\n{snippet}")


def _outline_line_count(outline: str) -> int:
    return len(_extract_outline_lines(outline))


def _extract_outline_lines(raw: str) -> list[str]:
    """Pull valid `N|...` lines out of model noise (thinking dumps, bullets, etc.)."""
    lines: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _OUTLINE_LINE_RE.match(s)
        if m:
            lines.append(f"{m.group(1)}|{m.group(2)}|{m.group(3).strip()}|{m.group(4).strip()}")
            continue
        m2 = _OUTLINE_LINE_FIND_RE.search(s)
        if m2:
            lines.append(
                f"{m2.group(1)}|{m2.group(2)}|{m2.group(3).strip()}|{m2.group(4).strip()}"
            )
    return lines


def _fetch_outline(*, client: OpenAI, model: str, topic: str, strict: bool) -> str:
    if strict:
        outline_user = (
            f"Tech topic: {topic}\n"
            "Reply with ONLY 6 lines in the pipe format. No other words.\n"
        )
        temp = 0.0
        max_tok = 350
    else:
        outline_user = (
            f"Tech topic: {topic}\n"
            "Write 6-8 outline lines in the pipe format.\n"
            "If you start explaining, you failed—delete it and output only lines.\n"
        )
        temp = 0.2
        max_tok = 500

    outline_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": OUTLINE_SYSTEM},
            {"role": "user", "content": outline_user},
        ],
        temperature=temp,
        max_tokens=max_tok,
    )
    outline_msg = outline_resp.choices[0].message
    outline = (getattr(outline_msg, "content", None) or getattr(outline_msg, "reasoning", None) or "").strip()
    if not outline:
        raise ValueError("LLM returned empty outline.")
    cleaned = "\n".join(_extract_outline_lines(outline))
    return cleaned


def generate_script(topic: str, cfg: AppConfig) -> Script:
    client = OpenAI(
        base_url=cfg.qwen.base_url,
        api_key=cfg.qwen.api_key,
        timeout=cfg.qwen.timeout_s,
    )

    outline = ""
    for strict in (True, False, False):
        outline = _fetch_outline(client=client, model=cfg.qwen.model, topic=topic, strict=strict)
        if _outline_line_count(outline) >= 6:
            break
    if _outline_line_count(outline) < 4:
        snippet = outline[:2000]
        raise ValueError(
            "Outline had too few valid lines. Expected format N|scene|emotion|text per line.\n"
            f"Got {_outline_line_count(outline)} valid lines. Snippet:\n{snippet}"
        )

    last_err: Exception | None = None
    for attempt in range(3):
        json_user = (
            f"Topic slug for JSON.topic field: {topic}\n"
            f"Pick a catchy JSON.title.\n"
            f"Convert EVERY line of the outline into EXACTLY one segment.\n"
            f"The segments array MUST contain 6-8 items (minimum 4).\n"
            f"Do not merge lines. Do not drop lines.\n\n"
            f"Outline:\n{outline}\n"
        )
        if attempt > 0 and last_err is not None:
            json_user += (
                f"\nYour previous JSON was invalid: {last_err}\n"
                "Fix it. Ensure segments length is 6-8.\n"
            )

        json_resp = client.chat.completions.create(
            model=cfg.qwen.model,
            messages=[
                {"role": "system", "content": JSON_FORMAT_SYSTEM},
                {"role": "user", "content": json_user},
            ],
            temperature=0.0,
            max_tokens=1800,
        )

        # Some OpenAI-compatible servers (notably certain Qwen/vLLM configs)
        # return text in `reasoning` with `content=None`.
        msg = json_resp.choices[0].message
        content = (getattr(msg, "content", None) or getattr(msg, "reasoning", None) or "").strip()
        try:
            data = _extract_json(content)
            data.setdefault("topic", topic)
            return Script.model_validate(data)
        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to generate a valid Script after retries. Last error: {last_err}")
