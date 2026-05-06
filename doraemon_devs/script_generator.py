from __future__ import annotations

import re

from openai import OpenAI

from .config import AppConfig
from .schema import Character, Script, Segment


OUTLINE_SYSTEM = """Write a Dev-aemon style script (Hinglish + 2026 dev slang).

Your ENTIRE reply must be EXACTLY 6 lines (no more, no less).
Each line MUST be ONLY this pipe format:

N|scene|emotion|spoken_line
or
D|scene|emotion|spoken_line

Rules:
- N = Nobita, D = Doraemon
- scene is 1-4 (integer)
- emotion is a short English label
- spoken_line is short, natural dialogue (may include Hindi+English mix)
- spoken_line MUST NOT contain the `|` character
- Do NOT copy template phrases like "Short line here"
- Do NOT explain, do NOT number steps, do NOT use markdown, do NOT use backticks

Story beats:
- Lines 1-2: Nobita stressed about the topic; Doraemon calms / teases
- Lines 3-4: Doraemon introduces a "gadget" fix; Nobita reacts
- Line 5: concrete terminal/command tip inside spoken_line (git/docker/etc.)
- Line 6: funny chaos ending

Keep it a generic parody vibe (no real anime/cartoon IP names).
"""


_OUTLINE_LINE_RE = re.compile(
    r"^\s*([ND])\s*\|\s*([1-4])\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$"
)
# Same pattern, but can appear anywhere in a noisy line (model adds bullets/backticks)
_OUTLINE_LINE_FIND_RE = re.compile(r"([ND])\s*\|\s*([1-4])\s*\|\s*([^|]+?)\s*\|\s*(.+)$")


def _clean_spoken_field(s: str) -> str:
    s = s.strip().strip("`").strip('"').strip("'")
    return s.strip()


def _is_junk_outline_line(spoken: str) -> bool:
    t = spoken.lower().strip()
    if len(t) < 8:
        return True
    if t.startswith("...") or re.match(r"^\s*\.\.\.\s*\(", spoken):
        return True
    if re.match(r"^\s*\(\s*(nobita|doraemon)\b", t):
        return True
    if "|" in spoken:
        return True
    junk_markers = (
        "short line here",
        "wait,",
        "the format says",
        "i'll follow",
        "thinking process",
        "analyze user",
        "example shows",
        "template",
        "```",
    )
    return any(m in t for m in junk_markers)


def _human_title(topic: str) -> str:
    s = topic.replace("-", " ").replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else "Episode"


def _mood_prompt(*, char: str, emotion: str, topic: str) -> str:
    return f"{char}, {emotion}, tech chaos about {topic}, 90s anime aesthetic, legally-distinct"


def _script_from_outline(*, topic: str, outline: str) -> Script:
    rows = _extract_outline_lines(outline)
    if len(rows) < 6:
        raise ValueError(f"Expected 6 outline lines, got {len(rows)}.")

    segments: list[Segment] = []
    for row in rows[:8]:
        parts = row.split("|")
        if len(parts) != 4:
            raise ValueError(f"Bad outline row (expected 4 fields): {row!r}")
        who, scene_s, emotion, text = parts
        who = who.strip().upper()
        if who not in {"N", "D"}:
            raise ValueError(f"Bad speaker in row: {row!r}")
        char: Character = "Nobita" if who == "N" else "Doraemon"
        scene = int(scene_s)
        emotion = emotion.strip()
        text = _clean_spoken_field(text)
        if not text or "|" in text:
            raise ValueError(f"Bad spoken text in row: {row!r}")
        segments.append(
            Segment(
                char=char,
                text=text,
                emotion=emotion,
                scene=scene,
                mood_prompt=_mood_prompt(char=char, emotion=emotion, topic=topic),
            )
        )

    return Script(title=_human_title(topic), topic=topic, segments=segments)


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
            spoken = _clean_spoken_field(m.group(4))
            emotion = m.group(3).strip()
            if _is_junk_outline_line(spoken):
                continue
            lines.append(f"{m.group(1)}|{m.group(2)}|{emotion}|{spoken}")
            continue
        m2 = _OUTLINE_LINE_FIND_RE.search(s)
        if m2:
            spoken = _clean_spoken_field(m2.group(4))
            emotion = m2.group(3).strip()
            if _is_junk_outline_line(spoken):
                continue
            lines.append(f"{m2.group(1)}|{m2.group(2)}|{emotion}|{spoken}")
    return lines


def _fetch_outline(*, client: OpenAI, model: str, topic: str, strict: bool) -> str:
    if strict:
        outline_user = (
            f"Tech topic: {topic}\n"
            "Output EXACTLY 6 lines.\n"
            "Each line must start with N| or D| and contain exactly 3 pipe characters.\n"
            "No blank lines. No other text.\n"
        )
        temp = 0.0
        max_tok = 500
    else:
        outline_user = (
            f"Tech topic: {topic}\n"
            "Output EXACTLY 6 lines in the pipe format.\n"
            "If you output anything except those 6 lines, you failed.\n"
        )
        temp = 0.1
        max_tok = 650

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
    for strict in (True, True, False, False):
        outline = _fetch_outline(client=client, model=cfg.qwen.model, topic=topic, strict=strict)
        if _outline_line_count(outline) >= 6:
            break
    if _outline_line_count(outline) < 6:
        snippet = outline[:2000]
        raise ValueError(
            "Outline had too few valid lines. Expected format N|scene|emotion|text per line.\n"
            f"Got {_outline_line_count(outline)} valid lines. Snippet:\n{snippet}"
        )

    script = _script_from_outline(topic=topic, outline=outline)
    return script
