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

OUTLINE_CONTINUE_SYSTEM = """Continue a Dev-aemon outline. Reply with ONLY new lines in pipe format.
Each line: N|scene|emotion|spoken_line OR D|scene|emotion|spoken_line
scene is 1-4. spoken_line must NOT contain the | character.
No markdown, no numbering, no explanation, no thinking."""


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


def character_visual_anchor(char: str) -> str:
    """Concrete SD-friendly looks (original archetypes, not IP names)."""
    if char.strip().lower() in {"nobita", "nobi"}:
        return (
            "teen boy with round glasses and short black hair, yellow polo shirt, "
            "expressive face, waist-up shot, simple apartment or desk background"
        )
    return (
        "small round blue robot mascot, red collar, white belly, golden bell on collar, "
        "no ears, friendly cartoon eyes, waist-up, futuristic gadget room background"
    )


def default_visual_prompt(*, char: str, emotion: str, topic: str) -> str:
    """Single-image Comfy prompt: one clear character, readable emotion, on-topic."""
    anchor = character_visual_anchor(char)
    return (
        f"masterpiece, best quality, single main character, centered composition, "
        f"{anchor}, emotion: {emotion}, context: software developer stress about {topic}, "
        f"1990s cel-shaded anime, clean lineart, solid colors, coherent anatomy, not crowded"
    )


def _mood_prompt(*, char: str, emotion: str, topic: str) -> str:
    return default_visual_prompt(char=char, emotion=emotion, topic=topic)


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


def _outline_join(rows: list[str]) -> str:
    return "\n".join(rows)


def _dedupe_append_rows(existing: list[str], new_rows: list[str]) -> int:
    """Append rows not already present (case-insensitive full-line match). Returns how many were added."""
    seen = {r.strip().lower() for r in existing}
    added = 0
    for r in new_rows:
        key = r.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        existing.append(r)
        added += 1
    return added


def _pad_outline_rows(topic: str, rows: list[str]) -> None:
    """
    Last resort: append template beats so the pipeline always has 6 lines.
    Wording references the topic slug so pads still feel on-theme.
    """
    t = topic.strip() or "tech"
    templates: list[tuple[str, str, str, str]] = [
        (
            "D",
            "2",
            "excited",
            f"Arre {t} ke liye mere paas ek naya pocket-gadget hai — ek baar try toh kar bhai.",
        ),
        (
            "N",
            "2",
            "skeptical",
            "Gadget se pehle bhi pipeline toot chuki hai… ab bharosa kaise karu?",
        ),
        (
            "D",
            "3",
            "focused",
            f"Chal practical: pehle conflict markers dhundh, fix kar, phir `git add` + commit — {t} tabhi settle hoga.",
        ),
        (
            "N",
            "4",
            "panicked",
            f"Bhai ab CI bhi roast kar raha hai — {t} ne poora deploy hi ulta kar diya!",
        ),
    ]
    i = 0
    while len(rows) < 6 and i < len(templates):
        who, scene, emotion, spoken = templates[i]
        rows.append(f"{who}|{scene}|{emotion}|{spoken}")
        i += 1
    while len(rows) < 6:
        rows.append(f"N|4|tired|Bas yaar, {t} se tang aa gaya — ab coffee break mandatory hai.")


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


def _fetch_outline_raw(*, client: OpenAI, model: str, topic: str, strict: bool) -> str:
    if strict:
        outline_user = (
            f"Tech topic: {topic}\n"
            "Output EXACTLY 6 lines.\n"
            "Each line must start with N| or D| and contain exactly 3 pipe characters.\n"
            "No blank lines. No other text.\n"
        )
        temp = 0.0
        max_tok = 1200
    else:
        outline_user = (
            f"Tech topic: {topic}\n"
            "Output EXACTLY 6 lines in the pipe format.\n"
            "If you output anything except those 6 lines, you failed.\n"
        )
        temp = 0.1
        max_tok = 1400

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
    return outline


def _fetch_outline_continue_raw(
    *, client: OpenAI, model: str, topic: str, existing: list[str]
) -> str:
    need = 6 - len(existing)
    if need <= 0:
        return ""
    user = (
        f"Tech topic: {topic}\n\n"
        "These lines are already written — do NOT repeat them; output ONLY new lines:\n"
        f"{_outline_join(existing)}\n\n"
        f"Write EXACTLY {need} new line(s) in the same pipe format.\n"
        "Continue story beats: gadget/tease (if missing), concrete terminal tip, then chaos ending.\n"
        "No blank lines, no other text.\n"
    )
    outline_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": OUTLINE_CONTINUE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.15,
        max_tokens=1200,
    )
    outline_msg = outline_resp.choices[0].message
    text = (getattr(outline_msg, "content", None) or getattr(outline_msg, "reasoning", None) or "").strip()
    return text


def generate_script(topic: str, cfg: AppConfig) -> Script:
    client = OpenAI(
        base_url=cfg.qwen.base_url,
        api_key=cfg.qwen.api_key,
        timeout=cfg.qwen.timeout_s,
    )

    rows: list[str] = []
    for strict in (True, True, False, False):
        raw = _fetch_outline_raw(client=client, model=cfg.qwen.model, topic=topic, strict=strict)
        _dedupe_append_rows(rows, _extract_outline_lines(raw))
        if len(rows) >= 6:
            break

    stagnant = 0
    for _ in range(10):
        if len(rows) >= 6:
            break
        before = len(rows)
        raw = _fetch_outline_continue_raw(
            client=client, model=cfg.qwen.model, topic=topic, existing=rows
        )
        _dedupe_append_rows(rows, _extract_outline_lines(raw))
        if len(rows) == before:
            stagnant += 1
            if stagnant >= 3:
                break
        else:
            stagnant = 0

    if len(rows) < 6:
        _pad_outline_rows(topic, rows)

    rows = rows[:6]
    outline = _outline_join(rows)
    return _script_from_outline(topic=topic, outline=outline)
