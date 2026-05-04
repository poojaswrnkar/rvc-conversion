from __future__ import annotations

from .schema import Script, Segment


def parse_dialogue_text(*, title: str, topic: str, text: str) -> Script:
    """
    One line per utterance, prefix with who speaks:

      N: ...  or  Nobita: ...
      D: ...  or  Doraemon: ...

    Blank lines ignored. At least 4 lines (schema minimum).
    """
    segments: list[Segment] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("nobita:"):
            char = "Nobita"
            rest = line.split(":", 1)[1].strip()
        elif lower.startswith("n:"):
            char = "Nobita"
            rest = line.split(":", 1)[1].strip()
        elif lower.startswith("doraemon:"):
            char = "Doraemon"
            rest = line.split(":", 1)[1].strip()
        elif lower.startswith("d:"):
            char = "Doraemon"
            rest = line.split(":", 1)[1].strip()
        else:
            raise ValueError(
                "Each non-empty line must start with N:, D:, Nobita:, or Doraemon:. "
                f"Problem line: {line[:80]!r}"
            )
        if not rest:
            raise ValueError(f"Empty line after speaker tag: {line[:80]!r}")
        segments.append(Segment(char=char, text=rest))

    if len(segments) < 4:
        raise ValueError(f"Need at least 4 dialogue lines; got {len(segments)}.")

    return Script(title=title.strip(), topic=topic.strip(), segments=segments)
