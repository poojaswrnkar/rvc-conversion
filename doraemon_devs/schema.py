from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Character = Literal["Nobita", "Doraemon"]
Emotion = str


class Segment(BaseModel):
    char: Character
    text: str = Field(min_length=1)
    emotion: Emotion = "explainer"
    scene: int = Field(ge=1, le=4, default=1)
    mood_prompt: str | None = None
    # Optional: richer visual direction (esp. for video workflows)
    video_prompt: str | None = None
    # Optional: planning hint; pipeline still uses real audio duration when available.
    duration_s: float | None = Field(default=None, gt=0)

    @field_validator("char", mode="before")
    @classmethod
    def normalize_char(cls, v: object) -> object:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"nobita", "nobi"}:
                return "Nobita"
            if s in {"doraemon", "dora", "robot"}:
                return "Doraemon"
        return v


class Script(BaseModel):
    title: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    segments: list[Segment] = Field(min_length=4)
