from __future__ import annotations

import json
from pathlib import Path

from .audio_engine import concat_wavs, render_segment_audio
from .comfyui_client import try_generate_image
from .config import AppConfig
from .schema import Script
from .utils import ensure_dir, slugify, timestamp_id
from .video_export import try_build_master_mp4


def run_pipeline(
    cfg: AppConfig,
    script: Script,
    *,
    make_mp4: bool = False,
) -> Path:
    """
    Render script.json, per-segment audio (optional RVC), optional Comfy stills, master.wav,
    and optionally master.mp4 (slideshow + audio; requires ffmpeg).
    """
    run_dir = ensure_dir(Path("outputs") / f"{slugify(script.topic)}_{timestamp_id()}")

    (run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")

    images_dir = ensure_dir(run_dir / "images")
    audio_dir = ensure_dir(run_dir / "audio_segments")

    segment_wavs: list[Path] = []
    for i, seg in enumerate(script.segments, start=1):
        prompt = seg.mood_prompt or f"{seg.emotion} {seg.char}, 90s anime aesthetic, legally-distinct"
        try:
            try_generate_image(
                cfg=cfg,
                prompt_text=prompt,
                out_dir=images_dir,
                filename_prefix=f"{i:03d}_{seg.char}_{seg.emotion}",
            )
        except Exception:
            pass

        wav = render_segment_audio(cfg=cfg, character=seg.char, text=seg.text, out_dir=audio_dir, idx=i)
        segment_wavs.append(wav)

    master_wav = run_dir / "master.wav"
    concat_wavs(segment_wavs, master_wav)

    manifest: dict = {
        "topic": script.topic,
        "title": script.title,
        "run_dir": str(run_dir),
        "files": {
            "script": str(run_dir / "script.json"),
            "master_audio": str(master_wav),
            "segments_dir": str(audio_dir),
            "images_dir": str(images_dir),
        },
    }

    if make_mp4:
        out_mp4 = run_dir / "master.mp4"
        try:
            try_build_master_mp4(
                script=script,
                segment_wavs=segment_wavs,
                images_dir=images_dir,
                out_mp4=out_mp4,
            )
            manifest["files"]["master_video"] = str(out_mp4)
        except Exception as e:
            manifest["video_error"] = str(e)

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir
