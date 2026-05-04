from __future__ import annotations

import argparse
import json
from pathlib import Path

from doraemon_devs.audio_engine import concat_wavs, render_segment_audio
from doraemon_devs.comfyui_client import try_generate_image
from doraemon_devs.config import load_config
from doraemon_devs.script_generator import generate_script
from doraemon_devs.utils import ensure_dir, slugify, timestamp_id


def run(topic: str, config_path: str) -> Path:
    cfg = load_config(config_path)
    run_dir = ensure_dir(Path("outputs") / f"{slugify(topic)}_{timestamp_id()}")

    script = generate_script(topic, cfg)
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
            # Visuals are optional; keep pipeline moving.
            pass

        wav = render_segment_audio(cfg=cfg, character=seg.char, text=seg.text, out_dir=audio_dir, idx=i)
        segment_wavs.append(wav)

    master_wav = run_dir / "master.wav"
    concat_wavs(segment_wavs, master_wav)

    manifest = {
        "topic": topic,
        "run_dir": str(run_dir),
        "files": {
            "script": str(run_dir / "script.json"),
            "master_audio": str(master_wav),
            "segments_dir": str(audio_dir),
            "images_dir": str(images_dir),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Dev-aemon style tech content locally.")
    ap.add_argument("--topic", required=True, help="Tech topic, e.g. 'Docker Merge Conflicts'")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = ap.parse_args()

    out = run(args.topic, args.config)
    print(str(out))


if __name__ == "__main__":
    main()
