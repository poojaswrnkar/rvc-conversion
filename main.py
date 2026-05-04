from __future__ import annotations

import argparse
from pathlib import Path

from doraemon_devs.config import load_config
from doraemon_devs.pipeline import run_pipeline
from doraemon_devs.schema import Script
from doraemon_devs.script_generator import generate_script


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Dev-aemon style tech content locally.")
    ap.add_argument("--topic", default=None, help="Tech topic for LLM script, e.g. 'Docker Merge Conflicts'")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument(
        "--script-json",
        default=None,
        help="Skip LLM: load Script JSON from this file (title, topic, segments).",
    )
    ap.add_argument(
        "--mp4",
        action="store_true",
        help="After audio, build master.mp4 (slideshow per segment + concat; needs ffmpeg).",
    )
    args = ap.parse_args()

    if not args.topic and not args.script_json:
        ap.error("Provide --topic (LLM) or --script-json (hand-authored).")

    cfg = load_config(args.config)

    if args.script_json:
        p = Path(args.script_json)
        script = Script.model_validate_json(p.read_text(encoding="utf-8"))
    else:
        script = generate_script(args.topic, cfg)

    out = run_pipeline(cfg, script, make_mp4=args.mp4)
    print(str(out))


if __name__ == "__main__":
    main()
