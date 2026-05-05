"""
Minimal web UI: paste Nobita / Doraemon lines, run the same pipeline as `main.py`.

  streamlit run ui_app.py

Dialogue format (one line per utterance):

  N: Hello...
  D: Here's the fix...
"""

from __future__ import annotations

import streamlit as st

from doraemon_devs.config import load_config
from doraemon_devs.pipeline import run_pipeline
from doraemon_devs.script_parse import parse_dialogue_text


def main() -> None:
    st.set_page_config(page_title="Dev-aemon", layout="centered")
    st.title("Dev-aemon — custom dialogue")
    st.caption("Each line: `N:` / `D:` (or `Nobita:` / `Doraemon:`). At least 4 lines.")

    cfg_path = st.text_input("Config file", value="config.yaml")
    title = st.text_input("Episode title", value="Custom episode")
    topic = st.text_input("Topic slug (output folder prefix)", value="custom-dialogue")
    dialogue = st.text_area(
        "Script",
        height=280,
        placeholder="N: Your Nobita line here.\nD: Your Doraemon line here.\n...",
    )
    make_mp4 = st.checkbox(
        "Build master.mp4 (needs ffmpeg on PATH; uses Comfy images if present, else black slides)",
        value=False,
    )
    make_mp4_clips = st.checkbox(
        "Build master.mp4 from per-line ComfyUI video clips (GPU recommended; requires a video workflow JSON)",
        value=False,
    )

    if st.button("Generate audio (and optional video)", type="primary"):
        if not dialogue.strip():
            st.warning("Paste dialogue first.")
            return
        try:
            cfg = load_config(cfg_path)
            script = parse_dialogue_text(title=title, topic=topic, text=dialogue)
            with st.spinner("Running pipeline…"):
                run_dir = run_pipeline(cfg, script, make_mp4=make_mp4, make_mp4_clips=make_mp4_clips)
            st.success(f"Output: `{run_dir}`")
            st.json({"master_wav": str(run_dir / "master.wav"), "script": str(run_dir / "script.json")})
            if make_mp4 and (run_dir / "master.mp4").is_file():
                st.video(str(run_dir / "master.mp4"))
            if make_mp4_clips and (run_dir / "master.mp4").is_file():
                st.video(str(run_dir / "master.mp4"))
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()
