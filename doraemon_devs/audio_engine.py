from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import edge_tts
import ffmpeg

from .config import AppConfig, Character
from .utils import ensure_dir


async def tts_to_wav(
    *,
    text: str,
    voice: str,
    rate: str,
    volume: str,
    out_wav: Path,
) -> None:
    """
    edge-tts outputs MP3 reliably; we convert to WAV for stitching.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = out_wav.with_suffix(".mp3")
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
    await communicate.save(str(tmp_mp3))

    (
        ffmpeg.input(str(tmp_mp3))
        .output(
            str(out_wav),
            acodec="pcm_s16le",
            ac=1,
            ar="48000",
            loglevel="error",
        )
        .overwrite_output()
        .run()
    )
    try:
        tmp_mp3.unlink(missing_ok=True)
    except Exception:
        pass


def run_rvc(
    *,
    cfg: AppConfig,
    character: Character,
    in_wav: Path,
    out_wav: Path,
) -> Path:
    """
    Wrapper around a local Applio/RVC CLI.

    Expected CLI interface:
      <python> <cli_path> --input <in.wav> --output <out.wav> --model <model.pth> [extra args...]

    Make your CLI wrapper accept those args, or edit this function to match your setup.
    """
    if not cfg.rvc.enabled:
        return in_wav

    model_pth = cfg.rvc.models.nobita if character == "Nobita" else cfg.rvc.models.doraemon
    index_path = cfg.rvc.models.nobita_index if character == "Nobita" else cfg.rvc.models.doraemon_index
    cli = Path(cfg.rvc.cli_path)
    if not cli.exists():
        raise FileNotFoundError(f"RVC cli_path not found: {cli}")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    template = os.environ.get("APPLIO_CMD_TEMPLATE", "").strip()
    if template:
        cmd_str = template.format(
            input=str(in_wav),
            output=str(out_wav),
            model=str(Path(model_pth)),
            index=str(Path(index_path)),
        )
        if cfg.rvc.extra_args:
            cmd_str = f"{cmd_str} {cfg.rvc.extra_args}".strip()
        subprocess.run(cmd_str, shell=True, check=True)
        return out_wav

    py = cfg.rvc.python or sys.executable
    cmd = [
        str(py),
        str(cli),
        "--input",
        str(in_wav),
        "--output",
        str(out_wav),
        "--model",
        str(Path(model_pth)),
        "--index",
        str(Path(index_path)),
    ]
    if cfg.rvc.extra_args:
        cmd += cfg.rvc.extra_args.split()

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip()
        if tail:
            tail = tail[-4000:]
        hint = (
            "RVC failed. Common causes: (1) this Python has no PyTorch — set rvc.python in config.yaml "
            "to your Applio/conda interpreter, or export APPLIO_PYTHON=/path/to/that/python; "
            "(2) wrong or missing .index path next to your .pth. "
            f"Command: {' '.join(cmd[:6])} ...\n--- stderr/stdout ---\n{tail}"
        )
        raise RuntimeError(hint) from None
    return out_wav


def concat_wavs(wavs: list[Path], out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # concat filter requires same format; we standardize to 48k mono PCM.
    inputs = [ffmpeg.input(str(p)) for p in wavs]
    joined = ffmpeg.concat(*inputs, v=0, a=1)
    (
        ffmpeg.output(
            joined,
            str(out_wav),
            acodec="pcm_s16le",
            ac=1,
            ar="48000",
            loglevel="error",
        )
        .overwrite_output()
        .run()
    )


def render_segment_audio(
    *,
    cfg: AppConfig,
    character: Character,
    text: str,
    out_dir: Path,
    idx: int,
) -> Path:
    ensure_dir(out_dir)
    base_wav = out_dir / f"{idx:03d}_{character.lower()}_base.wav"
    final_wav = out_dir / f"{idx:03d}_{character.lower()}_final.wav"

    voice = cfg.tts.nobita_voice if character == "Nobita" else cfg.tts.doraemon_voice
    asyncio.run(
        tts_to_wav(
            text=text,
            voice=voice,
            rate=cfg.tts.rate,
            volume=cfg.tts.volume,
            out_wav=base_wav,
        )
    )

    if cfg.rvc.enabled:
        return run_rvc(cfg=cfg, character=character, in_wav=base_wav, out_wav=final_wav)
    return base_wav
