from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .utils import slugify


def _run_ffmpeg(args: list[str]) -> None:
    r = subprocess.run(
        ["ffmpeg", "-y", *args],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {err[-4000:]}")


def _mux_clip_with_audio(*, clip: Path, wav: Path, out_mp4: Path, width: int, height: int) -> None:
    """
    Re-encode to a consistent format (H.264 + AAC, yuv420p) and attach segment audio.
    Uses -shortest so it ends with the shorter of audio/video.

    Still images (.png/.jpg/...) are treated as a single frame looped for the full audio length.
    """
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    )
    suffix = clip.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        _run_ffmpeg(
            [
                "-loop",
                "1",
                "-i",
                str(clip.resolve()),
                "-i",
                str(wav.resolve()),
                "-vf",
                vf,
                "-shortest",
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "1",
                str(out_mp4.resolve()),
            ]
        )
        return

    _run_ffmpeg(
        [
            "-i",
            str(clip.resolve()),
            "-i",
            str(wav.resolve()),
            "-vf",
            vf,
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(out_mp4.resolve()),
        ]
    )


def _black_clip(*, wav: Path, out_mp4: Path, width: int, height: int) -> None:
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r=30",
            "-i",
            str(wav.resolve()),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(out_mp4.resolve()),
        ]
    )


def build_master_mp4_from_clips(
    *,
    segment_clips: list[Path | None],
    segment_wavs: list[Path],
    out_mp4: Path,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """
    Given per-segment *video clips* (mp4/webm/etc) and per-segment WAV audio, mux each
    pair and concat to master.mp4.

    Any missing clip becomes a black screen for that segment.
    """
    if len(segment_clips) != len(segment_wavs):
        raise ValueError("segment_clips length must match segment_wavs")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="devaemon_clips_") as td:
        tmp = Path(td)
        muxed: list[Path] = []
        for i, (clip, wav) in enumerate(zip(segment_clips, segment_wavs, strict=True), start=1):
            out = tmp / f"mux_{i:03d}.mp4"
            if clip and clip.is_file():
                _mux_clip_with_audio(clip=clip, wav=wav, out_mp4=out, width=width, height=height)
            else:
                _black_clip(wav=wav, out_mp4=out, width=width, height=height)
            muxed.append(out)

        list_file = tmp / "concat.txt"
        list_file.write_text(
            "".join([f"file '{p.resolve().as_posix()}'\n" for p in muxed]),
            encoding="utf-8",
        )
        # Re-encode concat for compatibility (copy often fails when metadata differs).
        _run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file.resolve()),
                "-c",
                "copy",
                str(out_mp4.resolve()),
            ]
        )

    return out_mp4


def segment_clip_path(clips_dir: Path, idx: int, seg_char: str, seg_emotion: str) -> Path:
    stem = slugify(f"{idx:03d}_{seg_char}_{seg_emotion}")
    return clips_dir / f"{stem}.mp4"

