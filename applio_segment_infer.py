from __future__ import annotations

"""
Applio inference helper for a single WAV segment.

Supports two layouts:

1) **Legacy** — `infer_batch_rvc.py` at repo root (older Applio forks): batch CLI
   with temp input/output dirs.

2) **Current Applio** (e.g. IAHispano/Applio-RVC-Fork): no `infer_batch_rvc.py`; we call
   `core.run_infer_script` via `applio_core_infer_once.py` using the same Python as
   the Applio/conda environment (`rvc.python` / `APPLIO_PYTHON`).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _find_legacy_infer_batch(applio_root: Path) -> Path | None:
    direct = applio_root / "infer_batch_rvc.py"
    if direct.is_file():
        return direct
    found = next(applio_root.rglob("infer_batch_rvc.py"), None)
    return found if found is not None and found.is_file() else None


def _core_invoker_path() -> Path:
    return (Path(__file__).resolve().parent / "applio_core_infer_once.py").resolve()


def _rvc_repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_asset(path: str | Path, repo_root: Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (repo_root / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input wav file")
    ap.add_argument("--output", required=True, help="Output wav file")
    ap.add_argument("--model", required=True, help="Model .pth path")
    ap.add_argument("--index", required=True, help="Model .index path")
    ap.add_argument(
        "--applio-root",
        default=os.environ.get("APPLIO_ROOT", str(Path.home() / "Applio-RVC-Fork")),
        help="Path to cloned Applio repo root (legacy: infer_batch_rvc.py; new: core.py)",
    )
    ap.add_argument("--device", default=os.environ.get("APPLIO_DEVICE", "cuda:0"))
    ap.add_argument("--f0-method", default=os.environ.get("APPLIO_F0_METHOD", "rmvpe"))
    ap.add_argument("--f0-up-key", default=os.environ.get("APPLIO_F0_UP_KEY", "0"))
    ap.add_argument("--index-rate", default=os.environ.get("APPLIO_INDEX_RATE", "0.75"))
    ap.add_argument("--is-half", default=os.environ.get("APPLIO_IS_HALF", "true"))
    ap.add_argument("--filter-radius", default=os.environ.get("APPLIO_FILTER_RADIUS", "3"))
    ap.add_argument("--resample-sr", default=os.environ.get("APPLIO_RESAMPLE_SR", "0"))
    ap.add_argument("--rms-mix-rate", default=os.environ.get("APPLIO_RMS_MIX_RATE", "1"))
    ap.add_argument("--protect", default=os.environ.get("APPLIO_PROTECT", "0.33"))
    args = ap.parse_args()

    repo_root = _rvc_repo_root()
    applio_root = Path(args.applio_root).expanduser().resolve()
    in_wav = _resolve_asset(args.input, repo_root)
    out_wav = _resolve_asset(args.output, repo_root)
    model_pth = _resolve_asset(args.model, repo_root)
    index_path = _resolve_asset(args.index, repo_root)

    if not model_pth.is_file():
        raise FileNotFoundError(
            f"RVC model .pth not found: {model_pth}\n"
            "Applio will leave vc=None and then crash with "
            "'NoneType' object has no attribute 'pipeline'."
        )
    if not index_path.is_file():
        raise FileNotFoundError(f"RVC index not found: {index_path}")

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    python_exe = os.environ.get("APPLIO_PYTHON", sys.executable)
    legacy_infer = _find_legacy_infer_batch(applio_root)

    if legacy_infer is not None:
        _run_legacy_infer_batch(
            applio_root=applio_root,
            legacy_infer=legacy_infer,
            python_exe=python_exe,
            in_wav=in_wav,
            out_wav=out_wav,
            model_pth=model_pth,
            index_path=index_path,
            args=args,
        )
        return

    _run_applio_core_infer(
        applio_root=applio_root,
        python_exe=python_exe,
        in_wav=in_wav,
        out_wav=out_wav,
        model_pth=model_pth,
        index_path=index_path,
        args=args,
    )


def _run_legacy_infer_batch(
    *,
    applio_root: Path,
    legacy_infer: Path,
    python_exe: str,
    in_wav: Path,
    out_wav: Path,
    model_pth: Path,
    index_path: Path,
    args: argparse.Namespace,
) -> None:
    with tempfile.TemporaryDirectory(prefix="applio_seg_") as td:
        td_path = Path(td)
        in_dir = td_path / "in"
        out_dir = td_path / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        staged = in_dir / "segment.wav"
        shutil.copy2(in_wav, staged)

        cmd = [
            str(python_exe),
            str(legacy_infer),
            str(args.f0_up_key),
            str(in_dir),
            str(index_path),
            str(args.f0_method),
            str(out_dir),
            str(model_pth),
            str(args.index_rate),
            str(args.device),
            str(args.is_half),
            str(args.filter_radius),
            str(args.resample_sr),
            str(args.rms_mix_rate),
            str(args.protect),
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(applio_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()
            if tail:
                tail = tail[-6000:]
            raise RuntimeError(
                "Applio infer_batch_rvc.py failed (is PyTorch installed in this Python?).\n"
                f"Python: {python_exe}\n"
                f"Applio: {applio_root}\n"
                f"--- output ---\n{tail}"
            )

        produced = out_dir / "segment.wav"
        if not produced.exists():
            raise FileNotFoundError(f"Expected output not found: {produced}")
        shutil.move(str(produced), str(out_wav))


def _run_applio_core_infer(
    *,
    applio_root: Path,
    python_exe: str,
    in_wav: Path,
    out_wav: Path,
    model_pth: Path,
    index_path: Path,
    args: argparse.Namespace,
) -> None:
    invoker = _core_invoker_path()
    if not invoker.is_file():
        raise FileNotFoundError(
            f"Missing {invoker.name} next to applio_segment_infer.py "
            "(needed for current Applio layout without infer_batch_rvc.py)."
        )
    if not (applio_root / "core.py").is_file():
        raise FileNotFoundError(
            f"Applio repo root looks invalid (no core.py): {applio_root}\n"
            "Clone https://github.com/IAHispano/Applio-RVC-Fork or set APPLIO_ROOT."
        )

    # `run_infer_script` requires these positional args (no defaults in Applio core.py).
    payload: dict[str, object] = {
        "applio_root": str(applio_root),
        "pitch": int(float(args.f0_up_key)),
        "index_rate": float(args.index_rate),
        "volume_envelope": float(args.rms_mix_rate),
        "protect": float(args.protect),
        "f0_method": args.f0_method,
        "input_path": str(in_wav),
        "output_path": str(out_wav),
        "pth_path": str(model_pth),
        "index_path": str(index_path),
        "split_audio": False,
        "f0_autotune": False,
        "f0_autotune_strength": 1.0,
        "proposed_pitch": False,
        "proposed_pitch_threshold": 155.0,
        "clean_audio": False,
        "clean_strength": 0.5,
        "export_format": "WAV",
        "embedder_model": os.environ.get("APPLIO_EMBEDDER_MODEL", "contentvec"),
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as f:
        json.dump(payload, f)
        kwargs_path = Path(f.name)

    try:
        proc = subprocess.run(
            [str(python_exe), str(invoker), str(kwargs_path)],
            cwd=str(applio_root),
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()
            if tail:
                tail = tail[-6000:]
            extra = ""
            if "has no attribute 'pipeline'" in tail and "NoneType" in tail:
                extra = (
                    "\nHint: Applio's VoiceConverter.vc was never built — almost always the .pth "
                    "path was wrong or torch.load could not read the checkpoint (see Applio logs above).\n"
                )
            raise RuntimeError(
                "Applio core.run_infer_script failed (PyTorch / models / VRAM?).\n"
                f"Python: {python_exe}\n"
                f"Applio: {applio_root}\n"
                f"{extra}"
                f"--- output ---\n{tail}"
            )
    finally:
        kwargs_path.unlink(missing_ok=True)

    if not out_wav.is_file():
        raise FileNotFoundError(f"Expected RVC output not found: {out_wav}")


if __name__ == "__main__":
    main()
