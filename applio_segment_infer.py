from __future__ import annotations

"""
Applio batch inference helper for single WAV segments.

Applio's `infer_batch_rvc.py` expects:
- `input_path` as a DIRECTORY containing `.wav` files
- `opt_path` as a DIRECTORY to write outputs with the same filenames

This helper wraps that behavior so our pipeline can pass a single input wav
and a single output wav path.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input wav file")
    ap.add_argument("--output", required=True, help="Output wav file")
    ap.add_argument("--model", required=True, help="Model .pth path")
    ap.add_argument("--index", required=True, help="Model .index path")
    ap.add_argument(
        "--applio-root",
        default=os.environ.get("APPLIO_ROOT", str(Path.home() / "Applio-RVC-Fork")),
        help="Path to cloned Applio repo root (contains infer_batch_rvc.py)",
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

    applio_root = Path(args.applio_root).expanduser().resolve()
    infer_script = applio_root / "infer_batch_rvc.py"
    if not infer_script.exists():
        raise FileNotFoundError(f"infer_batch_rvc.py not found under: {applio_root}")

    in_wav = Path(args.input).expanduser().resolve()
    out_wav = Path(args.output).expanduser().resolve()
    model_pth = Path(args.model).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve()

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="applio_seg_") as td:
        td_path = Path(td)
        in_dir = td_path / "in"
        out_dir = td_path / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        staged = in_dir / "segment.wav"
        shutil.copy2(in_wav, staged)

        # Use the same interpreter as the caller by default (e.g. conda `applio-py310`),
        # because that's where Applio's dependencies should be installed.
        #
        # Override if needed:
        #   export APPLIO_PYTHON=/home/you/miniconda3/envs/applio-py310/bin/python
        python_exe = os.environ.get("APPLIO_PYTHON", sys.executable)
        cmd = [
            str(python_exe),
            str(infer_script),
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

        subprocess.run(cmd, cwd=str(applio_root), check=True)

        produced = out_dir / "segment.wav"
        if not produced.exists():
            raise FileNotFoundError(f"Expected output not found: {produced}")
        shutil.move(str(produced), str(out_wav))


if __name__ == "__main__":
    main()
