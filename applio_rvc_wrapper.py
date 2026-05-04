from __future__ import annotations

"""
Minimal wrapper CLI that matches this project's expected RVC interface:

  python3 applio_rvc_wrapper.py --input in.wav --output out.wav --model model.pth [--extra "..."]

This wrapper intentionally avoids hardcoding any Applio-specific CLI flags.

Configure it by setting an environment variable:

  export APPLIO_CMD_TEMPLATE='python3 /abs/path/to/applio_infer.py --input "{input}" --output "{output}" --model "{model}" --index "{index}"'

It will substitute:
  - {input}  -> input wav path
  - {output} -> output wav path
  - {model}  -> model .pth path
  - {index}  -> model .index path

Then the pipeline can stay stable while you swap Applio versions / flags.
"""

import argparse
import os
import subprocess
from pathlib import Path


def run_applio(*, in_wav: Path, out_wav: Path, model_pth: Path, index_path: Path, extra: str) -> None:
    """
    Runs a user-supplied Applio command template.
    """
    template = os.environ.get("APPLIO_CMD_TEMPLATE", "").strip()
    if not template:
        raise RuntimeError(
            "APPLIO_CMD_TEMPLATE is not set.\n\n"
            "Set it to the exact command you use to run Applio/RVC inference, e.g.\n"
            "  export APPLIO_CMD_TEMPLATE='python3 /abs/path/to/applio_infer.py --input \"{input}\" --output \"{output}\" --model \"{model}\" --index \"{index}\"'\n"
        )

    cmd_str = template.format(
        input=str(in_wav),
        output=str(out_wav),
        model=str(model_pth),
        index=str(index_path),
    )
    if extra:
        cmd_str = f"{cmd_str} {extra}".strip()

    # Use shell=True so users can include quotes/flags easily in the template.
    subprocess.run(cmd_str, shell=True, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--extra", default="")
    args = ap.parse_args()

    in_wav = Path(args.input)
    out_wav = Path(args.output)
    model_pth = Path(args.model)
    index_path = Path(args.index)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Placeholder to make it obvious where to integrate.
    run_applio(in_wav=in_wav, out_wav=out_wav, model_pth=model_pth, index_path=index_path, extra=args.extra)


if __name__ == "__main__":
    main()

