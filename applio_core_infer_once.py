"""
Run Applio `core.run_infer_script` from the Applio install (conda) interpreter.

`applio_segment_infer.py` writes a JSON file of kwargs and invokes:

  <applio-python> applio_core_infer_once.py <kwargs.json>

The JSON must include `"applio_root"`. Any other keys override `run_infer_script`
defaults (missing keys use the function signature defaults from Applio's core.py).
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: applio_core_infer_once.py <kwargs.json>", file=sys.stderr)
        sys.exit(2)
    kwargs_path = Path(sys.argv[1])
    raw: dict[str, Any] = json.loads(kwargs_path.read_text(encoding="utf-8"))
    applio_root = Path(raw.pop("applio_root")).expanduser().resolve()
    os.chdir(applio_root)
    sys.path.insert(0, str(applio_root))

    from core import run_infer_script  # noqa: E402 — after chdir/sys.path

    sig = inspect.signature(run_infer_script)
    call_kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name, param in sig.parameters.items():
        if name in raw:
            call_kwargs[name] = raw[name]
        elif param.default is not inspect.Parameter.empty:
            call_kwargs[name] = param.default
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"Missing infer kwargs (no default in Applio): {missing}")

    run_infer_script(**call_kwargs)


if __name__ == "__main__":
    main()
