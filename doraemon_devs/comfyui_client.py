from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from .config import AppConfig
from .utils import ensure_dir, slugify


class ComfyUIError(RuntimeError):
    pass


def _load_workflow(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def try_generate_image(
    *,
    cfg: AppConfig,
    prompt_text: str,
    out_dir: Path,
    filename_prefix: str,
    timeout_s: int = 30,
) -> Path | None:
    """
    Minimal ComfyUI integration:
    - Submits an existing workflow JSON (user-provided)
    - Replaces a best-effort text prompt node input if found
    - Downloads the first resulting image if history is available

    If ComfyUI isn't enabled/reachable, returns None.
    """
    if not cfg.comfyui.enabled:
        return None

    base = cfg.comfyui.base_url.rstrip("/")
    workflow_path = Path(cfg.comfyui.workflow_json_path)
    if not workflow_path.exists():
        raise FileNotFoundError(f"ComfyUI workflow_json_path not found: {workflow_path}")

    wf = _load_workflow(workflow_path)

    # ComfyUI HTTP API expects a payload like:
    #   { "prompt": { "<node_id>": { "class_type": "...", "inputs": { ... } }, ... }, ... }
    #
    # Users often save either:
    # - the full request object (already contains "prompt"), or
    # - just the prompt graph dict (node_id -> node_spec)
    if isinstance(wf, dict) and "prompt" in wf and isinstance(wf["prompt"], dict):
        prompt_graph = wf["prompt"]
        payload: dict[str, Any] = dict(wf)
    else:
        prompt_graph = wf if isinstance(wf, dict) else {}
        payload = {"prompt": prompt_graph}

    # Best-effort: set "text" field in the first node that looks like a CLIPTextEncode input.
    for _node_id, node in (prompt_graph or {}).items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "text" in inputs and isinstance(inputs["text"], str):
            inputs["text"] = prompt_text
            break

    r = requests.post(f"{base}/prompt", json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise ComfyUIError(f"ComfyUI /prompt failed: {r.status_code} {r.text[:200]}")
    prompt_id = r.json().get("prompt_id")
    if not prompt_id:
        return None

    # Try to fetch history and download first image output.
    hr = requests.get(f"{base}/history/{prompt_id}", timeout=timeout_s)
    if hr.status_code != 200:
        return None

    history = hr.json().get(prompt_id) or {}
    outputs = history.get("outputs") or {}
    for _node_id, out in outputs.items():
        images = (out or {}).get("images") or []
        if not images:
            continue
        img = images[0]
        filename = img.get("filename")
        subfolder = img.get("subfolder") or ""
        if not filename:
            continue
        view_params = {"filename": filename, "subfolder": subfolder, "type": img.get("type", "output")}
        img_bytes = requests.get(f"{base}/view", params=view_params, timeout=timeout_s).content

        ensure_dir(out_dir)
        out_path = out_dir / f"{slugify(filename_prefix)}.png"
        out_path.write_bytes(img_bytes)
        return out_path

    return None
