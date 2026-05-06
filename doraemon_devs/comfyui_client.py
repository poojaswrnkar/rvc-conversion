from __future__ import annotations

import json
import time
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


def _apply_checkpoint_override(prompt_graph: dict[str, Any], checkpoint_filename: str | None) -> None:
    if not checkpoint_filename or not prompt_graph:
        return
    for node in prompt_graph.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "CheckpointLoaderSimple":
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "ckpt_name" in inputs:
            inputs["ckpt_name"] = checkpoint_filename


def _submit_workflow(
    *,
    base: str,
    workflow_path: Path,
    prompt_text: str,
    timeout_s: int,
    checkpoint_filename: str | None = None,
) -> str | None:
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

    _apply_checkpoint_override(prompt_graph, checkpoint_filename)

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
    return prompt_id


def _poll_history(*, base: str, prompt_id: str, timeout_s: int) -> dict[str, Any] | None:
    deadline = time.time() + max(1, timeout_s)
    last_err: str | None = None
    while time.time() < deadline:
        try:
            hr = requests.get(f"{base}/history/{prompt_id}", timeout=30)
            if hr.status_code == 200:
                history = hr.json().get(prompt_id) or {}
                # If outputs exist, we're done.
                outputs = history.get("outputs") or {}
                if outputs:
                    return history
            else:
                last_err = hr.text[:200]
        except Exception as e:
            last_err = str(e)
        time.sleep(1.0)
    if last_err:
        raise ComfyUIError(f"ComfyUI history timeout: {last_err}")
    return None


def try_generate_media(
    *,
    cfg: AppConfig,
    prompt_text: str,
    out_dir: Path,
    filename_prefix: str,
    workflow_json_path: str | Path,
    timeout_s: int,
) -> Path | None:
    """
    Generic ComfyUI runner:
    - submits workflow
    - polls /history until output appears (or timeout)
    - downloads first output (image/video/other) via /view
    """
    if not cfg.comfyui.enabled:
        return None

    base = cfg.comfyui.base_url.rstrip("/")
    workflow_path = Path(workflow_json_path)
    if not workflow_path.exists():
        raise FileNotFoundError(f"ComfyUI workflow_json_path not found: {workflow_path}")

    prompt_id = _submit_workflow(
        base=base,
        workflow_path=workflow_path,
        prompt_text=prompt_text,
        timeout_s=30,
        checkpoint_filename=cfg.comfyui.checkpoint,
    )
    if not prompt_id:
        return None

    history = _poll_history(base=base, prompt_id=prompt_id, timeout_s=timeout_s) or {}
    outputs = history.get("outputs") or {}

    # Prefer images, but accept any output type with a filename.
    for _node_id, out in outputs.items():
        for key in ("images", "gifs", "videos", "files"):
            items = (out or {}).get(key) or []
            if not items:
                continue
            item = items[0]
            filename = item.get("filename")
            subfolder = item.get("subfolder") or ""
            if not filename:
                continue
            view_params = {"filename": filename, "subfolder": subfolder, "type": item.get("type", "output")}
            blob = requests.get(f"{base}/view", params=view_params, timeout=60).content

            ensure_dir(out_dir)
            suffix = Path(filename).suffix or ".bin"
            out_path = out_dir / f"{slugify(filename_prefix)}{suffix}"
            out_path.write_bytes(blob)
            return out_path

    return None


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

    # Maintain old behavior but with polling + generic output handling.
    return try_generate_media(
        cfg=cfg,
        prompt_text=prompt_text,
        out_dir=out_dir,
        filename_prefix=filename_prefix,
        workflow_json_path=cfg.comfyui.workflow_json_path,
        timeout_s=timeout_s,
    )
