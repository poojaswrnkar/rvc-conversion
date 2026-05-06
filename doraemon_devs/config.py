from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class QwenConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "local-no-key"
    model: str = "qwen3.6-35b"
    timeout_s: int = 180


class TTSConfig(BaseModel):
    nobita_voice: str = "en-IN-PrabhatNeural"
    doraemon_voice: str = "en-IN-NeerjaNeural"
    rate: str = "+0%"
    volume: str = "+0%"


class RVCModels(BaseModel):
    nobita: str = "./models/nobita.pth"
    doraemon: str = "./models/doraemon.pth"
    nobita_index: str = "./models/nobita.index"
    doraemon_index: str = "./models/doraemon.index"


class RVCConfig(BaseModel):
    enabled: bool = False
    """Python that has torch + Applio deps (e.g. conda env). Falls back to current interpreter."""
    python: str | None = None
    cli_path: str = "/absolute/path/to/applio_rvc_cli.py"
    extra_args: str = ""
    models: RVCModels = Field(default_factory=RVCModels)


class ComfyUIConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8188"
    workflow_json_path: str = "./comfy/workflows/character_gen.json"
    # Optional separate workflow for video clips (mp4/gif/webp) per segment.
    # If not set, `workflow_json_path` is used.
    video_workflow_json_path: str | None = None
    # Overrides `ckpt_name` on every CheckpointLoaderSimple node (bundled workflows ship a placeholder).
    checkpoint: str | None = None
    # API-format workflow node id (e.g. "12") whose string input receives the segment prompt.
    # Required for most LTX / custom graphs (they are not "first CLIPTextEncode wins").
    prompt_node_id: str | None = None
    # Input field on that node: "text", "prompt", "string", etc.
    prompt_input_key: str = "text"
    # How long to poll ComfyUI history for results.
    poll_timeout_s: int = 180


class AppConfig(BaseModel):
    qwen: QwenConfig = Field(default_factory=QwenConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    rvc: RVCConfig = Field(default_factory=RVCConfig)
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)


def load_config(path: str | Path) -> AppConfig:
    p = Path(path).expanduser()
    data: dict[str, Any] = {}
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    cfg = AppConfig.model_validate(data)

    # Resolve repo-relative paths against the config file directory (not CWD).
    base_dir = p.resolve().parent

    def _resolve_cfg_path(s: str) -> str:
        pp = Path(s)
        if pp.is_absolute():
            return str(pp)
        return str((base_dir / pp).resolve())

    # Environment overrides (so you can keep secrets out of git).
    # Example:
    #   export QWEN_BASE_URL="http://host:8000/v1"
    #   export QWEN_API_KEY="..."
    #   export QWEN_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
    if os.environ.get("QWEN_BASE_URL"):
        cfg.qwen.base_url = os.environ["QWEN_BASE_URL"]
    if os.environ.get("QWEN_API_KEY"):
        cfg.qwen.api_key = os.environ["QWEN_API_KEY"]
    if os.environ.get("QWEN_MODEL"):
        cfg.qwen.model = os.environ["QWEN_MODEL"]
    if os.environ.get("QWEN_TIMEOUT_S"):
        cfg.qwen.timeout_s = int(os.environ["QWEN_TIMEOUT_S"])

    if os.environ.get("APPLIO_PYTHON"):
        cfg.rvc.python = os.environ["APPLIO_PYTHON"]

    if os.environ.get("COMFY_CHECKPOINT"):
        cfg.comfyui.checkpoint = os.environ["COMFY_CHECKPOINT"]

    cfg.rvc.cli_path = _resolve_cfg_path(cfg.rvc.cli_path)
    cfg.rvc.models.nobita = _resolve_cfg_path(cfg.rvc.models.nobita)
    cfg.rvc.models.doraemon = _resolve_cfg_path(cfg.rvc.models.doraemon)
    cfg.rvc.models.nobita_index = _resolve_cfg_path(cfg.rvc.models.nobita_index)
    cfg.rvc.models.doraemon_index = _resolve_cfg_path(cfg.rvc.models.doraemon_index)
    cfg.comfyui.workflow_json_path = _resolve_cfg_path(cfg.comfyui.workflow_json_path)
    if cfg.comfyui.video_workflow_json_path:
        cfg.comfyui.video_workflow_json_path = _resolve_cfg_path(cfg.comfyui.video_workflow_json_path)

    if cfg.rvc.python:
        rp = Path(cfg.rvc.python).expanduser()
        cfg.rvc.python = str(rp if rp.is_absolute() else (base_dir / rp).resolve())
    return cfg


Character = Literal["Nobita", "Doraemon"]
