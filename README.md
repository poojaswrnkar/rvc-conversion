## Dev-aemon (Doraemon-style tech content) — local $0 pipeline

This is a local Python CLI that:
- calls your **OpenAI-compatible Qwen** endpoint to generate a structured script JSON
- generates per-line audio with **edge-tts** (free/no-key)
- optionally runs each line through **Applio/RVC** (local) using `nobita.pth` / `doraemon.pth`
- optionally triggers **ComfyUI** to generate character images from each segment’s mood
- concatenates all segment audio into a single `master.wav`

### Docker (optional, for reproducible OS deps)

Docker helps most with **system libraries** (example: Praat/`parselmouth`, `libsndfile`, consistent ffmpeg) and pinning **Python 3.10**.

It does **not** automatically solve Applio’s entire `requirements.txt` story unless you **bake** those installs into the image or mount a prebuilt env.

Build + run (CPU):

```bash
cd /home/pooja/experiment
docker compose build
docker compose run --rm doraemon-devs
```

Notes:
- `docker-compose.yml` mounts your host Applio checkout to `/opt/Applio-RVC-Fork`.
- Create `config.yaml` locally (`cp config.example.yaml config.yaml`) and mount it if you want container runs to use your real endpoints/models paths.
- You still need Applio’s Python deps available to the interpreter running `infer_batch_rvc.py`.
  - Easiest: install Applio deps into the image (extend `Dockerfile`), or run `pip install -r /opt/Applio-RVC-Fork/requirements.txt` inside a one-off container after bind-mounting Applio.

### Prereqs

- **Python**: 3.10+ (you have 3.12 which is fine)
- **ffmpeg** installed on your machine (`ffmpeg -version` should work)

Install Python deps:

```bash
python3 -m pip install -r requirements.txt
```

### Applio/RVC (recommended): use a Conda Python 3.10 env for Applio

Applio’s pinned dependencies are not reliable on **Python 3.12/3.13**. The practical setup is:

1) Install Miniconda (if you don’t already have it)

2) Create an env:

```bash
eval "$(/home/pooja/miniconda3/bin/conda shell.zsh hook)"
conda create -n applio-py310 python=3.10 -y
conda activate applio-py310
```

3) Install Applio deps inside `~/Applio-RVC-Fork` (you may need to relax the `faiss_cpu==1.7.3` pin for your platform)

4) Download `hubert_base.pt` into the Applio repo root (required by `infer_batch_rvc.py`):

```bash
cd ~/Applio-RVC-Fork
wget -O hubert_base.pt "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
```

5) Put your `.index` files next to your `.pth` files (or update paths in `config.yaml`):

- `./models/nobita.pth` + `./models/nobita.index`
- `./models/doraemon.pth` + `./models/doraemon.index`

6) Run the pipeline using the Applio conda python:

```bash
cd /home/pooja/experiment
conda activate applio-py310
python -m pip install -r requirements.txt
python main.py --topic "Docker Merge Conflicts"
```

This repo calls `./applio_segment_infer.py`, which wraps Applio’s `infer_batch_rvc.py` folder-based batching into a single-file in/out interface.

### Configure

Copy the example config and edit locally:

```bash
cd /home/pooja/experiment
cp config.example.yaml config.yaml
```

`config.yaml` is intentionally **gitignored** so you don’t commit secrets.

You can also override Qwen settings via env vars (recommended for CI):

```bash
export QWEN_BASE_URL="http://YOUR_HOST:8000/v1"
export QWEN_API_KEY="YOUR_KEY"
export QWEN_MODEL="YOUR_MODEL_ID"
```

### Run

```bash
python3 main.py --topic "Docker Merge Conflicts"
```

It will create an `outputs/...` folder containing:
- `script.json`
- `manifest.json`
- `images/` (if ComfyUI enabled and reachable)
- `audio_segments/` (per-line wavs)
- `master.wav`

