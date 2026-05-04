FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps:
# - build tools sometimes needed for scientific wheels
# - libsndfile for soundfile/librosa stacks
# - praat is required by praat-parselmouth (imports as `parselmouth`)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    build-essential \
    pkg-config \
    libsndfile1 \
    praat \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade "pip==24.0" && \
    python -m pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Default: Applio is expected to be mounted at runtime to /opt/Applio-RVC-Fork
ENV APPLIO_ROOT=/opt/Applio-RVC-Fork
ENV APPLIO_DEVICE=cpu

# NOTE: You still need to install Applio/RVC python deps inside this image OR mount
# a prebuilt Applio environment. The common pattern is:
# - mount your host Applio checkout to /opt/Applio-RVC-Fork
# - mount a conda env (advanced) OR bake deps into this Dockerfile.

ENTRYPOINT ["python", "main.py"]
