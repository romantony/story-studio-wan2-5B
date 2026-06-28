FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/wan22

# ── System packages ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── WAN2.2 library ───────────────────────────────────────────────────────────
# Clone into /opt/wan22 — handler.py adds this to sys.path at runtime.
# Use shallow clone to keep image lean (~10 MB vs ~250 MB with full history).
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/wan22

# Install WAN2.2 deps, excluding:
#   flash_attn  — needs full CUDA/C++ build (30+ min); torch SDPA is the fallback
#   dashscope   — only needed for LLM-based prompt expansion, we skip that feature
RUN grep -vE '^\s*(flash.attn|dashscope)' /opt/wan22/requirements.txt \
        > /tmp/wan_req.txt && \
    pip install --no-cache-dir -r /tmp/wan_req.txt

# ── Handler dependencies ─────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    runpod \
    boto3 \
    huggingface_hub \
    einops

# ── Copy handler ─────────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
