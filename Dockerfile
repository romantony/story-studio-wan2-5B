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

# ── flash_attn (build from source — requires devel image with nvcc, ~30 min) ─
# Placed early so this expensive layer is cached independently of WAN2.2 / handler changes.
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# ── WAN2.2 library ───────────────────────────────────────────────────────────
# Clone into /opt/wan22 — handler.py adds this to sys.path at runtime.
# Use shallow clone to keep image lean (~10 MB vs ~250 MB with full history).
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/wan22

# wan/__init__.py has top-level imports for optional features (speech2video,
# animate) that drag in heavy/missing deps (librosa, peft) at import time.
# We only use WanTI2V — strip everything else.
RUN sed -i '/speech2video\|WanS2V\|animate\|WanAnimate/d' /opt/wan22/wan/__init__.py

# Patch attention.py: the else-branch hard-asserts flash_attn v2 with no fallback.
# Replace it with a torch SDPA fallback so we skip the 30-min flash_attn build.
RUN python3 - << 'PATCH'
import pathlib

p = pathlib.Path('/opt/wan22/wan/modules/attention.py')
src = p.read_text()

old = '''    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))'''

new = '''    else:
        if FLASH_ATTN_2_AVAILABLE:
            x = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(k.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic).unflatten(0, (b, lq))
        else:
            import torch.nn.functional as F
            total_q = q.shape[0]
            if total_q == b * lq:
                q_b = q.unflatten(0, (b, lq))
                k_b = k.unflatten(0, (b, lk))
                v_b = v.unflatten(0, (b, lk))
            else:
                max_lq = int(q_lens.max().item())
                max_lk = int(k_lens.max().item())
                q_b = q.new_zeros(b, max_lq, q.shape[1], q.shape[2])
                k_b = k.new_zeros(b, max_lk, k.shape[1], k.shape[2])
                v_b = v.new_zeros(b, max_lk, v.shape[1], v.shape[2])
                oq, ok = 0, 0
                for i in range(b):
                    ql, kl = int(q_lens[i].item()), int(k_lens[i].item())
                    q_b[i, :ql] = q[oq:oq + ql]; oq += ql
                    k_b[i, :kl] = k[ok:ok + kl]
                    v_b[i, :kl] = v[ok:ok + kl]; ok += kl
                lq = max_lq
            x = F.scaled_dot_product_attention(
                q_b.transpose(1, 2), k_b.transpose(1, 2), v_b.transpose(1, 2),
                scale=softmax_scale, is_causal=causal, dropout_p=dropout_p,
            ).transpose(1, 2)'''

assert old in src, f"Pattern not found — attention.py may have changed upstream"
p.write_text(src.replace(old, new))
print("attention.py patched OK")
PATCH

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
    einops \
    decord \
    imageio-ffmpeg \
    timm

# ── Copy handler ─────────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
