"""
handler_fp8.py — Story Studio WAN2.2 TI2V-5B FP8 (diffusers)
=============================================================
RunPod serverless handler using the diffusers WAN pipeline with
shunyang90/Wan2.2-TI2V-5B-ModelOpt-FP8 weights (19.3 GB, Ada/Hopper/Blackwell).

Modes and payload identical to handler.py — drop-in replacement.

── PAYLOAD ──────────────────────────────────────────────────────────────────

t2v:
  { "mode": "t2v",
    "prompt": "Two cats boxing on a spotlit stage",
    "negative_prompt": "",
    "size": "1280*704",      # 1280*704 (landscape) | 704*1280 (portrait)
    "frame_num": 81,         # 4n+1: 49|65|81|97|113|121  (121=5s default)
    "steps": 50,
    "guidance": 5.0,
    "seed": -1 }

i2v:
  { "mode": "i2v",
    "prompt": "The cat slowly turns its head toward the camera",
    "image": "<base64_png>",
    "image_url": "https://...",
    "negative_prompt": "",
    "size": "1280*704",
    "frame_num": 81,
    "steps": 50,
    "guidance": 5.0,
    "seed": -1 }

── RESPONSE ────────────────────────────────────────────────────────────────
  { "mode": "t2v"|"i2v",
    "video": "https://pub-xxx.r2.dev/jobs/abc.mp4",
    "size": "1280x704",
    "frame_num": 81,
    "duration_s": 3.4,
    "fps": 24,
    "gen_time_s": 45.2 }
"""

import os
import io
import uuid
import base64
import tempfile
import traceback
import time
import urllib.request

import boto3
import torch
import runpod
import imageio
import numpy as np
from PIL import Image
from botocore.config import Config

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_BASE   = os.environ.get("MODEL_BASE",     "/runpod-volume/models")
WAN_FP8_MODEL = os.environ.get("WAN_FP8_MODEL", os.path.join(MODEL_BASE, "wan22-ti2v-5b-fp8"))

# ── Cloudflare R2 ─────────────────────────────────────────────────────────────
R2_ACCOUNT_ID        = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME       = os.environ.get("R2_BUCKET_NAME", "e2e-storystudio")
R2_PUBLIC_URL        = os.environ.get("R2_PUBLIC_URL",  "https://pub-bce4924e66d944668be30268ccf4492c.r2.dev")

_r2_client = None

def get_r2():
    global _r2_client
    if _r2_client is None:
        _r2_client = boto3.client(
            "s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
    return _r2_client

def upload_bytes(data: bytes, ext: str, content_type: str) -> str:
    key = f"jobs/{uuid.uuid4().hex}.{ext}"
    get_r2().put_object(Bucket=R2_BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
    return f"{R2_PUBLIC_URL}/{key}"

# ── Global pipeline handles ───────────────────────────────────────────────────
_pipe_t2v  = None
_pipe_i2v  = None
_load_time = None

_VALID_SIZES = {"1280*704", "704*1280"}
_FPS = 24


def _patch_transformer_config():
    """
    Remove quantization_config from transformer/config.json so diffusers loads
    the FP8 safetensors weights as plain BF16 tensors via torch_dtype casting.
    This avoids requiring nvidia-modelopt (which conflicts with torchvision in
    the base image). Still benefits from the 4x smaller transformer on disk.
    Runs once; subsequent workers find quantization_config already absent.
    """
    import json
    from pathlib import Path
    cfg_path = Path(WAN_FP8_MODEL) / "transformer" / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    if "quantization_config" in cfg:
        del cfg["quantization_config"]
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print("[LOADER] Removed quantization_config from transformer/config.json (FP8 weights load as BF16)")


def load_model():
    global _pipe_t2v, _pipe_i2v, _load_time
    if _pipe_t2v is not None:
        return

    from diffusers import WanPipeline, WanImageToVideoPipeline

    _patch_transformer_config()

    print(f"[LOADER] WAN2.2 TI2V-5B FP8 from {WAN_FP8_MODEL} ...")
    t0 = time.time()

    _pipe_t2v = WanPipeline.from_pretrained(
        WAN_FP8_MODEL,
        torch_dtype=torch.bfloat16,
    )
    _pipe_t2v.to("cuda")

    # Reuse loaded weights — no double VRAM cost
    _pipe_i2v = WanImageToVideoPipeline(**_pipe_t2v.components)

    _load_time = round(time.time() - t0, 1)
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[LOADER] Ready in {_load_time}s  ({vram_gb:.1f} GB VRAM used)")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _download_temp(url: str, suffix: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0", "Accept": "*/*"})
    data = urllib.request.urlopen(req, timeout=60).read()
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(data)
    f.close()
    return f.name

def _load_image(inp: dict) -> Image.Image | None:
    image_b64 = inp.get("image")
    image_url = inp.get("image_url")
    if image_b64:
        return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    if image_url:
        path = _download_temp(image_url, ".jpg")
        try:
            return Image.open(path).convert("RGB")
        finally:
            if os.path.exists(path):
                os.unlink(path)
    return None

def _frames_to_mp4(frames: list, fps: int = 24) -> bytes:
    """Convert list of PIL images to MP4 bytes via imageio."""
    buf = io.BytesIO()
    writer = imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264",
                                quality=8, output_params=["-movflags", "+faststart"])
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# RunPod handler
# ══════════════════════════════════════════════════════════════════════════════
def handler(event):
    try:
        cold_start = _pipe_t2v is None
        load_model()

        inp  = event.get("input", {})
        mode = inp.get("mode", "t2v").lower()

        if mode not in ("t2v", "i2v"):
            return {"error": f"Invalid mode '{mode}'. Valid: t2v, i2v"}

        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        neg_prompt = inp.get("negative_prompt", "")
        size_str   = inp.get("size", "1280*704")
        if size_str not in _VALID_SIZES:
            return {"error": f"Invalid size '{size_str}'. Valid: {sorted(_VALID_SIZES)}"}

        frame_num = int(inp.get("frame_num", 81))
        if (frame_num - 1) % 4 != 0:
            return {"error": f"frame_num must be 4n+1 (e.g. 49,65,81,97,113,121). Got {frame_num}"}

        steps    = int(inp.get("steps",    50))
        guidance = float(inp.get("guidance", 5.0))
        seed     = int(inp.get("seed", -1))
        w, h     = [int(x) for x in size_str.split("*")]

        generator = torch.Generator("cuda")
        if seed == -1:
            generator.seed()
        else:
            generator.manual_seed(seed)

        print(f"[WAN] mode={mode} size={size_str} frames={frame_num} steps={steps} seed={seed}")
        t_gen = time.time()

        if mode == "t2v":
            output = _pipe_t2v(
                prompt           = prompt,
                negative_prompt  = neg_prompt,
                num_frames       = frame_num,
                height           = h,
                width            = w,
                num_inference_steps = steps,
                guidance_scale   = guidance,
                generator        = generator,
            )
        else:
            img = _load_image(inp)
            if img is None:
                return {"error": "i2v mode requires 'image' (base64) or 'image_url'"}
            output = _pipe_i2v(
                image            = img,
                prompt           = prompt,
                negative_prompt  = neg_prompt,
                num_frames       = frame_num,
                height           = h,
                width            = w,
                num_inference_steps = steps,
                guidance_scale   = guidance,
                generator        = generator,
            )

        gen_time_s = round(time.time() - t_gen, 1)
        print(f"[WAN] generated in {gen_time_s}s, uploading...")

        frames    = output.frames[0]  # list of PIL images
        mp4_bytes = _frames_to_mp4(frames, fps=_FPS)
        video_url = upload_bytes(mp4_bytes, "mp4", "video/mp4")

        duration_s = round(frame_num / _FPS, 2)
        result = {
            "mode":       mode,
            "video":      video_url,
            "size":       f"{w}x{h}",
            "frame_num":  frame_num,
            "duration_s": duration_s,
            "fps":        _FPS,
            "gen_time_s": gen_time_s,
        }
        if cold_start:
            result["load_time_s"] = _load_time
        return result

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return {"error": str(e), "traceback": tb}


runpod.serverless.start({"handler": handler})
