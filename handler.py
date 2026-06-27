"""
handler.py — Story Studio WAN2.2 TI2V-5B Pod
==============================================
RunPod serverless handler for WAN2.2 TI2V-5B video generation.

Modes:
  "t2v"  → Text-to-video: text prompt → MP4
  "i2v"  → Image-to-video: image + text prompt → MP4 (image as first frame)

Both use the same TI2V-5B checkpoint.

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
    "image": "<base64_png>",    # base64 PNG/JPG  — OR —
    "image_url": "https://...", # public image URL
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

── VALID FRAME COUNTS (must be 4n+1) ────────────────────────────────────────
  @24fps:  49=2.0s  65=2.7s  81=3.4s  97=4.0s  113=4.7s  121=5.0s
"""

import os
import io
import sys
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

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_BASE = os.environ.get("MODEL_BASE", "/runpod-volume/models")
WAN_MODEL  = os.path.join(MODEL_BASE, "wan22-ti2v-5b")
WAN_LIB    = os.environ.get("WAN_LIB", "/opt/wan22")

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

# ── Global model handle ───────────────────────────────────────────────────────
wan_model   = None
_load_time  = None
_VALID_SIZES = {"1280*704", "704*1280"}
_FPS = 24

def load_model():
    global wan_model, _load_time

    if wan_model is not None:
        return

    # Add WAN2.2 library to path
    if WAN_LIB not in sys.path:
        sys.path.insert(0, WAN_LIB)

    import wan
    from wan.configs import WAN_CONFIGS

    print(f"[LOADER] WAN2.2 TI2V-5B from {WAN_MODEL} ...")
    t0 = time.time()

    config = WAN_CONFIGS["ti2v-5B"]

    # On A40 (48 GB) — no offloading needed for speed
    # On 24 GB GPUs — set offload_model=True at generate() time
    wan_model = wan.WanTI2V(
        config         = config,
        checkpoint_dir = WAN_MODEL,
        device_id      = 0,
        t5_cpu         = False,        # keep T5 on GPU for speed (A40 has headroom)
        offload_model  = False,        # override per-request for smaller GPUs
    )
    _load_time = round(time.time() - t0, 1)
    print(f"[LOADER] WAN2.2 TI2V-5B ready in {_load_time}s")


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
            if os.path.exists(path): os.unlink(path)
    return None

def _frames_to_mp4(frames: torch.Tensor, fps: int = 24) -> bytes:
    """Convert (C, N, H, W) float tensor [0,1] to MP4 bytes via imageio."""
    # frames shape: (C, N, H, W), range ~[-1, 1] or [0, 1] depending on WAN version
    arr = frames.permute(1, 2, 3, 0).cpu().float().numpy()  # (N, H, W, C)
    arr = np.clip((arr + 1.0) / 2.0, 0, 1)                  # normalize to [0,1]
    arr = (arr * 255).astype(np.uint8)

    buf = io.BytesIO()
    writer = imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264",
                                quality=8, output_params=["-movflags", "+faststart"])
    for frame in arr:
        writer.append_data(frame)
    writer.close()
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# RunPod handler
# ══════════════════════════════════════════════════════════════════════════════
def handler(event):
    try:
        cold_start = wan_model is None
        load_model()

        inp  = event.get("input", {})
        mode = inp.get("mode", "t2v").lower()

        if mode not in ("t2v", "i2v"):
            return {"error": f"Invalid mode '{mode}'. Valid: t2v, i2v"}

        prompt   = inp.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        neg_prompt = inp.get("negative_prompt", "")
        size_str   = inp.get("size", "1280*704")
        if size_str not in _VALID_SIZES:
            return {"error": f"Invalid size '{size_str}'. Valid: {sorted(_VALID_SIZES)}"}

        frame_num = int(inp.get("frame_num", 81))
        if (frame_num - 1) % 4 != 0:
            return {"error": f"frame_num must be 4n+1 (e.g. 49,65,81,97,113,121). Got {frame_num}"}

        steps     = int(inp.get("steps",    50))
        guidance  = float(inp.get("guidance", 5.0))
        seed      = int(inp.get("seed", -1))
        w, h      = [int(x) for x in size_str.split("*")]

        # I2V: load reference image
        img = None
        if mode == "i2v":
            img = _load_image(inp)
            if img is None:
                return {"error": "i2v mode requires 'image' (base64) or 'image_url'"}

        print(f"[WAN] mode={mode} size={size_str} frames={frame_num} steps={steps} seed={seed}")
        t_gen = time.time()

        frames = wan_model.generate(
            input_prompt  = prompt,
            img           = img,
            size          = (w, h),
            max_area      = w * h,
            frame_num     = frame_num,
            shift         = 5.0,
            sample_solver = "unipc",
            sampling_steps = steps,
            guide_scale   = guidance,
            n_prompt      = neg_prompt,
            seed          = seed,
            offload_model = False,
        )

        gen_time_s = round(time.time() - t_gen, 1)
        print(f"[WAN] generated in {gen_time_s}s, uploading...")

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
