# Story Studio WAN2.2 TI2V-5B — API Reference

WAN2.2 TI2V-5B is a 5B-parameter video generation model.
One model handles both **Text-to-Video (T2V)** and **Image-to-Video (I2V)** generation.

- Resolution: **720P** — `1280×704` (landscape) or `704×1280` (portrait)
- Frame rate: **24 FPS** (fixed)
- Max quality: 121 frames = 5.0 seconds

---

## Submit / Poll Pattern

All requests are **async** via RunPod's serverless queue:

```bash
# 1. Submit
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {...}}'
# → {"id": "abc123", "status": "IN_QUEUE"}

# 2. Poll until status == COMPLETED
curl https://api.runpod.ai/v2/{ENDPOINT_ID}/status/abc123 \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
# → {"id": "abc123", "status": "COMPLETED", "output": {...}}
```

---

## Modes

### `t2v` — Text to Video

Generate a video from a text prompt, no input image required.

**Request:**
```json
{
  "mode": "t2v",
  "prompt": "Two anthropomorphic cats in boxing gear fight on a spotlit stage",
  "negative_prompt": "",
  "size": "1280*704",
  "frame_num": 81,
  "steps": 50,
  "guidance": 5.0,
  "seed": -1
}
```

**Response:**
```json
{
  "mode": "t2v",
  "video": "https://pub-xxx.r2.dev/jobs/abc123.mp4",
  "size": "1280x704",
  "frame_num": 81,
  "duration_s": 3.37,
  "fps": 24,
  "gen_time_s": 48.2,
  "load_time_s": 32.0
}
```

---

### `i2v` — Image to Video

Animate a reference image from a text prompt. The image becomes the first frame.

**Request:**
```json
{
  "mode": "i2v",
  "prompt": "The cat slowly turns to look at the camera, blinking",
  "image_url": "https://your-storage.com/frame.jpg",
  "negative_prompt": "",
  "size": "1280*704",
  "frame_num": 81,
  "steps": 50,
  "guidance": 5.0,
  "seed": -1
}
```

Or with base64-encoded image:
```json
{
  "mode": "i2v",
  "prompt": "...",
  "image": "<base64_encoded_png_or_jpg>",
  ...
}
```

**Response:** Same as `t2v`.

---

## Field Reference

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `mode` | yes | — | `"t2v"` or `"i2v"` |
| `prompt` | yes | — | Text description of the video |
| `image` | i2v only | — | Base64-encoded image (PNG or JPG) |
| `image_url` | i2v only | — | Public URL to image (alternative to `image`) |
| `negative_prompt` | no | `""` | Content to avoid in generation |
| `size` | no | `"1280*704"` | Output resolution: `"1280*704"` or `"704*1280"` |
| `frame_num` | no | `81` | Number of frames — must be `4n+1` |
| `steps` | no | `50` | Diffusion steps (20–50; fewer = faster, lower quality) |
| `guidance` | no | `5.0` | Classifier-free guidance scale (3.0–7.0) |
| `seed` | no | `-1` | Random seed; `-1` = random |

---

## Frame Count → Duration Reference

Frame counts must satisfy `(frame_num - 1) % 4 == 0`:

| `frame_num` | Duration at 24 FPS | Notes |
|------------|-------------------|-------|
| 49 | 2.04 s | Quick preview |
| 65 | 2.71 s | Short |
| 81 | 3.37 s | Recommended default |
| 97 | 4.04 s | Medium |
| 113 | 4.71 s | Long |
| 121 | 5.04 s | Maximum quality |

---

## Resolution Guide

| `size` | Pixels | Use for |
|--------|--------|---------|
| `1280*704` | 720P landscape | Horizontal videos, cinematic shots |
| `704*1280` | 720P portrait | TikTok / Instagram Reels, vertical stories |

> For `i2v`, the input image is automatically resized to match the `size` parameter, preserving aspect ratio with center crop.

---

## Performance (A40 — 48 GB VRAM)

| `frame_num` | `steps` | Approx. Time |
|------------|---------|-------------|
| 81 | 50 | ~45–60 s |
| 121 | 50 | ~70–90 s |
| 81 | 20 | ~20–30 s |

Cold start (model load): ~30–40 s on first request.

---

## Python Example

```python
import requests, time, os

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def generate_video(payload: dict) -> dict:
    r = requests.post(f"{ENDPOINT}/run", json={"input": payload}, headers=HEADERS)
    job_id = r.json()["id"]
    while True:
        time.sleep(5)
        r = requests.get(f"{ENDPOINT}/status/{job_id}", headers=HEADERS)
        result = r.json()
        if result["status"] == "COMPLETED":
            return result["output"]
        if result["status"] == "FAILED":
            raise RuntimeError(result.get("error", "Job failed"))

# T2V example
output = generate_video({
    "mode": "t2v",
    "prompt": "A golden retriever puppy runs through a field of sunflowers at sunset",
    "size": "1280*704",
    "frame_num": 81
})
print(output["video"])   # → https://pub-xxx.r2.dev/jobs/abc.mp4

# I2V example
output = generate_video({
    "mode": "i2v",
    "prompt": "The flowers sway in a gentle breeze, petals catching the light",
    "image_url": "https://your-storage.com/frame.jpg",
    "frame_num": 97
})
print(output["video"])
```
