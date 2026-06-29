# WAN 2.2 TI2V-5B — API Reference

Endpoint: `https://api.runpod.ai/v2/hyfdmqik62okqs`

GPU: A40 (48 GB VRAM) · Model: WAN 2.2 TI2V-5B (BF16) · FPS: 24 (fixed)

All requests require header: `Authorization: Bearer <RUNPOD_API_KEY>`

---

## Job lifecycle

```
POST /run          → { "id": "<job_id>", "status": "IN_QUEUE" }
GET  /status/<id>  → { "status": "COMPLETED", "output": { ... } }
```

`status` values: `IN_QUEUE` → `IN_PROGRESS` → `COMPLETED` | `FAILED`

Poll every 10 s. Cold start (model load from network volume) takes **90–120 s** before
generation begins on a freshly launched worker.

---

## Modes

| Mode | Description |
|------|-------------|
| `i2v` | Image-to-video — animate a reference image guided by a text prompt |
| `t2v` | Text-to-video — generate a video from a text prompt only |

---

## mode: `"i2v"` — Image to Video *(primary mode)*

Animates a reference image into a video. The image becomes the first frame.

> **Use 720p for any content featuring human subjects.** 480p produces face distortion
> regardless of step count — confirmed in testing.

### Request

```json
{
  "input": {
    "mode": "i2v",

    "image_url": "https://...",
    "image":     "<base64_png_or_jpg>",

    "prompt":          "She walks gracefully along the garden path, turning slightly to look around, leaves gently moving in the breeze",
    "negative_prompt": "",

    "size":      "720*1280",
    "frame_num": 121,
    "steps":     30,
    "guidance":  5.0,
    "seed":      -1
  }
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `image_url` | string | — | Public URL to reference image (PNG/JPG). Required if no `image`. |
| `image` | string | — | Base64-encoded PNG/JPG. Required if no `image_url`. |
| `prompt` | string | **required** | Describe motion, action, camera movement, atmosphere. |
| `negative_prompt` | string | `""` | Things to avoid in the video. |
| `size` | string | `"1280*704"` | `"width*height"`. See resolution guide below. |
| `frame_num` | int | `81` | Must satisfy `(N-1) % 4 == 0`. See frame table below. |
| `steps` | int | `50` | Denoising steps. **30 recommended** for best quality/speed. |
| `guidance` | float | `5.0` | Guidance scale. Range: 3.0–7.0. |
| `seed` | int | `-1` | `-1` = random. Set a fixed value for reproducibility. |

---

## mode: `"t2v"` — Text to Video

Generates a video from a text prompt alone, no reference image needed.

### Request

```json
{
  "input": {
    "mode": "t2v",

    "prompt":          "Aerial view of a city at golden hour, slow cinematic pan, dramatic clouds",
    "negative_prompt": "",

    "size":      "1280*704",
    "frame_num": 121,
    "steps":     30,
    "guidance":  5.0,
    "seed":      -1
  }
}
```

Fields identical to `i2v`, minus `image_url` / `image`.

---

## Response

```json
{
  "mode":       "i2v",
  "video":      "https://pub-bce4924e66d944668be30268ccf4492c.r2.dev/jobs/<uuid>.mp4",
  "size":       "720x1280",
  "frame_num":  121,
  "duration_s": 5.04,
  "fps":        24,
  "gen_time_s": 574.5
}
```

`load_time_s` is only present on cold-start (first request after worker launch).

---

## Resolution guide

| `size` | Width | Height | Aspect | Notes |
|--------|-------|--------|--------|-------|
| `"720*1280"` | 720 | 1280 | 9:16 portrait | **Recommended — human subjects, Shorts/Reels** |
| `"704*1280"` | 704 | 1280 | 9:16 portrait | Alternative portrait |
| `"1280*704"` | 1280 | 704 | 16:9 landscape | YouTube landscape |
| `"832*480"` | 832 | 480 | 16:9 landscape | 480p landscape — non-human content only |
| `"480*832"` | 480 | 832 | 9:16 portrait | ⚠ Face distortion on human subjects — avoid |

---

## Frame count table

`(N - 1) % 4 == 0` — required by the WAN VAE temporal compression.

| Duration | `frame_num` |
|----------|-------------|
| ~2 s | 49 |
| ~3 s | 65 |
| ~4 s | 97 |
| **~5 s** | **121** ← recommended |
| ~6 s | 145 |
| **~7 s** | **169** |
| ~8 s | 193 |

Invalid frame counts return an error with the nearest valid values.

---

## Step count guide

Tested at 720p 9:16 portrait with human subject:

| `steps` | Gen time · 5 s video | Quality |
|---------|----------------------|---------|
| 25 | ~500 s | Good |
| **30** | **~575 s** | **Best — recommended** |
| 40 | ~760 s | Diminishing returns |
| 50 | ~950 s | Maximum |

---

## Timing benchmarks

| Scenario | Gen time (warm worker) |
|----------|------------------------|
| Cold start — model load | 90–120 s |
| 720p 5 s · 121 frames · 30 steps | ~575 s |
| 720p 7 s · 169 frames · 30 steps | ~800 s |
| 480p 5 s · 121 frames · 30 steps | ~290 s |
| 480p 7 s · 169 frames · 30 steps | ~320 s |

---

## Example payloads

**720p portrait · 5 s · i2v (recommended for human subjects)**
```json
{
  "input": {
    "mode": "i2v",
    "image_url": "https://pub-bce4924e66d944668be30268ccf4492c.r2.dev/jobs/abc.png",
    "prompt": "She walks gracefully along the garden path, turning slightly to look around, leaves gently moving in the breeze",
    "size": "720*1280",
    "frame_num": 121,
    "steps": 30,
    "guidance": 5.0,
    "seed": -1
  }
}
```

**720p portrait · 7 s · i2v**
```json
{
  "input": {
    "mode": "i2v",
    "image_url": "https://pub-bce4924e66d944668be30268ccf4492c.r2.dev/jobs/abc.png",
    "prompt": "She gazes at the horizon, a light breeze moves her hair, city lights begin to appear below",
    "size": "720*1280",
    "frame_num": 169,
    "steps": 30,
    "guidance": 5.0,
    "seed": -1
  }
}
```

**720p landscape · 5 s · t2v**
```json
{
  "input": {
    "mode": "t2v",
    "prompt": "Aerial view of a city at golden hour, slow cinematic pan, dramatic clouds",
    "size": "1280*704",
    "frame_num": 121,
    "steps": 30,
    "guidance": 5.0,
    "seed": 42
  }
}
```

---

## Error responses

```json
{
  "error": "Description of what went wrong",
  "traceback": "Full Python traceback (unexpected exceptions only)"
}
```

| Error | Cause |
|-------|-------|
| `"prompt is required"` | No prompt provided |
| `"i2v mode requires 'image' (base64) or 'image_url'"` | i2v with no image |
| `"Invalid size '480*900'. Valid: [...]"` | Size not in allowed list |
| `"frame_num must be 4n+1 (e.g. 49,65,81,...). Got 120"` | Invalid frame count |
| `"Invalid mode 'video'. Valid: t2v, i2v"` | Wrong mode string |

---

## curl

```bash
# Submit
curl -X POST https://api.runpod.ai/v2/hyfdmqik62okqs/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "input": {
      "mode": "i2v",
      "image_url": "https://example.com/portrait.png",
      "prompt": "She smiles warmly, a light breeze moves her hair",
      "size": "720*1280",
      "frame_num": 121,
      "steps": 30
    }
  }'

# Poll
curl https://api.runpod.ai/v2/hyfdmqik62okqs/status/<job_id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## Python

```python
import requests, time, os

API_KEY  = os.environ["RUNPOD_API_KEY"]
BASE_URL = "https://api.runpod.ai/v2/hyfdmqik62okqs"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def generate_video(payload: dict) -> dict:
    job_id = requests.post(f"{BASE_URL}/run", json={"input": payload}, headers=HEADERS).json()["id"]
    while True:
        time.sleep(10)
        result = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS).json()
        if result["status"] == "COMPLETED":
            return result["output"]
        if result["status"] == "FAILED":
            raise RuntimeError(result.get("output", {}).get("error", "Job failed"))

# i2v — 720p portrait, 5 s
output = generate_video({
    "mode":       "i2v",
    "image_url":  "https://example.com/portrait.png",
    "prompt":     "She smiles warmly, a light breeze moves her hair",
    "size":       "720*1280",
    "frame_num":  121,
    "steps":      30,
})
print(output["video"])  # → https://pub-....r2.dev/jobs/<uuid>.mp4
```
