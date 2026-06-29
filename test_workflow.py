#!/usr/bin/env python3
"""
test_workflow.py — Character consistency + video generation end-to-end test
===========================================================================

Steps:
  0. Scale up: Flux (rnqxi6c0mlq517) + WAN A40 (hyfdmqik62okqs) → min=1, max=1
     Wait 120 s for workers to warm up.

  1. Flux T2I → reference image (9:16 portrait, 576×1024)

  2. Flux I2I × 3 → scenario images using reference for character consistency
       A: café/coffee shop
       B: sunlit park
       C: rooftop at sunset

  3. WAN i2v (image A) → 5 s video, 480p 9:16 portrait, 25 steps
  4. WAN i2v (image B) → 7 s video, 480p 9:16 portrait, 25 steps
  5. WAN i2v (image C) → 5 s video, 720p 9:16 portrait, 25 steps

  6. Scale down: both endpoints → min=0

Usage:
  export RUNPOD_API_KEY=<your_key>
  python3 test_workflow.py
"""

import base64
import json
import os
import sys
import time
import urllib.request

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
if not API_KEY:
    sys.exit("ERROR: set RUNPOD_API_KEY first")

FLUX_EP = "rnqxi6c0mlq517"   # Pod 2 — FLUX + TTS
WAN_EP  = "hyfdmqik62okqs"   # WAN 2.2 TI2V-5B on A40

RUNPOD_BASE = "https://api.runpod.ai/v2"
GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

POLL_INTERVAL = 10   # seconds between status checks
JOB_TIMEOUT   = 900  # max seconds to wait for any single job

# ── Character & scene prompts ─────────────────────────────────────────────────

CHARACTER = (
    "A woman in her early 30s with dark wavy shoulder-length hair and warm brown eyes, "
    "wearing a deep cobalt blue structured blazer over a white top, minimal jewellery, "
    "confident yet approachable expression, soft natural lighting, sharp focus, "
    "vertical 9:16 portrait composition, photorealistic"
)

SCENARIOS = [
    {
        "label": "A — Café",
        "image_prompt": (
            f"{CHARACTER}, sitting at a warm wooden café table, ceramic coffee cup, "
            "soft window light, cozy European interior, shallow depth of field"
        ),
        "video_prompt": (
            "She glances up from her coffee with a gentle smile, soft ambient café sounds, "
            "warm light plays across her face, subtle shoulder movement"
        ),
        "wan_size":   "480*832",
        "wan_frames": 121,        # 5 s @ 24 fps
        "wan_fps":    24,
        "label_video": "Step 3 — 5 s 480p 9:16",
    },
    {
        "label": "B — Park",
        "image_prompt": (
            f"{CHARACTER}, walking through a sun-dappled botanical garden path, "
            "lush green trees, dappled golden light, relaxed stride, natural outdoor setting"
        ),
        "video_prompt": (
            "She walks slowly toward the camera, turns slightly to look at the surrounding trees, "
            "leaves gently rustle, warm sunlight filters through the canopy"
        ),
        "wan_size":   "480*832",
        "wan_frames": 169,        # 7 s @ 24 fps
        "wan_fps":    24,
        "label_video": "Step 4 — 7 s 480p 9:16",
    },
    {
        "label": "C — Rooftop Sunset",
        "image_prompt": (
            f"{CHARACTER}, standing on a rooftop terrace at golden hour, city skyline behind her, "
            "warm sunset glow, holding a glass of white wine, soft wind"
        ),
        "video_prompt": (
            "She gazes at the horizon with a serene expression, a light breeze moves her hair, "
            "city lights begin to twinkle below as the sky deepens to amber"
        ),
        "wan_size":   "720*1280",
        "wan_frames": 121,        # 5 s @ 24 fps
        "wan_fps":    24,
        "label_video": "Step 5 — 5 s 720p 9:16",
    },
]

# ── RunPod helpers ────────────────────────────────────────────────────────────

def _headers():
    return {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

def _http(method, url, body=None):
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(url, data=data, headers=_headers(), method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def set_workers(endpoint_id, min_workers, max_workers):
    """Set serverless endpoint worker counts via RunPod GraphQL."""
    query = """
    mutation UpdateEndpoint($id: String!, $workersMin: Int!, $workersMax: Int!) {
      updateEndpoint(input: { id: $id, workersMin: $workersMin, workersMax: $workersMax }) {
        id workersMin workersMax
      }
    }
    """
    payload = {
        "query": query,
        "variables": {"id": endpoint_id, "workersMin": min_workers, "workersMax": max_workers},
    }
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    if "errors" in resp:
        raise RuntimeError(f"GraphQL error setting {endpoint_id} workers: {resp['errors']}")
    result = resp["data"]["updateEndpoint"]
    print(f"  [{endpoint_id}] workers → min={result['workersMin']} max={result['workersMax']}")

def submit_job(endpoint_id, payload):
    resp = _http("POST", f"{RUNPOD_BASE}/{endpoint_id}/run", payload)
    job_id = resp.get("id")
    if not job_id:
        raise RuntimeError(f"Job submit failed: {resp}")
    return job_id

def poll_job(endpoint_id, job_id, label="job"):
    """Poll until COMPLETED or FAILED. Returns output dict."""
    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        resp    = _http("GET", f"{RUNPOD_BASE}/{endpoint_id}/status/{job_id}")
        status  = resp.get("status", "")
        print(f"  [{elapsed:4d}s] {label} — {status}")
        if status == "COMPLETED":
            output = resp.get("output", {})
            if isinstance(output, dict) and "error" in output:
                raise RuntimeError(f"Job error: {output['error']}")
            return output
        if status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Job {status}: {resp}")
        if elapsed > JOB_TIMEOUT:
            raise TimeoutError(f"Job {job_id} timed out after {JOB_TIMEOUT}s")
        time.sleep(POLL_INTERVAL)

def download_b64(url):
    """Download an image URL and return base64 string (PNG/JPG)."""
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return base64.b64encode(data).decode()

# ── Step runners ──────────────────────────────────────────────────────────────

def run_flux_t2i(label, image_prompt, reference_b64=None, width=576, height=1024, seed=None):
    """Generate one image. Returns the public image URL."""
    inp = {
        "mode":         "image",
        "image_prompt": image_prompt,
        "img_width":    width,
        "img_height":   height,
        "img_steps":    4,
        "img_guidance": 1.0,
    }
    if reference_b64:
        inp["reference_images"] = [reference_b64]
    if seed is not None:
        inp["seed"] = seed

    print(f"\n▶ Flux image — {label}")
    job_id = submit_job(FLUX_EP, {"input": inp})
    print(f"  job_id: {job_id}")
    output = poll_job(FLUX_EP, job_id, label)
    url = output.get("image") or output.get("image_url")
    if not url:
        raise RuntimeError(f"No image URL in output: {output}")
    gen = output.get("gen_time_s", "?")
    print(f"  ✓ {label}: {url}  ({gen}s)")
    return url

def run_wan_i2v(label, image_url, video_prompt, size, frame_num, steps=25, seed=-1):
    """Generate one video. Returns the public video URL."""
    inp = {
        "mode":            "i2v",
        "prompt":          video_prompt,
        "image_url":       image_url,
        "size":            size,
        "frame_num":       frame_num,
        "steps":           steps,
        "guidance":        5.0,
        "seed":            seed,
    }
    w, h = size.split("*")
    duration = round(frame_num / 24, 1)
    print(f"\n▶ WAN i2v — {label}  ({w}×{h}, {frame_num}f/{duration}s, {steps} steps)")
    job_id = submit_job(WAN_EP, {"input": inp})
    print(f"  job_id: {job_id}")
    output = poll_job(WAN_EP, job_id, label)
    url     = output.get("video")
    gen     = output.get("gen_time_s", "?")
    load    = output.get("load_time_s")
    timing  = f"gen={gen}s" + (f"  load={load}s" if load else "")
    print(f"  ✓ {label}: {url}  ({timing})")
    return url, output

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}

    # ── 0. Scale up ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 0 — Scale up endpoints (min=1, max=1)")
    print("=" * 60)
    set_workers(FLUX_EP, min_workers=1, max_workers=1)
    set_workers(WAN_EP,  min_workers=1, max_workers=1)
    print("Waiting 120 s for workers to warm up...")
    for i in range(12):
        time.sleep(10)
        print(f"  {(i+1)*10}s / 120s")

    # ── 1. Reference image ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1 — Generate reference character image (Flux, 576×1024)")
    print("=" * 60)
    ref_url = run_flux_t2i(
        "Reference character",
        CHARACTER,
        seed=42,
    )
    results["step1_reference"] = ref_url

    # Download as base64 for I2I reference
    print("  Downloading reference image for I2I conditioning...")
    ref_b64 = download_b64(ref_url)
    print(f"  Downloaded ({len(ref_b64)//1024} KB base64)")

    # ── 2. Scenario images (character consistency) ────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 — Generate 3 scenario images (Flux I2I, character consistency)")
    print("=" * 60)
    scenario_urls = []
    for i, sc in enumerate(SCENARIOS):
        url = run_flux_t2i(
            sc["label"],
            sc["image_prompt"],
            reference_b64=ref_b64,
            seed=100 + i,
        )
        scenario_urls.append(url)
        results[f"step2_{sc['label'][0].lower()}"] = url

    # ── 3-5. Videos from each scenario image ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Steps 3-5 — Generate videos (WAN i2v, A40, 25 steps)")
    print("=" * 60)
    for i, sc in enumerate(SCENARIOS):
        url, output = run_wan_i2v(
            sc["label_video"],
            image_url   = scenario_urls[i],
            video_prompt= sc["video_prompt"],
            size        = sc["wan_size"],
            frame_num   = sc["wan_frames"],
            steps       = 25,
            seed        = 200 + i,
        )
        results[f"step{3+i}_video"] = url
        results[f"step{3+i}_stats"] = output

    # ── 6. Scale down ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6 — Scale down endpoints (min=0)")
    print("=" * 60)
    set_workers(FLUX_EP, min_workers=0, max_workers=1)
    set_workers(WAN_EP,  min_workers=0, max_workers=1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nStep 1 — Reference image:")
    print(f"  {results['step1_reference']}")
    print(f"\nStep 2 — Scenario images (character consistency):")
    for i, sc in enumerate(SCENARIOS):
        key = f"step2_{sc['label'][0].lower()}"
        print(f"  {sc['label']}: {results[key]}")
    print(f"\nVideos:")
    for i, sc in enumerate(SCENARIOS):
        key = f"step{3+i}_video"
        stats = results.get(f"step{3+i}_stats", {})
        gen = stats.get("gen_time_s", "?")
        print(f"  {sc['label_video']}: {results[key]}  (gen={gen}s)")
    print()
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
