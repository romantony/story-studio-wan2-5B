"""
setup_volume.py — Download WAN2.2 TI2V-5B model to a RunPod network volume.

Run this INSIDE a temporary RunPod pod that has your volume mounted at /runpod-volume.
Safe to re-run — skips already-present files.

Usage:
    python3 setup_volume.py       # download everything (~20 GB, needs 22+ GB free)
    python3 setup_volume.py wan   # WAN2.2 TI2V-5B model only

Model size breakdown:
    wan22-ti2v-5b:
      diffusion_pytorch_model-00001-of-00003.safetensors  ~5 GB
      diffusion_pytorch_model-00002-of-00003.safetensors  ~5 GB
      diffusion_pytorch_model-00003-of-00003.safetensors  ~5 GB
      Wan2.2_VAE.pth                                      ~1 GB
      models_t5_umt5-xxl-enc-bf16.pth                     ~5 GB
      google/umt5-xxl/  (tokenizer)                       ~1 GB
    Total: ~20 GB
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Auto-install huggingface_hub if missing (bare RunPod pods often lack it)
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    print("[SETUP] Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    print("[SETUP] huggingface_hub installed.")

VOLUME    = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
MODEL_DIR = VOLUME / "models"

MODELS = {
    "wan": (
        "Wan-AI/Wan2.2-TI2V-5B",
        MODEL_DIR / "wan22-ti2v-5b",
        "config.json",
        20,
    ),
}


def free_gb():
    st = os.statvfs(VOLUME)
    return (st.f_bavail * st.f_frsize) / 1e9


def download(key: str):
    repo_id, local_path, marker, size_hint = MODELS[key]
    if (local_path / marker).exists():
        size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
        print(f"  [SKIP] {local_path.name} already present ({size/1e9:.1f} GB on disk)")
        return

    free = free_gb()
    if free < size_hint * 1.1:
        print(f"  [WARN] Only {free:.1f} GB free, need ~{size_hint} GB for {key}. Proceeding anyway...")

    from huggingface_hub import snapshot_download
    print(f"  [DL]  {repo_id}")
    print(f"        → {local_path}  (~{size_hint} GB, ~10 min on fast link)")
    local_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    elapsed = round(time.time() - t0, 1)
    total = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
    print(f"  [OK]  {local_path.name} — {total/1e9:.1f} GB in {elapsed}s")


if __name__ == "__main__":
    targets = [a.lower() for a in sys.argv[1:]] if len(sys.argv) > 1 else list(MODELS.keys())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Volume : {VOLUME}  (free: {free_gb():.1f} GB)")
    print(f"Models : {MODEL_DIR}")
    print(f"Targets: {targets}\n")

    for key in targets:
        if key not in MODELS:
            print(f"[ERROR] Unknown target '{key}'. Valid: {list(MODELS.keys())}")
            sys.exit(1)
        print(f"\n── {key.upper()} ─────────────────────────────────────────────")
        download(key)

    print(f"\n[DONE] Models on volume:")
    for p in sorted(MODEL_DIR.iterdir()):
        if p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            print(f"  {p.name:30s}  {size/1e9:.1f} GB")
    print(f"\nFree space remaining: {free_gb():.1f} GB")
    print("\nReady. Create a WAN2.2 endpoint pointing to this volume.")
