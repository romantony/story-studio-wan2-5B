"""
setup_volume_fp8.py — Download WAN2.2 TI2V-5B FP8 (ModelOpt) model to a RunPod network volume.

Model: shunyang90/Wan2.2-TI2V-5B-ModelOpt-FP8  (~19.3 GB)
Quantized with NVIDIA ModelOpt — native FP8 compute on Ada/Hopper/Blackwell GPUs.

Run this INSIDE a RunPod pod (dev/SSH) with the network volume attached.
Safe to re-run — skips already-present files.

Usage:
    python3 setup_volume_fp8.py
    python3 setup_volume_fp8.py --dst /workspace/models/wan22-ti2v-5b-fp8
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

# Auto-install huggingface_hub if missing
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[SETUP] Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import snapshot_download

REPO_ID   = "shunyang90/Wan2.2-TI2V-5B-ModelOpt-FP8"
SIZE_HINT = 20  # GB


def detect_volume() -> Path:
    env = os.environ.get("RUNPOD_VOLUME_PATH")
    if env:
        return Path(env)
    for candidate in ("/workspace", "/runpod-volume"):
        p = Path(candidate)
        if p.exists() and os.path.ismount(candidate):
            return p
    return Path("/workspace")


def free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=None,
                    help="Destination directory (default: <volume>/models/wan22-ti2v-5b-fp8)")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="HuggingFace token if repo requires auth (optional for this model)")
    args = ap.parse_args()

    volume = detect_volume()
    dst    = Path(args.dst) if args.dst else volume / "models" / "wan22-ti2v-5b-fp8"

    print(f"Volume : {volume}  (free: {free_gb(volume):.1f} GB)")
    print(f"Model  : {REPO_ID}")
    print(f"Dest   : {dst}")
    print(f"Size   : ~{SIZE_HINT} GB\n")

    # Check if already downloaded
    marker = dst / "model_index.json"
    if marker.exists():
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        print(f"[SKIP] Already present ({size/1e9:.1f} GB on disk)")
        print(f"       Delete {dst} to re-download.")
        return

    # Check free space
    free = free_gb(volume)
    if free < SIZE_HINT * 1.1:
        print(f"[WARN] Only {free:.1f} GB free, need ~{SIZE_HINT} GB. Proceeding anyway...")
    else:
        print(f"[OK]   {free:.1f} GB free — sufficient")

    dst.mkdir(parents=True, exist_ok=True)

    print(f"\n[DL]  Downloading {REPO_ID} ...")
    print(f"      (~{SIZE_HINT} GB, ~5-10 min on a fast link)\n")
    t0 = time.time()

    snapshot_download(
        repo_id            = REPO_ID,
        local_dir          = str(dst),
        local_dir_use_symlinks = False,
        resume_download    = True,
        token              = args.hf_token,
    )

    elapsed = round(time.time() - t0, 1)
    size    = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"\n[OK]  Downloaded {size/1e9:.1f} GB in {elapsed}s")

    # Show what we got
    print("\nContents:")
    for item in sorted(dst.iterdir()):
        if item.is_dir():
            dir_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            print(f"  {item.name}/  ({dir_size/1e9:.2f} GB)")
        else:
            print(f"  {item.name}  ({item.stat().st_size/1e6:.1f} MB)")

    print(f"\nFree space remaining: {free_gb(volume):.1f} GB")
    print(f"\n[DONE] FP8 model ready at: {dst}")
    print("       Set WAN_FP8_MODEL env var to this path in your serverless endpoint.")


if __name__ == "__main__":
    main()
