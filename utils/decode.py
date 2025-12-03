#!/usr/bin/env python3
"""
decode_bin.py

Decode a frames_XXXXXX.bin file produced by the archiver into NumPy arrays.

Assumptions:
  - frames are stored back-to-back, no headers
  - each frame is HEIGHT x WIDTH
  - dtype is uint16 (little-endian) by default
  - Save each frame as frames/frame_<index>.npy
  - Also save a stacked array frames.npy with shape (N, H, W)

You can turn off either with flags.
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Decode raw .bin frames into NumPy arrays."
    )
    ap.add_argument("bin_file", help="Path to frames_XXXXXX.bin")
    ap.add_argument("--height", type=int, required=True, help="Image height (pixels)")
    ap.add_argument("--width", type=int, required=True, help="Image width (pixels)")
    ap.add_argument(
        "--dtype",
        default="<u2",
        help="Numpy dtype (default '<u2' = little-endian uint16)",
    )
    ap.add_argument(
        "--out-npy",
        default="frames.npy",
        help="Output .npy file for stacked frames (default: frames.npy)",
    )
    ap.add_argument(
        "--no-per-frame",
        action="store_true",
        help="Do NOT save each frame as frames/frame_<index>.npy",
    )
    ap.add_argument(
        "--no-stack",
        action="store_true",
        help="Do NOT save stacked frames.npy",
    )
    args = ap.parse_args()

    bin_path = Path(args.bin_file)
    if not bin_path.exists():
        raise SystemExit(f"{bin_path} does not exist")

    # Read all data as flat array
    dtype = np.dtype(args.dtype)
    flat = np.fromfile(bin_path, dtype=dtype)

    H, W = args.height, args.width
    pixels_per_frame = H * W
    if flat.size % pixels_per_frame != 0:
        raise SystemExit(
            f"File size ({flat.size} pixels) is not a multiple of frame size ({pixels_per_frame} pixels)"
        )

    num_frames = flat.size // pixels_per_frame
    print(f"[INFO] {bin_path}")
    print(f"       dtype   : {dtype}")
    print(f"       shape   : ({num_frames}, {H}, {W})")
    print(f"       frames  : {num_frames}")

    frames = flat.reshape(num_frames, H, W)

    # Save stacked array unless disabled
    if not args.no_stack:
        out_path = Path(args.out_npy)
        if not out_path.is_absolute():
            out_path = bin_path.with_name(args.out_npy)
        np.save(out_path, frames)
        print(f"[INFO] Saved stacked frames to {out_path}")

    # Default: save per-frame npy (unless --no-per-frame)
    if not args.no_per_frame:
        frames_dir = bin_path.parent / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving per-frame .npy files under {frames_dir} ...")
        for i in range(num_frames):
            np.save(frames_dir / f"frame_{i:06d}.npy", frames[i])
        print("[INFO] Per-frame .npy export done.")


if __name__ == "__main__":
    main()
