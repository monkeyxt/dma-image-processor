#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_folder> [output_folder]")
        sys.exit(1)

    # Required: input folder
    image_folder = sys.argv[1]

    if not os.path.isdir(image_folder):
        print(f"Error: {image_folder} is not a directory")
        sys.exit(1)

    # Optional: output folder (default: <input_folder>/png_out)
    if len(sys.argv) >= 3:
        output_folder = sys.argv[2]
    else:
        output_folder = os.path.join(image_folder, "png_out")

    os.makedirs(output_folder, exist_ok=True)

    # List all .npy files in the directory
    npy_files = [f for f in os.listdir(image_folder) if f.endswith(".npy")]
    if not npy_files:
        print(f"No .npy files found in {image_folder}")
        sys.exit(0)

    print(f"Found {len(npy_files)} .npy files in {image_folder}")
    print(f"Saving PNGs to {output_folder}")

    for npy_file in npy_files:
        in_path = os.path.join(image_folder, npy_file)
        out_name = os.path.splitext(npy_file)[0] + ".png"
        out_path = os.path.join(output_folder, out_name)

        img = np.load(in_path)
        plt.imsave(out_path, img, cmap="gray")
        print(f"  {npy_file} -> {out_path}")

if __name__ == "__main__":
    main()
