#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} a.npy b.npy")
        sys.exit(1)

    path_a = Path(sys.argv[1])
    path_b = Path(sys.argv[2])

    if not path_a.exists():
        print(f"Error: {path_a} does not exist")
        sys.exit(1)
    if not path_b.exists():
        print(f"Error: {path_b} does not exist")
        sys.exit(1)

    a = np.load(path_a)
    b = np.load(path_b)

    print(f"A: {path_a}, shape={a.shape}, dtype={a.dtype}")
    print(f"B: {path_b}, shape={b.shape}, dtype={b.dtype}")

    if a.shape != b.shape:
        print("RESULT: shapes differ")
        return

    if a.dtype != b.dtype:
        print("Warning: dtypes differ")

    equal = np.array_equal(a, b)
    if equal:
        print("RESULT: arrays are exactly equal")
    else:
        # Just count how many elements differ
        diff_mask = a != b
        num_diff = int(np.count_nonzero(diff_mask))
        total = a.size
        print("RESULT: arrays differ")
        print(f"  differing elements: {num_diff} / {total}")

if __name__ == "__main__":
    main()
