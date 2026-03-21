"""
Generate float32 matmul test cases as binary files.
Runs locally -- no GPU needed.

Usage:
    python kernels/matmul/gen_tests.py
    python kernels/matmul/gen_tests.py --sizes 512x512x512 4096x4096x4096
"""

import argparse
import os
import numpy as np

DEFAULT_SIZES = ["512x512x512", "1024x1024x1024"]


def parse_size(s):
    parts = s.strip().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Size must be MxNxK format, got: {s}")
    return tuple(int(p) for p in parts)


def generate(M, N, K, outdir, rng):
    A = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    # Compute reference in float64 to avoid accumulation drift for large K
    C_ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

    name = f"{M}x{N}x{K}"
    path = os.path.join(outdir, name)
    os.makedirs(path, exist_ok=True)

    A.tofile(os.path.join(path, "A.bin"))
    B.tofile(os.path.join(path, "B.bin"))
    C_ref.tofile(os.path.join(path, "C_ref.bin"))

    with open(os.path.join(path, "meta.txt"), "w") as f:
        f.write(f"{M}\n{N}\n{K}\n")

    print(f"  {name}: A({M}x{K}), B({K}x{N}), C({M}x{N}) — {(A.nbytes + B.nbytes + C_ref.nbytes) / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Generate float32 matmul test cases.")
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES,
                        help="Sizes in MxNxK format (default: 512^3 1024^3 2048^3 4096^3)")
    parser.add_argument("--outdir", default="./testdata", help="Output directory (default: ./testdata)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Generating {len(args.sizes)} test case(s) in '{args.outdir}' (seed={args.seed}):")

    for size_str in args.sizes:
        M, N, K = parse_size(size_str)
        generate(M, N, K, args.outdir, rng)

    print("Done.")


if __name__ == "__main__":
    main()
