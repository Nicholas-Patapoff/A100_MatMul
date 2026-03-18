# A100 MatMul

A test harness for developing and evaluating custom CUDA matrix multiplication kernels on an A100 GPU. Instead of pass/fail, it reports accuracy metrics so you can tune precision as you iterate on your kernel implementations.

## How it works

1. `generate.py` creates float32 test matrices (A, B) and their reference products (C_ref = A @ B, computed in float64 for accuracy) and saves them to `data/`.
2. `harness.cu` loads each test case, calls your `solve()` function, and compares the output against C_ref.
3. You implement `solve()` in `solve.cu` — that's the only file you need to edit.

## Quick start

```bash
# Generate test data (default: 512³, 1024³, 2048³, 4096³)
make generate

# Build
make

# Run
./harness
```

## Writing your kernel

Edit `solve.cu`. The harness calls:

```cpp
void solve(float* A, float* B, float* C, int M, int N, int K);
```

- `A` is M×K, `B` is K×N, `C` is M×N — all row-major device pointers
- Fill `C` with `A @ B` before returning
- Launch whatever kernels you want inside `solve()`

The file ships with a naive baseline (one thread per output element) that you can use as a starting point or reference.

## Output

```
=== Test: M=512 N=512 K=512 ===
  Avg Abs Error : 2.910e-04
  Max Abs Error : 1.831e-03
  RMSE          : 3.724e-04
  Avg Rel Error : 8.901e-05

=== Overall Averages (4 tests) ===
  ...
```

- **Avg Abs Error** — typical element-wise error
- **Max Abs Error** — worst-case error (useful for catching race conditions or tile bugs)
- **RMSE** — like avg abs but penalizes large outliers more
- **Avg Rel Error** — error normalized by magnitude; useful for spotting scaling bugs

## Generating custom sizes

```bash
python3 generate.py --sizes 256x256x256 512x1024x512 --seed 0
```

Then rebuild and run `./harness`.

## Files

| File | Purpose |
|---|---|
| `generate.py` | Generates test matrices and saves to `data/` |
| `solve.h` | Interface declaration — do not modify |
| `solve.cu` | Your kernel implementation goes here |
| `harness.cu` | Test runner — loads data, calls solve, reports metrics |
| `Makefile` | Builds with `nvcc -arch=sm_80` (A100) |
