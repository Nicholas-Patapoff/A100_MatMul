# A100 MatMul

A test harness for developing and benchmarking custom CUDA kernels on a JarvisLabs A100. Each run builds your kernel, checks accuracy against a float64 reference, and captures an Nsight Compute profile — all automated via `run_harness.sh`.

## Structure

```
kernels/
└── matmul/
    ├── solutions/        # your .cu kernel implementations
    ├── harness.cu        # test runner
    ├── solve.h           # interface (do not modify)
    ├── gen_tests.py      # generates test data
    ├── Makefile
    ├── results/          # harness output (committed)
    └── profiles/         # ncu-rep files (committed)
```

## Workflow

```bash
# Run everything (build, test, profile) on the remote A100
./run_harness.sh

# Run a specific kernel
./run_harness.sh --kernel my_kernel

# Run a different problem
./run_harness.sh --problem reduction

# Change the profile size (default: 1024)
./run_harness.sh --size 512
```

`run_harness.sh` will:
1. Resume the JarvisLabs instance
2. `git pull` + build
3. Run the harness across all test sizes
4. Profile with `ncu --set full` at the specified size
5. Download the `.ncu-rep` to `kernels/<problem>/profiles/`
6. Commit results and push
7. Pause the instance

## Writing a kernel

Add a `.cu` file to `kernels/matmul/solutions/`. The harness calls:

```cpp
void solve(float* A, float* B, float* C, int M, int N, int K);
```

- `A` is M×K, `B` is K×N, `C` is M×N — all row-major device pointers
- Fill `C` with `A @ B` before returning

Then run with `--kernel <filename_without_extension>`.

## Output

```
=== Test: M=512 N=512 K=512 ===
  Avg Abs Error : 2.910e-04
  Max Abs Error : 1.831e-03
  RMSE          : 3.724e-04
  Avg Rel Error : 8.901e-05
```

- **Avg Abs Error** — typical element-wise error
- **Max Abs Error** — worst-case error (useful for catching tile/race bugs)
- **RMSE** — like avg abs but penalizes large outliers more
- **Avg Rel Error** — error normalized by magnitude; useful for scaling bugs

## Profiles

`.ncu-rep` files are committed to `kernels/<problem>/profiles/`. After a run, open locally with:

```bash
ncu-ui kernels/matmul/profiles/<timestamp>_<size>.ncu-rep
```

Or pull on another machine and open from there.

## First-time setup

```bash
./setup_instance.sh           # creates A100 instance, clones repo, generates test data, builds
./setup_instance.sh --gpu A100-80GB  # use 80GB variant
```

## Generating test data manually

```bash
python3 kernels/matmul/gen_tests.py
python3 kernels/matmul/gen_tests.py --sizes 256x256x256 512x512x512
```
