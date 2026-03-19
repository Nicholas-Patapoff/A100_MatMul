#include "solve.h"
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda::wmma;

#define TILE 8
__global__ void matrix_multiplication_kernel(float *A, float *B, float *C, int M, int N,
                                             int K)
{

    // tile assignment from C
    int Ccol = blockIdx.x * 16;
    int Crow = blockIdx.y * 16;

    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
    fragment<accumulator, 16, 16, 8, float> c_frag;
    fill_fragment(c_frag, 0.0f);
    // move the tiles and track A and B tiles (starting point)
    for (int i = 0; i < N; i += TILE)
    {
        int Arow = Crow;
        int Acol = i;
        int Brow = i;
        int Bcol = Ccol;

        load_matrix_sync(a_frag, A + Arow * N + Acol, N);
        load_matrix_sync(b_frag, B + Brow * K + Bcol, K);
        for (int t = 0; t < a_frag.num_elements; t++)
            a_frag.x[t] = __float_to_tf32(a_frag.x[t]);
        for (int t = 0; t < b_frag.num_elements; t++)
            b_frag.x[t] = __float_to_tf32(b_frag.x[t]);

        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    store_matrix_sync(C + Crow * K + Ccol, c_frag, K, mem_row_major);
}

__global__ void pad_matrix(float *src, float *dst, int srcRows, int srcCols, int dstCols)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row >= srcRows || col >= dstCols)
        return;

    float val = 0.0f;
    if (col < srcCols)
        val = src[row * srcCols + col];

    dst[row * dstCols + col] = (val);
}

__global__ void strip_matrix(float *src, float *dst, int dstRows, int dstCols, int srcCols)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row >= dstRows || col >= dstCols)
        return;

    dst[row * dstCols + col] = src[row * srcCols + col];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(float *A, float *B, float *C, int M, int N, int K)
{
    int Mp = ((M + 15) / 16) * 16;
    int Np = ((N + 15) / 16) * 16;
    int Kp = ((K + 15) / 16) * 16;

    float *Ap, *Bp;
    float *Cp;
    dim3 block(16, 16);

    cudaMalloc(&Ap, Mp * Np * sizeof(float));
    cudaMemset(Ap, 0, Mp * Np * sizeof(float));

    cudaMalloc(&Bp, Np * Kp * sizeof(float));
    cudaMemset(Bp, 0, Np * Kp * sizeof(float));

    cudaMalloc(&Cp, Mp * Kp * sizeof(float));
    cudaMemset(Cp, 0, Mp * Kp * sizeof(float));

    // A is [M x N], padded to [Mp x Np]
    dim3 gridA((Np) / 16, (Mp) / 16);
    pad_matrix<<<gridA, block>>>(A, Ap, M, N, Np);

    // B is [N x K], padded to [Np x Kp]
    dim3 gridB((Kp) / 16, (Np) / 16);
    pad_matrix<<<gridB, block>>>(B, Bp, N, K, Kp);

    cudaDeviceSynchronize();

    dim3 threadsPerBlock(32, 1);
    dim3 grid(Kp / 16, Mp / 16);
    matrix_multiplication_kernel<<<grid, threadsPerBlock>>>((float *)Ap, (float *)Bp, Cp, Mp, Np, Kp);

    dim3 gridC(Kp / 16, Mp / 16);
    strip_matrix<<<gridC, block>>>(Cp, C, M, K, Kp);

    cudaDeviceSynchronize();
    cudaFree(&Cp);
    cudaFree(&Bp);
    cudaFree(&Ap);
}
