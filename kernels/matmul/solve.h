#pragma once

// Implement this function in solutions/<kernel>.cu.
// A is M x K, B is K x N, C is M x N — all row-major, device pointers.
// Fill C with A @ B before returning.
void solve(float* A, float* B, float* C, int M, int N, int K);
