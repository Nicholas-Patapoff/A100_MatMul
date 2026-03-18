#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <dirent.h>

#include <cuda_runtime.h>
#include "solve.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

struct TestCase {
    std::string path;
    int M, N, K;
};

struct Metrics {
    double avg_abs;
    double max_abs;
    double rmse;
    double avg_rel;
};

// Load a binary float32 file into a newly malloc'd buffer.
// Returns nullptr and prints a message on failure.
static float* load_bin(const std::string& path, long expected_floats) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "  [skip] Cannot open %s\n", path.c_str());
        return nullptr;
    }
    float* buf = (float*)malloc(expected_floats * sizeof(float));
    if (!buf) { fclose(f); return nullptr; }
    long n = (long)fread(buf, sizeof(float), expected_floats, f);
    fclose(f);
    if (n != expected_floats) {
        fprintf(stderr, "  [skip] %s: expected %ld floats, got %ld\n",
                path.c_str(), expected_floats, n);
        free(buf);
        return nullptr;
    }
    return buf;
}

static Metrics compute_metrics(const float* C, const float* C_ref, long n) {
    double sum_abs = 0.0, max_abs = 0.0, sum_sq = 0.0, sum_rel = 0.0;
    for (long i = 0; i < n; i++) {
        double diff = fabs((double)C[i] - (double)C_ref[i]);
        double ref  = fabs((double)C_ref[i]);
        sum_abs += diff;
        sum_sq  += diff * diff;
        sum_rel += diff / (ref + 1e-6);
        if (diff > max_abs) max_abs = diff;
    }
    return {
        sum_abs / n,
        max_abs,
        sqrt(sum_sq / n),
        sum_rel / n
    };
}

static std::vector<TestCase> discover_tests(const std::string& datadir) {
    std::vector<TestCase> cases;
    DIR* dir = opendir(datadir.c_str());
    if (!dir) {
        fprintf(stderr, "Cannot open data directory: %s\n", datadir.c_str());
        return cases;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        std::string subdir = datadir + "/" + entry->d_name;
        std::string meta   = subdir + "/meta.txt";
        FILE* f = fopen(meta.c_str(), "r");
        if (!f) continue;
        int M, N, K;
        if (fscanf(f, "%d\n%d\n%d\n", &M, &N, &K) != 3) { fclose(f); continue; }
        fclose(f);
        cases.push_back({subdir, M, N, K});
    }
    closedir(dir);
    // Sort by total element count (small to large)
    std::sort(cases.begin(), cases.end(), [](const TestCase& a, const TestCase& b) {
        return (long)a.M * a.N < (long)b.M * b.N;
    });
    return cases;
}

int main(int argc, char** argv) {
    std::string datadir = (argc > 1) ? argv[1] : "./data";

    auto cases = discover_tests(datadir);
    if (cases.empty()) {
        fprintf(stderr, "No test cases found in '%s'. Run: python3 generate.py\n", datadir.c_str());
        return 1;
    }

    double total_avg_abs = 0.0, total_max_abs = 0.0, total_rmse = 0.0, total_avg_rel = 0.0;
    int passed = 0;

    for (const auto& tc : cases) {
        printf("\n=== Test: M=%d N=%d K=%d ===\n", tc.M, tc.N, tc.K);

        long sizeA = (long)tc.M * tc.K;
        long sizeB = (long)tc.K * tc.N;
        long sizeC = (long)tc.M * tc.N;

        float* h_A    = load_bin(tc.path + "/A.bin",     sizeA);
        float* h_B    = load_bin(tc.path + "/B.bin",     sizeB);
        float* h_Cref = load_bin(tc.path + "/C_ref.bin", sizeC);
        if (!h_A || !h_B || !h_Cref) {
            free(h_A); free(h_B); free(h_Cref);
            continue;
        }

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, sizeC * sizeof(float)));

        solve(d_A, d_B, d_C, tc.M, tc.N, tc.K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_C = (float*)malloc(sizeC * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

        Metrics m = compute_metrics(h_C, h_Cref, sizeC);
        printf("  Avg Abs Error : %.6e\n", m.avg_abs);
        printf("  Max Abs Error : %.6e\n", m.max_abs);
        printf("  RMSE          : %.6e\n", m.rmse);
        printf("  Avg Rel Error : %.6e\n", m.avg_rel);

        total_avg_abs += m.avg_abs;
        total_max_abs  = fmax(total_max_abs, m.max_abs);
        total_rmse    += m.rmse;
        total_avg_rel += m.avg_rel;
        passed++;

        free(h_A); free(h_B); free(h_Cref); free(h_C);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    if (passed > 1) {
        printf("\n=== Overall Averages (%d tests) ===\n", passed);
        printf("  Avg Abs Error : %.6e\n", total_avg_abs / passed);
        printf("  Max Abs Error : %.6e\n", total_max_abs);
        printf("  RMSE          : %.6e\n", total_rmse    / passed);
        printf("  Avg Rel Error : %.6e\n", total_avg_rel / passed);
    }
    printf("\n");
    return 0;
}
