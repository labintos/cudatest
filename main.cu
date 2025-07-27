// heavy_matmul.cu  ---  RTX 3070 を本気にさせる行列乗算
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(err) if (err != cudaSuccess){                     \
    printf("CUDA Error %s at %s:%d\n", cudaGetErrorString(err), \
           __FILE__, __LINE__); exit(EXIT_FAILURE); }

const int N = 4096;                 // 行列サイズ (N×N)
const int TILE = 16;                // ブロック 16×16 = 256 thread

__global__ void matmul_tiled(const float* A, const float* B, float* C, int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float val = 0.0f;

    for (int t = 0; t < n / TILE; ++t)
    {
        As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * n + col] = val;
}

int main()
{
    size_t bytes = N * N * sizeof(float);
    printf("Matrix size: %d x %d  (%.1f MB each)\n", N, N, bytes / (1024.0 * 1024));

    // --- host memory
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    // init
    for (int i = 0; i < N * N; ++i) {
        hA[i] = 1.0f;   // 適当
        hB[i] = 0.5f;
    }

    // --- device memory
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));

    // H→D
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    // --- timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_tiled<<<blocks, threads>>>(dA, dB, dC, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double gflop = 2.0 * N * N * N / 1e9;
    printf("GPU time: %.3f ms  →  %.2f GFLOP/s\n", ms, gflop / (ms / 1000));

    // D→H（チェック用に 1 要素だけ）
    CHECK(cudaMemcpy(hC, dC, sizeof(float), cudaMemcpyDeviceToHost));
    printf("C[0] = %.3f  (期待 0.5×%d = %.1f)\n", hC[0], N, 0.5f * N);

    // cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
