---
marp: true
title: CUDA環境の作り方
---

# CUDA環境の作り方

<div style="text-align:right">2211034  鈴木 智</div>

---

# CUDAとは
CUDAは、NVIDIAが開発したGPUを使用して並列計算を行うためのプラットフォームおよびAPIで、CUDAを使用することで、GPUの計算能力を利用して高速な処理を実現できます。特に、深層学習や科学技術計算など、大量のデータを扱うアプリケーションで威力を発揮する。

- [GPUとCUDAを一瞬で理解する！](https://qiita.com/microstone/items/34380357908784d1e183)

---
# Visual Studioのインストール
Visual Studioは、Microsoftが提供する統合開発環境で、CUDAの開発に必要なコンパイラやツールが含まれています。
- インストール時には「C++によるデスクトップ開発」にチェックを入れ、オプションで「MSVC v143 - VS 2022 C++ x64/x86 ビルド ツール（最新）」が選択されていることを確認する。

---

# CUDAのインストール
CUDAは、NVIDIAの公式サイトからダウンロードできます。インストール手順は以下の通りです。
- NVIDIAの[公式サイト](https://developer.nvidia.com/cuda-toolkit)にアクセスし、CUDA Toolkitをダウンロードします。
    * Download nowをクリック
    * windows / x86_64 / 11 / exe(local)を選択
    * ダウンロードしたexeファイルを実行し、インストールします。
今回ダウンロードするバージョンはCUDA 12.9です。

---

# cuDNNのインストール
cuDNNは、NVIDIAが提供する深層学習向けのGPUアクセラレーションライブラリで、CUDAと組み合わせて使用します。
- cuDNNは、[公式サイト](https://developer.nvidia.com/cuDNN)からダウンロードできます。
    * まず、NVIDIA Developer Programに登録する必要があります。(メールアドレス登録だけでOK)
    * 登録が完了したら、cuDNNの公式サイトのDownload cuDNN libraryをクリック
    * windows / x86_64 / Tarball / 12を選択
    * ダウンロードしたzipファイルを解凍し、bin、include、libの3つのフォルダをCUDA Toolkitのインストール先にコピーします。コピー先は何も設定していなければ、C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9になります。

---
インストールが完了したらcommand promptを開き、以下のコマンドを実行してCUDAとcuDNNが正しくインストールされているか確認します。
```bash
nvcc --version
```
- 正常にインストールされていれば、CUDAのバージョン情報が表示されます。
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Apr__9_19:29:17_Pacific_Daylight_Time_2025
Cuda compilation tools, release 12.9, V12.9.41
Build cuda_12.9.r12.9/compiler.35813241_0
```
cuDNNのインストール確認は以下のコマンドを実行します。
```bash
where cudnn64_9.dll
```
- 正常にインストールされていれば、cuDNNのパスが表示されます。
```bash
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cudnn64_9.dll
```

これで完了です。

---
# CUDAプログラムを実行してみる
windowsの検索でDeveloper command prompt for VS2022を開き、以下のコマンドを実行します。
自分のCUDAコードを実行したいフォルダPathをコピーして、cdコマンドで移動します。
```bash
cd cuda_code_path
```
次に、以下のコマンドでvscodeを開きます。
```bash
code .
```
vscodeで新しいファイルを作成。名前は適当に設定します。拡張子は.cuにします。
```cu
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
```

- 以下のコマンドでコンパイルします。
```bash
nvcc -arch=sm_86 main.cu -o main
```
- 実行します。
```bash
main main.cu
```
- 実行結果
```bash
The system cannot find the path specified.

c:\Users\sato2\Documents\大学院資料\cudatest>main main.cu
Matrix size: 4096 x 4096  (64.0 MB each)
GPU time: 83.639 ms  ↁE 1643.25 GFLOP/s
C[0] = 2048.000  (期征E0.5ÁEd = 0.0)
```

文字化けしている部分もありますが、実行できているので問題ありません。

---

# 僕が手こずったところ
- Visual Studioのインストール時に「C++によるデスクトップ開発」にチェックを入れなかったため、CUDAのインストールができなかった。
- Visual Studioの中の古いバージョンの個別コンポーネントが残っていたため、CUDAのインストールができなかった。
- cuDNNをずっとVersion選択で10を設定していたためwindows11にインストールできなかった。

