/**
 * Based on https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuFFT/3d_c2c
 */ 
#include <array>
#include <fftw3.h>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <stdexcept>
#include <LibSL/LibSL.h>
#include "indexOps.h"
#include "constants.h"
#include "types.h"
#include "error.h"
#include "voxelGrid.h"

using namespace std;

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL

// cufft API error checking
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL

__global__
void scaling_kernel(cufftComplex* data, int element_count, float scale) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;
  for (auto i = tid; i < element_count; i += stride) {
    data[i].x *= scale;
    data[i].y *= scale;
  }
}

__global__
void element_wise_cmplx_scalar_mul (cufftComplex *A, double scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  if (i < n) {
    A[i].x *= scalar;
    A[i].y *= scalar;
  }
}

__global__ 
void element_wise_cmplx_mul (cufftComplex *A, cufftComplex *B, cufftComplex *C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  if (i < n) {
    double real = A[i].x * B[i].x - A[i].y * B[i].y;
    double imag = A[i].x * B[i].y + A[i].y * B[i].x;
    C[i].x = real;
    C[i].y = imag; 
  }
}

__global__ 
void element_wise_round (cufftComplex *A, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  if (i < n) {
    A[i].x = round(A[i].x);
    A[i].y = round(A[i].y); 
  }
}

__global__ 
void extract_real_kernel(cufftComplex *A, int *B, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  if (i < n) 
    B[i] = (int) A[i].x;
}

__global__ 
void padding_kernel (cufftComplex * oA, cufftComplex *pA, int ox, int oy, int oz, int px, int py, int pz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < (ox * oy * oz)) {
    int k = tid % oz; 
    int j = ((tid - k) / oz) % oy; 
    int i = (((tid - k) / oz) - j) / oy;
    pA[(i * py + j) * pz + k].x = oA[tid].x;
    pA[(i * py + j) * pz + k].y = oA[tid].y;
  }
} 

void copyTo (fftw_complex *A, ComplexList &B, Index3 sz) { 
  LL volume = vol(sz); 
  for (LL i = 0; i < volume; i++) 
    B.push_back(Complex(A[i][0], A[i][1])); 
}

void copyFro (ComplexList &B, fftw_complex *A, Index3 sz) { 
  LL volume = vol(sz); 
  for (LL i = 0; i < volume; i++) {
    A[i][0] = B[i].real();
    A[i][1] = B[i].imag(); 
  }
}

void to_cufft_complex (const VoxelGrid &A, cufftComplex *B, Index3 size) {
  int cnt = 0;
  FOR_VOXEL(i, j, k, size) {
    B[cnt++] = make_float2(A[i][j][k], 0.0);
  }
}

// Fast version using FlatVoxelGrid - single pass instead of triple loop
void flat_to_cufft_complex(const FlatVoxelGrid &A, cufftComplex *B) {
  const int* src = A.ptr();
  size_t n = A.size();
  for (size_t i = 0; i < n; i++) {
    B[i] = make_float2(static_cast<float>(src[i]), 0.0f);
  }
}

void to_voxel_grid (const int *res, VoxelGrid &target, Index3 size) {
  auto [N, M, L] = size;
  resize3d(target, size, 0);
  int cnt = 0;
  FOR_VOXEL(i, j, k, size) {
    target[i][j][k] = res[cnt++];
  }
}

// Fast version - direct memcpy to FlatVoxelGrid
void to_flat_grid(const int *res, FlatVoxelGrid &target, Index3 size) {
  target.resize(size);
  std::memcpy(target.ptr(), res, target.size_bytes());
}

void pad_voxel_grid_cuda (cufftComplex *&d_voxel_grid, Index3 orig_size, Index3 padded_size) { 
  LL padded_volume = vol(padded_size);

  cufftComplex * d_padded_voxel_grid = nullptr; 

  CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_padded_voxel_grid), sizeof(cufftComplex) * padded_volume)); 
  CUDA_RT_CALL(cudaMemset(d_padded_voxel_grid, 0, sizeof(cufftComplex) * padded_volume));

  auto [ox, oy, oz] = orig_size;
  auto [px, py, pz] = padded_size;

  LL blockSize = 256;
  LL numBlocks = (vol(orig_size) + blockSize - 1) / blockSize;

  padding_kernel<<<numBlocks, blockSize>>>(d_voxel_grid, d_padded_voxel_grid, ox, oy, oz, px, py, pz); 

  CUDA_RT_CALL(cudaFree(d_voxel_grid));
  d_voxel_grid = d_padded_voxel_grid;
}

#if USE_PARALLEL_FFT3D 
void dft_conv3(const VoxelGrid &a, const VoxelGrid &b, VoxelGrid &result) {
  auto [N, M, L] = get_size(a);

  if (!same_size(get_size(a), get_size(b)))
    throw std::runtime_error("Input grids must be of the same size for convolution");

  Index3 padded_size = make_tuple(2 * N + 1, 2 * M + 1, 2 * L + 1);
  auto [nx, ny, nz] = padded_size;
  LL padded_volume = vol(padded_size); 

  LL init_volume = vol(get_size(a));
  cufftComplex *h_A, *h_B; // , *h_out_A; 
  int *h_out; 
  {
    Timer tm("(dft_conv3): Copying as is to cufftComplex");
    h_A = (cufftComplex *) malloc(sizeof(cufftComplex) * init_volume);
    h_B = (cufftComplex *) malloc(sizeof(cufftComplex) * init_volume);
    h_out = (int *) malloc(sizeof(int) * padded_volume);
    to_cufft_complex(a, h_A, get_size(a));
    to_cufft_complex(b, h_B, get_size(a));
  }

  cufftComplex *d_A = nullptr, *d_B = nullptr; 
  int * d_real_part = nullptr;
  cufftHandle plan; 

  { 
    Timer tm("(dft_conv3): CUDA operations");
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(cufftComplex) * init_volume)); 
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(cufftComplex) * init_volume)); 
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_real_part), sizeof(int) * padded_volume)); 

    CUDA_RT_CALL(cudaMemcpy(d_A, h_A, sizeof(cufftComplex) * init_volume, cudaMemcpyHostToDevice)); 
    CUDA_RT_CALL(cudaMemcpy(d_B, h_B, sizeof(cufftComplex) * init_volume, cudaMemcpyHostToDevice)); 

    pad_voxel_grid_cuda(d_A, get_size(a), padded_size); 
    pad_voxel_grid_cuda(d_B, get_size(b), padded_size); 

    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C)); 

    CUFFT_CALL(cufftExecC2C(plan, d_A, d_A, CUFFT_FORWARD)); 
    CUFFT_CALL(cufftExecC2C(plan, d_B, d_B, CUFFT_FORWARD)); 

    LL blockSize = 256;
    LL numBlocks = (padded_volume + blockSize - 1) / blockSize;

    element_wise_cmplx_mul<<<numBlocks, blockSize>>>(d_A, d_B, d_A, padded_volume); 

    CUFFT_CALL(cufftExecC2C(plan, d_A, d_A, CUFFT_INVERSE)); 

    double scalar = 1.0 / ((double) padded_volume);
    element_wise_cmplx_scalar_mul<<<numBlocks, blockSize>>>(d_A, scalar, padded_volume); 
    element_wise_round<<<numBlocks, blockSize>>>(d_A, padded_volume); 
    extract_real_kernel<<<numBlocks, blockSize>>>(d_A, d_real_part, padded_volume); 

    cudaMemcpy(h_out, d_real_part, sizeof(int) * padded_volume, cudaMemcpyDeviceToHost);
  }

  {
    Timer tm("(dft_conv3): Copying stuff back to result"); 
    {
      Timer a("(dft_conv3): to_voxel_grid");
      to_voxel_grid(h_out, result, padded_size); 
    }
    truncateto3d(result, get_size(a)); 
  }

  CUDA_RT_CALL(cudaFree(d_A))
  CUDA_RT_CALL(cudaFree(d_B))
  CUDA_RT_CALL(cudaFree(d_real_part))
  CUFFT_CALL(cufftDestroy(plan));
  // Note: cudaDeviceReset() removed - it was resetting the entire CUDA context
  // which is unnecessary and kills performance when called repeatedly
  free(h_A);
  free(h_B);
  free(h_out);
}

// Fast version using FlatVoxelGrid - avoids triple-loop conversions
void dft_conv3_flat(const FlatVoxelGrid &a, const FlatVoxelGrid &b, FlatVoxelGrid &result) {
  int N = a.nx, M = a.ny, L = a.nz;

  if (a.nx != b.nx || a.ny != b.ny || a.nz != b.nz)
    throw std::runtime_error("Input grids must be of the same size for convolution");

  Index3 padded_size = make_tuple(2 * N + 1, 2 * M + 1, 2 * L + 1);
  auto [nx, ny, nz] = padded_size;
  LL padded_volume = vol(padded_size);
  LL init_volume = a.size();

  cufftComplex *h_A, *h_B;
  int *h_out;
  {
    h_A = (cufftComplex *) malloc(sizeof(cufftComplex) * init_volume);
    h_B = (cufftComplex *) malloc(sizeof(cufftComplex) * init_volume);
    h_out = (int *) malloc(sizeof(int) * padded_volume);
    // Fast conversion - single loop instead of triple
    flat_to_cufft_complex(a, h_A);
    flat_to_cufft_complex(b, h_B);
  }

  cufftComplex *d_A = nullptr, *d_B = nullptr;
  int *d_real_part = nullptr;
  cufftHandle plan;

  {
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(cufftComplex) * init_volume));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(cufftComplex) * init_volume));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_real_part), sizeof(int) * padded_volume));

    CUDA_RT_CALL(cudaMemcpy(d_A, h_A, sizeof(cufftComplex) * init_volume, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_B, h_B, sizeof(cufftComplex) * init_volume, cudaMemcpyHostToDevice));

    pad_voxel_grid_cuda(d_A, a.dims(), padded_size);
    pad_voxel_grid_cuda(d_B, b.dims(), padded_size);

    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C));

    CUFFT_CALL(cufftExecC2C(plan, d_A, d_A, CUFFT_FORWARD));
    CUFFT_CALL(cufftExecC2C(plan, d_B, d_B, CUFFT_FORWARD));

    LL blockSize = 256;
    LL numBlocks = (padded_volume + blockSize - 1) / blockSize;

    element_wise_cmplx_mul<<<numBlocks, blockSize>>>(d_A, d_B, d_A, padded_volume);

    CUFFT_CALL(cufftExecC2C(plan, d_A, d_A, CUFFT_INVERSE));

    double scalar = 1.0 / ((double) padded_volume);
    element_wise_cmplx_scalar_mul<<<numBlocks, blockSize>>>(d_A, scalar, padded_volume);
    element_wise_round<<<numBlocks, blockSize>>>(d_A, padded_volume);
    extract_real_kernel<<<numBlocks, blockSize>>>(d_A, d_real_part, padded_volume);

    cudaMemcpy(h_out, d_real_part, sizeof(int) * padded_volume, cudaMemcpyDeviceToHost);
  }

  {
    // Fast conversion - memcpy instead of triple loop
    to_flat_grid(h_out, result, padded_size);
    // Truncate to original size (only keep the valid portion)
    // For flat grid, we need to copy row by row
    FlatVoxelGrid truncated(N, M, L);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        // Copy one row at a time
        std::memcpy(&truncated(i, j, 0), &result(i, j, 0), L * sizeof(int));
      }
    }
    result = std::move(truncated);
  }

  CUDA_RT_CALL(cudaFree(d_A));
  CUDA_RT_CALL(cudaFree(d_B));
  CUDA_RT_CALL(cudaFree(d_real_part));
  CUFFT_CALL(cufftDestroy(plan));
  free(h_A);
  free(h_B);
  free(h_out);
}

// Fast cross-correlation using FlatVoxelGrid
void dft_corr3_flat(const FlatVoxelGrid &a, const FlatVoxelGrid &b, FlatVoxelGrid &result) {
  FlatVoxelGrid flipped_a = a;
  flip3d_flat(flipped_a);
  dft_conv3_flat(flipped_a, b, result);
  flip3d_flat(result);
}

void fft3d (fftw_complex *in, fftw_complex *out, Index3 size, bool inverse) {
  cufftHandle plan; 
  cudaStream_t stream = NULL; 

  auto [N, M, L] = size;
  LL volume = vol(size); 

  ComplexList data; 
  copyTo(in, data, size); 

  cufftComplex *d_data = nullptr;

  CUFFT_CALL(cufftCreate(&plan));
  CUFFT_CALL(cufftPlan3d(&plan, N, M, L, CUFFT_C2C)); 

  CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUFFT_CALL(cufftSetStream(plan, stream));

  CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(Complex) * data.size()));
  CUDA_RT_CALL(cudaMemcpyAsync(d_data, data.data(), sizeof(Complex) * data.size(), cudaMemcpyHostToDevice, stream));

  if (!inverse) {
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUDA_RT_CALL(cudaMemcpyAsync(data.data(), d_data, sizeof(Complex) * data.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
  } else {
    LL blockSize = 256;
    LL numBlocks = (volume + blockSize - 1) / blockSize;
    scaling_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, data.size(), 1.f / volume);
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    CUDA_RT_CALL(cudaMemcpyAsync(data.data(), d_data, sizeof(Complex) * data.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
  }
  copyFro(data, out, size);
  CUDA_RT_CALL(cudaFree(d_data))
  CUFFT_CALL(cufftDestroy(plan));
  CUDA_RT_CALL(cudaStreamDestroy(stream));
  // Note: cudaDeviceReset() removed - unnecessary context reset
} 

__global__ 
void sweep_z (int * grid, int N, int L, int M) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N * L) { 
    int j = tid % L; 
    int i = (tid - j) / L;
    for (int k = M - 2; k >= 0; k--) {
      int idx = (i * L + j) * M + k;
      int idx_n = (i * L + j) * M + k + 1;
      grid[idx] = min(grid[idx], grid[idx_n] + 1);
    }
    for (int k = 1; k < M; k++) {
      int idx = (i * L + j) * M + k;
      int idx_n = (i * L + j) * M + k - 1;
      grid[idx] = min(grid[idx], grid[idx_n] + 1);
    }
  }
}

__global__
void sweep_y (int * grid, int N, int L, int M) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N * M) { 
    int k = tid % M; 
    int i = (tid - k) / M;
    for (int j = L - 2; j >= 0; j--) {
      int idx = (i * L + j) * M + k;
      int idx_n = (i * L + (j + 1)) * M + k;
      grid[idx] = min(grid[idx], grid[idx_n] + 1);
    }
    for (int j = 1; j < L; j++) {
      int idx = (i * L + j) * M + k;
      int idx_n = (i * L + (j - 1)) * M + k;
      grid[idx] = min(grid[idx], grid[idx_n] + 1);
    }
  }
}

__global__
void sweep_x (int * grid, int N, int L, int M) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < L * M) { 
    int k = tid % M; 
    int j = (tid - k) / M;
    for (int i = N - 2; i >= 0; i--) {
      int idx = (i * L + j) * M + k;
      int idx_n = ((i + 1) * L + j) * M + k;
      grid[idx] = min(grid[idx], grid[idx_n] + 1);
    }
    for (int i = 1; i < N; i++) {
      int idx = (i * L + j) * M + k;
      int idx_n = ((i - 1) * L + j) * M + k;
      grid[idx] = min(grid[idx], grid[idx_n] + 1);
    }
  }
}

__global__
void init_distance_grid (int *grid, int N, int L, int M) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N * L * M) 
    grid[tid] = (grid[tid] == 1) ? 0 : N + L + M + 10;
}

void calculate_distance (const VoxelGrid &occ, VoxelGrid &dist) { 
  auto [N, M, L] = get_size(occ);

  LL init_volume = vol(get_size(occ));
  int *h_occ;
  {
    Timer tm("(calculate_distance): Allocating stuff");
    h_occ = (int *) malloc(sizeof(int) * init_volume);
    int cnt = 0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < M; j++)
        for (int k = 0; k < L; k++)
          h_occ[cnt++] = occ[i][j][k];
  }

  int *d_occ = nullptr; 
  { 
    Timer tm("(calculate_distance): CUDA operations");
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_occ), sizeof(int) * init_volume)); 
    CUDA_RT_CALL(cudaMemcpy(d_occ, h_occ, sizeof(int) * init_volume, cudaMemcpyHostToDevice)); 

    LL blockSize = 256;
    LL numBlocks; 

    numBlocks = (N * M * L + blockSize - 1) / blockSize;
    init_distance_grid<<<numBlocks, blockSize>>>(d_occ, N, M, L);

    // One iteration is sufficient with correct sweep order
    for (int i = 0; i < 1; i++) {
      numBlocks = (M * L + blockSize - 1) / blockSize;
      sweep_x<<<numBlocks, blockSize>>>(d_occ, N, M, L);

      numBlocks = (N * L + blockSize - 1) / blockSize;
      sweep_y<<<numBlocks, blockSize>>>(d_occ, N, M, L);

      numBlocks = (N * M + blockSize - 1) / blockSize;
      sweep_z<<<numBlocks, blockSize>>>(d_occ, N, M, L);
    }
    cudaMemcpy(h_occ, d_occ, sizeof(int) * init_volume, cudaMemcpyDeviceToHost);
  }

  {
    Timer tm("(calculate_distance): Copying stuff back to result"); 
    int cnt = 0;
    resize3d(dist, get_size(occ)); 
    for (int i = 0; i < N; i++)
      for (int j = 0; j < M; j++) 
        for (int k = 0; k < L; k++) 
          dist[i][j][k] = h_occ[cnt++]; 
  }

  CUDA_RT_CALL(cudaFree(d_occ));
  // Note: cudaDeviceReset() removed - unnecessary context reset
  free(h_occ);
}
#endif

