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
#include "timing.h"

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

// =============================================================================
// GPU-Resident Score Computation Kernels (Phase 2)
// =============================================================================

// Compute penalty-adjusted scores on GPU, marking invalid positions with INF
__global__
void compute_scores_kernel(const int* collision_metric, const int* proximity_metric,
                           double* scores, int nx, int ny, int nz,
                           int max_x, int max_y, int max_z,
                           double height_penalty) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nx * ny * nz;
  if (tid < total) {
    int k = tid % nz;
    int j = (tid / nz) % ny;
    int i = tid / (ny * nz);

    // Check collision
    if (collision_metric[tid] != 0) {
      scores[tid] = 1e30;  // Invalid: collision
      return;
    }

    // Check out-of-bounds
    if (!((i + max_x <= nx - 1) && (j + max_y <= ny - 1) && (k + max_z <= nz - 1))) {
      scores[tid] = 1e30;  // Invalid: OOB
      return;
    }

    // Compute penalty-adjusted score
    double qz = (double)k / (double)nz;
    scores[tid] = (double)proximity_metric[tid] + height_penalty * qz * qz * qz;
  }
}

// =============================================================================
// Phase 3: Fused Int→Float Conversion Kernels
// =============================================================================

// Convert int array to cufftComplex on GPU (fused with padding)
// This reduces H2D bandwidth by 50% (4 bytes/int vs 8 bytes/float2)
__global__
void int_to_complex_and_pad_kernel(const int* src, cufftComplex* dst,
                                    int src_nx, int src_ny, int src_nz,
                                    int dst_nx, int dst_ny, int dst_nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int src_total = src_nx * src_ny * src_nz;
  int dst_total = dst_nx * dst_ny * dst_nz;

  // Initialize all destination elements to zero first
  // (This kernel will be called once to set zeros, then source will be copied)
  if (tid < dst_total) {
    dst[tid].x = 0.0f;
    dst[tid].y = 0.0f;
  }
}

// Copy and convert source ints to complex in padded destination
__global__
void int_to_complex_copy_kernel(const int* src, cufftComplex* dst,
                                 int src_nx, int src_ny, int src_nz,
                                 int dst_ny, int dst_nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int src_total = src_nx * src_ny * src_nz;

  if (tid < src_total) {
    // Convert linear index to 3D coordinates
    int k = tid % src_nz;
    int j = (tid / src_nz) % src_ny;
    int i = tid / (src_ny * src_nz);

    // Compute destination index (in padded array)
    int dst_idx = (i * dst_ny + j) * dst_nz + k;

    // Convert int to complex
    dst[dst_idx].x = (float)src[tid];
    dst[dst_idx].y = 0.0f;
  }
}

// Truncate and flip a padded result into tray-sized output (GPU version)
__global__
void truncate_flip_kernel(const int* padded, int* output,
                          int nx, int ny, int nz,
                          int px, int py, int pz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nx * ny * nz;
  if (tid < total) {
    int k = tid % nz;
    int j = (tid / nz) % ny;
    int i = tid / (ny * nz);

    // Flip coordinates
    int fi = nx - 1 - i;
    int fj = ny - 1 - j;
    int fk = nz - 1 - k;

    // Read from padded (using padded dimensions for indexing)
    int src_idx = (fi * py + fj) * pz + fk;
    output[tid] = padded[src_idx];
  }
}

// Parallel reduction to find argmin - each block finds its local minimum
__global__
void argmin_reduction_kernel(const double* scores, int n,
                             int* block_argmin, double* block_min) {
  extern __shared__ char shared_mem[];
  double* s_scores = (double*)shared_mem;
  int* s_indices = (int*)(shared_mem + blockDim.x * sizeof(double));

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  if (gid < n) {
    s_scores[tid] = scores[gid];
    s_indices[tid] = gid;
  } else {
    s_scores[tid] = 1e30;
    s_indices[tid] = -1;
  }
  __syncthreads();

  // Reduction within block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && tid + s < blockDim.x) {
      if (s_scores[tid + s] < s_scores[tid]) {
        s_scores[tid] = s_scores[tid + s];
        s_indices[tid] = s_indices[tid + s];
      }
    }
    __syncthreads();
  }

  // First thread writes block result
  if (tid == 0) {
    block_min[blockIdx.x] = s_scores[0];
    block_argmin[blockIdx.x] = s_indices[0];
  }
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
  SCOPED_TIMER("dft_conv3_flat_total");

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
    SCOPED_TIMER("to_cufft_complex");
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
    SCOPED_TIMER("cuda_malloc");
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(cufftComplex) * init_volume));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(cufftComplex) * init_volume));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_real_part), sizeof(int) * padded_volume));
  }

  {
    SCOPED_TIMER("memcpy_H2D");
    CUDA_RT_CALL(cudaMemcpy(d_A, h_A, sizeof(cufftComplex) * init_volume, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_B, h_B, sizeof(cufftComplex) * init_volume, cudaMemcpyHostToDevice));
  }

  {
    SCOPED_TIMER("pad_voxel_grid");
    pad_voxel_grid_cuda(d_A, a.dims(), padded_size);
    pad_voxel_grid_cuda(d_B, b.dims(), padded_size);
  }

  {
    SCOPED_TIMER("cufft_plan_create");
    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C));
  }

  {
    SCOPED_TIMER("cufft_forward");
    CUFFT_CALL(cufftExecC2C(plan, d_A, d_A, CUFFT_FORWARD));
    CUFFT_CALL(cufftExecC2C(plan, d_B, d_B, CUFFT_FORWARD));
    cudaDeviceSynchronize();  // Required for accurate timing
  }

  LL blockSize = 256;
  LL numBlocks = (padded_volume + blockSize - 1) / blockSize;

  {
    SCOPED_TIMER("cmplx_mul");
    element_wise_cmplx_mul<<<numBlocks, blockSize>>>(d_A, d_B, d_A, padded_volume);
    cudaDeviceSynchronize();
  }

  {
    SCOPED_TIMER("cufft_inverse");
    CUFFT_CALL(cufftExecC2C(plan, d_A, d_A, CUFFT_INVERSE));
    cudaDeviceSynchronize();
  }

  {
    SCOPED_TIMER("post_process_kernels");
    double scalar = 1.0 / ((double) padded_volume);
    element_wise_cmplx_scalar_mul<<<numBlocks, blockSize>>>(d_A, scalar, padded_volume);
    element_wise_round<<<numBlocks, blockSize>>>(d_A, padded_volume);
    extract_real_kernel<<<numBlocks, blockSize>>>(d_A, d_real_part, padded_volume);
    cudaDeviceSynchronize();
  }

  {
    SCOPED_TIMER("memcpy_D2H");
    cudaMemcpy(h_out, d_real_part, sizeof(int) * padded_volume, cudaMemcpyDeviceToHost);
  }

  {
    SCOPED_TIMER("to_flat_grid");
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
  SCOPED_TIMER("dft_corr3_flat_total");
  FlatVoxelGrid flipped_a = a;
  {
    SCOPED_TIMER("flip3d_input");
    flip3d_flat(flipped_a);
  }
  dft_conv3_flat(flipped_a, b, result);
  {
    SCOPED_TIMER("flip3d_output");
    flip3d_flat(result);
  }
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

// =============================================================================
// GPU-Resident Tray Context
// =============================================================================
// Keeps tray data on GPU between searches to avoid repeated transfers.
// Pre-computes FFT of tray and tray_phi for fast correlation.

// Helper kernel for padding: copy from small buffer to large padded buffer
__global__
void pad_copy_kernel(const cufftComplex* src, cufftComplex* dst,
                     int src_nx, int src_ny, int src_nz,
                     int dst_nx, int dst_ny, int dst_nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = src_nx * src_ny * src_nz;
  if (tid < total) {
    int k = tid % src_nz;
    int j = (tid / src_nz) % src_ny;
    int i = tid / (src_ny * src_nz);
    int dst_idx = (i * dst_ny + j) * dst_nz + k;
    dst[dst_idx] = src[tid];
  }
}

void pad_voxel_grid_cuda_inplace(const cufftComplex* d_src, cufftComplex* d_dst,
                                  Index3 src_size, Index3 dst_size) {
  auto [sx, sy, sz] = src_size;
  auto [dx, dy, dz] = dst_size;
  int total = sx * sy * sz;
  int blockSize = 256;
  int numBlocks = (total + blockSize - 1) / blockSize;
  pad_copy_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, sx, sy, sz, dx, dy, dz);
  CUDA_RT_CALL(cudaDeviceSynchronize());
}

class GPUTrayContext {
public:
  // Tray dimensions
  int nx, ny, nz;
  Index3 tray_size;
  Index3 padded_size;
  LL padded_volume;
  LL init_volume;

  // Raw padded data of flipped tray (NOT pre-FFT'd - compute fresh each time for precision)
  cufftComplex* d_tray_fft;
  // Raw padded data of flipped tray_phi (NOT pre-FFT'd - compute fresh each time for precision)
  cufftComplex* d_tray_phi_fft;
  // Temporary buffer for fresh FFT computation
  cufftComplex* d_tray_fft_temp;

  // Reusable buffers for item processing
  cufftComplex* d_item;
  int* d_real_part;

  // Phase 2: GPU-resident score computation buffers
  int* d_collision_result;     // Collision metric (stays on GPU)
  int* d_proximity_result;     // Proximity metric (stays on GPU)
  double* d_scores;            // Score buffer for argmin
  int* d_block_argmin;         // Block-level argmin results
  double* d_block_min;         // Block-level min values
  cufftComplex* d_item2;       // Second item buffer for parallel correlation

  // Phase 3: Raw int buffer for fused conversion (half H2D bandwidth)
  int* d_item_int;             // Raw int buffer for item upload

  // cuFFT plan (reusable)
  cufftHandle plan;

  bool initialized;

  GPUTrayContext() : d_tray_fft(nullptr), d_tray_phi_fft(nullptr),
                     d_item(nullptr), d_real_part(nullptr),
                     d_collision_result(nullptr), d_proximity_result(nullptr),
                     d_scores(nullptr), d_block_argmin(nullptr), d_block_min(nullptr),
                     d_item2(nullptr), d_item_int(nullptr),
                     initialized(false) {}

  ~GPUTrayContext() {
    cleanup();
  }

  void cleanup() {
    if (d_tray_fft) { cudaFree(d_tray_fft); d_tray_fft = nullptr; }
    if (d_tray_phi_fft) { cudaFree(d_tray_phi_fft); d_tray_phi_fft = nullptr; }
    if (d_item) { cudaFree(d_item); d_item = nullptr; }
    if (d_real_part) { cudaFree(d_real_part); d_real_part = nullptr; }
    if (d_collision_result) { cudaFree(d_collision_result); d_collision_result = nullptr; }
    if (d_proximity_result) { cudaFree(d_proximity_result); d_proximity_result = nullptr; }
    if (d_scores) { cudaFree(d_scores); d_scores = nullptr; }
    if (d_block_argmin) { cudaFree(d_block_argmin); d_block_argmin = nullptr; }
    if (d_block_min) { cudaFree(d_block_min); d_block_min = nullptr; }
    if (d_item2) { cudaFree(d_item2); d_item2 = nullptr; }
    if (d_item_int) { cudaFree(d_item_int); d_item_int = nullptr; }
    if (initialized) { cufftDestroy(plan); }
    initialized = false;
  }

  void initialize(const FlatVoxelGrid& tray, const FlatVoxelGrid& tray_phi) {
    cleanup();

    nx = tray.nx; ny = tray.ny; nz = tray.nz;
    tray_size = Index3(nx, ny, nz);
    padded_size = Index3(2 * nx + 1, 2 * ny + 1, 2 * nz + 1);
    padded_volume = vol(padded_size);
    init_volume = tray.size();

    // Flip tray and tray_phi for correlation
    FlatVoxelGrid flipped_tray = tray;
    flip3d_flat(flipped_tray);
    FlatVoxelGrid flipped_tray_phi = tray_phi;
    flip3d_flat(flipped_tray_phi);

    // Allocate host buffers for conversion
    cufftComplex* h_tray = (cufftComplex*)malloc(sizeof(cufftComplex) * init_volume);
    cufftComplex* h_tray_phi = (cufftComplex*)malloc(sizeof(cufftComplex) * init_volume);
    flat_to_cufft_complex(flipped_tray, h_tray);
    flat_to_cufft_complex(flipped_tray_phi, h_tray_phi);

    // Allocate GPU memory
    CUDA_RT_CALL(cudaMalloc(&d_tray_fft, sizeof(cufftComplex) * padded_volume));
    CUDA_RT_CALL(cudaMalloc(&d_tray_phi_fft, sizeof(cufftComplex) * padded_volume));
    CUDA_RT_CALL(cudaMalloc(&d_item, sizeof(cufftComplex) * padded_volume));
    CUDA_RT_CALL(cudaMalloc(&d_real_part, sizeof(int) * padded_volume));

    // Phase 2: Allocate GPU-resident score computation buffers
    CUDA_RT_CALL(cudaMalloc(&d_collision_result, sizeof(int) * init_volume));
    CUDA_RT_CALL(cudaMalloc(&d_proximity_result, sizeof(int) * init_volume));
    CUDA_RT_CALL(cudaMalloc(&d_scores, sizeof(double) * init_volume));
    CUDA_RT_CALL(cudaMalloc(&d_item2, sizeof(cufftComplex) * padded_volume));
    // Allocate block reduction buffers (one entry per block)
    int num_blocks = (init_volume + 255) / 256;
    CUDA_RT_CALL(cudaMalloc(&d_block_argmin, sizeof(int) * num_blocks));
    CUDA_RT_CALL(cudaMalloc(&d_block_min, sizeof(double) * num_blocks));

    // Phase 3: Allocate raw int buffer for fused conversion (half H2D bandwidth)
    CUDA_RT_CALL(cudaMalloc(&d_item_int, sizeof(int) * init_volume));

    // Create cuFFT plan
    auto [px, py, pz] = padded_size;
    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan3d(&plan, px, py, pz, CUFFT_C2C));

    // Upload and pad tray
    cufftComplex* d_temp;
    CUDA_RT_CALL(cudaMalloc(&d_temp, sizeof(cufftComplex) * init_volume));

    // Process tray
    CUDA_RT_CALL(cudaMemcpy(d_temp, h_tray, sizeof(cufftComplex) * init_volume, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemset(d_tray_fft, 0, sizeof(cufftComplex) * padded_volume));
    pad_voxel_grid_cuda_inplace(d_temp, d_tray_fft, tray_size, padded_size);
    CUFFT_CALL(cufftExecC2C(plan, d_tray_fft, d_tray_fft, CUFFT_FORWARD));

    // Process tray_phi
    CUDA_RT_CALL(cudaMemcpy(d_temp, h_tray_phi, sizeof(cufftComplex) * init_volume, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemset(d_tray_phi_fft, 0, sizeof(cufftComplex) * padded_volume));
    pad_voxel_grid_cuda_inplace(d_temp, d_tray_phi_fft, tray_size, padded_size);
    CUFFT_CALL(cufftExecC2C(plan, d_tray_phi_fft, d_tray_phi_fft, CUFFT_FORWARD));

    CUDA_RT_CALL(cudaFree(d_temp));
    free(h_tray);
    free(h_tray_phi);

    initialized = true;
  }

  // Compute correlation of item with pre-computed tray FFT
  void correlate_with_tray(const FlatVoxelGrid& item, FlatVoxelGrid& result) {
    correlate_internal(item, d_tray_fft, result);
  }

  // Compute correlation of item with pre-computed tray_phi FFT
  void correlate_with_tray_phi(const FlatVoxelGrid& item, FlatVoxelGrid& result) {
    correlate_internal(item, d_tray_phi_fft, result);
  }

private:
  void correlate_internal(const FlatVoxelGrid& item, cufftComplex* d_precomputed_fft,
                          FlatVoxelGrid& result) {
    // Pad item to tray size
    FlatVoxelGrid padded_item = item;
    padto3d_flat(padded_item, tray_size);

    // Convert to complex
    cufftComplex* h_item = (cufftComplex*)malloc(sizeof(cufftComplex) * init_volume);
    flat_to_cufft_complex(padded_item, h_item);

    // Upload item
    cufftComplex* d_temp;
    CUDA_RT_CALL(cudaMalloc(&d_temp, sizeof(cufftComplex) * init_volume));
    CUDA_RT_CALL(cudaMemcpy(d_temp, h_item, sizeof(cufftComplex) * init_volume, cudaMemcpyHostToDevice));

    // Pad on GPU
    CUDA_RT_CALL(cudaMemset(d_item, 0, sizeof(cufftComplex) * padded_volume));
    pad_voxel_grid_cuda_inplace(d_temp, d_item, tray_size, padded_size);
    CUDA_RT_CALL(cudaFree(d_temp));

    // FFT of item
    CUFFT_CALL(cufftExecC2C(plan, d_item, d_item, CUFFT_FORWARD));

    // Element-wise multiply with pre-computed FFT
    LL blockSize = 256;
    LL numBlocks = (padded_volume + blockSize - 1) / blockSize;
    element_wise_cmplx_mul<<<numBlocks, blockSize>>>(d_item, d_precomputed_fft, d_item, padded_volume);

    // Inverse FFT
    CUFFT_CALL(cufftExecC2C(plan, d_item, d_item, CUFFT_INVERSE));

    // Scale, round, extract real
    double scalar = 1.0 / ((double)padded_volume);
    element_wise_cmplx_scalar_mul<<<numBlocks, blockSize>>>(d_item, scalar, padded_volume);
    element_wise_round<<<numBlocks, blockSize>>>(d_item, padded_volume);
    extract_real_kernel<<<numBlocks, blockSize>>>(d_item, d_real_part, padded_volume);

    // Copy back
    int* h_out = (int*)malloc(sizeof(int) * padded_volume);
    CUDA_RT_CALL(cudaMemcpy(h_out, d_real_part, sizeof(int) * padded_volume, cudaMemcpyDeviceToHost));

    // Convert to result and truncate
    to_flat_grid(h_out, result, padded_size);
    FlatVoxelGrid truncated(nx, ny, nz);
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        std::memcpy(&truncated(i, j, 0), &result(i, j, 0), nz * sizeof(int));
      }
    }
    result = std::move(truncated);

    // Flip result (completing the correlation)
    flip3d_flat(result);

    free(h_item);
    free(h_out);
  }

  // Phase 2 & 3: Correlate and store result on GPU (no D2H transfer, fused int→float)
  void correlate_to_gpu(const FlatVoxelGrid& item, cufftComplex* d_precomputed_fft,
                        int* d_output, cufftComplex* d_work_buffer) {
    // Pad item to tray size on CPU
    FlatVoxelGrid padded_item = item;
    padto3d_flat(padded_item, tray_size);

    // Phase 3: Transfer raw ints to GPU (half the bandwidth vs float2)
    // Then convert to complex and pad on GPU
    CUDA_RT_CALL(cudaMemcpy(d_item_int, padded_item.ptr(), sizeof(int) * init_volume, cudaMemcpyHostToDevice));

    // Zero the destination buffer
    CUDA_RT_CALL(cudaMemset(d_work_buffer, 0, sizeof(cufftComplex) * padded_volume));

    // Fused int→complex conversion and padding on GPU
    int blockSize = 256;
    int numBlocks = (init_volume + blockSize - 1) / blockSize;
    auto [px, py, pz] = padded_size;
    int_to_complex_copy_kernel<<<numBlocks, blockSize>>>(
      d_item_int, d_work_buffer, nx, ny, nz, py, pz);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    // FFT of item
    CUFFT_CALL(cufftExecC2C(plan, d_work_buffer, d_work_buffer, CUFFT_FORWARD));

    // Element-wise multiply with pre-computed FFT
    LL blockSize2 = 256;
    LL numBlocks2 = (padded_volume + blockSize2 - 1) / blockSize2;
    element_wise_cmplx_mul<<<numBlocks2, blockSize2>>>(d_work_buffer, d_precomputed_fft, d_work_buffer, padded_volume);

    // Inverse FFT
    CUFFT_CALL(cufftExecC2C(plan, d_work_buffer, d_work_buffer, CUFFT_INVERSE));

    // Scale, round, extract real
    double scalar = 1.0 / ((double)padded_volume);
    element_wise_cmplx_scalar_mul<<<numBlocks2, blockSize2>>>(d_work_buffer, scalar, padded_volume);
    element_wise_round<<<numBlocks2, blockSize2>>>(d_work_buffer, padded_volume);
    extract_real_kernel<<<numBlocks2, blockSize2>>>(d_work_buffer, d_real_part, padded_volume);

    // Truncate and flip result entirely on GPU (no D2H transfer!)
    // Note: we need to get padded dimensions again since we're inside the function
    int total_output = init_volume;
    int numBlocks3 = (total_output + 255) / 256;
    auto [ppx, ppy, ppz] = padded_size;
    truncate_flip_kernel<<<numBlocks3, 256>>>(
      d_real_part, d_output, nx, ny, nz, ppx, ppy, ppz);
    CUDA_RT_CALL(cudaDeviceSynchronize());
    // No host memory to free - everything done on GPU!
  }

public:
  // Debug: Run correlate_to_gpu and return the result (for testing)
  void debug_correlate_to_gpu_proximity(const FlatVoxelGrid& item, FlatVoxelGrid& result) {
    correlate_to_gpu(item, d_tray_phi_fft, d_proximity_result, d_item2);

    // Download result from GPU
    int init_volume = nx * ny * nz;
    int* h_out = (int*)malloc(sizeof(int) * init_volume);
    cudaMemcpy(h_out, d_proximity_result, sizeof(int) * init_volume, cudaMemcpyDeviceToHost);

    result = FlatVoxelGrid(nx, ny, nz);
    for (int i = 0; i < init_volume; i++) {
      result.ptr()[i] = h_out[i];
    }
    free(h_out);
  }

  // Phase 2: Complete search on GPU - returns only best position
  Index3 search_on_gpu(const FlatVoxelGrid& item, bool& found, double& score, Index3 item_bounds_hi) {
    SCOPED_TIMER("search_on_gpu_total");

    auto [max_x, max_y, max_z] = item_bounds_hi;

    // Correlate item with tray (collision) - keep on GPU
    {
      SCOPED_TIMER("gpu_correlate_collision");
      correlate_to_gpu(item, d_tray_fft, d_collision_result, d_item);
    }

    // Correlate item with tray_phi (proximity) - keep on GPU
    {
      SCOPED_TIMER("gpu_correlate_proximity");
      correlate_to_gpu(item, d_tray_phi_fft, d_proximity_result, d_item2);
    }

    // Compute scores on GPU
    int total = init_volume;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    {
      SCOPED_TIMER("compute_scores_gpu");
      compute_scores_kernel<<<numBlocks, blockSize>>>(
        d_collision_result, d_proximity_result, d_scores,
        nx, ny, nz, max_x, max_y, max_z, P
      );
      CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    // Find argmin using block reduction
    int best_idx = -1;
    double best_score = 1e30;

    {
      SCOPED_TIMER("argmin_reduction_gpu");
      // Shared memory size: doubles for scores + ints for indices
      size_t shared_size = blockSize * (sizeof(double) + sizeof(int));
      argmin_reduction_kernel<<<numBlocks, blockSize, shared_size>>>(
        d_scores, total, d_block_argmin, d_block_min
      );
      CUDA_RT_CALL(cudaDeviceSynchronize());

      // Download block results and find global minimum on CPU
      // (small arrays - only numBlocks elements)
      std::vector<int> h_block_argmin(numBlocks);
      std::vector<double> h_block_min(numBlocks);
      CUDA_RT_CALL(cudaMemcpy(h_block_argmin.data(), d_block_argmin,
                              sizeof(int) * numBlocks, cudaMemcpyDeviceToHost));
      CUDA_RT_CALL(cudaMemcpy(h_block_min.data(), d_block_min,
                              sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));

      for (int b = 0; b < numBlocks; b++) {
        if (h_block_min[b] < best_score) {
          best_score = h_block_min[b];
          best_idx = h_block_argmin[b];
        }
      }
    }

    // Convert linear index back to 3D coordinates
    if (best_idx >= 0 && best_score < 1e29) {
      found = true;
      score = best_score;
      int k = best_idx % nz;
      int j = (best_idx / nz) % ny;
      int i = best_idx / (ny * nz);
      return Index3(i, j, k);
    } else {
      found = false;
      score = 0.0;
      return Index3(-1, -1, -1);
    }
  }
};

// Global context pointer (managed by Python)
static GPUTrayContext* g_gpu_context = nullptr;

void gpu_tray_context_init(const FlatVoxelGrid& tray, const FlatVoxelGrid& tray_phi) {
  if (!g_gpu_context) {
    g_gpu_context = new GPUTrayContext();
  }
  g_gpu_context->initialize(tray, tray_phi);
}

void gpu_tray_context_cleanup() {
  if (g_gpu_context) {
    delete g_gpu_context;
    g_gpu_context = nullptr;
  }
}

bool gpu_tray_context_is_initialized() {
  return g_gpu_context && g_gpu_context->initialized;
}

Index3 gpu_tray_context_dims() {
  if (!g_gpu_context || !g_gpu_context->initialized) {
    throw std::runtime_error("GPU tray context not initialized");
  }
  return g_gpu_context->tray_size;
}

void gpu_tray_correlate_collision(const FlatVoxelGrid& item, FlatVoxelGrid& result) {
  if (!g_gpu_context || !g_gpu_context->initialized) {
    throw std::runtime_error("GPU tray context not initialized");
  }
  g_gpu_context->correlate_with_tray(item, result);
}

void gpu_tray_correlate_proximity(const FlatVoxelGrid& item, FlatVoxelGrid& result) {
  if (!g_gpu_context || !g_gpu_context->initialized) {
    throw std::runtime_error("GPU tray context not initialized");
  }
  g_gpu_context->correlate_with_tray_phi(item, result);
}

// Debug: Use the GPU-resident correlate_to_gpu path and download results
void gpu_tray_correlate_proximity_fast(const FlatVoxelGrid& item, FlatVoxelGrid& result) {
  if (!g_gpu_context || !g_gpu_context->initialized) {
    throw std::runtime_error("GPU tray context not initialized");
  }
  g_gpu_context->debug_correlate_to_gpu_proximity(item, result);
}

// Complete FFT search using GPU-resident tray context
// Phase 2: Uses GPU-resident score computation to avoid large D2H transfers
Index3 fft_search_with_gpu_context(const FlatVoxelGrid& item, bool& found, double& score) {
  SCOPED_TIMER("gpu_context_search_total");

  if (!g_gpu_context || !g_gpu_context->initialized) {
    throw std::runtime_error("GPU tray context not initialized");
  }

  // Get item bounds for OOB checking
  Index3 tray_size = g_gpu_context->tray_size;
  FlatVoxelGrid padded_item = item;
  padto3d_flat(padded_item, tray_size);
  Index3 lo, hi;
  get_voxel_grid_bounds_flat(padded_item, lo, hi);

  // Use GPU-resident search (Phase 2 optimization)
  // This keeps collision and proximity metrics on GPU, does score computation
  // and argmin on GPU, only transfers the final result (12 bytes)
  return g_gpu_context->search_on_gpu(item, found, score, hi);
}

// =============================================================================
// Phase 4: Batch Orientation Processing
// =============================================================================

// Structure to hold batch search results
struct BatchSearchResult {
  Index3 position;
  bool found;
  double score;
};

// Batch search: process multiple orientations, return best placement
// This reduces Python↔C++ call overhead from 24 calls to 1 call per item
void fft_search_batch(const std::vector<FlatVoxelGrid>& orientations,
                      Index3& best_position, bool& found, double& best_score) {
  SCOPED_TIMER("batch_search_total");

  if (!g_gpu_context || !g_gpu_context->initialized) {
    throw std::runtime_error("GPU tray context not initialized");
  }

  Index3 tray_size = g_gpu_context->tray_size;
  best_position = Index3(-1, -1, -1);
  found = false;
  best_score = 1e30;

  for (const auto& item : orientations) {
    // Skip if item is larger than tray
    if (item.nx > std::get<0>(tray_size) ||
        item.ny > std::get<1>(tray_size) ||
        item.nz > std::get<2>(tray_size)) {
      continue;
    }

    // Get item bounds
    FlatVoxelGrid padded_item = item;
    padto3d_flat(padded_item, tray_size);
    Index3 lo, hi;
    get_voxel_grid_bounds_flat(padded_item, lo, hi);

    bool orientation_found = false;
    double orientation_score = 0.0;
    Index3 position = g_gpu_context->search_on_gpu(item, orientation_found, orientation_score, hi);

    if (orientation_found && orientation_score < best_score) {
      best_position = position;
      found = true;
      best_score = orientation_score;
    }
  }
}

