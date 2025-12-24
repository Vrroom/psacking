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

  // Pre-computed FFT of flipped tray (for collision correlation)
  cufftComplex* d_tray_fft;
  // Pre-computed FFT of flipped tray_phi (for proximity correlation)
  cufftComplex* d_tray_phi_fft;

  // Reusable buffers for item processing
  cufftComplex* d_item;
  int* d_real_part;

  // cuFFT plan (reusable)
  cufftHandle plan;

  bool initialized;

  GPUTrayContext() : d_tray_fft(nullptr), d_tray_phi_fft(nullptr),
                     d_item(nullptr), d_real_part(nullptr),
                     initialized(false) {}

  ~GPUTrayContext() {
    cleanup();
  }

  void cleanup() {
    if (d_tray_fft) { cudaFree(d_tray_fft); d_tray_fft = nullptr; }
    if (d_tray_phi_fft) { cudaFree(d_tray_phi_fft); d_tray_phi_fft = nullptr; }
    if (d_item) { cudaFree(d_item); d_item = nullptr; }
    if (d_real_part) { cudaFree(d_real_part); d_real_part = nullptr; }
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

// Complete FFT search using GPU-resident tray context
Index3 fft_search_with_gpu_context(const FlatVoxelGrid& item, bool& found, double& score) {
  if (!g_gpu_context || !g_gpu_context->initialized) {
    throw std::runtime_error("GPU tray context not initialized");
  }

  Index3 tray_size = g_gpu_context->tray_size;
  int L = get<2>(tray_size);

  // Get collision metric using GPU context
  FlatVoxelGrid collision_metric;
  g_gpu_context->correlate_with_tray(item, collision_metric);

  // Mark out-of-bounds as colliding
  Index3 lo, hi;
  FlatVoxelGrid padded_item = item;
  padto3d_flat(padded_item, tray_size);
  get_voxel_grid_bounds_flat(padded_item, lo, hi);
  auto [Mx, My, Mz] = hi;
  int N = get<0>(tray_size), M = get<1>(tray_size), Lz = get<2>(tray_size);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < Lz; k++) {
        if (!((i + Mx <= N - 1) && (j + My <= M - 1) && (k + Mz <= Lz - 1)))
          collision_metric(i, j, k) = max(collision_metric(i, j, k), 1);
      }
    }
  }

  // Get proximity metric using GPU context
  FlatVoxelGrid proximity_metric;
  g_gpu_context->correlate_with_tray_phi(item, proximity_metric);

  // Find best placement
  Index3 bestId(-1, -1, -1);
  found = false;

  vector<Index3> non_colliding_loc;
  where3d_flat(collision_metric, non_colliding_loc, 0);

  double bestVal = INF;
  for (auto id : non_colliding_loc) {
    auto [i, j, k] = id;
    double qz = (k + 0.0) / (L + 0.0);
    double metric_with_penalty = proximity_metric(i, j, k) + P * pow(qz, 3.0);
    if (metric_with_penalty < bestVal) {
      found = true;
      bestId = id;
      bestVal = metric_with_penalty;
      score = metric_with_penalty;
    }
  }

  return bestId;
}

