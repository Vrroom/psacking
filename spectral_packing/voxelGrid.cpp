#include "indexOps.h"
#include "voxelGrid.h" 
#include "constants.h"
#include <algorithm>

void print_voxel_grid (const VoxelGrid &a) {
  auto [N, M, L] = get_size(a);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < L; k++) 
        cout << a[i][j][k] << " "; 
      cout << endl;
    }
    cout << endl;
  }
}

bool same_grid (const VoxelGrid &a, const VoxelGrid &b) {
  if (!same_size(get_size(a), get_size(b)))
    return false;
  Index3 sz = get_size(a); 
  FOR_VOXEL(i, j, k, sz) if (a[i][j][k] != b[i][j][k]) 
    return false;
  return true;
}

int at3d (VoxelGrid &grid, Index3 id) {
  auto [i, j, k] = id;
  return grid[i][j][k]; 
}

double at3d (VoxelGridFP &grid, Index3 id) {
  auto [i, j, k] = id;
  return grid[i][j][k]; 
}

void resize3d(VoxelGrid& grid, const Index3& size, int value) {
  auto [x, y, z] = size;
  grid.resize(x);
  for (auto& plane : grid) {
    plane.resize(y);
    for (auto& row : plane) {
      row.resize(z, value);
    }
  }
}

void resize3dfp(VoxelGridFP& grid, const Index3& size, double value) {
  auto [x, y, z] = size;
  grid.resize(x);
  for (auto& plane : grid) {
    plane.resize(y);
    for (auto& row : plane) {
      row.resize(z, value);
    }
  }
}

void fill3d (VoxelGrid &grid, int value) {
  for (int i = 0; i < (int) grid.size(); i++)
    for (int j = 0; j < (int) grid[i].size(); j++)
      fill(grid[i][j].begin(), grid[i][j].end(), value);
}

void where3d (VoxelGrid &grid, vector<Index3> &idx, int value) {
  Index3 sz = get_size(grid); 
  FOR_VOXEL (i, j, k, sz) if (grid[i][j][k] == value)
    idx.emplace_back(i, j, k); 
}

void flip3d (VoxelGrid &a) {
  Index3 size = get_size(a);
  // flip z
  for (int i = 0; i < get<0>(size); i++)
    for (int j = 0; j < get<1>(size); j++)
      reverse(a[i][j].begin(), a[i][j].end());
  // flip y
  for (int i = 0; i < get<0>(size); i++)
    reverse(a[i].begin(), a[i].end());
  // flip x
  reverse(a.begin(), a.end());
}

void pad3d (VoxelGrid &a, Index3 pad, int value) {
  Index3 sz = get_size(a);
  VoxelGrid new_grid;
  resize3d(new_grid, sum(sz, pad), value);
  FOR_VOXEL (i, j, k, sz)
    new_grid[i][j][k] = a[i][j][k];
  a = new_grid;
}

void padto3d (VoxelGrid &a, Index3 padto, int value) {
  Index3 sz = get_size(a);
  VoxelGrid new_grid;
  resize3d(new_grid, padto, value);
  FOR_VOXEL (i, j, k, sz)
    new_grid[i][j][k] = a[i][j][k];
  a = new_grid;
}

void truncateto3d (VoxelGrid &a, Index3 truncateto) {
  Index3 sz = get_size(a);
  if (!is_in_range(truncateto, sum(sz, Index3(1, 1, 1))))
    raise_and_kill("(truncateto3d) Truncation size is bigger than original");
  VoxelGrid new_grid;
  resize3d(new_grid, truncateto);
  FOR_VOXEL (i, j, k, truncateto)
    new_grid[i][j][k] = a[i][j][k];
  a = new_grid;
}

int min3d (VoxelGrid &a) {
  Index3 sz = get_size(a);
  int minVal = INF;
  FOR_VOXEL(i, j, k, sz)
    minVal = min(minVal, a[i][j][k]);
  return minVal;
}

int max3d (VoxelGrid &a) {
  Index3 sz = get_size(a);
  int maxVal = -INF;
  FOR_VOXEL(i, j, k, sz)
    maxVal = max(maxVal, a[i][j][k]);
  return maxVal;
}

Index3 argmin3d(VoxelGrid &in) {
  Index3 sz = get_size(in);
  int minVal = INF;
  Index3 minIdx = {0, 0, 0}; 
  FOR_VOXEL(i, j, k, sz) if (in[i][j][k] < minVal) {
    minVal = in[i][j][k];
    minIdx = make_tuple(i, j, k);
  }
  return minIdx;
}

Index3 argmax3d(VoxelGrid &in) {
  Index3 sz = get_size(in);
  int maxVal = -INF;
  Index3 maxIdx = {0, 0, 0};  // Initialize to the first index
  FOR_VOXEL(i, j, k, sz) if (in[i][j][k] > maxVal) {
    maxVal = in[i][j][k];
    maxIdx = make_tuple(i, j, k);
  }
  return maxIdx;
}

void get_voxel_grid_bounds(const VoxelGrid &g, Index3 &lo, Index3 &hi) {
  Index3 sz = get_size(g);
  lo = Index3(INF, INF, INF); 
  hi = Index3(-INF, -INF, -INF); 
  FOR_VOXEL (i, j, k, sz) if (g[i][j][k] > 0) {
    get<0>(lo) = min(get<0>(lo), i); 
    get<1>(lo) = min(get<1>(lo), j); 
    get<2>(lo) = min(get<2>(lo), k); 

    get<0>(hi) = max(get<0>(hi), i); 
    get<1>(hi) = max(get<1>(hi), j); 
    get<2>(hi) = max(get<2>(hi), k); 
  }
}

void make_voxel_grid_tight (VoxelGrid &vg) {
  Index3 lo, hi; 
  get_voxel_grid_bounds(vg, lo, hi); 
  if (same_size(hi, Index3(-INF, -INF, -INF)))
    raise("(make_voxel_grid_tight) Voxel grid doesn't have any occupied cells"); 
  auto [a, b, c] = lo;
  Index3 new_dim = sum(diff(hi, lo), Index3(1, 1, 1)); 
  VoxelGrid new_grid;
  resize3d(new_grid, new_dim); 
  FOR_VOXEL(i, j, k, new_dim) 
    new_grid[i][j][k] = vg[i + a][j + b][k + c];
  vg = new_grid;
}

#if USE_PARALLEL_FFT3D == 0
void calculate_distance (const VoxelGrid &s_omega, VoxelGrid &distance_grid) {
  Index3 size = get_size(s_omega);
  int N = get<0>(size), M = get<1>(size), L = get<2>(size);
  resize3d(distance_grid, size, N + M + L + 100); 
  VoxelGrid visited;
  resize3d(visited, size);
  BFSQueue q;
  FOR_VOXEL(i, j, k, size) if (s_omega[i][j][k] == 1) {
    distance_grid[i][j][k] = 0;
    visited[i][j][k] = 1;
    q.push(Index3(i, j, k));
  }
  const int dx[] = {1, -1, 0, 0, 0, 0};
  const int dy[] = {0, 0, 1, -1, 0, 0};
  const int dz[] = {0, 0, 0, 0, 1, -1};
  while (!q.empty()) {
    auto [x, y, z] = q.front();
    q.pop();
    for (int dir = 0; dir < 6; ++dir) {
      int nx = x + dx[dir];
      int ny = y + dy[dir];
      int nz = z + dz[dir];
      if (is_in_range(Index3(nx, ny, nz), size) && !visited[nx][ny][nz]) {
          visited[nx][ny][nz] = 1;
          distance_grid[nx][ny][nz] = min(distance_grid[nx][ny][nz], distance_grid[x][y][z] + 1); 
          q.push(Index3(nx, ny, nz));
      }
    }
  }
}

#endif

// Conversion functions between VoxelGrid and FlatVoxelGrid
void voxel_grid_to_flat(const VoxelGrid &src, FlatVoxelGrid &dst) {
    Index3 sz = get_size(src);
    int nx = get<0>(sz), ny = get<1>(sz), nz = get<2>(sz);
    dst.resize(nx, ny, nz);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                dst(i, j, k) = src[i][j][k];
}

void flat_to_voxel_grid(const FlatVoxelGrid &src, VoxelGrid &dst) {
    resize3d(dst, src.dims());
    for (int i = 0; i < src.nx; i++)
        for (int j = 0; j < src.ny; j++)
            for (int k = 0; k < src.nz; k++)
                dst[i][j][k] = src(i, j, k);
}

FlatVoxelGrid to_flat(const VoxelGrid &src) {
    FlatVoxelGrid dst;
    voxel_grid_to_flat(src, dst);
    return dst;
}

VoxelGrid to_nested(const FlatVoxelGrid &src) {
    VoxelGrid dst;
    flat_to_voxel_grid(src, dst);
    return dst;
}

// Flip all 3 dimensions of a FlatVoxelGrid (x, y, z all reversed)
void flip3d_flat(FlatVoxelGrid &a) {
    int nx = a.nx, ny = a.ny, nz = a.nz;
    FlatVoxelGrid flipped(nx, ny, nz);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                // Map (i,j,k) to (nx-1-i, ny-1-j, nz-1-k)
                flipped(nx - 1 - i, ny - 1 - j, nz - 1 - k) = a(i, j, k);
            }
        }
    }
    a = std::move(flipped);
}

// Pad FlatVoxelGrid to a target size
void padto3d_flat(FlatVoxelGrid &a, Index3 padto, int value) {
    int old_nx = a.nx, old_ny = a.ny, old_nz = a.nz;
    auto [new_nx, new_ny, new_nz] = padto;

    FlatVoxelGrid padded(new_nx, new_ny, new_nz);
    padded.fill(value);

    for (int i = 0; i < old_nx; i++) {
        for (int j = 0; j < old_ny; j++) {
            // Copy row by row for efficiency
            std::memcpy(&padded(i, j, 0), &a(i, j, 0), old_nz * sizeof(int));
        }
    }
    a = std::move(padded);
}

// Get bounds of non-zero voxels in a FlatVoxelGrid
void get_voxel_grid_bounds_flat(const FlatVoxelGrid &g, Index3 &lo, Index3 &hi) {
    int nx = g.nx, ny = g.ny, nz = g.nz;
    lo = Index3(INF, INF, INF);
    hi = Index3(-INF, -INF, -INF);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                if (g(i, j, k) > 0) {
                    get<0>(lo) = min(get<0>(lo), i);
                    get<1>(lo) = min(get<1>(lo), j);
                    get<2>(lo) = min(get<2>(lo), k);

                    get<0>(hi) = max(get<0>(hi), i);
                    get<1>(hi) = max(get<1>(hi), j);
                    get<2>(hi) = max(get<2>(hi), k);
                }
            }
        }
    }
}

// Find positions where grid has specified value
void where3d_flat(const FlatVoxelGrid &grid, vector<Index3> &idx, int value) {
    int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                if (grid(i, j, k) == value)
                    idx.emplace_back(i, j, k);
            }
        }
    }
}
