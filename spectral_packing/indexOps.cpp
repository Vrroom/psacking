#include "indexOps.h"

Index3 sum(Index3 a, Index3 b) {
  return Index3(get<0>(a) + get<0>(b), get<1>(a) + get<1>(b), get<2>(a) + get<2>(b));
}

long long int vol (Index3 a) {
  auto [N, M, L] = a;
  return N * 1ll * M * 1ll * L; 
}

int max (Index3 a) {
  auto [N, M, L] = a; 
  return max(N, max(M, L)); 
}

Index3 diff(Index3 a, Index3 b) {
  return Index3(get<0>(a) - get<0>(b), get<1>(a) - get<1>(b), get<2>(a) - get<2>(b));
}

bool same_size (Index3 a, Index3 b) {
  return \
    (get<0>(a) == get<0>(b)) &&
    (get<1>(a) == get<1>(b)) &&
    (get<2>(a) == get<2>(b));
}

bool is_in_range(const Index3& index, const Index3& size) {
  auto [x, y, z] = index;
  auto [N, M, L] = size;
  return x >= 0 && x < N && y >= 0 && y < M && z >= 0 && z < L;
}

Index3 get_size (const VoxelGrid &grid) {
  if (grid.size() == 0 || grid[0].size() == 0 || grid[0][0].size() == 0)
    raise("(get_size) Unacceptable grid");
  return Index3(grid.size(), grid[0].size(), grid[0][0].size());
}

