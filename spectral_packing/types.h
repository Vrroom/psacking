#ifndef TYPES_H
#define TYPES_H

#include <tuple>
#include <vector>
#include <queue>
#include <complex>
#include <cstring>

using namespace std;

typedef tuple<int, int, int> Index3;
typedef vector<vector<vector<int> > > VoxelGrid;
typedef vector<vector<vector<double> > > VoxelGridFP;
typedef queue<Index3> BFSQueue;
typedef complex<float> Complex;
typedef vector<Complex> ComplexList;
typedef long long int LL;

/**
 * Flat (contiguous) voxel grid for efficient GPU transfers.
 * Uses row-major order: data[(i * ny + j) * nz + k]
 */
struct FlatVoxelGrid {
    vector<int> data;
    int nx, ny, nz;

    FlatVoxelGrid() : nx(0), ny(0), nz(0) {}

    FlatVoxelGrid(int x, int y, int z) : nx(x), ny(y), nz(z) {
        data.resize(static_cast<size_t>(x) * y * z, 0);
    }

    FlatVoxelGrid(Index3 size) : FlatVoxelGrid(get<0>(size), get<1>(size), get<2>(size)) {}

    inline int& operator()(int i, int j, int k) {
        return data[static_cast<size_t>(i * ny + j) * nz + k];
    }

    inline int operator()(int i, int j, int k) const {
        return data[static_cast<size_t>(i * ny + j) * nz + k];
    }

    inline int* ptr() { return data.data(); }
    inline const int* ptr() const { return data.data(); }
    inline size_t size() const { return data.size(); }
    inline size_t size_bytes() const { return data.size() * sizeof(int); }
    inline Index3 dims() const { return Index3(nx, ny, nz); }

    void resize(int x, int y, int z) {
        nx = x; ny = y; nz = z;
        data.resize(static_cast<size_t>(x) * y * z, 0);
    }

    void resize(Index3 size) {
        resize(get<0>(size), get<1>(size), get<2>(size));
    }

    void fill(int value) {
        std::fill(data.begin(), data.end(), value);
    }
};

#endif
