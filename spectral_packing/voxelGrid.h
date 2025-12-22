#ifndef VOXEL_GRID_H
#define VOXEL_GRID_H

#include <vector>
#include <iostream>
#include "types.h"

using namespace std;

#define FOR_VOXEL(i, j, k, size) \
  for (int i = 0; i < std::get<0>(size); i++) \
    for (int j = 0; j < std::get<1>(size); j++) \
      for (int k = 0; k < std::get<2>(size); k++)

void print_voxel_grid (const VoxelGrid &a); 

bool same_grid (const VoxelGrid &a, const VoxelGrid &b); 

int at3d (VoxelGrid &grid, Index3 id); 

double at3d (VoxelGridFP &grid, Index3 id); 

void resize3d(VoxelGrid& grid, const Index3& size, int value=0); 

void resize3dfp(VoxelGridFP& grid, const Index3& size, double value=0.0); 

void fill3d (VoxelGrid &grid, int value); 

void where3d (VoxelGrid &grid, vector<Index3> &idx, int value); 

void flip3d (VoxelGrid &a); 

void pad3d (VoxelGrid &a, Index3 pad, int value=0); 

void padto3d (VoxelGrid &a, Index3 padto, int value=0); 

void truncateto3d (VoxelGrid &a, Index3 truncateto); 

int min3d (VoxelGrid &a); 

int max3d (VoxelGrid &a); 

Index3 argmin3d(VoxelGrid &in); 

Index3 argmax3d(VoxelGrid &in); 

void get_voxel_grid_bounds(const VoxelGrid &g, Index3 &lo, Index3 &hi); 

void make_voxel_grid_tight (VoxelGrid &vg);

void calculate_distance (const VoxelGrid &s_omega, VoxelGrid &distance_grid);

// Conversion functions between VoxelGrid and FlatVoxelGrid
void voxel_grid_to_flat(const VoxelGrid &src, FlatVoxelGrid &dst);
void flat_to_voxel_grid(const FlatVoxelGrid &src, VoxelGrid &dst);
FlatVoxelGrid to_flat(const VoxelGrid &src);
VoxelGrid to_nested(const FlatVoxelGrid &src);

// FlatVoxelGrid operations (fast versions for GPU pipeline)
void flip3d_flat(FlatVoxelGrid &a);
void padto3d_flat(FlatVoxelGrid &a, Index3 padto, int value=0);
void get_voxel_grid_bounds_flat(const FlatVoxelGrid &g, Index3 &lo, Index3 &hi);
void where3d_flat(const FlatVoxelGrid &grid, vector<Index3> &idx, int value);

#endif
