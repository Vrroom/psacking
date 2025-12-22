#ifndef INDEX_OPS_H 
#define INDEX_OPS_H 

#include "error.h"
#include <tuple> 
#include <iostream>
#include "types.h"

void print_index (Index3 id);

Index3 sum(Index3 a, Index3 b);

long long int vol (Index3 a);

int max (Index3 a);

Index3 diff(Index3 a, Index3 b);

bool same_size (Index3 a, Index3 b);

bool is_in_range(const Index3& index, const Index3& size);

Index3 get_size (const VoxelGrid &grid);

#endif
