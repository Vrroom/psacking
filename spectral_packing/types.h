#ifndef TYPES_H
#define TYPES_H

#include <tuple> 
#include <vector>
#include <queue>
#include <complex>

using namespace std;

typedef tuple<int, int, int> Index3; 
typedef vector<vector<vector<int> > > VoxelGrid; 
typedef vector<vector<vector<double> > > VoxelGridFP; 
typedef queue<Index3> BFSQueue;
typedef complex<float> Complex;
typedef vector<Complex> ComplexList;
typedef long long int LL; 

#endif
