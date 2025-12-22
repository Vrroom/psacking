#ifndef CONSTANTS_H
#define CONSTANTS_H

#define USE_PARALLEL_FFT3D 1
#define VOXEL_RESOLUTION  128
// #define P 4.0
#define P 100000000.0
// #define VOXEL_FILL_INSIDE 1
#define VOXEL_FILL_INSIDE 0
#define VOXEL_ROBUST_FILL 0
#define FP_POW    16
#define FP_SCALE  (1<<FP_POW)
#define BOX_SCALE v3f(VOXEL_RESOLUTION*FP_SCALE)
#define ALONG_X  1
#define ALONG_Y  2
#define ALONG_Z  4
#define INSIDE   8
#define INSIDE_X 16
#define INSIDE_Y 32
#define INSIDE_Z 64
#define INF 1000000000


#endif
