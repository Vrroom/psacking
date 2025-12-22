#include <iostream>
#include <cmath> 
#include <algorithm>
#include <tuple> 
#include <fftw3.h>
#include <vector> 
#include <queue>
#include <string>
#include <chrono>
#include <filesystem>
#include <LibSL/LibSL.h>
#include <stdexcept>
#include "voxelGrid.h"
#include "indexOps.h"
#include "error.h"
#include "constants.h"
#include "types.h"

namespace fs = std::filesystem;
using namespace std;

#include "path.h"

// --------------------------------------------------------------

void fft3d(fftw_complex *in, fftw_complex *out, Index3 size, bool inverse=false);
void dft_conv3(const VoxelGrid &a, const VoxelGrid &b, VoxelGrid &result);
void calculate_distance (const VoxelGrid &occ, VoxelGrid &dist);

// --------------------------------------------------------------

float degree_to_rad (float degree) {  
  return 2.0 * M_PI * degree / 360.0; 
} 

m4x4f create_pitch_matrix(float pitchAngle) {
  float cosA = cos(pitchAngle);
  float sinA = sin(pitchAngle);
  return m4x4f(
    1,   0,    0, 0,
    0, cosA, -sinA, 0,
    0, sinA,  cosA, 0,
    0,   0,    0, 1
  );
}

m4x4f create_yaw_matrix(float yawAngle) {
  float cosA = cos(yawAngle);
  float sinA = sin(yawAngle);
  return m4x4f(
    cosA, 0, sinA, 0,
      0, 1,   0, 0,
   -sinA, 0, cosA, 0,
      0, 0,   0, 1
  );
}

m4x4f create_roll_matrix(float rollAngle) {
  float cosA = cos(rollAngle);
  float sinA = sin(rollAngle);
  return m4x4f(
    cosA, -sinA, 0, 0,
    sinA,  cosA, 0, 0,
      0,     0, 1, 0,
      0,     0, 0, 1
  );
}

m4x4f create_translation_matrix(v3f t) {
  return m4x4f(
    1, 0, 0, t[0],
    0, 1, 0, t[1],
    0, 0, 1, t[2],
    0, 0, 0, 1
  );
}

m4x4f rotation_matrix_from_euler_angles(float yaw, float pitch, float roll) {
  m4x4f yawMatrix = create_yaw_matrix(yaw);
  m4x4f pitchMatrix = create_pitch_matrix(pitch);
  m4x4f rollMatrix = create_roll_matrix(roll);
  return rollMatrix * pitchMatrix * yawMatrix;
}

void rotate_mesh_around_point (TriangleMesh_Ptr &mesh, float yaw, float pitch, float roll, v3f pt) {
  auto t1 = create_translation_matrix(-pt); 
  auto r  = rotation_matrix_from_euler_angles(yaw, pitch, roll); 
  auto t2 = create_translation_matrix(pt); 
  auto tr = t2 * r * t1;
  cout << pt << endl;
  cout << tr.mulPoint(pt) << endl;
  mesh->applyTransform(t2 * r * t1); 
}

// --------------------------------------------------------------

template <typename T> 
struct argsort_comparator {
  vector<T> *data;

  argsort_comparator (vector<T> *data): data(data) {}

  bool operator () (const int i, const int j) const {
    return (*data)[i] < (*data)[j];
  }
};

template <typename T, typename U> 
void sort_by_vals (vector<T> &data, vector<U> &vals) {
  argsort_comparator<U> comp(&vals);
  vector<int> indices(data.size()); 
  for (int i = 0; i < (int) indices.size(); i++) 
    indices[i] = i;
  sort(indices.begin(), indices.end(), comp);
  vector<T> data_clone; 
  for (int i : indices) 
    data_clone.push_back(data[i]); 
  data = data_clone;
}

// --------------------------------------------------------------

void elementWiseMultiplication(fftw_complex *a, fftw_complex *b, Index3 size) {
  auto [N, M, L] = size;
  for (int i = 0; i < N * M * L; i++) {
    double real = a[i][0] * b[i][0] - a[i][1] * b[i][1];
    double imag = a[i][0] * b[i][1] + a[i][1] * b[i][0];
    a[i][0] = real;
    a[i][1] = imag;
  }
}

void elementWiseScalarMultiplicationInPlace (fftw_complex *out, double scale, Index3 size) {
  auto [N, M, L] = size;
  for (int i = 0; i < N * M * L; i++) {
    out[i][0] *= scale;
    out[i][1] *= scale;
  }
}

void convertToFFTWComplex(const VoxelGrid &voxelGrid, fftw_complex *out, Index3 size) {
  auto [N, M, L] = size;
  for (int i = 0; i < N; i++) 
    for (int j = 0; j < M; j++) 
      for (int k = 0; k < L; k++) {
        out[(i * M + j) * L + k][0] = voxelGrid[i][j][k]; 
        out[(i * M + j) * L + k][1] = 0.0;
      }
}

#if USE_PARALLEL_FFT3D == 0
void fft3d(fftw_complex *in, fftw_complex *out, Index3 size, bool inverse) {
  auto [N, M, L] = size;
  fftw_plan plan;
  if (inverse) 
    plan = fftw_plan_dft_3d(N, M, L, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
  else 
    plan = fftw_plan_dft_3d(N, M, L, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  if (inverse) 
    elementWiseScalarMultiplicationInPlace(out, (1.0) / (N * M * L), size); 
}
#endif

void extractRealPart(const fftw_complex *fftwArray, VoxelGrid &target, Index3 size) {
  auto [N, M, L] = size; 
  resize3d(target, size, 0); 
  FOR_VOXEL(i, j, k, size) 
    target[i][j][k] = ((int) round(fftwArray[(i * M + j) * L + k][0])); 
}

#if USE_PARALLEL_FFT3D == 0
void dft_conv3(const VoxelGrid &a, const VoxelGrid &b, VoxelGrid &result) {
  auto [N, M, L] = get_size(a);
  if (!same_size(get_size(a), get_size(b)))
    throw std::runtime_error("Input grids must be of the same size for convolution");
  Index3 paddedSize = make_tuple(2 * N + 1, 2 * M + 1, 2 * L + 1);
  VoxelGrid a_cpy = a, b_cpy = b; 
  {
    Timer tm("(dft_conv3): Padding Stuff"); 
    padto3d(a_cpy, paddedSize);
    padto3d(b_cpy, paddedSize);
  }
  int totalSize = (2 * N + 1) * (2 * M + 1) * (2 * L + 1);
  fftw_complex *A, *B; 
  {
    Timer tm("(dft_conv3): Allocating and copying"); 
    A = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * totalSize);
    B = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * totalSize);
    for (int i = 0; i < totalSize; i++) {
      A[i][0] = 0.0; A[i][1] = 0.0;
      B[i][0] = 0.0; B[i][1] = 0.0;
    }
    convertToFFTWComplex(a_cpy, A, paddedSize);
    convertToFFTWComplex(b_cpy, B, paddedSize);
  }
  { 
    Timer tm("(dft_conv3): fft3d 1");  
    fft3d(A, A, paddedSize);
  }
  { 
    Timer tm("(dft_conv3): fft3d 2");  
    fft3d(B, B, paddedSize);
  }
  {
    Timer tm("(dft_conv3): Element wise mm");
    elementWiseMultiplication(A, B, paddedSize);
  }
  {
    Timer tm("(dft_conv3): fft3d 3"); 
    fft3d(A, A, paddedSize, true);
  }
  {
    Timer tm("(dft_conv3): extract real and truncate"); 
    extractRealPart(A, result, paddedSize); 
    truncateto3d(result, get_size(a)); 
  }
  fftw_free(A);
  fftw_free(B);
}
#endif

void dft_corr3(const VoxelGrid &a, const VoxelGrid &b, VoxelGrid &result) {
  VoxelGrid flipped_a = a; 
  {
    Timer tm("(dft_corr3): First Flip");
    flip3d(flipped_a);
  }
  {
    Timer tm("(dft_corr3): dft_conv3");
    dft_conv3(flipped_a, b, result);
  }
  {
    Timer tm("(dft_corr3): Second Flip"); 
    flip3d(result);
  }
}

void collision_grid (const VoxelGrid &tray, const VoxelGrid &item, VoxelGrid &corr) {
  Index3 size = get_size(tray);
  auto [N, M, L] = size;
  Index3 lo, hi;
  get_voxel_grid_bounds(item, lo, hi); 
  auto [Mx, My, Mz] = hi; 
  dft_corr3(tray, item, corr); 
  FOR_VOXEL (i, j, k, size) 
    if (!((i + Mx <= N - 1) && (j + My <= M - 1) && (k + Mz <= L - 1)))
      corr[i][j][k] = max(corr[i][j][k], 1); 
}

// -------------------------------------------------------------- 

// saves a voxel file (.slab.vox format, can be imported by MagicaVoxel)
void saveAsVox(const char *fname, const Array3D<uchar>& voxs)
{
  Array<v3b> palette(256); // RGB palette
  palette.fill(0);
  palette[123] = v3b(127, 0, 127);
  palette[124] = v3b(255, 0, 0);
  palette[125] = v3b(0, 255, 0);
  palette[126] = v3b(0, 0, 255);
  palette[127] = v3b(255, 255, 255);
  FILE *f;
  f = fopen(fname, "wb");
  sl_assert(f != NULL);
  long sx = voxs.xsize(), sy = voxs.ysize(), sz = voxs.zsize();
  fwrite(&sx, 4, 1, f);
  fwrite(&sy, 4, 1, f);
  fwrite(&sz, 4, 1, f);
  ForRangeReverse(i, sx - 1, 0) {
    ForIndex(j, sy) {
      ForRangeReverse(k, sz - 1, 0) {
        uchar v   = voxs.at(i, j, k);
        uchar pal = v != 0 ? 123 + (v % 5) : 255;
        if (v == INSIDE) {
          pal = 123;
        }
        fwrite(&pal, sizeof(uchar), 1, f);
      }
    }
  }
  fwrite(palette.raw(), sizeof(v3b), 256, f);
  fclose(f);
}

// --------------------------------------------------------------

inline bool isInTriangle(int i, int j, const v3i& p0, const v3i& p1, const v3i& p2, int& _depth,int& _aligned)
{
  v2i delta_p0 = v2i(i, j) - v2i(p0);
  v2i delta_p1 = v2i(i, j) - v2i(p1);
  v2i delta_p2 = v2i(i, j) - v2i(p2);
  v2i delta10 = v2i(p1) - v2i(p0);
  v2i delta21 = v2i(p2) - v2i(p1);
  v2i delta02 = v2i(p0) - v2i(p2);

  int64_t c0 = (int64_t)delta_p0[0] * (int64_t)delta10[1] - (int64_t)delta_p0[1] * (int64_t)delta10[0];
  int64_t c1 = (int64_t)delta_p1[0] * (int64_t)delta21[1] - (int64_t)delta_p1[1] * (int64_t)delta21[0];
  int64_t c2 = (int64_t)delta_p2[0] * (int64_t)delta02[1] - (int64_t)delta_p2[1] * (int64_t)delta02[0];
  // are we inside the triangle? (ignores orientation)
  bool inside = (c0 <= 0 && c1 <= 0 && c2 <= 0) || (c0 >= 0 && c1 >= 0 && c2 >= 0);
  // explicitly tracks cases where the sampling location is exactly on an edge
  _aligned = (c0 == 0) ? 0 : ((c1 == 0) ? 1 : ((c2 == 0) ? 2 : -1));
  // compute depth by barycentric interpolation
  if (inside) {
    int64_t area = c0 + c1 + c2;
    int64_t b0 = (c1 << 10) / area;
    int64_t b1 = (c2 << 10) / area;
    int64_t b2 = (1 << 10) - b0 - b1;
    _depth = (int)((b0 * p0[2] + b1 * p1[2] + b2 * p2[2]) >> 10);
  }
  return inside;
}

// --------------------------------------------------------------

class swizzle_xyz
{
public:
  inline v3i forward(const v3i& v)  const { return v; }
  inline v3i backward(const v3i& v) const { return v; }
  inline int along() const { return ALONG_Z; }
};

class swizzle_zxy
{
public:
  inline v3i   forward(const v3i& v)  const { return v3i(v[2], v[0], v[1]); }
  inline v3i   backward(const v3i& v) const { return v3i(v[1], v[2], v[0]); }
  inline uchar along() const { return ALONG_Y; }
};

class swizzle_yzx
{
public:
  inline v3i   forward(const v3i& v)  const { return v3i(v[1], v[2], v[0]); }
  inline v3i   backward(const v3i& v) const { return v3i(v[2], v[0], v[1]); }
  inline uchar along() const { return ALONG_X; }
};

// --------------------------------------------------------------

template <class S>
void rasterize(
  const v3u&                  tri,
  const std::vector<v3i>&     pts,
  Array3D<uchar>&             _voxs)
{
  const S swizzler;
  v3i tripts[3] = {
    swizzler.forward(pts[tri[0]]),
    swizzler.forward(pts[tri[1]]),
    swizzler.forward(pts[tri[2]])
  };
  // check if triangle is valid
  v2i delta10 = v2i(tripts[1]) - v2i(tripts[0]);
  v2i delta21 = v2i(tripts[2]) - v2i(tripts[1]);
  v2i delta02 = v2i(tripts[0]) - v2i(tripts[2]);
  if (delta10 == v2i(0)) return;
  if (delta21 == v2i(0)) return;
  if (delta02 == v2i(0)) return;
  if (delta02[0] * delta10[1] - delta02[1] * delta10[0] == 0) return;
  // proceed
  AAB<2, int> pixbx;
  pixbx.addPoint(v2i(tripts[0]) / FP_SCALE);
  pixbx.addPoint(v2i(tripts[1]) / FP_SCALE);
  pixbx.addPoint(v2i(tripts[2]) / FP_SCALE);
  int xs = _voxs.xsize(), ys = _voxs.ysize(), zs = _voxs.zsize(); 
  for (int j = pixbx.minCorner()[1]; j <= pixbx.maxCorner()[1]; j++) {
    for (int i = pixbx.minCorner()[0]; i <= pixbx.maxCorner()[0]; i++) {
      int depth; int aligned;
      if (isInTriangle(
        (i << FP_POW) + (1 << (FP_POW - 1)), // centered
        (j << FP_POW) + (1 << (FP_POW - 1)), // centered
        tripts[0], tripts[1], tripts[2], depth, aligned)) {
        v3i vx = swizzler.backward(v3i(i, j, depth >> FP_POW));
        // tag the voxel as occupied
        // NOTE: Voxels are likely to be hit multiple times (e.g. thin features)
        //       we flip the bit every time a hit occurs in a voxel.
        // NOTE: Deals with special case of perfect alignment with an edge,
        //       otherwise triangles on both sides are counted producing a hole.
        //       When aligned we keep a single triangle, the one where the first
        //       edge index is smallest ; works only if the mesh is properly indexed.
        //       (see call to mergeVerticesExact after loading the model)
        if (aligned == -1 || tri[aligned] < tri[(aligned + 1) % 3]) {
          if (is_in_range(Index3(vx[0], vx[1], vx[2]), Index3(xs, ys, zs)))
            _voxs.at(vx[0], vx[1], vx[2]) = (_voxs.at(vx[0], vx[1], vx[2]) ^ swizzler.along());
        }
      }
    }
  }
}

// --------------------------------------------------------------

// This version is more robust by using all three direction
// and voting among them to decide what is filled or not
void fillInsideVoting(Array3D<uchar>& _voxs) {

  // along x
  ForIndex(k, _voxs.zsize()) {
    ForIndex(j, _voxs.ysize()) {
      bool inside = false;
      ForIndex(i, _voxs.xsize()) {
        if (_voxs.at(i, j, k) & ALONG_X) {
          inside = !inside;
        }
        if (inside) {
          _voxs.at(i, j, k) |= INSIDE_X;
        }
      }
    }
  }
  // along y
  ForIndex(k, _voxs.zsize()) {
    ForIndex(j, _voxs.xsize()) {
      bool inside = false;
      ForIndex(i, _voxs.ysize()) {
        if (_voxs.at(j, i, k) & ALONG_Y) {
          inside = !inside;
        }
        if (inside) {
          _voxs.at(j, i, k) |= INSIDE_Y;
        }
      }
    }
  }
  // along z
  ForIndex(k, _voxs.ysize()) {
    ForIndex(j, _voxs.xsize()) {
      bool inside = false;
      ForIndex(i, _voxs.zsize()) {
        if (_voxs.at(j, k, i) & ALONG_Z) {
          inside = !inside;
        }
        if (inside) {
          _voxs.at(j, k, i) |= INSIDE_Z;
        }
      }
    }
  }
  // voting
  ForArray3D(_voxs, i, j, k) {
    uchar v = _voxs.at(i, j, k);
    int votes =
      (  (v & INSIDE_X) ? 1 : 0)
      + ((v & INSIDE_Y) ? 1 : 0)
      + ((v & INSIDE_Z) ? 1 : 0);
    // clean
    _voxs.at(i, j, k) &= ~(INSIDE_X | INSIDE_Y | INSIDE_Z);
    if (votes > 1) {
      // tag as inside
      _voxs.at(i, j, k) |= INSIDE;
    }
  }
}

// --------------------------------------------------------------

void fillInside(Array3D<uchar>& _voxs)
{
  ForIndex(k, _voxs.zsize()) {
    ForIndex(j, _voxs.ysize()) {
      bool inside = false;
      ForIndex(i, _voxs.xsize()) {
        if (_voxs.at(i, j, k) & ALONG_X) {
          inside = !inside;
        }
        if (inside) {
          _voxs.at(i, j, k) |= INSIDE;
        }
      }
    }
  }
}

// --------------------------------------------------------------

void array3d_to_binary_voxel_grid(const Array3D<uchar> &arr, VoxelGrid &vg) {
  int xsize = arr.xsize(), ysize = arr.ysize(), zsize = arr.zsize(); 
  resize3d(vg, Index3(xsize, ysize, zsize)); 
  for (int i = 0; i < xsize; i++)
    for (int j = 0; j < ysize; j++) 
      for (int k = 0; k < zsize; k++) 
        vg[i][j][k] = (arr.at(i, j, k) > 0); 
}

void array3d_to_voxel_grid(const Array3D<uchar> &arr, VoxelGrid &vg) {
  int xsize = arr.xsize(), ysize = arr.ysize(), zsize = arr.zsize(); 
  resize3d(vg, Index3(xsize, ysize, zsize)); 
  for (int i = 0; i < xsize; i++)
    for (int j = 0; j < ysize; j++) 
      for (int k = 0; k < zsize; k++) 
        vg[i][j][k] = arr.at(i, j, k);
}

void voxel_grid_to_array3d(const VoxelGrid &vg, Array3D<uchar> &arr) {
  Index3 sz = get_size(vg); 
  auto [N, M, L] = sz;
  arr.allocate(N, M, L); 
  FOR_VOXEL(i, j, k, sz) 
    arr.at(i, j, k) = vg[i][j][k]; 
}

void voxelize (TriangleMesh_Ptr &mesh, VoxelGrid &vg, int voxel_resolution=VOXEL_RESOLUTION) { 
  mesh->mergeVerticesExact();

  // produce (fixed fp) integer vertices and triangles
  std::vector<v3i> pts;
  std::vector<v3u> tris;
  {
    float factor = 0.95f;
    m4x4f boxtrsf = scaleMatrix(v3f(voxel_resolution * FP_SCALE))
      * scaleMatrix(v3f(1.f) / tupleMax(mesh->bbox().extent()))
      * translationMatrix((1 - factor) * 0.5f * mesh->bbox().extent())
      * scaleMatrix(v3f(factor))
      * translationMatrix(-mesh->bbox().minCorner());

    // transform vertices
    pts.resize(mesh->numVertices());
    ForIndex(p, mesh->numVertices()) {
      v3f pt   = mesh->posAt(p);
      v3f bxpt = boxtrsf.mulPoint(pt);
      v3i ipt  = v3i(clamp(round(bxpt), v3f(0.0f), v3f(voxel_resolution * FP_SCALE) - v3f(1.0f)));
      pts[p]   = ipt;
    }

    // prepare triangles
    tris.reserve(mesh->numTriangles());
    ForIndex(t, mesh->numTriangles()) {
      v3u tri = mesh->triangleAt(t);
      tris.push_back(tri);
    }
  }

  // rasterize into voxels
  v3u resolution(mesh->bbox().extent() / tupleMax(mesh->bbox().extent()) * float(voxel_resolution));
  cout << resolution << endl;
  Array3D<uchar> voxs(resolution);
  voxs.fill(0);
  {
    Timer tm("rasterization");
    Console::progressTextInit((int)tris.size());
    ForIndex(t, tris.size()) {
      Console::progressTextUpdate(t);
      rasterize<swizzle_xyz>(tris[t], pts, voxs); // xy view
      rasterize<swizzle_yzx>(tris[t], pts, voxs); // yz view
      rasterize<swizzle_zxy>(tris[t], pts, voxs); // zx view
    }
    Console::progressTextEnd();
    cerr << endl;
  }

  // add inner voxels
#if VOXEL_FILL_INSIDE
  {
    Timer tm("fill");
    cerr << "filling in/out ... ";
#if VOXEL_ROBUST_FILL
    fillInsideVoting(voxs);
#else
    fillInside(voxs);
#endif
    cerr << " done." << endl;
  }
#endif
  array3d_to_binary_voxel_grid(voxs, vg); 
}

Index3 fft_search_placement (const VoxelGrid &A, const VoxelGrid &tray, bool &found, double &score) { 
  VoxelGrid padded_a = A;
  Index3 tray_size = get_size(tray);
  padto3d(padded_a, tray_size); 

  VoxelGrid collision_metric; 
  {
    Timer tm("(fft_search_placement): collision grid"); 
    collision_grid(tray, padded_a, collision_metric); 
  }

  VoxelGrid tray_phi; 
  {
    Timer tm("(fft_search_placement): distance"); 
    calculate_distance(tray, tray_phi); 
  }

  VoxelGridFP promixity_metric_with_height_penalty;
  VoxelGrid promixity_metric; 
  {
    Timer tm("(fft_search_placement): proximity metric");
    dft_corr3(tray_phi, padded_a, promixity_metric); 
    resize3dfp(promixity_metric_with_height_penalty, tray_size, 0.0); 
    FOR_VOXEL(i, j, k, tray_size) {
      double qz = (k + 0.0) / (get<2>(tray_size) + 0.0); 
      promixity_metric_with_height_penalty[i][j][k] = promixity_metric[i][j][k] + P * pow(qz, 3.0); 
    }
  }

  Index3 bestId(-1, -1, -1); 
  found = false;
  {
    Timer tm("(fft_search_placement): optimal placement search"); 
    vector<Index3> non_colliding_loc; 
    where3d(collision_metric, non_colliding_loc, 0); 

    double bestVal = INF; 
    for (auto id : non_colliding_loc) {
      auto [i, j, k] = id;
      if (promixity_metric_with_height_penalty[i][j][k] < bestVal) { 
        found = true;
        bestId = id; 
        bestVal = promixity_metric_with_height_penalty[i][j][k];
        score = promixity_metric_with_height_penalty[i][j][k]; 
      }
    }
  }

  return bestId;
}

/**
 * Variant of fft_search_placement that uses a pre-computed distance field.
 * This allows caching the distance field across multiple orientation attempts,
 * since the distance field only depends on the tray (not the item orientation).
 *
 * @param A The item to place
 * @param tray The current tray state
 * @param tray_phi Pre-computed distance field of the tray (from calculate_distance)
 * @param found Output: whether a valid placement was found
 * @param score Output: the placement score (lower is better)
 * @return The optimal placement position
 */
Index3 fft_search_placement_with_cache(const VoxelGrid &A, const VoxelGrid &tray,
                                        const VoxelGrid &tray_phi, bool &found, double &score) {
  VoxelGrid padded_a = A;
  Index3 tray_size = get_size(tray);
  padto3d(padded_a, tray_size);

  VoxelGrid collision_metric;
  {
    Timer tm("(fft_search_placement_with_cache): collision grid");
    collision_grid(tray, padded_a, collision_metric);
  }

  // Note: tray_phi is passed in pre-computed, skipping calculate_distance call

  VoxelGridFP promixity_metric_with_height_penalty;
  VoxelGrid promixity_metric;
  {
    Timer tm("(fft_search_placement_with_cache): proximity metric");
    dft_corr3(tray_phi, padded_a, promixity_metric);
    resize3dfp(promixity_metric_with_height_penalty, tray_size, 0.0);
    FOR_VOXEL(i, j, k, tray_size) {
      double qz = (k + 0.0) / (get<2>(tray_size) + 0.0);
      promixity_metric_with_height_penalty[i][j][k] = promixity_metric[i][j][k] + P * pow(qz, 3.0);
    }
  }

  Index3 bestId(-1, -1, -1);
  found = false;
  {
    Timer tm("(fft_search_placement_with_cache): optimal placement search");
    vector<Index3> non_colliding_loc;
    where3d(collision_metric, non_colliding_loc, 0);

    double bestVal = INF;
    for (auto id : non_colliding_loc) {
      auto [i, j, k] = id;
      if (promixity_metric_with_height_penalty[i][j][k] < bestVal) {
        found = true;
        bestId = id;
        bestVal = promixity_metric_with_height_penalty[i][j][k];
        score = promixity_metric_with_height_penalty[i][j][k];
      }
    }
  }

  return bestId;
}

void saveVoxelGrid(const char *fname, const VoxelGrid& vg) {
  Array3D<uchar> new_voxs;
  voxel_grid_to_array3d(vg, new_voxs); 
  saveAsVox(fname, new_voxs);
}

void place_in_tray (const VoxelGrid &item, VoxelGrid &tray, Index3 st_id, int val=1) { 
  auto [a, b, c] = st_id; 
  FOR_VOXEL(i, j, k, get_size(tray)) {
    if (i >= a && j >= b && k >=c) {
      if (is_in_range(Index3(i - a, j - b, k - c), get_size(item))) {
        if (tray[i][j][k] > 0 && item[i - a][j - b][k - c] > 0) {
          throw std::runtime_error(
            "(place_in_tray): Item to be placed intersects with contents of tray at position (" +
            std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ")");
        }
        if (tray[i][j][k] == 0 && item[i - a][j - b][k - c] > 0)
          tray[i][j][k] = val;
      }
    }
  }
}

void run_tests () {
  try {
    {
      // simple fft_search_placement test
      cout << "Running Test 1 ... ";
      {
        VoxelGrid vg1 = {{{1}}, {{1}}, {{1}}}, vg2 = {{{0}, {1}, {1}}, {{0}, {0}, {1}}, {{0}, {1}, {1}}};
        bool found = false;
        double score; 
        Index3 id = fft_search_placement(vg1, vg2, found, score); 
        if (!found) 
          raise_and_kill("(run_tests): Failed Test 1"); 
        place_in_tray(vg1, vg2, id, 2); 
        saveVoxelGrid(SRC_PATH "/test1.slab.vox", vg2);
      }
      {
        VoxelGrid vg1 = {{{1}, {1}}}, vg2 = {{{0}, {1}, {1}}, {{0}, {0}, {1}}, {{0}, {1}, {1}}};
        bool found = false;
        double score;
        Index3 id = fft_search_placement(vg1, vg2, found, score); 
        if (!found) 
          raise_and_kill("(run_tests): Failed Test 1"); 
        place_in_tray(vg1, vg2, id, 2); 
        saveVoxelGrid(SRC_PATH "/test2.slab.vox", vg2);
      }
      {
       VoxelGrid tray = {{{0}, {0}, {0}}, {{0}, {0}, {0}}, {{0}, {0}, {0}}};
       vector<VoxelGrid> vgs = {
         {{{1}}, {{1}}, {{1}}},
         {{{1}, {1}}}, 
         {{{1}}},
       };
       int color_id = 1;
       for (auto vg: vgs) {
         bool found = false;
         double score;
         Index3 id = fft_search_placement(vg, tray, found, score); 
         if (found) 
           place_in_tray(vg, tray, id, color_id++); 
         else break;
       }
       saveVoxelGrid(SRC_PATH "/test3.slab.vox", tray);
      }
      cout << "done." << endl;
    }
    {
      cout << "Running Test 2 ... ";
      TriangleMesh_Ptr mesh1(loadTriangleMesh(SRC_PATH "/1382602.stl"));
      TriangleMesh_Ptr mesh2(loadTriangleMesh(SRC_PATH "/271303.stl"));

      vector<VoxelGrid> vgs = { VoxelGrid(), VoxelGrid() }; 

      voxelize(mesh1, vgs[0]);
      voxelize(mesh2, vgs[1]);

      VoxelGrid tray; 
      resize3d(tray, Index3(300, 300, 300)); 

      int color_id = 1;
      for (auto vg: vgs) {
        bool found = false;
        double score;
        Index3 id = fft_search_placement(vg, tray, found, score); 
        if (found) 
          place_in_tray(vg, tray, id, color_id++); 
        else break;
      }
      saveVoxelGrid(SRC_PATH "/test4.slab.vox", tray);
      cout << "done." << endl;
    }
    {
      // test loading and storing of voxel grids
      cout << "Running Test 3 ... ";
      TriangleMesh_Ptr mesh(loadTriangleMesh(SRC_PATH "/1382602.stl"));
      VoxelGrid vg;
      voxelize(mesh, vg);
      saveVoxelGrid(SRC_PATH "/test5.slab.vox", vg); 
      cout << "done." << endl;
    }
    { 
      // testing index operations
      cout << "Running Test 4 ... ";
      Index3 a(123, 12, 11), b (412, 214, 121); 
      if (!same_size(sum(a, b), Index3(412 + 123, 214 + 12, 121 + 11)))
        raise_and_kill("(run_tests): Failed Test 6"); 
      print_index(a); 
      print_index(b); 
      print_index(sum(a, b)); 
      if (is_in_range(a, a) || is_in_range(b, b))
        raise_and_kill("(run_tests): Failed Test 6"); 
      if (!is_in_range(a, sum(a, Index3(1, 1, 1))))
        raise_and_kill("(run_tests): Failed Test 6"); 
      cout << "done." << endl;
    }
    {
      // test voxel grid operations
      cout << "Running Test 5 ... ";
      {
        // min, max, argmin, argmax
        VoxelGrid tray = {{{1}, {2}, {1}}, {{1}, {120}, {10}}, {{0}, {123}, {0}}};
        if (min3d(tray) != at3d(tray, argmin3d(tray)) && min3d(tray) != 0) 
          raise_and_kill("(run_tests): Failed Test 5"); 
        if (max3d(tray) != at3d(tray, argmax3d(tray)) && max3d(tray) != 123) 
          raise_and_kill("(run_tests): Failed Test 5"); 
      }
      {
        // bfs for calculating distance
        {
          VoxelGrid tray = {{{0}, {0}, {0}}, {{0}, {1}, {0}}, {{0}, {0}, {0}}};
          VoxelGrid ans = {{{2}, {1}, {2}}, {{1}, {0}, {1}}, {{2}, {1}, {2}}};
          VoxelGrid dist; 
          calculate_distance(tray, dist);
          if (!same_grid(ans, dist)) 
            raise_and_kill("(run_tests): Failed Test 5"); 
          print_voxel_grid(dist);
        }
        {
          VoxelGrid tray = {{{0, 0, 0}}, {{0, 1, 0}}, {{0, 0, 0}}};
          VoxelGrid ans = {{{2, 1, 2}}, {{1, 0, 1}}, {{2, 1, 2}}};
          VoxelGrid dist; 
          calculate_distance(tray, dist);
          if (!same_grid(ans, dist)) 
            raise_and_kill("(run_tests): Failed Test 5"); 
          print_voxel_grid(dist);
        }
        {
          VoxelGrid tray = {{{0, 0, 0, 0}}, {{0, 1, 0, 0}}, {{0, 0, 0, 0}}};
          VoxelGrid ans = {{{2, 1, 2, 3}}, {{1, 0, 1, 2}}, {{2, 1, 2, 3}}};
          VoxelGrid dist; 
          calculate_distance(tray, dist);
          if (!same_grid(ans, dist)) 
            raise_and_kill("(run_tests): Failed Test 5"); 
          print_voxel_grid(dist);
        }
      }
      {
        // flip
        VoxelGrid a = {{{2, 1, 2, 3}}, {{1, 0, 1, 2}}, {{2, 1, 8, 5}}};
        VoxelGrid a_copy = a;
        flip3d(a_copy); 
        flip3d(a_copy); 
        if (!same_grid(a_copy, a))
          raise_and_kill("(run_tests): Failed Test 5"); 
        flip3d(a_copy); 
        VoxelGrid ans = {{{5, 8, 1, 2}}, {{2, 1, 0, 1}}, {{3, 2, 1, 2}}}; 
        if (!same_grid(a_copy, ans))
          raise_and_kill("(run_tests): Failed Test 5"); 
      }
      {
        // voxel bounds
        {
          VoxelGrid a = {{{2, 1, 2, 3}}, {{1, 0, 1, 2}}, {{2, 1, 8, 5}}};
          Index3 lo, hi;
          get_voxel_grid_bounds(a, lo, hi); 
          if (!(same_size(lo, Index3(0, 0, 0)) && same_size(hi, Index3(2, 0, 3))))
            raise_and_kill("(run_tests): Failed Test 5"); 
        }
        {
          VoxelGrid a = {
            {{0, 1, 0, 0}, {0, 1, 0, 0}}, 
            {{0, 1, 1, 0}, {0, 1, 1, 0}}, 
            {{1, 0, 0, 0}, {0, 0, 0, 0}}
          };
          Index3 lo, hi;
          get_voxel_grid_bounds(a, lo, hi); 
          if (!(same_size(lo, Index3(0, 0, 0)) && same_size(hi, Index3(2, 1, 2))))
            raise_and_kill("(run_tests): Failed Test 5"); 
        }
        {
          VoxelGrid a = {
            {{0, 1, 0, 0}, {0, 1, 0, 0}}, 
            {{0, 1, 1, 0}, {0, 1, 1, 0}}, 
            {{1, 0, 0, 0}, {0, 0, 0, 0}}
          };
          VoxelGrid res = {
            {{0, 1, 0}, {0, 1, 0}}, 
            {{0, 1, 1}, {0, 1, 1}}, 
            {{1, 0, 0}, {0, 0, 0}}
          };
          make_voxel_grid_tight(a); 
          print_voxel_grid(a);
          if (!same_grid(a, res))
            raise_and_kill("(run_tests): Failed Test 5"); 
        }
      }
      {
        // fill, where
        VoxelGrid a = {
          {{0, 1, 0, 0}, {0, 1, 0, 0}}, 
          {{0, 1, 1, 0}, {0, 1, 1, 0}}, 
          {{1, 0, 0, 0}, {0, 0, 0, 0}}
        };
        vector<Index3> idx;
        where3d(a, idx, 1); 
        if (idx.size() != 7) 
          raise_and_kill("(run_tests): Failed Test 5");
        if (!same_size(idx[0], Index3(0, 0, 1)))
          raise_and_kill("(run_tests): Failed Test 5"); 
        fill3d(a, 100); 
        if (min3d(a) != max3d(a) && min3d(a) != 100) 
          raise_and_kill("(run_tests): Failed Test 5"); 
        idx.clear();
        where3d(a, idx, 1);
        if (idx.size() != 0)
          raise_and_kill("(run_tests): Failed Test 5"); 
      }
      {
        VoxelGrid a = {
          {{0, 1, 0, 0}, {0, 1, 0, 0}}, 
          {{0, 1, 1, 0}, {0, 1, 1, 0}}, 
          {{1, 0, 0, 0}, {0, 0, 0, 0}}
        };
        VoxelGrid a1 = a, a2 = a;
        padto3d(a1, Index3(5, 5, 5)); 
        pad3d(a2, Index3(2, 3, 1)); 
        if (!same_size(get_size(a1), get_size(a2)))
          raise_and_kill("(run_tests): Failed Test 5"); 
        if (!same_grid(a1, a2))
          raise_and_kill("(run_tests): Failed Test 5"); 
        print_voxel_grid(a1);
      }
      cout << "done." << endl;
    }
    {
      cout << "Running Test 6 ... ";
      // fourier transforms
      {
        VoxelGrid a = {{{1, 1, 1}}}; 
        VoxelGrid b = {{{1, 2, 3}}}; 
        VoxelGrid c = {{{1, 3, 6}}}; 
        VoxelGrid res; 
        dft_conv3(a, b, res); 
        if (!same_grid(res, c))
          raise_and_kill("(run_tests): Failed Test 6"); 
        print_voxel_grid(res); 
      }
      {
        VoxelGrid a = {{{1}, {11}, {2}, {5}}}; 
        VoxelGrid b = {{{12}, {12}, {6}, {1}}}; 
        VoxelGrid c = {{{12}, {144}, {162}, {151}}}; 
        VoxelGrid res; 
        dft_conv3(a, b, res); 
        if (!same_grid(res, c))
          raise_and_kill("(run_tests): Failed Test 6"); 
        print_voxel_grid(res); 
      }
      {
        VoxelGrid a = {{{1}}, {{0}}, {{1}}, {{0}}}; 
        VoxelGrid b = {{{1}}, {{8}}, {{12}}, {{4}}}; 
        VoxelGrid c = {{{1}}, {{8}}, {{13}}, {{12}}}; 
        VoxelGrid res; 
        dft_conv3(a, b, res); 
        if (!same_grid(res, c))
          raise_and_kill("(run_tests): Failed Test 6"); 
        print_voxel_grid(res); 
      }
      {
        VoxelGrid a = {{{1, 1, 1}}}; 
        VoxelGrid b = {{{1, 2, 3}}}; 
        VoxelGrid c = {{{6, 3, 1}}}; 
        VoxelGrid res; 
        dft_corr3(a, b, res); 
        if (!same_grid(res, c))
          raise_and_kill("(run_tests): Failed Test 6"); 
        print_voxel_grid(res); 
      }
      {
        VoxelGrid a = {{{1}, {11}, {2}, {5}}}; 
        VoxelGrid b = {{{12}, {12}, {6}, {1}}}; 
        VoxelGrid c = {{{161}, {186}, {84}, {60}}}; 
        VoxelGrid res; 
        dft_corr3(a, b, res); 
        if (!same_grid(res, c))
          raise_and_kill("(run_tests): Failed Test 6"); 
        print_voxel_grid(res); 
      }
      {
        VoxelGrid a = {{{1}}, {{0}}, {{1}}, {{0}}}; 
        VoxelGrid b = {{{1}}, {{8}}, {{12}}, {{4}}}; 
        VoxelGrid c = {{{13}}, {{8}}, {{1}}, {{0}}}; 
        VoxelGrid res; 
        dft_corr3(a, b, res); 
        if (!same_grid(res, c))
          raise_and_kill("(run_tests): Failed Test 6"); 
        print_voxel_grid(res); 
      }
      {
        VoxelGrid a = {{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 1}, {0, 0, 0, 0}}}; 
        VoxelGrid b = {{{1, 1, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}}; 
        VoxelGrid res;
        collision_grid(a, b, res); 
        print_voxel_grid(a);
        print_voxel_grid(b);
        print_voxel_grid(res);
      }
      cout << "done." << endl;
    }
  } catch (Fatal& e) {
    cerr << "[ERROR] " << e.message() << endl;
  }
}

void listdir (string dirpath, vector<string> &paths) {
  try {
    if (fs::exists(dirpath) && fs::is_directory(dirpath)) {
      for (const auto& entry : fs::directory_iterator(dirpath)) {
        string filename = entry.path().filename();
        paths.push_back(string(THINGIVERSE_PATH) + "/" + filename); 
      }
    } else {
      std::cout << "Directory does not exist or is not a directory." << std::endl;
    }
  } catch (const fs::filesystem_error& e) {
    std::cerr << e.what() << std::endl;
  }
}

void build_triangle_mesh_list (vector<string> &paths, vector<TriangleMesh_Ptr> &trimesh_list, int totalCount=10) { 
  vector<long long int> vols; 
  vector<string> acc; 
  for (auto path: paths) {
    try {
      TriangleMesh_Ptr mesh(loadTriangleMesh(path.c_str()));
      int vox_res = ((int) ceil(tupleMax(mesh->bbox().extent())));
      VoxelGrid vg; 
      voxelize(mesh, vg, vox_res);
      make_voxel_grid_tight(vg); 
      Index3 sz = get_size(vg); 
      if (max(sz) < 30) {
        trimesh_list.push_back(mesh);
        // so that we sort by decreasing volume (biggest first)
        vols.push_back(-vol(sz)); 
        acc.push_back(path);
        if (((int) trimesh_list.size()) == totalCount) 
          break;
      }
    } catch (...) {}
  }
  sort_by_vals(trimesh_list, vols); 
  for (auto p: acc) {
    cout << p << endl;
  }
}

void pack (vector<TriangleMesh_Ptr> &meshes, VoxelGrid &tray) {
  int n_placed = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < (int) meshes.size(); i++) {
    TriangleMesh_Ptr mesh = meshes[i]; 
    int vox_res = ((int) ceil(tupleMax(mesh->bbox().extent())));
    VoxelGrid vg; 
    voxelize(mesh, vg, vox_res);
    make_voxel_grid_tight(vg); 
    bool found = false;
    double score;
    Index3 id = fft_search_placement(vg, tray, found, score); 
    if (found) 
      place_in_tray(vg, tray, id, n_placed++); 
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  cout << "Time taken per iteration " << (duration.count() / meshes.size()) << " microseconds" << endl;
  cout << "Placed " << n_placed << " Items" << endl;
  saveVoxelGrid(SRC_PATH "/pack.slab.vox", tray);
}

void print_packing_density (const VoxelGrid &tray) {
  Index3 lo, hi;
  get_voxel_grid_bounds(tray, lo, hi); 
  double tray_volume = vol(sum(diff(hi, lo), Index3(1, 1, 1))); 
  double packed_volume = 0.0; 
  FOR_VOXEL(i, j, k, get_size(tray))
    packed_volume += (double) (tray[i][j][k] > 0); 
  cout << "Packing Density = " << ((int) (100.0 * packed_volume / tray_volume)) << " %" << endl;
}

int main() {
  //run_tests();
  vector<string> paths;
  listdir(THINGIVERSE_PATH, paths);
  vector<TriangleMesh_Ptr> trimesh_list;
  build_triangle_mesh_list(paths, trimesh_list, 1000);
  VoxelGrid tray;
  resize3d(tray, Index3(100, 100, 100)); 
  pack(trimesh_list, tray); 
  print_packing_density(tray);
  return 0;
}

