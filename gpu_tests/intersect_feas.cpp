// End-to-end test for the GPU intersect_kernel.
// Generates random (aL, aR, bL, bR) 4-tuples with valid intersection
// configurations, runs both CPU fp64 Intersect() and the WGSL kernel,
// compares bit-exact.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gpu/gpu_hal.h"

using namespace manifold;

struct Vec3d { double x, y, z; };
struct Vec4d { double x, y, z, w; };

static Vec4d CpuIntersect(Vec3d aL, Vec3d aR, Vec3d bL, Vec3d bR) {
  const double dyL = bL.y - aL.y;
  const double dyR = bR.y - aR.y;
  const bool useL = std::fabs(dyL) < std::fabs(dyR);
  const double dx = aR.x - aL.x;
  double lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (!std::isfinite(lambda)) lambda = 0.0;
  Vec4d xyzz;
  xyzz.x = lambda * dx + (useL ? aL.x : aR.x);
  const double aDy = aR.y - aL.y;
  const double bDy = bR.y - bL.y;
  const bool useA = std::fabs(aDy) < std::fabs(bDy);
  xyzz.y = lambda * (useA ? aDy : bDy) +
           (useL ? (useA ? aL.y : bL.y) : (useA ? aR.y : bR.y));
  xyzz.z = lambda * (aR.z - aL.z) + (useL ? aL.z : aR.z);
  xyzz.w = lambda * (bR.z - bL.z) + (useL ? bL.z : bR.z);
  return xyzz;
}

static void Split(double v, uint32_t& lo, uint32_t& hi) {
  uint64_t u;
  std::memcpy(&u, &v, 8);
  lo = uint32_t(u); hi = uint32_t(u >> 32);
}
static double Join(uint32_t lo, uint32_t hi) {
  uint64_t u = (uint64_t(hi) << 32) | lo;
  double v; std::memcpy(&v, &u, 8); return v;
}

// Must match WGSL layout exactly: array<vec2<u32>, 3> is tightly packed 24B.
struct GpuInput {
  uint32_t aL[6];
  uint32_t aR[6];
  uint32_t bL[6];
  uint32_t bR[6];
  uint32_t pad[8];
};
static_assert(sizeof(GpuInput) == 128, "GpuInput layout");

struct GpuOutput {
  uint32_t xyzz[8];  // 4 × vec2<u32>
};
static_assert(sizeof(GpuOutput) == 32, "GpuOutput layout");

static void PackVec3(Vec3d v, uint32_t* dst) {
  Split(v.x, dst[0], dst[1]);
  Split(v.y, dst[2], dst[3]);
  Split(v.z, dst[4], dst[5]);
}

int main() {
  auto& ctx = gpu::GpuContext::instance();
  if (!ctx.isAvailable()) { printf("no GPU\n"); return 1; }
  auto pl = ctx.getPipeline("intersect_kernel");
  if (!pl) { printf("pipeline failed\n"); return 1; }

  const int N = 50000;
  std::vector<GpuInput> in(N);
  std::vector<Vec4d> cpuOut(N);
  std::vector<Vec3d> aL(N), aR(N), bL(N), bR(N);

  std::mt19937_64 rng(1337);
  std::uniform_real_distribution<double> d(-100, 100);

  int g = 0;
  while (g < N) {
    Vec3d pl_aL{d(rng), d(rng), d(rng)};
    Vec3d pl_aR{pl_aL.x + d(rng), d(rng), d(rng)};
    Vec3d pl_bL{d(rng), pl_aL.y + 0.1 * d(rng) - 2.0, d(rng)};
    Vec3d pl_bR{d(rng), pl_aR.y + 0.1 * d(rng) + 2.0, d(rng)};
    double dyL = pl_bL.y - pl_aL.y, dyR = pl_bR.y - pl_aR.y;
    if (dyL * dyR > 0) continue;
    aL[g] = pl_aL; aR[g] = pl_aR; bL[g] = pl_bL; bR[g] = pl_bR;
    cpuOut[g] = CpuIntersect(pl_aL, pl_aR, pl_bL, pl_bR);
    PackVec3(pl_aL, in[g].aL);
    PackVec3(pl_aR, in[g].aR);
    PackVec3(pl_bL, in[g].bL);
    PackVec3(pl_bR, in[g].bR);
    g++;
  }

  auto bufIn = ctx.upload(in.data(), N * sizeof(GpuInput));
  auto bufOut = ctx.allocate(N * sizeof(GpuOutput));
  struct P { uint32_t count; uint32_t _p[3]; } params{(uint32_t)N, {0, 0, 0}};

  auto batch = ctx.createBatch();
  batch->setBuffer(bufIn, 0);
  batch->setBuffer(bufOut, 1);
  batch->setBytes(&params, sizeof(params), 2);
  std::vector<GpuOutput> out(N);
  batch->dispatch(pl, (N + 63) / 64, 64);
  batch->addReadback(bufOut, out.data(), N * sizeof(GpuOutput));
  batch->commitAndWait();

  // Compare
  auto ulpDist = [](double a, double b) -> double {
    if (std::isnan(a) || std::isnan(b)) return 0.0;
    if (a == b) return 0.0;
    uint64_t ua, ub;
    std::memcpy(&ua, &a, 8); std::memcpy(&ub, &b, 8);
    if ((ua >> 63) != (ub >> 63)) return 1e30;
    int64_t d = int64_t(ua) - int64_t(ub);
    return double(d < 0 ? -d : d);
  };

  std::vector<double> ulpsAll;
  ulpsAll.reserve(N * 4);
  double maxUlp = 0;
  int worstI = 0, worstC = 0;
  for (int i = 0; i < N; i++) {
    double got[4] = {
      Join(out[i].xyzz[0], out[i].xyzz[1]),
      Join(out[i].xyzz[2], out[i].xyzz[3]),
      Join(out[i].xyzz[4], out[i].xyzz[5]),
      Join(out[i].xyzz[6], out[i].xyzz[7])};
    double exp[4] = {cpuOut[i].x, cpuOut[i].y, cpuOut[i].z, cpuOut[i].w};
    for (int c = 0; c < 4; c++) {
      double u = ulpDist(got[c], exp[c]);
      ulpsAll.push_back(u);
      if (u > maxUlp) { maxUlp = u; worstI = i; worstC = c; }
    }
  }
  std::sort(ulpsAll.begin(), ulpsAll.end());
  printf("WGSL fp64 Intersect() vs CPU on %d samples:\n", N);
  printf("  p50=%.1f ulp, p99=%.1f ulp, p999=%.1f ulp, max=%.1f ulp\n",
         ulpsAll[N * 4 / 2], ulpsAll[N * 4 * 99 / 100],
         ulpsAll[N * 4 * 999 / 1000], maxUlp);
  if (maxUlp > 5) {
    double got = Join(out[worstI].xyzz[worstC * 2],
                      out[worstI].xyzz[worstC * 2 + 1]);
    double* expc = reinterpret_cast<double*>(&cpuOut[worstI]);
    printf("  worst: sample %d comp %d  cpu=%.17g gpu=%.17g\n",
           worstI, worstC, expc[worstC], got);
  }
  return 0;
}
