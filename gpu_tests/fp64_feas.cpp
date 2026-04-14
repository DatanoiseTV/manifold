// Feasibility test for the WGSL IEEE 754 fp64 emulation.
// Runs random fp64 add/sub/mul/div pairs on GPU, compares bit-exact vs CPU.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gpu/gpu_hal.h"

using namespace manifold;

struct TestInput { uint32_t aLo, aHi, bLo, bHi; };
struct TestOutput { uint32_t sLo, sHi, dLo, dHi, pLo, pHi, qLo, qHi; };

static void SplitDouble(double v, uint32_t& lo, uint32_t& hi) {
  uint64_t u;
  std::memcpy(&u, &v, 8);
  lo = static_cast<uint32_t>(u);
  hi = static_cast<uint32_t>(u >> 32);
}

static double JoinDouble(uint32_t lo, uint32_t hi) {
  uint64_t u = (static_cast<uint64_t>(hi) << 32) | lo;
  double v;
  std::memcpy(&v, &u, 8);
  return v;
}

static double UlpDistance(double actual, double expected) {
  if (std::isnan(actual) || std::isnan(expected)) return 0.0;
  if (actual == expected) return 0.0;
  uint64_t ua, ue;
  std::memcpy(&ua, &actual, 8);
  std::memcpy(&ue, &expected, 8);
  if ((ua >> 63) != (ue >> 63)) return 1e30;
  int64_t da = static_cast<int64_t>(ua);
  int64_t de = static_cast<int64_t>(ue);
  return static_cast<double>(std::abs(da - de));
}

int main() {
  auto& ctx = gpu::GpuContext::instance();
  if (!ctx.isAvailable()) { printf("no GPU\n"); return 1; }
  auto pl = ctx.getPipeline("fp64_test");
  if (!pl) { printf("pipeline load failed\n"); return 1; }

  const int N = 100000;
  std::vector<TestInput> in(N);
  std::vector<double> aCpu(N), bCpu(N);

  std::mt19937_64 rng(1337);
  std::uniform_real_distribution<double> d_big(-1e12, 1e12);
  std::uniform_real_distribution<double> d_small(-1e-6, 1e-6);
  std::uniform_real_distribution<double> d_exp(-300, 300);

  for (int i = 0; i < N; i++) {
    // Mix of scales — large, small, wildly different exponents, cancellations.
    double a, b;
    switch (i % 5) {
      case 0: a = d_big(rng);   b = d_big(rng);   break;
      case 1: a = d_small(rng); b = d_small(rng); break;
      case 2: a = d_big(rng);   b = d_small(rng); break;  // huge exp diff
      case 3: {
        a = d_big(rng);
        // Cancellation: b = -a + tiny wobble
        b = -a + (d_small(rng) * std::fabs(a));
        break;
      }
      default: {
        double ea = d_exp(rng), eb = d_exp(rng);
        a = std::ldexp(d_big(rng) * 1e-12, static_cast<int>(ea));
        b = std::ldexp(d_big(rng) * 1e-12, static_cast<int>(eb));
        break;
      }
    }
    if (b == 0.0) b = 1.0;
    aCpu[i] = a;
    bCpu[i] = b;
    SplitDouble(a, in[i].aLo, in[i].aHi);
    SplitDouble(b, in[i].bLo, in[i].bHi);
  }

  auto bufIn = ctx.upload(in.data(), N * sizeof(TestInput));
  auto bufOut = ctx.allocate(N * sizeof(TestOutput));
  struct Params { uint32_t count; uint32_t _pad[3]; } p{(uint32_t)N, {0, 0, 0}};

  auto batch = ctx.createBatch();
  if (!batch) { printf("no batch\n"); return 1; }
  batch->setBuffer(bufIn, 0);
  batch->setBuffer(bufOut, 1);
  batch->setBytes(&p, sizeof(p), 2);
  std::vector<TestOutput> out(N);
  uint32_t groups = (N + 63) / 64;
  batch->dispatch(pl, groups, 64);
  batch->addReadback(bufOut, out.data(), N * sizeof(TestOutput));
  batch->commitAndWait();

  // Compare
  auto stats = [&](const char* name, auto op) {
    std::vector<double> ulps;
    ulps.reserve(N);
    double maxUlp = 0;
    int maxIdx = 0;
    for (int i = 0; i < N; i++) {
      double expected = op(aCpu[i], bCpu[i]);
      double gpu_val;
      if (name[0] == 'a') gpu_val = JoinDouble(out[i].sLo, out[i].sHi);
      else if (name[0] == 's') gpu_val = JoinDouble(out[i].dLo, out[i].dHi);
      else if (name[0] == 'm') gpu_val = JoinDouble(out[i].pLo, out[i].pHi);
      else gpu_val = JoinDouble(out[i].qLo, out[i].qHi);
      double u = UlpDistance(gpu_val, expected);
      ulps.push_back(u);
      if (u > maxUlp) { maxUlp = u; maxIdx = i; }
    }
    std::sort(ulps.begin(), ulps.end());
    printf("  %s: p50=%.1f ulp, p99=%.1f ulp, max=%.1f ulp",
           name, ulps[N / 2], ulps[N * 99 / 100], maxUlp);
    if (maxUlp > 3) {
      printf(" (worst case: a=%.15g b=%.15g expected=%.17g got=%.17g)",
             aCpu[maxIdx], bCpu[maxIdx], op(aCpu[maxIdx], bCpu[maxIdx]),
             name[0] == 'a' ? JoinDouble(out[maxIdx].sLo, out[maxIdx].sHi)
             : name[0] == 's' ? JoinDouble(out[maxIdx].dLo, out[maxIdx].dHi)
             : name[0] == 'm' ? JoinDouble(out[maxIdx].pLo, out[maxIdx].pHi)
             : JoinDouble(out[maxIdx].qLo, out[maxIdx].qHi));
    }
    printf("\n");
  };

  printf("WGSL fp64 precision vs CPU fp64 on %d random pairs:\n", N);
  stats("add", [](double x, double y) { return x + y; });
  stats("sub", [](double x, double y) { return x - y; });
  stats("mul", [](double x, double y) { return x * y; });
  stats("div", [](double x, double y) { return x / y; });
  return 0;
}
