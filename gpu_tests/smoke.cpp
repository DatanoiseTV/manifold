// Smoke test: produce a shape above the GPU sort threshold (8K tris), run a
// boolean operation both with and without GPU, and compare output hashes.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>

#include "manifold/manifold.h"

using namespace manifold;

static size_t HashMesh(const MeshGL64& m) {
  std::hash<double> hd;
  std::hash<uint64_t> hi;
  size_t h = 0xcbf29ce484222325ull;
  for (double v : m.vertProperties)
    h = (h ^ hd(v)) * 0x100000001b3ull;
  for (uint64_t i : m.triVerts)
    h = (h ^ hi(i)) * 0x100000001b3ull;
  return h;
}

int main() {
  // Two spheres big enough to cross the GPU threshold once unioned.
  auto a = Manifold::Sphere(1.0, 128).Translate({0, 0, 0});
  auto b = Manifold::Sphere(1.0, 128).Translate({0.5, 0, 0});

  printf("a tris=%zu, b tris=%zu\n", a.NumTri(), b.NumTri());

  using clock = std::chrono::high_resolution_clock;
  auto time_it = [&](const char* name, auto&& fn) {
    // Extra warmups to settle caches/thermals before measuring.
    for (int w = 0; w < 3; w++) { auto r = fn(); (void)r.NumTri(); }
    double best_ms = 1e30;
    size_t n_tri = 0;
    size_t hash = 0;
    double vol = 0;
    for (int i = 0; i < 10; i++) {
      auto t0 = clock::now();
      auto res = fn();
      MeshGL64 out = res.GetMeshGL64();
      auto t1 = clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      best_ms = std::min(best_ms, ms);
      if (i == 0) {
        n_tri = out.triVerts.size() / 3;
        hash = HashMesh(out);
        vol = res.Volume();
      }
    }
    printf("[%s] tris=%zu hash=%zx vol=%.6g best=%.2fms\n",
           name, n_tri, hash, vol, best_ms);
  };
  time_it("union", [&] { return a + b; });
  time_it("diff",  [&] { return a - b; });
  time_it("isect", [&] { return a ^ b; });

  std::vector<Manifold> cubes;
  for (int i = 0; i < 50; i++) {
    double t = i * 0.04;
    cubes.push_back(Manifold::Cube({1, 1, 1}).Translate({t, t * 0.5, 0}));
  }
  time_it("batch50",
          [&] { return Manifold::BatchBoolean(cubes, OpType::Add); });

  // One big intersection — the bigger GPU should win.
  auto hires_a = Manifold::Sphere(1.0, 384);
  auto hires_b = Manifold::Sphere(1.0, 384).Translate({0.5, 0, 0});
  printf("hires a/b tris=%zu\n", hires_a.NumTri());
  time_it("hires_union", [&] { return hires_a + hires_b; });
  time_it("hires_diff",  [&] { return hires_a - hires_b; });

  // Very large batch to stretch it further.
  std::vector<Manifold> cubes_big;
  for (int i = 0; i < 200; i++) {
    double t = i * 0.02;
    cubes_big.push_back(
        Manifold::Sphere(0.4, 96).Translate({t, t * 0.5, t * 0.3}));
  }
  time_it("batch200_sph",
          [&] { return Manifold::BatchBoolean(cubes_big, OpType::Add); });

  // Denser batch — more per-op work.
  std::vector<Manifold> dense;
  for (int i = 0; i < 60; i++) {
    double t = i * 0.06;
    dense.push_back(
        Manifold::Sphere(1.0, 256).Translate({t, t * 0.4, t * 0.2}));
  }
  time_it("batch60_hires",
          [&] { return Manifold::BatchBoolean(dense, OpType::Add); });

  // Huge single-pair boolean.
  auto huge_a = Manifold::Sphere(1.0, 512);
  auto huge_b = Manifold::Sphere(1.0, 512).Translate({0.5, 0, 0});
  printf("huge tris each=%zu\n", huge_a.NumTri());
  time_it("huge_union", [&] { return huge_a + huge_b; });
  return 0;
}
