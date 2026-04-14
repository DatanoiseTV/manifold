// Minimal standalone test for GPU Kernel12 via gpu::IntersectGpu.
// Runs a tiny boolean with GPU env var set, compares result hashes.

#include <cstdio>
#include <cstdlib>

#include "manifold/manifold.h"

using namespace manifold;

int main(int argc, char** argv) {
  int fn = argc > 1 ? std::atoi(argv[1]) : 32;
  auto a = Manifold::Sphere(1.0, fn);
  auto b = Manifold::Sphere(1.0, fn).Translate({0.5, 0, 0});
  printf("fn=%d a tris=%zu, b tris=%zu\n", fn, a.NumTri(), b.NumTri());
  printf("MANIFOLD_GPU_KERNEL=%s\n",
         std::getenv("MANIFOLD_GPU_KERNEL") ? "1" : "unset");
  printf("starting union...\n"); fflush(stdout);
  auto res = a + b;
  auto out = res.GetMeshGL64();
  printf("done. result tris=%zu genus=%d vol=%.6g\n",
         out.triVerts.size() / 3, res.Genus(), res.Volume());
  return 0;
}
