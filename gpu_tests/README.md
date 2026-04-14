# GPU feasibility tests

Standalone programs that validate the WebGPU/Dawn path end-to-end.
Written for the `feature/gpu-compute-rework` branch.

## Build

First build manifold with the WebGPU backend:

```
cmake -B build -G Ninja \
  -DMANIFOLD_GPU_WEBGPU=ON -DMANIFOLD_PAR=ON \
  -DMANIFOLD_TEST=OFF -DMANIFOLD_CROSS_SECTION=OFF -DMANIFOLD_PYBIND=OFF \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j --target manifold
cp src/gpu/kernels/*.wgsl build/
```

Then compile the tests against the static libs:

```
for t in gpu_tests/*.cpp; do
  g++ -std=c++17 -O3 -I include -I src -I build/include \
    "$t" \
    build/src/libmanifold.a \
    build/src/gpu/libmanifold_gpu.a \
    build/src/gpu/dawn/src/dawn/native/libwebgpu_dawn.a \
    -ltbb -lpthread -ldl \
    $(case "$(uname)" in
        Darwin) echo "-framework CoreFoundation -framework Foundation -framework IOSurface -framework QuartzCore -framework Cocoa -framework IOKit -framework Metal";;
        Linux) echo "-lvulkan";;
      esac) \
    -o "build/$(basename $t .cpp)"
done
```

## What each test does

- **fp64_feas** — runs 100K random fp64 ops (add/sub/mul/div) through the
  WGSL `fp64_lib.wgsl` IEEE-754 emulation, reports ULP distance vs CPU
  fp64. Target: ≤3 ULP max.
- **intersect_feas** — 50K random geometric 4-tuples through the WGSL
  port of `boolean3.cpp:Intersect()`. Target: ULP distances in the
  thousands at most, absolute error ≤ 1e-13.
- **smoke** — benchmark harness. Runs a suite of unions / differences /
  batch booleans. Prints best-of-N timings and result hashes. Run with
  `MANIFOLD_GPU_DISABLE=1` for the CPU baseline.
- **kernel12_feas** — small boolean through GPU Kernel12. Takes an `fn`
  argument (sphere resolution, default 32). Run with
  `MANIFOLD_GPU_KERNEL=1` to activate the GPU fp64 intersection math.

## Env vars

- `MANIFOLD_GPU_DISABLE=1` — force CPU path (disables all GPU dispatch)
- `MANIFOLD_GPU_KERNEL=1` — enables the experimental GPU fp64 Kernel12
  path (capped at 2048 pairs per chunk; on Apple Silicon slower than
  CPU; untested on Vulkan/D3D12)
