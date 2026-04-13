// Copyright 2024 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// GPU-accelerated SortGeometry.
// Delegates all fp64 operations to the CPU. Only the O(n log n) radix sort
// is dispatched to the GPU. Falls back to CPU sort if GPU fails.

#include "gpu_sort.h"

#include <cstring>
#include <vector>

#include "../collider.h"
#include "../impl.h"
#include "../parallel.h"
#include "gpu_hal.h"

namespace manifold {
namespace gpu {

namespace {

constexpr uint32_t kNoCode = 0xFFFFFFFFu;
constexpr uint32_t kWorkgroupSize = 256;

// Multi-workgroup GPU radix sort (VkRadixSort/Embree style).
// Uses all GPU compute units. Two passes per radix digit:
//   1. radix_sort_histograms: per-workgroup histogram
//   2. radix_sort_scatter: global prefix sum + scatter via bin_flags/popcount
bool GpuSortByMorton(Vec<int>& new2Old, const Vec<uint32_t>& morton,
                     size_t count) {
  auto& ctx = GpuContext::instance();
  if (!ctx.isAvailable()) return false;

  auto histPL = ctx.getPipeline("radix_sort_histograms");
  auto scatterPL = ctx.getPipeline("radix_sort_scatter");
  if (!histPL || !scatterPL) return false;

  uint32_t n = static_cast<uint32_t>(count);

  // Choose number of workgroups to use all compute units.
  // Each workgroup processes numBlocksPerWG * 256 elements.
  uint32_t numWorkgroups = 16;  // Reasonable for Apple Silicon (20 CUs)
  uint32_t elemsPerWG = (n + numWorkgroups - 1) / numWorkgroups;
  uint32_t numBlocksPerWG =
      (elemsPerWG + kWorkgroupSize - 1) / kWorkgroupSize;
  if (numBlocksPerWG == 0) numBlocksPerWG = 1;

  // Allocate buffers.
  auto keysA = ctx.upload(morton.data(), n * sizeof(uint32_t));
  auto valsA = ctx.upload(new2Old.data(), n * sizeof(int32_t));
  auto keysB = ctx.allocate(n * sizeof(uint32_t));
  auto valsB = ctx.allocate(n * sizeof(int32_t));
  auto histBuf =
      ctx.allocate(numWorkgroups * 256 * sizeof(uint32_t));
  if (!keysA || !valsA || !keysB || !valsB || !histBuf) return false;

  // 4 passes (8 bits per pass, 32-bit keys).
  bool swapped = false;
  for (uint32_t pass = 0; pass < 4; pass++) {
    uint32_t shift = pass * 8;

    auto batch = ctx.createBatch();
    if (!batch) return false;

    // Zero the histogram buffer.
    // (Upload zeros — small: numWorkgroups * 256 * 4 bytes = ~16KB)
    {
      std::vector<uint32_t> zeros(numWorkgroups * 256, 0);
      histBuf = ctx.upload(zeros.data(), zeros.size() * sizeof(uint32_t));
    }

    // Pass 1: Histograms.
    batch->setBuffer(swapped ? keysB : keysA, 0);
    batch->setBuffer(histBuf, 1);
    batch->setBytes(&n, sizeof(uint32_t), 2);
    batch->setBytes(&shift, sizeof(uint32_t), 3);
    batch->setBytes(&numBlocksPerWG, sizeof(uint32_t), 4);
    batch->dispatch(histPL, numWorkgroups, kWorkgroupSize);
    batch->barrier();

    // Pass 2: Scatter.
    batch->setBuffer(swapped ? keysB : keysA, 0);
    batch->setBuffer(swapped ? valsB : valsA, 1);
    batch->setBuffer(swapped ? keysA : keysB, 2);
    batch->setBuffer(swapped ? valsA : valsB, 3);
    batch->setBuffer(histBuf, 4);
    batch->setBytes(&n, sizeof(uint32_t), 5);
    batch->setBytes(&shift, sizeof(uint32_t), 6);
    batch->setBytes(&numWorkgroups, sizeof(uint32_t), 7);
    batch->setBytes(&numBlocksPerWG, sizeof(uint32_t), 8);
    batch->dispatch(scatterPL, numWorkgroups, kWorkgroupSize);

    batch->commitAndWait();
    swapped = !swapped;
  }

  // After 4 passes (even), result is in B buffers.
  ctx.download(swapped ? valsB : valsA, new2Old.data(),
               n * sizeof(int32_t));

  return true;
}

}  // namespace

void BuildColliderGpu(Collider& collider, const VecView<const Box>& leafBB,
                      const VecView<const uint32_t>& leafMorton) {
  auto& ctx = GpuContext::instance();
  int numLeaves = static_cast<int>(leafBB.size());
  if (numLeaves == 0) return;
  int numInternal = numLeaves - 1;
  int numNodes = 2 * numLeaves - 1;

  // Allocate Collider arrays.
  collider.nodeBBox_.resize_nofill(numNodes);
  collider.nodeParent_.resize(numNodes, -1);
  collider.internalChildren_.resize(numInternal, std::make_pair(-1, -1));

  // Try GPU CreateRadixTree.
  auto treePL = ctx.getPipeline("create_radix_tree");
  bool gpuDone = false;
  if (treePL && numInternal > 0) {
    auto gpuMorton = ctx.upload(leafMorton.data(),
                                numLeaves * sizeof(uint32_t));
    auto gpuParent = ctx.upload(collider.nodeParent_.data(),
                                numNodes * sizeof(int32_t));
    // int2 is layout-compatible with pair<int,int> on Metal (both 8 bytes)
    auto gpuChildren = ctx.upload(collider.internalChildren_.data(),
                                  numInternal * 2 * sizeof(int32_t));
    if (gpuMorton && gpuParent && gpuChildren) {
      auto batch = ctx.createBatch();
      if (batch) {
        uint32_t groups = (numInternal + 255) / 256;
        batch->setBuffer(gpuMorton, 0);
        batch->setBuffer(gpuParent, 1);
        batch->setBuffer(gpuChildren, 2);
        batch->setBytes(&numLeaves, sizeof(int), 3);
        batch->dispatch(treePL, groups, 256);
        batch->commitAndWait();

        ctx.download(gpuParent, collider.nodeParent_.data(),
                     numNodes * sizeof(int32_t));
        ctx.download(gpuChildren, collider.internalChildren_.data(),
                     numInternal * 2 * sizeof(int32_t));
        gpuDone = true;
      }
    }
  }

  if (!gpuDone) {
    // CPU fallback.
    for_each_n(autoPolicy(numInternal, 1e4), countAt(0), numInternal,
               collider_internal::CreateRadixTree(
                   {collider.nodeParent_, collider.internalChildren_,
                    leafMorton}));
  }

  // BuildInternalBoxes on CPU (requires fp64 Box unions).
  collider.UpdateBoxes(leafBB);
}

void SortGeometryGpu(Manifold::Impl& impl) {
  // --- SortVerts: compute Morton codes on CPU, sort on GPU ---
  const auto numVert = impl.NumVert();
  Vec<uint32_t> vertMorton(numVert);
  auto vertPolicy = autoPolicy(numVert, 1e5);
  for_each_n(vertPolicy, countAt(0), numVert,
             [&impl, &vertMorton](const int vert) {
               const vec3 pos = impl.vertPos_[vert];
               vertMorton[vert] = std::isnan(pos.x)
                                      ? kNoCode
                                      : Collider::MortonCode(pos, impl.bBox_);
             });

  Vec<int> vertNew2Old(numVert);
  sequence(vertNew2Old.begin(), vertNew2Old.end());

  if (!GpuSortByMorton(vertNew2Old, vertMorton, numVert)) {
    stable_sort(vertNew2Old.begin(), vertNew2Old.end(),
                [&vertMorton](const int& a, const int& b) {
                  return vertMorton[a] < vertMorton[b];
                });
  }

  impl.ReindexVerts(vertNew2Old, numVert);

  const auto newNumVert =
      std::lower_bound(vertNew2Old.begin(), vertNew2Old.end(), kNoCode,
                       [&vertMorton](const int vert, const uint32_t val) {
                         return vertMorton[vert] < val;
                       }) -
      vertNew2Old.begin();
  vertNew2Old.resize(newNumVert);
  Permute(impl.vertPos_, vertNew2Old);
  if (impl.vertNormal_.size() == numVert) {
    Permute(impl.vertNormal_, vertNew2Old);
  }

  // --- GetFaceBoxMorton on CPU (fp64), sort on GPU ---
  Vec<Box> faceBox;
  Vec<uint32_t> faceMorton;
  impl.GetFaceBoxMorton(faceBox, faceMorton);

  const auto numTri = impl.NumTri();
  Vec<int> faceNew2Old(numTri);
  sequence(faceNew2Old.begin(), faceNew2Old.end());

  if (!GpuSortByMorton(faceNew2Old, faceMorton, numTri)) {
    stable_sort(faceNew2Old.begin(), faceNew2Old.end(),
                [&faceMorton](const int& a, const int& b) {
                  return faceMorton[a] < faceMorton[b];
                });
  }

  // Trim, permute, gather — exactly as CPU path.
  const int newNumTri =
      std::lower_bound(faceNew2Old.begin(), faceNew2Old.end(), kNoCode,
                       [&faceMorton](const int face, const uint32_t val) {
                         return faceMorton[face] < val;
                       }) -
      faceNew2Old.begin();
  faceNew2Old.resize(newNumTri);
  Permute(faceMorton, faceNew2Old);
  Permute(faceBox, faceNew2Old);
  impl.GatherFaces(faceNew2Old);

  if (impl.halfedge_.size() == 0) {
    impl.collider_ = {};
    return;
  }

  BuildColliderGpu(impl.collider_, faceBox, faceMorton);
  impl.CompactProps();
}

}  // namespace gpu
}  // namespace manifold
