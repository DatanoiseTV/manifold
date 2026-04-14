// Copyright 2026 The Manifold Authors.
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
// GPU dispatch for manifold's Kernel12 (edge×face intersection w/ fp64 math).

#include "gpu_boolean.h"

#include <cstring>

#include "../parallel.h"
#include "../shared.h"
#include "gpu_hal.h"

namespace manifold {
namespace gpu {

namespace {

// Each vec3 fp64 coord packs as (lo:u32, hi:u32) × 3 = 24 bytes, matching
// WGSL's `struct Vec3F64 { x,y,z: vec2<u32> }`.
struct GpuVec3F64 { uint32_t data[6]; };
static_assert(sizeof(GpuVec3F64) == 24, "Vec3F64 layout");

struct GpuHalfedge {
  int32_t startVert;
  int32_t endVert;
  int32_t pairedHalfedge;
  int32_t propVert;
};
static_assert(sizeof(GpuHalfedge) == 16, "GpuHalfedge layout");

struct K12ParamsCpu {
  uint32_t count;
  uint32_t expandP;
  uint32_t forward;
  uint32_t _pad;
};

inline void PackDouble(double v, uint32_t& lo, uint32_t& hi) {
  uint64_t u;
  std::memcpy(&u, &v, 8);
  lo = static_cast<uint32_t>(u);
  hi = static_cast<uint32_t>(u >> 32);
}

inline GpuVec3F64 PackVec3(vec3 v) {
  GpuVec3F64 r{};
  PackDouble(v.x, r.data[0], r.data[1]);
  PackDouble(v.y, r.data[2], r.data[3]);
  PackDouble(v.z, r.data[4], r.data[5]);
  return r;
}

inline vec3 UnpackVec3(const GpuVec3F64& r) {
  uint64_t u;
  double xyz[3];
  for (int c = 0; c < 3; c++) {
    u = static_cast<uint64_t>(r.data[c * 2]) |
        (static_cast<uint64_t>(r.data[c * 2 + 1]) << 32);
    std::memcpy(&xyz[c], &u, 8);
  }
  return vec3(xyz[0], xyz[1], xyz[2]);
}

// Repack a manifold Vec<vec3> (fp64) to GPU-friendly layout.
Vec<GpuVec3F64> RepackVec3Array(const Vec<vec3>& src) {
  const size_t n = src.size();
  Vec<GpuVec3F64> dst(n);
  for_each_n(autoPolicy(n, 1e4), countAt(size_t{0}), n,
             [&](size_t i) { dst[i] = PackVec3(src[i]); });
  return dst;
}

}  // namespace

bool IntersectGpu(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                  VecView<const std::pair<int, int>> pairs, bool expandP,
                  bool forward, Vec<int>& outX, Vec<vec3>& outV) {
  auto& ctx = GpuContext::instance();
  if (!ctx.isAvailable()) return false;
  auto pl = ctx.getPipeline("kernel12_dispatch");
  if (!pl) return false;

  const size_t nPairs = pairs.size();
  outX.resize(nPairs);
  outV.resize(nPairs);
  if (nPairs == 0) return true;

  // Need vertNormal on both meshes. Manifold computes it in sort path; if
  // it's missing the GPU kernel results would be garbage.
  if (inP.vertNormal_.size() != inP.vertPos_.size() ||
      inQ.vertNormal_.size() != inQ.vertPos_.size()) {
    return false;
  }

  Vec<GpuVec3F64> pVertPos = RepackVec3Array(inP.vertPos_);
  Vec<GpuVec3F64> pVertNormal = RepackVec3Array(inP.vertNormal_);
  Vec<GpuVec3F64> pFaceNormal = RepackVec3Array(inP.faceNormal_);
  Vec<GpuVec3F64> qVertPos = RepackVec3Array(inQ.vertPos_);
  Vec<GpuVec3F64> qVertNormal = RepackVec3Array(inQ.vertNormal_);
  Vec<GpuVec3F64> qFaceNormal = RepackVec3Array(inQ.faceNormal_);

  auto upV3 = [&](const Vec<GpuVec3F64>& v) {
    return ctx.upload(v.data(), v.size() * sizeof(GpuVec3F64));
  };

  auto bufP_vp = upV3(pVertPos);
  auto bufP_vn = upV3(pVertNormal);
  auto bufP_he = ctx.upload(inP.halfedge_.data(),
                            inP.halfedge_.size() * sizeof(GpuHalfedge));
  auto bufP_fn = upV3(pFaceNormal);
  auto bufQ_vp = upV3(qVertPos);
  auto bufQ_vn = upV3(qVertNormal);
  auto bufQ_he = ctx.upload(inQ.halfedge_.data(),
                            inQ.halfedge_.size() * sizeof(GpuHalfedge));
  auto bufQ_fn = upV3(qFaceNormal);

  // Pack pair list as vec2<i32>.
  Vec<std::array<int32_t, 2>> pairsFlat(nPairs);
  for_each_n(autoPolicy(nPairs, 1e4), countAt(size_t{0}), nPairs,
             [&](size_t i) {
               pairsFlat[i][0] = pairs[i].first;
               pairsFlat[i][1] = pairs[i].second;
             });
  auto bufPairs = ctx.upload(pairsFlat.data(), nPairs * 8);

  // Chunk pairs into submits that fit under Apple Silicon's 2-second GPU
  // TDR. Each emulated-fp64 Kernel12 invocation is expensive; empirically
  // ~2048 pairs per submit stays well under the TDR threshold while
  // keeping submit overhead amortized.
  constexpr size_t kChunkSize = 2048;
  const size_t nChunks = (nPairs + kChunkSize - 1) / kChunkSize;

  Vec<GpuVec3F64> rawV(nPairs);

  for (size_t c = 0; c < nChunks; c++) {
    const size_t off = c * kChunkSize;
    const size_t cnt = std::min(kChunkSize, nPairs - off);

    // New pair buffer scoped to this chunk. Upload a slice.
    auto bufPairsChunk = ctx.upload(pairsFlat.data() + off, cnt * 8);
    auto bufOutXChunk = ctx.allocate(cnt * sizeof(int32_t));
    auto bufOutVChunk = ctx.allocate(cnt * sizeof(GpuVec3F64));
    if (!bufPairsChunk || !bufOutXChunk || !bufOutVChunk) return false;

    K12ParamsCpu p{static_cast<uint32_t>(cnt), expandP ? 1u : 0u,
                   forward ? 1u : 0u, 0u};

    auto batch = ctx.createBatch();
    if (!batch) return false;
    batch->setBuffer(bufP_vp, 0);
    batch->setBuffer(bufP_vn, 1);
    batch->setBuffer(bufP_he, 2);
    batch->setBuffer(bufP_fn, 3);
    batch->setBuffer(bufQ_vp, 4);
    batch->setBuffer(bufQ_vn, 5);
    batch->setBuffer(bufQ_he, 6);
    batch->setBuffer(bufQ_fn, 7);
    batch->setBuffer(bufPairsChunk, 8);
    batch->setBuffer(bufOutXChunk, 9);
    batch->setBuffer(bufOutVChunk, 10);
    batch->setBytes(&p, sizeof(p), 11);
    batch->dispatch(pl, static_cast<uint32_t>((cnt + 63) / 64), 64);
    batch->addReadback(bufOutXChunk, outX.data() + off,
                       cnt * sizeof(int32_t));
    batch->addReadback(bufOutVChunk, rawV.data() + off,
                       cnt * sizeof(GpuVec3F64));
    batch->commitAndWait();
  }

  for_each_n(autoPolicy(nPairs, 1e4), countAt(size_t{0}), nPairs,
             [&](size_t i) { outV[i] = UnpackVec3(rawV[i]); });
  return true;
}

}  // namespace gpu
}  // namespace manifold
