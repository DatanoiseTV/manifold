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
// GPU-side BVH collision query. Feeds collision pairs back to CPU Boolean3,
// where fp64 Kernel11/02/12 math runs as today.

#include "gpu_collider.h"

#include <cstring>
#include <limits>

#include "../collider.h"
#include "../parallel.h"
#include "gpu_hal.h"

namespace manifold {
namespace gpu {

namespace {

// 32-byte, 16-aligned Box matching the WGSL layout:
//   struct Box { min: vec3<f32>, _pad0: f32, max: vec3<f32>, _pad1: f32 }
struct GpuBox {
  float min[3];
  float _pad0;
  float max[3];
  float _pad1;
};
static_assert(sizeof(GpuBox) == 32, "GpuBox must be 32 bytes");

GpuBox ToGpuBox(const Box& b) {
  GpuBox out{};
  // Inflate slightly so fp32 precision loss never under-reports overlaps.
  // Downstream fp64 math re-tests on CPU so extra candidates cost nothing.
  const double kEps = 1e-6;
  out.min[0] = static_cast<float>(b.min.x - kEps);
  out.min[1] = static_cast<float>(b.min.y - kEps);
  out.min[2] = static_cast<float>(b.min.z - kEps);
  out.max[0] = static_cast<float>(b.max.x + kEps);
  out.max[1] = static_cast<float>(b.max.y + kEps);
  out.max[2] = static_cast<float>(b.max.z + kEps);
  return out;
}

void WriteEmpty(GpuBox& out) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  out.min[0] = kInf;
  out.min[1] = kInf;
  out.min[2] = kInf;
  out.max[0] = -kInf;
  out.max[1] = -kInf;
  out.max[2] = -kInf;
}

void PackBoxes(VecView<const Box> src, Vec<GpuBox>& dst) {
  const size_t n = src.size();
  dst.resize_nofill(n);
  for_each_n(autoPolicy(n, 1e4), countAt(size_t{0}), n, [&](size_t i) {
    const Box& b = src[i];
    if (b.min.x == std::numeric_limits<double>::infinity()) {
      WriteEmpty(dst[i]);
    } else {
      dst[i] = ToGpuBox(b);
    }
  });
}

struct JobBuffers {
  GpuBufferPtr queries;
  GpuBufferPtr nodes;
  GpuBufferPtr children;
  GpuBufferPtr pairs;
  GpuBufferPtr count;
  uint32_t capacity;
  uint32_t num_queries;
  // Host-side readback destinations; filled by the batch.
  uint32_t host_count = 0;
  // Pre-sized to capacity; the batch readback fills it up to host_count;
  // we truncate after.
  std::vector<std::pair<int, int>> host_pairs;
};

struct Params {
  uint32_t num_queries;
  uint32_t self_collision;
  uint32_t output_capacity;
  uint32_t _pad;
};

// Upload packed buffers for one (BVH, queries) job. Returns false on
// any allocation failure. Collider private members are extracted by the
// friended public wrappers below and passed in as VecView.
bool PrepareJob(GpuContext& ctx,
                VecView<const Box> nodeBBox,
                VecView<const std::pair<int, int>> internalChildren,
                VecView<const Box> queries, JobBuffers& out) {
  const size_t numNodes = nodeBBox.size();
  const size_t numInternal = internalChildren.size();
  const size_t numQueries = queries.size();
  if (numQueries == 0 || numNodes == 0 || numInternal == 0) {
    out.num_queries = 0;
    return true;  // no-op job
  }

  Vec<GpuBox> packedNodes, packedQueries;
  PackBoxes(nodeBBox, packedNodes);
  PackBoxes(queries, packedQueries);

  out.num_queries = static_cast<uint32_t>(numQueries);
  out.capacity = static_cast<uint32_t>(numQueries * 8 + 1024);
  out.queries = ctx.upload(packedQueries.data(), numQueries * sizeof(GpuBox));
  out.nodes = ctx.upload(packedNodes.data(), numNodes * sizeof(GpuBox));
  out.children = ctx.upload(internalChildren.data(),
                            numInternal * sizeof(std::pair<int, int>));
  out.pairs = ctx.allocate(out.capacity * sizeof(int32_t) * 2);
  uint32_t zero = 0;
  out.count = ctx.upload(&zero, sizeof(zero));
  return out.queries && out.nodes && out.children && out.pairs && out.count;
}

void BindJob(GpuCommandBatch& batch, const JobBuffers& j, bool selfCollision) {
  Params params{j.num_queries, selfCollision ? 1u : 0u, j.capacity, 0};
  batch.setBuffer(j.queries, 0);
  batch.setBuffer(j.nodes, 1);
  batch.setBuffer(j.children, 2);
  batch.setBuffer(j.pairs, 3);
  batch.setBuffer(j.count, 4);
  batch.setBytes(&params, sizeof(params), 5);
}

// Register the count and pairs buffers for in-batch readback, so they land in
// host memory as part of the single commitAndWait() submit.
void QueueReadback(GpuCommandBatch& batch, JobBuffers& j) {
  if (j.num_queries == 0) return;
  batch.addReadback(j.count, &j.host_count, sizeof(uint32_t));
  j.host_pairs.resize(j.capacity);
  batch.addReadback(j.pairs, j.host_pairs.data(),
                    j.capacity * sizeof(std::pair<int, int>));
}

bool FinalizeJob(JobBuffers& j, Vec<std::pair<int, int>>& out) {
  if (j.num_queries == 0) {
    out.clear();
    return true;
  }
  if (j.host_count > j.capacity) return false;  // overflow → CPU fallback
  out.resize_nofill(j.host_count);
  if (j.host_count > 0) {
    std::memcpy(out.data(), j.host_pairs.data(),
                j.host_count * sizeof(std::pair<int, int>));
  }
  return true;
}

}  // namespace

bool FindCollisionsGpu(const Collider& collider,
                       VecView<const Box> queries,
                       bool selfCollision,
                       Vec<std::pair<int, int>>& outPairs) {
  auto& ctx = GpuContext::instance();
  if (!ctx.isAvailable()) return false;

  auto findPL = ctx.getPipeline("find_collisions");
  if (!findPL) return false;

  JobBuffers job;
  if (!PrepareJob(ctx, {collider.nodeBBox_.data(), collider.nodeBBox_.size()},
                  {collider.internalChildren_.data(),
                   collider.internalChildren_.size()},
                  queries, job))
    return false;
  if (job.num_queries == 0) {
    outPairs.clear();
    return true;
  }

  auto batch = ctx.createBatch();
  if (!batch) return false;
  BindJob(*batch, job, selfCollision);
  uint32_t groups = (job.num_queries + 255) / 256;
  batch->dispatch(findPL, groups, 256);
  QueueReadback(*batch, job);
  batch->commitAndWait();

  return FinalizeJob(job, outPairs);
}

bool FindCollisionsGpuN(const std::vector<CollisionJob>& jobs) {
  if (jobs.empty()) return true;
  auto& ctx = GpuContext::instance();
  if (!ctx.isAvailable()) return false;
  auto findPL = ctx.getPipeline("find_collisions");
  if (!findPL) return false;

  std::vector<JobBuffers> bufs(jobs.size());
  for (size_t i = 0; i < jobs.size(); i++) {
    const Collider& c = *jobs[i].collider;
    if (!PrepareJob(ctx, {c.nodeBBox_.data(), c.nodeBBox_.size()},
                    {c.internalChildren_.data(),
                     c.internalChildren_.size()},
                    jobs[i].queries, bufs[i]))
      return false;
  }

  auto batch = ctx.createBatch();
  if (!batch) return false;
  for (size_t i = 0; i < jobs.size(); i++) {
    if (bufs[i].num_queries == 0) continue;
    BindJob(*batch, bufs[i], jobs[i].selfCollision);
    uint32_t groups = (bufs[i].num_queries + 255) / 256;
    batch->dispatch(findPL, groups, 256);
  }
  for (size_t i = 0; i < jobs.size(); i++) QueueReadback(*batch, bufs[i]);
  batch->commitAndWait();

  bool ok = true;
  for (size_t i = 0; i < jobs.size(); i++) {
    if (!FinalizeJob(bufs[i], *jobs[i].out)) ok = false;
  }
  return ok;
}

bool FindCollisionsGpu2(const Collider& c1, VecView<const Box> q1, bool self1,
                        Vec<std::pair<int, int>>& out1,
                        const Collider& c2, VecView<const Box> q2, bool self2,
                        Vec<std::pair<int, int>>& out2) {
  auto& ctx = GpuContext::instance();
  if (!ctx.isAvailable()) return false;

  auto findPL = ctx.getPipeline("find_collisions");
  if (!findPL) return false;

  JobBuffers j1, j2;
  if (!PrepareJob(ctx, {c1.nodeBBox_.data(), c1.nodeBBox_.size()},
                  {c1.internalChildren_.data(), c1.internalChildren_.size()},
                  q1, j1))
    return false;
  if (!PrepareJob(ctx, {c2.nodeBBox_.data(), c2.nodeBBox_.size()},
                  {c2.internalChildren_.data(), c2.internalChildren_.size()},
                  q2, j2))
    return false;

  auto batch = ctx.createBatch();
  if (!batch) return false;

  if (j1.num_queries > 0) {
    BindJob(*batch, j1, self1);
    uint32_t groups1 = (j1.num_queries + 255) / 256;
    batch->dispatch(findPL, groups1, 256);
  }
  if (j2.num_queries > 0) {
    BindJob(*batch, j2, self2);
    uint32_t groups2 = (j2.num_queries + 255) / 256;
    batch->dispatch(findPL, groups2, 256);
  }
  QueueReadback(*batch, j1);
  QueueReadback(*batch, j2);
  batch->commitAndWait();

  bool ok1 = FinalizeJob(j1, out1);
  bool ok2 = FinalizeJob(j2, out2);
  return ok1 && ok2;
}

}  // namespace gpu
}  // namespace manifold
