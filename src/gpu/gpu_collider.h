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

#pragma once

#include <utility>

#include "../impl.h"
#include "../vec.h"

namespace manifold {
class Collider;
namespace gpu {

// Run BVH collision queries on the GPU. Produces (queryIdx, leafIdx) pairs,
// the same set the CPU FindCollision would produce, but in unspecified order.
// Returns false if the GPU path couldn't run; caller should fall back to CPU.
bool FindCollisionsGpu(const Collider& collider,
                       VecView<const Box> queries,
                       bool selfCollision,
                       Vec<std::pair<int, int>>& outPairs);

// Batched variant: run two independent (collider, queries) jobs as a single
// GPU submit. Cuts per-Boolean3 GPU roundtrip count from 2 to 1 when both
// directions (forward/backward Intersect12) are executed in sequence.
bool FindCollisionsGpu2(const Collider& c1, VecView<const Box> q1, bool self1,
                        Vec<std::pair<int, int>>& out1,
                        const Collider& c2, VecView<const Box> q2, bool self2,
                        Vec<std::pair<int, int>>& out2);

// N-way batched collision queries (CollisionJob defined in collider.h for
// friend-declaration reach). Every job is dispatched within a single GPU
// command batch, so M pending Boolean3 operations at a BatchBoolean tree
// level can collect their 2M collision queries and amortize the fixed
// per-submit cost once. Call from the main thread before spawning TBB
// workers so all uploads land on the same Dawn device slot.

}  // namespace gpu
}  // namespace manifold
