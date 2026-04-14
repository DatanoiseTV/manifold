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

#include <array>
#include <utility>

#include "../impl.h"
#include "../vec.h"

namespace manifold {
namespace gpu {

// Runs Kernel12(a1, b2) on GPU for each (a1, b2) pair in `pairs`. Matches the
// CPU Kernel12<expandP, forward> output. Returns false if the GPU path
// couldn't execute; caller should fall back to CPU math.
//
// Outputs:
//   x12[i]  = Kernel12's x12 for pairs[i] (int winding direction)
//   v12[i]  = Kernel12's v12 for pairs[i] (intersection point; NaN = no hit)
bool IntersectGpu(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                  VecView<const std::pair<int, int>> pairs,
                  bool expandP, bool forward,
                  Vec<int>& outX, Vec<vec3>& outV);

}  // namespace gpu
}  // namespace manifold
