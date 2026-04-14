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

#include "gpu_hal.h"

// Null backend: used when no GPU is available or MANIFOLD_GPU is off.
// All methods return failure/empty. This file always compiles on every
// platform.

#if !defined(MANIFOLD_GPU_METAL) && !defined(MANIFOLD_GPU_OPENCL) && \
    !defined(MANIFOLD_GPU_WEBGPU)

namespace manifold {
namespace gpu {

namespace {

class NullContext : public GpuContext {
 public:
  Backend backend() const override { return Backend::None; }
  bool hasFloat64() const override { return false; }
  GpuBufferPtr allocate(size_t) override { return nullptr; }
  GpuBufferPtr upload(const void*, size_t) override { return nullptr; }
  void download(const GpuBufferPtr&, void*, size_t) override {}
  GpuCommandBatchPtr createBatch() override { return nullptr; }
  GpuPipelinePtr getPipeline(const std::string&) override { return nullptr; }
  std::string deviceName() const override { return "none"; }
};

// Null batch is never constructed (createBatch returns nullptr), but the
// abstract base still needs addReadback defined somewhere — inline via the
// base class default. Provided for builds that link the null backend
// alongside addReadback-calling code.

NullContext g_nullContext;

}  // namespace

GpuContext& GpuContext::instance() { return g_nullContext; }

}  // namespace gpu
}  // namespace manifold

#endif  // !MANIFOLD_GPU_METAL && !MANIFOLD_GPU_OPENCL
