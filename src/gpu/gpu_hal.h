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
// GPU compute Hardware Abstraction Layer.
// Provides a thin interface over Metal (macOS) and OpenCL (cross-platform).
// Designed for fused pipeline execution: multiple dispatches per command batch,
// submitted as a single unit to minimize per-dispatch overhead.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace manifold {
namespace gpu {

// Minimum triangle count before GPU dispatch is worthwhile.
// Below this, per-command-buffer overhead (~50-100us) exceeds the gains.
constexpr size_t kGpuSortThreshold = 20000;
constexpr size_t kGpuCollisionThreshold = 5000;

enum class Backend { None, Metal, OpenCL };

// ---------------------------------------------------------------------------
// GpuBuffer — opaque GPU memory handle
// ---------------------------------------------------------------------------
class GpuBuffer {
 public:
  virtual ~GpuBuffer() = default;

  // Size in bytes.
  virtual size_t size() const = 0;

  // Map buffer for CPU read/write. Returns nullptr on failure.
  // The buffer must be unmapped before any GPU dispatch that uses it.
  virtual void* map() = 0;
  virtual void unmap() = 0;

  // Native handle: MTLBuffer* (Metal) or cl_mem (OpenCL).
  virtual void* nativeHandle() = 0;
};

using GpuBufferPtr = std::shared_ptr<GpuBuffer>;

// ---------------------------------------------------------------------------
// GpuPipeline — compiled compute kernel
// ---------------------------------------------------------------------------
class GpuPipeline {
 public:
  virtual ~GpuPipeline() = default;
  virtual uint32_t maxThreadsPerGroup() const = 0;
};

using GpuPipelinePtr = std::shared_ptr<GpuPipeline>;

// ---------------------------------------------------------------------------
// GpuCommandBatch — records multiple dispatches, submitted as one unit
// ---------------------------------------------------------------------------
// This is the key abstraction: instead of submit+wait per primitive,
// an entire pipeline stage (e.g. all of SortGeometry) is recorded into
// one batch and submitted once.
class GpuCommandBatch {
 public:
  virtual ~GpuCommandBatch() = default;

  // Bind a buffer to an argument index for subsequent dispatches.
  virtual void setBuffer(GpuBufferPtr buf, uint32_t index) = 0;

  // Bind inline constant data (small, <4KB) at an argument index.
  virtual void setBytes(const void* data, size_t length, uint32_t index) = 0;

  // Dispatch a compute kernel.
  virtual void dispatch(GpuPipelinePtr pipeline, uint32_t threadgroups,
                        uint32_t threadsPerGroup) = 0;

  // Insert a memory + execution barrier between dispatches within the batch.
  // Required when a later dispatch reads data written by an earlier one.
  virtual void barrier() = 0;

  // Submit the entire batch to the GPU and wait for completion.
  // After this call, all output buffers are safe to read.
  virtual void commitAndWait() = 0;
};

using GpuCommandBatchPtr = std::unique_ptr<GpuCommandBatch>;

// ---------------------------------------------------------------------------
// GpuContext — singleton managing device and resource creation
// ---------------------------------------------------------------------------
class GpuContext {
 public:
  virtual ~GpuContext() = default;

  // Get the singleton instance. Thread-safe, lazy-initialized.
  // Returns the null backend if no GPU is available.
  static GpuContext& instance();

  // Which backend is active.
  virtual Backend backend() const = 0;
  bool isAvailable() const { return backend() != Backend::None; }

  // Does the GPU support native fp64 arithmetic?
  virtual bool hasFloat64() const = 0;

  // Allocate a GPU buffer of the given size (contents undefined).
  virtual GpuBufferPtr allocate(size_t bytes) = 0;

  // Upload CPU data to a new GPU buffer.
  virtual GpuBufferPtr upload(const void* data, size_t bytes) = 0;

  // Download GPU buffer contents to CPU memory.
  virtual void download(const GpuBufferPtr& buf, void* dst,
                        size_t bytes) = 0;

  // Create a new command batch for recording dispatches.
  virtual GpuCommandBatchPtr createBatch() = 0;

  // Get a compiled pipeline by kernel function name.
  // Pipelines are cached — the first call compiles, subsequent calls return
  // the cached version.
  virtual GpuPipelinePtr getPipeline(const std::string& kernelName) = 0;

  // Device info for logging.
  virtual std::string deviceName() const = 0;
};

}  // namespace gpu
}  // namespace manifold
