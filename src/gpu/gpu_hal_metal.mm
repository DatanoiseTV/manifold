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
// Metal GPU compute backend for macOS / Apple Silicon.

#ifdef MANIFOLD_GPU_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>
#include <string>
#include <unordered_map>

#include "gpu_hal.h"

namespace manifold {
namespace gpu {

// ---------------------------------------------------------------------------
// MetalBuffer
// ---------------------------------------------------------------------------
class MetalBuffer : public GpuBuffer {
 public:
  MetalBuffer(id<MTLBuffer> buf) : buf_(buf) {}

  size_t size() const override { return [buf_ length]; }

  void* map() override { return [buf_ contents]; }

  void unmap() override {
    // On shared memory (Apple Silicon), no-op. On discrete GPU,
    // would need didModifyRange: — but manifold targets Apple Silicon.
  }

  void* nativeHandle() override { return (__bridge void*)buf_; }

  id<MTLBuffer> mtlBuffer() const { return buf_; }

 private:
  id<MTLBuffer> buf_;
};

// ---------------------------------------------------------------------------
// MetalPipeline
// ---------------------------------------------------------------------------
class MetalPipeline : public GpuPipeline {
 public:
  MetalPipeline(id<MTLComputePipelineState> pso) : pso_(pso) {}

  uint32_t maxThreadsPerGroup() const override {
    return static_cast<uint32_t>([pso_ maxTotalThreadsPerThreadgroup]);
  }

  id<MTLComputePipelineState> pso() const { return pso_; }

 private:
  id<MTLComputePipelineState> pso_;
};

// ---------------------------------------------------------------------------
// MetalCommandBatch
// ---------------------------------------------------------------------------
class MetalCommandBatch : public GpuCommandBatch {
 public:
  MetalCommandBatch(id<MTLCommandQueue> queue) : queue_(queue) {
    cmdBuf_ = [queue_ commandBuffer];
    encoder_ = nil;
  }

  void setBuffer(GpuBufferPtr buf, uint32_t index) override {
    ensureEncoder();
    auto* mb = static_cast<MetalBuffer*>(buf.get());
    [encoder_ setBuffer:mb->mtlBuffer() offset:0 atIndex:index];
  }

  void setBytes(const void* data, size_t length, uint32_t index) override {
    ensureEncoder();
    [encoder_ setBytes:data length:length atIndex:index];
  }

  void dispatch(GpuPipelinePtr pipeline, uint32_t threadgroups,
                uint32_t threadsPerGroup) override {
    ensureEncoder();
    auto* mp = static_cast<MetalPipeline*>(pipeline.get());
    [encoder_ setComputePipelineState:mp->pso()];
    [encoder_ dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
  }

  void barrier() override {
    // End current encoder and start a new one.
    // This ensures all writes from previous dispatches are visible.
    if (encoder_) {
      [encoder_ endEncoding];
      encoder_ = nil;
    }
  }

  void commitAndWait() override {
    if (encoder_) {
      [encoder_ endEncoding];
      encoder_ = nil;
    }
    [cmdBuf_ commit];
    [cmdBuf_ waitUntilCompleted];
  }

 private:
  void ensureEncoder() {
    if (!encoder_) {
      encoder_ = [cmdBuf_ computeCommandEncoder];
    }
  }

  id<MTLCommandQueue> queue_;
  id<MTLCommandBuffer> cmdBuf_;
  id<MTLComputeCommandEncoder> encoder_;
};

// ---------------------------------------------------------------------------
// MetalContext
// ---------------------------------------------------------------------------
class MetalContext : public GpuContext {
 public:
  MetalContext() { init(); }

  Backend backend() const override {
    return device_ ? Backend::Metal : Backend::None;
  }

  bool hasFloat64() const override {
    // Apple Silicon GPUs do not have native fp64.
    return false;
  }

  GpuBufferPtr allocate(size_t bytes) override {
    if (!device_) return nullptr;
    @autoreleasepool {
      id<MTLBuffer> buf =
          [device_ newBufferWithLength:bytes
                               options:MTLResourceStorageModeShared];
      if (!buf) return nullptr;
      return std::make_shared<MetalBuffer>(buf);
    }
  }

  GpuBufferPtr upload(const void* data, size_t bytes) override {
    if (!device_) return nullptr;
    @autoreleasepool {
      id<MTLBuffer> buf =
          [device_ newBufferWithBytes:data
                               length:bytes
                              options:MTLResourceStorageModeShared];
      if (!buf) return nullptr;
      return std::make_shared<MetalBuffer>(buf);
    }
  }

  void download(const GpuBufferPtr& buf, void* dst, size_t bytes) override {
    // On Apple Silicon with shared memory, the data is already CPU-accessible.
    auto* mb = static_cast<MetalBuffer*>(buf.get());
    memcpy(dst, [mb->mtlBuffer() contents], bytes);
  }

  GpuCommandBatchPtr createBatch() override {
    if (!queue_) return nullptr;
    return std::make_unique<MetalCommandBatch>(queue_);
  }

  GpuPipelinePtr getPipeline(const std::string& kernelName) override {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = pipelineCache_.find(kernelName);
    if (it != pipelineCache_.end()) return it->second;

    @autoreleasepool {
      NSString* name =
          [NSString stringWithUTF8String:kernelName.c_str()];
      id<MTLFunction> fn = [library_ newFunctionWithName:name];
      if (!fn) return nullptr;

      NSError* error = nil;
      id<MTLComputePipelineState> pso =
          [device_ newComputePipelineStateWithFunction:fn error:&error];
      if (!pso) {
        NSLog(@"[manifold GPU] Failed to create pipeline '%s': %@",
              kernelName.c_str(), error);
        return nullptr;
      }
      auto pipeline = std::make_shared<MetalPipeline>(pso);
      pipelineCache_[kernelName] = pipeline;
      return pipeline;
    }
  }

  std::string deviceName() const override {
    if (!device_) return "unavailable";
    return std::string([[device_ name] UTF8String]);
  }

 private:
  void init() {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (!device_) return;

      queue_ = [device_ newCommandQueue];
      if (!queue_) {
        device_ = nil;
        return;
      }

      // Try loading pre-compiled metallib from bundle or working directory.
      NSError* error = nil;
      NSBundle* bundle = [NSBundle mainBundle];
      NSString* libPath =
          [bundle pathForResource:@"sort" ofType:@"metallib"];

      if (libPath) {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        library_ = [device_ newLibraryWithURL:url error:&error];
      }

      // Fallback: compile from source at runtime.
      if (!library_) {
        NSString* srcPath =
            [bundle pathForResource:@"sort" ofType:@"metal"];
        if (!srcPath) {
          // Search relative to executable.
          NSString* execDir =
              [[[NSProcessInfo processInfo].arguments firstObject]
                  stringByDeletingLastPathComponent];
          NSArray* paths = @[
            [execDir stringByAppendingPathComponent:@"sort.metal"],
            [execDir
                stringByAppendingPathComponent:@"../Resources/sort.metal"],
          ];
          for (NSString* p in paths) {
            if ([[NSFileManager defaultManager] fileExistsAtPath:p]) {
              srcPath = p;
              break;
            }
          }
        }
        if (srcPath) {
          NSString* src =
              [NSString stringWithContentsOfFile:srcPath
                                       encoding:NSUTF8StringEncoding
                                          error:&error];
          if (src) {
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.fastMathEnabled = YES;
            library_ =
                [device_ newLibraryWithSource:src options:opts error:&error];
          }
        }
      }

      if (!library_) {
        NSLog(@"[manifold GPU] Metal shader library not found — GPU "
              @"acceleration disabled");
        queue_ = nil;
        device_ = nil;
      }
    }
  }

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  std::mutex cacheMtx_;
  std::unordered_map<std::string, GpuPipelinePtr> pipelineCache_;
};

namespace {
std::once_flag g_metalInitFlag;
MetalContext* g_metalCtx = nullptr;
}  // namespace

GpuContext& GpuContext::instance() {
  std::call_once(g_metalInitFlag, [] { g_metalCtx = new MetalContext(); });
  return *g_metalCtx;
}

}  // namespace gpu
}  // namespace manifold

#endif  // MANIFOLD_GPU_METAL
