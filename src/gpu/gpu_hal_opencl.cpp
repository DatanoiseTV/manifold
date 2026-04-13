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
// OpenCL GPU compute backend. Cross-platform (Linux, Windows, macOS).
// On GPUs with cl_khr_fp64, enables fp64 operations on GPU.

#ifdef MANIFOLD_GPU_OPENCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_hal.h"

namespace manifold {
namespace gpu {

// ---------------------------------------------------------------------------
// OpenCL Buffer
// ---------------------------------------------------------------------------
class CLBuffer : public GpuBuffer {
 public:
  CLBuffer(cl_mem mem, size_t size, cl_command_queue queue)
      : mem_(mem), size_(size), queue_(queue) {}
  ~CLBuffer() {
    if (mem_) clReleaseMemObject(mem_);
  }

  size_t size() const override { return size_; }

  void* map() override {
    cl_int err;
    void* ptr = clEnqueueMapBuffer(queue_, mem_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                   0, size_, 0, nullptr, nullptr, &err);
    return (err == CL_SUCCESS) ? ptr : nullptr;
  }

  void unmap() override {
    void* ptr = map();
    if (ptr) clEnqueueUnmapMemObject(queue_, mem_, ptr, 0, nullptr, nullptr);
  }

  void* nativeHandle() override { return (void*)mem_; }
  cl_mem clMem() const { return mem_; }

 private:
  cl_mem mem_;
  size_t size_;
  cl_command_queue queue_;
};

// ---------------------------------------------------------------------------
// OpenCL Pipeline
// ---------------------------------------------------------------------------
class CLPipeline : public GpuPipeline {
 public:
  CLPipeline(cl_kernel kernel, size_t maxWG) : kernel_(kernel), maxWG_(maxWG) {}
  ~CLPipeline() {
    if (kernel_) clReleaseKernel(kernel_);
  }

  uint32_t maxThreadsPerGroup() const override {
    return static_cast<uint32_t>(maxWG_);
  }
  cl_kernel clKernel() const { return kernel_; }

 private:
  cl_kernel kernel_;
  size_t maxWG_;
};

// ---------------------------------------------------------------------------
// OpenCL Command Batch
// ---------------------------------------------------------------------------
class CLCommandBatch : public GpuCommandBatch {
 public:
  CLCommandBatch(cl_command_queue queue) : queue_(queue) {}

  void setBuffer(GpuBufferPtr buf, uint32_t index) override {
    pendingBufArgs_[index] = buf;
  }

  void setBytes(const void* data, size_t length, uint32_t index) override {
    auto vec = std::vector<uint8_t>(length);
    memcpy(vec.data(), data, length);
    pendingByteArgs_[index] = std::move(vec);
  }

  void dispatch(GpuPipelinePtr pipeline, uint32_t threadgroups,
                uint32_t threadsPerGroup) override {
    auto* cp = static_cast<CLPipeline*>(pipeline.get());
    cl_kernel k = cp->clKernel();

    // Set buffer args.
    for (auto& [idx, buf] : pendingBufArgs_) {
      auto* cb = static_cast<CLBuffer*>(buf.get());
      cl_mem m = cb->clMem();
      clSetKernelArg(k, idx, sizeof(cl_mem), &m);
    }
    // Set byte args.
    for (auto& [idx, vec] : pendingByteArgs_) {
      clSetKernelArg(k, idx, vec.size(), vec.data());
    }
    pendingBufArgs_.clear();
    pendingByteArgs_.clear();

    size_t globalSize = (size_t)threadgroups * threadsPerGroup;
    size_t localSize = threadsPerGroup;
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &globalSize, &localSize,
                           0, nullptr, nullptr);
  }

  void barrier() override {
    clEnqueueBarrierWithWaitList(queue_, 0, nullptr, nullptr);
  }

  void commitAndWait() override { clFinish(queue_); }

 private:
  cl_command_queue queue_;
  std::unordered_map<uint32_t, GpuBufferPtr> pendingBufArgs_;
  std::unordered_map<uint32_t, std::vector<uint8_t>> pendingByteArgs_;
};

// ---------------------------------------------------------------------------
// OpenCL Context
// ---------------------------------------------------------------------------
class CLContext : public GpuContext {
 public:
  CLContext() { init(); }
  ~CLContext() {
    pipelineCache_.clear();
    if (queue_) clReleaseCommandQueue(queue_);
    if (program_) clReleaseProgram(program_);
    if (ctx_) clReleaseContext(ctx_);
  }

  Backend backend() const override {
    return device_ ? Backend::OpenCL : Backend::None;
  }

  bool hasFloat64() const override { return hasFp64_; }

  GpuBufferPtr allocate(size_t bytes) override {
    if (!ctx_) return nullptr;
    cl_int err;
    cl_mem m = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS) return nullptr;
    return std::make_shared<CLBuffer>(m, bytes, queue_);
  }

  GpuBufferPtr upload(const void* data, size_t bytes) override {
    if (!ctx_) return nullptr;
    cl_int err;
    cl_mem m = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              bytes, const_cast<void*>(data), &err);
    if (err != CL_SUCCESS) return nullptr;
    return std::make_shared<CLBuffer>(m, bytes, queue_);
  }

  void download(const GpuBufferPtr& buf, void* dst, size_t bytes) override {
    auto* cb = static_cast<CLBuffer*>(buf.get());
    clEnqueueReadBuffer(queue_, cb->clMem(), CL_TRUE, 0, bytes, dst,
                        0, nullptr, nullptr);
  }

  GpuCommandBatchPtr createBatch() override {
    if (!queue_) return nullptr;
    return std::make_unique<CLCommandBatch>(queue_);
  }

  GpuPipelinePtr getPipeline(const std::string& kernelName) override {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = pipelineCache_.find(kernelName);
    if (it != pipelineCache_.end()) return it->second;

    if (!program_) return nullptr;
    cl_int err;
    cl_kernel k = clCreateKernel(program_, kernelName.c_str(), &err);
    if (err != CL_SUCCESS) return nullptr;

    size_t maxWG;
    clGetKernelWorkGroupInfo(k, device_, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(maxWG), &maxWG, nullptr);
    auto pipeline = std::make_shared<CLPipeline>(k, maxWG);
    pipelineCache_[kernelName] = pipeline;
    return pipeline;
  }

  std::string deviceName() const override { return deviceName_; }

 private:
  void init() {
    // Find GPU device.
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (numPlatforms == 0) return;

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    for (auto& p : platforms) {
      cl_uint numDevices;
      if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) !=
              CL_SUCCESS ||
          numDevices == 0)
        continue;

      clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);

      char name[256];
      clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
      deviceName_ = name;

      // Check fp64.
      char ext[4096];
      clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, sizeof(ext), ext, nullptr);
      hasFp64_ = (strstr(ext, "cl_khr_fp64") != nullptr);

      // Create context.
      cl_int err;
      ctx_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
      if (err != CL_SUCCESS) { device_ = nullptr; continue; }

      queue_ = clCreateCommandQueue(ctx_, device_, 0, &err);
      if (err != CL_SUCCESS) {
        clReleaseContext(ctx_); ctx_ = nullptr; device_ = nullptr;
        continue;
      }

      // Load and compile kernel source.
      if (!loadKernels()) {
        clReleaseCommandQueue(queue_); queue_ = nullptr;
        clReleaseContext(ctx_); ctx_ = nullptr;
        device_ = nullptr;
        continue;
      }

      return;  // Found a working GPU.
    }
  }

  bool loadKernels() {
    // Try to find sort.cl next to the executable.
    std::string src = loadFile("sort.cl");
    if (src.empty()) return false;

    const char* srcPtr = src.c_str();
    size_t srcLen = src.size();
    cl_int err;
    program_ = clCreateProgramWithSource(ctx_, 1, &srcPtr, &srcLen, &err);
    if (err != CL_SUCCESS) return false;

    std::string opts = "-cl-std=CL1.2";
    if (hasFp64_) opts += " -DHAS_FP64";

    err = clBuildProgram(program_, 1, &device_, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
      char log[8192];
      clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,
                            sizeof(log), log, nullptr);
      fprintf(stderr, "[manifold GPU] OpenCL build error: %s\n", log);
      clReleaseProgram(program_);
      program_ = nullptr;
      return false;
    }
    return true;
  }

  std::string loadFile(const char* filename) {
    // Search relative to executable, current dir, etc.
    const char* paths[] = {filename, "../Resources/%s", "../share/%s"};
    for (auto& p : paths) {
      char buf[512];
      snprintf(buf, sizeof(buf), p, filename);
      std::ifstream f(buf);
      if (f.good()) {
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
      }
    }
    return "";
  }

  cl_device_id device_ = nullptr;
  cl_context ctx_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  bool hasFp64_ = false;
  std::string deviceName_;
  std::mutex cacheMtx_;
  std::unordered_map<std::string, GpuPipelinePtr> pipelineCache_;
};

namespace {
std::once_flag g_clInitFlag;
CLContext* g_clCtx = nullptr;
}  // namespace

GpuContext& GpuContext::instance() {
  std::call_once(g_clInitFlag, [] { g_clCtx = new CLContext(); });
  return *g_clCtx;
}

}  // namespace gpu
}  // namespace manifold

#endif  // MANIFOLD_GPU_OPENCL
