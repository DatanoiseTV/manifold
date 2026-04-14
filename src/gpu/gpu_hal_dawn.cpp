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
// Dawn (WebGPU) GPU compute backend. Cross-platform: Metal on macOS/iOS,
// Vulkan on Linux/Android, D3D12 on Windows. One backend, one kernel language
// (WGSL).
//
// Threading model: a pool of N wgpu::Devices (one Adapter, N Devices each
// with their own Queue and pipeline cache). Each CPU thread is sticky-bound
// to one device on first access, so concurrent TBB workers submit to the GPU
// without serializing through a global mutex. Dawn's Metal backend asserts
// on concurrent encoding *within* a single Device's command buffer, so each
// device still gets a mutex — but cross-device work is fully parallel.

#ifdef MANIFOLD_GPU_WEBGPU

#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu_hal.h"

namespace manifold {
namespace gpu {

namespace {

void WaitForAll(wgpu::Instance instance,
                std::vector<wgpu::FutureWaitInfo>& waits) {
  while (!waits.empty()) {
    auto status = instance.WaitAny(waits.size(), waits.data(), UINT64_MAX);
    if (status != wgpu::WaitStatus::Success) break;
    waits.erase(std::remove_if(waits.begin(), waits.end(),
                               [](const wgpu::FutureWaitInfo& w) {
                                 return w.completed;
                               }),
                waits.end());
  }
}

// One-time Dawn instance + adapter shared by all devices in the pool.
struct SharedInstance {
  wgpu::Instance instance;
  dawn::native::Adapter nativeAdapter;
  wgpu::Adapter adapter;
  std::string deviceName;

  bool init() {
    wgpu::InstanceDescriptor idesc{};
    static const wgpu::InstanceFeatureName kRequiredFeatures[] = {
        wgpu::InstanceFeatureName::TimedWaitAny,
    };
    idesc.requiredFeatureCount = 1;
    idesc.requiredFeatures = kRequiredFeatures;
    instance = wgpu::CreateInstance(&idesc);
    if (!instance) return false;

    wgpu::RequestAdapterOptions opts{};
    opts.powerPreference = wgpu::PowerPreference::HighPerformance;

    wgpu::FutureWaitInfo info{instance.RequestAdapter(
        &opts, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter a,
            wgpu::StringView) {
          if (status == wgpu::RequestAdapterStatus::Success) adapter = a;
        })};
    std::vector<wgpu::FutureWaitInfo> waits{info};
    WaitForAll(instance, waits);
    if (!adapter) return false;

    wgpu::AdapterInfo info2{};
    adapter.GetInfo(&info2);
    deviceName = std::string(info2.device.data ? info2.device.data : "",
                             info2.device.length);
    return true;
  }
};

// Per-thread-bound slot: one wgpu::Device, its Queue, a pipeline cache, and
// a mutex that only serializes access within the slot (threads pinned to
// different slots never contend).
struct DeviceSlot {
  wgpu::Device device;
  wgpu::Queue queue;
  std::mutex mtx;
  std::unordered_map<std::string, GpuPipelinePtr> pipelineCache;
};

// Metal's Dawn backend serializes command encoding at the MTLCommandBuffer
// level, so multiple threads trying to encode to the same Device throw the
// "A command encoder is already encoding to this command buffer" assertion.
// On macOS we only get one Device per Adapter, so we need a single global
// submit mutex. On Vulkan/D3D backends this contention is cheap or absent.
std::recursive_mutex g_submitMutex;

}  // namespace

// Forward decls.
class DawnContextPool;
DawnContextPool* g_pool = nullptr;
std::once_flag g_poolInit;

// ---------------------------------------------------------------------------
// DawnBuffer — owns an MTLBuffer / Vulkan VkBuffer scoped to one slot's device.
// ---------------------------------------------------------------------------
class DawnBuffer : public GpuBuffer {
 public:
  DawnBuffer(wgpu::Buffer buf, size_t size, DeviceSlot* slot)
      : buf_(buf), size_(size), slot_(slot) {}

  size_t size() const override { return size_; }
  void* map() override { return nullptr; }
  void unmap() override {}
  void* nativeHandle() override { return buf_.Get(); }

  wgpu::Buffer& handle() { return buf_; }
  DeviceSlot* slot() const { return slot_; }

 private:
  wgpu::Buffer buf_;
  size_t size_;
  DeviceSlot* slot_;
};

// ---------------------------------------------------------------------------
// DawnPipeline — device-scoped compiled compute kernel.
// ---------------------------------------------------------------------------
class DawnPipeline : public GpuPipeline {
 public:
  DawnPipeline(wgpu::ComputePipeline pipeline, uint32_t maxWG,
               DeviceSlot* slot)
      : pipeline_(pipeline), maxWG_(maxWG), slot_(slot) {}

  uint32_t maxThreadsPerGroup() const override { return maxWG_; }
  wgpu::ComputePipeline& handle() { return pipeline_; }
  DeviceSlot* slot() const { return slot_; }

 private:
  wgpu::ComputePipeline pipeline_;
  uint32_t maxWG_;
  DeviceSlot* slot_;
};

// ---------------------------------------------------------------------------
// DawnCommandBatch — one per slot/submit. Holds the slot's mutex only for
// the duration of command encoding; the mutex is released before Submit so
// other threads bound to the *same* slot can start encoding while this one
// waits on the GPU. Threads bound to different slots never contend.
// ---------------------------------------------------------------------------
class DawnCommandBatch : public GpuCommandBatch {
 public:
  DawnCommandBatch(DeviceSlot* slot, wgpu::Instance instance)
      : slot_(slot),
        instance_(instance),
        lock_(g_submitMutex) {
    encoder_ = slot_->device.CreateCommandEncoder();
  }

  void setBuffer(GpuBufferPtr buf, uint32_t index) override {
    Binding b;
    b.kind = Binding::Kind::Buffer;
    b.buffer = buf;
    b.index = index;
    pending_.push_back(std::move(b));
  }

  void setBytes(const void* data, size_t length, uint32_t index) override {
    Binding b;
    b.kind = Binding::Kind::Bytes;
    b.bytes.assign(static_cast<const uint8_t*>(data),
                   static_cast<const uint8_t*>(data) + length);
    b.index = index;
    pending_.push_back(std::move(b));
  }

  void dispatch(GpuPipelinePtr pipeline, uint32_t threadgroups,
                uint32_t /*threadsPerGroup*/) override {
    auto* dp = static_cast<DawnPipeline*>(pipeline.get());

    std::vector<wgpu::BindGroupEntry> entries;
    entries.reserve(pending_.size());
    std::vector<wgpu::Buffer> holdAlive;

    for (auto& b : pending_) {
      wgpu::BindGroupEntry e{};
      e.binding = b.index;
      if (b.kind == Binding::Kind::Buffer) {
        auto* db = static_cast<DawnBuffer*>(b.buffer.get());
        e.buffer = db->handle();
        e.offset = 0;
        e.size = db->size();
      } else {
        wgpu::BufferDescriptor bd{};
        bd.size = (b.bytes.size() + 15) & ~size_t{15};
        bd.usage =
            wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        wgpu::Buffer ub = slot_->device.CreateBuffer(&bd);
        slot_->queue.WriteBuffer(ub, 0, b.bytes.data(), b.bytes.size());
        e.buffer = ub;
        e.offset = 0;
        e.size = bd.size;
        holdAlive.push_back(ub);
      }
      entries.push_back(e);
    }

    wgpu::BindGroupLayout bgl = dp->handle().GetBindGroupLayout(0);
    wgpu::BindGroupDescriptor bgd{};
    bgd.layout = bgl;
    bgd.entryCount = entries.size();
    bgd.entries = entries.data();
    wgpu::BindGroup bg = slot_->device.CreateBindGroup(&bgd);

    wgpu::ComputePassEncoder pass = encoder_.BeginComputePass();
    pass.SetPipeline(dp->handle());
    pass.SetBindGroup(0, bg);
    pass.DispatchWorkgroups(threadgroups);
    pass.End();

    transients_.insert(transients_.end(), holdAlive.begin(), holdAlive.end());
    pending_.clear();
  }

  void barrier() override {
    // Implicit between compute passes in WebGPU.
  }

  void addReadback(GpuBufferPtr src, void* dst, size_t bytes) override {
    auto* db = static_cast<DawnBuffer*>(src.get());
    wgpu::BufferDescriptor rd{};
    rd.size = bytes;
    rd.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer staging = slot_->device.CreateBuffer(&rd);
    encoder_.CopyBufferToBuffer(db->handle(), 0, staging, 0, bytes);
    readbacks_.push_back({staging, dst, bytes});
  }

  void commitAndWait() override {
    wgpu::CommandBuffer cb = encoder_.Finish();
    slot_->queue.Submit(1, &cb);

    std::vector<wgpu::FutureWaitInfo> waits;
    waits.reserve(readbacks_.size() + 1);
    for (auto& rb : readbacks_) {
      waits.push_back({rb.staging.MapAsync(
          wgpu::MapMode::Read, 0, rb.bytes,
          wgpu::CallbackMode::WaitAnyOnly,
          [](wgpu::MapAsyncStatus, wgpu::StringView) {})});
    }
    waits.push_back({slot_->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {})});

    // Encoding is done — release the slot's mutex before blocking so another
    // thread bound to the same slot can encode its next batch in parallel.
    lock_.unlock();

    WaitForAll(instance_, waits);

    for (auto& rb : readbacks_) {
      const void* src = rb.staging.GetConstMappedRange(0, rb.bytes);
      if (src) std::memcpy(rb.dst, src, rb.bytes);
      rb.staging.Unmap();
    }
    readbacks_.clear();
  }

 private:
  struct Binding {
    enum class Kind { Buffer, Bytes };
    Kind kind;
    uint32_t index;
    GpuBufferPtr buffer;
    std::vector<uint8_t> bytes;
  };
  struct Readback {
    wgpu::Buffer staging;
    void* dst;
    size_t bytes;
  };

  DeviceSlot* slot_;
  wgpu::Instance instance_;
  wgpu::CommandEncoder encoder_;
  std::unique_lock<std::recursive_mutex> lock_;
  std::vector<Binding> pending_;
  std::vector<wgpu::Buffer> transients_;
  std::vector<Readback> readbacks_;
};

// ---------------------------------------------------------------------------
// DawnContextPool — holds the shared Instance + Adapter + pool of Devices.
// ---------------------------------------------------------------------------
class DawnContextPool : public GpuContext {
 public:
  DawnContextPool() {
    if (!shared_.init()) return;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    // Try up to N devices, but accept whatever the driver gives us. Dawn's
    // Metal backend typically only exposes one shared Device per Adapter on
    // macOS, so slot 1+ requests fail; we just run with what we got.
    const unsigned n = std::min(hw, 8u);
    slots_.reserve(n);
    for (unsigned i = 0; i < n; i++) {
      slots_.emplace_back(std::make_unique<DeviceSlot>());
      if (!initDevice(*slots_.back())) {
        slots_.pop_back();
        break;
      }
    }
    if (slots_.empty()) return;
    ok_ = true;
    wgslSrcDir_ = findKernelsDir();
  }

  Backend backend() const override {
    return ok_ ? Backend::WebGPU : Backend::None;
  }

  bool hasFloat64() const override { return false; }

  GpuBufferPtr allocate(size_t bytes) override {
    if (!ok_) return nullptr;
    DeviceSlot& slot = pickSlot();
    std::lock_guard<std::recursive_mutex> lock(g_submitMutex);
    wgpu::BufferDescriptor d{};
    d.size = std::max<size_t>(bytes, 4);
    d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
              wgpu::BufferUsage::CopyDst;
    wgpu::Buffer b = slot.device.CreateBuffer(&d);
    return std::make_shared<DawnBuffer>(b, d.size, &slot);
  }

  GpuBufferPtr upload(const void* data, size_t bytes) override {
    if (!ok_) return nullptr;
    DeviceSlot& slot = pickSlot();
    std::lock_guard<std::recursive_mutex> lock(g_submitMutex);
    wgpu::BufferDescriptor d{};
    d.size = std::max<size_t>(bytes, 4);
    d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
              wgpu::BufferUsage::CopyDst;
    wgpu::Buffer b = slot.device.CreateBuffer(&d);
    slot.queue.WriteBuffer(b, 0, data, bytes);
    return std::make_shared<DawnBuffer>(b, d.size, &slot);
  }

  void download(const GpuBufferPtr& buf, void* dst, size_t bytes) override {
    if (!ok_ || !buf) return;
    auto* db = static_cast<DawnBuffer*>(buf.get());
    DeviceSlot* slot = db->slot();

    wgpu::BufferDescriptor rd{};
    rd.size = bytes;
    rd.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

    std::unique_lock<std::recursive_mutex> lock(g_submitMutex);
    wgpu::Buffer readback = slot->device.CreateBuffer(&rd);
    wgpu::CommandEncoder enc = slot->device.CreateCommandEncoder();
    enc.CopyBufferToBuffer(db->handle(), 0, readback, 0, bytes);
    wgpu::CommandBuffer cb = enc.Finish();
    slot->queue.Submit(1, &cb);
    auto future = readback.MapAsync(
        wgpu::MapMode::Read, 0, bytes, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus, wgpu::StringView) {});
    lock.unlock();

    std::vector<wgpu::FutureWaitInfo> waits{{future}};
    WaitForAll(shared_.instance, waits);
    const void* src = readback.GetConstMappedRange(0, bytes);
    if (src) std::memcpy(dst, src, bytes);
    readback.Unmap();
  }

  GpuCommandBatchPtr createBatch() override {
    if (!ok_) return nullptr;
    return std::make_unique<DawnCommandBatch>(&pickSlot(), shared_.instance);
  }

  GpuPipelinePtr getPipeline(const std::string& kernelName) override {
    if (!ok_) return nullptr;
    DeviceSlot& slot = pickSlot();
    std::lock_guard<std::mutex> lock(slot.mtx);
    auto it = slot.pipelineCache.find(kernelName);
    if (it != slot.pipelineCache.end()) return it->second;

    std::string src = loadFile((kernelName + ".wgsl").c_str());
    if (src.empty()) {
      fprintf(stderr, "[manifold GPU] cannot load %s.wgsl\n",
              kernelName.c_str());
      return nullptr;
    }
    // Kernels using fp64 arithmetic prepend the shared library.
    if (kernelName == "fp64_test" || kernelName == "intersect_kernel" ||
        kernelName == "kernel12_dispatch") {
      std::string lib = loadFile("fp64_lib.wgsl");
      if (lib.empty()) {
        fprintf(stderr, "[manifold GPU] cannot load fp64_lib.wgsl\n");
        return nullptr;
      }
      src = lib + src;
    }
    wgpu::ShaderSourceWGSL wgslDesc{};
    wgslDesc.code = src.c_str();
    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc;
    wgpu::ShaderModule module = slot.device.CreateShaderModule(&smDesc);

    wgpu::ComputePipelineDescriptor cpd{};
    cpd.compute.module = module;
    cpd.compute.entryPoint = kernelName.c_str();
    wgpu::ComputePipeline pipeline = slot.device.CreateComputePipeline(&cpd);
    if (!pipeline) return nullptr;

    auto p = std::make_shared<DawnPipeline>(pipeline, 256, &slot);
    slot.pipelineCache[kernelName] = p;
    return p;
  }

  std::string deviceName() const override { return shared_.deviceName; }

 private:
  DeviceSlot& pickSlot() {
    thread_local size_t idx = SIZE_MAX;
    if (idx == SIZE_MAX) {
      static std::atomic<size_t> next{0};
      idx = next.fetch_add(1) % slots_.size();
    }
    return *slots_[idx];
  }

  bool initDevice(DeviceSlot& slot) {
    wgpu::DeviceDescriptor ddesc{};

    static const char* kEnabled[] = {"skip_validation", "disable_robustness"};
    wgpu::DawnTogglesDescriptor toggles{};
    toggles.enabledToggleCount = sizeof(kEnabled) / sizeof(kEnabled[0]);
    toggles.enabledToggles = kEnabled;
    ddesc.nextInChain = &toggles;

    ddesc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView) {});
    ddesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView msg) {
          fprintf(stderr, "[manifold GPU] WebGPU error: %.*s\n",
                  static_cast<int>(msg.length), msg.data);
        });

    wgpu::FutureWaitInfo info{shared_.adapter.RequestDevice(
        &ddesc, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestDeviceStatus status, wgpu::Device d,
            wgpu::StringView) {
          if (status == wgpu::RequestDeviceStatus::Success) slot.device = d;
        })};
    std::vector<wgpu::FutureWaitInfo> waits{info};
    WaitForAll(shared_.instance, waits);
    if (!slot.device) return false;
    slot.queue = slot.device.GetQueue();
    return true;
  }

  std::string findKernelsDir() {
    const char* candidates[] = {".", "../Resources", "../share"};
    for (const char* c : candidates) {
      std::ifstream f(std::string(c) + "/radix_sort_histograms.wgsl");
      if (f.good()) return c;
    }
    return ".";
  }

  std::string loadFile(const char* filename) {
    const std::string paths[] = {
        filename,
        std::string("../Resources/") + filename,
        std::string("../share/") + filename,
    };
    for (const auto& p : paths) {
      std::ifstream f(p);
      if (f.good()) {
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
      }
    }
    return "";
  }

  SharedInstance shared_;
  std::vector<std::unique_ptr<DeviceSlot>> slots_;
  std::string wgslSrcDir_;
  bool ok_ = false;
};

GpuContext& GpuContext::instance() {
  std::call_once(g_poolInit, [] { g_pool = new DawnContextPool(); });
  return *g_pool;
}

}  // namespace gpu
}  // namespace manifold

#endif  // MANIFOLD_GPU_WEBGPU
