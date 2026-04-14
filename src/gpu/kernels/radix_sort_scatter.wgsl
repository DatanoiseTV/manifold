// Pass 2 of multi-workgroup radix sort: global prefix sum + scatter using
// bin_flags + popcount (VkRadixSort/Embree style).

struct SortParams {
  num_elements: u32,
  shift: u32,
  num_workgroups: u32,
  num_blocks_per_workgroup: u32,
}

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> vals_in: array<i32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals_out: array<i32>;
@group(0) @binding(4) var<storage, read> histograms: array<u32>;
@group(0) @binding(5) var<uniform> params: SortParams;

var<workgroup> global_offsets: array<atomic<u32>, 256>;

struct BinFlags {
  flags: array<atomic<u32>, 8>,  // 256 / 32
}
var<workgroup> bin_flags: array<BinFlags, 256>;

@compute @workgroup_size(256)
fn radix_sort_scatter(
    @builtin(local_invocation_index) lID: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
  let wID = wgid.x;

  // Compute global offset for this workgroup's bin.
  var local_hist_before_wg: u32 = 0u;
  var count: u32 = 0u;
  for (var j: u32 = 0u; j < params.num_workgroups; j = j + 1u) {
    let t = histograms[256u * j + lID];
    if (j == wID) { local_hist_before_wg = count; }
    count = count + t;
  }
  atomicStore(&global_offsets[lID], count);
  workgroupBarrier();

  // Thread 0 does exclusive prefix sum across 256 bins.
  if (lID == 0u) {
    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
      let c = atomicLoad(&global_offsets[i]);
      atomicStore(&global_offsets[i], sum);
      sum = sum + c;
    }
  }
  workgroupBarrier();

  atomicAdd(&global_offsets[lID], local_hist_before_wg);
  workgroupBarrier();

  let flags_bin = lID / 32u;
  let flags_bit = 1u << (lID % 32u);

  for (var index: u32 = 0u; index < params.num_blocks_per_workgroup;
       index = index + 1u) {
    let elementId =
        wID * params.num_blocks_per_workgroup * 256u + index * 256u + lID;

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      atomicStore(&bin_flags[lID].flags[i], 0u);
    }
    workgroupBarrier();

    var element_key: u32 = 0u;
    var element_val: i32 = 0;
    var binID: u32 = 0u;
    var binOffset: u32 = 0u;
    let is_active = elementId < params.num_elements;
    if (is_active) {
      element_key = keys_in[elementId];
      element_val = vals_in[elementId];
      binID = (element_key >> params.shift) & 0xFFu;
      binOffset = atomicLoad(&global_offsets[binID]);
      atomicOr(&bin_flags[binID].flags[flags_bin], flags_bit);
    }
    workgroupBarrier();

    if (is_active) {
      var prefix: u32 = 0u;
      var total: u32 = 0u;
      for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let bits = atomicLoad(&bin_flags[binID].flags[i]);
        let full_count = countOneBits(bits);
        let partial_count = countOneBits(bits & (flags_bit - 1u));
        if (i < flags_bin) { prefix = prefix + full_count; }
        else if (i == flags_bin) { prefix = prefix + partial_count; }
        total = total + full_count;
      }
      keys_out[binOffset + prefix] = element_key;
      vals_out[binOffset + prefix] = element_val;
      if (prefix == total - 1u) {
        atomicAdd(&global_offsets[binID], total);
      }
    }
    workgroupBarrier();
  }
}
