// Pass 1 of multi-workgroup radix sort: per-workgroup histograms.

struct SortParams {
  num_elements: u32,
  shift: u32,
  num_workgroups: u32,
  num_blocks_per_workgroup: u32,
}

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: SortParams;

var<workgroup> local_histogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn radix_sort_histograms(
    @builtin(local_invocation_index) lID: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
  let wID = wgid.x;

  atomicStore(&local_histogram[lID], 0u);
  workgroupBarrier();

  for (var index: u32 = 0u; index < params.num_blocks_per_workgroup;
       index = index + 1u) {
    let elementId =
        wID * params.num_blocks_per_workgroup * 256u + index * 256u + lID;
    if (elementId < params.num_elements) {
      let bin = (keys_in[elementId] >> params.shift) & 0xFFu;
      atomicAdd(&local_histogram[bin], 1u);
    }
  }
  workgroupBarrier();

  atomicStore(&histograms[256u * wID + lID],
              atomicLoad(&local_histogram[lID]));
}
