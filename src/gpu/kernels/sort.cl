// OpenCL radix sort + BVH tree construction kernels.
// Port of the Metal kernels (sort.metal) to OpenCL C.
// Based on VkRadixSort by Mirco Werner (MIT License).

#define WORKGROUP_SIZE 256
#define RADIX_SORT_BINS 256

// ---- Pass 1: Per-workgroup histograms ----
__kernel void radix_sort_histograms(
    __global const uint* keys_in,
    __global uint* histograms,
    const uint g_num_elements,
    const uint g_shift,
    const uint g_num_blocks_per_workgroup) {

  uint lID = get_local_id(0);
  uint wID = get_group_id(0);

  __local uint local_histogram[RADIX_SORT_BINS];
  if (lID < RADIX_SORT_BINS) local_histogram[lID] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint index = 0; index < g_num_blocks_per_workgroup; index++) {
    uint elementId = wID * g_num_blocks_per_workgroup * WORKGROUP_SIZE +
                     index * WORKGROUP_SIZE + lID;
    if (elementId < g_num_elements) {
      uint bin = (keys_in[elementId] >> g_shift) & 0xFFu;
      atomic_inc(&local_histogram[bin]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lID < RADIX_SORT_BINS) {
    histograms[RADIX_SORT_BINS * wID + lID] = local_histogram[lID];
  }
}

// ---- Pass 2: Global prefix sum + scatter ----
__kernel void radix_sort_scatter(
    __global const uint* keys_in,
    __global const int* vals_in,
    __global uint* keys_out,
    __global int* vals_out,
    __global uint* histograms,
    const uint g_num_elements,
    const uint g_shift,
    const uint g_num_workgroups,
    const uint g_num_blocks_per_workgroup) {

  uint lID = get_local_id(0);
  uint wID = get_group_id(0);

  __local uint global_offsets[RADIX_SORT_BINS];

  // Compute global offsets from per-workgroup histograms.
  uint local_histogram = 0;
  if (lID < RADIX_SORT_BINS) {
    uint count = 0;
    for (uint j = 0; j < g_num_workgroups; j++) {
      uint t = histograms[RADIX_SORT_BINS * j + lID];
      local_histogram = (j == wID) ? count : local_histogram;
      count += t;
    }
    global_offsets[lID] = count;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Sequential exclusive prefix sum across bins (thread 0).
  if (lID == 0) {
    uint sum = 0;
    for (uint i = 0; i < RADIX_SORT_BINS; i++) {
      uint c = global_offsets[i];
      global_offsets[i] = sum;
      sum += c;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lID < RADIX_SORT_BINS) {
    global_offsets[lID] += local_histogram;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Scatter using bin_flags + popcount.
  __local uint bin_flags[RADIX_SORT_BINS * (WORKGROUP_SIZE / 32)];

  const uint flags_bin = lID / 32;
  const uint flags_bit = 1u << (lID % 32);

  for (uint index = 0; index < g_num_blocks_per_workgroup; index++) {
    uint elementId = wID * g_num_blocks_per_workgroup * WORKGROUP_SIZE +
                     index * WORKGROUP_SIZE + lID;

    // Clear bin flags.
    if (lID < RADIX_SORT_BINS) {
      for (uint i = 0; i < WORKGROUP_SIZE / 32; i++)
        bin_flags[lID * (WORKGROUP_SIZE / 32) + i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint element_key = 0;
    int element_val = 0;
    uint binID = 0;
    uint binOffset = 0;
    if (elementId < g_num_elements) {
      element_key = keys_in[elementId];
      element_val = vals_in[elementId];
      binID = (element_key >> g_shift) & 0xFFu;
      binOffset = global_offsets[binID];
      atomic_or(&bin_flags[binID * (WORKGROUP_SIZE / 32) + flags_bin], flags_bit);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (elementId < g_num_elements) {
      uint prefix = 0;
      uint count = 0;
      for (uint i = 0; i < WORKGROUP_SIZE / 32; i++) {
        uint bits = bin_flags[binID * (WORKGROUP_SIZE / 32) + i];
        uint full_count = popcount(bits);
        uint partial_count = popcount(bits & (flags_bit - 1));
        prefix += (i < flags_bin) ? full_count : 0;
        prefix += (i == flags_bin) ? partial_count : 0;
        count += full_count;
      }
      keys_out[binOffset + prefix] = element_key;
      vals_out[binOffset + prefix] = element_val;
      if (prefix == count - 1) {
        atomic_add(&global_offsets[binID], count);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// ---- BVH CreateRadixTree ----

int cl_PrefixLength(uint a, uint b) {
  return clz(a ^ b);
}

int cl_PrefixLengthIdx(__global const uint* morton, int size, int i, int j) {
  if (j < 0 || j >= size) return -1;
  if (morton[i] == morton[j])
    return 32 + cl_PrefixLength((uint)i, (uint)j);
  return cl_PrefixLength(morton[i], morton[j]);
}

int cl_RangeEnd(__global const uint* morton, int size, int i) {
  int dir = cl_PrefixLengthIdx(morton, size, i, i + 1) -
            cl_PrefixLengthIdx(morton, size, i, i - 1);
  dir = (dir > 0) - (dir < 0);
  int commonPrefix = cl_PrefixLengthIdx(morton, size, i, i - dir);
  int max_length = 4;
  while (cl_PrefixLengthIdx(morton, size, i, i + dir * max_length) > commonPrefix)
    max_length *= 4;
  int length = 0;
  for (int step = max_length / 2; step > 0; step /= 2) {
    if (cl_PrefixLengthIdx(morton, size, i, i + dir * (length + step)) > commonPrefix)
      length += step;
  }
  return i + dir * length;
}

int cl_FindSplit(__global const uint* morton, int size, int first, int last) {
  int commonPrefix = cl_PrefixLengthIdx(morton, size, first, last);
  int split = first;
  int step = last - first;
  do {
    step = (step + 1) >> 1;
    int newSplit = split + step;
    if (newSplit < last) {
      int splitPrefix = cl_PrefixLengthIdx(morton, size, first, newSplit);
      if (splitPrefix > commonPrefix) split = newSplit;
    }
  } while (step > 1);
  return split;
}

__kernel void create_radix_tree(
    __global const uint* leafMorton,
    __global int* nodeParent,
    __global int2* internalChildren,
    const int numLeaves) {

  int internal = get_global_id(0);
  if (internal >= numLeaves - 1) return;

  int size = numLeaves;
  int first = internal;
  int last = cl_RangeEnd(leafMorton, size, first);
  if (first > last) { int tmp = first; first = last; last = tmp; }
  int split = cl_FindSplit(leafMorton, size, first, last);

  // Leaf2Node = leaf * 2, Internal2Node = internal * 2 + 1
  int child1 = (split == first) ? split * 2 : split * 2 + 1;
  int child2 = ((split + 1) == last) ? (split + 1) * 2 : (split + 1) * 2 + 1;

  internalChildren[internal] = (int2)(child1, child2);
  int node = internal * 2 + 1;
  nodeParent[child1] = node;
  nodeParent[child2] = node;
}
