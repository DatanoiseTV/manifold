// Multi-workgroup radix sort for Metal.
// Based on VkRadixSort by Mirco Werner (MIT License).
// https://github.com/MircoWerner/VkRadixSort
//
// Two-pass approach per radix digit:
//   Pass 1 (radix_sort_histograms): per-workgroup histograms
//   Pass 2 (radix_sort_scatter): prefix sum + scatter using bin_flags/popcount

#include <metal_stdlib>
using namespace metal;

#define WORKGROUP_SIZE 256
#define RADIX_SORT_BINS 256

// ---- Pass 1: Per-workgroup histograms ----
kernel void radix_sort_histograms(
    device const uint* keys_in [[buffer(0)]],
    device uint* histograms [[buffer(1)]],
    constant uint& g_num_elements [[buffer(2)]],
    constant uint& g_shift [[buffer(3)]],
    constant uint& g_num_blocks_per_workgroup [[buffer(4)]],
    uint lID [[thread_index_in_threadgroup]],
    uint wID [[threadgroup_position_in_grid]]) {

  threadgroup uint local_histogram[RADIX_SORT_BINS];
  if (lID < RADIX_SORT_BINS) local_histogram[lID] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint index = 0; index < g_num_blocks_per_workgroup; index++) {
    uint elementId = wID * g_num_blocks_per_workgroup * WORKGROUP_SIZE +
                     index * WORKGROUP_SIZE + lID;
    if (elementId < g_num_elements) {
      uint bin = (keys_in[elementId] >> g_shift) & 0xFF;
      atomic_fetch_add_explicit(
          reinterpret_cast<threadgroup atomic_uint*>(&local_histogram[bin]),
          1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lID < RADIX_SORT_BINS) {
    histograms[RADIX_SORT_BINS * wID + lID] = local_histogram[lID];
  }
}

// ---- Pass 2: Global prefix sum + scatter ----
kernel void radix_sort_scatter(
    device const uint* keys_in [[buffer(0)]],
    device const int* vals_in [[buffer(1)]],
    device uint* keys_out [[buffer(2)]],
    device int* vals_out [[buffer(3)]],
    device uint* histograms [[buffer(4)]],
    constant uint& g_num_elements [[buffer(5)]],
    constant uint& g_shift [[buffer(6)]],
    constant uint& g_num_workgroups [[buffer(7)]],
    constant uint& g_num_blocks_per_workgroup [[buffer(8)]],
    uint lID [[thread_index_in_threadgroup]],
    uint wID [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {

  threadgroup uint global_offsets[RADIX_SORT_BINS];

  // Each of the 256 threads (one per bin) computes its global offset
  // by scanning across all workgroups' histograms for its bin.
  uint local_histogram = 0;
  if (lID < RADIX_SORT_BINS) {
    uint count = 0;
    for (uint j = 0; j < g_num_workgroups; j++) {
      uint t = histograms[RADIX_SORT_BINS * j + lID];
      local_histogram = (j == wID) ? count : local_histogram;
      count += t;
    }
    // Now count = total elements with this digit across all workgroups.
    // local_histogram = elements with this digit in workgroups before wID.

    // Exclusive prefix sum across all 256 bins to get global offset.
    // Use sequential scan on thread 0 (simple, correct for 256 bins).
    // Store counts temporarily.
    global_offsets[lID] = count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Thread 0 does exclusive prefix sum across bins.
  if (lID == 0) {
    uint sum = 0;
    for (uint i = 0; i < RADIX_SORT_BINS; i++) {
      uint c = global_offsets[i];
      global_offsets[i] = sum;
      sum += c;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Add local_histogram (within-bin offset for this workgroup) to global offset.
  if (lID < RADIX_SORT_BINS) {
    global_offsets[lID] += local_histogram;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- Scatter using bin_flags + popcount ----
  struct BinFlags { uint flags[WORKGROUP_SIZE / 32]; };
  threadgroup BinFlags bin_flags[RADIX_SORT_BINS];

  const uint flags_bin = lID / 32;
  const uint flags_bit = 1u << (lID % 32);

  for (uint index = 0; index < g_num_blocks_per_workgroup; index++) {
    uint elementId = wID * g_num_blocks_per_workgroup * WORKGROUP_SIZE +
                     index * WORKGROUP_SIZE + lID;

    // Clear bin flags.
    if (lID < RADIX_SORT_BINS) {
      for (uint i = 0; i < WORKGROUP_SIZE / 32; i++)
        bin_flags[lID].flags[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint element_key = 0;
    int element_val = 0;
    uint binID = 0;
    uint binOffset = 0;
    if (elementId < g_num_elements) {
      element_key = keys_in[elementId];
      element_val = vals_in[elementId];
      binID = (element_key >> g_shift) & 0xFF;
      binOffset = global_offsets[binID];
      atomic_fetch_or_explicit(
          reinterpret_cast<threadgroup atomic_uint*>(
              &bin_flags[binID].flags[flags_bin]),
          flags_bit, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (elementId < g_num_elements) {
      uint prefix = 0;
      uint count = 0;
      for (uint i = 0; i < WORKGROUP_SIZE / 32; i++) {
        uint bits = bin_flags[binID].flags[i];
        uint full_count = popcount(bits);
        uint partial_count = popcount(bits & (flags_bit - 1));
        prefix += (i < flags_bin) ? full_count : 0;
        prefix += (i == flags_bin) ? partial_count : 0;
        count += full_count;
      }
      keys_out[binOffset + prefix] = element_key;
      vals_out[binOffset + prefix] = element_val;
      if (prefix == count - 1) {
        atomic_fetch_add_explicit(
            reinterpret_cast<threadgroup atomic_uint*>(
                &global_offsets[binID]),
            count, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
// ---------------------------------------------------------------------------
// Collider BVH construction — CreateRadixTree (pure integer ops)
// ---------------------------------------------------------------------------

inline int tree_Leaf2Node(int leaf) { return leaf * 2; }
inline int tree_Internal2Node(int internal) { return internal * 2 + 1; }

constant int kInitialLength = 4;
constant int kLengthMultiple = 4;

int PrefixLength(uint a, uint b) {
  return clz(a ^ b);
}

int PrefixLengthIdx(device const uint* morton, int size, int i, int j) {
  if (j < 0 || j >= size) return -1;
  if (morton[i] == morton[j])
    return 32 + PrefixLength(uint(i), uint(j));
  return PrefixLength(morton[i], morton[j]);
}

int RangeEnd(device const uint* morton, int size, int i) {
  int dir = PrefixLengthIdx(morton, size, i, i + 1) -
            PrefixLengthIdx(morton, size, i, i - 1);
  dir = (dir > 0) - (dir < 0);
  int commonPrefix = PrefixLengthIdx(morton, size, i, i - dir);
  int max_length = kInitialLength;
  while (PrefixLengthIdx(morton, size, i, i + dir * max_length) > commonPrefix)
    max_length *= kLengthMultiple;
  int length = 0;
  for (int step = max_length / 2; step > 0; step /= 2) {
    if (PrefixLengthIdx(morton, size, i, i + dir * (length + step)) >
        commonPrefix)
      length += step;
  }
  return i + dir * length;
}

int FindSplit(device const uint* morton, int size, int first, int last) {
  int commonPrefix = PrefixLengthIdx(morton, size, first, last);
  int split = first;
  int step = last - first;
  do {
    step = (step + 1) >> 1;
    int newSplit = split + step;
    if (newSplit < last) {
      int splitPrefix = PrefixLengthIdx(morton, size, first, newSplit);
      if (splitPrefix > commonPrefix) split = newSplit;
    }
  } while (step > 1);
  return split;
}

kernel void create_radix_tree(
    device const uint* leafMorton [[buffer(0)]],
    device int* nodeParent [[buffer(1)]],
    device int2* internalChildren [[buffer(2)]],  // pair<int,int> as int2
    constant int& numLeaves [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {

  int internal = int(gid);
  if (internal >= numLeaves - 1) return;

  int size = numLeaves;
  int first = internal;
  int last = RangeEnd(leafMorton, size, first);
  if (first > last) {
    int tmp = first; first = last; last = tmp;
  }
  int split = FindSplit(leafMorton, size, first, last);
  int child1 = (split == first) ? tree_Leaf2Node(split) : tree_Internal2Node(split);
  int child2 = ((split + 1) == last) ? tree_Leaf2Node(split + 1) :
                                        tree_Internal2Node(split + 1);

  internalChildren[internal] = int2(child1, child2);
  int node = tree_Internal2Node(internal);
  nodeParent[child1] = node;
  nodeParent[child2] = node;
}
