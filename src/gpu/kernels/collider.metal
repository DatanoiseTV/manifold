// Metal compute kernels for Collider BVH construction.
// Only CreateRadixTree runs on GPU (pure integer ops).
// BuildInternalBoxes and FindCollision require fp64 and stay on CPU.

#include <metal_stdlib>
using namespace metal;

// Tree node helpers (must match collider.h)
inline bool IsLeaf(int node) { return node % 2 == 0; }
inline int Leaf2Node(int leaf) { return leaf * 2; }
inline int Internal2Node(int internal) { return internal * 2 + 1; }
inline int Node2Internal(int node) { return (node - 1) / 2; }

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
  int child1 = (split == first) ? Leaf2Node(split) : Internal2Node(split);
  int child2 = ((split + 1) == last) ? Leaf2Node(split + 1) :
                                        Internal2Node(split + 1);

  internalChildren[internal] = int2(child1, child2);
  int node = Internal2Node(internal);
  nodeParent[child1] = node;
  nodeParent[child2] = node;
}
