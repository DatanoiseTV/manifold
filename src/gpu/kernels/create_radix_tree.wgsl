// BVH construction: one thread per internal node computes its two children
// by scanning prefix lengths on the sorted Morton code array.
// Pure integer ops, no fp required.

struct TreeParams {
  num_leaves: i32,
}

@group(0) @binding(0) var<storage, read> leaf_morton: array<u32>;
@group(0) @binding(1) var<storage, read_write> node_parent: array<i32>;
@group(0) @binding(2) var<storage, read_write> internal_children: array<vec2<i32>>;
@group(0) @binding(3) var<uniform> params: TreeParams;

fn tree_leaf_to_node(leaf: i32) -> i32 { return leaf * 2; }
fn tree_internal_to_node(internal: i32) -> i32 { return internal * 2 + 1; }

fn prefix_length(a: u32, b: u32) -> i32 {
  return i32(countLeadingZeros(a ^ b));
}

fn prefix_length_idx(i: i32, j: i32) -> i32 {
  if (j < 0 || j >= params.num_leaves) { return -1; }
  if (leaf_morton[i] == leaf_morton[j]) {
    return 32 + prefix_length(u32(i), u32(j));
  }
  return prefix_length(leaf_morton[i], leaf_morton[j]);
}

fn range_end(i: i32) -> i32 {
  let pAfter = prefix_length_idx(i, i + 1);
  let pBefore = prefix_length_idx(i, i - 1);
  var dir: i32 = 0;
  if (pAfter > pBefore) { dir = 1; }
  else if (pAfter < pBefore) { dir = -1; }

  let commonPrefix = prefix_length_idx(i, i - dir);
  var max_length: i32 = 4;
  loop {
    if (prefix_length_idx(i, i + dir * max_length) <= commonPrefix) { break; }
    max_length = max_length * 4;
  }
  var length: i32 = 0;
  var step: i32 = max_length / 2;
  loop {
    if (step <= 0) { break; }
    if (prefix_length_idx(i, i + dir * (length + step)) > commonPrefix) {
      length = length + step;
    }
    step = step / 2;
  }
  return i + dir * length;
}

fn find_split(first: i32, last: i32) -> i32 {
  let commonPrefix = prefix_length_idx(first, last);
  var split: i32 = first;
  var step: i32 = last - first;
  loop {
    step = (step + 1) >> 1u;
    let newSplit = split + step;
    if (newSplit < last) {
      let splitPrefix = prefix_length_idx(first, newSplit);
      if (splitPrefix > commonPrefix) { split = newSplit; }
    }
    if (step <= 1) { break; }
  }
  return split;
}

@compute @workgroup_size(256)
fn create_radix_tree(@builtin(global_invocation_id) gid: vec3<u32>) {
  let internal = i32(gid.x);
  if (internal >= params.num_leaves - 1) { return; }

  var first: i32 = internal;
  var last: i32 = range_end(first);
  if (first > last) {
    let tmp = first; first = last; last = tmp;
  }
  let split = find_split(first, last);
  var child1: i32;
  if (split == first) { child1 = tree_leaf_to_node(split); }
  else { child1 = tree_internal_to_node(split); }
  var child2: i32;
  if ((split + 1) == last) { child2 = tree_leaf_to_node(split + 1); }
  else { child2 = tree_internal_to_node(split + 1); }

  internal_children[internal] = vec2<i32>(child1, child2);
  let node = tree_internal_to_node(internal);
  node_parent[child1] = node;
  node_parent[child2] = node;
}
