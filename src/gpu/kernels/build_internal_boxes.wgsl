// Bottom-up propagation of node bounding boxes through the radix BVH.
// One thread per leaf; walks parent chain up to the root, atomically
// marking each internal node's first child as visited. The second-arriving
// child actually computes the union (since both children are then ready).
//
// Box is stored as two vec3<f32> (min, max). fp32 is fine for BVH prune —
// Boolean3's fp64 intersection math runs on CPU.

struct Box {
  min: vec3<f32>,
  max: vec3<f32>,
}

struct Params {
  num_leaves: u32,
}

@group(0) @binding(0) var<storage, read> node_parent: array<i32>;
@group(0) @binding(1) var<storage, read> internal_children: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read_write> node_bbox: array<Box>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

const ROOT_NODE: i32 = 1;

fn node_to_internal(node: i32) -> i32 { return (node - 1) / 2; }

fn box_union(a: Box, b: Box) -> Box {
  var out: Box;
  out.min = min(a.min, b.min);
  out.max = max(a.max, b.max);
  return out;
}

@compute @workgroup_size(256)
fn build_internal_boxes(@builtin(global_invocation_id) gid: vec3<u32>) {
  let leaf = i32(gid.x);
  if (u32(leaf) >= params.num_leaves) { return; }

  let leaf_node = leaf * 2;
  var node = leaf_node;
  loop {
    node = node_parent[node];
    if (node < 0) { return; }
    let internal = node_to_internal(node);
    let first = atomicAdd(&counters[u32(internal)], 1u);
    if (first == 0u) { return; }  // Wait for the other child.
    let children = internal_children[internal];
    node_bbox[node] =
        box_union(node_bbox[children.x], node_bbox[children.y]);
    if (node == ROOT_NODE) { return; }
  }
}
