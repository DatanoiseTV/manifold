// BVH collision query: one thread per query, DFS through the tree, appending
// (queryIdx, leafIdx) pairs to a flat output buffer. Output length is an
// atomic counter; callers pre-size the output buffer conservatively.
//
// Query box is read from an input array of Box (fp32). Tree bounding boxes
// are also fp32 — precision is CPU-handled in Kernel11/02/12 downstream.

struct Box {
  min: vec3<f32>,
  max: vec3<f32>,
}

struct Params {
  num_queries: u32,
  self_collision: u32,      // 0 or 1
  output_capacity: u32,     // max pairs; anything beyond is dropped
}

@group(0) @binding(0) var<storage, read> query_bbox: array<Box>;
@group(0) @binding(1) var<storage, read> node_bbox: array<Box>;
@group(0) @binding(2) var<storage, read> internal_children: array<vec2<i32>>;
@group(0) @binding(3) var<storage, read_write> pairs_out: array<vec2<i32>>;
@group(0) @binding(4) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

const ROOT_NODE: i32 = 1;

fn is_leaf(node: i32) -> bool { return (node & 1) == 0; }
fn node_to_internal(node: i32) -> i32 { return (node - 1) / 2; }
fn node_to_leaf(node: i32) -> i32 { return node / 2; }

fn box_overlap(a: Box, b: Box) -> bool {
  return all(a.min <= b.max) && all(a.max >= b.min);
}

fn box_is_empty(b: Box) -> bool {
  // A 'missing' face/leaf is marked by min.x = +inf on CPU.
  return b.min.x > b.max.x;
}

@compute @workgroup_size(256)
fn find_collisions(@builtin(global_invocation_id) gid: vec3<u32>) {
  let queryIdx = i32(gid.x);
  if (u32(queryIdx) >= params.num_queries) { return; }

  let query = query_bbox[queryIdx];
  if (box_is_empty(query)) { return; }

  // Fixed-size stack. BVH depth is bounded by 30 (Morton) + 32 (index) = 62.
  var stack: array<i32, 64>;
  var top: i32 = -1;
  var node: i32 = ROOT_NODE;

  loop {
    let internal = node_to_internal(node);
    let children = internal_children[internal];
    let c1 = children.x;
    let c2 = children.y;

    let b1 = node_bbox[c1];
    let b2 = node_bbox[c2];
    let o1 = box_overlap(query, b1);
    let o2 = box_overlap(query, b2);

    // Handle leaf hits.
    if (o1 && is_leaf(c1)) {
      let leafIdx = node_to_leaf(c1);
      if (params.self_collision == 0u || leafIdx != queryIdx) {
        let slot = atomicAdd(&pair_count[0], 1u);
        if (slot < params.output_capacity) {
          pairs_out[slot] = vec2<i32>(queryIdx, leafIdx);
        }
      }
    }
    if (o2 && is_leaf(c2)) {
      let leafIdx = node_to_leaf(c2);
      if (params.self_collision == 0u || leafIdx != queryIdx) {
        let slot = atomicAdd(&pair_count[0], 1u);
        if (slot < params.output_capacity) {
          pairs_out[slot] = vec2<i32>(queryIdx, leafIdx);
        }
      }
    }

    let t1 = o1 && !is_leaf(c1);
    let t2 = o2 && !is_leaf(c2);

    if (!t1 && !t2) {
      if (top < 0) { break; }
      node = stack[top];
      top = top - 1;
    } else if (t1 && t2) {
      node = c1;
      top = top + 1;
      stack[top] = c2;
    } else if (t1) {
      node = c1;
    } else {
      node = c2;
    }
  }
}
