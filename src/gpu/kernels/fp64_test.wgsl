// Bit-exact feasibility test for fp64_lib.wgsl. Caller supplies two fp64
// operands packed as vec2<u32> per input (matching the canonical IEEE 754
// binary64 bit layout). Kernel outputs add/sub/mul/div of each pair; host
// compares against CPU fp64 to verify precision.
//
// This file expects fp64_lib.wgsl to be prepended at load time.

struct TestInput {
  a: vec2<u32>,
  b: vec2<u32>,
}
struct TestOutput {
  sum: vec2<u32>,
  diff: vec2<u32>,
  prod: vec2<u32>,
  quot: vec2<u32>,
}
struct TestParams { count: u32, _pad: vec3<u32> }

@group(0) @binding(0) var<storage, read> t_in: array<TestInput>;
@group(0) @binding(1) var<storage, read_write> t_out: array<TestOutput>;
@group(0) @binding(2) var<uniform> t_params: TestParams;

@compute @workgroup_size(64)
fn fp64_test(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= t_params.count) { return; }
  let a = t_in[i].a;
  let b = t_in[i].b;
  t_out[i].sum = fp64_add(a, b);
  t_out[i].diff = fp64_sub(a, b);
  t_out[i].prod = fp64_mul(a, b);
  t_out[i].quot = fp64_div(a, b);
}
