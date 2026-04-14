// Port of Intersect() from boolean3.cpp to WGSL using fp64_lib.wgsl.
// Inputs: 4 vec3<f64> points per tuple (16 u32 words per coord pair).
// Output: vec4<f64> (8 u32 words).
//
// Must be loaded with fp64_lib.wgsl prepended.

struct IntersectInput {
  // 4 points × 3 coords × 2 u32 = 24 u32; pad to 32 (128 bytes).
  aL: array<vec2<u32>, 3>,
  aR: array<vec2<u32>, 3>,
  bL: array<vec2<u32>, 3>,
  bR: array<vec2<u32>, 3>,
  _pad: array<vec2<u32>, 4>,
}
struct IntersectOutput {
  xyzz: array<vec2<u32>, 4>,
}
struct IntersectParams { count: u32, _pad: vec3<u32> }

@group(0) @binding(0) var<storage, read> i_in: array<IntersectInput>;
@group(0) @binding(1) var<storage, read_write> i_out: array<IntersectOutput>;
@group(0) @binding(2) var<uniform> i_params: IntersectParams;

// vec4<u32> so we can pack (hi, lo) for a single f64 coord.
fn fp64_abs_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
  return fp64_lt(fp64_abs(a), fp64_abs(b));
}

// intersect_fp64 mirrors boolean3.cpp Intersect() line-for-line but in fp64.
fn intersect_fp64(in_: IntersectInput) -> IntersectOutput {
  let aL_x = in_.aL[0]; let aL_y = in_.aL[1]; let aL_z = in_.aL[2];
  let aR_x = in_.aR[0]; let aR_y = in_.aR[1]; let aR_z = in_.aR[2];
  let bL_x = in_.bL[0]; let bL_y = in_.bL[1]; let bL_z = in_.bL[2];
  let bR_x = in_.bR[0]; let bR_y = in_.bR[1]; let bR_z = in_.bR[2];

  let dyL = fp64_sub(bL_y, aL_y);
  let dyR = fp64_sub(bR_y, aR_y);
  let useL = fp64_abs_lt(dyL, dyR);

  let dx = fp64_sub(aR_x, aL_x);
  let denom = fp64_sub(dyL, dyR);
  var lambda: vec2<u32>;
  if (useL) { lambda = fp64_div(dyL, denom); }
  else      { lambda = fp64_div(dyR, denom); }
  if (!fp64_is_finite(lambda) || fp64_is_nan(lambda)) {
    lambda = fp64_zero();
  }

  var out: IntersectOutput;

  // x = lambda * dx + (useL ? aL.x : aR.x)
  let bx = select(aR_x, aL_x, useL);
  out.xyzz[0] = fp64_add(fp64_mul(lambda, dx), bx);

  // y: aDy = aR.y - aL.y; bDy = bR.y - bL.y; useA = |aDy| < |bDy|.
  let aDy = fp64_sub(aR_y, aL_y);
  let bDy = fp64_sub(bR_y, bL_y);
  let useA = fp64_abs_lt(aDy, bDy);
  let dy_use = select(bDy, aDy, useA);
  let by_L = select(bL_y, aL_y, useA);
  let by_R = select(bR_y, aR_y, useA);
  let basey = select(by_R, by_L, useL);
  out.xyzz[1] = fp64_add(fp64_mul(lambda, dy_use), basey);

  // z = lambda * (aR.z - aL.z) + (useL ? aL.z : aR.z)
  let dz = fp64_sub(aR_z, aL_z);
  let basez = select(aR_z, aL_z, useL);
  out.xyzz[2] = fp64_add(fp64_mul(lambda, dz), basez);

  // w = lambda * (bR.z - bL.z) + (useL ? bL.z : bR.z)
  let dw = fp64_sub(bR_z, bL_z);
  let basew = select(bR_z, bL_z, useL);
  out.xyzz[3] = fp64_add(fp64_mul(lambda, dw), basew);

  return out;
}

@compute @workgroup_size(64)
fn intersect_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= i_params.count) { return; }
  i_out[i] = intersect_fp64(i_in[i]);
}
