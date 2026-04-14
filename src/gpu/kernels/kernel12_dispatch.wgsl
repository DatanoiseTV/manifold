// WGSL port of boolean3.cpp's Kernel11/Kernel02/Kernel12 using fp64_lib.wgsl.
// Loaded with fp64_lib.wgsl prepended at pipeline load time.
//
// Layout conventions:
//   Vec3F64 = 3 × vec2<u32> = 24 bytes, IEEE 754 double per coord.
//   Halfedge = 4 × i32 = 16 bytes (startVert, endVert, pairedHalfedge, propVert).
//
// Dispatch kind:
//   0 = Kernel12 (edge_A, face_B) → (x12: i32, v12: Vec3F64)
//   1 = Kernel02 (vert_A, face_B) → (s02: i32, z02: f64)   [used internally]
//   2 = Kernel11 (edge_P, edge_Q) → (s11: i32, xyzz11: vec4<f64>)  [internal]
// For the manifold pipeline we only need kind=0 (Kernel12). The CPU feeds
// pairs from `Intersect12_` and receives the fp64 intersection points back.

struct Vec3F64 {
  x: vec2<u32>,
  y: vec2<u32>,
  z: vec2<u32>,
}

struct Halfedge {
  startVert: i32,
  endVert: i32,
  pairedHalfedge: i32,
  propVert: i32,
}

// ---- Bindings: meshes P and Q, pair list, outputs, params. ----
@group(0) @binding(0) var<storage, read> P_vertPos: array<Vec3F64>;
@group(0) @binding(1) var<storage, read> P_vertNormal: array<Vec3F64>;
@group(0) @binding(2) var<storage, read> P_halfedge: array<Halfedge>;
@group(0) @binding(3) var<storage, read> P_faceNormal: array<Vec3F64>;
@group(0) @binding(4) var<storage, read> Q_vertPos: array<Vec3F64>;
@group(0) @binding(5) var<storage, read> Q_vertNormal: array<Vec3F64>;
@group(0) @binding(6) var<storage, read> Q_halfedge: array<Halfedge>;
@group(0) @binding(7) var<storage, read> Q_faceNormal: array<Vec3F64>;
@group(0) @binding(8) var<storage, read> pair_list: array<vec2<i32>>;
@group(0) @binding(9) var<storage, read_write> out_x: array<i32>;
@group(0) @binding(10) var<storage, read_write> out_v: array<Vec3F64>;

struct K12Params {
  count: u32,
  expandP: u32,
  forward: u32,
  _pad: u32,
}
@group(0) @binding(11) var<uniform> params: K12Params;

// ---- Helpers for mesh access parameterized on "side": 0 = P, 1 = Q. ----

fn vp_get(side: u32, i: i32) -> Vec3F64 {
  if (side == 0u) { return P_vertPos[i]; }
  return Q_vertPos[i];
}
fn vn_get(side: u32, i: i32) -> Vec3F64 {
  if (side == 0u) { return P_vertNormal[i]; }
  return Q_vertNormal[i];
}
fn he_get(side: u32, i: i32) -> Halfedge {
  if (side == 0u) { return P_halfedge[i]; }
  return Q_halfedge[i];
}
fn fn_get(side: u32, i: i32) -> Vec3F64 {
  if (side == 0u) { return P_faceNormal[i]; }
  return Q_faceNormal[i];
}

// ---- fp64 convenience ----

fn f64_nan() -> vec2<u32> { return fp64_nan(); }
fn f64_zero() -> vec2<u32> { return fp64_zero(); }

fn withSign_f64(pos: bool, v: vec2<u32>) -> vec2<u32> {
  if (pos) { return v; }
  return fp64_neg(v);
}

// Shadows(p, q, dir) from boolean3.cpp:
//   p == q ? dir < 0 : p < q
fn shadows_pred(p: vec2<u32>, q: vec2<u32>, dir: vec2<u32>) -> bool {
  if (fp64_eq(p, q)) {
    return fp64_lt(dir, f64_zero());
  }
  return fp64_lt(p, q);
}

// ---- Interpolate: lambda = (useL ? dxL : dxR) / dLR.x ----
// Returns yz (vec2<f64>) packed as (y, z).
struct InterpolateResult {
  y: vec2<u32>,
  z: vec2<u32>,
}

fn interpolate_fp64(aL: Vec3F64, aR: Vec3F64, x: vec2<u32>) -> InterpolateResult {
  let dxL = fp64_sub(x, aL.x);
  let dxR = fp64_sub(x, aR.x);
  let useL = fp64_lt(fp64_abs(dxL), fp64_abs(dxR));
  let dLR_x = fp64_sub(aR.x, aL.x);
  let dLR_y = fp64_sub(aR.y, aL.y);
  let dLR_z = fp64_sub(aR.z, aL.z);
  var lambda = fp64_div(select(dxR, dxL, useL), dLR_x);
  var out: InterpolateResult;
  if (!fp64_is_finite(lambda) || fp64_is_nan(lambda) ||
      !fp64_is_finite(dLR_y) || !fp64_is_finite(dLR_z)) {
    out.y = aL.y;
    out.z = aL.z;
    return out;
  }
  out.y = fp64_add(fp64_mul(lambda, dLR_y), select(aR.y, aL.y, useL));
  out.z = fp64_add(fp64_mul(lambda, dLR_z), select(aR.z, aL.z, useL));
  return out;
}

// ---- Intersect (ported from intersect_kernel; kept here for self-containment) ----
struct IntersectResult {
  xyzz: array<vec2<u32>, 4>,
}

fn intersect_fp64_inline(aL: Vec3F64, aR: Vec3F64,
                         bL: Vec3F64, bR: Vec3F64) -> IntersectResult {
  let dyL = fp64_sub(bL.y, aL.y);
  let dyR = fp64_sub(bR.y, aR.y);
  let useL = fp64_lt(fp64_abs(dyL), fp64_abs(dyR));
  let dx = fp64_sub(aR.x, aL.x);
  let denom = fp64_sub(dyL, dyR);
  var lambda = fp64_div(select(dyR, dyL, useL), denom);
  if (!fp64_is_finite(lambda) || fp64_is_nan(lambda)) { lambda = f64_zero(); }

  let aDy = fp64_sub(aR.y, aL.y);
  let bDy = fp64_sub(bR.y, bL.y);
  let useA = fp64_lt(fp64_abs(aDy), fp64_abs(bDy));
  let dy_use = select(bDy, aDy, useA);
  let by_L = select(bL.y, aL.y, useA);
  let by_R = select(bR.y, aR.y, useA);
  let basey = select(by_R, by_L, useL);
  let basex = select(aR.x, aL.x, useL);
  let basez = select(aR.z, aL.z, useL);
  let basew = select(bR.z, bL.z, useL);
  let dz = fp64_sub(aR.z, aL.z);
  let dw = fp64_sub(bR.z, bL.z);

  var out: IntersectResult;
  out.xyzz[0] = fp64_add(fp64_mul(lambda, dx),    basex);
  out.xyzz[1] = fp64_add(fp64_mul(lambda, dy_use), basey);
  out.xyzz[2] = fp64_add(fp64_mul(lambda, dz),    basez);
  out.xyzz[3] = fp64_add(fp64_mul(lambda, dw),    basew);
  return out;
}

// ---- Shadow01 ported from boolean3.cpp ----
// Template parameters collapsed into runtime flags (expandP, forward).
// meshA/meshB are 0 (P) or 1 (Q).
struct Shadow01Result {
  s01: i32,
  yz_y: vec2<u32>,
  yz_z: vec2<u32>,
  finite: bool,
}

fn shadow01_fp64(
    a0: i32, b1: i32,
    fwd: bool, expandP: bool,
    meshA: u32, meshB: u32) -> Shadow01Result {
  let heB = he_get(meshB, b1);
  let b1s = heB.startVert;
  let b1e = heB.endVert;

  let a0x = vp_get(meshA, a0).x;
  let b1sx = vp_get(meshB, b1s).x;
  let b1ex = vp_get(meshB, b1e).x;
  let a0xp = vn_get(meshA, a0).x;
  let b1sxp = vn_get(meshB, b1s).x;
  let b1exp = vn_get(meshB, b1e).x;

  var s01: i32;
  if (fwd) {
    let d1 = fp64_sub(withSign_f64(expandP, a0xp), b1exp);
    let d0 = fp64_sub(withSign_f64(expandP, a0xp), b1sxp);
    let s_e = select(0, 1, shadows_pred(a0x, b1ex, d1));
    let s_s = select(0, 1, shadows_pred(a0x, b1sx, d0));
    s01 = s_e - s_s;
  } else {
    let d1 = fp64_sub(withSign_f64(expandP, b1sxp), a0xp);
    let d0 = fp64_sub(withSign_f64(expandP, b1exp), a0xp);
    let s_s = select(0, 1, shadows_pred(b1sx, a0x, d1));
    let s_e = select(0, 1, shadows_pred(b1ex, a0x, d0));
    s01 = s_s - s_e;
  }

  var out: Shadow01Result;
  out.s01 = s01;
  out.yz_y = f64_nan(); out.yz_z = f64_nan();
  out.finite = false;

  if (s01 != 0) {
    let interp = interpolate_fp64(vp_get(meshB, b1s),
                                  vp_get(meshB, b1e),
                                  vp_get(meshA, a0).x);
    out.yz_y = interp.y; out.yz_z = interp.z;
    out.finite = fp64_is_finite(interp.y);

    let b1pair = heB.pairedHalfedge;
    let fn1 = fn_get(meshB, b1 / 3);
    let fn2 = fn_get(meshB, b1pair / 3);
    let dir = fp64_add(fn1.y, fn2.y);

    let a0y = vp_get(meshA, a0).y;
    if (fwd) {
      if (!shadows_pred(a0y, interp.y, fp64_neg(dir))) { out.s01 = 0; }
    } else {
      if (!shadows_pred(interp.y, a0y, withSign_f64(expandP, dir))) {
        out.s01 = 0;
      }
    }
  }
  return out;
}

// ---- Kernel11: edge_P × edge_Q. ----
struct K11Result {
  s11: i32,
  xyzz: array<vec2<u32>, 4>,
}

fn kernel11_fp64(p1: i32, q1: i32, expandP: bool) -> K11Result {
  var out: K11Result;
  out.s11 = 0;
  out.xyzz[0] = f64_nan();
  out.xyzz[1] = f64_nan();
  out.xyzz[2] = f64_nan();
  out.xyzz[3] = f64_nan();

  var k: i32 = 0;
  var pRL: array<Vec3F64, 2>;
  var qRL: array<Vec3F64, 2>;
  var shadows: bool = false;

  let heP = he_get(0u, p1);
  let p0_0 = heP.startVert;
  let p0_1 = heP.endVert;
  // Unroll the for-loop over i in {0, 1}.
  for (var i: i32 = 0; i < 2; i = i + 1) {
    let p0 = select(p0_1, p0_0, i == 0);
    // Shadow01<expandP, true>(p0, q1, inP, inQ) → meshA=P, meshB=Q
    let r = shadow01_fp64(p0, q1, true, expandP, 0u, 1u);
    if (r.finite) {
      let sign = select(1, -1, i == 0);
      out.s11 = out.s11 + r.s01 * sign;
      if (k < 2 && (k == 0 || (r.s01 != 0) != shadows)) {
        shadows = r.s01 != 0;
        let pos = vp_get(0u, p0);
        pRL[k] = pos;
        qRL[k] = Vec3F64(pos.x, r.yz_y, r.yz_z);
        k = k + 1;
      }
    }
  }

  let heQ = he_get(1u, q1);
  let q0_0 = heQ.startVert;
  let q0_1 = heQ.endVert;
  for (var i: i32 = 0; i < 2; i = i + 1) {
    let q0 = select(q0_1, q0_0, i == 0);
    // Shadow01<expandP, false>(q0, p1, inQ, inP) → meshA=Q, meshB=P
    let r = shadow01_fp64(q0, p1, false, expandP, 1u, 0u);
    if (r.finite) {
      let sign = select(1, -1, i == 0);
      out.s11 = out.s11 + r.s01 * sign;
      if (k < 2 && (k == 0 || (r.s01 != 0) != shadows)) {
        shadows = r.s01 != 0;
        let pos = vp_get(1u, q0);
        qRL[k] = pos;
        pRL[k] = Vec3F64(pos.x, r.yz_y, r.yz_z);
        k = k + 1;
      }
    }
  }

  if (out.s11 == 0 || k < 2) {
    out.s11 = 0;
    return out;
  }
  let iv = intersect_fp64_inline(pRL[0], pRL[1], qRL[0], qRL[1]);
  out.xyzz = iv.xyzz;

  let p1pair = heP.pairedHalfedge;
  let dirP = fp64_add(fn_get(0u, p1 / 3).z, fn_get(0u, p1pair / 3).z);
  let q1pair = heQ.pairedHalfedge;
  let dirQ = fp64_add(fn_get(1u, q1 / 3).z, fn_get(1u, q1pair / 3).z);
  if (!shadows_pred(out.xyzz[2], out.xyzz[3],
                    fp64_sub(withSign_f64(expandP, dirP), dirQ))) {
    out.s11 = 0;
  }
  return out;
}

// ---- Kernel02: vert_A × face_B. ----
struct K02Result {
  s02: i32,
  z02: vec2<u32>,
}

fn kernel02_fp64(a0: i32, b2: i32, expandP: bool, fwd: bool,
                 meshA: u32, meshB: u32) -> K02Result {
  var out: K02Result;
  out.s02 = 0;
  out.z02 = f64_nan();

  var k: i32 = 0;
  var yzzRL: array<Vec3F64, 2>;
  var shadows: bool = false;

  for (var i: i32 = 0; i < 3; i = i + 1) {
    let b1 = 3 * b2 + i;
    let edgeB = he_get(meshB, b1);
    let edgeB_fwd = edgeB.startVert < edgeB.endVert;
    var b1F: i32;
    if (edgeB_fwd) { b1F = b1; } else { b1F = edgeB.pairedHalfedge; }
    let r = shadow01_fp64(a0, b1F, fwd, expandP, meshA, meshB);
    if (r.finite) {
      let sign = select(1, -1, fwd == edgeB_fwd);
      out.s02 = out.s02 + r.s01 * sign;
      if (k < 2 && (k == 0 || (r.s01 != 0) != shadows)) {
        shadows = r.s01 != 0;
        yzzRL[k] = Vec3F64(r.yz_y, r.yz_z, r.yz_z);
        k = k + 1;
      }
    }
  }

  if (out.s02 == 0 || k < 2) {
    out.s02 = 0;
    return out;
  }
  let vertPosA = vp_get(meshA, a0);
  let interp = interpolate_fp64(yzzRL[0], yzzRL[1], vertPosA.y);
  out.z02 = interp.z;

  let fnB = fn_get(meshB, b2);
  if (fwd) {
    if (!shadows_pred(vertPosA.z, out.z02, fp64_neg(fnB.z))) { out.s02 = 0; }
  } else {
    if (!shadows_pred(out.z02, vertPosA.z, withSign_f64(expandP, fnB.z))) {
      out.s02 = 0;
    }
  }
  return out;
}

// ---- Kernel12: edge_A × face_B → (x12, v12). ----
struct K12Result {
  x12: i32,
  v12: Vec3F64,
}

fn kernel12_fp64(a1: i32, b2: i32, expandP: bool, fwd: bool,
                 meshA: u32, meshB: u32) -> K12Result {
  var out: K12Result;
  out.x12 = 0;
  out.v12 = Vec3F64(f64_nan(), f64_nan(), f64_nan());

  var k: i32 = 0;
  var xzyLR0: array<Vec3F64, 2>;
  var xzyLR1: array<Vec3F64, 2>;
  var shadows: bool = false;

  let edgeA = he_get(meshA, a1);
  let vertA_s = edgeA.startVert;
  let vertA_e = edgeA.endVert;

  for (var i: i32 = 0; i < 2; i = i + 1) {
    let vertA = select(vertA_e, vertA_s, i == 0);
    let r = kernel02_fp64(vertA, b2, expandP, fwd, meshA, meshB);
    if (fp64_is_finite(r.z02) && !fp64_is_nan(r.z02)) {
      let is_start = (vertA == vertA_s);
      let sign = select(-1, 1, is_start == fwd);
      out.x12 = out.x12 + r.s02 * sign;
      if (k < 2 && (k == 0 || (r.s02 != 0) != shadows)) {
        shadows = r.s02 != 0;
        let vp = vp_get(meshA, vertA);
        // swap .y and .z:
        xzyLR0[k] = Vec3F64(vp.x, vp.z, vp.y);
        xzyLR1[k] = xzyLR0[k];
        xzyLR1[k].y = r.z02;
        k = k + 1;
      }
    }
  }

  for (var i: i32 = 0; i < 3; i = i + 1) {
    let b1 = 3 * b2 + i;
    let edgeB = he_get(meshB, b1);
    let edgeB_fwd = edgeB.startVert < edgeB.endVert;
    var b1F: i32;
    if (edgeB_fwd) { b1F = b1; } else { b1F = edgeB.pairedHalfedge; }
    // k11 called with (a1, b1F) if forward else (b1F, a1)
    var r: K11Result;
    if (fwd) { r = kernel11_fp64(a1, b1F, expandP); }
    else     { r = kernel11_fp64(b1F, a1, expandP); }
    if (fp64_is_finite(r.xyzz[0]) && !fp64_is_nan(r.xyzz[0])) {
      let sign = select(1, -1, edgeB_fwd);
      out.x12 = out.x12 - r.s11 * sign;
      if (k < 2 && (k == 0 || (r.s11 != 0) != shadows)) {
        shadows = r.s11 != 0;
        xzyLR0[k] = Vec3F64(r.xyzz[0], r.xyzz[2], r.xyzz[1]);
        xzyLR1[k] = xzyLR0[k];
        xzyLR1[k].y = r.xyzz[3];
        if (!fwd) {
          let tmp = xzyLR0[k].y;
          xzyLR0[k].y = xzyLR1[k].y;
          xzyLR1[k].y = tmp;
        }
        k = k + 1;
      }
    }
  }

  if (out.x12 == 0 || k < 2) {
    out.x12 = 0;
    return out;
  }
  let iv = intersect_fp64_inline(xzyLR0[0], xzyLR0[1], xzyLR1[0], xzyLR1[1]);
  // v12.x = xzyy[0]; v12.y = xzyy[2]; v12.z = xzyy[1];
  out.v12 = Vec3F64(iv.xyzz[0], iv.xyzz[2], iv.xyzz[1]);
  return out;
}

// ---- Entry point: runs Kernel12 for each pair. ----
@compute @workgroup_size(64)
fn kernel12_dispatch(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let pair = pair_list[i];
  let expandP = params.expandP != 0u;
  let fwd = params.forward != 0u;
  // "a" is inP when fwd, else inQ. "b" is the other.
  let meshA: u32 = select(1u, 0u, fwd);
  let meshB: u32 = select(0u, 1u, fwd);
  let r = kernel12_fp64(pair.x, pair.y, expandP, fwd, meshA, meshB);
  out_x[i] = r.x12;
  out_v[i] = r.v12;
}
