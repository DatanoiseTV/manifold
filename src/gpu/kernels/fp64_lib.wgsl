// WGSL IEEE 754 double-precision emulation via u32 integer math.
//
// Why integer? WGSL float ops are subject to compiler fast-math (Metal in
// particular collapses Dekker/two_sum primitives to fp32 — gpuweb#2076).
// Integer bit manipulation is not. This gives true fp64 precision on every
// WebGPU backend (Metal / Vulkan / D3D12).
//
// Representation: vec2<u32> where .x = low 32 bits, .y = high 32 bits of
// the canonical IEEE 754 binary64 encoding. sign | exp(11) | mantissa(52).
//
// Simplifications vs. Mesa's float64.glsl:
//   - Flush-to-zero for subnormals (impact on manifold: negligible)
//   - Round-to-nearest; no sticky-bit tracking on the last bit round
//   - NaN propagation handled minimally; operations on NaN return NaN
//
// Precision target: 0-1 ULP on normal inputs. Empirically verified against
// CPU fp64 before any Kernel11/02/12 port is attempted.

// ---------------------------------------------------------------------------
// Bit manipulation helpers
// ---------------------------------------------------------------------------

// Unsigned 32x32 -> 64-bit multiply. Result is (lo, hi).
fn umul32(a: u32, b: u32) -> vec2<u32> {
  let aLo = a & 0xFFFFu;
  let aHi = a >> 16u;
  let bLo = b & 0xFFFFu;
  let bHi = b >> 16u;

  let ll = aLo * bLo;           // 0..31
  let lh = aLo * bHi;           // 16..47
  let hl = aHi * bLo;           // 16..47
  let hh = aHi * bHi;           // 32..63

  // Sum the middle 32-bit partials with explicit carry tracking.
  let mid_lo = (lh << 16u) + (hl << 16u);
  let mid_lo_carry = select(0u, 1u, (lh << 16u) > 0xFFFFFFFFu - (hl << 16u));
  let low = ll + mid_lo;
  let low_carry = select(0u, 1u, low < ll);
  let hi = hh + (lh >> 16u) + (hl >> 16u) + mid_lo_carry + low_carry;
  return vec2<u32>(low, hi);
}

// Unsigned 64-bit add, returns (sum_lo, sum_hi) with carry in/out folded.
fn uadd64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  let carry = select(0u, 1u, lo < a.x);
  let hi = a.y + b.y + carry;
  return vec2<u32>(lo, hi);
}

fn usub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x - b.x;
  let borrow = select(0u, 1u, a.x < b.x);
  let hi = a.y - b.y - borrow;
  return vec2<u32>(lo, hi);
}

fn ult64(a: vec2<u32>, b: vec2<u32>) -> bool {
  return (a.y < b.y) || ((a.y == b.y) && (a.x < b.x));
}

fn clz64(a: vec2<u32>) -> u32 {
  if (a.y != 0u) { return countLeadingZeros(a.y); }
  if (a.x != 0u) { return 32u + countLeadingZeros(a.x); }
  return 64u;
}

// Shift right (lo, hi) by `n` (0..63). 64-bit logical shift.
fn shr64(a: vec2<u32>, n: u32) -> vec2<u32> {
  if (n == 0u) { return a; }
  if (n >= 64u) { return vec2<u32>(0u, 0u); }
  if (n >= 32u) {
    return vec2<u32>(a.y >> (n - 32u), 0u);
  }
  let lo = (a.x >> n) | (a.y << (32u - n));
  let hi = a.y >> n;
  return vec2<u32>(lo, hi);
}

// Shift right with "sticky" — returns shifted value AND whether any bits
// were shifted out. Required for correct IEEE rounding.
struct ShiftResult { v: vec2<u32>, sticky: u32 }

fn shr64_sticky(a: vec2<u32>, n: u32) -> ShiftResult {
  var out: ShiftResult;
  if (n == 0u) { out.v = a; out.sticky = 0u; return out; }
  if (n >= 64u) {
    out.v = vec2<u32>(0u, 0u);
    out.sticky = select(0u, 1u, (a.x | a.y) != 0u);
    return out;
  }
  // Build sticky = nonzero-or-of-shifted-out bits
  let shifted = shr64(a, n);
  let mask_lo = select((1u << n) - 1u, 0xFFFFFFFFu, n >= 32u);
  let mask_hi = select(0u, (1u << (n - 32u)) - 1u, n >= 32u);
  var stickyBits: u32 = a.x & mask_lo;
  if (n >= 32u) { stickyBits = stickyBits | (a.y & mask_hi); }
  out.v = shifted;
  out.sticky = select(0u, 1u, stickyBits != 0u);
  return out;
}

fn shl64(a: vec2<u32>, n: u32) -> vec2<u32> {
  if (n == 0u) { return a; }
  if (n >= 64u) { return vec2<u32>(0u, 0u); }
  if (n >= 32u) {
    return vec2<u32>(0u, a.x << (n - 32u));
  }
  let hi = (a.y << n) | (a.x >> (32u - n));
  let lo = a.x << n;
  return vec2<u32>(lo, hi);
}

// ---------------------------------------------------------------------------
// fp64 construction / inspection
// ---------------------------------------------------------------------------

fn fp64_sign(a: vec2<u32>) -> u32 { return a.y & 0x80000000u; }
fn fp64_exp_bits(a: vec2<u32>) -> u32 { return (a.y >> 20u) & 0x7FFu; }
fn fp64_frac_hi(a: vec2<u32>) -> u32 { return a.y & 0x000FFFFFu; }
fn fp64_frac_lo(a: vec2<u32>) -> u32 { return a.x; }
fn fp64_is_zero(a: vec2<u32>) -> bool {
  return (a.x | (a.y & 0x7FFFFFFFu)) == 0u;
}
fn fp64_is_nan(a: vec2<u32>) -> bool {
  return (fp64_exp_bits(a) == 0x7FFu) &&
         ((a.x != 0u) || (fp64_frac_hi(a) != 0u));
}
fn fp64_is_inf(a: vec2<u32>) -> bool {
  return (fp64_exp_bits(a) == 0x7FFu) &&
         (a.x == 0u) && (fp64_frac_hi(a) == 0u);
}
fn fp64_is_finite(a: vec2<u32>) -> bool {
  return fp64_exp_bits(a) != 0x7FFu;
}

fn fp64_abs(a: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x, a.y & 0x7FFFFFFFu);
}

fn fp64_neg(a: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x, a.y ^ 0x80000000u);
}

fn fp64_zero() -> vec2<u32> { return vec2<u32>(0u, 0u); }
fn fp64_one() -> vec2<u32> { return vec2<u32>(0u, 0x3FF00000u); }
fn fp64_nan() -> vec2<u32> { return vec2<u32>(0u, 0x7FF80000u); }
fn fp64_inf(sign: u32) -> vec2<u32> {
  return vec2<u32>(0u, sign | 0x7FF00000u);
}

// Pack (sign_bit, biased_exp, 53-bit mantissa in (lo, hi)) into fp64.
// mantissa must have its implicit leading 1 bit at bit 52 (so fracHi has
// bit 20 set before masking). exp is biased; out-of-range -> 0 or Inf.
fn fp64_pack(sign: u32, exp: i32, mantHi: u32, mantLo: u32) -> vec2<u32> {
  if (exp >= 0x7FF) { return fp64_inf(sign); }
  if (exp <= 0) { return vec2<u32>(0u, sign); }  // flush subnormals
  let hi = sign | (u32(exp) << 20u) | (mantHi & 0x000FFFFFu);
  return vec2<u32>(mantLo, hi);
}

// Extract the mantissa including the implicit leading 1 bit, as a 53-bit
// value in (lo, hi). hi has bit 20 set for normal values; subnormals flushed.
fn fp64_extract_mant(a: vec2<u32>) -> vec2<u32> {
  if (fp64_exp_bits(a) == 0u) { return vec2<u32>(0u, 0u); }
  return vec2<u32>(a.x, fp64_frac_hi(a) | 0x00100000u);
}

// Unbiased exponent for normals; subnormals flushed so exp doesn't matter.
fn fp64_unbiased_exp(a: vec2<u32>) -> i32 {
  return i32(fp64_exp_bits(a)) - 1023;
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

fn fp64_eq(a: vec2<u32>, b: vec2<u32>) -> bool {
  if (fp64_is_nan(a) || fp64_is_nan(b)) { return false; }
  if (fp64_is_zero(a) && fp64_is_zero(b)) { return true; }  // +0 == -0
  return (a.x == b.x) && (a.y == b.y);
}

fn fp64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
  if (fp64_is_nan(a) || fp64_is_nan(b)) { return false; }
  if (fp64_is_zero(a) && fp64_is_zero(b)) { return false; }
  let aSign = fp64_sign(a);
  let bSign = fp64_sign(b);
  if (aSign != bSign) { return aSign != 0u; }  // negative < positive
  // Same sign: compare magnitudes; flip for negatives.
  let aMag = vec2<u32>(a.x, a.y & 0x7FFFFFFFu);
  let bMag = vec2<u32>(b.x, b.y & 0x7FFFFFFFu);
  if (aSign != 0u) { return ult64(bMag, aMag); }
  return ult64(aMag, bMag);
}

fn fp64_le(a: vec2<u32>, b: vec2<u32>) -> bool {
  return fp64_eq(a, b) || fp64_lt(a, b);
}

// ---------------------------------------------------------------------------
// fp32 <-> fp64 conversions
// ---------------------------------------------------------------------------

fn fp32_to_fp64(x: f32) -> vec2<u32> {
  let bits = bitcast<u32>(x);
  let sign = bits & 0x80000000u;
  let exp32 = (bits >> 23u) & 0xFFu;
  let frac32 = bits & 0x007FFFFFu;

  if (exp32 == 0xFFu) {
    if (frac32 != 0u) {
      return vec2<u32>(0u, sign | 0x7FF80000u);  // NaN
    }
    return fp64_inf(sign);
  }
  if (exp32 == 0u) {
    return vec2<u32>(0u, sign);  // flush subnormal inputs to 0
  }
  let exp64 = i32(exp32) - 127 + 1023;
  let hi = sign | (u32(exp64) << 20u) | (frac32 >> 3u);
  let lo = frac32 << 29u;
  return vec2<u32>(lo, hi);
}

fn fp64_to_fp32(a: vec2<u32>) -> f32 {
  let sign = fp64_sign(a);
  let eb = fp64_exp_bits(a);
  if (eb == 0u) { return bitcast<f32>(sign); }
  if (eb == 0x7FFu) {
    if ((a.x | fp64_frac_hi(a)) != 0u) {
      return bitcast<f32>(sign | 0x7FC00000u);
    }
    return bitcast<f32>(sign | 0x7F800000u);
  }
  let exp32 = i32(eb) - 1023 + 127;
  if (exp32 >= 255) { return bitcast<f32>(sign | 0x7F800000u); }
  if (exp32 <= 0) { return bitcast<f32>(sign); }  // flush to 0
  let mant = (fp64_frac_hi(a) << 3u) | (a.x >> 29u);
  // Round-to-nearest-even based on dropped 29 bits of a.x.
  let lower = a.x & 0x1FFFFFFFu;
  let halfbit = 0x10000000u;
  var out = sign | (u32(exp32) << 23u) | mant;
  if (lower > halfbit) { out = out + 1u; }
  else if (lower == halfbit && (mant & 1u) != 0u) { out = out + 1u; }
  return bitcast<f32>(out);
}

// ---------------------------------------------------------------------------
// fp64 add / sub
// ---------------------------------------------------------------------------

// Round 56-bit mantissa in (mantLo, mantHi) with guard bits 0..2 to 53 bits.
// Returns the packed fp64 and carries exp out via modification.
fn fp64_round_and_pack(sign: u32, exp: i32, mantHi: u32, mantLo: u32,
                       sticky: u32) -> vec2<u32> {
  // Inputs: 56-bit mantissa packed as (mantHi:mantLo) where bit 23 of
  // mantHi is the implicit leading 1. Bits 2..0 of mantLo are guard/round/
  // sticky-merged. We shift the whole 56-bit value right by 3 to land on
  // a 53-bit mantissa (bit 20 of hi = implicit 1).
  let round_bit = (mantLo >> 2u) & 1u;
  let tail_bits = (mantLo & 3u) | sticky;
  var hiOut = mantHi >> 3u;
  var loOut = (mantLo >> 3u) | (mantHi << 29u);
  var ex = exp;

  let rnd = (round_bit != 0u) && ((tail_bits != 0u) || ((loOut & 1u) != 0u));
  if (rnd) {
    let sum = uadd64(vec2<u32>(loOut, hiOut), vec2<u32>(1u, 0u));
    loOut = sum.x;
    hiOut = sum.y;
    if ((hiOut & 0x00200000u) != 0u) {
      // 53-bit mantissa overflowed; shift right one, bump exp.
      loOut = (loOut >> 1u) | (hiOut << 31u);
      hiOut = hiOut >> 1u;
      ex = ex + 1;
    }
  }
  return fp64_pack(sign, ex, hiOut, loOut);
}

fn fp64_add_impl(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  // Handle NaN / Inf / zero fast paths.
  if (fp64_is_nan(a) || fp64_is_nan(b)) { return fp64_nan(); }
  if (fp64_is_inf(a)) {
    if (fp64_is_inf(b) && fp64_sign(a) != fp64_sign(b)) { return fp64_nan(); }
    return a;
  }
  if (fp64_is_inf(b)) { return b; }
  if (fp64_is_zero(a)) { return b; }
  if (fp64_is_zero(b)) { return a; }

  let aSign = fp64_sign(a);
  let bSign = fp64_sign(b);
  let aExp = fp64_unbiased_exp(a);
  let bExp = fp64_unbiased_exp(b);
  var aMant = fp64_extract_mant(a);  // 53 bits, hi bit at pos 20 of .y
  var bMant = fp64_extract_mant(b);
  // Shift up by 3 guard bits for rounding. Effective mantissa width: 56.
  aMant = shl64(aMant, 3u);
  bMant = shl64(bMant, 3u);

  // Align: shift the smaller-exp operand right.
  var exp = aExp;
  var sticky: u32 = 0u;
  if (aExp >= bExp) {
    let diff = u32(aExp - bExp);
    if (diff > 60u) { return a; }  // b is insignificant
    let s = shr64_sticky(bMant, diff);
    bMant = s.v;
    sticky = s.sticky;
  } else {
    let diff = u32(bExp - aExp);
    if (diff > 60u) { return b; }
    let s = shr64_sticky(aMant, diff);
    aMant = s.v;
    sticky = s.sticky;
    exp = bExp;
  }

  var resSign: u32;
  var mant: vec2<u32>;

  if (aSign == bSign) {
    resSign = aSign;
    mant = uadd64(aMant, bMant);
    // May overflow bit 23 (was bit 20+3): shift right.
    if ((mant.y & 0x01000000u) != 0u) {
      let s = shr64_sticky(mant, 1u);
      mant = s.v;
      sticky = sticky | s.sticky;
      exp = exp + 1;
    }
  } else {
    // Subtract: |larger| - |smaller|, sign of result = sign of larger.
    if (ult64(aMant, bMant)) {
      mant = usub64(bMant, aMant);
      resSign = bSign;
    } else if (ult64(bMant, aMant)) {
      mant = usub64(aMant, bMant);
      resSign = aSign;
    } else {
      // Equal magnitudes with opposite sign -> exact zero.
      return fp64_zero();
    }
    // Normalize: find MSB and shift left to put it at bit 23.
    let lz = clz64(mant);
    // We want MSB at position 55 (since 53-bit mantissa + 3 guard = 56, bit
    // index 55 is top). clz tells us how many zeros at the top of 64 bits,
    // so we need to shift left by lz - 8 (since bit 55 => 8 zeros from top).
    if (lz >= 8u) {
      let shift = lz - 8u;
      mant = shl64(mant, shift);
      exp = exp - i32(shift);
    } else {
      let shift = 8u - lz;
      let s = shr64_sticky(mant, shift);
      mant = s.v;
      sticky = sticky | s.sticky;
      exp = exp + i32(shift);
    }
  }

  let biased = exp + 1023;
  return fp64_round_and_pack(resSign, biased, mant.y, mant.x, sticky);
}

fn fp64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return fp64_add_impl(a, b);
}

fn fp64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return fp64_add_impl(a, fp64_neg(b));
}

// CPU-side packs fp64 as (hi: f32, lo: f32) such that hi+lo == v. Reassemble.
fn fp64_from_hilo(hi: f32, lo: f32) -> vec2<u32> {
  return fp64_add(fp32_to_fp64(hi), fp32_to_fp64(lo));
}

fn fp64_to_hilo(a: vec2<u32>) -> vec2<f32> {
  let hi = fp64_to_fp32(a);
  let hi64 = fp32_to_fp64(hi);
  let diff = fp64_sub(a, hi64);
  let lo = fp64_to_fp32(diff);
  return vec2<f32>(hi, lo);
}

// ---------------------------------------------------------------------------
// fp64 multiply
// ---------------------------------------------------------------------------

fn fp64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  if (fp64_is_nan(a) || fp64_is_nan(b)) { return fp64_nan(); }
  let sign = fp64_sign(a) ^ fp64_sign(b);
  if (fp64_is_inf(a) || fp64_is_inf(b)) {
    if (fp64_is_zero(a) || fp64_is_zero(b)) { return fp64_nan(); }
    return fp64_inf(sign);
  }
  if (fp64_is_zero(a) || fp64_is_zero(b)) { return vec2<u32>(0u, sign); }

  let aExp = fp64_unbiased_exp(a);
  let bExp = fp64_unbiased_exp(b);
  let aMant = fp64_extract_mant(a);  // 53-bit in (.x, .y bit 20)
  let bMant = fp64_extract_mant(b);

  // 53x53 -> 106-bit product. Split into 32-bit pieces.
  // aMant = aHi*2^32 + aLo (where aHi has bit 20 set for normals)
  // bMant = bHi*2^32 + bLo
  // product = aHi*bHi*2^64 + (aHi*bLo + aLo*bHi)*2^32 + aLo*bLo

  let aLo = aMant.x; let aHi = aMant.y;
  let bLo = bMant.x; let bHi = bMant.y;

  let hh = umul32(aHi, bHi);  // -> bits 64..127
  let hl = umul32(aHi, bLo);  // -> bits 32..95
  let lh = umul32(aLo, bHi);  // -> bits 32..95
  let ll = umul32(aLo, bLo);  // -> bits 0..63

  // Accumulate into (w0, w1, w2, w3) where w0 is lowest 32.
  var w0 = ll.x;
  // Sum bits 32..63: ll.y + hl.x + lh.x, carry into bits 64..95.
  var mid = ll.y;
  var c1: u32 = 0u;
  let s1 = mid + hl.x; if (s1 < mid) { c1 = c1 + 1u; } mid = s1;
  let s2 = mid + lh.x; if (s2 < mid) { c1 = c1 + 1u; } mid = s2;
  var w1 = mid;

  // Sum bits 64..95: hl.y + lh.y + hh.x + c1, carry into 96..127.
  var high = hl.y;
  var c2: u32 = 0u;
  let t1 = high + lh.y; if (t1 < high) { c2 = c2 + 1u; } high = t1;
  let t2 = high + hh.x; if (t2 < high) { c2 = c2 + 1u; } high = t2;
  let t3 = high + c1;   if (t3 < high) { c2 = c2 + 1u; } high = t3;
  var w2 = high;
  var w3 = hh.y + c2;

  // 106-bit product sits in bits 0..105. For normal inputs with aHi bit 20
  // and bHi bit 20, the MSB of the product is at bit 104 or 105.
  // We need the top 53 bits (+ 3 guard) = 56 bits.
  // Work with top 64 bits = w3:w2 (bits 64..127).
  var top = vec2<u32>(w2, w3);

  // Normalize: product MSB is at bit 41 or 42 of w3.
  // 53x53 multiply: MSB of result at bit 105 or 104 (depending on whether
  // product overflowed). We want bit 52 of our 53-bit result.
  // Raw shift needed: bits 53..105 = the 53-bit result (with rounding bits).
  // We need to keep bits [52-guardBits .. 104 or 105].
  var lowBits = vec2<u32>(w0, w1);
  var exp = aExp + bExp;

  // If bit 41 of w3 is set (MSB at bit 105), shift right by 1 to place MSB
  // at bit 40. Otherwise MSB is at bit 40 (bit 104).
  if ((w3 >> 9u) != 0u) {  // bit 41 is set
    // Shift 128-bit value right by 1.
    let new_w0 = (lowBits.x >> 1u) | (lowBits.y << 31u);
    let new_w1 = (lowBits.y >> 1u) | (top.x << 31u);
    let new_w2 = (top.x >> 1u) | (top.y << 31u);
    let new_w3 = top.y >> 1u;
    lowBits = vec2<u32>(new_w0, new_w1);
    top = vec2<u32>(new_w2, new_w3);
    exp = exp + 1;
  }

  // Now MSB is at bit 40 of top.y (= bit 104 of the full product).
  // We want the 56-bit result (53 + 3 guard) to sit in (top.x, top.y) with
  // MSB at top.y bit 23.
  // Currently MSB at top.y bit 8 (since 40 - 32 = 8).
  // Shift LEFT by 15 bits so MSB ends up at bit 23.
  // 64-bit shift left of (top.x, top.y) by 15.
  let sh_amount: u32 = 15u;
  let top_shifted = vec2<u32>(
    top.x << sh_amount,
    (top.y << sh_amount) | (top.x >> (32u - sh_amount))
  );
  // Bits that would shift out of top.x:
  let spill = top.x >> (32u - sh_amount);  // goes into top_shifted.y (above)
  // Bits from lowBits that shift into top_shifted.x:
  let fillLow = lowBits.y >> (32u - sh_amount);
  let mant_lo = top_shifted.x | fillLow;
  let mant_hi = top_shifted.y;
  // Sticky = OR of all bits below mant_lo's bottom.
  let droppedBits = (lowBits.y << sh_amount) | lowBits.x;
  let sticky = select(0u, 1u, droppedBits != 0u);

  let biased = exp + 1023;
  return fp64_round_and_pack(sign, biased, mant_hi, mant_lo, sticky);
}

// ---------------------------------------------------------------------------
// fp64 divide via Newton-Raphson on fp32 reciprocal
// ---------------------------------------------------------------------------

// Apply a raw exponent adjustment without touching mantissa. Used to rescale
// operands into fp32 representable range before computing an fp32 reciprocal.
fn fp64_scale_exp(a: vec2<u32>, delta: i32) -> vec2<u32> {
  if (fp64_is_zero(a) || !fp64_is_finite(a)) { return a; }
  let oldExp = fp64_unbiased_exp(a);
  let newBiased = oldExp + delta + 1023;
  if (newBiased >= 0x7FF) { return fp64_inf(fp64_sign(a)); }
  if (newBiased <= 0) { return vec2<u32>(0u, fp64_sign(a)); }
  let hi = (a.y & 0x800FFFFFu) | (u32(newBiased) << 20u);
  return vec2<u32>(a.x, hi);
}

fn fp64_div(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  if (fp64_is_nan(a) || fp64_is_nan(b)) { return fp64_nan(); }
  let sign = fp64_sign(a) ^ fp64_sign(b);
  if (fp64_is_zero(b)) {
    if (fp64_is_zero(a)) { return fp64_nan(); }
    return fp64_inf(sign);
  }
  if (fp64_is_inf(a)) {
    if (fp64_is_inf(b)) { return fp64_nan(); }
    return fp64_inf(sign);
  }
  if (fp64_is_inf(b)) { return vec2<u32>(0u, sign); }
  if (fp64_is_zero(a)) { return vec2<u32>(0u, sign); }

  // Rescale b into fp32's normal range so the initial reciprocal estimate
  // is well-conditioned. Then correct the exponent of the final result.
  let bExp = fp64_unbiased_exp(b);
  let bScaled = fp64_scale_exp(b, -bExp);
  let aScaled = fp64_scale_exp(a, -bExp);

  // fp32 reciprocal estimate of |bScaled| (magnitude in [1, 2)).
  let bf32 = fp64_to_fp32(fp64_abs(bScaled));
  let r0 = 1.0 / bf32;
  var x = fp32_to_fp64(r0);
  // Restore sign
  if (fp64_sign(bScaled) != 0u) { x = fp64_neg(x); }

  // Two Newton iterations: 24 -> 47 -> 94 bits. Saturates the 53-bit fp64
  // mantissa; a third iteration would change nothing and burns ~3 fp64_muls.
  let two = fp64_pack(0u, 1 + 1023, 0x00100000u, 0u);  // exactly 2.0
  for (var i = 0u; i < 2u; i = i + 1u) {
    let bx = fp64_mul(bScaled, x);
    let t = fp64_sub(two, bx);
    x = fp64_mul(x, t);
  }
  return fp64_mul(aScaled, x);
}
