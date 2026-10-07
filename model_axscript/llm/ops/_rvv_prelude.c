//
// Copyright Axelera AI, 2026
// SPDX-License-Identifier: Apache-2.0
//
// Shared fp32 RVV math helpers for the Europa DeltaNet kernels.
//
// The Europa AI core is built with -nostdlib against libkernsup_aicore, which
// exports memcpy and nothing else from libm: there is no expf/logf/sqrtf to
// call and RVV has no transcendental instruction. Every nonlinearity the gated
// DeltaNet needs is therefore open-coded here, once, in fp32.
//
// Every kernel's source is pasted into ONE translation unit per model, so this
// file is include-guarded (two ops both prepending it would otherwise redefine
// every symbol under -Werror) and every symbol is static and ax_-prefixed.
//
// Accuracy target is fp32 rounding (~1.2e-7 relative). Polynomial degrees are
// chosen so truncation sits below that, leaving rounding dominant.

#ifndef AX_RVV_PRELUDE
#define AX_RVV_PRELUDE

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// The cfunc ABI has no float argument type, so scalar floats arrive as fp32 bit
// patterns in a UINT64. Punning through a union avoids <string.h>/memcpy, which
// a freestanding -nostdlib target does not provide.
static inline float
ax_bits_to_f32(uint32_t bits)
{
  union {
    uint32_t u;
    float f;
  } pun;
  pun.u = bits;
  return pun.f;
}

// exp(x), fp32. x = n*ln2 + r with |r| <= ln2/2; exp(r) by a degree-7 Taylor
// polynomial (truncation ~5e-9) and 2^n by injecting n into the biased exponent.
// x is clamped to +-87 so that n+127 stays inside [1, 254] -- at n = -127 the
// injected exponent field would be 0, which encodes a denormal, not 2^-127.
static inline vfloat32m8_t
ax_vexp_f32m8(vfloat32m8_t x, size_t vl)
{
  const float LOG2E = 1.44269504088896340736f;
  // ln2 split hi/lo so that x - n*ln2 stays exact to more than fp32.
  const float LN2_HI = 0.693359375f;
  const float LN2_LO = -2.12194440e-4f;

  x = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(x, -87.0f, vl), 87.0f, vl);

  vint32m8_t ni = __riscv_vfcvt_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(x, LOG2E, vl), vl);
  vfloat32m8_t n = __riscv_vfcvt_f_x_v_f32m8(ni, vl);

  vfloat32m8_t r = __riscv_vfnmsac_vf_f32m8(x, LN2_HI, n, vl);
  r = __riscv_vfnmsac_vf_f32m8(r, LN2_LO, n, vl);

  vfloat32m8_t p = __riscv_vfmv_v_f_f32m8(1.0f / 5040.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), 1.0f / 720.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), 1.0f / 120.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), 1.0f / 24.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), 1.0f / 6.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), 0.5f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), 1.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), 1.0f, vl);

  vint32m8_t bits = __riscv_vsll_vx_i32m8(__riscv_vadd_vx_i32m8(ni, 127, vl), 23, vl);
  return __riscv_vfmul_vv_f32m8(p, __riscv_vreinterpret_v_i32m8_f32m8(bits), vl);
}

// log(1+u) for u in [0, 1], fp32. Uses 2*atanh(t) with t = u/(u+2), so t is in
// [0, 1/3] and the odd series converges fast; truncation after t^13 is ~1e-8
// relative. Restricting the domain to [0,1] avoids needing a general log (no
// exponent extraction, no argument reduction) -- softplus only ever needs this
// range, because it feeds log1p(exp(-|x|)).
static inline vfloat32m8_t
ax_vlog1p_unit_f32m8(vfloat32m8_t u, size_t vl)
{
  vfloat32m8_t t = __riscv_vfdiv_vv_f32m8(u, __riscv_vfadd_vf_f32m8(u, 2.0f, vl), vl);
  vfloat32m8_t s = __riscv_vfmul_vv_f32m8(t, t, vl);

  vfloat32m8_t p = __riscv_vfmv_v_f_f32m8(1.0f / 13.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, s, vl), 1.0f / 11.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, s, vl), 1.0f / 9.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, s, vl), 1.0f / 7.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, s, vl), 1.0f / 5.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, s, vl), 1.0f / 3.0f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, s, vl), 1.0f, vl);

  return __riscv_vfmul_vf_f32m8(__riscv_vfmul_vv_f32m8(p, t, vl), 2.0f, vl);
}

// softplus(x) = log(1 + e^x), fp32, via max(x,0) + log1p(exp(-|x|)).
// The identity keeps the exp argument non-positive, so it never overflows and
// the log1p argument always lands in (0, 1] -- the only domain
// ax_vlog1p_unit_f32m8 covers.
static inline vfloat32m8_t
ax_vsoftplus_f32m8(vfloat32m8_t x, size_t vl)
{
  vfloat32m8_t ax = __riscv_vfabs_v_f32m8(x, vl);
  vfloat32m8_t lo
      = ax_vlog1p_unit_f32m8(ax_vexp_f32m8(__riscv_vfneg_v_f32m8(ax, vl), vl), vl);
  return __riscv_vfadd_vv_f32m8(__riscv_vfmax_vf_f32m8(x, 0.0f, vl), lo, vl);
}

// 1/sqrt(x), fp32 scalar. sqrt is an F-extension instruction but there is no libm
// wrapper to call, so reach it through inline asm; vfrsqrt7 is only a 7-bit estimate
// and would need Newton steps to match. Used once per reduction, not per element.
static inline float
ax_rsqrtf(float x)
{
  float root;
  __asm__ volatile("fsqrt.s %0, %1" : "=f"(root) : "f"(x));
  return 1.0f / root;
}

// sqrt(x), fp32 scalar, same reasoning as ax_rsqrtf.
static inline float
ax_sqrtf(float x)
{
  float root;
  __asm__ volatile("fsqrt.s %0, %1" : "=f"(root) : "f"(x));
  return root;
}

// 1/x to fp32 accuracy, from the 7-bit estimate plus two Newton steps e <- e(2 - x e),
// which double the correct bits each time (7 -> 14 -> 28, past fp32's 24).
//
// Worth five instructions because division is not an ordinary vector op here: measured
// at vl=1024, vfdiv costs ~2950 cycles against 34 for an FMA, while this sequence costs
// 296. Anything with a divide in an elementwise inner loop is dominated by it.
static inline vfloat32m8_t
ax_vrecip_f32m8(vfloat32m8_t x, size_t vl)
{
  vfloat32m8_t e = __riscv_vfrec7_v_f32m8(x, vl);
  for (int step = 0; step < 2; step++) {
    e = __riscv_vfmul_vv_f32m8(e,
        __riscv_vfnmsac_vv_f32m8(__riscv_vfmv_v_f_f32m8(2.0f, vl), x, e, vl), vl);
  }
  return e;
}

// sigmoid(x), fp32. z = exp(-|x|) keeps the exponential bounded by 1, so the result is
// z/(1+z) for x < 0 and 1/(1+z) otherwise.
//
// The select is on the NUMERATOR, not the two quotients: computing both quotients and
// merging them costs two divides, and one divide costs 88 FMAs. Merging first leaves a
// single reciprocal, whose argument is in [1,2] and so is perfectly conditioned for the
// Newton iteration. That took the whole sigmoid from 6742 cycles at vl=1024 to ~1200.
static inline vfloat32m8_t
ax_vsigmoid_f32m8(vfloat32m8_t x, size_t vl)
{
  vfloat32m8_t z
      = ax_vexp_f32m8(__riscv_vfneg_v_f32m8(__riscv_vfabs_v_f32m8(x, vl), vl), vl);
  vfloat32m8_t num = __riscv_vmerge_vvm_f32m8(__riscv_vfmv_v_f_f32m8(1.0f, vl),
      z, __riscv_vmflt_vf_f32m8_b4(x, 0.0f, vl), vl);
  return __riscv_vfmul_vv_f32m8(
      num, ax_vrecip_f32m8(__riscv_vfadd_vf_f32m8(z, 1.0f, vl), vl), vl);
}

#endif // AX_RVV_PRELUDE
