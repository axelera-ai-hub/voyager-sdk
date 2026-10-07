//
// Copyright Axelera AI, 2026
// SPDX-License-Identifier: Apache-2.0
//
// Chunked gated delta rule, entirely fp32 on the AI core's RVV unit.
//
//     q, k      (bh, seq, dk) f32  -- L2-normed, head-major
//     v         (bh, seq, dv) f32
//     g, beta   (bh, seq)     f32  -- head-major; g is the LOG decay
//     state     (bh, dk, dv)  f32  -- entry state, updated in place
//     out       (bh, seq, dv) f32
//     scratch   flat f32 workspace, carved below
//     chunk     UINT64             -- tokens per chunk (T)
//
// Chunk algebra, for entry state S0 and A_t = prod_{j<=t} alpha_j:
//
//     M[t,i] = beta_t (A_t/A_i)(k_t . k_i),  i < t   (strictly lower, unit diag)
//     RHS[t] = beta_t (v_t - A_t (k_t^T S0))
//     (I + M) Vnew = RHS                             -> forward substitution
//     P[t,i] = (A_t/A_i)(q_t . k_i),  i <= t
//     O      = diag(A)(Q S0) + P Vnew
//     S_T    = A_T S0 + Kgamma^T Vnew,  Kgamma[i] = (A_T/A_i) k_i
//
// Decays are carried in LOG space: gc = cumsum(g) within the chunk and
// A_t/A_i = exp(gc_t - gc_i). Since g <= 0 every exponent is <= 0, so the ratio lands
// in (0,1] and can neither overflow nor divide by an underflowed A_i. The float18 DPU
// path instead had to build the ratio matrix with a pure-multiply recurrence purely to
// dodge division and the absent exponential.
//
// The ratio matrix is never materialised, because it is rank one in log space. Pivoting
// at the chunk's midpoint c = gc[T-1]/2,
//
//     A_t/A_i = exp(gc_t - c) exp(c - gc_i)
//
// so the (T,T) decay matrix is never built. One exp over T elements gives both halves:
//
//     ah[t] = exp(gc_t - c) = A_t/e^c        ea[i] = exp(c - gc_i) = e^c/A_i
//     M = diag(beta ah) (K K^T) diag(ea)     P = diag(ah) (Q K^T) diag(ea)
//     S_T = A_T S0 + e^c (khat^T Vnew),      khat_i = ea[i] k_i,  e^c = sqrt(A_T)
//
// The two Gram matmuls K K^T and Q K^T are the only matmuls here that do not depend on
// the carried state, so they are computed OUTSIDE this kernel on the int8 IMC, batched
// over every (head, chunk) -- 84.8M cycles of RVV became 13.5M -- and arrive as inputs.
// khat therefore exists only for the state update now.
//
// The midpoint pivot is what keeps both factors in range: each exponent is bounded by
// half the chunk's total decay rather than all of it. If even half would overflow fp32,
// the chunk length is halved until it fits, which is exact -- the algebra holds for any
// T, so a shorter chunk only carries the state more often.
//
// The solve is exact forward substitution, and on RVV it is already instruction-optimal:
// each (t,i) pair is one scalar load and one vfmacc at vl=dv, which is exactly what a
// blocked inversion's GEMMs would issue, because vl is capped at dv=128 either way. So
// blocking would only pay if the off-diagonal blocks could leave for the datapath. A
// truncated Neumann series for (I+M)^-1 is not an option regardless: ||M|| reaches ~8 on
// real activations, even though M is nilpotent.

// C(m,n) = A(m,kk) @ B(kk,n), fp32.
//
// Register-blocked 8 output rows at a time, all sharing ONE loaded B vector, which is
// the shape the in-tree reference matmuls use. What this buys is instruction count: a
// vector op on this core costs ~40 cycles of issue overhead plus vl/128 cycles of actual
// work, so at vl=128 it is ~98% overhead and nothing but the instruction count matters.
// The naive one-accumulator form issues a load per FMA; sharing B issues one per eight.
//
// LMUL is m1 because the accumulators must fit the register file: the reference keeps
// accumulators * LMUL at 16 registers, and n here is 128 = exactly VLMAX at m1. Going
// wider is not available to any of these matmuls -- m8 would reach vl=1024 and measures
// 6.9x per element, but n is dv or T, and batching heads to fill the group needs a
// coefficient vector holding one value per 128-lane block. Every way of building one
// costs what it saves: vrgather at m8 is trap-emulated at ~2.1M cycles per op, an
// indexed load faults, and eight m1 splats plus vset measure break-even to 3 digits.
//
// Leading dimensions are explicit because the (T,T) scratch arrays are always
// allocated with row stride `chunk` while a final short chunk has m = n = t_n < chunk.
static void
ax_mm_f32(const float *a, uint64_t lda, const float *b, uint64_t ldb, float *c,
    uint64_t ldc, uint64_t m, uint64_t kk, uint64_t n)
{
  uint64_t i = 0;
  for (; i + 8 <= m; i += 8) {
    const float *ar = a + i * lda;
    float *cr = c + i * ldc;
    for (uint64_t j = 0; j < n;) {
      size_t vl = __riscv_vsetvl_e32m1(n - j);
      vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      vfloat32m1_t a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
      const float *bp = b + j;
      for (uint64_t p = 0; p < kk; p++) {
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(bp, vl);
        bp += ldb;
        a0 = __riscv_vfmacc_vf_f32m1(a0, ar[p], vb, vl);
        a1 = __riscv_vfmacc_vf_f32m1(a1, ar[lda + p], vb, vl);
        a2 = __riscv_vfmacc_vf_f32m1(a2, ar[2 * lda + p], vb, vl);
        a3 = __riscv_vfmacc_vf_f32m1(a3, ar[3 * lda + p], vb, vl);
        a4 = __riscv_vfmacc_vf_f32m1(a4, ar[4 * lda + p], vb, vl);
        a5 = __riscv_vfmacc_vf_f32m1(a5, ar[5 * lda + p], vb, vl);
        a6 = __riscv_vfmacc_vf_f32m1(a6, ar[6 * lda + p], vb, vl);
        a7 = __riscv_vfmacc_vf_f32m1(a7, ar[7 * lda + p], vb, vl);
      }
      __riscv_vse32_v_f32m1(cr + j, a0, vl);
      __riscv_vse32_v_f32m1(cr + ldc + j, a1, vl);
      __riscv_vse32_v_f32m1(cr + 2 * ldc + j, a2, vl);
      __riscv_vse32_v_f32m1(cr + 3 * ldc + j, a3, vl);
      __riscv_vse32_v_f32m1(cr + 4 * ldc + j, a4, vl);
      __riscv_vse32_v_f32m1(cr + 5 * ldc + j, a5, vl);
      __riscv_vse32_v_f32m1(cr + 6 * ldc + j, a6, vl);
      __riscv_vse32_v_f32m1(cr + 7 * ldc + j, a7, vl);
      j += vl;
    }
  }
  for (; i < m; i++) {
    for (uint64_t j = 0; j < n;) {
      size_t vl = __riscv_vsetvl_e32m1(n - j);
      vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      for (uint64_t p = 0; p < kk; p++) {
        acc = __riscv_vfmacc_vf_f32m1(
            acc, a[i * lda + p], __riscv_vle32_v_f32m1(b + p * ldb + j, vl), vl);
      }
      __riscv_vse32_v_f32m1(c + i * ldc + j, acc, vl);
      j += vl;
    }
  }
}

static int
rvv_deltanet_chunk(cfunc_ctx_t *ctx)
{
  cfunc_dltensor_t q, k, v, g, beta, kkm_t, qkm_t, state, out, scratch;
  ctx->api->get_dltensor_arg(ctx, 0, &q);
  ctx->api->get_dltensor_arg(ctx, 1, &k);
  ctx->api->get_dltensor_arg(ctx, 2, &v);
  ctx->api->get_dltensor_arg(ctx, 3, &g);
  ctx->api->get_dltensor_arg(ctx, 4, &beta);
  ctx->api->get_dltensor_arg(ctx, 5, &kkm_t);
  ctx->api->get_dltensor_arg(ctx, 6, &qkm_t);
  ctx->api->get_dltensor_arg(ctx, 7, &state);
  ctx->api->get_dltensor_arg(ctx, 8, &out);
  ctx->api->get_dltensor_arg(ctx, 9, &scratch);
  uint64_t chunk = ctx->api->get_uint64_arg(ctx, 10);

  uint64_t bh = (uint64_t) q.shape[0];
  uint64_t seq = (uint64_t) q.shape[1];
  uint64_t dk = (uint64_t) q.shape[2];
  uint64_t dv = (uint64_t) v.shape[2];

  const float *q_data = (const float *) q.data;
  const float *k_data = (const float *) k.data;
  const float *v_data = (const float *) v.data;
  const float *g_data = (const float *) g.data;
  const float *beta_data = (const float *) beta.data;
  // The IMC Grams, (bh, seq, T): row t of chunk c0 is at (h*seq + c0 + t)*ldg.
  const float *kkm_data = (const float *) kkm_t.data;
  const float *qkm_data = (const float *) qkm_t.data;
  uint64_t ldg = (uint64_t) kkm_t.shape[kkm_t.ndim - 1];
  float *state_data = (float *) state.data;
  float *out_data = (float *) out.data;

  // Scratch carve. Sizes mirror DeltaNetChunkRvv._scratch_floats.
  float *ks0 = (float *) scratch.data; // (T, dv)
  float *qs0 = ks0 + chunk * dv; // (T, dv)
  float *vnew = qs0 + chunk * dv; // (T, dv)
  float *kkm = vnew + chunk * dv; // (T, T) M, the strictly lower solve operand
  float *qkm = kkm + chunk * chunk; // (T, T) P, the readout operand
  float *khat = qkm + chunk * chunk; // (T, dk) exp(c - gc_i) k_i
  float *gc = khat + chunk * dk; // (T,) cumulative log decay
  float *at = gc + chunk; // (T,) A_t
  float *ah = at + chunk; // (T,) A_t / e^c
  float *ea = ah + chunk; // (T,) e^c / A_t

  for (uint64_t h = 0; h < bh; h++) {
    float *s = state_data + h * dk * dv;
    uint64_t t_n = 0;

    for (uint64_t c0 = 0; c0 < seq; c0 += t_n) {
      // The Grams are precomputed on FIXED, chunk-aligned windows, so a chunk must
      // never straddle a window boundary -- cap t_n at the room left in the current
      // window. Without this the decay guard below silently corrupts everything
      // downstream of the first time it fires: it shrinks t_n, c0 stops being
      // aligned, and every later Gram lookup reads the wrong window (and, near the
      // end of a head's rows, reads past them). That cost +16% perplexity and broke
      // determinism, because the out-of-region reads pick up whatever else is in
      // DDR. It is invisible to synthetic tests, whose gates never fire the guard.
      uint64_t in_win = c0 % chunk;
      uint64_t room = chunk - in_win;
      t_n = seq - c0 < room ? seq - c0 : room;
      const float *qc = q_data + (h * seq + c0) * dk;
      const float *kc = k_data + (h * seq + c0) * dk;
      const float *vc = v_data + (h * seq + c0) * dv;
      // beta is read straight from L1 rather than copied into scratch: it is
      // only ever a scalar operand, and the copy cost two of the cumsum's five
      // scalar memory accesses per token.
      const float *be = beta_data + h * seq + c0;
      float *oc = out_data + (h * seq + c0) * dv;
      // Row (h*seq + c0) is already this chunk's first row within its window;
      // only the COLUMN needs the in-window offset, so the whole correction
      // folds into the base pointer and costs nothing per access.
      const float *kkm_in = kkm_data + (h * seq + c0) * ldg + in_win;
      const float *qkm_in = qkm_data + (h * seq + c0) * ldg + in_win;

      // Cumulative log decay within the chunk; it restarts at 0 each chunk
      // because the entry state S0 already carries everything before c0.
      float acc = 0.0f;
      for (uint64_t t = 0; t < t_n; t++) {
        acc += g_data[h * seq + c0 + t];
        gc[t] = acc;
      }

      // Shrink the chunk until every A_t stays normal, which also bounds both
      // halves of the factorisation by e^40. gc is a prefix sum, so a shorter
      // chunk needs no recomputation.
      //
      // This is NOT dead code. Layer 0's gates reach |g| = 5.8 for a single
      // token, and measured on real activations the worst chunk decays by 60
      // nats at T = 16 and 221 at T = 64 -- where it fires for 6 of 32 chunks,
      // and without it khat would need e^110, well past what fp32 can hold. The
      // median chunk decays by 0.03, so the distribution is extremely
      // long-tailed and a spot check on typical data will not show this.
      while (t_n > 1 && gc[t_n - 1] < -80.0f) {
        t_n >>= 1;
      }

      // A_t = exp(gc_t). One exponential is enough: with the pivot at the
      // midpoint, e^c = sqrt(A_T), so both halves of the factorisation are A_t
      // scaled by a single scalar and need no exponential of their own.
      for (uint64_t t = 0; t < t_n;) {
        size_t vl = __riscv_vsetvl_e32m8(t_n - t);
        __riscv_vse32_v_f32m8(
            at + t, ax_vexp_f32m8(__riscv_vle32_v_f32m8(gc + t, vl), vl), vl);
        t += vl;
      }
      float ec = ax_sqrtf(at[t_n - 1]);

      // Both halves of every decay ratio, as two vector reciprocals rather than
      // a scalar divide per row. The divides are the reason: a scalar fdiv is
      // not hidden by the vector queue, and one per row measured dearer than
      // the vector multiply it fed.
      for (uint64_t t = 0; t < t_n;) {
        size_t vl = __riscv_vsetvl_e32m8(t_n - t);
        vfloat32m8_t a = __riscv_vle32_v_f32m8(at + t, vl);
        __riscv_vse32_v_f32m8(ah + t, __riscv_vfmul_vf_f32m8(a, 1.0f / ec, vl), vl);
        __riscv_vse32_v_f32m8(ea + t, __riscv_vfrdiv_vf_f32m8(a, ec, vl), vl);
        t += vl;
      }

      // khat_i = exp(c - gc_i) k_i = (e^c / A_i) k_i, the shared right operand
      // of both Grams and of the state update.
      for (uint64_t i = 0; i < t_n; i++) {
        float eai = ea[i];
        for (uint64_t d = 0; d < dk;) {
          size_t vl = __riscv_vsetvl_e32m8(dk - d);
          __riscv_vse32_v_f32m8(khat + i * dk + d,
              __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(kc + i * dk + d, vl), eai, vl),
              vl);
          d += vl;
        }
      }

      ax_mm_f32(kc, dk, s, dv, ks0, dv, t_n, dk, dv);
      ax_mm_f32(qc, dk, s, dv, qs0, dv, t_n, dk, dv);

      // Turn the RAW Grams into M and P. The Grams arrive already computed on the
      // int8 IMC (llm.imc_mm_int8, batched over every (head, chunk) -- they are the
      // only matmuls here free of the sequential state dependency, so they leave the
      // loop entirely; 84.8M cycles of RVV became 13.5M). They hold k_t.k_i and
      // q_t.k_i with NO decay in them, so both halves of the ratio are applied here:
      // exp(gc_t - c) is the row scalar and exp(c - gc_i) the ea vector.
      //
      // Feeding the array raw L2-normed operands rather than khat is deliberate and
      // free -- measured 1.380e-02 against 1.376e-02 for the factorised form, since
      // khat's range lies along khat^T's columns, exactly what a per-column int8
      // scale absorbs. Raw operands just mean the IMC needs no khat at all.
      //
      // The inner loops below then get M and P with one contiguous scalar load per
      // iteration, which is what makes them fast: computing the coefficient inline
      // from two scalar loads and two multiplies measured 83.5 cycles per iteration
      // against the vector FMA's 40, because the scalar core hides about one access
      // per vector op, not four.
      for (uint64_t t = 0; t < t_n; t++) {
        float aht = ah[t], mrow = ah[t] * be[t];
        for (uint64_t i = 0; i <= t;) {
          size_t vl = __riscv_vsetvl_e32m8(t + 1 - i);
          vfloat32m8_t eav = __riscv_vle32_v_f32m8(ea + i, vl);
          __riscv_vse32_v_f32m8(kkm + t * chunk + i,
              __riscv_vfmul_vf_f32m8(
                  __riscv_vfmul_vv_f32m8(
                      __riscv_vle32_v_f32m8(kkm_in + t * ldg + i, vl), eav, vl),
                  mrow, vl),
              vl);
          __riscv_vse32_v_f32m8(qkm + t * chunk + i,
              __riscv_vfmul_vf_f32m8(
                  __riscv_vfmul_vv_f32m8(
                      __riscv_vle32_v_f32m8(qkm_in + t * ldg + i, vl), eav, vl),
                  aht, vl),
              vl);
          i += vl;
        }
      }

      // (I + M) Vnew = RHS by forward substitution. kkm already IS M, so the
      // inner loop is one scalar load and one vector FMA; only the RHS is
      // folded in here, as RHS[t] = beta_t (v_t - A_t KS0[t]).
      for (uint64_t t = 0; t < t_n; t++) {
        for (uint64_t j = 0; j < dv;) {
          size_t vl = __riscv_vsetvl_e32m8(dv - j);
          vfloat32m8_t rhs
              = __riscv_vfnmsac_vf_f32m8(__riscv_vle32_v_f32m8(vc + t * dv + j, vl),
                  at[t], __riscv_vle32_v_f32m8(ks0 + t * dv + j, vl), vl);
          rhs = __riscv_vfmul_vf_f32m8(rhs, be[t], vl);
          for (uint64_t i = 0; i < t; i++) {
            rhs = __riscv_vfnmsac_vf_f32m8(rhs, kkm[t * chunk + i],
                __riscv_vle32_v_f32m8(vnew + i * dv + j, vl), vl);
          }
          __riscv_vse32_v_f32m8(vnew + t * dv + j, rhs, vl);
          j += vl;
        }
      }

      // O = diag(A) QS0 + P Vnew, with P[t,i] = R[t,i] QK[t,i] for i <= t.
      for (uint64_t t = 0; t < t_n; t++) {
        for (uint64_t j = 0; j < dv;) {
          size_t vl = __riscv_vsetvl_e32m8(dv - j);
          vfloat32m8_t o = __riscv_vfmul_vf_f32m8(
              __riscv_vle32_v_f32m8(qs0 + t * dv + j, vl), at[t], vl);
          for (uint64_t i = 0; i <= t; i++) {
            o = __riscv_vfmacc_vf_f32m8(o, qkm[t * chunk + i],
                __riscv_vle32_v_f32m8(vnew + i * dv + j, vl), vl);
          }
          __riscv_vse32_v_f32m8(oc + t * dv + j, o, vl);
          j += vl;
        }
      }

      // S <- A_T S + e^c (khat^T Vnew), the same Kgamma update with the decay
      // taken out of K's rows and applied once to the whole product. The fp32
      // accumulator is the point of running this on RVV: the decay scales S down
      // every chunk, so in float18 the small increments were being rounded away.
      float a_t = at[t_n - 1];
      for (uint64_t d = 0; d < dk; d++) {
        for (uint64_t j = 0; j < dv;) {
          size_t vl = __riscv_vsetvl_e32m8(dv - j);
          vfloat32m8_t acc_s = __riscv_vfmv_v_f_f32m8(0.0f, vl);
          for (uint64_t i = 0; i < t_n; i++) {
            acc_s = __riscv_vfmacc_vf_f32m8(acc_s, khat[i * dk + d],
                __riscv_vle32_v_f32m8(vnew + i * dv + j, vl), vl);
          }
          acc_s = __riscv_vfmul_vf_f32m8(acc_s, ec, vl);
          __riscv_vse32_v_f32m8(s + d * dv + j,
              __riscv_vfmacc_vf_f32m8(
                  acc_s, a_t, __riscv_vle32_v_f32m8(s + d * dv + j, vl), vl),
              vl);
          j += vl;
        }
      }
    }
  }
  return 0;
}
