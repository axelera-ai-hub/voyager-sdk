//
// Copyright Axelera AI, 2026
// SPDX-License-Identifier: Apache-2.0
//
// Single-token gated delta rule, fp32 on the AI core's RVV unit.
//
//     q, k      (bh, dk)     f32  -- L2-normed, head-major, one token
//     v         (bh, dv)     f32
//     a, b      (bh, PWORD)  f32  -- in_proj_a / in_proj_b outputs, one row per head
//     pos_a     (bh, PWORD)  f32  -- exp(A_log) per head, one row per head
//     dt_bias   (bh, PWORD)  f32  -- softplus bias per head, one row per head
//     state     (bh, dk, dv) f32  -- entry state, updated in place
//     out       (bh, dv)     f32
//
// The chunked recurrence at T=1, with every term that vanishes dropped. For entry
// state S and alpha = exp(g):
//
//     vnew = beta (v - alpha (k^T S))
//     out  = alpha (q^T S) + (q . k) vnew
//     S    = alpha S + k (x) vnew
//
// The chunk form's decay pivot, triangular solve and Gram matrices are all identity
// or empty here: e^c = sqrt(alpha) cancels between khat and the state carry, and the
// strictly-lower solve matrix has no entries. So this shares no code with
// deltanet_chunk_rvv.c beyond the algebra it specialises.
//
// Two passes over the state, both vectorised along dv so the reduction over dk is a
// vertical vfmacc chain against a scalar coefficient -- no horizontal reduce, and
// nothing leaves the vector register file (see deltanet_l2norm_rvv.c). vl is dv, which
// is VLMAX at m1 for the 128 head dim this model uses; a longer dv blocks along dv and
// re-reads the state per block.
//
// The two passes cannot fuse: vnew needs the completed k^T S.
//
// The gates are computed here rather than by deltanet_gates_rvv, which packs the
// sequence into lanes and so wants seq a multiple of PWORD. One token is one lane word,
// so each head's gates are a vl=1 transcendental read straight out of the vector
// register file. Computing all the heads at once into an L1 scratch buffer and reading
// it back with scalar loads is what the obvious version does, and it is wrong: the
// scalar side goes through the data cache, which by then holds a stale line for that
// address left by an earlier RVV kernel in the same core program, so alpha comes back
// as whatever that kernel wrote and the recurrence diverges.

// The generated kernel runs with the AI core's data cache enabled and every
// operand arrives by DMA, which writes L1 without invalidating it; each layer's
// call stages at the same L1 offsets, so lines left resident from the previous
// layer shadow the new data (see _imc_mm_quant.c).
//
// Per buffer, and it has to be. dcache_invalidate() looks like a cheaper
// whole-cache equivalent but in libkernsup_aicore it disassembles to a bare
// `fence`, which orders memory and invalidates nothing; only
// dcache_invalidate_buffer does real work, via cbo.inval per line. Substituting
// it silently removes the invalidation: measured 8 of 16 full-model forwards
// corrupted, against 0 of 16 with this. Walking the lines costs ~36% of this
// kernel and that is the real price of correctness here, not an overhead to
// shave. (dcache_flush_buffer is not a way out either -- it degrades to the
// same bare fence for anything over 16 KB.)
extern void dcache_invalidate_buffer(const void *addr, size_t size);

static int
rvv_deltanet_step(cfunc_ctx_t *ctx)
{
  cfunc_dltensor_t q, k, v, a, b, pos_a, dt_bias, state, out;
  ctx->api->get_dltensor_arg(ctx, 0, &q);
  ctx->api->get_dltensor_arg(ctx, 1, &k);
  ctx->api->get_dltensor_arg(ctx, 2, &v);
  ctx->api->get_dltensor_arg(ctx, 3, &a);
  ctx->api->get_dltensor_arg(ctx, 4, &b);
  ctx->api->get_dltensor_arg(ctx, 5, &pos_a);
  ctx->api->get_dltensor_arg(ctx, 6, &dt_bias);
  ctx->api->get_dltensor_arg(ctx, 7, &state);
  ctx->api->get_dltensor_arg(ctx, 8, &out);

  // The head extent comes from the tile, not an argument: this runs once per head tile.
  uint64_t heads = (uint64_t) q.shape[q.ndim - 2];
  uint64_t dk = (uint64_t) q.shape[q.ndim - 1];
  uint64_t dv = (uint64_t) v.shape[v.ndim - 1];
  uint64_t gstride = (uint64_t) a.shape[a.ndim - 1];

  dcache_invalidate_buffer(q.data, (size_t) (heads * dk * sizeof(float)));
  dcache_invalidate_buffer(k.data, (size_t) (heads * dk * sizeof(float)));
  dcache_invalidate_buffer(v.data, (size_t) (heads * dv * sizeof(float)));
  dcache_invalidate_buffer(a.data, (size_t) (heads * gstride * sizeof(float)));
  dcache_invalidate_buffer(b.data, (size_t) (heads * gstride * sizeof(float)));
  dcache_invalidate_buffer(pos_a.data, (size_t) (heads * gstride * sizeof(float)));
  dcache_invalidate_buffer(dt_bias.data, (size_t) (heads * gstride * sizeof(float)));
  dcache_invalidate_buffer(state.data, (size_t) (heads * dk * dv * sizeof(float)));

  const float *qd = (const float *) q.data;
  const float *kd = (const float *) k.data;
  const float *vd = (const float *) v.data;
  float *sd = (float *) state.data;
  float *od = (float *) out.data;
  const float *ag = (const float *) a.data;
  const float *bg = (const float *) b.data;
  const float *pg = (const float *) pos_a.data;
  const float *dg = (const float *) dt_bias.data;

  for (uint64_t h = 0; h < heads; h++) {
    const float *qh = qd + h * dk;
    const float *kh = kd + h * dk;
    const float *vh = vd + h * dv;
    float *sh = sd + h * dk * dv;
    float *oh = od + h * dv;

    // g = -posA*softplus(a + dt_bias) <= 0, so alpha = exp(g) is in (0, 1] and
    // cannot overflow.
    size_t vl1 = __riscv_vsetvl_e32m8(1);
    float al = __riscv_vfmv_f_s_f32m8_f32(ax_vexp_f32m8(
        __riscv_vfneg_v_f32m8(
            __riscv_vfmul_vv_f32m8(__riscv_vle32_v_f32m8(pg + h * gstride, vl1),
                ax_vsoftplus_f32m8(__riscv_vfadd_vv_f32m8(__riscv_vle32_v_f32m8(ag + h * gstride, vl1),
                                       __riscv_vle32_v_f32m8(dg + h * gstride, vl1), vl1),
                    vl1),
                vl1),
            vl1),
        vl1));
    float be = __riscv_vfmv_f_s_f32m8_f32(
        ax_vsigmoid_f32m8(__riscv_vle32_v_f32m8(bg + h * gstride, vl1), vl1));

    // q . k, the only horizontal reduce here: one queue drain per head.
    float qk;
    {
      size_t vl = __riscv_vsetvl_e32m1(dk);
      vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      for (uint64_t j = 0; j < dk;) {
        vl = __riscv_vsetvl_e32m1(dk - j);
        acc = __riscv_vfmacc_vv_f32m1(acc, __riscv_vle32_v_f32m1(qh + j, vl),
            __riscv_vle32_v_f32m1(kh + j, vl), vl);
        j += vl;
      }
      qk = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(
          acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), __riscv_vsetvl_e32m1(dk)));
    }

    for (uint64_t c = 0; c < dv;) {
      size_t vl = __riscv_vsetvl_e32m1(dv - c);
      vfloat32m1_t ks = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      vfloat32m1_t qs = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      for (uint64_t i = 0; i < dk; i++) {
        vfloat32m1_t sr = __riscv_vle32_v_f32m1(sh + i * dv + c, vl);
        ks = __riscv_vfmacc_vf_f32m1(ks, kh[i], sr, vl);
        qs = __riscv_vfmacc_vf_f32m1(qs, qh[i], sr, vl);
      }
      vfloat32m1_t vn = __riscv_vfmul_vf_f32m1(
          __riscv_vfnmsac_vf_f32m1(__riscv_vle32_v_f32m1(vh + c, vl), al, ks, vl), be, vl);
      __riscv_vse32_v_f32m1(oh + c,
          __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(qs, al, vl), qk, vn, vl), vl);
      for (uint64_t i = 0; i < dk; i++) {
        vfloat32m1_t sr = __riscv_vle32_v_f32m1(sh + i * dv + c, vl);
        __riscv_vse32_v_f32m1(sh + i * dv + c,
            __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(sr, al, vl), kh[i], vn, vl), vl);
      }
      c += vl;
    }
  }
  return 0;
}
