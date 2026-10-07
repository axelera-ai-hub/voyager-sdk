//
// Copyright Axelera AI, 2026
// SPDX-License-Identifier: Apache-2.0
//
// Fused L2-norm(q) + L2-norm(k) + gated-delta step + gated RMSNorm, fp32 on RVV.
// The four ops chain 1:1 in the mixer (q/k are L2-normed, fed into the recurrence,
// whose readout is gated-RMSNormed) and are all per-head-independent over the same
// head axis, so this is the same algebra as deltanet_l2norm_rvv.c (x2) +
// deltanet_step_rvv.c + deltanet_gated_rmsnorm_rvv.c, just run back to back per head
// inside one kernel instead of four, with ql/kl/step-readout kept in L1 scratch
// (qn/kn/ot below) instead of round-tripping DDR between separate ops.
//
// The L2-norm and gated-RMSNorm reductions are done per head here (one reduction
// each, same style as the q.k dot product below), NOT batched across heads the way
// deltanet_l2norm_rvv.c/deltanet_gated_rmsnorm_rvv.c batch across rows. A batched
// version was tried and measured SLOWER at this model's dims (16 heads, dk=dv=128):
// the batched reduction's column-strided load (stride d*4=512B, one cache line per
// row per column) costs more here than the per-head drains it saves, since those
// source kernels' batching pays off at a different (row-count, d) scale than this
// one. Keep this simple per-head form unless re-measured otherwise.
//
//     q, k         (bh, dk)     f32  -- raw (NOT pre-normalized) query/key
//     v            (bh, dv)     f32
//     a, b         (bh, PWORD)  f32  -- in_proj_a / in_proj_b, one row per head
//     pos_a        (bh, PWORD)  f32
//     dt_bias      (bh, PWORD)  f32
//     z            (bh, dv)     f32  -- gate for the gated RMSNorm tail
//     gnorm_weight (1, dv)      f32  -- shared across heads
//     state        (bh, dk, dv) f32  -- updated in place
//     qn, kn       (bh, dk)     f32  -- L1 scratch: normalized q/k
//     ot           (bh, dv)     f32  -- L1 scratch: step readout, pre-gating
//     out          (bh, dv)     f32
//     q_scale, k_scale, l2norm_eps, gnorm_eps -- fp32 bit patterns
//
// See deltanet_step_rvv.c for the recurrence algebra and deltanet_l2norm_rvv.c /
// deltanet_gated_rmsnorm_rvv.c for the norms; unchanged here beyond sourcing q/k
// from local scratch and reading z/gnorm_weight/writing the final output directly
// instead of a chain of DDR buffers.
extern void dcache_invalidate_buffer(const void *addr, size_t size);

static inline float
ax_dot_m1(const float *x, const float *y, uint64_t n)
{
  size_t vl = __riscv_vsetvl_e32m1(n);
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  for (uint64_t j = 0; j < n;) {
    vl = __riscv_vsetvl_e32m1(n - j);
    acc = __riscv_vfmacc_vv_f32m1(acc, __riscv_vle32_v_f32m1(x + j, vl),
        __riscv_vle32_v_f32m1(y + j, vl), vl);
    j += vl;
  }
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(
      acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), __riscv_vsetvl_e32m1(n)));
}

static int
rvv_deltanet_mixer_core(cfunc_ctx_t *ctx)
{
  cfunc_dltensor_t q, k, v, a, b, pos_a, dt_bias, z, gnorm_weight, state, qn, kn, ot, out;
  ctx->api->get_dltensor_arg(ctx, 0, &q);
  ctx->api->get_dltensor_arg(ctx, 1, &k);
  ctx->api->get_dltensor_arg(ctx, 2, &v);
  ctx->api->get_dltensor_arg(ctx, 3, &a);
  ctx->api->get_dltensor_arg(ctx, 4, &b);
  ctx->api->get_dltensor_arg(ctx, 5, &pos_a);
  ctx->api->get_dltensor_arg(ctx, 6, &dt_bias);
  ctx->api->get_dltensor_arg(ctx, 7, &z);
  ctx->api->get_dltensor_arg(ctx, 8, &gnorm_weight);
  ctx->api->get_dltensor_arg(ctx, 9, &state);
  ctx->api->get_dltensor_arg(ctx, 10, &qn);
  ctx->api->get_dltensor_arg(ctx, 11, &kn);
  ctx->api->get_dltensor_arg(ctx, 12, &ot);
  ctx->api->get_dltensor_arg(ctx, 13, &out);
  float q_scale = ax_bits_to_f32((uint32_t) ctx->api->get_uint64_arg(ctx, 14));
  float k_scale = ax_bits_to_f32((uint32_t) ctx->api->get_uint64_arg(ctx, 15));
  float l2norm_eps = ax_bits_to_f32((uint32_t) ctx->api->get_uint64_arg(ctx, 16));
  float gnorm_eps = ax_bits_to_f32((uint32_t) ctx->api->get_uint64_arg(ctx, 17));

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
  dcache_invalidate_buffer(z.data, (size_t) (heads * dv * sizeof(float)));
  dcache_invalidate_buffer(gnorm_weight.data, (size_t) (dv * sizeof(float)));
  dcache_invalidate_buffer(state.data, (size_t) (heads * dk * dv * sizeof(float)));

  const float *qd = (const float *) q.data;
  const float *kd = (const float *) k.data;
  const float *vd = (const float *) v.data;
  const float *zd = (const float *) z.data;
  const float *gw = (const float *) gnorm_weight.data;
  float *sd = (float *) state.data;
  float *qnd = (float *) qn.data;
  float *knd = (float *) kn.data;
  float *otd = (float *) ot.data;
  float *od = (float *) out.data;
  const float *ag = (const float *) a.data;
  const float *bg = (const float *) b.data;
  const float *pg = (const float *) pos_a.data;
  const float *dg = (const float *) dt_bias.data;

  for (uint64_t h = 0; h < heads; h++) {
    const float *qh = qd + h * dk;
    const float *kh = kd + h * dk;
    const float *vh = vd + h * dv;
    const float *zh = zd + h * dv;
    float *sh = sd + h * dk * dv;
    float *qnh = qnd + h * dk;
    float *knh = knd + h * dk;
    float *oth = otd + h * dv;
    float *oh = od + h * dv;

    // -- L2-norm q/k (deltanet_l2norm_rvv.c's algebra, one reduction per head) --
    float inv_q = ax_rsqrtf(ax_dot_m1(qh, qh, dk) + l2norm_eps) * q_scale;
    float inv_k = ax_rsqrtf(ax_dot_m1(kh, kh, dk) + l2norm_eps) * k_scale;
    for (uint64_t j = 0; j < dk;) {
      size_t vl = __riscv_vsetvl_e32m1(dk - j);
      __riscv_vse32_v_f32m1(qnh + j,
          __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(qh + j, vl), inv_q, vl), vl);
      __riscv_vse32_v_f32m1(knh + j,
          __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(kh + j, vl), inv_k, vl), vl);
      j += vl;
    }

    // -- gated delta step (deltanet_step_rvv.c's algebra, using qnh/knh) --
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

    float qk = ax_dot_m1(qnh, knh, dk);

    for (uint64_t c = 0; c < dv;) {
      size_t vl = __riscv_vsetvl_e32m1(dv - c);
      vfloat32m1_t ks = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      vfloat32m1_t qs = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      for (uint64_t i = 0; i < dk; i++) {
        vfloat32m1_t sr = __riscv_vle32_v_f32m1(sh + i * dv + c, vl);
        ks = __riscv_vfmacc_vf_f32m1(ks, knh[i], sr, vl);
        qs = __riscv_vfmacc_vf_f32m1(qs, qnh[i], sr, vl);
      }
      vfloat32m1_t vn = __riscv_vfmul_vf_f32m1(
          __riscv_vfnmsac_vf_f32m1(__riscv_vle32_v_f32m1(vh + c, vl), al, ks, vl), be, vl);
      __riscv_vse32_v_f32m1(oth + c,
          __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(qs, al, vl), qk, vn, vl), vl);
      for (uint64_t i = 0; i < dk; i++) {
        vfloat32m1_t sr = __riscv_vle32_v_f32m1(sh + i * dv + c, vl);
        __riscv_vse32_v_f32m1(sh + i * dv + c,
            __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(sr, al, vl), knh[i], vn, vl), vl);
      }
      c += vl;
    }

    // -- gated RMSNorm tail (deltanet_gated_rmsnorm_rvv.c's algebra) --
    float inv_g = ax_rsqrtf(ax_dot_m1(oth, oth, dv) / (float) dv + gnorm_eps);
    for (uint64_t c2 = 0; c2 < dv;) {
      size_t vl2 = __riscv_vsetvl_e32m8(dv - c2);
      vfloat32m8_t nv
          = __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(oth + c2, vl2), inv_g, vl2);
      nv = __riscv_vfmul_vv_f32m8(nv, __riscv_vle32_v_f32m8(gw + c2, vl2), vl2);
      vfloat32m8_t g = __riscv_vle32_v_f32m8(zh + c2, vl2);
      vfloat32m8_t silu = __riscv_vfmul_vv_f32m8(g, ax_vsigmoid_f32m8(g, vl2), vl2);
      __riscv_vse32_v_f32m8(oh + c2, __riscv_vfmul_vv_f32m8(nv, silu, vl2), vl2);
      c2 += vl2;
    }
  }
  return 0;
}
