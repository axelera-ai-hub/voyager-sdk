//
// Copyright Axelera AI, 2026
// SPDX-License-Identifier: Apache-2.0
//
// Per-(head, token) gated-DeltaNet gates, fp32 on the AI core's RVV unit.
//
//     a, b     (seq, lanes) f32  -- in_proj_a / in_proj_b output, head in lane;
//                                   only lanes [0, heads) carry a head
//     pos_a    (1, lanes)   f32  -- exp(A_log) per head, folded on the host
//     dt_bias  (1, lanes)   f32  -- per-head softplus bias
//     mask     (1, seq)     f32  -- 1.0 for a real token, 0.0 for chunk padding
//     g        (heads, seq) f32  -- log decay = -pos_a * softplus(a + dt_bias)
//     beta     (heads, seq) f32  -- sigmoid(b)
//     heads    UINT64
//
// g is the LOG decay, not alpha = exp(g): the chunked delta rule needs cumsum(g)
// and the ratio matrix exp(gc[t] - gc[i]), so keeping the gate in log space means
// those never form a cumulative product that can underflow.
//
// The op also transposes head-in-lane to head-major, which is free here (a strided
// load feeds a contiguous store) and earns its keep downstream: the recurrence is
// per head and tiles over heads, so it needs each head's gates contiguous. It also
// removes the 32 lane_select ops the DPU path needed to reach the same layout.
//
// mask zeroes both gates at a padded position, which makes that position an exact
// no-op on the recurrent state: beta = 0 gives vnew = 0, and g = 0 gives alpha = 1, so
// S_new = alpha*S + k (x) vnew = S. That is what lets a short final prefill chunk be
// padded up to the compiled length instead of replayed token-by-token through decode.
// It is applied HERE, in fp32, rather than as a DPU multiply on the op's output: the
// datapath rounds to float18, which would cost 10 mantissa bits on every real token's
// gates to mask the padding.

static int
rvv_deltanet_gates(cfunc_ctx_t *ctx)
{
  cfunc_dltensor_t a, b, pos_a, dt_bias, mask, g, beta;
  ctx->api->get_dltensor_arg(ctx, 0, &a);
  ctx->api->get_dltensor_arg(ctx, 1, &b);
  ctx->api->get_dltensor_arg(ctx, 2, &pos_a);
  ctx->api->get_dltensor_arg(ctx, 3, &dt_bias);
  ctx->api->get_dltensor_arg(ctx, 4, &mask);
  ctx->api->get_dltensor_arg(ctx, 5, &g);
  ctx->api->get_dltensor_arg(ctx, 6, &beta);
  uint64_t heads = ctx->api->get_uint64_arg(ctx, 7);

  uint64_t seq = (uint64_t) a.shape[0];
  uint64_t lanes = (uint64_t) a.shape[1];

  const float *a_data = (const float *) a.data;
  const float *b_data = (const float *) b.data;
  const float *pos_a_data = (const float *) pos_a.data;
  const float *dt_bias_data = (const float *) dt_bias.data;
  const float *mask_data = (const float *) mask.data;
  float *g_data = (float *) g.data;
  float *beta_data = (float *) beta.data;

  ptrdiff_t bstride = (ptrdiff_t) (lanes * sizeof(float));

  for (uint64_t h = 0; h < heads; h++) {
    float neg_pos_a = -pos_a_data[h];
    float bias = dt_bias_data[h];

    for (uint64_t t = 0; t < seq;) {
      size_t vl = __riscv_vsetvl_e32m8(seq - t);

      vfloat32m8_t vm = __riscv_vle32_v_f32m8(mask_data + t, vl);

      vfloat32m8_t va = __riscv_vlse32_v_f32m8(a_data + t * lanes + h, bstride, vl);
      vfloat32m8_t sp = ax_vsoftplus_f32m8(__riscv_vfadd_vf_f32m8(va, bias, vl), vl);
      __riscv_vse32_v_f32m8(g_data + h * seq + t,
          __riscv_vfmul_vv_f32m8(__riscv_vfmul_vf_f32m8(sp, neg_pos_a, vl), vm, vl), vl);

      vfloat32m8_t vb = __riscv_vlse32_v_f32m8(b_data + t * lanes + h, bstride, vl);
      __riscv_vse32_v_f32m8(beta_data + h * seq + t,
          __riscv_vfmul_vv_f32m8(ax_vsigmoid_f32m8(vb, vl), vm, vl), vl);

      t += vl;
    }
  }
  return 0;
}
