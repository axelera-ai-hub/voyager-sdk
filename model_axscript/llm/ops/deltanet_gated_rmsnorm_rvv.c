//
// Copyright Axelera AI, 2026
// SPDX-License-Identifier: Apache-2.0
//
// Gated RMSNorm tail of the DeltaNet block, fp32 on the AI core's RVV unit.
//
//     x       (rows, d) f32  -- recurrence readout, one head's value dim per row
//     gate    (rows, d) f32  -- the z projection
//     weight  (1, d)    f32
//     y       (rows, d) f32  -- weight * x * rsqrt(mean(x^2) + eps) * silu(gate)
//     eps     UINT64         -- fp32 bit pattern
//
// Two float18 LUTs collapse into exact arithmetic here: the inverse square root
// and the SiLU. It also removes the eps bookkeeping the DPU path needed -- that
// kernel normalised with eps' = d*eps so its rsqrt saw variance + d*eps, and the
// LUT then had to be fitted over exactly that rescaled domain. Here eps is just
// eps, matching the torch GatedRMSNorm directly.

static int
rvv_deltanet_gated_rmsnorm(cfunc_ctx_t *ctx)
{
  cfunc_dltensor_t x, gate, weight, y;
  ctx->api->get_dltensor_arg(ctx, 0, &x);
  ctx->api->get_dltensor_arg(ctx, 1, &gate);
  ctx->api->get_dltensor_arg(ctx, 2, &weight);
  ctx->api->get_dltensor_arg(ctx, 3, &y);
  float eps = ax_bits_to_f32((uint32_t) ctx->api->get_uint64_arg(ctx, 4));

  uint64_t rows = (uint64_t) x.shape[0];
  uint64_t d = (uint64_t) x.shape[1];

  const float *x_data = (const float *) x.data;
  const float *gate_data = (const float *) gate.data;
  const float *weight_data = (const float *) weight.data;
  float *y_data = (float *) y.data;
  ptrdiff_t row_bstride = (ptrdiff_t) (d * sizeof(float));
  float inv_d = 1.0f / (float) d;

  // Transposed like deltanet_l2norm_rvv, and for the same two reasons: the
  // reduction becomes a vertical vfmacc chain instead of a horizontal reduce,
  // and nothing leaves the vector register file. The vector unit is a decoupled
  // coprocessor, so the per-row scalar round trip this replaced (vfredosum ->
  // vfmv.f.s -> scalar fsqrt -> broadcast) drained the queue once PER ROW.
  for (uint64_t r = 0; r < rows;) {
    size_t vl = __riscv_vsetvl_e32m8(rows - r);
    const float *xr = x_data + r * d;

    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    for (uint64_t j = 0; j < d; j++) {
      vfloat32m8_t v = __riscv_vlse32_v_f32m8(xr + j, row_bstride, vl);
      acc = __riscv_vfmacc_vv_f32m8(acc, v, v, vl);
    }
    acc = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(acc, inv_d, vl), eps, vl);
    vfloat32m8_t inv = __riscv_vfrdiv_vf_f32m8(__riscv_vfsqrt_v_f32m8(acc, vl), 1.0f, vl);

    // The reduction above needs this row-block transpose (see
    // deltanet_l2norm_rvv.c); the SiLU-gated write-back below does not -- each
    // row's `inv` is already a scalar, and vectorising it as d row-block-wide
    // (width vl) strided ops instead of vl row-wide (width d) contiguous ones
    // is why this kernel cost 8x deltanet_l2norm_rvv.c despite doing a similar
    // amount of work: every one of the d columns paid a full ax_vsigmoid_f32m8
    // at width vl instead of every row paying it once at width d. One vse32
    // moves `inv` out of the vector unit for the whole block (not a per-row
    // drain), then each row runs its own SiLU at full width.
    float inv_buf[vl];
    __riscv_vse32_v_f32m8(inv_buf, inv, vl);

    for (uint64_t rr = 0; rr < vl; rr++) {
      const float *xr2 = x_data + (r + rr) * d;
      const float *gr2 = gate_data + (r + rr) * d;
      float *yr2 = y_data + (r + rr) * d;
      for (uint64_t j = 0; j < d;) {
        size_t vl2 = __riscv_vsetvl_e32m8(d - j);
        vfloat32m8_t v = __riscv_vfmul_vf_f32m8(
            __riscv_vle32_v_f32m8(xr2 + j, vl2), inv_buf[rr], vl2);
        v = __riscv_vfmul_vv_f32m8(v, __riscv_vle32_v_f32m8(weight_data + j, vl2), vl2);
        vfloat32m8_t g = __riscv_vle32_v_f32m8(gr2 + j, vl2);
        vfloat32m8_t silu = __riscv_vfmul_vv_f32m8(g, ax_vsigmoid_f32m8(g, vl2), vl2);
        __riscv_vse32_v_f32m8(yr2 + j, __riscv_vfmul_vv_f32m8(v, silu, vl2), vl2);
        j += vl2;
      }
    }
    r += vl;
  }
  return 0;
}
