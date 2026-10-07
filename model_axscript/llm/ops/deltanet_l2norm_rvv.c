//
// Copyright Axelera AI, 2026
// SPDX-License-Identifier: Apache-2.0
//
// Row-wise L2 normalisation, fp32 on the AI core's RVV unit.
//
//     x      (rows, d) f32  -- one head's q or k per row, head dim contiguous
//     y      (rows, d) f32  -- x * rsqrt(sum(x^2) + eps) * scale
//     eps    UINT64         -- fp32 bit pattern
//     scale  UINT64         -- fp32 bit pattern, applied after normalising
//
// scale carries the delta rule's d^-0.5 query factor. It matters even though the
// readout is linear in q and a gated RMSNorm follows: that norm is only scale-free
// up to its own eps, and rescaling q by s shifts the norm's effective eps by 1/s^2
// (128x here), which is significant for the many rows whose readout is small.
//
// The loop is transposed: it vectorises across ROWS, not along d. Two reasons, both
// specific to this core:
//
//   * the reduction becomes a VERTICAL vfmacc chain at full width instead of a
//     horizontal reduce. The architectural vfred* are slow enough that the in-tree
//     RVV library shadows vfredusum with a vslidedown log-tree, and vfredosum is
//     sequential by definition;
//   * nothing leaves the vector register file. The vector unit is a decoupled
//     coprocessor, so a vector->scalar move (vfmv.f.s, or a scalar sqrt on a reduced
//     value) drains the whole queue -- reading a vector CSR is how the sync macro is
//     implemented. Per-row that would be one full pipeline drain per row.
//
// rsqrt is vfrsqrt7 plus two Newton steps, which lands inside fp32 rounding: the
// estimate is 7 bits, each step doubles it (7 -> 14 -> 28). The steps are per
// row-block, not per element, so they are free.

static int
rvv_deltanet_l2norm(cfunc_ctx_t *ctx)
{
  cfunc_dltensor_t x, y;
  ctx->api->get_dltensor_arg(ctx, 0, &x);
  ctx->api->get_dltensor_arg(ctx, 1, &y);
  float eps = ax_bits_to_f32((uint32_t) ctx->api->get_uint64_arg(ctx, 2));
  float post_scale = ax_bits_to_f32((uint32_t) ctx->api->get_uint64_arg(ctx, 3));

  uint64_t rows = (uint64_t) x.shape[0];
  uint64_t d = (uint64_t) x.shape[1];

  const float *x_data = (const float *) x.data;
  float *y_data = (float *) y.data;
  ptrdiff_t row_bstride = (ptrdiff_t) (d * sizeof(float));

  for (uint64_t r = 0; r < rows;) {
    size_t vl = __riscv_vsetvl_e32m8(rows - r);
    const float *xr = x_data + r * d;

    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(eps, vl);
    for (uint64_t j = 0; j < d; j++) {
      vfloat32m8_t v = __riscv_vlse32_v_f32m8(xr + j, row_bstride, vl);
      acc = __riscv_vfmacc_vv_f32m8(acc, v, v, vl);
    }

    // Exact, and still entirely in vector registers: vfsqrt + a reciprocal
    // divide, two instructions per row-block. vfrsqrt7 plus Newton steps was
    // measurably WORSE than its own 7-bit estimate here, so the estimate's
    // semantics are not what the obvious iteration assumes; at one instruction
    // per row-block the exact form costs nothing worth chasing that down for.
    vfloat32m8_t inv
        = __riscv_vfrdiv_vf_f32m8(__riscv_vfsqrt_v_f32m8(acc, vl), post_scale, vl);

    // The scale is per-row and already reduced to `inv`; the write-back below has
    // no cross-row dependency at all, so unlike the reduction above it does not
    // need to vectorise across this row-block. One vse32 gets every row's scalar
    // out of the vector unit at once (not a per-row drain -- see
    // deltanet_gated_rmsnorm_rvv.c), then each row's d elements go out as
    // contiguous loads/stores instead of d row-block-wide strided ones.
    float inv_buf[vl];
    __riscv_vse32_v_f32m8(inv_buf, inv, vl);

    for (uint64_t rr = 0; rr < vl; rr++) {
      const float *xr2 = x_data + (r + rr) * d;
      float *yr2 = y_data + (r + rr) * d;
      for (uint64_t j = 0; j < d;) {
        size_t vl2 = __riscv_vsetvl_e32m8(d - j);
        __riscv_vse32_v_f32m8(yr2 + j,
            __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(xr2 + j, vl2), inv_buf[rr], vl2),
            vl2);
        j += vl2;
      }
    }
    r += vl;
  }
  return 0;
}
