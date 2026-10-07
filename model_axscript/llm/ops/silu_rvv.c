//
// Copyright Axelera AI, 2026
// SPDX-License-Identifier: Apache-2.0
//
// Elementwise SiLU, fp32 on the AI core's RVV unit.
//
//     x  (rows, d) f32
//     y  (rows, d) f32  -- x * sigmoid(x)
//
// Replaces the float18 chained-LUT SiLU on the DeltaNet conv output. That LUT had to
// span roughly [-25, 40] with a handful of curvature-placed knots, so its absolute
// error (~0.016) is a large RELATIVE error wherever the activation is small -- which
// is most of a SiLU output, since the function is near zero for all negative inputs.

static int
rvv_silu(cfunc_ctx_t *ctx)
{
  cfunc_dltensor_t x, y;
  ctx->api->get_dltensor_arg(ctx, 0, &x);
  ctx->api->get_dltensor_arg(ctx, 1, &y);

  uint64_t n = 1;
  for (int32_t a = 0; a < x.ndim; a++)
    n *= (uint64_t) x.shape[a];

  const float *x_data = (const float *) x.data;
  float *y_data = (float *) y.data;

  for (uint64_t i = 0; i < n;) {
    size_t vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(x_data + i, vl);
    __riscv_vse32_v_f32m8(y_data + i,
        __riscv_vfmul_vv_f32m8(vx, ax_vsigmoid_f32m8(vx, vl), vl), vl);
    i += vl;
  }
  return 0;
}
