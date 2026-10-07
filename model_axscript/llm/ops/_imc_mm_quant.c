//
// Copyright Axelera AI, 2026
// SPDX-License-Identifier: Apache-2.0
//
// Quantize an fp32 matmul pair for the int8 IMC, on the AI core's RVV unit.
//
//     a   (M, K) f32   activation; quantized per ROW
//     b   (K, N) f32   IMC operand in (K, N) layout; quantized per COLUMN
//     qa  (M, K) i8
//     qb  (K, N) i8
//     sc  (M, (1 + N/PW) * PW) f32   per row: the row scale broadcast over PW lanes,
//                                    then the N column scales
//
// The two scale sets share ONE buffer because the MVM datapath consumes them as a single
// contiguous port-1 run per row; pushing them from two separate buffers into the same
// stream misaligns it, and since both operands here are L2-normed the row and column
// scales are of similar size, so the corruption is unbiased and looks like ~17% noise
// rather than anything obviously broken.
//
// The scale axes are forced by the hardware: the MVM accumulates over K in int32 and the
// only dequant available afterwards is the outer product sa[m]*sb[n], so the reduction
// axis gets no scale of its own. Measured against the fp64 recurrence, that costs 1.4e-2
// over all six of the chunk op's matmuls -- and the decay factorisation is free of charge,
// because khat's dynamic range lies along khat^T's COLUMNS, which is exactly what the
// per-column scale absorbs.
//
// sa is stored lane-broadcast because the DPU consumes it as a whole lane word (there is
// no scalar broadcast on that path); sb is already one value per lane and needs no
// expansion.

// The generated kernel enables the AI core's data cache (ecall_dcache_enable)
// and the fp32 operands arrive by DMA, which writes L1 without invalidating
// that cache. Every batch element stages at the SAME L1 offsets, so lines left
// resident from the previous element shadow the DMA's new data. Measured: one
// matmul of a 128-batch wrong about once per 10,000, always via the per-COLUMN
// scales -- the column absmax is the first pass over b, reads a few stale rows,
// and then evicts them itself, so the quantize pass that follows reads b
// correctly and only inherits the bad scale. Hence scales off by 0.75x to 1.52x
// (a max reduction cannot come out LOW over correct data) and qb wrong in
// exactly the columns whose scale is wrong.
//
// Invalidating is what is needed, not ordering: ecall_fence() before the first
// read and after the last store both measured no change in the rate.
extern void dcache_invalidate_buffer(const void *addr, size_t size);

// f32 -> i8 with round-to-nearest and saturation, narrowing m8 -> m4 -> m2.
static inline vint8m2_t
ax_quant_i8_f32m8(vfloat32m8_t x, size_t vl)
{
  x = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(x, -127.0f, vl), 127.0f, vl);
  return __riscv_vncvt_x_x_w_i8m2(
      __riscv_vncvt_x_x_w_i16m4(__riscv_vfcvt_x_f_v_i32m8(x, vl), vl), vl);
}

static int
rvv_imc_mm_quant(cfunc_ctx_t *ctx)
{
  cfunc_dltensor_t a, b, qa, qb, sc;
  ctx->api->get_dltensor_arg(ctx, 0, &a);
  ctx->api->get_dltensor_arg(ctx, 1, &b);
  ctx->api->get_dltensor_arg(ctx, 2, &qa);
  ctx->api->get_dltensor_arg(ctx, 3, &qb);
  ctx->api->get_dltensor_arg(ctx, 4, &sc);

  // Trailing two dims, so a unit batch axis on the staged slices is harmless.
  uint64_t m = (uint64_t) a.shape[a.ndim - 2];
  uint64_t k = (uint64_t) a.shape[a.ndim - 1];
  uint64_t n = (uint64_t) b.shape[b.ndim - 1];
  uint64_t scw = (uint64_t) sc.shape[sc.ndim - 1]; // (1 + N/PW) * PW
  uint64_t pw = scw - n; // one lane word for the row scale

  dcache_invalidate_buffer(a.data, (size_t) (m * k * sizeof(float)));
  dcache_invalidate_buffer(b.data, (size_t) (k * n * sizeof(float)));

  const float *ad = (const float *) a.data;
  const float *bd = (const float *) b.data;
  int8_t *qad = (int8_t *) qa.data;
  int8_t *qbd = (int8_t *) qb.data;
  float *scd = (float *) sc.data;
  float *sbd = scd + pw; // row 0's column-scale slots, computed first and then copied

  // Column absmax of B: the columns are the lane axis, so this one is a plain
  // vertical chain over unit-stride loads -- no transposition needed.
  for (uint64_t j = 0; j < n;) {
    size_t vl = __riscv_vsetvl_e32m8(n - j);
    vfloat32m8_t mx = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    for (uint64_t i = 0; i < k; i++) {
      mx = __riscv_vfmax_vv_f32m8(mx,
          __riscv_vfabs_v_f32m8(__riscv_vle32_v_f32m8(bd + i * n + j, vl), vl), vl);
    }
    mx = __riscv_vfmax_vf_f32m8(mx, 1e-30f, vl);
    __riscv_vse32_v_f32m8(sbd + j, __riscv_vfmul_vf_f32m8(mx, 1.0f / 127.0f, vl), vl);
    j += vl;
  }
  for (uint64_t i = 0; i < k; i++) {
    for (uint64_t j = 0; j < n;) {
      size_t vl = __riscv_vsetvl_e32m8(n - j);
      vfloat32m8_t inv = ax_vrecip_f32m8(__riscv_vle32_v_f32m8(sbd + j, vl), vl);
      __riscv_vse8_v_i8m2(qbd + i * n + j,
          ax_quant_i8_f32m8(
              __riscv_vfmul_vv_f32m8(__riscv_vle32_v_f32m8(bd + i * n + j, vl), inv, vl), vl),
          vl);
      j += vl;
    }
  }
  // Row absmax of A, then the row's scale broadcast and its quantization, in one pass
  // per row. A horizontal reduce per row costs one queue drain (~68 cycles), which at
  // this M beats vectorising across rows: that form needs a STRIDED load per column of
  // K, and a strided load measures ~2.3x a unit-stride one.
  for (uint64_t i = 0; i < m; i++) {
    vfloat32m8_t mx = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvl_e32m8(k));
    for (uint64_t j = 0; j < k;) {
      size_t vl = __riscv_vsetvl_e32m8(k - j);
      mx = __riscv_vfmax_vv_f32m8(mx,
          __riscv_vfabs_v_f32m8(__riscv_vle32_v_f32m8(ad + i * k + j, vl), vl), vl);
      j += vl;
    }
    size_t vlk = __riscv_vsetvl_e32m8(k);
    float amax = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m8_f32m1(mx, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlk));
    // A zero row would give a zero scale; the floor keeps 1/s finite and the
    // row quantizes to zero either way.
    float s = (amax < 1e-30f ? 1e-30f : amax) / 127.0f;
    float inv = 1.0f / s;
    for (uint64_t l = 0; l < pw;) {
      size_t vl = __riscv_vsetvl_e32m8(pw - l);
      __riscv_vse32_v_f32m8(scd + i * scw + l, __riscv_vfmv_v_f_f32m8(s, vl), vl);
      l += vl;
    }
    for (uint64_t j = 0; j < k;) {
      size_t vl = __riscv_vsetvl_e32m8(k - j);
      __riscv_vse8_v_i8m2(qad + i * k + j,
          ax_quant_i8_f32m8(
              __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(ad + i * k + j, vl), inv, vl), vl),
          vl);
      j += vl;
    }
    // Replicate the column scales into this row's run. Row 0 already holds them.
    if (i > 0) {
      for (uint64_t j = 0; j < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - j);
        __riscv_vse32_v_f32m8(
            scd + i * scw + pw + j, __riscv_vle32_v_f32m8(sbd + j, vl), vl);
        j += vl;
      }
    }
  }

  return 0;
}
