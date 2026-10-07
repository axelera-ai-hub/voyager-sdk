# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Fused SigLIP ViT vision-encoder kernels (device counterpart to vision_reference).

Faithfully adapted from ``models/dino/optimized_impl.py`` -- same fused datapath,
same fold conventions, same export helpers -- with the SigLIP-vs-DINOv2 deltas
applied throughout. The shared structure (kept identical to DINOv2):

  * every linear collapses a dequant chain into ONE llm.quantized_matmul_activation
    ((matmul - dz') * ds (+ LUT activation)) with the per-output-channel BIAS folded
    into the dequant zero-point (dz' = dz - bias/ds);
  * each LayerNorm + the following input-quantize fuse into ONE
    layer_norm_dpu_reduce datapath, with the LN bias folded into the input-quant
    zero-point (qz' = qz + bias*qs) so there is a single per-row IFD1 stream;
  * q/k/v share one input quantize (one SmoothQuant group) and each projection
    fuses the SDPA-input int8 quantize into its matmul write (fuse_output_quant);
  * bidirectional SDPA (is_causal=False), int8 1-head-broadcast padding mask.

SigLIP deltas vs DINOv2 (ALL applied here):

  * EMBED: NO CLS token concat -- the sequence is exactly num_patches (padded up
    to a multiple of 64). Position embedding is a plain LEARNED ABSOLUTE add
    (pos_embed param), NO grid interpolation. The patch conv stays a
    space-to-depth im2col matmul (stride == kernel).
  * BLOCK: NO LayerScale -- the o_proj / fc2 dequant scale is just ``ds`` (no ls
    fold). The MLP activation is the tanh-approx GELU (hidden_act
    "gelu_pytorch_tanh"), and the device LUT is fit to that curve rather than the
    erf one DINOv2 uses (vision_reference.gelu_lut_function picks by hidden_act).
  * POSTNORM: plain LayerNorm (gamma + beta), same as dino make_final_norm_kernel.
  * NO seg head.

Device packing, both exercised only by the 2.2B (SO400M) and device-validated there:

  * head_dim != 64. DINOv2 hard-requires 64 (one PWORD); SO400M is 1152/16 = 72, so each
    head pads to ceil(72/64) = 2 PWORDs (128 lanes), real channels in [0..71], zeros in
    [72..127]. No RoPE here, so the pad is a contiguous tail-zero (physical = h*128 + e),
    not Llama's design-P block-interleave. SDPA ``scale`` uses the LOGICAL head_dim.
  * intermediate_size not a multiple of 64. 4304 -> 4352 with zero weight rows (fc1 out) /
    columns (fc2 in); zero rows dequantize to 0, zero columns contribute 0.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from math import ceil, isfinite
import os
from typing import TYPE_CHECKING

# MODULE-level numpy: the Alto @A.prim_func parser resolves names used in a kernel
# body from module globals, so np.array/np.zeros inside a kernel main must be importable
# here. A function-local ``import numpy as np`` is invisible to the parser ("Undefined
# variable: np"). This matches llama/qwen3 (module-level np, inline np.* in the body).
import numpy as np

from ...export import export_linear_quant_params, export_lut_ranges, export_sdpa_quant_params
from ...layout import fit_lut, pack

PWORD = 64

# Residual/hidden stream dtype: fp16, as DINOv2 ships. Halves the block's DDR I/O and its
# promoted L2 footprint. embed output / block hidden / postnorm input all use this, so the
# single-pass graph threads one consistent dtype. Set SIGLIP_HIDDEN_FP16=0 for fp32.
#
# It defaulted to fp32 through bring-up, when the 256M's promoted workspace (11.25 MiB
# fp32 at seq 1024) did not fit a sub-device's L2 share (~7.7 MiB) and fp16 was the lever
# that made it fit. It fits either way now, so the default is no longer about fitting --
# it is that fp16 costs nothing measurable and buys headroom for the wider towers. Both
# measured on the 256M at cores 4, image_embeds vs the fp reference: fp16 cosine 0.9981 /
# rel-L2 0.0614, fp32 cosine 0.9979 / rel-L2 0.0650. A wash, if anything favouring fp16.
_HIDDEN_FP16 = os.environ.get("SIGLIP_HIDDEN_FP16", "1") == "1"
_HIDDEN_DTYPE = "float16x64" if _HIDDEN_FP16 else "float32x64"

# LUT chain lengths live on SiglipVisionConfig (sdpa_n_luts / gelu_n_luts / norm_n_luts),
# not here: each is a buffer shape below AND what vision_reference emulates.

# Shard the block's remaining single-core matmuls (o_proj, fc1, fc2) across the cores, the way
# dino does. Profiled on the 256M (seq 1024): a block is 9.81 ms single-core and 5.58 ms at
# cores=4, i.e. only 58% of it was parallel -- the serial 42% is exactly these three matmuls
# plus their residual adds. Set SIGLIP_MLP_SHARD=0 to fall back to the single-core path.
_MLP_SHARD = os.environ.get("SIGLIP_MLP_SHARD", "1") == "1"

if TYPE_CHECKING:
    from .vision_reference import SiglipBlock, SiglipVisionConfig


def kernel_flags() -> dict:
    """The SIGLIP_* knobs as resolved AT IMPORT, JSON-able.

    Reshapes the compiled kernel and is read from the environment, so the build records
    it in its manifest rather than leaving it invisible: a build made with
    SIGLIP_HIDDEN_FP16=1 looks identical afterwards to one made without.
    """
    return {"hidden_fp16": bool(_HIDDEN_FP16), "mlp_shard": bool(_MLP_SHARD)}


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _l2_resource_fraction(cores: int, generation: str = "OMEGA") -> float:
    """Fraction of the chip's L2 a ``cores``-wide build may promote into.

    A connection claims one sub-device per compiled core and the L2 arena is split across
    sub-devices, so a build gets ``cores / NUM_AI_CORES``, not all of it. Clamped to (0, 1].
    The denominator is per-generation (OMEGA has 4 AI cores, EUROPA 8), so an 8-core EUROPA
    build gets the whole arena where an 8-core count against OMEGA's 4 would over-claim.
    """
    from axelera.compiler.alto.compiler.config import HardwareGeneration
    from axelera.compiler.alto.compiler.device import NUM_AI_CORES

    gen = HardwareGeneration[generation] if isinstance(generation, str) else generation
    total = int(NUM_AI_CORES.get(gen, 4))
    return min(max(int(cores), 1) / total, 1.0)


def _fc1_shard_cores(inter_pw: int, cores: int) -> int:
    """N-shard width for fc1: how many cores split its output columns.

    ``split_output`` needs ``inter_pw % c == 0`` (it writes a rectangular (c, M, band)
    buffer), so a width that does not divide the PADDED intermediate is not available.
    Prefer dino's 3 when it divides -- 48/3 = 16 is one clean IMC tile per core, where
    48/4 = 12 sub-tiles into 2x6 -- else the widest c <= cores that does. On the 2.2B
    (inter_pw 68) that is 4, since 3 does not divide 68.
    """
    if cores <= 1:
        return 1
    for c in (min(int(cores), 3), *range(int(cores), 1, -1)):
        if inter_pw % c == 0:
            return c
    return 1


def _head_pad_dims(head_dim: int) -> dict:
    """Physical (padded) head dims. head_dim_pw = ceil(head_dim/PWORD) PWORD blocks;
    padded_head_dim = head_dim_pw * PWORD. For head_dim == 64 this is a no-op
    (head_dim_pw == 1), so the unit-test config degenerates exactly to DINOv2."""
    head_dim_pw = (head_dim + PWORD - 1) // PWORD
    return {"head_dim_pw": head_dim_pw, "padded_head_dim": head_dim_pw * PWORD}


# ---------------------------------------------------------------------------
# Patch-embedding kernel (graph kind "embed") -- space-to-depth matmul + pos add.
# ---------------------------------------------------------------------------
def make_siglip_embed_kernel(
    config: "SiglipVisionConfig", seq_len: int | None = None, cores: int = 1
):
    """Build the SigLIP patch-embed IRModule as a space-to-depth matmul.

    Non-overlapping patch projection (stride == kernel == P) == flattening each
    P*P*C patch into the channel dim and running one dense matmul (identical to
    dino make_embed_kernel, minus the CLS concat).

    Pipeline: host-im2col'd int8 pixels (1, num_patches, k_pw) ->
    quantized_matmul_activation (matmul + per-out-channel dequant, conv bias folded
    into dz', no LUT, fp out) -> zero-pad rows up to seq -> add the LEARNED ABSOLUTE
    positional embedding. Output (1, seq, hidden_pw) feeds block 0. ``cores`` is
    ignored (single-core). Graph input ``pixels`` is host space-to-depth + symmetric
    int8 quant.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.quantized_matmul_activation import quantized_matmul_activation

    hidden = config.hidden_size
    hidden_pw = hidden // PWORD
    H = W = config.image_size
    P = config.patch_size
    C = config.num_channels
    gh, gw = H // P, W // P
    num_patches = gh * gw
    patch_dim = C * P * P  # flattened patch length (K); matches host im2col + weight K order
    k = ceil(patch_dim / PWORD) * PWORD  # K padded to a PWORD multiple (pad zeros contribute 0)
    k_pw = k // PWORD
    seq = int(seq_len if seq_len is not None else _round_up(num_patches, PWORD))
    n_pad = seq - num_patches  # zero rows after the real patches (NO CLS row)

    # Two variants, chosen in Python: a ``if`` inside the prim_func would be traced as device
    # control flow, leaving an empty-tensor concat when n_pad == 0 (256M: 1024 patches, no CLS).
    # The class names MUST differ -- the parser's inspect.getsource(cls) matches the FIRST
    # ``class <name>:`` line, so a shared name makes both branches parse the same body.
    if n_pad > 0:

        @A.ir_module
        class SiglipEmbedPad:
            @A.prim_func
            def main(
                pixels: A.Buffer((1, num_patches, k_pw), "int8x64", scope=A.DDR),
                patch_w: A.Buffer((hidden_pw, k), "int8x64", scope=A.DDR),  # NK (N_BLOCKS, K)
                # interleaved per-out-channel [dz', ds] (conv bias folded into dz')
                patch_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
                # learned ABSOLUTE positional embedding, padded to seq (NO interpolation/CLS)
                pos_const: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
            ):  # noqa: F821
                p2 = quantized_matmul_activation(
                    pixels, patch_w, patch_dq, output=BufferInfo(dtype="float32x64")
                )  # (1, num_patches, hidden_pw) fp32 == patch conv dequantized + bias
                zpad = A.alloc_constant(
                    np.zeros((1, n_pad, hidden_pw, PWORD)), "float32x64", scope=A.DDR
                )
                tok = axol.core.concat(p2, zpad, dim=1)  # (1, seq, hidden_pw)
                A.output(axol.atomic.add(tok, pos_const, output=BufferInfo(dtype=_HIDDEN_DTYPE)))

        return SiglipEmbedPad

    else:  # seq == num_patches: no padding, patch output feeds block 0 directly

        @A.ir_module
        class SiglipEmbedNoPad:
            @A.prim_func
            def main(
                pixels: A.Buffer((1, num_patches, k_pw), "int8x64", scope=A.DDR),
                patch_w: A.Buffer((hidden_pw, k), "int8x64", scope=A.DDR),
                patch_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
                pos_const: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
            ):  # noqa: F821
                p2 = quantized_matmul_activation(
                    pixels, patch_w, patch_dq, output=BufferInfo(dtype="float32x64")
                )
                A.output(axol.atomic.add(p2, pos_const, output=BufferInfo(dtype=_HIDDEN_DTYPE)))

        return SiglipEmbedNoPad


def embed_buffers(model, spec: dict) -> dict:
    """Pack the space-to-depth embed kernel's constants from a calibrated
    SiglipVisionModel: the patch weight in NK layout (int8, K = C*P*P padded to a
    PWORD multiple), the interleaved [dz', ds] dequant table with the conv bias
    folded into dz', and the padded learned-absolute positional embedding.

    Adapted from dino embed_buffers: the CLS token is dropped and the positional
    embedding is the plain learned ``pos_embed`` parameter (no interpolation)."""
    import torch

    lin = export_linear_quant_params(model)
    p = lin["patch_embed.proj"]
    hidden = int(model.patch_embed.proj.out_features)
    hidden_pw = hidden // PWORD
    ps = int(model.patch_embed.patch_size)
    c = int(model.patch_embed.num_channels)
    patch_dim = c * ps * ps
    k = ceil(patch_dim / PWORD) * PWORD  # matches make_siglip_embed_kernel

    def blocks(v):  # (C,) -> (C//64, 64)
        return torch.as_tensor(v).to(torch.float32).reshape(-1, PWORD).contiguous()

    # NK weight: w (hidden, C*P*P) int8, K flatten order (c, kh, kw) == host im2col. Pad K to a
    # PWORD multiple, then reshape to (hidden_pw, k, 64) as core.matmul(b_transposed) wants.
    w = p["w"].to(torch.int8)  # (hidden, patch_dim)
    wpad = torch.zeros(hidden, k, dtype=torch.int8)
    wpad[:, :patch_dim] = w
    nk = wpad.reshape(hidden_pw, PWORD, k).permute(0, 2, 1).contiguous()  # (hidden_pw, k, 64)

    # [dz', ds] with the conv bias folded into the dequant zero-point:
    # (matmul - dz')*ds == matmul*ds + bias, dz' = dz - bias/ds (dz == 0 for symmetric qz).
    ds = p["ds"].to(torch.float32)
    bias = model.patch_embed.bias.detach().to(torch.float32)
    dz_folded = p["dz"].to(torch.float32) - bias / ds
    dq = torch.empty((2 * hidden_pw, PWORD), dtype=torch.float32)
    dq[0::2], dq[1::2] = blocks(dz_folded), blocks(ds)

    buffers = {"patch_w": nk}
    buffers["patch_dq"] = dq.reshape(1, 1, 2 * hidden_pw, PWORD).contiguous()
    # pos_embed (1, num_patches, hidden) -> pad rows to seq with zeros (masked/pad positions).
    seq_pad = spec["pos_const"][0][1]
    pos = model.pos_embed.detach().to(torch.float32)
    padded = torch.zeros(1, seq_pad, hidden, dtype=torch.float32)
    padded[:, : pos.shape[1]] = pos
    buffers["pos_const"] = pack(padded, spec["pos_const"])
    return buffers


# ---------------------------------------------------------------------------
# Fused SigLIP encoder-block kernel (graph kind "block").
# ---------------------------------------------------------------------------
def make_siglip_block_kernel(
    config: "SiglipVisionConfig", seq_len: int | None = None, cores: int = 1
):
    """Build the fused IRModule for one SigLIP encoder block.

    Datapath is the DINOv2 block's single-core fused path (fused LN+quant, fused
    q/k/v matmul+dequant+SDPA-input-quant, bidirectional SDPA with fused o_proj
    input quant, fused fc1 dequant+bias+GELU+fc2-input-quant, banded residual adds)
    -- with NO LayerScale and with head-dim / intermediate padding.

    ``cores`` > 1 head-shards SDPA (scaled_dot_product_attention(cores=)), M-shards the
    two LayerNorms, runs q/k/v concurrently, and -- with SIGLIP_MLP_SHARD (default on) --
    N-shards fc1, M-shards fc2 and o_proj, each fusing its residual add into the per-core
    tail. The three sharded matmuls reuse the shared/dino helpers verbatim (see _mlp_out below).
    q/k/v stay one-projection-per-core: sharding a single projection is fixed-overhead
    bound, which is why parallel_matmul_group parallelises the group instead.

    dino's whole-MLP M-shard (optimized_impl._mshard_mlp_residual, DINO_MLP_MSHARD) is NOT
    used: it needs BOTH fc1_w and fc2_w resident in shared L2 (4.7 MiB on the 256M) on top
    of the promoted workspace, and dino measured it a wash (+1.2% at seq 1408).
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod

    from ...ops.layer_norm_dpu_reduce import layer_norm_dpu_reduce_mshard
    from ...ops.parallel_matmul_group import parallel_matmul_group

    # Shared with dino, not copied: both helpers are shape-generic (they take the buffers +
    # seq/n_blocks/cores explicitly and nothing DINOv2-specific), and the SigLIP block is the
    # same datapath. _mlp_out dispatches the MLP layout in plain Python during tracing, so the
    # untaken branch is never parsed as TVMScript.
    # `mshard_matmul_residual` was promoted out of dino into a shared op (it is the same
    # ViT-block fc2/o_proj tail for dino, Qwen3-VL and here); `_mlp_out` is still dino's.
    from ...ops.mshard_matmul_residual import mshard_matmul_residual
    from ..dino.optimized_impl import _mlp_out

    hidden = config.hidden_size
    num_heads = config.num_heads
    head_dim = config.head_dim  # LOGICAL head_dim (e.g. 72 for SO400M)
    inter = config.intermediate_size
    eps = float(config.layernorm_eps)
    # Read here so they reach the LUT buffer shapes below; fit_lut then fits that many.
    sdpa_luts = int(config.sdpa_n_luts)
    gelu_luts = int(config.gelu_n_luts)
    norm_luts = int(config.norm_n_luts)

    hidden_pw = hidden // PWORD
    d = _head_pad_dims(head_dim)
    head_dim_pw = d["head_dim_pw"]  # PWORD blocks per PADDED head (2 for head_dim 72)
    padded_head_dim = d["padded_head_dim"]  # 128 for head_dim 72
    qkv_out_pw = num_heads * head_dim_pw  # PHYSICAL q/k/v output blocks (32 for SO400M)
    inter_pw = ceil(inter / PWORD)  # PADDED intermediate blocks (68 for inter 4304)
    padded_inter = inter_pw * PWORD
    seq = int(seq_len if seq_len is not None else _round_up(config.num_patches, PWORD))
    ctx_blocks = seq // PWORD

    _mc = bool(cores and cores > 1)
    _sdpa_kw = {"cores": cores} if _mc else {}
    _qkv_cores = (0, 1, 2) if (_mc and int(cores) >= 3) else None
    _ln_cores = tuple(range(int(cores))) if _mc else None
    # fc1 N-shard (contiguous split output, consumed split by fc2 -- no strided per-row output
    # DMA, no re-assembly) + fc2/o_proj M-shard over seq row-bands with the residual add banded
    # into each core's tail. The M-shards need seq % cores == 0: true for every shipped variant
    # (1024 and 768 at cores 1/2/4), and a width that does not divide just falls back to
    # single-core rather than failing the build.
    _shard = _mc and _MLP_SHARD
    _fc_cores = _fc1_shard_cores(inter_pw, int(cores)) if _shard else 1
    _ms_cores = int(cores) if (_shard and seq % int(cores) == 0) else None
    # Residual accumulator dtype, kept == the hidden stream for exactness (dino writes o6/d6
    # as fp16). Moves with SIGLIP_HIDDEN_FP16.
    _resid_dtype = _HIDDEN_DTYPE

    @A.ir_module
    class SiglipBlockFused:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq, hidden_pw), _HIDDEN_DTYPE, scope=A.DDR),
            ln1_w: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),  # D-major
            ln1_lut: A.Buffer((norm_luts,), "float32x64", scope=A.DDR),
            # q/k/v share one interleaved [scale, zp] input-quant table (one SmoothQuant group);
            # ln1 bias is FOLDED into its zero-point (see block_buffers), so no bias stream.
            qkv_qparams: A.Buffer((2 * hidden_pw,), "float32x64", scope=A.DDR),
            # NK weights (N_BLOCKS = PADDED head-major output blocks, K = hidden) + 4-wide
            # [dz', ds, q_attn_scale, 0] per PADDED output block: the SDPA-input int8 quantize
            # is fused into the matmul write (fuse_output_quant). Pad head lanes carry zero.
            q_proj_w: A.Buffer((qkv_out_pw, hidden), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 4 * qkv_out_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((qkv_out_pw, hidden), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 4 * qkv_out_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((qkv_out_pw, hidden), "int8x64", scope=A.DDR),
            v_proj_dq: A.Buffer((1, 1, 4 * qkv_out_pw), "float32x64", scope=A.DDR),
            sdpa_exp_lut: A.Buffer((sdpa_luts,), "float32x64", scope=A.DDR),
            sdpa_inv_lut: A.Buffer((sdpa_luts,), "float32x64", scope=A.DDR),
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            # int8 1-head-broadcast padding mask ({0, -127}): bidirectional attention over the
            # real patches; padded key columns (seq padded up to x64) are masked. is_causal=False.
            attn_mask: A.Buffer((1, 1, seq, ctx_blocks), "int8x64", scope=A.DDR),
            # o_proj input quant (SDPA-output quant): per PADDED head-major input block.
            o_proj_qparams: A.Buffer((2 * qkv_out_pw,), "float32x64", scope=A.DDR),
            o_proj_w: A.Buffer((hidden_pw, num_heads * padded_head_dim), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
            ln2_w: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),  # D-major
            ln2_lut: A.Buffer((norm_luts,), "float32x64", scope=A.DDR),
            fc1_qparams: A.Buffer((2 * hidden_pw,), "float32x64", scope=A.DDR),
            fc1_w: A.Buffer((inter_pw, hidden), "int8x64", scope=A.DDR),
            # 4-wide [dz', ds, fc2_in_scale, fc2_in_zp] per PADDED intermediate block: fc1's
            # dequant+bias+GELU AND fc2's input quantize fused into one write (fuse_output_quant).
            fc1_dq: A.Buffer((1, 1, 4 * inter_pw), "float32x64", scope=A.DDR),
            gelu_lut: A.Buffer((gelu_luts,), "float32x64", scope=A.DDR),
            fc2_w: A.Buffer((hidden_pw, padded_inter), "int8x64", scope=A.DDR),
            fc2_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant([eps] * PWORD, dtype="float32x64", scope=A.DDR)
            sdpa_w_zp = A.alloc_constant(
                np.array([0] * PWORD).reshape((1, 1, 1, 1, PWORD)), "float32x64", scope=A.DDR
            )
            sdpa_w_qmin = A.alloc_constant(
                np.array([-128] * PWORD).reshape((1, 1, 1, 1, PWORD)), "float32x64", scope=A.DDR
            )
            sdpa_w_qmax = A.alloc_constant(
                np.array([127] * PWORD).reshape((1, 1, 1, 1, PWORD)), "float32x64", scope=A.DDR
            )
            sdpa_q_zp = A.alloc_constant(
                np.array([0] * PWORD).reshape((1, 1, 1, 1, PWORD)), "float32x64", scope=A.DDR
            )
            sdpa_v_zp = A.alloc_constant(
                np.array([0] * PWORD).reshape((1, 1, 1, 1, PWORD)), "float32x64", scope=A.DDR
            )

            # --- Attention sub-block (NO LayerScale) ---
            # Fused LayerNorm 1 + q/k/v input quantize (int8 out) in one DPU-reduce datapath.
            # ln1 bias is folded into qkv_qparams' zero-point host-side (no bias stream).
            qkv_q = layer_norm_dpu_reduce_mshard(
                hidden_states,
                ln1_lut,
                eps_const[0:1],
                normalized_shape=(hidden,),
                cores=_ln_cores,
                weight_buffer=ln1_w,
                quant_params=qkv_qparams,
                output_dtype="int8x64",
            )

            # q/k/v projections (fuse_output_quant int8) + reshape to PADDED heads + transpose to
            # (1, num_heads, seq, head_dim_pw). head_dim_pw == 2 for SO400M (padded head 128).
            # reshape_shape folds the PHYSICAL padded head layout (num_heads, head_dim_pw);
            # the pad lanes carry zero from the zero weight rows.
            q8, k8, v8 = parallel_matmul_group(
                qkv_q,
                [q_proj_w, k_proj_w, v_proj_w],
                [q_proj_dq, k_proj_dq, v_proj_dq],
                cores=_qkv_cores,
                fuse_output_quant=True,
                output_dtype="int8x64",
                reshape_shape=(1, seq, num_heads, head_dim_pw),
                transpose_dims=(0, 2, 1, 3),
            )

            attn = axol.core.scaled_dot_product_attention(
                q8,
                k8,
                v8,
                sdpa_scores_dq_scale,
                sdpa_exp_lut,
                sdpa_inv_lut,
                sdpa_weights_q_scale,
                sdpa_w_zp,
                sdpa_w_qmin,
                sdpa_w_qmax,
                sdpa_output_dq_scale,
                sdpa_q_zp,
                sdpa_v_zp,
                attn_mask,
                is_causal=False,  # bidirectional encoder
                enable_gqa=False,
                # LOGICAL head_dim (72), NOT the padded 128: pad lanes are zero in Q/K (add 0
                # to Q.K^T) and in V (add 0 to P@V), so 128 padded lanes score as the 72 real
                # ones. Device-confirmed on the 2.2B.
                scale=1.0 / (head_dim**0.5),
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
                # Fuse the o_proj INPUT quantize into SDPA's P@V write: o_proj_qparams is per
                # PADDED head-major input block, exactly what the quantizer indexes.
                attention_output_quantizer=o_proj_qparams,
                **_sdpa_kw,
            )
            attn_t = axol.core.transpose(attn, dims=(0, 2, 1, 3))  # int8
            o3 = axol.atomic.reshape(attn_t, shape=(1, seq, qkv_out_pw))  # int8 o_proj input
            # o_proj (NO LayerScale -- dequant scale is just ds) + residual add res1 = hidden + o6.
            # o_proj reads the PADDED head-major input (K = num_heads*padded_head_dim); pad weight
            # columns are zero (see block_buffers), so pad input lanes contribute 0. M-sharded over
            # seq row-bands when enabled: o3 is a normal (un-split) input, so matrix_a_is_split is
            # False. o_proj's matmul is fixed-overhead bound like q/k/v, so the win here is mostly
            # the BANDED residual add, not the matmul.
            res1 = mshard_matmul_residual(
                o3,
                o_proj_w,
                o_proj_dq,
                hidden_states,
                seq=seq,
                n_blocks=hidden_pw,
                cores=(_ms_cores or 1),
                matrix_a_is_split=False,
                resid_dtype=_resid_dtype,
                out_dtype=_HIDDEN_DTYPE,
            )

            # --- MLP sub-block (gelu-tanh, NO LayerScale) ---
            # Fused LayerNorm 2 + fc1 input quantize; ln2 bias folded into fc1_qparams' zp.
            f2 = layer_norm_dpu_reduce_mshard(
                res1,
                ln2_lut,
                eps_const[0:1],
                normalized_shape=(hidden,),
                cores=_ln_cores,
                weight_buffer=ln2_w,
                quant_params=fc1_qparams,
                output_dtype="int8x64",
            )
            # fc1 matmul + dequant + bias + GELU + fc2-input int8 quantize (fuse_output_quant)
            # -> fc2 (NO LayerScale, reads the PADDED intermediate K = padded_inter; pad weight
            # columns are zero) -> residual add. gelu_lut is fit to whichever curve the block's
            # RangeObserver names -- "gelu_tanh" for SigLIP, "gelu" (erf) for a DINOv2-style
            # config (vision_reference.gelu_lut_function -> layout.fit_lut). Pad intermediate
            # channels: matmul 0 -> dequant 0 -> gelu(0)=0 -> quant (zp 0) -> 0.
            #
            # _mlp_out emits fc1 N-sharded over _fc_cores (split output) + fc2 M-sharded over
            # _ms_cores row-bands consuming that split K, with the residual add banded into each
            # core's tail; _fc_cores == 1 and _ms_cores is None give back the plain single-core
            # pair. mlp_mshard=False: see the whole-MLP note in this function's docstring.
            A.output(
                _mlp_out(
                    f2,
                    fc1_w,
                    fc1_dq,
                    gelu_lut,
                    fc2_w,
                    fc2_dq,
                    res1,
                    mlp_mshard=False,
                    mlp_cores=int(cores),
                    fc_cores=_fc_cores,
                    fc1_split=_fc_cores > 1,
                    fc2_ms_cores=_ms_cores,
                    seq=seq,
                    hidden_pw=hidden_pw,
                    resid_dtype=_resid_dtype,
                    out_dtype=_HIDDEN_DTYPE,
                )
            )

    return SiglipBlockFused


# Projection buffer prefix -> the module attribute holding the float bias (folded into dz').
_PROJ_BIAS = {
    "q_proj": "q_bias",
    "k_proj": "k_bias",
    "v_proj": "v_bias",
    "o_proj": "o_bias",
    "fc1": "fc1_bias",
    "fc2": "fc2_bias",
}
_SDPA_SCALARS = {
    "sdpa_scores_dq_scale": "scores_dq_scale",
    "sdpa_weights_q_scale": "qs_w",
    "sdpa_output_dq_scale": "output_dq_scale",
}
_NORM_W = {"ln1_w": "ln1.weight", "ln2_w": "ln2.weight"}


def _fit_lut_checked(entry: dict, spec_entry, site: str):
    """``fit_lut``, but say so first when the recorded range is not finite.

    An inf bound fits CLEANLY over an infinite domain, so the segments land on nonsense
    and the artifact is wrong on device -- the one LUT failure nothing else catches (an
    unrecorded site is refused at the calibration set, a nan bound raises in the fit).
    """
    lo, hi = float(entry["min"]), float(entry["max"])
    if not (isfinite(lo) and isfinite(hi)):
        print(f"[lut] WARNING: {site} range [{lo}, {hi}] not finite; table unusable", flush=True)
    return fit_lut(entry, spec_entry)


_LUT_SITES = {
    "ln1_lut": "ln1.inv_sqrt_range",
    "ln2_lut": "ln2.inv_sqrt_range",
    "gelu_lut": "gelu_range",
    "sdpa_exp_lut": "sdpa.exp_range",
    "sdpa_inv_lut": "sdpa.reciprocal_range",
}
# Interleaved input-quant [scale, zp] buffer -> the linear providing qs/qz; the LN bias
# folded into its zero-point (qz' = qz + ln_bias*qs) so the fused norm needs no bias stream.
_INPUT_QP = {"qkv_qparams": "q_proj", "fc1_qparams": "fc1"}
_QP_BIAS_FOLD = {"qkv_qparams": "ln1.bias", "fc1_qparams": "ln2.bias"}


def block_buffers(block: "SiglipBlock", spec: dict) -> dict:
    """Bind the fused SigLIP block kernel's buffers.

    Adapted from dino optimized_block_buffers, with: NO LayerScale fold; the head
    dimension padded to head_dim_pw PWORD blocks on q/k/v OUTPUT + o_proj INPUT; the
    intermediate dimension padded to a PWORD multiple on fc1 OUTPUT + fc2 INPUT.
    Pad channels carry zero weight (dequant to 0) so they never affect the output.
    """
    import torch

    lin = export_linear_quant_params(block)
    sdpa = export_sdpa_quant_params(block)["sdpa"]
    rng = export_lut_ranges(block)
    cfg = block.config
    hidden = cfg.hidden_size
    num_heads = cfg.num_heads
    head_dim = cfg.head_dim
    inter = cfg.intermediate_size
    hidden_pw = hidden // PWORD
    padded_head_dim = _head_pad_dims(head_dim)["padded_head_dim"]
    qkv_out = num_heads * padded_head_dim
    qkv_out_pw = qkv_out // PWORD
    inter_pw = ceil(inter / PWORD)
    padded_inter = inter_pw * PWORD

    buffers: dict = {}

    def put(name, value):
        if name in spec:
            buffers[name] = pack(value, spec[name])

    def blocks(v):  # (C,) -> (C//64, 64)
        return torch.as_tensor(v).to(torch.float32).reshape(-1, PWORD).contiguous()

    def interleave(scale, zp, c_pw):  # (2*c_pw, 64): [scale0, zp0, scale1, zp1, ...]
        qp = torch.empty((2 * c_pw, PWORD), dtype=torch.float32)
        qp[0::2], qp[1::2] = blocks(scale), blocks(zp)
        return qp

    def interleave4(dz, ds, qs, qz, c_pw):  # (4*c_pw, 64): [dz, ds, q_scale, q_zp] per block
        qp = torch.empty((4 * c_pw, PWORD), dtype=torch.float32)
        qp[0::4], qp[1::4], qp[2::4], qp[3::4] = blocks(dz), blocks(ds), blocks(qs), blocks(qz)
        return qp

    def nk(w):  # padded [out, in] int8 (out % 64 == 0) -> NK (out//64, in, 64)
        out, inn = w.shape
        return w.reshape(out // PWORD, PWORD, inn).permute(0, 2, 1).contiguous().to(torch.int8)

    def pad_head_out(t, fill=0.0):
        """Scatter t's axis-0 (num_heads*head_dim real channels, head-major) into the
        PADDED head-major layout (num_heads*padded_head_dim). physical = h*padded + e."""
        # Contiguous tail-zero pad (NO RoPE block-interleave), matching the device reshape's
        # head-major channel order.
        out = torch.full((qkv_out, *t.shape[1:]), fill, dtype=t.dtype)
        for h in range(num_heads):
            out[h * padded_head_dim : h * padded_head_dim + head_dim] = t[
                h * head_dim : (h + 1) * head_dim
            ]
        return out

    def pad_head_in_cols(w, fill=0):
        """Scatter weight columns (axis-1, head-major input) into the padded layout."""
        out = torch.full((w.shape[0], qkv_out), fill, dtype=w.dtype)
        for h in range(num_heads):
            out[:, h * padded_head_dim : h * padded_head_dim + head_dim] = w[
                :, h * head_dim : (h + 1) * head_dim
            ]
        return out

    def pad_head_vec(v, fill=0.0):  # per-channel head-major vector (num_heads*head_dim,) -> padded
        out = torch.full((qkv_out,), fill, dtype=torch.float32)
        for h in range(num_heads):
            out[h * padded_head_dim : h * padded_head_dim + head_dim] = torch.as_tensor(v)[
                h * head_dim : (h + 1) * head_dim
            ].to(torch.float32)
        return out

    def pad_tail(v, length, fill=0.0):  # 1D vector -> length, real prefix then fill
        out = torch.full((length,), fill, dtype=torch.float32)
        vt = torch.as_tensor(v).to(torch.float32).reshape(-1)
        out[: vt.numel()] = vt
        return out

    def pad_rows(w, length, fill=0):  # [out, in] -> [length, in], zero rows appended
        out = torch.full((length, w.shape[1]), fill, dtype=w.dtype)
        out[: w.shape[0]] = w
        return out

    def pad_cols(w, length, fill=0):  # [out, in] -> [out, length], zero cols appended
        out = torch.full((w.shape[0], length), fill, dtype=w.dtype)
        out[:, : w.shape[1]] = w
        return out

    def attn_scale_padded(field):
        """Per-PADDED-head-block SDPA input scale (uniform across heads, zp-pinned). The
        SDPA input quant is per-head-dim; scatter the real head_dim scales into the padded
        head then tile across heads -> (qkv_out,). Pad lanes -> 1.0 (they multiply zero)."""
        s = torch.as_tensor(sdpa[field]).to(torch.float32).reshape(-1)
        if s.numel() == 1:  # per-tensor: broadcast over the (logical) head-dim
            s = s.expand(head_dim)
        per_head = torch.ones(padded_head_dim, dtype=torch.float32)
        per_head[:head_dim] = s
        return per_head.repeat(num_heads)  # (qkv_out,)

    _FUSED_OQ = {"q_proj": "qs_q", "k_proj": "qs_k", "v_proj": "qs_v"}

    # --- q/k/v projections: pad the OUTPUT (head) axis ---
    for proj in ("q_proj", "k_proj", "v_proj"):
        p = lin[proj]
        if f"{proj}_w" in spec:
            buffers[f"{proj}_w"] = nk(pad_head_out(p["w"].to(torch.int8)))
        name = f"{proj}_dq"
        if name in spec:
            ds = p["ds"].to(torch.float32)
            bias = getattr(block, _PROJ_BIAS[proj]).detach().to(torch.float32)
            dz_folded = p["dz"].to(torch.float32) - bias / ds  # (matmul - dz')*ds = deq + bias
            ds_p = pad_head_vec(ds, fill=1.0)  # pad rows: ds arbitrary (matmul==0 there)
            dz_p = pad_head_vec(dz_folded, fill=0.0)
            qs_p = attn_scale_padded(_FUSED_OQ[proj])  # SDPA-input scale, zero-pinned
            qz_p = torch.zeros(qkv_out, dtype=torch.float32)
            buffers[name] = (
                interleave4(dz_p, ds_p, qs_p, qz_p, qkv_out_pw)
                .reshape(1, 1, 4 * qkv_out_pw, PWORD)
                .contiguous()
            )

    # --- o_proj: pad the INPUT (head) axis; output (hidden) unpadded, NO LayerScale ---
    p = lin["o_proj"]
    if "o_proj_w" in spec:
        buffers["o_proj_w"] = nk(pad_head_in_cols(p["w"].to(torch.int8)))
    if "o_proj_dq" in spec:
        ds = p["ds"].to(torch.float32)
        bias = block.o_bias.detach().to(torch.float32)
        dz_folded = p["dz"].to(torch.float32) - bias / ds
        buffers["o_proj_dq"] = (
            interleave(dz_folded, ds, hidden_pw).reshape(1, 1, 2 * hidden_pw, PWORD).contiguous()
        )
    if "o_proj_qparams" in spec:
        # SDPA-output quant = o_proj input quant (per input channel, head-major). Pad lanes:
        # scale -> 1.0 (multiply zero P@V lanes), zp -> 0.
        qs_p = pad_head_vec(p["qs"], fill=1.0)
        qz_p = pad_head_vec(p["qz"], fill=0.0)
        buffers["o_proj_qparams"] = interleave(qs_p, qz_p, qkv_out_pw)

    # --- fc1: pad the OUTPUT (intermediate) axis; fuse fc2's input quant into the write ---
    p = lin["fc1"]
    if "fc1_w" in spec:
        buffers["fc1_w"] = nk(pad_rows(p["w"].to(torch.int8), padded_inter))
    if "fc1_dq" in spec:
        ds = p["ds"].to(torch.float32)
        bias = block.fc1_bias.detach().to(torch.float32)
        dz_folded = p["dz"].to(torch.float32) - bias / ds
        ds_p = pad_tail(ds, padded_inter, fill=1.0)
        dz_p = pad_tail(dz_folded, padded_inter, fill=0.0)
        fq = lin["fc2"]  # fc1's output feeds fc2, so its output quant == fc2's input quant
        qs_p = pad_tail(fq["qs"].reshape(-1), padded_inter, fill=1.0)
        qz_p = pad_tail(fq["qz"].reshape(-1), padded_inter, fill=0.0)
        buffers["fc1_dq"] = (
            interleave4(dz_p, ds_p, qs_p, qz_p, inter_pw)
            .reshape(1, 1, 4 * inter_pw, PWORD)
            .contiguous()
        )

    # --- fc2: pad the INPUT (intermediate) axis; output (hidden) unpadded, NO LayerScale ---
    p = lin["fc2"]
    if "fc2_w" in spec:
        buffers["fc2_w"] = nk(pad_cols(p["w"].to(torch.int8), padded_inter))
    if "fc2_dq" in spec:
        ds = p["ds"].to(torch.float32)
        bias = block.fc2_bias.detach().to(torch.float32)
        dz_folded = p["dz"].to(torch.float32) - bias / ds
        buffers["fc2_dq"] = (
            interleave(dz_folded, ds, hidden_pw).reshape(1, 1, 2 * hidden_pw, PWORD).contiguous()
        )

    # --- input-quant tables (LN bias folded into zp) ---
    for buf, proj in _INPUT_QP.items():
        if buf in spec:
            p = lin[proj]
            c_pw = p["qs"].numel() // PWORD
            qs = p["qs"].to(torch.float32).reshape(-1)
            qz = p["qz"].to(torch.float32).reshape(-1)
            bias = block.get_parameter(_QP_BIAS_FOLD[buf]).detach().to(torch.float32).reshape(-1)
            qz = qz + bias * qs  # fold LayerNorm bias into the quant zero-point
            buffers[buf] = interleave(qs, qz, c_pw)

    for buf, field in _SDPA_SCALARS.items():
        put(buf, sdpa[field])
    for buf, path in _NORM_W.items():
        put(buf, block.get_parameter(path).detach())  # D-major reshape via pack
    for buf, site in _LUT_SITES.items():
        if buf in spec and site in rng:
            buffers[buf] = _fit_lut_checked(rng[site], spec[buf], site)
    return buffers


# ---------------------------------------------------------------------------
# Post-LayerNorm kernel (graph kind "postnorm") -- same as dino make_final_norm_kernel.
# ---------------------------------------------------------------------------
def make_siglip_postnorm_kernel(
    config: "SiglipVisionConfig", seq_len: int | None = None, cores: int = 1
):
    """Build the final post-LayerNorm IRModule over the last block's output. Plain
    LayerNorm (gamma + beta) via the single-pass DPU-reduce norm, M-sharded across
    ``cores`` (rows independent; weight is tiny -- no L2 staging)."""
    from axelera.compiler.alto import script as A

    from ...ops.layer_norm_dpu_reduce import layer_norm_dpu_reduce_mshard

    hidden = config.hidden_size
    hidden_pw = hidden // PWORD
    eps = float(config.layernorm_eps)
    norm_luts = int(config.norm_n_luts)
    seq = int(seq_len if seq_len is not None else _round_up(config.num_patches, PWORD))
    pn_cores = tuple(range(int(cores))) if (cores and cores > 1) else None

    @A.ir_module
    class SiglipPostNorm:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq, hidden_pw), _HIDDEN_DTYPE, scope=A.DDR),
            pn_w: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),  # D-major gamma
            pn_b: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),  # D-major beta
            pn_lut: A.Buffer((norm_luts,), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant([eps] * PWORD, dtype="float32x64", scope=A.DDR)
            A.output(
                layer_norm_dpu_reduce_mshard(
                    hidden_states,
                    pn_lut,
                    eps_const[0:1],
                    normalized_shape=(hidden,),
                    cores=pn_cores,
                    weight_buffer=pn_w,
                    bias_buffer=pn_b,
                    invsqrt_method="additive",
                )
            )

    return SiglipPostNorm


def postnorm_buffers(model, spec: dict) -> dict:
    """Pack the post-LayerNorm gamma/beta (D-major) + the inv_sqrt LUT (fit over the
    recorded post-norm range). Reads ``model.post_layernorm``."""
    rng = export_lut_ranges(model)
    buffers = {
        "pn_w": pack(model.post_layernorm.weight.detach(), spec["pn_w"]),
        "pn_b": pack(model.post_layernorm.bias.detach(), spec["pn_b"]),
    }
    pn_site = "post_layernorm.inv_sqrt_range"
    buffers["pn_lut"] = _fit_lut_checked(rng[pn_site], spec["pn_lut"], pn_site)
    return buffers


# ---------------------------------------------------------------------------
# Compile helper (mirrors dino compile_block) + host pixel quantize.
# ---------------------------------------------------------------------------
def compile_kernel(
    config: "SiglipVisionConfig",
    out_dir,
    *,
    make_kernel,
    target=None,
    seq_len: int | None = None,
    enable_buffer_promotion: bool = False,
    cores: int = 1,
    connection_cores: int | None = None,
    dpu_unroll: bool = False,
    generation: str = "OMEGA",
):
    """Compile one kernel for *target*. Returns (mod, model_json). target defaults to
    AICORE_SIM (the functional simulator); pass Target.DEVICE for Metis silicon.

    ``generation`` names the silicon ("OMEGA" / "EUROPA"); it picks the L1/L2/IMC specs the
    ops budget against and the sub-device count the L2 fraction divides by.

    ``cores`` is the kernel's width; ``connection_cores`` is how many sub-devices the
    connection it loads onto claims, which is what sets the L2 budget. They differ only for
    a kernel that is single-core while sharing a connection with kernels that are not."""
    from pathlib import Path

    from axelera.compiler.alto.compiler.config import AltoConfig, HardwareGeneration, Target
    from axelera.compiler.alto.compiler.driver import compile as alto_compile

    if target is None:
        target = Target.AICORE_SIM
    out_dir = Path(out_dir)
    mod = make_kernel(config, seq_len, cores)
    alto_config = AltoConfig(
        output_dir=out_dir,
        generation=(HardwareGeneration[generation] if isinstance(generation, str) else generation),
        target=target,
        num_cores=cores,
        # The L2 this kernel ACTUALLY gets at load time. Default resources=1.0 budgets the
        # whole chip (30.81 MiB), but the runtime splits the arena per sub-device and we open
        # num_sub_devices=cores, so a cores=1 build really gets ~7.7 MiB. Over-budgeting
        # compiles fine and then fails minutes later at executor creation with "Failed to
        # allocate memory pool 'shared_l2'". (ValidateConfig's own //= num_cores fires only
        # under auto_multicore_mode='batch'; ours is native/manual, so it never does.)
        resources=_l2_resource_fraction(
            cores if connection_cores is None else connection_cores, generation
        ),
        enable_tiling_injection=True,
        enable_buffer_promotion=enable_buffer_promotion,
        enable_llm_support=True,
        assign_unique_input_pools=True,
        unpack_axm_archive=True,
        clean_output_dir=True,
        emit_atex_artifacts=True,
        emit_alto_config=True,
        dpu_unroll=dpu_unroll,
    )
    alto_compile(mod, alto_config)
    model_jsons = sorted(out_dir.rglob("model.json"))
    if not model_jsons:
        raise RuntimeError(f"no model.json produced under {out_dir}")
    return mod, model_jsons[0]


def _quantize_pixels(pixel_values, pix_scale: float, patch_size: int):
    """(1,C,H,W) fp -> space-to-depth matmul input (1, num_patches, k_pw, 64) int8.

    Host im2col: each P*P patch flattened in (c, kh, kw) order (== the patch weight's
    K order), symmetric int8 quantized (zp 0, matching the symmetric patch conv), K
    padded up to a PWORD multiple. Reused verbatim from the dino host pipeline (see
    processor.py). Returns a numpy int8 array."""
    import torch

    pix_i8 = (
        torch.clamp(torch.round(pixel_values[0].float() / pix_scale), -127, 127)
        .to(torch.int8)
        .numpy()
    )
    c, h, w = pix_i8.shape
    p = int(patch_size)
    gh, gw = h // p, w // p
    # Crop a partial trailing patch, exactly as the stride==kernel conv does -- see
    # SiglipPatchEmbed._im2col. SmolVLM2-2.2B (384px tiles, patch 14) hits this: 27x27
    # patches over [0, 378), last 6 rows/columns unused. No-op at 512/16.
    pix_i8 = pix_i8[:, : gh * p, : gw * p]
    # (C, gh, kh, gw, kw) -> (gh, gw, C, kh, kw) -> (num_patches, C*P*P), flatten (c, kh, kw).
    cols = pix_i8.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, c * p * p)
    k = c * p * p
    k_pw = ceil(k / PWORD)
    data = np.zeros((1, gh * gw, k_pw * PWORD), np.int8)
    data[0, :, :k] = cols  # remaining K lanes stay zero (matches the padded patch weight)
    return data.reshape(1, gh * gw, k_pw, PWORD)
