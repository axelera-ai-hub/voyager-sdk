# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-VL vision tower in alto script: patch embed -> N SigLIP2 blocks -> connector.

The whole tower is authored as ONE device graph and runs in a single dispatch: flattened
int8 patches in, image embeddings out. Only the processor's normalization, the int8 patch
quantize, the merge-block permutation and an output slice stay on the host.

Two of the three kernels already exist in this tree:

* **the block** is the fused ViT block of ``models/dino/optimized_impl``. A SigLIP2
  encoder layer is a plain pre-norm ViT block -- LayerNorm, three separate biased
  projections, bidirectional attention, biased GELU MLP -- so that kernel and its binder
  apply as written, and with them the whole multicore story: SDPA head-sharded, both
  LayerNorms M-sharded, ``fc1`` N-sharded into an ``fc2`` that consumes the split
  directly, ``fc2``/``o_proj`` M-sharded with the residual add banded into the per-core
  tail, and q/k/v run as a parallel group on three cores. There is no rotary embedding in
  this tower, so nothing about the block changes at all. It is also what makes the So400m
  encoder of LFM2.5-VL-3B a build rather than a port: its head_dim (72) and MLP width
  (4304) are not PWORD multiples, and that kernel lays both out padded with zeros the
  binder cannot let leak.
* **the embed** is the space-to-depth matmul. SigLIP2 NaFlex patchifies on the host and
  projects with a plain Linear, so the patch embed is one dense int8 matmul over each
  patch's ``C * P * P`` features -- already the layout the image processor emits. The
  bias folds into the dequant zero point; the resampled position grid is a constant.

The one new kernel is the **connector** (:func:`make_connector_kernel`): the encoder's
``post_layernorm`` at vision width, the PixelUnshuffle fold (a pure reshape, because the
host handed the tower its patches in merge-block order -- see
``lfm2vl.merge_permutation``), optionally the connector's own LayerNorm at merged width,
then the GELU MLP down into the decoder's hidden size. Both matmuls N-shard over their
own output blocks, with ``fc1``'s split output feeding ``fc2`` as split K.

Every kernel takes the core count, so nothing in the tower is pinned to one core.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names
#
# NB: no `from __future__ import annotations` here. TVMScript reads the prim_func's
# parameter annotations at runtime, and PEP 563 would hand it strings instead of the
# A.Buffer objects it parses.

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...export import export_linear_quant_params, export_lut_ranges
from ...layout import broadcast_to_pword, extract_layer_buffer_spec, fit_lut, pack
from .helpers import DEFAULT_GENERATION
from ..dino.optimized_impl import (
    _HIDDEN_DTYPE,
    make_optimized_block_kernel,
    optimized_block_buffers,
)

if TYPE_CHECKING:
    from .lfm2vl import Lfm2VlVisionConfig, Lfm2VlVisionModel

#: int8 mask fill for padded key columns -- exp-negligible after score scaling.
_MASK_NEG = -127


def _fit_cores(blocks: int, cores: int | None) -> int:
    """The largest core count up to ``cores`` that divides ``blocks`` evenly.

    Every N-shard here writes a contiguous split output, which cannot be ragged, so a
    matmul can only use as many cores as divide its own output-block count. Taking the
    largest divisor rather than a fixed number lets each width use what it has.
    """
    return max(c for c in range(1, int(cores or 1) + 1) if blocks % c == 0)


# --------------------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------------------


def _pad_tokens(embedded, n_pad: int, hidden_pw: int):
    """Append ``n_pad`` zero rows so the token axis is a PWORD multiple (no-op if aligned)."""
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A

    if not n_pad:
        return embedded
    zpad = A.alloc_constant(np.zeros((1, n_pad, hidden_pw, 64)), "float32x64", scope=A.DDR)
    return axol.core.concat(embedded, zpad, dim=1)


def make_vision_embed_kernel(cfg: "Lfm2VlVisionConfig", cores: int = 1):
    """Patch projection: int8 patches -> hidden + the resampled position grid.

    SigLIP2 NaFlex projects already-patchified pixels with an ``nn.Linear``, so this is
    one dense matmul over each patch's ``C * P * P`` features and there is no im2col on
    device. The projection bias is folded into the dequant zero point
    (``dz' = dz - bias/ds``). Padding rows (patch count up to the PWORD multiple) are
    appended as zeros; the position constant is zero there too.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.quantized_matmul_activation import quantized_matmul_activation_multicore

    hidden_pw = cfg.hidden_size // 64
    k_pw = cfg.patch_dim // 64
    n_patches, seq = cfg.num_patches, cfg.seq_len
    n_pad = seq - n_patches
    # The output stays unified (the position add and the token pad read it whole), so this
    # is the strided-write N-shard: worth little on its own, but it stops the kernel
    # ignoring `cores` outright.
    proj_cores = _fit_cores(hidden_pw, cores)

    @A.ir_module
    class Lfm2VlEmbed:
        @A.prim_func
        def main(
            patches: A.Buffer((1, n_patches, k_pw), "int8x64", scope=A.DDR),
            patch_w: A.Buffer((hidden_pw, cfg.patch_dim), "int8x64", scope=A.DDR),  # NK
            patch_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),  # [dz', ds]
            pos_const: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            embedded = quantized_matmul_activation_multicore(
                patches,
                patch_w,
                patch_dq,
                cores=proj_cores,
                output=BufferInfo(dtype="float32x64"),
            )
            tokens = _pad_tokens(embedded, n_pad, hidden_pw)
            A.output(axol.atomic.add(tokens, pos_const, output=BufferInfo(dtype=_HIDDEN_DTYPE)))

    return Lfm2VlEmbed


def make_vision_block_kernel(cfg: "Lfm2VlVisionConfig", cores: int = 1):
    """One fused SigLIP2 encoder block, at this tower's shape.

    Nothing is added or removed: the shared ViT block kernel already spells this layer
    out, and LayerScale (which SigLIP2 lacks) is pinned at 1 and folded into a dequant
    scale where it disappears. Named here so the graph builder reads one module.

    The block config also carries the DEVICE widths, so a So400m encoder is built at its
    padded head stride and MLP width without anything here knowing about it.
    """
    return make_optimized_block_kernel(cfg.block_config(), cfg.seq_len, cores)


def _connector_dims(cfg, cores: int):
    """Widths, core counts and the norm choice shared by the two connector builders."""
    from types import SimpleNamespace

    proj_pw = cfg.projector_hidden_size // 64
    out_pw = cfg.out_hidden_size // 64
    # Both matmuls N-shard over their own output blocks. fc1's split output feeds fc2 as
    # split K (matrix_a_is_split), so there is no un-split in between.
    fc1_cores = _fit_cores(proj_pw, cores)
    return SimpleNamespace(
        hidden=cfg.hidden_size,
        hidden_pw=cfg.hidden_size // 64,
        merged_dim=cfg.merged_dim,
        merged_pw=cfg.merged_dim // 64,
        proj_dim=cfg.projector_hidden_size,
        proj_pw=proj_pw,
        out_pw=out_pw,
        seq=cfg.seq_len,
        merged_seq=cfg.seq_len // cfg.merge_unit,
        eps=float(cfg.layernorm_eps),
        # This tower's LUT chain lengths, not the shared DINO_* defaults -- see
        # Lfm2VlVisionConfig. They are the buffer shapes fit_lut fits into.
        norm_luts=int(cfg.norm_n_luts),
        gelu_luts=int(cfg.gelu_n_luts),
        ln_cores=tuple(range(int(cores))) if cores and cores > 1 else None,
        fc1_cores=fc1_cores,
        fc2_cores=_fit_cores(out_pw, cores),
        fc1_split=fc1_cores > 1,
    )


def _connector_mlp(merged, qparams, fc1_w, fc1_dq, gelu_lut, fc2_w, fc2_dq, d):
    """The connector MLP: quantize -> fc1 (+GELU, +fc2's input quantize) -> fc2.

    Called from a prim_func body during tracing, so the shard layout is chosen in plain
    Python and only the taken form reaches the parser.
    """
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.quantize import quantize
    from ...ops.quantized_matmul_activation import quantized_matmul_activation_multicore

    h1 = quantized_matmul_activation_multicore(
        quantize(merged, qparams),
        fc1_w,
        fc1_dq,
        cores=d.fc1_cores,
        luts=gelu_lut,
        fuse_output_quant=True,
        split_output=d.fc1_split,
        output=BufferInfo(dtype="int8x64"),
    )
    return quantized_matmul_activation_multicore(
        h1,
        fc2_w,
        fc2_dq,
        cores=d.fc2_cores,
        matrix_a_is_split=d.fc1_split,
        output=BufferInfo(dtype="float32x64"),
    )


def _make_connector_plain(cfg: "Lfm2VlVisionConfig", cores: int):
    """The connector with no LayerNorm of its own (both released LFM2.5-VLs).

    post_layernorm at vision width -> PixelUnshuffle fold -> GELU MLP. The norm's beta is
    folded into the fc1 input quantizer's zero point, tiled over the fold's sub-tokens,
    so the norm carries no bias stream.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A

    from ...ops.layer_norm_dpu_reduce import layer_norm_dpu_reduce_mshard

    d = _connector_dims(cfg, cores)

    @A.ir_module
    class Lfm2VlConnector:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, d.seq, d.hidden_pw), _HIDDEN_DTYPE, scope=A.DDR),
            pnorm_w: A.Buffer((d.hidden_pw,), "float32x64", scope=A.DDR),  # D-major
            pnorm_lut: A.Buffer((d.norm_luts,), "float32x64", scope=A.DDR),
            # [scale, zp] per merged channel, the norm's beta folded into zp.
            mfc1_qparams: A.Buffer((2 * d.merged_pw,), "float32x64", scope=A.DDR),
            mfc1_w: A.Buffer((d.proj_pw, d.merged_dim), "int8x64", scope=A.DDR),  # NK
            # 4-wide [dz', ds, fc2_in_scale, fc2_in_zp] -- GELU + fc2's quantize fused.
            mfc1_dq: A.Buffer((1, 1, 4 * d.proj_pw), "float32x64", scope=A.DDR),
            mgelu_lut: A.Buffer((d.gelu_luts,), "float32x64", scope=A.DDR),
            mfc2_w: A.Buffer((d.out_pw, d.proj_dim), "int8x64", scope=A.DDR),
            mfc2_dq: A.Buffer((1, 1, 2 * d.out_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant([d.eps] * 64, dtype="float32x64", scope=A.DDR)
            normed = layer_norm_dpu_reduce_mshard(
                hidden_states,
                pnorm_lut,
                eps_const[0:1],
                normalized_shape=(d.hidden,),
                cores=d.ln_cores,
                weight_buffer=pnorm_w,
                invsqrt_method="additive",
            )
            merged = axol.atomic.reshape(normed, shape=(d.merged_seq, d.merged_pw))
            A.output(
                _connector_mlp(
                    merged, mfc1_qparams, mfc1_w, mfc1_dq, mgelu_lut, mfc2_w, mfc2_dq, d
                )
            )

    return Lfm2VlConnector


def _make_connector_normed(cfg: "Lfm2VlVisionConfig", cores: int):
    """The connector with its own LayerNorm at merged width.

    post_layernorm at vision width (its beta a bias buffer, since another norm follows)
    -> PixelUnshuffle fold -> LayerNorm over the whole merge_unit * hidden vector ->
    GELU MLP. That width would need too many resident registers for the single-pass
    dpu_reduce norm, so it takes the transpose-reduce Kahan one, whose gamma is
    broadcast-per-lane rather than D-major (see connector_buffers).
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A

    from ...ops.layer_norm_dpu_reduce import layer_norm_dpu_reduce_mshard

    d = _connector_dims(cfg, cores)

    @A.ir_module
    class Lfm2VlConnectorNormed:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, d.seq, d.hidden_pw), _HIDDEN_DTYPE, scope=A.DDR),
            pnorm_w: A.Buffer((d.hidden_pw,), "float32x64", scope=A.DDR),  # D-major
            pnorm_b: A.Buffer((d.hidden_pw,), "float32x64", scope=A.DDR),  # D-major
            pnorm_lut: A.Buffer((d.norm_luts,), "float32x64", scope=A.DDR),
            mnorm_w: A.Buffer((d.merged_dim,), "float32x64", scope=A.DDR),  # broadcast
            mnorm_lut: A.Buffer((d.norm_luts,), "float32x64", scope=A.DDR),
            # [scale, zp] per merged channel, the merged norm's beta folded into zp.
            mfc1_qparams: A.Buffer((2 * d.merged_pw,), "float32x64", scope=A.DDR),
            mfc1_w: A.Buffer((d.proj_pw, d.merged_dim), "int8x64", scope=A.DDR),  # NK
            mfc1_dq: A.Buffer((1, 1, 4 * d.proj_pw), "float32x64", scope=A.DDR),
            mgelu_lut: A.Buffer((d.gelu_luts,), "float32x64", scope=A.DDR),
            mfc2_w: A.Buffer((d.out_pw, d.proj_dim), "int8x64", scope=A.DDR),
            mfc2_dq: A.Buffer((1, 1, 2 * d.out_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant([d.eps] * 64, dtype="float32x64", scope=A.DDR)
            normed = layer_norm_dpu_reduce_mshard(
                hidden_states,
                pnorm_lut,
                eps_const[0:1],
                normalized_shape=(d.hidden,),
                cores=d.ln_cores,
                weight_buffer=pnorm_w,
                bias_buffer=pnorm_b,
                invsqrt_method="additive",
            )
            folded = axol.atomic.reshape(normed, shape=(d.merged_seq, d.merged_pw))
            merged = axol.core.layer_norm_two_pass_kahan(
                folded,
                mnorm_lut,
                eps_const[0:1],
                normalized_shape=(d.merged_dim,),
                weight_buffer=mnorm_w,
                invsqrt_method="additive",
            )
            A.output(
                _connector_mlp(
                    merged, mfc1_qparams, mfc1_w, mfc1_dq, mgelu_lut, mfc2_w, mfc2_dq, d
                )
            )

    return Lfm2VlConnectorNormed


def make_connector_kernel(cfg: "Lfm2VlVisionConfig", cores: int = 1):
    """The connector: post_layernorm -> PixelUnshuffle -> [LayerNorm] -> GELU MLP.

    The fold is free: the host handed the tower its patches in merge-block order, so
    folding ``merge_unit`` consecutive tokens into one ``merge_unit * hidden`` row is a
    pure reshape of the packed buffer (see ``lfm2vl.merge_permutation``).

    Which norms run is a property of the checkpoint, so the two forms are separate
    builders -- a prim_func's buffer list has to be static.
    """
    builder = _make_connector_normed if cfg.projector_use_layernorm else _make_connector_plain
    return builder(cfg, cores)


# --------------------------------------------------------------------------------------
# Parameter binding
# --------------------------------------------------------------------------------------


def _nk(weight: torch.Tensor) -> torch.Tensor:
    """[out, in] int8 -> NK (out//64, in, 64), the layout core.matmul(b_transposed) wants."""
    out, inn = weight.shape
    return weight.reshape(out // 64, 64, inn).permute(0, 2, 1).contiguous().to(torch.int8)


def _interleave(*channels) -> torch.Tensor:
    """Interleave per-channel vectors on the channel axis: [a0, b0, a1, b1, ...].

    Each is first laid out (C,) -> (C//64, 64): one per-channel value per PWORD lane.
    """
    packed = [torch.as_tensor(c).to(torch.float32).reshape(-1, 64).contiguous() for c in channels]
    stride = len(packed)
    out = torch.empty((stride * packed[0].shape[0], 64), dtype=torch.float32)
    for i, part in enumerate(packed):
        out[i::stride] = part
    return out


def _dequant_table(params: dict, bias: torch.Tensor):
    """[dz', ds] with the linear's bias folded into the dequant zero.

    ``(matmul - dz') * ds == dequant(matmul) + bias`` for ``dz' = dz - bias/ds``.
    """
    ds = params["ds"].to(torch.float32)
    return params["dz"].to(torch.float32) - bias.to(torch.float32) / ds, ds


def embed_buffers(model: "Lfm2VlVisionModel", spec: dict) -> dict:
    """Patch weight (NK int8), bias-folded dequant table, and the position constant."""
    # Export just the patch embed: the tower-wide export would int8-quantize every
    # block's weights again for one buffer.
    params = export_linear_quant_params(model.patch_embed)["proj"]
    hidden = int(model.config.hidden_size)
    hidden_pw = hidden // 64
    dz, ds = _dequant_table(params, model.patch_embed.bias.detach())
    seq_pad = spec["pos_const"][0][1]
    pos = torch.zeros(seq_pad, hidden)
    pos[: model.config.num_patches] = model.pos_embed.detach().to(torch.float32)
    return {
        "patch_w": pack(_nk(params["w"]), spec["patch_w"]),
        "patch_dq": pack(_interleave(dz, ds).reshape(1, 1, 2 * hidden_pw, 64), spec["patch_dq"]),
        "pos_const": pack(pos, spec["pos_const"]),
    }


def block_buffers(block, spec: dict) -> dict:
    """Bind one SigLIP2 encoder block: the shared ViT block binder, unchanged.

    Named here so the graph builder reads one module. The block is that same block, so
    its NK weights, bias-folded dequant tables, fused input quants and LUT tables are
    packed by the same code.
    """
    return optimized_block_buffers(block, spec)


def connector_buffers(connector, spec: dict) -> dict:
    """Bind the connector: the two norms, the merged-width input quant, and the MLP.

    Whichever norm runs LAST before the MLP has its beta folded into the fc1 input
    quantizer's zero point -- repeated over the fold's sub-tokens when that is the
    vision-width post_layernorm (its beta covers ``hidden``, the quantizer ``merge_unit *
    hidden``), and once when it is the connector's own merged-width norm. The other norm,
    if any, carries a bias buffer.
    """
    lin = export_linear_quant_params(connector)
    ranges = export_lut_ranges(connector)
    fc1, fc2 = lin["linear_fc1"], lin["linear_fc2"]
    merged_dim = int(connector.linear_fc1.in_features)

    buffers: dict = {}
    # post_layernorm: D-major gamma for the dpu_reduce norm (pack reshapes it).
    buffers["pnorm_w"] = pack(connector.norm.weight.detach(), spec["pnorm_w"])
    buffers["pnorm_lut"] = fit_lut(ranges["norm.inv_sqrt_range"], spec["pnorm_lut"])
    if "pnorm_b" in spec:
        buffers["pnorm_b"] = pack(connector.norm.bias.detach(), spec["pnorm_b"])
        beta = connector.merged_norm.bias.detach().to(torch.float32).reshape(-1)
        gamma = connector.merged_norm.weight.detach().to(torch.float32)
        # (dim, PWORD) spec == the Kahan norm's broadcast weight; (dim/64, PWORD) == D-major.
        if spec["mnorm_w"][0][0] == gamma.numel():
            gamma = broadcast_to_pword(gamma)
        buffers["mnorm_w"] = pack(gamma, spec["mnorm_w"])
        buffers["mnorm_lut"] = fit_lut(ranges["merged_norm.inv_sqrt_range"], spec["mnorm_lut"])
    else:
        beta = connector.norm.bias.detach().to(torch.float32).reshape(-1)

    qs = fc1["qs"].to(torch.float32).reshape(-1)
    qz = fc1["qz"].to(torch.float32).reshape(-1) + beta.repeat(merged_dim // beta.numel()) * qs
    buffers["mfc1_qparams"] = pack(_interleave(qs, qz), spec["mfc1_qparams"])

    dz1, ds1 = _dequant_table(fc1, connector.fc1_bias.detach())
    dz2, ds2 = _dequant_table(fc2, connector.fc2_bias.detach())
    buffers["mfc1_w"] = pack(_nk(fc1["w"]), spec["mfc1_w"])
    buffers["mfc1_dq"] = pack(_interleave(dz1, ds1, fc2["qs"], fc2["qz"]), spec["mfc1_dq"])
    buffers["mgelu_lut"] = fit_lut(ranges["gelu_range"], spec["mgelu_lut"])
    buffers["mfc2_w"] = pack(_nk(fc2["w"]), spec["mfc2_w"])
    buffers["mfc2_dq"] = pack(_interleave(dz2, ds2), spec["mfc2_dq"])
    return buffers


# --------------------------------------------------------------------------------------
# Graph authoring + device driver
# --------------------------------------------------------------------------------------


def compile_kernel(
    module,
    out_dir: Path,
    *,
    target=None,
    cores: int = 1,
    promote: bool = True,
    generation: str = DEFAULT_GENERATION,
):
    """Compile one alto module; returns (model.json, model.atex).

    ``generation`` ("OMEGA" | "EUROPA") changes the core count, the memory sizes and hence
    the schedule, so it must be the same for every kernel of one artifact.

    ``resources`` is pinned to this kernel's fair share of device memory
    (``num_cores / NUM_AI_CORES``) rather than the default 1.0: a VLM keeps TWO artifacts
    loaded on one device, so neither may budget as if it owned all of it. It is also the
    minimum the config validator accepts for a given core count.
    """
    from axelera.compiler.alto.compiler.config import AltoConfig, HardwareGeneration, Target
    from axelera.compiler.alto.compiler.device.hw_specs import NUM_AI_CORES
    from axelera.compiler.alto.compiler.driver import compile as alto_compile

    out_dir = Path(out_dir)
    hw = HardwareGeneration[str(generation).upper()]
    alto_compile(
        module,
        AltoConfig(
            output_dir=out_dir,
            generation=hw,
            target=Target.AICORE_SIM if target is None else target,
            num_cores=cores,
            resources=max(cores, 1) / NUM_AI_CORES[hw],
            enable_tiling_injection=True,
            enable_buffer_promotion=promote,
            enable_llm_support=True,
            assign_unique_input_pools=True,
            unpack_axm_archive=True,
            clean_output_dir=True,
            emit_atex_artifacts=True,
            emit_alto_config=True,
        ),
    )
    model_jsons = sorted(out_dir.rglob("model.json"))
    atexes = sorted(out_dir.rglob("model.atex"))
    if not model_jsons or not atexes:
        raise RuntimeError(f"compile produced no model.json/model.atex under {out_dir}")
    return model_jsons[0], atexes[0]


def build_vision_graph(
    model: "Lfm2VlVisionModel",
    out_dir,
    *,
    target=None,
    cores=1,
    promote=True,
    generation: str = DEFAULT_GENERATION,
):
    """Author the whole tower as one graph: embed -> depth x block -> connector.

    The activation threads inputs_embeds -> hidden_a -> hidden_b -> ... -> the image
    embeddings, ping-ponging the two spine slots so depth costs 2 live buffers rather than
    one per block; the padding mask is a shared input to the blocks; every kernel's constants
    are bound as its own outlined weights. ``generation`` goes to every kernel compile and to
    the graph renderer. Returns (out_dir, specs).
    """
    from ...graph_descriptor import KindContract, author_graph_model

    out_dir = Path(out_dir)
    cfg = model.config

    def _compile(name, module):
        model_json, atex = compile_kernel(
            module,
            out_dir / f"_{name}",
            target=target,
            cores=cores,
            promote=promote,
            generation=generation,
        )
        return atex, extract_layer_buffer_spec(model_json)

    embed_atex, embed_spec = _compile("embed", make_vision_embed_kernel(cfg, cores))
    block_atex, block_spec = _compile("block", make_vision_block_kernel(cfg, cores))
    merge_atex, merge_spec = _compile("connector", make_connector_kernel(cfg, cores))
    kernel_atex = {
        ("encode", "embed"): embed_atex,
        ("encode", "block"): block_atex,
        ("encode", "connector"): merge_atex,
    }
    contracts = {
        "embed": KindContract(shared_inputs=frozenset({"patches"}), state=frozenset()),
        "block": KindContract(shared_inputs=frozenset({"attn_mask"}), state=frozenset()),
        "connector": KindContract(shared_inputs=frozenset(), state=frozenset()),
    }

    kinds = ["embed"] + ["block"] * cfg.depth + ["connector"]
    buffers = {0: embed_buffers(model, embed_spec)}
    for i, block in enumerate(model.blocks):
        buffers[1 + i] = block_buffers(block, block_spec)
    buffers[len(kinds) - 1] = connector_buffers(model.connector, merge_spec)

    author_graph_model(
        out_dir,
        "lfm2vl_vision",
        kernel_atex,
        kinds,
        contracts,
        buffers,
        threaded_inputs=frozenset({"hidden_states"}),
        generation=str(generation).lower(),
    )
    # The few scalars a run needs that are not in the graph: the geometry, the host
    # pixel-quant scale, the mask shape and the generation this was built for. With these
    # the artifact reloads and runs with no torch model -- see Lfm2VlVisionDevice.load.
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "patch_scale": float(
                    model.patch_embed.proj.act_observer.qparams()[0].reshape(-1)[0]
                ),
                "mask_shape": list(block_spec["attn_mask"][0]),
                "generation": str(generation).upper(),
                "cores": int(cores),
            },
            indent=2,
        )
    )
    return out_dir, {"embed": embed_spec, "block": block_spec, "connector": merge_spec}


class Lfm2VlVisionDevice:
    """The compiled vision tower: flattened patches -> image embeddings.

    ``encode`` is one device dispatch per image. The host does the merge-block
    permutation and the int8 patch quantize (one calibrated symmetric scale, the same the
    patch-embed matmul was calibrated with), and slices the padded rows off the output.
    """

    def _bind(self, graph_dir, config, patch_scale, mask_shape, generation, share_from=None):
        """Everything both construction paths set: geometry, padding mask, loaded runtime.

        The connection claims one sub-device per core the graph places work on, which
        RuntimeModel reads off the artifact -- unless it is sharing one, the first tower
        loaded having claimed them already.
        """
        from ...generate import RuntimeModel

        from .lfm2vl import merge_permutation

        self.graph_dir = Path(graph_dir)
        self.config = config
        self.generation = str(generation).upper()
        self.patch_scale = float(patch_scale)
        self.mask_shape = tuple(mask_shape)
        self._mask = self._padding_mask()
        self._order = merge_permutation(
            config.grid_h, config.grid_w, config.downsample_factor
        ).numpy()
        self.runtime = RuntimeModel(self.graph_dir / "model.json", share_from=share_from)

    def __init__(
        self,
        model: "Lfm2VlVisionModel",
        out_dir,
        *,
        target=None,
        cores=1,
        generation=DEFAULT_GENERATION,
    ):
        graph_dir, specs = build_vision_graph(
            model, out_dir, target=target, cores=cores, generation=generation
        )
        self._bind(
            graph_dir,
            model.config,
            model.patch_embed.proj.act_observer.qparams()[0].reshape(-1)[0],
            specs["block"]["attn_mask"][0],
            generation,
        )

    @classmethod
    def load(cls, graph_dir, *, share_from=None, generation=DEFAULT_GENERATION):
        """Reload a compiled tower (graph + meta.json) with no torch model or calibration.

        The core count comes off meta.json -- what the tower was compiled for is what the
        connection claims -- so a run needs no flag to repeat it.
        """
        from .lfm2vl import Lfm2VlVisionConfig

        meta = json.loads((Path(graph_dir) / "meta.json").read_text())
        self = cls.__new__(cls)
        self._bind(
            graph_dir,
            Lfm2VlVisionConfig(**meta["config"]),
            meta["patch_scale"],
            meta["mask_shape"],
            generation,
            share_from,
        )
        return self

    def _padding_mask(self) -> np.ndarray:
        """int8 1-head-broadcast mask: 0 for real patches, -127 for padding."""
        shape = self.mask_shape  # (1, 1, seq, ctx_blocks, 64)
        keys = np.arange(shape[-2] * 64).reshape(shape[-2], 64)
        row = np.where(keys < self.config.num_patches, 0, _MASK_NEG).astype(np.int8)
        return np.broadcast_to(row, shape).copy()

    def quantize_patches(self, patches) -> np.ndarray:
        """One image's patches (processor order) -> (1, num_patches, patch_dim/64, 64) int8.

        The merge-block permutation happens here, which is what makes the connector's
        PixelUnshuffle a reshape on device (see lfm2vl.merge_permutation).
        """
        cfg = self.config
        values = np.asarray(patches, dtype=np.float32)
        if values.shape != (cfg.num_patches, cfg.patch_dim):
            raise ValueError(
                f"expected {(cfg.num_patches, cfg.patch_dim)} patches for the compiled "
                f"{cfg.grid_h}x{cfg.grid_w} grid; got {tuple(values.shape)}"
            )
        ordered = values[self._order]
        quantized = np.clip(np.round(ordered / self.patch_scale), -127, 127).astype(np.int8)
        return quantized.reshape(1, cfg.num_patches, cfg.patch_dim // 64, 64)

    def encode(self, patches) -> np.ndarray:
        """Patches -> ``(n_images * num_merged, out_hidden)`` image embeddings.

        The processor concatenates every image's patches, so N images arrive as
        ``N * num_patches`` rows; each is one dispatch (the graph is compiled for one
        image's token count, and HF encodes and unpads them separately anyway), and the
        results concatenate in image order -- which is the order the decoder's
        placeholder rows expect.
        """
        cfg = self.config
        rows = np.asarray(patches)
        if rows.shape[0] % cfg.num_patches:
            raise ValueError(
                f"{rows.shape[0]} patch rows is not a whole number of "
                f"{cfg.num_patches}-patch images"
            )
        per_image = []
        # (dispatches, seconds) of the last encode -- the tower's own cost, separate from
        # whatever host work the caller wraps it in. Read by the demo's RunStats.
        self.last_encode = (0, 0.0)
        started = perf_counter()
        for start in range(0, rows.shape[0], cfg.num_patches):
            out = self.runtime.run(
                "encode",
                {
                    "patches": self.quantize_patches(rows[start : start + cfg.num_patches]),
                    "attn_mask": self._mask,
                },
            )
            out = out[0] if isinstance(out, list) else out
            per_image.append(np.asarray(out).reshape(-1, cfg.out_hidden_size)[: cfg.num_merged])
        self.last_encode = (len(per_image), perf_counter() - started)
        return np.concatenate(per_image, axis=0)
