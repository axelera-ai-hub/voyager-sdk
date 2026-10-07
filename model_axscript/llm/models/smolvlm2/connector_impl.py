# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM2 pixel-shuffle connector: host space-to-depth + one device matmul.

The HF ``SmolVLMConnector`` = ``pixel_shuffle`` (space-to-depth by ``scale_factor``,
folding each sf*sf spatial block into the channel dim: embed -> embed*sf^2) followed
by a single bias-free projection Linear into the text hidden size
(``SmolVLMSimpleMLP``, no bias, no activation).

Placement: the shuffle is a pure reindex, but on-device it would need a transpose/reshape
kind whose lane mapping is a device concern, so the shuffle + input quantize stay on the
host and only the projection is a device matmul. That host reshape sits between two
dispatches, so the connector is its OWN graph rather than threaded onto the vision one.
Revisit if the host reshape + requantize shows up as a latency floor.

The projection input dim is ``vision_hidden * sf^2``, a multiple of 64 on every
shipped checkpoint: 768*4^2 = 12288 (256M + 500M) and 1152*3^2 = 10368 (2.2B, SigLIP
SO400M with scale_factor 3). The output (text hidden) is padded up to a PWORD multiple
with zero weight rows (dequant to 0); the host slices back to text_hidden.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from math import ceil
from typing import TYPE_CHECKING

from ...export import export_linear_quant_params

PWORD = 64

if TYPE_CHECKING:
    from .smolvlm2 import SmolVLM2Config, SmolVLM2Connector


def _dims(config: "SmolVLM2Config") -> dict:
    sf = int(config.scale_factor)
    in_dim = int(config.vision_config.hidden_size) * sf * sf
    out_dim = int(config.text_config.hidden_size)
    out_pad = ceil(out_dim / PWORD) * PWORD
    if in_dim % PWORD != 0:
        # Every shipped checkpoint lands on a multiple of 64 (12288, 12288, 10368). A config
        # that breaks this needs the K axis zero-padded on host and weight alike (mirror the
        # fc2 input padding in vision_impl); an explicit failure until then.
        raise ValueError(f"connector in_dim {in_dim} must be a multiple of {PWORD}")
    return {
        "sf": sf,
        "in_dim": in_dim,
        "in_pw": in_dim // PWORD,
        "out_dim": out_dim,
        "out_pad": out_pad,
        "out_pw": out_pad // PWORD,
    }


def make_connector_kernel(config: "SmolVLM2Config", num_tokens: int):
    """Build the connector IRModule: one bias-free int8 matmul from the (host)
    pixel-shuffled vision features into the (padded) text hidden size. Single-core --
    see build_connector_graph for why sharding one 64-81 row matmul is not worth it.

    The threaded input ``hidden_states`` is the host pixel-shuffled + int8-quantized
    vision features (1, num_tokens, in_pw); the graph mechanism renames it to the
    graph input ``inputs_embeds`` (layer 0). Output is fp (1, num_tokens, out_pw);
    the host slices [:, :, :text_hidden]."""
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.quantized_matmul_activation import quantized_matmul_activation

    d = _dims(config)
    in_dim, in_pw, out_pw = d["in_dim"], d["in_pw"], d["out_pw"]

    @A.ir_module
    class SmolVLM2ConnectorKernel:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, num_tokens, in_pw), "int8x64", scope=A.DDR),
            proj_w: A.Buffer((out_pw, in_dim), "int8x64", scope=A.DDR),  # NK (N_BLOCKS, K)
            proj_dq: A.Buffer((1, 1, 2 * out_pw), "float32x64", scope=A.DDR),  # [dz', ds]
        ):  # noqa: F821
            # Bias-free projection: (matmul - dz)*ds. No LUT activation. dz is the plain
            # quant-zero-point dequant term (qz * sum_j w); there is no bias to fold.
            A.output(
                quantized_matmul_activation(
                    hidden_states, proj_w, proj_dq, output=BufferInfo(dtype="float32x64")
                )
            )

    return SmolVLM2ConnectorKernel


def connector_buffers(
    connector: "SmolVLM2Connector", config: "SmolVLM2Config", spec: dict
) -> dict:
    """Pack the connector matmul constants from the calibrated SmolVLM2Connector: the
    bias-free projection weight in NK layout (int8, out padded to a PWORD multiple)
    and the interleaved [dz', ds] dequant table (no bias fold)."""
    import torch

    lin = export_linear_quant_params(connector)
    p = lin["proj"]
    d = _dims(config)
    in_dim, out_pad, out_pw = d["in_dim"], d["out_pad"], d["out_pw"]

    def blocks(v):
        return torch.as_tensor(v).to(torch.float32).reshape(-1, PWORD).contiguous()

    # NK weight: (out, in) int8, out padded up to out_pad with zero rows (dequant to 0).
    w = p["w"].to(torch.int8)  # (out_dim, in_dim)
    wpad = torch.zeros(out_pad, in_dim, dtype=torch.int8)
    wpad[: w.shape[0]] = w
    nk = wpad.reshape(out_pw, PWORD, in_dim).permute(0, 2, 1).contiguous()

    # [dz', ds]: bias-free, so dz' == dz (qz * sum_j w). Pad rows: dz 0, ds arbitrary (matmul==0).
    ds = p["ds"].to(torch.float32)
    dz = p["dz"].to(torch.float32)
    ds_p = torch.ones(out_pad, dtype=torch.float32)
    ds_p[: ds.numel()] = ds
    dz_p = torch.zeros(out_pad, dtype=torch.float32)
    dz_p[: dz.numel()] = dz
    dq = torch.empty((2 * out_pw, PWORD), dtype=torch.float32)
    dq[0::2], dq[1::2] = blocks(dz_p), blocks(ds_p)

    return {"proj_w": nk, "proj_dq": dq.reshape(1, 1, 2 * out_pw, PWORD).contiguous()}


def pixel_shuffle_np(x, scale_factor: int):
    """numpy space-to-depth by ``scale_factor`` == smolvlm2.pixel_shuffle (same
    reindex). (bsz, h*w, embed) -> (bsz, h*w/sf^2, embed*sf^2).

    Requires a square patch grid whose side divides by scale_factor (the HF processor pads
    tiles to satisfy this) and only real patch rows -- the vision graph's pad rows are
    sliced off before this call.
    """
    import numpy as np

    bsz, seq, embed = x.shape
    height = width = int(round(seq**0.5))
    if height * width != seq:
        raise ValueError(f"pixel_shuffle expects a square patch grid; got seq={seq}")
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError(f"grid {height}x{width} not divisible by scale_factor {scale_factor}")
    x = np.asarray(x)
    x = x.reshape(bsz, height, width, embed)
    x = x.reshape(bsz, height, width // scale_factor, embed * scale_factor)
    x = np.transpose(x, (0, 2, 1, 3))
    x = x.reshape(bsz, width // scale_factor, height // scale_factor, embed * scale_factor**2)
    x = np.transpose(x, (0, 2, 1, 3))
    x = x.reshape(bsz, seq // (scale_factor**2), embed * (scale_factor**2))
    return np.ascontiguousarray(x)


def quantize_shuffled(shuffled, qs, qz, qmin: int = -128, qmax: int = 127):
    """Host int8 quantize of the pixel-shuffled features for the connector matmul:
    q = clamp(round(x*qs + qz), qmin, qmax).

    ``qs`` MUST be the EXPORTED reciprocal scale ``1/(smooth * act_scale)``, per-in-channel
    (export.export_linear_quant_params), NOT the raw ``act_observer.qparams()[0]``: multiplying
    by that divide-convention scale collapses everything onto the zero-point (dz then zeroes
    the output), and dividing by it drops the SmoothQuant fold (rel-L2 4.55). The patch-embed
    may divide by its raw scale only because it is ungrouped, so its smooth is 1.
    Returns (1, num_tokens, in_pw, 64) int8 numpy."""
    import numpy as np

    x = np.asarray(shuffled, dtype=np.float32)
    bsz, num_tokens, in_dim = x.shape
    qs = np.asarray(qs, dtype=np.float32).reshape(-1)  # per-in-channel (or scalar)
    q = np.clip(np.round(x * qs + float(qz)), qmin, qmax).astype(np.int8)
    in_pw = ceil(in_dim / PWORD)
    data = np.zeros((bsz, num_tokens, in_pw * PWORD), np.int8)
    data[..., :in_dim] = q
    return data.reshape(bsz, num_tokens, in_pw, PWORD)
