# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3 decoder kernel of one or more layers, generated from its signature table.

TVMScript reads a prim_func's parameter list from source, so ``main`` is generated: the
shared inputs once, then every per-layer buffer once per sub-layer with the ``sublayer_tag``
suffix (none for a one-layer kernel), which graph_descriptor binds to consecutive layers.
Sub-layer j+1 reads sub-layer j's output as both hidden_states and residual_in.
"""

from ...graph_descriptor import sublayer_tag
from ..multicore_layer import PWORD

#: Rows of each SDPA LUT buffer (``sdpa_exp_lut``, ``sdpa_inv_lut``).
SDPA_LUT_ROWS = 2


def default_layers_per_kernel(num_layers: int) -> int:
    """Layers per decode kernel for the fused multicore schedule: 4, or the largest smaller
    divisor of the layer count (a call must run whole groups)."""
    return next(n for n in (4, 3, 2, 1) if num_layers % n == 0)


def shared_params(d) -> tuple:
    """The five inputs a kernel declares once: (name, shape, dtype, scope expression)."""
    return (
        ("seq_pos", (1,), "uint64x64", "A.DDR"),
        ("hidden_states", (1, d.seq_len, d.hidden_dim_pw), "float32x64", "A.DDR"),
        ("residual_in", (1, d.seq_len, d.hidden_dim_pw), "float32x64", "A.DDR"),
        ("rope_trig", (1, d.seq_len, d.head_dim_pw), "float32x64", "A.DDR"),
        ("attn_mask", (1, 1, d.seq_len, d.context_len // PWORD), "int8x64", "A.DDR"),
    )


def layer_params(d, *, sdpa_packed_rows=None) -> tuple:
    """Every per-layer buffer of the kernel signature, in declaration order, the DDR1 shards
    last (the runtime binds inputs pool-grouped). Each weight's lower K half is the DDR
    shard, the upper the ``<name>1`` shard in DDR1, so both regions stream at once;
    ``sdpa_packed_rows`` adds the band SDPA's packed constant buffer."""
    qkv_k_lo, o_k_lo = d.hidden_dim // 2, d.q_out_dim // 2
    gate_up_k_lo, down_k_lo = d.hidden_dim // 2, d.intermediate_dim // 2
    qkv_out_dim_pw = d.q_out_dim_pw + 2 * d.kv_out_dim_pw
    cache = (1, d.num_kv_heads, d.context_len, d.head_dim_pw)
    params = (
        # [inv-sqrt LUT rows, 1/D, eps] in one buffer: see pack_norm_consts.
        ("rms1_consts", (4,), "float32x64", "A.DDR"),
        ("q_proj_qparams", (2 * d.hidden_dim_pw,), "float32x64", "A.DDR"),
        # q||k||v stacked on N, so a core's band is one contiguous weight slab.
        ("qkv_proj_w", (qkv_out_dim_pw, qkv_k_lo), "int8x64", "A.DDR"),
        ("q_proj_dq", (1, 1, 2 * d.q_out_dim_pw), "float32x64", "A.DDR"),
        ("k_proj_dq", (1, 1, 2 * d.kv_out_dim_pw), "float32x64", "A.DDR"),
        ("v_proj_dq", (1, 1, 2 * d.kv_out_dim_pw), "float32x64", "A.DDR"),
        # The per-head norm weights D-major (dpu_reduce).
        ("q_norm_w", (d.head_dim_pw,), "float32x64", "A.DDR"),
        ("q_norm_consts", (4,), "float32x64", "A.DDR"),
        ("k_norm_w", (d.head_dim_pw,), "float32x64", "A.DDR"),
        ("k_norm_consts", (4,), "float32x64", "A.DDR"),
        ("k_cache_qparams", (2 * d.head_dim_pw,), "float32x64", "A.DDR"),
        ("v_qparams", (2 * d.head_dim_pw,), "float16x64", "A.DDR"),
        ("k_out", cache, "int8x64", "A.DDR"),
        ("q_attn_qparams", (2 * d.head_dim_pw,), "float32x64", "A.DDR"),
        ("sdpa_exp_lut", (SDPA_LUT_ROWS,), "float32x64", "A.DDR"),
        ("sdpa_inv_lut", (SDPA_LUT_ROWS,), "float32x64", "A.DDR"),
        ("sdpa_scores_dq_scale", (1, 1, 1, 1), "float32x64", "A.DDR"),
        ("sdpa_weights_q_scale", (1, 1, 1, 1), "float32x64", "A.DDR"),
        ("sdpa_output_dq_scale", (1, 1, 1, 1), "float32x64", "A.DDR"),
        ("o_qparams", (2 * d.q_out_dim_pw,), "float16x64", "A.DDR"),
        ("o_proj_w", (d.hidden_dim_pw, o_k_lo), "int8x64", "A.DDR"),
        ("o_proj_dq", (1, 1, 2 * d.hidden_dim_pw), "float32x64", "A.DDR"),
        ("rms2_consts", (4,), "float32x64", "A.DDR"),
        ("gate_proj_qparams", (2 * d.hidden_dim_pw,), "float32x64", "A.DDR"),
        ("silu_lut", (2,), "float32x64", "A.DDR"),
        ("gate_up_proj_w", (2 * d.intermediate_dim_pw, gate_up_k_lo), "int8x64", "A.DDR"),
        ("mlp_dq_params", (1, 1, 6 * d.intermediate_dim_pw), "float32x64", "A.DDR"),
        ("down_proj_w", (d.hidden_dim_pw, down_k_lo), "int8x64", "A.DDR"),
        ("down_dq_params", (1, 1, 2 * d.hidden_dim_pw), "float32x64", "A.DDR"),
        (
            "gate_up_proj_w1",
            (2 * d.intermediate_dim_pw, d.hidden_dim - gate_up_k_lo),
            "int8x64",
            "_ddr1",
        ),
        ("down_proj_w1", (d.hidden_dim_pw, d.intermediate_dim - down_k_lo), "int8x64", "_ddr1"),
        ("qkv_proj_w1", (qkv_out_dim_pw, d.hidden_dim - qkv_k_lo), "int8x64", "_ddr1"),
        ("o_proj_w1", (d.hidden_dim_pw, d.q_out_dim - o_k_lo), "int8x64", "_ddr1"),
        # With the weight shards balanced across regions, the KV cache evens them out.
        ("v_out", cache, "int8x64", "_ddr1"),
    )
    if sdpa_packed_rows is not None:
        params += (("sdpa_packed", (sdpa_packed_rows,), "float32x64", "A.DDR"),)
    return params


def grouped_module(n, layer, shared, layer_params, env, chain=False):
    """The IRModule of ``n`` decoder layers in one kernel.

    ``layer(hidden_states, residual_in, seq_pos, attn_mask, rope_trig, **per_layer)`` traces
    one sub-layer and returns layer_out. With ``chain`` (the fused decode schedule)
    sub-layer j's region tail computes sub-layer j+1's qkv: ``layer`` also takes
    ``next_layer`` (sub-layer j+1's buffers) and returns that layer's q5/k5/v5 after
    layer_out, which the next call takes as ``prev_qkv``. ``env`` supplies the names the
    declarations' scope expressions use (``A``, ``_ddr1``).
    """
    from tvm.runtime import String
    from tvm.script.parser import parse
    from tvm.script.parser.core.entry import _default_globals

    def decl(name, shape, dtype, scope):
        return f"        {name}: A.Buffer({tuple(shape)!r}, {dtype!r}, scope={scope}),\n"

    def kwargs(tag):
        return "".join(f"            {name}={name}{tag},\n" for name, *_ in layer_params)

    # A one-layer kernel keeps the untagged names.
    tag = (lambda j: "") if n == 1 else sublayer_tag
    src = "@A.ir_module\nclass QwenLayerFunctional:\n    @A.prim_func\n    def main(\n"
    for name, shape, dtype, scope in shared:
        src += decl(name, shape, dtype, scope)
    for j in range(n):
        for name, shape, dtype, scope in layer_params:
            src += decl(name + tag(j), shape, dtype, scope)
    src += "    ):\n"
    for j in range(n):
        t, t1 = tag(j), tag(j + 1)
        prev = f"layer_out{tag(j - 1)}"
        spine = ["hidden_states", "residual_in"] if j == 0 else [prev, prev]
        chained = chain and j < n - 1
        targets = [f"layer_out{t}"] + ([f"q5{t1}", f"k5{t1}", f"v5{t1}"] if chained else [])
        src += f"        {', '.join(targets)} = _layer({', '.join(spine)}, seq_pos, attn_mask, rope_trig,\n"
        src += kwargs(t)
        if chained:
            src += f"            next_layer=dict(\n{kwargs(t1)}            ),\n"
        if chain and j > 0:
            src += f"            prev_qkv=(q5{t}, k5{t}, v5{t}),\n"
        src += "        )\n"
    src += f"        A.output(layer_out{tag(n - 1)})\n"

    mod = parse(src, {**_default_globals(), **env, "dict": dict, "_layer": layer})
    mod = mod.with_attr("dialect", String("axelerascript"))
    # What A.ir_module sets on a class-defined module.
    mod.__name__ = "QwenLayerFunctional"
    return mod
