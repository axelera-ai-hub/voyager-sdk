# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Op-level parallel projection group: run several INDEPENDENT quantized matmuls that share one
input CONCURRENTLY, one whole matmul per AI core.

Motivation (DINOv2 q/k/v @518, seq 1408):
    Sharding a SINGLE projection across cores is flat (~1.1x). A projection here is fixed-overhead
    / weight-load bound -- ~30us of actual int8 compute vs ~410us observed -- so splitting its M
    (rows) or N (cols) just replicates the weight load / IMC program per core and buys nothing.
    But q, k, v are three DIFFERENT matmuls over the SAME input (the ln1 int8 output) with distinct
    weights and outputs: genuinely independent work. Pinning one whole projection to each core
    overlaps them, so the group drops from ~3x a single matmul (sequential today) toward ~1x, with
    NO weight replication. Measured qkv matmul WORK: 617us sequential -> ~11us over 3 cores.

Structure mirrors quantized_matmul_activation_multicore's AICore/barrier bracketing, but instead of
one matmul N-sharded across cores, each core runs a DISTINCT (weight, dequant) on the full input and
writes its OWN full output (no N-shard, no cross-core reduction). Outputs are allocated in the shared
(outer) scope so downstream ops -- on any core -- can read them (e.g. the head transposes / SDPA).
"""

# flake8: noqa: F821

from axelera.compiler.alto import script as A

from .quantized_matmul_activation import (
    QuantizedMatMulActivation,
    _b_l2_stage_max_bytes,
    quantized_matmul_activation,
)


def parallel_matmul_group(
    matrix_a,
    weights,
    dequants,
    *,
    cores,
    fuse_output_quant=False,
    output_dtype="int8x64",
    reshape_shape=None,
    transpose_dims=None,
    **kwargs,
):
    """Run ``len(weights)`` independent ``quantized_matmul_activation`` ops -- shared ``matrix_a``,
    distinct ``weights[i]`` / ``dequants[i]`` -- one per core in ``cores``, concurrently. Returns a
    list of per-projection output buffers, allocated in the shared scope so any core can consume them.

    The projections need NOT share an output width: each one's shape is inferred from its own
    weight. That is what lets a GQA attention block group q with k/v (q is num_q_heads wide, k/v
    num_kv_heads) rather than only same-shape triples like a ViT's.

    Nor need they share an output DTYPE or fuse setting: ``output_dtype`` and
    ``fuse_output_quant`` each take either one value for the group or one per projection.
    That is what lets a *vision* block group q/k/v, where q and k must reach RoPE in float
    (fp16, unfused) while v's SDPA-input quantize is fused into its write (int8). Before
    this the group forced one dtype on all three, which is precisely why the Qwen3-VL
    vision q/k/v had to stay sequential -- and sequential meant single-core.

    Optional per-core epilogue: if ``reshape_shape`` and ``transpose_dims`` are given, each core also
    reshapes its matmul output to ``reshape_shape`` and transposes it by ``transpose_dims`` -- INSIDE
    its own AICore region -- so the transpose parallelises too (and stays local to the core that
    produced the data). The transposed result is written into a shared, pre-allocated buffer so the
    (head-sharded) consumer on any core can read it. Returned buffers are the transposed ones.
    Either may be given per projection (a list/tuple of one shape per weight) when the group's
    outputs reshape differently -- again, the GQA case.

    Falls back to a plain sequential emission (single core) when ``cores`` has <= 1 entry, so callers
    can gate purely on the core count. All ``kwargs`` (e.g. ``luts``) pass through to each matmul.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto.axol.common import BufferInfo

    n = len(weights)
    assert len(dequants) == n, "weights and dequants must be the same length"
    do_transpose = reshape_shape is not None and transpose_dims is not None

    def _per_projection(value, what):
        """Broadcast one shared value over the group, or take one entry per projection."""
        if value is None or not isinstance(value[0], (list, tuple)):
            return [value] * n
        assert len(value) == n, f"{what}: {len(value)} entries for {n} projections"
        return [tuple(v) for v in value]

    def _per_scalar(value, what):
        """Same, for values that are themselves scalars (a dtype string, a bool).

        A list/tuple means one entry per projection; anything else -- including a string,
        which is a sequence but not a per-projection list -- is the whole group's value.
        """
        if isinstance(value, (list, tuple)):
            assert len(value) == n, f"{what}: {len(value)} entries for {n} projections"
            return list(value)
        return [value] * n

    reshapes = _per_projection(reshape_shape, "reshape_shape")
    transposes = _per_projection(transpose_dims, "transpose_dims")
    out_dtypes = _per_scalar(output_dtype, "output_dtype")
    fuse_quants = _per_scalar(fuse_output_quant, "fuse_output_quant")

    core_ids = tuple(int(c) for c in cores) if cores is not None else ()
    parallel = len(core_ids) > 1
    if parallel:
        assert len(core_ids) == n, (
            f"parallel_matmul_group needs one core per projection: {n} projections, "
            f"{len(core_ids)} cores"
        )

    # Infer each projection's matmul output shape/scope from its OWN weight, so the group can mix
    # widths. Honor the requested output dtype (the op infers float16x64 otherwise).
    infos = [
        QuantizedMatMulActivation(
            matrix_a,
            weights[i],
            dequants[i],
            output=BufferInfo(dtype=out_dtypes[i]),
            fuse_output_quant=fuse_quants[i],
            **kwargs,
        ).infer_outputs()["output"]
        for i in range(n)
    ]

    # Each core loads a DISTINCT weight (no replication), so the L2-stage budget is a per-weight
    # check (not the aggregate). Stage iff one weight fits.
    stage_l2 = [
        # int8x64: 64 B/elem
        int(w.shape[-2]) * int(w.shape[-1]) * 64 <= _b_l2_stage_max_bytes()
        for w in weights
    ]

    # Allocate every output up front in the shared (outer) scope -- NOT inside an AICore region --
    # so cross-core consumers can read them. `outs` hold the matmul results; `finals` (if transposing)
    # hold the transposed results (the shape is reshape_shape permuted by transpose_dims).
    outs = [
        A.alloc_buffer(tuple(int(s) for s in info.shape), dtype=out_dtypes[i], scope=info.scope)
        for i, info in enumerate(infos)
    ]
    if do_transpose:
        finals = [
            A.alloc_buffer(
                tuple(int(reshapes[i][d]) for d in transposes[i]),
                dtype=out_dtypes[i],
                scope=infos[i].scope,
            )
            for i in range(n)
        ]

    def _emit(i, core):
        quantized_matmul_activation(
            matrix_a,
            weights[i],
            dequants[i],
            output=outs[i],
            core=core,
            fuse_output_quant=fuse_quants[i],
            stage_weight_l2=stage_l2[i],
            **kwargs,
        )
        if do_transpose:
            x = axol.atomic.reshape(outs[i], shape=reshapes[i])
            axol.core.transpose(x, dims=transposes[i], output=finals[i], core=core)

    if parallel:
        for i in range(n):
            with A.AICore(core=core_ids[i]):
                A.barrier(*core_ids)
                _emit(i, core_ids[i])
                A.barrier(*core_ids)
    else:
        for i in range(n):
            _emit(i, None)
    return finals if do_transpose else outs
