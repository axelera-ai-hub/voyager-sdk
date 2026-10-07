#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Device perf harness: llm.sdpa_flash vs axol.core.scaled_dot_product_attention.

sdpa_flash tiles the KV (and optionally seq) axis so L1/IMC footprint is
independent of context length; the core op keeps the full (seq, context)
score row resident per head. For shapes that fit the core op's L1 budget,
that extra generality is pure overhead -- this harness quantifies it by
building both ops on the SAME (query, key, value, causal mask, quant/LUT)
inputs and timing each on real hardware, sweeping context length.

llm.sdpa_flash and axol.core.scaled_dot_product_attention share one positional
call signature (query, key, value, scores_dq_scale, exp_lut, inv_lut,
weights_q_scale, weights_q_zp, weights_q_qmin, weights_q_qmax, output_dq_scale,
query_zp, value_zp, mask) -- see ``_build_module``. llm.sdpa_flash_europa drops
exp_lut/inv_lut (it computes exp/1/x with EXP2/RECIP hardware instructions
instead of LUT chains), so it gets its own trimmed builder, ``_build_europa_module``.
Both builders draw from the same ``_sdpa_constants``, so all three ops still see
identical quant scales on identical (query, key, value, mask) inputs.

THIS TOOL RUNS IN THE COMPILER CHECKOUT ONLY, same as layer_benchmark.py: it
reaches ``utils.compile_and_run`` (via ``measure_module``), which lives beside
the ALTO integration tests and is not part of the model tree. Run it there:

    source build-Release/workspace/chroot/bin/axelera-activate.sh
    cd host/tvm/axelera_compiler
    PYTHONPATH="$PWD:$PYTHONPATH" python -m tests.integration.alto.llm.ops._bench_sdpa_flash \\
        --seq-len 128 --ctx-lens 512 1024 2048 4096
"""

import argparse
import logging
import tempfile
from functools import partial
from pathlib import Path

import axelera.compiler.alto.script as A
import numpy as np
from axelera.compiler.alto import axol
from axelera.compiler.alto.axol.utils.lut_chain import (
    ChainedLUTMethod,
    compute_chained_lut_params,
    fit_exp_lut_params_sdpa_softmax,
    fit_inv_lut_params,
    pack_lut_params_to_pword,
)
from axelera.compiler.alto.compiler import interpret
from axelera.compiler.alto.compiler.config import HardwareGeneration

from ..evaluation.layer_benchmark import device_sanity_check, measure_module
from .sdpa_flash import sdpa_flash
from .sdpa_flash_europa import sdpa_flash_europa

# mypy: disable-error-code="attr-defined, index, arg-type"

logger = logging.getLogger(__name__)

PWORD = 64


def _causal_mask(batch: int, heads: int, seq_len: int, ctx_len: int) -> np.ndarray:
    """int8 additive causal mask (0 visible / -127 masked), PWORD-padded ctx.

    Queries are the LAST seq_len positions of ctx_len, matching both ops'
    chunked-prefill convention (cf. _default_causal_mask in the op tests).
    """
    ctx_padded = ((ctx_len + PWORD - 1) // PWORD) * PWORD
    query_pos = np.arange(ctx_len - seq_len, ctx_len)[:, None]
    key_pos = np.arange(ctx_padded)[None, :]
    masked = (key_pos > query_pos) | (key_pos >= ctx_len)
    return (
        np.where(masked, -127, 0)
        .astype("int8")
        .reshape(1, 1, seq_len, ctx_padded // PWORD, PWORD)
        .repeat(heads, axis=1)
        .repeat(batch, axis=0)
    )


def _sdpa_constants(context_len: int, head_dim: int, n_luts: int = 1) -> dict:
    """The quant-scale + LUT constants both ops take, built once for both.

    ``n_luts`` chains that many LUTs (ADDITIVE) for exp/inv -- more LUTs cost more
    per-element ops (one lookup + one add per extra chain link) but track the true
    function tighter; n_luts=1 is the cheapest, least accurate case.
    """
    exp_params = compute_chained_lut_params(
        0.0, 5.53, n_luts, method=ChainedLUTMethod.ADDITIVE, fit_fn=fit_exp_lut_params_sdpa_softmax
    )
    inv_params = compute_chained_lut_params(
        1.0,
        float(context_len),
        n_luts,
        method=ChainedLUTMethod.ADDITIVE,
        fit_fn=partial(fit_inv_lut_params, epsilon=0.0),
    )
    scores_dq_scale = (1.0 / 100.0) / np.sqrt(head_dim)
    zero = np.zeros((1, PWORD), dtype=np.float32)
    return {
        "exp_lut": np.stack([pack_lut_params_to_pword(p) for p in exp_params]),
        "inv_lut": np.stack([pack_lut_params_to_pword(p) for p in inv_params]),
        "scores_dq_scale": np.array([[scores_dq_scale] * PWORD], dtype=np.float32),
        "weights_q_scale": np.array([[127.0] * PWORD], dtype=np.float32),
        "zero": zero,
        "qmin": np.array([[-128.0] * PWORD], dtype=np.float32),
        "qmax": np.array([[127.0] * PWORD], dtype=np.float32),
        "output_dq_scale": np.array([[1.0] * PWORD], dtype=np.float32),
    }


def _random_qkv(
    batch: int, heads: int, kv_heads: int, seq_len: int, ctx_len: int, head_dim: int, seed: int
) -> tuple:
    """int8 (query, key, value) arrays, PWORD-lane-expanded, for a shared A/B input."""
    rng = np.random.default_rng(seed)
    hd_blocks = head_dim // PWORD
    q = rng.integers(-128, 128, (batch, heads, seq_len, hd_blocks, PWORD), dtype=np.int8)
    k = rng.integers(-128, 128, (batch, kv_heads, ctx_len, hd_blocks, PWORD), dtype=np.int8)
    v = rng.integers(-128, 128, (batch, kv_heads, ctx_len, hd_blocks, PWORD), dtype=np.int8)
    return q, k, v


def _build_module(
    op,
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    ctx_len: int,
    head_dim: int,
    *,
    scale: float,
    op_kwargs: dict,
    n_luts: int = 1,
):
    """One IRModule calling *op* (core SDPA or sdpa_flash) on a shared shape/mask/quant setup."""
    hd_blocks = head_dim // PWORD
    c = _sdpa_constants(ctx_len, head_dim, n_luts=n_luts)
    mask = _causal_mask(batch, heads, seq_len, ctx_len)

    def _build_body(query, key, value):
        out = op(
            query,
            key,
            value,
            A.alloc_constant(c["scores_dq_scale"], dtype="float32x64"),
            A.alloc_constant(c["exp_lut"], dtype="float32x64"),
            A.alloc_constant(c["inv_lut"], dtype="float32x64"),
            A.alloc_constant(c["weights_q_scale"], dtype="float32x64"),
            A.alloc_constant(c["zero"], dtype="float32x64"),
            A.alloc_constant(c["qmin"], dtype="float32x64"),
            A.alloc_constant(c["qmax"], dtype="float32x64"),
            A.alloc_constant(c["output_dq_scale"], dtype="float32x64"),
            A.alloc_constant(c["zero"], dtype="float32x64"),
            A.alloc_constant(c["zero"], dtype="float32x64"),
            A.alloc_constant(mask, dtype="int8x64"),
            enable_gqa=heads != kv_heads,
            scale=scale,
            **op_kwargs,
        )
        A.evaluate(A.output(out))

    @A.ir_module
    class Module:
        @A.prim_func
        def main(
            query: A.Buffer((batch, heads, seq_len, hd_blocks), "int8x64", scope=A.DDR),
            key: A.Buffer((batch, kv_heads, ctx_len, hd_blocks), "int8x64", scope=A.DDR),
            value: A.Buffer((batch, kv_heads, ctx_len, hd_blocks), "int8x64", scope=A.DDR),
        ):
            _build_body(query, key, value)

    return Module


def _build_europa_module(
    op,
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    ctx_len: int,
    head_dim: int,
    *,
    scale: float,
    op_kwargs: dict,
    num_phases: int = 1,
):
    """Like ``_build_module``, but for sdpa_flash_europa's trimmed signature (no
    exp_lut/inv_lut -- it computes exp/1/x with EXP2/RECIP, not LUT chains).

    ``num_phases`` > 1 calls the op that many times in one dispatch (only the
    last call's output is wired up; earlier calls' outputs are dead but
    harmless). This isolates the steady-state marginal cost of one call/barrier
    -- once the one-time per-core launch stagger is already absorbed by the
    first call -- as the delta between num_phases=k and k+1, entirely from the
    harness: the op itself is never modified. See the barrier-cost
    investigation doc for why that stagger otherwise dominates a naive,
    single-call measurement.
    """
    hd_blocks = head_dim // PWORD
    c = _sdpa_constants(ctx_len, head_dim)
    mask = _causal_mask(batch, heads, seq_len, ctx_len)

    def _build_body(query, key, value):
        for _ in range(num_phases):
            out = op(
                query,
                key,
                value,
                A.alloc_constant(c["scores_dq_scale"], dtype="float32x64"),
                A.alloc_constant(c["weights_q_scale"], dtype="float32x64"),
                A.alloc_constant(c["zero"], dtype="float32x64"),
                A.alloc_constant(c["qmin"], dtype="float32x64"),
                A.alloc_constant(c["qmax"], dtype="float32x64"),
                A.alloc_constant(c["output_dq_scale"], dtype="float32x64"),
                A.alloc_constant(c["zero"], dtype="float32x64"),
                A.alloc_constant(c["zero"], dtype="float32x64"),
                # keyword, not positional: sdpa_flash_europa's mask operand is optional
                # (it synthesises the mask by default) and only keyword operands may be None.
                attention_mask=A.alloc_constant(mask, dtype="int8x64"),
                enable_gqa=heads != kv_heads,
                scale=scale,
                **op_kwargs,
            )
        A.evaluate(A.output(out))

    @A.ir_module
    class Module:
        @A.prim_func
        def main(
            query: A.Buffer((batch, heads, seq_len, hd_blocks), "int8x64", scope=A.DDR),
            key: A.Buffer((batch, kv_heads, ctx_len, hd_blocks), "int8x64", scope=A.DDR),
            value: A.Buffer((batch, kv_heads, ctx_len, hd_blocks), "int8x64", scope=A.DDR),
        ):
            _build_body(query, key, value)

    return Module


def build_core_module(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    ctx_len: int,
    head_dim: int,
    *,
    scale: float | None = None,
    cores: int | tuple[int, ...] | None = None,
    generation: HardwareGeneration = HardwareGeneration.OMEGA,
):
    """axol.core.scaled_dot_product_attention -- the dense, untiled baseline."""
    scale = scale if scale is not None else 1.0 / np.sqrt(head_dim)
    return _build_module(
        axol.core.scaled_dot_product_attention,
        batch,
        heads,
        kv_heads,
        seq_len,
        ctx_len,
        head_dim,
        scale=scale,
        op_kwargs={"generation": generation, "cores": cores},
    )


def build_flash_module(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    ctx_len: int,
    head_dim: int,
    *,
    scale: float | None = None,
    kv_tile_blocks: int = 0,
    seq_tile: int = 0,
    cores: int | tuple[int, ...] | None = None,
    generation: HardwareGeneration = HardwareGeneration.OMEGA,
    n_luts: int = 1,
):
    """llm.sdpa_flash -- KV-tiled (+ optionally seq-tiled); 0/0 auto-picks both.

    ``n_luts`` chains that many exp/inv LUTs (see ``_sdpa_constants``); 1 (default)
    is the cheapest, least accurate case.
    """
    scale = scale if scale is not None else 1.0 / np.sqrt(head_dim)
    return _build_module(
        sdpa_flash,
        batch,
        heads,
        kv_heads,
        seq_len,
        ctx_len,
        head_dim,
        scale=scale,
        n_luts=n_luts,
        op_kwargs={
            "kv_tile_blocks": kv_tile_blocks,
            "seq_tile": seq_tile,
            "generation": generation,
            "cores": cores,
        },
    )


def build_flash_europa_module(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    ctx_len: int,
    head_dim: int,
    *,
    scale: float | None = None,
    kv_tile_blocks: int = 0,
    seq_tile: int = 0,
    cores: int | tuple[int, ...] | None = None,
    num_phases: int = 1,
):
    """llm.sdpa_flash_europa -- exp/1x via EXP2/RECIP instead of LUT chains; EUROPA-only.

    ``num_phases`` -- see ``_build_europa_module``.
    """
    scale = scale if scale is not None else 1.0 / np.sqrt(head_dim)
    return _build_europa_module(
        sdpa_flash_europa,
        batch,
        heads,
        kv_heads,
        seq_len,
        ctx_len,
        head_dim,
        scale=scale,
        op_kwargs={
            "kv_tile_blocks": kv_tile_blocks,
            "seq_tile": seq_tile,
            "generation": HardwareGeneration.EUROPA,
            "cores": cores,
        },
        num_phases=num_phases,
    )


def _rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    """Mean relative error over entries that are a significant fraction of ref's peak.

    Near-zero reference entries would otherwise blow up a plain relative error;
    cf. the ops' own tests, which use the same significance mask.
    """
    got = got.reshape(ref.shape).astype(np.float64)
    ref = ref.astype(np.float64)
    sig = np.abs(ref) > 0.01 * np.abs(ref).max()
    return float((np.abs(got - ref)[sig] / np.abs(ref)[sig]).mean())


def check_correctness(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    ctx_len: int,
    head_dim: int,
    seed: int = 0,
    *,
    cores: int | tuple[int, ...] | None = None,
) -> float:
    """Relative error of sdpa_flash (auto tiles) vs the core op, same inputs, no device.

    Both quantize/softmax slightly differently (flash's online rescale adds float18
    rounding the core op's one-shot softmax doesn't -- see sdpa_flash's module
    docstring), so this is a tolerance check (~2%), not bit-exactness.
    """
    core_mod = build_core_module(batch, heads, kv_heads, seq_len, ctx_len, head_dim, cores=cores)
    flash_mod = build_flash_module(batch, heads, kv_heads, seq_len, ctx_len, head_dim, cores=cores)
    q, k, v = _random_qkv(batch, heads, kv_heads, seq_len, ctx_len, head_dim, seed)
    (core_out,) = interpret(core_mod, q, k, v)
    (flash_out,) = interpret(flash_mod, q, k, v)
    return _rel_err(flash_out, core_out)


def check_flash_europa_correctness(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    ctx_len: int,
    head_dim: int,
    seed: int = 0,
    *,
    cores: int | tuple[int, ...] | None = None,
) -> float:
    """Relative error of sdpa_flash_europa vs sdpa_flash (auto tiles), same inputs, no device.

    Currently sdpa_flash_europa is an exact copy of sdpa_flash, so this is expected to be
    ~0 (bit-exact in emulate(), which both share unmodified) until the LUT->dpu_unary swap.
    """
    flash_mod = build_flash_module(batch, heads, kv_heads, seq_len, ctx_len, head_dim, cores=cores)
    europa_mod = build_flash_europa_module(
        batch, heads, kv_heads, seq_len, ctx_len, head_dim, cores=cores
    )
    q, k, v = _random_qkv(batch, heads, kv_heads, seq_len, ctx_len, head_dim, seed)
    (flash_out,) = interpret(flash_mod, q, k, v)
    (europa_out,) = interpret(europa_mod, q, k, v)
    return _rel_err(europa_out, flash_out)


def compare_shapes(
    ctx_lens: list,
    *,
    batch: int = 1,
    heads: int = 4,
    kv_heads: int = 4,
    seq_len: int = 128,
    head_dim: int = 128,
    kv_tile_blocks: int = 0,
    seq_tile: int = 0,
    cores: int | tuple[int, ...] | None = None,
    generation: HardwareGeneration = HardwareGeneration.OMEGA,
    iterations: int = 200,
    out_dir: Path | None = None,
    check: bool = True,
    include_flash_europa: bool = False,
) -> list:
    """Compile + time both ops for each context length, on device, and compare.

    One result dict per ctx_len: cycles/us for each op that compiled, their
    speedup ratio, and the flash-vs-core relative error (skipped if a shape
    fails to build/compile -- e.g. the core op overflowing L1 at long context,
    the exact case sdpa_flash exists to fix). ``cores`` shards the query heads
    across that many AIPU cores (both ops support it); None/1 is single-core.
    ``generation`` must be passed to ``measure_module`` too (it drives a SEPARATE
    outer AltoConfig, not just the op args) or the op silently compiles+runs on
    the wrong chip -- see ``_device_alto_config``.

    ``include_flash_europa`` adds a third leg, llm.sdpa_flash_europa (EUROPA-only --
    requires ``generation=HardwareGeneration.EUROPA``), plus its relative error and
    speedup vs plain flash -- an A/B of the two flash implementations, currently
    expected to land at ~1.0x/~0 error since sdpa_flash_europa is still an exact copy.
    """
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="bench_sdpa_flash_"))
    results = []
    for ctx_len in ctx_lens:
        row = {"ctx_len": ctx_len, "seq_len": seq_len}
        if check:
            try:
                row["rel_err"] = check_correctness(
                    batch, heads, kv_heads, seq_len, ctx_len, head_dim, cores=cores
                )
            except Exception as e:  # noqa: BLE001 -- report and keep timing anyway
                logger.warning("ctx_len=%d correctness check failed: %s", ctx_len, e)
            if include_flash_europa:
                try:
                    row["europa_rel_err"] = check_flash_europa_correctness(
                        batch, heads, kv_heads, seq_len, ctx_len, head_dim, cores=cores
                    )
                except Exception as e:  # noqa: BLE001 -- report and keep timing anyway
                    logger.warning(
                        "ctx_len=%d flash_europa correctness check failed: %s", ctx_len, e
                    )

        q, k, v = _random_qkv(batch, heads, kv_heads, seq_len, ctx_len, head_dim, seed=0)
        legs = [
            (
                "core",
                build_core_module(
                    batch,
                    heads,
                    kv_heads,
                    seq_len,
                    ctx_len,
                    head_dim,
                    cores=cores,
                    generation=generation,
                ),
            ),
            (
                "flash",
                build_flash_module(
                    batch,
                    heads,
                    kv_heads,
                    seq_len,
                    ctx_len,
                    head_dim,
                    kv_tile_blocks=kv_tile_blocks,
                    seq_tile=seq_tile,
                    cores=cores,
                    generation=generation,
                ),
            ),
        ]
        if include_flash_europa:
            legs.append(
                (
                    "flash_europa",
                    build_flash_europa_module(
                        batch,
                        heads,
                        kv_heads,
                        seq_len,
                        ctx_len,
                        head_dim,
                        kv_tile_blocks=kv_tile_blocks,
                        seq_tile=seq_tile,
                        cores=cores,
                    ),
                )
            )
        # measure_module's profiler needs the real core count to time every core
        # (not just core 0) -- see layer_benchmark.py's _device_alto_config note.
        num_cores = len(cores) if isinstance(cores, tuple) else (cores or 1)
        for name, mod in legs:
            try:
                res = measure_module(
                    mod,
                    [q, k, v],
                    out_dir=out_dir / f"{name}_{ctx_len}",
                    iterations=iterations,
                    generation=generation,
                    num_cores=num_cores,
                )
                row[f"{name}_cycles"] = res["cycles"]
                row[f"{name}_us"] = res["us"]
            except Exception as e:  # noqa: BLE001 -- e.g. core op overflowing L1 at long ctx
                logger.warning("ctx_len=%d %s failed: %s", ctx_len, name, e)
                row[f"{name}_error"] = str(e)
        if "core_us" in row and "flash_us" in row:
            row["speedup"] = row["core_us"] / row["flash_us"]
        if "flash_us" in row and "flash_europa_us" in row:
            row["europa_speedup"] = row["flash_us"] / row["flash_europa_us"]
        results.append(row)
        logger.info("ctx_len=%d: %s", ctx_len, row)
    return results


def _format(results: list) -> str:
    show_europa = any("flash_europa_us" in r or "europa_rel_err" in r for r in results)
    header = f"{'ctx_len':>8} {'core us':>10} {'flash us':>10} {'speedup':>8} {'rel_err':>8}"
    if show_europa:
        header += f" {'europa us':>10} {'eu_speedup':>10} {'eu_rel_err':>10}"
    lines = [header]
    for r in results:
        core_us = f"{r['core_us']:.2f}" if "core_us" in r else "FAILED"
        flash_us = f"{r['flash_us']:.2f}" if "flash_us" in r else "FAILED"
        speedup = f"{r['speedup']:.2f}x" if "speedup" in r else "-"
        rel_err = f"{r['rel_err']:.4f}" if "rel_err" in r else "-"
        line = f"{r['ctx_len']:>8} {core_us:>10} {flash_us:>10} {speedup:>8} {rel_err:>8}"
        if show_europa:
            europa_us = f"{r['flash_europa_us']:.2f}" if "flash_europa_us" in r else "FAILED"
            eu_speedup = f"{r['europa_speedup']:.2f}x" if "europa_speedup" in r else "-"
            eu_rel_err = f"{r['europa_rel_err']:.4f}" if "europa_rel_err" in r else "-"
            line += f" {europa_us:>10} {eu_speedup:>10} {eu_rel_err:>10}"
        lines.append(line)
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ctx-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--kv-tile-blocks", type=int, default=0, help="0 = flash auto-tiles")
    p.add_argument("--seq-tile", type=int, default=0, help="0 = flash auto-tiles")
    p.add_argument(
        "--cores", type=int, default=None, help="shard query heads over this many AIPU cores"
    )
    p.add_argument("--iterations", type=int, default=200, help="profiler iterations")
    p.add_argument("--no-check", action="store_true", help="skip the flash-vs-core rel_err gate")
    p.add_argument("--no-sanity", action="store_true", help="skip the device wedge check")
    p.add_argument(
        "--flash-europa",
        action="store_true",
        help="also build+time llm.sdpa_flash_europa (EUROPA-only); implies --generation europa",
    )
    p.add_argument(
        "--generation",
        choices=[g.value for g in HardwareGeneration],
        default=HardwareGeneration.OMEGA.value,
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> list:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.no_sanity and not device_sanity_check():
        raise SystemExit("device sanity check failed; refusing to report perf on a wedged device")

    generation = HardwareGeneration(args.generation)
    if args.flash_europa and generation is not HardwareGeneration.EUROPA:
        raise SystemExit(
            "--flash-europa requires --generation europa (sdpa_flash_europa is EUROPA-only)"
        )

    results = compare_shapes(
        args.ctx_lens,
        batch=args.batch,
        heads=args.heads,
        kv_heads=args.kv_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        kv_tile_blocks=args.kv_tile_blocks,
        seq_tile=args.seq_tile,
        cores=args.cores,
        generation=generation,
        iterations=args.iterations,
        check=not args.no_check,
        include_flash_europa=args.flash_europa,
    )
    print(_format(results))
    return results


if __name__ == "__main__":
    main()
