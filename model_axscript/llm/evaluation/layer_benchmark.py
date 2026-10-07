#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Single Qwen3 decoder-layer benchmark + correctness harness (on real Metis).

Only end-to-end (full-layer or subgraph) numbers come from the device, so this
times *one* compiled decoder layer on hardware under the DEVICE_MAIN cycle
profiler and reads back median cycles -> us/layer. The Metis schedule is static
(no data-dependent control flow), so the cycle count is exact for a given shape
regardless of the data -- but we still bind *real* per-layer weights/LUTs from a
saved QuantizedModel bundle and synthesize the host inputs (mask/rope/seq_pos)
exactly as the runtime does, so device numerics stay in range (no NaN/saturation)
and the same inputs double as a correctness fixture.

Three entry points:

  * ``measure_layer``  -- compile one (impl, phase, ctx_len) layer for DEVICE,
    run it under the profiler, return {cycles, us, per-op breakdown}.
  * ``measure_module`` -- the same for an arbitrary IRModule (op/subgraph
    microbench): the fast A/B loop, since a small kernel compiles in seconds.
  * ``rel_l2_vs``      -- interpret two kernels on shared inputs (no compile) and
    return their relative-L2; the correctness gate a perf change must pass.

A multicore Europa decode build stacks several layers into one kernel by default
(``layers_per_kernel``), so a "layer" timing is that group's unless ``layers_per_kernel=1``
is asked for; every result carries ``layers_per_kernel`` and ``us_per_layer``.

It reaches ``utils.compile_and_run``, which lives beside the ALTO integration tests and is
not part of the model tree. Both checkouts resolve that import: the compiler checkout has
the harness itself, and the framework checkout mirrors the package path with a vendored
subset of it.

Run directly (script, not pytest-collected, see bench_ddr_l1_bandwidth.py). In the
compiler checkout:

    source build-Release/workspace/chroot/bin/axelera-activate.sh
    cd host/tvm/axelera_compiler
    PYTHONPATH="$PWD:$PYTHONPATH" python -m tests.integration.alto.llm.evaluation.layer_benchmark \\
        --impl qwen3-metis-opt --phases prefill decode --ctx-len 1024

In the framework checkout, the same tool under its module path there:

    python -m model_axscript.llm.evaluation.layer_benchmark \\
        --impl qwen3-metis-opt --phases prefill decode --ctx-len 1024
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
from pathlib import Path
import re
import tempfile

import numpy as np
import torch as _torch

# --- avoid the (unused) qtoolsv2 native ext, which axelera.compiler imports eagerly ---
# The prebuilt .so in this chroot is ABI-mismatched against the active torch and nothing
# on the compile/run path here calls a qtools op; make *only* that load non-fatal. See
# the qtoolsv2-abi-mismatch note. A real qtools op would still raise if ever invoked.
# axelera.compiler is imported lazily (inside the functions below), so this shim is
# installed before any eager qtoolsv2 load.
_orig_load_library = _torch.ops.load_library


def _tolerant_load_library(path):
    try:
        _orig_load_library(path)
    except OSError:
        if "qtoolsv2" not in str(path):
            raise


_torch.ops.load_library = _tolerant_load_library
# --------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

PWORD = 64

# AI-core clock per generation, the divisor for DEVICE_MAIN cycles. Europa runs at
# 1.2 GHz; the profile's own frequency field cannot express it (see _read_cycles).
CORE_CLOCK_HZ = {"omega": 800_000_000, "europa": 1_200_000_000}

# Default saved Qwen3-0.6B bundle (28 layers, hidden 1024). Override with --bundle.
_DEFAULT_BUNDLE = (
    Path(__file__).resolve().parents[2] / "build" / "qwen3-0.6b-2k" / "quantized_model"
)

# Threaded residual-stream activation scale. Qwen3's pre-norm residual stream is
# O(1); the in-layer RMSNorms renormalise anyway, so the exact value only matters
# for keeping the int8 GEMMs off their saturation rails -- unit-ish is realistic.
_HIDDEN_STD = 1.0


def _spec_from_kernel(mod) -> dict:
    """{logical_name: (shape_with_lane, dtype)} read straight off the prim_func.

    The compiled model.json buffer spec is each x64 buffer's declared shape with
    the 64-lane appended and the dtype's ``x64`` stripped -- so we can synthesize
    it without a multi-minute compile (same trick as test_qwen3_5_deltanet_layer).
    """
    func = mod["main"]
    spec: dict = {}
    for var in func.params:
        buf = func.buffer_map[var]
        dtype = buf.dtype.replace("x64", "")
        logical = re.sub(r"_\d+$", "", buf.name)
        spec[logical] = (tuple(int(d) for d in buf.shape) + (PWORD,), dtype)
    return spec


def _logical_names(mod) -> list[str]:
    """Param logical names in prim_func order (== device input order)."""
    func = mod["main"]
    return [re.sub(r"_\d+$", "", func.buffer_map[v].name) for v in func.params]


def load_bundle(bundle_dir: Path):
    from ..quantized_model import QuantizedModel

    return QuantizedModel.load(Path(bundle_dir))


def _model_config(bundle):
    from ..models.architecture import get_architecture

    return get_architecture(bundle.architecture).config_from_dict(bundle.config)


def _layer_kernel(impl, model_config):
    """The (single) decoder LayerKernel for a uniform-stack implementation."""
    kernels = impl.layer_kernels(model_config)
    layer_kernels = [k for k in kernels.values()]
    if len(layer_kernels) != 1:
        raise ValueError(f"expected one layer kind, got {sorted(kernels)}")
    return layer_kernels[0]


def _bind_layer_buffers(kernel, bundle, spec, layer: int, num_layers: int) -> dict:
    """One call's packed constants, whether the kernel runs one layer or N.

    The packers key on the untagged names (``check_norm_folds`` looks up ``q_proj_qparams``),
    so a stacked kernel's spec is collapsed for them and the tag put back on the result:
    sub-layer j is packed from layer ``layer + j``.
    """
    from ..graph_descriptor import group_size, per_layer_spec, sublayer_tag

    group = group_size(spec)
    if group == 1:
        return kernel.layer_buffers(bundle, spec, layer)
    if layer + group > num_layers:
        raise ValueError(
            f"layer {layer} starts a group of {group} sub-layers, which runs past the model's "
            f"{num_layers} layers; the last valid --layer for this kernel is {num_layers - group}"
        )
    untagged = per_layer_spec(spec)
    return {
        name + sublayer_tag(j): value
        for j in range(group)
        for name, value in kernel.layer_buffers(bundle, untagged, layer + j).items()
    }


def build_layer_inputs(
    mod,
    impl,
    bundle,
    model_config,
    *,
    ctx_len: int,
    phase: str,
    layer: int = 0,
    position: int | None = None,
    seed: int = 0,
    chunk_len: int | None = None,
) -> list[np.ndarray]:
    """Ordered (prim_func order) device inputs for one decoder layer.

    Real per-layer weights/LUTs come from the bundle (via the kernel's
    ``layer_buffers``); host runtime inputs (mask/rope/seq_pos) are synthesized
    exactly as DecoderLayerImplementation would; threaded activations are a
    fixed-seed residual-stream draw; persistent KV state is zero (fresh instance).

    ``position`` (decode only) is the KV write index / query position; default is
    the last context slot so decode reads a representative cache depth.
    """
    from ..implementation import make_causal_mask, make_rope_cos_sin, make_seq_pos

    kernel = _layer_kernel(impl, model_config)
    spec = _spec_from_kernel(mod)
    bound = _bind_layer_buffers(kernel, bundle, spec, layer, model_config.num_hidden_layers)

    if phase == "decode":
        seq_len = 1
        pos = ctx_len - 1 if position is None else position
        positions = pos
        chunk_start = 0
    else:
        # Chunked prefill: the chunk_len query rows are the LAST chunk (positions
        # ctx_len-chunk_len .. ctx_len-1) attending the full ctx_len KV cache -- the
        # representative, most-expensive chunk. chunk_len=None => full prefill (seq=ctx).
        seq_len = chunk_len if chunk_len else ctx_len
        chunk_start = ctx_len - seq_len
        positions = np.arange(chunk_start, ctx_len)

    head_dim = int(bundle.host_scalars["head_dim"])
    num_q_heads = int(bundle.host_scalars["num_q_heads"])
    rope_theta = float(bundle.host_scalars["rope_theta"])
    cos, sin = make_rope_cos_sin(positions, head_dim, rope_theta)
    hidden_pw = model_config.hidden_size // PWORD

    rng = np.random.default_rng(seed)
    hidden = (rng.standard_normal((1, seq_len, hidden_pw, PWORD)) * _HIDDEN_STD).astype(np.float32)

    # Build the mask in the head count AND dtype the kernel actually declares: a kernel
    # may pass a (1, 1, ...) head-broadcast int8 mask (the SDPA op broadcasts it across
    # heads and casts it), which is cheaper to stream than a full per-head fp32 mask.
    # The dtype has to be passed in, not cast afterwards -- the fp sentinel (-30000)
    # does not survive a cast to int8, whereas make_causal_mask picks -127 for int8.
    mask_spec = spec.get("attn_mask")
    mask_heads = int(mask_spec[0][1]) if mask_spec else num_q_heads
    mask_dtype = np.dtype(mask_spec[1]) if mask_spec else np.float32
    host_inputs = {
        "attn_mask": make_causal_mask(positions, mask_heads, ctx_len, dtype=mask_dtype),
        "q_rope_cos": cos,
        "q_rope_sin": sin,
        "k_rope_cos": cos,
        "k_rope_sin": sin,
        # Host-precomputed combined trig (sin||cos), if the kernel takes one input
        # instead of the 4 separate sin/cos (q and k share it -- same positions).
        "rope_trig": np.ascontiguousarray(np.concatenate([sin, cos], axis=2)),
        "seq_pos": make_seq_pos(chunk_start if phase == "prefill" else pos),
        "hidden_states": hidden,
        "residual_in": hidden,  # the layer input feeds both threaded slots
    }

    ordered: list[np.ndarray] = []
    for logical in _logical_names(mod):
        if logical in host_inputs:
            # Coerce to the kernel's declared dtype (e.g. an fp16 attn_mask).
            dt = np.dtype(spec[logical][1])
            ordered.append(np.ascontiguousarray(host_inputs[logical].astype(dt)))
        elif logical in bound:
            ordered.append(np.ascontiguousarray(bound[logical].numpy()))
        else:
            # Persistent per-layer state (k_cache / v_cache): zero on a fresh instance.
            shape, dtype = spec[logical]
            ordered.append(np.zeros(shape, dtype=np.dtype(dtype)))
    return ordered


def _device_alto_config(out_dir: Path, kernel, generation=None, num_cores=None):
    """AltoConfig mirroring DecoderLayerImplementation._compile_phase for DEVICE.

    ``generation`` must match whatever generation the op/module itself was built
    for (e.g. via the op's own ``generation=`` kwarg) -- this is a SEPARATE,
    outer compile-level setting; a mismatch doesn't fail loudly in general (only
    once a pass hits something the wrong generation can't express, e.g. a core
    index out of range), so it's easy to silently compile+run on the wrong chip.

    ``num_cores`` (default 1) must likewise match the op's own ``cores=`` kwarg --
    it's what tells PerformanceProfiling how many per-core TritonTrace collectors
    to spin up. Left at the default, only ONE collector runs regardless of how many
    cores the kernel actually uses, so _read_cycles only ever sees core 0's own
    self-timed `[M] ... cycles` interval -- trustworthy only because a trailing
    barrier forces core 0 to block until every other core has also finished; with
    no such barrier (or if some other core is slower for a barrier-independent
    reason), core 0's own number silently under-reports the true kernel duration.

    NB: the pass flags come from the kernel, as in _compile_phase -- benchmarking a kind
    with fusion/promotion set differently to the shipped artifact measures the wrong graph.
    Fusion is off unless the kind opts in (it miscompiles hand-built datapaths).
    """
    from axelera.compiler.alto.compiler.config import AltoConfig, HardwareGeneration, Target

    return AltoConfig(
        output_dir=out_dir,
        generation=generation or HardwareGeneration.OMEGA,
        target=Target.DEVICE,
        num_cores=num_cores or 1,
        enable_tiling_injection=True,
        enable_buffer_promotion=getattr(kernel, "enable_buffer_promotion", False),
        enable_datapath_fusion=getattr(kernel, "enable_datapath_fusion", False),
        enable_llm_support=True,
        assign_unique_input_pools=True,
        unpack_axm_archive=True,
        clean_output_dir=True,
        emit_atex_artifacts=True,
    )


def _read_cycles(rundir: Path, generation=None) -> dict:
    """max(median_cycles) + frequency from the DEVICE_MAIN profile, across every core.

    Each core times only its OWN instruction stream (its own `[M] ... cycles` line),
    so with more than one core's perf_stats file present, the true kernel duration
    is the max over cores (the last one to finish), not any single core's number --
    see _device_alto_config's num_cores note. The frequency is NOT the profile's own
    ``frequency`` field: that is the compiler config's, whose default and maximum are
    both 800 MHz, so on Europa it inflates every time by 1.5x -- look it up by
    generation instead.
    """
    perf_files = list(Path(rundir).glob("perf_stats_*core_*.json"))
    if not perf_files:
        raise RuntimeError(f"no device-cycle profile written under {rundir}")
    per_core = []
    for f in perf_files:
        dm = json.loads(f.read_text())["perf_stats"]["device_main"]
        if "median_cycles" not in dm:
            # The level was enabled but the device produced no samples (happens at
            # higher core counts); the placeholder entry carries no cycle count.
            continue
        per_core.append(dm)
    if not per_core:
        raise RuntimeError(f"device-cycle profile in {rundir} has no samples")
    slowest = max(per_core, key=lambda dm: dm["median_cycles"])
    cycles = slowest["median_cycles"]
    freq_hz = CORE_CLOCK_HZ.get(
        str(getattr(generation, "value", generation)), slowest["frequency"]
    )
    return {"cycles": cycles, "freq_hz": freq_hz, "us": cycles / freq_hz * 1e6}


def _device_report(rundir: Path) -> list[dict] | None:
    """Best-effort per-op breakdown from a DEVICE_REPORT dump, if one was written."""
    reports = list(Path(rundir).glob("*device_report*.json")) + list(
        Path(rundir).glob("*report*.json")
    )
    for f in reports:
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
    return None


def measure_module(
    mod,
    inputs: list[np.ndarray] | None = None,
    *,
    out_dir: Path,
    iterations: int = 200,
    kernel=None,
    device_report: bool = True,
    generation=None,
    num_cores=None,
    host_timing: bool = False,
    device_cycles: bool = True,
) -> dict:
    """Compile ``mod`` for DEVICE, run ``inputs`` under the profiler, return timing.

    The microbench primitive: works for a full layer or any standalone op/subgraph
    IRModule. ``inputs=None`` lets the runtime synthesize shape-correct random
    inputs (fine for a data-independent timing). ``kernel`` (optional) carries the
    per-kind compile flags (buffer-promotion); a bare op passes None (defaults off).
    ``generation`` (default OMEGA) must match the generation ``mod``'s op(s) were
    built for -- see ``_device_alto_config``. ``num_cores`` (default 1) must match
    the op's own ``cores=`` kwarg or the reported cycles silently reflect only one
    core's own timer -- see ``_device_alto_config``'s and ``_read_cycles``'s notes.
    ``host_timing`` also records the host-side kernel wall clock, the only timing
    left when DEVICE_MAIN yields no samples, as it does at high core counts.
    ``device_cycles=False`` drops DEVICE_MAIN entirely: it raises the AI-core log
    level and returns no samples above one core, which perturbs what it measures.
    """
    from axelera.compiler.alto.compiler.config import AltoRuntimeConfig, ProfilingLevel

    from tests.integration.alto.utils import compile_and_run

    out_dir = Path(out_dir)
    cfg = _device_alto_config(out_dir, kernel, generation, num_cores)
    levels = [ProfilingLevel.DEVICE_MAIN] if device_cycles else []
    if device_report:
        levels.append(ProfilingLevel.DEVICE_REPORT)
    if host_timing:
        levels.append(ProfilingLevel.HOST_TEMPLATE)
    rt = AltoRuntimeConfig(
        profiling_levels=levels,
        iterations=iterations,
        profiling_output_dir=out_dir,
    )
    _, outputs = compile_and_run(mod, cfg, config_runtime=rt, inputs=inputs)
    result = _read_cycles(out_dir, generation) if device_cycles else {}
    result["out_dir"] = str(out_dir)
    report = _device_report(out_dir) if device_report else None
    if report is not None:
        result["device_report"] = report
    result["outputs"] = outputs
    return result


def _build_kwargs(
    build, cores: int, chunk_len: int | None, phase: str, layers_per_kernel: int | None = None
) -> dict:
    """The builder kwargs for this run, refusing any the builder cannot accept.

    Dropping one silently would time a single-core or full-width build under a
    multicore or chunked label, or a stacked kernel under a one-layer label.
    """
    kwargs: dict = {}
    if cores > 1:
        kwargs["cores"] = cores
    if chunk_len and phase == "prefill":
        kwargs["chunk_len"] = chunk_len
    if layers_per_kernel is not None:
        kwargs["layers_per_kernel"] = layers_per_kernel
    accepted = inspect.signature(build).parameters
    missing = [name for name in kwargs if name not in accepted]
    if missing:
        raise ValueError(f"builder {build!r} takes no {missing}, so it cannot be timed with them")
    return kwargs


def measure_layer(
    impl_name: str,
    *,
    bundle,
    ctx_len: int,
    phase: str,
    layer: int = 0,
    iterations: int = 200,
    out_dir: Path | None = None,
    generation=None,
    cores: int = 1,
    chunk_len: int | None = None,
    layers_per_kernel: int | None = None,
) -> dict:
    """Compile + time one decoder-layer kernel of ``impl_name`` for ``phase``.

    ``generation`` / ``cores`` / ``chunk_len`` / ``layers_per_kernel`` are threaded to the
    builder exactly as ``_compile_phase`` does, so a multicore implementation is timed at the
    core count and generation it ships at. The result's ``us`` is the kernel's;
    ``layers_per_kernel`` says how many layers that kernel runs and ``us_per_layer`` divides
    it out, so a stacked decode build is never reported as one layer's time.
    """
    from axelera.compiler.alto.compiler.config import alto_config_context

    from ..graph_descriptor import group_size
    from ..implementation import get_implementation

    impl = get_implementation(impl_name)
    model_config = _model_config(bundle)
    kernel = _layer_kernel(impl, model_config)
    build_kwargs = _build_kwargs(kernel.build, cores, chunk_len, phase, layers_per_kernel)
    # Ops resolve an unset generation from the ambient config at trace time.
    with alto_config_context(generation=generation) if generation else contextlib.nullcontext():
        mod = kernel.build(model_config, ctx_len, phase, **build_kwargs)
    inputs = build_layer_inputs(
        mod,
        impl,
        bundle,
        model_config,
        ctx_len=ctx_len,
        phase=phase,
        layer=layer,
        chunk_len=chunk_len,
    )
    group = group_size(_spec_from_kernel(mod))
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix=f"layer_{impl_name}_{phase}_{ctx_len}_"))
    logger.info(
        "[%s/%s ctx=%d] compiling + timing a %d-layer kernel (%d iters)",
        impl_name,
        phase,
        ctx_len,
        group,
        iterations,
    )
    res = measure_module(
        mod,
        inputs,
        out_dir=out_dir,
        iterations=iterations,
        kernel=kernel,
        generation=generation,
        num_cores=cores,
    )
    res.update(
        impl=impl_name,
        phase=phase,
        ctx_len=ctx_len,
        layer=layer,
        cores=cores,
        layers_per_kernel=group,
        us_per_layer=res["us"] / group,
    )
    return res


def rel_l2_vs(mod_a, mod_b, inputs: list[np.ndarray], out_index: int = 0) -> float:
    """Relative-L2 between two kernels interpreted on the *same* inputs (no compile).

    The correctness gate: a candidate kernel (mod_b) must track the baseline
    (mod_a) within the int8/LUT band before its device timing is trusted.
    """
    from axelera.compiler.alto.compiler import interpret

    a = np.asarray(interpret(mod_a, *inputs)[out_index], dtype=np.float64)
    b = np.asarray(interpret(mod_b, *inputs)[out_index], dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def device_sanity_check() -> bool:
    """Trivial DDR->L1->DDR copy must be bit-exact before trusting any device run.

    A prior DEVICE_LOST wedges the device (still enumerates, every kernel returns
    garbage); this catches that state so a real perf result is never read off a
    wedged device. See the metis-device-wedge-recovery note for the reload path.
    """
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol import atomic as aops
    from axelera.compiler.alto.compiler.config import AltoConfig, AltoRuntimeConfig, Target

    from tests.integration.alto.utils import compile_and_run

    @A.ir_module
    class RoundTrip:
        @A.prim_func
        def main(
            src: A.Buffer((64, 1), "int8x64", scope=A.DDR),  # noqa: F821 (TVMScript dtype str)
        ):
            l1 = A.alloc_buffer((64, 1), "int8x64", scope=A.L1_0)
            dst = A.alloc_buffer((64, 1), "int8x64", scope=A.DDR)
            aops.copy(src, output=l1)
            aops.copy(l1, output=dst)
            A.output(dst)

    rundir = Path(tempfile.mkdtemp(prefix="device_sanity_"))
    cfg = AltoConfig(output_dir=rundir, target=Target.DEVICE)
    ins, outs = compile_and_run(mod=RoundTrip, config=cfg, config_runtime=AltoRuntimeConfig())
    ok = np.array_equal(ins[0], outs[0])
    if not ok:
        logger.error("device sanity check FAILED (device may be wedged); reload firmware")
    return ok


def _format(results: list[dict]) -> str:
    lines = ["decoder-layer kernel device timing (median cycles @ core clock):"]
    for r in results:
        lines.append(
            f"  {r['impl']:>22} {r['phase']:>7} ctx={r['ctx_len']:<5} "
            f"{r['cycles']:>12,.0f} cyc  {r['us']:>9.2f} us/kernel  "
            f"({r['layers_per_kernel']} layer(s), {r['us_per_layer']:.2f} us/layer)"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--impl", default="qwen3-metis-opt", help="registered implementation name")
    p.add_argument("--phases", nargs="+", default=["prefill", "decode"])
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--layer", type=int, default=0, help="which layer's weights to bind")
    p.add_argument("--iterations", type=int, default=200, help="profiler iterations")
    p.add_argument("--cores", type=int, default=1, help="AI cores (multicore builders only)")
    p.add_argument("--generation", default=None, help="OMEGA or EUROPA; default OMEGA")
    p.add_argument("--chunk-len", type=int, default=None, help="chunked-prefill query width")
    p.add_argument(
        "--layers-per-kernel",
        type=int,
        default=None,
        help="decode layers stacked per kernel (builder default when unset; 1 times one layer)",
    )
    p.add_argument("--bundle", type=Path, default=_DEFAULT_BUNDLE)
    p.add_argument("--no-sanity", action="store_true", help="skip the device wedge check")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> list[dict]:
    from axelera.compiler.alto.compiler.config import HardwareGeneration

    from .. import models as _models  # noqa: F401  -- registers the implementations

    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.no_sanity and not device_sanity_check():
        raise SystemExit("device sanity check failed; refusing to report perf on a wedged device")

    bundle = load_bundle(args.bundle)
    results = []
    for phase in args.phases:
        r = measure_layer(
            args.impl,
            bundle=bundle,
            ctx_len=args.ctx_len,
            phase=phase,
            layer=args.layer,
            iterations=args.iterations,
            cores=args.cores,
            generation=(HardwareGeneration[args.generation.upper()] if args.generation else None),
            chunk_len=args.chunk_len,
            layers_per_kernel=args.layers_per_kernel,
        )
        r.pop("outputs", None)  # don't carry big arrays into the report
        results.append(r)
        logger.info(
            "  %s/%s: %.2f us per %d-layer kernel (%d cyc)",
            args.impl,
            phase,
            r["us"],
            r["layers_per_kernel"],
            int(r["cycles"]),
        )
    print(_format(results))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                [{k: v for k, v in r.items() if k != "device_report"} for r in results], indent=2
            )
        )
    return results


if __name__ == "__main__":
    main()
