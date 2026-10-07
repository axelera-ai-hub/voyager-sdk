#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Where a Qwen3-VL frame and a decode step actually go -- per kernel, and per operator.

`RunStats` (hf_model.py) splits a *run* into vision / prefill / decode, host against
device. This module goes the two levels below that, and it exists because the obvious
route does not work:

  * **Per kernel -- free, no compile.** A built artifact already contains every kernel
    compiled on its own (`vision/_block`, `text/decode_layer_decode`, ...). Running one
    of those under the DEVICE_MAIN cycle profiler gives its cycles directly, so the
    per-kernel table costs nothing but device time. `kernel_table()`.

  * **Per operator -- one compile per operator.** There is no in-band per-op timer on
    this firmware, and it is worth writing down why, because all three profiling levels
    look like they should provide one:

      - `DEVICE_MAIN` `[M]` lines DO carry a name, but it is the AI *core*
        (`ai_core_kernel_0..3`), not the operator -- the per-core JSONs already split by
        it. Whole-kernel totals only.
      - `DEVICE_KERNEL` `[K]` is declared in the ProfilingLevel enum and parsed in
        `axelera/compiler/utils/profiling.py`, but nothing in the ALTO lowering path
        emits it. Asking for it yields nothing.
      - `DEVICE_PERF_EVAL` `[PE]` is the only level with per-task timings, and it is
        firmware-gated: on PCIe production firmware the run comes back "No perf data
        collected". Its logging alone also inflates a vision block from 2.24 to 6.63
        Mcycles, so even where it works the timings it perturbs are not the ones you
        wanted to measure.

    What `[PE]` DOES give everywhere is *structural*: the task list and every DMA's
    size. That is what `dma_profile()` reads -- bytes moved and DMA count per kernel,
    which answers "is this bandwidth-bound?" without any timing at all.

    So an operator's cost has to be priced *by difference*: rebuild the kernel with that
    one operator neutralised and re-measure. The delta is its cost in real schedule
    context -- buffer promotion, overlap and DMAs included -- which is the number a design
    decision turns on. A different program is a different binary, so this costs one compile
    per operator; there is no way around it.

Ablation is a measurement, not a correctness-preserving edit: the neutralised kernel
computes the wrong answer on purpose. Nothing here writes an artifact.

Two traps this hit, both silent. A kernel compiled for N cores that places work on fewer
produces an EMPTY trace, and the profiler writes a JSON `NaN` median -- which Python parses
as a truthy float, so it slips past an `if median_cycles` check and poisons the result
(`_valid_cycles` filters it). And a micro-benchmark of a LUT-driven op needs real LUTs:
random values where an invsqrt table belongs wedges the card (`ZE_RESULT_ERROR_DEVICE_LOST`,
recovered with `axdevice --reload-firmware` after killing the stale `axtrace` daemons that
keep the device claimed).

Run it from the compiler root, as `tests.integration.alto....` -- the same import-path
rule `evaluation/layer_benchmark.py` documents. A `sys.path` shortcut that makes `llm` the
top-level package breaks the `from ....utils import` this needs:

    cd host/tvm/axelera_compiler
    PYTHONPATH="$PWD" python -m tests.integration.alto.llm.models.qwen3vl.profiling \
        --artifact-dir /path/to/qwen3vl-2b --kernels          # 0 compiles
    ... --ablate vision --cores 4                             # 1 compile per operator
    ... --lm-head                                             # the host gemv, timed
"""

import argparse
import contextlib
import json
import logging
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

logger = logging.getLogger(__name__)

#: The compiled kernels a run dispatches, vision tower then decoder. How MANY times each
#: is dispatched per frame / per decode step is a property of the CHECKPOINT, not of this
#: module -- 2B is 24 vision blocks and 3 DeepStack taps over 28 decoder layers, 8B is
#: neither -- so the multipliers are read off the artifact by `kernel_counts`.
VISION_KERNELS = ("vision/_embed", "vision/_block", "vision/_deepstack", "vision/_merge")
TEXT_KERNELS = (
    "text/decode_layer_prefill",
    "text/decode_layer_decode",
    "text/deepstack_add_prefill",
    "text/deepstack_add_decode",
)

_NP_DTYPE = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
}


# --- running an already-compiled kernel ---------------------------------------------


def _synth_inputs(spec: dict) -> list[np.ndarray]:
    """Shape/dtype-correct random inputs for a compiled model.json.

    Random is fine and it is worth saying why: the Metis schedule is static -- no
    data-dependent control flow -- so a kernel's cycle count is exact for a given shape
    whatever the data. Small values keep the int8 paths off their saturation limits.
    """
    inputs = []
    for spec_in in spec["inputs"]:
        dtype = _NP_DTYPE[spec_in["dtype"]]
        shape = tuple(spec_in["shape"])
        if np.issubdtype(dtype, np.unsignedinteger):
            # Unsigned inputs here are positions (seq_pos), not data: a negative cast
            # would wrap to an enormous index and walk off the KV cache. Keep them small
            # and valid -- and the cycle count does not depend on the value anyway.
            inputs.append(np.zeros(shape, dtype=dtype))
        elif np.issubdtype(dtype, np.integer):
            inputs.append(np.random.randint(-8, 8, size=shape).astype(dtype))
        else:
            inputs.append(((np.random.rand(*shape) - 0.5) * 0.1).astype(dtype))
    return inputs


@contextlib.contextmanager
def _capture_trace():
    """Collect the raw device trace lines the profiler parses and then discards.

    The parsed JSON keeps only aggregates; the DMA sizes and the task list are in the raw
    lines, so `dma_profile` needs them before they are thrown away.
    """
    from axelera.compiler.utils import profiling as prof

    lines: list[str] = []
    original = prof.PerformanceProfiling._parse_stats

    def spy(self, perf_stats):
        lines.extend(perf_stats)
        return original(self, perf_stats)

    prof.PerformanceProfiling._parse_stats = spy
    try:
        yield lines
    finally:
        prof.PerformanceProfiling._parse_stats = original


def _valid_cycles(value) -> float | None:
    """A core's median cycle count, or None if that core contributed nothing.

    A kernel compiled for N cores need not place work on all N. An idle core's trace is
    empty, its median is computed over an empty slice, and the profiler writes a JSON
    literal `NaN`. Python's json parses that into float('nan'), which is TRUTHY -- so a
    plain `if median_cycles` lets it through and one idle core turns the whole kernel's
    max() into nan. That is what "DeepStack @4 cores = nan" was: a measurement artifact,
    not a kernel that failed.
    """
    if value is None:
        return None
    number = float(value)
    return None if (number != number or number <= 0) else number  # NaN != NaN


def ensure_alto_config(kernel_dir: Path) -> None:
    """Give a compiled kernel dir the `alto_config.json` the runner insists on.

    The vision build compiles with `emit_alto_config=True` and so writes one; the text
    build does not, and `tests/integration/alto/utils.run` refuses a directory without it.
    Only four fields of it are read at RUN time (target, generation, num_cores,
    resources) -- the rest are compile-time flags -- so a minimal one is enough, and the
    core count is recovered by counting the per-core ELFs the compile actually emitted.
    Written next to the model it describes; a no-op when the build already wrote one.
    """
    kernel_dir = Path(kernel_dir)
    config_path = kernel_dir / "alto_config.json"
    if config_path.exists():
        return
    cores = max(1, len(list(kernel_dir.glob("ai_core_kernel_*.elf"))))
    config_path.write_text(
        json.dumps(
            {
                "target": "device",
                "generation": "omega",
                "num_cores": cores,
                "resources": 1.0,
            },
            indent=1,
        )
    )
    logger.info("wrote a minimal alto_config.json (%d cores) for %s", cores, kernel_dir.name)


def run_compiled(kernel_dir: Path, *, iterations: int = 20, level: str = "DEVICE_MAIN"):
    """Run one already-compiled kernel under the profiler. Returns (per-core cycles, freq, trace).

    No compile: this is the exact binary the artifact ships, so the number is the one the
    pipeline pays rather than a rebuild that might schedule differently.
    """
    from axelera.compiler.alto.compiler.config import AltoRuntimeConfig, ProfilingLevel

    from ....utils import run as alto_run

    kernel_dir = Path(kernel_dir)
    ensure_alto_config(kernel_dir)
    spec = json.loads((kernel_dir / "model.json").read_text())
    out_dir = Path(tempfile.mkdtemp(prefix=f"prof_{kernel_dir.name}_"))
    with _capture_trace() as trace:
        alto_run(
            kernel_dir / "model.json",
            _synth_inputs(spec),
            AltoRuntimeConfig(
                profiling_levels=[getattr(ProfilingLevel, level)],
                iterations=iterations,
                profiling_output_dir=out_dir,
            ),
        )
    cycles, frequency = [], 800e6
    for path in sorted(out_dir.glob("perf_stats_*.json")):
        main = json.loads(path.read_text())["perf_stats"].get("device_main") or {}
        value = _valid_cycles(main.get("median_cycles"))
        if value is not None:
            cycles.append(value)
            frequency = float(main.get("frequency") or frequency)
    return cycles, frequency, list(trace)


def kernel_ms(kernel_dir: Path, *, iterations: int = 20) -> float:
    """Wall-clock ms for one dispatch of a compiled kernel.

    The MAX over cores, not the mean: a multicore kernel finishes when its slowest core
    does, so that is the latency the graph pays.
    """
    cycles, frequency, _ = run_compiled(kernel_dir, iterations=iterations)
    return max(cycles) / frequency * 1e3 if cycles else float("nan")


def dma_profile(kernel_dir: Path, *, iterations: int = 3) -> dict:
    """Bytes moved and DMA transfers issued per dispatch, off the firmware task log.

    Timing under this level is inflated (the logging costs more than the work), so only
    the COUNTS are read here -- they are exact, and they are what says whether a kernel is
    bandwidth-bound. Pair the bytes with `kernel_ms` from a clean run for GB/s.
    """
    _, _, trace = run_compiled(kernel_dir, iterations=iterations, level="DEVICE_PERF_EVAL")
    total_bytes = transfers = 0
    for line in trace:
        if line.startswith("Start DMA transfer") and "size:" in line:
            transfers += 1
            try:
                total_bytes += int(line.split("size:")[1].split(",")[0].split(")")[0])
            except (IndexError, ValueError):
                pass
    return {"bytes": total_bytes / iterations, "transfers": transfers / iterations}


def _read_json(path: Path):
    """One of the artifact's own JSON records, or None with a warning if it is missing."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning(
            "%s: %s -- assuming one dispatch for the kernels it would have counted", path, exc
        )
        return None


def kernel_counts(artifact_dir: Path) -> dict[str, int]:
    """Dispatches per frame / per decode step, read off the artifact that will be timed.

    The multipliers are what turn a per-kernel time into a share of the thing a user waits
    for, and they differ per checkpoint -- so they come from the built artifact rather than
    from a constant that happens to be one model's. Both towers already record what they
    were compiled as: the vision tower's `meta.json` carries the config it was built from
    (`depth` blocks, one DeepStack tap per `deepstack_visual_indexes` entry) and the
    decoder's `pipeline.json` carries its spine (`layer_kinds`, one entry per dispatch, so
    the layer and DeepStack-add counts are exact rather than inferred).

    A file that is not there leaves that kernel at ONE dispatch and says so: an unknown
    multiplier must not silently scale a measured time into a total nobody can check.
    """
    artifact_dir = Path(artifact_dir)
    counts = {name: 1 for name in VISION_KERNELS + TEXT_KERNELS}
    vision = _read_json(artifact_dir / "vision" / "meta.json")
    if vision is not None:
        config = vision.get("config") or {}
        counts["vision/_block"] = int(config.get("depth", 1))
        counts["vision/_deepstack"] = len(config.get("deepstack_visual_indexes") or ())
    text = _read_json(artifact_dir / "text" / "pipeline.json")
    if text is not None:
        kinds = text.get("layer_kinds") or []
        for phase in ("prefill", "decode"):
            counts[f"text/decode_layer_{phase}"] = sum(k == "decode_layer" for k in kinds)
            counts[f"text/deepstack_add_{phase}"] = sum(k == "deepstack_add" for k in kinds)
    return counts


def kernel_table(artifact_dir: Path, *, iterations: int = 20, dma: bool = True) -> list[dict]:
    """Every compiled kernel of an artifact: ms/dispatch, count, share, and DMA traffic.

    Costs no compiles at all -- see the module docstring.
    """
    artifact_dir = Path(artifact_dir)
    counts = kernel_counts(artifact_dir)
    rows = []
    for rel in VISION_KERNELS + TEXT_KERNELS:
        count = counts[rel]
        kernel_dir = artifact_dir / rel
        if not (kernel_dir / "model.json").exists():
            logger.warning("no compiled kernel at %s", kernel_dir)
            continue
        row = {"kernel": rel, "count": count}
        try:
            row["ms"] = kernel_ms(kernel_dir, iterations=iterations)
            if dma:
                traffic = dma_profile(kernel_dir)
                row["mb"] = traffic["bytes"] / 1e6
                row["dma_transfers"] = traffic["transfers"]
                row["gb_s"] = (traffic["bytes"] / 1e9) / (row["ms"] / 1e3) if row["ms"] else 0.0
            row["total_ms"] = row["ms"] * count
        except Exception as exc:  # noqa: BLE001 -- one bad kernel must not lose the sweep
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("kernel %s failed: %s", rel, exc)
        rows.append(row)
    return rows


# --- compiling a variant (the per-operator route) ------------------------------------


def compile_and_measure(
    module,
    *,
    cores: int = 4,
    promote: bool = True,
    iterations: int = 20,
    generation: str = "OMEGA",
    inputs=None,
    return_outputs: bool = False,
):
    """Compile one alto module for DEVICE exactly as the artifact does, and time it.

    Mirrors `vision_impl.compile_kernel` (core count, `resources` share, tiling, buffer
    promotion) so an ablation is compared against the schedule that actually ships, not a
    default-configured rebuild of it.

    Pass `inputs` to pin the data (the same arrays across two builds make a correctness
    A/B) and `return_outputs=True` to get `(ms, outputs)` back. That device A/B is the only
    correctness gate available for the vision block: `layer_benchmark.rel_l2_vs` interprets
    on the host, and the host interpreter has no emulation for `axol.core.convolution`,
    which every projection in this kernel lowers to.
    """
    from axelera.compiler.alto.compiler.config import (
        AltoConfig,
        AltoRuntimeConfig,
        HardwareGeneration,
        ProfilingLevel,
        Target,
    )
    from axelera.compiler.alto.compiler.device.hw_specs import NUM_AI_CORES
    from axelera.compiler.alto.compiler.driver import compile as alto_compile

    from ....utils import create_inputs, run as alto_run

    hw = HardwareGeneration[str(generation).upper()]
    out_dir = Path(tempfile.mkdtemp(prefix="ablate_"))
    model_path = alto_compile(
        module,
        AltoConfig(
            output_dir=out_dir,
            generation=hw,
            target=Target.DEVICE,
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
    if inputs is None:
        inputs = create_inputs(module)
    with _capture_trace():
        outputs = alto_run(
            model_path,
            inputs,
            AltoRuntimeConfig(
                profiling_levels=[ProfilingLevel.DEVICE_MAIN],
                iterations=iterations,
                profiling_output_dir=out_dir,
            ),
        )
    cycles, frequency = [], 800e6
    for path in sorted(Path(out_dir).rglob("perf_stats_*.json")):
        main = json.loads(path.read_text())["perf_stats"].get("device_main") or {}
        value = _valid_cycles(main.get("median_cycles"))
        if value is not None:
            cycles.append(value)
            frequency = float(main.get("frequency") or frequency)
    if not cycles:
        raise RuntimeError(f"no device profile written under {out_dir}")
    ms = max(cycles) / frequency * 1e3
    return (ms, outputs) if return_outputs else ms


# --- the vision block's operators ----------------------------------------------------
#
# Each entry patches ONE helper with a shape- and dtype-compatible stand-in, so the
# kernel still builds and still schedules, minus that operator's work. The patch targets
# are the modules the helpers live in, because `make_vision_block_kernel` imports them
# INSIDE the function -- the lookup happens when the kernel is built, so patching the
# source module is what takes effect.


@contextlib.contextmanager
def _ablate_vision(op: str):
    """Neutralise one operator of the vision block for the duration of a kernel build."""
    from axelera.compiler.alto import axol
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops import layer_norm_dpu_reduce as ln_mod
    from ...ops import mshard_matmul_residual as resid_mod
    from ..dino import optimized_impl as dino_mod
    from . import vision_impl

    patches: list[tuple[object, str, object]] = []

    if op == "rope":
        # The documented one: y = x*cos + rotate_half(x)*sin becomes y = x, so the
        # swizzle and the multiply-add both go.
        patches.append((vision_impl, "_rope", lambda x, cos, sin, perm: x))
    elif op == "sdpa":
        # Attention output has v's shape and (thanks to the fused output quantize) v's
        # dtype, so passing v through is shape-legal. Removes QK^T, the softmax LUT
        # chain and P@V.
        def _no_sdpa(q, k, v, *args, **kwargs):
            return axol.atomic.dpu_cast(v, output=BufferInfo(dtype="int8x64"))

        patches.append((axol.core, "scaled_dot_product_attention", _no_sdpa))
    elif op == "layernorm":
        # Both LayerNorms emit int8 directly (their output quantize is fused in), so the
        # stand-in has to be a cast, not a passthrough -- that keeps the dtype contract
        # and prices the reduce, not the quantize.
        def _no_ln(x, lut, eps, *, output_dtype=None, **kwargs):
            return axol.atomic.dpu_cast(x, output=BufferInfo(dtype=output_dtype or "int8x64"))

        patches.append((ln_mod, "layer_norm_dpu_reduce_mshard", _no_ln))
    elif op == "mlp":
        # fc1 + GELU + fc2 + the residual add, replaced by a copy of the residual they
        # would have added to. It has to be a COPY, not the buffer itself: handing back an
        # existing buffer makes one variable both the block's input and its output, and
        # AssignDefaultBufferScopes rejects that ("Conflicting buffer kinds"). The copy is
        # a DDR-to-DDR move of one activation (256x1024 fp16 = 0.5 MB), so this ablation
        # UNDER-states the MLP's cost by that move rather than over-stating it.
        def _no_mlp(f2, *args, out_dtype=None, **kwargs):
            return axol.atomic.dpu_cast(args[5], output=BufferInfo(dtype=out_dtype))

        patches.append((dino_mod, "_mlp_out", _no_mlp))
    elif op == "o_proj":
        # o_proj + its residual add -> a copy of the residual, for the same aliasing reason.
        def _no_o_proj(a, b, dq, residual, *, out_dtype=None, **kwargs):
            return axol.atomic.dpu_cast(residual, output=BufferInfo(dtype=out_dtype))

        patches.append((resid_mod, "mshard_matmul_residual", _no_o_proj))
    else:
        raise ValueError(f"unknown vision ablation {op!r}")

    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    try:
        for obj, name, replacement in patches:
            setattr(obj, name, replacement)
        yield
    finally:
        for obj, name, original in saved:
            setattr(obj, name, original)


VISION_OPS = ("rope", "sdpa", "layernorm", "mlp", "o_proj")


def ablate_vision_block(
    *, cores: int = 4, iterations: int = 20, ops=VISION_OPS, image_size: int = 256
) -> list[dict]:
    """Price each operator of the vision block by removing it and re-measuring.

    Returns the baseline row plus one row per operator: `delta_ms` is what that operator
    costs inside the real kernel, and the leftover (baseline minus the deltas) is what the
    ablations do not reach -- the q/k/v projections and the head transposes.
    """
    from .qwen3vl import Qwen3VLVisionConfig
    from . import vision_impl

    config = Qwen3VLVisionConfig(grid_h=image_size // 16, grid_w=image_size // 16)
    baseline = compile_and_measure(
        vision_impl.make_vision_block_kernel(config, cores), cores=cores, iterations=iterations
    )
    rows = [{"op": "baseline (whole block)", "ms": baseline, "delta_ms": 0.0, "share": 1.0}]
    for op in ops:
        try:
            with _ablate_vision(op):
                module = vision_impl.make_vision_block_kernel(config, cores)
            without = compile_and_measure(module, cores=cores, iterations=iterations)
            rows.append(
                {
                    "op": op,
                    "ms": without,
                    "delta_ms": baseline - without,
                    "share": (baseline - without) / baseline if baseline else 0.0,
                }
            )
            logger.info("ablate %-12s %.3f ms -> costs %.3f ms", op, without, baseline - without)
        except Exception as exc:  # noqa: BLE001 -- keep the sweep alive
            rows.append({"op": op, "error": f"{type(exc).__name__}: {exc}"})
            logger.warning("ablation %s failed: %s", op, exc)
    attributed = sum(r.get("delta_ms", 0.0) for r in rows[1:])
    rows.append(
        {
            "op": "unattributed (q/k/v + transposes)",
            "delta_ms": baseline - attributed,
            "share": (baseline - attributed) / baseline if baseline else 0.0,
        }
    )
    return rows


# --- the lm_head, host against device ------------------------------------------------


def lm_head_host(vocab: int = 151936, hidden: int = 2048, repeats: int = 30) -> dict:
    """Time the host lm_head exactly as `implementation.compute_logits` runs it.

    fp16 weights, fp32 activation cast down, `torch.mv`, result back to fp32.
    """
    import torch

    weight = torch.randn(vocab, hidden, dtype=torch.float16)
    activation = torch.randn(hidden, dtype=torch.float32)
    times = []
    for _ in range(repeats + 3):
        started = time.perf_counter()
        logits = torch.mv(weight, activation.to(torch.float16)).to(torch.float32)
        times.append(time.perf_counter() - started)
    times = times[3:]  # warm only: the first calls fault the weight in
    median = statistics.median(times)
    argmax_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        int(torch.argmax(logits))
        argmax_times.append(time.perf_counter() - started)
    return {
        "ms": median * 1e3,
        "argmax_ms": statistics.median(argmax_times) * 1e3,
        "weight_gb": vocab * hidden * 2 / 1e9,
        "gb_s": (vocab * hidden * 2 / 1e9) / median,
    }


# --- reporting -----------------------------------------------------------------------


def format_kernels(rows: list[dict]) -> str:
    out = [
        f"{'kernel':32s} {'ms':>8s} {'x':>4s} {'total ms':>9s} {'MB':>8s} {'GB/s':>7s}",
        "-" * 74,
    ]
    for row in rows:
        if "error" in row:
            out.append(f"{row['kernel']:32s}  {row['error']}")
            continue
        out.append(
            f"{row['kernel']:32s} {row['ms']:8.3f} {row['count']:4d} {row['total_ms']:9.1f} "
            f"{row.get('mb', float('nan')):8.2f} {row.get('gb_s', float('nan')):7.2f}"
        )
    frame = sum(r["total_ms"] for r in rows if r["kernel"].startswith("vision") and "ms" in r)
    step = sum(r["total_ms"] for r in rows if r["kernel"].endswith("_decode") and "ms" in r)
    out += [
        "-" * 74,
        f"vision frame (sum of kernels): {frame:.1f} ms",
        f"decode step  (sum of kernels): {step:.1f} ms",
    ]
    return "\n".join(out)


def format_ablation(rows: list[dict]) -> str:
    out = [f"{'operator':34s} {'kernel ms':>10s} {'costs ms':>9s} {'share':>7s}", "-" * 64]
    for row in rows:
        if "error" in row:
            out.append(f"{row['op']:34s}  {row['error']}")
            continue
        ms = row.get("ms")
        out.append(
            f"{row['op']:34s} {ms if ms is not None else float('nan'):10.3f} "
            f"{row.get('delta_ms', float('nan')):9.3f} {row.get('share', 0) * 100:6.1f}%"
        )
    return "\n".join(out)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen3vl profiling", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Built artifact; its kernels are timed without recompiling.",
    )
    parser.add_argument("--kernels", action="store_true", help="per-kernel table (0 compiles)")
    parser.add_argument(
        "--ablate",
        choices=("vision",),
        default=None,
        help="per-operator attribution (one compile per operator)",
    )
    parser.add_argument(
        "--lm-head",
        action="store_true",
        help="time the lm_head on the host AND as a device matmul",
    )
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--no-dma", action="store_true", help="skip the DMA/traffic pass")
    parser.add_argument("--json", type=Path, default=None, help="also write results here")
    parser.add_argument("--log-level", default="WARNING")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    results: dict = {}

    if args.kernels:
        if not args.artifact_dir:
            raise SystemExit("--kernels needs --artifact-dir")
        rows = kernel_table(args.artifact_dir, iterations=args.iterations, dma=not args.no_dma)
        results["kernels"] = rows
        print("\n== per kernel (no compiles: the artifact's own binaries) ==")
        print(format_kernels(rows))

    if args.ablate == "vision":
        rows = ablate_vision_block(
            cores=args.cores, iterations=args.iterations, image_size=args.image_size
        )
        results["vision_ablation"] = rows
        print(f"\n== vision block, per operator ({args.cores} cores) ==")
        print(format_ablation(rows))

    if args.lm_head:
        host = lm_head_host()
        results["lm_head_host"] = host
        print(
            f"\n== lm_head ==\nhost   {host['ms']:.2f} ms  "
            f"({host['weight_gb']:.2f} GB fp16, {host['gb_s']:.1f} GB/s), "
            f"argmax {host['argmax_ms']:.2f} ms"
        )

    if args.json:
        args.json.write_text(json.dumps(results, indent=1, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
