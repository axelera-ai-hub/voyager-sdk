# Copyright Axelera AI, 2026
"""Vendored subset of the ALTO integration test harness.

Source: axelera-ai/host.compiler.alto, axelera_compiler/tests/integration/alto/utils.py,
revision bbcd69fa93.

Trimmed to the functions the imported op tests in model_axscript/llm/tests/
actually call (compile_and_run, create_inputs, set_seed, and their helpers run/
get_io_info/make_data). Everything else in the upstream file (covertable-based
pairwise sampling, ParamDict, assert_close_in_aggregate, closest_multiple,
flatten_pwords, sample_int_space) is unused here and was left out rather than
carried over unread.

One local deviation: ``run`` connects to the generation-matched device and pins
``TritonTrace._DEVICE_HANDLE`` to it, since axtrace enumerates devices on its own and
would otherwise trace a multi-device host's default device.
"""

import os
from pathlib import Path
import random
from typing import Any, Callable

import axelera.aipu.atex as aipu_atex
from axelera.compiler.alto.compiler.annotations import NUM_INPUTS, NUM_OUTPUTS
from axelera.compiler.alto.compiler.config import (
    AltoConfig,
    AltoRuntimeConfig,
    ProfilingLevel,
    Target,
    get_subdevices_used,
)
from axelera.compiler.alto.compiler.device import NUM_AI_CORES
from axelera.compiler.alto.compiler.driver import compile
from axelera.compiler.alto.compiler.dtypes import unpack_vector_dtype
from axelera.compiler.alto.compiler.passes import LiftIOToParams
from axelera.compiler.config import ProfilingLevel as LegacyProfilingLevel
from axelera.compiler.config import Target as LegacyTarget
from axelera.compiler.utils.profiling import PerformanceProfiling
from axelera.compiler.utils.profiling_report import dump_device_report, dump_host_report
from axelera.compiler.utils.triton_trace import TritonTrace
from axelera.config import HardwareGeneration as LegacyHardwareGeneration
import axelera.runtime as axe_runtime
import numpy as np
import torch
import tvm
from tvm.relay.backend.contrib.axelera.utils.float18 import np_round_float32_to_float18


def make_data(
    shape: tuple[int, ...],
    dtype: str,
    interval: tuple[int | float, int | float] | None = None,
    allow_subnormal: bool = False,
    round_float18: bool = True,
) -> np.ndarray:
    """Generate test data with deterministic random values."""

    np_dtype = np.dtype(dtype)
    is_integer = np.issubdtype(np_dtype, np.integer)
    is_floating = np.issubdtype(np_dtype, np.floating)

    if not (is_integer or is_floating):
        raise ValueError(f"Unsupported dtype: {dtype}")

    if interval is None:
        if is_integer:
            if np_dtype == np.uint64:
                interval = (0, np.iinfo(np.int64).max + 1)
            else:
                info = np.iinfo(np_dtype)
                interval = (info.min, info.max + 1)
                interval = (-10, 10)
        else:  # is_floating
            interval = (-1.0, 1.0)

    low, high = interval
    size = int(np.prod(shape))
    if is_integer:
        data = np.random.randint(int(low), int(high), size=size, dtype=np.int64)
    else:
        data = np.random.uniform(float(low), float(high), size=size)  # type: ignore

    data = data.reshape(shape).astype(np_dtype)

    # The Omega DPU does not support subnormal numbers due to a hardware bug.
    # The simulator threshold (FP16_TINY = 6.104e-05) is slightly above the
    # IEEE float16 smallest normal (6.103515625e-05), so also flush the
    # smallest normal value to avoid false positives.
    if is_floating and not allow_subnormal:
        finfo = np.finfo(np_dtype)
        threshold = finfo.smallest_normal * (1.0 + finfo.eps)
        subnormal_mask = (data != 0) & (np.abs(data) < threshold)
        data = data.copy()
        data[subnormal_mask] = 0

    # Ensure the float data we pass to the DPU is representable in float18.
    if is_floating and round_float18:
        data = np_round_float32_to_float18(data)

    return data


def _extract_buffer_info(
    buf: tvm.tir.Buffer,
    shapes: list[tuple[int, ...]],
    dtypes: list[str],
) -> None:
    shape = tuple(int(v) for v in buf.shape)
    dtype = str(buf.dtype)
    shape, dtype = unpack_vector_dtype(shape, dtype)
    shapes.append(shape)
    dtypes.append(dtype)


def get_io_info(
    mod: tvm.IRModule,
) -> tuple[list[tuple[int, ...]], list[str], list[tuple[int, ...]], list[str]]:
    """Extract input and output shapes and dtypes from an IRModule."""

    mod = LiftIOToParams()(mod)
    main_func = mod["main"]
    attrs = dict(main_func.attrs)
    num_in = int(attrs.get(NUM_INPUTS, 0))
    num_out = int(attrs.get(NUM_OUTPUTS, 0))
    if num_in + num_out != len(main_func.params):
        raise ValueError(
            f"Expected {num_in} inputs and {num_out} outputs, "
            f"but found {len(main_func.params)} params on main func."
        )

    in_shapes: list[tuple[int, ...]] = []
    in_dtypes: list[str] = []
    out_shapes: list[tuple[int, ...]] = []
    out_dtypes: list[str] = []

    for param in main_func.params[:num_in]:
        _extract_buffer_info(main_func.buffer_map[param], in_shapes, in_dtypes)
    for param in main_func.params[num_in:]:
        _extract_buffer_info(main_func.buffer_map[param], out_shapes, out_dtypes)

    return in_shapes, in_dtypes, out_shapes, out_dtypes


def create_inputs(
    mod_or_input_infos: tvm.IRModule | list[tuple[tuple[int, ...], str]],
    data_generator: Callable | None = None,
    interval: tuple[int | float, int | float] | None = None,
    additional_specs: list[tuple[tuple[int, ...], str]] | None = None,
) -> list[np.ndarray]:
    """Generate input arrays for module parameters."""

    if isinstance(mod_or_input_infos, tvm.IRModule):
        input_shapes, input_dtypes, _, _ = get_io_info(mod_or_input_infos)
    else:
        input_shapes, input_dtypes = zip(*mod_or_input_infos)

    data_generator = data_generator or make_data

    all_specs = list(zip(input_shapes, input_dtypes)) + (additional_specs or [])

    inputs = []
    for in_shape, in_dtype in all_specs:
        inputs.append(data_generator(in_shape, in_dtype, interval))
    return inputs


def _device_for_generation(devices: list, generation: Any) -> Any | None:
    """Device whose runtime HWGeneration matches the compiled *generation*, or None.

    ``axe_runtime.axelera_load_model`` connects to whatever device enumerates
    first, with no generation filter -- on a host with more than one generation
    attached (e.g. a metis card alongside europa cards) that can silently hand an
    OMEGA-only kernel to a europa device, failing deep in native module load
    rather than here. None (nothing attached matches, or an unmapped generation)
    leaves device_connect's own default-device behaviour untouched, so a normal
    single-generation test host is unaffected.
    """
    hw_name = {"omega": "metis", "europa": "europa"}.get(generation.value)
    return next((d for d in devices if d.hw_generation.name == hw_name), None)


def run(model_path: Path, inputs: list, rt_config: AltoRuntimeConfig | None = None) -> list:
    """Run a TIR function on the AIPU. Only used for testing and profiling."""

    if model_path.suffix == ".axm":
        model_path = model_path.parent / "model.json"
    if not model_path.exists():
        raise ValueError(f"Model file not found at {model_path}")

    if not rt_config:
        rt_config = AltoRuntimeConfig()
    if not rt_config.profiling_output_dir:
        rt_config.profiling_output_dir = model_path.parent

    # Read-back the AltoConfig from the output directory
    alto_config_path = model_path.parent / "alto_config.json"
    alto_config = None
    if alto_config_path.exists():
        alto_config = AltoConfig.load_json(alto_config_path)
    if not alto_config:
        raise ValueError(f"Could not find a valid AltoConfig at {alto_config_path}. ")

    perf_profiler: PerformanceProfiling | None = None
    if rt_config.profiling_levels and alto_config.target == Target.DEVICE:
        levels = [LegacyProfilingLevel(lvl.to_capture) for lvl in rt_config.profiling_levels]
        perf_profiler = PerformanceProfiling(
            target=LegacyTarget(alto_config.target),
            generation=LegacyHardwareGeneration(alto_config.generation),
            num_processes=1,
            process_id=0,
            num_ai_cores=alto_config.num_cores,
            resources=alto_config.resources,
            model_name="alto_model",
            iterations=rt_config.iterations,
            profiling_levels=levels,
        )

    # resources is a fraction of the generation's cores, so hardcoding 4 here capped the
    # count and a kernel whose barriers span more cores than that hung the device.
    num_sub_devices = len(
        get_subdevices_used(
            alto_config.resources, aipu_cores_max=NUM_AI_CORES[alto_config.generation]
        )
    )

    runtime_params: dict[str, Any] = {
        "aipu_cores": alto_config.num_cores,
        "num_sub_devices": num_sub_devices,
        "target": alto_config.target.to_runtime(),
        "double_buffer": False,
        "stream_tasklist": True,
        "elf_in_ddr": False,
        "host_profiling": bool(perf_profiler and perf_profiler.is_host_tracing_required()),
    }

    # Each stage of a synchronous pipeline needs an extra iteration to fill/drain.
    atex_path = model_path.parent / "model.atex"
    if atex_path.exists():
        atex = aipu_atex.from_file(model_path.parent / "model.atex")
        if isinstance(atex, aipu_atex.AbstractSynchronousPipeline):
            rt_config.iterations += len(atex.stages) - 1

    # axelera_load_model() has no device-selection parameter, so pin the generation-matched
    # device ourselves (see _device_for_generation) instead of going through it -- this is
    # its own body inlined, plus an explicit `device=`.
    ctx = axe_runtime.Context()
    device = None
    if alto_config.target == Target.DEVICE:
        device = _device_for_generation(ctx.list_devices(), alto_config.generation)
    if device is not None:
        TritonTrace._DEVICE_HANDLE = device.name
    model = ctx.load_model(f"{model_path.parent}/{model_path.name}")
    target = runtime_params.pop("target", "silicon")
    conn = ctx.device_connect(
        device=device, num_sub_devices=runtime_params.get("num_sub_devices", 1), target=target
    )
    instance = conn.load_model_instance(model, **runtime_params)
    executor = axe_runtime.tvm_compat.GraphExecutor(ctx, model, instance)

    for idx, value in enumerate(inputs):
        executor.set_input(idx, value)

    if perf_profiler:
        perf_profiler.start_tracing()

    for _ in range(rt_config.iterations):
        executor.run()

    if perf_profiler:
        perf_profiler.stop_tracing()
        profiling_dir = str(rt_config.profiling_output_dir)
        stats_jsons = perf_profiler.save_as_json(dir=profiling_dir)
        stats_path = Path(stats_jsons[0])
        if ProfilingLevel.DEVICE_REPORT in rt_config.profiling_levels:
            dump_device_report(stats_path)
        if ProfilingLevel.HOST_REPORT in rt_config.profiling_levels:
            dump_host_report(stats_path)

    outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]

    executor.release()
    return outputs


def compile_and_run(
    mod: tvm.IRModule,
    config: AltoConfig,
    config_runtime: AltoRuntimeConfig | None = None,
    inputs: list[np.ndarray] | None = None,
    data_generator: Callable | None = None,
    no_outputs: bool = False,
    check_saturation: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compile module and execute on AIPU with test inputs."""

    config.clean_output_dir = True
    config.unpack_axm_archive = True
    config.emit_alto_config = True
    config.emit_atex_artifacts = True
    # The three diagrams are DEBUGGING AIDS and are off unless asked for. Each is a PDF
    # rendered per compile into the per-test `rundir`, which is a `tmp_path` that nothing reads
    # afterwards, so a test run paid for them and threw them away. Worse, they are rendered
    # through matplotlib, which nothing in the wheel graph declares: on a machine without it
    # `omnimalloc.plot_allocation` raises INSIDE the compile, every device test fails, and the
    # ImportError REPLACES whatever the compiler was about to say about the kernel. Set
    # AXSCRIPT_EMIT_DIAGRAMS=1 when investigating one compile and they come back.
    emit_diagrams = os.environ.get("AXSCRIPT_EMIT_DIAGRAMS", "") not in ("", "0")
    config.emit_allocation_diagram = emit_diagrams
    config.emit_memory_diagram = emit_diagrams
    config.emit_dependency_diagram = emit_diagrams
    config.emit_input_axscript = True

    model_path = compile(mod, config)

    inputs = inputs or create_inputs(mod, data_generator)
    if not inputs:
        raise ValueError("No inputs. Ensure inputs are provided or generated.")

    outputs = run(model_path, inputs, config_runtime)

    if not outputs and not no_outputs:
        raise ValueError("No outputs. Ensure the model produces outputs.")

    # Naive check to ensure that we are not just getting min/max/zero values
    if check_saturation and len(np.unique(outputs)) <= 3:
        raise ValueError(f"Output tensor appears saturated, unique vals: {np.unique(outputs)}")

    return inputs, outputs


def set_seed(seed: int = 42) -> None:
    """Set random seeds and ensure reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)

    # Only applicable when running on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
