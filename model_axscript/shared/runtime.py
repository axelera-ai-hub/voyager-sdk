# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
# The one runtime-plumbing home (RuntimeModel/_select_device/probe_runtime_load), shared by every family driver.
"""Runtime plumbing: load + run a compiled artifact on the Axelera runtime executor.

RuntimeModel / _select_device / probe_runtime_load / _NP_DTYPE are family-neutral -
the LLM KV-cache path and the vision single-graph path both invoke the same executor
seam (a named graph run via instance._run_graph with a pre-allocated output buffer).
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

import numpy as np

from model_axscript.shared.run_sizes import RUN_SIZE_KEYS

logger = logging.getLogger(__name__)


def required_cores(spec: dict) -> int:
    """How many AI cores a compiled build spans.

    A build's per-graph ``call_graphs`` list has one entry per AI core the graph was
    compiled onto: 1 for a single-core build, N for an N-core shard. The max over graphs
    is the core count. Defaults to 1 when no graph declares ``call_graphs``.

    This is the core SPAN, and it is a lower bound on the reservation rather than the
    whole answer - see ``required_sub_devices``, which is what a loader should call.
    """
    return max(
        (len(g.get("call_graphs") or [None]) for g in spec.get("graphs", {}).values()),
        default=1,
    )


def required_pool_bytes(spec: dict, tier: str) -> int:
    """Total bytes the build's declared memory pools ask for in one memory ``tier``.

    Both tiers a reservation scales are asked the same way, because the arithmetic that
    turns them into a sub-device count is the same: ``l2`` is the on-chip SRAM arena and
    ``ddr`` is the off-chip pool, and a build whose demand in EITHER exceeds one
    sub-device's share cannot load at a narrower reservation than that demand implies.
    """
    tier = tier.lower()
    return sum(
        int(pool.get("size") or 0)
        for pool in spec.get("memory_pools", []) or []
        if str(pool.get("memory", "")).lower() == tier
    )


def required_l2_bytes(spec: dict) -> int:
    """Total L2 (on-chip SRAM) the build's declared memory pools ask for."""
    return required_pool_bytes(spec, "l2")


def required_ddr_bytes(spec: dict) -> int:
    """Total DDR the build's declared memory pools ask for.

    Dominated by the outlined weights, so it is roughly the model's int8 size plus the KV
    cache and the prefill scratch - and unlike L2 it is large enough that a small model can
    still exceed a single sub-device's share while spanning one core.
    """
    return required_pool_bytes(spec, "ddr")


def _sub_devices_for_pool(needed: int, total: int, subdevices: int, tier: str, device) -> int:
    """How many sub-devices one memory tier's demand implies, capped at the device's width.

    Both tiers are partitioned PER RESERVED SUB-DEVICE, so the share available is
    ``total / subdevices`` times the reservation, and the count a demand implies is the
    ceiling of the ratio. Over capacity is capped and warned rather than silently clamped,
    so the runtime's own allocator names the shortfall instead of this hiding it.

    An UNKNOWN capacity is not an absent demand. ``DeviceInfo.max_memory`` is documented as
    "not populated and will always be 0" - it IS populated on the parts measured here, but a
    part or firmware where it reads 0 would otherwise make this arm vanish silently, and the
    reservation fall back to the span. That is exactly how qwen3.5-2b reserved one sub-device
    and died in memory-pool setup naming `w_oproj_L17`. Sizing is impossible either way, so
    the fallback stands - but it is said out loud rather than inferred from a crash.
    """
    if needed <= 0 or subdevices <= 0:
        return 0
    if total <= 0:
        logger.warning(
            "%s reports no %s capacity, so the %.2f MiB this build asks for cannot be sized "
            "into a reservation; falling back to the core span, which may under-reserve and "
            "fail in memory-pool setup naming a pool",
            getattr(device, "name", device),
            tier.upper(),
            needed / 2**20,
        )
        return 0
    per_sub_device = total // subdevices
    count = -(-needed // per_sub_device)  # ceil
    if count > subdevices:
        logger.warning(
            "build asks for %.2f MiB of %s but %s holds only %.2f MiB across %d sub-devices; "
            "reserving all of them, so the load will fail in memory-pool setup",
            needed / 2**20,
            tier.upper(),
            getattr(device, "name", device),
            total / 2**20,
            subdevices,
        )
        count = subdevices
    return count


def required_sub_devices(spec: dict, device=None) -> int:
    """How many sub-devices to reserve for a build: core span, L2 or DDR - whichever binds.

    Three independent demands, all read off the BUILD, never hand-picked:

    * the core SPAN (``required_cores``) - reserving fewer under-loads a shard (DEVICE_LOST);
    * the L2 ARENA (``required_l2_bytes``) - on-chip SRAM;
    * the DDR POOLS (``required_ddr_bytes``) - the outlined weights, KV cache and scratch.

    Both memory tiers are partitioned per reserved sub-device, so a build can span one core
    and still need a wider reservation for either. Neither case is hypothetical:

    * a kernel builder that ignores a ``cores`` argument emits a single-core program while
      ``AltoConfig.num_cores`` still sizes its L2 arena for the requested count (the three
      LFM2 builds), so it spans 1 core and needs 3 cores' L2;
    * a kernel that does not shard at all still has to fit its WEIGHTS somewhere, and
      qwen3.5-2b spans 1 core with 0 bytes of L2 while asking 1474 MiB of DDR against the
      ~1016 MiB one sub-device provides.

    Both failed in memory-pool setup naming a pool (`shared_l2`, `w_oproj_L17`), which reads
    as a broken artifact rather than an under-reservation - so getting this wrong costs a
    debugging session, not an error message.

    Reserving for memory is still a CONSEQUENCE of the compile, so this is not the
    "reserve N, compile 1" pattern the bring-up contract rejects: nothing here is a
    constant, and both per-sub-device shares come from the device, not from a guess.

    ``device`` is a runtime ``DeviceInfo``; without one (or without its capacity fields, as
    on a simulator) only the span is knowable and it is returned unchanged.
    """
    span = required_cores(spec)
    subdevices = int(getattr(device, "subdevice_count", 0) or 0)
    if subdevices <= 0:
        return span
    demands = {
        "l2": (required_l2_bytes(spec), int(getattr(device, "l2_size", 0) or 0)),
        "ddr": (required_ddr_bytes(spec), int(getattr(device, "max_memory", 0) or 0)),
    }
    counts = {
        tier: _sub_devices_for_pool(needed, total, subdevices, tier, device)
        for tier, (needed, total) in demands.items()
    }
    widest = max(counts.values(), default=0)
    if widest > span:
        tier = max(counts, key=lambda t: counts[t])
        needed, total = demands[tier]
        logger.info(
            "reserving %d sub-devices for a %d-core build: its %s pools need %.2f MiB, more "
            "than the %.2f MiB a %d-sub-device reservation makes available",
            widest,
            span,
            tier.upper(),
            needed / 2**20,
            span * (total // subdevices) / 2**20,
            span,
        )
    return max(span, widest)


def probe_runtime_load(artifact_dir) -> bool:
    """Confirm the runtime's own loader recognises the artifact (host-side parse).

    Complements the structural contract with the runtime's own parser - it
    catches manifest/spec fields the structural check does not model. Host-side
    only: `load_model` parses the spec, with no `device_connect` / weight transfer
    / instance load, so a plain build never becomes a device session and the
    subsequent `--run` is not a second full load. Returns False (skip + warn) when
    the runtime is not installed or no device is present; raises
    AxScriptArtifactError only when the runtime is available but rejects the
    artifact. Actually running on-device is `--run`, or the family's eval harness.
    """
    try:
        from axelera.runtime import Context
    except ImportError:
        logger.warning(
            "axelera.runtime not installed; skipped the runtime load-probe for %s", artifact_dir
        )
        return False

    # The single shared AxScript validator (task-agnostic); reused by import.
    from model_axscript.shared.artifact_contract import AxScriptArtifactError

    try:
        ctx = Context()
        devices = ctx.list_devices()
    except Exception as e:  # runtime present but unavailable (driver/init) - don't fail the build
        logger.warning("runtime unavailable (%s); skipped the load-probe for %s", e, artifact_dir)
        return False
    if not devices:
        logger.warning(
            "no Axelera device/simulator present; skipped the runtime load-probe for %s "
            "(structural contract only - run `--run` on a device to confirm it loads)",
            artifact_dir,
        )
        return False
    try:
        ctx.load_model(str(Path(artifact_dir) / "model.json"))
    except Exception as e:  # device present + runtime OK, but the artifact was rejected
        raise AxScriptArtifactError(
            f"runtime could not load {artifact_dir}/model.json: {e}"
        ) from e
    return True


# model.json dtype strings -> numpy dtypes for buffer allocation.
_NP_DTYPE = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "float16": np.float16,
    "float32": np.float32,
    "uint64": np.uint64,
}


# Run sizes worth naming in the banner: the ones that decide whether a result is
# comparable. Read from the artifact, which is the source of truth for compiled params.
_RUN_SIZE_KEYS = RUN_SIZE_KEYS


def run_banner(model_json: Path, device, num_sub_devices: int) -> str:
    """One line naming the device, the cores reserved, and the compiled run size.

    Every reported number (fps, perplexity, a kNN gap) is only comparable against another
    at the SAME run size and core count, so the run states them rather than relying on
    whoever writes the result down to remember. Values come from the artifact's
    pipeline.json, never from a build config the run may not have seen.
    """
    pipeline_path = Path(model_json).parent / "pipeline.json"
    pipeline = {}
    if pipeline_path.is_file():
        try:
            pipeline = json.loads(pipeline_path.read_text())
        except ValueError:  # a malformed artifact must not break the run
            pipeline = {}
    sizes = " ".join(f"{k}={pipeline[k]}" for k in _RUN_SIZE_KEYS if k in pipeline)
    name = getattr(device, "name", device)
    return f"[axscript] device={name} cores={num_sub_devices} {sizes}".rstrip()


# Kept for callers that still classify a re-raised failure by its message (e.g. the LLM
# context-sweep harness's own cross-device reproduction path constructs a message
# containing "DEVICE_LOST" deliberately). Not used by device_job itself any more: the
# driver's distinguishing text ("DEVICE_LOST", "Failed to wait for MSI") is printed
# straight to stderr by the C library and is never part of the Python exception, so
# matching against it here always failed silently - device_job's wrapped message never
# fired for a real device failure. See ExecutionError below for the reliable signal.
_WEDGED_DEVICE_MARKERS = ("DEVICE_LOST", "Failed to wait for MSI", "wait for MSI")


@contextlib.contextmanager
def device_job(device):
    """Turn a device-execution failure into an instruction, the way `hf_access` does for HF.

    A Metis device that stops completing jobs fails with `ZE_RESULT_ERROR_DEVICE_LOST` or an
    MSI wait timeout after the silicon timeout elapses, both surfaced to Python as a bare
    `axelera.runtime.objects.ExecutionError` with a generic message ("Failed to execute
    command lists") - the distinguishing driver text is a stderr print, not part of the
    exception. So this classifies by exception TYPE, not message content: any
    `ExecutionError` reaching here came from a job that was actually submitted to the device
    and failed to complete, which is exactly the "may be wedged" case, regardless of what its
    message says. The runtime not being importable at all means the caller could not have
    raised one, so every exception then passes through unchanged.

    ponytail: no health probe. Probing costs a connect on every load and still cannot tell a
    wedged device from a busy one; naming the device and the retry is what the caller needs.
    Pin a good device for a whole session with $AXELERA_METIS_DEVICE.
    """
    try:
        from axelera.runtime.objects import ExecutionError
    except ImportError:
        # No runtime means nothing here can BE an ExecutionError, so match nothing: the
        # caller's own failure must reach it, never a report about the missing import.
        ExecutionError = ()
    name = getattr(device, "name", device)
    try:
        yield
    except ExecutionError as e:
        raise RuntimeError(
            f"metis device {name} did not complete the job ({type(e).__name__}: {e}). "
            f"The device may be wedged - results from the others are unaffected. List them "
            f"with `axdevice`, then retry on a different one: `--device <name>`, or set "
            f"AXELERA_METIS_DEVICE=<name> to pin it for the session."
        ) from e


#  Our OMEGA/EUROPA vocabulary (GenerationConfig.generation) -> axelera.runtime's HWGeneration name.
_HW_GENERATION = {"OMEGA": "metis", "EUROPA": "europa"}


def _select_device(devices, selector, generation: str | None = None):
    """Pick ONE runtime device by name, index, or default.

    ``selector``: ``None`` => ``$AXELERA_METIS_DEVICE`` if set, else the first matching
    device; an ``int`` or all-digit ``str`` index; or a device name (e.g.
    ``"metis-0:2d:0"``). An explicit selector (flag or env var) always wins, unfiltered -
    the caller is trusted to know what they asked for. ``generation`` ("OMEGA"/"EUROPA",
    from the artifact's own pipeline.json) only narrows the *default* pick, so a mixed
    fleet does not hand an OMEGA-built artifact to a EUROPA card (or vice versa) just
    because it happened to enumerate first. Lets a caller avoid a faulty/busy device
    (device 0 is not always healthy). A Metis device is one chip with 4 cores; a model
    reserves only the cores its compiled build spans (``RuntimeModel`` derives that from
    ``call_graphs``), so a single-core model leaves the other cores free. Running one
    model across MULTIPLE devices is not supported by the executor, so a comma list is
    rejected rather than silently using only the first.
    """
    import os

    if selector is None:
        selector = os.environ.get("AXELERA_METIS_DEVICE") or None
    if selector is None:
        candidates = devices
        if generation is not None:
            hw = _HW_GENERATION.get(generation, generation.lower())
            candidates = [d for d in devices if d.hw_generation.name == hw]
            if not candidates:
                have = ", ".join(f"{d.name}({d.hw_generation.name})" for d in devices)
                raise RuntimeError(
                    f"no {generation} device present (have: {have}) - rebuild the artifact "
                    "for this fleet, or pin one explicitly with --device/$AXELERA_METIS_DEVICE."
                )
        return candidates[0]
    if isinstance(selector, str):
        if "," in selector:
            raise RuntimeError(
                "one model runs on a single Metis device (its 4 cores); running "
                "across multiple devices is not supported - pass one device name or index."
            )
        selector = int(selector) if selector.isdigit() else selector
    if isinstance(selector, int):
        # Not a bare `devices[selector]`: a negative index wraps round to a device the caller
        # never named, and an index past the end raises an IndexError that says nothing about
        # what is attached. Both fall through to the one refusal below.
        if 0 <= selector < len(devices):
            return devices[selector]
    else:
        for d in devices:
            if getattr(d, "name", None) == selector:
                return d
    names = ", ".join(getattr(d, "name", "?") for d in devices)
    raise RuntimeError(f"metis device {selector!r} not found; available: {names}")


def release_runtime(runtime) -> None:
    """Give back the device a loaded model reserved. Idempotent, and shape-agnostic.

    A process that measures TWO artifacts (a 1-core baseline and a 4-core shard, say)
    reserves sub-devices for the first and then asks for the second: without an explicit
    release that second reservation fails with `requested num_sub_devices(4) > available (4)`,
    which reads like a part too small for the build rather than a model still holding the
    part. `Context.release` frees the context and every child of it, so one call covers the
    instance and the connection.

    Written against the ATTRIBUTES rather than against a class, because two loaders reach a
    device in this tree: this module's `RuntimeModel` and the imported implementation's own.
    Both hold a `ctx`, and only one has a `close`, so a caller that reached for the method
    released one of them and raised on the other.
    """
    ctx, runtime.ctx = getattr(runtime, "ctx", None), None
    runtime.instance = runtime.connection = None
    if ctx is not None:
        ctx.release()


class RuntimeModel:
    """Thin wrapper over axelera.runtime that loads and runs the artifact.

    The model is type: "graph" (v2 executor): a named graph is invoked via
    instance._run_graph(graph, inputs, outputs) with all of that graph's inputs
    (mapped by position) and a pre-allocated output buffer. A vision encoder has a
    single "encode" graph run once per image and no state across calls; an LLM has
    "prefill" and "decode" over a cache pool the executor persists, which is what
    `reset_state` re-zeros.

    One class for both families. They had diverged into two near-identical copies whose
    only behavioural difference was that this one wraps a dispatch in `device_job`, an
    additive 1.75 us that turns a wedged device into an instruction instead of a bare
    "Failed to execute command lists".
    """

    def __init__(
        self,
        model_json: Path,
        num_sub_devices: int | None = None,
        device=None,
        *,
        share_from=None,
        generation: str | None = None,
    ):
        from axelera.runtime import Context  # device or simulator backend

        model_json = Path(model_json)
        self._spec = json.loads(model_json.read_text())
        # The core SPAN alone, published because callers ask what the artifact shards over
        # (qwen3vl's pipeline reads it). It is NOT the reservation floor: see `_resolve_width`,
        # which needs a device before it can answer that.
        self.cores = required_cores(self._spec)
        if share_from is not None:
            # Reuse another RuntimeModel's context + connection. Two models that must be
            # resident at once (a VLM's vision tower beside its decoder) share one: a second
            # Context would try to claim the same sub-devices and fail. The context belongs
            # to whoever opened it, so `close` here frees only this model's instance.
            self._owns_context = False
            self.ctx, self.connection = share_from.ctx, share_from.connection
            self._device = share_from._device
            self.generation = share_from.generation
            num_sub_devices = self._resolve_width(model_json, num_sub_devices, self._device)
            with device_job(self._device):
                self._model = self.ctx.load_model(str(model_json))
                self.instance = self.connection.load_model_instance(self._model)
            logger.info("%s", run_banner(Path(model_json), self._device, num_sub_devices))
            return
        self._owns_context = True
        self.ctx = Context()
        devices = self.ctx.list_devices()
        if not devices:
            raise RuntimeError("no Axelera device/simulator available")
        # Narrow the default pick to the artifact's own hardware generation, so a mixed
        # fleet does not hand e.g. an OMEGA-built artifact to a EUROPA card just because
        # it happened to enumerate first (an explicit `device` still overrides this).
        # pipeline.json is LLM-specific and may be absent (e.g. a vision artifact, which
        # has no OMEGA/EUROPA split yet) -- fall back to no generation filter rather than
        # fail a family this class also serves.
        # An explicit `generation` wins over the artifact's own record, so a caller can steer
        # a mixed fleet without a rebuild (this is what the chat CLI's --generation exposes).
        if generation is None:
            try:
                pipeline = json.loads((model_json.parent / "pipeline.json").read_text())
                generation = pipeline.get("generation")
            except (OSError, json.JSONDecodeError):
                pass
        #: The artifact's hardware generation ("OMEGA"/"EUROPA") as recorded in its
        #: pipeline.json, or the caller's override; None for an artifact that records none.
        self.generation = generation
        selected = _select_device(devices, device, generation=generation)
        num_sub_devices = self._resolve_width(model_json, num_sub_devices, selected)
        with device_job(selected):
            self.connection = self.ctx.device_connect(
                device=selected, num_sub_devices=num_sub_devices
            )
            # Kept, not discarded: reloading the instance is how `reset_state` re-zeros
            # the cache pool, and that needs the model again.
            self._model = self.ctx.load_model(str(model_json))
            self.instance = self.connection.load_model_instance(self._model)
        self._device = selected
        self._num_sub_devices = num_sub_devices
        self._model_json = model_json
        logger.info("%s", run_banner(model_json, selected, num_sub_devices))

    def _resolve_width(self, model_json, num_sub_devices: int | None, device) -> int:
        """The reservation to open: what the build needs, or the caller's ask if it is wider.

        Sized against the DEVICE, never the core span alone. Both memory tiers are
        partitioned per reserved sub-device, so a build that shards over ONE core can still
        need several: qwen3.5-2b spans 1 core and 0 bytes of L2 while asking 1474 MiB of DDR
        against the ~1016 MiB one sub-device provides.

        That is why the floor is `required_sub_devices` and not `required_cores`. Checking
        the span would pass an explicit `num_sub_devices=1` for that build -- and the eval
        harnesses DO pass one (`evaluation/evaluate.py`, `hf_model.py`, `regression.py`, and
        dino's `num_sub_devices or cores`) -- so the shortfall would arrive in memory-pool
        setup naming a pool, having cleared a guard that reported it checked.

        Claiming MORE stays legal: an eval harness reserves the whole generation's worth so a
        later, wider artifact loads into the same connection.
        """
        needed = required_sub_devices(self._spec, device)
        if num_sub_devices is None:
            return needed
        if num_sub_devices < needed:
            # Measured, both ways this bites: a 2-core artifact on a 1-sub-device connection
            # dies with ZE_RESULT_ERROR_DEVICE_LOST and the card needs
            # `axdevice --reload-firmware`; an under-reserved memory pool dies in setup
            # naming the pool. Refuse here, where the numbers are still legible.
            raise ValueError(
                f"{model_json} needs {needed} sub-device(s) on "
                f"{getattr(device, 'name', device)} -- core span {self.cores}, "
                f"{required_l2_bytes(self._spec) / 2**20:.2f} MiB L2, "
                f"{required_ddr_bytes(self._spec) / 2**20:.2f} MiB DDR -- but "
                f"{num_sub_devices} were requested; the connection must own every core the "
                "artifact places work on and every byte its memory pools ask for"
            )
        return num_sub_devices

    def reset_state(self) -> None:
        """Re-zero the persistent cache pool (KV + conv + recurrent state) by reloading the instance.

        The renderer marks every cache pool ``zero_init`` and the executor memsets it at
        instance load, so a reload is a reset.
        Recurrent and conv state are sequential accumulators, not position-indexed like the KV
        cache, so they must be reset at each new sequence; teacher-forced perplexity scores many
        independent windows and must reset between them or state leaks across.
        """
        self.instance.release()  # free the device memory pools first, else the reload OOMs
        try:
            self.instance = self.connection.load_model_instance(self._model)
            return
        except Exception:
            # The release has already happened, so a failed reload leaves this object
            # holding a dead instance and every later run() fails -- reconnecting is the
            # only way back. Seen on a two-graph (prefill+decode) artifact, where both
            # graphs' pools are live and the reload cannot allocate.
            logger.warning("instance reload failed after release; reconnecting the device")
        # Release the old connection FIRST: reconnecting without it leaks the sub-device
        # handles, and a few leaked connections take the whole context down with
        # ZE_RESULT_ERROR_INVALID_NULL_POINTER from zeContextCreateEx.
        try:
            self.connection.release()
        except Exception:  # noqa: BLE001 - already on the failure path
            pass
        self.connection = self.ctx.device_connect(
            device=self._device, num_sub_devices=self._num_sub_devices
        )
        self._model = self.ctx.load_model(str(self._model_json))
        self.instance = self.connection.load_model_instance(self._model)

    def close(self) -> None:
        """Release the device this model reserved. Idempotent.

        A model on a SHARED connection releases only its own instance -- the device memory
        its weights and pools hold. Tearing down the context would take the tower that
        opened it down too.
        """
        if getattr(self, "_owns_context", True):
            release_runtime(self)
        else:
            instance, self.instance = getattr(self, "instance", None), None
            if instance is not None:
                instance.release()
            self.ctx = self.connection = None
        self._model = None

    def release(self) -> None:
        """The imported LLM tree's name for `close`; delegates, so the two cannot diverge.

        Both names are load-bearing: `release` matches the `instance.release()` /
        `ctx.release()` calls this class wraps, and `close` is what the context-manager
        protocol and the vision side call. What must not come back is a second teardown
        BODY under the other name -- see `release_runtime` for the bug that caused (one
        loader with a `close`, one without, and a caller that raised on whichever it
        guessed wrong).
        """
        self.close()

    def __enter__(self) -> "RuntimeModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def graph_names(self) -> set:
        """The graphs the loaded model exposes (e.g. {"encode"}, or {"prefill", "decode"})."""
        return set(self._spec.get("graphs", {}))

    def _prepare(self, graph_name: str, named_inputs: dict) -> tuple[list, list]:
        """Order named_inputs per the graph spec and allocate the output buffers.

        The executor maps inputs AND outputs by index to the graph's lists, so we emit
        them in that order (casting each input to the declared dtype).
        """
        graph = self._spec["graphs"][graph_name]
        ordered = [
            np.ascontiguousarray(named_inputs[i["name"]], dtype=_NP_DTYPE[i["dtype"]])
            for i in graph["inputs"]
        ]
        outs = [np.zeros(tuple(o["shape"]), dtype=_NP_DTYPE[o["dtype"]]) for o in graph["outputs"]]
        return ordered, outs

    def output_names(self, graph_name: str) -> list[str]:
        """The graph's output tensor names, in the order `run` returns them."""
        return [o["name"] for o in self._spec["graphs"][graph_name]["outputs"]]

    def run(self, graph_name: str, named_inputs: dict):
        """Run a named graph; returns its output buffer, filled in place.

        A multi-output graph (one with tap kernels, e.g. the Qwen3-VL vision tower's
        DeepStack mergers) returns the list of buffers instead -- pair it with
        output_names() rather than relying on the order.
        """
        ordered, outs = self._prepare(graph_name, named_inputs)
        with device_job(self._device):
            self.instance._run_graph(graph=graph_name, inputs=ordered, outputs=outs)
        return outs[0] if len(outs) == 1 else outs
