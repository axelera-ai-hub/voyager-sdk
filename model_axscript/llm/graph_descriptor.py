# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Author the runtime model.json via the production render_graph_model.

Builds the codegen axelera.codegen.axmodel_graph descriptor (the
MultiGraphFormat schema) and lets render_graph_model render the ATEX,
lay out pools, compile the ELF, and emit the manifest. The LLM is expressed as one
KernelCall per decoder layer (or per group of layers a stacked kernel runs in one
call); each layer names the kernel for its kind (the
structural layer type — e.g. "full_attn" vs "deltanet"). A uniform stack is just
the degenerate case of a single kind. Per-layer constants + each layer's
persistent state are phase-independent shared_tensors.

Two descriptor contracts (validated by the engine):
  * a kernel's arguments must be ordered as axr_model.inputs + outputs and
    match each IO's dtype + byte size (validated positionally), so we render the
    ATEX here to read that canonical order;
  * each per-layer KernelCall binds those IO, in the same order, to the
    layer's concrete tensor names.

The kernel symbol in the descriptor is "{phase}_{kind}" — unique across the
merged source even when one kind appears in several phases (decode + prefill) or
several kinds appear in one phase (a hybrid stack).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
import zipfile

# Logical IO names (_<n> suffix stripped) that are host-supplied each step.
DEFAULT_SHARED_INPUTS = frozenset(
    {"attn_mask", "seq_pos", "q_rope_cos", "q_rope_sin", "k_rope_cos", "k_rope_sin"}
)
# Activations threaded layer-to-layer (the previous layer's output).
DEFAULT_THREADED_INPUTS = frozenset({"hidden_states", "residual_in"})
# Persistent per-layer KV-cache buffers (read + written in place).
DEFAULT_KV_CACHE = frozenset({"k_out", "v_out"})

# The graph-input tensor the host seeds with token embeddings (layer 0's input).
EMBED_TENSOR = "inputs_embeds"
# The graph-output tensor (the final layer's hidden state).
OUTPUT_TENSOR = "output"
# The ping-ponged activation spine. Two slots is both enough at any depth (the buffer is live
# only from L's write to L+1's read) and the minimum (one would alias a call's read and write).
SPINE_TENSORS = ("hidden_a", "hidden_b")

_DESCRIPTOR_VERSION = 1


@dataclass(frozen=True)
class TailGraph:
    """A standalone one-call graph run after the layer stack: the device lm_head.

    It owns no persistent state and needs no spine slot; its single fed input is the
    stack's hidden state, which the host passes back in. Its constants carry no
    ``_L{layer}`` suffix because there is exactly one instance of each.
    """

    kind: str  #: kernel kind, and the graph name the runtime invokes
    phase: str  #: the compiled (phase, kind) whose atex backs this graph
    buffers: dict  #: logical -> packed constant tensor


@dataclass(frozen=True)
class KindContract:
    """The per-kind IO contract the descriptor classifies against.

    shared_inputs: logical IO this kind consumes from the host each step (a
    full-attention layer needs rope + mask + seq_pos; a deltanet layer needs
    none of them). state: this kind's persistent per-layer buffers (a KV cache,
    or a deltanet recurrent/conv state) — codegen-allocated, one per layer.
    """

    shared_inputs: frozenset = field(default_factory=frozenset)
    state: frozenset = field(default_factory=frozenset)
    #: Logical IO the host supplies PER LAYER rather than once for the whole stack: the
    #: tensor is named "<logical>_L<layer>", so one kind can sit at several layers, each
    #: reading its own host buffer (Qwen3-VL's DeepStack adds — one vision feature map
    #: per injected layer). Takes precedence over shared_inputs for the same name.
    per_layer_inputs: frozenset = field(default_factory=frozenset)
    #: This kind READS the activation spine without advancing it: its output is a graph
    #: output of its own ("output_L<layer>") and the next layer consumes the same
    #: activation this one did — a branch off the trunk, not a link in it (Qwen3-VL's
    #: DeepStack mergers, which read a mid-tower hidden state and emit an extra
    #: image-feature output).
    tap: bool = False


def fused_graph_name(phase: str, tail_kind: str) -> str:
    """The graph that runs ``phase``'s layer stack and the tail in one device call (graphs
    and kernel symbols are separate namespaces, so it may coincide with kind_symbol)."""
    return f"{phase}_{tail_kind}"


def kind_symbol(phase: str, kind: str) -> str:
    """The kernel symbol for a (phase, kind): unique across the merged source."""
    return f"{phase}_{kind}"


def _strip_index(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def sublayer_tag(j: int) -> str:
    """The suffix an N-layer kernel gives sub-layer ``j``'s per-layer buffers.

    ``__l`` followed by digits is therefore reserved at the end of a kernel IO name:
    ``split_sublayer`` reads any such name as a tagged sub-layer buffer.
    """
    return f"__l{j}"


def split_sublayer(logical: str) -> tuple[str, int | None]:
    """(base name, sub-layer) of a kernel IO name.

    A kernel that runs N consecutive layers per call (``layers_per_kernel``)
    declares each sub-layer's per-layer buffers as ``<name>`` + ``sublayer_tag(j)``; an
    untagged name is shared by the call (host inputs, the spine).
    """
    m = re.fullmatch(r"(.+)__l(\d+)", logical)
    return (m.group(1), int(m.group(2))) if m else (logical, None)


def per_layer_spec(spec: dict) -> dict:
    """The packer's per-layer view of a kernel's buffer spec: an N-layer kernel declares
    every per-layer buffer once per sub-layer, all with one shape, and the packer binds one
    layer at a time under the base names."""
    by_base: dict[str, dict] = {}
    for name, entry in spec.items():
        by_base.setdefault(split_sublayer(name)[0], {})[name] = entry
    for base, entries in by_base.items():
        if len(set(entries.values())) != 1:
            raise ValueError(f"sub-layers declare {base!r} differently: {entries}")
    return {base: next(iter(entries.values())) for base, entries in by_base.items()}


def group_size(names) -> int:
    """Layers one call of a kernel runs: 1 + the highest sub-layer tag among its IO names."""
    subs = (split_sublayer(name)[1] for name in names)
    return 1 + max((sub for sub in subs if sub is not None), default=0)


# The region a kernel names for a pool it does not pin: the renderer's own default, so the
# descriptor leaves the field out.
_UNPINNED_MEMORY = "ddr"


def _kernel_decl(sym: str, io: list) -> dict:
    """A kernel declaration: its arguments are the rendered IO in canonical order.

    The engine validates arguments positionally by dtype + byte size, so the declared
    "input" type is a placeholder the per-call binding overrides.
    """
    return {
        "path": f"compiled_artifacts/{sym}/model.atex",
        "arguments": [
            {"name": logical, "tensor": {"type": "input", "shape": list(shape), "dtype": dtype}}
            for shape, dtype, logical, _ in io
        ],
    }


def _write_blob(path: Path, tensor) -> None:
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
    path.write_bytes(arr.tobytes())


def _render_phase_io(atex_path: Path, kernel_name: str) -> list[tuple[tuple, str, str, str]]:
    """Render a compiled (phase, kind)'s ATEX and return its IO in canonical order.

    Each entry is (shape, dtype, logical_name, memory) for axr_model.inputs then
    axr_model.outputs (outputs use OUTPUT_TENSOR). ``memory`` is the region the kernel's
    own pool for that buffer declares (`ddr0`, `ddr1`, or the unpinned `ddr`); it is
    carried because the graph, not the kernel, allocates the pool, so a region the kernel
    pinned survives only if the descriptor repeats it.
    """
    import axelera.aipu.atex as atex_base
    from axelera.codegen.axmodel import AxModelConfig
    from axelera.codegen.axmodel._engine import AxrModelEngine

    atex = atex_base.from_file(str(atex_path))
    atex.name = kernel_name
    axr = AxrModelEngine(AxModelConfig()).render(atex)
    if len(axr.outputs) != 1:
        # All outputs would take the OUTPUT_TENSOR name, i.e. one buffer (a live spine slot).
        raise ValueError(
            f"{kernel_name} renders {len(axr.outputs)} outputs; the graph path needs 1"
        )
    region = {p.name: str(getattr(p.memory, "value", p.memory)) for p in axr.memory_pools}
    io = [
        (tuple(i.shape), i.dtype, _strip_index(i.name), region[i.memory_pool]) for i in axr.inputs
    ]
    io += [(tuple(o.shape), o.dtype, OUTPUT_TENSOR, region[o.memory_pool]) for o in axr.outputs]
    return io


class _DescriptorBuilder:
    """Builds the MultiGraphFormat descriptor + the per-layer binary map.

    kinds[layer] is the structural kind of each layer; contracts[kind] gives that
    kind's shared-input / state classification. threaded_inputs (the activation
    spine) is uniform across kinds.
    """

    def __init__(self, model_name, kinds, contracts, threaded_inputs):
        self.model_name = model_name
        self.kinds = list(kinds)
        self.num_layers = len(self.kinds)
        self.contracts = contracts
        self.threaded_inputs = threaded_inputs
        self.shared_tensors: dict[str, dict] = {}
        self.graphs: dict[str, dict] = {}
        # tensor_name -> (layer, logical) for every per-layer constant to serialise.
        self.binaries: dict[str, tuple[int, str]] = {}
        # The tail graph's constants, whose tensor name IS their logical name.
        self.tail_binaries: set[str] = set()
        self.spine = self._plan_spine({})  # one layer per call until add_graph plans a graph

    def _spine(self, step: int) -> str:
        """The spine slot the trunk step ``step`` writes, and ``step + 1`` reads."""
        return SPINE_TENSORS[step % len(SPINE_TENSORS)]

    def _plan_spine(self, groups: dict[str, int]) -> list[tuple[str, str, str]]:
        """Per layer: (activation read, activation written, written type).

        The trunk threads call to call through the two ping-ponged spine slots: the first
        call reads the graph input, the last writes the graph output, and a tap layer
        (KindContract.tap) reads the live activation into its own output without taking a
        slot. The ping-pong advances once per CALL, not per layer, so an even group cannot
        land its read and its write on one slot; every layer of a call carries its ends.
        """
        trunk = [i for i, kind in enumerate(self.kinds) if not self.contracts[kind].tap]
        if not trunk:
            raise ValueError("every kind is a tap: the stack has no activation spine")
        plan: list[tuple[str, str, str]] = [("", "", "")] * self.num_layers
        step = 0  # trunk CALLS emitted so far == the index of the live activation
        layer = 0
        while layer < self.num_layers:
            kind = self.kinds[layer]
            source = EMBED_TENSOR if step == 0 else self._spine(step - 1)
            if self.contracts[kind].tap:
                plan[layer] = (source, f"{OUTPUT_TENSOR}_L{layer}", "output")
                layer += 1
                continue
            last = min(layer + groups.get(kind, 1) - 1, self.num_layers - 1)
            is_last = last >= trunk[-1]
            sink = OUTPUT_TENSOR if is_last else self._spine(step)
            for at in range(layer, last + 1):
                plan[at] = (source, sink, "output" if is_last else "temporary")
            step += 1
            layer = last + 1
        return plan

    def _tensor(self, logical: str, layer: int) -> tuple[str, str, bool]:
        """(tensor_name, type, needs_binary) for an IO logical at a layer."""
        contract = self.contracts[self.kinds[layer]]
        if logical in contract.per_layer_inputs:
            return f"{logical}_L{layer}", "input", False  # host-supplied, one per layer
        if logical in contract.shared_inputs:
            return logical, "input", False  # host-supplied each step
        if logical in contract.state:
            return f"{logical}_L{layer}", "cache", False  # persistent, codegen-allocated
        source, sink, sink_type = self.spine[layer]
        if logical in self.threaded_inputs:
            return source, ("input" if source == EMBED_TENSOR else "temporary"), False
        if logical == OUTPUT_TENSOR:
            return sink, sink_type, False
        return f"{logical}_L{layer}", "constant", True  # per-layer constant (has data)

    def add_graph(self, phase: str, io_by_kind: dict[str, list]) -> None:
        """Add one phase's graph: a kernel per kind + one call per group of layers.

        io_by_kind maps each kind appearing in this stack to its rendered IO (in
        canonical order). A call runs ``group_size`` consecutive layers of one kind; it
        names that kind's kernel and binds each sub-layer's tagged IO to its own layer's
        concrete tensors.
        """
        used = list(dict.fromkeys(self.kinds))  # distinct kinds, order-preserving
        # Arguments: IO in canonical order (validated by dtype + byte size).
        kernels_decl = {}
        for kind in used:
            sym = kind_symbol(phase, kind)
            kernels_decl[sym] = _kernel_decl(sym, io_by_kind[kind])
        # The grouping is this graph's: decode may stack layers where prefill does not.
        groups = {kind: group_size(e[2] for e in io_by_kind[kind]) for kind in used}
        self.spine = self._plan_spine(groups)
        graph_tensors: dict[str, dict] = {}
        calls = []
        layer = 0
        while layer < self.num_layers:
            kind = self.kinds[layer]
            sym = kind_symbol(phase, kind)
            # A call runs `group` consecutive layers of one kind: sub-layer j's tagged IO
            # binds layer + j's constants and state, the untagged inputs bind the first
            # layer's (spine source, host inputs), the output the last layer's spine sink.
            group = groups[kind]
            last = layer + group - 1
            if last >= self.num_layers or any(k != kind for k in self.kinds[layer : last + 1]):
                raise ValueError(
                    f"kernel {kind!r} runs {group} layers per call, but layers {layer}..{last} "
                    f"are not {group} consecutive layers of that kind"
                )
            arg_values = []
            per_layer_inputs = self.contracts[kind].per_layer_inputs
            for shape, dtype, logical, memory in io_by_kind[kind]:
                base, sub = split_sublayer(logical)
                at = layer + sub if sub is not None else (last if base == OUTPUT_TENSOR else layer)
                tname, ttype, needs_bin = self._tensor(base, at)
                # An untagged per-layer buffer in a stacked kernel would bind the first
                # layer's tensor to every sub-layer, with nothing downstream to notice.
                per_layer = ttype in ("constant", "cache") or base in per_layer_inputs
                if group > 1 and sub is None and per_layer:
                    raise ValueError(
                        f"kernel {kind!r} runs {group} layers per call, but its per-layer "
                        f"{ttype} IO {logical!r} carries no sub-layer tag "
                        f"('{logical}{sublayer_tag(0)}'..'{logical}{sublayer_tag(group - 1)}')"
                    )
                decl = {"type": ttype, "shape": list(shape), "dtype": dtype}
                if needs_bin:
                    decl["binary_path"] = f"outlined_weights/{tname}.bin"
                    self.binaries[tname] = (at, base)
                # Constants and state keep the kernel's DDR region: unpinned, the driver
                # scatters a layer's K-shards across DDR0/DDR1 and a matmul then streams from
                # one region's bandwidth. Host IO and the spine stay unpinned.
                if memory != _UNPINNED_MEMORY and ttype in ("constant", "cache"):
                    decl["memory"] = memory
                # Two kinds threading different shapes/dtypes would mis-size one spine slot
                # for the whole stack, and nothing downstream cross-checks the two.
                if ttype == "temporary" and graph_tensors.get(tname, decl) != decl:
                    raise ValueError(
                        f"spine slot {tname!r} declared as {graph_tensors[tname]} and {decl}"
                    )
                # Per-layer KernelCall binds IO (same order) to the layer's tensors.
                # Constants + state are shared across phases; per-graph IO stays local.
                dest = self.shared_tensors if ttype in ("constant", "cache") else graph_tensors
                # A tensor shared across phases must not be claimed by two regions: the
                # graph allocates it once, so the second declaration would silently move
                # the first kernel's buffer out from under it.
                if dest.get(tname, decl).get("memory") != decl.get("memory"):
                    raise ValueError(
                        f"{tname!r} is declared in {dest[tname].get('memory')} and "
                        f"{decl.get('memory')}"
                    )
                dest[tname] = decl
                arg_values.append({"name": logical, "tensor_name": tname})
            calls.append({"kernel": sym, "argument_values": arg_values})
            layer += group
        self.graphs[phase] = {
            "tensors": graph_tensors,
            "kernels": kernels_decl,
            "graph": calls,
        }

    def add_tail_graph(self, tail: TailGraph, io: list) -> None:
        """Add the tail graph: one call, whose constants are named by their logical name.

        Classification is by data, not by name: an IO the tail has a packed buffer for is a
        constant, the kernel's output is the graph output, and the one left over is the
        hidden state the host feeds each step.
        """
        sym = kind_symbol(tail.phase, tail.kind)
        graph_tensors: dict[str, dict] = {}
        arg_values = []
        fed = []
        for shape, dtype, logical, memory in io:
            decl = {"shape": list(shape), "dtype": dtype}
            if logical == OUTPUT_TENSOR:
                graph_tensors[logical] = {"type": "output", **decl}
            elif logical in tail.buffers:
                if logical in self.shared_tensors:
                    raise ValueError(
                        f"tail constant {logical!r} collides with a layer constant: they share "
                        "the shared_tensors declaration and the outlined_weights blob name"
                    )
                decl["binary_path"] = f"outlined_weights/{logical}.bin"
                if memory != _UNPINNED_MEMORY:  # the head K-splits its vocab projection too
                    decl["memory"] = memory
                self.shared_tensors[logical] = {"type": "constant", **decl}
                self.tail_binaries.add(logical)
            else:
                graph_tensors[logical] = {"type": "input", **decl}
                fed.append(logical)
            arg_values.append({"name": logical, "tensor_name": logical})
        if len(fed) != 1:
            raise ValueError(
                f"tail graph {tail.kind!r} must take exactly one host-fed input (the hidden "
                f"state); got {fed}. An unexpected name here means its packer did not produce "
                f"a buffer the kernel declares."
            )
        self.graphs[tail.kind] = {
            "tensors": graph_tensors,
            "kernels": {sym: _kernel_decl(sym, io)},
            "graph": [{"kernel": sym, "argument_values": arg_values}],
        }

    def add_fused_graph(self, phase: str, tail: TailGraph) -> str | None:
        """Add ``phase``'s stack with the tail appended, as one graph: the head reads the
        stack's output on the device instead of the host reading it back and feeding it in.

        The two graphs stay as they are (prefill scoring still feeds the tail by itself);
        the fused one is a third graph over the same kernels and tensors. Dropping the host
        round trip saves the output and input DMAs, a job load and a launch.

        Returns the new graph's name, or None when ``phase`` has no single-output stack to
        fuse (a tap kind's extra outputs would need a second host read-back anyway).
        """
        if phase not in self.graphs or tail.kind not in self.graphs:
            raise ValueError(f"add both the {phase!r} graph and the tail before fusing them")
        stack, head = self.graphs[phase], self.graphs[tail.kind]
        outputs = [n for n, t in stack["tensors"].items() if t["type"] == "output"]
        if outputs != [OUTPUT_TENSOR]:
            return None
        (fed,) = [n for n, t in head["tensors"].items() if t["type"] == "input"]
        hidden = {k: v for k, v in stack["tensors"][OUTPUT_TENSOR].items() if k != "type"}
        expected = {k: v for k, v in head["tensors"][fed].items() if k != "type"}
        if hidden != expected:
            raise ValueError(
                f"the {phase!r} stack writes {hidden} but the tail reads {expected}: they "
                "cannot be chained on the device"
            )
        tensors = {n: t for n, t in stack["tensors"].items() if n != OUTPUT_TENSOR}
        tensors[fed] = {"type": "temporary", **hidden}
        tensors[OUTPUT_TENSOR] = head["tensors"][OUTPUT_TENSOR]
        calls = [
            {
                "kernel": call["kernel"],
                "argument_values": [
                    {
                        **a,
                        "tensor_name": (
                            fed if a["tensor_name"] == OUTPUT_TENSOR else a["tensor_name"]
                        ),
                    }
                    for a in call["argument_values"]
                ],
            }
            for call in stack["graph"]
        ] + [dict(call) for call in head["graph"]]
        name = fused_graph_name(phase, tail.kind)
        self.graphs[name] = {
            "tensors": tensors,
            "kernels": {**stack["kernels"], **head["kernels"]},
            "graph": calls,
        }
        return name

    def descriptor(self) -> dict:
        return {
            "name": self.model_name,
            "version": _DESCRIPTOR_VERSION,
            "shared_tensors": self.shared_tensors,
            "graphs": self.graphs,
        }


def author_graph_model(
    out_dir: Path,
    model_name: str,
    phase_kind_to_atex: dict[tuple[str, str], Path],
    kinds: list[str],
    contracts: dict[str, KindContract],
    layer_buffers: dict[int, dict],
    *,
    threaded_inputs: frozenset = DEFAULT_THREADED_INPUTS,
    generation: str = "omega",
    tail: TailGraph | None = None,
) -> dict:
    """Build the descriptor + run render_graph_model, unpacking into out_dir.

    phase_kind_to_atex maps each (phase, kind) to its compiled model.atex; kinds
    is the per-layer kind list; contracts[kind] classifies each kind's IO;
    layer_buffers[layer][logical] provides the packed per-layer constants.
    tail adds one standalone graph after the stack (the device lm_head), whose
    (phase, kind) must also appear in phase_kind_to_atex; when its phase is one of the
    stack's, a third graph fuses that phase's stack and the tail into one device call
    (see _DescriptorBuilder.add_fused_graph).
    Returns the emitted model.json.
    """
    from axelera.codegen.axmodel_graph._engine import render_graph_model

    out_dir = Path(out_dir)
    builder = _DescriptorBuilder(model_name, kinds, contracts, threaded_inputs)
    # Group the compiled atexes by phase, then add one graph per phase.
    phases: dict[str, dict[str, Path]] = {}
    for (phase, kind), atex_path in phase_kind_to_atex.items():
        if tail is not None and (phase, kind) == (tail.phase, tail.kind):
            continue  # a standalone graph, not one of the stack's phases
        phases.setdefault(phase, {})[kind] = atex_path
    for phase, kind_atex in phases.items():
        io_by_kind = {
            kind: _render_phase_io(atex_path, kind_symbol(phase, kind))
            for kind, atex_path in kind_atex.items()
        }
        builder.add_graph(phase, io_by_kind)
    if tail is not None:
        tail_atex = phase_kind_to_atex[(tail.phase, tail.kind)]
        sym = kind_symbol(tail.phase, tail.kind)
        builder.add_tail_graph(tail, _render_phase_io(tail_atex, sym))
        if tail.phase in phases:
            builder.add_fused_graph(tail.phase, tail)
    descriptor = builder.descriptor()

    # Lay out the single input artifact render_graph_model consumes.
    src = out_dir / "_graph_input"
    if src.exists():
        shutil.rmtree(src)
    (src / "metadata_params").mkdir(parents=True)
    weights = src / "outlined_weights"
    weights.mkdir(parents=True)
    for (phase, kind), atex_path in phase_kind_to_atex.items():
        kdir = src / "compiled_artifacts" / kind_symbol(phase, kind)
        kdir.mkdir(parents=True)
        shutil.copyfile(atex_path, kdir / "model.atex")
    (src / "metadata_params" / "runtime_metadata.json").write_text(
        json.dumps(descriptor, indent=2)
    )
    for tname, (layer, logical) in builder.binaries.items():
        _write_blob(weights / f"{tname}.bin", layer_buffers[layer][logical])
    if tail is not None:
        for logical in builder.tail_binaries:
            _write_blob(weights / f"{logical}.bin", tail.buffers[logical])

    # Render, then unpack the AXM so model.json + its blobs sit at the top of out_dir.
    render_out = out_dir / "_graph_render"
    render_out.mkdir(parents=True, exist_ok=True)
    render_graph_model(
        model_name=model_name, generation=generation, input_paths=[src], output_path=render_out
    )
    axm = render_out / f"{model_name}.axm"
    with zipfile.ZipFile(axm) as z:
        z.extractall(out_dir)
        model_json = json.loads(z.read("model.json"))
    shutil.rmtree(src)
    shutil.rmtree(render_out)
    # Cache pools need no init blob: the renderer marks them zero_init and the runtime
    # memsets them at instance load.
    return model_json
