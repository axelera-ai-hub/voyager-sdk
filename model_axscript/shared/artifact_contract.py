# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Runtime-loadable contract check for an AxScript build artifact (stdlib only).

A dependency-free precursor to ``axelera.runtime.boundary_spec``: it
checks a compiled artifact holds the ``model.json`` graph contract the runtime
consumes (named graphs, each with typed inputs and a shaped, typed output),
plus ``pipeline.json`` and the ``host_params`` blob. Kept import-light (no torch
/ codegen) so it can gate any hand-edited or AI-modified recipe at build time,
not later at op.load.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


class AxScriptArtifactError(RuntimeError):
    """The compiled artifact does not meet the runtime-loadable contract."""


def validate_artifact(artifact_dir, known_dtypes: Iterable[str]) -> None:
    """Raise ``AxScriptArtifactError`` on a cross-layer inconsistency in the artifact.

    This is a cheap, dependency-free check. It does NOT prove the model runs;
    that is the runtime load-probe (``generate.probe_runtime_load``, the runtime's
    own recogniser) plus a numerical check (``generate.device_perplexity``). ALTO
    *generates* ``model.json``, so checking its shape alone would be tautological.
    What ALTO does not guarantee end-to-end - and what a hand-edited or
    AI-modified recipe desyncs - is that the runtime-side pieces line up with it:

    - ``pipeline.json`` references a ``host_params`` blob and a ``model.json`` that
      both exist;
    - every graph a device stage will run is declared in ``model.json``;
    - each graph's input/output dtypes are ones the runtime can bind
      (``known_dtypes`` is the caller's own dtype map, so the two never drift) and
      each output carries a shape (``RuntimeModel`` allocates the buffer from it);
    - every graph compiles at batch 1 (``shape[0] == 1`` on all inputs/outputs) -
      parallelism is multi-core sharding or scheduler fan-out, never a graph batch.

    ``known_dtypes`` is the runtime's model.json dtype vocabulary.
    """
    known = set(known_dtypes)
    artifact_dir = Path(artifact_dir)
    pipe_path = artifact_dir / "pipeline.json"
    if not pipe_path.is_file():
        raise AxScriptArtifactError(f"missing pipeline.json in {artifact_dir}")
    pipeline = json.loads(pipe_path.read_text())

    host_params = pipeline.get("host_params")
    if not host_params or not (artifact_dir / host_params).is_file():
        raise AxScriptArtifactError(f"host_params blob {host_params!r} missing in {artifact_dir}")

    model_name = pipeline.get("device_model_json", "model.json")
    model_path = artifact_dir / model_name
    if not model_path.is_file():
        raise AxScriptArtifactError(f"model.json {model_name!r} missing in {artifact_dir}")
    graphs = json.loads(model_path.read_text()).get("graphs")
    if not graphs:
        raise AxScriptArtifactError(f"{model_name} declares no graphs in {artifact_dir}")

    # Every graph a device stage will run must actually be declared.
    stage_graphs = {g for s in pipeline.get("stages", []) for g in (s.get("graphs") or [])}
    missing = stage_graphs - set(graphs)
    if missing:
        raise AxScriptArtifactError(
            f"pipeline stages reference graphs absent from {model_name}: {sorted(missing)}"
        )

    # Each graph must be runtime-consumable: named, runtime-bindable dtypes, shaped output.
    for name, g in graphs.items():
        ins, outs = g.get("inputs"), g.get("outputs")
        if not ins or any(i.get("name") is None or i.get("dtype") not in known for i in ins):
            raise AxScriptArtifactError(f"graph {name!r}: input missing name / runtime dtype")
        if not outs or "shape" not in outs[0] or outs[0].get("dtype") not in known:
            raise AxScriptArtifactError(f"graph {name!r}: output missing shape / runtime dtype")
        # Compiled batch must be 1. The op.load scheduler + result decoders + RuntimeModel
        # all assume a batch-1 graph; runtime batching (Seq.batch / more cores) is a
        # scheduler-level fan-out of many batch-1 runs, never a batched device call. A
        # batch>1 graph (e.g. an accidental auto_multicore_mode="batch" compile) has no
        # consumer in the stack, so reject it at build time rather than at op.load.
        for io in (*ins, *outs):
            shape = io.get("shape")
            if shape and shape[0] != 1:
                raise AxScriptArtifactError(
                    f"graph {name!r}: {io.get('name')!r} has compiled batch {shape[0]} != 1 "
                    f"(shape {shape}); AxScript graphs must compile at batch 1 - parallelism "
                    f"comes from multi-core sharding or scheduler fan-out, not a graph batch."
                )


TOOLCHAIN_PACKAGES = (
    ("alto", "axelera.compiler", "axelera-compiler"),
    ("tvm", "tvm", "axelera-tvm"),
    ("graph_compiler", "axelera.graph_compiler", "axelera-graph-compiler"),
    ("runtime", "axelera.runtime", "axelera-runtime"),
)
"""What a build may compile through, as (report key, module, distribution).

The list is a superset on purpose and costs a build nothing it did not already pay: an entry
appears only when that package was actually LOADED, so an AxScript build, which lowers
through ALTO, TVM and the runtime, records no `graph_compiler` key at all, and an artifact
produced through a path that does reach it says so rather than going unrecorded. A key that
is present is therefore evidence the build ran that code, never a version list copied off
the shell. Add `axelera.model_optimizer` here the day a build path reaches it.
"""


_GIT_TIMEOUT_S = 10
"""How long a provenance query may take before the build gives up on it.

This runs at the END of a compile that already cost minutes, so a git that never returns (an
fsmonitor hook, a network-mounted worktree) would throw away the expensive part after it had
already succeeded. Losing the provenance is the cheap failure and is the one to take.
"""


def _git(path: Path, *args: str) -> str | None:
    """The stdout of one read-only git command in `path`, or None when it did not answer.

    Did not answer covers all three: git is absent, the command failed, or it ran out of
    `_GIT_TIMEOUT_S`. A caller cannot act differently on those, and the difference between
    them is not a fact about the artifact.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _git_revision(path: Path) -> str | None:
    """The revision of the worktree `path` sits in, plus what is known about its edits.

    Three states, not two. A bare revision says the tree was checked and was clean, `.dirty`
    says it was checked and had edits, and `.unverified` says the check itself could not run:
    `git status` fails on index.lock contention with a concurrent build, on a permission
    problem, on a broken worktree. Reading any of those as clean asserts the one thing the
    suffix exists to establish, which is that the recorded revision IS the code that was
    compiled.

    None when `path` is not in a worktree, and never for an installed wheel: a wheel under
    site-packages can sit inside an unrelated repository, and reporting THAT repository's
    revision as the package's would be a fabricated provenance rather than a missing one.
    """
    if any(part == "site-packages" for part in path.parts):
        return None
    revision = _git(path, "rev-parse", "--short", "HEAD")
    if revision is None:
        return None
    status = _git(path, "status", "--porcelain")
    if status is None:
        return revision.strip() + ".unverified"
    return revision.strip() + (".dirty" if status.strip() else "")


def _package_revision(module: str, distribution: str) -> str | None:
    """What the loaded `module` actually is: its worktree revision, else its wheel version.

    Read from ``sys.modules`` rather than imported, so a build that never reached a package
    records nothing about it instead of importing it to ask.

    The worktree branch is the load-bearing one. An editable install SHADOWS the wheel while
    ``importlib.metadata`` keeps reporting the version of the wheel it replaced, so recording
    that version names code which never ran. That is not a corner case here: a compiler
    worktree on the path is the normal shape of a bring-up shell.
    """
    loaded = sys.modules.get(module)
    origin = getattr(loaded, "__file__", None)
    if origin is None:
        return None
    revision = _git_revision(Path(origin).resolve().parent)
    if revision:
        return revision
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def build_toolchain() -> dict[str, str]:
    """What produced this artifact: the compiler packages that ran, plus this checkout.

    Provenance for a number, never a gate. `axzoo report --check` does not compare it, and
    must not: a wheel bump moves every entry here at once and would turn every committed
    report red without a single measurement having changed.

    `framework` is the AxScript source tree's own revision, which no wheel version carries:
    the recipes, kernels and graph authoring that decide what gets compiled live here rather
    than in any installed distribution.
    """
    entries = {"framework": _git_revision(Path(__file__).resolve().parent)}
    for key, module, distribution in TOOLCHAIN_PACKAGES:
        entries[key] = _package_revision(module, distribution)
    return {key: value for key, value in entries.items() if value}


def artifact_metadata(artifact_dir) -> dict:
    """``<artifact_dir>/metadata.json``, or ``{}`` when there is nothing readable there.

    One reader for the file this module writes, because every caller wants the same fallback
    and each hand-rolled copy has to decide the same three things: that a missing file is
    "does not say" rather than an error, that an unreadable one is too, and that a directory
    with no build in it answers the same way. `artifact_checkpoint` is deliberately not
    routed through here: it reads one field out of whichever of THREE files carries it, so
    the file is its variable rather than its constant.
    """
    try:
        recorded = json.loads((Path(artifact_dir) / "metadata.json").read_text())
    except (OSError, ValueError):
        return {}
    return recorded if isinstance(recorded, dict) else {}


def artifact_toolchain(artifact_dir) -> dict[str, str]:
    """The toolchain recorded in ``<artifact_dir>/metadata.json``, empty when there is none.

    Read back rather than recomputed, because `axzoo report` REUSES an artifact it finds
    already built. Recomputing would stamp the shell that measured the numbers onto a
    directory some earlier shell compiled, which is the one claim this field exists to stop.
    """
    toolchain = artifact_metadata(artifact_dir).get("toolchain")
    return toolchain if isinstance(toolchain, dict) else {}


def write_artifact_metadata(
    artifact_dir,
    *,
    compiled_seq_lens: Iterable[int],
    model_name: str,
    num_calibration_samples: int | None = None,
    image_size: Iterable[int] | None = None,
) -> None:
    """Write ``<artifact_dir>/metadata.json``: what this artifact was actually built at.

    The eval driver (``llm_runner._built_seq_len``) reads this back to bound the
    scored sequence and to label the accuracy CSV, refusing rather than falling back to a
    build config's declared default. A config default describes the model's stated
    intent, not necessarily what THIS directory was compiled at (a sweep or a `--seq-len`
    override can compile a different length into the same shape of artifact). So the
    compile step that knows the true value writes it here instead of leaving eval to
    infer it. ``compiled_seq_lens`` is a list (today always one entry) so a future
    multi-bucket compile has somewhere to declare every length it produced.

    Neutral toolkit (stdlib only, no torch/numpy), so the vision AxScript build can adopt
    the same file for its own compiled run sizes (e.g. image_size) without a new writer.

    ``toolchain`` is what COMPILED this directory (`build_toolchain`), recorded at build time
    because `axzoo report` measures an artifact it may find already built: asked later, the
    shell doing the asking is not necessarily the one that produced the numbers.

    ``num_calibration_samples`` and ``image_size`` are the two facts a VISION-language build
    knows and nothing downstream can recover. The calibration count is the load-bearing one:
    the DocVQA eval scores rows the build did NOT calibrate on, and how many to skip is a
    property of this directory, not of the config in the reader's hand. Read from the config
    instead, a build made with `--num-cal 500` would be scored on 452 of its own calibration
    rows and report a score it earned by memorising them. Both are optional so a text model
    writes the same file it always did.
    """
    payload: dict = {
        "compiled_seq_lens": [int(n) for n in compiled_seq_lens],
        "model_name": model_name,
        "toolchain": build_toolchain(),
    }
    if num_calibration_samples is not None:
        payload["num_calibration_samples"] = int(num_calibration_samples)
    if image_size is not None:
        payload["image_size"] = [int(n) for n in image_size]
    (Path(artifact_dir) / "metadata.json").write_text(json.dumps(payload, indent=2))


def artifact_checkpoint(artifact_dir) -> str | None:
    """The checkpoint an artifact was built from, or None when nothing here records one.

    :func:`write_artifact_metadata` records it as ``model_name``; a text build also writes
    ``repo_id`` into its ``pipeline.json``, and an artifact built before either did carries
    the quantization cache key. Read in that order. A missing or unreadable file means "does
    not say" rather than an error, so a caller can treat "nothing built here" and "a build
    that did not record it" the same way.

    Both entry points read the checkpoint through this, because the weights are compiled in:
    a run over an existing directory has to tokenize and score with the checkpoint THAT
    directory was built from, and a card's default describes what a build would use today.
    """
    directory = Path(artifact_dir)
    for name, field in (
        ("metadata.json", "model_name"),
        ("pipeline.json", "repo_id"),
        (Path("quantized_model") / "cache_key.json", "repo_id"),
    ):
        try:
            recorded = json.loads((directory / name).read_text()).get(field)
        except (OSError, ValueError):
            continue
        if recorded:
            return recorded
    return None


def checkpoint_conflict(artifact_dir, requested: str | None) -> str | None:
    """Why ``requested`` cannot be honoured against the artifact already in the directory.

    Returns None when there is nothing to refuse: no checkpoint asked for, no artifact to
    read one off, or the two agree. The weights are compiled in as hard as the context
    length, so honouring the artifact that is there and labelling it with the checkpoint
    that was asked for is how a number gets attributed to weights nobody measured.

    One text for both families, so a text card and a vision-language card cannot drift into
    saying it two different ways.
    """
    built_from = artifact_checkpoint(artifact_dir)
    if requested is None or built_from is None or requested == built_from:
        return None
    return (
        f"{artifact_dir} was built from {built_from}, not {requested}. Run `build` to "
        f"compile it from your checkpoint, or pass --output-dir to keep both."
    )


def validate_run_size(value: int, supported: tuple[int, ...], name: str) -> None:
    """Reject a run size the model has not been measured at, before the compile starts.

    ``context_len`` (LLM) and ``image_size`` (vision) size the device buffers. An
    unmeasured size does not fail at compile: it fails on chip as a hang at prefill or a
    saturated LUT, minutes later and far from the cause. So each model declares the set it
    has been measured at and this rejects anything outside it. Widening the set is a
    deliberate edit backed by a measurement on the target part - the ceiling is AIPU DRAM,
    so a 16GB part admits sizes a 4GB part does not.
    """
    if value not in supported:
        raise AxScriptArtifactError(
            f"{name}={value} has not been validated for this model; measured sizes are "
            f"{sorted(supported)}. Measure it on the target device first, then add it to "
            f"the model's supported set - do not widen the set to make a build pass."
        )


def decoder_tower(artifact_dir):
    """The directory inside a build that holds the decoder's ``pipeline.json``.

    A text-only build writes ``pipeline.json`` at the top of its artifact directory. A
    multi-tower build (a VLM: a vision encoder plus a decoder) writes one directory per
    tower and keeps the artifact root for the HuggingFace metadata that makes the whole
    thing loadable, so its decoder contract lives one level down under ``text/``.

    Resolving that here, rather than at each caller, is what lets ONE eval adapter score
    every LLM in the tree. A VLM routed through a second adapter would report a perplexity
    that is comparable with no other number in the zoo, which is the failure the single
    accuracy path exists to prevent.

    Returns ``artifact_dir`` unchanged when it already holds a ``pipeline.json`` (so a
    caller that has already resolved a tower may pass it straight back in), the ``text/``
    subdirectory when that holds one, and otherwise ``artifact_dir`` again -- the caller's
    own "no pipeline.json here" error names the directory the user actually passed.
    """
    artifact_dir = Path(artifact_dir)
    if (artifact_dir / "pipeline.json").is_file():
        return artifact_dir
    text = artifact_dir / "text"
    if (text / "pipeline.json").is_file():
        return text
    return artifact_dir
