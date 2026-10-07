# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Measure one vision model's PERF for its report: latency and fps per core count.

The vision analogue of `llm/shared/context_sweep.py`, and the perf half `axzoo report` calls
for this family. It exists so a vision block is GENERATED: the same numbers typed by hand read
identically and cannot be re-derived, which is how a run size measured at two core counts once
became two invented keys on one row.

One image size is measured at every shard width the family ships, because the shard is a
COMPILE decision (`EncoderConfig.shard`), so each is a different artifact and a different row.
The 1-core baseline and the shipped shard answer different questions and neither substitutes
for the other.

ACCURACY is not here and must not come back. A metric belongs to the zoo, keyed by task: a
dataset iterator in `axelera/zoo/datasets` and an evaluator in `axelera/zoo/evaluators`, which
`axzoo report` and `axzoo eval` both reach through `axelera/zoo/runner.py`. This module once
dispatched to a harness per task inside `model_axscript`, and that gave one metric two
implementations free to disagree about one card.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import io
import json
import re
import sys
import textwrap
from pathlib import Path
from time import perf_counter
from typing import Any

from .build import timeit
from .builder import artifact_exists, device_name
from .config import EncoderConfig

# Driver lines worth keeping with a failure: the ones that name a resource it could not get.
_CAUSE = re.compile(r'^\s*(ERROR|FATAL)[:\s].*(pool|memory|core|L1|L2|DDR).*$', re.M | re.I)


class _Tee:
    """Write to both streams, so capturing the driver's output does not silence the console."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


#: Where a formatted traceback starts inside an exception message. Everything before it is
#: the caller's own context; the last line after it is the exception that ended the chain.
_TRACEBACK_START = 'Traceback (most recent call last):'


def _build_failure_note(exc: Exception) -> str:
    """What the build was doing and WHY it stopped, without the frames between the two.

    A compiler exception carries one or more nested worker tracebacks, and a report is a
    committed file a person reads: the caller's context names what was being compiled, the last
    line names the constraint that was not met, and every frame between them is noise in a
    yaml. The whole text stays in the build log, which the block's own `command` reproduces.

    Split on the traceback marker rather than filtered line by line. A filter has to enumerate
    what noise looks like, and each spelling it misses becomes the note's headline: successive
    attempts here were headed by "Traceback (most recent call last):" and then by
    "rv = local_pyfunc(*pyargs)", which are the frames, not the reason.
    """
    text = str(exc)
    if _TRACEBACK_START in text:
        context, _, frames = text.partition(_TRACEBACK_START)
        lines = [line.strip() for line in frames.splitlines() if line.strip()]
        reason = ' '.join(
            part for part in (' '.join(context.split()), lines[-1:] and lines[-1]) if part
        )
    else:
        reason = ' '.join(text.split())
    return 'build failed: ' + textwrap.shorten(
        reason or type(exc).__name__, width=400, placeholder=' ...'
    )


def _failure_note(exc: Exception, captured: str) -> str:
    """Return a note that says what failed AND the driver line explaining why, when there is one."""
    seen = list(dict.fromkeys(m.group(0).strip() for m in _CAUSE.finditer(captured)))
    return f'run failed: {exc}' + (f' | driver: {"; ".join(seen)}' if seen else '')


class PreparationFailure(Exception):
    """The model could not be prepared, so no size was attempted and no row exists.

    Same meaning as the LLM sweep's: the checkpoint, its processor or its calibration set
    reads no run size, so a failure there says nothing about the size being measured.
    """


def shard_widths(config: EncoderConfig | None = None) -> tuple[int, ...]:
    """The shard widths a vision model is measured at: the 1-core baseline and its shard.

    Derived from the config rather than listed, so a model that ships a different shard is
    measured at the shard it ships. Deduplicated, because a model whose shard IS one core
    has one configuration, not the same one twice.
    """
    shard = int((config or EncoderConfig()).shard)
    return (1,) if shard == 1 else (1, shard)


def _package(stem: str):
    """The model package for a config stem, which is where its seams and `make_config` live.

    Asked of the registry rather than joined onto `model_axscript.vision.`: a family may group
    its models by task, and a fixed-depth path would stop resolving the moment one does. The
    directory basename IS the config stem, which `test_axscript_model_dir_matches_stem` gates.
    """
    from model_axscript import registry

    for tag, directory in registry.models().items():
        if directory.name == stem and registry.family(tag) == 'vision':
            return importlib.import_module(registry.package(tag))
    raise PreparationFailure(
        f'no vision model package registers a directory named {stem!r}; the registry holds '
        f'{sorted(registry.models())}'
    )


def _stages(artifact_dir: Path) -> list[str] | None:
    """The ordered stage placements of a compiled artifact, or None for a single stage.

    Read from the artifact's own `pipeline.json`, never assumed: a row states how the model
    was CUT, and the writer of that file is the only thing that knows.
    """
    pipeline = json.loads((artifact_dir / 'pipeline.json').read_text())
    placements = [stage.get('placement') for stage in pipeline.get('stages') or []]
    return placements if len(placements) > 1 else None


def _sample_image(config) -> Any:
    """A deterministic image to time on: a mid-grey frame at the model's own input size.

    Perf is what this measures, and the CONTENT of the input does not change a static-shape
    dispatch, so a synthetic frame keeps the sweep independent of any dataset being present.
    Accuracy is the other half and it uses real images.
    """
    from PIL import Image

    size = int(config.image_size or 224)
    return Image.new('RGB', (size, size), (114, 114, 114))


def measure_one(stem: str, image_size: int, shard: int, device, out_root: Path, reuse: bool):
    """Build (or reuse) one artifact at `image_size` x `shard`, and time it. Returns a row."""
    import os

    package = _package(stem)
    artifact_dir = Path(out_root) / stem / f'{image_size}px-c{shard}'
    config = dataclasses.replace(
        package.make_config(artifact_dir), shard=shard, image_size=image_size
    )
    # instances is fixed at 1 until a run-time replica count exists; ai_cores is the derived
    # total a reader compares across rows without doing the multiplication themselves.
    row: dict[str, Any] = {
        'image_size': image_size,
        'shard': shard,
        'instances': 1,
        'ai_cores': shard,
    }
    if device is not None:
        os.environ['AXELERA_METIS_DEVICE'] = str(device)

    built = None
    # NOT model.json: an axten artifact keeps its graph model one directory down, so the old
    # check saw no artifact there and rebuilt every card on every run. `artifact_exists` is the
    # contract's own entry point and is where every builder writes it.
    if not (reuse and artifact_exists(artifact_dir)):
        started = perf_counter()
        try:
            # The card's own `build` seam, not a builder named here: two build paths exist
            # under `shared/`, and a card that bakes a head compiles through the other one.
            # Naming one here measured a different artifact than the card ships, or, once
            # that path started refusing what it cannot honour, refused to measure at all.
            package.build(config)
        except Exception as exc:  # noqa: BLE001 - a build failure is a ROW, not a crash
            return row | {'status': 'FAILED', 'note': _build_failure_note(exc)}
        built = round(perf_counter() - started)

    driver = None
    measured_on = None
    # The runtime's exception says an executor could not be created; WHY is a driver print on
    # stderr ("Failed to create memory pool shared_l2"), which `shared.runtime.device_job`
    # documents as not being part of the exception. A row that records only the exception
    # sends the next reader to re-run the case to learn what this run already found out, so
    # the driver's own lines are captured and kept with the failure.
    driver_output = io.StringIO()
    try:
        # The family seams, so this measures whatever a card actually ships: `load_device`
        # finds the artifact already built above, and `prepare` is hoisted OUT of the timed
        # call because what is timed is the model, not the image decoder.
        with contextlib.redirect_stderr(_Tee(sys.stderr, driver_output)):
            driver = package.load_device(config)
            prepared = package.prepare(_sample_image(config), config)
            timing = timeit(lambda: package.infer(driver, prepared))
            # Read while the part is still held: the release below is what makes it unknowable.
            measured_on = device_name(config, driver)
    except Exception as exc:  # noqa: BLE001 - a device failure is a ROW, not a crash
        return row | {'status': 'FAILED', 'note': _failure_note(exc, driver_output.getvalue())}
    finally:
        # The next core count reserves the same part, and the accuracy leg after it reserves
        # it again: a measurement that keeps its reservation makes the next one fail.
        _release(config, driver)

    row |= {'latency_ms': round(timing['ms'], 2), 'fps': round(timing['fps'], 1)}
    row['device_used'] = measured_on
    if built is not None:
        row['build_s'] = built
    stages = _stages(artifact_dir)
    if stages:
        row['stages'] = stages
    return row


def _release(config, driver) -> None:
    """Give the part back through the builder that loaded `driver`.

    Resolved from the config, never by probing the driver for a `runtime` attribute: the two
    builders hold their reservation in different places, and an axten card holds it in a
    module-level cache the driver has no handle on. A probe therefore gave the part back for one
    builder and silently kept it for the other, so a sweep held a device for every step after it.
    """
    if driver is not None:
        from .builder import release

        release(config, driver)


def sweep_one(
    stem: str, image_size: int, device, out_root: Path, *, reuse_existing: bool = False
) -> dict:
    """Measure one image size at every shard width the model ships. Returns a report block.

    `{'runs': [...], 'device': ..., 'shard': ...}`, the shape `axzoo report` merges: one row
    per shard width, each carrying its own `shard`, so the two are never mistaken for one
    measurement taken twice.
    """
    package = _package(stem)
    widths = shard_widths(package.make_config(Path(out_root) / stem))
    rows = [
        measure_one(stem, image_size, shard, device, out_root, reuse_existing) for shard in widths
    ]
    # The device the rows were actually measured ON, lifted out of them: the block states one
    # part for one measurement. Reported rather than echoed from `device`, because a block that
    # repeats the request cannot show a request that was not honoured, and that is precisely how
    # a run on the wrong device stays invisible.
    measured_on = [row.pop('device_used', None) for row in rows]
    return {'runs': rows, 'device': next((n for n in measured_on if n), None), 'shard': widths[-1]}
