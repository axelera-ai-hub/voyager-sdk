# Copyright Axelera AI, 2026
"""The command line every recipe offers, and what each verb does.

Four verbs. `--build` compiles, and is the only one that writes. `--run` shows
the model working on an input. `--benchmark` reports throughput and latency.
`--eval` reports accuracy.

A recipe declares what is true of ITS model and hands it here, so the flags,
the help text and the order of operations are the same in all of them, and
reading one recipe teaches you every recipe. What each recipe keeps is the part
that differs: its preprocess, its pipelines, and the compiler call.

Imports no zoo. `--eval` needs one, and `accuracy` is where that dependency is
declared and refused when absent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import accuracy as _accuracy


@dataclass(frozen=True)
class Accuracy:
    """What `--eval` needs from a recipe, or None on a model with no eval dataset.

    `pipeline` takes the model stage rather than a path, because the two legs
    differ only in that stage: the host leg has no `.axm` to be given.
    """

    target: _accuracy.Target
    pipeline: Callable[[Any], Any]
    source: Callable[[], Path]
    postamble: Any = None
    device_config: Any = None


@dataclass(frozen=True)
class Recipe:
    """The recipe, as the command line needs to see it.

    `accuracy` is None for a model the zoo declares no eval dataset for, and
    that absence is what removes `--eval` from the parser: a verb that exists
    and then refuses would advertise something this model cannot do.
    """

    model_name: str
    output_path: Path
    build: Callable[[], None]
    pipeline: Callable[[Path], Any]
    renderers: tuple[str, ...]
    accuracy: Accuracy | None = None


def _parser(recipe: Recipe):
    import argparse

    scores = recipe.accuracy is not None
    parser = argparse.ArgumentParser(
        description='Build, run, benchmark or score this model.'
        if scores
        else 'Build, run, or benchmark this model.'
    )
    parser.add_argument(
        '--build',
        action='store_true',
        help='compile the model (minutes; the only action that writes)',
    )
    parser.add_argument(
        '--run',
        metavar='INPUT',
        help='run end-to-end inference on an image, video or stream and render each frame',
    )
    parser.add_argument(
        '--benchmark',
        metavar='INPUT',
        help='benchmark the pipeline over an image or video; print fps/latency',
    )
    if scores:
        dataset = recipe.accuracy.target.dataset
        parser.add_argument(
            '--eval',
            action='store_true',
            help=f'score accuracy on {dataset} and print one metric (needs axelera-zoo)',
        )
        parser.add_argument(
            '--engine',
            choices=_accuracy.ENGINES,
            default=_accuracy.AIPU,
            help='which leg --eval scores: aipu (default; the compiled .axm on the '
            'device) or host (the fp32 source weight on this machine, the reference '
            'the device number is a gap from)',
        )
        parser.add_argument(
            '--data-root',
            metavar='DIR',
            help='directory holding one subdirectory per dataset (default: '
            '$AXZOO_EVAL_DATA_ROOT, else ./data, else the zoo cache)',
        )
        parser.add_argument(
            '--frames',
            type=int,
            default=0,
            metavar='N',
            help='stop --eval after N frames (default: 0, the whole split). A split '
            'stored in class order is truncated, not sampled',
        )
    parser.add_argument(
        '--axm',
        metavar='PATH',
        help='run, benchmark or score this already-built .axm instead of compiling first',
    )
    parser.add_argument(
        '--display',
        choices=recipe.renderers,
        default='auto',
        help='renderer for --run (default: auto, which prefers Vulkan and works '
        "headless). 'none' prints one line per result object and needs no renderer "
        'installed',
    )
    parser.add_argument(
        '--window-size',
        metavar='WxH | W | fullscreen',
        help="window size for --run: 'WxH', a bare width (16:10 height), or "
        "'fullscreen' (default: the display's own)",
    )
    parser.add_argument(
        '--fullscreen',
        action='store_const',
        const='fullscreen',
        dest='window_size',
        help='alias for --window-size fullscreen',
    )
    parser.add_argument(
        '--optimize',
        choices=('auto', 'cl', 'simd', 'none'),
        # Not in `choices`, and deliberately: it is the runtime's own word for
        # "the policy already in effect" (auto), and a value no caller typed is
        # what lets --eval keep a different default from --run without a second
        # flag. See `accuracy.FUSION_OFF`.
        default='inherit',
        help='preprocess fusion kernel: auto (the shipped pipeline; the default '
        "for --run and --benchmark), cl or simd to force one, or none for the raw "
        'op sequence. --eval defaults to none instead, because the policy selects '
        'the resize kernel and every accuracy number on record was taken raw',
    )
    return parser


def _resolve_artifact(recipe: Recipe, args, parser, *, reference_only: bool) -> Path | None:
    """The `.axm` the requested verbs will use, building it first if needed.

    Returns None only when `reference_only`: the host leg runs the source weight
    the build consumes, so a model can be scored fp32 before it has ever been
    compiled, and compiling one to satisfy an unused argument would be minutes
    of work nothing reads.
    """
    if args.axm:
        if args.build:
            parser.error('--axm runs an already-built model; it cannot be combined with --build')
        return Path(args.axm)
    if reference_only:
        return None
    # --run/--benchmark still build first: this path has no cached-artifact
    # check. Point --axm at an existing one to skip the compile.
    recipe.build()
    return recipe.output_path


def _show(recipe: Recipe, args, axm: Path) -> None:
    """`--run`: stream the input through the pipeline and render each frame."""
    pipeline = recipe.pipeline(axm).optimized(args.optimize)
    if args.display == 'none':
        # No window: stream and print what the pipeline found. Still display-free,
        # because `display`'s module scope pulls no renderer (the backends are
        # imported only when one is chosen), so a headless box with no display
        # stack runs the model end to end.
        from axelera.runtime import display

        for _frame, results in pipeline.stream(args.run, max_in_flight=1):
            print(display.format_objects(results))
        return
    from axelera.runtime import cv, display

    with display.create_display(
        renderer=args.display, window_size=args.window_size, title=recipe.model_name
    ) as show:
        for image, objects in cv.iter_results(pipeline, args.run):
            show(image, objects)


def _score(recipe: Recipe, args, axm: Path | None) -> None:
    """`--eval`: build the scoring pipeline for one leg and print its metrics."""
    leg = recipe.accuracy
    # A policy nobody typed is `--eval`'s own default, not `--run`'s: see
    # `accuracy.FUSION_OFF`. Applied inside `measure`, which is also what
    # `axzoo eval` reaches, so one number cannot mean two preprocesses.
    optimize = _accuracy.FUSION_OFF if args.optimize == 'inherit' else args.optimize
    stage = _accuracy.model_stage(
        args.engine,
        axm_path=axm,
        source=leg.source,
        postamble=leg.postamble,
        device_config=leg.device_config,
    )
    result = _accuracy.measure(
        leg.target,
        leg.pipeline(stage),
        engine=args.engine,
        optimize=optimize,
        data_root=args.data_root,
        frames=args.frames,
    )
    print(
        _accuracy.format_result(
            result,
            leg.target,
            model_name=recipe.model_name,
            engine=args.engine,
            optimize=optimize,
        )
    )


def run(recipe: Recipe) -> None:
    """Parse the command line and carry out every verb it names."""
    parser = _parser(recipe)
    args = parser.parse_args()
    scores = recipe.accuracy is not None
    asked = bool(args.build or args.run or args.benchmark or (scores and args.eval))
    # No action requested: show the commands rather than starting a multi-minute
    # compile. `axzoo build` calls the recipe's build entry, not this.
    if not asked:
        parser.print_help()
        raise SystemExit(0)
    if args.benchmark:
        # Before the build, not after it: a report gathered under per-frame DEBUG
        # logging is measurably wrong, and refusing once a multi-minute compile has
        # already run makes the flag mistake expensive rather than cheap.
        from axelera.runtime import bench

        try:
            bench.refuse_if_debug_logging()
        except bench.BenchmarkRefused as refused:
            parser.exit(2, f'Error: {refused}\n')
    scoring_host_only = False
    if scores and args.eval:
        # Before the build, not after it: see `accuracy.require_support`.
        _accuracy.require_support()
        if args.axm and args.engine == _accuracy.HOST:
            parser.error(
                '--axm names the compiled artifact to score, so it applies to --engine aipu; '
                '--engine host runs the source weight the build consumes and reads no .axm'
            )
        scoring_host_only = args.engine == _accuracy.HOST and not (
            args.build or args.run or args.benchmark
        )
    axm = _resolve_artifact(recipe, args, parser, reference_only=scoring_host_only)
    if args.run:
        _show(recipe, args, axm)
    if args.benchmark:
        from axelera.runtime import bench

        pipeline = recipe.pipeline(axm).optimized(args.optimize)
        # The early guard above catches the ordinary case, before the build. This
        # catches the same refusal raised late, if anything raised the log level
        # between the two, so every path out of `--benchmark` is one line and an
        # exit code rather than a traceback.
        try:
            print(bench.format_benchmark(bench.benchmark_pipeline(pipeline, args.benchmark)))
        except bench.BenchmarkRefused as refused:
            parser.exit(2, f'Error: {refused}\n')
    if scores and args.eval:
        # A refusal `shared/accuracy` raises (an absent weight, an artifact that
        # was never built, an unreadable suffix) names its own fix, so it reads
        # as one line and exit 2, the way `axzoo eval` prints the same refusal.
        # The type comes from `accuracy`, which is the only module here allowed
        # to name the zoo, and `require_support` above has already proved it
        # imports.
        try:
            _score(recipe, args, axm)
        except _accuracy.refusals() as refused:
            parser.exit(2, f'Error: {refused}\n')
