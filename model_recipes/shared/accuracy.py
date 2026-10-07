# Copyright Axelera AI, 2026
"""The `--eval` legs: what a model scores, and what that score is a gap from.

Accuracy is the one action a recipe cannot do alone. The datasets, the ground
truth and the per-task evaluators live in the Model Zoo, so this is the only
module here that needs `axelera-zoo` installed, and it says which package to
install rather than failing on an import.

Two legs, and the pair is the point:

    aipu    the compiled .axm, on the device
    host    the fp32 weight the build consumes, on this machine

The dataset, the preprocess, the decode tail, the evaluator and the metric are
the same on both, because the recipe hands one pipeline factory to both legs and
only the model stage differs. So the difference between the two numbers is what
deployment cost, and not a difference in how they were measured.

One call scores one leg. Scoring both in one call would double a caller's wall
clock over the dataset without being asked for it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

AIPU = 'aipu'
HOST = 'host'
ENGINES = (AIPU, HOST)

#: `--eval`'s fusion policy when the caller names none: the raw op sequence.
#:
#: `--run` and `--benchmark` default to `auto`, which is what a deployment runs,
#: and the difference is deliberate. The policy selects the preprocess KERNEL
#: (`auto` swaps a cv2 resize for an OpenCL one, `simd` for a SIMD one), so it
#: moves pixels and can move the metric. Every accuracy number on record was
#: taken on the raw sequence, so that is what a bare `--eval` keeps reporting;
#: naming a policy scores the fused preprocess instead, and the result line says
#: which was used.
FUSION_OFF = 'none'

NEEDS_ZOO = (
    '--eval scores against the Model Zoo datasets, ground truth and evaluators; '
    'install `axelera-zoo[eval]`.'
)


@dataclass(frozen=True)
class Target:
    """What a recipe scores against: the task, the split, and the evaluator's arguments.

    A recipe declares this once, as literals, because a recipe IS its resolved
    catalog entry. `task` selects the evaluator, `dataset` and `split` select
    the iterator, and the two kwargs carry what each of those needs and no other
    does: `evaluator_kwargs` a detector's class count, `iterator_kwargs` the
    crop size or mask stride the split must be read at.

    `data_dir` names a copy of the split this card reads instead of the one under
    `--data-root`, and saying so is also what suppresses the fetch. Empty (the
    usual case) means read it from `--data-root` and fetch it there if absent.
    """

    task: str
    dataset: str
    split: str
    data_dir: str = ''
    evaluator_kwargs: dict = field(default_factory=dict)
    iterator_kwargs: dict = field(default_factory=dict)


def require_support() -> None:
    """Refuse a zoo-less `--eval` BEFORE anything expensive runs.

    `--eval` scores the compiled leg by default, so it builds first, and a build
    is minutes. Finding out about a missing package after that wait is the same
    refusal delivered uselessly late, so this is a cheap import at the front.

    What it proves is that the zoo is importable, which is the failure that stops
    every task. The per-task dataset dependencies inside `axelera-zoo[eval]`
    (pycocotools, motmetrics, h5py) are imported by the iterator that needs one,
    so those still surface when the split opens: no single declaration says
    which of them a given dataset needs, and a table here would rot.
    """
    try:
        import axelera.zoo.runner  # noqa: F401
    except ImportError as error:
        raise SystemExit(NEEDS_ZOO) from error


def refusals() -> tuple[type[BaseException], ...]:
    """The exception types a caller should REPORT rather than raise: the refusals.

    `model_stage` names the zoo's `EvalRequestError` for a leg that cannot start,
    and every one of those names its own fix, so a command line prints it as one
    line rather than a traceback. Handed out from here because this is the only
    module in `shared/` that may name the zoo: `cli.py` is imported by every
    verb, including the ones that run with no zoo installed.
    """
    from axelera.zoo.interfaces import EvalRequestError

    return (EvalRequestError,)


def model_stage(
    engine: str,
    *,
    axm_path: Path | None,
    source: Callable[[], Path],
    postamble=None,
    device_config=None,
):
    """The one stage that differs between the two legs.

    `host` runs the source weight through a runtime2 executor, which the zoo
    builds so that this leg and `axzoo eval --engine host` are the same stage
    rather than two that agree today. `aipu` loads the compiled artifact with
    the recipe's own postamble and board settings, the same way its demo
    pipeline does.

    `source` is the recipe's `load_source`, not a path: resolving it downloads
    the weight, and the compiled leg has no use for it.

    A leg that cannot start says so in the words of the leg it is: an absent
    source weight and an absent artifact are different failures with different
    fixes. Both convert HERE rather than at a call site, so a recipe and
    `axzoo eval` refuse identically instead of one of them printing a traceback.
    """
    require_support()
    from axelera.runtime import op
    from axelera.zoo.interfaces import EvalRequestError

    if engine == HOST:
        from axelera.zoo.eval.engines import host_model

        try:
            return host_model(source())
        except EvalRequestError:
            # Already names its own fix (an unreadable suffix, a missing extra).
            raise
        except (FileNotFoundError, ValueError) as error:
            raise EvalRequestError(
                f'--engine {HOST} runs the source weight the build consumes, and it could '
                f'not be materialized ({error}). Use --engine {AIPU} to score the compiled '
                f'artifact instead.'
            ) from error
    kwargs = {}
    if postamble is not None:
        kwargs['using_postamble'] = postamble
    if device_config is not None:
        # The chip leg runs thousands of frames, which is exactly when a board
        # setting the model asked for has to be set. The host leg has no device.
        kwargs['device_config'] = device_config
    try:
        return op.load(str(axm_path), **kwargs)
    except FileNotFoundError as error:
        raise EvalRequestError(
            f'--engine {AIPU} scores the compiled artifact, and it could not be loaded '
            f'({error}). Build it first (`axzoo build <model>`, or the recipe\'s own '
            f'--build), or use --engine {HOST} to score the source weight, which needs '
            f'no artifact.'
        ) from error


def measure(
    target: Target,
    pipeline,
    *,
    engine: str,
    optimize: str = FUSION_OFF,
    data_root=None,
    frames: int = 0,
):
    """Score `pipeline` on `target` and return the zoo's `EvalResult`.

    `pipeline` is already built around the stage `model_stage` returned, so this
    function chooses nothing about WHAT runs: it applies the fusion policy,
    resolves the dataset and the evaluator, and runs the loop.

    `optimize` is the preprocess fusion policy, applied here rather than by the
    caller so a recipe and `axzoo eval` cannot end up scoring two different
    preprocesses. See `FUSION_OFF` for why its default is not `--run`'s.

    `data_root` left unset resolves the same root `axzoo eval` uses, so a recipe
    and the CLI score the same copy of a dataset on one machine.
    """
    require_support()
    from axelera.zoo.interfaces import EvalConfig
    from axelera.zoo.runner import eval_data_root, run_evaluation

    evaluation = EvalConfig(
        task=target.task,
        pipeline=scoring_pipeline(pipeline, optimize),
        dataset=eval_dataset(target),
        evaluator_kwargs=dict(target.evaluator_kwargs),
    )
    return run_evaluation(evaluation, data_root or eval_data_root(), frames=frames, leg=engine)


def scoring_pipeline(pipeline, optimize: str = FUSION_OFF):
    """Apply the preprocess fusion policy `optimize` to `pipeline`.

    One function so the two front doors cannot drift: `measure` calls it for a
    recipe's own `--eval`, and `axzoo eval` calls it on the pipeline it took from
    this recipe. `'none'` rewrites nothing and returns the sequence as declared.
    """
    return pipeline.optimized(optimize)


def eval_dataset(target: Target):
    """The zoo's `EvalDataset` for `target`: one construction, both front doors.

    `axzoo eval` scores through this recipe and builds its own `EvalConfig` around
    the same target, so this is where the two doors agree on how the split is
    read. Written once because listing the fields twice is what dropped
    `iterator_kwargs` and then `data_dir` on one side of the pair: a field added
    to `Target` now reaches both callers, or neither.
    """
    from axelera.zoo.interfaces import EvalDataset

    return EvalDataset(
        name=target.dataset,
        split=target.split,
        data_dir=target.data_dir,
        iterator_kwargs=dict(target.iterator_kwargs),
    )


def format_result(
    result, target: Target, *, model_name: str, engine: str, optimize: str = FUSION_OFF
) -> str:
    """Two lines: every metric, then which one is the headline.

    The engine is printed with the numbers because a number without it is not a
    claim about anything: `host` and `aipu` are two different statements about
    one model. The fusion policy is printed for the same reason, one level down:
    it selects the preprocess kernel, so two runs of one engine under two
    policies are two statements as well.
    """
    metrics = ', '.join(f'{name}={value:.4f}' for name, value in result.as_dict().items())
    # `samples` is what the evaluator scored, which is not the frame count for a
    # task that reads an index set and a query set. 0 means it reports none, so
    # it is left out rather than printed as a sample count of zero.
    counted = f'   samples: {result.samples}' if result.samples else ''
    return (
        f'{model_name} {target.dataset}/{target.split} on {engine} '
        f'(preprocess: {optimize}): {metrics}\n'
        f'key metric: {result.key_metric}{counted}'
    )
