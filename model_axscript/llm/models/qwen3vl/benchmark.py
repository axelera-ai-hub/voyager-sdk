# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Performance benchmark for a compiled Qwen3-VL: the two towers, separately and together.

Two things get measured, because they are different questions:

  * **vision, repeated** -- the same frame encoded `--iters` times: median and best
    latency, and the token cost of a frame. That decides how many camera frames per second
    the tower can absorb, and a repeated dispatch is the only way to see it without the
    rest of a generation in the way.
  * **one whole generation** -- vision, prefill and decode as they actually happen, read
    off `model.stats` (`hf_model.RunStats`). That is the same object the CLI's report and
    the UI's panels print, so a benchmark number and a demo number can be compared.

    A two-token generation primes the decoder first: the first prefill and decode dispatch
    of an instance's life carry one-time kernel setup that no later prompt pays.
    `--no-warmup` reports that cold first run instead.

    python -m model_axscript.llm.models.qwen3vl.benchmark --artifact-dir build/qwen3vl-2b \\
        --image photo.jpg --max-new-tokens 64

    python -m model_axscript.llm.models.qwen3vl.benchmark --artifact-dir ... --vision-only --iters 20

  * **the synthetic sweep** (`--prompt-tokens`) -- a prompt of exactly N filler tokens
    into the compiled context, with the answer length pinned, plus the repeated vision
    dispatch. That is the one path that answers all four spec-sheet figures in a single
    load of the artifact, and it prints them as a `headline` block:

        ViT latency at batch 1 | ViT FPS | LLM TTFT @Nk of Mk | LLM decode tok/s @Nk of Mk

    python -m model_axscript.llm.models.qwen3vl.benchmark --artifact-dir build/benchmark_qwen3vl-2b \\
        --prompt-tokens 1024 --max-new-tokens 64

With no --image a synthetic grey frame is used, so the vision numbers are available
without a dataset. Generation is greedy, so a repeat run is comparable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
from time import perf_counter

import numpy as np

from .pipeline import Qwen3VLPipeline, add_artifact_args, load_image

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = "Describe this image in detail."


def synthetic_image(config):
    """A mid-grey frame at the compiled geometry: the tower's timing input.

    Public because the zoo's perf writer times the same tower and must time it on the SAME
    frame. A vision dispatch is data-independent -- the patch count and the graph are baked
    at build time -- so any frame of the right geometry gives the same latency; what a
    second frame construction would cost is the guarantee that the card's number and the
    report's number are the same measurement.
    """
    from PIL import Image

    height, width = config.vision.image_shape
    return Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))


def benchmark_vision(pipeline: Qwen3VLPipeline, image, iters: int, warmup: int = 3) -> dict:
    """Time the vision tower alone: patches in -> image embeddings out.

    `RunStats` reports the single dispatch a generation made; this repeats it, so the
    median is not one sample.
    """
    from .pipeline import render

    inputs = render(
        pipeline.processor,
        pipeline.config,
        [{"role": "user", "content": [{"type": "image"}]}],
        [image],
    )
    for _ in range(warmup):
        pipeline.vision.encode(inputs.patches)
    times = []
    for _ in range(iters):
        start = perf_counter()
        embeds, deepstack = pipeline.vision.encode(inputs.patches)
        times.append(perf_counter() - start)
    vision = pipeline.config.vision
    return {
        "iters": iters,
        "latency_ms": 1e3 * statistics.median(times),
        "latency_ms_min": 1e3 * min(times),
        "fps": 1.0 / statistics.median(times),
        "patches_per_frame": vision.num_patches,
        "image_tokens_per_frame": int(len(embeds)),
        # The DeepStack mergers are taps on the same dispatch, so they are already in
        # the latency above; count them so a frame's real output size is visible.
        "deepstack_maps": len(deepstack),
        "grid": f"{vision.grid_h}x{vision.grid_w}",
    }


#: Ordinary prose, tiled to whatever length is asked for. Real tokens rather than random
#: ids: a random id can be a special token (an early EOS, an image placeholder with no
#: picture behind it), and then the timed run is not the run that was asked for.
_FILLER = (
    "The harbour wakes slowly in the winter, and the boats come back one at a time. "
    "A list of names is read out on the quay, and somebody writes the weather down. "
)


def _answer(model, inputs, generated) -> str:
    """The generated text alone, with the prompt's own tokens sliced off the front.

    Best-effort: this is evidence a reader compares between reports, and a decoder that
    cannot be read back is not a reason to lose a whole perf row, so a failure here returns
    nothing and `perf_row` then writes no request at all rather than a partial one.
    """
    try:
        prompt_len = inputs["input_ids"].shape[1]
        return model.pipeline.processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0]
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning("could not decode the benchmark answer", exc_info=True)
        return None


def measure_for_report(
    model, *, max_new_tokens: int, vision_iters: int, prompt: str = _DEFAULT_PROMPT, image=None
) -> dict:
    """Both towers timed for `axzoo report`, in the shape its perf writer files a row from.

    The report needs ONE shape from every VLM family. Each family times itself through its
    own API -- this seam exists because the writer used to name THIS family's functions
    directly and could therefore measure only this family: the others reached it and died on
    a vocabulary mismatch (an unregistered `ax_lfm2_vl` model type, then a KeyError on the
    `image_grid_thw` only this family's processor emits), recorded as a FAILED row against
    the MODEL for what was a property of the harness.

    So the family-specific half lives here, beside the functions it calls, and the writer
    holds no family's vocabulary. `prompt` is passed in rather than defaulted per family:
    two families answering different questions are not comparable rows.
    """
    import dataclasses

    # Deferred like `main`'s: the run CLI imports this module, so a module-scope import back
    # into it is a cycle.
    from .qwen3vl_run import build_inputs

    # `image` of None keeps the synthetic frame this module's own CLI times on; the report
    # passes a real one, because its row records the ANSWER and a flat grey rectangle cannot
    # produce one that shows a tower regression. Fitted either way, so the patch grid -- and
    # therefore the dispatch this times -- is the compiled one whatever came in.
    image = model.fit_images(
        image if image is not None else synthetic_image(model.pipeline.config)
    )
    vision = benchmark_vision(model.pipeline, image, vision_iters)

    inputs = build_inputs(model, prompt, images=[image])
    # One warmup request: the first prefill and first decode step on a fresh instance pay
    # kernel setup no second request sees. Two tokens primes both graphs, and the timed run
    # prefills from position 0 again, so it cannot change what the measurement answers.
    model.generate(**inputs.to(model.device), max_new_tokens=2, do_sample=False)
    model.history.clear()  # a warmup is not a run; keep it out of the averages
    # Greedy: a benchmark that samples is not comparable with itself.
    generated = model.generate(
        **inputs.to(model.device), max_new_tokens=max_new_tokens, do_sample=False
    )
    stats = model.stats
    return {
        "prompt": prompt,
        "response": _answer(model, inputs, generated),
        "repo_id": model.pipeline.repo_id,
        "context_len": model.context_len,
        "generation": model.pipeline.generation,
        "cores": model.num_cores,
        "image_size": list(model.pipeline.config.vision.image_shape),
        "load_s": model.load_s,
        "vision": vision,
        "request": {
            "prompt_tokens": stats.prompt_tokens,
            "prefill_ms": stats.prefill_s * 1e3,
            "ttft_ms": stats.ttft_s * 1e3,
            "generated": stats.generated_tokens,
            "decode_tok_s": stats.decode_tok_s,
            "ms_per_token": stats.ms_per_token,
            "end_to_end_tok_s": stats.end_to_end_tok_s,
        },
        "stats": dataclasses.asdict(stats),
    }


def synthetic_ids(model, prompt_tokens: int):
    """Exactly `prompt_tokens` token ids of filler text, as a [1, n] tensor.

    Ids rather than a chat template because the point is an exact length: a template's
    preamble moves with the checkpoint, and 1024 tokens on one model against 1039 on
    another cannot be compared.
    """
    import torch

    tokenizer = model.tokenizer
    ids: list[int] = []
    while len(ids) < prompt_tokens:  # tile until long enough, then cut to length
        ids += tokenizer(_FILLER, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids[:prompt_tokens]], dtype=torch.long)


def benchmark_prompt_length(model, prompt_tokens: int, max_new_tokens: int) -> dict:
    """One greedy generation from a synthetic prompt of exactly `prompt_tokens`.

    `min_new_tokens` pins the decode step count: the filler is not a question, so an
    unpinned run stops wherever it likes and the per-token figures lose their sample size.
    """
    room = model.context_len - prompt_tokens
    if max_new_tokens > room:
        raise ValueError(
            f"{prompt_tokens} prompt + {max_new_tokens} answer exceeds the artifact's "
            f"{model.context_len}-token context; at this prompt length the answer can be "
            f"at most {room} tokens"
        )
    import torch

    ids = synthetic_ids(model, prompt_tokens)
    model.generate(
        input_ids=ids.to(model.device),
        attention_mask=torch.ones_like(ids).to(model.device),
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
    )
    stats = model.stats
    return {
        "prompt_tokens": stats.prompt_tokens,
        "prefill_ms": stats.prefill_s * 1e3,
        "prefill_device_ms": stats.prefill_device_s * 1e3,
        "prefill_host_ms": stats.prefill_host_total_s * 1e3,
        "ttft_ms": stats.ttft_s * 1e3,
        "generated": stats.generated_tokens,
        "decode_tok_s": stats.decode_tok_s,
        "decode_wall_tok_s": stats.decode_wall_tok_s,
        "ms_per_token": stats.ms_per_token,
        "framework_ms_per_token": stats.other_s * 1e3 / max(stats.decode_tokens, 1),
        "end_to_end_tok_s": stats.end_to_end_tok_s,
        "wall_s": stats.total_s,
    }


def benchmark_lengths(model, lengths, max_new_tokens: int, warmup: bool = True) -> list[dict]:
    """`benchmark_prompt_length` over several prompt lengths, warmed up once.

    One warmup for the whole sweep: the decoder's graphs are compiled for the whole context
    and do not vary with the prompt, so the one-time setup is paid once.
    """
    if warmup:
        import torch

        ids = synthetic_ids(model, min(lengths))
        model.generate(
            input_ids=ids.to(model.device),
            # Passed for the same reason the timed runs pass it: without a mask
            # transformers infers one from a pad token that equals EOS, and warns.
            attention_mask=torch.ones_like(ids).to(model.device),
            max_new_tokens=2,
            min_new_tokens=2,
            do_sample=False,
        )
        model.history.clear()  # the warmup is not a run; keep it out of the averages
    return [benchmark_prompt_length(model, length, max_new_tokens) for length in lengths]


def _report_lengths(rows: list[dict], context_len: int) -> None:
    """The sweep as a table, one line per prompt length.

    `prefill` sits next to `prompt` on purpose: the prefill graph ingests the whole
    compiled context whatever the prompt's length, so that column stays flat while the
    prompt grows tenfold.
    """
    print(f"\nsynthetic prompts, greedy, into a {context_len}-token context")
    print(
        f"  {'prompt':>8}  {'prefill':>9}  {'TTFT':>8}  {'decode':>10}  "
        f"{'wall':>10}  {'per token':>10}  {'end to end':>11}"
    )
    for row in rows:
        print(
            f"  {row['prompt_tokens']:>8}  {row['prefill_ms']:>7.0f} ms  "
            f"{row['ttft_ms']:>6.0f} ms  {row['decode_tok_s']:>6.2f} tok/s  "
            f"{row['decode_wall_tok_s']:>6.2f} tok/s  {row['ms_per_token']:>7.1f} ms  "
            f"{row['end_to_end_tok_s']:>6.2f} tok/s"
        )
    if len(rows) > 1:
        first, last = rows[0], rows[-1]
        growth = last["prompt_tokens"] / max(first["prompt_tokens"], 1)
        change = last["prefill_ms"] / max(first["prefill_ms"], 1e-9)
        print(
            f"\n  {growth:.0f}x the prompt cost {change:.2f}x the prefill: the graph is "
            f"compiled for {context_len} positions and the prompt is padded up to it, so "
            "TTFT is a property of the artifact, not of the prompt."
        )


def _kilo(tokens: int) -> str:
    """``1024`` -> "1k", ``4096`` -> "4k", ``1500`` -> "1500": how a spec sheet says it."""
    return f"{tokens // 1024}k" if tokens and tokens % 1024 == 0 else str(tokens)


def _report_headline(results: dict, context_len: int) -> None:
    """The four figures a spec sheet quotes, in the units it quotes them in.

    Both towers in one block because a card is chosen on both: the tower's batch-1 latency
    and the frame rate that follows from it, and the decoder's TTFT and decode rate at a
    stated occupancy of a stated context ("@1k of 4k"). Nothing is recomputed here -- these
    are the same numbers the tables above print, said once and named the way they are asked
    for.
    """
    vision = results.get("vision")
    print("\nheadline")
    if vision:
        print(
            f"  ViT  latency (batch 1)   {vision['latency_ms']:>8.1f} ms   "
            f"(best {vision['latency_ms_min']:.1f} ms of {vision['iters']} runs, "
            f"{vision['grid']} grid)"
        )
        print(f"  ViT  throughput          {vision['fps']:>8.1f} FPS")
    for row in results.get("lengths") or []:
        at = f"@{_kilo(row['prompt_tokens'])} of {_kilo(context_len)}"
        print(f"  LLM  TTFT {at:<14} {row['ttft_ms']:>8.0f} ms")
        print(
            f"  LLM  decode {at:<12} {row['decode_tok_s']:>8.2f} tok/s "
            f"(wall {row['decode_wall_tok_s']:.2f} tok/s)"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen3vl benchmark", description=__doc__)
    add_artifact_args(parser)
    parser.add_argument(
        "--image",
        default=None,
        help="frame to encode -- a path or an http(s) URL (default: synthetic grey)",
    )
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10, help="vision-tower repetitions")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="time the first generation cold, one-time kernel setup included",
    )
    parser.add_argument(
        "--prompt-tokens",
        default=None,
        metavar="N[,N...]",
        help="synthetic text prompt(s) of exactly N tokens -- the decode rate at a given "
        "context occupancy, e.g. --prompt-tokens 1024 on a 4096-token artifact for "
        "\"tok/s @1k of 4k\". A comma-separated list sweeps them in one run.",
    )
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--text-only", action="store_true", help="decoder without an image")
    parser.add_argument("--json", action="store_true", help="emit the raw results as JSON")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    from .hf_model import AxQwen3VLForConditionalGeneration
    from .qwen3vl_run import build_inputs, print_performance

    # The HF model rather than the bare pipeline: `generate()` is what fills in
    # `model.stats`, and the tower is still reachable through `model.pipeline`.
    model = AxQwen3VLForConditionalGeneration.from_artifact(
        args.artifact_dir, args.repo_id, generation=args.generation
    )

    # A synthetic sweep is a decoder measurement: no picture, exact prompt lengths, a
    # pinned answer length. It answers "tok/s at N of the compiled context", which the
    # single-prompt path below cannot, its length being whatever the question tokenizes to.
    if args.prompt_tokens:
        lengths = [int(n) for n in str(args.prompt_tokens).split(",") if n.strip()]
        results = {
            "repo_id": model.pipeline.repo_id,
            "context_len": model.context_len,
            "generation": model.pipeline.generation,
            "cores": model.num_cores,
            "max_new_tokens": args.max_new_tokens,
        }
        # The tower is timed in the same run: a card is quoted on both its frame rate and
        # its decode rate, and reading them off two invocations of this script means two
        # loads of the same artifact. --text-only skips it.
        if not args.text_only:
            image = (
                model.fit_images(load_image(args.image))
                if args.image
                else synthetic_image(model.pipeline.config)
            )
            results["vision"] = benchmark_vision(model.pipeline, image, args.iters)
        results["lengths"] = benchmark_lengths(
            model, lengths, args.max_new_tokens, warmup=not args.no_warmup
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(
                f"model: {results['repo_id']}  ({results['generation']}, "
                f"context_len {results['context_len']}, {results['cores']} core(s))"
            )
            _report_lengths(results["lengths"], model.context_len)
            _report_headline(results, model.context_len)
        return 0

    image = None
    if not args.text_only:
        image = (
            model.fit_images(load_image(args.image))
            if args.image
            else synthetic_image(model.pipeline.config)
        )

    results: dict = {
        "repo_id": model.pipeline.repo_id,
        "context_len": model.context_len,
        "generation": model.pipeline.generation,
        "cores": model.num_cores,
    }
    if image is not None:
        results["vision"] = benchmark_vision(model.pipeline, image, args.iters)
    if not args.vision_only:
        inputs = build_inputs(model, args.prompt, images=[image] if image is not None else None)
        if not args.no_warmup:
            # As evaluation.benchmark does for the other models: the first prefill and
            # first decode step on a fresh instance pay one-time kernel setup that no
            # second prompt sees. Two tokens primes both graphs; the timed run prefills
            # from position 0 again, so the warmup cannot change what it answers.
            model.generate(**inputs.to(model.device), max_new_tokens=2, do_sample=False)
            model.history.clear()  # the warmup is not a run; keep it out of the averages
        # Greedy: a benchmark that samples is not comparable with itself.
        model.generate(
            **inputs.to(model.device), max_new_tokens=args.max_new_tokens, do_sample=False
        )
        results["stats"] = dataclasses.asdict(model.stats)
        results["derived"] = {f"{s}.{m}": v for s, m, v in model.stats.rows()}

    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    _report(results)
    if "stats" in results:
        print_performance(model)
    return 0


def _report(results: dict) -> None:
    print(
        f"model: {results['repo_id']}  ({results['generation']}, "
        f"context_len {results['context_len']}, {results['cores']} core(s))"
    )
    vision = results.get("vision")
    if vision:
        print("\nvision tower, repeated dispatch")
        print(f"  grid                 {vision['grid']} ({vision['patches_per_frame']} patches)")
        print(f"  image tokens/frame   {vision['image_tokens_per_frame']}")
        print(
            f"  latency              {vision['latency_ms']:.1f} ms "
            f"(best {vision['latency_ms_min']:.1f} ms, {vision['iters']} runs)"
        )
        print(f"  throughput           {vision['fps']:.1f} img/s")


if __name__ == "__main__":
    raise SystemExit(main())
