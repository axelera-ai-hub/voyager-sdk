# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""What a compiled LFM2.5-VL artifact costs: vision, prefill, decode, end to end.

Three figures, each measured where the work happens and each runnable on its own::

    python -m model_axscript.llm.models.lfm2vl.benchmark --artifact-dir build/lfm2.5-vl-450m --image photo.jpg
    python -m model_axscript.llm.models.lfm2vl.benchmark --artifact-dir build/lfm2.5-vl-450m --vision-only
    python -m model_axscript.llm.models.lfm2vl.benchmark --artifact-dir build/lfm2.5-vl-450m --text-only --json
    python -m model_axscript.llm.models.lfm2vl.benchmark --artifact-dir build/lfm2.5-vl-450m \
        --image photo.jpg --prompt-tokens 1024     # one picture in a 1k-token prompt

* **vision** -- one dispatch per compiled grid, timed over ``--iters`` runs after a warm
  one. A split picture is several dispatches, so ``--tiles`` measures what a
  high-resolution prompt actually pays.
* **prefill** -- the decoder ingesting the whole prompt in one call. The graph is compiled
  for the artifact's context, so this is a constant, not a function of the prompt.
* **decode** -- device dispatch plus the host lm_head, per generated token, reported
  separately because the host share is not noise: the lm_head is a [vocab x hidden] fp16
  gemv on the host, and it grows with the model like everything else.

Medians, not means: a single slow first run (page faults, clock ramp) would otherwise
dominate, and the median is what a steady stream of prompts sees.

One device setting dominates every figure below and is not this code's to set:
``axtrace --slog-level err``, worth ~15x on every stage. The benchmark cannot detect it.
"""

from __future__ import annotations

import argparse
import json as jsonlib
import logging
import statistics
import time

import numpy as np

logger = logging.getLogger(__name__)

_PROMPT = "Describe this image in detail."


def _parser() -> argparse.ArgumentParser:
    from .pipeline import add_artifact_args

    parser = argparse.ArgumentParser(prog="lfm2vl benchmark", description=__doc__.splitlines()[0])
    add_artifact_args(parser)
    parser.add_argument("--image", default=None, metavar="PATH_OR_URL")
    parser.add_argument("--iters", type=int, default=10, help="Timed runs after the warm one.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Pad the question with filler until the whole prompt is ~N tokens, image "
        "tokens included. What a long context actually costs, without needing a long "
        "document to hand.",
    )
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--tiles",
        dest="tiles",
        action="store_true",
        default=None,
        help="Time a SPLIT picture: several dispatches and a much longer prompt.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the numbers as JSON.")
    return parser


def _synthetic_image(model):
    """A picture at the compiled geometry, so no download is needed to time the tower."""
    from PIL import Image

    height, width = model.image_shape
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def synthetic_prompt(model, target: int, images, *, tiles=None) -> str:
    """The question padded with filler so the rendered prompt is about ``target`` tokens.

    Rendered, not estimated: the chat template, the image markers and one row per image
    token all count, and how many of those there are depends on the geometry. The filler's
    token cost is measured rather than assumed to be 1, then the count is corrected against
    the real length a few times; the closest candidate that does not exceed the target
    wins, and the benchmark reports the length it actually got.
    """
    filler = " lorem ipsum dolor sit amet"

    def length(text: str) -> int:
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in images or []),
                    {"type": "text", "text": text},
                ],
            }
        ]
        inputs = model.process(messages, images=images or None, split=tiles)
        return int(inputs["input_ids"].shape[-1])

    base = length(_PROMPT)
    if base >= target:  # the image rows alone are already longer than asked for
        return _PROMPT
    per = max(1.0, (length(_PROMPT + filler * 8) - base) / 8)
    best, best_len, count = _PROMPT, base, round((target - base) / per)
    for _ in range(6):
        count = max(0, count)
        text = _PROMPT + filler * count
        got = length(text)
        if best_len < got <= target:
            best, best_len = text, got
        if got == target:
            break
        count += round((target - got) / per)
    return best


def vision_benchmark(model, image, iters: int, tiles=None) -> dict:
    """Median seconds per dispatch and per picture, plus the token count."""
    from .pipeline import render

    pipeline = model.pipeline
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    inputs = render(pipeline.processor, pipeline.config, messages, [image], split=tiles)
    dispatches = inputs.patches.shape[0] // pipeline.config.vision.num_patches
    pipeline.vision.encode(inputs.patches)  # warm
    times = []
    for _ in range(iters):
        started = time.perf_counter()
        embeds = pipeline.vision.encode(inputs.patches)
        times.append(time.perf_counter() - started)
    per_picture = statistics.median(times)
    return {
        "dispatches": dispatches,
        "tokens": int(embeds.shape[0]),
        "picture_ms": per_picture * 1e3,
        "dispatch_ms": per_picture / dispatches * 1e3,
        "pictures_per_s": 1.0 / per_picture,
    }


def text_benchmark(model, iters: int, max_new_tokens: int, prompt: str = _PROMPT) -> dict:
    """Prefill latency and decode tok/s on a text-only prompt of the same shape."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = model.process(messages)
    stats = None
    prefill, decode = [], []
    for _ in range(iters + 1):  # the first is the warm one
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        stats = model.stats
        prefill.append(stats.prefill_device_s)
        decode.extend(stats.step_device_s)
    host = statistics.median(stats.step_host_s) if stats.step_host_s else float("nan")
    return {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "context_len": model.context_len,
        "prefill_ms": statistics.median(prefill) * 1e3,
        "decode_device_ms": statistics.median(decode) * 1e3,
        "decode_host_ms": host * 1e3,
        "decode_tok_s": 1.0 / (statistics.median(decode) + host),
    }


def _answer(model, inputs, generated) -> str | None:
    """The generated text alone, with the prompt's own tokens sliced off the front.

    Best-effort: this is evidence a reader compares between reports, and a decoder that
    cannot be read back is not a reason to lose a whole perf row, so a failure returns
    nothing and the writer then records no request at all rather than a partial one.
    """
    try:
        prompt_len = inputs["input_ids"].shape[1]
        return model.processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning("could not decode the benchmark answer", exc_info=True)
        return None


def end_to_end(model, image, max_new_tokens: int, tiles=None, prompt: str = _PROMPT) -> dict:
    """One real prompt: time to first token and the wall clock, from the run's own stats.

    `answer` rides along because this is the call that generated it: a report row records
    what the model replied, and re-generating it afterwards would be a second run whose text
    is not the one that was timed.
    """
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    inputs = model.process(messages, images=[image], split=tiles)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    stats = model.stats
    return {
        "answer": _answer(model, inputs, generated),
        "prompt_tokens": stats.prompt_tokens,
        "vision_ms": stats.vision_s * 1e3,
        "prefill_ms": stats.prefill_device_s * 1e3,
        "ttft_s": stats.ttft_s,
        "decode_tok_s": stats.decode_tok_s,
        "generated": stats.generated_tokens,
        "wall_s": stats.total_s,
    }


def measure_for_report(
    model, *, max_new_tokens: int, vision_iters: int, prompt: str = _PROMPT, image=None
) -> dict:
    """Both towers timed for `axzoo report`, in the shape its perf writer files a row from.

    The report needs ONE shape from every VLM family. Each family times itself through its
    own API -- this seam exists because the writer used to name one family's functions
    directly and could therefore only measure that family: the others reached it and died on
    a vocabulary mismatch -- a KeyError on `image_grid_thw`, which only Qwen3-VL's processor
    emits -- and the report committed that as a FAILED row against the MODEL, for what was a
    property of the harness.

    So the family-specific half lives here, beside the functions it calls, and the writer
    holds no family's vocabulary. `prompt` is passed in rather than defaulted per family:
    two families answering different questions are not comparable rows.
    """
    import dataclasses

    # None keeps the synthetic frame this module's CLI times on; the report passes a real
    # picture, because its row records the ANSWER and noise cannot produce one that shows a
    # tower regression. `process` fits whatever comes in to the compiled geometry, so the
    # dispatch being timed is the same either way.
    image = image if image is not None else _synthetic_image(model)
    vision = vision_benchmark(model, image, vision_iters)
    # UNSPLIT, deliberately: a tile is one dispatch of the same graph, so a split picture
    # times the same tower several times over and carries a much longer prompt with it. One
    # dispatch is the comparable number; `--tiles` on this module's CLI measures the split.
    #
    # One warmup request first -- a fresh instance pays kernel setup on its first prefill
    # and first decode step that no second request sees. The timed run prefills from
    # position 0 again, so priming cannot change what it answers.
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    model.generate(
        **model.process(messages, images=[image], split=None),
        max_new_tokens=2,
        do_sample=False,
    )
    model.history.clear()  # a warmup is not a run; keep it out of the session averages
    request = end_to_end(model, image, max_new_tokens, prompt=prompt)
    stats = model.stats
    return {
        "prompt": prompt,
        "response": request["answer"],
        "repo_id": model.pipeline.repo_id,
        "context_len": model.context_len,
        "generation": model.pipeline.generation,
        "cores": model.num_cores,
        "image_size": list(model.image_shape),
        "load_s": model.load_s,
        # `latency_ms` is the report's name for one picture through the tower; this family
        # calls the same median `picture_ms`. Renamed HERE so the writer keeps one name.
        "vision": {**vision, "latency_ms": vision["picture_ms"]},
        "request": {
            "prompt_tokens": request["prompt_tokens"],
            # The WHOLE prefill -- device dispatch plus the host input prep and lm_head --
            # taken from the stats and not from `end_to_end`, whose `prefill_ms` is
            # `prefill_device_s` alone. Two reasons, and the first one is inside this row:
            # `ttft_ms` below is `stats.ttft_s`, which is vision + the full prefill, so with
            # the narrower number here `ttft_ms - vision_ms` did not equal `prefill_ms`. The
            # second is across rows: Qwen3-VL files `stats.prefill_s`, and `axzoo benchmark`
            # prints both under one `prefill` label, so a reader comparing two families was
            # comparing two different quantities. `end_to_end` keeps the narrower meaning for
            # this family's own CLI, where the device/host split is the point.
            "prefill_ms": stats.prefill_s * 1e3,
            "ttft_ms": request["ttft_s"] * 1e3,
            "generated": request["generated"],
            "decode_tok_s": request["decode_tok_s"],
            "ms_per_token": stats.ms_per_token,
            "end_to_end_tok_s": stats.end_to_end_tok_s,
        },
        "stats": dataclasses.asdict(stats),
    }


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    from .lfm2vl_run import load
    from .pipeline import load_image

    model = load(args)
    image = load_image(args.image) if args.image else _synthetic_image(model)
    results = {
        "artifact": str(args.artifact_dir),
        "geometry": model.image_geometry,
        "cores": model.num_cores,
        "generation": model.pipeline.generation,
        "image_tokens": model.pipeline.config.vision.num_merged,
    }
    if not args.text_only:
        results["vision"] = vision_benchmark(model, image, args.iters, args.tiles)
    if not args.vision_only:
        # --prompt-tokens counts the image tokens too, so the text-only figure is padded to
        # the same total as the end-to-end one: the decoder sees the same length either way.
        text_prompt = (
            synthetic_prompt(model, args.prompt_tokens, None) if args.prompt_tokens else _PROMPT
        )
        results["text"] = text_benchmark(
            model, args.iters, args.max_new_tokens, prompt=text_prompt
        )
    if not (args.vision_only or args.text_only):
        prompt = (
            synthetic_prompt(model, args.prompt_tokens, [image], tiles=args.tiles)
            if args.prompt_tokens
            else _PROMPT
        )
        results["end_to_end"] = end_to_end(
            model, image, args.max_new_tokens, args.tiles, prompt=prompt
        )

    if args.json:
        print(jsonlib.dumps(results, indent=2))
        return 0
    print(
        f"\n{results['geometry']} geometry, {results['image_tokens']} image tokens, "
        f"{results['cores']} core(s), {results['generation']}"
    )
    for section in ("vision", "text", "end_to_end"):
        if section not in results:
            continue
        print(f"  {section}")
        width = max(len(k) for k in results[section])
        for name, value in results[section].items():
            shown = f"{value:.2f}" if isinstance(value, float) else str(value)
            print(f"    {name:<{width}} {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
