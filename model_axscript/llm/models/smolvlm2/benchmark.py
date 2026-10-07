# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""On-device SmolVLM2 performance, vision and text measured separately.

Two numbers, from one compiled artifact (built by ``llm compile
--implementation smolvlm2-metis-multicore``), on one device connection:

  * VISION -- frames/s. Times ``SiglipVisionDevice.encode`` on one untiled frame,
    which is exactly what a video frame costs: HF's video scaffold gives each frame
    a single image block, so frames/s == 1/encode_latency. Reported alongside the
    encoder-only latency (``encode_vision``), so a low fps can be attributed to the
    SigLIP graph rather than to the host pixel-shuffle + connector dispatch that
    ``encode`` adds on top.
  * TEXT -- decode tokens/s (steady state, prefill excluded) + TTFT, from a
    text-only prompt through the text decoder artifact. No image embeds are
    spliced, so nothing of the vision path is in this number.

Deliberately NOT one end-to-end figure: the two stages have different units and
scale with different things (frames vs prompt length), so a combined tok/s would
hide both. A real image/video turn costs
``n_frames / vision_fps + ttft(prompt + n_frames*image_seq_len) + n_new/tok_per_s``.

Run it (needs a device):

    python -m model_axscript.llm.models.smolvlm2.benchmark --artifact-dir ~/smolvlm-artifact
    python -m model_axscript.llm.models.smolvlm2.benchmark --artifact-dir ~/smolvlm-artifact \
        --frames 32 --max-new-tokens 128 --json

Frames are synthetic noise at the checkpoint's own tile geometry: latency is
data-independent, and this keeps the measurement free of image fetching and frame
decoding (host costs that are not the device's throughput). Run a real clip through
``smolvlm2_run`` when you want the host preprocessing in the total.
"""

from __future__ import annotations

import argparse
import json
import logging
from math import isqrt
from pathlib import Path
from statistics import median
from time import perf_counter

from ...shared.config import GenerationConfig

#: The text seed for a direct caller of `measure_for_report`. Taken from the one place a
#: default prompt is declared rather than spelled again here; `axzoo report` passes the VLM
#: question of its own, so this is only what a bare call gets.
_DEFAULT_PROMPT = GenerationConfig.prompt


def measure_vision(vision, *, frames: int, warmup: int) -> dict:
    """Time the vision path per frame: full ``encode`` and encoder-only
    ``encode_vision``, both on one untiled frame. Returns medians + fps."""
    import torch

    # One frame at the tile geometry the graphs were compiled for. seq_real patches of
    # patch_size => a (patch_size*sqrt(seq_real))^2 tile; that is the checkpoint's tile size
    # up to the trailing partial patch the conv crops anyway (2.2B: 378 vs 384), so the
    # patch count -- and therefore the latency -- is the compiled one.
    side = vision.patch_size * isqrt(vision.seq_real)
    pixels = torch.randn(1, 3, side, side)

    for _ in range(warmup):  # prime the graphs (first dispatch pays kernel setup)
        vision.encode(pixels)

    def _times(fn):
        out = []
        for _ in range(frames):
            t = perf_counter()
            fn(pixels)
            out.append(perf_counter() - t)
        return median(out)

    encode_s = _times(vision.encode)
    encoder_s = _times(vision.encode_vision)
    return {
        "tile_px": side,
        "patches": vision.seq_real,
        "tokens_per_frame": vision.num_tokens,
        "frames": frames,
        "encode_ms": encode_s * 1e3,
        "encoder_only_ms": encoder_s * 1e3,
        "connector_ms": (encode_s - encoder_s) * 1e3,
        "fps": 1.0 / encode_s,
    }


def _synthetic_tile(side: int):
    """One noise picture at the compiled tile geometry, as PIL RGB.

    Noise rather than a photograph for the reason the module header gives: latency is
    data-independent, and a fetched picture would put image download and decode in a number
    that is meant to be the device's. RGB because the tower is 3-channel by construction.
    """
    import numpy as np
    from PIL import Image

    pixels = np.random.default_rng(0).integers(0, 256, (side, side, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _checkpoint_tile(model) -> int:
    """The tile side the checkpoint states, off the loaded artifact's own config.

    This is the geometry the card declares and therefore the one a report row is filed under.
    Refused rather than defaulted: the compiled patch grid is a plausible-looking alternative
    that is a few pixels smaller on the 2.2B, and a row filed under it would read as an
    undeclared run size in a file whose whole job is to be comparable.
    """
    vision = getattr(model.config, "vision_config", None)
    side = (
        vision.get("image_size")
        if isinstance(vision, dict)
        else getattr(vision, "image_size", None)
    )
    if not side:
        raise RuntimeError(
            "this artifact's config states no `vision_config.image_size`, so the tile the "
            "checkpoint was built for -- what the card declares and what a row is filed "
            "under -- cannot be read; the compiled patch grid is NOT a substitute for it."
        )
    return int(side)


logger = logging.getLogger(__name__)


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


def measure_for_report(
    model, *, max_new_tokens: int, vision_iters: int, prompt: str = _DEFAULT_PROMPT, image=None
) -> dict:
    """Both towers timed for `axzoo report`, in the shape its perf writer files a row from.

    The report needs ONE shape from every VLM family, and each family times itself through
    its own API: this family's vision number comes from `measure_vision` (the median of
    `vision_iters` encodes of one frame) and its request numbers from the `RunStats` the
    shim's own `forward` fills in, because nothing else here knows where a dispatch ends and
    the host's lm_head begins.

    `latency_ms` is the report's name for one picture through the tower; this family calls
    the same median `encode_ms`, and it is the FULL encode -- SigLIP plus the host
    pixel-shuffle and the connector dispatch -- because that is what a picture costs a
    caller. `encoder_only_ms` rides along in the same block, so a tower regression and a
    connector regression are still separable by a reader of the measurement.

    `prompt` is passed in rather than defaulted per family: two families answering different
    questions are not comparable rows.
    """
    import dataclasses

    # Deferred: the run module imports this one for its own benchmark verb, so a module-scope
    # import back into it is a cycle.
    from .smolvlm2_run import build_inputs

    tower = model.pipeline.vision
    vision = measure_vision(tower, frames=vision_iters, warmup=1)
    # The CHECKPOINT's tile, not the compiled patch grid. They differ on the 2.2B: its tower
    # spans 27 patches of 14 px = 378, because the patch conv crops the trailing partial
    # patch of a 384 px tile (`measure_vision` says so where it makes its frame). 378 is the
    # right number to time a graph with and the WRONG number to file a row under -- the card
    # declares 384 (`TILE_SIZE`), so a row carrying 378 states an identity nobody declared
    # and can never pair with a committed one.
    side = _checkpoint_tile(model)
    # None keeps the synthetic tile this module's CLI times on; the report passes a real
    # picture, because its row records the ANSWER and noise cannot produce one that shows a
    # tower regression. Resized to the CHECKPOINT tile for the reason the side is computed
    # that way -- the tower spans that many pixels -- so the dispatch is unchanged.
    frame = _synthetic_tile(side) if image is None else image.convert("RGB").resize((side, side))
    inputs = build_inputs(model, prompt, images=[frame])
    # One warmup request: a fresh instance pays kernel setup on its first prefill and first
    # decode step that no second request sees. Two tokens primes both graphs, and the timed
    # run prefills from position 0 again, so it cannot change what the measurement answers.
    model.generate(**inputs, max_new_tokens=2, do_sample=False)
    model.history.clear()  # a warmup is not a run; keep it out of the session averages
    # Greedy: a benchmark that samples is not comparable with itself.
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    stats = model.stats
    return {
        "prompt": prompt,
        "response": _answer(model, inputs, generated),
        "repo_id": model.repo_id,
        "context_len": model.context_len,
        "generation": model.generation,
        "cores": model.num_cores,
        "image_size": [side, side],
        "load_s": model.load_s,
        "vision": {**vision, "latency_ms": vision["encode_ms"]},
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


def measure_text(
    runner,
    artifact_dir: Path,
    manifest: dict,
    *,
    prompt: str,
    max_new_tokens: int,
    prompt_tokens: int | None = None,
):
    """Time text-only generation on the decoder artifact: TTFT + steady-state tok/s.

    ``prompt_tokens`` benchmarks a synthetic prompt of that many tokens (the seed text
    tiled, via ``EvalContext.perf_prompt``) instead of the seed itself: TTFT scales with
    prompt length, so a number is only comparable at a stated length (e.g. 1k of a 4k
    context). Clamped to context_len - 1 there, so one token can still be generated.
    """
    from ...generate import GenerationConfig, benchmark_generation
    from ...evaluation.metrics import EvalContext

    if runner.tokenizer is None:
        raise SystemExit(
            "no tokenizer for this artifact (offline HF cache?); the text measurement "
            "needs one to tokenize the prompt"
        )
    config = GenerationConfig(
        implementation=manifest["implementation"],
        repo_id=manifest["repo_id"],
        context_len=int(manifest["context_len"]),
        chunk_len=manifest.get("chunk_len"),
        max_new_tokens=max_new_tokens,
    )
    resolved = EvalContext(
        config, prompt=prompt, prompt_tokens=prompt_tokens, tokenizer=runner.tokenizer
    ).perf_prompt()
    result = benchmark_generation(
        runner.text_runtime,
        runner.tokenizer,
        artifact_dir / manifest["text_dir"],
        resolved,
        config,
    )
    result["prompt_tokens"] = (
        len(resolved)
        if not isinstance(resolved, str)
        else len(runner.tokenizer(resolved).input_ids)
    )
    return result


def format_report(vision: dict, text: dict, manifest: dict) -> str:
    # encode and encode_vision are timed in separate loops, so a run too short to settle can
    # put the full path UNDER the encoder alone. Say so rather than print a negative split.
    split = (
        f"{vision['encoder_only_ms']:.1f} encoder + {vision['connector_ms']:.1f} shuffle+connector"
        if vision["connector_ms"] >= 0
        else f"encoder alone read {vision['encoder_only_ms']:.1f}, ABOVE the full path"
        f" -- run has not settled, raise --frames/--warmup"
    )
    return "\n".join(
        [
            f"smolvlm2 on-device performance -- {manifest['repo_id']}"
            f" (cores={manifest.get('num_cores', 1)}, context_len={manifest['context_len']})",
            "",
            f"  vision  {vision['fps']:.3g} fps   ({vision['encode_ms']:.1f} ms/frame: {split})",
            f"          {vision['tile_px']}px tile, {vision['patches']} patches ->"
            f" {vision['tokens_per_frame']} prompt tokens/frame,"
            f" median of {vision['frames']}",
            "",
            f"  text    {text['decode_tok_per_s']:.3g} tok/s decode"
            f"   (ttft {text['ttft_s'] * 1e3:.0f} ms at a {text['prompt_tokens']}-token"
            f" prompt of {manifest['context_len']} ctx, {text['tokens']} generated)",
        ]
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m model_axscript.llm.models.smolvlm2.benchmark",
        description="Measure SmolVLM2 vision fps and text tok/s on device, separately.",
    )
    p.add_argument("--artifact-dir", required=True, help="compiled SmolVLM2 artifact dir")
    p.add_argument(
        "--frames",
        type=int,
        default=64,
        help="frames to time (median is reported). Below ~32 the median is still riding the "
        "first dispatches and reads fast enough to invert the encoder/connector split.",
    )
    p.add_argument("--warmup", type=int, default=8, help="untimed warmup frames")
    # None, so `EvalContext.perf_prompt` falls through to the config's declared prompt
    # instead of this parser restating it. Today's string is unchanged either way.
    p.add_argument(
        "--prompt",
        default=None,
        help="text-only prompt for the tok/s run (default: the declared seed prompt)",
    )
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="benchmark a synthetic prompt of this many tokens (--prompt tiled) instead of "
        "--prompt itself: TTFT scales with prompt length, so quote it at a stated length "
        "(e.g. 1024 of a 4096 context). Clamped to context_len - 1.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--skip-text", action="store_true")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    return p


def measure(argv=None) -> dict:
    args = _parser().parse_args(argv)
    from .multicore_impl import VLM_MANIFEST, load_runner

    artifact_dir = Path(args.artifact_dir).expanduser()
    manifest = json.loads((artifact_dir / VLM_MANIFEST).read_text())
    runner = load_runner(artifact_dir)  # one connection: text + vision + connector

    vision = (
        {}
        if args.skip_vision
        else measure_vision(runner.vision, frames=args.frames, warmup=args.warmup)
    )
    text = (
        {}
        if args.skip_text
        else measure_text(
            runner,
            artifact_dir,
            manifest,
            prompt=args.prompt,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
    )
    results = {"repo_id": manifest["repo_id"], "vision": vision, "text": text}
    # The text table wants both halves; a --skip-* run just prints what it measured.
    if args.json or not (vision and text):
        print(json.dumps(results, indent=2))
    else:
        print(format_report(vision, text, manifest))
    return results


def main(argv=None) -> int:
    """CLI entry: measure, print, and report a STATUS -- not the measurements.

    The family entry does ``return action_main(argv) or 0`` and the model's ``__main__``
    passes that to ``sys.exit``, so returning the results dict here would exit 1 with the
    dict dumped to stderr. Callers that want the numbers call ``measure`` directly.
    """
    measure(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
