# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL on an Axelera accelerator, driven exactly like a HuggingFace model.

Every feature a VLM customer expects from `transformers` -- chat templates, one or
several images, video, sampling parameters, token-by-token streaming -- against a
compiled Axelera artifact. `example()` below is the Qwen3-VL model card's own
quickstart, runnable, with two lines changed: the model class
(`AutoModelForImageTextToText.from_pretrained("build/qwen3vl-2b")`, pointed at the
artifact) and a `fit_images` call. Running the module with no flag but --artifact-dir
runs it.

`fit_images` is the one Axelera-specific call: the vision tower is compiled for ONE patch
grid, so pictures are resized to it before the processor picks a size of its own. Without
it the run stops with a message naming the size to use -- never a quietly wrong answer
about a differently-cropped picture. It is also why images are handed to the template as
objects rather than as URLs for the processor to fetch at its own size.

Run it (needs a device):

    python -m model_axscript.llm.models.qwen3vl.qwen3vl_run --artifact-dir build/qwen3vl-2b
    python -m model_axscript.llm.models.qwen3vl.qwen3vl_run --artifact-dir build/qwen3vl-2b --image photo.jpg \\
        --prompt "What is in this picture?" --top-p 0.9 --min-p 0.05
    python -m model_axscript.llm.models.qwen3vl.qwen3vl_run --artifact-dir build/qwen3vl-2b \\
        --image a.jpg https://example.com/b.jpg --prompt "Compare them." --greedy
    python -m model_axscript.llm.models.qwen3vl.qwen3vl_run --artifact-dir build/qwen3vl-2b \\
        --video https://upload.wikimedia.org/.../clip.webm --frames 8
    python -m model_axscript.llm.models.qwen3vl.qwen3vl_run --artifact-dir build/qwen3vl-2b \\
        --image photo.jpg --interactive        # keep asking; /image /video /reset /exit

`--interactive` is the same `generate()` call in a loop: the conversation so far is
re-rendered and re-prefilled each turn (the device holds one position-indexed KV cache),
and `/image` or `/video` swaps what is attached and clears the history with it.

`axelera.llm.demo_qwen3vl` is the same demo with a browser page instead of flags
(upload or paste paths/links, type a prompt, watch it answer) --
`python -m axelera.llm.demo_qwen3vl`.

Either way the device timings come out as they are known: the vision + prefill cost the
moment the first token lands, then the full line when the answer ends --

    [vision 89 ms (1 dispatch) + prefill 337 ms (78 tokens) = TTFT 425 ms]
    This is a painting titled "The Forest of the Garden of the Palace of the King…
    [vision 89 ms (1 dispatch) | prefill 337 ms (78 tokens, 232 tok/s) | TTFT 425 ms |
     38 tokens at 9.87 tok/s (101 ms/token, 14% host) | total 4.17 s]

Sampler flags left unset are not passed to `generate()` at all -- the artifact's own
`generation_config` decides, which is what a stock HF model would do. That config is the
checkpoint's, saved into the artifact at build time and unmodified (Qwen3-VL-2B ships
`do_sample=true, temperature=0.7, top_p=0.8, top_k=20`). `--greedy` pins it down; the
Technical Report's own evaluation settings for these sizes are different again
(`--temperature 1.0 --top-p 1.0 --top-k 40`, 5.11).

With no --image/--video/--prompt the two default pictures are fetched (cached under
$AXELERA_IMAGE_CACHE) and `example()` runs; anything attached streams token by token
through `TextIteratorStreamer`.
"""

from __future__ import annotations

import argparse
import logging
from queue import Empty
from threading import Thread

from .helpers import device_name
from .pipeline import VIDEO_SUFFIXES, add_artifact_args
from .pipeline import open_media as pipeline_media

logger = logging.getLogger(__name__)

#: Wikimedia pictures, the same two the SmolVLM examples use.
DEFAULT_IMAGES = (
    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
    "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg",
)


def example(model, max_new_tokens: int = 128):
    """The Qwen3-VL model card's quickstart, run against the compiled artifact."""
    from transformers.image_utils import load_image

    processor = model.processor

    image1, image2 = model.fit_images([load_image(url) for url in DEFAULT_IMAGES])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": "Describe the images."},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text


def open_media(model, sources, *, frames: int | None = None, max_new_tokens: int = 128):
    """Paths or links -> (fitted images, (clip, metadata) or None), ready for the processor.

    `pipeline.open_media` does the picking-apart and the refusals; this fits the result to
    the compiled grid, the one step the processor cannot do for itself. Its two error
    kinds pass straight through: ValueError for a refusal, OSError for a source that will
    not open.
    """
    images, clip = pipeline_media(
        model.pipeline, sources, frames=frames, max_new_tokens=max_new_tokens
    )
    if clip is not None:
        return [], model.fit_video(*clip)
    return model.fit_images(images), None


def sampling_kwargs(args) -> dict:
    """Only the generate() parameters the user asked for.

    Everything left out stays whatever the repo's own `generation_config` says, as it
    would for a stock HF model.
    """
    knobs = {"max_new_tokens": args.max_new_tokens}
    if args.greedy:
        knobs["do_sample"] = False
    elif args.do_sample:
        knobs["do_sample"] = True
    for name in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty"):
        value = getattr(args, name)
        if value is not None:
            knobs[name] = value
    return knobs


def user_turn(question: str, *, images=None, video=None, first: bool = True) -> dict:
    """One user message: the question, with the media placeholders on the FIRST turn."""
    content = (
        []
        if not first
        else (
            [{"type": "image"} for _ in images or []] + [{"type": "video"}] * (video is not None)
        )
    )
    return {"role": "user", "content": content + [{"type": "text", "text": question}]}


def build_inputs(model, question: str, *, images=None, video=None, history=None):
    """The processor call for a turn: pictures OR a clip, plus the question.

    `history` is the turns before this one. The media placeholders ride on the FIRST user
    turn only, but the device holds one position-indexed KV cache, so the whole
    conversation is re-rendered and re-prefilled from position 0 every turn -- which is
    why the processor still needs the pictures, to expand those placeholders again.
    """
    processor = model.processor
    messages = list(history or [])
    messages.append(user_turn(question, images=images, video=video, first=not messages))
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    extra = {}
    if video is not None:
        clip, metadata = video
        # The frames are already sampled and sized, so the processor only needs to know
        # when each was taken (it writes <X.X seconds> labels into the prompt). do_resize
        # off as in pipeline.render: the video processor would re-shrink them off the grid.
        extra = {
            "videos": [clip],
            "video_metadata": [metadata],
            "do_sample_frames": False,
            "do_resize": False,
        }
    return processor(text=prompt, images=images or None, return_tensors="pt", **extra)


#: How long a stream waits for the next token before assuming generation is wedged. Well
#: clear of a slow first token (vision + a full prefill) but not an infinite hang.
_STREAM_TIMEOUT_S = 300.0


def stream_answer(model, inputs, **generate_kwargs):
    """Yield the answer chunk by chunk: HF's documented streaming pattern.

    `generate` runs in a thread and `TextIteratorStreamer` hands back text as tokens
    land; `model.stats` is live throughout, so a caller can show tok/s while it types.

    A `generate` that raises never sends the streamer's end signal, so an unguarded
    `yield from streamer` would wait for ever. The thread's exception is captured and
    re-raised here, and the read has a timeout: however generation dies, the caller hears.
    """
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        model.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=_STREAM_TIMEOUT_S
    )
    failure = []

    def run():
        try:
            model.generate(**inputs, streamer=streamer, **generate_kwargs)
        except BaseException as exc:  # re-raised in the caller below
            failure.append(exc)

    thread = Thread(target=run)
    thread.start()
    try:
        try:
            yield from streamer
        except Empty:
            # Nothing arrived within the timeout. A dead generate is the likely cause and
            # its exception is the useful one, so let the finally clause raise that.
            if not failure and thread.is_alive():
                raise TimeoutError(
                    f"no token in {_STREAM_TIMEOUT_S:.0f}s and generate is still running"
                ) from None
    finally:
        thread.join()
        if failure:
            raise failure[0]


def ask(
    model,
    question: str,
    *,
    images=None,
    video=None,
    stream=True,
    history=None,
    **generate_kwargs,
):
    """One user turn through the HF API, with the timings printed as they are known.

    `images` are already fitted PIL images, `video` the (clip, metadata) pair from
    `model.fit_video`, `history` the earlier turns. Returns the answer text.
    """
    inputs = build_inputs(model, question, images=images, video=video, history=history)

    if not stream:
        ids = model.generate(**inputs, **generate_kwargs)
        answer = model.processor.batch_decode(
            ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        print(answer)
        print_performance(model)
        return answer

    chunks = []
    for chunk in stream_answer(model, inputs, **generate_kwargs):
        if not chunks:  # the vision + prefill numbers are known once the first token lands
            stats = model.stats
            plural = "" if stats.vision_dispatches == 1 else "es"
            print(
                f"[vision {stats.vision_s * 1e3:.0f} ms ({stats.vision_dispatches} "
                f"dispatch{plural}) + prefill {stats.prefill_s * 1e3:.0f} ms "
                f"({stats.prompt_tokens} tokens) = TTFT {stats.ttft_s * 1e3:.0f} ms]"
            )
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print()
    print_performance(model)
    return "".join(chunks)


_BANNER = (
    "Qwen3-VL -- /image <path…> attaches pictures, /video <path> [frames] a clip, "
    "/reset clears the history, /exit quits"
)


def _attach(model, line: str, frames: int | None, max_new_tokens: int):
    """Handle /image and /video: (images, video, note). Raises ValueError on bad usage.

    Attaching REPLACES whatever was attached before (pictures and a clip share one vision
    budget), and the caller clears the history with it.
    """
    command, *sources = line.split()
    if command == "/image":
        if not sources:
            raise ValueError("usage: /image <path or url> [more…]")
        images, _ = open_media(model, sources, max_new_tokens=max_new_tokens)
        return images, None, f"[image] {len(images)} attached"
    if not 1 <= len(sources) <= 2:
        raise ValueError("usage: /video <path or url> [frames]")
    count = int(sources[1]) if len(sources) == 2 else frames
    _, video = open_media(model, sources[:1], frames=count, max_new_tokens=max_new_tokens)
    if video is None:
        # `open_media` tells a clip from a picture by SUFFIX, so /video on a .jpg comes back
        # as a picture and there is no clip to attach. Said here as a refusal the REPL
        # prints, rather than counting the frames of a None and taking the loop down with a
        # TypeError.
        raise ValueError(
            f"{sources[0]} is not a video (recognised: {', '.join(VIDEO_SUFFIXES)}); "
            "attach a picture with /image"
        )
    return [], video, f"[video] {len(video[0])} frames attached"


def _questions(first: str | None):
    """`--prompt` if there was one, then whatever is typed, until EOF/Ctrl-D."""
    if first:
        print(f"\n> {first}")
        yield first
    while True:
        try:
            yield input("\n> ").strip()
        except EOFError:
            print()
            return


def repl(
    model,
    images,
    video,
    *,
    frames: int | None,
    max_new_tokens: int,
    stream=True,
    first_question: str | None = None,
    **generate_kwargs,
) -> int:
    """Multi-turn chat over whatever is attached, until /exit or Ctrl-D.

    Every turn is a plain `generate()` call with the conversation so far, so the sampler
    flags, the streamer and the stopping criteria are the single-shot path's. The vision
    tower re-encodes each turn, which buys a stateless loop: nothing to invalidate when
    /image or /reset lands.
    """
    history: list[dict] = []
    print(_BANNER)
    for line in _questions(first_question):
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/reset":
            history = []
            print("[history cleared]")
            continue
        if line.startswith(("/image", "/video")):
            try:
                images, video, note = _attach(model, line, frames, max_new_tokens)
            # OSError too: a missing file, an unreachable link and a file that is not an
            # image all arrive as subclasses of it, and any of them ended the session.
            except (ValueError, OSError) as refusal:
                print(refusal)
                continue
            history = []  # the old conversation was about the old media
            print(note)
            continue
        turn = user_turn(line, images=images, video=video, first=not history)
        answer = ask(
            model,
            line,
            images=images,
            video=video,
            stream=stream,
            history=history,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        history += [turn, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    return 0


def print_performance(model) -> None:
    """Everything the UI's Performance panel shows, as text.

    Headline, where the wall clock went, the per-stage breakdown, and -- after more than
    one run -- the session averages. Same numbers in the same order as the UI panel.
    """
    from .hf_model import session_rows

    stats = model.stats
    print(f"\n[{stats.summary()}]")
    print("\nperformance")
    print("  " + stats.time_bar().replace("\n", "\n  "))
    print(stats.report())
    if len(model.history) > 1:
        print("\n  session averages (every run so far)")
        width = max(len(label) for label, _ in session_rows(model.history))
        for label, value in session_rows(model.history):
            print(f"    {label:<{width}} {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen3vl run", description=__doc__)
    add_artifact_args(parser)
    parser.add_argument(
        "--image",
        nargs="+",
        default=None,
        metavar="PATH_OR_URL",
        help=f"picture(s); default: {DEFAULT_IMAGES[0]} and {DEFAULT_IMAGES[1]}",
    )
    parser.add_argument("--video", default=None, metavar="PATH_OR_URL")
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="frames to sample from the video, equally distributed over the whole clip",
    )
    parser.add_argument("--prompt", default=None, help="ask this instead of the built-in tour")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    # Unset sampler flags are not passed to generate() at all: the repo's own
    # generation_config decides, exactly as it would for a stock HF model.
    parser.add_argument("--do-sample", action="store_true", help="sample (repo default here)")
    parser.add_argument("--greedy", action="store_true", help="do_sample=False, reproducible")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="keep asking: a multi-turn REPL over whatever is attached "
        "(/image, /video, /reset, /exit)",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    from .hf_model import AxQwen3VLForConditionalGeneration

    model = AxQwen3VLForConditionalGeneration.from_artifact(
        args.artifact_dir,
        args.repo_id,
        generation=args.generation,
    )
    knobs = sampling_kwargs(args)
    print(
        f"[model] {model.config.name_or_path or ''} on "
        f"{device_name(model.pipeline.generation)}: {model.image_geometry}px vision, "
        f"{model.context_len}-token context, {model.num_cores} core(s)"
    )

    sources = list(args.image or []) + ([args.video] if args.video else [])
    if not sources and not args.prompt and not args.interactive:  # the copy-paste example
        print("\n> [2 images] Describe the images.")
        example(model, args.max_new_tokens)
        print_performance(model)
        return 0

    # A one-shot run with nothing attached still shows the tour's two pictures; the REPL
    # starts empty instead, because /image is how it attaches anything.
    default = [] if args.interactive else DEFAULT_IMAGES
    try:
        images, video = open_media(
            model,
            sources or default,
            frames=args.frames,
            max_new_tokens=args.max_new_tokens,
        )
    except (ValueError, OSError) as refusal:  # see the REPL's note: a bad path is not a crash
        raise SystemExit(refusal) from None

    if args.interactive:  # --prompt, if given, is simply the first question asked
        return repl(
            model,
            images,
            video,
            frames=args.frames,
            stream=args.stream,
            first_question=args.prompt,
            **knobs,
        )

    question = args.prompt or ("What happens in this video?" if video else "Describe the images.")
    what = f"video, {len(video[0])} frames" if video else f"{len(images)} image(s)"
    # Name the pictures, not just how many: an answer about the built-in tour reads exactly
    # like an answer about a picture the caller attached, and the count alone does not tell
    # the two apart. Asking a question with nothing attached is the case that misleads.
    if not sources:
        what += ", the built-in example pictures -- attach your own with --image"
    print(f"\n> [{what}] {question}")
    ask(model, question, images=images, video=video, stream=args.stream, **knobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
