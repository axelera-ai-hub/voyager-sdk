# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Run a compiled LFM2.5-VL artifact through the HuggingFace image-text-to-text API.

The inference half of ``pipeline.build_artifacts`` (which the family entry drives through
``shared/vl_entry.py``), and a worked example of the API a customer meets: the model class is Axelera's, everything above it -- the processor, the chat
template, the sampling parameters, the streamer -- is stock transformers.

    python -m model_axscript.llm.models.lfm2vl.lfm2vl_run --artifact-dir build/lfm2.5-vl-450m
    python -m model_axscript.llm.models.lfm2vl.lfm2vl_run --artifact-dir build/lfm2.5-vl-450m \\
        --image photo.jpg --prompt "What is in this picture?"
    python -m model_axscript.llm.models.lfm2vl.lfm2vl_run --artifact-dir build/lfm2.5-vl-450m \\
        --image a.jpg https://example.com/b.jpg --prompt "Compare them." --greedy
    python -m model_axscript.llm.models.lfm2vl.lfm2vl_run --artifact-dir build/lfm2.5-vl-450m \\
        --image poster.jpg --tiles          # split a high-resolution picture
    python -m model_axscript.llm.models.lfm2vl.lfm2vl_run --artifact-dir build/lfm2.5-vl-450m \\
        --video clip.webm --frames 5 --prompt "What happens?"

    python -m model_axscript.llm.models.lfm2vl.lfm2vl_run --artifact-dir build/lfm2.5-vl-450m --interactive
        /image <path|url> ...   attach pictures      /tiles on|off   splitting
        /video <path|url> [N]   attach a clip        /reset          start a new turn
        /exit

A CLIP is multi-image input, not a video path: LFM2.5-VL has no video processor and no
frame scaffold in its chat template, so --video samples N frames, encodes each as a whole
picture (one dispatch each, never tiled) and labels it with its timestamp as ordinary
prompt text. That is how Liquid presents video for this family, and multi-frame reasoning
is a headline improvement in the 3B. What it is NOT: video-native temporal encoding, an
audio track, or long-video handling. `--frames` defaults to EVERY frame in the clip and
the context caps how many fit (7 x 256 tokens of 2048 on the 450M), so a clip longer than
a few frames is refused until you say how to sample it -- deliberately, because silently
keeping a handful would drop most of a clip without saying so.

Anywhere a picture is accepted it may be a local path or an http(s) URL; fetched media is
cached in ``$AXELERA_IMAGE_CACHE`` (default ``~/.cache/axelera_images``).

One host-side setting is worth knowing and this code cannot set it for you:
``axtrace --slog-level err`` silences the device log, which is worth ~15x on every stage.

The sampling defaults are the checkpoint's own, saved into the artifact at build time.
The model card recommends ``temperature=0.1, min_p=0.15, repetition_penalty=1.05`` for
this size; ``--greedy`` turns sampling off entirely.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)

#: The picture the quickstart uses when none is given.
_EXAMPLE_IMAGE = (
    "https://cdn.britannica.com/61/93061-050-99147DCE/" "Statue-of-Liberty-Island-New-York-Bay.jpg"
)
_EXAMPLE_PROMPT = "What is in this image?"
#: The question a clip gets when the caller names none.
_VIDEO_PROMPT = "What happens in this video?"


def frame_budget(model, max_new_tokens: int) -> int:
    """How many frames fit this artifact's context, alongside a question and its answer.

    Derived, not tabulated: image tokens per dispatch and the context are both compiled
    properties that differ per checkpoint and per build, so a constant here would be right
    for one artifact and quietly wrong for the next. Untiled is assumed because that is how
    a frame is always encoded (see :func:`video_messages`).
    """
    from ...shared.video import frames_that_fit

    return frames_that_fit(
        model.pipeline.config.vision.num_merged, model.context_len, max_new_tokens
    )


def video_messages(history: list[dict], labels, prompt: str) -> list[dict]:
    """A clip as a multi-image turn: every frame introduced by its own timestamp.

    LFM2.5-VL has no video path -- no video processor, no frame scaffold in its chat
    template -- and multi-image input is how Liquid presents video for this family. So a
    clip IS a sequence of pictures here, and the only thing this adds is the label before
    each one.

    The labels are load-bearing rather than decoration: a bare run of pictures carries no
    order and no spacing, so "what happens next" has nothing to refer to and the model is
    free to read a sequence as one scene. They are ordinary prompt text -- the template has
    no timestamp token -- which is also why they cost tokens (see `_TEXT_RESERVE`).

    What this does NOT give: video-native temporal encoding, an audio track, or long
    clips. The caller owns the sampling, and the context caps the frame count, so this is
    for a handful of moments rather than minute-scale temporal reasoning.
    """
    content: list[dict] = []
    for label in labels:
        stamp = f"Frame at {label}" if ":" in label else f"Frame {label}"
        content.append({"type": "text", "text": stamp})
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return [*history, {"role": "user", "content": content}]


def load_clip(source: str, num_frames: int | None):
    """(PIL frames, labels) for a clip -- the family's loader, then this model's labels.

    ``num_frames`` of None means EVERY frame in the clip, which is the default: sampling is
    the caller's to choose, and silently picking a handful would drop most of a clip without
    saying so. What does not fit the context is refused by name (see `frame_budget`), never
    quietly thinned.

    The frames arrive from ``load_video`` as one numpy array; they are handed on as PIL
    images because that is what the picture path takes, and a frame is a picture as far as
    this model is concerned.
    """
    from PIL import Image

    from ...shared.video import frame_labels, load_video_frames

    frames, metadata = load_video_frames(source, num_frames)
    images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    return images, frame_labels(metadata, len(images))


#: Seconds the streamer waits for a token before concluding generation is not progressing.
#: Generous: it has to cover a full prefill on the longest supported context, which is
#: ~0.8 s here but is a compile-time property, so this is a wedge detector and not a
#: latency budget.
_STREAM_TIMEOUT_S = 300.0


def _parser() -> argparse.ArgumentParser:
    from .pipeline import add_artifact_args

    parser = argparse.ArgumentParser(prog="lfm2vl run", description=__doc__.splitlines()[0])
    add_artifact_args(parser)
    parser.add_argument(
        "--image",
        nargs="+",
        default=None,
        metavar="PATH_OR_URL",
        help="Picture(s) to ask about. Omit for the quickstart example, or for a "
        "text-only prompt.",
    )
    parser.add_argument(
        "--video",
        default=None,
        metavar="PATH_OR_URL",
        help="A clip to ask about. LFM2.5-VL has no video path: its frames are encoded as "
        "whole pictures and labelled with their timestamps, so this answers questions about "
        "a set of moments, not fine temporal reasoning. Not with --image.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        metavar="N",
        help="Sample N frames from --video, equally distributed over the whole clip. "
        "Default: EVERY frame. Each frame costs one dispatch and a whole picture's prompt "
        "tokens, so the artifact's context caps this -- a clip that does not fit is refused "
        "with the arithmetic rather than thinned behind your back.",
    )
    parser.add_argument("--prompt", default=None, help="The question. Default: the quickstart's.")
    parser.add_argument(
        "--tiles",
        dest="tiles",
        action="store_true",
        default=None,
        help="Split a high-resolution picture into tiles plus a thumbnail -- more detail, "
        "several dispatches, many more prompt tokens. Default: the artifact's setting.",
    )
    parser.add_argument(
        "--no-tiles",
        dest="tiles",
        action="store_false",
        help="Encode each picture whole, in one dispatch.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--greedy", action="store_true", help="Turn sampling off.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--interactive", action="store_true", help="A multi-turn REPL instead of one answer."
    )
    parser.add_argument("--no-stats", action="store_true", help="Skip the timing report.")
    return parser


def sampling_kwargs(args) -> dict:
    """The generate() sampling arguments the flags ask for -- transformers' own names."""
    if args.greedy:
        return {"do_sample": False}
    named = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
    }
    chosen = {name: value for name, value in named.items() if value is not None}
    # Any explicit knob means sampling, whatever the artifact's generation_config says.
    return {**chosen, "do_sample": True} if chosen else {}


def load(args):
    """The artifact, as a transformers model."""
    from . import hf_model

    if hf_model.AxLfm2VlForConditionalGeneration is None:
        raise SystemExit("transformers is required to run an LFM2-VL artifact")
    model = hf_model.AxLfm2VlForConditionalGeneration.from_artifact(
        args.artifact_dir, args.repo_id, generation=args.generation
    )
    print(
        f"loaded {model.config.name_or_path} from {args.artifact_dir} in "
        f"{model.load_s:.1f}s -- {model.image_geometry} geometry, "
        f"{model.pipeline.config.vision.num_merged} image tokens per dispatch, "
        f"{model.num_cores} core(s), {model.context_len}-token context"
    )
    return model


def build_messages(history: list[dict], images, prompt: str) -> list[dict]:
    """The chat-template form: the pictures ride on the turn that introduces them."""
    content = [{"type": "image"} for _ in images or []]
    content.append({"type": "text", "text": prompt})
    return [*history, {"role": "user", "content": content}]


def answer(model, messages, images, *, tiles=None, max_new_tokens=128, stream=True, **sampling):
    """One generation, streamed to stdout; returns (text, stats)."""
    from transformers import TextIteratorStreamer

    inputs = model.process(messages, images=images, split=tiles)
    prompt_len = int(inputs["input_ids"].shape[-1])
    if prompt_len + max_new_tokens > model.context_len:
        raise SystemExit(
            f"{prompt_len} prompt tokens + {max_new_tokens} answer exceed the artifact's "
            f"{model.context_len}-token context. Drop --tiles (or lower --frames, for a "
            "clip), shorten the question, lower --max-new-tokens, or rebuild with a larger "
            "--compile-seq-len."
        )
    if not stream:
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        text = model.processor.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)[0]
        return text, model.stats

    # Token-by-token output is transformers' documented pattern -- the streamer in one
    # thread, generate() in another, this thread printing what arrives -- with the one
    # guard that pattern needs. A `generate` that raises never reaches the streamer's end
    # signal, so an unguarded read waits FOR EVER: every refusal the model makes before
    # its first token (a batch > 1, beam search, a tile that does not match the compiled
    # patch count, a wedged device) would hang the default --run path with a traceback on
    # stderr and nothing to catch. So the thread's exception is captured, the streamer is
    # ended on the way out so the reader wakes at once rather than waiting out the
    # timeout, and it is re-raised here. The timeout covers a generation that wedges
    # without raising at all.
    import threading
    from queue import Empty

    streamer = TextIteratorStreamer(
        model.processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=_STREAM_TIMEOUT_S,
    )
    failure: list[BaseException] = []

    def run():
        try:
            model.generate(**inputs, max_new_tokens=max_new_tokens, streamer=streamer, **sampling)
        except BaseException as exc:  # re-raised in the caller below
            failure.append(exc)
            streamer.end()

    thread = threading.Thread(target=run)
    thread.start()
    pieces = []
    try:
        for piece in streamer:
            pieces.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
    except Empty:
        # Nothing arrived within the timeout. A dead generate is the likely cause and its
        # exception is the useful one, so let the finally clause raise that instead.
        if not failure:
            raise TimeoutError(
                f"no token in {_STREAM_TIMEOUT_S:.0f}s -- generation is not progressing"
            ) from None
    finally:
        thread.join()
        if failure:
            raise failure[0]
    sys.stdout.write("\n")
    return "".join(pieces), model.stats


def report(stats) -> None:
    """The timing block after an answer."""
    print()
    print(stats.time_bar())
    print(stats.report())


def interactive(model, args) -> int:
    """A multi-turn REPL: attach pictures, ask, keep the conversation."""
    from .hf_model import session_rows

    history: list[dict] = []
    images: list = []
    labels: list | None = None
    tiles = args.tiles
    print(
        "Type a question. /image <path|url> ... attaches pictures, /video <path|url> [N] a "
        "clip (N frames, default all), /tiles on|off toggles splitting, /reset starts over, "
        "/exit quits."
    )
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/reset":
            history, images, labels = [], [], None
            print("(conversation cleared)")
            continue
        if line.startswith("/tiles"):
            parts = line.split()
            tiles = None if len(parts) < 2 else parts[1].lower() in ("on", "true", "1", "yes")
            if tiles and labels is not None:
                # main() refuses --tiles with --video outright; say so rather than accept a
                # setting that cannot apply while a clip is attached.
                print("(tiling: on for pictures -- frames are always encoded whole)")
            else:
                print(f"(tiling: {'artifact default' if tiles is None else tiles})")
            continue
        if line.startswith("/image"):
            # A bare `/image` used to clear the attachment silently and report "0
            # picture(s)", which reads like it worked. It is a usage error, and a
            # ValueError is what this loop recovers from.
            attached = line.split()[1:]
            if not attached:
                print("\nusage: /image <path or url> [more...]   (/reset clears them)")
                continue
            images, labels = attached, None
            print(f"({len(images)} picture(s) attached to the next question)")
            continue
        if line.startswith("/video"):
            # Same slot as /image -- a clip and pictures are alternative fillings of one
            # turn -- so attaching either REPLACES the other rather than appending to it.
            parts = line.split()[1:]
            if not parts:
                print("\nusage: /video <path or url> [frames]   (/reset clears it)")
                continue
            from ...shared.video import clip_frame_count

            asked = None  # no count given: every frame, as --video does
            if len(parts) > 1:
                try:
                    asked = int(parts[-1])
                except ValueError:
                    print(f"\n{parts[-1]!r} is not a frame count")
                    continue
            fits = frame_budget(model, args.max_new_tokens)
            try:
                holds = asked if asked is not None else clip_frame_count(parts[0])
            except SystemExit as unusable:
                # The probe is best-effort for decode failures, but `_local_video_path`
                # reports a bad path with SystemExit, which its `except Exception` cannot
                # catch. A clean line for the CLI; the end of the session for this loop.
                print(f"\n{unusable}")
                continue
            if holds is not None and holds > fits:
                print(
                    f"\nat most {fits} frame(s) fit this artifact's context; {holds} asked. "
                    f"Try /video {parts[0]} {fits}"
                )
                continue
            try:
                images, labels = load_clip(parts[0], asked)
            except ValueError as unusable:
                print(f"\n{unusable}")
                continue
            print(f"({len(images)} frame(s) attached: {', '.join(labels)})")
            continue
        messages = (
            video_messages(history, labels, line)
            if labels is not None
            else build_messages(history, images, line)
        )
        try:
            text, stats = answer(
                model,
                messages,
                images,
                # A frame is never tiled (see the --tiles refusal in main()). Derived from
                # what is attached rather than by overwriting `tiles`, which would leave a
                # later /image untiled for the rest of the session.
                tiles=False if labels is not None else tiles,
                max_new_tokens=args.max_new_tokens,
                **sampling_kwargs(args),
            )
        except (ValueError, SystemExit, NotImplementedError) as refusal:
            print(f"\n{refusal}")
            continue
        if not args.no_stats:
            report(stats)
        history = [*messages, {"role": "assistant", "content": [{"type": "text", "text": text}]}]
        # A follow-up must not re-announce the media: the placeholders are already in the
        # history, and re-attaching would ask the tower to encode the same frames again.
        images, labels = [], None
    rows = session_rows(model.history)
    if rows:
        width = max(len(name) for name, _ in rows)
        print("\nsession")
        for name, value in rows:
            print(f"  {name:<{width}} {value}")
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    # Refused before the artifact is opened: both are wrong the moment they are typed, and
    # a device load first would report them a couple of seconds late for no gain.
    if args.video and args.image:
        raise SystemExit("pass --image or --video, not both -- a clip IS the pictures here")
    if args.video and args.tiles:
        raise SystemExit(
            "--tiles is not available for --video: a tiled frame is several dispatches "
            "(min_tiles..max_tiles each), so a handful of frames would exceed any context. "
            "Frames are always encoded whole."
        )
    if args.frames is not None and not args.video:
        raise SystemExit("--frames only means something with --video")

    model = load(args)
    if args.interactive:
        return interactive(model, args)

    tiles = args.tiles
    labels = None
    if args.video:
        # The frame count is checked against the artifact BEFORE the clip is decoded: eight
        # frames of a 4K clip is real work to throw away, and the context it would not fit
        # in is known from the build alone.
        from ...shared.video import clip_frame_count

        # No --frames means every frame the clip holds. Resolved from the container's header
        # BEFORE it is decoded, so the common answer -- "that is more than fits" -- costs a
        # header read rather than a full decode of a clip that was never going to be used.
        asked = args.frames or clip_frame_count(args.video)
        fits = frame_budget(model, args.max_new_tokens)
        if asked is not None and asked > fits:
            whole = " (the whole clip)" if args.frames is None else ""
            raise SystemExit(
                f"{asked} frames{whole} x {model.pipeline.config.vision.num_merged} image "
                f"tokens do not fit the artifact's {model.context_len}-token context "
                f"alongside a {args.max_new_tokens}-token answer: at most {fits} frame(s) "
                f"here. Pass --frames N (N <= {fits}) to sample the clip, lower "
                "--max-new-tokens, or rebuild at a longer context."
            )
        try:
            images, labels = load_clip(args.video, asked)
        except ValueError as unusable:  # the loader names the container and the reason
            raise SystemExit(f"{unusable}") from None
        # A container that would not declare its length is checked after the decode, which
        # is the only place its real frame count exists.
        if len(images) > fits:
            raise SystemExit(
                f"{args.video} decoded to {len(images)} frames, and at most {fits} fit this "
                f"artifact's {model.context_len}-token context. Pass --frames N (N <= {fits})."
            )
        prompt = args.prompt or _VIDEO_PROMPT
        messages = video_messages([], labels, prompt)
        attached = f"video, {len(images)} frame(s) at {', '.join(labels)}"
        # Never tiled, whatever the artifact's default: see the --tiles refusal above.
        tiles = False
    else:
        images = args.image
        prompt = args.prompt
        if images is None and prompt is None:  # the quickstart
            images, prompt = [_EXAMPLE_IMAGE], _EXAMPLE_PROMPT
        prompt = prompt or _EXAMPLE_PROMPT
        messages = build_messages([], images, prompt)
        # Said out loud: a question about "this image" with no --image is a text-only prompt,
        # and the answer will be about nothing in particular.
        attached = f"{len(images)} picture(s)" if images else "no picture -- text-only"
    print(f"\n> {prompt}  [{attached}]")
    started = time.perf_counter()
    text, stats = answer(
        model,
        messages,
        images,
        tiles=tiles,
        max_new_tokens=args.max_new_tokens,
        **sampling_kwargs(args),
    )
    assert text is not None
    if not args.no_stats:
        report(stats)
        print(f"\n{stats.summary()} (wall {time.perf_counter() - started:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
