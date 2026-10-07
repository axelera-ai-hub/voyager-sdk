# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM2 on Metis, driven exactly like a HuggingFace model.

Every feature a VLM customer expects from ``transformers`` -- chat templates, one or
several images, video, sampling parameters, token-by-token streaming -- against a compiled
Axelera artifact. The only line that changes is the model class.

This is the SmolVLM2 model card's own quickstart, with ONE line different -- the import:

    ```python
    >>> from transformers import AutoModelForImageTextToText, AutoProcessor
    >>> from model_axscript.llm.models.smolvlm2.hf_model import register_auto
    >>> register_auto()   # registers the model type
    >>>
    >>> # the card's `AutoModelForImageTextToText.from_pretrained(repo, device_map="auto")`
    >>> # -- here the weights are the compiled artifact, and the device is Metis
    >>> model = AutoModelForImageTextToText.from_pretrained("build/smolvlm2-256m")
    >>> processor = AutoProcessor.from_pretrained("build/smolvlm2-256m")  # == model.processor
    >>>
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             {"type": "image", "url": "https://.../bee.jpg"},
    ...             {"type": "text", "text": "Can you describe this image?"},
    ...         ],
    ...     },
    ... ]
    >>>
    >>> inputs = processor.apply_chat_template(
    ...     messages, add_generation_prompt=True, tokenize=True,
    ...     return_dict=True, return_tensors="pt",
    ... )
    >>>
    >>> generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    >>> print(generated_texts[0])
    ```

``example()`` below is exactly that, runnable: the front door with no other flag runs it.

    python -m model_axscript.llm run smolvlm2-256m-video-instruct
    python -m model_axscript.llm run smolvlm2-256m-video-instruct "What is the total due?" \\
        --image invoice.png
    python -m model_axscript.llm run smolvlm2-256m-video-instruct --video clip.webm --frames 8
    python -m model_axscript.llm run smolvlm2-256m-video-instruct --image photo.jpg --interactive

Sampler flags left unset are not passed to ``generate()`` at all -- the artifact's own
``generation_config`` decides, which is what a stock HF model would do. That config is the
checkpoint's, saved into the artifact at build time and unmodified. ``--greedy`` pins it
down.

``-i`` / ``--interactive`` is the same ``generate()`` call in a loop: ``/image`` and
``/video`` attach media, ``/reset`` clears the history, ``/exit`` quits. The conversation
so far is re-rendered and re-prefilled each turn -- the device holds one position-indexed
KV cache, so replaying it is what keeps it honest.
"""

from __future__ import annotations

import argparse
import logging
from queue import Empty
from threading import Thread

logger = logging.getLogger(__name__)

#: The picture the SmolVLM2 model card's own example uses.
DEFAULT_IMAGE = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
)
_DEFAULT_QUESTION = "Can you describe this image?"


def example(model, max_new_tokens: int = 64):
    """The SmolVLM2 model card's quickstart, run against the compiled artifact.

    Line for line the snippet in the module docstring: the card's code, with the model
    class as the only difference. The ``url`` content part is the card's own spelling --
    the processor fetches and tiles it, so no Axelera call appears anywhere.
    """
    processor = model.processor

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": DEFAULT_IMAGE},
                {"type": "text", "text": _DEFAULT_QUESTION},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_texts[0])
    return generated_texts[0]


def sampling_kwargs(args) -> dict:
    """Only the generate() parameters the user actually asked for.

    Everything left out stays whatever the artifact's own ``generation_config`` says -- the
    same thing that would happen with a stock HF model. Passing our own defaults instead
    would quietly override the checkpoint's.
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


def user_turn(text: str, *, images=None, video=None, processor=None, first: bool = True) -> dict:
    """One user message. The media placeholders belong to the FIRST turn only.

    Every turn re-renders the whole conversation, so repeating them would ask the tower for
    two pictures' worth of placeholder rows against one encoded picture -- the splice
    refuses, mid-conversation.

    Video is a single ``<video>`` token that the processor expands into the scaffold
    SmolVLM2 was TRAINED on (an intro naming the frame count and duration, then a
    ``Frame from MM:SS`` label per frame). The chat template itself has no video branch, so
    a ``{"type": "video"}`` part would only produce a bag of unordered ``<image>`` entries.
    """
    if not first:
        return {"role": "user", "content": [{"type": "text", "text": text}]}
    if video is not None:
        return {
            "role": "user",
            "content": [{"type": "text", "text": processor.video_token + text}],
        }
    n = 0 if images is None else len(images)
    return {"role": "user", "content": [{"type": "image"}] * n + [{"type": "text", "text": text}]}


def build_inputs(model, question: str, *, images=None, video=None, history=None):
    """The processor call for a turn: pictures OR a clip, plus the question.

    ``history`` is the turns before this one. The media placeholders ride on the FIRST user
    turn only, but the whole conversation is re-rendered (and re-prefilled from position 0)
    every turn, so the processor still needs the media on a later turn -- to expand the
    placeholders the first turn left behind.

    Tiling is not a knob and does not need to be: a still goes through the IMAGE processor,
    which splits it into a global view plus high-res crops (up to 17 blocks); a video goes
    through the VIDEO processor, which never splits, so each frame is exactly one block --
    which is what HF's frame scaffold counts on.
    """
    processor = model.processor
    messages = list(history or [])
    messages.append(
        user_turn(question, images=images, video=video, processor=processor, first=not messages)
    )
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    if video is not None:
        frames, metadata = video
        # videos/video_metadata are nested per text sample: [[frames]].
        return processor(
            text=[text], videos=[[frames]], video_metadata=[[metadata]], return_tensors="pt"
        )
    return processor(text=text, images=list(images or []) or None, return_tensors="pt")


#: How long a stream waits for the next token before assuming generation is wedged. Well
#: clear of a slow first token (vision + a full prefill) but not an infinite hang. A backstop
#: only: a `generate` that RAISES ends the stream at once (see below).
_STREAM_TIMEOUT_S = 300.0


def stream_answer(model, inputs, **generate_kwargs):
    """Yield the answer chunk by chunk: HF's documented streaming pattern.

    ``generate`` runs in a thread and ``TextIteratorStreamer`` hands back text as tokens
    land.

    A ``generate`` that raises never reaches the streamer's end signal, so an unguarded
    ``yield from streamer`` waits FOR EVER -- every refusal the model makes before its first
    token (a prompt and an answer that cannot share the context, a splice the tile count
    disagrees with, a device that will not open) would hang the caller with its traceback on
    stderr and nothing to catch. So the thread's exception is captured, the streamer is
    ended on the way out so the reader wakes immediately, and it is re-raised here: the
    refusal reaches the REPL's own handler, which reports it and takes the next question.
    The read also has a timeout, for a generation that wedges without raising.
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
            # Wake the reader now rather than leaving it on the timeout: the exception is
            # already known, and 300s of silence is not a report.
            streamer.end()

    thread = Thread(target=run)
    thread.start()
    try:
        try:
            yield from streamer
        except Empty:
            # Nothing arrived within the timeout. A dead generate is the likely cause and its
            # exception is the useful one, so let the finally clause raise that instead.
            if not failure and thread.is_alive():
                raise TimeoutError(
                    f"no token in {_STREAM_TIMEOUT_S:.0f}s and generate is still running"
                ) from None
    finally:
        thread.join()
        if failure:
            raise failure[0]


def ask(model, question: str, *, images=None, video=None, stream=True, history=None, **kwargs):
    """One user turn through the HF API. Returns the answer text.

    ``images`` are PIL images, ``video`` the (frames, metadata) pair from
    ``load_video_frames``, ``history`` the earlier turns of a conversation (see
    build_inputs).
    """
    inputs = build_inputs(model, question, images=images, video=video, history=history)

    if not stream:
        ids = model.generate(**inputs, **kwargs)
        answer = model.processor.batch_decode(
            ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        print(answer)
        return answer

    chunks = []
    for chunk in stream_answer(model, inputs, **kwargs):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print()
    return "".join(chunks)


_BANNER = (
    "SmolVLM2 -- /image <path…> attaches pictures, /video <path> [frames] a clip, "
    "/reset clears the history, /exit quits"
)


def _attach(line: str, frames: int):
    """Handle /image and /video: (images, video, note). Raises ValueError on bad usage.

    Attaching REPLACES whatever was attached before -- pictures and a clip are alternative
    fillings of the same prompt -- and the caller clears the history with it, because the
    old conversation refers to the old media.

    EVERY refusal is a ``ValueError``, because that is the only thing the loop above can
    report and carry on from. The two kinds are told apart by SUFFIX and mismatches are
    named as such: handed the other kind, the loaders raise from inside a library
    (``UnidentifiedImageError`` for a clip given to PIL, an arithmetic error out of the
    frame sampler for a picture given to the video decoder) and take the session down over
    what is a typo.
    """
    from .processor import is_video, load_image, load_video_frames

    command, *sources = line.split()
    if command == "/image":
        if not sources:
            raise ValueError("usage: /image <path or url> [more…]")
        clips = [src for src in sources if is_video(src)]
        if clips:
            raise ValueError(
                f"{clips[0]} is a video, not a picture; attach a clip with "
                "/video <path> [frames]"
            )
        try:
            images = [load_image(src) for src in sources]
        except (Exception, SystemExit) as unreadable:
            raise ValueError(
                f"could not read a picture ({type(unreadable).__name__}: {unreadable})"
            ) from unreadable
        return images, None, f"[image] {len(images)} attached"
    if not 1 <= len(sources) <= 2:
        raise ValueError("usage: /video <path or url> [frames]")
    count = int(sources[1]) if len(sources) == 2 else frames
    video = load_video_frames(sources[0], count)  # refuses a picture with a ValueError
    return None, video, f"[video] {len(video[0])} frames attached"


def _questions(first: str | None):
    """``--prompt`` if there was one, then whatever is typed, until EOF/Ctrl-D."""
    if first:
        print(f"\n> {first}")
        yield first
    while True:
        try:
            yield input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return


def repl(model, images, video, *, frames: int, stream=True, first_question=None, **kwargs) -> int:
    """Multi-turn chat over whatever is attached, until /exit or Ctrl-D.

    Every turn is a plain ``generate()`` call with the conversation so far, so the sampler
    flags, the streamer and the stopping criteria are the same HF ones the single-shot path
    uses. The vision tower re-encodes each turn, which buys a stateless loop: nothing to
    invalidate when /image or /reset lands.
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
                images, video, note = _attach(line, frames)
            except ValueError as refusal:
                print(refusal)
                continue
            history = []  # the old conversation was about the old media
            print(note)
            continue
        turn = user_turn(
            line, images=images, video=video, processor=model.processor, first=not history
        )
        try:
            answer = ask(
                model, line, images=images, video=video, stream=stream, history=history, **kwargs
            )
        except ValueError as too_long:  # the context check in hf_model.forward
            print(f"\n{too_long}\n[turn dropped; try /reset or fewer images]")
            continue
        history += [turn, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smolvlm2 run", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifact-dir", required=True, help="Artifact directory written by a --build run."
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="the checkpoint the artifact was built from, normally read from its vlm.json. "
        "Only needed for an artifact built before the processor was saved into it, and it "
        "must be that same checkpoint -- another one tiles and chat-templates differently.",
    )
    parser.add_argument(
        "--image",
        nargs="+",
        default=None,
        metavar="PATH_OR_URL",
        help="picture(s) to ask about. Omit for a text-only prompt, or -- with no --prompt "
        f"either -- for the quickstart example ({DEFAULT_IMAGE})",
    )
    parser.add_argument("--video", default=None, metavar="PATH_OR_URL")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="sample N frames from the video, equally distributed over the whole clip; "
        "default EVERY frame. Each costs one untiled image block (64 prompt tokens on the "
        "256M/500M, 81 on the 2.2B) plus its scaffold label, so the artifact's context caps "
        "this -- a clip that does not fit is refused with the arithmetic rather than thinned "
        "behind your back.",
    )
    parser.add_argument("--prompt", default=None, help="ask this instead of the built-in example")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    # Unset sampler flags are not passed to generate() at all: the artifact's own
    # generation_config decides, exactly as it would for a stock HF model.
    parser.add_argument("--do-sample", action="store_true")
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
    parser.add_argument(
        "--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    if args.video is not None and args.image is not None:
        raise SystemExit("pass --image or --video, not both")

    from .hf_model import AxSmolVLMForConditionalGeneration
    from .processor import load_image, load_video_frames

    model = AxSmolVLMForConditionalGeneration.from_pretrained(
        args.artifact_dir, repo_id=args.repo_id
    )
    print(
        f"[model] {model.config.name_or_path or ''} on Metis: "
        f"{model.context_len}-token context"
    )
    knobs = sampling_kwargs(args)

    if not args.image and not args.video and not args.prompt and not args.interactive:
        print(f"\n> [1 image] {_DEFAULT_QUESTION}")  # the copy-paste example
        example(model, args.max_new_tokens)
        return 0

    # Unreadable or mismatched media is this CLI's own refusal -- a message and a status --
    # rather than a traceback out of PIL or the frame sampler. The loaders say what is wrong
    # (see processor.load_video_frames); only the shape of the report belongs here.
    if args.video:
        # No --frames means every frame the clip holds, and most clips hold more than a
        # 2048-token context can take. Resolved from the container's header BEFORE the
        # decode, so the common refusal costs a header read rather than a full decode.
        from ...shared.video import clip_frame_count, frames_that_fit

        per_frame = model.processor.image_seq_len
        fits = frames_that_fit(per_frame, model.context_len, args.max_new_tokens)
        asked = args.frames or clip_frame_count(args.video)
        if asked is not None and asked > fits:
            whole = " (the whole clip)" if args.frames is None else ""
            raise SystemExit(
                f"{asked} frames{whole} x {per_frame} tokens do not fit this artifact's "
                f"{model.context_len}-token context alongside a {args.max_new_tokens}-token "
                f"answer: at most {fits} frame(s) here. Pass --frames N (N <= {fits}), lower "
                "--max-new-tokens, or rebuild with a larger --seq-len."
            )

    try:
        video = load_video_frames(args.video, args.frames) if args.video else None
        # A question with no --image and no --video is a TEXT-ONLY prompt, answered by the
        # decoder alone (`pixel_values=None` all the way down). It is never quietly turned
        # into a question about the example picture: the caller asked about no picture, and
        # substituting one answers about an image they never passed -- which reads as a
        # hallucination rather than as a default. The zero-argument quickstart above is the
        # other case, where nothing was asked at all. The REPL likewise starts empty,
        # because /image is how it attaches anything.
        if video is not None or not args.image:
            images = None
        else:
            images = [load_image(src) for src in args.image]
    except ValueError as unusable:
        raise SystemExit(f"{unusable}") from None

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

    question = args.prompt or ("What happens in this video?" if video else _DEFAULT_QUESTION)
    # Said out loud, so a text-only answer is never read as one about a picture the run
    # silently dropped (an unreadable path is refused above, not skipped).
    if video:
        what = f"video, {len(video[0])} frames"
    else:
        what = f"{len(images)} image(s)" if images else "no picture -- text-only"
    print(f"\n> [{what}] {question}")
    ask(model, question, images=images, video=video, stream=args.stream, **knobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
