# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The family's clip loader: a path or URL -> sampled frames + the container's metadata.

All three VLMs here take video. Two load it through this module -- LFM2.5-VL directly,
SmolVLM2 via ``models.smolvlm2.processor``, which re-exports these -- to share
:func:`_pyav_sample_indices_fn`, the repair for headers that declare no frame count.
Qwen3-VL keeps its own copy of that, in ``models/qwen3vl/pipeline.load_video``, and takes
only :data:`VIDEO_SUFFIXES` / :func:`is_video` from here -- what counts as a clip is one
answer for the family even where the decoding is not.

What each does with the frames differs completely:

* SmolVLM2 stamps every frame with a real timestamp off this metadata, which its
  ``Frame from MM:SS`` scaffold counts on.
* LFM2.5-VL has no video path at all: a clip is a multi-image turn, each frame a whole
  picture ordered only by the :func:`frame_labels` text beside it in the prompt, with the
  context bounding how many fit (:func:`frames_that_fit`).
* Qwen3-VL is video-native -- temporal patches, not a stack of pictures.

Fetched clips are cached under ``$AXELERA_VIDEO_CACHE``, else ``$XDG_CACHE_HOME`` (or
``~/.cache``) ``/axelera_videos`` -- see :func:`video_cache_dir`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.parse
import urllib.request

# Sent on every media fetch, pictures and clips alike. Hosts like Wikimedia answer
# urllib's default "Python-urllib/3.x" with 403, and a decoder handed that HTML error body
# reports only "Invalid data found when processing input" -- naming neither the status nor
# the URL. Identify as a normal client instead.
_USER_AGENT = "Mozilla/5.0 (compatible; axelera-vlm/1.0)"


def video_cache_dir() -> Path:
    """Where downloaded clips are kept between runs.

    ``$AXELERA_VIDEO_CACHE`` names it outright; otherwise ``$XDG_CACHE_HOME`` (or
    ``~/.cache``) ``/axelera_videos``. Family-neutral rather than under one model's name,
    because every VLM that fetches through this module shares it and a clip cached by one is
    the same file for the next. The env-var name follows LFM's ``$AXELERA_IMAGE_CACHE``.
    """
    named = os.environ.get("AXELERA_VIDEO_CACHE")
    if named:
        path = Path(named).expanduser()
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        path = root / "axelera_videos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _local_video_path(src: str) -> str:
    """A local filesystem path for ``src`` (a path on this machine, or an http(s) URL).

    A local path is expanded (~ and $VARS) and used in place. A URL is downloaded once
    and cached, for three reasons: HF's ``load_video`` fetches URLs itself with the
    default agent and is 403'd by some hosts; the frame-count probe in
    ``_pyav_sample_indices_fn`` opens the container a second time, which is a seek on a
    file but a re-fetch over HTTP; and clips are large (53 MB for the Wikimedia lava
    sample), so re-downloading one per run is the slowest part of a video demo.

    Cached under ``video_cache_dir()`` keyed by a hash of the URL, so it survives across
    runs and is never re-fetched. Delete that directory to force a refresh -- nothing
    ages entries out, and nothing revalidates against the server, so a URL whose content
    changes under a stable address will keep serving the first copy.
    """
    if not src.startswith(("http://", "https://")):
        path = Path(os.path.expandvars(src)).expanduser()
        if not path.is_file():
            raise SystemExit(
                f"--video: no such file: {path}\n"
                f"Pass a path to a video on this machine, or an http(s) URL."
            )
        return str(path)

    suffix = Path(urllib.parse.urlparse(src).path).suffix or ".mp4"
    cached = video_cache_dir() / (hashlib.sha256(src.encode()).hexdigest()[:16] + suffix)
    if cached.is_file() and cached.stat().st_size:
        print(f"[video] already downloaded, reusing {cached}", flush=True)
        return str(cached)

    print(f"[video] downloading {src}\n            -> {cached}", flush=True)
    # Fetch to a sibling temp file and rename: an interrupted download then leaves no
    # cache entry, rather than a truncated one that every later run would happily reuse.
    fd, tmp = tempfile.mkstemp(dir=cached.parent, prefix=".partial-", suffix=suffix)
    try:
        req = urllib.request.Request(src, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req) as r, os.fdopen(fd, "wb") as out:  # noqa: S310
            shutil.copyfileobj(r, out)
        os.replace(tmp, cached)
    except BaseException:
        os.unlink(tmp)
        raise
    print(f"[video] downloaded {cached.stat().st_size / 1e6:.1f} MB -> {cached}", flush=True)
    return str(cached)


#: Container suffixes the video path recognises. A clip is told from a picture HERE, before
#: a decoder is asked to prove it: pyav opens a JPEG quite happily as a one-frame image2
#: stream and reports no frame count, at which point HF's sampler divides by zero -- an
#: arithmetic error from three libraries down, in place of "that is a picture".
VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".gif")

#: The suffixes that make ``/video photo.jpg`` a certain mistake rather than an unlucky
#: container, so it can be refused by name. ``.gif`` is deliberately absent -- it is a clip
#: as far as the decoders are concerned, and it is listed above.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".ppm")


def media_suffix(src) -> str:
    """The lowercased suffix of a path or a URL, any query string dropped."""
    return Path(str(src).split("?")[0]).suffix.lower()


def is_video(src) -> bool:
    """Whether ``src`` names a clip, by suffix -- the same question ``load_video_frames``
    asks before it hands anything to a decoder, so a caller can route on it."""
    return media_suffix(src) in VIDEO_SUFFIXES


def _pyav_sample_indices_fn(src: str, num_frames: int):
    """Repair pyav's missing frame count/duration, or None if the file is fine.

    VP8/webm (e.g. the Wikimedia clips) carries no frame count in its header, so
    pyav reports total_num_frames=0 AND duration=0.0, and HF's
    ``default_sample_indices_fn`` then divides by zero. The container DOES know its
    duration and frame rate, so probe those directly and hand the sampler a
    repaired ``VideoMetadata`` -- 8.008s x 29.97fps = 240 frames for the kitten clip.

    Take the rate from ``guessed_rate`` (ffmpeg's own estimate) in preference to
    ``average_rate``, which some webm containers report as the RATIO NUMERATOR rather
    than the rate: a 29.97fps (30000/1001) clip reports average_rate 30000, and seeding
    the repair with that overstates the frame count 1000x. The sampler then spreads its
    indices over frames that do not exist, only a handful decode, and the prompt (whose
    <image> count came from num_frames) no longer matches the pixel stack -- a shape
    mismatch in the embed splice, far from the cause. Verified on a 165.5s VP8 clip:
    guessed_rate -> 4959 frames == the true decode count, average_rate -> 4,964,040.

    Note HF stops passing ``num_frames`` through once a custom sample_indices_fn is
    supplied, so it is captured here; otherwise the sampler returns every frame.
    """
    try:
        import av
    except ImportError:  # pyav unavailable -> the backend itself will fail over
        return None
    try:
        with av.open(src) as container:
            stream = container.streams.video[0]
            rate_field = stream.guessed_rate or stream.base_rate or stream.average_rate
            rate = float(rate_field) if rate_field else 0.0
            duration = (container.duration / 1e6) if container.duration else 0.0
            total = stream.frames or (int(round(duration * rate)) if duration and rate else 0)
    except Exception:  # noqa: BLE001 - probing is best-effort; let the decoder report
        return None
    if not total:  # nothing to repair with
        return None

    from transformers.video_utils import default_sample_indices_fn

    def sample_indices_fn(metadata, **kwargs):
        if not metadata.total_num_frames:
            metadata.total_num_frames = total
        if not metadata.duration:
            metadata.duration = duration
        # fps comes from the decoder's own average_rate and is wrong on the same
        # containers (30000 for a 29.97fps clip). It is not used for sampling, but the
        # processor stamps each frame's timestamp as index/fps, so leaving it overstated
        # collapses every label to "Frame from 00:00" -- eight frames presented as one
        # instant, which is the whole temporal signal the video scaffold exists to give.
        if rate and metadata.fps and metadata.fps > rate * 1.5:
            metadata.fps = rate
        return default_sample_indices_fn(metadata, num_frames=num_frames)

    return sample_indices_fn


#: Prompt tokens held back for the question and the chat scaffold when working out how many
#: frames fit. Short, but not free.
TEXT_RESERVE = 128

#: Prompt tokens allowed PER FRAME on top of its image tokens: its timestamp label plus the
#: processor's own per-image markers. Deliberately generous. This whole budget exists to
#: refuse a clip before it is decoded, so erring high costs a frame that would have fitted,
#: while erring low promises a frame count that then dies at the exact gate in the run
#: driver -- after the decode, the vision dispatches and the banner. Measured at 7 frames on
#: the 450M: 1792 image tokens and 1923 prompt tokens, i.e. 19 a frame, against a flat
#: 128-token reserve that had promised the caller 7 would fit.
FRAME_OVERHEAD = 24


def frames_that_fit(
    tokens_per_frame: int, context_len: int, max_new_tokens: int, reserve: int = TEXT_RESERVE
) -> int:
    """How many frames fit a context alongside a question and its answer.

    One arithmetic for both models, fed each one's OWN numbers: a frame costs 64 prompt
    tokens on SmolVLM2's 256M and 256 on LFM2.5-VL's 450M, and both are compiled properties
    that differ per checkpoint and per build. A constant anywhere here would be right for
    one artifact and quietly wrong for the next.

    An ESTIMATE, and one that rounds against itself: the label lengths and the processor's
    markers are not known here. The authority is each driver's exact prompt-length check,
    which runs once the turn is rendered; this only has to be safe enough that the exact
    check never fires on a count this function blessed.
    """
    if not tokens_per_frame:
        return 0
    return max(0, (context_len - max_new_tokens - reserve) // (tokens_per_frame + FRAME_OVERHEAD))


def clip_frame_count(src: str) -> int | None:
    """How many frames ``src`` actually holds, or None if the container will not say.

    What "every frame" resolves to when a caller names no count. Probed with pyav directly
    rather than decoded, so it costs a header read instead of the whole clip -- which
    matters precisely because the answer is often "too many to fit", and that refusal
    should not have to decode a 4K clip first.

    Falls back to duration x rate for containers carrying no frame count (VP8/webm), using
    ``guessed_rate`` for the same reason :func:`_pyav_sample_indices_fn` does: some webm
    files report ``average_rate`` as the ratio numerator (30000 for 29.97fps), which
    overstates the count a thousandfold.
    """
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(_local_video_path(src)) as container:
            stream = container.streams.video[0]
            if stream.frames:
                return int(stream.frames)
            rate_field = stream.guessed_rate or stream.base_rate or stream.average_rate
            rate = float(rate_field) if rate_field else 0.0
            duration = (container.duration / 1e6) if container.duration else 0.0
            return int(round(duration * rate)) or None
    except Exception:  # noqa: BLE001 - probing is best-effort; the decoder reports for real
        return None


def load_video_frames(src: str, num_frames: int | None = None, backend: str = "pyav"):
    """Sample ``num_frames`` from a video on this machine, or at an http(s) URL
    -> (frames, metadata).

    Uses HF's own ``load_video`` so the returned ``VideoMetadata`` (fps, duration,
    frame indices) is exactly what the processor needs to stamp each frame with its
    real timestamp. Pre-sampled frames without metadata make SmolVLM assume fps=24
    and mislabel the timeline.

    ``backend`` defaults to HF's own ``pyav``. Containers that omit a frame count in
    their header (VP8/webm) are handled by _pyav_sample_indices_fn rather than by
    switching decoders -- decoders disagree on fps/frame counts, and those numbers
    end up in the prompt's frame timestamps, so silently swapping one in would change
    the timeline instead of fixing it.

    A URL is downloaded once and cached (see _local_video_path) rather than handed to
    the decoder to stream, so every path below this point is a plain file.

    EVERY failure leaves here as a ``ValueError`` naming the source, because the callers are
    an interactive loop and two CLIs: the REPL recovers from a refusal and dies on anything
    else, so a picture passed to ``/video`` has to arrive as one. Left to the libraries it
    is a ``ZeroDivisionError`` out of HF's frame sampler (pyav opens a JPEG as a one-frame
    image2 stream and reports no count), or a ``SystemExit`` from the path check -- neither
    of which a loop can report and continue from.
    """
    from transformers.video_utils import load_video

    if media_suffix(src) in IMAGE_SUFFIXES:
        raise ValueError(
            f"{src} is a picture, not a video (recognised video suffixes: "
            f"{', '.join(VIDEO_SUFFIXES)}); attach it as an image instead"
        )
    try:
        path = _local_video_path(src)
        kwargs = {}
        # num_frames=None means EVERY frame, which is what HF's sampler does when the count
        # is not passed through at all. The frame-count repair is skipped with it for the
        # same reason: it exists to spread N indices over a clip whose header lies about its
        # length, and there is nothing to spread when the answer is "all of them".
        if num_frames is not None:
            kwargs["num_frames"] = num_frames
            if backend == "pyav":
                fn = _pyav_sample_indices_fn(path, num_frames)
                if fn is not None:
                    kwargs["sample_indices_fn"] = fn
        frames, metadata = load_video(path, backend=backend, **kwargs)
    except (Exception, SystemExit) as undecodable:
        # SystemExit included: _local_video_path raises it for a missing file, which is a
        # clean message for a CLI and an unrecoverable one for the REPL.
        raise ValueError(
            f"{src} could not be read as a video ({type(undecodable).__name__}: "
            f"{undecodable}); recognised video suffixes: {', '.join(VIDEO_SUFFIXES)}"
        ) from undecodable
    # fps/duration are the decoder's own and are None on containers that declare neither, so
    # the banner formats defensively: a missing number must not turn a decoded clip into a
    # TypeError in an f-string.
    print(
        f"[video] decoder={backend}  fps={_number(metadata.fps)}  "
        f"duration={_number(metadata.duration)}s"
    )
    return frames, metadata


def _number(value, unknown: str = "?") -> str:
    """A metadata number for the banner: two decimals, or ``?`` if the container had none."""
    return unknown if value is None else f"{float(value):.2f}"


def frame_labels(metadata, count: int) -> list[str]:
    """A short label per sampled frame: ``"00:03"``, or ``"#12"`` with no frame rate.

    For the model with no frame scaffold of its own (LFM2.5-VL), where these go into the
    prompt as plain text. A clip presented as a bag of pictures has no order and no
    spacing -- "what happens next" has nothing to refer to -- so the label carries the one
    thing the pixels do not.

    Timestamps come from the container's OWN sampled indices and rate, not from the
    frame's position in the batch, so eight frames spread over a three-minute clip are
    labelled 00:00, 00:26, 00:52 ... and not 0..7. A container that declares no rate gets
    frame NUMBERS instead: an invented 00:00 on every frame is worse than admitting the
    clip did not say, because it reads as "all at once".
    """
    fps = float(getattr(metadata, "fps", 0) or 0)
    # Spelled out rather than `... or []`: the decoders hand back the sampled indices as a
    # numpy ARRAY, and `array or default` raises "truth value ... is ambiguous" instead of
    # falling back -- a clip that decoded perfectly, refused while labelling it.
    indices = getattr(metadata, "frames_indices", None)
    indices = [] if indices is None else list(indices)
    if not fps:
        return [f"#{indices[i] if i < len(indices) else i}" for i in range(count)]

    times = [(indices[i] if i < len(indices) else i) / fps for i in range(count)]

    def render(decimals: int) -> list[str]:
        width = 2 + (decimals and decimals + 1)
        return [
            f"{int(minutes):02d}:{seconds:0{width}.{decimals}f}"
            for minutes, seconds in (divmod(moment, 60) for moment in times)
        ]

    # Whole seconds unless they would COLLIDE, and then tenths. The tenth is not cosmetic:
    # 5 frames of a 4-second clip are 0.8s apart, so whole seconds label two of them
    # identically however they are rounded (0, 1, 2, 2, 3) -- two distinct moments presented
    # as one instant, which is the signal these labels exist to carry. Tested for rather
    # than inferred from a spacing threshold, because collision is the thing that actually
    # matters and a threshold only approximates it: frames exactly 1s apart need no tenths.
    labels = render(0)
    return labels if len(set(labels)) == len(labels) else render(1)
