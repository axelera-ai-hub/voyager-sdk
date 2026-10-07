# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Qwen3-VL: preprocess -> vision graph -> decoder artifact -> text.

Two device artifacts, one per tower, sharing a device connection:

    image ──> processor ──> patches ──[vision graph, 1 dispatch]──> image embeds
                                                                        │
    prompt ──> tokenizer ──> input_ids ──────────────────────────> host embed
                                                                  (splice image
                                                                   embeds in)
                                                                        │
                                            [decoder artifact: prefill + decode]
                                                                        │
                                                                     tokens

The vision tower is compiled for ONE patch grid (square or not -- see
Qwen3VLVisionConfig), so images are resized to it before the processor runs; a mismatch
raises rather than silently producing the wrong token count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ...shared.video import is_video, VIDEO_SUFFIXES  # noqa: F401 -- re-exported
from .helpers import DEVICE_NAMES, supported_generation
from .qwen3vl import Qwen3VLConfig, mrope_cos_sin, mrope_position_ids
from .text_impl import Qwen3VLPrompt, Qwen3VLTextImplementation

logger = logging.getLogger(__name__)

_USER_AGENT = "axelera-qwen3vl/1.0 (alto llm tests; image fetch)"


# --------------------------------------------------------------------------------------
# Host preprocessing
# --------------------------------------------------------------------------------------


@dataclass
class VisionInputs:
    """One preprocessed (images/videos, text) sample."""

    input_ids: np.ndarray  # (T,) int64
    patches: np.ndarray | None  # (num_patches, patch_dim) float32
    grids: list[tuple[int, int, int]]  # per-image (t, h, w) patch grid
    #: Video patches, laid out exactly like image ones: a video's t temporal patches are
    #: t independent frames as far as the tower is concerned.
    video_patches: np.ndarray | None = None
    video_grids: list[tuple[int, int, int]] = field(default_factory=list)


def image_cache_dir() -> Path:
    """Where downloaded images live (override with $AXELERA_IMAGE_CACHE).

    Outside the repo on purpose: demo pictures are fetched, not committed.
    """
    return Path(os.environ.get("AXELERA_IMAGE_CACHE", Path.home() / ".cache" / "axelera_images"))


def fetch_source(source) -> Path:
    """A local path for a path or an http(s)/file URL, downloading once into the cache.

    Cached by the URL's hash, written via a temporary file + rename so a killed download
    leaves nothing half-written behind.
    """
    import hashlib
    import shutil
    import urllib.parse
    import urllib.request

    text = str(source)
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme in ("", "file") and not parsed.netloc:
        return Path(urllib.parse.unquote(parsed.path) if parsed.scheme else text)
    if parsed.scheme not in ("http", "https", "file"):
        raise ValueError(f"unsupported source scheme {parsed.scheme!r}: {text}")

    cache = image_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    suffix = Path(parsed.path).suffix.lower()
    dest = cache / (hashlib.sha1(text.encode()).hexdigest()[:16] + (suffix or ".img"))
    if not dest.exists():
        logger.info("fetching %s -> %s", text, dest)
        tmp = dest.with_suffix(dest.suffix + ".part")
        # A descriptive User-Agent is not optional for some hosts -- Wikimedia (the
        # obvious source of demo pictures and videos) 403s urllib's default.
        request = urllib.request.Request(text, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        tmp.rename(dest)
    return dest


def load_image(source):
    """Open an image given a local path OR an http(s)/file URL (cached)."""
    from PIL import Image

    if isinstance(source, Image.Image):
        return source
    return Image.open(fetch_source(source))


def probe_video(source) -> tuple[int, float, float]:
    """(frame count, seconds, rate) without decoding a single frame.

    What "analyse the whole video" would actually cost, cheaply enough to refuse before
    decoding. The count is the container's when it has one, otherwise duration x rate.
    """
    import av

    with av.open(str(fetch_source(source))) as container:
        if not container.streams.video:
            # An audio-only container opens fine and indexes empty, so without this the
            # caller gets `IndexError: tuple index out of range` from inside pyav rather
            # than the refusal `open_media` documents.
            raise ValueError(f"{Path(str(source)).name} has no video stream to read")
        stream = container.streams.video[0]
        rate = float(stream.guessed_rate or stream.average_rate or 30)
        seconds = (container.duration or 0) / 1e6 or None
        total = int(stream.frames) or (int(seconds * rate) if seconds else 0)
        return total, seconds or (total / rate if total else 0.0), rate


def load_video(source, *, frames: int | None = None, fps: float | None = None):
    """Decode a video from a path or URL into (frames, metadata).

    ``frames`` is how many to sample, equally distributed over the whole clip; omit it and
    EVERY frame is decoded (check `probe_video` first -- they all have to fit the model's
    context). ``fps`` samples at a rate instead -- ``duration * fps`` frames, still capped
    by ``frames``.

    pyav directly rather than ``transformers.load_video``: plenty of containers (the
    Wikimedia webm demos among them) report no frame count, and the library's sampler
    divides by it. Here it is estimated from duration x rate.

    The metadata keeps the SOURCE frame indices and rate, so the prompt's
    ``<X.X seconds>`` labels are the video's own clock, not the sampling loop's.
    """
    import av
    from transformers.video_utils import VideoMetadata

    path = fetch_source(source)
    total, seconds, rate = probe_video(path)  # the container's own count/length/rate
    duration = seconds or None
    with av.open(str(path)) as container:
        # No requested rate: exactly `frames`, spread over the clip. With one:
        # duration * fps frames, still capped by `frames`.
        wanted = frames or total or 1
        if fps is not None:
            wanted = max(1, round((duration or 1) * fps))
        if frames is not None:
            wanted = min(wanted, frames)
        wanted = min(wanted, total) if total else wanted
        # Sample the CENTRE of each of `wanted` equal segments rather than the endpoints:
        # the first and last frames of a real clip are usually a title card and a fade.
        step = max(1, round(rate / fps)) if fps else 1
        targets = (
            {int((index + 0.5) * total / wanted) for index in range(wanted)}
            if total
            else set(range(step // 2, wanted * step, step))
        )
        last = max(targets)
        sampled, indices = [], []
        for index, frame in enumerate(container.decode(video=0)):
            if index in targets:
                sampled.append(frame.to_ndarray(format="rgb24"))
                indices.append(index)
            if index >= last:
                break
    if not sampled:
        raise ValueError(f"decoded no frames from {source}")
    logger.info(
        "video %s: %d frames over %.1fs = %.2f fps effective (source %.2f fps%s)",
        source,
        len(sampled),
        duration or 0.0,
        len(sampled) / duration if duration else float("nan"),
        rate,
        f", requested {fps} fps" if fps else "",
    )
    metadata = VideoMetadata(
        total_num_frames=len(sampled),
        fps=rate,
        duration=duration or len(sampled) / rate,
        video_backend="pyav",
        frames_indices=indices,
    )
    return np.asarray(sampled), metadata


#: Mid-grey. The processor normalizes with mean 0.5 / std 0.5, so letterbox bars land at
#: ~0 -- the least signal a pad can carry into the tower.
_PAD = (128, 128, 128)


def fit_image(config: Qwen3VLConfig, image):
    """Force an image onto the compiled geometry (RGB), the way the artifact was built.

    Qwen's smart-resize leaves an already patch-aligned resolution alone, so this is what
    pins the processor's patch grid to the one the tower was compiled for. A raw HxWx3
    array is accepted too -- that is how video frames usually arrive.

    `config.vision.fit_mode` decides HOW a non-square picture is made to fit: `letterbox`
    keeps the aspect and spends patches on grey bars, `stretch` (the default) keeps every
    patch and distorts. stretch measured better at 256x256 -- ~5 ANLS points on DocVQA,
    ~4 on InfographicVQA -- because at that grid the bars cost more than the aspect they
    restore. A build whose own grid matches the pictures' aspect narrows the gap.
    """
    from PIL import Image

    vision = config.vision
    height, width = vision.image_shape  # PIL sizes are (w, h); the grid is (h, w)
    if not hasattr(image, "convert"):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    image = image.convert("RGB")
    if getattr(vision, "fit_mode", "stretch") != "letterbox":
        return image.resize((width, height))
    src_width, src_height = image.size
    # min() of the two ratios: the largest scale that still fits inside BOTH sides, which
    # for a square target is the old side / max(w, h).
    scale = min(width / src_width, height / src_height)
    fitted = image.resize((max(1, round(src_width * scale)), max(1, round(src_height * scale))))
    canvas = Image.new("RGB", (width, height), _PAD)
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def fit_video(config: Qwen3VLConfig, frames):
    """Frames -> (num_frames, height, width, 3) uint8 at the compiled geometry.

    Padded up to a whole number of temporal patches by repeating the last frame, which
    is what the video processor would do anyway -- doing it here keeps the frame count,
    the timestamps and the patch rows consistent.
    """
    pad_to = config.vision.temporal_patch_size
    frames = [np.asarray(fit_image(config, frame), dtype=np.uint8) for frame in frames]
    if not frames:
        raise ValueError("a video needs at least one frame")
    while len(frames) % pad_to:
        frames.append(frames[-1])
    return np.stack(frames)


def fit_metadata(meta, clip, fps: float):
    """Metadata matching a fitted clip: real frame times where we have them.

    Qwen3-VL separates frames with literal ``<X.X seconds>`` text, so the processor refuses
    to build a video prompt without frame times. `load_video`'s metadata carries the
    video's own clock; raw frames with none get 0..n-1 at `fps`. Frames added by temporal
    padding repeat the last index, keeping the list and the frame count in step.
    """
    from transformers.video_utils import VideoMetadata

    if meta is None:
        return VideoMetadata(
            total_num_frames=len(clip),
            fps=float(fps),
            duration=len(clip) / float(fps),
            video_backend="none",
            frames_indices=list(range(len(clip))),
        )
    indices = list(meta.frames_indices)
    indices += [indices[-1]] * (len(clip) - len(indices))
    meta.frames_indices = indices[: len(clip)]
    meta.total_num_frames = len(clip)
    return meta


def check_grids(config: Qwen3VLConfig, grids: list[tuple[int, int, int]]):
    """The processor's (t, h, w) grids, or a ValueError naming the fix.

    The tower is compiled for one grid, so a picture the processor sized its own way
    would produce the wrong number of vision tokens. Caught here, at the one place both
    the built-in rendering and a caller's own ``processor(...)`` output pass through.
    """
    vision = config.vision
    height, width = vision.image_shape
    for grid in grids:
        if grid[1:] != (vision.grid_h, vision.grid_w):
            raise ValueError(
                f"the processor produced a {grid[1:]} patch grid but the vision tower is "
                f"compiled for ({vision.grid_h}, {vision.grid_w}). Resize the image(s) to "
                f"{width}x{height} (width x height) first (model.fit_images / "
                "pipeline.fit_image) and process them again."
            )
    return grids


def render(
    processor,
    config: Qwen3VLConfig,
    messages: list[dict],
    images=None,
    videos=None,
    fps: float = 2.0,
    metadata=None,
) -> VisionInputs:
    """Render a conversation (+ its images/videos) into ids and flattened patches.

    `messages` is the chat-template form: [{"role": ..., "content": [{"type": "image"},
    {"type": "text", "text": ...}]}, ...]; a video is {"type": "video"}. Video frames are
    fitted exactly like images -- a video's temporal patches go through the same tower.
    `metadata` is the per-video `load_video` metadata, if the frames came from a real video.
    """
    images = [fit_image(config, image) for image in (images or [])]
    clips = [fit_video(config, frames) for frames in (videos or [])]
    metas = list(metadata) if metadata else [None] * len(clips)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    extra = {}
    if clips:
        # Frames are pre-sampled and pre-sized; the processor only needs the timestamps.
        # do_resize=False because the VIDEO processor splits its pixel budget across the
        # clip's frames, so it would shrink already-fitted frames to a smaller grid (a
        # 288x512 build got 12x22, a 256 square 18x18) and check_grids would reject them.
        extra = {
            "videos": clips,
            "video_metadata": [fit_metadata(meta, clip, fps) for meta, clip in zip(metas, clips)],
            "do_sample_frames": False,
            "do_resize": False,
        }
    batch = processor(text=prompt, images=images or None, return_tensors="pt", **extra)

    def _grids(key: str) -> list[tuple[int, int, int]]:
        return check_grids(config, [tuple(int(v) for v in row) for row in batch[key]])

    patches = video_patches = None
    grids: list[tuple[int, int, int]] = []
    video_grids: list[tuple[int, int, int]] = []
    if images:
        grids = _grids("image_grid_thw")
        patches = batch["pixel_values"].to(torch.float32).numpy()
    if clips:
        video_grids = _grids("video_grid_thw")
        video_patches = batch["pixel_values_videos"].to(torch.float32).numpy()
    return VisionInputs(batch["input_ids"][0].numpy(), patches, grids, video_patches, video_grids)


def build_prompt(
    config: Qwen3VLConfig,
    inputs: VisionInputs,
    image_embeds: np.ndarray | None,
    deepstack: list[np.ndarray] | None = None,
) -> Qwen3VLPrompt:
    """Bundle the prompt's MRoPE tables + image-embedding placement for the decoder.

    ``deepstack`` is the vision tower's per-layer feature maps; they land on the same
    placeholder rows as the embeddings, so they must have the same row count.
    """
    ids = np.asarray(inputs.input_ids).reshape(-1)
    position_ids, delta = mrope_position_ids(ids, config, inputs.grids, inputs.video_grids)
    cos, sin = mrope_cos_sin(
        position_ids, config.text.head_dim, config.text.rope_theta, config.mrope_section
    )
    # Images and videos share one row space: the decoder just sees vision placeholders,
    # filled in token order (see join_vision).
    positions = np.flatnonzero((ids == config.image_token_id) | (ids == config.video_token_id))
    placed = [("embeddings", image_embeds)]
    placed += [(f"DeepStack feature map {i}", f) for i, f in enumerate(deepstack or [])]
    for what, rows in placed:
        if rows is not None and len(positions) != len(rows):
            raise ValueError(
                f"{len(positions)} vision placeholder tokens but {len(rows)} vision "
                f"{what} -- the compiled grid and the prompt disagree"
            )
    return Qwen3VLPrompt(
        input_ids=ids,
        image_positions=positions,
        image_embeds=image_embeds,
        deepstack=deepstack,
        rope_cos=cos,
        rope_sin=sin,
        delta=delta,
    )


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def join_vision(config: Qwen3VLConfig, inputs: VisionInputs, encoded: dict) -> tuple:
    """Per-modality tower outputs -> (embeddings, DeepStack maps) in token order.

    ``encoded`` maps a placeholder token id to that modality's (merged, deepstack).
    A prompt can interleave images and videos in any order, so the rows are placed at
    each modality's positions WITHIN the union of vision positions rather than
    concatenated modality by modality (the numpy twin of qwen3vl.join_vision_rows).
    """
    ids = np.asarray(inputs.input_ids).reshape(-1)
    if len(encoded) == 1:  # the common case: one modality, already in order
        return next(iter(encoded.values()))
    vision = (ids == config.image_token_id) | (ids == config.video_token_id)
    total = int(vision.sum())

    def _join(rows_by_token):
        first = next(iter(rows_by_token.values()))
        joint = np.zeros((total, first.shape[-1]), dtype=first.dtype)
        for token_id, rows in rows_by_token.items():
            joint[(ids == token_id)[vision]] = rows
        return joint

    merged = _join({token: value[0] for token, value in encoded.items()})
    slots = len(next(iter(encoded.values()))[1])
    deepstack = [
        _join({token: value[1][slot] for token, value in encoded.items()}) for slot in range(slots)
    ]
    return merged, deepstack


#: The default calibration set: DocVQA (document photographs + real questions about
#: them). Read one parquet shard at a time, and only as many as the sample count needs.
DOCVQA = ("lmms-lab-encoder/DocVQA", "DocVQA", "validation")


def docvqa_calibration(num_samples: int = 48, *, source=DOCVQA) -> tuple[list, list[str]]:
    """(images, questions) from DocVQA -- what calibration uses when given nothing else.

    Real (image, question) pairs, because the activation scales should be set from what
    the deployed model actually sees. Pass your own to `multimodal_calibration` instead.

    The shards are read directly rather than through ``load_dataset(repo, config)``: that
    route resolves the corpus through the hub's dataset server, which hangs on this one
    (`lmms-lab/DocVQA` was renamed and now redirects). A shard is ~900 rows, so any sane
    calibration set comes out of the first.
    """
    from datasets import load_dataset
    from huggingface_hub import HfFileSystem

    path, name, split = source
    shards = sorted(HfFileSystem().glob(f"datasets/{path}/{name}/{split}-*.parquet"))
    if not shards:
        raise ValueError(f"no {split} shards in {path} ({name})")
    images, questions = [], []
    for shard in shards:
        for row in load_dataset("parquet", data_files=f"hf://{shard}", split="train"):
            images.append(row["image"])
            # `question` on DocVQA; `query` on the smaller community mirrors of it.
            question = row.get("question") or row["query"]
            questions.append(question["en"] if isinstance(question, dict) else question)
            if len(images) == num_samples:
                return images, questions
    logger.warning(
        "%s (%s/%s) has only %d samples, %d were asked for",
        path,
        name,
        split,
        len(images),
        num_samples,
    )
    return images, questions


def multimodal_calibration(
    processor, config: Qwen3VLConfig, images=None, prompts=None, *, num_samples: int = 48
) -> list[dict]:
    """Calibration samples for the joint PTQ pass: real (image, question) pairs.

    Both towers are calibrated together -- the decoder's activation scales have to see
    the image embeddings the vision tower actually produces, and the vision tower's
    LayerNorm/GELU ranges are nothing like a random tensor's.

    `images` may be PIL images, paths, or URLs (fetched through the image cache); omit
    them and `num_samples` DocVQA pairs are used.
    """
    if images is None:
        images, docvqa_prompts = docvqa_calibration(num_samples)
        prompts = prompts if prompts is not None else docvqa_prompts
    samples = []
    for image, prompt in zip(images, prompts):
        image = load_image(image)
        item = render(
            processor,
            config,
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            [image],
        )
        samples.append(
            {
                "input_ids": torch.as_tensor(item.input_ids)[None],
                "patches": torch.as_tensor(item.patches),
                "grids": item.grids,
            }
        )
    return samples


#: Shared with SmolVLM2, which needs the same move for the same reason -- a second copy is
#: how the two drift.
from ...calibration import require_vision_samples, to_device  # noqa: E402


def build_artifacts(
    repo_id: str,
    out_dir,
    *,
    calibration=None,
    num_samples: int = 48,
    image_size: int | tuple[int, int] = 256,
    fit_mode: str = "stretch",
    context_len: int = 2048,
    target: str = "device",
    num_cores: int = 1,
    generation: str = "OMEGA",
    smooth_alpha: float = 0.8,
    qk_alpha: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Quantize once, then compile both towers into ``out_dir``.

    Returns {"vision": dir, "text": dir, "bundle": dir}.

    ``calibration`` is a sample list from `multimodal_calibration`; omitted, it is
    ``num_samples`` DocVQA pairs at this repo's geometry.

    ``image_size`` is the geometry the vision tower is pinned to: a square side, or an
    (height, width) pair (288x512 -> an 18x32 grid). One artifact serves exactly ONE
    geometry -- the patch count, the position embeddings, the 2-D RoPE tables and the
    attention mask are all baked at it -- so a second resolution is a second call.

    ``fit_mode`` ("stretch" | "letterbox") is how non-square pictures are made to fit it;
    it is recorded in the bundle, so inference reproduces the geometry calibration saw.

    ``generation`` ("OMEGA" | "EUROPA") goes to both towers: they share a device, so they
    must agree. ``device`` is the **torch** device the reference calibrates on ("cpu",
    "cuda:1", ...), distinct from that target -- calibration is the bulk of a build, so a
    GPU is worth having. The model comes back to the host before export, which reads numpy.
    """
    import torch as _torch

    from ...generate import GenerationConfig, build_quantized_model
    from ...implementation import get_implementation
    from .qwen3vl import Qwen3VLForConditionalGeneration, quantize_model
    from .vision_impl import build_vision_graph

    out_dir = Path(out_dir)
    calib_device = _torch.device(device)
    if calib_device.type == "cuda" and not _torch.cuda.is_available():
        raise RuntimeError(f"calibration device {device!r} requested but torch sees no CUDA")
    # quantize_attention=True: the decoder layer kernel binds calibrated SDPA scales.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        repo_id, image_size=image_size, device=calib_device, quantize_attention=True
    )
    # Set before calibration and carried in the bundle: the scales are set on the same
    # geometry inference will produce, and the runtime reads back what was built.
    if fit_mode not in ("letterbox", "stretch"):
        raise ValueError(f"fit_mode must be 'letterbox' or 'stretch', not {fit_mode!r}")
    model.config.vision.fit_mode = fit_mode
    if calibration is None:
        from transformers import AutoProcessor

        calibration = multimodal_calibration(
            AutoProcessor.from_pretrained(repo_id), model.config, num_samples=num_samples
        )
    require_vision_samples(calibration, "patches", "qwen3vl.pipeline.multimodal_calibration")
    # The samples must live where the model does; the (t, h, w) grids stay python.
    calibration = [to_device(sample, calib_device) for sample in calibration]
    logger.info(
        "calibrating %s on %d multimodal samples (%s)", repo_id, len(calibration), calib_device
    )
    # Timed and announced: on a CPU host this is most of the build, and in silence there is
    # no telling a slow calibration from a hung one.
    started = perf_counter()
    quantize_model(model, calibration, smooth_alpha=smooth_alpha, qk_alpha=qk_alpha)
    logger.info("quantized in %.0fs; fitting LUT ranges", perf_counter() - started)
    from ...quant import record_lut_ranges

    lut_started = perf_counter()
    record_lut_ranges(model, calibration)
    logger.info(
        "calibration done in %.0fs (quantize %.0fs + LUT ranges %.0fs); compiling",
        perf_counter() - started,
        lut_started - started,
        perf_counter() - lut_started,
    )
    # Export and the vision binder go through numpy, so come home before either.
    model.to("cpu")

    bundle = build_quantized_model("qwen3vl", model)
    bundle.save(out_dir / "quantized_model")
    # The multicore decoder kernel shards SDPA heads + the MLP matmuls; below two cores
    # its extra machinery buys nothing, so the single-core kernel stays the default.
    implementation = "qwen3vl-metis-multicore" if num_cores > 1 else "qwen3vl-metis-opt"
    text_config = GenerationConfig(
        implementation=implementation,
        repo_id=repo_id,
        artifact_dir=out_dir / "text",
        context_len=context_len,
        num_cores=num_cores,
        generation=supported_generation(generation),
        target=target,
        device="cpu",  # the reference is back on the host by now
    )
    text_dir = get_implementation(implementation).compile(bundle, text_config)

    from axelera.compiler.alto.compiler.config import Target

    vision_dir, _ = build_vision_graph(
        model.visual,
        out_dir / "vision",
        target=Target.DEVICE if target == "device" else Target.AICORE_SIM,
        cores=num_cores,
        generation=generation,
    )
    # Last, so a half-built directory is not mistaken for a loadable one. The geometry
    # comes off the config, which has already normalized and validated whatever spelling
    # of `image_size` the caller used.
    save_hf_metadata(out_dir, repo_id, model.config.vision.image_shape)
    return {
        "vision": Path(vision_dir),
        "text": Path(text_dir),
        "bundle": out_dir / "quantized_model",
    }


# --------------------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------------------


def _weight_gb(directory: Path) -> float:
    """What a built tower weighs on disk -- what the runtime actually tried to upload."""
    return sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file()) / 1e9


def artifacts(artifact_dir) -> dict:
    """The three directories build_artifacts writes, as the loaders want them."""
    artifact_dir = Path(artifact_dir)
    return {
        "vision": artifact_dir / "vision",
        "text": artifact_dir / "text",
        "bundle": artifact_dir / "quantized_model",
    }


def hf_metadata_dir(artifact_dir) -> Path | None:
    """The artifact directory itself if `save_hf_metadata` wrote HF files into it.

    One check ("is there a config.json?") shared by everything that would otherwise
    reach for the hub: artifacts built before this returns None, and those callers fall
    back to the repo id recorded in the artifact.
    """
    path = Path(artifact_dir)
    return path if (path / "config.json").exists() else None


def save_hf_metadata(out_dir, repo_id: str, image_shape: tuple[int, int]) -> None:
    """Write the HuggingFace-side files into a built artifact.

    What makes the artifact directory loadable through the stock Auto classes
    (`AutoConfig`/`AutoProcessor`/`AutoModelForImageTextToText`.from_pretrained) with no
    hub and no network.

    The saved processor is pinned to the compiled geometry's PIXEL COUNT (``size``, which
    is how the Qwen image processor spells min/max pixels), so an image already at the
    compiled height x width passes through untouched whatever its aspect. A budget cannot
    force an aspect though -- a differently shaped picture still needs `fit_images` first.
    """
    from transformers import AutoProcessor, GenerationConfig as HFGenerationConfig

    from .hf_model import artifact_config

    out_dir = Path(out_dir)
    height, width = image_shape
    pixels = height * width
    processor = AutoProcessor.from_pretrained(
        repo_id, size={"shortest_edge": pixels, "longest_edge": pixels}
    )
    processor.save_pretrained(out_dir)
    artifact_config(repo_id).save_pretrained(out_dir)
    # Sampling defaults travel with the artifact too, exactly as the checkpoint ships them
    # (2B: do_sample, 0.7 / 0.8 / 20), so a script that never mentions the hub repo still
    # behaves like the model it was built from. The Technical Report evaluates these sizes
    # at temperature 1.0 / top-p 1.0 / top-k 40 (5.11) -- pass those to generate() to
    # reproduce its numbers.
    HFGenerationConfig.from_pretrained(repo_id).save_pretrained(out_dir)


def add_artifact_args(parser, *, log_level: str = "WARNING", required: bool = True):
    """The flags every tool here needs to open a built artifact (qwen3vl_run, ui, benchmark).

    `required=False` is for a tool with somewhere else to get an artifact from: the demo
    page downloads a prebuilt one.
    """
    parser.add_argument("--artifact-dir", required=required, default=None, type=Path)
    parser.add_argument(
        "--repo-id", default=None, help="processor/tokenizer repo (default: the artifact's)"
    )
    # Choices from DEVICE_NAMES rather than spelled again here (see helpers.py).
    parser.add_argument(
        "--generation",
        default=None,
        choices=tuple(DEVICE_NAMES),
        help="hardware generation the artifact was compiled for (default: the artifact's "
        "own record; only needed for a tower built before that was written)",
    )
    parser.add_argument(
        "--log-level", default=log_level, choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def open_artifact(artifact_dir, repo_id: str | None = None) -> tuple[dict, str]:
    """(tower directories, repo id) for a built artifact -- the repo id off it if unset."""
    from ...generate import artifact_repo_id

    dirs = artifacts(artifact_dir)
    repo_id = repo_id or artifact_repo_id(dirs["text"])
    if repo_id is None:
        raise ValueError(
            f"no repo id recorded in {artifact_dir}; pass --repo-id (repo_id=) explicitly"
        )
    return dirs, repo_id


#: Tokens a temporal patch costs BEYOND its placeholders: its `<X.X seconds>` label plus
#: the vision_start/end pair (measured at 7; rounded up).
_PER_PATCH_OVERHEAD = 8
#: Tokens for the chat template and the question itself (measured at ~21; rounded up).
_PROMPT_RESERVE = 24


def resolve_frames(pipeline, source, frames, max_new_tokens: int) -> int:
    """How many frames to take from `source`, or a ValueError saying what would fit.

    A count means that many, equally distributed. NO count means the whole video: every
    frame it has. Either way the frames have to fit the artifact's context alongside the
    answer, and if they do not this refuses rather than quietly sampling fewer -- which
    would answer a different question than the one asked.
    """
    budget = frame_budget(pipeline, max_new_tokens)
    # `frame_budget` floors at one temporal patch, so a large enough --max-new-tokens
    # leaves a "budget" that does not itself fit. Whether the budget fits is therefore a
    # separate question from whether the request is within it, and both are asked here --
    # otherwise an impossible request passes this gate and fails in prefill instead, which
    # is precisely what this function exists to say first.
    affordable = frame_cost(pipeline, budget) + max_new_tokens <= pipeline.context_len
    whole = frames is None
    if whole:
        total, seconds, rate = probe_video(source)
        logger.info(
            "no frame count: the whole video, %d frames (%.1fs at %.2f fps)", total, seconds, rate
        )
        frames = max(1, total)
    if frames > budget or not affordable:
        needed = frame_cost(pipeline, frames) + max_new_tokens
        asked = (
            f"the whole video is {frames} frames, which need"
            if whole
            else f"--frames {frames} needs"
        )
        remedy = (
            f"Use --frames {budget} or fewer, lower --max-new-tokens, or build the "
            "artifact with a larger context_len."
            if affordable
            else f"No video fits: a {max_new_tokens}-token answer leaves no room for even "
            f"one temporal patch ({frame_cost(pipeline, 1)} tokens with the prompt). Lower "
            "--max-new-tokens, or build the artifact with a larger context_len."
        )
        raise ValueError(
            f"{asked} about {needed} tokens ({frame_cost(pipeline, frames)} prompt + "
            f"{max_new_tokens} answer) but this artifact's context holds "
            f"{pipeline.context_len}. {remedy}"
        )
    return frames


def open_media(pipeline, sources, *, frames: int | None = None, max_new_tokens: int = 128):
    """Paths or links -> (images, (clip, metadata) or None), or an error saying why not.

    Two kinds: a ValueError for a request this refuses (mixed media, too many videos, a
    clip that cannot fit), and an OSError for a source that will not open at all -- a
    missing file, an unreachable URL, bytes that are not an image. Callers that turn a
    refusal into a message have to catch both, or a typo'd path is a traceback.

    One picture, several pictures, or one video, told apart by the FAMILY's suffix list
    rather than a copy of it: two lists drift into a clip this refuses and its neighbour
    accepts. How many of a video's frames fit is `resolve_frames`' problem. Pictures and a
    clip share one vision budget, so mixing them is refused here, once, for every tool
    that takes media.
    """
    images, videos = [], []
    for source in (str(s).strip() for s in sources or []):
        if source:
            (videos if is_video(source) else images).append(source)
    if videos and images:
        raise ValueError(
            f"pictures and a video at once ({len(images)} picture(s) + "
            f"{Path(videos[0]).name}): they share one vision budget, so drop one -- clear "
            "the uploads, or the paths/links box"
        )
    if len(videos) > 1:
        raise ValueError(f"one video at a time (got {len(videos)})")
    if videos:
        count = resolve_frames(pipeline, videos[0], frames, max_new_tokens)
        return [], load_video(videos[0], frames=count)
    return [load_image(source) for source in images], None


def frame_cost(pipeline, frames: int, reserve: int = _PROMPT_RESERVE) -> int:
    """Prompt tokens a video of `frames` frames costs, answer excluded.

    The inverse of frame_budget, sharing its constants so an error message can quote the
    number that made the request impossible.
    """
    vision = pipeline.config.vision
    patches = -(-frames // vision.temporal_patch_size)  # frames pair up, rounding up
    return patches * (vision.num_merged + _PER_PATCH_OVERHEAD) + reserve


def frame_budget(pipeline, max_new_tokens: int, reserve: int = _PROMPT_RESERVE) -> int:
    """How many video frames fit the artifact's context alongside the answer.

    A *temporal patch* -- `temporal_patch_size` frames, paired by the video processor --
    costs `num_merged` placeholder tokens, so at 256x256 that is 64 tokens per TWO frames.
    On top of that each patch carries its timestamp label and vision markers, and the
    prompt and the answer share the same context.

    Deliberately a patch or so conservative: overflowing fails the prefill, one frame
    fewer costs a little detail. It is the ceiling a request is checked against
    (`resolve_frames`), not a default -- `--frames` decides that. At context 512: 6 frames
    beside a 256-token answer, 10 beside a 64-token one (the measured ceiling is 12).

    Never returns zero: one temporal patch is the floor, so the UI slider this bounds
    always has a range and an error message always has a count to quote. That makes the
    floor a CEILING that need not itself fit -- ask for an answer longer than the context
    and the one patch costs more than is left -- so `resolve_frames` checks affordability
    on top of this, and a caller that only compares against the return value is admitting
    a request that will fail in prefill.
    """
    vision = pipeline.config.vision
    budget = pipeline.context_len - max_new_tokens - reserve
    patches = max(1, budget // (vision.num_merged + _PER_PATCH_OVERHEAD))
    return patches * vision.temporal_patch_size


def artifact_generation(vision_dir) -> str | None:
    """The generation a compiled vision tower recorded, or None if it recorded none.

    `build_vision_graph` writes it into meta.json, so which device an artifact is for is
    a fact about the artifact -- not something a caller has to remember and can get wrong.
    """
    import json

    meta = Path(vision_dir) / "meta.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text()).get("generation")


class Qwen3VLPipeline:
    """A loaded Qwen3-VL: both compiled towers plus the host glue between them."""

    def __init__(
        self, artifacts: dict, repo_id: str, *, generation: str | None = None, device=None
    ):
        import json

        from transformers import AutoProcessor

        from ...generate import RuntimeModel
        from ...implementation import get_implementation
        from .vision_impl import Qwen3VLVisionDevice

        text_dir = Path(artifacts["text"])
        self.repo_id = repo_id
        # The artifact's own processor: the chat template and patch geometry it was
        # calibrated and compiled against, loadable with no hub and no network. The repo id
        # is the fallback for artifacts built before that was saved, and the override.
        self.processor = AutoProcessor.from_pretrained(hf_metadata_dir(text_dir.parent) or repo_id)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.pipeline = json.loads((text_dir / "pipeline.json").read_text())
        self.host_params = dict(np.load(text_dir / self.pipeline["host_params"]))
        self.impl: Qwen3VLTextImplementation = get_implementation(self.pipeline["implementation"])
        # The artifact's own record wins; `generation` is for towers built before it was
        # written, and a mismatch is a mistake worth saying out loud -- it names the wrong
        # device everywhere and sizes the memory share wrong.
        recorded = artifact_generation(artifacts["vision"])
        if recorded and generation and supported_generation(generation) != recorded.upper():
            raise ValueError(
                f"this artifact was compiled for {recorded.upper()} but "
                f"generation={str(generation).upper()} was asked for; drop the flag to use "
                "the artifact's own"
            )
        self.generation = supported_generation(recorded or generation or "OMEGA")
        # No sub-device count anywhere here: RuntimeModel claims one per core the artifact's
        # call graphs actually use, so cores this run does not need stay free. WHICH part to
        # claim them on is the caller's, though: a host holds several and one of them can be
        # unhealthy, so the count stays derived while the part stays askable.
        self.text_runtime = RuntimeModel(text_dir / "model.json", device=device)
        # Both towers share one device connection: a second Context would claim the
        # sub-devices again and fail.
        try:
            self.vision = Qwen3VLVisionDevice.load(
                artifacts["vision"], share_from=self.text_runtime, generation=self.generation
            )
        except Exception as refusal:
            # Out of device memory, almost always: the decoder is resident by now, and from
            # 8B up it alone is most of the card. Hand the card back before raising, or the
            # NEXT run fails too for a reason that has nothing to do with it.
            self.text_runtime.release()
            raise RuntimeError(
                f"the decoder ({_weight_gb(text_dir):.2f} GB) and the vision tower "
                f"({_weight_gb(artifacts['vision']):.2f} GB) do not fit this device "
                "together. They must be resident at once; run this model on a device "
                "with more memory (`axdevice` prints what this one has)."
            ) from refusal
        # The model config travels with the quantized bundle (written next to the towers).
        bundle_dir = Path(artifacts.get("bundle", text_dir.parent / "quantized_model"))
        manifest = json.loads((bundle_dir / "quantized_model.json").read_text())
        self.config = Qwen3VLConfig(**manifest["config"])
        # The tower's tap outputs feed the decoder's per-layer adds one for one. Both counts
        # come off the built artifacts rather than the config, so a pre-DeepStack pair (no
        # taps, no adds) still loads -- and a mismatched pair says so here, not at the first
        # prompt.
        self.num_deepstack = self.vision.num_deepstack
        wanted = int(self.host_params.get("deepstack_layers", 0))
        if self.num_deepstack != wanted:
            raise ValueError(
                f"the vision tower emits {self.num_deepstack} DeepStack feature maps but "
                f"the decoder injects into {wanted} layers -- the two artifacts are from "
                "different builds"
            )

    @property
    def context_len(self) -> int:
        return int(self.pipeline["context_len"])

    @property
    def num_cores(self) -> int:
        """AI cores this artifact places work on -- read off it, never asked for.

        The runtime claims one sub-device per compiled core, so this is also how much of the
        card the model gets. Nothing at load time can change it.
        """
        return int(self.text_runtime.cores)

    def default_config(self, **overrides):
        from ...generate import GenerationConfig

        return GenerationConfig(context_len=self.context_len, **overrides)

    # --- the two towers, drivable separately ---------------------------------------

    def encode_images(self, images) -> tuple[np.ndarray, list[np.ndarray]]:
        """Image(s) -> (merged embeddings, DeepStack features): the vision tower alone.

        Everything comes back in image order, which is the order the decoder's
        image-placeholder rows are filled in.
        """
        images = list(images)
        inputs = render(
            self.processor,
            self.config,
            [{"role": "user", "content": [{"type": "image"} for _ in images]}],
            images,
        )
        return self.vision.encode(inputs.patches)

    def encode_video(self, frames, fps: float = 2.0, metadata=None):
        """Video frames -> (merged embeddings, DeepStack features), same as images.

        A video's temporal patches are independent frame encodings -- HF blocks vision
        attention per frame and repeats the position and rotary tables -- so this is the
        multi-image path with the frames paired up by the video processor.
        """
        inputs = render(
            self.processor,
            self.config,
            [{"role": "user", "content": [{"type": "video"}]}],
            videos=[frames],
            fps=fps,
            metadata=None if metadata is None else [metadata],
        )
        return self.vision.encode(inputs.video_patches)

    def prepare(
        self, messages: list[dict], images=None, videos=None, vision=None, fps=2.0, metadata=None
    ):
        """Render a conversation and encode its images/videos; returns the decoder prompt.

        `vision` reuses an earlier `encode_image`/`encode_images`/`encode_video` result
        -- the (embeddings, deepstack) pair -- so a chat turn does not re-encode the same
        picture. With both modalities in one prompt the rows are interleaved into token
        order by `join_vision`.
        """
        inputs = render(self.processor, self.config, messages, images, videos, fps, metadata)
        return self.prepare_inputs(inputs, vision)

    def prepare_inputs(self, inputs: VisionInputs, vision=None) -> Qwen3VLPrompt:
        """Preprocessed inputs -> decoder prompt: encode the vision side, build the tables.

        The half of `prepare` below the processor, so a caller who ran the HF processor
        itself (the HF-API shim does) reaches the same path with the same checks.
        """
        if vision is None:
            encoded = {
                token_id: self.vision.encode(patches)
                for token_id, patches in (
                    (self.config.image_token_id, inputs.patches),
                    (self.config.video_token_id, inputs.video_patches),
                )
                if patches is not None
            }
            vision = join_vision(self.config, inputs, encoded) if encoded else None
        image_embeds, deepstack = vision if vision is not None else (None, None)
        return build_prompt(self.config, inputs, image_embeds, deepstack)

    def _drive(self, generate, prompt: Qwen3VLPrompt, config):
        """Run one of the implementation's generators with the prompt context set."""
        self.impl.set_prompt(prompt)
        try:
            yield from generate(
                self.text_runtime,
                self.tokenizer,
                [int(t) for t in prompt.input_ids],
                config or self.default_config(),
                host_params=self.host_params,
            )
        finally:
            self.impl.set_prompt(None)

    def stream(self, prompt: Qwen3VLPrompt, config=None):
        """Drive the decoder over a prepared prompt, yielding cumulative text."""
        yield from self._drive(self.impl.generate, prompt, config)

    def stream_token_ids(self, prompt: Qwen3VLPrompt, config=None):
        """The token-id form of `stream` (what the benchmark times)."""
        yield from self._drive(self.impl.generate_token_ids, prompt, config)
