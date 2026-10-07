# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""End to end: preprocess -> vision graph -> decoder artifact -> text.

Two device artifacts, one per tower, sharing a device connection::

    image ──> processor ──> patches ──[vision graph, 1 dispatch per tile]──> embeddings
                                                                                 │
    prompt ──> tokenizer ──> input_ids ───────────────────────────────────> host embed
                                                                            (scatter the
                                                                             embeddings)
                                                                                 │
                                                     [decoder artifact: prefill + decode]
                                                                                 │
                                                                              tokens

The vision tower is compiled for ONE patch grid, which the report calls out as the knob
that trades accuracy for latency. Two things follow:

* a picture is fitted to that grid before the processor runs (:func:`fit_image`), and a
  mismatch raises rather than silently producing the wrong token count;
* **dynamic tiling still works**, because a tile is exactly that grid. A high-resolution
  picture is reshaped to the closest supported tile aspect and cut into ``tile_size``
  squares, plus a whole-image thumbnail -- each one dispatch. The tiles come from the
  image processor's own ``crop_image_to_patches`` and the prompt from its own
  ``expand_text_with_placeholders``, so the row/column markers, the thumbnail token and
  the placeholder counts are HuggingFace's, not a restatement of them. Only the
  thumbnail is re-encoded, at the compiled grid instead of the processor's own
  variable one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .helpers import (
    DEFAULT_GENERATION,
    DEVICE_NAMES,
    attached_generation,
    device_name,
    supported_generation,
)
from .lfm2vl import Lfm2VlConfig
from .text_impl import FULL_PREFILL_GRAPH, Lfm2VlPrompt, Lfm2VlTextImplementation

logger = logging.getLogger(__name__)

_USER_AGENT = "axelera-lfm2vl/1.0 (alto llm tests; image fetch)"

#: Mid-grey. The processor normalizes with mean 0.5 / std 0.5, so letterbox bars land at
#: ~0 -- the least signal a pad can carry into the tower.
_PAD = (128, 128, 128)


# --------------------------------------------------------------------------------------
# Host preprocessing
# --------------------------------------------------------------------------------------


@dataclass
class VisionInputs:
    """One preprocessed (images, text) sample."""

    input_ids: np.ndarray  # (T,) int64
    patches: np.ndarray | None  # (n_dispatches * num_patches, patch_dim) float32
    #: Per image, the (rows, cols) tile layout the processor chose -- (1, 1) when the
    #: picture was small enough to encode whole. A split picture is rows * cols tiles
    #: plus its thumbnail, so it is that many dispatches plus one.
    layouts: list[tuple[int, int]] = field(default_factory=list)


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
        # A descriptive User-Agent is not optional for some hosts.
        request = urllib.request.Request(text, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        tmp.rename(dest)
    return dest


def _as_rgb(image):
    """The picture as 3-channel RGB, converting only when it is not already.

    The tower is 3-channel by construction, and the image processor identifies the channel
    axis by looking for a 1 or a 3 in the shape (`infer_channel_dimension_format`). Four
    channels match neither, so an RGBA screenshot does not arrive as a wrong picture -- it
    raises "Unable to infer channel dimension format" from inside transformers, naming
    neither alpha nor the file. Normalizing here is what keeps a PNG with an alpha channel, a
    palette GIF and an L-mode scan ordinary inputs rather than that traceback.

    Guarded on the mode rather than converted unconditionally, because `convert` copies and
    the common case -- a JPEG, every DocVQA calibration row -- is already RGB.

    The conversion DROPS alpha rather than compositing it, so a genuinely transparent picture
    comes through against black. That is right for the opaque screenshots this exists for, and
    compositing would be a choice about a background colour that nothing here knows.
    """
    return image if image.mode == "RGB" else image.convert("RGB")


def load_image(source):
    """Open an image given a local path OR an http(s)/file URL (cached), as RGB.

    Every failure arrives as a ValueError naming what went wrong, because that is the only
    kind the run CLI's REPL recovers from -- it catches ValueError/SystemExit and prints
    them, and anything else ends the session. The three that reach here are all ordinary
    typing rather than exotic: a path that does not exist (FileNotFoundError), a clip or
    any non-image handed to `/image` (PIL's UnidentifiedImageError), and a URL the server
    refuses (urllib's HTTPError -- Wikimedia's thumbnail policy returns 400 for a
    hand-edited size). All three are OSError subclasses, so none of them was recoverable
    and one typo ended a multi-turn session.

    `Image.open` is also lazy: it reads the header and defers the pixels, so a truncated
    or mislabelled file would otherwise raise later, from inside the processor, where there
    is no longer a source path to name. `load()` pulls the decode forward to here.

    The mode is normalized for the same reason (`_as_rgb`): the TILING path hands the picture
    straight to the image processor, which cannot infer a channel axis from four, so an RGBA
    file failed there while the unsplit path -- which goes through `fit_image`, already
    converting -- took it. One door, one mode, and the two paths cannot disagree again.
    """
    from PIL import Image, UnidentifiedImageError

    if isinstance(source, Image.Image):
        return _as_rgb(source)
    try:
        path = fetch_source(source)
    except OSError as error:  # HTTPError/URLError are OSError subclasses
        raise ValueError(f"could not fetch {source!r}: {error}") from error
    try:
        image = Image.open(path)
        image.load()
    except FileNotFoundError as error:
        raise ValueError(f"no such image: {path}") from error
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"{path} is not an image this model can read ({type(error).__name__}). "
            "Pictures only -- LFM2.5-VL takes stills, not clips."
        ) from error
    return _as_rgb(image)


def fit_image(config: Lfm2VlConfig, image):
    """Force an image onto the compiled geometry (RGB), the way the artifact was built.

    The processor's own resize preserves the input's aspect ratio, so this is what pins
    its patch grid to the one the tower was compiled for. A raw HxWx3 array is accepted
    too.

    ``config.vision.fit_mode`` decides HOW a non-square picture is made to fit:
    ``letterbox`` keeps the aspect and spends patches on grey bars, ``stretch`` (the
    default) keeps every patch and distorts.
    """
    from PIL import Image

    vision = config.vision
    height, width = vision.image_shape  # PIL sizes are (w, h); the grid is (h, w)
    if not hasattr(image, "convert"):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    image = _as_rgb(image)
    if getattr(vision, "fit_mode", "stretch") != "letterbox":
        return image.resize((width, height))
    src_width, src_height = image.size
    # min() of the two ratios: the largest scale that still fits inside BOTH sides.
    scale = min(width / src_width, height / src_height)
    fitted = image.resize((max(1, round(src_width * scale)), max(1, round(src_height * scale))))
    canvas = Image.new("RGB", (width, height), _PAD)
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def check_grid(config: Lfm2VlConfig, shape) -> tuple[int, int]:
    """One (h, w) patch grid, or a ValueError naming the fix."""
    vision = config.vision
    grid = (int(shape[0]), int(shape[1]))
    if grid != (vision.grid_h, vision.grid_w):
        height, width = vision.image_shape
        raise ValueError(
            f"the processor produced a {grid} patch grid but the vision tower is compiled "
            f"for ({vision.grid_h}, {vision.grid_w}). Resize the image(s) to {width}x{height} "
            "(width x height) first (model.fit_images / pipeline.fit_image) and process "
            "them again."
        )
    return grid


def _unpad(batch, config: Lfm2VlConfig, rows) -> np.ndarray:
    """The chosen rows of a processor batch as (len(rows) * num_patches, patch_dim).

    Each row is checked against the compiled grid and sliced back to its real patch count
    (the processor right-pads every image to one width).
    """
    vision = config.vision
    values = batch["pixel_values"].to(torch.float32).numpy()
    out = []
    for row in rows:
        check_grid(config, batch["spatial_shapes"][row])
        out.append(values[row, : vision.num_patches])
    return np.concatenate(out, axis=0)


def encode_thumbnail(processor, config: Lfm2VlConfig, image) -> np.ndarray:
    """One whole picture at the compiled grid: ``(num_patches, patch_dim)``.

    Fitted first, so the processor's aspect-preserving resize is a no-op, and its pixel
    budget is pinned to this grid's token count so it cannot pick a size of its own.
    """
    vision = config.vision
    batch = processor.image_processor(
        [fit_image(config, image)],
        return_tensors="pt",
        do_image_splitting=False,
        min_image_tokens=vision.num_merged,
        max_image_tokens=vision.num_merged,
    )
    return _unpad(batch, config, [0])


def tile_image(processor, config: Lfm2VlConfig, image):
    """Split a picture the way the image processor would: (patches, rows, cols).

    The tiles are the processor's own ``crop_image_to_patches`` output -- the image is
    reshaped to the closest supported tile aspect and cut into ``tile_size`` squares, each
    of which IS the compiled grid -- and the thumbnail is re-encoded at that grid instead
    of the variable one the processor would have chosen. ``rows``/``cols`` are the
    processor's own (they name the ``<|img_row_r_col_c|>`` markers).

    Returns ``(1, 1)`` and the whole picture when the processor would not have split it.
    """
    vision = config.vision
    if not vision.tiling_supported:
        raise ValueError(
            f"this artifact is compiled for a {vision.image_geometry} grid, and a tile is "
            f"{vision.tile_size}x{vision.tile_size}: a tile has to be exactly one compiled "
            "dispatch, so tiling needs an artifact built at that square"
        )
    if not vision.use_thumbnail:
        raise NotImplementedError(
            "use_thumbnail=False is not supported: the image processor counts an unsplit "
            "picture's placeholders as thumbnail tokens, so dropping the thumbnail also "
            "drops them"
        )
    image = load_image(image)
    fast = processor.image_processor
    width, height = image.size
    too_large = fast._is_image_too_large(
        height=height,
        width=width,
        max_image_tokens=vision.num_merged,
        encoder_patch_size=vision.patch_size,
        downsample_factor=vision.downsample_factor,
        max_pixels_tolerance=fast.max_pixels_tolerance,
    )
    if not too_large:
        return encode_thumbnail(processor, config, image), 1, 1

    # Every knob is passed explicitly, splitting included: an artifact's saved image
    # processor is pinned to the UNSPLIT path (so a stock `processor(...)` call lands on
    # the compiled grid), and this is the one call that must override that.
    batch = fast(
        [image],
        return_tensors="pt",
        return_row_col_info=True,
        do_image_splitting=True,
        min_image_tokens=vision.num_merged,
        max_image_tokens=vision.num_merged,
        tile_size=vision.tile_size,
        min_tiles=vision.min_tiles,
        max_tiles=vision.max_tiles,
    )
    rows, cols = int(batch["image_rows"][0]), int(batch["image_cols"][0])
    if rows * cols == 1:  # the processor declined to split after all
        return encode_thumbnail(processor, config, image), 1, 1
    # The tiles come first and the thumbnail last; only the thumbnail is re-encoded.
    tiles = _unpad(batch, config, range(rows * cols))
    return np.concatenate([tiles, encode_thumbnail(processor, config, image)], axis=0), rows, cols


def _expand_placeholders(processor, config: Lfm2VlConfig, prompt: str, images, layouts):
    """The prompt with each ``<image>`` expanded, by the processor's own expansion.

    Row/column markers, the thumbnail token and every placeholder count are HuggingFace's;
    only ``image_sizes`` is ours, pinned to the compiled square so the thumbnail's token
    count is the one the tower actually produces.
    """
    vision = config.vision
    height, width = vision.image_shape
    return processor.expand_text_with_placeholders(
        [prompt],
        [list(images)],
        image_rows=[rows for rows, _ in layouts],
        image_cols=[cols for _, cols in layouts],
        image_sizes=[[height, width]] * len(layouts),
        use_image_special_tokens=True,
        tile_size=vision.tile_size,
        downsample_factor=vision.downsample_factor,
        encoder_patch_size=vision.patch_size,
        use_thumbnail=True,
    )[0]


def render(
    processor, config: Lfm2VlConfig, messages: list[dict], images=None, *, split=None
) -> VisionInputs:
    """Render a conversation (+ its images) into ids and flattened patches.

    ``messages`` is the chat-template form: ``[{"role": ..., "content": [{"type":
    "image"}, {"type": "text", "text": ...}]}, ...]``.

    ``split`` overrides the artifact's ``do_image_splitting``: True tiles a picture the
    processor considers too large for one frame, False always encodes it whole (one
    dispatch, the cheapest and the least detailed).
    """
    images = [load_image(image) for image in (images or [])]
    tiling = config.vision.do_image_splitting if split is None else bool(split)
    patches, layouts = [], []
    for image in images:
        if tiling:
            rows_patches, rows, cols = tile_image(processor, config, image)
        else:
            rows_patches, rows, cols = encode_thumbnail(processor, config, image), 1, 1
        patches.append(rows_patches)
        layouts.append((rows, cols))

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # The template renders each {"type": "image"} entry as a bare image token, which only
    # means anything once it has been expanded into a placeholder run. A count that does
    # not match the pictures would silently leave one un-expanded (or pair the wrong
    # picture with the wrong marker), so it is refused here, as the processor refuses it.
    markers = prompt.count(processor.image_token)
    if markers != len(images):
        raise ValueError(
            f"the prompt has {markers} image marker(s) but {len(images)} picture(s) were "
            "given; there must be one {'type': 'image'} entry per picture"
        )
    if images:
        prompt = _expand_placeholders(processor, config, prompt, images, layouts)
    input_ids = processor.tokenizer(prompt, add_special_tokens=False).input_ids
    return VisionInputs(
        input_ids=np.asarray(input_ids, dtype=np.int64),
        patches=np.concatenate(patches, axis=0) if patches else None,
        layouts=layouts,
    )


def build_prompt(
    config: Lfm2VlConfig, inputs: VisionInputs, image_embeds: np.ndarray | None
) -> Lfm2VlPrompt:
    """Bundle the prompt's image-embedding placement for the decoder."""
    ids = np.asarray(inputs.input_ids).reshape(-1)
    positions = np.flatnonzero(ids == config.image_token_id)
    if image_embeds is not None and len(positions) != len(image_embeds):
        raise ValueError(
            f"{len(positions)} image placeholder tokens but {len(image_embeds)} image "
            "embeddings -- the compiled grid and the prompt disagree"
        )
    return Lfm2VlPrompt(input_ids=ids, image_positions=positions, image_embeds=image_embeds)


#: The vision special tokens ``Lfm2VlProcessor`` reads off its tokenizer, and the names
#: the LFM2-VL family gives them. transformers v5 carries these as the tokenizer's
#: model_specific_special_tokens; a tokenizer loaded any other way has to be told.
_IMAGE_TOKENS = {
    "image_token": "<image>",
    "image_start_token": "<|image_start|>",
    "image_end_token": "<|image_end|>",
    "image_thumbnail": "<|img_thumbnail|>",
}


def load_processor(repo_id: str, **image_processor_kwargs):
    """The processor for a hub repo, a checkpoint directory, or a built artifact.

    Assembled rather than taken from ``AutoProcessor`` for two reasons, both about the
    tokenizer: the LFM2.5 checkpoints name a ``tokenizer_class`` ("TokenizersBackend")
    that transformers releases before v5 do not have, which makes ``AutoProcessor`` refuse
    the whole processor; and whichever way the tokenizer arrives it has to carry the four
    vision special tokens the processor reads off it, which a plain fast tokenizer does
    not restore. ``load_tokenizer`` handles the first, this handles the second, and the
    two halves are put together here -- so a repo and an artifact load by the same path.
    """
    import json

    from transformers import AutoImageProcessor
    from transformers.models.lfm2_vl.processing_lfm2_vl import Lfm2VlProcessor

    from ...checkpoint import repo_file
    from ...generate import load_tokenizer

    tokenizer = load_tokenizer(repo_id)
    config_path = repo_file(repo_id, "tokenizer_config.json", optional=True)
    tokenizer_config = json.load(open(config_path)) if config_path else {}
    for attribute, default in _IMAGE_TOKENS.items():
        setattr(tokenizer, attribute, tokenizer_config.get(attribute, default))
    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
    return Lfm2VlProcessor(
        image_processor=AutoImageProcessor.from_pretrained(repo_id, **image_processor_kwargs),
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
    )


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------

#: The default calibration set: DocVQA (document photographs + real questions about
#: them). Read one parquet shard at a time, and only as many as the sample count needs.
DOCVQA = ("lmms-lab-encoder/DocVQA", "DocVQA", "validation")


def docvqa_calibration(num_samples: int = 48, *, source=DOCVQA) -> tuple[list, list[str]]:
    """(images, questions) from DocVQA -- what calibration uses when given nothing else.

    Real (image, question) pairs, because the activation scales should be set from what
    the deployed model actually sees. Pass your own to :func:`multimodal_calibration`.

    The shards are read directly rather than through ``load_dataset(repo, config)``: that
    route resolves the corpus through the hub's dataset server, which hangs on this one.
    A shard is ~900 rows, so any sane calibration set comes out of the first.
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
    processor, config: Lfm2VlConfig, images=None, prompts=None, *, num_samples: int = 48
) -> list[dict]:
    """Calibration samples for the joint PTQ pass: real (image, question) pairs.

    Both towers are calibrated together -- the decoder's activation scales have to see
    the image embeddings the vision tower actually produces, and the encoder's
    LayerNorm/GELU ranges are nothing like a random tensor's.

    Calibration always uses the UNSPLIT path (one dispatch per picture): the tower is the
    same for a tile and for a whole picture, so tiling would only multiply the sample
    count by the tile count.

    ``images`` may be PIL images, paths, or URLs (fetched through the image cache); omit
    them and ``num_samples`` DocVQA pairs are used.
    """
    if images is None:
        images, docvqa_prompts = docvqa_calibration(num_samples)
        prompts = prompts if prompts is not None else docvqa_prompts
    samples = []
    for image, prompt in zip(images, prompts):
        item = render(
            processor,
            config,
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            [load_image(image)],
            split=False,
        )
        samples.append(
            {
                "input_ids": torch.as_tensor(item.input_ids)[None],
                "patches": torch.as_tensor(item.patches),
            }
        )
    return samples


#: Shared with the other two VLM builds, which move samples and resolve a calibration device
#: for the same reasons -- three copies is how they drift.
from ...calibration import (  # noqa: E402
    calibration_device,
    require_vision_samples,
    to_device,
)


def apply_calibration_method(model, calib_method: str, percentile: float = 99.99) -> int:
    """Point the VISION tower's affine activation observers at ``calib_method``.

    Percentile clipping is the ViT fix DINOv2's loader already ships (models/dino/model.py).
    SigLIP2's LayerNorm outputs carry heavy per-channel outliers, and min/max is monotone in
    sample count, so more calibration data only widens every scale. Measured on the released
    towers at 512px: device-vs-torch cosine 0.9550 -> 0.9612 (450M, 48 samples) and
    0.9832 -> 0.9852 (3B, 12 samples); and because tighter ranges give the PWL LUT chains finer
    resolution over the range they are fit on, the pure-torch LUT error falls with it
    (0.9856 -> 0.9915 on the 3B).

    ``model.visual`` ONLY, deliberately. The same clip applied to the decoder costs accuracy:
    on the 450M its logits go from cosine 0.9448 / top-1 0.551 against fp32 to 0.8944 / 0.343.
    A ViT's activation tails are outliers; an LM head's tails are the answer.

    Returns the number of observers changed (0 for "minmax", which is the toolkit default).
    """
    if calib_method == "minmax":
        return 0
    from ...quant import set_calibration_method

    visual = getattr(model, "visual", model)
    n_obs = set_calibration_method(visual, calib_method, percentile=percentile)
    logger.info(
        "%s clipping (p=%s) on %d vision activation observers", calib_method, percentile, n_obs
    )
    return n_obs


def build_artifacts(
    repo_id: str,
    out_dir,
    *,
    calibration=None,
    num_samples: int = 48,
    image_size: int | tuple[int, int] = 512,
    fit_mode: str = "stretch",
    lut_chains: tuple[int, int, int] | None = None,
    context_len: int = 2048,
    chunk_len: int | None = None,
    target: str = "device",
    num_cores: int = 1,
    generation: str = DEFAULT_GENERATION,
    smooth_alpha: float = 0.8,
    qk_alpha: float = 0.5,
    calib_method: str = "percentile",
    percentile: float = 99.99,
    device: str = "cpu",
) -> dict:
    """Quantize once, then compile both towers into ``out_dir``.

    Returns ``{"vision": dir, "text": dir, "bundle": dir}``.

    ``calibration`` is a sample list from :func:`multimodal_calibration`; omitted, it is
    ``num_samples`` DocVQA pairs at this repo's geometry.

    ``image_size`` is the geometry the vision tower is pinned to: a square side, or an
    (height, width) pair. One artifact serves exactly ONE geometry -- the patch count, the
    resampled position grid and the attention mask are all baked at it -- so a second
    resolution is a second call. Tiling is not: every tile is this same geometry, which is
    why the default (512, the checkpoint's own tile size) is the one that can tile.

    ``fit_mode`` ("stretch" | "letterbox") is how non-square pictures are made to fit it;
    it is recorded in the bundle, so inference reproduces the geometry calibration saw.

    ``calib_method`` ("percentile" | "minmax") is how the VISION tower's affine activation
    observers pick their clip. Percentile (the default, p=99.99) is the ViT-standard fix for
    the heavy per-channel outliers SigLIP2's LayerNorms produce; "minmax" restores the old
    behaviour and is monotone in ``num_samples``, so it gets worse as calibration grows. The
    decoder stays on min/max either way -- percentile clipping measurably hurts its logits.

    ``lut_chains`` is (sdpa, gelu, norm): how many hardware LUTs each nonlinearity's
    additive chain gets on device. It has to be set before the reference is built -- the
    chain length is a constructor argument of the modules that model it -- so it goes in
    here rather than onto the loaded model, and the artifact records it with the rest of
    the vision config. None keeps the checkpoint's defaults.

    ``chunk_len`` compiles the prefill graph for that many query positions instead of the
    whole ``context_len``, and the host then ingests a prompt in chunks. The prefill graph
    costs the same whatever the prompt's length, so a short prompt on a long-context
    artifact pays for every position it does not use; chunking is what removes that. Keep
    it SMALL: the last, partial chunk is replayed token by token through the decode graph
    (a padded chunk would leave pad in the conv mixer's carried window), so a prompt
    shorter than one chunk gets no prefill at all. See the README for the measured
    trade-off.

    ``generation`` ("OMEGA" | "EUROPA") goes to both towers: they share a device, so they
    must agree. ``device`` is the **torch** device the reference calibrates on ("cpu",
    "cuda:1", ...), distinct from that target -- calibration is the bulk of a build, so a
    GPU is worth having. The model comes back to the host before export, which reads numpy.
    """
    from ...generate import GenerationConfig, build_quantized_model
    from ...implementation import get_implementation
    from .lfm2vl import Lfm2VlForConditionalGeneration, quantize_model
    from .vision_impl import build_vision_graph

    out_dir = Path(out_dir)
    calib_device = calibration_device(device)
    if fit_mode not in ("letterbox", "stretch"):
        raise ValueError(f"fit_mode must be 'letterbox' or 'stretch', not {fit_mode!r}")

    # Said BEFORE the work, not after it: an artifact carries its generation into its ELF and
    # the runtime refuses a mismatch, so a build for the part this machine does not have is
    # minutes spent producing something that cannot be run here. A warning rather than a
    # refusal -- building for a part you will run on elsewhere is a real thing to do.
    target_generation = supported_generation(generation)
    attached = attached_generation()
    if attached and attached != target_generation:
        logger.warning(
            "building for %s (%s), but the board attached here is %s (%s) -- the runtime "
            "refuses a generation mismatch, so this artifact will not run on this machine",
            target_generation,
            device_name(target_generation),
            attached,
            device_name(attached),
        )

    # quantize_attention=True: the decoder layer kernel binds calibrated SDPA scales.
    chains = dict(zip(("sdpa_n_luts", "gelu_n_luts", "norm_n_luts"), lut_chains or ()))
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        repo_id,
        image_size=image_size,
        device=calib_device,
        quantize_attention=True,
        vision_overrides=chains or None,
    )
    # Set before calibration and carried in the bundle: the scales are set on the same
    # geometry inference will produce, and the runtime reads back what was built.
    model.config.vision.fit_mode = fit_mode
    if calibration is None:
        calibration = multimodal_calibration(
            load_processor(repo_id), model.config, num_samples=num_samples
        )
    require_vision_samples(calibration, "patches", "lfm2vl.pipeline.multimodal_calibration")
    calibration = [to_device(sample, calib_device) for sample in calibration]
    logger.info(
        "calibrating %s on %d multimodal samples (%s)", repo_id, len(calibration), calib_device
    )
    # Timed and announced: on a CPU host this is most of the build, and in silence there
    # is no telling a slow calibration from a hung one.
    started = perf_counter()
    # Percentile clipping of the VISION tower's affine activation observers -- the ViT fix
    # DINOv2's loader already ships (models/dino/model.py). SigLIP2's LayerNorm outputs carry
    # heavy per-channel outliers, and min/max is monotone in sample count, so more calibration
    # data only widens every scale. Measured on the released towers at 512px: device-vs-torch
    # cosine 0.9550 -> 0.9612 (450M, 48 samples) and 0.9832 -> 0.9852 (3B, 12 samples), and
    # because tighter ranges give the PWL LUT chains finer resolution over the range they are
    # fit on, the pure-torch LUT error falls with it (0.9856 -> 0.9915 on the 3B).
    #
    # model.visual ONLY, deliberately. The same clip applied to the decoder costs accuracy:
    # on the 450M its logits go from cosine 0.9448 / top-1 0.551 against fp32 to 0.8944 / 0.343.
    # A ViT's activation tails are outliers; an LM head's tails are the answer.
    apply_calibration_method(model, calib_method, percentile)
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

    bundle = build_quantized_model("lfm2vl", model)
    bundle.save(out_dir / "quantized_model")
    text_config = GenerationConfig(
        implementation="lfm2vl-metis-opt",
        repo_id=repo_id,
        artifact_dir=out_dir / "text",
        context_len=context_len,
        chunk_len=chunk_len,
        # A chunked build carries BOTH prefill graphs -- the chunk-sized one and a
        # full-context one -- so the host can pick whichever is cheaper for the prompt in
        # front of it. Costs two more codegen units; buys a guarantee that a chunked
        # artifact is never slower than a plain one. See text_impl.plan_prefill.
        phases=(("prefill", FULL_PREFILL_GRAPH, "decode") if chunk_len else ("prefill", "decode")),
        num_cores=num_cores,
        generation=supported_generation(generation),
        target=target,
        device="cpu",  # the reference is back on the host by now
    )
    text_dir = get_implementation("lfm2vl-metis-opt").compile(bundle, text_config)

    from axelera.compiler.alto.compiler.config import Target

    vision_dir, _ = build_vision_graph(
        model.visual,
        out_dir / "vision",
        target=Target.DEVICE if target == "device" else Target.AICORE_SIM,
        cores=num_cores,
        generation=generation,
    )
    # Last, so a half-built directory is not mistaken for a loadable one.
    save_hf_metadata(out_dir, repo_id, model.config.vision)
    return {
        "vision": Path(vision_dir),
        "text": Path(text_dir),
        "bundle": out_dir / "quantized_model",
    }


# --------------------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------------------


def weight_gb(directory: Path) -> float:
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
    """The artifact directory itself if :func:`save_hf_metadata` wrote HF files into it.

    One check ("is there a config.json?") shared by everything that would otherwise reach
    for the hub.
    """
    path = Path(artifact_dir)
    return path if (path / "config.json").exists() else None


def save_hf_metadata(out_dir, repo_id: str, vision) -> None:
    """Write the HuggingFace-side files into a built artifact.

    What makes the artifact directory loadable through the stock Auto classes
    (``AutoConfig`` / ``AutoProcessor`` / ``AutoModelForImageTextToText``.from_pretrained)
    with no hub and no network.

    The saved image processor is pinned to the compiled geometry's token count and to the
    unsplit path, so a picture that has been through ``fit_images`` comes out at exactly
    the grid the tower encodes. Tiling asks for a different prompt shape than a plain
    ``processor(...)`` call produces, so it goes through the pipeline instead (see
    :func:`tile_image`).
    """
    import json

    from transformers import GenerationConfig as HFGenerationConfig

    from .hf_model import artifact_config

    out_dir = Path(out_dir)
    processor = load_processor(
        repo_id,
        do_image_splitting=False,
        min_image_tokens=vision.num_merged,
        max_image_tokens=vision.num_merged,
        tile_size=vision.tile_size,
    )
    processor.save_pretrained(out_dir)
    # save_pretrained writes the tokenizer as a plain fast one, which loses the vision
    # special tokens the processor reads off it. Record them, so a reload of THIS
    # directory recovers the checkpoint's own names rather than falling back to defaults.
    tokenizer_config_path = out_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        tokenizer_config.update(
            {name: getattr(processor.tokenizer, name) for name in _IMAGE_TOKENS}
        )
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2))
    artifact_config(repo_id).save_pretrained(out_dir)
    # Sampling defaults travel with the artifact too, so a script that never mentions the
    # hub repo still behaves like the model it was built from. The model card's own
    # recommendation for this size is temperature=0.1, min_p=0.15, repetition_penalty=1.05.
    HFGenerationConfig.from_pretrained(repo_id).save_pretrained(out_dir)


def add_artifact_args(parser, *, log_level: str = "WARNING"):
    """The flags every tool here needs to open a built artifact."""
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument(
        "--repo-id", default=None, help="processor/tokenizer repo (default: the artifact's)"
    )
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


def artifact_generation(vision_dir) -> str | None:
    """The generation a compiled vision tower recorded, or None if it recorded none."""
    import json

    meta = Path(vision_dir) / "meta.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text()).get("generation")


def graph_cores(model_json) -> int:
    """AI cores a compiled graph places work on: one call graph per core.

    Read off the artifact, so nothing has to remember what it was built with.
    """
    import json

    spec = json.loads(Path(model_json).read_text())
    graphs = spec.get("graphs", {}).values()
    return max((len(g.get("call_graphs", [])) for g in graphs), default=1) or 1


def image_token_cost(config: Lfm2VlConfig, rows: int = 1, cols: int = 1) -> int:
    """Prompt tokens one picture costs at a given tile layout, markers included.

    An unsplit picture is ``image_start`` + its placeholders + ``image_end``. A split one
    adds a ``<|img_row_r_col_c|>`` marker per tile and a thumbnail (its own token plus a
    whole grid's placeholders), which is what makes tiling a context decision as much as a
    latency one -- the report's accuracy/latency knob, priced.
    """
    merged = config.vision.num_merged
    if rows * cols == 1:
        return merged + 2
    return rows * cols * (merged + 1) + merged + 3


class Lfm2VlPipeline:
    """A loaded LFM2.5-VL: both compiled towers plus the host glue between them."""

    def __init__(
        self, artifacts: dict, repo_id: str, *, generation: str | None = None, device=None
    ):
        import json

        from ...generate import RuntimeModel
        from ...implementation import get_implementation
        from .vision_impl import Lfm2VlVisionDevice

        text_dir = Path(artifacts["text"])
        self.repo_id = repo_id
        # The artifact's own processor: the chat template and patch geometry it was
        # calibrated and compiled against, loadable with no hub and no network. The repo
        # id is the fallback and the override.
        self.processor = load_processor(hf_metadata_dir(text_dir.parent) or repo_id)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.pipeline = json.loads((text_dir / "pipeline.json").read_text())
        self.host_params = dict(np.load(text_dir / self.pipeline["host_params"]))
        self.impl: Lfm2VlTextImplementation = get_implementation(self.pipeline["implementation"])
        # The artifact's own record wins; `generation` is for towers built before it was
        # written, and a mismatch is a mistake worth saying out loud.
        recorded = artifact_generation(artifacts["vision"])
        if recorded and generation and supported_generation(generation) != recorded.upper():
            raise ValueError(
                f"this artifact was compiled for {recorded.upper()} but "
                f"generation={str(generation).upper()} was asked for; drop the flag to use "
                "the artifact's own"
            )
        self.generation = supported_generation(recorded or generation or DEFAULT_GENERATION)
        # The connection is claimed once, for BOTH towers, and must own every core either
        # of them places work on. They are compiled with the same core count but need not
        # END UP using it: how far a kernel's shards actually spread depends on its own
        # widths, so a decoder whose widest shard is three cores can sit beside a vision
        # tower that uses four. Claiming only the decoder's share and then running the
        # tower does not fail cleanly -- the run dies inside the driver and the card needs
        # `axdevice --reload-firmware` -- so the larger of the two is what is asked for.
        vision_json = Path(artifacts["vision"]) / "model.json"
        self._cores = max(graph_cores(text_dir / "model.json"), graph_cores(vision_json))
        self.text_runtime = RuntimeModel(
            text_dir / "model.json", num_sub_devices=self._cores, device=device
        )
        # Both towers share that one connection: a second Context would claim the
        # sub-devices again and fail.
        try:
            self.vision = Lfm2VlVisionDevice.load(
                artifacts["vision"], share_from=self.text_runtime, generation=self.generation
            )
        except Exception as refusal:
            # Out of device memory, almost always: the decoder is resident by now. Hand
            # the card back before raising, or the NEXT run fails too for a reason that
            # has nothing to do with it.
            self.text_runtime.release()
            raise RuntimeError(
                f"the decoder ({weight_gb(text_dir):.2f} GB) and the vision tower "
                f"({weight_gb(artifacts['vision']):.2f} GB) do not fit this device "
                "together. They must be resident at once; run this model on a device with "
                "more memory (`axdevice` prints what this one has)."
            ) from refusal
        # The model config travels with the quantized bundle (written next to the towers).
        bundle_dir = Path(artifacts.get("bundle", text_dir.parent / "quantized_model"))
        manifest = json.loads((bundle_dir / "quantized_model.json").read_text())
        self.config = Lfm2VlConfig(**manifest["config"])

    @property
    def context_len(self) -> int:
        return int(self.pipeline["context_len"])

    @property
    def chunk_len(self) -> int | None:
        """Width of the chunked prefill graph, or None if the build has only a full one.

        Read off the artifact: which graphs exist is a property of what was built. Which of
        them a given prompt uses is decided per prompt, from this and the length -- see
        :func:`~.text_impl.plan_prefill`.
        """
        return self.pipeline.get("chunk_len") or None

    @property
    def num_cores(self) -> int:
        """AI cores the pair claims -- read off the artifacts, never asked for.

        The wider of the two towers, since one connection serves both; that is also how
        much of the card this model gets.
        """
        return int(self._cores)

    def default_config(self, **overrides):
        from ...generate import GenerationConfig

        # chunk_len has to travel with the config: it is what tells the decode loop to
        # ingest the prompt in chunks rather than in one full-context call.
        settings = {"context_len": self.context_len, "chunk_len": self.chunk_len}
        settings.update(overrides)
        return GenerationConfig(**settings)

    # --- the two towers, drivable separately ---------------------------------------

    def encode_images(self, images, *, split=None) -> np.ndarray:
        """Image(s) -> image embeddings: the vision tower alone.

        Everything comes back in image order (and, within a picture, tiles then
        thumbnail), which is the order the decoder's placeholder rows are filled in.
        """
        inputs = render(
            self.processor,
            self.config,
            [{"role": "user", "content": [{"type": "image"} for _ in images]}],
            list(images),
            split=split,
        )
        return self.vision.encode(inputs.patches)

    def prepare(self, messages: list[dict], images=None, *, split=None, vision=None):
        """Render a conversation and encode its images; returns the decoder prompt.

        ``vision`` reuses an earlier :meth:`encode_images` result, so a chat turn does not
        re-encode the same picture.
        """
        inputs = render(self.processor, self.config, messages, images, split=split)
        return self.prepare_inputs(inputs, vision)

    def prepare_inputs(self, inputs: VisionInputs, vision=None) -> Lfm2VlPrompt:
        """Preprocessed inputs -> decoder prompt: encode the vision side.

        The half of :meth:`prepare` below the processor, so a caller who ran the processor
        itself reaches the same path with the same checks.
        """
        if vision is None and inputs.patches is not None:
            vision = self.vision.encode(inputs.patches)
        return build_prompt(self.config, inputs, vision)

    def _drive(self, generate, prompt: Lfm2VlPrompt, config):
        """Run one of the implementation's generators with the prompt context set.

        The prefill graph is chosen here, per prompt: a chunked artifact carries a
        full-context graph too, and which one wins depends on the length (see
        text_impl.plan_prefill). The choice has to reach the config as well, since the
        shared decode loop reads ``chunk_len`` from it.
        """
        config = config or self.default_config()
        chunk_len = self.impl.select_prefill(
            len(prompt.input_ids),
            self.chunk_len,
            self.context_len,
            self.text_runtime.graph_names,
        )
        config = replace(config, chunk_len=chunk_len)
        self.impl.set_prompt(prompt)
        try:
            yield from generate(
                self.text_runtime,
                self.tokenizer,
                [int(t) for t in prompt.input_ids],
                config,
                host_params=self.host_params,
            )
        finally:
            self.impl.set_prompt(None)

    def stream(self, prompt: Lfm2VlPrompt, config=None):
        """Drive the decoder over a prepared prompt, yielding cumulative text."""
        yield from self._drive(self.impl.generate, prompt, config)

    def stream_token_ids(self, prompt: Lfm2VlPrompt, config=None):
        """The token-id form of :meth:`stream` (what the benchmark times)."""
        yield from self._drive(self.impl.generate_token_ids, prompt, config)
