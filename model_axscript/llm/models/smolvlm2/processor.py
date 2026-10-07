# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""HF ``AutoProcessor`` wrapper for SmolVLM2: image tiling + normalization ->
``pixel_values``, and chat template + ``<image>`` expansion -> ``input_ids``.

Thin wrapper so the host side (splice + int8 pixel quant) stays independent of the
exact HF preprocessing. Image int8 quantize + space-to-depth im2col reuse the dino
host pipeline (see ``vision_impl._quantize_pixels``); this module only produces the
fp ``pixel_values`` and the tokenized prompt.
"""

from __future__ import annotations

import os
from pathlib import Path
import urllib.request

# The clip loader is the FAMILY's, two levels up: LFM2.5-VL reads the same containers, and
# the pyav frame-count repair is too subtle to keep two copies of. Re-exported here because
# this module is the media entry point every SmolVLM2 caller already imports from.
from ...shared.video import (  # noqa: F401
    _number,
    _USER_AGENT,
    IMAGE_SUFFIXES,
    is_video,
    load_video_frames,
    media_suffix,
    video_cache_dir,
    VIDEO_SUFFIXES,
)


def resolve_image_source(src: str) -> "str | Path":
    """An http(s) URL unchanged, or a local path with ~ and $VARS expanded.

    Local paths are the common case for a real image on this machine; expanding here
    means ``--image ~/photos/cat.jpg`` works from any shell, including the ones that do
    not expand ~ inside a quoted argument.
    """
    if src.startswith(("http://", "https://")):
        return src
    path = Path(os.path.expandvars(src)).expanduser()
    if not path.is_file():
        raise SystemExit(
            f"--image: no such file: {path}\n"
            f"Pass a path to an image on this machine, or an http(s) URL."
        )
    return path


def load_image(src: str):
    """Load a PIL RGB image from a local path on this machine or an http(s) URL."""
    from PIL import Image

    resolved = resolve_image_source(src)
    if isinstance(resolved, str):  # URL
        import io

        req = urllib.request.Request(resolved, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req) as r:  # noqa: S310 - user-supplied test image
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(resolved).convert("RGB")


def make_processor(repo_id: str, *, do_image_splitting: bool | None = None):
    """Splitting follows the INPUT KIND, not user preference: ``True`` for stills, ``False``
    for video frames (HF's video scaffold gives each frame exactly one block, and the frame
    budget is computed against that); ``None`` keeps the
    checkpoint default, which splits.

    ``True`` tiles into a global view + high-res crops (up to 4x4 + 1 = 17 blocks), costing
    ``n_tiles * image_seq_len`` prompt tokens -- 1142 on the 500M, 1431 on the 2.2B for a
    500x375 photo -- and one vision dispatch per tile. That is where fine detail comes from.
    ``False`` feeds ONE tile: right for video (HF's scaffold gives each frame one block) and
    wrong for stills. The per-tile shape is identical either way, so the compiled graph is
    unchanged; only the dispatch count and token cost move."""
    from transformers import AutoProcessor

    kwargs = {} if do_image_splitting is None else {"do_image_splitting": do_image_splitting}
    return AutoProcessor.from_pretrained(repo_id, **kwargs)


def build_inputs(
    processor, image, prompt: str, *, system: str | None = None, answer: str | None = None
):
    """Apply the chat template with one image + a text prompt and return the HF
    processor outputs (``input_ids``, ``attention_mask``, ``pixel_values`` with the
    ``<image>`` tokens already expanded to the per-tile token count).

    ``answer`` appends a completed assistant turn instead of the generation prompt, so the
    ids end in ``<end_of_utterance>``. Inference wants the generation prompt (answer=None);
    CALIBRATION wants both regimes -- a prompt-only sample never shows the quantizer an
    assistant-turn activation or the eos logit, which is how a W8A8 decoder ends up unable
    to stop (see llm/calibration.py)."""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append(
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    )
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    text = processor.apply_chat_template(messages, add_generation_prompt=answer is None)
    return processor(text=text, images=[image], return_tensors="pt")


def build_video_turn(
    processor,
    frames,
    metadata,
    prompt: str,
    *,
    system: str | None = None,
    answer: str | None = None,
):
    """Tokenize a VIDEO turn the way SmolVLM2 was TRAINED, via HF's own expansion.

    The model's chat_template.json has no video branch, so ``apply_chat_template``
    alone can only produce a bag of unordered ``<image>`` entries. The real scaffold
    lives in ``SmolVLMProcessor.expand_text_with_video_tokens``: a ``<video>`` token
    expands to an intro naming the frame count and duration, then per frame a
    ``\\nFrame from MM:SS:`` label wrapped in the fake-image/global-image tokens the
    checkpoint expects. We emit ``<video>`` and let the processor expand it, so the
    prompt matches training exactly rather than approximating it.

    Note ``videos``/``video_metadata`` are nested per text sample: [[frames]].

    ``answer`` closes the turn instead of asking for a generation, for calibration -- see
    build_inputs.
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": processor.video_token + prompt}],
        }
    )
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    text = processor.apply_chat_template(messages, add_generation_prompt=answer is None)
    return processor(
        text=[text], videos=[[frames]], video_metadata=[[metadata]], return_tensors="pt"
    )
