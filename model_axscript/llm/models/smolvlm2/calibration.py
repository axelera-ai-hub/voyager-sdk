# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Multimodal calibration set for SmolVLM2 PTQ.

Produces samples consumed directly by ``SmolVLM2Reference.forward(batch)`` (so one
``quantize_model`` / ``record_lut_ranges`` pass calibrates vision + connector + text
jointly, in distribution). Two kinds, both accepted by that one forward, and both derived
from the SAME DocVQA entries -- so a budget of one entry covers both:

* ``{"input_ids", "pixel_values"}`` -- a real image with a real question, tokenized +
  tiled by the model's own HF processor (global view + up to 16 crops). Half carry the
  dataset's assistant answer too. Stills only: the video regime (untiled frames + the
  frame-timestamp scaffold) needs real clips to calibrate honestly, and is left out rather
  than approximated with stills stitched into a fake clip.
* bare ``input_ids`` tensors -- the same entries' (question, answer) rendered as a chat
  turn with the page left out: decoder only, no vision pass, ending in the eos token.

Images, questions and answers come from DocVQA (see ``DOCVQA``): photographs of real
documents with real questions about them. Document pages are the demanding regime for a
W8A8 vision tower -- PTQ fits each activation scale to what it sees, and natural
photographs alone contain no small, low-amplitude glyph detail, so scales fitted on them
spend their int8 levels on bulk image structure and quantise fine text to nothing.
Measured on the 256M: the fp reference reads the dates off a plain invoice, a variant
calibrated on photos alone misreads them. There is no offline fallback: an unreachable
source raises, because a calibration set that silently shrank (or quietly swapped in
different images) still produces an artifact -- one whose int8 scales were fitted on a
recipe nobody recorded, discovered much later as unexplained accuracy loss.

The text-only half is not optional. SmolVLM2's decoder IS a Llama, and an
image-conditioned prompt is >90% image tokens (1142 of ~1240 for one photo on the
500M), so a purely multimodal set calibrates the text trunk on a few hundred tokens of a
handful of questions. The scales it fits then don't cover the regime the model generates
in -- most sharply at the assistant turn and the eos logit, which is how a W8A8 decoder
ends up unable to stop.

That half is drawn from DocVQA too, not from the text-only family's chat mix. One source
means one download, which is what lets a budget of a single entry calibrate both regimes;
the cost is breadth -- the assistant-turn/eos scales are now fitted on short document
answers rather than on varied prose, arithmetic and code. Deliberate: this model is asked
document questions in production. Pass ``calibration=`` to the builder to substitute a
wider set.
"""

from __future__ import annotations

import io

from .processor import build_inputs, make_processor

_ROWS_PER_BATCH = 64  # parquet rows decoded at a time; a shard holds ~900

#: The default calibration corpus: DocVQA validation -- photographs of real documents,
#: each with a real question and its answers. (repo, config, split).
DOCVQA = ("lmms-lab-encoder/DocVQA", "DocVQA", "validation")


def calibration_turns(n, source=DOCVQA):
    """n (image, question, answer) triples from ``source``, in dataset order.

    ONE question per document page. DocVQA asks several questions about each scan, so
    consecutive rows repeat the same image -- taking them as they come, a budget of 8 buys
    5 distinct pages and pays 8 vision forwards for them. Since the vision half is what a
    still costs (up to 17 tile dispatches against a text sample's none), a duplicated page
    is the most expensive way to calibrate nothing new; the question diversity it gives up
    is what the text-only half of the set is for.

    The parquet shards are read directly rather than through ``load_dataset(repo, config)``
    or its streaming reader. Two reasons: the hub's dataset server hangs on this corpus
    (``lmms-lab/DocVQA`` was renamed and now redirects), and ``datasets``' streaming reader
    leaves a native thread behind that aborts the interpreter during finalisation
    (``PyGILState_Release``, exit 134) -- long after the work is done, so a complete
    artifact would be reported as a failed build. A shard is ~900 rows, so a calibration
    set of any sane size comes out of the first one, and the prefix is deterministic, so a
    rebuild at the same budget calibrates on the same samples.
    """
    from PIL import Image
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    repo, config, split = source
    fs = HfFileSystem()
    try:
        shards = sorted(fs.glob(f"datasets/{repo}/{config}/{split}-*.parquet"))
    except Exception as exc:  # noqa: BLE001 - name the source; a partial set is not usable
        raise RuntimeError(f"calibration source {repo}:{config} unavailable: {exc}") from exc
    if not shards:
        raise RuntimeError(f"no {split} shards in {repo} ({config})")

    turns, seen = [], set()
    for shard in shards:
        try:
            with fs.open(shard) as handle:
                batches = pq.ParquetFile(handle).iter_batches(
                    batch_size=_ROWS_PER_BATCH,
                    columns=["docId", "question", "image", "answers"],
                )
                for batch in batches:
                    for row in batch.to_pylist():
                        if row["docId"] in seen:  # one question per page; see above
                            continue
                        seen.add(row["docId"])
                        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                        answers = row["answers"] or []
                        # Several human answers per question; the first is as good as any,
                        # and calibration only needs ONE plausible assistant turn.
                        turns.append((image, row["question"], answers[0] if answers else None))
                        if len(turns) == n:
                            print(f"[calib] docvqa -> {len(turns)} pages", flush=True)
                            return turns
        except Exception as exc:  # noqa: BLE001 - name the source; a partial set is not usable
            raise RuntimeError(f"calibration source {shard} unreadable: {exc}") from exc
    raise RuntimeError(f"{repo} ({config}/{split}) has only {len(turns)} rows, {n} were asked for")


# How a budget is spent. It counts DocVQA ENTRIES, and every entry is spent TWICE: once as a
# still (its page and question through the vision tower) and once as text alone (the same
# question with the dataset's answer, decoder only). So a budget of 1 is one document that
# calibrates BOTH regimes, and no budget can round a regime to zero -- which is why there is no
# minimum above 1 and no kind mix to split. Note this is not a count of forwards: n entries cost
# 2n of them.
#
# Both regimes are load-bearing and one entry carries both: the still fixes the tiling geometry
# and the fine-glyph scales, the text-only pass fixes the assistant-turn / eos regime that an
# image-conditioned prompt (>90% image tokens) barely touches. They are nowhere near equal in
# COST -- a still is a vision forward over every tile (up to 17), the text pass none -- so an
# entry is ~97% still by time.
#
# ONE source for both halves, rather than the text-only family's chat mix (wikitext / gsm8k /
# humaneval / alpaca / mmlu): a DocVQA turn is what this model is actually asked in production,
# and it keeps the whole set to one download that a budget of 1 can satisfy. What that gives up
# is breadth -- the mix covered general prose, arithmetic and code -- so the text half now
# calibrates the assistant-turn/eos regime on short document answers rather than on varied
# prose. A deliberate trade, not an oversight.
#
# There is no CLIP kind. The video path (untiled frames + the Frame from MM:SS scaffold) is a
# genuinely different regime, but calibrating it needs real clips: a "clip" assembled from
# unrelated stills fits the scales to frame-to-frame statistics no video has, which is worse than
# leaving the regime out, because it looks covered. Video prompts therefore run on scales fitted
# to stills and text only.
DEFAULT_BUDGET = 26  # -> 26 stills + 26 text samples, 52 forwards per pass

MIN_BUDGET = 1  # one entry calibrates both regimes, so one is a legitimate build


def text_calibration(tokenizer, turns, window: int) -> list:
    """Text-only samples ([1, <=window] id tensors) from DocVQA ``turns``.

    The same entries the stills come from, with the page left out, so the decoder sees a real
    question and a real answer ending in the model's eos token -- the regime an
    image-conditioned prompt barely reaches. ``typed=True`` is required: SmolVLM2's chat
    template indexes ``content[0]['type']`` and silently drops plain-string content (see
    llm/calibration.chat_ids).

    A turn whose answer is missing is skipped, and so is one that renders to fewer than 8
    tokens (the floor ``build_calibration`` uses): the assistant half is the whole point here,
    and a user turn on its own calibrates neither the answer nor the eos position.
    """
    import torch

    # The one renderer for a (user, assistant) turn, shared with the text-only models so both
    # families calibrate on identically-templated chat rather than on two spellings of it.
    from ...calibration import chat_ids

    samples: list = []
    for _image, question, answer in turns:
        if not answer:
            continue
        ids = chat_ids(tokenizer, question, answer, typed=True)[:window]
        if len(ids) >= 8:
            samples.append(torch.tensor([ids]))
    return samples


def _sample(enc) -> dict:
    """One calibration sample from processor output. The pixel stack is flattened to
    (blocks, 3, H, W) -- tiles for a still, frames for a clip; the reference encodes one
    block per vision pass either way."""
    return {
        "input_ids": enc["input_ids"],
        "pixel_values": enc["pixel_values"].reshape(-1, *enc["pixel_values"].shape[-3:]),
    }


def multimodal_calibration(
    repo_id: str,
    *,
    n: int = DEFAULT_BUDGET,
    n_still: int | None = None,
    n_text: int | None = None,
    window: int | None = None,
) -> list:
    """Calibration samples from ``n`` DocVQA entries, each spent as a still AND as text.

    ``n`` counts dataset ENTRIES, not forwards: n entries produce n still samples plus up to n
    text-only samples, so 2n forwards per calibration pass. That is what makes ``n=1`` a
    legitimate build -- one document calibrates both regimes -- where a budget counted in
    forwards had to reserve one per kind and could not go below two.

    ``n_still`` / ``n_text`` cap one regime for callers that want the other alone. The entries
    are fetched once and shared, so the same pages back both halves.

    Everything is tokenized by the model's own HF processor, so pixel normalization and
    ``<image>`` expansion match inference exactly. Every other still stops at the generation
    prompt, the way inference starts; the rest continue into the dataset's answer, so the
    quantizer sees an assistant-turn activation with the page attached as well as without it.

    Stills only on the vision side. Video prompts (untiled frames plus the frame-timestamp
    scaffold) are a different regime, and it is NOT calibrated here -- covering it honestly needs
    real clips, and stitching unrelated stills into a fake one fits the scales to frame-to-frame
    statistics no video has while making the gap invisible.
    """
    still = n if n_still is None else n_still
    text_count = n if n_text is None else n_text
    entries = max(still, text_count)
    if entries < MIN_BUDGET:
        raise ValueError(
            f"calibration budget {entries} leaves both regimes uncalibrated; "
            f"pass at least {MIN_BUDGET} (one DocVQA entry calibrates both)"
        )
    print(
        f"[calib] {entries} docvqa entries -> {still} images + {text_count} text samples, "
        f"{still + text_count} forwards per pass",
        flush=True,
    )
    processor = make_processor(repo_id)
    turns = calibration_turns(entries)

    samples: list = []
    for i, (image, question, answer) in enumerate(turns[:still]):
        enc = build_inputs(processor, image, question, answer=answer if i % 2 else None)
        samples.append(_sample(enc))
    if window is None:
        from ...generate import GenerationConfig

        window = GenerationConfig().calib_window
    text = text_calibration(processor.tokenizer, turns[:text_count], window)
    # What was actually built, which can fall short of what was asked for -- a row with no
    # image is skipped upstream, and a turn with no answer carries no text sample.
    print(
        f"[calib] calibrating on {len(samples)} images and {len(text)} text samples "
        f"-- {len(samples) + len(text)} forwards per pass",
        flush=True,
    )
    return samples + text
