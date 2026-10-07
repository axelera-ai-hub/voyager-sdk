# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Diverse, chat-aware calibration set, mirroring the production quantizer.

W8A8 activation scales are set from the calibration data, so calibration must cover
the deployment distribution. Raw text alone leaves the chat / assistant-turn regime
out-of-distribution, which mis-quantises the end-of-turn (eos) logit -- the model
then never stops. A diverse chat-templated mix (math, code, instructions, knowledge)
plus raw text gives robust scales. Sources mirror production: WikiText (raw) +
GSM8K / HumanEval / Alpaca / MMLU (chat-templated).
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_HUMANEVAL_INSTR = "Complete the following Python function:\n\n"
_LETTERS = "ABCDEFGHIJ"


def _fmt_gsm8k(row) -> tuple[str, str]:
    return row["question"], row["answer"]


def _fmt_humaneval(row) -> tuple[str, str]:
    return _HUMANEVAL_INSTR + row["prompt"], row["canonical_solution"]


def _fmt_alpaca(row) -> tuple[str, str]:
    user = row["instruction"]
    if row.get("input", "").strip():
        user += "\n\n" + row["input"]
    return user, row["output"]


def _fmt_mmlu(row) -> tuple[str, str]:
    choices = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    answer = int(row["answer"])
    return f"{row['question']}\n{choices}", f"{_LETTERS[answer]}. {row['choices'][answer]}"


# name -> (path, config, split, formatter). formatter=None means raw text (no chat).
_SOURCES: dict[str, tuple] = {
    "wikitext": ("wikitext", "wikitext-2-raw-v1", "train", None),
    "gsm8k": ("gsm8k", "main", "train", _fmt_gsm8k),
    "humaneval": ("openai_humaneval", None, "test", _fmt_humaneval),
    "alpaca": ("tatsu-lab/alpaca", None, "train", _fmt_alpaca),
    "mmlu": ("cais/mmlu", "all", "validation", _fmt_mmlu),
}


_TEXT_SUFFIXES = (".txt", ".md", ".jsonl")


def _local_text(source: str) -> str | None:
    """The concatenated text of a local file or directory, or None when `source` is not a path.

    Calibration data is domain data, so a customer's corpus is a file on disk far more often
    than a hub dataset. A directory is read recursively, sorted, so two builds of the same
    corpus observe the same scales. A `.jsonl` file contributes its `text` field per line.
    """
    import json

    path = Path(source).expanduser()
    if not path.exists():
        return None
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.rglob("*") if p.suffix.lower() in _TEXT_SUFFIXES)
    )
    chunks = []
    for one in files:
        if one.suffix.lower() == ".jsonl":
            for line in one.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(str(json.loads(line)["text"]))
                except (ValueError, KeyError, TypeError):
                    continue
        else:
            chunks.append(one.read_text(encoding="utf-8", errors="replace"))
    text = "\n\n".join(c for c in chunks if c.strip())
    if not text:
        raise RuntimeError(f"no calibration text read from {path}")
    return text


def to_device(sample, device):
    """Move a calibration sample's tensors to ``device``; non-tensors pass through.

    Samples are nested dicts/lists of tensors plus plain metadata (grids, ids), so this
    recurses rather than assuming a flat mapping. The model and its samples have to agree
    on a device or the first forward fails on the mismatch.
    """
    import torch

    if torch.is_tensor(sample):
        return sample.to(device)
    if isinstance(sample, dict):
        return {k: to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, (list, tuple)):
        return type(sample)(to_device(v, device) for v in sample)
    return sample


def require_vision_samples(calib, key: str, source: str) -> None:
    """Refuse a VLM calibration set that carries no images.

    All three VLM references accept a bare tensor as a text-only sample, so a set of
    those calibrates the DECODER alone: the vision tower is never forwarded, its LUT
    ranges and activation scales stay unrecorded, and the build dies stages later with
    no table to bind. ``key`` is the model's pixel field -- pixel_values for SmolVLM2,
    patches for the two patch-fed towers.
    """
    if not any(isinstance(s, dict) and s.get(key) is not None for s in calib):
        raise ValueError(
            f"calibration set has {len(calib)} sample(s) and not one carries {key!r}. A VLM "
            f"needs image+text samples: text alone never reaches the vision tower, so its "
            f"LUT ranges and activation scales stay unrecorded. Pass dicts carrying {key!r} "
            f"(see {source}), or leave calibration=None for the default set."
        )


def calibration_device(requested):
    """Resolve a requested calibration device, refusing one torch cannot provide.

    A CUDA device that is not there must fail HERE rather than be quietly downgraded: the
    forwards are the bulk of a build, so a silent fallback bills the user the full CPU cost
    while they believe a GPU is in use.
    """
    import torch

    device = torch.device(requested or "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"calibration device {requested!r} requested but torch sees no CUDA")
    return device


def distribute_budget(budget: int, mix) -> list[tuple[str, int]]:
    """Split a total sample budget across (name, weight) sources, proportional to weight.

    Each per-source count is capped at the source's dataset size downstream (in
    build_calibration), so naturally-small sources (e.g. HumanEval's 164) just take
    what they have and the rest keep their proportional share.
    """
    total = sum(w for _, w in mix) or 1
    return [(name, max(1, round(budget * w / total))) for name, w in mix]


def chat_ids(tokenizer, user: str, assistant: str, *, typed: bool = False) -> list[int]:
    """Render a (user, assistant) turn to ids, ending in the model's turn/eos token.

    Public because it is a real seam: a VLM that already holds its (question, answer) turns
    calibrates its text regime through this, not through ``build_calibration`` -- that one
    fetches from NAMED sources, which is a different input entirely (see
    ``models/smolvlm2/calibration.text_calibration``).

    typed wraps each content in the multimodal ``[{"type": "text", "text": ...}]`` form.
    Required by templates that index ``content[0]['type']`` to decide text vs image (every
    VLM checkpoint, SmolVLM2 included): handed a plain string those iterate it CHARACTER by
    character, match neither branch, and emit an empty turn -- silently calibrating on
    "User: <end_of_utterance>" instead of on the text.
    """
    if typed:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": user}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
        ]
    else:
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    try:
        out = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:  # template doesn't accept enable_thinking
        out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    # Some tokenizers return a BatchEncoding/Mapping rather than a flat list[int] (note:
    # BatchEncoding is a UserDict, not a dict subclass); pull out the ids so the caller's
    # length check / slicing operate on tokens, not on the encoding's fields.
    if isinstance(out, Mapping):
        out = out["input_ids"]
    return list(out)


def build_calibration(sources, tokenizer, window: int) -> list:
    """Build calibration samples ([1, <=window] int tensors) from (name, count) pairs.

    Chat sources are rendered with the tokenizer's chat template (assistant turns
    ending in the eos token); WikiText is cut into raw windows. Unavailable sources
    are skipped with a warning so calibration degrades gracefully.

    `datasets` is imported inside the hub branch, not here: calibrating from a local corpus
    is the case that needs no hub at all, and importing it up front made a purely local
    calibration fail with ModuleNotFoundError on an environment that never needed it.
    """
    has_chat = getattr(tokenizer, "chat_template", None) is not None
    samples: list = []
    for name, count in sources:
        if name not in _SOURCES:
            # A local corpus is a first-class source: raw windows, same as WikiText.
            text = _local_text(name)
            if text is None:
                raise ValueError(
                    f"unknown calibration source {name!r}: it is neither one of "
                    f"{sorted(_SOURCES)} nor a path that exists. Skipping it silently was "
                    f"worse: a typo left the model calibrated on less data than asked for."
                )
            ids = tokenizer(text, return_tensors="pt").input_ids[0]
            added = 0
            for j in range(min(count, ids.numel() // window)):
                samples.append(ids[j * window : (j + 1) * window].unsqueeze(0))
                added += 1
            logger.info("calibration: %s (local) -> %d samples", name, added)
            continue
        path, cfg, split, fmt = _SOURCES[name]
        import datasets

        try:
            ds = (
                datasets.load_dataset(path, cfg, split=split)
                if cfg
                else datasets.load_dataset(path, split=split)
            )
        except Exception as exc:  # noqa: BLE001 - network/availability; skip, don't abort
            logger.warning("calibration source %s unavailable (%s); skipping", name, exc)
            continue

        added = 0
        if fmt is None:  # raw windows
            ids = tokenizer(
                "\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt"
            ).input_ids[0]
            for j in range(min(count, ids.numel() // window)):
                samples.append(ids[j * window : (j + 1) * window].unsqueeze(0))
                added += 1
        else:  # chat-formatted turns
            for i in range(min(count, len(ds))):
                user, assistant = fmt(ds[i])
                if has_chat:
                    ids = chat_ids(tokenizer, user, assistant)[:window]
                else:
                    ids = tokenizer(user + "\n" + assistant).input_ids[:window]
                if len(ids) >= 8:
                    samples.append(torch.tensor([ids]))
                    added += 1
        logger.info("calibration: %s -> %d samples", name, added)
    logger.info("calibration: %d samples total", len(samples))
    return samples
