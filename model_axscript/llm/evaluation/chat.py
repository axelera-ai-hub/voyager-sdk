# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Interactive chat over a compiled LLM artifact.

Loads a built artifact (model.json + host_params + pipeline.json), then runs a
multi-turn REPL: each turn the full conversation is rendered with the model's
chat template, prefilled in one call, and decoded token-by-token with streaming
output. The executor's KV cache is overwritten from position 0 each turn, so the
conversation is re-prefilled statelessly -- simple and robust for a small model.

Run it:
    python -m model_axscript.llm.evaluation.chat --artifact-dir build/qwen3-0.6b-fix
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import logging
import os
from pathlib import Path
import select
import shutil
import sys
import time

from ..generate import GenerationConfig, RuntimeModel, _load_host_params
from ..implementation import get_implementation

logger = logging.getLogger(__name__)

# Context kept free for a reply when no explicit --max-new-tokens is given.
_REPLY_RESERVE = 512

# How long to keep collecting after a line arrives before treating the turn as finished.
# A paste lands as a burst of lines; readline returns only the first and leaves the rest
# queued, which would fire one turn per line. Short enough that a human cannot type a
# whole further line inside it.
_PASTE_DRAIN_S = 0.12

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, *codes: str) -> str:
    if not _COLOR or not text:
        return text
    return "".join(f"\033[{c}m" for c in codes) + text + "\033[0m"


def _read_turn(prompt: str) -> str:
    """One user turn, joining a pasted multi-line block into a single message."""
    lines = [input(prompt)]
    while select.select([sys.stdin], [], [], _PASTE_DRAIN_S)[0]:
        more = sys.stdin.readline()
        if not more:
            break
        lines.append(more.rstrip("\n"))
    return "\n".join(lines).strip()


def _render(tokenizer, messages, enable_thinking: bool):
    """Render the conversation to token ids via the chat template.

    Falls back to plain tokenization for tokenizers without a chat template, and
    drops the enable_thinking kwarg for templates that do not accept it.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"
        return tokenizer(text, return_tensors="np").input_ids[0].tolist()
    try:
        out = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=enable_thinking
        )
    except TypeError:  # template doesn't support enable_thinking
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    # transformers >=5 defaults apply_chat_template to return_dict=True, yielding a
    # BatchEncoding/Mapping rather than a flat list[int]; pull out the ids explicitly.
    if isinstance(out, Mapping):
        out = out["input_ids"]
    return list(out)


def _render_text(tokenizer, messages, enable_thinking: bool) -> str | None:
    """The chat template's rendered STRING, or None if it has no template."""
    if getattr(tokenizer, "chat_template", None) is None:
        return None
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def derive_turn_fragment(tokenizer, enable_thinking: bool) -> tuple[str, str] | None:
    """(before, after) wrapping a follow-up user turn's text, or None if undetectable.

    Incremental chat cannot re-render the history each turn, because the template is not
    append-only: Qwen renders a trailing assistant turn with a <think></think> block and
    then STRIPS that block when the same turn becomes history, so the token stream the
    device was fed is neither a prefix nor an extension of the next render. We therefore
    own the fed stream and extend it with this fragment, which is recovered from the
    template itself (with sentinels) rather than hardcoded, and self-checked below.

    `before` keeps the assistant-closing eos: the decode loop breaks ON eos without
    feeding it, so the fed stream never contains the model's own turn terminator and the
    fragment has to supply it -- which is also the right thing when generation instead ran
    out of budget mid-sentence.
    """
    s1, s2, aa = "\x00U1\x00", "\x00U2\x00", "\x00A1\x00"
    two = _render_text(
        tokenizer,
        [
            {"role": "user", "content": s1},
            {"role": "assistant", "content": aa},
            {"role": "user", "content": s2},
        ],
        enable_thinking,
    )
    one = _render_text(tokenizer, [{"role": "user", "content": s1}], enable_thinking)
    if not two or not one or s2 not in two or aa not in two:
        return None
    left, after = two.split(s2, 1)
    before = left.split(aa, 1)[-1]
    eos = getattr(tokenizer, "eos_token", None)
    # Two invariants we CAN check. The fragment must end with the very generation prompt
    # the first turn already used (so every turn opens the assistant the same way), and it
    # must begin with the eos the model emits at the end of its own turn (so we strip
    # exactly one copy rather than double-feeding it). Deliberately NOT checked against
    # the template's history render -- that render drops the <think> block and so can
    # never match the stream the device was actually fed.
    if after != one.split(s1, 1)[-1]:
        return None
    if not (eos and before.startswith(eos)):
        return None
    return before, after


def _stream(impl, runtime, tokenizer, host_params, delta, config, base_position: int):
    """Drive one turn, yielding (text so far, generated ids so far).

    Yields the ids as well as the text because the caller has to append exactly what the
    executor consumed to its fed stream, and detokenize->retokenize is not an identity.
    """
    ids: list[int] = []
    for tok in impl.generate_token_ids(
        runtime, tokenizer, delta, config, host_params=host_params, base_position=base_position
    ):
        ids.append(int(tok))
        yield tokenizer.decode(ids, skip_special_tokens=True), ids


def _eos_token_ids(repo_id: str) -> tuple[int, ...]:
    """The model's full end-of-turn id set (Qwen3 uses <|im_end|> and <|endoftext|>).

    tokenizer.eos_token_id is a single id; the generation_config carries the full set.
    Returns () if unavailable (generate still stops on tokenizer.eos_token_id).
    """
    try:
        from transformers import GenerationConfig as HFGenerationConfig

        eos = HFGenerationConfig.from_pretrained(repo_id).eos_token_id
    except Exception:
        return ()
    if eos is None:
        return ()
    return tuple(eos) if isinstance(eos, (list, tuple)) else (int(eos),)


def _fit_context(tokenizer, messages, enable_thinking, max_len):
    """Render to ids, dropping the oldest non-system turns until they fit max_len."""
    ids = _render(tokenizer, messages, enable_thinking)
    while len(ids) > max_len and sum(m["role"] != "system" for m in messages) > 1:
        drop = next(i for i, m in enumerate(messages) if m["role"] != "system")
        del messages[drop]
        ids = _render(tokenizer, messages, enable_thinking)
    return ids


def chat(
    artifact_dir: str | Path,
    *,
    repo_id: str = "Qwen/Qwen3-0.6B",
    max_new_tokens: int | None = None,
    temperature: float = 0.7,
    system: str | None = None,
    enable_thinking: bool = False,
    generation: str | None = None,
    device: str | None = None,
    host_head: bool = False,
) -> None:
    from ..generate import load_tokenizer

    artifact_dir = Path(artifact_dir)
    pipeline = json.loads((artifact_dir / "pipeline.json").read_text())
    context_len = int(pipeline["context_len"])
    config = GenerationConfig(
        implementation=pipeline["implementation"],
        repo_id=repo_id,
        context_len=context_len,
        chunk_len=pipeline.get("chunk_len"),  # present only for chunked-prefill artifacts
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        artifact_dir=artifact_dir,
        stop_token_ids=_eos_token_ids(repo_id),
    )
    tokenizer = load_tokenizer(repo_id)
    # Reserve what the artifact actually spans, not the whole chip: RuntimeModel derives
    # required_cores() from the graph spec when this is None. A 1-core build asking for
    # all 8 EUROPA sub-devices fails as soon as anything else holds the device.
    # `generation` is passed through rather than resolved here: RuntimeModel falls back to
    # the artifact's own pipeline.json when it is None, so an absent --generation keeps the
    # artifact's record as the device filter.
    runtime = RuntimeModel(artifact_dir / "model.json", device=device, generation=generation)
    runtime.force_host_head = host_head
    impl = get_implementation(pipeline["implementation"])
    host_params = _load_host_params(artifact_dir, pipeline)
    # Leave room for the response so prefill never fills the whole cache. This is a
    # RESERVATION, not the cap: max_new_tokens None lets a reply run to the end of the
    # context, but the history still has to stop short of filling it or there would be
    # nothing left to answer with.
    prompt_budget = max(context_len - (max_new_tokens or _REPLY_RESERVE), 1)

    # Incremental turns: own the fed token stream and extend it, instead of re-rendering
    # the whole conversation each turn. Re-prefilling is quadratic in the conversation, and
    # for a linear-attention mixer it is also WRONG -- its recurrent state is an accumulator
    # with no position addressing, so re-feeding the history double-counts it (and the state
    # cannot be rewound). None means the template is not extendable; fall back to
    # re-rendering, which needs a reset per turn to be correct.
    fragment = derive_turn_fragment(tokenizer, enable_thinking)
    incremental = fragment is not None
    if not incremental:
        logging.getLogger(__name__).warning(
            "chat template is not extendable; falling back to stateless re-prefill "
            "(quadratic, and inexact for a recurrent mixer)"
        )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    print(_c("bot chat", "1") + _c("  ctrl-c interrupts a reply, ctrl-d quits", "2"))
    print(_c(f"  {artifact_dir}", "2"))
    print(
        _c(
            f"  context {context_len} · temperature {temperature} · "
            f"reply {'unbounded' if max_new_tokens is None else max_new_tokens} · "
            f"{'incremental' if incremental else 'stateless re-prefill'} prefill",
            "2",
        )
    )
    print(_c("  /reset clears history · /exit quits", "2") + "\n")

    fed: list[int] = []  # exactly what the executor has consumed

    def reset():
        fed.clear()
        if hasattr(runtime, "reset_state"):
            try:
                runtime.reset_state()
            except Exception as exc:  # noqa: BLE001 - a failed reset is recoverable-ish
                logging.getLogger(__name__).warning("reset_state failed: %s", exc)

    while True:
        try:
            user = _read_turn(_c("you", "1", "36") + _c(" › ", "2"))
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print(_c("  (ctrl-d to quit)", "2"))
            continue
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            messages = [m for m in messages if m["role"] == "system"]
            reset()
            print(_c("  history cleared", "2") + "\n")
            continue

        messages.append({"role": "user", "content": user})
        if incremental and fed:
            delta = tokenizer(fragment[0] + user + fragment[1], add_special_tokens=False)
            delta = list(delta["input_ids"])
        else:
            # First turn (or non-extendable template): render the conversation and start over.
            if fed:
                reset()
            delta = _fit_context(tokenizer, messages, enable_thinking, prompt_budget)
        # The recurrent state cannot be rewound, so once the stream outgrows the cache the
        # only correct move is to start the sequence again from the trimmed history.
        if len(fed) + len(delta) > prompt_budget:
            reset()
            delta = _fit_context(tokenizer, messages, enable_thinking, prompt_budget)

        print(_c("bot", "1", "35") + _c(" › ", "2"), end="", flush=True)
        prev, full, out_ids, live, col = "", "", [], 0, 0
        started, first_at, interrupted = time.perf_counter(), None, False
        try:
            for full, out_ids in _stream(
                impl, runtime, tokenizer, host_params, delta, config, len(fed)
            ):
                if first_at is None:
                    first_at = time.perf_counter()
                new = full[len(prev) :]
                prev = full
                # The rate rides on the current line and is rubbed out before the next
                # token lands, so it never ends up inside the reply text.
                print("\b" * live + " " * live + "\b" * live + new, end="")
                col = len(new) - new.rfind("\n") - 1 if "\n" in new else col + len(new)
                elapsed = time.perf_counter() - first_at
                rate = f"  {len(out_ids) / elapsed:.1f} tok/s" if elapsed > 0.2 else ""
                if col + len(rate) >= shutil.get_terminal_size().columns:
                    rate = ""
                print(_c(rate, "2"), end="", flush=True)
                live = len(rate)
        except KeyboardInterrupt:
            interrupted = True
        print("\b" * live + " " * live + "\b" * live)

        took = time.perf_counter() - started
        decode = len(out_ids) / (time.perf_counter() - first_at) if first_at else 0.0
        note = "  interrupted" if interrupted else ""
        print(
            _c(
                (
                    f"  {len(out_ids)} tok · {took:.1f}s · {decode:.1f} tok/s"
                    f" · {(first_at - started):.2f}s to first{note}"
                    if first_at
                    else f"  no output · {took:.1f}s{note}"
                ),
                "2",
            )
            + "\n"
        )

        if interrupted and not out_ids:
            # The prompt was part-ingested and there is no way to tell how far it got,
            # so the only sound position to resume from is the start.
            reset()
            messages = [m for m in messages if m["role"] == "system"]
            print(_c("  interrupted during prefill; history cleared", "2") + "\n")
            continue
        fed.extend(delta)
        # generate_token_ids yields a token BEFORE feeding it, so an interrupted reply's
        # last token never reached the device and must not count towards the position.
        fed.extend(out_ids[:-1] if interrupted else out_ids)
        messages.append({"role": "assistant", "content": full})


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m model_axscript.llm.evaluation.chat",
        description="Interactive chat over a compiled LLM artifact.",
    )
    p.add_argument("--artifact-dir", required=True, help="compiled artifact directory")
    p.add_argument(
        "--repo-id",
        default=None,
        help="tokenizer / chat-template source; defaults to the artifact's pipeline.json",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="response length cap; default is as much as the context still holds",
    )
    p.add_argument("--temperature", type=float, default=0.7, help="0 => greedy")
    p.add_argument("--system", default=None, help="optional system prompt")
    p.add_argument(
        "--think", action="store_true", help="enable the model's thinking mode (Qwen3 <think>)"
    )
    p.add_argument(
        "--generation",
        default=None,
        choices=("OMEGA", "EUROPA"),
        help="target hardware generation; defaults to the artifact's pipeline.json",
    )
    p.add_argument(
        "--device",
        default=None,
        help="device name (e.g. europa-0:ac:0) or index; defaults to $AXELERA_METIS_DEVICE, "
        "else the first device of the artifact's generation -- which is not necessarily a "
        "healthy one",
    )
    p.add_argument(
        "--host-head",
        action="store_true",
        help="run the lm_head as the host fp16 gemv even though the artifact carries a device "
        "head graph; one dispatch per token instead of two, no rebuild",
    )
    p.add_argument("--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    from ..generate import artifact_repo_id

    repo_id = args.repo_id or artifact_repo_id(args.artifact_dir)
    if repo_id is None:
        raise SystemExit(f"no repo_id in {args.artifact_dir}/pipeline.json; pass --repo-id")
    chat(
        args.artifact_dir,
        repo_id=repo_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        system=args.system,
        enable_thinking=args.think,
        generation=args.generation,
        device=args.device,
        host_head=args.host_head,
    )


if __name__ == "__main__":
    main()
