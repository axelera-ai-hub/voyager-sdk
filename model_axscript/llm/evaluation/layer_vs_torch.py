# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Accuracy of one compiled decoder layer against the fakequant torch reference.

Answers "is the kernel computing the math the quantized reference model intends?" -- as
distinct from "what does quantization cost?", which is a fakequant-vs-fp32 question about
the model, not the kernel. So the reference is the torch module with W8A8 + int8 SDPA
fakequant ENABLED: the same numerical intent the kernel implements.

Both sides come from ONE calibration in ONE process: the bundle the kernel binds its
constants from is exported (build_quantized_model) from the very torch module used as the
reference, so the two cannot disagree on a scale. Rebuilding the reference from a saved
bundle instead would risk reporting calibration drift as kernel error.

The layer input is the real hidden state at that depth (embeddings pushed through the
preceding layers in torch), not noise: the activation quant scales were calibrated for that
distribution, so noise would exercise the kernel outside the range its scales were fitted
to and report a pessimistic error.

Everything under test is this tree's ``model_axscript`` code. Only ``run`` (a generic
run-a-compiled-model-on-device helper) is imported from the compiler checkout, because it
lives beside the ALTO integration tests and has no copy here -- which is also why
``layer_benchmark.py``'s ``from ...utils import compile_and_run`` cannot resolve in this
tree. Point --compiler-root at that checkout if it is not in the default place.

``--emulate`` adds a third leg (the same module through the compiler's emulation) which is
what makes a bad number diagnosable: device-vs-emulate isolates a lowering bug, while both
agreeing yet differing from torch points at the intended math or the reference itself.

Usage:

    python -m model_axscript.llm.evaluation.layer_vs_torch --layer 3 --seq 512 --cores 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

_logger = logging.getLogger(__name__)
PWORD = 64
_DEFAULT_COMPILER_ROOT = "/home/axuser/software-platform/host/tvm/axelera_compiler"


def _device_run(compiler_root: str):
    """The compiler checkout's run-on-device helper (no equivalent in this tree).

    Deliberately ``run`` and not ``compile_and_run``: the spec needed to pack the layer's
    constants comes from the compiled model.json, and those constants are inputs to the run,
    so the compile must happen first anyway -- compile_and_run would then compile the very
    same module a SECOND time, which dominates this harness's wall-clock.
    """
    if compiler_root not in sys.path:
        sys.path.insert(0, compiler_root)
    try:
        from tests.integration.alto.utils import run
    except ImportError as e:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            f"cannot import tests.integration.alto.utils from {compiler_root!r}: {e}. "
            "Pass --compiler-root pointing at the axelera_compiler checkout."
        ) from e
    return run


def _metrics(got: np.ndarray, ref: np.ndarray) -> dict:
    """rel-L2 / cosine / max-abs of a device output against its reference."""
    a = np.asarray(got, np.float64).ravel()
    b = np.asarray(ref, np.float64).ravel()
    denom = np.linalg.norm(b) + 1e-12
    return {
        "rel_l2": float(np.linalg.norm(a - b) / denom),
        "cosine": float(a @ b / ((np.linalg.norm(a) + 1e-12) * denom)),
        "max_abs": float(np.max(np.abs(a - b))),
        "ref_rms": float(np.sqrt(np.mean(b**2))),
    }


def _lane_pack(x) -> np.ndarray:
    """Split a trailing dim that is a multiple of PWORD into (blocks, PWORD) lanes."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, np.float32)
    return np.ascontiguousarray(x.reshape(*x.shape[:-1], x.shape[-1] // PWORD, PWORD))


def cached_reference(repo_id, layer, seq, calib_budget, cal_device, cache_dir):
    """(bundle, layer_in, layer_out) -- calibrating and running torch ONCE, then reusing.

    Calibration (~2-3 min) and the torch reference (~1 min: the deltanet layers ahead of
    `layer` are a per-timestep Python loop) are identical on every rerun, and together dwarf
    the compile. Caching them turns an iteration into compile+device-run, which is what makes
    bisecting the kernel practical. Keyed by everything either output depends on, so a changed
    seq/layer/budget cannot silently reuse the wrong tensors -- note the KERNEL is deliberately
    NOT part of the key: recompiling it is exactly what we are iterating on.
    """
    from ..quantized_model import QuantizedModel

    key = f"{repo_id.replace('/', '_')}_L{layer}_s{seq}_cb{calib_budget}"
    root = Path(cache_dir) / key
    refs, bundle_dir = root / "refs.npz", root / "bundle"
    if refs.is_file() and (bundle_dir / "quantized_model.json").is_file():
        _logger.info("reusing cached bundle + torch reference from %s", root)
        z = np.load(refs)
        return QuantizedModel.load(bundle_dir), z["layer_in"], z["layer_out"]

    import torch

    model, bundle, _ = load_quantized_pair(repo_id, cal_device, calib_budget, seq)
    from ..generate import load_tokenizer

    ids = _calibration_text(load_tokenizer(repo_id), seq)
    input_ids = torch.tensor([ids], dtype=torch.long, device=cal_device)
    _logger.info("torch reference through layer %d ...", layer)
    layer_in, layer_out = torch_layer_reference(model, layer, input_ids)
    li = layer_in.detach().cpu().numpy()
    lo = layer_out.detach().cpu().numpy()
    root.mkdir(parents=True, exist_ok=True)
    bundle.save(bundle_dir)
    np.savez(refs, layer_in=li, layer_out=lo)
    _logger.info("cached bundle + reference under %s", root)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return bundle, li, lo


def load_quantized_pair(repo_id: str, cal_device: str, calib_budget: int, context_len: int):
    """(torch fakequant model, exported bundle, config) -- one calibration, so scales agree."""
    import torch

    from ..generate import GenerationConfig, _tokenize_calibration, build_quantized_model
    from ..models.architecture import get_architecture

    cfg = GenerationConfig(
        implementation="qwen3.5-europa-multicore",
        repo_id=repo_id,
        context_len=context_len,
        cal_device=cal_device,
        calib_budget=calib_budget,
    )
    arch = get_architecture("qwen3.5")
    _logger.info("loading %s on %s ...", repo_id, cal_device)
    model = arch.load_pretrained(
        repo_id, dtype=torch.float32, device=cal_device, quantize_attention=True
    )
    calib = _tokenize_calibration(cfg)
    _logger.info("calibrating + quantizing (%d samples) ...", len(calib))
    arch.quantize(model, calib, smooth_alpha=cfg.smooth_alpha, qk_alpha=cfg.qk_alpha)
    arch.record_lut_ranges(model, calib)
    return model, build_quantized_model("qwen3.5", model), cfg


def torch_layer_reference(model, layer: int, input_ids):
    """(layer input, layer output) from the fakequant torch model at ``layer``.

    Uses the model's own rotary_emb / _causal_mask rather than rebuilding either, so the
    reference cannot disagree with itself about RoPE or masking conventions. Returns the
    layer's input as well, so the device is handed the identical tensor.
    """
    import torch

    trunk = model.model
    with torch.no_grad():
        hidden = trunk.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        cos, sin = trunk.rotary_emb(position_ids)
        mask = trunk._causal_mask(seq_len, hidden.dtype, hidden.device)
        for i in range(layer):
            hidden = trunk.layers[i](hidden, cos, sin, mask)
        layer_in = hidden.clone()
        layer_out = trunk.layers[layer](hidden, cos, sin, mask)
    return layer_in, layer_out


def run_device_layer(
    bundle,
    model_config,
    impl,
    layer,
    seq,
    cores,
    out_dir,
    layer_in,
    compiler_root,
    with_emulation: bool = False,
    which: str = "europa",
):
    """Compile the layer of the given kind, bind the real constants, run it on device."""
    from axelera.compiler.alto.compiler.config import AltoConfig, HardwareGeneration, Target
    from axelera.compiler.alto.compiler.driver import compile as alto_compile

    from ..implementation import make_causal_mask, make_mask_query_positions, make_seq_pos
    from ..layout import extract_layer_buffer_spec

    # Which tree's layer to measure. The europa layer was cloned from the metis one, so
    # measuring BOTH is what separates "this change regressed accuracy" from "this layer
    # always had this error" -- the same reference, harness and input for each.
    if which == "metis":
        from ..models.qwen3_5.deltanet_impl import (
            deltanet_layer_buffers,
            full_attention_layer_buffers,
            make_deltanet_layer_kernel,
            make_full_attention_layer_kernel,
        )

        generation, kernel_kwargs = HardwareGeneration.OMEGA, {}
    else:
        from ..models.qwen3_5.europa_multicore_impl import (
            deltanet_layer_buffers,
            full_attention_layer_buffers,
            make_deltanet_layer_kernel,
            make_full_attention_layer_kernel,
        )

        generation, kernel_kwargs = HardwareGeneration.EUROPA, {"cores": cores}

    device_run = _device_run(compiler_root)

    # One-shot prefill (ctx == seq). For full_attention the KV cache then holds exactly
    # these tokens and the causal mask spans 0..seq-1 -- the same computation torch's full
    # self-attention does. For linear_attention the recurrent and conv states start empty,
    # which likewise matches torch running the block over the whole sequence.
    kind = model_config.layer_type(layer)
    if kind == "linear_attention":
        mod = make_deltanet_layer_kernel(model_config, ctx_len=seq, phase="prefill", chunk_len=seq)
        layer_buffers = deltanet_layer_buffers
        # In-place persistent state, codegen-allocated rather than host-bound.
        state_names = {"conv_state", "recurrent_state"}
    else:
        mod = make_full_attention_layer_kernel(
            model_config, ctx_len=seq, phase="prefill", chunk_len=seq, **kernel_kwargs
        )
        layer_buffers = full_attention_layer_buffers
        # The compiled params carry the OUTPUT names (k_out/v_out), not the kernel's
        # k_cache/v_cache.
        state_names = {"k_out", "v_out", "k_cache", "v_cache"}
    cfg = AltoConfig(
        output_dir=Path(out_dir),
        generation=generation,
        target=Target.DEVICE,
        num_cores=cores if which == "europa" else 1,
        enable_tiling_injection=True,
        enable_buffer_promotion=False,
        # enable_datapath_fusion is deliberately left unset: AltoConfig defaults it to
        # False, which is what both qwen3.5 kernels want and what the production compile
        # path (Implementation._compile_phase) also gets, since it never sets it either.
        enable_llm_support=True,
        assign_unique_input_pools=True,
        unpack_axm_archive=True,
        clean_output_dir=True,
        emit_atex_artifacts=True,
        # `run` reads alto_config.json back from the output dir; compile_and_run used to set
        # this for us, so compiling directly has to ask for it explicitly.
        emit_alto_config=True,
    )
    model_path = alto_compile(mod, cfg)
    spec = extract_layer_buffer_spec(sorted(Path(out_dir).rglob("model.json"))[0])
    buffers = layer_buffers(bundle, spec, layer)

    positions = np.arange(seq)
    cos, sin = impl._rope_tables(positions, model_config.head_dim, model_config.rope_theta)
    hidden_np = _lane_pack(np.asarray(layer_in)[0].reshape(1, seq, -1))
    host = {
        "hidden_states": hidden_np,
        # The kernel adds residual_in to the mixer output where torch reuses its own input
        # as the residual, so for a standalone layer the two must be the same tensor.
        "residual_in": hidden_np,
        "seq_pos": make_seq_pos(0),
        "q_rope_cos": cos,
        "q_rope_sin": sin,
        "k_rope_cos": cos,
        "k_rope_sin": sin,
        # The europa layer takes per-row positions and regenerates the mask in-DPU; the
        # metis layer takes the dense mask. Offer both; only the declared one is bound.
        "attn_mask_positions": make_mask_query_positions(positions),
        "attn_mask": make_causal_mask(positions, model_config.num_attention_heads, seq),
    }
    prim = mod["main"]  # inputs are positional, in prim_func parameter order
    inputs, missing = [], []
    for var in prim.params:
        buf = prim.buffer_map[var]
        name = buf.name
        if name in host:
            inputs.append(np.ascontiguousarray(host[name]))
        elif name in buffers:
            inputs.append(np.ascontiguousarray(np.asarray(buffers[name])))
        elif name in state_names:
            # Zero-fill from each param's own declared shape rather than hardcoding
            # the geometry.
            shape = [int(d) for d in buf.shape] + [PWORD]
            inputs.append(np.zeros(shape, buf.dtype.split("x")[0]))
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"unbound layer inputs {missing}: neither host-supplied nor emitted by "
            f"{layer_buffers.__name__}. Refusing to substitute zeros -- that would "
            "silently produce a plausible-looking but wrong accuracy number."
        )
    # Emulation of the SAME module on the SAME inputs, to triangulate: it separates a
    # kernel-lowering error (device disagrees with emulate) from an intended-math or
    # reference-equivalence error (both agree, yet differ from torch).
    emulated = None
    if with_emulation:
        from axelera.compiler.alto.compiler import interpret

        emulated = np.asarray(interpret(mod, *inputs)[0])
    return np.asarray(device_run(model_path, inputs)[0]), emulated


def _calibration_text(tok, seq: int) -> list:
    """`seq` token ids of natural text (wikitext if reachable, else a repeated passage)."""
    text = None
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in ds["text"] if t.strip())
    except Exception as e:  # offline / no datasets -- fall back, but say so
        _logger.warning("wikitext unavailable (%s); using a repeated built-in passage", e)
        text = (
            "The Axelera Metis and Europa accelerators execute quantized transformer "
            "inference on an in-memory-compute array. Attention is tiled over the key "
            "and value axes so that the working set is independent of context length. "
        ) * 64
    ids = tok(text, return_tensors=None)["input_ids"]
    while len(ids) < seq:
        ids = ids + ids
    return ids[:seq]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="Qwen/Qwen3.5-2B")
    p.add_argument(
        "--layer", type=int, default=3, help="full_attention or linear_attention layer index"
    )
    p.add_argument("--seq", type=int, default=512, help="one-shot prefill length (== ctx)")
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--calib-budget", type=int, default=16)
    p.add_argument("--cal-device", default="cuda")
    p.add_argument("--compiler-root", default=_DEFAULT_COMPILER_ROOT)
    p.add_argument(
        "--impl",
        default="europa",
        choices=("europa", "metis"),
        help="which tree's layer to measure; metis is the pre-change baseline "
        "the europa one was cloned from, so comparing both separates a regression from "
        "a pre-existing error level",
    )
    p.add_argument(
        "--emulate",
        action="store_true",
        help="also emulate the same module (triangulates lowering vs intended-math error)",
    )
    p.add_argument("--out", default="/tmp/layer_vs_torch")
    p.add_argument(
        "--cache-dir",
        default="/tmp/layer_vs_torch_cache",
        help="where the calibrated bundle + torch reference are cached between runs",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level)

    from ..implementation import _REGISTRY
    from ..models.architecture import get_architecture

    bundle, layer_in, layer_out = cached_reference(
        args.repo_id,
        args.layer,
        args.seq,
        args.calib_budget,
        args.cal_device,
        args.cache_dir,
    )
    model_config = get_architecture("qwen3.5").config_from_dict(bundle.config)

    _logger.info("compiling + running the device layer ...")
    device_out, emulated = run_device_layer(
        bundle,
        model_config,
        _REGISTRY["qwen3.5-europa-multicore"],
        args.layer,
        args.seq,
        args.cores,
        args.out,
        layer_in,
        args.compiler_root,
        with_emulation=args.emulate,
        which=args.impl,
    )

    ref = np.asarray(layer_out)[0].reshape(args.seq, -1)
    got = np.asarray(device_out).reshape(args.seq, -1)
    print()
    print(
        f"layer {args.layer} ({model_config.layer_type(args.layer)})  impl={args.impl}  seq={args.seq}  "
        f"cores={args.cores if args.impl == 'europa' else 1}"
    )

    def _show(label, a, b):
        m = _metrics(a, b)
        print(
            f"  {label:26s} rel_l2={m['rel_l2']:.4f}  cosine={m['cosine']:.6f}  "
            f"max_abs={m['max_abs']:.4f}"
        )

    _show("device vs torch", got, ref)
    if emulated is not None:
        emu = np.asarray(emulated).reshape(args.seq, -1)
        _show("emulate vs torch", emu, ref)
        _show("device vs emulate", got, emu)
    print(f"  reference rms              : {np.sqrt(np.mean(ref.astype(np.float64) ** 2)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
