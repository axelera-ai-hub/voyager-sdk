# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM2 device composition: SigLIP vision graph + connector + Llama text
decoder, bridged by a host embed-splice.

The text decoder is SmolLM2 -- its own weights, a Llama-shape architecture (RMSNorm,
GQA, SwiGLU, plain RoPE, head_dim 64 -> design-P head padding). Since the shapes and
op sequence are Llama's, the text device graph is authored by the existing
``LlamaMulticoreImplementation`` unchanged; ``SmolVLM2TextImplementation`` subclasses
it (registering ``smolvlm2-metis-multicore``) and overrides ``embed_tokens`` to splice
image embeddings into the ``<image>`` placeholder positions before x64-packing.

The vision encoder is a single-pass bidirectional ViT with no KV cache, so it
CANNOT share the KV/prefill-decode text executor; it is a separate device graph
(``model.SiglipVisionDevice``). ``SmolVLM2Runner`` holds the two opened dispatches
together and ``hf_model`` drives them: encode image -> connector -> image_embeds (once,
at prefill), splice, then the standard text prefill/decode loop. Image tokens enter
only at prefill; every decode step is pure text, so the whole existing decode loop is
reused unchanged.

Two artifacts: ``text`` (a Llama-keyed QuantizedModel bundle compiled by the inherited
implementation) and ``vision`` (built directly by SiglipVisionDevice). All three
checkpoints are device-validated image-in/text-out. Outstanding framework seam: the
vision graph needs the LIVE torch reference (its calibrated observers), so build_artifact
bypasses the usual export->compile split and the vision quant params never reach the bundle.
"""

from __future__ import annotations

import numpy as np

from ...implementation import register_implementation
from ..llama.multicore_impl import LlamaMulticoreImplementation


class SmolVLM2TextImplementation(LlamaMulticoreImplementation):
    """The SmolVLM2 text decoder graph == Llama multicore, plus an image-embed
    splice in ``embed_tokens``.

    The driver sets ``self._image_embeds`` (a ``(num_image_tokens, text_hidden)``
    array from the vision device) before a multimodal prefill; ``embed_tokens`` then
    scatters those rows into the ``<image>`` placeholder positions. When
    ``_image_embeds`` is None the behaviour is identical to Llama (pure text).
    """

    name = "smolvlm2-metis-multicore"
    architecture = "smolvlm2"

    _image_embeds: np.ndarray | None = None

    def set_image_embeds(self, image_embeds: np.ndarray | None) -> None:
        """Attach (or clear) the image embeddings spliced at the next prefill."""
        self._image_embeds = None if image_embeds is None else np.asarray(image_embeds)

    def embed_tokens(self, token_ids: np.ndarray, host_params: dict) -> np.ndarray:
        """Host embedding gather + image splice, returning x64-packed embeddings.

        Identical to the base gather, except the rows at ``image_token_id``
        positions are overwritten (in order) by the connector's image embeddings.
        """
        ids = np.asarray(token_ids)
        rows = host_params["embed_weight"][ids].astype(np.float32)  # (1, T, hidden)
        img = self._image_embeds
        if img is not None and "image_token_id" in host_params:
            itid = int(host_params["image_token_id"])
            mask = ids[0] == itid
            n = int(mask.sum())
            if n:
                # One image row per placeholder, in order. The counts come from two
                # independent places -- the pixel stack the vision graph encoded, and the
                # <image> expansion the chat template produced -- so check rather than
                # assume: a bare [:n] slice would raise obscurely when there are too few
                # rows and, worse, SILENTLY DROP the tail when there are too many, leaving
                # a plausible answer built from a partially spliced image.
                have = img.reshape(-1, img.shape[-1])
                if have.shape[0] != n:
                    raise ValueError(
                        f"vision produced {have.shape[0]} image embed rows but the prompt "
                        f"has {n} <image> placeholders; the pixel stack and the chat "
                        f"template disagree (check the tile/frame count against "
                        f"image_seq_len)."
                    )
                rows[0, mask] = have.astype(np.float32)
        batch, seq, hidden = rows.shape
        return np.ascontiguousarray(rows.reshape(batch, seq, hidden // 64, 64))


class SmolVLM2Runner:
    """The opened halves of an artifact, held together: both towers and their host IO.

    ``vision`` is a ``model.SiglipVisionDevice``; ``text_impl`` is a
    ``SmolVLM2TextImplementation``; ``text_runtime`` a ``RuntimeModel`` over the
    compiled text artifact; ``host_params`` the exported text host IO (incl.
    ``image_token_id``). Image tokens enter only at prefill.

    Deliberately NOT a driver: generation is ``hf_model``'s, so that the samplers, the
    streamers and the stopping criteria are transformers' own rather than a second
    implementation of them here. This class is what a load produces and what that shim,
    ``benchmark`` and the fidelity checks read their pieces out of -- ``_image_embeds``
    being the one piece of work it does, since encoding a tile stack is neither tower's
    job alone.
    """

    def __init__(self, vision, text_impl, text_runtime, host_params, tokenizer=None):
        self.vision = vision
        self.text_impl = text_impl
        self.text_runtime = text_runtime
        self.host_params = host_params
        self.tokenizer = tokenizer

    def _image_embeds(self, pixel_values) -> np.ndarray:
        """Run the vision + connector dispatches over each tile -> stacked image
        embeddings (num_image_tokens, text_hidden), matching the <image> count."""
        tiles = [pixel_values[i : i + 1] for i in range(pixel_values.shape[0])]
        embeds = [self.vision.encode(t)[0] for t in tiles]  # each (num_tokens, text_hidden)
        return np.concatenate(embeds, axis=0)


def build_text_bundle(ref):
    """Build the SmolVLM2 text-decoder ``QuantizedModel`` bundle.

    ``ref.text`` is a plain ``LlamaForCausalLM``, so exporting it under architecture "llama"
    gives Llama-native keys that the reused packer and kernels already understand. The two
    splice scalars are injected into the host scalars because Llama's own ``export_host_io``
    does not emit them and the runtime ``embed_tokens`` override needs them.
    """
    from ...generate import build_quantized_model

    bundle = build_quantized_model("llama", ref.text)
    bundle.host_scalars["image_token_id"] = int(ref.config.image_token_id)
    bundle.host_scalars["scale_factor"] = int(ref.config.scale_factor)
    return bundle


_TEXT_SUBDIR = "text"
_VISION_SUBDIR = "vision"
_BUNDLE_SUBDIR = "quantized_model"
#: The multi-tower manifest at the artifact root, tying the two towers together.
#:
#: Deliberately NOT "pipeline.json": that name is the DECODER contract in this tree, and
#: `shared.artifact_contract.decoder_tower` treats any directory holding one as the decoder
#: tower. A root `pipeline.json` would therefore route `--eval` at the artifact root instead
#: of `text/`, and the shim would look for the decoder's graph where the HF metadata lives.
VLM_MANIFEST = "vlm.json"


def buffer_promotion(config) -> bool:
    """Whether to stage vision temporaries in L2, from ``config.enable_buffer_promotion``
    (None == on, the same convention the decoder implementations use). Promotion fits inside
    the per-sub-device budget ``vision_impl.compile_kernel`` declares, so turning it off is
    only an A/B knob -- ``llm.build --no-buffer-promotion``."""
    return getattr(config, "enable_buffer_promotion", None) is not False


def build_artifact(
    repo_id, config, out_dir, *, calib_n=None, cores=1, num_sub_devices=None, calibration=None
):
    """Compile a full SmolVLM2 device artifact to ``out_dir`` (the analog of
    ``generate.build_artifact`` for the VLM). Loads + joint-W8A8-quantizes the model
    on multimodal calibration, compiles the text decoder into ``out_dir/text`` and the
    vision + connector graphs into ``out_dir/vision``, and writes a top-level
    ``vlm.json`` manifest tying them together. Returns ``out_dir``.

    Requires the Axelera toolchain (clang) and a device to open the runtimes. ``config``
    is a ``GenerationConfig``; ``config.parallel`` folds LUT-range recording into the
    calibration passes (3 full multimodal passes -> 2). There is no reuse path: a build
    overwrites ``out_dir`` unconditionally, and whether one is needed at all is the family
    entry's decision (it reads the artifact's own ``metadata.json``).

    ``calib_n`` is the number of DocVQA ENTRIES to calibrate on, each spent twice (as a still
    and as a text-only turn) so the pass costs 2x``calib_n`` forwards; None takes
    ``calibration.DEFAULT_BUDGET``. ``calibration`` overrides the set outright with a caller's
    own sample list, in which case ``calib_n`` is unused.

    Every shape comes from the resolved HF config, so all three checkpoints build through
    this one path, but they are not interchangeable in cost:

        repo_id                              vision            text        tokens/tile
        SmolVLM2-256M-Video-Instruct   768x12L @512/16   576x30L  9q/3kv     64
        SmolVLM2-500M-Video-Instruct   768x12L @512/16   960x32L 15q/5kv     64
        SmolVLM2-2.2B-Instruct        1152x27L @384/14  2048x24L 32q/32kv    81

    The 500M reuses the 256M's tower exactly. The 2.2B is SigLIP SO400M (head_dim 72 ->
    128 lanes, intermediate 4304 -> 4352, 729 patches -> seq 768 + mask) plus a no-GQA
    ~1.7B decoder, so it is heavier on L2 and the KV cache; it builds and runs at
    ``--num-cores 4 --seq-len 2048`` with promotion on and fp16 off.

    ``config.chunk_len`` is REFUSED rather than dropped: see below, before anything is
    imported, so a mistaken knob costs no torch import.
    """
    # Chunked prefill cannot carry an image. The splice is a HOST operation over the whole
    # prompt -- `SmolVLM2TextImplementation.embed_tokens` fills every <image> placeholder in
    # the ids it is handed from one stacked vision result -- while `_chunk_inputs` embeds one
    # chunk at a time, so the row count and the placeholder count disagree by construction
    # and the run fails on the SPLICE, naming the tile count. The value would also be
    # recorded in vlm.json and fed back to `prefill_hidden` by `hf_model.forward`, making
    # every later run of the artifact fail the same way. `pipeline.build_artifacts` (the
    # family's route in) pins it to None; this is the direct caller's gate.
    chunk_len = getattr(config, "chunk_len", None)
    if chunk_len:
        raise ValueError(
            f"chunk_len={chunk_len} is not supported for SmolVLM2: the vision embeddings are "
            "spliced into the WHOLE prompt on the host, and a chunked prefill embeds one "
            "chunk at a time, so no chunk holds the placeholders the tiles were encoded for. "
            "Build with chunk_len unset (the prompt is ingested in one prefill call)."
        )

    import dataclasses
    import json
    from pathlib import Path
    import time

    import torch

    from . import vision_impl as _vis
    from ...calibration import calibration_device, require_vision_samples, to_device
    from ...quant import record_lut_ranges, reset_lut_ranges
    from ..architecture import get_architecture
    from .calibration import multimodal_calibration
    from .model import SiglipVisionDevice
    from .smolvlm2 import load_pretrained

    def _stage(msg):
        print(f"[build] {msg}", flush=True)

    from .calibration import DEFAULT_BUDGET

    out_dir = Path(out_dir)
    calib_n = DEFAULT_BUDGET if calib_n is None else calib_n
    out_dir.mkdir(parents=True, exist_ok=True)
    arch = get_architecture("smolvlm2")
    # Refused, never downgraded: the forwards are the bulk of a build, so a silent fallback
    # to CPU bills the full cost while the caller believes --cal-device took effect.
    cal_device = calibration_device(getattr(config, "device", None))
    _stage(f"loading + HF weights: {repo_id} (calibrating on {cal_device})")
    ref = load_pretrained(repo_id, dtype=torch.float32, device=cal_device)
    # Echo the resolved geometry: it varies across the family (see the table above) and
    # decides the L2 levers, so it belongs in the build log rather than in the reader's head.
    _v, _t = ref.config.vision_config, ref.config.text_config
    _seq = -(-_v.num_patches // 64) * 64
    _stage(
        f"geometry: vision {_v.hidden_size}x{_v.num_hidden_layers}L head_dim={_v.head_dim} "
        f"@{_v.image_size}/{_v.patch_size} -> {_v.num_patches} patches (seq {_seq}); "
        f"sf={ref.config.scale_factor} -> {_v.num_patches // ref.config.scale_factor**2} "
        f"tokens/tile; text {_t.hidden_size}x{_t.num_hidden_layers}L "
        f"{_t.num_attention_heads}q/{_t.num_key_value_heads}kv"
    )
    if calibration is None:
        _stage(f"fetching calibration samples (budget {calib_n})")
        calib = multimodal_calibration(repo_id, n=calib_n)
    else:
        # A caller's own set: pictures from the domain the model will actually run on, which
        # is what sets the int8 activation scales. The budget is then whatever was handed in.
        calib = list(calibration)
        _stage(f"using {len(calib)} caller-supplied calibration samples")
    # Refuse HERE, not stages later when the vision graph cannot find a LUT buffer to bind.
    require_vision_samples(calib, "pixel_values", "smolvlm2.calibration.multimodal_calibration")
    calib = [to_device(sample, cal_device) for sample in calib]
    # --parallel folds LUT-range recording into the two calibrate passes instead of paying a
    # third full multimodal pass (one forward is a vision pass per tile + the text trunk). Ranges
    # are then captured on fp OBSERVE activations, not post-quantize: sound here because every
    # SmolVLM2 LUT site is a GEMM output or reads the residual stream, and qk_alpha is None,
    # so all are SmoothQuant-invariant. Same trade the bundle path makes -- see
    # LLMArchitecture.set_calibration_recording.
    fold_lut = bool(getattr(config, "parallel", False))
    _stage(
        "calibrating + W8A8 quantizing (see [quant] forwards)"
        + ("; LUT ranges folded in (--parallel)" if fold_lut else "")
    )
    if fold_lut:
        # set_calibration_recording only flips the flag; record_lut_ranges resets first, so
        # reset here too or a re-quantized model unions stale ranges into the new fit.
        reset_lut_ranges(ref)
        arch.set_calibration_recording(ref, True)
    try:
        arch.quantize(ref, calib, smooth_alpha=config.smooth_alpha, qk_alpha=None)
    finally:
        if fold_lut:
            arch.set_calibration_recording(ref, False)

    if not fold_lut:
        _stage("recording LUT ranges (see [lut] forwards)")
        ref._quant_verbose = True
        ref._verbose_label = "lut"
        ref._fwd_count = 0
        ref._quant_t0 = ref._quant_t_last = time.perf_counter()
        try:
            record_lut_ranges(ref, calib)
        finally:
            ref._quant_verbose = False
        _stage(f"LUT ranges done ({ref._fwd_count} forwards)")

    # Export and the vision binder both go through numpy, so come home before either.
    ref.to("cpu")

    # Text decoder artifact (Llama-keyed bundle + splice scalars).
    _stage("compiling text decoder (clang codegen + author) -> text/")
    t = time.perf_counter()
    text_config = dataclasses.replace(config, artifact_dir=out_dir / _TEXT_SUBDIR)
    bundle = build_text_bundle(ref)
    # Saved beside the towers, the shape the family's other VLM writes: calibration is the
    # bulk of a build, so keeping the quantized weights means a recompile at a different
    # context or core count does not have to re-quantize from the fp checkpoint.
    bundle.save(out_dir / _BUNDLE_SUBDIR)
    SmolVLM2TextImplementation().compile(bundle, text_config)
    _stage(f"text decoder compiled in {time.perf_counter() - t:.1f}s")

    # Vision + connector graphs (compiles + writes vision/{vision,connector}/*).
    # Promotion is on by default and fits, because compile_kernel budgets L2 as
    # cores/NUM_AI_CORES.
    promote = buffer_promotion(config)
    _stage(f"compiling vision + connector graphs -> vision/ (promote={promote})")
    t = time.perf_counter()
    compiled = SiglipVisionDevice(
        ref.vision_model,
        ref.connector,
        ref.config,
        seq_real=ref.config.vision_config.num_patches,
        out_dir=out_dir / _VISION_SUBDIR,
        reference_model=ref,
        cores=cores,
        num_sub_devices=num_sub_devices,
        enable_buffer_promotion=promote,
        generation=getattr(config, "generation", "OMEGA") or "OMEGA",
        # Constructed for the compile ALONE, so it opens no device. It used to open two
        # RuntimeModels and close them again, which made an AIPU a requirement for building
        # -- and a build is the step most likely to happen on a machine that has none (the
        # cost is calibration, hours of CPU or GPU, and the artifact is carried to the board
        # afterwards). It also claimed the part for the length of the compile, since a device
        # is claimed whole by one connection.
        open_device=False,
    )
    del compiled  # the graphs are on disk; nothing here runs them
    _stage(f"vision + connector compiled in {time.perf_counter() - t:.1f}s")

    _stage("writing vlm.json manifest")
    manifest = {
        "implementation": "smolvlm2-metis-multicore",
        "repo_id": repo_id,
        "context_len": config.context_len,
        "chunk_len": getattr(config, "chunk_len", None),
        "text_dir": _TEXT_SUBDIR,
        "vision_dir": _VISION_SUBDIR,
        "num_cores": int(cores),
        # Hardware generation, which sets how many sub-devices a load claims (4 on OMEGA,
        # 8 on EUROPA). Recorded because load_runner has no other way to know it.
        "generation": getattr(config, "generation", "OMEGA") or "OMEGA",
        "image_token_id": int(ref.config.image_token_id),
        "scale_factor": int(ref.config.scale_factor),
        # Tokens one tile/frame occupies in the prompt (== the processor's image_seq_len):
        # 64 on the 256M/500M, 81 on the 2.2B. Recorded so a frame-budget or prompt-cost
        # check can read it off the artifact instead of reconstructing the processor.
        "image_seq_len": int(ref.config.vision_config.num_patches)
        // int(ref.config.scale_factor) ** 2,
        # Vision-graph shaping knobs, resolved from the environment at import time. Recorded
        # because they are otherwise unrecoverable from the artifact: a build made with
        # SIGLIP_HIDDEN_FP16=1 or promotion off looks identical afterwards to one without.
        "buffer_promotion": promote,
        "siglip_flags": _vis.kernel_flags(),
    }
    # The HF face of the artifact: config + processor + generation defaults, so the
    # directory loads through AutoModelForImageTextToText/AutoProcessor with no hub and no
    # network (see hf_model.save_hf_metadata).
    _stage("writing HuggingFace metadata (config.json, processor, generation_config)")
    from .hf_model import save_hf_metadata

    save_hf_metadata(out_dir, repo_id)

    # vlm.json is the last thing a build writes, so its presence is what marks the
    # multi-tower artifact complete.
    (out_dir / VLM_MANIFEST).write_text(json.dumps(manifest, indent=2))
    return out_dir


def load_runner(out_dir, *, device=None):
    """Reopen a ``build_artifact`` directory into a runnable ``SmolVLM2Runner`` --
    no torch / recompilation (the loader ``benchmark`` measures through).
    Requires a device to open the runtimes.

    The sub-device COUNT is not a choice worth offering (see below) and the core count is
    read off the artifact. WHICH part to claim is a choice: ``device`` names one the way
    ``--device`` spells it, and defaults to the first of the host."""
    import json
    from pathlib import Path

    import numpy as np

    from ...generate import RuntimeModel, load_tokenizer
    from .model import SiglipVisionDevice

    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / VLM_MANIFEST).read_text())
    text_art = out_dir / manifest["text_dir"]
    host_params = dict(np.load(text_art / "host_params.npz"))
    # The reservation is DERIVED from the decoder build (`shared.runtime.required_cores`
    # off its call_graphs), not from the generation: this tree's rule is that a build
    # reserves exactly the cores it spans, and needing more is a compile-time decision
    # (shard it) rather than a bigger claim. For the shipped 4-core Omega build the two
    # answers coincide.
    #
    # The L2 arena is shared across exactly the sub-devices claimed, and SmolVLM2 carries a
    # Llama KV cache sized by context_len -- so an artifact compiled for fewer cores than
    # its cache needs fails AT LOAD rather than silently: RuntimeModel refuses a connection
    # smaller than the build instead of dying later as ZE_RESULT_ERROR_DEVICE_LOST.
    text_runtime = RuntimeModel(text_art / "model.json", device=device)
    # One device is claimed whole by one connection, so the vision pipeline shares the
    # text decoder's connection instead of opening a second (which the driver refuses).
    # On a failed vision load that same exclusivity turns a leaked text_runtime into a
    # device the NEXT launch cannot open either, reported as "another process already
    # holding the device" -- accurate, and pointing away from here.
    try:
        vision = SiglipVisionDevice.load(out_dir / manifest["vision_dir"], share_from=text_runtime)
    except BaseException:
        text_runtime.close()
        raise
    tokenizer = None
    try:
        tokenizer = load_tokenizer(manifest["repo_id"])
    except Exception:  # noqa: BLE001 - tokenizer optional for embed-only use
        pass
    return SmolVLM2Runner(
        vision, SmolVLM2TextImplementation(), text_runtime, host_params, tokenizer
    )


register_implementation(SmolVLM2TextImplementation())
