# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Full DINOv2 inference: host front-end + device fused encoder.

First-step full-model wiring that keeps the non-repeated parts on the host and
runs only the (12x identical) encoder on device:

  * HOST: patch embedding + CLS token + interpolated positional embeddings (the
    HF Dinov2Embeddings module, run in torch) produce the initial hidden states;
    the final LayerNorm (HF Dinov2Model.layernorm) is applied on the host to the
    encoder output. Reusing the HF modules verbatim is the lowest-risk front-end.
  * DEVICE: the fused block kernel (optimized_impl) is compiled ONCE and invoked
    once per layer, threading the packed hidden states layer->layer and binding
    that layer's own int8 weights. The block output layout is exactly the block
    input layout, so activations feed straight back with no repacking.

DINOv2's token count (num_patches + 1 CLS) is not a multiple of 64, so the
sequence is padded up to one and the padded key columns are masked out in
attention (an additive -30000 mask). Real query rows attend only to real keys,
so their outputs match an unpadded run; the padded rows are sliced off before the
final LayerNorm. Calibration happens at the real (unpadded) sequence length, so
the int8 scales see only real tokens.

This does not use the LLM v2 graph/KV executor (which bakes in a KV cache and
prefill/decode phases a bidirectional encoder has none of); it is a plain host
loop over one single-graph kernel -- the low-risk reuse boundary.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ...layout import extract_layer_buffer_spec
from ...quant import enable_w8a8, quantize_model, record_lut_ranges, set_calibration_method
from .calibration import image_calibration
from .hf import config_from_hf, load_full_weights
from .optimized_impl import compile_block, optimized_block_buffers
from .reference import DinoV2Model, smoothquant_groups

# int8 masked-key fill: exp-negligible in the SDPA score space (the op's own int8 mask
# sentinel), so a padded key gets ~0 softmax weight. Matches the qwen int8 mask convention.
_MASK_NEG = -127


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def load_dinov2(
    repo_id: str = "facebook/dinov2-small",
    *,
    n_calib: int = 32,
    calibration_dataset: str = "zh-plus/tiny-imagenet",
    pixel_values: list | None = None,
    smooth_alpha: float | None = 0.5,
    image_size: int | None = None,
    calib_device: str | None = None,
    calib_method: str = "minmax",
    percentile: float = 99.99,
    seg_head: bool = False,
):
    """Load a HF DINOv2 checkpoint and build the full int8 model, calibrated on real images.

    Builds the whole reference (patch embed + encoder + final LayerNorm) and quantize_model's
    it in ONE pass on a representative set of real natural images (n_calib from
    calibration_dataset, an ImageNet proxy -- see calibration.py), the LLM-style joint PTQ:
    every int8 site (patch conv, blocks) sees its real activation, and the final-norm LUT
    range is recorded. Pass pixel_values to supply your own calibration images.

    Returns (hf_model, model, config, seq_real): hf_model (fp reference), the calibrated
    reference.DinoV2Model, the DinoBlockConfig (seq_len = padded), and the real token count.
    """
    from transformers import Dinov2Model

    hf_model = Dinov2Model.from_pretrained(repo_id).eval()
    if pixel_values is None:
        pixel_values = image_calibration(
            repo_id, n=n_calib, dataset=calibration_dataset, image_size=image_size
        )
    image_size = int(pixel_values[0].shape[-1])

    with torch.no_grad():
        seq_real = int(hf_model.embeddings(pixel_values[0]).shape[1])
    seq_pad = _round_up(seq_real, 64)
    config = config_from_hf(hf_model.config, seq_pad, image_size=image_size)

    model = DinoV2Model(
        config,
        hf_model.config.num_hidden_layers,
        hf_model.config.num_channels,
        hf_model.config.patch_size,
        seq_real,
    )
    load_full_weights(model, hf_model, image_size)
    model.eval()

    # Calibration is host torch (CPU by default); calib_device="cuda" runs the fake-quant
    # forwards on GPU (~10-20x faster, same numerics) so large n_calib at high res is cheap.
    calib = pixel_values
    if calib_device is not None:
        model.to(calib_device)
        calib = [px.to(calib_device) for px in pixel_values]
    if calib_method != "minmax":
        n = set_calibration_method(model, calib_method, percentile=percentile)
        print(f"[dino calib] {calib_method} (p={percentile}) on {n} activation observers")
    quantize_model(
        model,
        calib,
        groups=smoothquant_groups(model),
        smooth_alpha=smooth_alpha,
        qk_alpha=None,
    )
    enable_w8a8(model)
    record_lut_ranges(model, calib)
    if seg_head:
        # Calibrate the symmetric int8 scale for the final-norm output (the on-device head's
        # activation input) from the fake-quant model over the calibration images.
        with torch.no_grad():
            amax = max(float(model(px).abs().max()) for px in calib)
        model.seg_feat_scale = amax / 127.0
    if calib_device is not None:
        model.to("cpu")
    return hf_model, model, config, seq_real


def build_encoder_graph(
    hf_model,
    model,
    config,
    out_dir,
    *,
    enable_buffer_promotion=True,
    cores=1,
    dpu_unroll=False,
    seg_head_w=None,
    seg_head_b=None,
    seg_feat_scale=None,
    final_norm=True,
):
    """Author the full DINOv2 as ONE device graph via the LLM graph mechanism: patch-embed
    (kind "embed", input = host-quantized pixels) -> N fused blocks (kind "block") -> final
    LayerNorm (kind "final_norm"), which `final_norm=False` omits so the graph ends at the
    last block: a head trained on pre-norm patch tokens, such as a MixVPR mixer, reads those
    and applying the norm first would feed it the wrong features. The activation threads
    inputs_embeds -> hidden_a -> hidden_b -> ... -> output, ping-ponging the two spine slots
    (graph_descriptor.SPINE_TENSORS) so depth costs 2 live buffers rather than one per block;
    the padding mask is a single shared input to the blocks; each kernel's constants are bound
    as its own outlined weights. One _run_graph dispatch runs everything. Device-only.

    Returns (out_dir, specs): the graph artifact dir + {"embed","block","final_norm"} buffer
    specs (for the host pixel/mask layout).
    """
    from axelera.compiler.alto.compiler.config import Target

    from ...graph_descriptor import KindContract, author_graph_model
    from .optimized_impl import (
        embed_buffers,
        final_norm_buffers,
        final_seg_buffers,
        make_embed_kernel,
        make_final_norm_kernel,
        make_final_seg_kernel,
        make_optimized_block_kernel,
    )

    if seg_head_w is not None and not final_norm:
        raise ValueError(
            "final_norm=False cannot carry a segmentation head: the head reads the normalized "
            "features. Drop the head, or leave final_norm at its default."
        )
    out_dir = Path(out_dir)
    blocks = list(model.encoder.blocks)
    L = len(blocks)
    seg_head = seg_head_w is not None  # emit per-token class logits instead of features

    def _compile(name, make_kernel, ncores):
        _, mj = compile_block(
            config,
            out_dir / f"_{name}",
            target=Target.DEVICE,
            make_kernel=make_kernel,
            enable_buffer_promotion=enable_buffer_promotion,
            cores=ncores,
            dpu_unroll=dpu_unroll,
        )
        atex = sorted((out_dir / f"_{name}").rglob("model.atex"))
        if not atex:
            raise RuntimeError(f"no model.atex under {out_dir / ('_' + name)}")
        return mj, atex[0], extract_layer_buffer_spec(mj)

    if seg_head:
        n_cls = int(seg_head_w.shape[0])
        fn_maker = functools.partial(make_final_seg_kernel, n_classes=n_cls)
    else:
        fn_maker = make_final_norm_kernel

    embed_mj, embed_atex, embed_spec = _compile("embed", make_embed_kernel, 1)
    _, block_atex, block_spec = _compile("block", make_optimized_block_kernel, cores)

    kinds = ["embed"] + ["block"] * L
    contracts = {
        "embed": KindContract(shared_inputs=frozenset({"pixels"}), state=frozenset()),
        "block": KindContract(shared_inputs=frozenset({"attn_mask"}), state=frozenset()),
    }
    kernels = {("encode", "embed"): embed_atex, ("encode", "block"): block_atex}
    layer_buffers = {0: embed_buffers(model, embed_spec)}
    for i, block in enumerate(blocks):
        layer_buffers[i + 1] = optimized_block_buffers(block, block_spec)

    specs = {"embed": embed_spec, "block": block_spec}
    if final_norm:
        _, fn_atex, fn_spec = _compile("finalnorm", fn_maker, 1)
        kinds.append("final_norm")
        contracts["final_norm"] = KindContract(shared_inputs=frozenset(), state=frozenset())
        kernels[("encode", "final_norm")] = fn_atex
        if seg_head:
            layer_buffers[L + 1] = final_seg_buffers(
                model, fn_spec, head_w=seg_head_w, head_b=seg_head_b, feat_scale=seg_feat_scale
            )
        else:
            layer_buffers[L + 1] = final_norm_buffers(model, fn_spec)
        specs["final_norm"] = fn_spec

    author_graph_model(
        out_dir,
        "dinov2",
        kernels,
        kinds,
        contracts,
        layer_buffers,
        threaded_inputs=frozenset({"hidden_states"}),
    )
    return out_dir, specs


class DinoV2Device:
    """Full DINOv2 on device in ONE dispatch: patch-embed + N fused blocks + final LayerNorm
    authored as a single graph via the LLM graph mechanism. The host does only image-processor
    normalization, an int8 pixel quantize+pad, and a trivial output slice. `encode(pixels)`
    runs the whole model in one `_run_graph` call. Device-only (graph executor). Buffer
    promotion on by default; cores>1 head-shards SDPA across cores (matmuls single-core)."""

    def __init__(
        self,
        hf_model,
        model,
        config,
        seq_real,
        out_dir,
        *,
        cores=1,
        num_sub_devices=None,
        enable_buffer_promotion=True,
        dpu_unroll=False,
        seg_head_w=None,
        seg_head_b=None,
        seg_feat_scale=None,
    ):
        self.hf_model = hf_model
        self.model = model
        self.config = config
        self.seq_real = seq_real
        self.seq_pad = config.seq_len
        self.hidden = config.hidden_size
        self.image_size = int(config.image_size)
        self.patch_size = int(config.patch_size)
        self.grid = int(round((seq_real - 1) ** 0.5))
        self.seg_head = seg_head_w is not None
        self.n_classes = int(seg_head_w.shape[0]) if self.seg_head else None
        self.n_cls_pad = int(_round_up(self.n_classes, 64)) if self.seg_head else None
        graph_dir, specs = build_encoder_graph(
            hf_model,
            model,
            config,
            out_dir,
            enable_buffer_promotion=enable_buffer_promotion,
            cores=cores,
            dpu_unroll=dpu_unroll,
            seg_head_w=seg_head_w,
            seg_head_b=seg_head_b,
            seg_feat_scale=seg_feat_scale,
        )
        self.graph_dir = Path(graph_dir)
        self.embed_spec = specs["embed"]
        self.block_spec = specs["block"]
        self._mask = self._padding_mask()
        # Host pixel int8 quant scale: the patch-embed's calibrated (symmetric) activation scale.
        self._pix_scale = float(model.patch_embed.proj.act_observer.qparams()[0].reshape(-1)[0])
        self._write_meta()
        from ....shared.runtime import RuntimeModel

        self.runtime = RuntimeModel(
            graph_dir / "model.json", num_sub_devices=num_sub_devices or cores
        )

    def _write_meta(self) -> None:
        """Persist the small scalars a run needs (alongside the compiled graph) so the artifact
        can be reloaded and run WITHOUT re-calibrating or re-compiling -- see `load`."""
        meta = {
            "repo_id": getattr(self.config, "repo_id", None),
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "seq_real": self.seq_real,
            "seq_pad": self.seq_pad,
            "hidden": self.hidden,
            "grid": self.grid,
            "pix_scale": self._pix_scale,
            "mask_shape": list(self.block_spec["attn_mask"][0]),
            "seg_head": self.seg_head,
            "seg_num_classes": self.n_classes,
        }
        (self.graph_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, graph_dir, *, cores=1, num_sub_devices=None):
        """Reconstruct a runnable device from a cached artifact dir (compiled graph + meta.json)
        WITHOUT any torch model, calibration, or compilation. `encode` works; `reference` (the fp
        comparison) does not (hf_model is None -- load it separately if you need fp). The int8
        weights live in the compiled graph, so this is purely a device load."""
        from ....shared.runtime import RuntimeModel

        graph_dir = Path(graph_dir)
        meta = json.loads((graph_dir / "meta.json").read_text())
        self = cls.__new__(cls)
        self.hf_model = self.model = self.config = None
        self.graph_dir = graph_dir
        self.seq_real = meta["seq_real"]
        self.seq_pad = meta["seq_pad"]
        self.hidden = meta["hidden"]
        self.image_size = meta["image_size"]
        self.patch_size = meta["patch_size"]
        self.grid = meta["grid"]
        self._pix_scale = meta["pix_scale"]
        self.seg_head = bool(meta.get("seg_head", False))
        self.n_classes = meta.get("seg_num_classes")
        self.n_cls_pad = int(_round_up(self.n_classes, 64)) if self.seg_head else None
        self.block_spec = {"attn_mask": (tuple(meta["mask_shape"]), "int8")}
        self._mask = self._padding_mask()
        self.runtime = RuntimeModel(
            graph_dir / "model.json", num_sub_devices=num_sub_devices or cores
        )
        return self

    def _padding_mask(self) -> np.ndarray:
        """int8 1-head-broadcast padding mask: 0 for real keys, -127 for padded keys."""
        shape = self.block_spec["attn_mask"][0]  # (1, 1, seq_pad, ctx_blocks, 64) int8
        ctx_blocks = shape[-2]
        keys = np.arange(ctx_blocks * 64).reshape(ctx_blocks, 64)
        pad = np.where(keys < self.seq_real, 0, _MASK_NEG).astype(np.int8)
        return np.broadcast_to(pad, shape).copy()

    def _quantize_pixels(self, pixel_values: torch.Tensor) -> np.ndarray:
        """(1,C,H,W) fp -> space-to-depth matmul input (1, num_patches, k_pw, 64) int8.

        Host im2col: each P*P patch is flattened in (c, kh, kw) order (== the patch weight's K
        order) into the channel dim, symmetric int8 quantized, and K padded up to a PWORD
        multiple. Replaces the old (1,H,W,1,64) conv layout that zero-padded 3 channels to 64."""
        from math import ceil

        # Quantize with torch.round (multithreaded; ~2.5x faster than np.round, which dominated the
        # host cost -- ~3ms of a ~2.8ms _quantize_pixels @518). /scale (not *inv) keeps it bit-exact
        # vs the old numpy path. -> (C, H, W) int8 numpy for the im2col below.
        pix_i8 = (
            torch.clamp(torch.round(pixel_values[0].float() / self._pix_scale), -127, 127)
            .to(torch.int8)
            .numpy()
        )
        c, h, w = pix_i8.shape
        p = self.patch_size
        gh, gw = h // p, w // p
        # (C, gh, kh, gw, kw) -> (gh, gw, C, kh, kw) -> (num_patches, C*P*P), flatten (c, kh, kw).
        cols = pix_i8.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, c * p * p)
        k = c * p * p
        k_pw = ceil(k / 64)
        data = np.zeros((1, gh * gw, k_pw * 64), np.int8)
        data[0, :, :k] = cols  # remaining K lanes stay zero (matches the padded patch weight)
        return data.reshape(1, gh * gw, k_pw, 64)

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> np.ndarray:
        """pixel_values (1,C,H,W) -> (1, seq_real, D) in one dispatch. D = hidden features by
        default (matching HF Dinov2Model.last_hidden_state); with an on-device seg head, D =
        n_classes per-token LOGITS (token 0 is the CLS row; patch tokens are [1:])."""
        pixels = self._quantize_pixels(pixel_values)
        out = self.runtime.run("encode", {"pixels": pixels, "attn_mask": self._mask})
        if self.seg_head:
            return out.reshape(1, self.seq_pad, self.n_cls_pad)[
                :, : self.seq_real, : self.n_classes
            ]
        return out.reshape(1, self.seq_pad, self.hidden)[:, : self.seq_real]

    @torch.no_grad()
    def reference(self, pixel_values: torch.Tensor) -> np.ndarray:
        """HF fp last_hidden_state for the same input (the fidelity reference)."""
        return self.hf_model(pixel_values=pixel_values).last_hidden_state.float().numpy()


def _artifact_key(
    repo_id,
    image_size,
    cores,
    n_calib,
    smooth_alpha,
    enable_buffer_promotion,
    calib_method="minmax",
    percentile=99.99,
    dpu_unroll=False,
    seg_head=False,
):
    """Short hash of everything that changes the compiled+calibrated artifact. Calibration is
    deterministic (fixed calib images, deterministic PTQ), so identical inputs -> identical
    artifact -> safe to reuse."""
    sig = f"{repo_id}|{image_size}|{cores}|{n_calib}|{smooth_alpha}|{int(enable_buffer_promotion)}"
    if calib_method != "minmax":  # keep the default key stable for existing artifacts
        sig += f"|{calib_method}|{percentile}"
    if dpu_unroll:  # compile-only flag; append only when set so existing keys stay stable
        sig += "|dpu_unroll"
    if seg_head:  # bakes the ADE20K seg head into the graph (output = class logits)
        sig += "|seg_head"
    return f"{repo_id.split('/')[-1]}-{hashlib.sha1(sig.encode()).hexdigest()[:12]}"


def default_cache_root() -> Path:
    """Where reusable DINOv2 artifacts live (override with $AXELERA_DINO_CACHE). Outside the repo
    so a rebuild never touches tracked files."""
    return Path(os.environ.get("AXELERA_DINO_CACHE", Path.home() / ".cache" / "axelera_dino"))


def load_dino_device(
    repo_id: str = "facebook/dinov2-base",
    *,
    cache_dir=None,
    cores=4,
    n_calib=256,
    image_size=None,
    smooth_alpha: float | None = 0.5,
    enable_buffer_promotion=True,
    num_sub_devices=None,
    calib_device=None,
    want_fp=False,
    force=False,
    calib_method: str = "minmax",
    percentile: float = 99.99,
    dpu_unroll: bool = False,
    seg_head: bool = False,
):
    """Get a runnable DinoV2Device, reusing a cached compiled artifact when one exists.

    seg_head=True bakes Meta's ADE20K linear segmentation head into the graph, so `encode` returns
    per-token class LOGITS (pixels -> segmentation in one dispatch) instead of features.

    On a cache MISS (or force=True) this calibrates + compiles the single-dispatch graph into a
    keyed cache dir and writes meta.json. On a cache HIT it reloads the artifact directly -- no
    calibration, no compile -- so repeated demo/eval runs pay the multi-minute build only once.

    Returns (dev, hf_model): hf_model is the fp reference (needed for accuracy/PCA-basis/--show-fp)
    -- loaded (uncalibrated, cheap) on a cache hit only if want_fp; None otherwise.
    """
    from transformers import Dinov2Model

    key = _artifact_key(
        repo_id,
        image_size,
        cores,
        n_calib,
        smooth_alpha,
        enable_buffer_promotion,
        calib_method,
        percentile,
        dpu_unroll,
        seg_head,
    )
    cache_dir = Path(cache_dir) if cache_dir else default_cache_root() / key
    hit = not force and (cache_dir / "model.json").exists() and (cache_dir / "meta.json").exists()

    if hit:
        print(f"[dino cache] reusing artifact {cache_dir}")
        dev = DinoV2Device.load(cache_dir, cores=cores, num_sub_devices=num_sub_devices)
        hf_model = Dinov2Model.from_pretrained(repo_id).eval() if want_fp else None
        return dev, hf_model

    print(f"[dino cache] building artifact -> {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_model, model, config, seq_real = load_dinov2(
        repo_id,
        n_calib=n_calib,
        image_size=image_size,
        smooth_alpha=smooth_alpha,
        calib_device=calib_device,
        calib_method=calib_method,
        percentile=percentile,
        seg_head=seg_head,
    )
    seg_w = seg_b = seg_fs = None
    if seg_head:
        from .seg_demo import folded_head  # ADE20K head, BN folded into the linear

        seg_w, seg_b = folded_head()
        seg_fs = float(model.seg_feat_scale)
        print(f"[dino seg] on-device head: {seg_w.shape[0]} classes, feat_scale={seg_fs:.4g}")
    dev = DinoV2Device(
        hf_model,
        model,
        config,
        seq_real,
        cache_dir,
        cores=cores,
        num_sub_devices=num_sub_devices,
        enable_buffer_promotion=enable_buffer_promotion,
        dpu_unroll=dpu_unroll,
        seg_head_w=seg_w,
        seg_head_b=seg_b,
        seg_feat_scale=seg_fs,
    )
    return dev, hf_model
