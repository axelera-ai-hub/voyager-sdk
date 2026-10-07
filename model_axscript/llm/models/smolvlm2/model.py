# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM2 vision encoder on device: host front-end + fused SigLIP graph + connector.

Adapted from ``models/dino/model.py``. The (27x identical) SigLIP encoder is
authored as ONE single-pass device graph and invoked in a single dispatch; the
non-repeated front-end (int8 pixel quantize + space-to-depth im2col) and the
pixel-shuffle connector stay on the host boundary.

  * VISION GRAPH (one dispatch): patch-embed (kind "embed") -> N fused blocks
    (kind "block") -> post-LayerNorm (kind "postnorm"). The activation threads
    inputs_embeds -> hidden_a -> hidden_b -> ... -> output, ping-ponging the two
    spine slots so depth costs 2 live buffers rather than one per block; the int8
    padding mask is a single shared block input; each kernel's constants are its
    own outlined weights.
  * HOST: SigLIP's num_patches (e.g. 729 for SO400M @384) is not a multiple of 64,
    so the sequence is padded up to one and the padded key columns are masked out
    of attention (int8 -127). Real query rows attend only to real keys, so their
    outputs match an unpadded run; the padded rows are sliced off before the
    connector. Calibration happens at the real (unpadded) length.
  * CONNECTOR (second dispatch): host pixel-shuffle (space-to-depth by
    scale_factor) + int8 quantize of the vision features, then one device matmul
    into the text hidden size (see connector_impl for why the reshape is host-side).

Device-only (graph executor). This does NOT use the LLM KV executor (a bidirectional
encoder has no KV cache / prefill-decode phases); it is a plain host driver over two
single-graph kernels -- the low-risk reuse boundary, exactly like dino.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import connector_impl as _conn
from . import vision_impl as _vis
from ...graph_descriptor import KindContract, author_graph_model
from ...layout import extract_layer_buffer_spec
from .vision_impl import _round_up

# int8 masked-key fill: exp-negligible in the SDPA score space (the op's own int8 mask
# sentinel), so a padded key gets ~0 softmax weight. Matches the qwen/dino convention.
_MASK_NEG = -127


def _generation_name(generation) -> str:
    """The lower-case generation name author_graph_model stamps into the native module.

    The runtime refuses a module whose stamp does not match the card ("hardware
    generation mismatch (runtime is europa, module built for omega)"), so this has to
    travel with the kernels' AltoConfig generation and not default separately.
    """
    name = generation.name if hasattr(generation, "name") else str(generation)
    return name.lower()


def build_vision_graph(
    vision_model,
    vision_config,
    out_dir,
    *,
    seq_len: int | None = None,
    enable_buffer_promotion: bool = True,
    cores: int = 1,
    generation: str = "OMEGA",
):
    """Author the SigLIP vision tower as ONE device graph: patch-embed (kind
    "embed", input = host-quantized pixels) -> N fused blocks (kind "block") ->
    post-LayerNorm (kind "postnorm"). Returns (out_dir, specs) with the
    {"embed","block","postnorm"} buffer specs (for the host pixel/mask layout).
    Device-only.
    """
    from axelera.compiler.alto.compiler.config import Target

    out_dir = Path(out_dir)
    # Always authored at FULL depth: a truncated block list still compiles, loads and runs,
    # just numerically wrong. To bisect kinds, slice this in a scratch copy, never behind a
    # flag a real build can pick up.
    blocks = list(vision_model.encoder.blocks)
    L = len(blocks)
    seq = int(seq_len if seq_len is not None else _round_up(vision_config.num_patches, 64))

    def _compile(name, make_kernel, ncores):
        _, mj = _vis.compile_kernel(
            vision_config,
            out_dir / f"_{name}",
            make_kernel=make_kernel,
            target=Target.DEVICE,
            seq_len=seq,
            enable_buffer_promotion=enable_buffer_promotion,
            cores=ncores,
            # L2 follows the connection all three kinds share, not the kernel width.
            connection_cores=cores,
            generation=generation,
        )
        atex = sorted((out_dir / f"_{name}").rglob("model.atex"))
        if not atex:
            raise RuntimeError(f"no model.atex under {out_dir / ('_' + name)}")
        return mj, atex[0], extract_layer_buffer_spec(mj)

    # embed authors the same program at any core count; block and postnorm shard on it.
    embed_mj, embed_atex, embed_spec = _compile("embed", _vis.make_siglip_embed_kernel, 1)
    _, block_atex, block_spec = _compile("block", _vis.make_siglip_block_kernel, cores)
    _, pn_atex, pn_spec = _compile("postnorm", _vis.make_siglip_postnorm_kernel, cores)

    kinds = ["embed"] + ["block"] * L + ["postnorm"]
    contracts = {
        "embed": KindContract(shared_inputs=frozenset({"pixels"}), state=frozenset()),
        "block": KindContract(shared_inputs=frozenset({"attn_mask"}), state=frozenset()),
        "postnorm": KindContract(shared_inputs=frozenset(), state=frozenset()),
    }
    layer_buffers = {0: _vis.embed_buffers(vision_model, embed_spec)}
    for i, block in enumerate(blocks):
        layer_buffers[i + 1] = _vis.block_buffers(block, block_spec)
    layer_buffers[L + 1] = _vis.postnorm_buffers(vision_model, pn_spec)

    author_graph_model(
        out_dir,
        "smolvlm2_vision",
        {
            ("encode", "embed"): embed_atex,
            ("encode", "block"): block_atex,
            ("encode", "postnorm"): pn_atex,
        },
        kinds,
        contracts,
        layer_buffers,
        threaded_inputs=frozenset({"hidden_states"}),
        generation=_generation_name(generation),
    )
    return out_dir, {"embed": embed_spec, "block": block_spec, "postnorm": pn_spec}


def build_connector_graph(
    connector,
    config,
    num_tokens,
    out_dir,
    *,
    enable_buffer_promotion: bool = True,
    cores: int = 1,
    generation: str = "OMEGA",
):
    """Author the pixel-shuffle connector projection as a single-dispatch device
    graph (kind "connector"): one bias-free int8 matmul from the host-shuffled
    vision features into the text hidden size. Returns (out_dir, spec).

    The KERNEL is single-core: it is one matmul over num_tokens (64-81) rows, ~0.4% of
    the encoder's MACs, so sharding it would buy ~0.3% of vision latency and is not worth
    a second kernel to validate. ``cores`` is therefore NOT the kernel width -- it is how
    many sub-devices the shared connection owns, which is what sets the L2 budget.
    """
    from axelera.compiler.alto.compiler.config import AltoConfig, HardwareGeneration, Target
    from axelera.compiler.alto.compiler.driver import compile as alto_compile

    out_dir = Path(out_dir)
    kdir = out_dir / "_connector"
    mod = _conn.make_connector_kernel(config, int(num_tokens))
    alto_config = AltoConfig(
        output_dir=kdir,
        generation=(HardwareGeneration[generation] if isinstance(generation, str) else generation),
        target=Target.DEVICE,
        num_cores=1,
        # L2 share is a property of the CONNECTION, not of the kernel's core count: this
        # graph loads onto the vision runtime's connection (share_from=vision_rt), which
        # claimed `cores` sub-devices -- so it may promote into cores/NUM_AI_CORES of the
        # arena even though the kernel itself runs on one core. See
        # vision_impl._l2_resource_fraction for why the 1.0 default over-budgets.
        resources=_vis._l2_resource_fraction(cores, generation),
        enable_tiling_injection=True,
        enable_buffer_promotion=enable_buffer_promotion,
        enable_llm_support=True,
        assign_unique_input_pools=True,
        unpack_axm_archive=True,
        clean_output_dir=True,
        emit_atex_artifacts=True,
        emit_alto_config=True,
    )
    alto_compile(mod, alto_config)
    mjs = sorted(kdir.rglob("model.json"))
    atex = sorted(kdir.rglob("model.atex"))
    if not mjs or not atex:
        raise RuntimeError(f"no connector artifact under {kdir}")
    spec = extract_layer_buffer_spec(mjs[0])

    author_graph_model(
        out_dir,
        "smolvlm2_connector",
        {("project", "connector"): atex[0]},
        ["connector"],
        {"connector": KindContract(shared_inputs=frozenset(), state=frozenset())},
        {0: _conn.connector_buffers(connector, config, spec)},
        threaded_inputs=frozenset({"hidden_states"}),
        generation=_generation_name(generation),
    )
    return out_dir, spec


class SiglipVisionDevice:
    """The SmolVLM2 vision path on device: the fused SigLIP encoder as ONE dispatch
    (patch-embed + N blocks + post-LayerNorm) followed by the host pixel-shuffle +
    a connector matmul dispatch. ``encode(pixel_values)`` returns image_embeds in the
    TEXT hidden size (matching HF SmolVLMConnector output); ``encode_vision`` returns
    the raw vision features; ``reference`` / ``reference_embeds`` give the fp
    comparison via the SmolVLM2Reference vision path. Device-only (graph executor).
    """

    def __init__(
        self,
        vision_model,
        connector,
        config,
        seq_real,
        out_dir,
        *,
        reference_model=None,
        cores: int = 1,
        num_sub_devices=None,
        # On by default, matching what a build does (multicore_impl.buffer_promotion
        # resolves GenerationConfig's None before it reaches here).
        enable_buffer_promotion: bool = True,
        generation: str = "OMEGA",
        open_device: bool = True,
    ):
        self.reference_model = reference_model  # fp SmolVLM2Reference for fidelity (optional)
        self.vision_model = vision_model
        self.connector = connector
        self.config = config
        self.vision_config = config.vision_config
        self.scale_factor = int(config.scale_factor)
        self.seq_real = int(seq_real)
        self.seq_pad = _round_up(self.seq_real, 64)
        self.hidden = int(self.vision_config.hidden_size)
        self.text_hidden = int(config.text_config.hidden_size)
        self.patch_size = int(self.vision_config.patch_size)
        self.num_tokens = self.seq_real // (self.scale_factor**2)
        self.cores = int(cores)

        out_dir = Path(out_dir)
        vis_dir = out_dir / "vision"
        conn_dir = out_dir / "connector"
        _, vspecs = build_vision_graph(
            vision_model,
            self.vision_config,
            vis_dir,
            seq_len=self.seq_pad,
            enable_buffer_promotion=enable_buffer_promotion,
            cores=cores,
            generation=generation,
        )
        _, cspec = build_connector_graph(
            connector,
            config,
            self.num_tokens,
            conn_dir,
            enable_buffer_promotion=enable_buffer_promotion,
            cores=cores,  # sub-devices on the shared connection, not the kernel width
            generation=generation,
        )
        self.vision_dir = vis_dir
        self.connector_dir = conn_dir
        self.block_spec = vspecs["block"]
        d = _conn._dims(config)
        self.conn_in_pw = d["in_pw"]
        self.conn_out_pad = d["out_pad"]
        self._mask_shape = tuple(self.block_spec["attn_mask"][0])
        self._mask = self._padding_mask()

        # Host int8 pixel quant scale: the patch-embed's calibrated (symmetric) activation scale.
        self._pix_scale = float(
            vision_model.patch_embed.proj.act_observer.qparams()[0].reshape(-1)[0]
        )
        # Connector input quant. Must be the EXPORTED reciprocal scale 1/(smooth*act_scale),
        # per-in-channel -- the same qs the device matmul's ds/dz were built against. The raw
        # act_observer scale is the wrong convention AND omits the SmoothQuant fold (the proj
        # is in a SmoothQuant group), which silently costs a gain error; see quantize_shuffled.
        from ...export import export_linear_quant_params

        _cp = export_linear_quant_params(connector)["proj"]
        self._conn_qs = _cp["qs"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        self._conn_qz = float(_cp["qz"].reshape(-1)[0])
        self._conn_qmin = int(_cp["qmin"])
        self._conn_qmax = int(_cp["qmax"])
        self._write_meta()

        # A COMPILE needs no device, and `open_device=False` is how it says so. Everything
        # above this point -- both graphs, the host quant params, meta.json -- is produced by
        # the compiler and numpy alone; only the two runtimes below want a part. A build that
        # opened them anyway could not run on a machine with no AIPU attached, which is where
        # most builds happen: calibration is hours of CPU (or GPU) and the artifact is carried
        # to the board afterwards. `load()` is the path that DOES open a compiled directory.
        self.vision_rt = None
        self.connector_rt = None
        if not open_device:
            return

        from ...generate import RuntimeModel

        # num_sub_devices is left to RuntimeModel unless the caller named one: it derives
        # the reservation from the build's own call_graphs (`shared.runtime.required_cores`),
        # which is this tree's rule -- a build that needs more cores is a COMPILE-time
        # decision (shard it), never a bigger reservation.
        self.vision_rt = RuntimeModel(vis_dir / "model.json", num_sub_devices=num_sub_devices)
        # Share the vision connection: one device is claimed whole by one connection, so
        # the connector graph loads onto vision_rt's connection instead of opening a second.
        self.connector_rt = RuntimeModel(conn_dir / "model.json", share_from=self.vision_rt)

    def close(self) -> None:
        """Release both runtimes. The connector shares the vision connection, so it goes
        first; closing is idempotent, so a partly-built tower can be closed too."""
        for attr in ("connector_rt", "vision_rt"):
            runtime = getattr(self, attr, None)
            if runtime is not None:
                runtime.close()
                setattr(self, attr, None)

    def __enter__(self) -> "SiglipVisionDevice":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _write_meta(self) -> None:
        meta = {
            "seq_real": self.seq_real,
            "seq_pad": self.seq_pad,
            "hidden": self.hidden,
            "text_hidden": self.text_hidden,
            "patch_size": self.patch_size,
            "scale_factor": self.scale_factor,
            "num_tokens": self.num_tokens,
            "pix_scale": self._pix_scale,
            # qs is per-in-channel (in_dim floats), so it rides in a sidecar .npy rather
            # than bloating meta.json.
            "conn_qs_file": "conn_qs.npy",
            "conn_qz": self._conn_qz,
            "conn_in_pw": self.conn_in_pw,
            "conn_out_pad": self.conn_out_pad,
            "conn_qmin": self._conn_qmin,
            "conn_qmax": self._conn_qmax,
            "mask_shape": list(self._mask_shape),
            # The connection must claim one sub-device per compiled core; load() has no
            # other way to know the build-time core count.
            "cores": self.cores,
        }
        np.save(self.vision_dir / "conn_qs.npy", np.asarray(self._conn_qs, dtype=np.float32))
        (self.vision_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def _padding_mask(self) -> np.ndarray:
        """int8 1-head-broadcast padding mask: 0 for real keys, -127 for padded keys."""
        shape = self._mask_shape  # (1, 1, seq_pad, ctx_blocks, 64) int8
        ctx_blocks = shape[-2]
        keys = np.arange(ctx_blocks * 64).reshape(ctx_blocks, 64)
        pad = np.where(keys < self.seq_real, 0, _MASK_NEG).astype(np.int8)
        return np.broadcast_to(pad, shape).copy()

    @classmethod
    def load(cls, out_dir, *, num_sub_devices=None, share_from=None):
        """Reopen a prebuilt vision device from its compiled artifacts + meta.json,
        WITHOUT torch / recompilation (mirrors dino's DinoV2Device.load). ``out_dir``
        is the directory a prior ``SiglipVisionDevice(...)`` wrote (holds ``vision/``
        and ``connector/`` graph dirs; ``vision/meta.json`` carries the host scalars).
        """
        from ...generate import RuntimeModel

        out_dir = Path(out_dir)
        self = cls.__new__(cls)
        self.reference_model = None
        self.vision_dir = out_dir / "vision"
        self.connector_dir = out_dir / "connector"
        meta = json.loads((self.vision_dir / "meta.json").read_text())
        self.seq_real = int(meta["seq_real"])
        self.seq_pad = int(meta["seq_pad"])
        self.hidden = int(meta["hidden"])
        self.text_hidden = int(meta["text_hidden"])
        self.patch_size = int(meta["patch_size"])
        self.scale_factor = int(meta["scale_factor"])
        self.num_tokens = int(meta["num_tokens"])
        self._pix_scale = float(meta["pix_scale"])
        self._conn_qs = np.load(self.vision_dir / meta["conn_qs_file"])
        self._conn_qz = float(meta["conn_qz"])
        self.conn_in_pw = int(meta["conn_in_pw"])
        self.conn_out_pad = int(meta["conn_out_pad"])
        self._conn_qmin = int(meta["conn_qmin"])
        self._conn_qmax = int(meta["conn_qmax"])
        self._mask_shape = tuple(meta["mask_shape"])
        self.cores = int(meta.get("cores", 1))  # pre-"cores" artifacts were single-core
        self._mask = self._padding_mask()
        # share_from lets the whole vision pipeline share an already-open connection (e.g.
        # the text decoder's), since one device is claimed whole by one connection.
        if share_from is not None:
            self.vision_rt = RuntimeModel(self.vision_dir / "model.json", share_from=share_from)
        else:
            # None means "derive it from the build" (`required_cores`), which is what this
            # tree reserves on; meta.json's `cores` is kept for the banner, not the claim.
            self.vision_rt = RuntimeModel(
                self.vision_dir / "model.json", num_sub_devices=num_sub_devices
            )
        # Share the vision connection (one device == one connection; see __init__).
        self.connector_rt = RuntimeModel(
            self.connector_dir / "model.json", share_from=self.vision_rt
        )
        return self

    def encode_vision(self, pixel_values) -> np.ndarray:
        """pixel_values (1,C,H,W) -> device vision features (1, num_patches, hidden)
        in one dispatch (patch-embed + blocks + post-LayerNorm), padded rows sliced off.

        The tile is checked against the geometry the graph was COMPILED at. Nothing below
        this line would say so: the host im2col happily produces however many patches the
        pixels give it, `RuntimeModel` orders inputs by name without comparing shapes, and
        the output buffer is allocated from the graph's own spec -- so a tile of the wrong
        size reads as a plausible embedding rather than as a mistake. The way to get one is
        a processor from a different checkpoint than the artifact (the family tiles at
        512/16 and 384/14), which is a load-time mix-up, not a bad picture.
        """
        _, _, height, width = pixel_values.shape
        patches = (height // self.patch_size) * (width // self.patch_size)
        if patches != self.seq_real:
            raise ValueError(
                f"this tower was compiled for {self.seq_real} patches "
                f"({self.patch_size}px) and a {height}x{width} tile gives {patches}: the "
                "pictures are being tiled by a processor that does not match the artifact "
                "(check --repo-id against the checkpoint the artifact was built from)"
            )
        if self.vision_rt is None:
            raise RuntimeError(
                "this tower was constructed to COMPILE (open_device=False), so it holds no "
                "device runtimes and cannot encode. Reopen the directory it wrote with "
                "`SiglipVisionDevice.load(out_dir)`, which is the run path."
            )
        pixels = _vis._quantize_pixels(pixel_values, self._pix_scale, self.patch_size)
        out = self.vision_rt.run("encode", {"pixels": pixels, "attn_mask": self._mask})
        return out.reshape(1, self.seq_pad, self.hidden)[:, : self.seq_real]

    def encode(self, pixel_values) -> np.ndarray:
        """pixel_values (1,C,H,W) -> image_embeds (1, num_tokens, text_hidden): the
        device vision features, host pixel-shuffled + int8-quantized, then projected
        by the connector matmul dispatch. Matches HF SmolVLMConnector output."""
        feats = self.encode_vision(pixel_values)  # (1, num_patches, hidden) fp
        shuffled = _conn.pixel_shuffle_np(feats, self.scale_factor)  # (1, num_tokens, in_dim)
        q = _conn.quantize_shuffled(
            shuffled, self._conn_qs, self._conn_qz, self._conn_qmin, self._conn_qmax
        )
        emb = self.connector_rt.run("project", {"inputs_embeds": q})
        return emb.reshape(1, self.num_tokens, self.conn_out_pad)[..., : self.text_hidden]

    def reference(self, pixel_values) -> np.ndarray:
        """fp SigLIP vision last_hidden_state (post-LayerNorm output) for the same
        input, via the SmolVLM2Reference vision path (the vision-encoder fidelity ref)."""
        import torch

        with torch.no_grad():
            return self.reference_model.vision_model(pixel_values).float().numpy()

    def reference_embeds(self, pixel_values) -> np.ndarray:
        """fp connector output (image_embeds in text hidden) for the same input --
        the fidelity reference for ``encode``."""
        import torch

        with torch.no_grad():
            feats = self.reference_model.vision_model(pixel_values)
            return self.reference_model.connector(feats).float().numpy()
