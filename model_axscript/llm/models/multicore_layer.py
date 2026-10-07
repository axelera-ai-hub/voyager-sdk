# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Shared multicore-capable decoder-layer backbone for the Metis "multicore" impls.

Backs the qwen3 / llama "multicore" implementations (registered as
``qwen3-metis-multicore`` / ``llama-metis-multicore``). Shares the host-side contract
(:class:`MulticoreDecoderImplementation`: per-step input prep, RoPE trig, the kernel
list), the per-layer size resolution (:func:`layer_dims`), the fused norm/quantize
helpers, and the qparam interleave packer.

The kernel op *sequence* lives in each model's own ``make_*_layer_kernel`` rather than
here, for two reasons the TVMScript parser forces: (1) qwen3 has per-head q/k RMSNorm
and so declares extra q_norm/k_norm buffers in its ``main`` signature that llama lacks;
(2) the parser names buffers from the assignment targets in the parsed ``main`` body, so
authoring the sequence in ``main`` (not a plain helper) is what gives the KV-cache outputs
their ``k_out``/``v_out`` names that the state contract depends on. The two bodies are
otherwise near-identical and call the shared helpers below.

Convergence (qwen3 optimizations applied to both): int8 attn_mask, conditional
hidden-norm dispatch (dpu_reduce if it fits the DPU register file, else Kahan), and core
SDPA for both phases (the decode-specialized sdpa_decode is dropped). Multicore (SDPA
head-shard + MLP N-shard) is driven purely by ``cores > 1`` -- no env gating.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import numpy as np

from ..graph_descriptor import fused_graph_name, split_sublayer
from ..implementation import HostKernel, LayerKernel, UniformDecoderImplementation

PWORD = 64


def upper_shard_scope(generation=None):
    """Region for a K-split weight's upper shard: the second DDR where the target has one.

    Omega has a single region, so both shards land in it and stream on the two DMA
    channels instead of the two regions.

    ``generation`` should be passed by any builder that wants the split to actually happen,
    rather than left to the ambient AltoConfig. ``_compile_phase`` evaluates
    ``kernel.build(...)`` as an ARGUMENT to alto_compile, so the trace does not see that
    config directly; it reads the one ``alto_config_context`` publishes around the build.
    Relying on that is a two-step dependency which, when the wrap was briefly absent, read
    the OMEGA default and silently put every upper shard in the same region as its lower
    half. So builders declare a ``generation`` parameter and ``_compile_phase`` passes the
    target through explicitly -- correct either way, and independent of the ambient value.
    """
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.compiler.config import get_alto_config
    from axelera.compiler.alto.compiler.device import AVAILABLE_MEMORIES, MemoryScope

    supported = AVAILABLE_MEMORIES[generation or get_alto_config().generation]
    return A.DDR1 if MemoryScope.DDR1 in supported else A.DDR


def ddr1_unsharded_spec(spec, weight_names):
    """The spec a packer should pack against: each DDR1-split weight restored to its full K.

    ddr1_split_shards cuts the packed result afterwards, so the packer itself needs no
    knowledge of the split.
    """
    full = dict(spec)
    for name in weight_names:
        shard = f"{name}1"
        if name in spec and shard in spec:
            (n_blocks, k_lo, lane), dtype = spec[name]
            k_hi = spec[shard][0][1]
            full[name] = ((n_blocks, k_lo + k_hi, lane), dtype)
    return full


def ddr1_split_shards(buffers, spec, weight_names) -> None:
    """Cut each packed NK weight (out/64, in, 64) along K into its DDR/DDR1 shards.

    No-op for a weight whose kernel didn't declare a "<name>1" buffer, so calling this
    unconditionally from a shared packer is safe for callers that never split.
    """
    for name in weight_names:
        shard = f"{name}1"
        if name not in buffers or shard not in spec:
            continue
        k_lo = spec[name][0][1]
        packed = buffers[name]
        buffers[name] = packed[:, :k_lo].contiguous()
        buffers[shard] = packed[:, k_lo:].contiguous()


# Max hidden-norm D_blocks (hidden_size/64) whose resident input vectors fit the ~72 usable A+B
# DPU registers less the norm's ~4 constants, so 64 has no headroom. Above it: the Kahan norm.
DPU_REDUCE_MAX_D_BLOCKS = 64


@dataclass(frozen=True)
class NormFold:
    """A hidden RMSNorm whose affine weight the packer folds into the quantize scale after it.

    ``qparams`` names the kernel buffer holding that scale and ``site`` the norm's
    per-layer path in the bundle; hidden_norm_quant checks one against the other, so
    renaming a half is a trace-time error rather than a dropped weight.
    """

    qparams: str
    site: str


# hidden_norm_quant carries no weight buffer: the layer packer folds each hidden RMSNorm's
# affine weight into the following activation-quantize scale.
INPUT_NORM_FOLD = NormFold("q_proj_qparams", "input_layernorm")
POST_ATTN_NORM_FOLD = NormFold("gate_proj_qparams", "post_attention_layernorm")
HIDDEN_NORM_FOLDS = (INPUT_NORM_FOLD, POST_ATTN_NORM_FOLD)


def check_norm_folds(buffers: dict, folds=HIDDEN_NORM_FOLDS) -> None:
    """Assert every declared fold produced a buffer, since a missing one silently drops a norm."""
    missing = [f.qparams for f in folds if f.qparams not in buffers]
    if missing:
        raise ValueError(
            f"hidden RMSNorm fold targets {missing} are absent from the packed buffers; the "
            "kernel and the fold table disagree, so the norms' affine weights would be dropped"
        )


@dataclass(frozen=True)
class ModelSpec:
    """The small per-model divergences the shared backbone is parameterized over."""

    has_qk_norm: bool  # qwen3: True (per-head q/k RMSNorm before RoPE); llama: False
    has_lm_head: bool = False  # llama appends a host lm_head kernel; qwen3 does not
    dpu_reduce_max_d_blocks: int = DPU_REDUCE_MAX_D_BLOCKS


def layer_dims(
    config,
    *,
    head_dim_pw: int,
    logical_head_dim: int,
    ctx_len: int,
    phase: str,
    chunk_len: int | None,
    spec: ModelSpec,
) -> SimpleNamespace:
    """Resolve the per-layer sizes the kernel builders need.

    ``head_dim_pw`` and ``logical_head_dim`` are the only model-specific layout inputs
    (qwen3: ``head_dim//64`` and ``head_dim``; llama: design-P ``head_dim_pw`` and the
    logical 64). The padded head dim is ``head_dim_pw * 64``; the attention ``scale``
    uses the *logical* dim (the padding lanes are zeros).
    """
    hidden_dim = config.hidden_size
    hidden_dim_pw = hidden_dim // 64
    padded_head_dim = head_dim_pw * 64
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    intermediate_dim = config.intermediate_size

    use_dpu_hidden_norm = hidden_dim_pw <= spec.dpu_reduce_max_d_blocks
    seq_len = 1 if phase == "decode" else (chunk_len if chunk_len else ctx_len)

    return SimpleNamespace(
        hidden_dim=hidden_dim,
        hidden_dim_pw=hidden_dim_pw,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_pw=head_dim_pw,
        padded_head_dim=padded_head_dim,
        logical_head_dim=logical_head_dim,
        q_out_dim=num_q_heads * padded_head_dim,
        q_out_dim_pw=(num_q_heads * padded_head_dim) // 64,
        kv_out_dim=num_kv_heads * padded_head_dim,
        kv_out_dim_pw=(num_kv_heads * padded_head_dim) // 64,
        intermediate_dim=intermediate_dim,
        intermediate_dim_pw=intermediate_dim // 64,
        context_len=ctx_len,
        seq_len=seq_len,
        use_dpu_hidden_norm=use_dpu_hidden_norm,
    )


def quantize_activation(x, qparams, leading, c_pw):
    """Fuse mul(scale)+add(zp)+cast-int8 into one llm.quantize datapath. ``x`` is
    (1, *leading, c_pw); ``qparams`` is the host-interleaved (2*c_pw,) [scale, zp, ...]
    per-channel buffer (on-device strided interleave does not lower). Reshapes are free
    views (datapath fusion is buggy, so the chain is fused manually). Called from the
    kernel ``main`` during tracing -- ops register into the active prim_func context."""
    from axelera.compiler.alto import axol

    from ..ops.quantize import quantize

    n = int(np.prod(leading))
    x2d = axol.atomic.reshape(x, shape=(n, c_pw))
    qp1 = axol.atomic.reshape(qparams, shape=(2 * c_pw,))
    q = quantize(x2d, qp1)
    return axol.atomic.reshape(q, shape=(1, *leading, c_pw))


def hidden_norm_quant(
    x, consts, qparams, dims, fold: NormFold, *, output=None, force_tiled: bool = False
):
    """RMSNorm over the hidden dim, result quantized to int8 for the next matmul (q/k/v
    or gate/up share one SmoothQuant group). Small hidden dims use the single-pass
    rms_norm_dpu_reduce with the quantize fused in-op (~3x faster, int8 out halves the
    write); large hidden dims would overflow the DPU register file, so they use the
    transpose-reduce Kahan norm + the streaming llm.quantize datapath.

    The norm has NO weight buffer on either path: both its affine weight and the quantize
    scale are per-channel constants, so the packer folds ``fold``'s weight into the scale
    (see interleave_qparams), freeing the D_blocks weight registers the DPU op used to hold.

    ``output`` and ``force_tiled`` are forwarded to rms_norm_dpu_reduce.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto.axol.common import BufferInfo

    from ..ops.rms_norm_dpu_reduce import rms_norm_dpu_reduce

    # An N-layer kernel suffixes the name with its sub-layer tag (graph_descriptor).
    if split_sublayer(qparams.name)[0] != fold.qparams:
        raise ValueError(
            f"the packer folds {fold.site!r} into {fold.qparams!r}, but this call passed "
            f"{qparams.name!r}; the fold is keyed by buffer name, so the two must agree"
        )

    if dims.use_dpu_hidden_norm:
        return rms_norm_dpu_reduce(
            x,
            consts,
            normalized_shape=(dims.hidden_dim,),
            quant_params=qparams,
            invsqrt_method="additive",
            output=BufferInfo(dtype="int8x64") if output is None else output,
            force_tiled=force_tiled,
        )
    if output is not None or force_tiled:
        raise ValueError(
            "output and force_tiled are the dpu_reduce norm's; the Kahan arm allocates its own"
        )
    # The Kahan path takes the LUT and eps as separate operands, so hand it the two slices
    # of the packed buffer it wants (see pack_norm_consts for the row order).
    n_luts = int(consts.shape[0]) - 2
    rms = axol.core.rms_norm_kahan(
        x,
        consts[0:n_luts],
        consts[n_luts + 1 : n_luts + 2],
        (dims.hidden_dim,),
        invsqrt_method="additive",
    )
    return quantize_activation(rms, qparams, (dims.seq_len,), dims.hidden_dim_pw)


def norm_consts(bundle, rng: dict, normalized_shape: tuple[int, ...], spec_entry):
    """Pack one norm's init_consts: its fitted inv-sqrt LUT, then 1/D, then eps.

    ``spec_entry`` sizes the buffer, so the LUT row count follows the kernel's declaration.
    """
    from ..layout import fit_lut
    from ..ops.rms_norm_dpu_reduce import pack_norm_consts

    (shape, dtype) = spec_entry
    n_luts = shape[0] - 2
    lut = fit_lut(rng, ((n_luts, *shape[1:]), dtype))
    return pack_norm_consts(lut, normalized_shape, float(bundle.host_scalars["rms_eps"]))


def add_norm_consts(buffers: dict, spec: dict, bundle, layer: int, sites: dict) -> None:
    """Pack each declared norm's init_consts, ``sites`` mapping its buffer name to
    (the RangeObserver site its LUT is fit over, the length of the axis it reduces)."""
    from ..layout import layer_subset

    rng = layer_subset(bundle.lut_ranges, layer)
    for name, (site, reduction_dim) in sites.items():
        if name not in spec:
            continue  # this kernel does not declare this norm's consts
        # A declared buffer with no recorded range used to be skipped silently. Nothing else
        # fills it, so the kernel kept declaring a buffer that was never packed and the
        # failure surfaced as a bare KeyError in the graph's blob loop, naming neither the
        # layer nor the observer. Loud here, like check_norm_folds beside it.
        if site not in rng:
            raise KeyError(
                f"layer {layer}: no recorded LUT range {site!r} for the declared norm "
                f"buffer {name!r} (recorded: {sorted(rng)}). Run record_lut_ranges over "
                "the calibration set, or correct the observer name."
            )
        buffers[name] = norm_consts(bundle, rng[site], (reduction_dim,), spec[name])


def interleave_qparams(qs, qz, spec_entry, norm_weight=None):
    """Interleave per-channel scale/zero-point into the [scale, zp, ...] qparams buffer
    the fused ops consume (the device can't do a strided interleave). ``spec_entry`` is
    the packer spec ((2*blocks, PWORD), dtype); zp=None pins zero (SDPA quants).
    ``norm_weight`` folds a preceding RMSNorm's affine weight into the scale so the norm
    kernel needs no weight buffer (see hidden_norm_quant)."""
    import torch

    from ..layout import pack

    blocks = spec_entry[0][0] // 2

    def _blocks(v):
        t = torch.as_tensor(v).float().reshape(-1)
        t = t.expand(blocks * PWORD) if t.numel() == 1 else t
        return t.reshape(blocks, PWORD)

    qs_b = _blocks(qs)
    if norm_weight is not None:
        qs_b = qs_b * _blocks(norm_weight)
    qz_b = torch.zeros(blocks, PWORD) if qz is None else _blocks(qz)
    qp = torch.empty(2 * blocks, PWORD)
    qp[0::2] = qs_b
    qp[1::2] = qz_b
    return pack(qp, spec_entry)


class MulticoreDecoderImplementation(UniformDecoderImplementation):
    """Base for the multicore Metis decoder implementations (qwen3 / llama).

    Subclasses set ``name``/``architecture``, ``spec``, ``layer_build`` (the model's
    ``make_*_layer_kernel``) and ``layer_buffers_fn`` (the model's buffer packer). The
    RoPE default here is standard (qwen3); llama overrides ``_trig`` for llama3 rescaling.
    """

    spec: ModelSpec = None
    layer_build = None
    layer_buffers_fn = None

    # One combined RoPE trig (q/k share it, sin||cos) + a 1-head broadcast int8 causal
    # mask, instead of 4 sin/cos inputs + a per-head fp32 mask.
    shared_inputs = frozenset({"attn_mask", "seq_pos", "rope_trig"})
    # Attention-only (KV cache is the sole threaded state): a partial last chunk can be
    # ingested as one padded prefill chunk rather than replayed token-by-token.
    chunked_remainder_via_padded_chunk = True

    #: A decode step costs about as much as 128 prefill tokens on both generations, so a
    #: padded chunk wins from chunk_len // 128 real tokens. Measured on the Qwen3 fused
    #: schedule and applied to every multicore artifact. Declared rather than keyed on
    #: `runtime.generation`: a build targets OMEGA or EUROPA and nothing else (build.py's
    #: `--generation` choices), so a branch answering both with 128 selects nothing while
    #: shadowing whatever ratio a subclass declares -- which is how this became dead code
    #: once already.
    #:
    #: The one Metis-side cross-check brackets lower and is not yet reconciled: the shipped
    #: smolvlm2-2.2b row (reports/llm/smolvlm2_2_2b_instruct.yaml -- this stack's text side,
    #: metis-0:1:0, ctx 2048, one full-context prefill) gives 4251.7 ms for 2048 positions
    #: against a 139 ms decode step, i.e. ~67 positions per step, and LFM2-VL measured 46
    #: for the same quantity (models/lfm2vl/text_impl.py REPLAY_COST_IN_POSITIONS). A family
    #: that measures its own Omega figure declares it and wins over this default.
    padded_chunk_breakeven_ratio = 128

    def _trig(self, positions, host_params):
        from ..implementation import make_rope_cos_sin

        cos, sin = make_rope_cos_sin(
            positions, int(host_params["head_dim"]), float(host_params["rope_theta"])
        )
        return np.ascontiguousarray(np.concatenate([sin, cos], axis=2))

    def prefill_inputs(self, host_params, token_ids, context_len):
        from ..implementation import make_causal_mask, make_seq_pos

        if len(token_ids) > context_len:
            raise ValueError(f"prompt length {len(token_ids)} exceeds context_len {context_len}")
        positions = np.arange(context_len)
        embeds = self.embed_tokens(np.array([token_ids]), host_params)
        pad = context_len - len(token_ids)
        if pad:
            embeds = np.pad(embeds, ((0, 0), (0, pad), (0, 0), (0, 0)))
        inputs = {
            "inputs_embeds": np.ascontiguousarray(embeds),
            "attn_mask": make_causal_mask(positions, 1, context_len, dtype=np.int8),
            "rope_trig": self._trig(positions, host_params),
            "seq_pos": make_seq_pos(0),
        }
        assert set(inputs) == {"inputs_embeds"} | set(self.shared_inputs)
        return inputs

    def _chunk_inputs(self, host_params, chunk_tokens, base_pos, context_len, chunk_len):
        """Chunked-prefill inputs for one chunk_len-query chunk starting at base_pos
        (combined rope_trig, 1-head int8 causal mask); last chunk is right-padded."""
        from ..implementation import make_causal_mask, make_seq_pos

        positions = np.arange(base_pos, base_pos + chunk_len)
        embeds = self.embed_tokens(np.array([chunk_tokens]), host_params)
        pad = chunk_len - len(chunk_tokens)
        if pad:
            embeds = np.pad(embeds, ((0, 0), (0, pad), (0, 0), (0, 0)))
        inputs = {
            "inputs_embeds": np.ascontiguousarray(embeds),
            "attn_mask": make_causal_mask(positions, 1, context_len, dtype=np.int8),
            "rope_trig": self._trig(positions, host_params),
            "seq_pos": make_seq_pos(base_pos),
        }
        assert set(inputs) == {"inputs_embeds"} | set(self.shared_inputs)
        return inputs

    def step_inputs(self, host_params, token_id, position, context_len):
        from ..implementation import make_causal_mask, make_seq_pos

        inputs = {
            "inputs_embeds": self.embed_tokens(np.array([[token_id]]), host_params),
            "attn_mask": make_causal_mask(position, 1, context_len, dtype=np.int8),
            "rope_trig": self._trig(position, host_params),
            "seq_pos": make_seq_pos(position),
        }
        assert set(inputs) == {"inputs_embeds"} | set(self.shared_inputs)
        return inputs

    @property
    def provides_device_head(self) -> bool:
        """A kind that already appends a host lm_head kernel would otherwise get two heads."""
        return not self.spec.has_lm_head

    def tail_kernel(self, config):
        """The device lm_head, unless the build opted out (config.device_lm_head=False).

        Decode phase only: the head is on the decode critical path, while prefill scores its
        logits on the host through compute_logits_seq, off that path.
        """
        if not self.uses_device_head(config):
            return None
        # Local import: lm_head_kernel reads layer_dims from here, so a module-level import
        # would close the cycle.
        from .lm_head_kernel import LM_HEAD_KIND, lm_head_buffers, make_lm_head_kernel

        return LayerKernel(
            LM_HEAD_KIND,
            # The head takes this implementation's norm arm, not the module default.
            build=partial(make_lm_head_kernel, spec=self.spec),
            layer_buffers=lm_head_buffers,
            phases=("decode",),
            enable_buffer_promotion=True,
        )

    def decode_step(self, runtime, host_params, token_id: int, position: int, context_len: int):
        """One decode step through the fused stack+head graph when the artifact has one,
        else the two-call path."""
        from .lm_head_kernel import LM_HEAD_KIND

        fused = fused_graph_name(self._decode_graph_name(), LM_HEAD_KIND)
        if not self.has_device_head(runtime) or fused not in getattr(runtime, "graph_names", ()):
            return super().decode_step(runtime, host_params, token_id, position, context_len)
        logits = runtime.run(fused, self.step_inputs(host_params, token_id, position, context_len))
        return np.asarray(logits, dtype=np.float32).reshape(-1)

    def kernels(self):
        ks = [
            HostKernel("embedding", params=("embed_weight",)),
            LayerKernel(
                "decode_layer",
                build=self.layer_build,
                layer_buffers=self.layer_buffers_fn,
                phases=("prefill", "decode"),
                shared_inputs=self.shared_inputs,
                state=frozenset({"k_out", "v_out"}),
                enable_buffer_promotion=True,
            ),
        ]
        if self.spec.has_lm_head:
            ks.append(
                HostKernel("lm_head", params=("final_norm_weight", "embed_weight", "rms_eps"))
            )
        return ks
