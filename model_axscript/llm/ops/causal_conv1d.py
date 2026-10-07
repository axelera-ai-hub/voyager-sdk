#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Depthwise causal 1D convolution -- the LFM2 short-conv token mixer (llm.causal_conv1d).

LFM2 (and the Qwen3.5 gated-DeltaNet input conv) mix tokens with a depthwise
causal 1D convolution: a per-channel FIR of width K (conv_L_cache, 3 for
LFM2) along the sequence axis, groups == channels so every channel has its own
K taps and there is no cross-channel mixing. For output timestep t and
channel c (matching torch.nn.Conv1d(..., padding=K-1)[..., :T]):

    out[t, c] = sum_{j=0}^{K-1} w[j, c] * x[t + j - (K-1), c]

with inputs at negative time coming from state (the last K-1 timesteps of
the previous chunk) -- zero for the first prefill chunk. state is updated in
place to the final K-1 inputs, so the caller persists the conv window across
invocations exactly like a KV cache (cf. axol.core.update_kv_cache); the single
in-place state buffer (rather than a separate state_out) is what lets the op
be a layer's persistent state cache. It therefore covers decode (T = 1,
window entirely from state) and prefill chunks uniformly. The conv is purely
linear: any activation/gating (LFM2's B*x / C* gates, the in_proj /
out_proj GEMMs) is the caller's job, done with the existing element-wise /
matmul ops.

Layout / packing
----------------
Channels are packed into the PWORD lanes (C = C_blocks * PWORD): every lane
is an independent conv channel, so each output is a per-lane FMA over the K taps
with no cross-lane reduction and no transpose -- the cheapest possible DPU
mapping. The weight is therefore channel-major (like rms_norm_dpu_reduce's
D-major weight, not broadcast): weight is (K, C_blocks) with lane j of
block cb holding channel cb*PWORD+j's tap. Pack the torch conv weight
((C, 1, K)) as weight[:, 0, :].T.reshape(K, C_blocks, PWORD).

Operand shapes (vectorized, *x64):

* input   (B, T, C_blocks)     -- the (already gated) conv input.
* weight  (K, C_blocks)        -- per-channel taps, channel-major.
* state   (B, K-1, C_blocks)   -- previous K-1 timesteps (oldest first);
  updated in place to the last K-1 inputs after the call.
* output  (B, T, C_blocks)     -- auto-inferred.

The T timesteps are tiled into chunks of <= _CONV_CHUNK (the conv has no
recurrence -- each output reads only a K-wide causal window of inputs -- so
chunks are independent), and one self-contained Datapath handles each (batch,
C_block, chunk). Within a chunk the K-1 + Tc window is held resident in DPU
registers and indexed statically (a rolling window does not persist across an
in-DPU serial loop). Tiling is what bounds the register footprint: a single resident
row of the whole T needs O(T) registers and overflows regalloc for large
prefill (T=256 -> 258 vectors). For each chunk a single IFD0 feeds the
(K-1 + Tc) window (halo + this chunk's inputs, staged into L1 -- the first chunk's
halo is the carried state, or zero for a fresh prefill) and IFD1 feeds the K
taps (one IFD0 read per datapath -- two hangs the device, see llm.gated_deltanet).
The new window is the last K-1 inputs -- pure data movement -- so it is written
into state with atomic.copy outside the datapath rather than as a second ODR.
DPU arithmetic is float18; emulate mirrors the op order with emu.rf18.

Two hardware paths, dispatched on the input dtype:

* float (float18x64 / float16x64 input): the DPU FIR described above.
* int8 (int8x64 input, weight, and state): a W8A8 conv on the DWPU. The same int8
  extended input xe = [halo ++ input] is fed through a depthwise VALID conv on the
  DWPU (int8 x int8 dot -> int32 accumulate), the DPU casts the int32 to float32,
  and the output is float32x64. The per-channel dequant scale (ds = s_x*s_w) and the
  activation-zero-point correction (dz) are applied by the CALLER (a per-channel
  scale/bias on the float32 output), so this op stays a plain integer conv. The state
  window is int8 and written back unchanged. See _implementation_dwpu / _emulate_dwpu.
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.core.dw_convolution_2d import dw_convolution_2d
from axelera.compiler.alto.axol.utils import to_int_tuple, to_tuple
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.dtypes import MetisFloats, MetisTypes
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901

# Storage width per lane, to address the window in bytes for the runtime function.
# float18x64 is deliberately absent: its packed device width is not one of these, so a
# valid_len write-back on the float path fails by name instead of copying the wrong span.
_DTYPE_BYTES = {"int8x64": 1, "float16x64": 2, "float32x64": 4}
PWORD = 64

# Chunk size for tiling the FIR along T so the DPU's resident window (K-1 + chunk)
# fits the register file. K-1 + 64 = 66 vectors is validated to allocate; a full-T
# resident row overflows for large prefill (e.g. T=256 -> 258 regs, regalloc fails).
# Applies only to the float DPU path; the int8 DWPU path tiles T itself.
#
# The register file is not the only bound: the same chunk sizes the L1 staging, which
# also scales with C_blocks, and that combination is what actually overflows. See
# _float_chunk.
_CONV_CHUNK = 64

#: L1 the float path's three staging buffers may take together. Below the 4 MB pool because
#: they are live alongside the rest of the layer's staging and the memory pass counts what
#: OVERLAPS, not what one op asks for -- the 9B failure had the conv at 4.22 MB and the pool
#: 1.05x over, so the rest came to only ~0.2 MB. 3.5 MB leaves that margin while still
#: admitting a 64-timestep chunk at 96 channel blocks (3.16 MB), which is what Qwen3.5-2B
#: compiles and was measured at; a tighter budget would silently halve its chunk.
_CONV_L1_BUDGET = 3584 * 1024


def _float_chunk(t: int, k: int, c_blocks: int) -> int:
    """Timesteps per float-path chunk: the largest that fits BOTH bounds, and divides T.

    _CONV_CHUNK bounds the DPU's resident window (register file). The L1 bound is the
    other half and is the one that bites as channels grow, since the staging is
    (K-1 + tc, C_blocks) + (K, C_blocks) + (tc, C_blocks) lane words -- linear in
    C_blocks, which _CONV_CHUNK alone does not see. At Qwen3.5-9B's 128 channel blocks a
    64-timestep chunk wants 4.22 MB and the compile fails in enforce_memory_constraints
    ("all tileable loops have been maximally tiled"), where the 2B's 96 blocks fit at
    3.16 MB. tc must divide T: the chunk loop is t // tc with no remainder handling.
    """
    from axelera.compiler.alto.compiler.device import PWORD

    per_timestep = c_blocks * PWORD * 4  # one lane word per channel block, fp32
    # 2*tc for the input window and the output, plus the K-1 halo and the K taps.
    budget_tc = (_CONV_L1_BUDGET // per_timestep - (k - 1) - k) // 2
    tc = min(t, _CONV_CHUNK, max(1, budget_tc))
    while t % tc:  # the chunk loop has no remainder handling
        tc -= 1
    return tc


class CausalConv1d(ScriptOp):
    """Depthwise causal 1D convolution with a persistent conv-window state.

    See the module docstring for the layout and state contract. Batched over
    (batch, C_blocks); the T timesteps are walked sequentially, carrying
    the K-1 vector window in registers.
    """

    input = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(B, T, C_blocks) conv input; channels packed in the lanes.",
    )
    weight = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(K, C_blocks) channel-major per-channel taps (lane = channel).",
    )
    state = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(B, K-1, C_blocks) previous K-1 timesteps (oldest first). Updated "
        "IN PLACE to the final window after T steps, so it persists across calls "
        "like a KV cache (cf. axol.core.update_kv_cache).",
    )

    output = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Conv output, auto-inferred (B, T, C_blocks) DDR.",
    )
    zero_initial_state = Arg(
        bool,
        default=False,
        kw=True,
        description="Seed the leading K-1 window with zeros instead of reading "
        "state -- the semantically-correct mode for the FIRST chunk (prefill from "
        "position 0, which has no history, == torch Conv1d left zero-padding). The "
        "persistent state cache is uninitialised on a fresh instance, so reading it "
        "for the first chunk corrupts the leading K-1 outputs; the final window is still "
        "written back to state for any decode continuation.",
    )
    state_scale = Arg(
        MetisFloats[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="Optional runtime (1, 1, 1) broadcast-scalar multiplicative gate "
        "on the carried K-1 window (broadcast over B/K-1/C_blocks, all PWORD lanes equal): "
        "1.0 continues the persistent state, 0.0 zeros it == zero_initial_state, so "
        "the host can start a fresh sequence per-call without recompiling or reloading "
        "the instance. This is the sequence-start signal chunked prefill / decode need "
        "to reset the accumulator between prompts (and the hook a future incremental "
        "chat sets only at true sequence boundaries). Overrides zero_initial_state "
        "when given; the window write-back is unaffected.",
    )
    valid_len = Arg(
        MetisTypes[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="Optional (1,) uint64x64 broadcast scalar: how many of the T input "
        "timesteps are REAL (the rest being right-pad). The window write-back then takes "
        "inputs [valid_len-(K-1) : valid_len) instead of the tail of the buffer, so a "
        "padded prefill chunk leaves the window on the last real tokens rather than on "
        "pad. Without it a chunk carrying N<T real tokens persists a window of pad, and "
        "the next decode step -- and every KV entry it writes -- is computed from it. "
        "None keeps the static tail copy; valid_len == T is equivalent to None.",
    )
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )

    # Shape helpers
    def _dims(self) -> tuple[int, int, int, int]:
        """(B, T, C_blocks, K) from the operand shapes.

        The float path's weight is (K, C_blocks); the int8/DWPU path's is
        dw_convolution_2d's native (C_blocks, K, 1, 1) (K on axis 1), packed
        channel-major on the host so no device-side transpose is needed.
        """
        b, t, c_blocks = to_int_tuple(self.input.shape)
        k = int(self.weight.shape[1]) if self._is_int8() else int(self.weight.shape[0])
        return b, t, c_blocks, k

    def _is_int8(self) -> bool:
        """Whether this is the W8A8 int8/DWPU conv (vs the float18/DPU conv)."""
        return self.input.dtype == "int8x64"

    def validate(self) -> None:
        if len(self.input.shape) != 3:
            raise ValueError(f"input must be (B, T, C_blocks); got {tuple(self.input.shape)}.")
        b, t, c_blocks, k = self._dims()
        if k < 2:
            raise ValueError(f"kernel size K must be >= 2; got {k}.")
        if self._is_int8():
            # int8/DWPU path: int8 operands, weight in dw_convolution_2d's native
            # (C_blocks, K, 1, 1) layout. The conv is delegated to
            # axol.core.dw_convolution_2d, which tiles the sequence itself.
            if to_int_tuple(self.weight.shape) != (c_blocks, k, 1, 1):
                raise ValueError(
                    f"int8 conv weight must be (C_blocks, K, 1, 1) = {(c_blocks, k, 1, 1)}; "
                    f"got {tuple(self.weight.shape)}."
                )
            if self.weight.dtype != "int8x64":
                raise ValueError(f"int8 conv weight must be int8x64; got {self.weight.dtype}.")
            if self.state.dtype != "int8x64":
                raise ValueError(f"int8 conv state must be int8x64; got {self.state.dtype}.")
            if to_int_tuple(self.state.shape) != (b, k - 1, c_blocks):
                raise ValueError(
                    f"state must be (B, K-1, C_blocks) = {(b, k - 1, c_blocks)}; "
                    f"got {tuple(self.state.shape)}."
                )
            return
        if len(self.weight.shape) != 2:
            raise ValueError(f"weight must be (K, C_blocks); got {tuple(self.weight.shape)}.")
        if int(self.weight.shape[1]) != c_blocks:
            raise ValueError(
                f"weight C_blocks {int(self.weight.shape[1])} must match input C_blocks {c_blocks}."
            )
        if to_int_tuple(self.state.shape) != (b, k - 1, c_blocks):
            raise ValueError(
                f"state must be (B, K-1, C_blocks) = {(b, k - 1, c_blocks)}; "
                f"got {tuple(self.state.shape)}."
            )
        if t > _CONV_CHUNK and t % _CONV_CHUNK != 0:
            raise ValueError(
                f"T={t} must be a multiple of the conv chunk size {_CONV_CHUNK} when "
                f"T > {_CONV_CHUNK} (the FIR is tiled along T to bound DPU registers)."
            )

    def infer_outputs(self) -> dict[str, BufferInfo]:
        b, t, c_blocks, k = self._dims()
        # int8/DWPU accumulates int32 and the DPU casts to float; the float18 path
        # keeps the input dtype.
        out_dtype = "float32x64" if self._is_int8() else self.input.dtype
        return {"output": BufferInfo(shape=(b, t, c_blocks), dtype=out_dtype, scope=A.DDR)}

    def implementation(self) -> None:
        if self._is_int8():
            self._implementation_dwpu()
            return
        b, t, c_blocks, k = self._dims()
        km1 = k - 1
        f18 = "float18x64"
        in_dtype = self.input.dtype
        w_dtype = self.weight.dtype
        out_dtype = self.output.dtype

        # The FIR is tiled along T into chunks of <= _CONV_CHUNK timesteps so the
        # DPU's resident window (K-1 + Tc) stays within the register file: a full-T
        # resident row needs O(T) registers and overflows for large prefill (T=256 ->
        # 258, regalloc fails). The conv has no recurrence -- each output reads only
        # inputs in a K-wide causal window -- so chunks are independent: chunk j reads
        # inputs [j*Tc - (K-1) : (j+1)*Tc] and writes outputs [j*Tc : (j+1)*Tc].
        # xe_ddr = [halo (K-1) ++ input] is materialised once so every chunk reads a
        # uniform (K-1 + Tc) slice with the halo built in.
        #
        # One self-contained Datapath per (b, c_block, chunk): per-cell operands staged
        # into L1 first; the datapath reads/writes L1 (not DDR indexed by the grid var,
        # which lowers incorrectly for >1 block); grid is outside the Datapath so each
        # cell fuses into one datapath with its own IFD0/IFD1/DPU/ODR.
        #   IFD0 : the (K-1 + Tc) window vectors.   IFD1 : the K taps.
        #   DPU  : load taps + the resident window, emit Tc outputs.
        #   ODR  : the Tc output vectors.
        tc = _float_chunk(t, k, c_blocks)
        n_chunks = t // tc

        # xe_ddr = [halo (K-1) ++ input (T)]. The halo is the carried window state,
        # or zero for the first prefill chunk (no history == torch Conv1d left zero-padding;
        # the persistent cache is uninitialised on a fresh instance). The write-back reads
        # xe_ddr (built from self.state first), so it is ordered after the read -- no
        # intermediate snapshot buffer needed.
        xe_ddr = A.alloc_buffer((b, km1 + t, c_blocks), in_dtype, scope=A.DDR)
        if self.state_scale is not None:
            # Runtime per-call reset: halo = state * state_scale (1.0 continues the
            # persistent window, 0.0 zeros it == zero_initial_state). The masked read
            # still feeds xe_ddr[:, 0:km1], so the write-back (which reads xe_ddr) stays
            # ordered after the state read -- no aliasing with the in-place update below.
            masked = atomic.mul(self.state, self.state_scale)
            atomic.copy(masked, output=xe_ddr[:, 0:km1, :])
        elif self.zero_initial_state:
            halo_zero = A.alloc_constant(
                np.zeros((b, km1, c_blocks, 64), dtype=np.float32), in_dtype, scope=A.DDR
            )
            atomic.copy(halo_zero, output=xe_ddr[:, 0:km1, :])
        else:
            atomic.copy(self.state, output=xe_ddr[:, 0:km1, :])
        atomic.copy(self.input, output=xe_ddr[:, km1:, :])

        # Channel blocks live INSIDE the datapath, not in the outer grid. With them
        # outside, one (b, c_block, chunk) step meant its own Datapath plus three tiny
        # per-block copies -- 96 datapaths and 288 copies for a single decode token here,
        # and the cost is per-datapath setup, not arithmetic. Now the whole channel range
        # streams through one datapath per (batch, chunk), fed by three slab copies.
        # The stream walks are channel-OUTER (stride c_blocks over the L1 slabs) so the
        # DPU sees one channel's K taps and K-1+Tc window contiguously and needs only
        # k + km1 + tc registers, independent of c_blocks. Channel-inner order would
        # instead need k * c_blocks registers resident at once.
        x_l1 = A.alloc_buffer((km1 + tc, c_blocks), in_dtype, scope=A.L1)
        w_l1 = A.alloc_buffer((k, c_blocks), w_dtype, scope=A.L1)
        o_l1 = A.alloc_buffer((tc, c_blocks), out_dtype, scope=A.L1)

        with A.grid(b, n_chunks) as idx:
            bi, ji = to_tuple(idx)
            lo = ji * tc
            atomic.copy(xe_ddr[bi, lo : lo + km1 + tc, :], output=x_l1)
            atomic.copy(self.weight, output=w_l1)

            with A.Datapath(core=None):
                with (
                    A.decl_stream(in_dtype) as in_stream,
                    A.decl_stream(w_dtype) as w_stream,
                    A.decl_stream(out_dtype) as out_stream,
                ):
                    # IFD0: each channel's (K-1 + Tc) window vectors, channel-outer.
                    with A.grid(c_blocks, km1 + tc) as cw:
                        cb_i, wi = to_tuple(cw)
                        with A.IFD(num=0):
                            A.stream_push(in_stream, x_l1[wi, cb_i])
                    # IFD1: each channel's K taps, in the same channel-outer order.
                    with A.grid(c_blocks, k) as cj:
                        cb_j, j = to_tuple(cj)
                        with A.IFD(num=1):
                            A.stream_push(w_stream, w_l1[j, cb_j])

                    with A.DPU():
                        w_regs = A.array(f18, size=k)
                        # Only this chunk's window is resident: K-1 + Tc registers,
                        # independent of T. Indexed statically (a rolling window does
                        # not persist across an in-DPU serial loop).
                        xe = A.array(f18, size=km1 + tc)
                        acc = A.var(f18)
                        prod = A.var(f18)

                        # Serial over channel blocks: each iteration re-pops its own
                        # taps and window, so nothing has to persist across the loop
                        # (which is what stops the window being rolled across T).
                        with A.serial(0, c_blocks) as _cb:
                            with A.unroll(0, k) as j:
                                A.buffer_store(w_regs, A.cast(w_stream.pop(), f18), (j,))
                            with A.unroll(0, km1 + tc) as i:
                                A.buffer_store(xe, A.cast(in_stream.pop(), f18), (i,))

                            # out[i] = w[K-1]*xe[i+K-1] + sum_{j<K-1} w[j]*xe[i+j]
                            # (newest tap first, matching the emulate accumulation order).
                            with A.unroll(0, tc) as i:
                                A.buffer_store(acc, w_regs[km1] * xe[i + km1], (0,))
                                with A.unroll(0, km1) as j:
                                    A.buffer_store(prod, w_regs[j] * xe[i + j], (0,))
                                    A.buffer_store(acc, acc[0] + prod[0], (0,))
                                A.stream_push(out_stream, A.cast(acc[0], out_dtype))

                    # ODR: drain every channel's Tc outputs, same channel-outer order.
                    with A.grid(c_blocks, tc) as ct:
                        cb_o, ti = to_tuple(ct)
                        with A.ODR():
                            A.buffer_store(o_l1, out_stream.pop(), (ti, cb_o))

            atomic.copy(o_l1, output=self.output[bi, lo : lo + tc, :])

        # Update the window state IN PLACE from xe_ddr's tail (xe_ddr[:, T:T+K-1, :]).
        # Sourcing from xe_ddr (which the datapath already read) ensures the write is
        # ordered after window consumption and avoids aliasing with the halo copy above
        # that seeded xe_ddr from self.state -- otherwise the device scheduler can
        # reorder the write-back ahead of that halo copy and corrupt the K-1 history.
        if not self._use_runtime_window(t, km1):
            atomic.copy(xe_ddr[:, t : t + km1, :], output=self.state)
        else:
            self._write_back_window(t, km1, b, c_blocks)

    def _use_runtime_window(self, t: int, km1: int) -> bool:
        """Whether to persist the window at a runtime offset rather than from the tail."""
        if self.valid_len is None:
            return False
        # A call with T <= K-1 cannot be right-padded and still leave a meaningful
        # window (valid_len == T is its only possibility), and its window straddles the
        # carried halo, which the runtime function cannot address from `input`. Decode
        # (T == 1) is exactly this case, and the static tail copy is already correct there.
        if t <= km1:
            return False
        return True

    def _write_back_window(self, t: int, km1: int, b: int, c_blocks: int) -> None:
        """Persist the conv window from a RUNTIME offset, via the `copy_conv_window` function.

        A buffer slice needs compile-time bounds, so `xe[valid_len : valid_len+K-1]` is not
        expressible in the script. The runtime function takes valid_len as an ADDRESS and does
        the pointer arithmetic on the DMA -- the same shape of solution axol.core.update_kv_cache
        uses for seq_pos, and for the same reason. Every extent stays static; only the offset moves.

        Sourced from `input`, not from the xe scratch: an ADDRESS arg must be an operand
        buffer (an op-allocated intermediate reaches the invoke as a re-declared Buffer with
        no span, which the source-location checker rejects). That costs nothing, because
        xe = [halo(K-1) ++ input(T)] means xe[valid_len+j] == input[valid_len-K+1+j] for the
        whole window whenever valid_len >= K-1 -- the runtime function refuses a shorter
        chunk rather than silently reading the halo it cannot see.
        """
        dtype = self.input.dtype
        if dtype not in _DTYPE_BYTES:
            raise ValueError(
                f"valid_len needs a known per-lane storage width to address the window in "
                f"bytes; {dtype!r} is not one of {sorted(_DTYPE_BYTES)}. Pass valid_len=None "
                f"to keep the static tail copy."
            )
        # ADDRESS args arrive as BufferLoads, not bare Buffers, so pass full-extent slices.
        src = self.input[tuple(slice(0, int(d)) for d in self.input.shape)]
        with A.script():
            with A.Invoke(core=None):
                A.call_extern(
                    "void",
                    "copy_conv_window",
                    input=src,
                    state=self.state,
                    valid_len=self.valid_len,
                    batch_size=b,
                    static_T=t,
                    window=km1,
                    row_bytes=c_blocks * PWORD * _DTYPE_BYTES[dtype],
                )

    def _implementation_dwpu(self) -> None:  # noqa: C901
        """W8A8 depthwise causal conv on the int8 DWPU.

        Builds the int8 extended input xe = [halo (K-1) ++ input (T)] and runs a
        depthwise VALID 1D conv over it on the DWPU (the halo is the carried window
        state, or zero for a fresh prefill), producing T outputs. The DWPU computes a
        per-channel int32 dot product sum_j q_x[t+j]*q_w[j]; the DPU casts it to
        float32 -- the per-channel dequant scale is applied by the caller. The last
        K-1 int8 inputs are written back into state in place (module docstring).

        The DWPU depthwise conv itself is delegated to the proven on-branch
        axol.core.dw_convolution_2d: the sequence maps onto its H (row) axis with a
        degenerate width of 1 (KH=K, KW=1), and it handles the DWPU datapath, its own
        H-tiling for L1, and the channel-block unroll. A hand-rolled 1D DWPU datapath
        (per-position or single-datapath) mis-lowered for C_blocks > 1 -- reusing the
        one physical scratchpad/weight buffer across the channel-block unroll corrupted
        every block; dw_convolution_2d gets this right.
        """
        b, t, c_blocks, k = self._dims()
        km1 = k - 1
        in_dtype = self.input.dtype  # int8x64
        in_L = km1 + t

        # Build xe = [halo (K-1) ++ input (T)] directly as a (B, in_L, 1, C_blocks) image
        # for dw_convolution_2d: the sequence on its H (row) axis, a degenerate width of
        # 1. The staging copies below are BULK 4D slices, and the atomic.reshape that
        # feeds them only inserts that degenerate axis -- row-major (B, T, C_blocks) and
        # (B, T, 1, C_blocks) hold their elements in the same order, so it is a view and
        # leaves the channels alone. (The per-(timestep, channel-block) CoreDMA form is
        # still what the output and the state write-back further down use, where every
        # index is addressed explicitly.) The halo is the carried window state, or zero
        # for a fresh prefill chunk.
        xe_ddr = A.alloc_buffer((b, in_L, 1, c_blocks), in_dtype, scope=A.DDR)
        # halo_src seeds the leading K-1 window, or is None to leave it at zero.
        # state_scale overrides zero_initial_state exactly as on the float path: mask the
        # int8 window in the DPU (int8 -> float broadcast multiply) and cast straight back
        # to int8. The gate is exactly 0.0 or 1.0, so the round trip is lossless.
        if self.state_scale is not None:
            masked = atomic.mul(self.state, self.state_scale)
            halo_src = atomic.dpu_cast(masked, output=BufferInfo(dtype=in_dtype))
        elif self.zero_initial_state:
            halo_src = None
            halo_zero = A.alloc_constant(
                np.zeros((b, km1, 1, c_blocks, 64), dtype=np.int8), in_dtype, scope=A.DDR
            )
            atomic.copy(halo_zero, output=xe_ddr[:, 0:km1, :, :])
        else:
            halo_src = self.state

        # Bulk slice copies, not per-(timestep, channel-block) CoreDMA writes. Each of
        # these four moves is contiguous, so it is one descriptor rather than T*C_blocks
        # of them; at decode (T=1, C_blocks=96, K=4) the elementwise form issued ~768
        # 64-byte transfers at ~370ns each and dominated the op (0.282ms for ~24k MACs).
        # Same atomic.copy 4D-slice idiom as the zero-halo path above.
        atomic.copy(
            atomic.reshape(self.input, shape=(b, t, 1, c_blocks)),
            output=xe_ddr[:, km1 : km1 + t, :, :],
        )
        # The halo copy is guarded OUTSIDE A.script(). Inside it, `if` is parsed as IR and
        # both arms are traced, so `halo_src[...]` was dereferenced even when halo_src is
        # None -- zero_initial_state=True failed with "'NoneType' object is not
        # subscriptable". The op's int8 test only ever passes a real state plus
        # state_scale, so nothing exercised that branch.
        if halo_src is not None:
            atomic.copy(
                atomic.reshape(halo_src, shape=(b, km1, 1, c_blocks)),
                output=xe_ddr[:, 0:km1, :, :],
            )

        # Delegate the int8 depthwise conv to the proven axol.core.dw_convolution_2d
        # (int8 x int8 dot -> int32 -> DPU float32, its own H-tiling + channel-block
        # unroll). self.weight is already dw_convolution_2d's native (C_blocks, K, 1, 1)
        # layout (packed channel-major on the host), so it is passed straight through --
        # a device-side (K, C_blocks) -> (C_blocks, K, 1, 1) transpose mis-lowers for
        # C_blocks > 1. Output (B, T, 1, C_blocks) is the raw accumulation; the caller
        # applies the per-channel dequant scale.
        conv_2d = dw_convolution_2d(
            xe_ddr, self.weight, stride=(1, 1), dilation=(1, 1), padding=(0, 0)
        )
        with A.script():
            for bi, ti, cb in A.grid(b, t, c_blocks):
                with A.CoreDMA():
                    self.output[bi, ti, cb] = conv_2d[bi, ti, 0, cb]
            # Update the window state IN PLACE: the last K-1 int8 inputs. Without
            # valid_len that is the tail of the (possibly right-padded) buffer; with it,
            # the last K-1 REAL rows, addressed at a runtime offset (_write_back_window).
            if not self._use_runtime_window(t, km1):
                for bi, li, cb in A.grid(b, km1, c_blocks):
                    with A.CoreDMA():
                        self.state[bi, li, cb] = xe_ddr[bi, t + li, 0, cb]
        if self._use_runtime_window(t, km1):
            self._write_back_window(t, km1, b, c_blocks)

    # Emulation (float18-stepped, mirrors the DPU op order)
    def emulate(self) -> None:  # type: ignore[override]
        # In emulate the operands are numpy arrays, so dispatch on the numpy dtype
        # (self.input.dtype is np.int8 here, not the "int8x64" buffer string that
        # _is_int8 checks in validate/infer_outputs/implementation).
        if np.issubdtype(self.input.dtype, np.integer):
            self._emulate_dwpu()
            return
        # Operands carry an explicit trailing PWORD lane dim: (B, T, C_blocks, PWORD);
        # fold the lanes into the channel axis.
        b, t = (int(s) for s in self.input.shape[:2])
        pword = int(self.input.shape[-1])
        c_blocks = int(self.input.shape[-2])
        c = c_blocks * pword
        k = int(self.weight.shape[0])
        km1 = k - 1

        x = self.input.astype(np.float32).reshape(b, t, c)
        w = emu.rf18(self.weight.astype(np.float32).reshape(k, c))
        # First chunk: the leading window is zero (not the persistent cache).
        if self.state_scale is not None:
            # Runtime gate: 1.0 continues the carried window, 0.0 resets it. All
            # lanes/blocks hold the same scalar; mirror the device broadcast multiply.
            scale = float(np.asarray(self.state_scale).reshape(-1)[0])
            s0 = self.state.astype(np.float32).reshape(b, km1, c) * scale
        elif self.zero_initial_state:
            s0 = np.zeros((b, km1, c), dtype=np.float32)
        else:
            s0 = self.state.astype(np.float32).reshape(b, km1, c)

        # Extended input: the K-1 window seeds (oldest first) then the T inputs.
        xe_raw = np.concatenate([s0, x], axis=1)  # (B, K-1 + T, C)
        xe = emu.rf18(xe_raw)

        out = np.zeros((b, t, c), dtype=np.float32)
        for ti in range(t):
            # newest tap first (matches the DPU acc seed), then the past taps.
            acc = emu.rf18(w[km1] * xe[:, ti + km1])
            for j in range(km1):
                acc = emu.rf18(acc + emu.rf18(w[j] * xe[:, ti + j]))
            out[:, ti] = acc

        np.copyto(self.output, out.reshape(self.output.shape))
        # Update the window state IN PLACE: the last K-1 of the extended input. It
        # is a plain copy on device (no DPU / float18 rounding), so use raw values.
        new_state = xe_raw[:, self._window_start(t) : self._window_start(t) + km1].reshape(
            self.state.shape
        )
        np.copyto(self.state, new_state)

    def _window_start(self, t: int) -> int:
        """First xe row of the persisted window: valid_len when given, else the buffer tail.

        Mirrors what `copy_conv_window` computes on device, so emulation and device agree
        by construction rather than by two hand-kept copies of the same index arithmetic.
        """
        if not self._use_runtime_window(t, int(self.state.shape[1])):
            return t
        vl = int(np.asarray(self.valid_len).reshape(-1)[0])
        if not 0 <= vl <= t:
            raise ValueError(f"valid_len {vl} outside [0, T={t}]")
        return vl

    def _emulate_dwpu(self) -> None:
        """NumPy reference for the int8 DWPU conv: an exact integer dot then a single
        int32->float18 cast (the DPU cast), no dequant (the caller applies the per-
        channel scale). Mirrors axol.core.dw_convolution_1d + quantized_matmul_activation.
        """
        b, t = (int(s) for s in self.input.shape[:2])
        pword = int(self.input.shape[-1])
        c_blocks = int(self.input.shape[-2])
        c = c_blocks * pword
        # int8 weight is (C_blocks, K, 1, 1) [+ PWORD lane]; fold to (K, C) with the
        # channel = cb*PWORD + lane order used for x/state.
        k = int(self.weight.shape[1])
        km1 = k - 1

        x = self.input.astype(np.int32).reshape(b, t, c)
        w = (
            self.weight.astype(np.int32)
            .reshape(c_blocks, k, pword)
            .transpose(1, 0, 2)
            .reshape(k, c)
        )
        if self.state_scale is not None:
            # Runtime gate: 1.0 continues the carried window, 0.0 resets it. All
            # lanes/blocks hold the same scalar; the device masks the int8 window and
            # casts back to int8, and the gate is exactly 0.0 or 1.0, so rounding the
            # product back to an integer here mirrors it exactly.
            scale = float(np.asarray(self.state_scale).reshape(-1)[0])
            s0 = np.rint(self.state.astype(np.float32).reshape(b, km1, c) * scale).astype(np.int32)
        elif self.zero_initial_state:
            s0 = np.zeros((b, km1, c), dtype=np.int32)
        else:
            s0 = self.state.astype(np.int32).reshape(b, km1, c)

        # Extended input: the K-1 window seeds (oldest first) then the T inputs.
        xe = np.concatenate([s0, x], axis=1)  # (B, K-1 + T, C), int32
        out = np.zeros((b, t, c), dtype=np.float32)
        for ti in range(t):
            # out[t] = sum_j q_w[j] * q_xe[t+j] (exact int accumulation), then the DPU
            # casts the int32 accumulation to float18.
            acc = np.zeros((b, c), dtype=np.int32)
            for j in range(k):
                acc += w[j] * xe[:, ti + j]
            out[:, ti] = emu.rf18(acc.astype(np.float32))

        np.copyto(self.output, out.reshape(self.output.shape))
        # State: the last K-1 int8 inputs, a plain copy (no rounding). casting="unsafe"
        # narrows the int32 window back to the state buffer's int8 (values already fit).
        state = self.state
        w0 = self._window_start(t)
        np.copyto(state, xe[:, w0 : w0 + km1].reshape(state.shape), casting="unsafe")


causal_conv1d = register_op(
    CausalConv1d,
    "llm.causal_conv1d",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)
