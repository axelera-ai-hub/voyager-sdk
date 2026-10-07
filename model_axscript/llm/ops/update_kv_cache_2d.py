#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""KV-cache slot write as one 2D DMA per cache (llm.update_kv_cache_2d).

``axol.core.update_kv_cache`` issues one 1D DMA per head per cache and polls each to
completion. The heads of the update are contiguous and the cache slots of one position are
``ctx_len * head_dim`` bytes apart, so one two-dim DMA (x = one head's slab, y = the heads)
writes each cache: two DMAs and two polls per call. The update buffers are only read.
"""

# mypy: disable-error-code="attr-defined, index, arg-type"

from axelera.compiler.alto import script as A
from axelera.compiler.alto.axol import intrinsic_registry
from axelera.compiler.alto.axol.core.update_kv_cache import PWORD, UpdateKvCache
from axelera.compiler.alto.axol.intrinsic_registry import FunctionArgument, FunctionArgumentType
from axelera.compiler.alto.compiler.op_registry import Arg, register_op
from axelera.config import HardwareGeneration

_INTRINSIC_NAME = "update_cache_2d"

# Guarded: the simulator renders every core's functions into one C module.
_IMPLEMENTATION = r"""
#ifndef AXSCRIPT_UPDATE_CACHE_2D
#define AXSCRIPT_UPDATE_CACHE_2D
#include <module.h>
#include <dma.h>

static int update_cache_2d(cfunc_ctx_t* ctx) {
    void* k_update = ctx->api->get_address_arg(ctx, 0);
    void* v_update = ctx->api->get_address_arg(ctx, 1);
    void* k_full   = ctx->api->get_address_arg(ctx, 2);
    void* v_full   = ctx->api->get_address_arg(ctx, 3);
    uint64_t seq_pos  = *(uint64_t*)ctx->api->get_address_arg(ctx, 4);
    uint64_t batch_size = ctx->api->get_uint64_arg(ctx, 5);
    uint64_t num_kv_heads = ctx->api->get_uint64_arg(ctx, 6);
    uint64_t head_dim = ctx->api->get_uint64_arg(ctx, 7);
    uint64_t static_seq_full_len = ctx->api->get_uint64_arg(ctx, 8);
    uint64_t static_seq_update_len = ctx->api->get_uint64_arg(ctx, 9);
    uint64_t head_lo = ctx->api->get_uint64_arg(ctx, 10);
    uint64_t k_src_head_lo = ctx->api->get_uint64_arg(ctx, 11);
    uint64_t v_src_head_lo = ctx->api->get_uint64_arg(ctx, 12);

    if (static_seq_update_len == 0 ||
        static_seq_update_len > static_seq_full_len) {
        LOG_ERR("Invalid static_seq_update_len");
        return 1;
    }
    if (seq_pos + static_seq_update_len > static_seq_full_len) {
        LOG_ERR("Invalid seq_pos %llu + update_len %llu exceeds full_len %llu",
                seq_pos, static_seq_update_len, static_seq_full_len);
        return 1;
    }

    /* One row per (batch, head): the update's rows are contiguous, the cache's are one
     * full sequence apart; every row is the update_len tokens of that head. `num_kv_heads`
     * is the number of heads to write and `head_lo` the first of them: a core can write
     * just its own band of heads out of a full-width update buffer (batch must be 1). */
    const uint64_t row_bytes = static_seq_update_len * head_dim;
    const uint64_t rows = batch_size * num_kv_heads;
    const uint64_t cache_row_stride = static_seq_full_len * head_dim;
    const uint64_t slot_offset = (head_lo * static_seq_full_len + seq_pos) * head_dim;
    /* The update buffers are full-width (one row per head), so the band starts k/v_src_head_lo
     * rows in -- normally head_lo, but a caller may pack k and v (and more) into one buffer. */
    const uint64_t k_src_offset = k_src_head_lo * row_bytes;
    const uint64_t v_src_offset = v_src_head_lo * row_bytes;

    const module_id_t dma_module_id = MODULE_ID_HP_DMA_0;
    dma_command_format_two_dim_t dma_cmd = {
        .header = {
            .triggers = 0,
            .id = dma_module_id,
            .tokens_produce = 0,
            .tokens_consume = 0,
            .command_format = DMA_COMMAND_FORMAT_ID_TWO_DIM,
            .config = 0,
        },
        .payload = {
            .src_xbytesize = row_bytes,
            .dst_xbytesize = row_bytes,
            .src_xaddrinc = 1,
            .dst_xaddrinc = 1,
            .src_yrowsize = rows,
            .dst_yrowsize = rows,
            .src_yaddrstride = row_bytes,
            .dst_yaddrstride = cache_row_stride,
            .tran_size = 0,
            .xtype = 1,
            .ytype = 1,
            .src_burstlen = 63,
            .dst_burstlen = 63,
            .src_osr_lmt = 31,
            .dst_osr_lmt = 31,
            .enable = 1,
        }
    };

    dma_cmd.payload.src_address = (uintptr_t)((uint8_t*)k_update + k_src_offset);
    dma_cmd.payload.dst_address = (uintptr_t)((uint8_t*)k_full + slot_offset);
    if (module_load_command(dma_module_id, (const module_command_t*)&dma_cmd) != MODULE_STATUS_OK) {
        LOG_ERR("Failed to load DMA command for K");
        return 1;
    }
    if (module_poll_idle(dma_module_id) != MODULE_STATUS_OK) {
        LOG_ERR("DMA K transfer failed");
        return 1;
    }

    dma_cmd.payload.src_address = (uintptr_t)((uint8_t*)v_update + v_src_offset);
    dma_cmd.payload.dst_address = (uintptr_t)((uint8_t*)v_full + slot_offset);
    if (module_load_command(dma_module_id, (const module_command_t*)&dma_cmd) != MODULE_STATUS_OK) {
        LOG_ERR("Failed to load DMA command for V");
        return 1;
    }
    if (module_poll_idle(dma_module_id) != MODULE_STATUS_OK) {
        LOG_ERR("DMA V transfer failed");
        return 1;
    }
    return 0;
}
#endif
"""

intrinsic_registry.set(
    name=_INTRINSIC_NAME,
    arguments=[
        FunctionArgument(name="k_update", type=FunctionArgumentType.ADDRESS),
        FunctionArgument(name="v_update", type=FunctionArgumentType.ADDRESS),
        FunctionArgument(name="k_full", type=FunctionArgumentType.ADDRESS),
        FunctionArgument(name="v_full", type=FunctionArgumentType.ADDRESS),
        FunctionArgument(name="seq_pos", type=FunctionArgumentType.ADDRESS),
        FunctionArgument(name="batch_size", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="num_kv_heads", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="head_dim", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="static_seq_full_len", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="static_seq_update_len", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="head_lo", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="k_src_head_lo", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="v_src_head_lo", type=FunctionArgumentType.UINT64),
    ],
    implementation=_IMPLEMENTATION,
)


class UpdateKvCache2D(UpdateKvCache):
    """``axol.core.update_kv_cache`` with the two caches written by one 2D DMA each.

    ``head_lo`` / ``heads`` write only cache heads ``[head_lo, head_lo + heads)``, from heads
    ``k_src_head_lo`` / ``v_src_head_lo`` (default ``head_lo``) of the update buffers, which
    may be wider than the cache on the head axis (a core's band out of a packed scratch).
    """

    head_lo = Arg(int, default=0, kw=True, description="First head to write.")
    heads = Arg(
        int | None,
        default=None,
        kw=True,
        description="Heads to write from head_lo (default: all remaining); batch must be 1 "
        "for a band.",
    )
    k_src_head_lo = Arg(
        int | None, default=None, kw=True, description="First source head of the k update."
    )
    v_src_head_lo = Arg(
        int | None, default=None, kw=True, description="First source head of the v update."
    )

    def _band(self) -> tuple[int, int, int, int]:
        """(head_lo, heads, k_src_head_lo, v_src_head_lo) with the defaults resolved."""
        lo = int(self.head_lo)
        n = int(self.k_full.shape[1]) - lo if self.heads is None else int(self.heads)
        k_src = lo if self.k_src_head_lo is None else int(self.k_src_head_lo)
        v_src = lo if self.v_src_head_lo is None else int(self.v_src_head_lo)
        return lo, n, k_src, v_src

    def validate(self) -> None:
        # Not super().validate(): it refuses an update wider than the cache on the head axis.
        batch, n_heads, full_len, dim = (int(x) for x in self.k_full.shape)
        lo, n, k_src, v_src = self._band()
        if lo < 0 or n < 1 or lo + n > n_heads:
            raise ValueError(f"heads [{lo}, {lo + n}) exceed the cache's {n_heads}")
        if (lo, n) != (0, n_heads) and batch != 1:
            raise ValueError("a head-band update needs batch 1")
        if tuple(int(x) for x in self.v_full.shape) != (batch, n_heads, full_len, dim):
            raise ValueError("v_full shape is incompatible with k_full shape")
        update_len = int(self.k_update.shape[2])
        for name, upd, src in (("k", self.k_update, k_src), ("v", self.v_update, v_src)):
            ub, uheads, ulen, udim = (int(x) for x in upd.shape)
            if (ub, ulen, udim) != (batch, update_len, dim):
                raise ValueError(f"{name}_update shape is incompatible with {name}_full")
            if src < 0 or src + n > uheads:
                raise ValueError(
                    f"{name} source heads [{src}, {src + n}) exceed the update's {uheads}"
                )
        for buf in (self.k_update, self.v_update, self.k_full, self.v_full):
            if buf.dtype != "int8x64":
                raise ValueError("Only int8x64 dtype is supported for kv caches")
        if self.seq_pos.dtype != "uint64x64" or [int(x) for x in self.seq_pos.shape] != [1]:
            raise ValueError("seq_pos must be a single uint64x64 element")

    def emulate(self) -> None:  # type: ignore[override]
        lo, n, k_src, v_src = self._band()
        pos = int(self.seq_pos[0, 0])
        length = int(self.k_update.shape[2])
        self.k_full[:, lo : lo + n, pos : pos + length] = self.k_update[:, k_src : k_src + n]
        self.v_full[:, lo : lo + n, pos : pos + length] = self.v_update[:, v_src : v_src + n]

    def implementation(self) -> None:
        lo, n, k_src, v_src = self._band()
        with A.script():
            with A.Invoke(core=self.core):
                A.call_extern(
                    "void",
                    _INTRINSIC_NAME,
                    k_update=self.k_update,
                    v_update=self.v_update,
                    k_full=self.k_full,
                    v_full=self.v_full,
                    seq_pos=self.seq_pos,
                    batch_size=self.k_full.shape[0],
                    num_kv_heads=n,
                    head_dim=self.k_full.shape[3] * PWORD,
                    static_seq_full_len=self.k_full.shape[2],
                    static_seq_update_len=self.k_update.shape[2],
                    head_lo=lo,
                    k_src_head_lo=k_src,
                    v_src_head_lo=v_src,
                )


_update_kv_cache_2d = register_op(
    UpdateKvCache2D,
    "llm.update_kv_cache_2d",
    generation=[HardwareGeneration.EUROPA],
    atomic=False,
)


def update_kv_cache_2d(k_update, v_update, k_full, v_full, seq_pos, **kwargs):
    """In-place KV write; returns (k_full, v_full) like ``axol.core.update_kv_cache``."""
    _update_kv_cache_2d(k_update, v_update, k_full, v_full, seq_pos, **kwargs)
    return k_full, v_full
