# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The build+run config a decoder-only model in this family declares.

Moved out of :mod:`model_axscript.llm.generate` and re-exported from it, so the name every
caller already imports keeps working while the class itself costs nothing to read. The
orchestrator imports torch, safetensors and the compiler at module scope; this class is
plain data, and every text model's ``make_config`` returns it. Asking a model what it
declares -- the front door refusing an unmeasured ``--seq-len``, a ``--help`` printing the
sizes, the core count a generation implies -- was pulling in the whole numerical stack to
read a dataclass, which is also why it could not be read at all in an env without one.

Stdlib only, therefore, exactly as ``vl_config.py`` is for the vision-language half: the two
families' configs are now symmetric in what they cost to import as well as in what they
declare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from model_axscript.shared.hardware import default_cal_device


@dataclass
class GenerationConfig:
    """Build + run config for one LLM artifact (defaults: Qwen3-0.6B on Metis)."""

    implementation: str = "qwen3-metis"
    repo_id: str = "Qwen/Qwen3-0.6B"
    target: str = "device"  # "device" (real Metis) or "sim" (AICORE_SIM)
    # torch device for the reference model: load + calibration/quantization and the
    # fake-quant perplexity forwards run here ("cpu", "cuda", "cuda:0", ...). Distinct
    # from target (the Metis/sim backend the compiled artifact runs on); the export
    # pulls every tensor back to the host, so the artifact is identical either way.
    device: str = "cpu"
    # The torch device for PTQ CALIBRATION only, separate from `device` (the reference
    # and eval device) and from the Metis run device. A GPU accelerates calibration and
    # nothing else, since the ALTO compile and export always run on CPU. That is what
    # lets the build complete on a host with no AIPU.
    #
    # Empty means "pick one": CUDA when it is available, else CPU. Resolved in
    # __post_init__ so every consumer sees a concrete device. Leaving it empty and
    # resolving at each use site would let one path fall back while another raised.
    cal_device: str = ""
    artifact_dir: Path = Path("build/qwen3-0.6b")  # holds model.json + blobs + host_params
    # Compiled KV-cache capacity (prompt + generated tokens). 1024 is the smallest size
    # any card ships and the cheapest to run: the prefill attention mask is QUADRATIC in
    # this value (measured on LFM2-700M: 96 MiB at 1024 against 1536 MiB at 4096), so a
    # larger default would let a config that forgets to set it compile a size that no
    # longer fits a 4GB part -- and that fails as a device hang at prefill minutes in,
    # not as a compile error. Every shipped card overrides this explicitly.
    context_len: int = 1024
    # The context lengths this model has been measured at on chip. `context_len` must be
    # one of them: the KV cache and the prefill kernel are sized from it, and an unmeasured
    # size fails as a device hang at prefill rather than a compile error, so the set is
    # declared instead of left open. Widen it only after measuring on the target part -
    # the ceiling is AIPU DRAM, so a 16GB part admits sizes a 4GB part does not.
    # This set decides which sizes a model card must carry evidence for; filing that
    # evidence is a separate step.
    # None => just the size being compiled, so a config that states one size does not have
    # to state it twice and cannot contradict itself. A card that ships several measured
    # sizes lists them all; that list, not this fallback, is what a report owes evidence for.
    supported_context_lens: tuple[int, ...] | None = None
    # Device-kernel phases to compile. With a prefill graph present, the prompt is
    # ingested in one call; set ("decode",) to skip the (large at big context_len)
    # prefill kernel and prefill by replaying the decode graph instead.
    phases: tuple[str, ...] = ("prefill", "decode")
    # Chunked prefill: compile the prefill graph for chunk_len query positions (not the full
    # context_len) and ingest the prompt in chunk_len-token chunks, threading the recurrent/conv/KV
    # state across chunks. Bounds the sequential-recurrence program size + compile time regardless
    # of context_len. None => single full-context prefill call.
    chunk_len: int | None = None
    # The per-context chunk widths, positionally aligned with `supported_context_lens`: entry i
    # is the chunk this model prefills `supported_context_lens[i]` in. One chunk cannot serve a
    # whole declared set once that set spans a wide range -- the prefill graph is compiled for
    # chunk_len query positions, so a chunk LARGER than the context it runs at is not a slower
    # build but an impossible one, and a card declaring 128 alongside 4096 has no single value
    # that is legal at both ends. Stated as a list rather than derived (say, min(chunk, context))
    # because the right width at each size is a kernel property measured per model: Qwen3.5's
    # gated-DeltaNet needs a multiple of 128 and overflows L1 above it, where the dense-attention
    # cards run 1024 happily at 4096.
    #
    # None => `chunk_len` is used at every size, which is what a config declaring ONE size means
    # and what the `--chunk-len` front door passes. When it IS set it WINS over `chunk_len`:
    # `context_len` moves (a `--seq-len`, a context sweep, an artifact read back off disk) and the
    # chunk has to move with it, so the resolved value is derived from the pair at every
    # construction rather than pinned once by whoever built the config first.
    chunk_lens: tuple[int, ...] | None = None
    # Run the lm_head on the DEVICE as an int8 vocab projection instead of the host fp16 gemv.
    # Quantizes the head into the bundle (so it changes the quant cache key) and compiles one
    # extra decode-width graph, at the cost of vocab*hidden bytes of device DDR. A different
    # artifact, so it is never resolved from the implementation: a card that ships the
    # device head says so. True on an implementation without one is an error.
    device_lm_head: bool = False
    # Number of AIPU cores to compile the layer for. >1 distributes the attention
    # heads of the (core) SDPA op across cores (see scaled_dot_product_attention's
    # cores arg); it also sets AltoConfig.num_cores so the system is configured with
    # that many cores. Other ops run on core 0; only multicore-aware ops (SDPA) shard.
    # 1 (default) = single-core, unchanged behaviour.
    num_cores: int = 1
    # Fraction of device memory the compile may use -> AltoConfig.resources, which scales
    # the L2 budget: (MEMORY_SIZES[generation][L2] - L2_RESERVED_SIZE) * resources.
    # It must match the share the RUNTIME will actually be given, because the driver
    # partitions L2 statically into NUM_AI_CORES equal windows and hands a context
    # `num_sub_devices` of them (libaxldev `axl_dev_config_ctx`), while the reservation
    # is derived from the artifact (`shared/runtime.py::required_cores`). So the honest
    # value is cores_the_artifact_spans / NUM_AI_CORES: 1.0 for a build that spans the
    # whole chip, 0.25 for a single-core build on OMEGA's four AI cores. Left at 1.0 a
    # single-core build is compiled against ~30.8 MB of L2 but loaded into ~8 MB, and the
    # mismatch surfaces only on the device, as a memory-pool failure at model load.
    # AltoConfig refuses num_cores / NUM_AI_CORES > resources, so lower it with num_cores.
    resources: float = 1.0
    # HardwareGeneration name -> AltoConfig.generation. OMEGA (default) or EUROPA.
    generation: str = "OMEGA"
    # Run the whole eval as a dependency graph: overlap the (CPU) device codegen with the (GPU)
    # calibration + run the per-(kind,phase) codegens in parallel, and let metrics that don't
    # need the device artifact (perplexity) score concurrently with the build. Also folds the
    # LUT-range recording into the calibration passes. Quant-independent codegen makes this safe;
    # off by default.
    parallel: bool = False
    #: Codegen compiles to run at once under `parallel`. Each is a spawn process holding a whole
    #: kernel compile, alongside the calibration process holding the fp32 reference, so a big
    #: model can exhaust host memory before it exhausts cores. 1 keeps the recorder folding (the
    #: halved calibration passes) and compiles one kernel at a time.
    codegen_jobs: int = 4
    # Override the layer kernel's buffer-promotion default (PromoteBuffers -> L2). None keeps the
    # kernel's default; False forces it off (e.g. large models where L2 promotion overflows).
    enable_buffer_promotion: bool | None = None
    # Override the layer kernel's datapath-fusion default (DatapathFusion pass). None keeps the
    # kernel's default (off for every kind that has not verified fusion on its op chain); True
    # forces it on everywhere, False off -- for A/B-ing the pass without editing kernels.
    enable_datapath_fusion: bool | None = None
    # Profile every compilation pass (time + host RSS) -> pass_profile.json +
    # host_memory_timeline.jsonl per compiled (kind, phase) dir. Off by default.
    emit_compilation_profile: bool = False
    # None caps generation only at what the context can still hold.
    max_new_tokens: int | None = 256
    temperature: float = 0.0  # 0 => greedy
    # The seed prompt a `run` / `benchmark` uses when the caller names none. A run-time
    # knob like the two above, not a build one: it never reaches the compiler, and the
    # artifact is identical whatever it says.
    #
    # DECLARED ONCE HERE for every path that holds a `GenerationConfig`: the front door,
    # the zoo's `benchmark` bridge, the context sweep, `evaluation/metrics.EvalContext` and
    # `evaluation/benchmark.benchmark_implementations`. They each carried their own copy of
    # this string with no gate holding them together, so an override here reached some of
    # them and not others -- two `ttft_ms` numbers under one name.
    #
    # ONE PATH READS THIS FIELD AND STILL CANNOT SEE AN OVERRIDE.
    # `models/smolvlm2/benchmark.measure_text` resolves its seed through
    # `EvalContext.perf_prompt`, so it does reach this default -- but the `GenerationConfig`
    # it reads is built from the ARTIFACT'S MANIFEST (implementation, repo_id, context_len,
    # chunk_len) rather than from the model's `make_config`, so the value is always this
    # class default and a card's override never arrives. Teaching that path to load the card
    # would change what it measures, not where a default is declared, so it is left alone.
    #
    # KEEP IT THE SAME FOR EVERY MODEL unless a model's own prompt is genuinely wrong for
    # it. `prompt_tokens` and `ttft_ms` are reported per model and read side by side in a
    # report table, so a per-model prompt makes those columns mean a different thing in
    # each row -- the same reason the context sweep pins its eval token count across sizes.
    # An override is for a checkpoint the shared prompt does not fit, not for taste.
    #
    # NOTE it is passed RAW: the benchmark path does not chat-template it (only `--eval`
    # has `--apply-chat-template`), so an instruct checkpoint is measured on a completion
    # prompt. Overriding this with pre-formatted chat markup would hardcode a template
    # that then rots against the checkpoint's own; templating is the missing feature.
    prompt: str = "The capital of France is"
    # Extra end-of-sequence ids to stop on, beyond tokenizer.eos_token_id (which is a
    # single id). Qwen3 ends turns with either <|im_end|> or <|endoftext|>, so the
    # caller populates this from the model's generation_config.eos_token_id.
    stop_token_ids: tuple[int, ...] = ()
    smooth_alpha: float = 0.8
    qk_alpha: float = 0.5
    calibration_texts: list[str] = field(default_factory=list)
    calib_window: int = 1024  # max tokens per calibration sample
    #: Calibration windows to run per forward. The samples are independent, so stacking them
    #: is the SAME arithmetic -- observers see the same values and their min/max is unchanged
    #: -- but it amortises Python and thread dispatch, which dominate on CPU: a model whose
    #: linear attention is a per-token loop spends most of calibration in dispatch, not FLOPs
    #: (measured 3.07x for a 9B at batch 8). Costs activation memory linearly, so raise it
    #: only as far as host RAM allows.
    calib_batch: int = 1
    #: Dtype the torch reference is calibrated in. float32 is the safe default; bfloat16
    #: halves its weights (36 GB -> 18 GB for a 9B) and the memory traffic that dominates a
    #: CPU calibration. The gated-delta scan stays fp32 either way -- it accumulates over the
    #: whole sequence, so a narrower type drifts -- but the observed activation RANGES are
    #: then bf16-rounded, which perturbs the quantization scales slightly. Gate it on the
    #: perplexity check rather than assuming it is free.
    calib_dtype: str = "float32"
    num_calib_samples: int = 16  # windows, only for the calibration_texts override path
    # Diverse, chat-aware calibration mirroring the production quantizer: raw WikiText
    # for general LM coverage plus chat-templated task data (math/code/instructions/
    # knowledge) so the activation scales cover the assistant-turn / eos regime.
    # Raw-only calibration mis-quantises the eos logit (the model never stops).
    # (source, weight) pairs, the default weights are the production counts.
    calib_sources: tuple[tuple[str, int], ...] = (
        ("wikitext", 1024),
        ("gsm8k", 1024),
        ("humaneval", 164),
        ("alpaca", 1024),
        ("mmlu", 1024),
    )
    # Total calibration samples (forwards) to spread across calib_sources in
    # proportion to their weights; each source is capped at its dataset size. None
    # uses the weights directly as per-source counts (the full production mix, ~4232).
    calib_budget: int | None = None

    def __post_init__(self) -> None:
        if not self.cal_device:
            self.cal_device = default_cal_device()
        # Derived, never widened: a config that names one size declares that size. Widening
        # an EXPLICIT set to admit an unmeasured `context_len` is what validate_run_size
        # refuses, and this does not do it -- it only fills a set nobody stated.
        if self.supported_context_lens is None:
            self.supported_context_lens = (self.context_len,)
        self.supported_context_lens = tuple(int(n) for n in self.supported_context_lens)
        if self.chunk_lens is not None:
            self.chunk_lens = tuple(int(n) for n in self.chunk_lens)
            self._check_chunk_lens()
            # Resolved HERE, so every reader of `config.chunk_len` -- the compile, the host
            # prefill loop, the report row, the regression manifest -- keeps reading one field
            # and none of them has to know the list exists. `dataclasses.replace` is the reason
            # this cannot be left to a property: it copies the resolved `chunk_len` forward, so
            # a `replace(config, context_len=128)` would otherwise carry 4096's chunk to 128.
            # Re-deriving it from the list here overwrites exactly that stale copy.
            self.chunk_len = self.chunk_len_for(self.context_len)

    def _check_chunk_lens(self) -> None:
        """Refuse a `chunk_lens` that does not line up with the declared sizes.

        Both failures it catches are silent otherwise. A misaligned list hands one size's
        chunk to another, which compiles and runs and is simply the wrong prefill width; a
        chunk wider than its own context asks the prefill graph for more query positions than
        the KV cache holds, which fails deep in codegen with no mention of either number.
        """
        if len(self.chunk_lens) != len(self.supported_context_lens):
            raise ValueError(
                f"chunk_lens {self.chunk_lens} has {len(self.chunk_lens)} entries for the "
                f"{len(self.supported_context_lens)} sizes in supported_context_lens "
                f"{self.supported_context_lens}; they are positionally aligned, so a chunk "
                f"is owed for each declared size"
            )
        too_wide = [
            (context, chunk)
            for context, chunk in zip(self.supported_context_lens, self.chunk_lens)
            if chunk > context
        ]
        if too_wide:
            raise ValueError(
                f"chunk_lens entries wider than the context they prefill: {too_wide}; the "
                f"prefill graph is compiled for chunk_len query positions, which cannot "
                f"exceed the compiled context_len"
            )

    def chunk_len_for(self, context_len: int) -> int | None:
        """The chunked-prefill width this config uses at `context_len`.

        The declared pairing when `context_len` is one of the declared sizes, which is every
        build the front door admits (`validate_run_size` refuses the rest). A context sweep
        measuring a size the card has not declared yet is the one caller that asks outside the
        set, and what it gets is the widest width the card DECLARED that still fits.

        Never a width derived by clamping. A chunk is a kernel-legal value, not a number on a
        scale: qwen3.5's gated DeltaNet refuses anything that is not a multiple of its internal
        C=128, so `min(max(chunk_lens), context_len)` answering 64 for a 64-token context, or
        200 for a 200-token one, hands the compiler a width no card ever claimed to compile and
        fails inside codegen rather than by name. A context under every declared width is
        refused here instead, where the message can say so.
        """
        if self.chunk_lens is None:
            return self.chunk_len
        pairs = dict(zip(self.supported_context_lens, self.chunk_lens))
        if context_len in pairs:
            return pairs[context_len]
        fits = [chunk for chunk in self.chunk_lens if chunk <= context_len]
        if not fits:
            raise ValueError(
                f"context_len {context_len} is under every declared chunk width "
                f"{sorted(set(self.chunk_lens))}, so this model has no prefill width for it"
            )
        return max(fits)
