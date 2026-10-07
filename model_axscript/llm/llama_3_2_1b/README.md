# Llama-3.2-1B

This folder is the AxScript (ALTO-direct) port of Llama-3.2-1B. Its zoo card is
`llama-3.2-1b`, and the artifact is compiled locally, not downloaded. That card also carries
an `axgraph` backend built through the graph compiler; this directory is the ALTO-direct
path, a second backend on one card, not a second model.

**The two backends do NOT ship the same weights, so their accuracy numbers are not
comparable.** This AxScript path builds the BASE checkpoint `unsloth/Llama-3.2-1B`,
deliberately, because this backend's reference measurements were taken on base;
the `axgraph` path builds `meta-llama/Llama-3.2-1B-Instruct`, which is also what the
card is named after. Differencing an axscript score
against an axgraph one therefore measures base-vs-Instruct, not the compile path, and
is the wrong conclusion waiting to be drawn. Compare a backend only against its own
fp32 reference. (The sibling `llama-3.2-3b` ships Instruct on both, so this divergence
is specific to the 1B card.)

## How it is compiled (AxScript, ALTO-direct)

Instead of exporting to the graph compiler, the decoder layer is authored as an
ALTO kernel directly: RMSNorm, RoPE, quantized scaled-dot-product attention, and a
SwiGLU MLP, quantized per layer (W8A8 + SmoothQuant on the linears). Llama 3 is the
plain GQA decoder the rest of this family is described against - no QKV bias, no
QK-Norm - so the kernel is the design-P layer in its simplest form. Two traits are
worth knowing:

- **head_dim = 64 (design-P padding).** Llama-3.2-1B has hidden 2048 / 32 heads =
  head_dim 64, whose RoPE `rotate_half` pairs elements within a single 64-lane
  PWORD (the lane-aligned DPU cannot reach across it). Each head is padded to 2
  PWORD blocks (block-interleaved, pad lanes 0) entirely in host-side weight/table
  packing; the torch model stays at the real head_dim and the SDPA scale / RoPE
  frequencies use the logical head_dim. The dims come from
  `models/llama/naive_impl.py::_design_p_dims`, which the LFM2 kernels share.
- **Llama-3 frequency-band RoPE scaling.** `rope_scaling` splits the frequencies
  into three bands by wavelength against the original 8192-token context: short
  wavelengths are kept, long ones are divided by `factor` (32), and the band
  between ramps linearly. This is a change to the FREQUENCIES only, so no rope
  kernel of its own is needed - but the device must read the reference's exported
  `rope_inv_freq` rather than recompute the plain series from `rope_theta`, or the
  low-frequency bands would silently disagree with the torch model.

Today the compile emits a **build directory** with a named-graph contract (a
`prefill` graph and a `decode` graph) plus the packed per-layer parameters, which
the runtime loads directly.

## How to build and run it

One entry builds and runs; the compiled artifact is device-agnostic, so switching
the run device never recompiles:

```bash
# from a source checkout root
python -m model_axscript.llm build llama-3.2-1b                    # compile the artifact
python -m model_axscript.llm run llama-3.2-1b "The capital of France is" # build if needed, then run
python -m model_axscript.llm benchmark llama-3.2-1b "The capital of France is"
python -m model_axscript.llm run llama-3.2-1b "..." --device metis-0:2d:0 # run on a specific device (no recompile)

# --backend axscript is REQUIRED on the zoo verbs: this model also declares a codegen
# `build`, so a bare `axzoo build llama-3.2-1b` resolves to axgraph and compiles the
# OTHER backend (a different checkpoint - see below), not this directory.
axzoo build llama-3.2-1b --backend axscript                    # compile by zoo tag (runs --build)
axzoo eval llama-3.2-1b --backend axscript                     # score accuracy (lm-eval)
```

Calibration uses the GPU automatically (`--cal-device cpu` to force CPU).
This path builds `unsloth/Llama-3.2-1B`, an UNGATED mirror of the base weights, so no
Hugging Face login is needed. `--device` picks one Metis device by name or index - a model runs on one
device (its 4 cores), and device 0 is not always healthy, so pick another when it
is faulty. Because the artifact is device-agnostic, `run ... --device <other>` reuses
the same build with no recompile.


**Validated context lengths:** 128, 256, 512, 1024, 2048, 4096

Any other `context_len` is refused at build time. Sizes outside this set have not been
measured on this part, and an unmeasured one fails as a device hang rather than an
error - so the set is widened only by a measurement, never to make a build pass.
The live list is `supported_context_lens` in this folder's `make_config`, and the refusal
prints it, so the number above is a copy and that field is the source of truth.

A build with no `--seq-len` compiles **2048**, the useful middle of that range rather than
its top: the prefill attention mask grows with the SQUARE of the context.

On-chip measurements exist for 1024, 2048 and 4096. 128, 256 and 512 build, and are strictly
below the smallest measured size, so they do not approach the DRAM ceiling, but they have no
measurement on the part yet.

Each size prefills at its own chunk width, `chunk_lens` beside the set above: 64 / 128 / 256
at 128 / 256 / 512, and 1024 at the three measured sizes. The prefill graph is compiled for
`chunk_len` query positions, so one width cannot serve a set this wide - 1024 is not merely
slow at a 128-token context, it is wider than the whole KV cache.

## Adapt it to your own model

The `Llama3*` classes track the ARCHITECTURE, not this checkpoint's size, so a
same-architecture Llama 3 checkpoint (a fine-tune, or Llama-3.1-8B /
Llama-3.2-3B) needs no new code: point `make_config`'s `repo_id` at your weights
and rebuild; the design-P padding, the band-scaled RoPE and every dim
are derived from the HF config. A learned QKV bias is refused rather than silently
dropped (the kernel has no bias buffers).
A different architecture means authoring your own decoder-layer ALTO kernel on the
same shared harness (`../shared/`). See *A different architecture* in the top-level
[`model_axscript/README.md`](../../README.md).

`repo_id` is a Hugging Face Hub repo id or a local directory in checkpoint layout (the
`config.json` beside the safetensors shards). For one build, `--weights <dir-or-id>` does the
same without editing `make_config`.

Match the calibration data to YOUR domain. `calib_sources` defaults to a general mix
(wikitext, gsm8k, humaneval, alpaca, mmlu), and the int8 scales and activation LUT ranges
are fitted from whatever you supply. Out-of-domain calibration does not error - it produces
a model that quantizes badly, so change `calib_sources` when your deployment text differs.
