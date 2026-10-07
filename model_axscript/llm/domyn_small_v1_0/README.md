# Domyn-Small-v1.0

This folder is the AxScript (ALTO-direct) recipe for `domyn/Domyn-Small-v1.0`, a
9.82B-parameter `NemotronForCausalLM`. Its zoo card is `domyn-small-v1.0`, and the artifact is
compiled locally, not downloaded. Like the LFM2 cards this one declares only the AxScript path:
there is no `axgraph` backend for it.

**It is the first card on the Nemotron architecture.** The device kernel is `domyn-metis`
(`models/domyn/metis_impl.py`) and the torch reference is `models/domyn/domyn.py`
(architecture `domyn`). Do not reason from a Llama or Qwen3 card about this one: three
traits mean none of the shipped RMSNorm/SwiGLU kernels carry over.

Measured results, perf at 1024, 2048 and 4096 and the device-vs-fp32 accuracy gap at 1024 and
2048, are under "Measured on chip" below.

## The architecture

| | |
|---|---|
| layers / hidden / intermediate | 40 / 4096 / 16384 |
| attention | 48 query heads over 8 KV heads (GQA), head_dim **128** |
| norm | **LayerNorm1P** - subtracts the mean, adds a bias, gain is `weight + 1` |
| MLP | **squared ReLU** - `down(relu(up(x))**2)`, no gate |
| RoPE | **partial**, `partial_rotary_factor = 0.5`: the leading 64 of 128 channels |
| vocab | 256000, and `lm_head` runs on the HOST |
| tied embeddings | **no** - `lm_head` has its own ~2 GB of fp16 weights host-side |
| rope_theta | 5e5 |

Two details are easy to get wrong and both are asserted in the tests:

- `head_dim` is **not** `hidden_size // num_attention_heads`. It is 128, while
  `4096 // 48 = 85`; `q_proj` widens 4096 -> 6144.
- the partial-RoPE inverse frequencies use the **rotary** width as the exponent's
  denominator (64), not `head_dim` (128). Using `head_dim` changes every frequency and
  still loads and runs.

## How each trait reaches the device

None of the three needed a new compiler primitive.

**Partial RoPE -> a channel permutation.** `axol.core.rope` is a half-rotation over PWORD
*blocks*: it pairs channel `p` with `p + head_dim/2` lanewise. Nemotron pairs channel `i`
with `i + rotary_dim/2` inside the 64 rotary channels - and at `head_dim=128` those 64
channels are exactly ONE PWORD, so the partner of lane `i` is lane `i+32` of the same
PWORD. That is precisely the lane-alignment case a block-wise op cannot express.
`partial_rope_head_permutation` reorders each head's channels so every rotary pair's two
halves land a half-head apart, where the hardware pairs them, and the pass-through channels
fill the remaining pairs. The frequency table is then padded to `head_dim/2` lanes with
frequency **zero**, which is `cos = 1`, `sin = 0`: an exact identity rotation for those
pairs. `q_proj` / `k_proj` rows and their per-output-channel dequant are permuted together
at bind time; V and `o_proj` are untouched, because RoPE never reaches them and `q.k` is
invariant under a permutation applied to both sides. Generic in `rotary_dim`, and the
identity when `rotary_dim == head_dim`.

**Squared-ReLU MLP -> `llm.relu2_quantize`.** One new op (`ops/relu2_quantize.py`):
`int8(round_clip(max(x,0)**2 * scale + zp))` in a single DPU datapath, using the native
`prelu(x, 0)` for the relu. It replaces what would be three passes (`maximum` -> `mul` ->
`llm.quantize`) over the layer's widest tensor. Exact, **not** a LUT: `relu^2` is unbounded,
so the fixed-domain piecewise-linear trick that works for SiLU has no good outer bin. The
MLP is therefore matmul -> `relu2_quantize` -> matmul, with no composed gate/up weight
because there is no gate.

**LayerNorm1P -> the two-pass Kahan norm, affine folded into the quantize.**
`axol.core.layer_norm_two_pass_kahan` takes weight and bias buffers, but they do not fit:
each is `normalized_shape` PWORD rows, 1 MiB apiece at hidden 4096, so 2 MiB of fixed L1
overhead against a 4 MiB L1 - and the op then refuses to tile at all, even for `seq_len=1`
decode. `fold_norm_affine_into_quant` folds the affine into the `llm.quantize` that consumes
the norm output: `scale = norm_weight * act_scale`, `zp = norm_bias * act_scale + act_zero`.
Algebraically the same affine, one rounding step shorter, and it removes both buffers from
the kernel. Every norm here is immediately quantized for the next matmul
(`input_layernorm` -> the shared q/k/v quant, `post_attention_layernorm` -> `up_proj`), so
nothing consumes an un-affined norm output. The single-pass `rms_norm_dpu_reduce` /
`layer_norm_dpu_reduce` variants are not an option at this width - they keep `~3 * D_blocks`
vectors register-resident and hidden 4096 is `D_blocks = 64`, twice the ~32 that fits the DPU
register banks - so `DOMYN_SPEC` pins `dpu_reduce_max_d_blocks = 0`, which makes a small test
config take the same path as the real model instead of silently diverging.

**`up_proj` emits fp16.** It is the widest matmul in the layer (`N = intermediate/64 = 256`
blocks). With a float32 output its L1 tile is 2.7 MiB of the 4 MiB L1 and prefill codegen is
infeasible at any chunk length (the tiling floor does not depend on `seq_len`). float16
halves it, and precision is not the binding constraint: `relu2_quantize` squares in float18
and the result lands on an int8 grid far coarser than float16's 10 mantissa bits.

## Validated context lengths

**Validated context lengths:** 128, 256, 512, 1024, 2048, 4096

Any other `context_len` is refused at build time, before the compile starts
(`validate_run_size`), and the refusal prints the set. The live list is
`supported_context_lens` in this folder's `make_config`, never this sentence. 1024, 2048 and
4096 were run on a Metis part (16 GiB, 4 cores); 128, 256 and 512 build but have no measurement
on the part yet. Widening the set is a measurement, never an edit to make a build pass.

`chunk_len` follows the size (`chunk_lens`: 64 / 128 / 256 / 512 / 512 / 512), because a
prefill graph compiled for 512 query positions is wider than a 128-token KV cache. It tops out
at 512, not the 1024 the other 8B-class cards use: at hidden 4096 / intermediate 16384 the
prefill kernel already sits at 3.85 MiB of the 4 MiB L1 at 512. The KV cache scales linearly
with the context (80, 160 and 320 MiB at 1024, 2048 and 4096), putting total device memory at
7.50, 7.58 and 7.74 GiB, so all three fit a 16 GiB part and none fits a smaller one.

## Cores

The card builds `num_cores = 4`, which reserves 4 sub-devices at load, so it needs a part
that has them. The two phases are limited by different things: prefill is compute-bound and
scales cleanly with cores, while decode is DDR-bandwidth-bound - the layer stack reads
7.72 GiB of int8 weights per token however many cores read it - so sharding buys only the
non-DDR part of a decode step. Reading fewer bytes per token, not adding cores, is the lever
left for decode.

## Host cost, and host RAM

The artifact is 7.72 GiB of device memory pools (160 MiB of it KV cache), so it needs a 16GB
part, plus 4.0 GB of host params: the fp16 embedding and the untied `lm_head`. A decode step
therefore carries a host `256000 x 4096` fp16 gemv plus the final LayerNorm1P.

Building it needs ~47 GB of host RAM to load the checkpoint and ~52 GB through calibration.
The reference is built in float32 and the load **streams** shard by shard
(`checkpoint.stream_hf_safetensors_into`) straight into the already-allocated model; merging
the checkpoint into one state dict first and calling `load_state_dict` would hold both at
once, ~79 GB, which is what made this checkpoint unloadable on a 62 GB host.

Measured: **9.815B parameters in 92.8 s at 46.7 GB peak RSS** (float32, cold page cache). `transformers` loads the same checkpoint at 44.1 GB, so the streaming loader
costs ~2.6 GB over `from_pretrained` and buys the ~32 GB that merging would have cost.

## How to build and run it

```bash
# from a source checkout root
python -m model_axscript.llm build domyn-small-v1.0                   # compile the artifact
python -m model_axscript.llm run domyn-small-v1.0 "The capital of Italy is"
python -m model_axscript.llm benchmark domyn-small-v1.0 "The capital of Italy is"
python -m model_axscript.llm run domyn-small-v1.0 "..." --device metis-0:1:0   # no recompile
python -m model_axscript.llm eval domyn-small-v1.0 --tasks wikitext   # score accuracy (lm-eval)

axzoo build domyn-small-v1.0 --backend axscript                       # compile by zoo tag
axzoo eval  domyn-small-v1.0 --backend axscript --tasks wikitext      # score accuracy (lm-eval)
```

`domyn/Domyn-Small-v1.0` is a public repo, so no `huggingface-cli login` is needed. The
artifact is device-agnostic, so `--device <other>` reuses the same build with no recompile.

A prompt shorter than `chunk_len` still pays a full 512-position prefill, so time-to-first-token
is quantised to whole chunks: a 128-token prompt costs what 512 costs.

## Measured on chip

On a 16 GiB Metis part with 4 cores:

| context_len | ttft_ms | decode_tok_s | kv_cache_mib | ddr_mib |
|---|---|---|---|---|
| 1024 | 2258.5 | 2.0 | 80.0 | 7499.46 |
| 2048 | 2280.0 | 2.0 | 160.0 | 7579.96 |
| 4096 | 2340.9 | 1.9 | 320.0 | 7740.96 |

End-to-end **device + host**, not AIPU-only: every token pays a 256000 x 4096 fp16 gemv plus
the final LayerNorm1P on the host. `run` answers "The capital of Italy is" with "Rome.".

Accuracy is lm-eval wikitext `word_perplexity`, 200 samples, 0-shot, on the device and on the
fp32 reference:

| context_len | device | fp32 reference |
|---|---|---|
| 1024 | 18.0841 | 16.9771 |
| 2048 | 15.7186 | 14.7263 |

4096 carries perf only. Scoring the fp32 reference leg there needs more host RAM than the
measurement host had: lm-eval materialises a `[max_length, 256000]` fp32 logits tensor (4.0 GiB
at 4096) on top of the ~30 GiB fp32 model. A host with more RAM scores all three sizes with the
same command.

`axzoo eval --tasks wikitext` and the `eval` verb report a **word-denominated** perplexity. A
token-denominated perplexity over the same text is a different number; never difference one
against the other.

### The torch reference

The torch reference was checked against `transformers`' own `NemotronForCausalLM` on the
real 9.82B checkpoint (both legs float32, same 41 tokens, one process each because two
~39 GB models do not co-reside):

| | |
|---|---|
| logit scale (max abs logit) | 56.567 |
| max abs logit error | 1.20e-04 |
| max abs error / logit scale | **2.12e-06** |
| max relative error where `abs(logit) > 1` | 5.66e-05 |
| KL(HF, ours) | **+1.76e-11 nats** |
| top-1 agreement | **100.00%** |

That is float32 rounding, so the thing being quantized is right; the error reported is
against the logit SCALE, since a 256000-wide vocabulary always holds logits near zero and a
per-element ratio would be dominated by whichever landed closest to 0.

### The build

At the shipped configuration, on a 16 vCPU host with no GPU:

```
python -m model_axscript.llm build domyn-small-v1.0 --output-dir <out> --num-cal 64
```

| | |
|---|---|
| wall clock | **6 min 55 s** (calibration mix resolved to 62 samples of the 64 budget) |
| `pipeline.json` | context 2048, chunk 512, `num_cores` 4, graphs `prefill` + `decode` |
| `required_cores` (derived) | 4 |
| outlined device weights | 7.3 GB |
| host params (fp16 embedding + untied lm_head) | 4.0 GB |
| quant bundle (`quantized_model/`, reusable with `--quantized-model`) | 12 GB |
| `validate_artifact` | PASS |

The calibration is real, not short-circuited: the 40 per-layer `down_proj` input scales are
all distinct and span 0.225 to 383.9, and the softmax LUT ranges are recorded values (`exp`
over [-398.58, 0], `reciprocal` over [1.0, 177.16]).

The LayerNorm1P export was checked against the raw checkpoint on the real weights: for
`model.layers.0.input_layernorm`, `exported_gain - checkpoint_weight` is **exactly 1.0** at
every one of the 4096 channels, and the bias is non-zero in all 4096, which is the tensor an
RMSNorm-only export would drop. The gain itself averages 0.157 because this checkpoint's raw 1P
weight averages -0.843 (range -1.156 to -0.081); that is the model, not a missing offset.

## Licence and intended use

`domyn/Domyn-Small-v1.0` is published by Domyn S.p.A. under its MIT-style licence
(`licenses/LICENSE-domyn-small.txt`, listed in `licenses/Credits.txt`). The only condition is
that the copyright and permission notice accompanies every copy or substantial portion of the
model. An artifact this recipe compiles is a substantial portion, so ship that notice with it.

Domyn releases the model as a general-purpose AI model under the EU AI Act and states that it
discharges the Article 53 provider obligations itself. Quantizing and compiling the weights here
does not make Axelera or you a provider of the model. If you build a product on the artifact,
the Article 50 transparency duties (telling users they are interacting with an AI, marking
generated content) fall on you as the provider or deployer of that product.

Domyn's model card states the model is not designed or evaluated for all downstream purposes,
and that developers should evaluate accuracy, safety and fairness before applying it to specific
scenarios, particularly high-risk ones. That caveat carries over unchanged to anything built from
this recipe.

## Adapt it to your own model

The `Domyn*` classes track the ARCHITECTURE, not this checkpoint's size, so another Nemotron
checkpoint (a fine-tune, or a different size) needs no new code: copy this folder, point
`repo_id` at your weights and change `MODEL_TAG`. `DomynConfig.from_hf_dict` refuses the
Nemotron variants this reference does not model - a `hidden_act` other than `relu2`, and
`attention_bias` / `mlp_bias` - rather than loading a silently-wrong graph, since
`QuantLinear` is bias-free and those weights would be dropped outright.

Set `calibration_texts` to your own domain if it differs from the default mix: per-tensor
W8A8 on this architecture is harder than on the SwiGLU models, because Q and K carry the
projection's full dynamic range into int8 SDPA (no QK-Norm), and `down_proj`'s input is
`relu(u)**2` - non-negative and quadratically scaled, so a relative error on `u` is doubled
on the value being quantized. `smoothquant_groups` also gives this MLP two singletons rather
than a `{gate, up}` pair, so there is less for SmoothQuant to equalise.
