# DINOv2 ViT-S/14

This folder is the AxScript (ALTO-direct) port of the DINOv2
ViT-S/14 backbone, the checkpoint upstream publishes as `facebook/dinov2-small`.
The artifact is compiled locally, not downloaded.

DINOv2-small is a headless ViT backbone, so its `task` is `feature_extraction` (it emits
embeddings, not labels); downstream heads (segmentation, depth, detection, ...) are separate
cards under their own task, reusing this backbone. The device recipe is the DINOv2
implementation under `model_axscript/llm/models/dino/`, reached through this family's
`EncoderSource` seam (`model_axscript/vision/shared/`), so this package is a configuration
and the seams its driver calls, and nothing else.

## How it is compiled (AxScript, ALTO-direct)

The whole model - patch-embedding conv, the 12 identical encoder blocks, and the final
LayerNorm - is authored as ONE device graph and runs in a single `_run_graph` dispatch
(pixels -> last_hidden_state). The host does only image-processor normalization, an int8 pixel
quantize + pad, and an output slice. It reuses the LLM device op stack (int8 matmul + dequant,
scaled-dot-product attention, GELU LUT) without any autoregressive-decoder machinery (no KV
cache, no causal mask, no RoPE, no token embedding / lm_head). Two pieces the decoder-LLM path
lacks and this block adds:

- **LayerNorm** (mean-subtract + gamma + beta), not the LLM's scale-only RMSNorm.
- a plain **GELU MLP**, not a gated SwiGLU.
- **bidirectional attention**: `QuantSDPA` is called with `attn_mask=None` (full attention over
  all patches); the device analogue is an all-zero mask. DINOv2's token count (num_patches + 1
  CLS) is not a multiple of 64, so the sequence is padded to a multiple and the padded key
  columns are masked out with an additive `-127` sentinel (`_MASK_NEG`).

## How to build and run it

One command line reaches every verb; the compiled artifact is device-agnostic, so switching
the run device never recompiles:

```bash
python -m model_axscript.vision build dinov2-vits14                # compile the artifact
python -m model_axscript.vision run dinov2-vits14 image.jpg        # build if needed, then extract features
python -m model_axscript.vision benchmark dinov2-vits14 image.jpg  # + ms / fps
python -m model_axscript.vision run dinov2-vits14 image.jpg --device metis-0:2d:0  # a named device
```

Calibration uses the GPU automatically (`--cal-device cpu` to force CPU) on real natural images
(COCO by default - see `vision/shared/calibration.py` for why the images must be DOWNSCALED, not
upscaled, to the input resolution). The `facebook/dinov2-small` checkpoint is public (no Hugging
Face token). `--device` picks one Metis device by name or index; because the artifact is
device-agnostic, `run ... --device <other>` reuses the same build with no recompile.

**Validated input sizes:** 224

Any other `image_size` is refused at build time, by an error naming the whole measured set.
Sizes outside it have not been measured on this part, and an unmeasured one fails as a device
hang rather than an error, so the set is widened only by a measurement, never to make a build
pass. The live list is `supported_image_sizes` on the configuration this model's `make_config`
returns, and the refusal prints it.

## Bring your own weights (upstream is pluggable)

HuggingFace is the DEFAULT upstream, not a hard requirement. Weight sourcing lives behind
`vision/shared/source.py::EncoderSource`; `vision/shared/hf.py::HFSource` is one
implementation. To bring the same
ViT architecture from a different upstream (torchvision, a local checkpoint, our export + SoT
flow), implement `EncoderSource` (load the fp reference, its processor, its per-block config, and
map its weights into `reference.DinoV2Model`) and pass it as `EncoderConfig(source=...)`. The
on-chip recipe (the torch fake-quant reference + the ALTO graph) is unchanged; only where the fp
weights come from differs. `repo_id` stays the identity used for naming / `meta.json`.

Match the calibration data to YOUR domain. `calibration_dataset` defaults to a general
image set (`sayakpaul/coco-30-val-2014`), and the int8 scales and activation LUT ranges are
fitted from whatever you supply. Out-of-domain calibration does not error - it under-covers
the recorded LayerNorm `inv_sqrt` range, so real inputs go out of range and the LUT
extrapolates, which shows up as some images scoring fine and others collapsing rather than
as a graceful drop. Point `calibration_dataset` at images from your deployment domain.

## Accuracy

A bare backbone has no head of its own to score, so the numeric proof is a training-free kNN
top-1 over CLS features against tiny-imagenet, run twice - device int8 vs HF fp - reporting
both and the quantization **gap**:

```bash
axzoo eval dinov2_vits14 --backend axscript
```

This runs through the zoo's `feature_extraction` evaluator. The device-vs-fp gap converges far
faster than absolute top-1, so the gap is the meaningful quantity.
