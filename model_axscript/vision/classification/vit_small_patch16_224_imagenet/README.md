# ViT-S/16 (224) - AxScript

This folder is the AxScript (ALTO-direct) port of ViT-S/16, the
`vit_small_patch16_224.augreg_in21k_ft_in1k` checkpoint. The
zoo already ships this model on the axgraph backend, so this is a SECOND backend on the same card
(`configs/classification/vit_small_patch16_224_imagenet.py`).
The artifact is compiled locally, not downloaded.

Its zoo `task` is `classification`: it emits ImageNet-1k labels, not embeddings. The encoder runs
on the AIPU and the checkpoint's own linear classifier runs on the host.

## How it is compiled (AxScript, ALTO-direct)

The whole encoder (patch-embedding projection, the 12 identical blocks, and the final LayerNorm)
is ONE device graph running in a single dispatch, exactly as for `dinov2_vits14`: this card adds no
ALTO kernel and no code. A plain ViT is the LayerScale-free case of the block the shared reference
already implements, so the recipe is the shared one (`../../shared/imagenet_vit.py`), driven by this
model's shapes and weights. ViT-B/16 and ViT-L/16 are the same recipe at a different width and
depth:

| | DINOv2-small | ViT-S/16 |
|---|---|---|
| patch size | 14 | 16 |
| real tokens at 224 | 257 | 197 |
| padded sequence (multiple of 64) | 320 | 256 |
| patch projection K (C*P*P) | 588, padded to 640 | 768 |
| LayerScale | learned per block | absent, so the reference keeps `ls1`/`ls2` at ones |
| upstream | HuggingFace `Dinov2Model` | timm `VisionTransformer` |
| output | tokens | ImageNet-1k logits (host head) |

Two pieces are specific to this upstream, and both live behind the `EncoderSource` seam
(`../../shared/timm_vit.py`): timm fuses q, k and v into one linear, which the weight mapping splits into the
reference's three projections, and timm stores a learned positional embedding for its own
resolution rather than interpolating one. Preprocessing comes from timm's resolved eval transform,
so the checkpoint's own normalization (mean and std 0.5, crop_pct 0.9, bicubic) is what reaches
both calibration and the run path.

The host does image normalization, an int8 pixel quantize plus pad, the output slice, and the
1000x384 classifier. The classifier weights travel in the artifact's `host_params.npz`, so a run
needs no upstream checkpoint.

## How to build and run it

```bash
python -m model_axscript.vision build vit_small_patch16_224-imagenet                  # compile
python -m model_axscript.vision run vit_small_patch16_224-imagenet image.jpg          # top-5 classes
python -m model_axscript.vision benchmark vit_small_patch16_224-imagenet image.jpg    # + ms / fps
python -m model_axscript.vision run vit_small_patch16_224-imagenet image.jpg --device metis-0:2d:0
```

Calibration uses the GPU automatically (`--cal-device cpu` to force CPU) on real natural images
(COCO by default). `--device` picks one Metis device by name or index; because the artifact is
device-agnostic, naming a different `--device` on a `run` reuses the same build with no recompile.

**Validated input sizes:** 224

Any other `image_size` is refused at build time, by an error naming the whole measured set.
Sizes outside it have not been measured on this part, and an unmeasured one fails as a device
hang rather than an error, so the set is widened only by a measurement, never to make a build
pass. This checkpoint also stores one positional embedding for 224, so another size needs a
checkpoint that carries it. The live list is `supported_image_sizes` on the configuration this
model's `make_config` returns, and the refusal prints it.

## Bring your own weights

The upstream is pluggable: `TimmViTSource` is one `EncoderSource`
(`model_axscript/vision/shared/source.py`). A fine-tune of the same architecture only changes
`repo_id` in `make_config`. A different upstream (torchvision, a local checkpoint, our export and
SoT flow) implements `EncoderSource` and maps its weights into `reference.DinoV2Model`; the
mapping helper `load_vit_state_dict` already covers any timm-layout plain ViT, and refuses a
checkpoint carrying LayerScale, QK-norm or a gated MLP rather than dropping those weights in
silence.

Match the calibration data to YOUR domain (`calibration_dataset`): the int8 scales and the
activation LUT ranges are fitted from whatever you supply.

## Accuracy

Top-1 on a labelled ImageNet val sample, measured twice on the same images (device int8 and the fp
reference, through the same host head) and reported with the gap:

```bash
axzoo eval vit_small_patch16_224_imagenet --backend axscript --frames 2000
```

The gap is the meaningful quantity, because the same host head runs on both legs, so it isolates
encoder quantization. The card's published figures were measured this way over the full
50,000-image validation set (without `--frames`).
