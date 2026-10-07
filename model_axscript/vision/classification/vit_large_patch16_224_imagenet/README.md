# ViT-L/16 (224) - AxScript

This folder is the AxScript (ALTO-direct) port of ViT-L/16,
the `vit_large_patch16_224.augreg_in21k_ft_in1k` checkpoint. The zoo already ships this model on
the axgraph backend, so this is a SECOND backend on the same card
(`configs/classification/vit_large_patch16_224_imagenet.py`).
The artifact is compiled locally, not downloaded.

Its zoo `task` is `classification`: it emits ImageNet-1k labels, not embeddings. The encoder runs
on the AIPU and the checkpoint's own linear classifier runs on the host.

## How it is compiled (AxScript, ALTO-direct)

The whole encoder (patch-embedding projection, the 24 identical blocks, and the final
LayerNorm) is ONE device graph running in a single dispatch. This card adds no ALTO kernel and no
code: a plain ViT is the LayerScale-free case of the block the shared reference already
implements, the recipe is the shared one (`../../shared/imagenet_vit.py`), and every shape below is
read off the checkpoint rather than declared here.

| | ViT-S/16 | ViT-L/16 |
|---|---|---|
| width | 384 | 1024 |
| blocks | 12 | 24 |
| attention heads | 6 | 16 (head dimension stays 64) |
| real tokens at 224 | 197 | 197 |
| padded sequence (multiple of 64) | 256 | 256 |
| patch projection K (C*P*P) | 768 | 768 |

Two pieces are specific to this upstream, and both live behind the `EncoderSource` seam
(`../../shared/timm_vit.py`): timm fuses q, k and v into one linear, which the weight mapping splits
into the reference's three projections, and timm stores a learned positional embedding for its own
resolution rather than interpolating one. Preprocessing comes from timm's resolved eval transform,
so the checkpoint's own normalization is what reaches both calibration and the run path.

The host does image normalization, an int8 pixel quantize plus pad, the output slice, and the
1000x1024 classifier. The classifier weights travel in the artifact's `host_params.npz`, so a
run needs no upstream checkpoint.

## How to build and run it

```bash
python -m model_axscript.vision build vit_large_patch16_224-imagenet                  # compile
python -m model_axscript.vision run vit_large_patch16_224-imagenet image.jpg          # top-5 classes
python -m model_axscript.vision benchmark vit_large_patch16_224-imagenet image.jpg    # + ms / fps
python -m model_axscript.vision run vit_large_patch16_224-imagenet image.jpg --device metis-0:2d:0
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
axzoo eval vit_large_patch16_224_imagenet --backend axscript --frames 2000
```

The gap is the meaningful quantity, because the same host head runs on both legs, so it isolates
encoder quantization. The card's published figures were measured this way over the full
50,000-image validation set (without `--frames`).
