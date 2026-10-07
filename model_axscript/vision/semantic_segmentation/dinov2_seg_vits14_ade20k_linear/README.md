# DINOv2-S/14 + ADE20K linear head

This folder is the AxScript (ALTO-direct) recipe for
semantic segmentation with a DINOv2 ViT-S/14 backbone and Meta's ADE20K linear head.
The artifact is compiled locally, not downloaded.

The head is a BatchNorm and a 1x1 convolution onto the 150 ADE20K classes, published as a
file rather than as a model card, so this model cannot be reached by a checkpoint id alone:
it is assembled from a backbone plus that head. The assembly happens at build time, and the
BatchNorm is folded into the convolution first, so what reaches the device is one linear.

## How it is compiled

Everything runs as ONE device graph in a single dispatch: the patch-embedding convolution,
the 12 identical encoder blocks, the final LayerNorm, and the head. Because the head is fused
into the final kernel, the graph emits per-token class LOGITS directly rather than features,
so there is no host-side head and no second model. The host does image-processor
normalization, an int8 pixel quantize and pad, and an output slice.

Token 0 of the output is the CLS row, which the head was never trained to classify. The patch
rows are `[1:]` and reshape to the `grid x grid` class map; a consumer upsamples that map to
the input resolution and takes the argmax per pixel. At the shipped 518 the grid is 37x37, the
resolution the head was trained at; at 224 it is 16x16, which costs less to run and gives a
fifth of the map.

The backbone recipe is the DINOv2 implementation under `model_axscript/llm/models/dino/`,
reached through this family's `EncoderSource` seam rather than copied. On the same
checkpoint, calibration set and device, that implementation agrees with the fp reference at
CLS cosine 0.98, where the earlier copy of it reached 0.92.

## How to build and run it

```bash
python -m model_axscript.vision build dinov2_seg_vits14_ade20k_linear
python -m model_axscript.vision run dinov2_seg_vits14_ade20k_linear image.jpg
python -m model_axscript.vision benchmark dinov2_seg_vits14_ade20k_linear image.jpg
python -m model_axscript.vision run dinov2_seg_vits14_ade20k_linear image.jpg --device metis-0:2d:0
```

`run` prints, per image, how often the device takes the same ADE20K label as the fp reference
and the class mix it predicted. The compiled artifact is device-agnostic, so naming a
different `--device` reuses the same build with no recompile.

Calibration uses real natural images (COCO by default) and the GPU when one is present
(`--cal-device cpu` forces CPU). The backbone checkpoint and the head file are both public.

**Validated input sizes:** 224, 518

Any other `image_size` is refused at build time, by an error naming the whole measured set.
Sizes outside it have not been measured on this part, and an unmeasured one fails as a device
hang rather than an error, so the set is widened only by a measurement, never to make a build
pass. The live list is `supported_image_sizes` on the configuration this model's `make_config`
returns, and the refusal prints it.

## Bring your own weights

The backbone comes from whatever `EncoderSource` the configuration names, and HuggingFace is
only the default one (see `model_axscript/vision/shared/source.py`). The head is selected by
the backbone size the configuration declares, because Meta publishes one head per size and
pairing a head with a backbone it was not trained on is refused by name.

Match the calibration data to YOUR domain. Out-of-domain calibration does not error: it
under-covers the recorded LayerNorm range, which shows up as some images segmenting well and
others collapsing rather than as a graceful drop.
