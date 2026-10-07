# DINOv2-S/14 + NYU linear depth head

This folder is the AxScript (ALTO-direct) recipe for
monocular depth with a DINOv2 ViT-S/14 backbone and Meta's NYU linear depth head. The artifact is
compiled locally, not downloaded.

The head is published as a file rather than as a model card
(`dinov2_vits14_nyu_linear_head.pth` on `dl.fbaipublicfiles.com`), so this model cannot be
reached by a checkpoint id alone: it is assembled from a backbone plus that head, at build
time. Despite Meta calling it `BNHead` it carries no BatchNorm, so there is nothing to fold.

The output is one distance per input pixel, in metres. That is metric depth, not the
scale-free relative depth `depth_anything_v2_small` beside it emits, and the two are not
interchangeable in a consumer.

## How it is compiled

Everything runs as ONE device graph in a single dispatch: the patch-embedding convolution,
the 12 encoder blocks, the CLS broadcast and channel concatenation, the 4x bilinear upsample,
the 1x1 convolution to 256 depth bins, the bin normalisation and the bin-centre weighted sum,
and the resize back to the input grid. There is no host-side head and no second model. The
host does image-processor normalization, an int8 pixel quantize and pad, and nothing else.

This card has ONE builder, not a choice of two. `imported_dino` authors a ViT graph with an
optional baked LINEAR head, and this head is not linear, so `implementation` is
`dinov2-depth-axten` and the graph comes from `axelera.compiler.axten`.

The head reads the last encoder block's tokens BEFORE the backbone's final LayerNorm, which
is what Meta's `final_norm=False` means. Reading the normed tokens instead compiles and runs,
and is a different model.

## How to build and run it

```bash
python -m model_axscript.vision build dinov2_depth_vits14_nyu_linear
python -m model_axscript.vision run dinov2_depth_vits14_nyu_linear image.jpg
python -m model_axscript.vision benchmark dinov2_depth_vits14_nyu_linear image.jpg
python -m model_axscript.vision run dinov2_depth_vits14_nyu_linear image.jpg --device metis-0:2d:0
```

`run` prints, per image, the depth range the device predicted and how far it is from the fp
reference on the same pixels (AbsRel and cosine). The compiled artifact is device-agnostic, so
naming a different `--device` reuses the same build with no recompile.

That AbsRel is a FIDELITY number, device against the model's own fp32 leg. The card's accuracy
is a different measurement, against ground truth, and comes from `axzoo eval`.

**Validated input sizes:** 224

Any other `image_size` is refused at build time, by an error naming the whole measured set.
Sizes outside it have not been measured on this part, and an unmeasured one fails as a device
hang rather than an error, so the set is widened only by a measurement, never to make a build
pass. The live list is `supported_image_sizes` on the configuration this model's `make_config`
returns, and the refusal prints it.

## Calibration

The default calibration set is `tanganke/nyuv2`, indoor scenes, and NOT the COCO set the other
vision cards use. Indoor depth spans roughly 0.5 to 10 m; calibrating a depth model on
outdoor-heavy images sets int8 ranges that clip where this model's output actually lives, and
no evaluator's scale alignment recovers a clipped value.

Match the calibration data to YOUR domain for the same reason. Out-of-domain calibration does
not error: it shows up as some scenes reading plausible depths and others saturating.
