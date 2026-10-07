# Depth Anything V2 Small

This folder is the AxScript (ALTO-direct) recipe for
monocular relative depth with Depth Anything V2 Small: a DINOv2 ViT-S/14 backbone under a DPT
neck and head, shipped already joined as `depth-anything/Depth-Anything-V2-Small-hf`.
The artifact is compiled locally, not downloaded.

The output is RELATIVE INVERSE depth: a larger value means nearer, and the scale is arbitrary.
That is not the metric depth in metres `dinov2_depth_vits14_nyu_linear` beside it emits, and
the two are not interchangeable in a consumer. Scoring it needs its own alignment: the zoo's
depth evaluator's median alignment fixes a scale without inverting one, so it would score this
output backwards. `evaluation.evaluator_kwargs={'align': 'disparity'}` on this card selects the
mode built for that shape instead: two least-squares affine fits, disparity space then depth
space, the protocol Depth Anything V2 itself reports numbers under.

## How it is compiled

Everything runs as ONE device graph in a single dispatch: the patch-embedding convolution, the
12 encoder blocks, the four intermediate-layer taps the neck reads, the reassemble
convolutions, the bilinear fusions and the head. There is no host-side head and no second
model.

This card has ONE builder, not a choice of two. `imported_dino` authors a ViT graph and a DPT
neck is not one, so `implementation` is `depth-anything-axten` and the graph comes from
`axelera.compiler.axten`.

The backbone is a `Dinov2Backbone`, not a `Dinov2Model`: it taps several intermediate layers
and hands each to the neck, which reassembles them at four scales and fuses them top-down.

## The input is pinned SQUARE, and the upstream's is not

The published processor preserves each image's aspect ratio, so every image would reach the
model at its own shape. One compiled graph is one shape, so the recipe turns that off and
pins a square (`DepthAnythingSource.make_processor`).

Two consequences, and only the first matters for a fidelity number:

* A device-versus-fp comparison is unaffected. Both legs read identical pixels.
* An absolute accuracy measured here is NOT comparable to a published one, which uses the
  aspect-preserving resize and runs each image at more than one scale.

## How to build and run it

```bash
python -m model_axscript.vision build depth_anything_v2_small
python -m model_axscript.vision run depth_anything_v2_small image.jpg
python -m model_axscript.vision benchmark depth_anything_v2_small image.jpg
python -m model_axscript.vision run depth_anything_v2_small image.jpg --device metis-0:2d:0
```

`run` prints, per image, the range the device predicted and how far it is from the fp
reference on the same pixels (AbsRel and cosine). The compiled artifact is device-agnostic, so
naming a different `--device` reuses the same build with no recompile.

**Validated input sizes:** 224

Any other `image_size` is refused at build time, by an error naming the whole measured set.
Sizes outside it have not been measured on this part, and an unmeasured one fails as a device
hang rather than an error, so the set is widened only by a measurement, never to make a build
pass. The live list is `supported_image_sizes` on the configuration this model's `make_config`
returns, and the refusal prints it.

## Calibration

The default calibration set is `tanganke/nyuv2`, indoor scenes, and NOT the COCO set the other
vision cards use, for the reason the sibling card's README gives: a depth model calibrated on
outdoor-heavy images sets int8 ranges that clip where its own output lives.
