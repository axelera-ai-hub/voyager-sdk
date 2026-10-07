# DINOv2-L/14 + ADE20K linear head

This folder is the AxScript (ALTO-direct) recipe for
semantic segmentation with a DINOv2 ViT-L/14 backbone and Meta's ADE20K linear head.
The artifact is compiled locally, not downloaded.

It is the large sibling of `dinov2_seg_vits14_ade20k_linear`, and everything about
the recipe is the same: the head is published as a file rather than as a model card, so the
model is assembled from a backbone plus that head, with the head's BatchNorm folded into its
1x1 convolution first. See that model's README for how the graph is put together and why the
backbone comes from the imported implementation.

## How to build and run it

```bash
python -m model_axscript.vision build dinov2_seg_vitl14_ade20k_linear
python -m model_axscript.vision run dinov2_seg_vitl14_ade20k_linear image.jpg
python -m model_axscript.vision benchmark dinov2_seg_vitl14_ade20k_linear image.jpg
```

`run` prints, per image, how often the device takes the same ADE20K label as the fp reference
and the class mix it predicted.

**Validated input sizes:** 518

Any other `image_size` is refused at build time, by an error naming the whole measured set.
Sizes outside it have not been measured on this part, and an unmeasured one fails as a device
hang rather than an error, so the set is widened only by a measurement, never to make a build
pass. The live list is `supported_image_sizes` on the configuration this model's `make_config`
returns, and the refusal prints it.
