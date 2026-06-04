---
title: "Cascaded Pipelines"
---
# Cascaded Pipelines

How to chain two models together so the output of the first feeds the input of the second.

A typical use case: a detector finds objects of interest, then a classifier runs on each detected region to make a finer prediction.

---

## How it works

In a cascade pipeline, the second model's `input` section specifies `source: roi` — meaning it receives region-of-interest crops from a previous model's detections, not the full frame:

```yaml
pipeline:
  - first-model:
      postprocess:
        - decoder:
            label_filter: bottle     # only pass 'bottle' detections downstream

  - second-model:
      input:
        type: image
        source: roi                  # crop from detections, not the full frame
        where: first-model           # which upstream task to take ROIs from
        label_filter: bottle         # filter by class
        which: CENTER                # selection strategy: CENTER, AREA, or SCORE
        top_k: 10                    # max number of ROIs to process per frame
```

The `which` field controls which crops get passed when there are many detections:
- `CENTER` — crop closest to the frame center
- `AREA` — largest crop by bounding-box area
- `SCORE` — highest-confidence detection

---

## Example: detect bottles → classify wine type

The SDK ships a ready-to-run cascade: SSD-MobileNetV1 detects objects (filtering for `bottle`), then ResNet50 classifies each detected bottle into subcategories (red wine, white wine, beer).

```bash
./inference.py ssd-mobilenetv1-resnet50 usb:0
```

> [!NOTE]
> The first run will compile both models for your hardware, which takes longer than single-model pipelines. This is a one-time process — subsequent runs use the cached compiled models.


The YAML definition for this pipeline is at `ax_models/cascade/ssd-mobilenetv1-resnet50.yaml`:

```yaml
pipeline:
  - SSD-MobileNetV1-COCO:
      template_path: $AXELERA_FRAMEWORK/pipeline-template/ssd-tensorflow.yaml
      postprocess:
        - decode-ssd-mobilenet:
            conf_threshold: 0.4
            label_filter: bottle
            overwrite_labels: True
        - tracker:
            algorithm: sort    # optional: adds persistent track IDs

  - ResNet50-ImageNet1K:
      input:
        type: image
        source: roi
        where: SSD-MobileNetV1-COCO
        label_filter: bottle
        which: CENTER
        top_k: 10
      preprocess:
        - torch-totensor:
        - normalize:
            mean: 0.485, 0.456, 0.406
            std: 0.229, 0.224, 0.225
      postprocess:
        - topk:
            k: 1
            labels: $$labels$$
            num_classes: $$num_classes$$
```

The optional `tracker` step adds persistent object IDs across frames before the classifier runs. This is useful if you want to track which tracked object has which wine classification over time.

---

## Building your own cascade

1. **Start with two Model Zoo models** that cover your detection and classification tasks.
2. **Copy the cascade YAML pattern** — set `label_filter` in the detector's postprocessing to the class you want to classify.
3. **Set `where:` in the second model's input** to the name of the first pipeline task.
4. **Choose `which:`** based on your use case (`CENTER` is a good default).
5. **Deploy both models together:**

```bash
./deploy.py customers/mymodels/my-cascade.yaml
```

---

## Adding a tracker

Any object detection stage can include a tracker before the ROI crops are passed downstream:

```yaml
postprocess:
  - decode-ssd-mobilenet:
      label_filter: car
  - tracker:
      algorithm: oc-sort    # sort, oc-sort, bytetrack, scalarmot
```

See [GStreamer Operators — inplace_tracker](../reference/pipeline/gst-operators.md#inplace_tracker) for tracking algorithm options.

---

## See also

- [Deploy Custom Weights](custom-weights.md) — use your own models in a cascade
- [GStreamer Operators](../reference/pipeline/gst-operators.md) — the operators used in pipeline YAML
- [Run Inference in Python](run-inference-in-python.md) — access per-task metadata from cascades in Python
