![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Build a cascade pipeline

- [Build a cascade pipeline](#build-a-cascade-pipeline)

A cascade pipeline with ssd-mobilenetv1 to detect selected objects and resnet50 for further
classification is included with the SDK to demonstrate this functionality. The YAML pipeline
definition for this example is found at `ax_models/cascade/ssd-mobilenetv1-resnet50.yaml`. You can
deploy and run the model by:

```bash
inference.py ssd-mobilenetv1-resnet50 <usb/video>
```

In this example, the object detector will find all bottles, and the classifier further categorizes
the object into subcategories like red wine, white wine, and beer. Since "bottle" is among the
classes identified by the COCO dataset, and ImageNet includes specific categories such as
"red wine," "white wine," and "beer," it's feasible to seamlessly integrate the detector and
classifier trained on these datasets. In object detection pipelines, we can easily add a "tracker"
to track objects of a certain class. Below is a high-level representation of the pipeline:

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
            algorithm: sort # support list: oc-sort, bytetrack, sort, scalarmot
  - ResNet50-ImageNet1K:
      input:
        type: image
        source: roi
        where: SSD-MobileNetV1-COCO
        classes: bottle
        which: CENTER # AREA, SCORE, CENTER
        top_k: 10
        expand_margin: 0
        resize_width: 224
        resize_height: 224
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

In actual applications, this pipeline can have many extended uses, such as first detecting cars,
then classifying the cars into corresponding makes/models or model years.
