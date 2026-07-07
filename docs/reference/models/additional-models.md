---
title: "Additional Models"
---
# Additional Models

## Additional Models without a YAML configuration

Models that have been verified on Metis but are not yet listed in the Model Zoo with dedicated YAML configurations. You can deploy them by adapting an existing template.
Accuracy drop is measured as FP32 top-1 accuracy minus quantized (int8 on AIPU) top-1 accuracy.

---

### Image Classification

These classification models have been compiled and accuracy-verified on Metis. To use one, copy the `mobilenetv4_small-imagenet.yaml` template and update the `timm_model_args.name` field and preprocessing configuration to match your target model.

| Model | Accuracy drop vs FP32 |
|-------|-----------------------|
| `dla34.in1k` | 0.59 |
| `dla60.in1k` | 0.55 |
| `dla60_res2net.in1k` | 0.15 |
| `dla102.in1k` | 0.03 |
| `dla169.in1k` | 0.27 |
| `efficientnet_es.ra_in1k` | 0.02 |
| `efficientnet_es_pruned.in1k` | 0.13 |
| `efficientnet_lite0.ra_in1k` | 0.22 |
| `dla46_c.in1k` | 1.54 |
| `fbnetc_100.rmsp_in1k` | 0.24 |
| `gernet_m.idstcv_in1k` | 0.05 |
| `gernet_s.idstcv_in1k` | 0.18 |
| `mnasnet_100.rmsp_in1k` | 0.28 |
| `mobilenetv2_050.lamb_in1k` | 0.92 |
| `mobilenetv2_120d.ra_in1k` | 0.44 |
| `mobilenetv2_140.ra_in1k` | 0.89 |
| `res2net50_14w_8s.in1k` | 0.17 |
| `res2net50_26w_4s.in1k` | 0.17 |
| `res2net50_26w_6s.in1k` | 0.06 |
| `res2net50_48w_2s.in1k` | 0.09 |
| `res2net50d.in1k` | 0.00 |
| `res2net101_26w_4s.in1k` | 0.19 |
| `res2net101d.in1k` | 0.08 |
| `resnet10t.c3_in1k` | 1.61 |
| `resnet14t.c3_in1k` | 0.85 |
| `resnet50c.gluon_in1k` | 0.03 |
| `resnet50s.gluon_in1k` | 0.19 |
| `resnet101c.gluon_in1k` | 0.08 |
| `resnet101d.gluon_in1k` | 0.10 |
| `resnet101s.gluon_in1k` | 0.18 |
| `resnet152d.gluon_in1k` | 0.15 |
| `selecsls42b.in1k` | 0.25 |
| `selecsls60.in1k` | 0.05 |
| `selecsls60b.in1k` | 0.20 |
| `spnasnet_100.rmsp_in1k` | 0.25 |
| `tf_efficientnet_es.in1k` | 0.26 |
| `tf_efficientnet_lite0.in1k` | 0.33 |
| `tf_mobilenetv3_large_minimal_100.in1k` | 1.68 |
| `wide_resnet101_2.tv2_in1k` | 0.26 |


## Additional Models with a YAML configuration
These models have a YAML configuration but they have yet to be fully verified for full speed and accuracy before they eventually move to the Model Zoo.


### Image Classification
| Model                                                                                          | ONNX                                                                                        | Repo                                                             | Resolution | Dataset     | Ref FP32 Top1 | Model license |
| :--------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ | :--------------------------------------------------------------- | :--------- | :---------- | ----------: | ------------: |
| [MobileNetV3-large](../../../ax_models/zoo/torchvision/classification/mobilenetv3_large-imagenet.yaml) | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/mobilenetv3_large-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 74.05         | BSD-3-Clause  |
| [MobileNetV3-small](../../../ax_models/zoo/torchvision/classification/mobilenetv3_small-imagenet.yaml) | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/mobilenetv3_small-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 67.67         | BSD-3-Clause  |


### Object Detection
| Model                                                                                                             | ONNX                        | Repo                                                                | Resolution | Dataset                   | Ref FP32 mAP | Model license |
| :---------------------------------------------------------------------------------------------------------------- | :-------------------------- | :------------------------------------------------------------------ | :--------- | :------------------------ | -----------: | ------------: |
| [YOLOv4](../../../ax_models/zoo/yolo/object_detection/yolov4-416-coco.yaml)                                               |                             | [&#x1F517;](https://github.com/AlexeyAB/darknet)                    | 416x416    | COCO2017                  | 25.00        | GPL-3.0       |
| [YOLOv4-CSP-Leaky](../../../ax_models/zoo/yolo/object_detection/yolov4-csp-leaky-coco.yaml)                               |                             | [&#x1F517;](https://github.com/WongKinYiu/CrossStagePartialNetworks)| 640x640    | COCO2017                  | 29.57        | GPL-3.0       |


### Keypoint Detection
| Model                                                                        | ONNX                                                                           | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Model license |
| :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: |
| [YOLO26x-pose](https://github.com/ultralytics/ultralytics)                   | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolo26xpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 72.75        | AGPL-3.0      |


### Semantic Segmentation
| Model                                                                                | ONNX                                                                                   | Repo                                                                             | Resolution | Dataset    | Ref FP32 mIoU | Model license |
| :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: |
| [YOLO26n-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26nsem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 71.42         | AGPL-3.0      |
| [YOLO26s-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26ssem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 76.50         | AGPL-3.0      |
| [YOLO26m-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26msem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 79.23         | AGPL-3.0      |
| [YOLO26l-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26lsem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 79.73         | AGPL-3.0      |
| [YOLO26x-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26xsem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 80.76         | AGPL-3.0      |


### Image Enhancement Super Resolution
| Model                                                        | ONNX                                                           | Repo                                                | Resolution | Dataset                         | Ref FP32 PSNR | Model license |
| :----------------------------------------------------------- | :------------------------------------------------------------- | :-------------------------------------------------- | :--------- | :------------------------------ | ------------: |  ------------: |
| [Real-ESRGAN-x4plus](https://github.com/xinntao/Real-ESRGAN) | [&#x1F517;](../../../ax_models/zoo/torch/real-esrgan-x4plus-onnx.yaml) | [&#x1F517;](https://github.com/xinntao/Real-ESRGAN) | 128x128    | SuperResolutionCustomSet128x128 | 24.77         |  BSD-3-Clause  |

---

## See also

- [Model Zoo](model-zoo.md) — fully integrated models with ready-to-use YAML configs
- [Deploy Custom Weights](../../tutorials/custom-weights.md) — how to use a YAML template for any model
