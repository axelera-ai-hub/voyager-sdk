![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Custom model deployment [Experimental]

- [Custom model deployment \[Experimental\]](#custom-model-deployment-experimental)
  - [Voyager APIs](#voyager-apis)
  - [Related documents](#related-documents)

The Voyager SDK is built on a flexible and modular framework that makes it easy to deploy
custom models on Metis-based systems through the use of simple Python
APIs, standardized dataset adapters and evaluators.

Deploying a model is sometimes as simple as combining pretrained weights from PyTorch or
ONNX with an Axelera [dataset adapter](/docs/tutorials/custom_weights.md) that is compatible with your existing data,
for example using an industry-standard labelling format.
If a suitable data adapter does not already exist then you can create your own custom adapter, and if you wish to measure an accuracy metric not yet supported you can implement
your own custom evaluator. Once validated on one model, you can apply your custom dataset adapter and evaluators with other models trained on the same dataset.
The only requirement is that the model, dataset and evaluator are all implemented for same *task category*, for example an object detector and dataset providing bounding boxes
which is evaluated using the metric *mean average precision* (mAP).

To run a deployed model within the Voyager framework you must also implement a *decoder* that converts the raw tensor output from your model to the Voyager
metadata representation for the model task category. You specify this decoder in the Voyager YAML file for your model along with any host-based image pre-processing and post-processing
operators required by the model, such as resizing, letterboxing, tensor conversion and non-maximal suppression (NMS). The Voyager toolchain uses this YAML file to generate an executable
pipeline that implements the complete model end-to-end on your target platform. If required, it is also possible to implement and register your own custom operators
for use by the toolchain.

> [!CAUTION]
> The Axelera model compiler removes any image operators from the source model definition prior to compilation
> for Metis hardware. You must therefore ensure that
> these operators are correctly specified in your Voyager YAML file to be able to run the model end-to-end.
> Limited beta-level features are provided in the toolchain to support this workflow.

## Voyager APIs

The Voyager framework provides APIs for implementing deployable models, custom pre and post-processing operators,
dataset adapters and evaluators. The API design ensures that these components can be defined and implemented independently of
one another, interfacing using a common representation of metadata for each supported task category (`AxTaskMeta`).

| API | Description | Example |
| :-- | :---------- | :------ |
| `types.Model` | A deployable PyTorch or ONNX model. Specified in Voyager YAML files in the `models` section (fields `class` and `class_path`) | [`AxYolo`](/ax_models/yolo/ax_yolo.py) |
| `types.DataAdapter` | A dataset adapter that outputs images and related ground truth metadata in the format `AxTaskMeta`. Specified in Voyager YAML files in the `datasets` section (fields `class` and `class_path`) | [`ObjDataAdapter`](/ax_datasets/objdataadapter.py) |
| `types.Evaluator` | A evaluator that measures model accuracy by comparing model inference results with ground truth data (both with the type `AxTaskMeta`). Defined as a property of a data adapter (class method `evaluator`) | [`YoloEvaluator`](/ax_evaluators/yolo_eval.py) |
| `AxOperator` | A pipeline element implemented on the host processor, typically image pre-processing and post-processing operators | [`Resize`](/axelera/app/operators/preprocessing.py) |
| `AxOperator` model decoder | An `AxOperator` that converts model raw output tensor data to `AxTaskMeta` metadata | [`YoloDecode`](/ax_models/decoders/yolo.py) |

## Related documents

The [custom weights tutorial](/docs/tutorials/custom_weights.md) provides examples of dataset adapters.

The [advanced deployment tutorials](/ax_models/tutorials/general/tutorials.md) explains experimental advanced deployment options.
