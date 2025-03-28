![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Preparing ONNX Models for Metis Platform

- [Preparing ONNX Models for Metis Platform](#preparing-onnx-models-for-metis-platform)
  - [Getting Started: Fruits-360 Classification Example](#getting-started-fruits-360-classification-example)
    - [Dataset and Model Setup](#dataset-and-model-setup)
      - [Option 1: Train the Model Yourself](#option-1-train-the-model-yourself)
      - [Option 2: Use a Pre-trained Model](#option-2-use-a-pre-trained-model)
  - [Next Steps](#next-steps)
- [Appendix: Converting Models to ONNX](#appendix-converting-models-to-onnx)
  - [1. Converting PyTorch Models to ONNX](#1-converting-pytorch-models-to-onnx)
  - [2. Converting TensorFlow Models to ONNX](#2-converting-tensorflow-models-to-onnx)
    - [a. TensorFlow 2.x: Converting SavedModel to ONNX](#a-tensorflow-2x-converting-savedmodel-to-onnx)
    - [b. TensorFlow 2.x: Converting HDF5 Model to ONNX](#b-tensorflow-2x-converting-hdf5-model-to-onnx)
    - [TensorFlow 1.x: Converting Frozen Graph to ONNX](#tensorflow-1x-converting-frozen-graph-to-onnx)
  - [3. Converting PaddlePaddle Models to ONNX](#3-converting-paddlepaddle-models-to-onnx)
  - [4. Reducing ONNX Opset Version](#4-reducing-onnx-opset-version)

This tutorial is the starting point of the ONNX series, guiding you through the process of preparing your ONNX models for deployment on the Metis platform. By following this series, you'll learn how to fully leverage the Voyager SDK for both model and pipeline deployment.

## Getting Started: Fruits-360 Classification Example

To illustrate the concepts in this tutorial, we’ll use a practical example: deploying a ResNet34 model trained on the Fruits-360 dataset for fruit and vegetable classification.

### Dataset and Model Setup

You have two options to get started:

#### Option 1: Train the Model Yourself
Under the Voyager SDK root directory, run the training script:

```bash
cd ${AXELERA_FRAMEWORK}
python ax_models/tutorials/resnet34_fruit360.py
```

This script will:

- Download the Fruits-360 dataset from GitHub and save it to `$AXELERA_FRAMEWORK/data/fruits-360-100x100`. The dataset includes both training and validation sets and requires approximately 1.4GB of disk space.
- Save the pre-trained weights to `~/.cache/axelera/weights/tutorials/resnet34_fruits360.pth`.
- Export the ONNX model to `~/.cache/axelera/weights/tutorials/resnet34_fruits360.onnx`.

The training script will automatically use GPU or Apple MPS if available; otherwise, it will default to CPU.

#### Option 2: Use a Pre-trained Model

If you prefer to skip the training process, you can use the pre-trained model and dataset by following these steps:

1. Download the dataset:

You can download the test dataset by running the following command:
```bash
wget --show-progress -P $AXELERA_FRAMEWORK/data/ https://d1o2y3tc25j7ge.cloudfront.net/artifacts/tutorials/fruits-360-100x100-testset.zip && \
unzip -q $AXELERA_FRAMEWORK/data/fruits-360-100x100-testset.zip -d $AXELERA_FRAMEWORK/data/ && \
rm $AXELERA_FRAMEWORK/data/fruits-360-100x100-testset.zip
```

Alternatively, you can download the full dataset by running:
```bash
python ax_models/tutorials/resnet34_fruit360.py --download
```


2. Download the pre-trained weights and class names:

```bash
mkdir -p ~/.cache/axelera/weights/tutorials/ && \
wget --show-progress -O ~/.cache/axelera/weights/tutorials/resnet34_fruits360.onnx https://d1o2y3tc25j7ge.cloudfront.net/artifacts/tutorials/resnet34_fruits360_classifier.onnx && \
wget --show-progress -O ~/.cache/axelera/weights/tutorials/resnet34_fruits360.pth https://d1o2y3tc25j7ge.cloudfront.net/artifacts/tutorials/resnet34_fruits360.pth && \
wget --show-progress -O ~/.cache/axelera/weights/tutorials/fruits360.names https://d1o2y3tc25j7ge.cloudfront.net/artifacts/tutorials/fruits360.names
```
The model is trained to classify 131 different types of fruits and vegetables. The `fruits360.names` file contains all class labels, which you can reference to understand the model's classification capabilities. We download both pt and onnx models, so you can play with both.


3. Download the representative images:

We randomly sample 186 images from the validation set to use as representative images. You can prepare the representative images by running the following command:

```bash
wget --show-progress -P ~/.cache/axelera/fruits-360-100x100/ https://d1o2y3tc25j7ge.cloudfront.net/artifacts/tutorials/fruits-360-100x100-repr200.zip && \
unzip -q ~/.cache/axelera/fruits-360-100x100/fruits-360-100x100-repr200.zip -d $AXELERA_FRAMEWORK/data && \
rm ~/.cache/axelera/fruits-360-100x100/fruits-360-100x100-repr200.zip
```

## Next Steps

Once your model is set up using either of the above options, you’re ready to move on to the next tutorials in this series. These will cover advanced techniques for model and pipeline deployment using the Voyager SDK. Refer to [tutorials.md](/ax_models/tutorials/general/tutorials.md) for detailed instructions.

---
# Appendix: Converting Models to ONNX

If you have a model in another format (e.g., PyTorch, TensorFlow, PaddlePaddle, or other frameworks), you can always convert it to ONNX for deployment on the Metis platform. Our compiler offers strong support for **opset=14**, with **opset=17** being the maximum version currently supported by the Voyager SDK. The ONNX [Version Converter](https://github.com/onnx/tutorials/blob/master/tutorials/VersionConversion.md) is available to further assist in reducing the opset version of your ONNX model if needed.


## 1. Converting PyTorch Models to ONNX
To convert a PyTorch model to ONNX, use the torch.onnx.export function. Here's an example:

```python
import torch
import torchvision.models as models

# Load your PyTorch model (e.g., ResNet34)
model = models.resnet34(pretrained=True)
model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "resnet34.onnx",
    opset_version=17, # Specify opset version
    input_names=["input"],
    output_names=["output"]
)
```

## 2. Converting TensorFlow Models to ONNX
TensorFlow models can be saved in different formats, such as SavedModel, HDF5, or frozen graphs. Below are examples for each format. Note: TensorFlow 1.x and TensorFlow 2.x have different workflows, which are explained below.

### a. TensorFlow 2.x: Converting SavedModel to ONNX
```python
import tensorflow as tf
import tf2onnx

# Load your TensorFlow SavedModel
model = tf.saved_model.load("saved_model_directory")

# Convert to ONNX
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_saved_model("saved_model_directory", input_signature=spec, opset=17)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
```

### b. TensorFlow 2.x: Converting HDF5 Model to ONNX
```python
import tensorflow as tf
import tf2onnx

# Load your TensorFlow HDF5 model
model = tf.keras.models.load_model("model.h5")

# Convert to ONNX
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=17)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
```

### TensorFlow 1.x: Converting Frozen Graph to ONNX

For TensorFlow 1.x models, you need to first freeze the graph and then convert it to ONNX. Here's an example:

```python
import tensorflow as tf
import tf2onnx

# Load your TensorFlow 1.x frozen graph
model = tf.graph_util.import_graph_def(graph_def, name="")

# Convert to ONNX
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_graph_def(
    graph_def=frozen_graph_path,
    input_names=["input:0"],
    output_names=["output:0"],
    opset=17
)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
```

## 3. Converting PaddlePaddle Models to ONNX
For PaddlePaddle models, use the paddle2onnx library. Here's an example:

```bash
pip install paddle2onnx
```

```python
import paddle
from paddle2onnx import export

# Load your PaddlePaddle model
model = paddle.jit.load("paddle_model_directory")

# Convert to ONNX
export(
    model,
    input_shape=[[1, 3, 224, 224]],
    save_file="model.onnx",
    opset_version=17
)
```

## 4. Reducing ONNX Opset Version
If your ONNX model uses a higher opset version than 17, you can reduce it using the onnx Python library. Here's an example:

```python
import onnx
from onnx import version_converter

# Load the ONNX model
model_path = "model_with_higher_opset.onnx"
model = onnx.load(model_path)

#Convert to opset 17
converted_model = version_converter.convert_version(model, 17)

# Save the converted model
onnx.save(converted_model, "model_opset17.onnx")
```

Not all operators can be reduced to Opset 17. If you encounter this error, please try re-exporting the ONNX from the source model or file an issue on the Axelera community
