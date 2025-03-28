![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Preparing PyTorch Models for Metis Platform

- [Preparing PyTorch Models for Metis Platform](#preparing-pytorch-models-for-metis-platform)
  - [Getting Started: Fruits-360 Classification Example](#getting-started-fruits-360-classification-example)
    - [Dataset and Model Setup](#dataset-and-model-setup)
      - [Option 1: Train the Model Yourself](#option-1-train-the-model-yourself)
      - [Option 2: Use a Pre-trained Model](#option-2-use-a-pre-trained-model)
  - [Next Steps](#next-steps)

This tutorial is the starting point of the PyTorch series, teaching you how to prepare your PyTorch models for deployment on the Metis platform. Through this series, you'll learn how to harness the full potential of the Voyager SDK for model and pipeline deployment.

## Getting Started: Fruits-360 Classification Example

To illustrate the concepts in this tutorial, we’ll use a practical example: training a ResNet34 model on the Fruits-360 dataset for fruit and vegetable classification.

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
