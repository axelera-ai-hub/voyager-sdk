![](../../../docs/images/Ax_Page_Banner_2500x168_01.png)
# Advanced model and pipeline deployment [Experimental]

- [Advanced model and pipeline deployment \[Experimental\]](#advanced-model-and-pipeline-deployment-experimental)
  - [What You'll Learn in the Tutorials](#what-youll-learn-in-the-tutorials)
  - [How to Use The Tutorials](#how-to-use-the-tutorials)
    - [Key Components to Accelerate Development](#key-components-to-accelerate-development)
      - [types.Model](#typesmodel)
      - [types.DataAdapter](#typesdataadapter)
      - [types.Evaluator](#typesevaluator)
      - [AxOperators](#axoperators)
    - [Reusing Existing Datasets and Evaluators](#reusing-existing-datasets-and-evaluators)
    - [Best Practices for Organizing Your Workspace](#best-practices-for-organizing-your-workspace)
  - [Tutorial-1: Getting Started with Model Deployment](#tutorial-1-getting-started-with-model-deployment)
    - [YAML File Structure](#yaml-file-structure)
      - [Explanation of Key Sections:](#explanation-of-key-sections)
    - [Deploying a Single Model](#deploying-a-single-model)
    - [PyTorch path](#pytorch-path)
    - [Takeaways](#takeaways)
    - [Appendix](#appendix)
      - [`extra_kwargs` in YAML models Section](#extra_kwargs-in-yaml-models-section)
  - [Tutorial-2: Inspecting the Model Pipeline and Measuring Model Performance](#tutorial-2-inspecting-the-model-pipeline-and-measuring-model-performance)
    - [AxOperator](#axoperator)
      - [AxOperator Methods](#axoperator-methods)
      - [Pipeline Overview](#pipeline-overview)
      - [Registering an AxOperator](#registering-an-axoperator)
      - [Example: *TopKDecoder* Implementation](#example-topkdecoder-implementation)
      - [Verification of Your Work and Measuring Model Performance](#verification-of-your-work-and-measuring-model-performance)
    - [PyTorch path](#pytorch-path-1)
    - [Takeaways](#takeaways-1)
  - [Tutorial-3: Working with Abstracted Inference Results](#tutorial-3-working-with-abstracted-inference-results)
    - [Understanding AxMeta and AxTaskMeta](#understanding-axmeta-and-axtaskmeta)
      - [Deep dive into AxTaskMeta](#deep-dive-into-axtaskmeta)
      - [Helper Functions for Organizing Model Outputs](#helper-functions-for-organizing-model-outputs)
    - [Connecting Everything with application.py and FrameResult](#connecting-everything-with-applicationpy-and-frameresult)
      - [FrameResult Internals](#frameresult-internals)
    - [Takeaways](#takeaways-2)
  - [Tutorial-4: Working with Custom Datasets and Accuracy Measurement](#tutorial-4-working-with-custom-datasets-and-accuracy-measurement)
    - [Building End-to-End GStreamer Pipelines](#building-end-to-end-gstreamer-pipelines)
      - [YAML Configuration Breakdown](#yaml-configuration-breakdown)
    - [Custom DataAdapter](#custom-dataadapter)
      - [Implementing a Custom DataAdapter](#implementing-a-custom-dataadapter)
      - [types.BaseEvalSample](#typesbaseevalsample)
    - [The other built-in DataAdapter](#the-other-built-in-dataadapter)
    - [Deploying with Existing Calibration Data](#deploying-with-existing-calibration-data)
    - [Takeaways](#takeaways-3)
    - [Appendix](#appendix-1)
  - [Tutorial-5: Building End-to-End GStreamer Pipelines](#tutorial-5-building-end-to-end-gstreamer-pipelines)
    - [Gstreamer as backend in application.py](#gstreamer-as-backend-in-applicationpy)
    - [Takeaways](#takeaways-4)
  - [Tutorial-6: Developing Your Own C/C++ Decoders](#tutorial-6-developing-your-own-cc-decoders)
    - [Compile as a Shared Library](#compile-as-a-shared-library)
    - [Add Properties](#add-properties)
    - [Takeaways](#takeaways-5)
    - [Appendix - Access Metadata in C++](#appendix---access-metadata-in-c)
  - [Tutorial-7: Defining Custom Abstracted Inference Metadata for Your Vision Task](#tutorial-7-defining-custom-abstracted-inference-metadata-for-your-vision-task)
    - [Takeaways](#takeaways-6)
  - [Tutorial-8: Evaluating Model Performance with Your Own Metrics](#tutorial-8-evaluating-model-performance-with-your-own-metrics)
    - [Understanding AxTaskMeta::to\_evaluation and types.BaseEvalSample](#understanding-axtaskmetato_evaluation-and-typesbaseevalsample)
    - [Exploring the Evaluator Implementation](#exploring-the-evaluator-implementation)
    - [Executing the Offline Evaluator](#executing-the-offline-evaluator)
    - [Introducing the Online Evaluation Approach](#introducing-the-online-evaluation-approach)
    - [Takeaways](#takeaways-7)
  - [Tutorial-9: Cascaded and parallel pipelines](#tutorial-9-cascaded-and-parallel-pipelines)
    - [Filtering and Selecting ROIs](#filtering-and-selecting-rois)
      - [Label Filtering](#label-filtering)
      - [Top-K ROI Selection](#top-k-roi-selection)
      - [Complete Pipeline Example with Label Filtering](#complete-pipeline-example-with-label-filtering)
    - [Accessing Submeta in Your Application](#accessing-submeta-in-your-application)
      - [C++ Implementation](#c-implementation)
      - [Python Implementation](#python-implementation)
    - [Running the Cascaded Pipeline](#running-the-cascaded-pipeline)
    - [Takeaways](#takeaways-8)
  - [Tutorial-10: Face detection and recognition pipeline](#tutorial-10-face-detection-and-recognition-pipeline)
    - [Creating an Embeddings File](#creating-an-embeddings-file)
    - [The Cascaded Pipeline](#the-cascaded-pipeline)
      - [Register yourself in the embeddings file](#register-yourself-in-the-embeddings-file)
    - [Adding a Tracker](#adding-a-tracker)
    - [Takeaways](#takeaways-9)
  - [Other References](#other-references)

The following tutorials explain experimental advanced model and pipeline deployment features of the Voyager SDK.


## What You'll Learn in the Tutorials

To get familiar with Axelera YAML and build your applications, we provide a tutorial series that guides you through the entire process of deploying your model on the Metis platform. You'll learn how to measure its performance on the AIPU and create a high-performance, end-to-end pipeline without sacrificing accuracy. By the end of the series, you’ll be equipped to build your own optimized pipelines with any model.

- If your model is a **PyTorch model**, start with [t0-prepare-your-torch.md](../../../ax_models/tutorials/torch/t0-prepare-your-torch.md).
- If your model is an **ONNX model** or can be converted to ONNX, begin with [t0-prepare-your-onnx.md](../../../ax_models/tutorials/onnx/t0-prepare-your-onnx.md). This tutorial also covers how to convert models from various frameworks to ONNX.

Both versions include a **tutorial-1** to help you deploy your model. From **tutorial-2** onward, all tutorials are consolidated in the `tutorials/general/` directory, with an introduction in [tutorials.md](../../../ax_models/tutorials/general/tutorials.md). While ONNX models are used as examples, the concepts and techniques are fully applicable to PyTorch models as well. The tutorial series will guide you through:

1. **Getting Started with Model Deployment**
   - *Key Concepts Covered:* `types.Model`, preprocess function
   - Learn how to wrap your model into a `types.Model` and implement the preprocess function using your existing transforms.
   - Deploy your model using representative images.
   - Find your generated artifacts in the `build` directory, ready for pipeline creation with AxRuntime.

2. **Inspecting the Model Pipeline and Measuring Model Performance**
   - *Key Concepts Covered:* `AxOperator`, Model FPS, Pipeline Preview
   - Learn to implement your postprocessing steps as an `AxOperator`.
   - Inspect your complete model pipeline (preprocess, model, postprocess) using a PyTorch implementation.
   - Inspect your Axelera-accelerated model pipeline, where preprocessing and postprocessing run on the CPU with PyTorch, and the model runs on the AIPU.
   - Measure the latency and throughput of your model inference on the AIPU.

3. **Working with Abstracted Inference Results**
   - *Key Concepts Covered:* `AxTaskMeta`, `application.py`, `FrameResult`
   - Integrate your inference results into `AxTaskMeta`.
   - Utilize built-in CV and GL components to visualize the results.
   - Start building your own `application.py` for proof-of-concept development.
   - Learn how to use `FrameResult` to access the results from your pipeline.

4. **Working with Custom Datasets and Accuracy Measurement**
   - *Key Concepts Covered:* `types.DataAdapter`, Applicable Accuracy
   - Learn how to use existing `DataAdapter` classes for supported label formats.
   - Learn how to create a custom `types.DataAdapter` from your existing datasets.
   - Measure your model's applicable accuracy using an in-house evaluator.

5. **Building End-to-End GStreamer Pipelines**
   - *Key Concepts Covered:* GStreamer, `AxOperator`
   - Build high-performance end-to-end GStreamer model pipelines with a low-code approach.
   - Use existing `AxOperator` classes to describe your preprocessing steps.
   - Measure the overall performance of your pipeline, including FPS and accuracy.
   - Adapting  the GStreamer pipeline as backend for your application.py

6. **Developing Your Own C/C++ Decoders**
   - *Key Concepts Covered:* C/C++ decoder, `AxMeta`, `AxOperator`, `inference.py`
   - Write a C/C++ decoder without needing any knowledge of GStreamer.
   - Structure the output of your decoder into the `AxMeta` format.
   - Integrate your C/C++ plugins into an `AxOperator` and build the end-to-end pipeline from `inference.py`.
   - Organize and build your decoder within your own development directory using `make operators`.

7. **Defining Custom Abstracted Inference Metadata for Your Vision Task**
   - *Key Concepts Covered:* `AxTaskMeta`
   - Create a new `AxTaskMeta` to accommodate new types of vision tasks or models with specific output data structures.
   - Transfer specific data from your C/C++ decoder to Python and register it within your custom `AxTaskMeta`.

8. **Evaluating Model Performance with Your Own Metrics**
   - *Key Concepts Covered:* `types.Evaluator`, `types.EvalResult`
   - Wrap your existing evaluation logic into a `types.Evaluator` class.
   - Enhance your offline evaluator to work online, reducing memory consumption.
   - Measure your model's accuracy using your custom evaluator and encapsulate the results in a `types.EvalResult`.

9. **Building Cascade Pipelines for Real-World Applications**
   - *Key Concepts Covered:* Cascade Pipeline
   - Modify your decoder to enable your model to participate in a cascade pipeline.
   - Run a classification model on Regions of Interest (ROIs) generated by a detection model.
   - Access the combined results from the cascade pipeline within your `application.py` or `inplace_plugin.cc`.

*Stay tuned! The upcoming tutorials will be available soon:*

10. **Integrating Multi-Object Tracking into Your Pipelines**
    - *Key Concepts Covered:* YAML pipeline, Multiple Object Tracking (MOT)
    - Create a YAML pipeline that integrates MOT functionality.
    - Access and utilize the tracking results directly in your `application.py`.

11. **Preprocessing Images before Your Inference Pipeline**
    - *Key Concepts Covered:* YAML pipeline, Image Preprocessing, `PreprocessOperator`
    - Learn how to create a YAML pipeline that includes image preprocessing steps before model inference.
    - Return the preprocessed images to your `application.py` for further processing or use.
    - Create your own `PreprocessOperator` for custom preprocessing logic.

12. **Visualizing Inference Data**
    - *Key Concepts Covered:* Visualizer, `AxTaskMeta`
    - Master the use of built-in visualization methods to display your `AxTaskMeta` data within `application.py`.
    - Develop a custom drawing method for your specific `AxTaskMeta` visualization needs.

13. **Making Your Metadata More Readable**
    - *Key Concepts Covered:* Object-view, `AxTaskMeta`
    - Create and implement a custom object-view for your `AxTaskMeta` to organize and present data in a specific way.


## How to Use The Tutorials

Choose the tutorial that aligns with your learning goals. This series is designed to help you understand the core components of the Voyager SDK step by step. Once you grasp these concepts, you can build your own pipelines with minimal effort. Note that not all steps in the tutorial are required to create a pipeline.

### Key Components to Accelerate Development

The Voyager SDK Model-Zoo is built on a modular architecture where models, datasets, and evaluators are treated as interchangeable plugins. A model can pair with a dataset and an evaluator to measure accuracy, but datasets and evaluators are designed to work independently as well. For instance, you can use a dataset plugin with a custom evaluator implementation. This modularity allows you to reuse existing components while seamlessly integrating your own plugins to achieve tailored accuracy measurements, ensuring consistent, apple-to-apple benchmarking.

This design not only promotes reusability but also provides the flexibility to support a wide variety of vision tasks and models, making it easier to adapt to diverse use cases. While datasets and evaluators focus on enabling accuracy measurement, the AxOperator serves as a key component for pipeline deployment, bridging the gap between model development and production workflows.

While there are Voyager SDK-specific concepts to learn, mastering its built-in components—**AxOperators**, **types.Model**, **types.Evaluators**, and **types.DataAdapter**—can significantly accelerate your development process and drive the creation of innovative solutions.

#### types.Model
- To deploy a model on the Metis, you must wrap your model as a `types.Model`.
- If you are using an **ONNX model** and describing your preprocess with AxOperators in the YAML pipeline section, you can simply declare:
  ```yaml
  class: AxONNXModel
  class_path: $AXELERA_FRAMEWORK/ax_models/base_onnx.py
  ```
in the YAML `models` section without any additional coding.
- For supported training frameworks, you can also declare the corresponding `class` and `class_path` in the YAML `models` section. For example:
  - A **YOLOv5 model** trained with Ultralytics can use:

  ```yaml
  class: AxYolo
  class_path: $AXELERA_FRAMEWORK/ax_models/yolo/ax_yolo.py
  ```

- Other supported frameworks include **torchvision** and **mmseg**.
- This approach eliminates the need for new implementation, allowing you to deploy models quickly through the configuration.

#### types.DataAdapter
- We focus on **data formats**, not specific datasets.
- For popular detection formats like **COCO**, **VOC**, and **YOLO**, you can reuse our built-in DataAdapter through YAML-based dataset descriptions.
- For classification tasks, we support the **ImageFolder** format (see [torchvision.datasets.ImageFolder](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)).
- For segmentation and pose estimation, we support the **COCO format**.
- After completing **Tutorial 4**, you’ll find it easy to wrap your existing dataset implementation with the `types.DataAdapter`.

#### types.Evaluator
- The evaluator takes **AxTaskMeta** as input.
- As long as your model supports standard vision tasks and you organize the results into the in-house AxTaskMeta format, you can evaluate applicable accuracy without implementing a new AxEvaluator.
- If needed, you can also wrap your existing evaluator with the `types.Evaluator`.


#### AxOperators
- We provide **mega-operators** built upon general operators, fusing them into highly optimized kernels that offload to host accelerators whenever possible.
- Preprocessing is typically a combination of a few operations, and you can almost always form your preprocess pipeline using existing AxOperators.
- Tutorials **2** and **5** will guide you on how to use AxOperators effectively.
- For further reference, check the YAML files under `pipeline-template`, which include common patterns from selected training frameworks.
- **General Rule**: If your preprocess can be composed using `torchvision`'s preprocess operators, you should be able to describe it entirely using existing AxOperators.

### Reusing Existing Datasets and Evaluators

The `types.DataAdapter` and `types.Evaluator` are designed to **wrap your existing implementations**, not force you to reimplement everything. This allows you to perform **apple-to-apple benchmarking** with minimal effort.

For further reference, check:
- [ax_datasets/mmseg.py](../../../ax_datasets/mmseg.py) for dataset wrapping examples.
- [ax_evaluators/mmlab.py](../../../ax_evaluators/mmlab.py) for evaluator wrapping examples.

By following the tutorials, you will gain a deeper understanding of these components, learn how to leverage these built-in tools effectively, and discover how to contribute to the ecosystem by building your own plugins to integrate with your framework.


### Best Practices for Organizing Your Workspace

The tutorial files are organized as follows:

- YAMLs, Python files, and the `tutorials.md` file are located in the [ax_models/tutorials/](../../../ax_models/tutorials/) directory.
- C++ files are located in the [customers/tutorials/](../../../customers/tutorials/) directory.

For your own projects, we recommend creating a dedicated folder within the `customers` directory, especially if you plan to develop custom C++ plugins. Our C/C++ build system automatically scans the `customers` folder for valid `CMakeLists.txt` files that use our helper functions. By placing your source code in a dedicated folder, you can easily reuse your work across SDK versions by simply copying your folder into the updated environment. This folder can include:

- C++ source files for custom plugins
- Custom Python scripts
- YAML configuration files

If you are a Python-only user, you can place your Python files and YAMLs anywhere. However, to use `axelera.app` to run your pipeline, you need to add the appropriate path to your script:

```python
framework = os.environ.get("AXELERA_FRAMEWORK", '.')
sys.path.append(framework)
```

The `AXELERA_FRAMEWORK` environment variable points to the root of the Voyager SDK, where the virtual environment is activated.


## Tutorial-1: Getting Started with Model Deployment

This tutorial introduces the fundamental structure of Axelera YAML files, which are essential for defining and deploying models on the Axelera platform. We will then walk through an example of deploying a single model. 

Before starting this tutorial ensure you have completed Tutorial 0.
- If your model is a **PyTorch model**, start with [t0-prepare-your-torch.md](../../../ax_models/tutorials/torch/t0-prepare-your-torch.md).
- If your model is an **ONNX model** or can be converted to ONNX, begin with [t0-prepare-your-onnx.md](../../../ax_models/tutorials/onnx/t0-prepare-your-onnx.md). This tutorial also covers how to convert models from various frameworks to ONNX.

### YAML File Structure
A valid Axelera YAML file must include the following six sections + an optional section:

``` yaml
axelera-model-format: 1.0.0
name: <model_name> 
description: <model_description>
pipeline: <pipeline_definition>
models: <model_definition>
datasets: <dataset_definition>

# Optional section
operators: <operator_definition>
```

The `axelera-model-format` field is used for backward compatibility. This version number will only be incremented if there are significant changes or improvements to the YAML specification. It ensures that older YAML files remain compatible with future versions of the SDK.  

#### Explanation of Key Sections:

- `name`: A unique identifier for your model deployment. It's crucial that the value of this field matches the name of your YAML file.
- `description`: A human-readable summary of the model, displayed when you use the make help command.
- `pipeline`: The `pipeline` section defines the end-to-end workflow of your model deployment. It describes how different models and datasets interact. Pipelines can be simple, involving a single model, or complex, orchestrating multiple models running in parallel or sequentially.
- `models`: The `models` section is a dictionary that defines the specific model(s) you want to deploy. Each entry in this dictionary represents a model and will contain further details about that model (which we will explore in later tutorials). These models are referenced by the `pipeline` section to define the workflow.
- `datasets`: The `datasets` section is a dictionary that defines the datasets used by your models. Each entry in this dictionary represents a dataset and will contain details about the data source and format. Datasets are associated with one or more models.

- `operators` (optional): The `operators` section is an optional dictionary used for registering custom operators. If your model utilizes custom operators not included in the standard Axelera library, you would define them in this section. These custom operators can then be referenced by the `pipeline`. We will delve into custom operators in more advanced tutorials.


### Deploying a Single Model

The YAML file for tutorial-1 can be found at [ax_models/tutorials/onnx/t1-simplest-onnx.yaml](../../../ax_models/tutorials/onnx/t1-simplest-onnx.yaml). The `pipeline` section is set to `null`, indicating that we are deploying a single model without a pipeline. Below is the `models` section from the YAML file:

```yaml
models:
  model1:
    class: CustomONNXModel
    class_path: simplest_onnx.py
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 100, 100]
    input_color_format: RGB
    weight_path: ~/.cache/axelera/weights/tutorials/resnet34_fruits360.onnx
    dataset: RepresentativeDataset
    extra_kwargs:
```

When deploying an ONNX model, you must implement a subclass of `types.ONNXModel` to load the ONNX model specified in the `weight_path` of the YAML file into the `self.onnx_model` attribute. In this example, the model is implemented in [simplest_onnx.py](../../../ax_models/tutorials/onnx/simplest_onnx.py), located alongside the YAML file, and the subclass name is `CustomONNXModel`. You can define your own class name as long as it subclasses `types.ONNXModel` and specify the `class_path` as either a relative or absolute path to the file.

To simplify implementation, we provide the `base_onnx.AxONNXModel` helper class, which includes the `init_model_deploy` function. This function automatically loads the ONNX model path from `types.ModelInfo` and assigns it to the `self.onnx_model` attribute. The `ModelInfo` class encapsulates the model information declared in the YAML file, including parameters like `task_category`, `input_tensor_layout`, `input_tensor_shape`, `input_color_format`, `weight_path` and `dataset`. We suggest studying extra_kwargs in conjunction with ModelInfo after completing this tutorial to avoid distractions for now. For more details, refer to the [extra_kwargs section](#extra_kwargs-in-yaml-models-section).


**Implementing Preprocessing:**
You can use [simplest_onnx.py](../../../ax_models/tutorials/onnx/simplest_onnx.py) as a foundation to implement your own `override_preprocess` function, which is responsible for converting input data into a `torch.Tensor` that matches the requirements of the ONNX/PyTorch model. This function should encapsulate all necessary preprocessing steps, such as resizing, normalization and format conversion, to ensure the input data aligns with the model's expected input shape and format. In most cases, your existing codebase may already include preprocessing logic implemented using `torchvision.transforms`. You can adapt this logic by integrating it into the `override_preprocess` function. The function takes a `PIL.Image.Image` or `np.ndarray` as input, following the conventions of `torchvision`, and applies the defined transformations to produce a properly formatted tensor as output. This ensures compatibility with the ONNX/PyTorch model while maintaining flexibility for customization.

```python
class CustomONNXModel(base_onnx.AxONNXModel):
    def override_preprocess(self, img: PIL.Image.Image | np.ndarray) -> torch.Tensor:
        return transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(img)
```

The `input_tensor_layout` and `input_tensor_shape` in the YAML file define the input shape of the ONNX model, so your preprocessing function must match these specifications. The `input_color_format` specifies the color format expected by the model, typically `RGB` (the default for `torchvision` and `PIL.Image`). If your model expects `GRAY` or `BGR`, ensure your preprocessing function includes a step to convert the input image accordingly.

Alternatively, you can specify the `repr_imgs_dataloader_color_format` in the dataset section to define the color format of the input images for the dataloader. This ensures the input images passed to the `override_preprocess` function are in the correct format. Let's see the dataset section in the yaml file:

```yaml
datasets:
  RepresentativeDataset:
    class: DataAdapter
    repr_imgs_dir_path: $AXELERA_FRAMEWORK/data/fruits-360-100x100/repr_imgs/
    repr_imgs_dataloader_color_format: RGB  # RGB or BGR
```

Each model declared in the `models` section must reference a dataset declared in the `datasets` section. In this example, the dataset is `RepresentativeDataset`, which uses the default `DataAdapter` class. This class points to a directory of representative images. The `repr_imgs_dir_path` specifies the absolute path to the image directory, and `repr_imgs_dataloader_color_format` defines the color format of the input images.

To use your own dataset, simply update the `repr_imgs_dir_path` to point to your dataset directory. We recommend using 100–400 images for the representative dataset. Deploy the model using the following command:


```bash
./deploy.py t1-simplest-onnx -v
```

By default, the deploy command uses 200 images. However, if your dataset contains fewer than 200 images, you may encounter the following error:


```
ERROR   : Cannot use 200 calibration images when dataset only contains 186 images. Please either:
ERROR   :   1. Reduce --num-cal-images to 186 or less
ERROR   :   2. Add more images to the calibration dataset
```

In this case, since your representative dataset contains only 186 images, you can specify the number of images to match your dataset size using the --num-cal-images parameter:

```bash
./deploy.py t1-simplest-onnx --num-cal-images=186 -v
```

The `-v` flag provides verbose output, while `-vv` offers even more detailed logs for debugging. If deployment fails, include the verbose output when reporting issues to the Axelera team.


**Deployment Results:**
The deployment results are stored in the `build/t1-simplest-onnx` directory. The Axelera portable model, `model.json`, can be found in `build/t1-simplest-onnx/model1/1`. Here:
- `model1` is the model name from the YAML file.
- `1` indicates the number of cores used for deployment.

On Metis, there are 4 cores available. You can use multiple cores in two ways:
1. **Multi-Process**:
   Deploy with `./deploy.py t1-simplest-onnx --aipu-cores=1` and infer with `./inference.py t1-simplest-onnx --aipu-cores=<number>`. The number of cores should not exceed 4. This is the default method and generally provides good performance.
2. **Multi-Core**:
   Deploy with `./deploy.py t1-simplest-onnx --aipu-cores=<number>` and infer with the same number of cores. The number of cores must not exceed 4 and should be an integer multiple of the deployment cores. This method may offer better performance for certain models.

For multi-core deployment, the portable model will be located in `build/t1-simplest-onnx/model1/<number>`. Alongside the `model.json` file, you will find the `manifest.json` file, which contains deployment details such as input/output tensor shapes, quantization and padding. The manifest will be loaded as TensorInfo. See its usages in [Python example](../../../examples/axruntime/python/axruntime_quickstart.py) and [C++ example](../../../examples/axruntime/axruntime_example.cpp).


**Notes on Multiple Models Deployment:**
- You can declare multiple models in the `models` section, but each model name must be unique.
- Each model can reference the same or different datasets.
- You can declare multiple datasets in the `datasets` section, but each dataset name must be unique.
- To deploy all models in a YAML file, use:

  ./deploy.py <yaml_file_name>

- To deploy a specific model, use:

  ./deploy.py <yaml_file_name> --model=<model_name>


### PyTorch path

The YAML file for PyTorch models is located at [ax_models/tutorials/torch/t1-simplest-pytorch.yaml](../../../ax_models/tutorials/torch/t1-simplest-pytorch.yaml). The main differences are:

```yaml
models:
  model1:
    class: CustomAxPytorchModelWithPreprocess
    class_path: simplest_torch.py
```

Additionally, the `weight_path` points to a `.pt` file instead of an ONNX file. We provide the `base_torch.TorchModel` helper class to simplify implementation. However, you must implement the `init_model_deploy` function to initialize your PyTorch model, load the weights, and assign it to the `self.torch_model` attribute. The `override_preprocess` function is implemented in the same way as for ONNX models.

### Takeaways

- To deploy an ONNX model, the simplest way is:
  1. Copy `simplest_onnx.py`
  2. Implement the `override_preprocess` function using your existing preprocessing code
  3. Copy `t1-simplest-onnx.yaml` and rename it to your own yaml file name. Rename `name` to align with your yaml file name.
  4. Declare `input_tensor_layout`, `input_tensor_shape`, `input_color_format` and `weight_path` in the `models` section.
  5. Declare `repr_imgs_dir_path` and `repr_imgs_dataloader_color_format` in the `datasets` section.
  6. Deploy the model using `./deploy.py <yaml_file_name>`.
  7. Check the deployment results in `build/<yaml_file_name>`.

- For PyTorch models, follow a similar process but use the PyTorch-specific YAML file and helper class.


### Appendix

This tutorial assumes that the model you want to deploy is either not supported by the Axelera model-zoo or you want to learn the Voyager SDK from scratch. If you are using a model-zoo model with your own weights, the best approach is to copy the YAML file and replace the weights. For more details, refer to the [custom weights tutorial](../../../docs/tutorials/custom-weights.md) and the examples in [yolov8n-weapons-and-knives.yaml](../../../ax_models/tutorials/yolo/yolov8n-weapons-and-knives.yaml) and [yolov8n-license-plate.yaml](../../../ax_models/tutorials/yolo/yolov8n-license-plate.yaml).


#### `extra_kwargs` in YAML models Section

The `extra_kwargs` field allows you to pass additional arguments to the model, making it useful for providing custom parameters to initialize your model or for configurations of runtime and compilation. In this example, the field is left empty. For more details on how to use this field for custom compilation, refer to the [Compiler Configuration Parameters](../../../docs/reference/compiler/compiler-configs.md) documentation. 

To see how `extra_kwargs` can be used for model initialization, refer to the [/ax_models/model_cards/timm/resnet10t-imagenet.yaml](../../../ax_models/zoo/timm/resnet10t-imagenet.yaml) file, which includes the following configuration:

```yaml
extra_kwargs:
  timm_model_args:
    name: resnet10t.c3_in1k
```

In the `AxTimmModel` class (located in [ax_models/torch/ax_timm.py](../../../ax_models/torch/ax_timm.py)), this field is accessed as follows:

```python
model_name = YAML.attribute(timm_model_args, 'name')
self.torch_model = timm.create_model(model_name, pretrained=True)
```

This approach allows you to flexibly select and configure any timm model directly through the YAML file, without requiring additional coding. By simply modifying the YAML configuration, you can easily integrate variant models, such as those listed in [Hugging Face's TIMM ResNet10t model comparison](https://huggingface.co/timm/resnet10t.c3_in1k#model-comparison), into the existing `AxTimmModel` implementation. For example, you could specify `name: resnet50.c1_in1k` in the YAML file to use a different model variant.

The only consideration is whether the preprocessing transform for the new model differs from the example model. If it does, you may need to implement the `override_preprocess` function in your model to handle the specific preprocessing requirements.


---
## Tutorial-2: Inspecting the Model Pipeline and Measuring Model Performance

In [Tutorial-1: Getting Started with Model Deployment](#tutorial-1-getting-started-with-model-deployment), you learned that an Axelera YAML file contains two primary sections—`models` and `datasets`. Now, let's move on to how to define the `pipeline` section. Unlike the other sections—which can contain multiple models and datasets—the `pipeline` section can only hold a single pipeline composed of a sequence of tasks.

A "Task" in this context is a combination of:
1. A model (already defined in the `models` section)
2. An input operator, used to process incoming data
3. (Optional) Pre-processing operators (if `override_preprocess` is not implemented in the model)
4. Post-processing operators

Below is an example snippet of the `pipeline` section from the YAML file located at [ax_models/tutorials/general/t2-learn-axoperator.yaml](../../../ax_models/tutorials/general/t2-learn-axoperator.yaml)

```yaml
pipeline:
  - classifications:            # Task name
      model_name: model1   # References model defined in models section
      input:
          type: image
          color_format: RGB # Must match repr_imgs_dataloader_color_format
      postprocess:
        - my_topk_decoder: # Post-processing operator
            k: 3           # A parameter for topk decoder to control how many top predictions to return
            softmax: true  # A parameter for topk decoder to control whether to apply softmax to the predictions
```

**Key components:**

1. Task Definition: each task in the pipeline is defined with a descriptive name (e.g., classifications)
2. Model Reference:
   - model_name links to a model defined in the models section
   - This tells the pipeline which model to use for inference
3. A task always owns an input operator and a (possibly empty) list of operators for preprocess and postprocess steps:
   - In this example, there is no dedicated preprocess operator, because the model's `override_preprocess` handles that internally.
4. Input Operator:
   - type: image specifies that this task processes images.
   - `color_format` must match the `repr_imgs_dataloader_color_format` from your dataset configuration. Keep in mind that this color format setting applies to the pipeline, not necessarily the model. Any color conversions for the model’s input format happen in the model's or the pipeline's preprocessing.
5. Postprocess and AxOperator:
   - The `postprocess` section lists operators (in order) that run after model inference.
   - In this example we have `my_topk_decoder`, a custom AxOperator that:
     - Returns the top 3 predictions (`k: 3`)
     - Applies a softmax (`softmax: true`) to the model's output

### AxOperator

An **AxOperator** in the Voyager SDK is an interface that abstracts the pre- and post-processing logic for a model. Each operator supports three main pipeline types, or "paths," depending on the processing requirements:

1. **Python path (`--pipe=torch`)**:  
Built on PyTorch for flexible, Python-based processing. This path provides an end-to-end FP32 implementation and serves as the accuracy benchmark baseline.
2. **C++ path (`--pipe=gst`)**:  
Built on GStreamer for optimized streaming capabilities. This path can utilize one or more AIPU cores for high-performance processing.
3. **Mixed path (`--pipe=torch-aipu`)**:  
Combines PyTorch and AIPU processing to quickly verify how a deployed model performs on AIPU. Currently, this path supports only a single AIPU core. It is particularly useful for evaluating accuracy drop when offloading a model to AIPU and gaining a basic understanding of the model's performance on AIPU.

#### AxOperator Methods
`AxOperator` provides two key methods for building the above pipeline types:

- `exec_torch`:  
This method implements the operator in Python, typically using PyTorch or other Python frameworks. For example, in the case of resizing, PyTorch's resize method can be used, but OpenCV's resize is preferred for half-precision support (commonly used in TensorFlow workflows). This flexibility allows developers to choose the most suitable implementation for their needs.

- `build_gst`:  
This method constructs the GStreamer pipeline using either general GStreamer elements or Voyager's built-in elements. Built-in elements are primarily used for model inference, while general GStreamer elements handle I/O and other tasks.

#### Pipeline Overview

The following figure illustrates the three pipeline types. A pipeline is constructed as a sequence of `AxOperators`, with `Model Inference` being a special built-in `AxOperator` that is automatically included in the pipeline. 

![pipe_types](../../../ax_models/tutorials/general/images/pipe_types.png)

Below is a summary of how the inference operator behaves across the three pipeline types:

- `--pipe=torch`:  
The inference operator uses ONNXRuntime for ONNX models or PyTorch backends for PyTorch models. If a GPU or MPS is available, it will leverage these for inference.
- `--pipe=gst`:  
When offloading to AIPU, the inference operator quantizes and pads the input tensor, invokes the AIPU for inference, and then dequantizes and crops the output tensor.
- `--pipe=torch-aipu`:  
Processing is implemented in Python, while the GStreamer path is implemented in C++. Additionally, certain operations at the beginning or end of the model may be offloaded to the host for better performance. These operations are fused into pre-processing or post-processing operators to optimize the pipeline.


We provide many built-in operators to cover common use cases. For details, see [Tutorial-5](#tutorial-5-building-end-to-end-gstreamer-pipelines). In this section, you'll learn how to implement a custom operator—TopKDecoder—and use it for verification of what you have implemented till now.

#### Registering an AxOperator

To register your own operator, add an operators block in your YAML file (anywhere, not necessarily before pipeline):
```yaml
operators:
  my_topk_decoder:
    class: TopKDecoder
    class_path: tutorial_decoders.py
```
Here, `class: TopKDecoder` and `class_path: tutorial_decoders.py` inform the pipeline that whenever `my_topk_decoder` is referenced (as it is in the postprocess section), the `TopKDecoder` class implemented in the file [tutorial_decoders.py](../../../ax_models/tutorials/general/tutorial_decoders.py) should be used.

#### Example: *TopKDecoder* Implementation

Below is a sample Python implementation for `TopKDecoder`. In this example, it uses PyTorch to apply optional softmax followed by extracting the top K predictions:

```python
class TopKDecoder(AxOperator):
    k: int = 1
    largest: bool = True
    sorted: bool = True
    softmax: bool = False

    def _post_init(self):
        pass

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        raise NotImplementedError("This is a dummy implementation")

    def exec_torch(self, image, predict, axmeta):
        if self.softmax:
            predict = TF.softmax(predict, dim=1)
        top_scores, top_ids = torch.topk(
            predict, k=self.k, largest=self.largest, sorted=self.sorted
        )
        top_scores = top_scores.cpu().detach().numpy()[0]
        top_ids = top_ids.cpu().detach().numpy()[0]
        LOG.info(f"Output tensor shape: {predict.shape}")
        LOG.info(f"Top {self.k} results: classified as {top_ids} with score {top_scores}")

        return image, predict, axmeta
```

1. Class Inheritance
 - `TopKDecoder(AxOperator)` indicates that this class is an AxOperator—allowing it to work within the Voyager SDK's pipeline framework.
2. Parameters
 - `k`, `largest`, `sorted` and `softmax` each have default values. You can omit these parameters in your YAML configuration if you want to use the defaults, or override them if needed.
 - If a parameter in your AxOperator has no default value, that parameter becomes mandatory in the YAML file.
 - Although parameter definitions resemble a dataclass, the AxOperator is not a dataclass. Each parameter must include a type hint, and you can use the `_post_init` method to perform any initialization or parameter validation after the operator is instantiated.
3. Execution Paths
 - `build_gst` is responsible for building the GStreamer pipeline for this operator. Details on how to implement this are covered in [Tutorial-6: Developing Your Own C/C++ Decoders](#tutorial-6-developing-your-own-cc-decoders).
  -  Even if you do not enable GStreamer now, you still need at least a dummy implementation (for example, by raising a `NotImplementedError`).
- `exec_torch` is responsible for running the operator’s logic using `pipe=torch` or `pipe=torch-aipu`.
  - The function signature is: `def exec_torch(self, image, predict, axmeta):`
    - `image` is the input image from the input operator, represented in Axelera's `types.Image` format.
    - `predict` is the input to this operator; as a postprocess operator directly following the model, it is the output of the model.
    - `axmeta` is the AxMeta object, which passes through the entire pipeline and serves as the container for the output of the model inference. This will be covered in the next tutorial.


#### Verification of Your Work and Measuring Model Performance

1. Deploy the model

```bash
./deploy.py t2-learn-axoperator
```

2. Inspect with torch Pipeline

```bash
./inference.py t2-learn-axoperator "data/fruits-360-100x100/Test/Plum 3/r_200_100.jpg" --pipe=torch --no-display
```

You will see console output, including the model’s output tensor shape and top 3 predictions. For example:

```
INFO    : Top 3 results: classified as [112  48  36] with score [1.0000000e+00 2.7341653e-09 1.0651212e-09]
```

You can then verify the top 3 predictions by checking the `fruits360.names` file. They are "Plum 3", "Eggplant long 1", and "Cherry Wax Yellow 1", where "Plum 3" is the correct prediction.

You can verify the model running on AIPU by

```bash
./inference.py t2-learn-axoperator "data/fruits-360-100x100/Test/Plum 3/r_200_100.jpg" --pipe=torch-aipu --no-display
```

Example console output:

```
INFO    : Top 3 results: classified as [112  36  48] with score [1.00000e+00 3.00744e-08 3.00744e-08]
```

The top-1 result matches the Torch pipeline. The other two predictions are reversed but still within the top 3. Additionally, performance metrics are displayed in the console output:

```
INFO    : Metis : 578.3fps
INFO    : Host : 54.1fps
```
- `Metis` shows the performance of the model running on AIPU.
- `Host` includes model inference and data transfer to/from AIPU. Single-image inference is not necessarily meaningful for performance metrics; using a video input can yield more realistic results:

```bash
./inference.py t2-learn-axoperator media/traffic1_480p.mp4 --pipe=torch-aipu --no-display --frames=2000
```

On a typical i5 machine, you might see:
```
INFO    : Metis : 579.5fps
INFO    : Host : 429.1fps
```

Providing too few frames can under-represent your model's true throughput.


This is how you can verify that your preprocess and postprocess are well integrated into the Voyager SDK. If a model fails to deploy and reports an error, we suggest verifying your Python path by:

```bash
./deploy.py t2-learn-axoperator --pipe=torch
./inference.py t2-learn-axoperator <image_path> --pipe=torch
```
Without `--no-display`, the image will be displayed in a window.


### PyTorch path

For pytorch path, you can use the same YAML file as the ONNX path. The only difference is the `class`, `class_path` and `weight_path` in the `models` section:

```yaml
models:
  model1:
    class: CustomAxPytorchModelWithPreprocess
    class_path: simplest_torch.py
    weight_path: ~/.cache/axelera/weights/tutorials/resnet34_fruits360.pth
```

Uncomment the 3 lines for PyTorch in [t2-learn-axoperator.yaml](../../../ax_models/tutorials/general/t2-learn-axoperator.yaml) and comment out the 3 lines pointing to the ONNX path. After that, you can redeploy and run inference with the model.

### Takeaways

Axelera YAML pipeline configuration allows you to:
- Define the flow of data through your model in a ML-oriented manner
- Specify input requirements
- Register and use your own post-processing operations for verifying if your `override_preprocess` and model within the SDK

With the pipeline section, you can run the end-to-end model pipeline using `--pipe=torch` or `--pipe=torch-aipu` to verify your model. The `torch-aipu` path provides a basic understanding of the model's performance on AIPU.

A minimal implementation of a subclass of `AxOperator` requires:
- Define the parameters, either by assigning a default value or requiring them to be explicitly declared
- Implementing the `exec_torch` function to parse the input from `predict` and using `LOG.info` to print the result.

## Tutorial-3: Working with Abstracted Inference Results

In [Tutorial-2: Inspecting the Model Pipeline and Measuring Model Performance](#tutorial-2-inspecting-the-model-pipeline-and-measuring-model-performance), you learned how to enable a pipeline in a low-code fashion. By providing a YAML file, end users can easily configure parameters under the pipeline section and run the inference pipeline with a single command (using `inference.py`). However, to make this pipeline more than a command-line tool—and to serve as an inference engine for user-facing applications—you need a structured way to share inference results. That’s where `AxTaskMeta` comes in.

`AxTaskMeta` is the base class which provides a container for model inference results. It abstracts the inference results into a unified format so that developers can retrieve and visualize them easily—without delving into pipeline internals. It also aligns well with in-house CV and GL components in the Voyager SDK for displaying or further processing the inference outcomes.

In this tutorial, we introduce [t3-learn-axtaskmeta.yaml](../../../ax_models/tutorials/general/t3-learn-axtaskmeta.yaml), which is very similar to [t2-learn-axoperator.yaml](../../../ax_models/tutorials/general/t2-learn-axoperator.yaml) except for two key differences:

1. We replace the `TopKDecoder` (postprocess operator) with a `TopKDecoderOutputMeta` implementation that populates results into `AxTaskMeta`.
2. We add a `labels_path` entry in the `datasets` section that points to the `fruits360.names` file. This file contains the dataset labels, used here to populate the `AxTaskMeta` with human-readable class names.


```python
class TopKDecoderOutputMeta(AxOperator):
    k: int = 1
    largest: bool = True
    sorted: bool = True
    softmax: bool = False

    def _post_init(self):
        pass

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph: graph.DependencyGraph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        raise NotImplementedError("This is a dummy implementation")

    def exec_torch(self, image, predict, axmeta):
        model_meta = meta.ClassificationMeta(
            labels=self.labels,
            num_classes=self.num_classes,
        )

        if self.softmax:
            predict = TF.softmax(predict, dim=1)
        top_scores, top_ids = torch.topk(
            predict, k=self.k, largest=self.largest, sorted=self.sorted
        )
        top_scores = top_scores.cpu().detach().numpy()[0]
        top_ids = top_ids.cpu().detach().numpy()[0]

        model_meta.add_result(top_ids, top_scores)
        for i in range(self.k):
            LOG.info(
                f"Top {i+1} result: classified as {self.labels(top_ids[i]).name} with score {top_scores[i]}"
            )

        axmeta.add_instance(self.task_name, model_meta)
        return image, predict, axmeta
```

**Key Points in TopKDecoderOutputMeta**
1. configure_model_and_context_info
  - Retrieves model details (like width, height, labels, etc.) from model_info.
  - Passes pipeline context data through. This ensures subsequent operators can access the same context info instance. You’ll see more on this in [Tutorial-5](#tutorial-5-building-end-to-end-gstreamer-pipelines).
2. ClassificationMeta
  - We instantiate `ClassificationMeta` to store classification results.
  - After parsing inference outputs (top IDs and scores), results are appended to `model_meta` via `add_result`.
3. axmeta.add_instance
  - The method `axmeta.add_instance(self.task_name, model_meta)` registers the classification metadata in the pipeline’s top-level container for metadata `axmeta`.
  - Because we’re naming the task “classifications” in the pipeline, you can later retrieve `axmeta["classifications"]` to see the classification results.


You can run the pipeline with `inference.py` and see the results in the console:

```bash
./inference.py t3-learn-axtaskmeta "data/fruits-360-100x100/Test/Plum 3/r_200_100.jpg" --pipe=torch-aipu -v
```

You will see the following output in the console:

```
DEBUG   :tutorial_decoders: Top 1 result: classified as plum_3 with score 17.662796020507812
DEBUG   :tutorial_decoders: Top 2 result: classified as chestnut_1 with score 8.615998268127441
DEBUG   :tutorial_decoders: Top 3 result: classified as blueberry_1 with score 3.446399211883545
```

This time, you see the label name instead of the class ID because the decoder maps the ID to the label name using `self.labels(top_ids[i]).name`. The label name should also appear in the display window.

Here, `self.labels` refers to the labels loaded from your `.names` file, which are represented as an Enum class. You can disable this automatic conversion by setting `disable_enumeration: True` in the `datasets` section, right after specifying your `labels_path`. When this option is enabled, `self.labels` will remain a list of strings instead of being converted to an Enum class.

The reason for converting labels to an Enum class is to provide ease of use when integrating at the application level. For example, this allows you to identify specific classes more intuitively, such as using `object.is_plum_3()` without requiring additional coding. This feature simplifies interaction with the model. We will provide more details about this functionality in a future tutorial.

Note: You may find it interesting that you didn’t explicitly run deploy.py, yet the log shows the deployment flow before running inference. This is because `inference.py` automatically performs incremental deployment for you. During the pipeline development stage, it is still recommended to run `deploy.py` explicitly to ensure there are no deployment issues. However, during the application development stage, you can take advantage of the incremental deployment performed automatically within the inference engine.


### Understanding AxMeta and AxTaskMeta

- The object `axmeta`, which is the only instance of class `AxMeta`, is the top-level container for inference results. Each step in the pipeline (preprocess, model inference, postprocess) can add or modify metadata within `axmeta`. When your pipeline has multiple tasks, they’re all stored under distinct keys (usually the task name).

- `AxTaskMeta` is an abstract base class for different computer vision tasks. We build domain-specific subclasses (e.g., `ClassificationMeta`, `ObjectDetectionMeta`, `SemanticSegmentationMeta`, etc.). Each subclass provides convenient properties and methods to handle relevant post-inference data, visualization and evaluation logic (like bounding boxes, keypoints, or class labels).

In this tutorial, you work with `ClassificationMeta`, a built-in class for classification tasks. If your pipeline uses other tasks, each would have corresponding metadata. For example, a pipeline with multiple classifiers referencing the same or different models would store results under separate keys in `axmeta`:
`axmeta["classifications1"], axmeta["classifications2"], …`

#### Deep dive into AxTaskMeta

`AxTaskMeta` is a foundational class in the Voyager SDK that provides a consistent and structured way to handle task-specific metadata for computer vision. By subclassing or leveraging existing subclasses (e.g., `ClassificationMeta`, `ObjectDetectionMeta`), developers can store, visualize, and evaluate inference outputs in a uniform manner across various pipelines and models. The key methods in `AxTaskMeta` include:

- `draw`: Draws abstracted task results (like bounding boxes, labels, segmentation masks, etc.) directly onto a `types.Image`. This simplifies visualization for integrated tasks and ensures efficiency by leveraging an OpenGL implementation to offload rendering to the eGPU hardware.
- `to_evaluation`: Converts the task results into an evaluation-plugin-friendly format (e.g., for COCO or custom evaluation). This standardizes how you pass data to different evaluators or metric calculators—ensuring you don’t need ad hoc scripts or transformations per model.
- `aggregate`: Combines multiple AxTaskMeta objects representing the same task type into a single instance. This is particularly useful when you’re running inference on ROIs and want to generate summarized results for evaluation.
- `decode`: Decodes task results passed from a raw byte stream, typically when running on the GStreamer (C++) path. This method ensures your Python-based metadata stays in sync with results produced by the lower-level engine.
- `objects`: Exposes an object-oriented view of the metadata, making it easier to access subsets of your inference results. For example, in a detection task, objects might return a list of bounding box entities—each containing class IDs, confidence scores, or keypoints. If you store AxTaskMeta in axmeta["classifications"], you can directly reference `frame_result.classifications` to retrieve an easy-to-consume array of classified objects.

Instead of creating a different `AxTaskMeta` subclass for each new model, the SDK encourages you to align model outputs with one of the built-in task-specific classes. This lets you leverage the built-in functions and components for visualization, evaluation and I/O. Common subclasses include:

- `ClassificationMeta`
- `ObjectDetectionMeta`
- `SemanticSegmentationMeta`
- `InstanceSegmentationMeta`
- `TrackerMeta`

For keypoint detection, the Voyager SDK provides several specialized classes to accommodate different output formats:

- `TopDownKeypointDetectionMeta` – uses a bounding box detector before keypoint determination.
- `BottomUpKeypointDetectionMeta` – processes keypoints directly in the full frame without explicit bounding boxes.
- `FaceLandmarkTopDownMeta` – specialized for five key face landmarks (top-down approach).
- `CocoBodyKeypointsMeta` – supports 17 COCO-format body keypoints.
- `FaceLandmarkLocalizationMeta` – handles five facial landmarks (bottom-up style).

By adhering to these built-in classes, you gain immediate access to standardized drawing, evaluation and metadata-handling methods—streamlining both application development and model iteration. In some cases, you may need to create your own AxTaskMeta subclass. Refer to [Tutorial-7: Defining Custom Abstracted Inference Metadata for Your Vision Task](#tutorial-7-defining-custom-abstracted-inference-metadata-for-your-vision-task) for more details.


#### Helper Functions for Organizing Model Outputs

When you have raw model outputs (e.g., bounding boxes, scores and classes) from tasks like object detection, instance segmentation, or keypoint detection, the Voyager SDK provides helper functions to ensure these outputs align with the expected format of each `AxTaskMeta` subclass.

For example, when working with object detection models:

```python
from axelera.app.meta import BBoxState

state = BBoxState(
   self.model_width,
   self.model_height,
   src_img_width,
   src_img_height,
   self.box_format,
   self.normalized_coord,
   self.scaled,
   self.max_nms_boxes,
   self.nms_iou_threshold,
   self.nms_class_agnostic,
   self.nms_top_k,
)
boxes, scores, classes = state.organize_bboxes(boxes, scores, classes)

model_meta = ObjectDetectionMeta.create_immutable_meta(
   boxes=boxes,
   scores=scores,
   class_ids=classes,
   labels=self.labels,
)

axmeta.add_instance(self.task_name, model_meta)
```

Here’s what’s happening:

1. You provide model-specific details (model width/height, source image width/height, whether coordinates are normalized, etc.) to initialize the `BBoxState`.
2. The raw predictions (boxes, scores, classes) from your model then get passed to `organize_bboxes`. This function standardizes the bounding-box format and applies NMS (non-maximum suppression) if configured.
3. The resulting data is then used to create an immutable `ObjectDetectionMeta` instance, which is finally registered in the `axmeta` dictionary via `axmeta.add_instance(self.task_name, model_meta)`.


Similar helper functions exist for keypoint detection (`organize_bboxes_and_kpts`) and instance segmentation (`organize_bboxes_and_instance_seg`). They handle specialized data wrangling to ensure you end up with a properly formatted AxTaskMeta subclass.

By combining:
- The built-in task-specific subclasses of `AxTaskMeta` (e.g., `ObjectDetectionMeta`)
- Helper classes like `BBoxState` that prepare and normalize raw model outputs
- The convenience methods (`draw`, `to_evaluation`, `aggregate`, `decode`, `objects`)

…you can quickly integrate your model outputs into a production-ready pipeline. This approach avoids duplicating logic for NMS, coordinate normalization, or bounding-box interpretation. It also ensures you get standardized drawing, evaluation and data access with minimal effort—so you can focus on model performance and application-specific functionality.


### Connecting Everything with application.py and FrameResult

`application.py` is an entry-level Python script demonstrating how to build your own custom application on top of Voyager pipelines. It accomplishes the following:

1. Creates an `InferenceConfig`
 - Specifies the pipeline (YAML file) via `network`.
 - Defines input media sources (videos, images, RTSP streams, etc.).
 - Chooses the pipeline type (torch, torch-aipu, or gst).
2. Initializes an `InferenceStream`
 - The `create_inference_stream` function creates an InferenceStream object, iterating over incoming frames and generating one `FrameResult` per frame.
3. Retrieves Results from `FrameResult`
 - `FrameResult` has an `image`, a `meta` attribute (the `AxMeta`), the `stream_id` (to identify which input stream has been returned), and timestamps.
 - Because each named task’s results are stored in meta under matching keys, you can access them either by `frame_result.meta["classifications"]` or—thanks to a custom `__getattr__`—using `frame_result.classifications`.
 - Optionally call `window.show(frame_result.image, frame_result.meta, frame_result.stream_id)` to visualize results in a window.

Example outline:
```python
stream = create_inference_stream(
    InferenceConfig(
        network="t3-learn-axtaskmeta",
        sources=["~/.cache/axelera/media/traffic1_480p.mp4"],
        pipe_type='torch-aipu',
    )
)


for frame_result in stream:
    window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
    print(frame_result.classifications.class_id)
stream.close()
```

In this snippet:
- `frame_result.classifications` references the classification task named “classifications” from your YAML configuration.
- `frame_result.classifications.class_id` gives you the predicted class for each object.

The for loop is where you can implement your business logic. For instance, you might log `frame_result.classifications.class_id` to a database or use it in other downstream processes.

Additionally, remember that looping over the stream with your business logic is synchronous: if that logic is time-consuming, it could slow the entire pipeline. To address this, you should measure the end-to-end FPS achievable with your business logic. Based on this measurement, you can configure the `frame_rate=<measured_fps>` within InferenceConfig. This setting allows you to specify a target frame rate, which will then be applied to the incoming stream. The system will automatically adjust the stream's frame rate to match your target by either dropping or duplicating frames as needed. Keep in mind that this frame rate adjustment is applied globally to all streams. Therefore, if you are working with multiple streams, they will all be adjusted to the same target frame rate. Per-stream frame rate control is a feature we plan to support in a future update.

#### FrameResult Internals

`FrameResult` is a dataclass that holds key information about each inference result:

- image – The frame’s input image (`types.Image`).
- meta – An AxMeta object containing the model outputs for each named task.
- src_timestamp, sink_timestamp, render_timestamp – Various timestamps indicating when the frame was received from the source pipeline, processed by the engine and rendered.
- stream_id – The ID of the stream (based on the index in the sources list if you have multiple inputs).

Thanks to the `__getattr__` method defined in `FrameResult`, you can conveniently access any stored `AxTaskMeta` by simply calling `frame_result.<task_name>`. Internally, this is equivalent to `frame_result.meta["<task_name>"]`, but returns the `objects` attribute directly, giving you a more object-oriented way to interact with inference results.

Here’s the relevant portion of FrameResult:

```python
@dataclasses.dataclass
class FrameResult:
    image: Optional[types.Image] = None
    tensor: Optional[np.ndarray] = None
    meta: Optional[AxMeta] = None
    stream_id: int = 0
    src_timestamp: int = 0
    sink_timestamp: int = 0
    render_timestamp: int = 0

    def __getattr__(self, attr):
        try:
            return self.meta[attr].objects
        except KeyError:
            raise AttributeError(f"'FrameResult' object has no attribute '{attr}'")
```


### Takeaways

`AxTaskMeta` provides a consistent schema for storing and referencing inference results according to the task type rather than the model architecture. It allows you to return structured data (“classifier,” “detector,” etc.) to downstream applications. By defining your custom operator classes to populate an appropriate subclass of `AxTaskMeta`, you can:

- Leverage built-in visualization (via `draw`) for quick debugging and real-time display.
- Integrate with the SDK’s built-in evaluation plugins (via `to_evaluation` and `aggregate`).
- Seamlessly share or decode results for more complex pipelines (via `decode`).
- Provide easy-to-access results integrated into `FrameResult` (via `objects` and `__getattr__`).

Collectively, these features help ensure that your model outputs—no matter the architecture or domain—are packaged in a reusable, consistent manner for both debugging and production-level applications.

To focus on application development or provide the AI inference engine as a backend for end users to build value on top of the prebuilt AI inference engine, `application.py` demonstrates how easily business logic can be integrated. With minimal configuration, users can utilize a YAML pipeline as the AI inference engine, retrieve pipeline results via `FrameResult` and `AxTaskMeta`. For handling complex real-world applications, we also provide examples of running business logic asynchronously and using a worker pool for business logic execution, which are valuable for end users to explore.


## Tutorial-4: Working with Custom Datasets and Accuracy Measurement

Before deploying a model to production, it can be crucial to evaluate its accuracy on real-world datasets. This ensures that the model performs as expected in practical scenarios. The Voyager SDK provides tools to measure accuracy using an in-house evaluator, and `types.DataAdapter` plays a key role in loading and preparing dataset plugins for evaluation.

We will use two YAML configuration files to demonstrate the process:

- [t4.1-measurement-with-meta.yaml](../../../ax_models/tutorials/general/t4.1-measurement-with-meta.yaml): Uses a built-in `TorchvisionDataAdapter` to load the Fruits360 dataset and measure accuracy.
- [t4.2-dataadapter.yaml](../../../ax_models/tutorials/general/t4.2-dataadapter.yaml): Demonstrates how to build your own `DataAdapter` to load the same dataset and measure accuracy.
- [t4.3-deploy-with-cal-data.yaml](../../../ax_models/tutorials/general/t4.3-deploy-with-cal-data.yaml): An example demonstrating how to deploy a model with calibration data.

### Building End-to-End GStreamer Pipelines

To measure the accuracy of your model using the Fruits360 dataset, run the following commands:
```bash
# ./deploy.py t4.1-measurement-with-meta
# ./inference.py t4.1-measurement-with-meta dataset --pipe=torch-aipu --no-display
```

This process may take some time. During execution, you will see progress updates in the console:

![Progress of the measurement](../../../ax_models/tutorials/general/images/measurement.png)

Once the measurement is complete, you will see a summary like this:

```
INFO    : Model:           model1
INFO    : Dataset:         Fruits360
INFO    : Date:            2025-01-06 14:45:09.485552
INFO    : Inference Time:  109543.46ms
INFO    : Evaluation Time: 3143.69ms
INFO    : Evaluation Metrics:
INFO    : ===================================
INFO    : | accuracy-top-1_average | 91.57% |
INFO    : | accuracy-top-5_average | 98.88% |
INFO    : ===================================
INFO    : Key Metric (accuracy-top-1_average): 91.57%
INFO    : Metis : 579.5fps
INFO    : Host : 429.2fps
INFO    : CPU % : 2.1%
INFO    : End-to-end : 241.8fps
```

This output provides key metrics such as:

- **Top-1 Accuracy**: The percentage of predictions where the top result matches the ground truth.
- **Top-5 Accuracy**: The percentage of predictions where the correct result is among the top 5 predictions.
- **Performance Metrics**: Frames per second (FPS) for different components (e.g., Metis, Host, End-to-end).

To compare the accuracy of the AIPU model with the FP32 ONNX/PyTorch model, rerun the measurement with the `--pipe=torch` option:

```bash
./inference.py t4.1-measurement-with-meta dataset --pipe=torch --no-display
```

The FP32 accuracy results will look like this:
```
INFO    : | accuracy-top-1_average | 91.82% |
INFO    : | accuracy-top-5_average | 98.94% |
```

The accuracy on AIPU is very close to the FP32 accuracy. This is because the Voyager SDK automatically calibrates the model into a proprietary mixed-precision format, optimizing performance while maintaining accuracy.

#### YAML Configuration Breakdown

Let’s take a closer look at [t4.1-measurement-with-meta.yaml](../../../ax_models/tutorials/general/t4.1-measurement-with-meta.yaml). The first difference is that we explicitly set the `num_classes` in the `models` section:

```yaml
    num_classes: 141
```

This is required for tasks like classification, object detection and segmentation, as the evaluator needs to know the number of classes to calculate accuracy.

Next, the dataset is loaded using the built-in `TorchvisionDataAdapter`:

```yaml
  Fruits360:
    class: TorchvisionDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/torchvision.py
    data_dir_name: fruits-360-100x100
    val_data: Test
    labels_path: ~/.cache/axelera/weights/tutorials/fruits360.names
    repr_imgs_dir_path: $AXELERA_FRAMEWORK/data/fruits-360-100x100/repr_imgs/
```

- data_dir_name: Specifies the dataset directory.
- val_data: Points to the validation data directory (e.g., Test).
- labels_path: Path to the label file.

The `TorchvisionDataAdapter` supports the widely used `ImageFolder` dataset format, where images are organized as follows:

```
data_root/Test/dog/xxx.png
data_root/Test/dog/xxy.png
data_root/Test/dog/[...]/xxz.png

data_root/Test/cat/123.png
data_root/Test/cat/nsdf3.png
data_root/Test/cat/[...]/asd932_.png
```

The `data_root` is defined as `root_dir` combined with the directory named `data_dir_name` in the `TorchvisionDataAdapter` section. By default, `root_dir` points to `$AXELERA_FRAMEWORK/data`, but it can be modified by setting `data_root` in the `InferenceConfig` or using `--data-root=</your/data/root>` with `deploy.py` or `inference.py`.

The `val_data` refers to the directory within `data_root` that contains the validation data, such as `Test` in this case. Alternatively, it can be set to `Train` or `Val` to use the corresponding data subsets, depending on your dataset structure. Note that `Test`, `Val` and `Train` are not strict required names but simply examples of subfolders representing different data subsets.

The advantage of the built-in DataAdapter with YAML configuration is that it eliminates the need to reorganize your dataset for deployment on different platforms. Our goal is to support an increasing number of dataset formats, including popular data labeling formats, to simplify usage. If you have a custom dataset format, refer to the next section to create your own DataAdapter.


### Custom DataAdapter

Let’s examine [t4.2-dataadapter.yaml](../../../ax_models/tutorials/general/t4.2-dataadapter.yaml). The key difference is that it points the dataset to `Custom-Fruits360`, which utilizes `CustomDataAdapter` instead of `Fruits360`. Now, let’s proceed to run the measurement using the `torch-aipu` backend:

```bash
./inference.py t4.2-dataadapter dataset --pipe=torch-aipu --no-display
```

The results will be similar to the Fruits360 dataset:
```
INFO    : | accuracy-top-1_average | 91.43% |
INFO    : | accuracy-top-5_average | 98.88% |
```

The slight difference is due to the calibration process, which is not always identical.

#### Implementing a Custom DataAdapter

The `CustomDataAdapter` is a user-defined implementation of a `DataAdapter`, designed to load and prepare datasets for evaluation. It is implemented in the file [tutorial_data_adapter.py](../../../ax_models/tutorials/general/tutorial_data_adapter.py). To create a custom DataAdapter, you need to implement four key methods. Below, we will walk through these methods step by step.

1. Initializing the `DataAdapter`

The `__init__` method is used to initialize the DataAdapter. It takes two arguments:

- `dataset_config`: The dataset configuration from the YAML file's datasets section.
- `model_info`: The model information from the YAML file's models section.

These configurations allow the DataAdapter to access the necessary dataset and model details. Typically, you store these arguments in instance variables (e.g., self.dataset_config and self.model_info) for later use.

2. Creating a Validation DataLoader

The `create_validation_data_loader` method is responsible for creating a DataLoader for the validation dataset. This DataLoader will load batches of data that are later passed to the `reformat_for_validation` method for preprocessing.
- The method takes the dataset's root directory (`root`) and the target split (`target_split`) as arguments.
  - The target_split is an optional argument passed via the CLI. For example:
    ```bash
    ./inference.py t4.2-dataadapter dataset:val --pipe=torch-aipu --no-display
    ```
    Here, val is the target_split. This allows you to use the CLI to specify which dataset split to use.
- Additional arguments (e.g., `val_data`) are passed via `kwargs` and must be defined in the YAML dataset configuration.
- The `val_data` argument specifies the directory containing the validation data.

Here’s an example implementation:
```python
def create_validation_data_loader(self, root, target_split, **kwargs):
    assert 'val_data' in kwargs, "val_data is required in the YAML dataset config"
    return torch.utils.data.DataLoader(
        build_dataset(root / kwargs['val_data'], transform=None),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
    )
```

- `build_dataset`: An existing function of the dataset implementation. The `transform` function is set to None because the pipeline applies transformations from `override_preprocess` from your types.Model implementation automatically.
- `batch_size=1`: Ensures that data is loaded one sample at a time.
- `shuffle=False`: Disables shuffling to maintain the order of the data, which is useful for debugging and development.
- `collate_fn=lambda x: x`: Ensures that the data is returned as-is without additional processing.
- `num_workers=0`: Sets the number of workers to 0, which means the data loading is done sequentially.


3. Reformatting Data for Validation

The `reformat_for_validation` method processes the batched data from the DataLoader and converts it into a format compatible with the Voyager SDK's measurement pipeline. Specifically, it returns a list of `types.FrameInput` objects. 

As the validation dataloader returns an image and a label, where the label represents the class ID of the image. Therefore, the input to `reformat_for_validation` is a list of tuples, `batched_data`, where each tuple contains an image and its corresponding label as the ground truth. Since you are implementing your dataset, you are responsible for understanding the structure of `batched_data`, to ensure the data can be accessed by the Voyager SDK measurement pipeline.

Here’s an example implementation:

```python
def reformat_for_validation(self, batched_data):
    return [
        types.FrameInput.from_image(
            img=img,
            color_format=types.ColorFormat.RGB,
            ground_truth=eval_interfaces.ClassificationGroundTruthSample(class_id=target),
            img_id='',
        )
        for img, target in batched_data
    ]
```

Explanation:
- `types.FrameInput.from_image`: A convenience function to create a FrameInput object.
- `img`: The image data (as a NumPy array or PIL image).
- `color_format`: Specifies the color format of the image (e.g., types.ColorFormat.RGB).
- `ground_truth`: A ground truth object (e.g., ClassificationGroundTruthSample) containing the class ID.
- `img_id`: An optional identifier for the image (set to an empty string in this example).



4. Bypassing built-in evaluator

The 4th method is the evaluator which expected to return a `BaseEvaluator` object. For example,

```python
def evaluator(self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False):
    from ax_evaluators.classification import ClassificationEvaluator
    return ClassificationEvaluator()
```

Depending on the vision task, use a built-in evaluator such as ClassificationEvaluator for classification tasks. You can also implement your own evaluator by subclassing `BaseEvaluator` and then reusing it here. For more details, refer to [Tutorial-8: Evaluating Model Performance with Your Own Metrics](#tutorial-8-evaluating-model-performance-with-your-own-metrics).


#### types.BaseEvalSample

`BaseEvalSample` is an abstract base class that represents a sample of data to be evaluated. It is a very simple class that contains a `data` attribute. The `data` attribute is a property that returns the data of the sample which can be any type of data. For classification tasks, it is the class id of the image. For object detection tasks, it is a list of bounding boxes, scores and class ids.

`BaseEvalSample` is used for both ground truth and predictions to build a pair for accuracy evaluation. The prediction side comes from AxTaskMeta’s `to_evaluation` method, which returns a `BaseEvalSample` object to the evaluator.


### The other built-in DataAdapter

For YOLO tasks like object detection, instance segmentation and keypoint detection, you can use `ObjDataAdapter`:

```yaml
  CocoFormatDataset:
    class: ObjDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py
    data_dir_name: your_dataset_dir_name
    cal_data: train.txt
    val_data: val.txt
    labels_path: $AXELERA_FRAMEWORK/path/to/your/labels.names
```

For object detection, it supports COCO, YOLO and VOC formats, for the keypoint detection and instance segmentation, it supports YOLO format.


### Deploying with Existing Calibration Data

Throughout the previous sections, we have been using representative images for model calibration. These images are consolidated into a single directory, and the Voyager SDK automatically processes them through the preprocessing pipeline to calibrate the model for quantization.

As you may recall from sections 4.1 and 4.2, we specified the representative images directory path in the dataset configuration:

```yaml
    repr_imgs_dir_path: $AXELERA_FRAMEWORK/data/fruits-360-100x100/repr_imgs/
```

> [!IMPORTANT]
> By default, `./deploy.py` uses 400 selected COCO images as representative images for model calibration. Since COCO images are captured from real-world scenarios, they generally provide good calibration accuracy. However, if your target application uses images that differ significantly from the COCO dataset (for example, specialized images from automated optical inspection systems with unique lighting and color conditions), you should use your own representative images for better model calibration. This can be done by specifying the `repr_imgs_dir_path` in the YAML dataset section, which will override the default representative image path.

A common scenario is when you already have a calibration or training dataset that was used during model training. In such cases, rather than preparing a separate set of representative images, you can leverage the existing dataset using the `cal_data` parameter in the YAML configuration file.

To implement this approach, refer to [t4.3-deploy-with-cal-data.yaml](../../../ax_models/tutorials/general/t4.3-deploy-with-cal-data.yaml), where we replace `repr_imgs_dir_path` with `cal_data`:

```yaml
  Custom-Fruits360:
    class: CustomDataAdapterWithCalData
    class_path: tutorial_data_adapter.py
    data_dir_name: fruits-360-100x100
    val_data: Test
    labels_path: ~/.cache/axelera/weights/tutorials/fruits360.names
    cal_data: Training
```

The configuration points to the `CustomDataAdapterWithCalData` class, which extends `CustomDataAdapter` and implements the necessary `create_calibration_data_loader` and `reformat_for_calibration` methods. In this implementation, `reformat_for_calibration` is identical to `reformat_for_validation`, while `create_calibration_data_loader` uses the `cal_data` parameter instead of `val_data` and enables data shuffling.

Alternatively, you can use the built-in `TorchvisionDataAdapter` with the `cal_data: Training` parameter, eliminating the need for a custom data adapter:

```yaml
  Fruits360:
    class: TorchvisionDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/torchvision.py
    data_dir_name: fruits-360-100x100
    val_data: Test
    labels_path: ~/.cache/axelera/weights/tutorials/fruits360.names
    cal_data: Training
```

Examining the model section:

```yaml
  model1:
    class: CustomONNXModelWithoutResize
    class_path: ../onnx/simplest_onnx.py
```

We use `CustomONNXModelWithoutResize` here to demonstrate that when using training data already sized at 100x100 pixels, the preprocessing pipeline works effectively even without the Resize transform:

```python
class CustomONNXModelWithoutResize(base_onnx.AxONNXModel):
    def override_preprocess(self, img: PIL.Image.Image | np.ndarray) -> torch.Tensor:
        return transforms.Compose([
            # transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(img)
```

If your input dimensions exceed the expected size, you will see a warning:
```
WARNING : The test set shape of the calibration dataloader does not match model input shape: [1, 3, 480, 640] vs [1, 3, 100, 100]. This can lead to incorrect quantization!
```

In such cases, you should review your preprocessing pipeline immediately, as continuing with calibration could adversely affect model accuracy. Conversely, if the input dimensions are smaller than expected, you will encounter an error due to insufficient data for processing.

To deploy using your calibration data, execute the following command:

```bash
./deploy.py t4.3-deploy-with-cal-data --no-default-representative-images
```

This command disables the default representative images and utilizes the `cal_data` specified in your YAML configuration file. Ensure that the `data/fruits-360-100x100/Training` directory contains the training images; otherwise, you will encounter an error. If needed, you can download a subset of training images using:

```bash
[ ! -d "$AXELERA_FRAMEWORK/data/fruits-360-100x100/Training" ] && wget --show-progress -P $AXELERA_FRAMEWORK/data/ https://media.axelera.ai/artifacts/tutorials/fruits-360-100x100-trainsubset.zip && unzip -q $AXELERA_FRAMEWORK/data/fruits-360-100x100-trainsubset.zip -d $AXELERA_FRAMEWORK/data/ && rm $AXELERA_FRAMEWORK/data/fruits-360-100x100-trainsubset.zip || echo "Training directory exists, skipping download"
```

This training subset contains 282 images. The default calibration process randomly selects 200 images. If you attempt to use more images than are available:

```bash
 ./deploy.py t4.3-deploy-with-cal-data --no-default-representative-images --num-cal-images=400
 ```

You will receive an error:

```
ERROR   : Cannot use 400 calibration images when dataset only contains 282 images. Please either:
ERROR   :   1. Reduce --num-cal-images to 282 or less
ERROR   :   2. Add more images to the calibration dataset
```

To resolve this, either reduce the `--num-cal-images` parameter to match or fall below your available dataset size, or add more images to your calibration dataset. The default value of 200 is recommended for typical calibration scenarios, but you can experiment with values between 100 and 400 images. Research indicates that using more than 400 images typically produces diminishing returns for calibration quality.


### Takeaways

- Voyager SDK provides a built-in DataAdapter for most of the common dataset formats. By simply invoking the corresponding DataAdapter through the YAML datasets section, you can easily use `./inference.py <the_model> dataset --pipe=torch-aipu --no-display` to measure the model accuracy and compare with the FP32 model by setting the `--pipe=torch`.
- You can create your own DataAdapter to handle custom dataset formats. This involves implementing the methods illustrated above and reusing your existing dataset class.
- By default, `./deploy.py` uses COCO representative images for model calibration
- You can override this default by either:
  - Specifying `repr_imgs_dir_path` in the YAML dataset section to use your own representative images
  - Specifying `cal_data` in the YAML dataset section to use your existing calibration data

### Appendix

Accuracy measurement is an optional step in model deployment and can be skipped if your priority is quick POC over accuracy. In the Voyager SDK’s Model-Zoo, accuracy measurement for all models is already provided. You can simply run the following command:

```bash
./inference.py <the_model> dataset --pipe=torch/torch-aipu --no-display
```
This measures model accuracy efficiently. Accuracy measurement is particularly valuable for production deployment, offering a more comprehensive evaluation compared to other SDKs that only measure the core model while handling pre/post-processing in Python, which is not a true end-to-end C/C++ pipeline evaluation.

The knowledge gained here can be seamlessly applied to GStreamer pipeline measurement, which will be discussed in the next tutorial. During measurement, the DataAdapter and evaluator are directly reused, eliminating the need for engineers to rebuild or customize accuracy measurement methods for different models or datasets. Instead, this functionality is implemented as a plugin, enabling meaningful accuracy comparisons between FP32, torch-aipu and gst pipelines. You can also inspect operations across pipelines to ensure that no critical accuracy is lost during the complex deployment process.

The SDK further supports accuracy measurement for cascade pipelines. For instance, you can verify classifier accuracy within an object detection + classification pipeline. As we understand so far, this is the only SDK that enables such cascade pipeline accuracy measurement and allows direct comparisons between different pipelines.

For embedded or application engineers, the Voyager Model-Zoo is a reliable, production-ready solution that has been thoroughly verified.

## Tutorial-5: Building End-to-End GStreamer Pipelines

We have now verified a qualified model running on the AIPU platform. The next step is to deploy the model as a high-performance, end-to-end C/C++ pipeline, making it production-ready.

First, let’s examine the file [t5-gst-pipeline.yaml](../../../ax_models/tutorials/general/t5-gst-pipeline.yaml), specifically the model section. For this pipeline, we don’t need a custom `CustomONNXModel` because the base model `AxONNXModel` is sufficient. The `override_preprocess` method is not required here.

```yaml
  model1:
    class: AxONNXModel
    class_path: $AXELERA_FRAMEWORK/ax_models/base_onnx.py
```

Similarly, but with a slight difference, for the PyTorch model, we still need to use a custom version to load the model. In this case, we use `CustomAxPytorchModel` instead of `CustomAxPytorchModelWithPreprocess`.

```yaml
  model1:
    class: CustomAxPytorchModel
    class_path: ../torch/simplest_torch.py
```

The preprocessing transform is implemented directly in the YAML pipeline using built-in AxOperators. This eliminates the need for custom preprocessing code. Below is an example of the preprocessing transform we are using now:

```python
transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```
In the YAML pipeline, we can use the built-in AxOperators to replicate the functionality of custom transforms (override_preprocess). Refer to the [YAML Operator](../../../docs/reference/pipeline/yaml-operators.md) to see the list of supported operators in YAML.

```yaml
preprocess:
  - resize:
      width: 100
      height: 100
  - torch-totensor:
  - normalize:
      mean: 0.485, 0.456, 0.406
      std: 0.229, 0.224, 0.225
```

For postprocessing, we also use the built-in decoders to decode the results without writing any code.

```yaml
postprocess:
  - topk:
      k: 1
```

Now we have a complete YAML pipeline. Isn't it easy? Let's run the pipeline:

```bash
./inference.py t5-gst-pipeline dataset --no-display
```

The `--pipe` option defaults to `gst`, so there’s no need to specify it explicitly. Using a dataset with 23,619 samples, the pipeline runs at over 1300 FPS and provides top-1 and top-5 accuracy as follows:


```
INFO    : | accuracy-top-1_average | 91.32% |
INFO    : | accuracy-top-5_average | 98.87% |
```

You can run again with the `--pipe=torch-aipu` to compare the accuracy with the previous measurement. For my case, it is 91.51% and 98.85%. This shows that the accuracy of the end-to-end C++ pipeline is very close to that of the AIPU model with Python-based preprocessing and postprocessing. To measure the FPS of the end-to-end pipeline using a video stream, run:

```bash
./inference.py t5-gst-pipeline media/traffic1_4480.mp4 --no-display -v
```

You will see the fps of the e2e pipeline:

```
INFO    : Metis Core 0 : 546.0fps
INFO    : Metis Core 1 : 545.9fps
INFO    : Metis Core 2 : 546.0fps
INFO    : Metis Core 3 : 544.9fps
INFO    : Metis : 2183.0fps
INFO    : Memcpy-in : 18.1us
INFO    : Memcpy-out : 9.7us
INFO    : Patch Parameters : 0.3us
INFO    : Kernel : 1968.8us
INFO    : Host : 1998.2fps
INFO    : CPU % : 6.8%
INFO    : End-to-end : 1342.8fps
```

In this example, the GST pipeline uses 4 cores, each running at ~545 FPS, resulting in a total of ~2183 FPS for the core model on the AIPU. Including memory copy and patch parameter times, the host achieves ~1998 FPS. With preprocessing and postprocessing, the end-to-end pipeline achieves ~1342 FPS.

### Gstreamer as backend in application.py

To use the GST pipeline in application.py, you can simply set `pipe_type='gst'` from the InferenceConfig.

```python
stream = create_inference_stream(
    InferenceConfig(
        network="t5-gst-pipeline",
        sources=["~/.cache/axelera/media/traffic1_480p.mp4"],
        pipe_type='gst',
    )
)
```

Everything else remains the same, and you can run the pipeline with `python application.py`.

### Takeaways
- Using the built-in `AxOperators` allows you to build a high-performance end-to-end GStreamer model pipeline with minimal code.
- The `--pipe=gst` option enables easy performance measurement of the end-to-end pipeline, including FPS and accuracy (when using the dataset input option).
- The `override_preprocess` method is not required when building the end-to-end GST pipeline.
- By setting `pipe_type='gst'` in `application.py`, you can seamlessly use the GST pipeline as the backend.

To deploy a model to production, you can start directly from this tutorial without going through Tutorials 1–4. At this stage, you should have a solid understanding of the Axelera YAML configuration. You can deploy a model by describing the pipeline in YAML and using the following model configuration:


```yaml
  model1:
    class: AxONNXModel
    class_path: $AXELERA_FRAMEWORK/ax_models/base_onnx.py
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 100, 100] # remember to set the input shape!
    input_color_format: RGB
    weight_path: /path/to/your/model.onnx
    num_classes: 10 # remember to set the num_classes!
    dataset: RepresentativeDataset
```

With this setup, you can deploy the model and run inference.py with the GST pipeline if the model is already available in the Model-Zoo.


## Tutorial-6: Developing Your Own C/C++ Decoders

In the previous tutorial, we ran a GStreamer pipeline using the default tensor decoder for classification networks provided by the application framework. A tensor decoder is an operator which gets the raw output tensor data of the model as input and returns the relevant parts of the result in a structured way. If you introduce a new model then the need may arise to write your own decoder.

In Tutorial 2, we already implemented a **custom decoder** in Python. The goal of this tutorial is to do the decoding inside of the GStreamer pipeline. No knowledge of GStreamer is required. The C++ code you write will be compiled into a **shared library** file that is loaded by GStreamer plugins.

The description of the interfaces designed to exchange data between your custom code and GStreamer can be found in the [pipeline operator reference](../../../docs/reference/pipeline/pipeline-basics.md). At least the function `decode_to_meta` must be defined in the source code file that will be compiled into the shared library.

```cpp
extern "C" void
decode_to_meta(const AxTensorsInterface &tensors,
    const void *, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &,
    const AxDataInterface &, Ax::Logger &)
{
    auto &tensor = tensors[0];
    size_t total = tensor.total();
    auto *data = static_cast<const float *>(tensor.data);
    auto max_itr = std::max_element(data, data + total);
    auto max_score = *max_itr;
    auto max_class_id = std::distance(data, max_itr);
    std::cout << "Class ID " << max_class_id << std::endl;
}
```

The first parameter is the output of the model.  In our case, there is only one tensor, we can get its total number of elements and then loop over all of them to find the position of the largest value, which gives us the top class ID.

We would like to pass the result of the decoder to the Python code (e.g. for measurements) or to another GStreamer element. To achieve both of this, we have to **store the result** in the unordered meta map, which is the fifth parameter of `decode_to_meta`. The values inside the map are objects with the base class `AxMetaBase`. Classes to store the results already exist in the application framework for the most common network types. For those C++ classes, corresponding `AxTaskMeta` classes exist in the Python code. In our example, [AxMetaClassification](../../../operators/axstreamer/include/AxMetaClassification.hpp) is appropriate. We can use the helper function `insert_meta` to construct an instance of this class inside of the meta map.

```cpp
  ax_utils::insert_meta<AxMetaClassification>(map, "classifier", {}, 0, 1,
      std::vector<std::vector<float>>{ { max_score } },
      std::vector<std::vector<int32_t>>{ { max_class_id } },
      std::vector<std::vector<std::string>>{ { std::to_string(max_class_id) } });
```

The first parameter of `insert_meta` is the meta map. The second one is the key into the meta map and has to be equal to the task name in the YAML file. Parameters three to five are only relevant for cascaded pipelines. Everything that follows after the first 5 parameters is passed into the constructor of the class specified as the template argument, in this case `AxMetaClassification`. When the C++ code returns to the Python code after each frame, the contents of the C++ meta map will be translated to instances of the corresponding Python classes derived from `AxTaskMeta` and stored in `axmeta`, which can be regarded as the meta map of the Python code.

### Compile as a Shared Library

In order to run the decoder above, we have to **compile** the code into a shared library and implement the function `build_gst` in the Python decoder file. The source code of the C++ decoder for this part of the tutorial can be found in [customers/tutorials/src/decode_tu6a.cc](../../../customers/tutorials/src/decode_tu6a.cc). To compile this C++ file, first add it to [customers/tutorials/CMakeLists.txt](../../../customers/tutorials/CMakeLists.txt).

```cmake
add_customer_libraries(tutorials
    SOURCES
        src/decode_tu6a.cc
)
```

After that, run the command `make operators`. This command will compile all built-in GStreamer plugins located in the `operators` directory. Additionally, it will iterate through the `customers` folder to check for a valid CMakeLists.txt file that uses our helper functions. In this example, since the source file is named `decode_tu6a.cc`, the resulting library will be generated as `libdecode_tu6a.so`. If you encounter any errors during compilation, resolve them and rerun `make operators`. If you need to clean up all existing artifacts for a fresh build, you can use the `make clobber-libs` command. Now, let’s explore how to integrate this library into our pipeline builder.

We now implement the function `build_gst` in `class TopKDecoderWithMySimplifiedGstPlugin` of the Python decoder file [ax_models/tutorials/general/tutorial_decoders.py](../../../ax_models/tutorials/general/tutorial_decoders.py). Remember that we specified this path in the YAML file. The Python code below makes sure that the GStreamer element for decoding loads the shared library resulting from the compilation above. We just have to assign the filename to the lib property.

```python
def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
    gst.decode_muxer(lib='libdecode_tu6a.so', options='')
```

If we have compiled our C++ decoder correctly, we can now use it in the GStreamer pipeline by running

```bash
./inference.py t6a-gst-decoder dataset --no-display
```

### Add Properties

In our YAML files from previous tutorials, we specified **options** for the Python decoder, e.g. we assigned the number 3 for `k` so that the top 3 results are returned. In the following, we want to pass those options to our C++ decoder as well. As a first step, we adapt the function `build_gst` of `class TopKDecoderWithMyGstPlugin` in the Python decoder file [ax_models/tutorials/general/tutorial_decoders.py](../../../ax_models/tutorials/general/tutorial_decoders.py).

```python
def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
    gst.decode_muxer(
        lib='libdecode_tu6b.so',
        options=f'meta_key:{str(self.task_name)};'
        f'top_k:{self.k};'
    )
```

Notice that we also pass the task name via the option `meta_key`. Inside of the C++ source file [customers/tutorials/src/decode_tu6b.cc](../../../customers/tutorials/src/decode_tu6b.cc) we can then use it as the key into the meta map in the helper function `insert_meta`. As a consequence, if we change the task name in our YAML, the changes will be reflected in C++ as well without recompiling.

We have to store the options inside C++ somehow. To do this, first we declare a struct.

```cpp
struct classification_properties
{
    std::string key;
    int top_k;
}
```

After that we have to define two functions with the signatures shown below, see [customers/tutorials/src/decode_tu6b.cc](../../../customers/tutorials/src/decode_tu6b.cc).
The keywords must be registered in the function `allowed_properties`.

```cpp
extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "top_k",
  };
  return allowed_properties;
}
```

The struct declared above is initialized and stored inside a shared pointer in the function `init_and_set_static_properties`.

```cpp
extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &)
{
  std::shared_ptr<classification_properties> prop =
    std::make_shared<classification_properties>();
  prop->key = Ax::get_property(
    input, "meta_key", "classification_properties", prop->key);
  prop->top_k = Ax::get_property(
    input, "top_k", "classification_properties", prop->top_k);
  return prop;
}
```

The second parameter of the helper function `get_property` corresponds to the key that has been chosen in the Python function `build_gst`.

### Takeaways
   - Write a C/C++ decoder without having any knowledge of GStreamer
   - Organize the results of your decoder into a class derived from `AxMetaBase`
   - Place your decoder in your own development directory and build it as a shared library with `make operators`
   - Pass options to the C/C++ decoder
   - Integrate the shared library into AxOperator and build the e2e pipeline from inference.py

### Appendix - Access Metadata in C++
It was mentioned above that the results created in the decoder are passed to other elements. The interface to all AxStreamer plugins always exposes the meta map. Once an entry is created there, the data can be retrieved by a standard map access. A helper function `get_meta` is available which returns a pointer to the data in case the entry exists and is of the type specified as the template argument. Otherwise, it throws an exception.

```cpp
AxMetaClassification *meta = ax_utils::get_meta<AxMetaClassification>("classifier", map, "tutorial_decoder")
```

## Tutorial-7: Defining Custom Abstracted Inference Metadata for Your Vision Task

In the previous tutorial, we used predefined classes derived from `AxMetaBase` to store the decoded results of our neural networks. For some tasks, those might not meet the requirements. In the following, we will write our own C++ class `MyCppClassificationMeta` and the corresponding Python class `MyPyClassificationMeta` derived from `AxTaskMeta` and use it in our pipeline.

The C++ decoder file [customers/tutorials/src/decode_tu7.cc](../../../customers/tutorials/src/decode_tu7.cc), where we add our custom metadata format, is almost equal to [customers/tutorials/src/decode_tu6a.cc](../../../customers/tutorials/src/decode_tu6a.cc), except that it constructs an instance of `MyCppClassificationMeta`.

```cpp
  ax_utils::insert_meta<MyCppClassificationMeta>(map, prop->key, {}, 0, 1,
        std::vector<float>{ max_score },
        std::vector<int32_t>{ max_class_id },
        prop->num_classes);
```

The definitions of `MyCppClassificationMeta` can be found in the header file [customers/tutorials/include/MyCppClassificationMeta_tu7.h](../../../customers/tutorials/include/MyCppClassificationMeta_tu7.h), which gets included in the decoder file. To make sure the file is found, we have to add the folder to [customers/tutorials/CMakeLists.txt](../../../customers/tutorials/CMakeLists.txt).

```
add_customer_libraries(tutorials
    SOURCES
        src/decode_tu7.cc
    INCLUDE_DIRECTORIES
        include
)
```

Our custom class `MyCppClassificationMeta`, which inherits from `AxMetaBase`, has three private member variables that are initialized in the constructor and contain the scores, class IDs and the number of classes. It is possible to implement C++ business logic in member functions and run them in the decoder or other GStreamer plugins. Alternatively, we can pass the data to the Python code.

The member function `get_extern_meta` serves as an interface between C++ and Python. For our `MyCppClassificationMeta`, we implemented the following

```cpp
  std::vector<extern_meta> get_extern_meta() const override
  {
    auto results = std::vector<extern_meta>();
    results.push_back({ "MyPyClassificationMeta", "scores_subtype",
        int(scores.size() * sizeof(float)),
        reinterpret_cast<const char *>(scores.data()) });
    results.push_back({ "MyPyClassificationMeta", "classes_subtype",
        int(classes.size() * sizeof(int)),
        reinterpret_cast<const char *>(classes.data()) });
    ...
    return results;
  }
```

Each `extern_meta` struct contains four values:
1. The name of the type. The char array must be valid until the data is copied to Python, so it must not be a local string. This name must correspond to the name of the class derived from `AxTaskMeta` in the Python code.
1. The name of the subtype. The char array must be valid until the data is copied to Python, so it must not be a local string. The subtype is used in the `decode` function of the class derived from `AxTaskMeta` in the Python code.
1. The size of the data in number of bytes.
1. The pointer to the data. It must stay valid until the data is copied to Python, so it must not point to data that is local to the function.

In the Python code, we must define a corresponding class derived from `AxTaskMeta` and member variables that store the required data. We created the class `MyPyClassificationMeta` in the file [ax_models/tutorials/general/tutorial_decoders_tu7.py](../../../ax_models/tutorials/general/tutorial_decoders_tu7.py). Here, just to get a runnable example using dataset, we added some member functions that are required to do the evaluation.

To read the metadata into the Python object, we define the function `decode` of `MyPyClassificationMeta`.

```cpp
@classmethod
def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> 'MyPyClassificationMeta':
    scores = data.get("scores_subtype", b"")
    scores = np.frombuffer(scores, dtype=np.float32)
    classes = data.get("classes_subtype", b"")
    classes = np.frombuffer(classes, dtype=np.int32)
    model_meta = cls(class_ids=classes, scores=scores, ...)
    return model_meta

```

The input into this function, i.e. `data`, is a dictionary with the subtypes as keys and a bytearray containing the actual data as values. You can easily see the analogy of this `decode` function to the `get_extern_meta` function of `MyCppClassificationMeta`. When we return from the `decode` function in Python, we automatically add the result into `axmeta`, the meta map of the Python code.

In its `exec_torch` function, the decoder `TopKDecoderWithMySimplifiedGstPluginAndMyAxTaskMeta` in the file mentioned above creates an instance of the Python class `MyPyClassificationMeta` as well.

A subclass of `AxTaskMeta` must be immutable. Once you call `add_instance` to put it into `axmeta`, it cannot be modified further. This ensures data consistency and stability for use in components like the visualizer and evaluator, while preventing accidental modifications during cascade pipeline execution.

We can now run inference with our new custom metadata format using `--pipe=torch` as well as using GStreamer.

```bash
./inference.py t7-axtaskmeta dataset --no-display
```

### Takeaways
   - Create a new Python class derived from `AxTaskMeta` according to your new type of vision task or models with specific output data and use it in the Python decoder
   - Create a corresponding C++ class derived from `AxMetaBase` and use it in your C++ decoder
   - Pass your specific data from C++ to Python

## Tutorial-8: Evaluating Model Performance with Your Own Metrics

This tutorial explains how to evaluate model performance using custom metrics. We'll start by using `TopKDecoderWithMyAxTaskMeta` from [tutorial_decoders_tu8.py](../../../ax_models/tutorials/general/tutorial_decoders_tu8.py) instead of [tutorial_decoders_tu7.py](../../../ax_models/tutorials/general/tutorial_decoders_tu7.py) in the YAML operators section. Then, in the YAML datasets section, we'll switch from `CustomDataAdapter` to either `CustomDataAdapterWithOfflineEvaluator` or `CustomDataAdapterWithOnlineEvaluator`, both implemented in [tutorial_data_adapter.py](../../../ax_models/tutorials/general/tutorial_data_adapter.py).

### Understanding AxTaskMeta::to_evaluation and types.BaseEvalSample

Here's a high-level overview of the `MyAxTaskMeta` class defined in [tutorial_decoders_tu8.py](../../../ax_models/tutorials/general/tutorial_decoders_tu8.py):

```python
@dataclass(frozen=True)
class MyAxTaskMeta(meta.AxTaskMeta):
    class_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def __post_init__(self):
        # inspect the data type of the class_ids and scores

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) ->
        # decode the data passed from the C++ pipeline into AxTaskMeta

    def to_evaluation(self):
        return MyClassificationEvalSample(
                class_ids=self.class_ids,
            )
```
The `to_evaluation` method is implemented to wrap the prediction results into a `MyClassificationEvalSample` object, a subclass of `BaseEvalSample`. In this example, `MyClassificationEvalSample` simply holds the `class_ids`, which are necessary for accuracy evaluation. This provides a clean interface for integrating with evaluators. This method can return various data structures like tuples, lists, or custom class instances.

```python
class MyClassificationEvalSample(types.BaseEvalSample):
    def __init__(self, class_ids: np.ndarray):
        self.class_ids = class_ids

    @property
    def data(self) -> Any:
        return self.class_ids
```

### Exploring the Evaluator Implementation

Now, let's investigate how the evaluator utilizes this data by looking at the following class:

```python
class OfflineEvaluator(types.Evaluator):
    def __init__(self, top_k: int = 1, **kwargs):
        # init all containers

    def process_meta(self, meta) -> None:
        # append the prediction and ground truth into the containers
        class_ids = meta.to_evaluation().data
        self.labels.append(class_ids[0])
        self.gt_labels.append(meta.access_ground_truth().data)

    def collect_metrics(self, key_metric_aggregator: str | None = None) -> types.EvalResult:
        # compute the accuracy by using sklearn
        top1 = accuracy_score(self.gt_labels, self.labels)
        # return the results in types.EvalResult
        eval_result = types.EvalResult(
            metric_names=['Top-1 Accuracy'],
            key_metric='Top-1 Accuracy',
        )
        eval_result.set_metric_result('Top-1 Accuracy', top1, is_percentage=True)
        return eval_result
```
The `OfflineEvaluator` class is designed for clarity and ease of understanding. It gathers prediction and ground truth data and then computes the accuracy using a library like scikit-learn, allowing integration of existing evaluation methodologies.

When accessing the prediction, the `to_evaluation` method retrieves the `class_ids`. You might ask why we use `meta.to_evaluation().data` instead of directly accessing `meta.class_ids`. While functionally equivalent in this case, using `to_evaluation` is a standard practice for evaluator integration. It abstracts the internal structure of `AxTaskMeta`, letting users focus on the data relevant for evaluation. This becomes particularly beneficial with more complex prediction structures, such as those found in MMLab, where `to_evaluation` can adapt our format to their specific requirements.

The ground truth is accessed via the `access_ground_truth` method. The source of this data is the dataset implementation's `reformat_for_validation` method, as detailed in [Tutorial-4](#tutorial-4-working-with-custom-datasets-and-accuracy-measurement). The ground truth isn't stored directly within `AxTaskMeta` but is accessed from its parent container, `AxMeta`, through the `access_ground_truth` helper function.

In `collect_metrics`, we compute the accuracy using scikit-learn. This step allows you to incorporate your established evaluation techniques. To display the results in the terminal, we encapsulate them within a `types.EvalResult` object using `set_metric_result`. The essential elements of an `EvalResult` object are the metric's name and its calculated value. You can customize how results are aggregated within the `EvalResult` object. Common aggregators include `average`, `max`, `min`, `median`, `std` and `best` for more informative logging. Here's how to configure an `EvalResult` object:

```python
eval_result = types.EvalResult(
    metric_names=['Top-1 Accuracy'],
    aggregators={'Top-1 Accuracy': 'average'},
    key_metric='Top-1 Accuracy',
    key_aggregator='average',
)
eval_result.set_metric_result('Top-1 Accuracy', top1, 'average', is_percentage=True)
```

The `OfflineEvaluator` implementation in [tutorial_data_adapter.py](../../../ax_models/tutorials/general/tutorial_data_adapter.py) might appear more intricate to showcase the ability to calculate and register multiple metrics within the `EvalResult`, allowing selection of a primary metric. The `organize_results` method demonstrates how aggregators can present average and weighted average metrics.

The `OfflineEvaluator` is integrated through the `CustomDataAdapterWithOfflineEvaluator`. This illustrates how a dataset is linked to its evaluator. `CustomDataAdapterWithOfflineEvaluator` inherits from `CustomDataAdapter` and overrides the `evaluator` method to return an instance of `OfflineEvaluator`. The dataloader/dataset object itself is still created by the `create_validation_data_loader` method within `CustomDataAdapter`.

### Executing the Offline Evaluator

First, install scikit-learn, as it's used for accuracy calculation:

```bash
pip install scikit-learn
```

Now, run the pipeline using the offline evaluator with the first 1000 frames:

```bash
./inference.py t8-evaluator dataset --no-display --pipe=torch --frames=1000
```

Note that in the dataloader implementation, shuffle is turned off, allowing consistent results with the first *N* frames. This is useful for debugging, but please be aware that, in many cases, the first *N* frames may belong to the same class. Relying only on those frames instead of the entire dataset can lead to meaningless results.

You should observe output similar to this:

```
INFO    : ===================================
INFO    : | Top-1 Accuracy_average | 82.40% |
INFO    : | Top-5 Accuracy_average | 96.20% |
INFO    : | Precision_macro_avg    | 34.28% |
INFO    : | Precision_weighted_avg | 97.69% |
INFO    : | Recall_macro_avg       | 29.45% |
INFO    : | Recall_weighted_avg    | 82.40% |
INFO    : | F1-score_macro_avg     | 31.08% |
INFO    : | F1-score_weighted_avg  | 87.55% |
INFO    : ===================================
INFO    : Key Metric (Top-1 Accuracy_average): 82.40%
```

To evaluate the entire dataset, remove the `--frames=1000` option. Use `--pipe=torch-aipu` to compare accuracy with models running on AIPU.

### Introducing the Online Evaluation Approach

The term "offline evaluator" signifies that it collects all data before computing metrics, a common practice in machine learning. However, this can be memory-intensive for real-time pipelines, especially with rapid model inference, potentially leading to garbage collection bottlenecks in Python. An "online evaluator," which calculates accuracy incrementally, offers a more memory-efficient alternative.

To switch to the online evaluator, comment out `CustomDataAdapterWithOfflineEvaluator` and uncomment `CustomDataAdapterWithOnlineEvaluator` in the YAML datasets section:
```yaml
# class: CustomDataAdapterWithOfflineEvaluator
class: CustomDataAdapterWithOnlineEvaluator
```

Run the pipeline again:

```bash
./inference.py t8-evaluator dataset --no-display --pipe=torch --frames=1000
```

You will see similar results to the offline evaluator, but the computation happens in real-time, processing data as it arrives.

The `CustomDataAdapterWithOnlineEvaluator` encapsulates the `OnlineEvaluator`. The `OnlineEvaluator` receives the `top_k` value from the `custom_config` and the `num_classes` parameter from the model information defined in the models section of your YAML pipeline. The `custom_config` dictionary originates from the operators implementations.

In [tutorial_decoders_tu8.py](../../../ax_models/tutorials/general/tutorial_decoders_tu8.py), `TopKDecoderWithMyAxTaskMeta` invokes the `register_validation_params` method within its `_post_init` method. This mechanism allows passing parameters from the operators to the evaluator. Since post-processing parameters can influence accuracy, it's beneficial to pass them to the evaluator. Here, the `top_k` parameter is passed, which is why it's present in the `OnlineEvaluator`'s `custom_config` dictionary.

Referring back to the [t8-evaluator.yaml](../../../ax_models/tutorials/general/t8-evaluator.yaml) pipelines section, you'll find the `eval` parameter configured within the `postprocess` section, like this:

```yaml
postprocess:
- my_topk_decoder:
    k: 1
    eval:
        k: 5
```
The `eval` parameter is specifically for controlling parameters during evaluation, distinguishing it from general inference settings. When running `inference.py` with a dataset, the `eval` parameters are used. For other input types, these parameters are ignored, and the default parameters (here, `k=1`) are used for inference. This separation is important because some models require specific parameter settings for fair evaluation benchmarking (e.g., YOLO often uses a confidence threshold of 0.001 for evaluation, which is impractical for general inference due to the high number of false positives).

In our case, the `OnlineEvaluator` will use `top_k` as 5, as specified in the `eval` parameter. Experiment by changing the `top_k` value under `eval` and re-running the pipeline to observe the impact on the results. This enables a low-code approach to accuracy measurement, allowing you to adjust evaluation parameters directly in the YAML configuration without code modifications.

Besides `model_info` and `custom_config`, the evaluator method can also access parameters from `dataset_config`, passed from the YAML datasets section. For instance, in your YAML file:

```yaml
datasets:
  Custom-Fruits360:
    class: CustomDataAdapterWithOfflineEvaluator
    class_path: tutorial_data_adapter.py
    my_param: 100
```
You can then access `my_param` within the evaluator method using `dataset_config['my_param']`. This is useful for controlling evaluator behavior based on dataset-specific configurations, such as different file paths. The `pair_validation` flag indicates if pair validation is active, and `dataset_root` provides the dataset's root path, which might be needed for loading specific evaluation data.

The `OnlineEvaluator` initializes metrics directly instead of creating separate containers. In the `process_meta` method, it updates these metrics incrementally. This is a more memory-efficient design. Once all data is processed and `collect_metrics` is called, the evaluator aggregates the metrics and structures the results into a `types.EvalResult` object.

While not strictly mandatory, using an online evaluator is a recommended practice, particularly for real-time applications. The Axelera Model-Zoo utilizes online evaluators for all its tasks.

### Takeaways
From this tutorial, you have learned:

- How to integrate a custom evaluator into the `DataAdapter` for tailored accuracy measurement.
- How to utilize the `eval` parameter in the YAML pipeline to specify evaluation-specific parameters and how to pass parameters from operators or the YAML datasets section to the evaluator.
- The advantages of implementing an online evaluator for reduced memory consumption and the implementation process.
- How to structure your metrics within `types.EvalResult` for clear display in the terminal.

## Tutorial-9: Cascaded and parallel pipelines

To run several models in parallel, the individual pipeline descriptions are just pasted one after the other as list items under the top level key `pipeline` in the yaml file. With a key corresponding to the task name of each of the models, the code will populate the meta map with metadata, all of which the user can access in the business logic.

In contrast to the above, a cascaded pipeline consists of multiple models arranged in a hierarchical structure. In this setup, a master model serves as the primary model, with its output feeding into one or more dependent models. The results of these submodels refer to data from the master. A typical example is a detection-classification pipeline, where the master model is a detector that identifies regions of interest (ROIs) and the classifier assigns a label to each ROI.

An example of a YOLO detector cascaded into the Resnet34 classifier known from the previous tutorials can be found in the file [t9-cascaded.yaml](../../../ax_models/tutorials/general/t9-cascaded.yaml). In this yaml file we now have two list items under the top-level key `pipeline`. The first list item is the `detections` task, the second one is the `classifier` task. The associations between the two models are established in the input operator of the dependent model. As `detections` serves as the master, no changes are required there. We specify that `detections` is the master of `classifier` with the help of the key `where` in the input operator of the subtask `classifier`. In addition, we have to set the source to `roi`.

```yaml
pipeline:
  - detections:
      model_name: model1
      ...
  - classifier:
      model_name: model2
      input:
          ...
          source: roi
          where: detections
```

The code will now run the second model for every detection result of the first model. While the master model operates on the video frame, we call the frames that the dependent models work on *subframes*.

In addition to the changes to the pipeline mentioned above, we also have to adapt the model section of the yaml file, because the number of AIPU cores for each of the models must be specified. The key `aipu_cores` is used for this purpose.

When deciding how many AIPU cores to allocate, it might seem intuitive to assign more cores to computationally more demanding models. For example, YOLOv8n may have fewer FPS compared to ResNet34. However, in practice, you should consider how many ROIs need to be processed by the second model. For instance, if your master model runs at 200 FPS and outputs 10 ROIs per frame, your second model will need to handle 2000 FPS to process all the ROIs. This can be achieved by allocating the appropriate number of AIPU cores.

```yaml
models:
  model1:
    ...
    extra_kwargs:
      aipu_cores: 3
  model2:
    ...
    extra_kwargs:
      aipu_cores: 1
```

The Python decoder in file [ax_models/tutorials/general/tutorial_decoders_tu9.py](../../../ax_models/tutorials/general/tutorial_decoders_tu9.py) needs only minor adaptations compared to the previous tutorials. The function `configure_model_and_context_info` of the base class `AxOperator`, which is called in the application framework for each operator, assigns the master meta name to the member variable `self._where`. In the GST pipeline, this variable is passed to the subplugins, while in the torch pipeline, it is used in `add_instance`.

```python
axmeta.add_instance(self.task_name, model_meta, self._where)
```

In our C++ decoder, we have to make some minor adaptations as well. The file [customers/tutorials/src/decode_tu9_master.cc](../../../customers/tutorials/src/decode_tu9_master.cc) looks familiar from tutorial 6. The difference is that we now have an additional property which is required to pass the name of the master meta into the decoder. Furthermore, we use the third and fourth parameter in the signature of `decode_to_meta`. Those specify the index of the current subframe and the total number of subframes, in our case the number of classification models that have been executed for each detection model. The `insert_meta` helper functions uses those three additional parameters to store the results of the submodels in such a way that the connection between the primary task and the dependent tasks is maintained.

```cpp
extern "C" void
decode_to_meta(const AxTensorsInterface &tensors, const classification_properties *prop,
    unsigned int subframe_index, unsigned int number_of_subframes, ...)
{
  ...
  ax_utils::insert_meta<AxMetaClassification>(map, prop->key,
      prop->master_meta_key, subframe_index, number_of_subframes,
      std::vector<std::vector<float>>{ { max_score } },
      std::vector<std::vector<int32_t>>{ { max_class_id } },
      std::vector<std::vector<std::string>>{ { std::to_string(max_class_id) } });
}
```

Where does the `number_of_subframes` parameter inside of the decoder of the dependent network come from? The answer is: from the master meta! The C++ base class for metadata has the virtual function `get_number_of_subframes`. It can be overridden by the user to return the number of subframes that should be created. For detection models or any other models where the meta format is derived from class `AxMetaBbox`, this is just the number of bounding boxes.

```cpp
class AxMetaBase
{
  virtual size_t get_number_of_subframes() const
  {
    return 1;
  }
}
```

### Filtering and Selecting ROIs

In cascaded pipelines, the first model usually serves as a detector and the dependent models as classifiers that add more information about the detection result. In the following, two ways to control which ROIs are processed by secondary networks are presented: **label filtering** and **top-K selection**.

#### Label Filtering

Imagine a detector which finds bottles and persons, and classifiers that are applicable to only one of either bottle or person, e.g. one secondary network distinguishes between wine and beer bottles while another secondary network determines the age of a person. To solve this problem, the ROIs must be filtered before entering the secondary networks. This can be achieved by using the YAML keyword `label_filter` in the input operator.

In the case above, we might want to add a filter which selects only those ROIs that are labeled `bottle` to the input operator of the secondary model:

```yaml
pipeline:
  ...
  - classifier:
      model_name: model2
      input:
          ...
          label_filter: bottle
```

#### Top-K ROI Selection

The platform supports advanced functionality to enable the selection of the top-K ROIs, prioritizing them by different criteria. You can configure how many ROIs to handle by setting the `top_k` parameter and specify which ROIs to prioritize using the `which` parameter.

```yaml
pipeline:
  - master_detections:    # first task's name
      model_name: yolov5m
      preprocess:
         - resize:
         - totensor:
      postprocess:
        - decodeyolo:
            max_nms_boxes: 30000
            conf_threshold: 0.3
            nms_iou_threshold: 0.5
            nms_class_agnostic: False
            nms_top_k: 300
            use_multi_label: False
  - classifications:
      model_name: ResNet50-ImageNet1K
      input:
        type: image
        source: roi
        where: master_detections   #  take ROI from the first task
        which: AREA # SCORE, CENTER
        top_k: 10
```

**Selection criteria and use cases:**

- **`SCORES`**: Selects ROIs with the highest detection confidence scores. **Use case:** Quality control applications where you only want to classify the most confident detections, such as defect detection in manufacturing.
- **`CENTER`**: Selects ROIs closest to the image center. **Use case:** Driver assistance systems where objects in the center of view (like the car ahead) are more critical than peripheral objects.
- **`AREA`**:  Selects ROIs with the largest areas. **Use case:** Security surveillance where larger objects (closer people or vehicles) are prioritized over smaller, distant objects.

> [!IMPORTANT]  
> **Performance Consideration for Real-Time Applications**
> 
> The `top_k` value directly impacts pipeline performance. Calculate your throughput capacity:
> 
> **Example:** Detection at 10 FPS with 20 ROIs/frame = 200 ROIs/second  
> If classifier runs at 100 FPS → **Maximum safe `top_k` = 10** (100 ÷ 10)
> 
> Setting `top_k: 15` would require 150 FPS from classifier, exceeding capacity and causing frame drops.

> [!TIP]
> **Choosing `top_k`:**
> 1. Calculate: `classifier_fps ÷ detector_fps` = maximum safe `top_k`
> 2. Monitor performance metrics to validate settings
> 3. Consider that fewer, higher-quality ROIs often perform better than processing all detections

When using filters in the input operator, internally a new hidden meta object is created, which contains the filtered boxes and the associations between those boxes and the boxes inside the original master meta. The name of this association meta can be taken from the context and assigned to a member variable inside of the function `configure_model_and_context_info` in the Python decode operator file [ax_models/tutorials/general/tutorial_decoders_tu9.py](../../../ax_models/tutorials/general/tutorial_decoders_tu9.py).

```python
self._association = context.association or None
```

If the results of the secondary models are assigned as submeta to the original master meta, the name of the hidden meta must be specified as well. Instead of the function `insert_meta`, the function `insert_and_associate_meta` has to be used, which takes the hidden meta as one additional parameter. An example can be found in the file [customers/tutorials/src/decode_tu9.cc](../../../customers/tutorials/src/decode_tu9.cc). The additional parameter must be passed from the Python code as a property, just like the master meta key.

```cpp
...
ax_utils::insert_and_associate_meta<AxMetaClassification>(map, prop->key,
  prop->master_meta_key, subframe_index, number_of_subframes,
  prop->association_meta_key, std::vector<std::vector<float>>{ { max_score } },
  std::vector<std::vector<int32_t>>{ { max_class_id } },
  std::vector<std::vector<std::string>>{ { std::to_string(max_class_id) } }
);
```

#### Complete Pipeline Example with Label Filtering

Here's a complete YAML configuration showing label filtering in action:

```yaml
pipeline:
  - detections:
      model_name: model1
      # ... detection configuration ...
  - classifier:
      model_name: model2
      input:
          type: image
          color_format: RGB
          source: roi
          where: detections
          label_filter: [banana, apple, orange, bowl]  # Only process these classes
      preprocess:
        - resize:
            width: 100
            height: 100
        - torch-totensor:
        - normalize:
            mean: 0.485, 0.456, 0.406
            std: 0.229, 0.224, 0.225
      postprocess:
        - my_topk_decoder:
        - access_submeta_operator:  # C++ business logic (comment out for Python-only)
```

### Accessing Submeta in Your Application

After setting up filtering, you'll need to access the classification results in your application code. The platform provides APIs for both C++ and Python to handle cases where some detections may not have secondary metadata (due to filtering).

#### C++ Implementation

To access the submeta inside of a C++ plugin, first the pointer to the master is required. Each master meta contains a shared pointer to a map of submetas, called `submeta_map`. This data structure maps strings which represent the submeta names to a vector of pointers to objects derived from `AxMetaBase`. The size of the vector is just the number of subframes. The overloaded function `get` is used to retrieve either the complete vector, or the submeta of a specified subframe. In your C++ code, you need to handle the fact that not all master detections will have corresponding submeta due to filtering. The key steps are shown in [customers/tutorials/src/inplace_tu9.cc](../../../customers/tutorials/src/inplace_tu9.cc):

**1. Check for subtask existence:**
```cpp
// Check if the subtask exists in the submeta names
auto submeta_names = master_meta->submeta_names();
if (std::none_of(submeta_names.begin(), submeta_names.end(),
        [&prop](const char *name) { return prop->subtask_name == name; })) {
  std::cout << "Subtask name not found. Normal for frames with no objects after filtering." << std::endl;
  return;
}
```

**2. Get submetas with null-checking:**
```cpp
// Get all submetas (some may be null due to filtering)
std::vector<AxMetaClassification *> submetas
    = master_meta->get_submetas<AxMetaClassification>(prop->subtask_name);

// Check if this box has classification results (only filtered objects should)
if (i < submetas.size() && submetas[i] != nullptr) {
  auto class_id = submetas[i]->get_classes()[0][0];  // Always a 2D array
  auto score = submetas[i]->get_scores()[0][0];      // Always a 2D array
  // Process classification results...
} else {
  // This detection was filtered out - handle accordingly
}
```

#### Python Implementation

For Python applications, you have multiple ways to access submeta. The complete implementation can be found in [ax_models/tutorials/general/t9-application.py](../../../ax_models/tutorials/general/t9-application.py). Here are the key approaches:

**Option 1: Low-level metadata access (similar to C++)**

Direct access to underlying `AxTaskMeta` - for custom AxTaskMeta implementations where no object view exists
```python
if ('classifier' in detection_meta.secondary_frame_indices and 
    i in detection_meta.secondary_frame_indices['classifier']):
    
    classifier_meta = detection_meta.get_secondary_meta('classifier', i)
    class_ids, scores = classifier_meta.get_result(0)
    class_id = class_ids[0]  # Raw array access
    score = scores[0]        # Raw array access
    # Process results...
else:
    # This detection was filtered out
    print(f"Detection not classified (filtered out)")
```

**Option 2: Object-oriented metadata access**

Access via metadata objects - good when master is standard (detection/segmentation) but secondary is custom
```python
for master in frame_result.detections:
    if 'classifier' in master.secondary_task_names:
        classifier_meta = master.get_secondary_meta('classifier')
        # Access classification results...
    else:
        print("Detection has no classification (filtered out)")
```

**Option 3: High-level object view**

Simplified object access - best for built-in AxTaskMeta
```python
for master in frame_result.detections:
    if 'classifier' in master.secondary_task_names:
        secondary_objects = master.get_secondary_objects('classifier')
        for secondary in secondary_objects:
            class_id = secondary.class_id[0]
            score = secondary.score[0]
    else:
        print("No classifier results for this detection")
```

> [!IMPORTANT]  
> **Handling Filtered Objects**
> 
> When using `label_filter`, not all master detections will have corresponding submeta. Your code must check for null/missing submeta before accessing classification results. The examples above show proper null-checking patterns for both C++ and Python.

> [!TIP]
> **Python vs C++ Processing**
> 
> - **C++ approach**: Use `access_submeta_operator` in your YAML pipeline for processing submeta within the inference pipeline
> - **Python approach**: Comment out the C++ operator and handle submeta access in your Python application code using the three methods shown above

### Running the Cascaded Pipeline

First, download the test video:

```bash
# Create media directory and download test video
mkdir -p ./media
wget https://media.axelera.ai/artifacts/test_videos/bowl-of-fruit.mp4 -O ./media/bowl-of-fruit.mp4
```

> [!NOTE]  
> **Focus on Pipeline Flow, Not Classification Accuracy**
> 
> This example demonstrates the detection→classification cascaded pattern commonly used in real-world applications. The fruit classifier is trained on controlled dataset images and may not accurately classify real-world objects, but it effectively shows how to implement and access submeta in cascaded pipelines.

**Run the C++ implementation (with access_submeta_operator in YAML):**
```bash
./inference.py t9-cascaded ./media/bowl-of-fruit.mp4
```

**Run the Python implementation:**
```bash
# First comment out 'access_submeta_operator' line in your YAML pipeline
# Then run the Python application directly:
python ./ax_models/tutorials/general/t9-application.py
```

> [!TIP]  
> The Python example (`t9-application.py`) has the video path configured internally and demonstrates three different ways to access submeta programmatically.

### Takeaways

From this tutorial, you have learned:

- How to build cascaded pipelines where a master model (detector) feeds ROIs to dependent models (classifiers) for hierarchical processing.
- How to implement ROI filtering using `label_filter` and `top_k` selection with different prioritization criteria (`SCORES`, `CENTER`, `AREA`) to optimize real-time performance.
- How to access submeta results in both C++ and Python applications, including proper null-checking for filtered objects that don't have secondary classifications.
- How to calculate and balance throughput requirements between master and secondary models to maintain real-time performance in production systems.

## Tutorial-10: Face detection and recognition pipeline

A real-world implementation of a cascaded pipeline is demonstrated through the face detection and recognition system. This pipeline operates in two distinct phases: first, it determines the bounding box coordinates around detected faces, and second, it establishes the identity of each person. The recognition models typically generate embedding vectors, which serve as numerical representations of facial features that can be compared against a dictionary containing mappings between embeddings and individual names.

### Creating an Embeddings File

Face recognition models are typically trained on extensive datasets comprising publicly available images. To enable identification of specific individuals, their facial embeddings must be computed and stored alongside their corresponding names. This reference dictionary allows the recognition model to compare newly computed embeddings against stored embeddings using distance metrics such as cosine similarity. A positive identification is achieved when the computed distance falls below a predefined threshold and represents the minimum distance among all stored individuals in the embeddings database.
For this implementation, we utilize the FaceNet model for face recognition and RetinaFace for face detection. To create the embeddings database, configure the file located at [ax_models/reference/cascade/face-recognition.yaml](../../../ax_models/reference/cascade/face-recognition.yaml). 

Deploy the model using the following command:

```bash
./deploy.py face-recognition
```

While the model is being deployed, prepare the test images and video files:

```bash
wget https://media.axelera.ai/artifacts/reference/face_recog/famous_faces.zip -O ./famous_faces.zip
unzip -q ./famous_faces.zip -d ./

wget https://media.axelera.ai/artifacts/reference/face_recog/famous_people.mp4 -O ./media/famous_people.mp4
```

After successful model deployment, execute the following command to generate the embeddings file:

```bash
GST_DEBUG=ax*:4 ./inference.py face-recognition famous_faces --no-display 2>&1 | grep -E "(Recognized person|facenet_recog)"
```

The environment variable GST_DEBUG=ax*:4 enables debug logging from the C++ components. You should observe output similar to:

```text
0:00:00.525100100 3583698 0x609b8b0cd8f0 INFO          axinferencenet AxDecodeFacenet.cc:117:decode_to_meta: facenet_recog: Added new person: 'Ed_Sheeran'
0:00:00.525449582 3583698 0x609b8b0cd8f0 INFO          axinferencenet AxDecodeFacenet.cc:117:decode_to_meta: facenet_recog: Added new person: 'Elon'
0:00:00.528724121 3583698 0x609b8b0cd8f0 INFO          axinferencenet AxDecodeFacenet.cc:117:decode_to_meta: facenet_recog: Added new person: 'Emma_Watson'
0:00:00.528965061 3583698 0x609b8b0cd8f0 INFO          axinferencenet AxDecodeFacenet.cc:117:decode_to_meta: facenet_recog: Added new person: 'Fabrizio'
0:00:00.529262009 3583698 0x609b8b0cd8f0 INFO          axinferencenet AxDecodeFacenet.cc:117:decode_to_meta: facenet_recog: Added new person: 'Lionel_Messi'
0:00:00.532228250 3583698 0x609b8b0cd8f0 INFO          axinferencenet AxDecodeFacenet.cc:117:decode_to_meta: facenet_recog: Added new person: 'Psy'
0:00:00.532642970 3583698 0x609b8b0cd8f0 INFO          axinferencenet AxDecodeFacenet.cc:117:decode_to_meta: facenet_recog: Added new person: 'Shohei_Ohtani'
0:00:00.533104312 3583698 0x609b8b0cd8f0 INFO          axinferencenet AxDecodeFacenet.cc:117:decode_to_meta: facenet_recog: Added new person: 'Taylor_Swift'
```

This debug output is invaluable for troubleshooting purposes. The logging level "ax*:4" corresponds to AX_INFO level, while "ax*:5" includes AX_DEBUG level logging. Within the GStreamer framework, we recommend utilizing our debug logging system to monitor pipeline operations rather than relying on print statements.

Verify that the `famous_embeddings.json` file contains 8 entries with their corresponding names and embeddings. The names are derived from the image filenames in the `famous_faces` directory, while the embeddings are computed by the face-recognition pipeline.

Next, disable embedding updates by setting `update_embeddings` to `False` in the configuration file [ax_models/reference/cascade/face-recognition.yaml](../../../ax_models/reference/cascade/face-recognition.yaml). This prevents the embeddings from being modified during subsequent runs with new faces.

```yaml
      postprocess:
        - recognition:
            ...
            update_embeddings: False
```

Execute the model again using the same command:

```bash
GST_DEBUG=ax*:4 ./inference.py face-recognition famous_faces --no-display 2>&1 | grep -E "(Recognized person|facenet_recog)"
```

The output should display:
```text
(rundev-release) ubuntu@ein0-3060-01:~/software-platform/host/application/framework$ GST_DEBUG=ax*:4 ./inference.py face-recognition famous_faces --no-display 2>&1 | grep -E "(Recognized person|facenet_recog)"
0:00:00.357525903 3584023 0x5cf2d87b2270 INFO          axinferencenet AxDecodeFacenet.cc:264:init_and_set_static_properties: facenet_recog: Loaded 8 existing embeddings from: /home/ubuntu/software-platform/host/application/framework/famous_embeddings.json
0:00:00.500568545 3584023 0x5cf2d87b2550 INFO          axinferencenet AxDecodeFacenet.cc:207:decode_to_meta: facenet_recognition: Recognized person: 'Ed_Sheeran' with score: 1 (threshold: 0.352)
0:00:00.503771113 3584023 0x5cf2d87b2550 INFO          axinferencenet AxDecodeFacenet.cc:207:decode_to_meta: facenet_recognition: Recognized person: 'Elon' with score: 1 (threshold: 0.352)
0:00:00.503792531 3584023 0x5cf2d87b2550 INFO          axinferencenet AxDecodeFacenet.cc:207:decode_to_meta: facenet_recognition: Recognized person: 'Emma_Watson' with score: 0.999999 (threshold: 0.352)
0:00:00.503846089 3584023 0x5cf2d87b2550 INFO          axinferencenet AxDecodeFacenet.cc:207:decode_to_meta: facenet_recognition: Recognized person: 'Fabrizio' with score: 1 (threshold: 0.352)
0:00:00.507021044 3584023 0x5cf2d87b2550 INFO          axinferencenet AxDecodeFacenet.cc:207:decode_to_meta: facenet_recognition: Recognized person: 'Lionel_Messi' with score: 0.999999 (threshold: 0.352)
0:00:00.507120091 3584023 0x5cf2d87b2550 INFO          axinferencenet AxDecodeFacenet.cc:207:decode_to_meta: facenet_recognition: Recognized person: 'Psy' with score: 1 (threshold: 0.352)
0:00:00.507147470 3584023 0x5cf2d87b2550 INFO          axinferencenet AxDecodeFacenet.cc:207:decode_to_meta: facenet_recognition: Recognized person: 'Shohei_Ohtani' with score: 1 (threshold: 0.352)
0:00:00.510035887 3584023 0x5cf2d87b2550 INFO          axinferencenet AxDecodeFacenet.cc:207:decode_to_meta: facenet_recognition: Recognized person: 'Taylor_Swift' with score: 0.999999 (threshold: 0.352)
```

This demonstrates perfect recognition accuracy, as the system is now processing the same images used to generate the original embeddings.

### The Cascaded Pipeline

The cascaded pipeline comprises two primary components: the RetinaFace model, which provides detection results and 5 facial landmarks, and the FaceNet model for recognition functionality.


```yaml
pipeline:
  - detections:
      model_name: retinaface-resnet50-widerface-onnx
      ...
  - recognitions:
      model_name: facenet-lfw
      input:
          type: image
          source: roi
          where: detections
          which: AREA # AREA, SCORE, CENTER
          top_k: 1
          min_width: 80
          min_height: 80
          image_processing_on_roi:
          - facealign:
              keypoints_submeta_key: detections
              width: 192
              height: 192
              save_aligned_images: False # for debugging
      postprocess:
        - recognition:
```

Key configuration parameters include:

1. The `which` parameter is set to `AREA`, prioritizing the region of interest (ROI) with the largest area for selection.
2. The `top_k` parameter is configured to 1, ensuring only the ROI with the largest area is processed.
3. The `image_processing_on_roi` section handles ROI preprocessing. The `facealign` operator utilizes keypoints from RetinaFace to transform the detected face into a normalized frontal view.
  - Enabling `save_aligned_images` will save aligned images to the `face_align_debug` directory for debugging purposes.
  - Face alignment is optional; removing the `image_processing_on_roi` section will not compromise pipeline functionality.
  - This demonstrates the capability to perform custom image processing on ROIs before feeding them to subsequent models.
4. The `keypoints_submeta_key` parameter references `detections`, indicating that RetinaFace keypoints will be used for facial transformation.
5. The `width` and `height` parameters are set to 192 pixels, transforming detected faces to 192×192 resolution. This optimization enhances performance since the FaceNet model requires 192×192 input images.
6. The `min_width` and `min_height` parameters filter out ROIs smaller than 80×80 pixels.


Execute the pipeline with the test video using:

```bash
./inference.py face-recognition media/famous_people.mp4
```

The results should be displayed on your screen.

#### Register yourself in the embeddings file

To register yourself in the embeddings database, enable `update_embeddings` in `ax_models/reference/cascade/face-recognition.yaml` and place your photograph in the `famous_faces` directory, using your name as the filename (supporting PNG or JPG formats).

```bash
./inference.py face-recognition famous_faces
```

Subsequently, disable `update_embeddings` and test the pipeline with your USB camera to verify personal recognition:

```bash
./inference.py face-recognition usb:0
```

This assumes your webcam is connected and accessible via `/dev/video0`. Modify the device name if using an alternative video device.


### Adding a Tracker

Continuously executing face recognition for every frame is computationally expensive and can degrade performance. However, repeated identification becomes unnecessary once a person's identity is established with confidence. By implementing a tracker, we can monitor facial movement across frames. The tracker assigns unique IDs to each face, ideally maintaining consistency throughout the video sequence. This approach allows recognition to be performed once per tracked individual. In practice, recognition accuracy is improved by running the sub-model a limited number of times and selecting the most frequently occurring identification result.

The enhanced configuration file [ax_models/reference/cascade/with_tracker/face-recognition-with-vote.yaml](../../../ax_models/reference/cascade/with_tracker/face-recognition-with-vote.yaml) incorporates the OC-Sort tracker within the pipeline:

```yaml
pipeline:
  - detections:
      model_name: retinaface-resnet50-widerface-onnx
  ...
  - tracking:
      model_name: tracker
      input:
        source: full
        color_format: RGB
      cv_process:
        - tracker:
            algorithm: oc-sort
            bbox_task_name: detections
            ...
            filter_callbacks:
              recognitions:
                lib: libtrackerfilter_numsubtaskruns.so
                num_subtask_runs: 10
            determine_object_attribute_callbacks:
              recognitions:
                lib: libtrackerattribute_mostfrequent.so
  - recognitions:
      model_name: facenet-lfw
      input:
          type: image
          source: roi
          where: tracking
          which: AREA # AREA, SCORE, CENTER
          top_k: 10
```

The `bbox_task_name` property establishes the connection between the `detections` task (which generates bounding boxes) and the `tracking` task. The `filter_callbacks` section limits the maximum number of recognition runs to 10 per track ID. The `determine_object_attribute_callbacks` section implements a voting mechanism, selecting the most frequently occurring name from the 10 recognition attempts as the final result for each track ID.

Download the test video:

```bash
wget https://media.axelera.ai/artifacts/test_videos/Fabrizio_talk.mp4 -O ./media/Fabrizio_talk.mp4
```

Execute the enhanced pipeline:

```bash
./inference.py face-recognition-with-vote Fabrizio_talk.mp4
```

You will notice higher frame rates after the initial few seconds, as individuals are recognized once and then tracked, rather than being re-identified in every frame. Note that when a person disappears and later reappears, they will be assigned a new track ID, which restarts the voting mechanism as the system treats them as a different individual. Additionally, this tracker version is not compatible with the `famous_people.mp4` video, as it rapidly switches between multiple people, making it unsuitable for tracking.

### Takeaways

In this tutorial, you've learned how to build efficient application pipelines using YAML configuration. Key learnings include:

- **Building cascaded pipelines**: You've created a real-world two-stage pipeline where your detector (RetinaFace) locates faces and feeds ROIs to your recognition model (FaceNet) - a pattern you can apply to many other vision tasks.
- **Configuring with YAML**: Using declarative YAML files to define sophisticated processing pipelines, making your applications easier to maintain, and modify the parameters without recompiling code.
- **Optimizing with tracking**: You now know how to easily plug in a tracker to your pipeline, allowing for efficient tracking of objects across frames and reducing the need for repeated recognition with the callback mechanism.
- **Debugging effectively**: You now have experience with `GST_DEBUG` with Axelera Logger to monitor pipeline operations and diagnose issues at various levels of detail.

**Try this next**: Modify the recognition threshold in the YAML file to see how it affects the balance between false positives and false negatives in your face recognition system. You can also experiment with implementing a multi-stage cascade by adding additional attribute classifiers (such as age, gender, etc.) after face detection. Each task should have a unique name, but all secondary models can share the same input configuration:
```yaml
      input:
          type: image
          source: roi
          where: detections
```
With this setup, the ROIs from the detection stage will be distributed in parallel to all second-level models.

## Other References

The `tutorials/resnet34_caltech101/` directory provides a complete example of how to deploy a **torchvision model** to the Metis platform in a low-code manner. This example demonstrates the following components:

- **[tutorial_resnet34_caltech101.py](../../../ax_models/tutorials/resnet34_caltech101/tutorial_resnet34_caltech101.py)**: The training script for the ResNet34 model on the Caltech101 dataset.
- **[tutorial_resnet34_caltech101.yaml](../../../ax_models/tutorials/resnet34_caltech101/tutorial_resnet34_caltech101.yaml)**: A YAML configuration file for deploying the model and pipeline. It leverages existing **AxOperators**, **types.DataAdapter**, and **types.Evaluator** to simplify the deployment process.

This example serves as a practical reference for using torchvision models with the Metis platform, showcasing how to train, configure, and deploy with minimal coding effort.

The `ax_models/cascade` directory contains additional references for building **cascade** and **parallel pipelines** using existing AxOperators. Each YAML file in this directory is designed to have a corresponding `application.py` for your reference, making it easier to understand and implement these pipelines in your own projects.
