# Copyright Axelera AI, 2024
import functools
from pathlib import Path

from .generated import generate_compilation_configs, generate_operators
from .types import (
    Bool,
    ColorFormat,
    EmptyDict,
    EmptyList,
    Enum,
    Float,
    ImageReaderBackend,
    InputSource,
    Int,
    List,
    MapCombined,
    MapPattern,
    ModelType,
    Null,
    Optional,
    Required,
    Str,
    SupportedLabelType,
    TaskCategory,
    TensorLayout,
    TopKRanking,
    Union,
    compile_schema,
)

extra_kwargs = lambda operators, compilation_configs: {
    "_type": MapCombined,
    Optional["YOLO"]: {
        Optional["anchors"]: List[List[Int]],
        Optional["anchors_path"]: Str,
        Optional["anchors_url"]: Str,
        Optional["anchors_md5"]: Str,
        Optional["strides"]: List[Int],
        Optional["focus_layer_replacement"]: Bool,
    },
    Optional["compilation_config"]: compilation_configs,
    Optional["compiler_config_file"]: Str,
    # Path to a TOML file containing compiler configuration. Supports:
    # - Absolute paths: /path/to/config.toml
    # - Relative paths: configs/model.toml (relative to YAML file)
    # - Filename only: model-config.toml (searches compiler configs directory first)
    # Settings from inline compilation_config take precedence over TOML file settings.
    Optional["timm_model_args"]: {
        Required["name"]: Str,
    },
    Optional["RetinaFace"]: {
        Optional["cfg"]: {
            Optional["min_sizes"]: List[List[Int]],
            Optional["steps"]: List[Int],
            Optional["variance"]: List[Float],
            Optional["clip"]: Bool,
        }
    },
    Optional["torchvision_args"]: {
        Optional["block"]: Str,
        Optional["layers"]: List[Int],
        Optional["alpha"]: Float,
        Optional["groups"]: Int,
        Optional["width_per_group"]: Int,
        Optional["torchvision_weights_args"]: {
            Required["object"]: Str,
            Required["name"]: Str,
        },
    },
    Optional["mmseg"]: {
        Required["config_file"]: Str,
    },
    Optional["m2"]: {
        Optional["mvm_limitation"]: Int,
    },
    Optional["aipu_cores"]: Int,
    Optional["max_compiler_cores"]: Int,
    Optional["resize_size"]: Int,
    Optional["darknet_cfg_path"]: Str,
    Optional["convert_first_node_to_1_channel"]: Bool,
}


'''Represents a model.'''
model = lambda operators, compilation_configs: {
    Optional["model_type"]: ModelType,
    # The type of model, this is used to determine how to parse the model.
    Optional["task_category"]: TaskCategory,
    # The task category of the model
    Optional["input_tensor_shape"]: List[Int],
    # The shape of the input tensor, this is usually given in flow style [1, 2, 224, 224] and
    # the shape must match tensor layout.
    Optional["input_color_format"]: ColorFormat,
    # The color format of the input tensor, this may be RGB or BGR for 3 channel input, or GREY for
    # greyscale.
    Optional["input_tensor_layout"]: TensorLayout,
    # The layout of the input tensor, this may be NCHW, NHWC or CHWN.
    Optional["labels"]: List[Str],
    # Labels used in the model.  Normally this is obtained from the dataset.
    Optional["label_filter"]: List[Str],
    # Configure the post processing to drop classes that are not in the list.
    #
    # For example a label_filter of "dog, cat" will drop all classes except dog and cat.
    Optional["weight_path"]: Str,
    # Path to the model weights, relative to the directory containing the pipeline yaml file.
    Optional["weight_url"]: Str,
    # URI of a download location for the weights, used if the weight_path does not exist or if the
    # md5 checksum does not match.
    #
    # The download will be saved to `weight_path`.
    Optional["weight_md5"]: Str,
    # MD5 of the weights, to verify if the downloaded file is up-to-date, and that the download
    # was completed without corruption.
    Optional["prequantized_url"]: Str,
    # URI of a prequantized model including binaries and weights.
    Optional["prequantized_md5"]: Str,
    # MD5 of the prequantized model, to verify the download.
    Optional["prequantized_path"]: Str,
    # Path to the model prequantized weights, relative to the directory containing the pipeline yaml file.
    Optional["precompiled_url"]: Str,
    # URI of a precompiled model including binaries and weights.
    Optional["precompiled_md5"]: Str,
    # MD5 of the precompiled model, to verify the download.
    Optional["precompiled_path"]: Str,
    # Path to the model precompiled weights, relative to the directory containing the pipeline yaml file.
    Optional["dataset"]: Str,
    # The name of the dataset to use for evaluation, if present then the dataset
    # should be defined in the datasets section of the network.
    Optional["base_dir"]: Str,
    # The base directory used when importing the model or data loader.  If not present then the
    # parent directory of `class_path` is used.
    Optional["class"]: Str,
    # The name of the model class, this is used to import the model class from the module specified
    # by `class_path`.
    # Note that the yaml key is `class`.  It is named `class_` here because class is a python keyword.
    Optional["class_path"]: Str,
    # Path to the python module containing the model class. This path should be relative to the
    # directory containing the pipeline yaml file.
    Optional["version"]: Str,
    # The version of the model, for example Squeezenet v1.0.
    #
    # This is not used by the application framework, but the pytorch model may use it to configure
    # behaviour.
    Optional["num_classes"]: Int,
    # Number of classes in the output model. For example for coco-80 this will be 80.
    Optional["extra_kwargs"]: extra_kwargs(operators, compilation_configs),
    # Extra keyword arguments to pass to the model and data loader constructor.
    #
    # This is not used by the application framework so it can be used to configure
    # the pytorch model.
}

'''Determine where the input to a model should come from.'''
input_operator = lambda operators, compilation_configs: {
    Optional["source"]: InputSource,
    # Used when the input to this model should be taken from the output of a previous model.
    #
    # For the first model in a pipeline this must be full, defaults to full.
    #
    # For subsequent models this may be roi, indicating that an roi should be extracted.
    # For example a model that performs object detection may be followed by a model that
    # performs model classification on the roi extracted by the object detection model.
    #
    # For subsequent models that follow models that produce meta data this may be 'meta',
    # for example a model lightened or upscaled image.
    Optional["where"]: Str,
    # Used with source == 'meta' or 'roi.' The name of the model to extract the roi or meta
    # information from.
    Optional["extra_info_key"]: Str,
    # Used when source == 'meta', and determines the field from the previous model's
    # meta information that should be used as the input to this model.
    Optional["min_width"]: Int,
    # Used when source == 'roi', and determines the minimum width of the roi to extract.
    Optional["min_height"]: Int,
    # Used when source == 'roi', and determines the minimum height of the roi to extract.
    # Optional["expand_margin"]: Float,
    # # Used when source == 'roi'.  When given and not zero the roi region will be expanded by
    # this amount/2 in the width, and in the height.
    Optional["top_k"]: Int,
    # Used when source == 'roi'.  How many of the object detections to use as input to the model.
    Optional["which"]: TopKRanking,
    # Used when source == 'roi'.  How to determine the top_k ranking.
    Optional["label_filter"]: Union[List[Union[Int, Str]], Int, Str],
    # Used when source == 'roi'.  Filter the object detections to only include the given classes.
    #
    # label_filter may be given as integer class ids, or as string class names.
    Optional["image_processing"]: List[operators],
    # Used when source == 'image_processing'. Used to specify the image processing operators
    Optional["image_processing_on_roi"]: List[operators],
    # Used when source == 'roi'. Used to specify the image processing operators on the roi
    Optional["color_format"]: ColorFormat,
    # The color format of the input tensor, this may be RGB or BGR for 3 channel input, or GREY for
    # greyscale.
    Optional["imreader_backend"]: ImageReaderBackend,
    # The backend library to read images. May be OPENCV or PIL
    Optional["type"]: Enum["image"],
    # We now support only 'image'.
    Optional["margin"]: Float,
    # Margin for extending ROISs
}

'''Determine the inference settings for a model.'''
inference_operator = lambda operators, compilation_configs: {
    Optional["handle_all"]: Bool,
    # Optional convenience parameter to set all 4 handle flags at once.
    # - If None (default): individual flags are used as specified
    # - If True: all 4 handle flags are set to True and the input/output will totally follow the source ONNX input/output nodes.
    #   This simplifies integration but might not be optimal for performance in all cases.
    # - If False: all 4 handle flags are set to False
    #
    # Performance considerations:
    # - For small models: Setting handle_all=False and moving operations to postprocessing may improve performance
    #   through fusion optimizations
    # - For large models: Using handle_all=True typically doesn't impact performance significantly and simplifies integration
    # Individual handle flags
    Optional["handle_dequantization_and_depadding"]: Bool,
    # Whether the C++ decoder should handle dequantization and depadding
    Optional["handle_transpose"]: Bool,
    # Whether the C++ decoder should handle transposition
    Optional["handle_postamble"]: Bool,
    # Whether the C++ decoder should handle postamble processing
    Optional["handle_preamble"]: Bool,
    # Whether the C++ decoder should handle preamble processing
    # Other configuration options
    Optional["dequantize_using_lut"]: Bool,
    # Whether to use lookup tables for dequantization
    Optional["postamble_onnxruntime_intra_op_num_threads"]: Int,
    # Number of threads for ONNX Runtime intra-op parallelism (must be >= 1)
    Optional["postamble_onnxruntime_inter_op_num_threads"]: Int,
    # Number of threads for ONNX Runtime inter-op parallelism (must be >= 1)
    Optional["postamble_onnx"]: Str,
    # Optional path to a manual-cut postamble ONNX model
}


'''
Make a custom operator available to the pipeline.

An operator is defined by a python class, and a path to the python module containing the class.

operators:
    custom-operator:
        class: CustomOperator
        class_path: custom_operator.py
'''
custom_operator = lambda operators, compilation_configs: {
    Required["class"]: Str,
    # The name of the python class that implements the operator.
    Required["class_path"]: Str,
    # Path to the python module containing the operator class. This path should be relative to the
    # directory containing the yaml file.
}

'''Information about the Axelera internal model card.'''
internal_model_card = lambda operators, compilation_configs: {
    Optional["card_name"]: Str,
    # Name of the model card.
    Optional["model_card"]: Str,
    # Name of the model card.
    Optional["model_repository"]: Str,
    # Path to the Git repository containing the model card.
    Optional["git_commit"]: Str,
    # Commit SHA for this model card in the model repository.
    Optional["model_subversion"]: Str,
    # Subversion number for this model card.
    Optional["production_ML_framework"]: Str,
    # Framework name and version used for this model card in production, e.g. PyTorch 1.13.1.
    Optional["key_metric"]: Str,
    # Key metric used to evaluate the model card.
    Optional["license"]: Str,
    # License for the model card.
    Optional["dependencies"]: List[Str],
    Optional["runtime_dependencies"]: List[Str],
}

'''A task is one stage of a pipeline. Every pipeline must have at least one task.'''
task = lambda operators, compilation_configs: {
    Optional["task_type"]: Enum["deep_learning", "classical_cv"],
    # The type of task, this is used to determine how to parse the task. Default is deep_learning.
    Optional["model_name"]: Str,
    # The name of the model to use for this task. This should be defined in the
    # models section of the network yaml file.
    Optional["input"]: Union[EmptyDict, input_operator(operators, compilation_configs)],
    # Optional for the first model in a pipeline, this determines where the input to the model
    # should come from. See InputOperator for more information.
    Optional["template_path"]: Str,
    # Path to a template file, this is used to place common preprocess and postprocessing options
    # in a central location.
    #
    # For example a template may be used to configure the preprocessing and postprocessing for
    # imagenet classification models, or for coco based object detection models.
    #
    # The schema for a template is :class:`Task`.
    #
    # To aid with making a template more generic, several variables are available for substitution:
    #
    # ============== ========================================================================
    # Variable       Description
    # ============== ========================================================================
    # num_classes    Number of classes as defined by the Dataset.num_classes.
    # input_width    Input tensor width of the model.
    # input_height   Input tensor height of the model.
    # labels         List of labels as loaded from the file specified by Dataset.labels_path.
    # label_filter   List of labels to filter, as defined by Dataset.label_filter.
    # ============== ========================================================================
    #
    # Where a task redeclares an operator, the settings in the task override those of the template.
    # For example if a template is ::
    #
    #     preprocess:
    #       - resize:
    #           size: 256
    #       - centercrop:
    #           width: {{input_width}}
    #           height: {{input_height}}
    #       - torch-totensor:
    #       - normalize:
    #           mean: 0.485, 0.456, 0.406
    #           std: 0.229, 0.224, 0.225
    #
    # If the network yaml file specifies ::
    #
    #     preprocess:
    #        - normalize:
    #            mean: 0.5
    #
    # This indicates that the normalization's mean parameter is overridden with 0.5, while all other parameters and the overall process remain unchanged. The mean used for normalization is 0.5, but the standard deviation remains at its original value. (Prior to normalization, the resize, center crop, and to-tensor operations are still applied sequentially.)
    Optional["preprocess"]: Union[EmptyList, List[operators]],
    # List of operators to apply to the input before passing it to the model.
    # Used when task_type is "deep_learning".
    #
    # Lists of operators are given as a sequence of dicts containing a single item - the name of the
    # operator ::
    #     preprocess:
    #         - resize:
    #             width: 224
    #             height: 224
    #             half_pixel_centers: True
    #         - torch-to-tensor:
    #             scale: False
    #         - normalize:
    #             mean: 127.5
    #             std: 127.5
    Optional["inference"]: Union[EmptyDict, inference_operator(operators, compilation_configs)],
    # Optional inference settings for the model. Example usage:
    #     inference:
    #       handle_all: True
    #       dequantize_using_lut: True
    Optional["postprocess"]: Union[EmptyList, List[operators]],
    # List of operators to apply to the output of the model.
    # Used when task_type is "deep_learning".
    Optional["cv_process"]: Union[EmptyList, List[operators]],
    # List of computer vision operators to apply when task_type is "classical_cv".
    # This is used instead of preprocess and postprocess for classical computer vision tasks.
    #
    # Example:
    #     cv_process:
    #       - tracker:
    #           algorithm: oc-sort
    #           bbox_task_name: detections
    #           history_length: 30
    #           algo_params:
    #             max_age: 50
    #             min_hits: 1
    #             iou_threshold: 0.3
    #             max_id: 0
    Optional["render"]: {
        Optional["show_annotations"]: Bool,
        Optional["show_labels"]: Bool,
    },
    # Settings for rendering metadata for a specific task.
    # - show_annotations: Whether to draw visual elements like bounding boxes, keypoints, etc.
    # - show_labels: Whether to draw class labels and score text.
    Optional["operators"]: MapPattern[custom_operator(operators, compilation_configs)],
    # List of custom operators to make available to the pipeline.
}

'''Configures the dataset used by a model.'''
dataset = lambda operators, compilation_configs: {
    "_type": MapCombined,
    Optional["dataset_name"]: Str,
    Optional["image_set"]: Str,
    Optional["class"]: Str,
    # The name of the python class that is used to load the dataset.
    #
    # At present it must be one of the known classes:
    #
    #    CocoObjectDetection, TorchvisionDataset
    #
    # Or an overload of the set_calibration_data_loader method of the Model can use this to return the correct dataloader
    # class.
    Optional["class_path"]: Str,
    Optional["format"]: Str,
    Optional["dataset_url"]: Str,
    Optional["dataset_md5"]: Str,
    Optional["dataset_drop_dirs"]: Int,
    Optional["data_dir_name"]: Str,
    Optional["ultralytics_data_yaml"]: Str,
    # Directory in which to store and load the dataset, this is prefixed with `--data-root`
    # (which defaults to ./data).
    Optional["cal_data_url"]: Str,
    Optional["cal_data"]: Str,
    Optional["val_data"]: Str,
    Optional["val_data_url"]: Str,
    Optional["val_data_md5"]: Str,
    Optional["val_drop_dirs"]: Int,
    Optional["repr_imgs_dir_path"]: Str,
    Optional["repr_imgs_dir_name"]: Str,
    Optional["repr_imgs_url"]: Str,
    Optional["repr_imgs_md5"]: Str,
    Optional["val"]: Str,
    # The name of the validation dataset, this is used to configure the model for evaluation.
    Optional["test"]: Str,
    # The name of the test dataset, this is used to configure the model for evaluation.
    Optional["labels_path"]: Str,
    # The path to the labels file, these should be one label per line, and must match
    # num_classes of the model.
    Optional["split"]: Str,
    # The default dataset to load for evaluation, this defaults to val.
    Optional["label_type"]: SupportedLabelType,
}

pipeline_asset = lambda operators, compilation_configs: {
    Required["md5"]: Str,
    Required["path"]: Str,
}


'''This is the top level of the network yaml file.'''
network = lambda operators, compilation_configs: {
    Optional["axelera-model-format"]: Str,
    # Version of the model format used for the network yaml
    Optional["name"]: Str,
    # Name of the model, this is used to populate models.yaml.
    Optional["description"]: Str,
    # Description of the model, this is used to populate models.yaml.
    Optional["internal-model-card"]: internal_model_card(operators, compilation_configs),
    Optional["model-env"]: internal_model_card(operators, compilation_configs),
    Optional["pipeline"]: Union[
        EmptyList, Null, List[MapPattern[task(operators, compilation_configs)]]
    ],
    # List of tasks in the network, and how they are connected.
    #
    # Each task in the pipeline is a dictionary containing a single item:
    #
    # pipeline:
    #    - model1:
    #       preprocess: ...
    Optional["operators"]: MapPattern[custom_operator(operators, compilation_configs)],
    # List of custom operators to make available to the pipeline.
    Optional["models"]: MapPattern[model(operators, compilation_configs)],
    # Mapping of models used in the pipeline.
    Optional["datasets"]: MapPattern[dataset(operators, compilation_configs)],
    # Mapping of datasets used by the models in the pipeline.
    Optional["pipeline_assets"]: MapPattern[pipeline_asset(operators, compilation_configs)],
}


def load(schema, path=None, check_required=True, load_compiler_config=False):
    path = Path(path) if path else None
    base = path.parent if path else None
    operators = generate_operators(path, base)
    compilation_configs = generate_compilation_configs(load_compiler_config)
    return compile_schema(schema(operators, compilation_configs), check_required)


@functools.lru_cache(maxsize=32)
def load_network(path=None, check_required=True, load_compiler_config=False):
    return load(network, path, check_required, load_compiler_config)


@functools.lru_cache(maxsize=32)
def load_task(path=None, check_required=True, load_compiler_config=False):
    return load(task, path, check_required, load_compiler_config)
