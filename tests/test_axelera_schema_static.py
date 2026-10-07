# Copyright Axelera AI, 2024

from collections import defaultdict
import re
import tempfile

import pytest
import strictyaml as sy

from axelera.app.schema import compile_schema, generate_operators, static
from axelera.app.schema.types import Any, MapPattern

_seq = r'(sequence|commaseparated)'

# StrictYAML error validators
str_not_map = lambda x: rf"when expecting a mapping(?s:.)*?found arbitrary text(?s:.)*?{x}"

seq_not_map = lambda x: rf"when expecting a mapping(?s:.)*?({x})?(?s:.)*?found a {_seq}({x})?"
seq_not_str = lambda x: rf"when expecting a str(?s:.)*?{x}(?s:.)*?found a {_seq}"

map_not_seq = lambda x: rf"when expecting a {_seq}(?s:.)*?({x})?(?s:.)*?found a mapping({x})?"
seq_not_int = lambda x: rf"when expecting a int(?s:.)*?{x}(?s:.)*?found a {_seq}"

bnk_not_map = lambda x: rf"when expecting a mapping(?s:.)*?found a blank string(?s:.)*?{x}"

unexpected_key = lambda x: rf"unexpected key not in schema '{x}'"
missing_key = lambda x: rf"while parsing a mapping(?s:.)*?'{x}' not found"


not_enumerated = lambda x: rf"when expecting one of(?s:.)*?found arbitrary text(?s:.)*?{x}"
case_insensitive_enum_error = lambda values, invalid: re.compile(
    not_enumerated(invalid).replace("arbitrary text", ".*")
)

NETWORK_YAML = """\
axelera-model-format: 1.0.0

name: schema-test-network

description: schema network for testing

pipeline:
    - task1:
        model_name: model1
        template_path: $AXELERA_FRAMEWORK/test_templates/template.py
        postprocess:
            - decodeyolo:
                max_nms_boxes: 30000
                conf_threshold: 0.3
                nms_iou_threshold: 0.5
                nms_class_agnostic: False
                nms_top_k: 300
                use_multi_label: False
            - tracker:
                algorithm: oc-sort
                history_length: 30
                algo_params:
                    max_age: 50
                    min_hits: 1
                    iou_threshold: 0.25
                    max_id: 0

models:
    model1:
        class: AxOnnxObjDetection
        class_path: $AXELERA_FRAMEWORK/test_models/model.py
        weight_path: test_weights/model_weights.onnx
        weight_url: https://test.com/model_weights.onnx
        weight_md5: a123456789
        task_category: ObjectDetection
        input_tensor_layout: NCHW
        input_tensor_shape: [1, 3, 640, 640]
        input_color_format: RGB
        num_classes: 80
        dataset: dataset1
        extra_kwargs:
            YOLO:
                anchors: # or, specified explicitly
                - [10,13, 16,30, 33,23]
                - [30,61, 62,45, 59,119]
                - [116,90, 156,198, 373,326]
                strides: [8, 16, 32]

datasets:
    dataset1:
        class: DatasetClass
        class_path: $AXELERA_FRAMEWORK/test_datasets/dataset.py
        data_dir_name: test_dir
        labels_path: $AXELERA_FRAMEWORK/test_labels/labels/test.names
        label_type: coco2017
        repr_imgs_dir_name: test_images/representative
        repr_imgs_url: http://test.com/representative_images
        repr_imgs_md5: b123456789
"""


def _get_ids(cases):
    counts = defaultdict(int)
    ids = []
    for case in cases:
        ids.append(f"{case[0]}_{counts[case[0]]}")
        counts[case[0]] += 1
    return ids


def _get_compiled(name, check_required):
    definitions = (
        getattr(static, name)(get_test_operators(), get_test_compilation_config())
        if name != 'operator'
        else get_test_operators()
    )
    compiled = compile_schema(definitions, check_required)
    if name == 'network':
        return compiled
    return sy.MapPattern(sy.Str(), compiled)


def get_test_operators():
    network_yaml = '''\
    operators:
      decodeyolo:
        class: DecodeYolo
        class_path: $AXELERA_FRAMEWORK/ax_models/decoders/yolo.py
    '''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as temp_file:
        temp_file.write(network_yaml)
        temp_file.seek(0)
        return generate_operators(temp_file.name, ".")


def get_test_compilation_config():
    return MapPattern[Any]


def yaml_load(in_yaml, compiled):
    return sy.dirty_load(in_yaml, compiled, allow_flow_style=True).data


non_conformant_cases = [
    (
        "task",
        """\
        FaceDetection: hello
        """,
        str_not_map("FaceDetection"),
        True,
    ),
    (
        "task",
        """\
        FaceDetection:
            - template_path:
        """,
        seq_not_map("FaceDetection"),
        True,
    ),
    (
        "task",
        """\
        FaceDetection:
            template_path:
                - templates/face_detection.yaml
        """,
        seq_not_str("template_path"),
        True,
    ),
    (
        "task",
        """
        FaceDetection:
            unknown_key: templates/face_detection.yaml
        """,
        unexpected_key("unknown_key"),
        True,
    ),
    (
        "task",
        """\
        FaceDetection:
            input:
                - type: image
        """,
        seq_not_map("input"),
        True,
    ),
    (
        "task",
        """\
        FaceDetection:
            preprocess:
                torchtotensor:
        """,
        map_not_seq("preprocess"),
        True,
    ),
    (
        "task",
        """\
        FaceDetection:
            postprocess:
                - torchtotensor:
                    -  positional
        """,
        seq_not_map("positional"),
        True,
    ),
    (
        "task",
        """\
        Tracking:
            cv_process:
                tracker:
        """,
        map_not_seq("cv_process"),
        True,
    ),
    (
        "task",
        """\
        FaceDetection:
        """,
        bnk_not_map("FaceDetection"),
        True,
    ),
    (
        "extra_kwargs",
        """\
        extra_kwargs:
            - YOLO:
            - mmseg:
            - aipu_cores: 4
        """,
        seq_not_map("extra_kwargs"),
        True,
    ),
    (
        "extra_kwargs",
        """\
        extra_kwargs:
            YOLO:
                - anchors: [[1,2,3,4], [5,6,7,8]]
        """,
        seq_not_map("YOLO"),
        True,
    ),
    (
        "extra_kwargs",
        """\
        extra_kwargs:
            mmseg: test_config/config.py
        """,
        str_not_map("mmseg"),
        True,
    ),
    (
        "extra_kwargs",
        """\
        extra_kwargs:
            aipu_cores:
                - 4
        """,
        seq_not_int("aipu_cores"),
        True,
    ),
    (
        "model",
        """\
        test_model:
            task_category: ImageSegmentation
            input_tensor_shape: [1, 3, 512, 1024]
            input_color_format: BGR
            input_tensor_layout: NCHW
            labels: cat, dog, mouse
        """,
        not_enumerated("ImageSegmentation"),
        True,
    ),
    (
        "model",
        """\
        test_model:
            task_category: ObjectDetection
            input_tensor_shape: [1, 3, 512, 1024]
            input_color_format: YMCA
            input_tensor_layout: NCHW
            labels: cat, dog, mouse
        """,
        case_insensitive_enum_error(["RGB", "BGR", "RGBA", "BGRA", "GRAY"], "YMCA"),
        True,
    ),
    (
        "model",
        """\
        test_model:
            task_category: ObjectDetection
            input_tensor_shape: [1, 3, 512, 1024]
            input_color_format: BGR
            input_tensor_layout: WHD
            labels: cat, dog, mouse
        """,
        not_enumerated("WHD"),
        True,
    ),
    (
        "model",
        """\
        test_model:
            task_category: ObjectDetection
            input_tensor_shape: [1, 3, 512, 1024]
            input_color_format: BGR
            input_tensor_layout: NCHW
            output_tensor_layout: NCHW
            labels: cat, dog, mouse
        """,
        unexpected_key("output_tensor_layout"),
        True,
    ),
    (
        "operator",  # FAIL
        """\
        test_operators:
            decodeyolo:
                max_nms_boxes: 30000
                conf_threshold: 0.3
                nms_iou_threshold: 0.5
                nms_class_agnostic: False
                nms_top_k: 300
                use_multi_label: False
                eval:
                    - conf_threshold: 0.1
                    - nms_iou_threshold: 0.2
        """,
        seq_not_map("conf_threshold"),
        True,
    ),
    (
        "operator",
        """\
        test_operators:
            - decodeyolo:
                max_nms_boxes: 30000
                conf_threshold: 0.3
                nms_iou_threshold: 0.5
                nms_class_agnostic: False
                nms_top_k: 300
                use_multi_label: False
                eval:
                    conf_threshold: 0.1
                    nms_iou_threshold: 0.2
        """,
        seq_not_map("decodeyolo"),
        True,
    ),
    (
        "input_operator",
        """\
        input:
            - source: meta
            - where: my_input_model
            - extra_info_key: other_key
            - color_format: RGB
            - imreader_backend: OpenCV
        """,
        seq_not_map("input"),
        True,
    ),
    (
        "input_operator",
        """\
        input:
            source: half
            where: my_input_model
            extra_info_key: other_key
            color_format: RGB
            imreader_backend: OpenCV
        """,
        not_enumerated("half"),
        True,
    ),
    (
        "input_operator",
        """\
        input:
            source: full
            where: my_input_model
            extra_info_key: other_key
            color_format: RGB
            imreader_backend: OpenGL
        """,
        case_insensitive_enum_error(["PIL", "OPENCV"], "OpenGL"),
        True,
    ),
    (
        "input_operator",
        """\
        input:
            source: meta
            where: my_input_model
            extra_info_key: other_key
            color_format: RGB
            imreader_backend: OpenCV
            type: txt
        """,
        not_enumerated("txt"),
        True,
    ),
    (
        "input_operator",
        """\
        input:
            source: meta
            where: my_input_model
            extra_info_key: other_key
            color_format: RGB
            imreader_backend: OpenCV
            image_processing:
                perspective:
                decodeyolo:
            type: image
        """,
        map_not_seq("image_processing"),
        True,
    ),
    (
        "custom_operator",
        """\
        my_custom_operator:
            class: MyCustomOperator
        """,
        missing_key("class_path"),
        True,
    ),
    (
        "custom_operator",
        """\
        my_custom_operator:
            class_path: test_ops/my_custom_operator.py
        """,
        missing_key("class"),
        True,
    ),
    (
        "custom_operator",
        """\
        my_custom_operator:
            class: MyCustomOperator
            class_path: test_ops/my_custom_operator.py
            class_sub_key: sub_key
        """,
        unexpected_key("class_sub_key"),
        True,
    ),
    (
        "internal_model_card",
        """\
        internal-model-card:
            model_card: MC10
            model_repository: https://test.com/model_repo
            git_commit: 1234567890
            model_subversion: 1.0
            production_ML_framework: PyTorch 1.13.1
            key_metric: metric
            dependencies:
                dependency1:
                dependency2:
                dependency3:
        """,
        map_not_seq("dependencies"),
        True,
    ),
    (
        "pipeline_asset",
        """\
        pipeline_asset:
            md5: a1234567890
        """,
        missing_key("path"),
        True,
    ),
    (
        "pipeline_asset",
        """\
        pipeline_asset:
            path: test_assets/test_asset.py
        """,
        missing_key("md5"),
        True,
    ),
    (
        "pipeline_asset",
        """\
        pipeline_asset:
            md5: a1234567890
            path: test_assets/test_asset.py
            git_commit: 1234567890
        """,
        unexpected_key("git_commit"),
        True,
    ),
    (
        "network",
        f"{NETWORK_YAML}\nextra_key: extra_value",
        unexpected_key("extra_key"),
        False,
    ),
    (
        "network",
        NETWORK_YAML.replace("    model1:", "    - model1:"),
        seq_not_map("models"),
        False,
    ),
    (
        "network",
        NETWORK_YAML.replace("    - task1:", "    task1:"),
        map_not_seq("pipeline"),
        False,
    ),
    (
        "network",
        NETWORK_YAML.replace("    dataset1:", "    - dataset1:"),
        seq_not_map("datasets"),
        False,
    ),
]


@pytest.mark.parametrize(
    "name, input, error, check_required",
    non_conformant_cases,
    ids=_get_ids(non_conformant_cases),
)
def test_non_conformant_yaml(name, input, error, check_required):
    compiled = _get_compiled(name, check_required)
    with pytest.raises(sy.exceptions.YAMLValidationError, match=error):
        yaml_load(input, compiled)


conformant_cases = [
    (
        "task",
        False,
        """\
        test_task:
            model_name: test_model
            input:
            template_path: test_templates/template.yaml
            preprocess:
            postprocess:
            operators:
                test_operator:
                    class: TestOperator
                    class_path: test_operators/operator.py
        """,
        {
            "test_task": {
                "model_name": "test_model",
                "input": {},
                "template_path": "test_templates/template.yaml",
                "preprocess": [],
                "postprocess": [],
                "operators": {
                    "test_operator": {
                        "class": "TestOperator",
                        "class_path": "test_operators/operator.py",
                    }
                },
            }
        },
    ),
    (
        "task",
        False,
        """\
        test_task:
            model_name: test_model
            input:
                source: meta
                where: test_model
            template_path: test_templates/template.yaml
            preprocess:
                - torchtotensor:
                - resize:
                    width: 224
                    height: 224
            postprocess:
                - resize:
                    width: 224
                    height: 224
            operators:
                test_operator:
                    class: TestOperator
                    class_path: test_operators/operator.py
        """,
        {
            "test_task": {
                "model_name": "test_model",
                "input": {"source": "meta", "where": "test_model"},
                "template_path": "test_templates/template.yaml",
                "preprocess": [
                    {"torchtotensor": {}},
                    {"resize": {"width": 224, "height": 224}},
                ],
                "postprocess": [{"resize": {"width": 224, "height": 224}}],
                "operators": {
                    "test_operator": {
                        "class": "TestOperator",
                        "class_path": "test_operators/operator.py",
                    }
                },
            }
        },
    ),
    (
        "extra_kwargs",
        True,
        """\
        extra_kwargs:
            YOLO:
                anchors:
                - [1,2,3,4]
                - [5,6,7,8]
                anchors_path: test_anchors/anchors.yaml
                anchors_url: https://test.com/anchors.yaml
                anchors_md5: a1234567890
                strides: [8, 16, 32]
                focus_layer_replacement: True
            mmseg:
                config_file: test_config/config.py
            aipu_cores: 4
        """,
        {
            "extra_kwargs": {
                "YOLO": {
                    "anchors": [[1, 2, 3, 4], [5, 6, 7, 8]],
                    "anchors_path": "test_anchors/anchors.yaml",
                    "anchors_url": "https://test.com/anchors.yaml",
                    "anchors_md5": "a1234567890",
                    "strides": [8, 16, 32],
                    "focus_layer_replacement": True,
                },
                "mmseg": {"config_file": "test_config/config.py"},
                "aipu_cores": 4,
            }
        },
    ),
    (
        "model",
        True,
        """\
        test_model:
            task_category: ObjectDetection
            input_tensor_shape: [1, 3, 512, 1024]
            input_color_format: BGR
            input_tensor_layout: NCHW
            labels: cat, dog, mouse
            label_filter: [cat, dog]
            weight_path: test_weights/weights
            weight_url: https://test.com/weights
            weight_md5: a1234567890
            prequantized_url: https://test.com/prequantized
            prequantized_md5: b1234567890
            dataset: COCO-Test
            base_dir: test_base_dir
            class: TestModel
            class_path: test_models/model.py
            version: 1.0
            num_classes: 3
            extra_kwargs:
                YOLO:
                    anchors:
                    - [1,2,3,4]
                    - [5,6,7,8]
                    anchors_path: test_anchors/anchors.yaml
                    anchors_url: https://test.com/anchors.yaml
                    anchors_md5: a1234567890
                    strides: [8, 16, 32]
                    focus_layer_replacement: True
        """,
        {
            "test_model": {
                "task_category": "ObjectDetection",
                "input_tensor_shape": [1, 3, 512, 1024],
                "input_color_format": "BGR",
                "input_tensor_layout": "NCHW",
                "labels": ["cat", "dog", "mouse"],
                "label_filter": ["cat", "dog"],
                "weight_path": "test_weights/weights",
                "weight_url": "https://test.com/weights",
                "weight_md5": "a1234567890",
                "prequantized_url": "https://test.com/prequantized",
                "prequantized_md5": "b1234567890",
                "dataset": "COCO-Test",
                "base_dir": "test_base_dir",
                "class": "TestModel",
                "class_path": "test_models/model.py",
                "version": "1.0",
                "num_classes": 3,
                "extra_kwargs": {
                    "YOLO": {
                        "anchors": [[1, 2, 3, 4], [5, 6, 7, 8]],
                        "anchors_path": "test_anchors/anchors.yaml",
                        "anchors_url": "https://test.com/anchors.yaml",
                        "anchors_md5": "a1234567890",
                        "strides": [8, 16, 32],
                        "focus_layer_replacement": True,
                    }
                },
            }
        },
    ),
    (
        "operator",  # FAIL
        True,
        '''\
        known_operator:
            letterbox:
                width: 240
                height: ${{height}}
                scaleup: True
                image_width: 240
                image_height: 360
                pad_val: 5
                half_pixel_centers: False
            normalize:
        ''',
        {
            "known_operator": {
                "letterbox": {
                    "width": 240,
                    "height": "${{height}}",
                    "scaleup": True,
                    "image_width": 240,
                    "image_height": 360,
                    "pad_val": 5,
                    "half_pixel_centers": False,
                },
                "normalize": {},
            }
        },
    ),
    (
        "input_operator",
        True,
        '''\
        input:
            source: meta
            where: my_input_model
            extra_info_key: other_key
            color_format: RGB
            imreader_backend: OpenCV
        ''',
        {
            "input": {
                "source": "meta",
                "where": "my_input_model",
                "extra_info_key": "other_key",
                "color_format": "RGB",
                "imreader_backend": "OPENCV",
            }
        },
    ),
    (
        "input_operator",
        True,
        '''\
        input:
            source: meta
            where: my_input_model
            extra_info_key: other_key
            color_format: rgb
            imreader_backend: pil
        ''',
        {
            "input": {
                "source": "meta",
                "where": "my_input_model",
                "extra_info_key": "other_key",
                "color_format": "RGB",
                "imreader_backend": "PIL",
            }
        },
    ),
    (
        "input_operator",
        True,
        '''\
        input:
            source: roi
            where: my_input_model
            min_width: 10
            min_height: 10
            top_k: 5
            which: SCORE
            label_filter: [cat, dog, mouse]
            color_format: RGB
            imreader_backend: OpenCV
        ''',
        {
            "input": {
                "source": "roi",
                "where": "my_input_model",
                "min_width": 10,
                "min_height": 10,
                "top_k": 5,
                "which": "SCORE",
                "label_filter": ["cat", "dog", "mouse"],
                "color_format": "RGB",
                "imreader_backend": "OPENCV",
            }
        },
    ),
    (
        "input_operator",  # FAIL
        True,
        '''\
        input:
            source: image_processing
            image_processing:
                - perspective:
                    camera_matrix: 1.019,-0.697,412.602,0.918,1.361,-610.083,0.0,0.0,1.0
            color_format: RGB
            imreader_backend: OpenCV
        ''',
        {
            "input": {
                "source": "image_processing",
                "image_processing": [
                    {
                        "perspective": {
                            "camera_matrix": [
                                1.019,
                                -0.697,
                                412.602,
                                0.918,
                                1.361,
                                -610.083,
                                0.0,
                                0.0,
                                1.0,
                            ]
                        }
                    }
                ],
                "color_format": "RGB",
                "imreader_backend": "OPENCV",
            }
        },
    ),
    (
        "custom_operator",
        True,
        '''\
        my_custom_operator:
            class: MyCustomOperator
            class_path: my_custom_operator.py
        ''',
        {
            "my_custom_operator": {
                "class": "MyCustomOperator",
                "class_path": "my_custom_operator.py",
            }
        },
    ),
    (
        "internal_model_card",
        True,
        '''\
        internal-model-card:
            model_card: MC10
            model_repository: https://test.com/model_repo
            git_commit: 1234567890
        ''',
        {
            "internal-model-card": {
                "model_card": "MC10",
                "model_repository": "https://test.com/model_repo",
                "git_commit": "1234567890",
            }
        },
    ),
    (
        "internal_model_card",
        True,
        '''\
        internal-model-card:
            model_card: MC10
            model_repository: https://test.com/model_repo
            git_commit: 1234567890
            model_subversion: 1.0
            production_ML_framework: PyTorch 1.13.1
            key_metric: metric
            dependencies:
                - dependency1
                - dependency2
                - dependency3
        ''',
        {
            "internal-model-card": {
                "model_card": "MC10",
                "model_repository": "https://test.com/model_repo",
                "git_commit": "1234567890",
                "model_subversion": "1.0",
                "production_ML_framework": "PyTorch 1.13.1",
                "key_metric": "metric",
                "dependencies": ["dependency1", "dependency2", "dependency3"],
            }
        },
    ),
    (
        "pipeline_asset",
        True,
        '''\
        test_asset:
            md5: a1234567890
            path: test_assets/test_asset.py
        ''',
        {
            "test_asset": {
                "md5": "a1234567890",
                "path": "test_assets/test_asset.py",
            }
        },
    ),
    (
        "dataset",
        True,
        '''\
        test_dataset:
            class: COCOTest
            data_dir_name: test/test_data
        ''',
        {
            "test_dataset": {
                "class": "COCOTest",
                "data_dir_name": "test/test_data",
            }
        },
    ),
    (
        "dataset",
        True,
        '''\
        test_dataset:
            class: COCOTest
            data_dir_name: test/test_data
            val: validation_coco_test
            test: test_coco_test
            labels_path: test_labels/labels.txt
            split: eval_dataset
        ''',
        {
            "test_dataset": {
                "class": "COCOTest",
                "data_dir_name": "test/test_data",
                "val": "validation_coco_test",
                "test": "test_coco_test",
                "labels_path": "test_labels/labels.txt",
                "split": "eval_dataset",
            }
        },
    ),
    (
        "dataset",
        True,
        '''\
        test_dataset:
            class: COCOTest
            data_dir_name: test/test_data
            val: validation_coco_test
            test: test_coco_test
            labels_path: test_labels/labels.txt
            split: eval_dataset
            another_random_field: False
        ''',
        {
            "test_dataset": {
                "class": "COCOTest",
                "data_dir_name": "test/test_data",
                "val": "validation_coco_test",
                "test": "test_coco_test",
                "labels_path": "test_labels/labels.txt",
                "another_random_field": "False",
                "split": "eval_dataset",
            }
        },
    ),
    (
        "network",
        False,
        NETWORK_YAML,
        {
            "axelera-model-format": "1.0.0",
            "name": "schema-test-network",
            "description": "schema network for testing",
            "pipeline": [
                {
                    "task1": {
                        "model_name": "model1",
                        "template_path": "$AXELERA_FRAMEWORK/test_templates/template.py",
                        "postprocess": [
                            {
                                "decodeyolo": {
                                    "max_nms_boxes": 30000,
                                    "conf_threshold": 0.3,
                                    "nms_iou_threshold": 0.5,
                                    "nms_class_agnostic": False,
                                    "nms_top_k": 300,
                                    "use_multi_label": False,
                                }
                            },
                            {
                                "tracker": {
                                    "algorithm": "oc-sort",
                                    "history_length": 30,
                                    "algo_params": {
                                        "max_age": 50,
                                        "min_hits": 1,
                                        "iou_threshold": 0.25,
                                        "max_id": 0,
                                    },
                                }
                            },
                        ],
                    }
                }
            ],
            "models": {
                "model1": {
                    "class": "AxOnnxObjDetection",
                    "class_path": "$AXELERA_FRAMEWORK/test_models/model.py",
                    "weight_path": "test_weights/model_weights.onnx",
                    "weight_url": "https://test.com/model_weights.onnx",
                    "weight_md5": "a123456789",
                    "task_category": "ObjectDetection",
                    "input_tensor_layout": "NCHW",
                    "input_tensor_shape": [1, 3, 640, 640],
                    "input_color_format": "RGB",
                    "num_classes": 80,
                    "dataset": "dataset1",
                    "extra_kwargs": {
                        "YOLO": {
                            "anchors": [
                                [10, 13, 16, 30, 33, 23],
                                [30, 61, 62, 45, 59, 119],
                                [116, 90, 156, 198, 373, 326],
                            ],
                            "strides": [8, 16, 32],
                        }
                    },
                }
            },
            "datasets": {
                "dataset1": {
                    "class": "DatasetClass",
                    "class_path": "$AXELERA_FRAMEWORK/test_datasets/dataset.py",
                    "data_dir_name": "test_dir",
                    "labels_path": "$AXELERA_FRAMEWORK/test_labels/labels/test.names",
                    "label_type": "COCO2017",
                    "repr_imgs_dir_name": "test_images/representative",
                    "repr_imgs_url": "http://test.com/representative_images",
                    "repr_imgs_md5": "b123456789",
                }
            },
        },
    ),
]


@pytest.mark.parametrize(
    "name, check_required, input, expected",
    conformant_cases,
    ids=_get_ids(conformant_cases),
)
def test_conformant_yaml(name, check_required, input, expected):
    compiled = _get_compiled(name, check_required)
    assert yaml_load(input, compiled) == expected


def test_label_type_variations():
    """Test that different label type formats are correctly validated."""

    # Test different label type formats
    test_cases = [
        # Original format in your test
        "coco2017",
        # Other formats and variations
        "COCO2017",
        "Coco2017",  # Mixed case
        "COCO2014",
        "coco2014",
        # JSON formats with different casings and separators
        "COCO JSON",
        "coco json",
        "COCO_JSON",
        "coco_json",
        "COCOJSON",
        "cocojson",
        # YOLO formats
        "YOLOv8",
        "yolov8",
        "YOLOv5",
        "yolov5",
        # Pascal VOC formats
        "PascalVOCXML",
        "Pascal VOC XML",
        "pascal_voc_xml",
    ]

    for label_type in test_cases:
        yaml_str = f"""
datasets:
    dataset1:
        class: DatasetClass
        class_path: $AXELERA_FRAMEWORK/test_datasets/dataset.py
        data_dir_name: test_dir
        labels_path: $AXELERA_FRAMEWORK/test_labels/labels/test.names
        label_type: {label_type}
        repr_imgs_dir_name: test_images/representative
        repr_imgs_url: http://test.com/representative_images
        repr_imgs_md5: b123456789
"""
        try:
            # This should not raise an exception if validation works correctly
            compiled = _get_compiled("network", False)
            result = yaml_load(yaml_str, compiled)
            # Verify the dataset was parsed correctly in the nested structure
            assert "datasets" in result
            assert "dataset1" in result["datasets"]
            assert "label_type" in result["datasets"]["dataset1"]
            # Print for debugging if needed
            # print(f"'{label_type}' validated as: {result['datasets']['dataset1']['label_type']}")
        except Exception as e:
            pytest.fail(f"Validation failed for '{label_type}': {str(e)}")


def test_invalid_label_types():
    """Test that invalid label types are rejected."""

    invalid_types = [
        "unknown_type",
        "coco2018",  # non-existent version
        "YOLOv9",  # non-existent version
        "JSON",  # incomplete name
        "COCOv2",  # invalid format
        "RandomFormat",  # completely invalid
    ]

    for label_type in invalid_types:
        yaml_str = f"""
datasets:
    dataset1:
        class: DatasetClass
        class_path: $AXELERA_FRAMEWORK/test_datasets/dataset.py
        data_dir_name: test_dir
        labels_path: $AXELERA_FRAMEWORK/test_labels/labels/test.names
        label_type: {label_type}
        repr_imgs_dir_name: test_images/representative
        repr_imgs_url: http://test.com/representative_images
        repr_imgs_md5: b123456789
"""
        compiled = _get_compiled("network", False)
        with pytest.raises(sy.exceptions.YAMLValidationError):
            yaml_load(yaml_str, compiled)
