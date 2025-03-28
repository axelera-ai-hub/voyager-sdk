# Copyright Axelera AI, 2024
import pathlib
import textwrap
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

import numpy as np
import yaml

from axelera import types
from axelera.app import gst_builder, network, operators, pipeline, utils
from axelera.app.operators import EvalMode
from axelera.app.operators.custom_preprocessing import ConvertColorInput

FACE_DETECTION_MODEL_INFO = types.ModelInfo('FaceDetection', 'ObjectDetection', [3, 640, 480])
FACE_RECOGNITION_MODEL_INFO = types.ModelInfo('FaceRecognition', 'Classification', [3, 160, 160])


def make_model_infos(the_model_info):
    model_infos = network.ModelInfos()
    model_infos.add_model(the_model_info, pathlib.Path('/path'))
    return model_infos


@pytest.mark.parametrize(
    'in_yaml',
    [
        """\
FaceDetection:
""",
        """\
FaceDetection: {}
""",
    ],
)
def test_parse_task_empty(in_yaml):
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with pytest.raises(ValueError, match='No pipeline config for FaceDetection'):
        pipeline.parse_task(in_dict, {}, model_infos)


def test_parse_task_default_input_type():
    in_yaml = """\
FaceDetection:
  input:
  preprocess:
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.input == operators.Input()


@pytest.mark.parametrize(
    "in_yaml, expected_exception, expected_message, expected_type, expected_processing",
    [
        (
            """\
FaceDetection:
  input:
    source: image_processing
    type: image
  preprocess:
""",
            AssertionError,
            "Please specify the image processing operator",
            None,
            None,
        ),
        (
            """\
FaceDetection:
  input:
    source: image_processing
    type: image
    image_processing:
      - resize:
          width: 640
          height: 480
  preprocess:
""",
            None,
            None,
            operators.InputWithImageProcessing,
            [operators.Resize(width=640, height=480)],
        ),
        (
            """\
FaceDetection:
  input:
    source: image_processing
    type: image
    image_processing:
      - normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
  preprocess:
""",
            ValueError,
            "Unsupported image type: <class 'torch.Tensor'>",
            None,
            None,
        ),
    ],
)
def test_parse_task_input_source_image_processing(
    in_yaml, expected_exception, expected_message, expected_type, expected_processing
):
    import torch

    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    if expected_exception:
        with pytest.raises(expected_exception, match=expected_message):
            mp = pipeline.parse_task(in_dict, {}, model_infos)
            if expected_type:
                assert isinstance(mp.input, expected_type)
                assert mp.input.image_processing == expected_processing
            if expected_exception is ValueError:
                img = torch.randn(3, 640, 480)
                mp.input.exec_torch(img, None, None)
    else:
        mp = pipeline.parse_task(in_dict, {}, model_infos)
        assert isinstance(mp.input, expected_type)
        assert mp.input.image_processing == expected_processing


def test_parse_task_from_full():
    in_yaml = """\
FaceDetection:
  input:
    source: full
  preprocess:
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.input == operators.Input()


def test_parse_task_custom_operator():
    in_yaml = """\
FaceDetection:
  input:
  preprocess:
  - myop:
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)

    class MyOp(operators.AxOperator):
        def exec_torch(self, img, result, meta):
            return img, result, meta

        def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
            pass

    ops = dict(myop=MyOp)
    mp = pipeline.parse_task(in_dict, ops, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.input == operators.Input()
    assert mp.preprocess == [MyOp()]
    assert mp.postprocess == []


def test_parse_task():
    in_yaml = """\
FaceDetection:
  input:
  preprocess:
  - resize:
      width: 640
      height: 480
  - convert-color:
      format: RGB2BGR
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.input == operators.Input()
    assert mp.preprocess == [
        operators.Resize(width=640, height=480),
        operators.ConvertColor(format='RGB2BGR'),
    ]
    assert mp.postprocess == []


def test_parse_task_with_template_preprocess():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  input:
    type: image
  postprocess:
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(utils, 'load_yaml_by_reference') as mock_template:
        mock_template.return_value = dict(
            input=dict(type='image'), preprocess=[dict(resize=dict(width=1024, height=768))]
        )
        mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.input == operators.Input()
    assert mp.preprocess == [
        operators.Resize(width=1024, height=768),
    ]
    assert mp.postprocess == []


def test_parse_task_with_template_preprocess_overridden_in_yaml():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  preprocess:
  - resize:
      width: 1280
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(utils, 'load_yaml_by_reference') as mock_template:
        mock_template.return_value = dict(
            input=dict(type='image'), preprocess=[dict(resize=dict(width=1024, height=768))]
        )
        mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.input == operators.Input()
    assert mp.preprocess == [
        operators.Resize(width=1280, height=768),
    ]
    assert mp.postprocess == []


def test_parse_task_with_template_preprocess_with_extra_operator_in_yaml():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  preprocess:
  - resize:
  - torch-totensor:
  - newop:
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(utils, 'load_yaml_by_reference') as mock_template:
        mock_template.return_value = dict(
            input=dict(type='image'), preprocess=[dict(resize=dict(width=1024, height=768))]
        )
        with pytest.raises(AssertionError):
            mp = pipeline.parse_task(in_dict, {}, model_infos)


def test_parse_task_with_template_postprocess_with_extra_operator_before_template_operators_in_yaml():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  postprocess:
    - tracker:
    - topk:

"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(utils, 'load_yaml_by_reference') as mock_template:
        mock_template.return_value = dict(
            input=dict(type='image'), postprocess=[dict(topk=dict())]
        )
        with pytest.raises(AssertionError):
            mp = pipeline.parse_task(in_dict, {}, model_infos)


def test_parse_task_with_template_postprocess_with_extra_operator_after_template_operators_in_yaml():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  postprocess:
    - topk: # simple treat topk as a decoder not a real case
    - tracker:

"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(utils, 'load_yaml_by_reference') as mock_template:
        mock_template.return_value = dict(
            input=dict(type='image'), postprocess=[dict(topk=dict())]
        )
        mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.input == operators.Input()
    assert mp.preprocess == []
    assert mp.postprocess == [
        operators.postprocessing.TopK(),
        operators.postprocessing.Tracker(),
    ]


def test_parse_task_with_template_postprocess_with_extra_operator_in_yaml():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  postprocess:
    - tracker:

"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(utils, 'load_yaml_by_reference') as mock_template:
        mock_template.return_value = dict(
            input=dict(type='image'), postprocess=[dict(topk=dict())]
        )
        mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.input == operators.Input()
    assert mp.preprocess == []
    assert mp.postprocess == [
        operators.postprocessing.TopK(),
        operators.postprocessing.Tracker(),
    ]


@pytest.fixture
def mock_template():
    with patch.object(utils, 'load_yaml_by_reference') as mock_template:
        yield mock_template


@pytest.mark.parametrize(
    "in_yaml, expected_input",
    [
        (
            """\
FaceDetection:
    template_path: templates/face_detection.yaml
    input:
        type: image
        source: roi
        where: ObjectDetection
        which: CENTER
        top_k: 5
        expand_margin: 0
""",
            operators.InputFromROI(
                where='ObjectDetection',
                which='CENTER',
                top_k=5,
                expand_margin=0,
            ),
        ),
        (
            """\
FaceDetection:
    template_path: templates/face_detection.yaml
    input:
        type: image
        source: image_processing
        image_processing:
            - convert-color:
                format: rgb2bgr
""",
            operators.InputWithImageProcessing(
                image_processing=[operators.ConvertColor(format='rgb2bgr')],
            ),
        ),
    ],
)
def test_parse_task_with_template_input_overridden_in_yaml(mock_template, in_yaml, expected_input):
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    mock_template.return_value = dict(
        input=dict(type='image'), preprocess=[dict(resize=dict(width=1024, height=768))]
    )
    mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.input == expected_input


MY_RECOG_TEMPLATE = """
class {class_name}(operators.AxOperator):
    distance_metric: str = 'Cosine'
    distance_threshold: float = 0.5
    k_fold: int = 0
    param_flag: bool = False
    embedding_size: int = 160

    {post_init}

    def exec_torch(self, img, result, meta):
        return img, result, meta

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        pass
"""


def create_custom_myrecog(
    class_name='CustomMyRecog',
    add_post_init=False,
):
    if add_post_init:
        post_init_impl = """
    def _post_init(self):
        if self.eval_mode == EvalMode.PAIR_EVAL:
            self.register_validation_params(
                {'distance_threshold': 0.2, 'k_fold': 10, 'distance_metric': 'Euclidean'}
            )
        elif self.eval_mode == EvalMode.EVAL:
            self.register_validation_params(
                {'distance_threshold': 0.2}
            )
            """
    else:
        post_init_impl = ""

    class_def = MY_RECOG_TEMPLATE.format(post_init=post_init_impl, class_name=class_name)
    namespace = {}
    exec(textwrap.dedent(class_def), globals(), namespace)
    return namespace[class_name]


MyRecog = create_custom_myrecog()


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "eval_overridden",
            "yaml": """
            FaceRecognition:
              preprocess:
                - torch-totensor:
              postprocess:
                - myrecog:
                    distance_metric: Cosine
                    distance_threshold: 0.5
                    k_fold: 0
                    param_flag: False
                    eval:
                        param_flag: True
                        distance_threshold: 0
                        k_fold: 5
        """,
            "expected": {
                "name": "FaceRecognition",
                "postprocess": [
                    MyRecog(
                        distance_threshold=0,
                        distance_metric='Cosine',
                        k_fold=5,
                        param_flag=True,
                    ),
                ],
            },
        },
        {
            "name": "pair_eval_overridden",
            "yaml": """
            FaceRecognition:
              preprocess:
                - torch-totensor:
              postprocess:
                - myrecog:
                    distance_metric: Cosine
                    distance_threshold: 0.5
                    k_fold: 0
                    pair_eval:
                        distance_threshold: 0
                        k_fold: 5
        """,
            "expected": {
                "name": "FaceRecognition",
                "postprocess": [
                    MyRecog(
                        distance_threshold=0,
                        distance_metric='Cosine',
                        k_fold=5,
                        embedding_size=160,
                    ),
                ],
            },
        },
    ],
)
def test_parse_operator_with_overrides(test_case):
    model_infos = make_model_infos(FACE_RECOGNITION_MODEL_INFO)
    in_dict = yaml.safe_load(test_case["yaml"])
    ops = dict(myrecog=MyRecog)
    mp = pipeline.parse_task(in_dict, ops, model_infos, eval_mode=True)

    assert mp.name == test_case["expected"]["name"]
    assert mp.postprocess == test_case["expected"]["postprocess"]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "eval_overridden",
            "yaml": """
            FaceRecognition:
              preprocess:
                - torch-totensor:
              postprocess:
                - myrecog:
                    distance_metric: Cosine
                    distance_threshold: 0.5
                    k_fold: 0
                    param_flag: False
                    eval:
                        param_flag: True
                        distance_threshold: 0
                        k_fold: 5
        """,
            "expected": {
                "name": "FaceRecognition",
                "postprocess": [
                    MyRecog(
                        distance_threshold=0,
                        distance_metric='Cosine',
                        k_fold=5,
                        param_flag=True,
                    ),
                ],
                "validation_settings": {
                    "distance_threshold": 0.2,
                    "pair_validation": False,
                },
            },
        },
        {
            "name": "pair_eval_overridden",
            "yaml": """
            FaceRecognition:
              preprocess:
                - torch-totensor:
              postprocess:
                - myrecog:
                    distance_metric: Cosine
                    distance_threshold: 0.5
                    k_fold: 0
                    pair_eval:
                        distance_threshold: 0
                        k_fold: 5
        """,
            "expected": {
                "name": "FaceRecognition",
                "postprocess": [
                    MyRecog(
                        distance_threshold=0,
                        distance_metric='Cosine',
                        k_fold=5,
                        embedding_size=160,
                    ),
                ],
                "validation_settings": {
                    "distance_threshold": 0.2,
                    "k_fold": 10,
                    "distance_metric": 'Euclidean',
                    "pair_validation": True,
                },
            },
        },
    ],
)
def test_parse_operator_with_register_validation_params(test_case):
    MyRecog = create_custom_myrecog(add_post_init=True)
    model_infos = make_model_infos(FACE_RECOGNITION_MODEL_INFO)
    in_dict = yaml.safe_load(test_case["yaml"])
    ops = dict(myrecog=MyRecog)
    mp = pipeline.parse_task(in_dict, ops, model_infos, eval_mode=True)

    assert mp.name == test_case["expected"]["name"]
    assert mp.validation_settings == test_case["expected"]["validation_settings"]


def test_parse_operator_with_duplicate_validation_settings():
    in_yaml = """
        FaceRecognition:
          preprocess:
            - torch-totensor:
          postprocess:
            - myrecog:
                distance_threshold: 0
                distance_metric: Cosine
                k_fold: 5
            - myrecog2:
                distance_metric: Euclidean
                distance_threshold: 0.5
                k_fold: 10
    """
    MyRecog = create_custom_myrecog(add_post_init=True)
    model_infos = make_model_infos(FACE_RECOGNITION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    ops = dict(myrecog=MyRecog, myrecog2=MyRecog)

    with pytest.raises(
        ValueError,
        match=(
            r"Operator .* has validation settings \{'distance_threshold'\} that are already registered"
        ),
    ):
        pipeline.parse_task(in_dict, ops, model_infos, eval_mode=True)


@pytest.mark.parametrize(
    'in_yaml,expected_error',
    [
        (
            """\
FaceDetection: hello
""",
            r'Task properties must be dict \(found type str\)',
        ),
        (
            """\
FaceDetection:
- template_path:
""",
            r'Task properties must be dict \(found type list\)',
        ),
        (
            """\
FaceDetection:
  template_path:
  - templates/face_detection.yaml
""",
            r'template_path must be str \(found type list\)',
        ),
        (
            """
FaceDetection:
  unknown_key: templates/face_detection.yaml
""",
            r'unknown_key is not a valid property of a Task',
        ),
        (
            """
FaceDetection:
  unknown_key: templates/face_detection.yaml
  other: huh
""",
            r'unknown_key, other are not valid properties of a Task',
        ),
        (
            """\
FaceDetection:
  input:
  - type: image
""",
            r'Task input must be dict \(found type list\)',
        ),
        (
            """\
FaceDetection:
  preprocess:
    torch-totensor:
""",
            r'preprocess operations must be list \(found type dict\)',
        ),
        (
            """\
FaceDetection:
  postprocess:
  - torch-totensor:
    -  positional
""",
            r'postprocess operator properties must be dict\|None \(found type list\)',
        ),
        (
            """\
FaceDetection:
  postprocess:
  - unknownop:
""",
            r'unknownop: Unsupported postprocess operator',
        ),
        (
            """\
UnknownModel:
  postprocess:
  - torch-totensor:
""",
            r'Model UnknownModel not found in models',
        ),
    ],
)
def test_parse_task_non_conformant(in_yaml, expected_error):
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with pytest.raises(ValueError, match=expected_error):
        pipeline.parse_task(in_dict, {}, model_infos)


def test_trace_model_info():
    mi = types.ModelInfo(
        'face', 'ObjectDetection', [3, 640, 480], labels='a b c d e f g h i'.split()
    )
    lines = []
    pipeline._trace_model_info(mi, lines.append)
    assert (
        '\n'.join(lines)
        == '''\
               Field Value
                name face
       task_category ObjectDetection
  input_tensor_shape 1, 3, 640, 480
  input_color_format RGB
 input_tensor_layout NCHW
         num_classes 1
              labels a, b, c, d, e
                     f, g, h, i
        label_filter []
         weight_path
          weight_url
          weight_md5
    prequantized_url
    prequantized_md5
             dataset
            base_dir
          class_name
          class_path
             version
        extra_kwargs {}
         input_width 480
        input_height 640
       input_channel 3'''
    )


MISSING = object()


class DataLoader:
    def __init__(self, sampler=MISSING):
        if sampler is not MISSING:
            self.sampler = sampler


class CheckedModel(types.Model):
    def __init__(self, res=None, sampler=None):
        self.res = res
        self.dl = DataLoader(sampler)

    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        pass

    def create_calibration_data_loader(self, **kwargs):
        return self.dl

    def check_calibration_data_loader(self, data_loader):
        assert self.dl is data_loader
        return self.res


@pytest.mark.parametrize(
    'model, warning',
    [
        # TODO: shim (types.Model(), 'Unable to determine'),
        (CheckedModel(), 'Unable to determine'),
        (CheckedModel(False), 'does not appear to'),
        (CheckedModel(True), ''),
        (CheckedModel(sampler=None), 'Unable to determine'),
        (CheckedModel(sampler=object()), 'does not appear to'),
        (CheckedModel(sampler=torch.utils.data.sampler.RandomSampler([], num_samples=10)), ''),
    ],
)
def test_check_calibration_dataloader(caplog, model, warning):
    dataloader = model.create_calibration_data_loader()
    pipeline._check_calibration_data_loader(model, dataloader)
    if warning:
        assert warning in caplog.text
    else:
        assert '' == caplog.text


def test_operator_eval_mode_affected_by_pipeline_eval():
    in_yaml = """\
FaceRecognition:
    preprocess:
    - torch-totensor:
    postprocess:
    - myrecog:
        distance_metric: Cosine
        distance_threshold: 0.5
        k_fold: 0
        param_flag: False
        eval:
            param_flag: True
"""
    model_infos = make_model_infos(FACE_RECOGNITION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    ops = dict(myrecog=MyRecog)
    mp = pipeline.parse_task(in_dict, ops, model_infos, eval_mode=True)
    assert bool(mp.postprocess[0].eval_mode) is True
    assert mp.postprocess[0].eval_mode == EvalMode.EVAL
    # pipeline says eval_mode=False
    mp = pipeline.parse_task(in_dict, ops, model_infos, eval_mode=False)
    assert bool(mp.postprocess[0].eval_mode) is False
    assert mp.postprocess[0].eval_mode == EvalMode.NONE


def test_operator_has_no_eval_mode_affected_by_pipeline_eval():
    in_yaml_no_eval = """\
FaceRecognition:
    preprocess:
    - torch-totensor:
    postprocess:
    - myrecog:
        distance_metric: Cosine
        distance_threshold: 0.5
        k_fold: 0
        param_flag: False
"""
    model_infos = make_model_infos(FACE_RECOGNITION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml_no_eval)
    ops = dict(myrecog=MyRecog)
    mp = pipeline.parse_task(in_dict, ops, model_infos, eval_mode=True)
    assert bool(mp.postprocess[0].eval_mode) is True
    assert mp.postprocess[0].eval_mode == EvalMode.EVAL


MY_DECODE_TEMPLATE = """
class {class_name}(operators.AxOperator):
    conf_threshold: 0.25

    {post_init}

    def exec_torch(self, img, result, meta):
        return img, result, meta

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        pass
"""


def create_custom_decode(
    class_name='CustomDecode',
    add_post_init=False,
):
    if add_post_init:
        post_init_impl = """
    def _post_init(self):
        self.gst_decoder_does_dequantization_and_depadding = True
            """
    else:
        post_init_impl = ""

    class_def = MY_DECODE_TEMPLATE.format(post_init=post_init_impl, class_name=class_name)
    namespace = {}
    exec(textwrap.dedent(class_def), globals(), namespace)
    return namespace[class_name]


Decode = create_custom_decode()


@pytest.mark.parametrize(
    "add_post_init, include_model_config",
    [
        (False, True),
        (True, True),
        (False, False),
        (True, False),
    ],
)
def test_yaml_with_decoder_dequant_config(add_post_init, include_model_config):
    Decode = create_custom_decode(add_post_init=add_post_init)

    in_yaml = f"""
FaceDetection:
    preprocess:
    - torch-totensor:
    postprocess:
    - decode:
        conf_threshold: 0.25
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    model_infos.model('FaceDetection').extra_kwargs['YOLO'] = {
        'focus_layer_replacement': include_model_config
    }
    in_dict = yaml.safe_load(in_yaml)
    ops = dict(decode=Decode)
    mp = pipeline.parse_task(in_dict, ops, model_infos)

    if include_model_config:
        assert mp.inference_config.gst_focus_layer_on_host is True
    else:
        assert mp.inference_config.gst_focus_layer_on_host is False

    decode_op = mp.postprocess[0]
    assert isinstance(decode_op, Decode)
    assert decode_op.conf_threshold == 0.25

    if add_post_init:
        assert decode_op.gst_decoder_does_dequantization_and_depadding
    else:
        assert not decode_op.gst_decoder_does_dequantization_and_depadding
