# Copyright Axelera AI, 2023
import pathlib
import textwrap
from unittest.mock import MagicMock, patch

import pytest
import yaml

torch = pytest.importorskip("torch")

from axelera import types
from axelera.app import (
    config,
    gst_builder,
    network,
    operators,
    pipeline,
    schema,
    utils,
    yaml_parser,
)
from axelera.app.operators import EvalMode

FACE_DETECTION_MODEL_INFO = types.ModelInfo('FaceDetection', 'ObjectDetection', [3, 640, 480])
FACE_RECOGNITION_MODEL_INFO = types.ModelInfo('FaceRecognition', 'Classification', [3, 160, 160])
TRACKER_MODEL_INFO = types.ModelInfo(
    'Tracker', 'ObjectTracking', model_type=types.ModelType.CLASSICAL_CV
)


@pytest.fixture(scope="function", autouse=True)
def _reset_parse_cache():
    schema.load_task.cache_clear()
    schema.load_network.cache_clear()
    yaml_parser.get_network_yaml_info.cache_clear()


def make_model_infos(the_model_info):
    model_infos = network.ModelInfos()
    model_infos.add_model(the_model_info, pathlib.Path('/path'))
    return model_infos


@pytest.mark.parametrize(
    'in_yaml',
    [
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
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(utils, 'load_yaml_by_reference') as mock_template:
            with patch.object(schema, 'load_task') as mock_schema:
                mock_schema.return_value = None
                mock_template.return_value = dict(
                    input=dict(type='image'),
                    preprocess=[dict(resize=dict(width=1024, height=768))],
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
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(utils, 'load_yaml_by_reference') as mock_template:
            with patch.object(schema, 'load_task') as mock_schema:
                mock_schema.return_value = None
                mock_template.return_value = dict(
                    input=dict(type='image'),
                    preprocess=[dict(resize=dict(width=1024, height=768))],
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
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(utils, 'load_yaml_by_reference') as mock_template:
            with patch.object(schema, 'load_task') as mock_schema:
                mock_schema.return_value = None
                mock_template.return_value = dict(
                    input=dict(type='image'),
                    preprocess=[dict(resize=dict(width=1024, height=768))],
                )
                with pytest.raises(AssertionError):
                    pipeline.parse_task(in_dict, {}, model_infos)


def test_parse_task_with_template_postprocess_with_extra_operator_before_template_operators_in_yaml():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  postprocess:
    - c:
    - topk:

"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(utils, 'load_yaml_by_reference') as mock_template:
            with patch.object(schema, 'load_task') as mock_schema:
                mock_schema.return_value = None
                mock_template.return_value = dict(
                    input=dict(type='image'), postprocess=[dict(topk=dict())]
                )
                with pytest.raises(AssertionError):
                    pipeline.parse_task(in_dict, {}, model_infos)


def test_parse_task_with_template_postprocess_with_extra_operator_after_template_operators_in_yaml():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  postprocess:
    - topk: # simple treat topk as a decoder not a real case
    - ctc-decoder:

"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(utils, 'load_yaml_by_reference') as mock_template:
            with patch.object(schema, 'load_task') as mock_schema:
                mock_schema.return_value = None
                mock_template.return_value = dict(
                    input=dict(type='image'), postprocess=[dict(topk=dict())]
                )
                mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.input == operators.Input()
    assert mp.preprocess == []
    assert mp.postprocess == [
        operators.postprocessing.TopK(),
        operators.postprocessing.CTCDecoder(),
    ]


def test_parse_task_with_template_postprocess_with_extra_operator_in_yaml():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  postprocess:
    - ctc-decoder:

"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(utils, 'load_yaml_by_reference') as mock_template:
            with patch.object(schema, 'load_task') as mock_schema:
                mock_schema.return_value = None
                mock_template.return_value = dict(
                    input=dict(type='image'), postprocess=[dict(topk=dict())]
                )
                mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.input == operators.Input()
    assert mp.preprocess == []
    assert mp.postprocess == [
        operators.postprocessing.TopK(),
        operators.postprocessing.CTCDecoder(),
    ]


@pytest.fixture
def mock_template():
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(utils, 'load_yaml_by_reference') as mock_template:
            with patch.object(schema, 'load_task') as mock_schema:
                mock_schema.return_value = None
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
""",
            operators.InputFromROI(
                where='ObjectDetection',
                which='CENTER',
                top_k=5,
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
        (
            """\
FaceDetection:
  postprocess:
  - tracker:
""",
            r'tracker is a classical CV operator, not allowed in postprocess',
        ),
    ],
)
def test_parse_task_non_conformant(in_yaml, expected_error):
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    with pytest.raises(ValueError, match=expected_error):
        in_yaml = yaml.safe_load(in_yaml)
        pipeline.parse_task(in_yaml, {}, model_infos)


def test_parse_task_non_classical_cv_operator_in_cv_process():
    in_yaml = """
Tracker:
  input:
    source: full
    color_format: RGB
  cv_process:
  - topk:
"""
    model_infos = make_model_infos(TRACKER_MODEL_INFO)
    with pytest.raises(ValueError, match=r'topk: Not a valid classical CV operator'):
        in_yaml = yaml.safe_load(in_yaml)
        pipeline.parse_task(in_yaml, {}, model_infos)


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
         output_info []
          model_type DEEP_LEARNING
         weight_path
          weight_url
          weight_md5
   prequantized_path
    prequantized_url
    prequantized_md5
    precompiled_path
     precompiled_url
     precompiled_md5
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


def test_parse_task_with_task_render_config():
    in_yaml = """\
FaceDetection:
  input:
  preprocess:
  render:
    show_annotations: True
    show_labels: True
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.task_render_config.show_annotations is True
    assert mp.task_render_config.show_labels is True


def test_parse_task_with_template_and_task_render_config():
    in_yaml = """\
FaceDetection:
  template_path: templates/face_detection.yaml
  input:
    type: image
  render:
    show_annotations: True
    show_labels: False
"""
    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(utils, 'load_yaml_by_reference') as mock_template:
            with patch.object(schema, 'load_task') as mock_schema:
                mock_schema.return_value = None
                mock_template.return_value = dict(
                    input=dict(type='image'),
                    preprocess=[dict(resize=dict(width=1024, height=768))],
                )
                mp = pipeline.parse_task(in_dict, {}, model_infos)
    assert mp.name == 'FaceDetection'
    assert mp.task_render_config.show_annotations is True
    assert mp.task_render_config.show_labels is False


def test_parse_classical_cv_task_with_task_render_config():
    in_yaml = """
Tracker:
  input:
    source: full
    color_format: RGB
  cv_process:
  - tracker:
      algorithm: oc-sort
      bbox_task_name: detections
      algo_params:
        det_thresh: 0.6
  render:
    show_annotations: True
    show_labels: True
"""
    model_infos = make_model_infos(TRACKER_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)

    # Create a concrete mock class for BaseClassicalCV
    mock_tracker = MagicMock(spec=operators.BaseClassicalCV)
    mock_tracker.__repr__ = lambda _: "MockTracker"

    # The correct way to patch the tracker
    with patch.object(operators, 'builtins_classical_cv', {'tracker': mock_tracker}):
        mp = pipeline.parse_task(in_dict, {}, model_infos)

    assert mp.name == 'Tracker'
    assert mp.task_render_config.show_annotations is True
    assert mp.task_render_config.show_labels is True


def test_parse_task_render_config():
    phases = {
        'render': {
            'show_annotations': True,
            'show_labels': False,
        }
    }
    task_render_config = pipeline._parse_task_render_config(phases)
    assert task_render_config is not None
    assert task_render_config.show_annotations is True
    assert task_render_config.show_labels is False
    assert 'render' not in phases  # Check if render was popped from phases


def test_parse_task_render_config_with_additional_params():
    # Mock the TaskRenderConfig.from_dict method to accept our test parameters
    original_from_dict = config.TaskRenderConfig.from_dict

    def mock_from_dict(cls, settings_dict):
        # Only keep the keys that are valid for TaskRenderConfig
        valid_keys = {'show_annotations', 'show_labels'}
        filtered_dict = {k: v for k, v in settings_dict.items() if k in valid_keys}
        return original_from_dict(filtered_dict)

    with patch.object(config.TaskRenderConfig, 'from_dict', classmethod(mock_from_dict)):
        phases = {
            'render': {
                'show_annotations': True,
                'show_labels': False,
                'bbox_thickness': 2,
                'text_thickness': 1,
                'font_scale': 0.5,
            }
        }
        task_render_config = pipeline._parse_task_render_config(phases)
        assert task_render_config is not None
        assert task_render_config.show_annotations is True
        assert task_render_config.show_labels is False
        assert 'render' not in phases  # Check if render was popped from phases


def test_parse_task_render_config_empty():
    phases = {}
    task_render_config = pipeline._parse_task_render_config(phases)
    assert task_render_config is None


@pytest.mark.parametrize(
    "inference_config, expected_handle_flags",
    [
        # Test handle_all=True sets all flags to True
        (
            {"handle_all": True},
            {
                "handle_dequantization_and_depadding": True,
                "handle_transpose": True,
                "handle_postamble": True,
                "handle_preamble": True,
            },
        ),
        # Test handle_all=False sets all flags to False
        (
            {"handle_all": False},
            {
                "handle_dequantization_and_depadding": False,
                "handle_transpose": False,
                "handle_postamble": False,
                "handle_preamble": False,
            },
        ),
        # Test individual flags without handle_all (should use defaults or specified values)
        (
            {
                "handle_preamble": True,
                "handle_transpose": False,
                "handle_dequantization_and_depadding": False,
                "handle_postamble": False,
            },
            {
                "handle_dequantization_and_depadding": False,
                "handle_transpose": False,
                "handle_postamble": False,
                "handle_preamble": True,
            },
        ),
        # Test empty config uses defaults
        (
            {},
            {
                "handle_dequantization_and_depadding": True,
                "handle_transpose": True,
                "handle_postamble": True,
                "handle_preamble": True,
            },
        ),
    ],
)
def test_yaml_with_inference_config(inference_config, expected_handle_flags):
    """Test that YAML parsing correctly sets InferenceOpConfig flags."""
    in_yaml = f"""
FaceDetection:
    preprocess:
    - torch-totensor:
    {'inference:' if inference_config else ''}
"""
    # Add inference config to YAML if provided
    if inference_config:
        for key, value in inference_config.items():
            in_yaml += f"      {key}: {str(value).lower()}\n"
    in_yaml += """
    postprocess:
    - decode:
        conf_threshold: 0.25
"""

    class CustomDecode(operators.AxOperator):
        conf_threshold: float = 0.25

        def exec_torch(self, *a, **kw):
            pass

        def build_gst(self, *a, **kw):
            pass

    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    in_dict = yaml.safe_load(in_yaml)
    ops = dict(decode=CustomDecode)
    mp = pipeline.parse_task(in_dict, ops, model_infos)

    # Verify inference op config flags
    for flag_name, expected_value in expected_handle_flags.items():
        actual_value = getattr(mp.inference_op_config, flag_name)
        assert (
            actual_value == expected_value
        ), f"Expected {flag_name}={expected_value}, got {actual_value}"

    # Verify decode operator
    decode_op = mp.postprocess[0]
    assert isinstance(decode_op, CustomDecode)
    assert decode_op.conf_threshold == 0.25


@pytest.mark.parametrize(
    "template_config, phase_config, expected_result",
    [
        # Template has handle_all, phase has individual flags -> phase individual flags win
        # Note: handle_transpose is forced to True because handle_postamble is True (dependency)
        (
            {"handle_all": True},
            {"handle_preamble": True, "handle_transpose": False},
            {
                "handle_preamble": True,
                "handle_transpose": True,  # forced to True due to postamble dependency
                "handle_dequantization_and_depadding": True,  # default
                "handle_postamble": True,  # default
            },
        ),
        # Phase has handle_all, template has individual flags -> phase handle_all wins
        (
            {"handle_preamble": True, "handle_transpose": False},
            {"handle_all": False},
            {
                "handle_preamble": False,
                "handle_transpose": False,
                "handle_dequantization_and_depadding": False,
                "handle_postamble": False,
            },
        ),
        # Both have individual flags -> phase overrides template
        # Note: handle_transpose and handle_dequantization_and_depadding forced to True due to postamble dependency
        (
            {"handle_preamble": True, "handle_transpose": True},
            {"handle_preamble": False},
            {
                "handle_preamble": False,
                "handle_transpose": True,  # from template + forced due to postamble dependency
                "handle_dequantization_and_depadding": True,  # default + forced due to postamble dependency
                "handle_postamble": True,  # default
            },
        ),
        # Only template has config -> template is used
        (
            {"handle_all": False},
            {},
            {
                "handle_preamble": False,
                "handle_transpose": False,
                "handle_dequantization_and_depadding": False,
                "handle_postamble": False,
            },
        ),
    ],
)
def test_inference_config_priority_from_yaml_dict(template_config, phase_config, expected_result):
    """Test InferenceOpConfig.from_yaml_dict priority handling."""
    config = operators.InferenceOpConfig.from_yaml_dict(phase_config, template_config)

    for flag_name, expected_value in expected_result.items():
        actual_value = getattr(config, flag_name)
        assert (
            actual_value == expected_value
        ), f"Expected {flag_name}={expected_value}, got {actual_value}"


def test_inference_config_conflicting_flags():
    """Test that setting both handle_all and individual flags raises an error."""
    with pytest.raises(
        ValueError, match="Cannot set both 'handle_all' and individual handle flags"
    ):
        operators.InferenceOpConfig(handle_all=True, handle_preamble=False)


def test_inference_config_postamble_dependencies():
    """Test that postamble dependencies are enforced with warnings."""
    # Test that dependencies are enforced correctly
    config = operators.InferenceOpConfig(
        handle_postamble=True, handle_transpose=False, handle_dequantization_and_depadding=False
    )
    # Dependencies should be forced to True
    assert config.handle_transpose is True
    assert config.handle_dequantization_and_depadding is True


def test_yaml_with_focus_layer_replacement():
    """Test YOLO focus layer replacement functionality."""
    in_yaml = """
FaceDetection:
    preprocess:
    - torch-totensor:
    inference:
      handle_all: false
    postprocess:
    - decode:
        conf_threshold: 0.25
"""

    class CustomDecode(operators.AxOperator):
        conf_threshold: float = 0.25

        def exec_torch(self, *a, **kw):
            pass

        def build_gst(self, *a, **kw):
            pass

    model_infos = make_model_infos(FACE_DETECTION_MODEL_INFO)
    # Test focus layer replacement via extra_kwargs
    model_infos.model('FaceDetection').extra_kwargs['YOLO'] = {'focus_layer_replacement': True}
    in_dict = yaml.safe_load(in_yaml)
    ops = dict(decode=CustomDecode)
    mp = pipeline.parse_task(in_dict, ops, model_infos)

    # Verify inference op config
    assert mp.inference_op_config.handle_all is False
    assert mp.inference_op_config.handle_preamble is False

    # Verify YOLO config is preserved in model info
    assert mp.model_info.extra_kwargs['YOLO']['focus_layer_replacement'] is True


@pytest.mark.parametrize("handle_all_value", [True, False, None])
def test_inference_config_handle_all_property(handle_all_value):
    """Test the handle_all convenience parameter."""
    if handle_all_value is None:
        # Test individual flags when handle_all is None
        config = operators.InferenceOpConfig(
            handle_preamble=True,
            handle_transpose=False,
            handle_dequantization_and_depadding=True,
            handle_postamble=False,
        )
        assert config.handle_preamble is True
        assert config.handle_transpose is False
        assert config.handle_dequantization_and_depadding is True
        assert config.handle_postamble is False
    else:
        # Test handle_all parameter
        config = operators.InferenceOpConfig(handle_all=handle_all_value)
        assert config.handle_preamble is handle_all_value
        assert config.handle_transpose is handle_all_value
        assert config.handle_dequantization_and_depadding is handle_all_value
        assert config.handle_postamble is handle_all_value
