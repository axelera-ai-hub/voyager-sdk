# Copyright Axelera AI, 2024

import re
import tempfile

import pytest
import strictyaml as sy

from axelera.app import operators
from axelera.app.schema import compile_schema, generate_operators, static
from axelera.app.schema.generated import _generate_operator
from axelera.app.schema.types import Any, MapPattern

DEFINITON = '''\
from axelera.app.operators import AxOperator
from typing import Optional, List

class DecodeYolo(AxOperator):
    box_format: str
    normalized_coord: bool
    label_filter: Optional[List[str] | str] = None
    label_exclude: Optional[List[str] | str] = None
    conf_threshold: float = 0.25
    max_nms_boxes: int = 30000
    use_multi_label: bool = False
    nms_iou_threshold: float = 0.45
    nms_class_agnostic: bool = False
    nms_top_k: int = 300
    generic_gst_decoder: bool = False'''


def _get_compiled(name, check_required):
    definitions = (
        getattr(static, name)(get_test_operators(), get_test_compilation_config)
        if name != 'operator'
        else get_test_operators()
    )
    compiled = compile_schema(definitions, check_required)
    if name == 'network':
        return compiled
    return sy.MapPattern(sy.Str(), compiled)


def get_test_operators(definition=DEFINITON):
    with (
        tempfile.NamedTemporaryFile(mode='w', suffix='.py') as d_file,
        tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as n_file,
    ):
        d_file.write(definition)
        d_file.seek(0)
        network_yaml = f'''\
        operators:
            decodeyolo:
                class: DecodeYolo
                class_path: {d_file.name}
        '''
        n_file.write(network_yaml)
        n_file.seek(0)
        return generate_operators(n_file.name, ".")


def get_test_compilation_config():
    return MapPattern[Any]


def test_operator_missing_class():
    network_yaml = '''\
    operators:
      decodeyolo:
        # Missing 'class' attribute
        class_path: $AXELERA_FRAMEWORK/ax_models/decoders/yolo.py
    '''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as temp_file:
        temp_file.write(network_yaml)
        temp_file.seek(0)
        with pytest.raises(ValueError) as excinfo:
            generate_operators(temp_file.name, ".")
        assert "must have a 'class' and 'class_path' attribute" in str(excinfo.value)


def test_operator_missing_class_path():
    network_yaml = '''\
    operators:
      decodeyolo:
        class: DecodeYolo
        # Missing 'class_path' attribute
    '''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as temp_file:
        temp_file.write(network_yaml)
        temp_file.seek(0)
        with pytest.raises(ValueError) as excinfo:
            generate_operators(temp_file.name, ".")
        assert "must have a 'class' and 'class_path' attribute" in str(excinfo.value)


def yaml_load(in_yaml, compiled):
    return sy.dirty_load(in_yaml, compiled, allow_flow_style=True).data


def test_generate_operator():
    compiled = _get_compiled('task', True)

    operator_yaml = '''\
    test_task:
        preprocess:
            - decodeyolo:
                box_format: xywh
                normalized_coord: False
                label_filter: [a, b, c]
                label_exclude: a, b, c
                conf_threshold: 0.25
                max_nms_boxes: 30000
                use_multi_label: False
                nms_iou_threshold: 0.45
                nms_class_agnostic: False
    '''
    output = yaml_load(operator_yaml, compiled)
    assert output["test_task"] == {
        "preprocess": [
            {
                "decodeyolo": {
                    "box_format": "xywh",
                    "normalized_coord": False,
                    "label_filter": ["a", "b", "c"],
                    "label_exclude": ["a", "b", "c"],
                    "conf_threshold": 0.25,
                    "max_nms_boxes": 30000,
                    "use_multi_label": False,
                    "nms_iou_threshold": 0.45,
                    "nms_class_agnostic": False,
                }
            }
        ]
    }


def test_generate_all_builtin_operators():
    for name, op in operators.builtins.items():
        try:
            compile_schema(_generate_operator(op), True)
        except Exception as e:
            raise ValueError(
                f"Error parsing operator '{name}', class {op.__name__}: {str(e)}"
            ) from e


def test_operator_sentinel():
    compiled = _get_compiled('task', True)

    operator_yaml = '''\
    test_task:
        preprocess:
            - decodeyolo:
                box_format: xywh
                normalized_coord: False
                label_filter: $$label_filter$$
                label_exclude: a, b, c
                conf_threshold: 0.25
                max_nms_boxes: 30000
                use_multi_label: False
                nms_iou_threshold: 0.45
                nms_class_agnostic: False
    '''
    output = yaml_load(operator_yaml, compiled)
    assert output["test_task"]["preprocess"][0]["decodeyolo"]["label_filter"] == "$$label_filter$$"


def test_operator_sentinel_when_disallowed():
    compiled = _get_compiled('task', True)

    operator_yaml = '''\
    test_task:
        preprocess:
            - decodeyolo:
                box_format: xywh
                normalized_coord: $$normalized_coord$$
                label_filter: $$label_filter$$
                label_exclude: a, b, c
                conf_threshold: 0.25
                max_nms_boxes: 30000
                use_multi_label: False
                nms_iou_threshold: 0.45
                nms_class_agnostic: False
    '''
    with pytest.raises(sy.exceptions.YAMLValidationError) as e:
        yaml_load(operator_yaml, compiled)
    assert "found arbitrary text" in str(e.value)
    assert "normalized_coord" in str(e.value)


@pytest.mark.parametrize(
    "section, operator_name",
    [
        ("preprocess", "undeclared_operator"),
        ("postprocess", "missing_decoder"),
        ("cv_process", "tracker_missing"),
        ("preprocess", "custom-missing-operator"),
    ],
)
def test_undeclared_operators(section, operator_name):
    compiled = _get_compiled('task', True)
    operator_yaml = f'''\
    test_task:
        {section}:
            - {operator_name}:
                param1: test
    '''
    with pytest.raises(sy.exceptions.YAMLValidationError) as e:
        yaml_load(operator_yaml, compiled)
    exp_op_name = re.sub(
        r'[-_]', '', operator_name
    )  # Names are reported without dashes or underscores
    error_msg = str(e.value)
    assert f"unexpected key not in schema '{exp_op_name}'" in error_msg
    assert f"did you forget to declare '{exp_op_name}' in the operators section" in error_msg
