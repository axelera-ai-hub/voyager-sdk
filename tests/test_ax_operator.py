# Copyright Axelera AI, 2024
from pathlib import Path
from unittest.mock import Mock

import pytest

from axelera import types
from axelera.app import gst_builder
from axelera.app.operators import AxOperator, PipelineContext


def _members(op):
    return [op.required, op.defaults, op.supported]


def test_with_no_attributes_members_are_empty():
    assert not any(_members(ImplementsGenGstAndTorch))


def test_ax_operator():
    with pytest.raises(Exception, match="got an unexpected keyword argument 'a'"):
        ImplementsGenGstAndTorch(a=1)


def test_derived_ax_operator():
    class Derived(ImplementsGenGstAndTorch):
        a: int
        b: int
        c: int = 1
        d = 2

    assert _members(Derived) == [['a', 'b'], {'c': 1, 'd': 2}, ['a', 'b', 'c', 'd']]
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'x'"):
        Derived(x=1)
    with pytest.raises(Exception, match="missing 1 required positional argument: 'b'"):
        Derived(a=1)
    Derived(a=1, b=2)
    assert Derived.add_defaults({'a': 1, 'b': 2}) == {'a': 1, 'b': 2, 'c': 1, 'd': 2}


class NameRequired(AxOperator):
    a: int
    b: int
    c: int = 1
    d: int = 2


class NameRequiredTyped(AxOperator):
    a: int
    b: int
    c: int = 1
    d: int = 2


class ImplementsGenGstAndTorch(AxOperator):
    def exec_torch(self, img, result, meta):
        raise Exception('overridden torch')

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        raise Exception('overridden gst')


class NameNotRequiredExplicitTyped(ImplementsGenGstAndTorch):
    a: int
    needs_model_name: bool = False


class NameNotRequiredExplicit(ImplementsGenGstAndTorch):
    a: int
    needs_model_name = False


class NameNotRequiredImplicit(ImplementsGenGstAndTorch):
    a: int


def test_bad_model_name():
    with pytest.raises(
        AttributeError,
    ):

        class BadModelName(AxOperator):
            a: int
            model_name: str

    with pytest.raises(
        AttributeError,
    ):

        class BadModelName(AxOperator):
            a: int
            _model_name: str


def test_gen_torch_unimplemented():
    class ImplementsGstOnly(AxOperator):
        def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
            raise Exception('overridden gst')

    with pytest.raises(TypeError, match='abstract methods? exec_torch'):
        ImplementsGstOnly()


def test_gen_torch_implemented():
    op = ImplementsGenGstAndTorch()
    with pytest.raises(Exception, match='overridden torch'):
        op.exec_torch(None, None, None)


def test_gen_gst_unimplemented():
    class ImplementsTorchOnly(AxOperator):
        def exec_torch(self, img, result, meta):
            raise Exception('overridden torch')

    with pytest.raises(TypeError, match='abstract methods? build_gst'):
        ImplementsTorchOnly()


def test_gen_gst_implemented():
    op = ImplementsGenGstAndTorch()
    with pytest.raises(Exception, match='overridden gst'):
        op.build_gst(None, '')


class WithMember(ImplementsGenGstAndTorch):
    a: int


class NoMember(ImplementsGenGstAndTorch):
    pass


def test_repr_with_modelname():
    op = WithMember(a=1)
    op._model_name = 'somemodel'
    assert repr(op) == 'WithMember(a=1, model_name=\'somemodel\')'


def test_repr_without_modelname():
    op = WithMember(a=1)
    assert repr(op) == 'WithMember(a=1)'


def test_repr_no_members_with_modelname():
    op = NoMember()
    op._model_name = 'somemodel'
    assert repr(op) == 'NoMember(model_name=\'somemodel\')'


def test_repr_no_members_without_modelname():
    op = NoMember()
    assert repr(op) == 'NoMember()'


def test_eq_with_modelname():
    op = WithMember(a=1)
    op._model_name = 'somemodel'
    op1 = WithMember(a=1)
    op1._model_name = 'somemodel'
    assert op == op1
    op2 = WithMember(a=2)
    op2._model_name = 'somemodel'
    assert op != op2
    assert op != WithMember(a=1)


def test_eq_without_modelname():
    op = WithMember(a=1)
    assert op == WithMember(a=1)
    assert op != WithMember(a=2)


def test_eq_no_members_with_modelname():
    op = NoMember()
    op._model_name = 'somemodel'
    op1 = NoMember()
    op1._model_name = 'somemodel'
    assert op == op1
    op2 = NoMember()
    op2._model_name = 'somemodel2'
    assert op != op2


def test_eq_no_members_without_modelname():
    op = NoMember()
    assert op == NoMember()


@pytest.mark.parametrize(
    "class_name, attribute, error_message",
    [
        (
            "EvalIsSpecialKeyword",
            "eval: dict = {}",
            "eval is a reserved keyword and cannot be used as parameters",
        ),
        (
            "PairEvalIsSpecialKeyword",
            "pair_eval: dict = {}",
            "pair_eval is a reserved keyword and cannot be used as parameters",
        ),
        (
            "EvalOverrideFromYamlIsSpecialKeyword",
            "_override_params_for_eval: bool = False",
            "_override_params_for_eval is a reserved keyword and cannot be used as parameters",
        ),
    ],
)
def test_reserved_keywords(class_name, attribute, error_message):
    with pytest.raises(ValueError, match=error_message):
        exec(
            f"""
class {class_name}(AxOperator):
    {attribute}
"""
        )


def test_operator_override_validation_settings():
    with pytest.raises(
        TypeError,
        match="BadOp tries to override final property 'validation_settings'; please use register_validation_params instead",
    ):

        class BadOp(AxOperator):
            distance_threshold: float = 0.5
            kfold: int = 0
            distance_metric: str = 'Cosine'

            def exec_torch(self, img, result, meta):
                return img, result, meta

            def build_gst(self, gst, stream_idx):
                pass

            @property
            def validation_settings(self):
                return {
                    'distance_threshold': self.distance_threshold,
                    'k_fold': self.kfold,
                }

    class GoodOp(AxOperator):
        distance_threshold: float = 0.5
        kfold: int = 0
        distance_metric: str = 'Cosine'

        def _post_init(self):
            self.register_validation_params(
                {'distance_threshold': self.distance_threshold, 'kfold': self.kfold}
            )
            with pytest.raises(
                AttributeError,
                match="Parameter 'kfold' is already registered in validation settings.",
            ):
                self.register_validation_params({'kfold': self.kfold})

        def exec_torch(self, img, result, meta):
            return img, result, meta

        def build_gst(self, gst, stream_idx):
            pass

    op = GoodOp()
    assert op.validation_settings == {
        'distance_threshold': 0.5,
        'kfold': 0,
        'pair_validation': False,
    }


def test_ax_operator_configure_model_and_context_info():
    mi = types.ModelInfo(
        'modelname',
        types.TaskCategory.Classification,
        [3, 224, 244],
    )
    mi.name = 'modelname'
    mi.manifest = types.Manifest(
        'modellib',
        quantize_params=[(0.1, -14)],
        dequantize_params=[(0.3, 0.4)],
        input_shapes=[(1, 3, 224, 224)],
    )

    class TestOp(AxOperator):
        def exec_torch(self, img, result, meta):
            return img, result, meta

        def build_gst(self, gst, stream_idx):
            pass

    op = TestOp()
    op.configure_model_and_context_info(mi, PipelineContext(), "task_name", 0, '', Path('.'))
    assert op.model_name == 'modelname'
    assert op.task_name == 'task_name'
    assert op.compiled_model_dir == Path('.')
    assert op.required == []
    assert op.defaults == {}
    assert op.supported == []
