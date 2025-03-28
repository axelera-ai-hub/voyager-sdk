# Copyright Axelera AI, 2024

import pytest

from axelera.app import compile

n_padded_ch_inputs_1 = (
    (1, 2, 3, 4, 5, 6, 7, 8),
    (9, 10, 11, 12, 13, 14, 15, 16),
    (17, 18, 19, 20, 21, 22, 23, 24),
)
expected_result_1 = ((3, 4), (11, 12), (19, 20))


@pytest.mark.parametrize(
    "n_padded_ch_inputs, tensor_layout, where, expected",
    [
        (n_padded_ch_inputs_1, 'NHWC', 'H', expected_result_1),
        (((1, 2, 3, 4, 5, 6, 7, 8),), 'NHWC', 'H', ((3, 4),)),
        (((1, 2, 3, 4, 5, 6, 7, 8),), 'NCHW', 'H', ((5, 6),)),
    ],
)
def test_get_padded_low_high(n_padded_ch_inputs, tensor_layout, where, expected):
    result = compile.get_padded_low_high(n_padded_ch_inputs, tensor_layout, where)
    assert result == expected


def test_invalid_where():
    with pytest.raises(ValueError):
        compile.get_padded_low_high(n_padded_ch_inputs_1, 'NHWC', 'X')


def test_invalid_tensor_layout():
    with pytest.raises(ValueError):
        compile.get_padded_low_high(n_padded_ch_inputs_1, 'XYZW', 'H')


def test_invalid_n_padded_ch_inputs():  # (not enough values)
    with pytest.raises(IndexError):
        compile.get_padded_low_high(([1, 2, 3],), 'NHWC', 'H')


@pytest.mark.parametrize(
    "output_shapes, n_padded_ch, current_layout, expected_layout, expected_result",
    [
        (
            ((1, 40, 40, 256), (1, 20, 20, 256), (1, 80, 80, 256)),
            ((0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1)),
            'NHWC',
            'NCHW',
            [(1, 255, 40, 40), (1, 255, 20, 20), (1, 255, 80, 80)],
        ),
        (
            ((1, 40, 40, 256), (1, 20, 20, 256), (1, 80, 80, 256)),
            ((0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1)),
            'NHWC',
            'NHWC',
            [(1, 40, 40, 255), (1, 20, 20, 255), (1, 80, 80, 255)],
        ),
        (
            ((1, 256, 40, 40), (1, 256, 20, 20), (1, 256, 80, 80)),
            ((0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 1, 0)),
            'NCHW',
            'NHWC',
            [(1, 40, 39, 256), (1, 20, 19, 256), (1, 80, 79, 256)],
        ),
        (
            ((1, 40, 40, 256), (1, 20, 20, 256), (1, 80, 80, 256)),
            ((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0)),
            'NHWC',
            None,
            [(1, 40, 40, 256), (1, 20, 20, 256), (1, 80, 80, 256)],
        ),
        pytest.param(
            ((1, 40, 40), (1, 20, 20, 256), (1, 80, 80, 256)),
            ((0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1)),
            'NHWC',
            'NHWC',
            ValueError,  # Invalid input shapes
        ),
        pytest.param(
            ((1, 40, 40, 256), (1, 20, 20, 256)),
            ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1)),
            'NHWC',
            'NHWC',
            ValueError,  # Invalid n_padded_ch length
        ),
        pytest.param(
            ((1, 256, 40, 40), (1, 256, 20, 20), (1, 256, 80, 80)),
            ((0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 1, 0)),
            'NCHW',
            'NSHW',
            ValueError,  # Different layouts including non-matching dimensions
        ),
        pytest.param(
            ((1, 40, 40, 256), (1, 20, 20, 256), (1, 80, 80, 256)),
            ((0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1)),
            'NHWC',
            'HA',
            ValueError,  # "Invalid expected layout"
        ),
    ],
)
def test_get_original_shape(
    output_shapes, n_padded_ch, current_layout, expected_layout, expected_result
):
    if isinstance(expected_result, type) and issubclass(expected_result, Exception):
        with pytest.raises(expected_result):
            compile.get_original_shape(output_shapes, n_padded_ch, current_layout, expected_layout)
    else:
        result = compile.get_original_shape(
            output_shapes, n_padded_ch, current_layout, expected_layout
        )
        assert result == expected_result, f"Expected {expected_result}, but got {result}"
