# Copyright Axelera AI, 2023

from axelera import types


def test_construction():
    manifest = types.Manifest(
        'path',
        quantize_params=((0.007, 0),),
        dequantize_params=(
            (0.084, 59),
            (0.14, -8),
        ),
    )
    assert manifest.quantized_model_file == 'path'
    assert manifest.quantize_params == ((0.007, 0),)
    assert manifest.dequantize_params == ((0.084, 59), (0.14, -8))
