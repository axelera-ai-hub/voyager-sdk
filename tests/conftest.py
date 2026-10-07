# Copyright Axelera AI, 2026
import pytest

_NO_RUNTIME2 = 'axelera.runtime2 is not installed'


@pytest.fixture
def display_cv():
    '''The OpenCV renderer lives in axelera.runtime2, which is an optional dependency.'''
    return pytest.importorskip('axelera.app.display_cv', reason=_NO_RUNTIME2)


@pytest.fixture
def rt2_img():
    '''The image type the renderers draw on, also from the optional axelera.runtime2.'''
    return pytest.importorskip('axelera.runtime2.img', reason=_NO_RUNTIME2)
