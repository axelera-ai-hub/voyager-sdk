# Copyright Axelera AI, 2023
import numpy as np
import pytest

from axelera.app.meta import gst

NULL_TS = b'\x00' * 8
NULL_INF = b'\x00' * 4
FORTY_TWO_TS = np.uint64(40_000_000_002).tobytes()


@pytest.mark.parametrize(
    'sid, ts, inf, exp',
    [
        (b'\x00\x00\x00\x00', NULL_TS, NULL_INF, (0, 0, 0)),
        (b'\xff\x00\x00\x00', NULL_TS, NULL_INF, (255, 0, 0)),
        (b'\xff\x00\x00\x00', FORTY_TWO_TS, NULL_INF, (255, 40.000000002, 0)),
        (b'\xff\x00\x00\x00', NULL_TS, b'\x03\x00\x00\x00', (255, 0, 3)),
        (b'\xff', NULL_TS, NULL_INF, 'Expecting stream_id to be a byte stream 4 bytes long'),
        (b'\xff' * 5, NULL_TS, NULL_INF, 'Expecting stream_id to be a byte stream 4 bytes long'),
        (b'\x00' * 4, None, NULL_INF, "Expecting timestamp meta element"),
        (
            b'\x00' * 4,
            b'\x00' * 7,
            NULL_INF,
            "Expecting timestamp to be a byte stream 8 bytes long",
        ),
        (
            b'\x00' * 4,
            b'\x00' * 9,
            NULL_INF,
            "Expecting timestamp to be a byte stream 8 bytes long",
        ),
        (
            b'\x00' * 4,
            NULL_TS,
            b'\x00' * 3,
            "Expecting inferences to be a byte stream 4 bytes long",
        ),
        (
            b'\x00' * 4,
            NULL_TS,
            b'\x00' * 5,
            "Expecting inferences to be a byte stream 4 bytes long",
        ),
        (b'\x00' * 4, NULL_TS, None, "Expecting inferences meta element"),
        (None, NULL_TS, NULL_INF, "Expecting stream_id meta element"),
    ],
)
def test_decode_stream_meta(sid, ts, inf, exp):
    data = {} if sid is None else {'stream_id': sid}
    if ts is not None:
        data['timestamp'] = ts
    if inf is not None:
        data['inferences'] = inf
    if isinstance(exp, str):
        with pytest.raises(RuntimeError, match=exp):
            gst.decode_stream_meta(data)
    else:
        assert exp == gst.decode_stream_meta(data)
