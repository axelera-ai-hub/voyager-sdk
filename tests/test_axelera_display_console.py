# Copyright Axelera AI, 2025

from axelera.app import display_console


def test_count_as_bar():
    full = '\u2588'
    half = '\u258C'
    assert full * 8 == display_console._count_as_bar(0, 8, 0)
    assert full * 8 == display_console._count_as_bar(8, 8, 8)
    assert full * 4 + ' ' * 4 == display_console._count_as_bar(4, 8, 8)
    assert full * 3 + half + ' ' * 4 == display_console._count_as_bar(7, 8, 16)
