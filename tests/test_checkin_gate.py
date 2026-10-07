# Copyright Axelera AI, 2026
"""The checkin gate decides whether the whole axelera.app suite runs at all.

A pattern that matched every path, or a comparison that ran the wrong way round,
would stop the product being tested, and no job would go red to say so. So the
pattern AND the grep flags are read out of the workflow rather than restated
here, and the deny list is pinned as a whole rather than by example.
"""

import pathlib
import re
import subprocess

WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / '.github/workflows/checkin.yml'

DENY_LIST = {
    'model_axscript',
    'axelera_zoo',
    'model_recipes',
    'axelera_llm',
    'axelera_runtime2',
    'worklogs',
}

_GATE = re.compile(
    r"unreachable='(?P<pattern>[^']+)'.*?\|\s*grep\s+(?P<flags>-\S+)\s+\"\$unreachable\"",
    re.S,
)


def _gate():
    found = _GATE.search(WORKFLOW.read_text())
    assert found, f'no unreachable= assignment followed by a grep in {WORKFLOW}'
    return found.group('pattern'), found.group('flags')


def _runs_app_checkin(paths):
    """Ask grep the question the workflow asks it, with the workflow's own pattern and flags."""
    pattern, flags = _gate()
    non_md = [p for p in paths if not p.endswith('.md')]
    if not non_md:
        return False
    grep = subprocess.run(['grep', flags, pattern], input='\n'.join(non_md), text=True)
    return grep.returncode == 0


def test_the_deny_list_is_exactly_these_trees():
    """Pins the complement: adding a tree that DOES reach axelera.app fails here."""
    pattern, _ = _gate()
    assert set(pattern.strip('^()/').split('|')) == DENY_LIST


def test_a_change_axelera_app_can_reach_runs_the_checkin():
    reachable = [
        'axelera/app/pipe/gst.py',
        'ax_models/model_cards/yolov8n-coco.yaml',
        'ax_datasets/dataset_private.yaml',
        # the image runs `make operators` over this tree
        'operators/axstreamer/src/AxOpenClExtensions.cpp',
        # not installed in the image either, but nobody has classified it
        'trackers/CMakeLists.txt',
        # run_tests.py keys should_check_models on these two
        'internal-apps/axelera/internal/apps/smoke_networks.py',
        'internal_tools/model_cards.yaml',
        'deploy.py',
        'requirements.axelera.txt',
        # a tree nobody has classified is reachable until someone says otherwise
        'some_new_tree/thing.py',
    ]
    for path in reachable:
        assert _runs_app_checkin([path]), path


def test_a_change_it_cannot_reach_skips_the_checkin():
    assert not _runs_app_checkin(['model_axscript/llm/build.py'])
    assert not _runs_app_checkin(['axelera_zoo/python/axelera/zoo/configs/llm/qwen3.py'])
    assert not _runs_app_checkin(['model_recipes/llm/qwen3.py'])
    assert not _runs_app_checkin(['axelera_llm/pyproject.toml'])
    # the image installs axelera-rt as a released wheel, never this tree
    assert not _runs_app_checkin(['axelera_runtime2/python/axelera/runtime2/op/tracker.py'])
    assert not _runs_app_checkin(['model_axscript/llm/build.py', 'axelera_zoo/tests/test_x.py'])


def test_documents_alone_skip_the_checkin():
    assert not _runs_app_checkin(['README.md', 'docs/tutorials/custom_model.md'])


def test_one_reachable_file_among_many_is_enough():
    assert _runs_app_checkin(['model_axscript/llm/build.py', 'axelera/app/config.py'])
