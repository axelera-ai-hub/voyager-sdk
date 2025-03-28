# Copyright Axelera AI, 2024

import pytest
import yaml

model_subsets = ['VALIDATION', 'MINIMAL', 'PERFORMANCE']


@pytest.mark.parametrize('subset', model_subsets)
def test_model_release_candidates_consistent(subset):
    with open("internal_tools/model_release_candidates.yaml", "r") as f:
        data = yaml.safe_load(f)
    ready = set(data['READY_FOR_RELEASE'])
    in_subset = set(x for x in data[subset] if not x.startswith("#"))
    missing = in_subset - ready
    assert not missing, f"{subset} has models not ready for release: {sorted(missing)}"
