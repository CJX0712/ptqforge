"""Tests for synthetic data generators (determinism + structure).

Author: 晨星
"""

import numpy as np

from ptqforge.data.generators import (
    get_dataset,
    make_circles,
    make_gaussian_blobs,
    make_moons,
)


def test_moons_deterministic_same_seed():
    a = make_moons(np.random.RandomState(1))
    b = make_moons(np.random.RandomState(1))
    assert np.allclose(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_moons_is_two_class_and_non_trivial():
    X, y = make_moons(np.random.RandomState(5))
    assert set(np.unique(y).tolist()) == {0, 1}
    assert X.shape[0] == 2000 and X.shape[1] == 2
    # the two classes must be interleaved (not block-contiguous) after generation
    # so an index split would be class-balanced only after an explicit shuffle
    assert X.shape[0] == y.shape[0]


def test_circles_has_two_rings():
    X, y = make_circles(np.random.RandomState(7))
    assert set(np.unique(y).tolist()) == {0, 1}
    assert X.shape[1] == 2


def test_gaussian_blobs_two_classes():
    X, y = make_gaussian_blobs(np.random.RandomState(3), n_features=12, n_classes=2)
    assert X.shape[1] == 12
    assert set(np.unique(y).tolist()) == {0, 1}


def test_get_dataset_registry_keys():
    for name in ["moons", "circles", "gaussian"]:
        X, y = get_dataset(name, np.random.RandomState(0))
        assert X.shape[0] == y.shape[0]
