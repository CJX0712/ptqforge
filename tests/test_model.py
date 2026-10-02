"""Tests for the reference MLP model.

Author: 晨星
"""

import numpy as np

from ptqforge.core.seed import set_all
from ptqforge.data.generators import get_dataset
from ptqforge.model.mlp import build_default_mlp


def _fit_once(seed, dataset="moons"):
    set_all(seed)
    rng = np.random.RandomState(seed)
    X, y = get_dataset(dataset, rng)
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    n_tr = int(X.shape[0] * 0.8)
    m = build_default_mlp(int(X.shape[1]), int(y.max()) + 1)
    m.fit(X[:n_tr], y[:n_tr], rng)
    return m, X[n_tr:], y[n_tr:]


def test_mlp_trains_above_chance():
    m, Xte, yte = _fit_once(1)
    acc = m.score(Xte, yte)
    assert acc > 0.80  # moons is linearly separable-ish but needs the MLP


def test_mlp_deterministic_same_seed():
    m1, Xte, yte = _fit_once(2)
    m2, _, _ = _fit_once(2)
    assert m1.score(Xte, yte) == m2.score(Xte, yte)


def test_mlp_quantized_weights_consistent():
    m, Xte, yte = _fit_once(3)
    a_full = m.score(Xte, yte)
    a_same = m.score(Xte, yte, W=m.W)  # passing quantized==fp32 should match
    assert a_full == a_same


def test_mlp_layer_inputs_shape():
    m, Xte, _ = _fit_once(4)
    ins = m.layer_inputs(Xte)
    assert len(ins) == len(m.W)
    # first layer input is the raw features (n_samples, in_features)
    assert ins[0].shape[1] == m.W[0].shape[1]
    assert ins[0].shape[0] == Xte.shape[0]
