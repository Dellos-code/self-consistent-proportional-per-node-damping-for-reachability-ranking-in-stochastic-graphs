import json
import multiprocessing

import numpy as np
import scipy.sparse as sp

import mixing_lib as M
import prop_damping_lib as L


def test_config_id_stable():
    cfg = {"b": [2, 1], "a": 3}
    assert M.config_id(cfg) == M.config_id(json.loads(json.dumps(cfg)))


def test_fusion_endpoints():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 1.0, 2.0])
    assert np.array_equal(L.ranks_of(M.fuse([a, b], beta=0)), L.ranks_of(a))
    assert np.array_equal(L.ranks_of(M.fuse([a, b], beta=1)), L.ranks_of(b))


def test_scaled_matches_direct_when_finite():
    T = sp.csr_matrix([[0.0, 0.5], [0.2, 0.0]])
    x = np.array([0.4, 0.6])
    ops = {"Pbar": 0.5}
    tr, _ = M.phase_B_trajectory(T, x, ops, 0.7, x, (1, 2, 3), "none")
    y = x.copy()
    for k in range(1, 4):
        y = (1 - 0.7) * x + 0.7 * (T @ y)
        assert np.allclose(tr[k], y)


def test_same_operator_polynomial():
    T = sp.csr_matrix([[0.0, 0.5], [0.2, 0.0]])
    x = np.array([0.4, 0.6])
    y = M.linear_rank(T, x, 0.3, 3)
    expected = sum((0.7 ** 0.0) * np.zeros_like(x) for _ in [])  # shape check
    direct = np.zeros_like(x)
    z = x.copy()
    for _ in range(3):
        z = 0.7 * x + 0.3 * (T @ z)
        direct += z
    assert np.allclose(y, direct)


def test_operator_mixture_lambda_zero_matches_diffuse():
    T = sp.csr_matrix([[0.0, 0.5], [0.2, 0.0]])
    x = np.array([0.4, 0.6])
    got, _ = M.operator_mixture(T, T, x, 0.7, 0.0, c=0.9, steps=3)
    expected, _ = M.phase_A(T, x, {"Pbar": 0.0}, damping=0.7, steps=3)
    assert np.allclose(got, expected)
