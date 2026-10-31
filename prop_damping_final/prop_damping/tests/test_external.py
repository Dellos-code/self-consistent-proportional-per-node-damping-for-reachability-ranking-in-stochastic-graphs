import sys

import numpy as np

import mixing_lib as M
from build_panels import load_instance

sys.path.insert(0, "external/escape_probability")
import escape_lib as external


def test_irie_exact_matches_vendored_port():
    inst, _, _, _ = load_instance("data/dev/0000.npz")
    other = external.Instance(inst.n, inst.edges[:, 0], inst.edges[:, 1], inst.p, inst.x)
    assert np.allclose(M.irie_exact(inst), external.m_irie(other, 0.7), atol=1e-12, rtol=0)
