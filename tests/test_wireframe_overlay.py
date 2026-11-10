import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import wireframe_overlay as w  # noqa: E402


def test_feature_edges_of_a_box_are_its_twelve_edges():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    tri = np.array([[v[a], v[b], v[c]] for a, b, c, d in quads for (a, b, c) in ((a, b, c), (a, c, d))])
    assert len(w.feature_edges(tri)) == 12


def test_project_puts_the_optical_axis_on_the_principal_point():
    k = np.array([[2000.0, 0, 1200.0], [0, 2000.0, 1000.0], [0, 0, 1]])
    assert np.allclose(w.project(np.array([[0.0, 0.0, 1000.0], [100.0, 0.0, 1000.0]]), k), [[1200, 1000], [1400, 1000]])
