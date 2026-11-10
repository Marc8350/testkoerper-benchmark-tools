"""Tests for the pure parts of tools/tk_project.py (no camera, no zivid needed)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import tk_project  # noqa: E402


def _calibration(tmp_path, pocket=(50.0, 40.0)):
    T = np.eye(4)
    T[:3, 3] = [100.0, 200.0, 1000.0]
    T2 = T.copy()
    T2[0, 3] = 160.0
    plate = lambda side, names: {"side": side, "floor_depth_mm": 17.0, "pocket_size_mm": list(pocket),
                                 "slots": {n: (T if i == 0 else T2).tolist() for i, n in enumerate(names)}}
    raw = {"positions": {"P1": {"plates": {"L": plate("L", ["L1", "L2"]), "R": plate("R", ["R1"])}}}}
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_corners_identity_floor_and_top():
    T = np.eye(4)
    floor = tk_project.pocket_corners(T, (50.0, 40.0), 0.0)
    assert floor.tolist() == [[-25, -20, 0], [25, -20, 0], [25, 20, 0], [-25, 20, 0]]
    assert np.allclose(tk_project.pocket_corners(T, (50.0, 40.0), 17.0)[:, 2], 17.0)


def test_corners_follow_the_slot_frame():
    # slot frame rotated 90 degrees about z and moved: local x axis becomes world y
    T = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]], dtype=float)
    corners = tk_project.pocket_corners(T, (50.0, 40.0), 0.0)
    assert np.allclose(corners[1], [10 + 20, 20 + 25, 30])  # local (25, -20) -> (10 - (-20), 20 + 25)


def test_load_plates_replaces_nan_pocket(tmp_path):
    path = _calibration(tmp_path, pocket=(float("nan"), float("nan")))
    plates = tk_project.load_plates(path, "P1")
    assert plates["L"]["pocket"] == (tk_project.DEFAULT_POCKET_MM,) * 2
    assert set(plates) == {"L", "R"} and plates["L"]["floor_depth"] == 17.0


def test_unknown_position_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        tk_project.load_plates(_calibration(tmp_path), "P9")


def test_select_slots(tmp_path):
    plates = tk_project.load_plates(_calibration(tmp_path), "P1")
    assert tk_project.select_slots(plates, ["R1", "L2"], None) == ["R1", "L2"]
    assert tk_project.select_slots(plates, [], "L") == ["L1", "L2"]
    with pytest.raises(SystemExit):
        tk_project.select_slots(plates, ["L7"], None)
    with pytest.raises(SystemExit):
        tk_project.select_slots(plates, [], "X")


def test_polygons_level(tmp_path):
    plates = tk_project.load_plates(_calibration(tmp_path), "P1")
    floor = tk_project.polygons_for(plates, ["L1"], "floor")[0]
    top = tk_project.polygons_for(plates, ["L1"], "top")[0]
    assert np.allclose(top - floor, [0, 0, 17.0])
    assert math.isclose(floor[:, 2].mean(), 1000.0)


def test_draw_outline_and_fill():
    poly = [np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 40.0], [10.0, 40.0]])]
    outline = tk_project.draw_polygons(poly, (60, 80), (0, 255, 0), fill=False, thickness=2)
    filled = tk_project.draw_polygons(poly, (60, 80), (0, 255, 0), fill=True, thickness=2)
    assert outline.shape == (60, 80, 4) and (outline[..., 3] == 255).all()
    assert outline[25, 30, 1] == 0 and outline[10, 30, 1] == 255  # interior dark, edge lit
    assert filled[25, 30, 1] == 255
    assert filled[..., 1].sum() > outline[..., 1].sum()
    assert outline[0, 0, :3].tolist() == [0, 0, 0]  # black = no light


def test_count_outside():
    inside = np.array([[1.0, 1.0], [79.0, 59.0]])
    outside = np.array([[-1.0, 5.0], [80.0, 5.0], [5.0, 60.0], [5.0, -0.5]])
    assert tk_project.count_outside([inside], (60, 80)) == 0
    assert tk_project.count_outside([inside, outside], (60, 80)) == 4
