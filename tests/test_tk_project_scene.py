import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import tk_project_scene as ps  # noqa: E402


def test_parse_cells():
    assert ps.parse_cells("L1:UO/q3 R2:H") == {"L1": ("UO", 3), "R2": ("H", 0)}


def test_unterteil_turns_ninety_degrees_per_quarter():
    yaws = [ps.parts_for_code("U", q)[0][3] for q in range(4)]
    assert yaws == [90.0, 180.0, 270.0, 360.0]


def test_oberteil_on_halter_sits_25_mm_up_and_turns_with_the_slot():
    parts = dict((k, (z, xy, yaw)) for k, z, xy, yaw in ps.parts_for_code("HO", 1))
    assert parts["Oberteil"][0] == 25.0 and parts["Oberteil"][2] == 90.0 and parts["Halter"][0] == 0.0


def test_oberteil_in_unterteil_is_offset_along_the_groove():
    (_, _, _, yaw_u), (_, z, xy, yaw_o) = ps.parts_for_code("UO", 0)
    assert yaw_u == yaw_o == 90.0 and z == 25.0
    assert np.allclose(xy, (3.5, 0.0), atol=1e-9)           # (0, -3.5) in the Unterteil frame turned by 90 deg


def test_flatten_and_transform_put_edges_on_the_plate_top():
    edges = np.array([[[0.0, 0.0, 10.0], [10.0, 0.0, 30.0]]])
    flat = ps.flatten(ps.edges_in_slot(edges, 5.0, (1.0, 2.0), 90.0), 17.0)
    assert np.allclose(flat[0, :, 2], 17.0) and np.allclose(flat[0, 1, :2], (1.0, 12.0))
    t = np.eye(4)
    t[:3, 3] = (100.0, 0.0, 1000.0)
    assert np.allclose(ps.to_camera(flat, t)[0, 0], (101.0, 2.0, 1017.0))


def test_unknown_code_is_an_error():
    try:
        ps.parts_for_code("X", 0)
    except ValueError:
        return
    raise AssertionError("no error")


def test_syrz_unterteil_wall_to_wall_direction_is_horizontal_at_q0():
    (_, _, _, yaw), = ps.parts_for_code("U", 0, "syrz")
    c, sn = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    wall_dir = (-sn, c)                                     # CAD y axis (wall to wall) in the slot frame
    assert np.isclose(abs(wall_dir[0]), 1.0) and np.isclose(wall_dir[1], 0.0, atol=1e-9)


def test_syrz_oberteil_sits_25_mm_up_and_turns_with_the_unterteil_in_uo():
    (_, _, _, yaw_u), (kind, z, xy, yaw_o) = ps.parts_for_code("UO", 1, "syrz")
    assert kind == "Oberteil" and z == 25.0 and xy == (0.0, 0.0) and yaw_o == yaw_u


def test_syrz_step_uses_its_own_meshes():
    assert all("SY_RZ" in f for f in ps.STEP_MESHES["syrz"].values())


def test_simple_shapes_are_polygons_of_four_edges():
    assert all(len(e) % 4 == 0 for e in ps.SIMPLE_SHAPES.values()) and len(ps.SIMPLE_SHAPES["Unterteil"]) == 8
