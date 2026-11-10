import json

import numpy as np
import pytest

import cell_ground_truth as cg
import marker_ground_truth as mgt
from test_marker_ground_truth import MARKER_MM, make_scene, part_pose, pose_errors

# ---------------------------------------------------------------------------
# Synthetic plate: tilted top plane with a 3x3 grid of square pockets
# ---------------------------------------------------------------------------

K_PLATE = np.array([[2479.0, 0.0, 650.0], [0.0, 2477.0, 500.0], [0.0, 0.0, 1.0]])
PLATE_SHAPE = (1000, 1300)
TILT = np.radians(12.5)
NORMAL = np.array([0.0, np.sin(TILT), -np.cos(TILT)])  # towards the camera
E1 = np.array([1.0, 0.0, 0.0])
E2 = np.array([0.0, np.cos(TILT), np.sin(TILT)])  # image down, in the plane
PITCH, OPENING, DEPTH = 60.5, 50.5, 17.2
ORIGIN = np.array([20.0, -10.0, 1050.0])  # centre pocket on the top plane


def pocket_centre(col, row):
    return ORIGIN + (col - 1) * PITCH * E1 + (row - 1) * PITCH * E2


def plate_sample(dead_top=False, noise=0.15, seed=0):
    """Ray-cast depth of the plate. Walls have no depth; ``dead_top`` removes the plate top (dark glossy)."""
    v, u = np.mgrid[0 : PLATE_SHAPE[0], 0 : PLATE_SHAPE[1]]
    rays = np.stack([(u + 0.5 - K_PLATE[0, 2]) / K_PLATE[0, 0], (v + 0.5 - K_PLATE[1, 2]) / K_PLATE[1, 1], np.ones(u.shape)], axis=-1)
    floor_origin = ORIGIN - DEPTH * NORMAL
    t_top = (ORIGIN @ NORMAL) / (rays @ NORMAL)
    t_floor = (floor_origin @ NORMAL) / (rays @ NORMAL)

    def inside(points):
        a, b = (points - ORIGIN) @ E1, (points - ORIGIN) @ E2
        ia, ib = np.round(a / PITCH), np.round(b / PITCH)
        return (np.abs(ia * PITCH - a) < OPENING / 2) & (np.abs(ib * PITCH - b) < OPENING / 2) & (np.abs(ia) <= 1) & (np.abs(ib) <= 1)

    in_top = inside(rays * t_top[..., None])
    in_floor = inside(rays * t_floor[..., None])
    depth = np.where(in_top & in_floor, t_floor, np.where(in_top, np.nan, t_top))
    if dead_top:
        depth = np.where(in_top & in_floor, depth, np.nan)
    depth = depth + np.random.default_rng(seed).normal(0, noise, depth.shape)
    xyz = (rays * depth[..., None]).astype(np.float32)
    return mgt.Sample("empty", np.zeros(PLATE_SHAPE + (3,), np.uint8), xyz, K_PLATE)


ROI = (330, 200, 980, 800)


@pytest.fixture(scope="module")
def plate():
    return plate_sample()


def test_slot_calibration_recovers_the_pocket_lattice(plate):
    result = cg.calibrate_sample(plate, {"L": ROI})["L"]
    assert result.n_pockets_measured == 9
    assert result.floor_depth_mm == pytest.approx(DEPTH, abs=0.1)
    assert result.pitch_mm[0] == pytest.approx(PITCH, abs=0.15)
    assert result.pitch_mm[1] == pytest.approx(PITCH, abs=0.15)
    assert result.grid_angle_deg == pytest.approx(90.0, abs=0.3)
    assert result.lattice_rms_mm < 0.3
    for row in range(3):
        for col in range(3):
            T = result.slots[f"L{row * 3 + col + 1}"]
            truth = pocket_centre(col, row) - DEPTH * NORMAL  # slot origin: pocket centre on the floor
            assert np.linalg.norm(T[:3, 3] - truth) < 0.35
            assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
            assert T[:3, 2] == pytest.approx(NORMAL, abs=2e-3)  # z towards the camera
            assert T[:3, 0] @ E1 > 0.999  # x along the columns (image right)
            assert T[1, 1] < 0  # y is image up


def test_slot_names_run_row_major_from_the_top_left(plate):
    slots = cg.calibrate_sample(plate, {"L": ROI})["L"].slots
    assert slots["L1"][0, 3] < slots["L2"][0, 3] < slots["L3"][0, 3]  # left to right
    assert slots["L1"][1, 3] < slots["L4"][1, 3] < slots["L7"][1, 3]  # top to bottom in the image


def test_slot_calibration_without_plate_top_uses_floor_and_flags_it():
    sample = plate_sample(dead_top=True)
    # no top plane: only a floor-depth prior makes the calibration possible
    with pytest.raises(ValueError, match="top plane not measurable"):
        cg.measure_pockets(sample, "L", ROI, floor_depth_prior_mm=None)
    m = cg.measure_pockets(sample, "L", ROI, floor_depth_prior_mm=DEPTH)
    assert not m.top_plane_found and np.isnan(m.rim_uv).all()
    assert any("top plane not measurable" in w for w in m.warnings)


def test_rigid_lattice_fit_recovers_origin_and_rotation():
    ij = np.array([[c, r] for r in range(3) for c in range(3)], float)
    phi = np.radians(1.2)
    a, b = np.array([np.cos(phi), np.sin(phi)]) * 60.5, np.array([-np.sin(phi), np.cos(phi)]) * 60.5
    obs = np.array([5.0, -3.0]) + ij[:, :1] * a + ij[:, 1:] * b
    weights = np.ones_like(obs)
    weights[[2, 5, 8]] = 0.0  # a hidden column
    o, a_fit, b_fit = cg._fit_rigid_lattice(ij, obs, weights, (60.5, 60.5))
    assert o == pytest.approx([5.0, -3.0], abs=0.02)
    assert a_fit == pytest.approx(a, abs=0.02)


# ---------------------------------------------------------------------------
# Layout, captures, positions
# ---------------------------------------------------------------------------

LOG_A01 = "PO_TY a: L1-L8 U, R1-R8 HO"
LOG_A02 = "PO_TY a: L1,L3,L5,L7 U, L2,L4,L6,L8 HO, R2,R4,R6,R8 U, R1,R3,R5,R7 HO (read from image)"
LOG_A03 = (
    "PO_TY a, plate position P2: L1,L3,L5,L7 U, L2,L4,L6,L8 HO, R2,R4,R6,R8 U, R1,R3,R5,R7 HO; "
    "L3,L6,L9 partly behind robot arm (read from image)"
)


def test_layout_ranges_and_lists():
    layout = cg.parse_layout(LOG_A01)
    assert {k: v for k, v in layout.items() if k[0] == "L"} == {f"L{k}": "U" for k in range(1, 9)}
    assert {k: v for k, v in layout.items() if k[0] == "R"} == {f"R{k}": "HO" for k in range(1, 9)}
    assert "L9" not in layout and "R9" not in layout


def test_layout_checkerboard_and_notes_are_parsed_without_picking_up_the_note():
    expected = {f"L{k}": ("U" if k % 2 else "HO") for k in range(1, 9)}
    expected.update({f"R{k}": ("HO" if k % 2 else "U") for k in range(1, 9)})
    assert cg.parse_layout(LOG_A02) == expected
    assert cg.parse_layout(LOG_A03) == expected  # "L3,L6,L9 partly behind robot arm" is a note, not a layout


def test_layout_review_log_syntax_with_empty_and_unknown_slots():
    layout = cg.parse_layout("L1,L2 U; L3 HO; L4 H; R1-R9 empty; L9 ?")
    assert layout == {"L1": "U", "L2": "U", "L3": "HO", "L4": "H", "L9": "?"}  # empty slots are simply absent


def test_layout_of_empty_scenes_and_odd_texts():
    assert cg.parse_layout("both plates empty, plate position P2") == {}
    assert cg.parse_layout("as found") == {}
    assert cg.parse_layout("x: L1-3 H, R4 E") == {"L1": "H", "L2": "H", "L3": "H", "R4": "E"}
    assert cg.parse_layout("L1 Halter, L2 Unterteil") == {}  # words are not codes


def test_layout_conflicts_and_bad_ranges_raise():
    with pytest.raises(ValueError, match="L2"):
        cg.parse_layout("L1-L3 U, L2 H")
    with pytest.raises(ValueError, match="descending"):
        cg.parse_layout("L5-L2 U")
    with pytest.raises(ValueError, match="across plates"):
        cg.parse_layout("L5-R7 U")


def test_step_key_from_capture_name():
    assert cg.parse_step_key("S02_fill_poty-a_03") == ("fill", "poty")
    assert cg.parse_step_key("S02_empty_plates-p2_01") == ("empty", "plates")
    assert cg.parse_step_key("whatever") is None


def _positions():
    return [
        cg.PositionSpec("P1", ["e1"], {"L": (0, 0, 1, 1)}),
        cg.PositionSpec("P2", ["e2"], {"L": (0, 0, 1, 1)}, from_sample="S02_fill_poty-a_03"),
    ]


def test_position_follows_capture_time_from_the_log():
    log = {
        "S02_fill_poty-a_01": {"time": "2026-10-01T08:54:49Z"},
        "S02_fill_poty-a_02": {"time": "2026-10-01T09:05:53Z"},
        "S02_fill_poty-a_03": {"time": "2026-10-01T09:07:56Z"},
        "S02_fill_poty-a_09": {"time": "2026-10-01T10:00:00Z"},
    }
    assert [cg.resolve_position(k, _positions(), log) for k in log] == ["P1", "P1", "P2", "P2"]
    assert cg.resolve_position("not_in_log", _positions(), log) == "P1"


def _cfg():
    step = cg.StepParts("poty", {}, np.eye(4), ["Einsetzen_(PO_TY)"])
    return cg.CellConfig("DICT_4X4_50", 15.0, {"poty": step}, [], None)


def test_capture_selection_skips_rejected_teach_and_unknown_steps():
    log = {
        "S02_fill_poty-a_01": {"layout": LOG_A01, "time": "2026-10-01T08:54:49Z"},
        "S02_fill_poty-a_04": {"layout": "REJECTED: does not match dictated layout"},
        "S02_teach_grip-u-l1_01": {"layout": "teaching"},
        "S02_fill_wirz-a_01": {"layout": LOG_A01},
        "S02_empty_plates_01": {"layout": "both plates empty"},
        "S02_fill_poty-a_05": {"layout": ""},
        "S02_fill_poty-a_07": {"layout": "PENDING_REVIEW"},
        "S02_fill_poty-a_08": {"layout": "L1,L2 U; L3 ?"},
    }
    ids = sorted(log) + ["S02_fill_poty-a_06"]
    chosen, skipped = cg.select_captures(ids, log, _cfg(), explicit=False, positions=_positions())
    assert [c.sample_id for c in chosen] == ["S02_fill_poty-a_01", "S02_fill_poty-a_05"]
    assert chosen[0].positions == {"L": "P1", "R": "P1"} and chosen[0].source == "capture_log"
    assert "REJECTED" in skipped["S02_fill_poty-a_04"]
    assert "teach" in skipped["S02_teach_grip-u-l1_01"]
    assert "no mesh mapping" in skipped["S02_fill_wirz-a_01"]
    assert "capture log" in skipped["S02_fill_poty-a_06"]
    assert "PENDING_REVIEW" in skipped["S02_fill_poty-a_07"]
    assert "unknown slots (?): L3" in skipped["S02_fill_poty-a_08"]


REVIEW_CSV = """file,verdict,step,pos_L,pos_R,layout,notes,reviewed_at
S02_auto_001.zdf,accept,Einsetzen_(PO_TY),P1,P2,"L1,L2 U; L3 HO; L4 H; R1-R9 empty",,2026-10-01T10:00:00Z
S02_auto_002.zdf,accept,Einsetzen_(PO_TY),P1,P1,L1 U; L9 ?,,2026-10-01T10:01:00Z
S02_auto_003.zdf,reject,Einsetzen_(PO_TY),P1,P1,L1 U,bad,2026-10-01T10:02:00Z
S02_auto_004.zdf,accept,Klammern_(SY_RZ),P1,P1,L1 U,,2026-10-01T10:03:00Z
S02_auto_005.zdf,accept,Einsetzen_(PO_TY),P1,P1,L1 U,,2026-10-01T10:04:00Z
S02_auto_005.zdf,reject,Einsetzen_(PO_TY),P1,P1,L1 U,changed mind,2026-10-01T10:05:00Z
S02_auto_006.zdf,reject,Einsetzen_(PO_TY),P1,P1,L1 U,first,2026-10-01T10:06:00Z
S02_auto_006.zdf,accept,Einsetzen_(PO_TY),P2,P1,L1-L3 U,second,2026-10-01T10:07:00Z
"""


def test_review_log_gives_step_layout_and_per_plate_positions(tmp_path):
    (tmp_path / "review_log.csv").write_text(REVIEW_CSV)
    review = cg.read_review_log(tmp_path / "review_log.csv")
    log = {sid: {"layout": "PENDING_REVIEW"} for sid in review}
    ids = sorted(review) + ["S02_auto_007"]
    chosen, skipped = cg.select_captures(ids, log, _cfg(), explicit=False, positions=_positions(), review=review)
    by_id = {c.sample_id: c for c in chosen}
    assert sorted(by_id) == ["S02_auto_001", "S02_auto_006"]
    assert by_id["S02_auto_001"].positions == {"L": "P1", "R": "P2"}  # mixed plate positions
    assert by_id["S02_auto_001"].step_key == "poty" and by_id["S02_auto_001"].source == "review_log"
    assert cg.parse_layout(by_id["S02_auto_001"].layout_text) == {"L1": "U", "L2": "U", "L3": "HO", "L4": "H"}
    assert by_id["S02_auto_006"].positions == {"L": "P2", "R": "P1"}  # the latest row wins
    assert "unknown slots" in skipped["S02_auto_002"]
    assert "reject" in skipped["S02_auto_003"]
    assert "no mesh mapping" in skipped["S02_auto_004"]
    assert "reject" in skipped["S02_auto_005"]  # accepted first, rejected later
    assert "S02_auto_007" in skipped


def test_step_names_resolve_through_aliases_and_the_bracket_token():
    cfg = _cfg()
    assert cg.resolve_step_key("Einsetzen_(PO_TY)", cfg) == "poty"
    assert cg.resolve_step_key("Einlegen_(PO_TY)", cfg) == "poty"
    assert cg.resolve_step_key("poty", cfg) == "poty"
    assert cg.resolve_step_key("Klammern_(SY_RZ)", cfg) is None


def test_background_takes_each_image_half_from_its_plates_position(tmp_path):
    cal = cg.Calibration({"P1": {}, "P2": {}}, {}, {"P1": ["a"], "P2": ["b"]}, {}, {}, {})
    cache = {"P1": np.full((4, 10), 1.0, np.float32), "P2": np.full((4, 10), 2.0, np.float32)}
    mixed = cg.composite_background({"L": "P1", "R": "P2"}, tmp_path, cal, cache)
    assert (mixed[:, :5] == 1.0).all() and (mixed[:, 5:] == 2.0).all()
    assert cg.composite_background({"L": "P2", "R": "P2"}, tmp_path, cal, cache)[0, 0] == 2.0


def test_position_check_is_per_plate(plate):
    top_point, floor_depth = ORIGIN, DEPTH
    cal = cg.Calibration(
        {}, {"P1": {"L": cg.CalibratedPlate(top_point, NORMAL, floor_depth)}, "P2": {"L": cg.CalibratedPlate(top_point + np.array([0.0, 0.0, 40.0]), NORMAL, floor_depth)}},
        {}, {"P1": {"L": ROI}, "P2": {"L": ROI}}, {},
    )
    fractions = cg.position_fractions(plate, cal)
    assert fractions["L"]["P1"] > 0.5 > fractions["L"]["P2"]


# ---------------------------------------------------------------------------
# Fitting parts against depth
# ---------------------------------------------------------------------------


def l_mesh():
    """A 40x40x20 slab with a 10 mm tall bar along one edge: no rotational symmetry."""

    def box(x0, x1, y0, y1, z0, z1):
        v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
        f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        return v, f

    v1, f1 = box(-20, 20, -20, 20, 0, 20)
    v2, f2 = box(-20, 20, 10, 20, 20, 30)
    return mgt.Mesh(np.vstack([v1, v2]), np.vstack([f1, f2 + 8]))


T_SLOT = mgt.make_transform(np.column_stack([E1, np.cross(NORMAL, E1), NORMAL]), ORIGIN - DEPTH * NORMAL)
SHAPE_FIT = PLATE_SHAPE


def fit_scene(truth_params, yaw_deg=90.0, mesh=None, background_z=None):
    """Measured depth of the L-shaped part at ``slot_pose(truth)`` in front of a flat background."""
    mesh = mesh or l_mesh()
    T = cg.slot_pose(T_SLOT, np.asarray(truth_params, float), yaw_deg)
    bg = np.full(SHAPE_FIT, np.inf)
    v, u = np.mgrid[0 : SHAPE_FIT[0], 0 : SHAPE_FIT[1]]
    rays = np.stack([(u + 0.5 - K_PLATE[0, 2]) / K_PLATE[0, 0], (v + 0.5 - K_PLATE[1, 2]) / K_PLATE[1, 1], np.ones(u.shape)], axis=-1)
    bg = ((T_SLOT[:3, 3] @ NORMAL) / (rays @ NORMAL)).astype(np.float32)
    depth = np.minimum(bg, mgt.render_depth(mesh, T, K_PLATE, SHAPE_FIT)).astype(np.float32)
    return mesh, depth, bg


def run_fit(depth, bg, mesh, lower=None, upper=None):
    members = [(mesh, np.eye(4))]
    roi = cg.union_roi([[(mesh, cg.slot_pose(T_SLOT, np.zeros(4), y))] for y in cg.YAW_HYPOTHESES_DEG], K_PLATE, SHAPE_FIT)
    model = cg.DepthModel(depth, bg, K_PLATE, roi)
    lower = np.array([-3.0, -3.0, -1.0, -3.0]) if lower is None else lower
    return cg.fit_slot_part(model, T_SLOT, [members], np.eye(4), lower, -lower if upper is None else upper)


def test_fit_recovers_yaw_hypothesis_and_offsets_within_bounds():
    mesh, depth, bg = fit_scene([1.2, -0.8, 0.3, 1.5], yaw_deg=180.0)
    fit, yaw, info = run_fit(depth, bg, mesh)
    assert yaw == 180.0
    assert fit.params == pytest.approx([1.2, -0.8, 0.3, 1.5], abs=0.25)
    assert fit.at_bound == []
    assert info["n_yaw_classes"] == 4  # an L-shape has no symmetry
    assert info["yaw_margin"] > 0.5


def test_fit_flags_corrections_that_hit_a_bound():
    mesh, depth, bg = fit_scene([5.0, 0.0, 0.0, 0.0], yaw_deg=0.0)
    fit, _, _ = run_fit(depth, bg, mesh)
    assert "x" in fit.at_bound
    assert fit.params[0] == pytest.approx(3.0, abs=0.06)
    assert np.all(np.abs(fit.params) <= np.array([3.0, 3.0, 1.0, 3.0]) + 1e-9)  # never leaves the bounds


def test_symmetric_parts_collapse_hypotheses_instead_of_flagging_ambiguity():
    from test_marker_ground_truth import box_mesh

    mesh, depth, bg = fit_scene([0.0, 0.0, 0.0, 0.0], yaw_deg=0.0, mesh=box_mesh(40.0, 40.0, 20.0))
    _, yaw, info = run_fit(depth, bg, mesh)
    assert info["n_yaw_classes"] == 1  # a square box looks the same at 0/90/180/270
    assert yaw == 0.0


def test_variants_pick_the_turned_arrangement_of_linked_meshes():
    base = l_mesh()
    bar = mgt.Mesh(base.vertices * [0.4, 0.4, 0.5] + [0, 0, 30], base.faces)  # a second, small L-shaped part on top
    turned = [(base, np.eye(4)), (bar, cg.pose_from_rz(90.0))]
    straight = [(base, np.eye(4)), (bar, np.eye(4))]
    T = cg.slot_pose(T_SLOT, np.zeros(4), 0.0)
    v, u = np.mgrid[0 : SHAPE_FIT[0], 0 : SHAPE_FIT[1]]
    rays = np.stack([(u + 0.5 - K_PLATE[0, 2]) / K_PLATE[0, 0], (v + 0.5 - K_PLATE[1, 2]) / K_PLATE[1, 1], np.ones(u.shape)], axis=-1)
    depth = ((T_SLOT[:3, 3] @ NORMAL) / (rays @ NORMAL)).astype(np.float32)
    for mesh, Tm in turned:
        depth = np.minimum(depth, mgt.render_depth(mesh, T @ Tm, K_PLATE, SHAPE_FIT)).astype(np.float32)
    bg = ((T_SLOT[:3, 3] @ NORMAL) / (rays @ NORMAL)).astype(np.float32)
    roi = cg.union_roi([[(m, T @ Tm) for m, Tm in straight]], K_PLATE, SHAPE_FIT, margin_px=40)
    model = cg.DepthModel(depth, bg, K_PLATE, roi)
    lower = np.array([-3.0, -3.0, -1.0, -3.0])
    fit, yaw, info = cg.fit_slot_part(model, T_SLOT, [straight, turned], np.eye(4), lower, -lower)
    assert info["variant"] == 1 and yaw == 0.0
    assert np.abs(fit.params).max() < 0.3


def test_pattern_search_respects_bounds():
    x, c, _ = cg.pattern_search(
        lambda p: float(np.sum((p - 10.0) ** 2)), np.zeros(2), np.array([-1.0, -2.0]), np.array([1.0, 2.0]), np.ones(2), np.full(2, 0.01)
    )
    assert x == pytest.approx([1.0, 2.0])


# ---------------------------------------------------------------------------
# Masks: occlusion and missing depth
# ---------------------------------------------------------------------------


def twin_part(mesh, T, kind="U", marker_id=-1, T_marker_part=None):
    spec = mgt.PartSpec(marker_id, "label", mesh, np.eye(4) if T_marker_part is None else T_marker_part)
    return cg.TwinPart("L1", kind, spec, T, T)


def test_masks_drop_pixels_with_nearer_surfaces_but_keep_pixels_without_depth():
    mesh, depth, bg = fit_scene([0.0, 0.0, 0.0, 0.0], yaw_deg=0.0)
    T = cg.slot_pose(T_SLOT, np.zeros(4), 0.0)
    ys, xs = np.nonzero(np.isfinite(mgt.render_depth(mesh, T, K_PLATE, SHAPE_FIT)))
    x_mid = int((xs.min() + xs.max()) / 2)
    y_mid = int((ys.min() + ys.max()) / 2)
    measured = depth.copy()
    measured[:, x_mid:] = 600.0  # the arm: much closer than the part, right half
    measured[:y_mid, :x_mid] = np.nan  # black part: no depth on the upper left quarter
    sample = mgt.Sample("s", np.zeros(SHAPE_FIT + (3,), np.uint8), np.stack([np.zeros_like(measured), np.zeros_like(measured), measured], -1), K_PLATE)
    part = twin_part(mesh, T)
    cg.compute_twin_masks([part], sample)
    assert part.bbox[2] <= x_mid + 1  # the right half is gone, the box shrinks to the visible part
    assert not part.mask[:, x_mid + 1 :].any()
    assert part.mask[ys.min() + 3 : y_mid - 3, xs.min() + 3 : x_mid - 3].all()  # pixels without depth stay
    assert part.occluded_fraction == pytest.approx(0.5, abs=0.1)
    assert any("partly occluded" in n for n in part.notes)


def test_nearer_part_wins_in_the_shared_z_buffer():
    mesh, depth, _ = fit_scene([0.0, 0.0, 0.0, 0.0], yaw_deg=0.0)
    near = twin_part(mesh, cg.slot_pose(T_SLOT, np.array([0.0, 0.0, 12.0, 0.0]), 0.0))
    far = twin_part(mesh, cg.slot_pose(T_SLOT, np.array([10.0, 0.0, 0.0, 0.0]), 0.0))
    sample = mgt.Sample("s", np.zeros(SHAPE_FIT + (3,), np.uint8), None, K_PLATE)
    cg.compute_twin_masks([near, far], sample)
    assert not (near.mask & far.mask).any()
    assert near.mask.sum() > far.mask.sum()


# ---------------------------------------------------------------------------
# Marker check
# ---------------------------------------------------------------------------


def test_marker_check_reports_the_offset_between_prediction_and_detection():
    T_true = part_pose(x_mm=10.0, yaw_deg=20.0)
    sample, spec, _ = make_scene([T_true])
    markers = mgt.detect_markers(sample.rgb, "DICT_4X4_50")
    part = cg.TwinPart("L1", "O", spec, T_true, T_true)
    part.support_px = 10000
    assert cg.check_markers([part], markers, sample, MARKER_MM) == []
    check = part.marker_check
    assert check["status"] == "ok" and check["pose_method"] == "depth_plane"
    assert check["distance_mm"] < 1.0
    assert check["tilt_deg"] < 2.0

    shifted = cg.TwinPart("L1", "O", spec, T_true @ cg.translate(3.0, 0.0, 0.0), T_true)  # ground truth 3 mm off in x
    shifted.support_px = 10000
    cg.check_markers([shifted], markers, sample, MARKER_MM)
    assert shifted.marker_check["distance_mm"] == pytest.approx(3.0, abs=1.0)


def test_marker_check_without_a_nearby_marker_says_so():
    T_true = part_pose()
    sample, spec, _ = make_scene([T_true])
    part = cg.TwinPart("L1", "O", spec, T_true @ cg.translate(200.0, 0.0, 0.0), T_true)
    part.support_px = 10000
    cg.check_markers([part], mgt.detect_markers(sample.rgb, "DICT_4X4_50"), sample, MARKER_MM)
    assert "no marker" in part.marker_check["status"]


def test_marker_offsets_give_median_and_spread_per_label():
    def report(dx, yaw):
        return {"parts": [{"label": "O", "marker_check": {"dx_mm": dx, "dy_mm": 0.0, "dz_mm": 0.0, "tilt_deg": 1.0, "yaw_deg": yaw, "pose_method": "depth_plane"}}]}

    out = cg.marker_offsets([report(0.1, 89.0), report(0.3, 91.0), report(0.2, 90.0)])
    assert out["O"]["n"] == 3
    assert out["O"]["median_offset"]["dx_mm"] == pytest.approx(0.2)
    assert out["O"]["median_offset"]["yaw_deg"] == pytest.approx(90.0, abs=1.0)


# ---------------------------------------------------------------------------
# Transforms and configuration
# ---------------------------------------------------------------------------


def test_average_transforms_and_rotation_helpers():
    a = cg.pose_from_rz(10.0, np.array([1.0, 0, 0]))
    b = cg.pose_from_rz(20.0, np.array([3.0, 0, 0]))
    m = cg.average_transforms([a, b])
    assert m[0, 3] == pytest.approx(2.0)
    assert cg.yaw_of(m[:3, :3]) == pytest.approx(15.0, abs=0.1)
    assert cg.rotation_angle_deg(a[:3, :3].T @ b[:3, :3]) == pytest.approx(10.0)
    assert cg.wrap_deg(190.0) == pytest.approx(-170.0)
    assert cg.wrap_deg(100.0, 180.0) == pytest.approx(-80.0)


def test_cell_config_builds_gui_view_with_all_parts(tmp_path):
    from test_marker_ground_truth import box_mesh, write_binary_stl

    for name in ("u", "h", "o"):
        write_binary_stl(tmp_path / f"{name}.stl", box_mesh(50.0, 50.0, 25.0))
    cfg = {
        "marker_size_mm": 15.0,
        "steps": {
            "poty": {
                "U": {"label": "U_label", "mesh": "u.stl"},
                "H": {"label": "H_label", "mesh": "h.stl"},
                "O": {"label": "O_label", "mesh": "o.stl", "marker_id": 1, "marker_on_top_face": True},
            }
        },
    }
    (tmp_path / "cell.json").write_text(json.dumps(cfg))
    loaded = cg.load_cell_config(tmp_path / "cell.json")
    assert loaded.class_names == ["U_label", "H_label", "O_label"]
    assert {s.label for s in loaded.config.parts.values()} == {"U_label", "H_label", "O_label"}
    assert 1 in loaded.config.parts  # the Oberteil is reachable by its marker ID
    # Oberteil origin sits on the Halter top face
    assert loaded.steps["poty"].T_halter_oberteil[2, 3] == pytest.approx(25.0)
    import review_gui

    assert review_gui.load_any_config(tmp_path / "cell.json").class_names == loaded.class_names


def test_weak_plates_and_plate_shifts_apply_in_the_slot_frame():
    T = cg.pose_from_rz(0.0, np.array([100.0, 200.0, 1000.0]))
    cal = cg.Calibration(
        {"P1": {"L1": T, "R1": T}, "P2": {"L1": T, "R1": T}}, {}, {}, {}, {},
        {"P1": {"L": 9, "R": 9}, "P2": {"L": 0, "R": 8}},
    )
    assert cg.weak_plates(cal) == [("P2", "L")]
    shifts = {("P2", "L"): cg.PlateShift(0.5, -2.0, 6, 2, "test")}
    slots, record = cg.slots_for(cal, {"L": "P2", "R": "P1"}, shifts)
    assert slots["L1"][:3, 3] == pytest.approx([100.5, 198.0, 1000.0])  # shifted along the slot x and y axes
    assert slots["R1"][:3, 3] == pytest.approx([100.0, 200.0, 1000.0])
    assert record["L"]["dy_mm"] == -2.0 and "R" not in record
    unshifted, none = cg.slots_for(cal, {"L": "P1", "R": "P1"}, shifts)  # the shift belongs to position P2 only
    assert unshifted["L1"][1, 3] == 200.0 and none == {}


def test_layout_code_uo_is_not_confused_with_u_or_o():
    assert cg.parse_layout("L7 H; R1 UO; R6 U; L2,L3 HO") == {"L7": "H", "R1": "UO", "R6": "U", "L2": "HO", "L3": "HO"}
    assert cg.parse_layout("L5 UO; R2 UO; others empty") == {"L5": "UO", "R2": "UO"}


def test_oberteil_slides_along_the_groove_and_the_fit_finds_where():
    unterteil = l_mesh()
    bar = mgt.Mesh(np.array([[x, y, z] for z in (0, 12) for y in (-5, 5) for x in (-12, 12)], float), l_mesh().faces[:12])
    joint = {"z_mm": 20.0, "y_mm": 0.0, "y_half_range_mm": 9.0}
    T_u = cg.slot_pose(T_SLOT, np.array([0.0, 0.0, 0.0, 0.0]), 0.0)
    true = T_u @ cg.translate(0.4, 6.5, 20.3)
    v, u = np.mgrid[0 : SHAPE_FIT[0], 0 : SHAPE_FIT[1]]
    rays = np.stack([(u + 0.5 - K_PLATE[0, 2]) / K_PLATE[0, 0], (v + 0.5 - K_PLATE[1, 2]) / K_PLATE[1, 1], np.ones(u.shape)], axis=-1)
    bg = ((T_SLOT[:3, 3] @ NORMAL) / (rays @ NORMAL)).astype(np.float32)
    depth = np.minimum(bg, mgt.render_depth(unterteil, T_u, K_PLATE, SHAPE_FIT))
    depth = np.minimum(depth, mgt.render_depth(bar, true, K_PLATE, SHAPE_FIT)).astype(np.float32)
    roi = cg.union_roi([[(unterteil, T_u)], [(bar, T_u @ cg.translate(0, -9.0, 20.0))], [(bar, T_u @ cg.translate(0, 9.0, 20.0))]], K_PLATE, SHAPE_FIT)
    model = cg.DepthModel(depth, bg, K_PLATE, roi)
    model.set_fixed([(unterteil, T_u)])
    T_o, fit, info = cg.fit_oberteil_in_unterteil(model, T_u, bar, joint)
    assert np.linalg.norm(T_o[:3, 3] - true[:3, 3]) < 0.4
    assert info["y_in_groove_mm"] == pytest.approx(6.5, abs=0.4)
    assert fit.at_bound == []
    # an Oberteil outside the play is reported at the bound
    far = T_u @ cg.translate(0.0, 14.0, 20.0)
    depth2 = np.minimum(np.minimum(bg, mgt.render_depth(unterteil, T_u, K_PLATE, SHAPE_FIT)), mgt.render_depth(bar, far, K_PLATE, SHAPE_FIT)).astype(np.float32)
    model2 = cg.DepthModel(depth2, bg, K_PLATE, roi)
    model2.set_fixed([(unterteil, T_u)])
    _, fit2, _ = cg.fit_oberteil_in_unterteil(model2, T_u, bar, joint)
    assert "y" in fit2.at_bound


def test_marker_chooses_between_the_two_equivalent_oberteil_poses():
    T_true = part_pose(x_mm=5.0, yaw_deg=30.0)
    sample, spec, _ = make_scene([T_true])
    pose = mgt.estimate_marker_pose(6, mgt.detect_markers(sample.rgb, "DICT_4X4_50")[0][1], sample, MARKER_MM)
    turned = cg.TwinPart("L1", "O", spec, T_true @ cg.pose_from_rz(180.0), T_true)
    cg.orient_from_marker(turned, pose)
    assert pose_errors(turned.T_cam_part, T_true)[1] < 3.0 and turned.refinement["turned_180_by_marker"]
    right = cg.TwinPart("L1", "O", spec, T_true, T_true)
    cg.orient_from_marker(right, pose)
    assert pose_errors(right.T_cam_part, T_true)[1] < 3.0 and "turned_180_by_marker" not in right.refinement
    cg.orient_from_marker(right, None)  # no marker: nothing changes


def test_review_gui_card_lines_show_corrections_and_marker_check():
    import review_gui

    html_lines = review_gui._twin_lines(
        {
            "refinement": {"correction": {"dx_mm": 0.5, "dy_mm": -0.25, "dz_mm": 0.1, "dyaw_deg": 1.0}, "yaw_hypothesis_deg": 90.0, "stage": "single"},
            "marker_check": {"dx_mm": 0.1, "dy_mm": 0.2, "dz_mm": 0.3, "tilt_deg": 1.5, "yaw_deg": -0.5, "pose_method": "depth_plane"},
        }
    )
    assert "dx +0.50" in html_lines and "marker check" in html_lines and "depth_plane" in html_lines
    assert "no marker found" in review_gui._twin_lines({"marker_check": {"status": "no marker found near the predicted position"}})
