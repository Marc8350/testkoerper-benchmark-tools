import json
import struct

import cv2
import numpy as np
import pytest

import marker_ground_truth as mgt

K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
SHAPE = (480, 640)
MARKER_MM = 30.0
TABLE_MM = 520.0


def box_mesh(sx=40.0, sy=40.0, sz=20.0):
    x, y = sx / 2, sy / 2
    v = np.array(
        [[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0], [-x, -y, sz], [x, -y, sz], [x, y, sz], [-x, y, sz]],
        dtype=float,
    )
    f = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    )
    return mgt.Mesh(v, f)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def part_pose(x_mm=0.0, y_mm=0.0, yaw_deg=25.0):
    """Part standing on the table, Z up (towards the camera), camera looks down."""
    R = np.diag([1.0, -1.0, -1.0]) @ rot_z(yaw_deg)
    return mgt.make_transform(R, np.array([x_mm, y_mm, TABLE_MM]))


def xyz_from_depth(depth):
    v, u = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    x = (u + 0.5 - K[0, 2]) * depth / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, depth], axis=-1).astype(np.float32)


def draw_marker(rgb, marker_id, T_cam_marker, size_mm=MARKER_MM):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tile = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)
    h = size_mm / 2
    corners_m = np.array([[-h, h, 0, 1], [h, h, 0, 1], [h, -h, 0, 1], [-h, -h, 0, 1]]).T
    corners_cam = (T_cam_marker @ corners_m)[:3].T
    dst = mgt.project_points(K, corners_cam).astype(np.float32) - 0.5
    src = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR), H, (SHAPE[1], SHAPE[0]), flags=cv2.INTER_LINEAR)
    region = cv2.warpPerspective(np.full((200, 200), 255, np.uint8), H, (SHAPE[1], SHAPE[0])) > 0
    rgb[region] = warped[region]
    return dst


def make_scene(poses, with_depth=True, marker_ids=None):
    """Grey table, light part tops, one marker per part. Returns sample, spec, depth."""
    mesh = box_mesh()
    T_marker_part = mgt.marker_offset_from_top_face(mesh, yaw_deg=0.0)
    rgb = np.full(SHAPE + (3,), 60, np.uint8)
    depth = np.full(SHAPE, TABLE_MM)
    for T in poses:
        d = mgt.render_depth(mesh, T, K, SHAPE)
        hit = np.isfinite(d) & (d < depth)
        depth[hit] = d[hit]
        rgb[np.isfinite(d)] = 200
    marker_ids = marker_ids or [6] * len(poses)
    for marker_id, T in zip(marker_ids, poses):
        draw_marker(rgb, marker_id, T @ mgt.invert_transform(T_marker_part))
    sample = mgt.Sample("s0", rgb, xyz_from_depth(depth) if with_depth else None, K)
    spec = mgt.PartSpec(6, "Einlegen_WI_RZ_Oberteil", mesh, T_marker_part)
    return sample, spec, depth


def config_for(spec):
    return mgt.Config("DICT_4X4_50", MARKER_MM, {spec.marker_id: spec}, [spec.label])


def pose_errors(T_est, T_true):
    dt = np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])
    dR = T_est[:3, :3].T @ T_true[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return dt, angle


def write_binary_stl(path, mesh):
    tris = mesh.vertices[mesh.faces]
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", 0, 0, 0) + t.astype("<f4").tobytes() + b"\0\0")


def test_stl_binary_and_ascii_load_same_mesh(tmp_path):
    mesh = box_mesh()
    write_binary_stl(tmp_path / "b.stl", mesh)
    lines = ["solid box"]
    for t in mesh.vertices[mesh.faces]:
        lines += ["facet normal 0 0 0", "outer loop"] + [f"vertex {a} {b} {c}" for a, b, c in t] + ["endloop", "endfacet"]
    (tmp_path / "a.stl").write_text("\n".join(lines + ["endsolid box"]))
    for name in ("a.stl", "b.stl"):
        loaded = mgt.load_stl(tmp_path / name)
        assert loaded.vertices.shape == (8, 3)
        assert loaded.faces.shape == (12, 3)
        assert np.allclose(np.sort(loaded.vertices, axis=0), np.sort(mesh.vertices, axis=0))


def test_quaternion_and_euler_describe_the_rotation():
    R = rot_z(30) @ np.array([[1, 0, 0], [0, np.cos(0.4), -np.sin(0.4)], [0, np.sin(0.4), np.cos(0.4)]])
    w, x, y, z = mgt.rotation_to_quaternion_wxyz(R)
    Rq = np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    )
    assert np.allclose(Rq, R, atol=1e-9)
    a, b, c = np.radians(mgt.rotation_to_euler_xyz_deg(R))
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    assert np.allclose(Rz @ Ry @ Rx, R, atol=1e-9)


def test_render_depth_shows_top_face_at_its_distance():
    depth = mgt.render_depth(box_mesh(), part_pose(yaw_deg=0), K, SHAPE)
    assert depth[240, 320] == pytest.approx(TABLE_MM - 20.0, abs=1e-6)
    ys, xs = np.nonzero(np.isfinite(depth))
    # 40 mm top face at 500 mm is 64 px wide; the side walls reach slightly beyond.
    assert 64 <= xs.max() - xs.min() + 1 <= 70


def test_marker_offset_puts_marker_on_top_face_centre():
    T_part_marker = mgt.invert_transform(mgt.marker_offset_from_top_face(box_mesh(), yaw_deg=90))
    assert np.allclose(T_part_marker[:3, 3], [0, 0, 20])
    assert np.allclose(T_part_marker[:3, 0], [0, 1, 0], atol=1e-12)


def test_marker_pose_from_depth_recovers_part_pose():
    T_true = part_pose(x_mm=15.0, y_mm=-10.0, yaw_deg=25.0)
    sample, spec, _ = make_scene([T_true])
    corners = dict(mgt.detect_markers(sample.rgb, "DICT_4X4_50"))[6]
    marker = mgt.estimate_marker_pose(6, corners, sample, MARKER_MM)
    assert marker.method == "depth_plane"
    dt, angle = pose_errors(marker.T_cam_marker @ spec.T_marker_part, T_true)
    assert dt < 1.0
    assert angle < 1.0


def test_marker_pose_falls_back_to_pnp_without_depth():
    T_true = part_pose(yaw_deg=-40.0)
    sample, spec, _ = make_scene([T_true], with_depth=False)
    corners = dict(mgt.detect_markers(sample.rgb, "DICT_4X4_50"))[6]
    marker = mgt.estimate_marker_pose(6, corners, sample, MARKER_MM)
    assert marker.method == "pnp"
    dt, angle = pose_errors(marker.T_cam_marker @ spec.T_marker_part, T_true)
    # PnP on a ~48 px, nearly camera-facing marker gets distance only to a few
    # mm and its tilt is ambiguous by several degrees. That is why the report
    # flags PnP poses and rejects such samples for the test set.
    assert dt < 5.0
    assert angle < 10.0


def test_pnp_poses_reject_the_sample(tmp_path):
    sample, spec, _ = make_scene([part_pose()], with_depth=False)
    report = mgt.process_sample(sample, config_for(spec), tmp_path)
    assert report["rejected"] is True
    assert "pose_from_pnp" in report["parts"][0]["flags"]


def test_process_sample_writes_consistent_ground_truth(tmp_path):
    T_true = part_pose(x_mm=10.0, yaw_deg=10.0)
    sample, spec, _ = make_scene([T_true])
    report = mgt.process_sample(sample, config_for(spec), tmp_path, expected={spec.label: 1})

    assert report["rejected"] is False
    part = report["parts"][0]
    assert part["depth_residual_mm"] < 0.5
    silhouette = np.isfinite(mgt.render_depth(spec.mesh, T_true, K, SHAPE))
    ys, xs = np.nonzero(silhouette)
    true_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    assert np.abs(np.array(part["bbox_2d"]) - true_box).max() <= 1

    label = (tmp_path / "labels" / "s0.txt").read_text().split()
    assert label[0] == "0" and len(label) == 5
    assert (tmp_path / "klassen.txt").read_text() == spec.label + "\n"
    poses = json.loads((tmp_path / "annotations" / "s0_poses.json").read_text())
    assert poses["frame"] == "CAM" and poses["poses"][0]["uncertainty"] is None
    dets = json.loads((tmp_path / "annotations" / "s0_detections.json").read_text())
    assert dets["detections"][0]["bbox_2d"] == part["bbox_2d"]

    inpainted = cv2.imread(str(tmp_path / "images" / "s0.png"))
    marker_mask = cv2.imread(str(tmp_path / "masks" / "s0_markers.png"), cv2.IMREAD_GRAYSCALE) > 0
    assert np.array_equal(inpainted[~marker_mask], sample.rgb[~marker_mask])
    assert mgt.detect_markers(inpainted, "DICT_4X4_50") == []


def test_parts_hidden_by_unmodelled_objects_lose_those_pixels():
    T_true = part_pose(yaw_deg=0.0)
    sample, spec, depth = make_scene([T_true])
    occluded = depth.copy()
    occluded[:, 330:] = 400.0  # e.g. the robot arm, closer than the part
    sample.xyz = xyz_from_depth(occluded)
    marker = mgt.MarkerPose(6, np.zeros((4, 2)), T_true @ mgt.invert_transform(spec.T_marker_part), "depth_plane", 100)
    part = mgt.PartGroundTruth(marker, spec, T_true)
    mgt.compute_visible_masks([part], sample)
    assert part.bbox is not None
    assert part.bbox[2] <= 330
    assert not part.mask[:, 330:].any()


def test_nearer_part_wins_where_parts_overlap():
    near = part_pose(yaw_deg=0.0)
    near[2, 3] = TABLE_MM - 30.0
    far = part_pose(x_mm=20.0, yaw_deg=0.0)
    sample, spec, _ = make_scene([far])
    sample.xyz = None
    marker = mgt.MarkerPose(6, np.zeros((4, 2)), np.eye(4), "pnp", 0)
    parts = [mgt.PartGroundTruth(marker, spec, T) for T in (near, far)]
    mgt.compute_visible_masks(parts, sample)
    assert not (parts[0].mask & parts[1].mask).any()
    assert parts[0].mask[240, 330] and not parts[1].mask[240, 330]


def test_expected_mismatch_rejects_the_sample(tmp_path):
    sample, spec, _ = make_scene([part_pose()])
    report = mgt.process_sample(sample, config_for(spec), tmp_path, expected={spec.label: 2})
    assert report["rejected"] is True
    assert report["expected_mismatch"] == {spec.label: {"expected": 2, "found": 1}}


def test_unknown_markers_are_reported_and_still_filled_in(tmp_path):
    sample, spec, _ = make_scene([part_pose(x_mm=-40.0), part_pose(x_mm=40.0)], marker_ids=[6, 9])
    report = mgt.process_sample(sample, config_for(spec), tmp_path)
    assert report["unknown_marker_ids"] == [9]
    assert [p["marker_id"] for p in report["parts"]] == [6]
    inpainted = cv2.imread(str(tmp_path / "images" / "s0.png"))
    assert mgt.detect_markers(inpainted, "DICT_4X4_50") == []


def test_load_config_derives_top_face_offset(tmp_path):
    write_binary_stl(tmp_path / "box.stl", box_mesh())
    cfg = {
        "marker_size_mm": 15.0,
        "parts": {"6": {"label": "Einlegen_WI_RZ_Oberteil", "mesh": "box.stl", "marker_on_top_face": True, "yaw_deg": 90}},
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    config = mgt.load_config(tmp_path / "cfg.json")
    assert config.class_names == ["Einlegen_WI_RZ_Oberteil"]
    T_part_marker = mgt.invert_transform(config.parts[6].T_marker_part)
    assert np.allclose(T_part_marker[:3, 3], [0, 0, 20])


def test_load_config_rejects_missing_offset(tmp_path):
    write_binary_stl(tmp_path / "box.stl", box_mesh())
    cfg = {"marker_size_mm": 15.0, "parts": {"6": {"label": "x", "mesh": "box.stl"}}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="T_marker_part"):
        mgt.load_config(tmp_path / "cfg.json")


def test_load_sample_rejects_misaligned_xyz(tmp_path):
    cv2.imwrite(str(tmp_path / "a_rgb.png"), np.zeros((10, 12, 3), np.uint8))
    np.save(tmp_path / "a_xyz.npy", np.zeros((10, 11, 3), np.float32))
    np.savetxt(tmp_path / "a_cam_K.txt", K)
    assert mgt.list_sample_ids(tmp_path) == ["a"]
    with pytest.raises(ValueError, match="pixel-aligned"):
        mgt.load_sample(tmp_path, "a")


def test_copies_with_the_same_marker_id_each_get_ground_truth(tmp_path):
    sample, spec, _ = make_scene([part_pose(x_mm=-45.0), part_pose(x_mm=45.0)])
    markers = mgt.detect_markers(sample.rgb, "DICT_4X4_50")
    assert [marker_id for marker_id, _ in markers] == [6, 6]
    report = mgt.process_sample(sample, config_for(spec), tmp_path, expected={spec.label: 2})
    assert report["rejected"] is False
    assert len(report["parts"]) == 2
    assert (tmp_path / "labels" / "s0.txt").read_text().count("\n") == 2


def test_surface_fill_leaves_no_bright_sticker_on_a_black_part():
    rng = np.random.default_rng(1)
    rgb = np.clip(rng.normal(28, 2, (200, 300, 3)), 0, 255).astype(np.uint8)
    rgb[:, 240:] = (60, 200, 150)  # bright plate next to the part
    centre, side = np.array([120.0, 100.0]), 36.0
    half = side / 2
    corners = centre + np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    cv2.rectangle(rgb, (int(centre[0] - half - 6), int(centre[1] - half - 6)), (int(centre[0] + half + 6), int(centre[1] + half + 6)), (190, 230, 200), -1)
    cv2.rectangle(rgb, (int(centre[0] - half), int(centre[1] - half)), (int(centre[0] + half), int(centre[1] + half)), (20, 20, 20), -1)

    assert 4.0 <= mgt.quiet_zone_px(rgb, corners) <= 7.0
    filled, mask = mgt.inpaint_markers(rgb, [corners], "surface", 3)

    assert np.array_equal(filled[mask == 0], rgb[mask == 0])
    assert abs(float(filled[mask > 0].mean()) - 28.0) < 4.0
    assert filled[mask > 0].max() < 60
