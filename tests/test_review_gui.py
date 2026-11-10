import json
import threading
import urllib.error
import urllib.request
from urllib.parse import quote

import cv2
import numpy as np
import pytest

import marker_ground_truth as mgt
import review_gui as rg
from test_marker_ground_truth import K, SHAPE, box_mesh, make_scene, part_pose, write_binary_stl


def test_box_mesh_has_twelve_sharp_edges():
    edges = rg.sharp_edges(box_mesh())
    assert edges.shape == (12, 2)
    v = box_mesh().vertices
    lengths = sorted(np.round(np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1), 6))
    assert lengths == [20.0] * 4 + [40.0] * 8  # no face diagonals


def test_quaternion_round_trip():
    R = part_pose(yaw_deg=37.0)[:3, :3]
    assert np.allclose(rg.quaternion_wxyz_to_rotation(mgt.rotation_to_quaternion_wxyz(R)), R, atol=1e-9)


def test_wireframe_lies_on_rendered_silhouette():
    mesh = box_mesh()
    T = part_pose(20.0, -10.0, yaw_deg=25.0)
    depth = mgt.render_depth(mesh, T, K, SHAPE)
    silhouette = np.isfinite(depth)
    zmax = cv2.dilate(depth.astype(np.float32), np.ones((3, 3), np.uint8))
    verts = (T[:3, :3] @ mesh.vertices.T).T + T[:3, 3]
    visible, hidden = rg.edge_polylines(verts, rg.sharp_edges(mesh), K, zmax)
    pts = np.vstack(visible)
    # Every visible edge point is on the silhouette (1 px tolerance) ...
    near = cv2.dilate(silhouette.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ix, iy = np.floor(pts[:, 0]).astype(int), np.floor(pts[:, 1]).astype(int)
    assert near[iy, ix].all()
    # ... and the silhouette outline is covered by the wireframe.
    contours, _ = cv2.findContours(silhouette.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = contours[0].reshape(-1, 2) + 0.5  # pixel centres
    dense = np.vstack([np.linspace(p[:-1], p[1:], 5, axis=0).reshape(-1, 2) for p in visible])
    dist = np.min(np.linalg.norm(outline[:, None, :] - dense[None, :, :], axis=2), axis=1)
    assert np.percentile(dist, 99) < 1.5
    # Seen from above, the four bottom edges are hidden behind the top face.
    assert hidden


def test_review_round_trip(tmp_path):
    rg.update_review(tmp_path, "s0", "Marc", gt_created="t1", part_index=1, part_label="A", part_marker_id=6,
                     part_verdict="pose_wrong", part_note="shifted 2 mm")
    rg.update_review(tmp_path, "s0", "Marc", gt_created="t1", part_index=0, part_verdict="correct")
    rg.update_review(tmp_path, "s0", "Marc", gt_created="t1", part_index=1, part_note="shifted 3 mm")
    review = rg.update_review(tmp_path, "s0", "Marc", gt_created="t1", capture_verdict="reject", capture_note="bad")
    loaded = json.loads((tmp_path / "review" / "s0.json").read_text())
    assert loaded == review == rg.load_review(tmp_path, "s0")
    assert [p["index"] for p in loaded["parts"]] == [0, 1]
    assert loaded["parts"][1]["verdict"] == "pose_wrong"  # a note alone keeps the verdict
    assert loaded["parts"][1]["note"] == "shifted 3 mm"
    assert loaded["parts"][1]["label"] == "A"
    assert loaded["capture"]["verdict"] == "reject"
    assert loaded["reviewer"] == "Marc" and "T" in loaded["updated"]
    assert rg.review_status(loaded, 2, "t1") == "rejected"
    assert rg.review_status(loaded, 2, "t2").startswith("stale")
    with pytest.raises(ValueError):
        rg.update_review(tmp_path, "s0", "Marc", part_index=0, part_verdict="maybe")
    with pytest.raises(ValueError):
        rg.update_review(tmp_path, "s0", " ", capture_verdict="accept")


def _make_run(tmp_path, poses):
    """Sample folder, config and ground-truth run of a synthetic scene."""
    sample, spec, _ = make_scene(poses)
    samples, run = tmp_path / "samples", tmp_path / "run"
    samples.mkdir()
    cv2.imwrite(str(samples / "s0_rgb.png"), sample.rgb)
    np.save(samples / "s0_xyz.npy", sample.xyz)
    np.savetxt(samples / "s0_cam_K.txt", K)
    write_binary_stl(tmp_path / "box.stl", spec.mesh)
    cfg = {"marker_size_mm": 30.0, "parts": {"6": {"label": spec.label, "mesh": "box.stl", "T_marker_part": spec.T_marker_part.tolist()}}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    config = mgt.load_config(tmp_path / "cfg.json")
    mgt.process_sample(mgt.load_sample(samples, "s0"), config, run)
    return samples, run, config


def test_capture_view_matches_ground_truth_and_tolerates_missing_mask(tmp_path):
    samples, run, config = _make_run(tmp_path, [part_pose(-60, 0, 10), part_pose(60, 20, -30)])
    (run / "masks" / "s0_det_1.png").unlink()
    view = rg.load_capture(run, samples, "s0", config)
    assert len(view.parts) == 2
    assert view.parts[0].mask is not None and view.parts[1].mask_missing
    for part in view.parts:
        assert abs(part.residual_bias_mm) < 0.5  # model agrees with the synthetic depth
        assert part.wire_visible
    img = rg.render_view(view, rg.LAYERS, "original", 1.0)
    assert img.shape == SHAPE + (3,)
    crop = rg.render_view(view, rg.LAYERS, "filled", 3.0, rg.part_roi(view, 1))
    assert crop.shape[0] > 100


def test_only_part_draws_wireframe_and_label_of_that_part_only(tmp_path):
    samples, run, config = _make_run(tmp_path, [part_pose(-60, 0, 10), part_pose(60, 20, -30)])
    view = rg.load_capture(run, samples, "s0", config)
    plain = rg.render_view(view, [], "filled", 1.0)
    only_first = rg.render_view(view, rg.ZOOM_LAYERS, "filled", 1.0, None, 0)
    both = rg.render_view(view, rg.ZOOM_LAYERS, "filled", 1.0)

    def changed(img, index):
        x0, y0, x1, y1 = rg.part_roi(view, index, 0)
        return bool((img[y0:y1, x0:x1] != plain[y0:y1, x0:x1]).any())

    assert changed(only_first, 0) and not changed(only_first, 1)
    assert changed(both, 0) and changed(both, 1)


def test_capture_without_parts_renders(tmp_path):
    samples, run, config = _make_run(tmp_path, [])
    view = rg.load_capture(run, samples, "s0", config)
    assert view.parts == []
    assert rg.render_view(view, rg.LAYERS, "filled", 0.5).shape == (SHAPE[0] // 2, SHAPE[1] // 2, 3)
    assert rg.scan_run(run)[0]["n_parts"] == 0


def test_server_serves_list_page_image_and_review(tmp_path):
    samples, run, config = _make_run(tmp_path, [part_pose(0, 0, 25)])
    server = rg.make_server(rg.ReviewApp(run, samples, config), port=0)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "s0" in page and "accepted" in page
        capture = urllib.request.urlopen(base + "/capture/" + quote("s0")).read().decode()
        assert "Einlegen_WI_RZ_Oberteil" in capture
        img = urllib.request.urlopen(base + "/img/s0?layers=masks,wire,axes,bbox,resid&w=320").read()
        decoded = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (240, 320, 3)
        req = urllib.request.Request(
            base + "/api/review/s0",
            data=json.dumps({"reviewer": "Marc", "part_index": 0, "part_verdict": "correct"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        review = json.loads(urllib.request.urlopen(req).read())
        assert review["parts"][0]["verdict"] == "correct"
        assert (run / "review" / "s0.json").exists()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/capture/..%2Fetc")
    finally:
        server.shutdown()
        server.server_close()


def test_open_parts_are_counted_and_filtered(tmp_path):
    samples, run, config = _make_run(tmp_path, [part_pose(-60, 0, 10), part_pose(60, 20, -30)])
    assert rg.scan_run(run)[0]["open_parts"] == 2
    rg.update_review(run, "s0", "me", part_index=0, part_verdict="correct")
    assert rg.scan_run(run)[0]["open_parts"] == 1
    rg.update_review(run, "s0", "me", part_index=1, part_verdict="pose_wrong")
    assert rg.scan_run(run)[0]["open_parts"] == 0
    rows = rg.scan_run(run)
    assert "s0" in rg.render_index(run, rows) and "s0" not in rg.render_index(run, rows, only_open=True).split("<table>")[1]
    rg.update_review(run, "s0", "me", capture_verdict="reject")
    assert rg.scan_run(run)[0]["open_parts"] == 0  # a rejected capture needs no part verdicts


def test_capture_page_in_open_mode_flags_the_data_and_keeps_the_filter_in_links(tmp_path):
    samples, run, config = _make_run(tmp_path, [part_pose(0, 0, 25)])
    view = rg.load_capture(run, samples, "s0", config)
    page = rg.render_capture_page(view, rg.load_review(run, "s0"), ["s0"], "me", only_open=True)
    assert '"open": true' in page and "/?open=1" in page
    assert '"open": false' in rg.render_capture_page(view, rg.load_review(run, "s0"), ["s0"], "me")
