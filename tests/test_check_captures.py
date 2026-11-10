"""Offline tests for tools/check_captures.py. They never import zivid."""

import csv
import json

import cv2
import numpy as np
import pytest

import check_captures as cc

K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
SHAPE = (480, 640)


def marker_image(placements):
    """White image with DICT_4X4_50 markers. placements: [(id, x, y, side_px)]."""
    img = np.full((*SHAPE, 3), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, x, y, side in placements:
        m = cv2.aruco.generateImageMarker(d, marker_id, side, borderBits=1)
        img[y : y + side, x : x + side] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return img


def write_sample(samples_dir, stem, placements, settings="Engine: stripe\n", z=500.0, hole=False):
    samples_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(samples_dir / f"{stem}_rgb.png"), marker_image(placements))
    xyz = np.full((*SHAPE, 3), z, np.float32)
    if hole:
        xyz[:, :400] = np.nan
    np.save(samples_dir / f"{stem}_xyz.npy", xyz)
    np.savetxt(samples_dir / f"{stem}_cam_K.txt", K)
    (samples_dir / f"{stem}_meta.json").write_text(json.dumps({"sample_id": stem}))
    (samples_dir / f"{stem}_settings.yml").write_text(settings)


# --- names ------------------------------------------------------------------


def test_parse_capture_name():
    p = cc.parse_capture_name("S01_calib_oberteil_s1-1_r090_02")
    assert (p.session, p.block, p.scene, p.index) == ("S01", "calib", "oberteil_s1-1_r090", 2)
    assert p.group_key == "S01_calib_oberteil_s1-1_r090"
    assert cc.parse_capture_name("S01_rep_f01_30").index == 30


@pytest.mark.parametrize("stem", ["Einlegen_(WI_RZ)_Testbild", "S1_rep_f01_01", "S01_rep_f01", "s01_rep_f01_01"])
def test_parse_rejects_other_names(stem):
    assert cc.parse_capture_name(stem) is None


def test_marker_expectation_by_block():
    exp = lambda s: cc.marker_expectation(cc.parse_capture_name(s))  # noqa: E731
    assert exp("S01_empty_spots_01") == "none"
    assert exp("S01_tape_f05_01") == "none"
    assert exp("S01_calib_halter_s1-1_r000_01") == "none"
    assert exp("S01_calib_oberteil_s1-1_r000_01") == "some"
    assert exp("S01_rep_f01_05") == "some"
    assert exp("S01_test_idcheck_01") == "any"
    assert cc.marker_expectation(None) == "some"


# --- markers ----------------------------------------------------------------


def test_duplicate_ids_do_not_crash_and_are_counted():
    rgb = marker_image([(6, 50, 50, 40), (6, 200, 50, 40), (9, 350, 200, 40)])
    assert [i for i, _ in cc.mgt.detect_markers(rgb, "DICT_4X4_50")] == [6, 6, 9]
    markers, duplicates = cc.detect_all_markers(rgb, "DICT_4X4_50")
    assert duplicates
    assert sorted(i for i, _ in markers) == [6, 6, 9]
    assert cc.side_length_px(markers[0][1]) == pytest.approx(40, abs=2)


def test_single_ids_use_ground_truth_detector():
    markers, duplicates = cc.detect_all_markers(marker_image([(6, 50, 50, 40)]), "DICT_4X4_50")
    assert not duplicates and [i for i, _ in markers] == [6]


def test_valid_fraction_and_marker_depth():
    xyz = np.full((*SHAPE, 3), 500.0, np.float32)
    xyz[:, :320] = np.nan
    assert cc.valid_depth_fraction(xyz) == pytest.approx(0.5)
    inside = np.array([[400, 100], [440, 100], [440, 140], [400, 140]], float)
    assert cc.valid_depth_fraction(xyz, inside) == 1.0
    outside = np.array([[10, 100], [50, 100], [50, 140], [10, 140]], float)
    assert cc.valid_depth_fraction(xyz, outside) == 0.0


# --- per-capture warnings ---------------------------------------------------


def test_missing_expected_id_warns(tmp_path):
    write_sample(tmp_path, "S01_fill_f02_01", [(4, 50, 50, 40)])
    res = cc.analyse_capture(tmp_path, "S01_fill_f02_01", [6])
    assert any("expected marker ID [6] missing" in w and "4x1" in w for w in res.warnings)
    ok = cc.analyse_capture(tmp_path, "S01_fill_f02_01", [4])
    assert ok.warnings == []


def test_markers_in_empty_capture_warn(tmp_path):
    write_sample(tmp_path, "S01_empty_spots_01", [(6, 50, 50, 40)])
    res = cc.analyse_capture(tmp_path, "S01_empty_spots_01", [6])
    assert any("none expected" in w for w in res.warnings)


def test_low_valid_depth_and_small_marker_warn(tmp_path):
    write_sample(tmp_path, "S01_fill_f02_01", [(6, 450, 50, 20)], hole=True)
    res = cc.analyse_capture(tmp_path, "S01_fill_f02_01", [6])
    joined = " ".join(res.warnings)
    assert "low valid depth" in joined and "marker side" in joined


# --- blocks -----------------------------------------------------------------


def repeat_results(tmp_path, shifts):
    stems = []
    for n, dx in enumerate(shifts, start=1):
        stem = f"S01_rep_f01_{n:02d}"
        write_sample(tmp_path, stem, [(6, 50 + dx, 50, 40), (6, 250, 60, 40)])
        stems.append(stem)
    return [cc.analyse_capture(tmp_path, s, [6]) for s in stems]


def test_repeatability_within_limit(tmp_path):
    summary, warnings = cc.check_repeatability(repeat_results(tmp_path, [0, 0, 0]))
    assert warnings == [] and "3 captures, 2 markers" in summary


def test_repeatability_flags_shift_above_one_pixel(tmp_path):
    _, warnings = cc.check_repeatability(repeat_results(tmp_path, [0, 0, 3]))
    assert len(warnings) == 1 and "3." in warnings[0]


def test_repeatability_flags_missing_marker(tmp_path):
    results = repeat_results(tmp_path, [0, 0])
    write_sample(tmp_path, "S01_rep_f01_03", [(6, 50, 50, 40)])
    results.append(cc.analyse_capture(tmp_path, "S01_rep_f01_03", [6]))
    _, warnings = cc.check_repeatability(results)
    assert any("1 markers found, reference" in w for w in warnings)


def test_tape_pair_depth_check(tmp_path):
    write_sample(tmp_path, "S01_fill_f05_01", [(6, 50, 50, 40)])
    write_sample(tmp_path, "S01_tape_f05_01", [])
    assert cc.check_tape_pair(tmp_path, "S01_fill_f05_01", "S01_tape_f05_01") is None
    write_sample(tmp_path, "S01_tape_f05_01", [], z=510.0)
    assert "something moved" in cc.check_tape_pair(tmp_path, "S01_fill_f05_01", "S01_tape_f05_01")


def test_check_log_reports_both_directions(tmp_path):
    log = tmp_path / "capture_log.csv"
    with open(log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "block"])
        w.writerow(["S01_fill_f02_01.zdf", "fill"])
        w.writerow(["S01_fill_f02_02", "fill"])
    warnings = cc.check_log(["S01_fill_f02_01", "S01_fill_f02_03"], log)
    assert warnings == ["S01_fill_f02_03: no row in capture_log.csv", "S01_fill_f02_02: in capture_log.csv but no .zdf file"]
    # a file outside the checked pattern still counts as present
    assert cc.check_log(["S01_fill_f02_01"], log, ["S01_fill_f02_01", "S01_fill_f02_02"]) == []


# --- run() with a fake converter ---------------------------------------------


def test_run_converts_only_new_files_and_reports(tmp_path):
    zdf_dir, samples = tmp_path / "zdf", tmp_path / "samples"
    zdf_dir.mkdir()
    for name in ("S01_fill_f02_01", "S01_fill_f02_02"):
        (zdf_dir / f"{name}.zdf").write_bytes(b"x")
    write_sample(samples, "S01_fill_f02_01", [(6, 50, 50, 40)])  # already converted
    calls = []

    def fake_converter(zdf, samples_dir, settings_only=False):
        calls.append((zdf.stem, settings_only))
        write_sample(samples_dir, zdf.stem, [(6, 50, 50, 40), (6, 200, 50, 40)])

    lines = []
    n = cc.run(zdf_dir, samples, [6], converter=fake_converter, out=lines.append)
    assert calls == [("S01_fill_f02_02", False)]
    text = "\n".join(lines)
    assert "converted 1 new, 1 already there" in text
    assert "ids 6x2 dup-ID" in text and "ids 6x1" in text
    assert n == 0


def test_run_warns_on_changed_settings_and_missing_id(tmp_path):
    zdf_dir, samples = tmp_path / "zdf", tmp_path / "samples"
    zdf_dir.mkdir()
    for name, settings, marker in (("S01_fill_f02_01", "a\n", 6), ("S01_fill_f02_02", "b\n", 4)):
        (zdf_dir / f"{name}.zdf").write_bytes(b"x")
        write_sample(samples, name, [(marker, 50, 50, 40)], settings=settings)
    lines = []
    n = cc.run(zdf_dir, samples, [6], out=lines.append)
    text = "\n".join(lines)
    assert n == 2
    assert "Zivid settings differ" in text and "expected marker ID [6] missing" in text


def test_run_without_files_returns_error(tmp_path):
    (tmp_path / "zdf").mkdir()
    assert cc.run(tmp_path / "zdf", tmp_path / "s", [6], out=lambda _: None) == 1
