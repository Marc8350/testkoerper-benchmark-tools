# testkoerper-benchmark-tools

Capture and ground-truth labelling tools for a real-world detection and 6D-pose benchmark of
joining-process test bodies (Zivid 2+ L110 fixed above two 3x3 tray plates, LARA5 robot cell).

Ground truth is computed from the cell geometry (slot frames from empty-plate captures x CAD seat,
fitted to depth); ArUco markers on the parts are only a consistency check and are filled in digitally.

## Contents

| Path | Purpose |
|---|---|
| `tools/capture_ui.py`, `capture_feed.py` | Web UI and loop that drive a capture session card by card (capture on the Jetson, transfer, convert, ground truth) |
| `tools/tk_shot.sh`, `tk_transfer.sh`, `tk_autocapture.py`, `tk_live_capture.py` | Jetson-side and transfer helpers |
| `tools/check_captures.py` | Converts ZDF captures to samples (sRGB colour, depth, point cloud), detects markers, warns on drift |
| `tools/cell_ground_truth.py`, `marker_ground_truth.py` | Ground truth: poses, masks, boxes per capture |
| `tools/review_gui.py` | Review tool: per-capture and per-part verdicts on the computed ground truth |
| `tools/tk_project.py`, `tk_project_scene.py`, `wireframe_overlay.py` | Projector helper that lights pockets or part outlines on the table; wireframe overlay |
| `tools/make_scene_cards.py`, `derive_slot_poses.py` | Balanced scene cards for capture sessions; slot poses from CAD |
| `docs/researcheffort/scene_cards_*.csv` | Generated scene cards (PO_TY, SY_RZ) |

## Tests

```bash
python3 -m pytest -q
```

They run offline and need no hardware (numpy, OpenCV, trimesh, pytest).

## Not included / assumptions

- No data, CAD files, calibration files or captures.
- The SSH aliases `lara5roboter` (capture PC on the robot net) and `Workstationlernfabrik`
  (processing workstation) and the paths under `~/testkoerper_benchmark_capture` and
  `~/Desktop/Gruppe3/testkoerper_benchmark` are specific to our lab setup; change them in
  `tools/capture_feed.py` and `tools/tk_*.sh`.
- `tk_capture.py` (the Jetson capture script that `capture_feed.py` calls) is not part of this repo.
