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

## Setup

Python 3.9+. Install the dependencies on every machine that runs a tool:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy opencv-contrib-python pytest
pip install zivid   # only where ZDF files are read or captured (needs the matching Zivid SDK)
```

Run the offline tests first (no hardware needed):

```bash
python3 -m pytest -q
```

## How the pieces fit together

Three machines take part. The tools expect this layout; adapt the constants at the top of
`tools/capture_feed.py` if yours differs.

| Machine | Role | Runs |
|---|---|---|
| **Mac / laptop** | operator desk | `capture_ui.py`, `capture_feed.py` (they drive the others over SSH) |
| **Capture PC** (SSH alias `lara5roboter`, next to the camera) | talks to the Zivid camera, disk is small | `tk_autocapture.py`, `tk_live_capture.py`, `tk_capture.py` (not in this repo) |
| **Workstation** (SSH alias `Workstationlernfabrik`) | stores data, computes ground truth | `check_captures.py`, `cell_ground_truth.py`, `review_gui.py` |

On the workstation the tools expect a work folder `~/Desktop/Gruppe3/testkoerper_benchmark/` (venv in `.venv/`,
this repo's `tools/` copied to `tools/`) with:

```
data/zdf/<session>/            raw .zdf captures, capture_log.csv, review_log.csv
data/samples/<session>/        converted samples (<name>_rgb.png, _xyz.npy, _cam_K.txt, ...)
data/cell_parts_<step>.json    part configuration (CAD meshes, seat heights)  -- not in this repo
data/plate_positions_*.json    where the two tray plates stood                -- not in this repo
runs/<session>/                ground-truth output, review verdicts
```

The CAD STL files, part configurations, plate positions and calibration are **not** part of this repo.

## Quick start: capture a session

1. **Make scene cards** (only once per step; the generated CSVs for PO_TY and SY_RZ are in `docs/researcheffort/`):

   ```bash
   python3 tools/make_scene_cards.py --step poty --out docs/researcheffort/scene_cards_poty_v1
   ```

2. **Calibrate the slots** from empty-plate captures (once per plate position; needs the plate positions file):

   ```bash
   python3 tools/check_captures.py data/zdf/<session> --samples-dir data/samples/<session> --pattern "S0*_empty*.zdf"
   python3 tools/cell_ground_truth.py calibrate data/samples/<session> \
       --positions data/plate_positions_<date>.json --out runs/<session>/calibration.json
   ```

   Point `CALIBRATION` / `POSITIONS` (or the step entry in `STEPS`) in `tools/capture_feed.py` at these files.

3. **Start the capture tool** on the Mac. Stop any running autocapture first, because the camera can only be held by one process:

   ```bash
   python3 tools/capture_ui.py --session S03 --sess-dir benchmark_2026-10-06 --step poty
   ```

   Open http://localhost:8770. The page shows the current scene card and both plate grids. Set up the scene on the
   plates, step out of the camera view and press the capture button. The tool then captures on the capture PC,
   streams the ZDF to the workstation (checksum verified, the capture-PC copy is deleted), converts it to a sample,
   logs the card's layout and, with `--gt`, computes the ground truth. A rejected capture does not advance the card;
   the next press counts as a new attempt (`<scene> + 100`). Useful options:
   `--gt` (ground truth after every capture, slower), `--no-project` (no projector wireframes), `--port`,
   `--step poty|syrz`, `--calibration`, `--positions`.

   The page shows the scene as a render from `docs/researcheffort/scene_card_renders/`; those renders come from an
   Isaac Sim script that is not in this repo, so without them the preview stays empty (capturing still works).

4. **Compute ground truth for the whole session** when you captured without `--gt`:

   ```bash
   python3 tools/capture_feed.py gt --sess-dir benchmark_2026-10-06 --session S03
   ```

### Alternative: hands-free autocapture

`capture_feed.py` can instead let the capture PC shoot automatically once the scene changed and stayed still:

```bash
python3 tools/capture_feed.py start  --session S03 --sess-dir benchmark_2026-10-06   # start autocapture on the capture PC
python3 tools/capture_feed.py next   --session S03 --sess-dir benchmark_2026-10-06   # process ONE capture, then exit
python3 tools/capture_feed.py status --session S03 --sess-dir benchmark_2026-10-06
python3 tools/capture_feed.py skip   --session S03 --sess-dir benchmark_2026-10-06   # skip the current card
python3 tools/capture_feed.py stop   --session S03 --sess-dir benchmark_2026-10-06   # stop autocapture
```

Autocapture can fire in the middle of a set-up, so the manual `capture_ui.py` is the better default.

### Single captures from the shell

```bash
tools/tk_shot.sh benchmark_2026-10-01 S02_fill_poty-a_01 "L1-L8:U R1-R8:HO" 1
#                session dir          name                layout (logged)     expected marker ID
```

`tools/tk_transfer.sh <session-dir>` moves finished autocaptures from the capture PC to the workstation.

## Quick start: check captures and label

All of this runs on the workstation, inside the work folder described above.

1. **Check a batch of captures** (converts ZDF to samples, detects markers, warns on drift; exit code 1 on warnings):

   ```bash
   python tools/check_captures.py data/zdf/<session> --samples-dir data/samples/<session> --expect 1 --log data/zdf/<session>/capture_log.csv
   ```

2. **Compute ground truth** (poses, masks, boxes per capture):

   ```bash
   python tools/cell_ground_truth.py run data/samples/<session> \
       --config data/cell_parts_poty.json \
       --calibration runs/<session>/calibration.json \
       --positions data/plate_positions_<date>.json \
       --capture-log data/zdf/<session>/capture_log.csv \
       --out runs/<session> [--sample-id NAME ...]
   ```

3. **Review it by hand** in the web tool. It binds to `127.0.0.1` only, so open it through an SSH tunnel:

   ```bash
   python tools/review_gui.py runs/<session> --samples data/samples/<session> \
       --config data/cell_parts_poty.json --port 8765 --reviewer "<your name>"
   ```

   ```bash
   ssh -N -L 8765:localhost:8765 Workstationlernfabrik     # on your laptop, then open http://localhost:8765
   ```

   Per part mark the ground truth as `correct`, `pose wrong`, `mask wrong` or `wrong label`; per capture `accept` or
   `reject`. Verdicts are written to `runs/<session>/review/<sample>.json`; the ground-truth files are never changed.
   The page rescans the run folder on every load, so a session that is still growing shows new captures on reload.

### Projector helper

`tools/tk_project_scene.py` lights part outlines of a scene card on the table with the camera's projector (run on the
capture PC; `capture_ui.py` starts it for you unless `--no-project` is given):

```bash
python3 tk_project_scene.py --card S05_fill_poty-c_10 --cards scene_cards_poty_v1.csv \
    --calibration <slot calibration>.json --cad <folder with STL files>
```

## Not included / assumptions

- No data, CAD files, calibration files or captures.
- The SSH aliases `lara5roboter` (capture PC on the robot net) and `Workstationlernfabrik`
  (processing workstation) and the paths under `~/testkoerper_benchmark_capture` and
  `~/Desktop/Gruppe3/testkoerper_benchmark` are specific to our lab setup; change them in
  `tools/capture_feed.py` and `tools/tk_*.sh`.
- `tk_capture.py` (the Jetson capture script that `capture_feed.py` calls) is not part of this repo.
