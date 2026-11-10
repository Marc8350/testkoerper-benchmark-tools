"""Project a scene card onto the table: per slot the footprint wireframe of every part at its pose and orientation.

Runs on the Jetson (needs ``zivid``, the camera, numpy and cv2). The part poses in a slot follow the scene-card code
(U, H, HO, UO) and quarter turn qN (counter-clockwise, as seen in the camera image), see ``parts_for_code``. The CAD
edges are moved into the slot frame, flattened onto the plate top (so the light lands on the plate) and projected.

    python3 tk_project_scene.py --card S05_fill_poty-c_10 --cards scene_cards_poty_v1.csv --calibration cell_calibration_2026-10-01.json --cad cad
    python3 tk_project_scene.py --cells "L3:HO/q0 R5:U/q2" --pos-L P1 --pos-R P1 --calibration ... --cad cad --no-project --save-image proj.png

The projection ends after ``--seconds`` (0 = until Enter) or when another ``capture`` runs on the camera.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tk_project as tp  # noqa: E402
import wireframe_overlay as wo  # noqa: E402

# Bright colours only: the projector brightness cannot be set through the SDK, so full-intensity green / cyan / amber
# (blue and red look much dimmer to the eye) and thick lines are the way to a brighter image.
COLORS_BGR = {"Unterteil": (0, 255, 0), "Halter": (255, 255, 0), "Oberteil": (0, 190, 255)}
SEAT_Z_OBERTEIL = 25.0           # Oberteil on the Halter / in the Unterteil groove: its z = 0 plane is 25 mm over the pocket floor
UO_OFFSET_Y = -3.5               # Oberteil play position along the Unterteil groove (CAD joining pose), Unterteil frame
# Measured from the accepted captures (docs/researcheffort/slot_part_poses_poty_derived.json, tools/derive_slot_poses.py):
# Unterteil q0 = 90 deg in the slot frame, +90 deg per quarter turn; Halter and Oberteil are symmetric (90 / 180 deg),
# the Oberteil bar lies along the slot x axis for even q and along y for odd q; in UO the Oberteil turns with the Unterteil.
# Einlegen_SY_RZ, derived from Paul's CAD (no captures yet, to be checked against the first ones): the Unterteil is a U-channel
# with walls at CAD y = +-(19.9..24.9) and the channel along CAD x; q0 turns the wall-to-wall direction (CAD y) to the slot x
# axis (yaw 270 deg; the part is 180 deg symmetric). The Oberteil cone seats in the round Unterteil / Halter hole (xy = 0) with
# its bar (CAD x) on the floor / Halter top, 25 mm up (voxel overlap with both parts reaches 0 at dz = 25-26 mm). Rotation about
# z is free in the Unterteil (that is the designed RZ error), so the nominal pose turns with the Unterteil.
SY_RZ_MESHES = {kind: f"Einlegen_(SY_RZ)_{kind}_49,8mm.stl" for kind in ("Unterteil", "Halter", "Oberteil")}
STEP_MESHES = {"poty": wo.MESHES, "syrz": SY_RZ_MESHES}
# SY_RZ is drawn as plain shapes (the CAD detail with cone circles was not readable on the table): the two Unterteil walls and the
# Oberteil bar as filled areas (green / yellow), the Halter as a thick cyan square. Values from the CAD, see above.
SIMPLE_COLORS_BGR = {"Unterteil": (0, 255, 0), "Halter": (255, 255, 0), "Oberteil": (0, 255, 255)}


def _rect_edges(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return np.array([[(*corners[i], 0.0), (*corners[(i + 1) % 4], 0.0)] for i in range(4)])


# Four consecutive edges = one polygon (filled where ``SIMPLE_FILLED``).
SIMPLE_SHAPES = {
    "Unterteil": np.concatenate([_rect_edges(-24.9, 19.9, 24.9, 24.9), _rect_edges(-24.9, -24.9, 24.9, -19.9)]),
    "Halter": _rect_edges(-24.9, -24.9, 24.9, 24.9),
    "Oberteil": _rect_edges(-17.5, -10.0, 17.5, 10.0),
}
SIMPLE_FILLED = {"Unterteil": 2, "Halter": 0, "Oberteil": 1}      # leading polygons that are filled, the others are outlines


def parse_cells(text: str) -> dict[str, tuple[str, int]]:
    """``'L1:UO/q3 R2:H'`` -> ``{'L1': ('UO', 3), 'R2': ('H', 0)}``."""
    cells = {}
    for token in (text or "").split():
        slot, _, rest = token.partition(":")
        code, _, turn = rest.partition("/")
        cells[slot] = (code, int(turn[1:]) if turn else 0)
    return cells


def parts_for_code(code: str, quarter: int, step: str = "poty") -> list[tuple[str, float, tuple[float, float], float]]:
    """Parts of one slot as ``(kind, z_mm, (x, y) in the slot frame, yaw_deg)``; z is over the pocket floor."""
    if step == "syrz":
        yaw_u = 270.0 + 90.0 * quarter
        if code == "U":
            return [("Unterteil", 0.0, (0.0, 0.0), yaw_u)]
        if code == "H":
            return [("Halter", 0.0, (0.0, 0.0), 0.0)]
        if code == "HO":
            return [("Halter", 0.0, (0.0, 0.0), 0.0), ("Oberteil", SEAT_Z_OBERTEIL, (0.0, 0.0), 90.0 * quarter)]
        if code == "UO":
            return [("Unterteil", 0.0, (0.0, 0.0), yaw_u), ("Oberteil", SEAT_Z_OBERTEIL, (0.0, 0.0), yaw_u)]
        raise ValueError(f"unknown code {code!r}")
    yaw_u = 90.0 + 90.0 * quarter
    if code == "U":
        return [("Unterteil", 0.0, (0.0, 0.0), yaw_u)]
    if code == "H":
        return [("Halter", 0.0, (0.0, 0.0), 0.0)]
    if code == "HO":
        return [("Halter", 0.0, (0.0, 0.0), 90.0 * quarter), ("Oberteil", SEAT_Z_OBERTEIL, (0.0, 0.0), 90.0 * quarter)]
    if code == "UO":
        rad = math.radians(yaw_u)
        offset = (-UO_OFFSET_Y * math.sin(rad), UO_OFFSET_Y * math.cos(rad))
        return [("Unterteil", 0.0, (0.0, 0.0), yaw_u), ("Oberteil", SEAT_Z_OBERTEIL, offset, yaw_u)]
    raise ValueError(f"unknown code {code!r}")


def rot_z(yaw_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def edges_in_slot(edges: np.ndarray, z_mm: float, xy: tuple[float, float], yaw_deg: float) -> np.ndarray:
    """CAD edges (M, 2, 3) placed at the part pose in the slot frame."""
    return (edges.reshape(-1, 3) @ rot_z(yaw_deg).T + np.array([xy[0], xy[1], z_mm])).reshape(-1, 2, 3)


def flatten(edges_slot: np.ndarray, level_mm: float) -> np.ndarray:
    """Collapse the edges onto the plane ``level_mm`` above the pocket floor (top view of the part)."""
    out = edges_slot.copy()
    out[..., 2] = level_mm
    return out


def to_camera(edges_slot: np.ndarray, t_cam_slot: np.ndarray) -> np.ndarray:
    pts = np.concatenate([edges_slot.reshape(-1, 3), np.ones((edges_slot.shape[0] * 2, 1))], axis=1)
    return ((t_cam_slot @ pts.T).T[:, :3]).reshape(-1, 2, 3)


def scene_edges(cells: dict[str, tuple[str, int]], plates: dict[str, dict], meshes: dict[str, np.ndarray],
                step: str = "poty") -> list[tuple[str, str, np.ndarray]]:
    """``[(slot, kind, edges in the camera frame (M, 2, 3))]`` for every part of the scene; edges are flattened on the plate top."""
    out = []
    for slot, (code, quarter) in cells.items():
        plate = plates[slot[0]]
        for kind, z, xy, yaw in parts_for_code(code, quarter, step):
            placed = flatten(edges_in_slot(meshes[kind], z, xy, yaw), plate["floor_depth"])
            out.append((slot, kind, to_camera(placed, plate["slots"][slot])))
    return out


def draw_scene(items: list[tuple[str, str, np.ndarray]], pixel_fn, shape: tuple[int, int], thickness: int, fill: bool = False,
               colors: dict[str, tuple[int, int, int]] = COLORS_BGR, simple: bool = False) -> np.ndarray:
    """BGRA projector image; ``pixel_fn`` maps camera mm (N, 3) to projector pixels (N, 2); ``simple`` draws ``SIMPLE_SHAPES``."""
    import cv2

    image = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
    image[..., 3] = 255
    for slot, kind, edges in items:
        px = np.asarray(pixel_fn(edges.reshape(-1, 3)), dtype=float).reshape(-1, 2, 2)
        color = (*colors[kind], 255)
        if simple:                                         # one polygon per four edges
            for i, poly in enumerate(px[:, 0].reshape(-1, 4, 2)):
                pts = np.round(poly).astype(np.int32)
                if i < SIMPLE_FILLED[kind]:
                    cv2.fillPoly(image, [pts], color)
                cv2.polylines(image, [pts], True, color, thickness if kind == "Halter" else 3, cv2.LINE_AA)
            continue
        hull = cv2.convexHull(np.round(px.reshape(-1, 2)).astype(np.int32))
        if fill:                                           # footprint tinted at 35 % intensity under the lines
            cv2.fillPoly(image, [hull], tuple(int(0.35 * c) for c in colors[kind]) + (255,))
        for a, b in px:                                    # CAD feature edges
            cv2.line(image, tuple(int(v) for v in a), tuple(int(v) for v in b), color, 2, cv2.LINE_AA)
        cv2.polylines(image, [hull], True, color, thickness, cv2.LINE_AA)   # footprint outline, thick
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cad", type=Path, required=True, help="folder with the STL files")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--card", help="scene card name, e.g. S05_fill_poty-c_10")
    source.add_argument("--cells", help="e.g. 'L3:HO/q0 R5:U/q2'")
    parser.add_argument("--cards", type=Path, help="scene cards CSV (with --card)")
    parser.add_argument("--step", default="poty", choices=sorted(STEP_MESHES))
    parser.add_argument("--pos-L", default="P1")
    parser.add_argument("--pos-R", default="P1")
    parser.add_argument("--thickness", type=int, default=5, help="footprint outline width in projector px")
    parser.add_argument("--fill", action="store_true", help="tint the footprints as well")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--no-project", action="store_true")
    parser.add_argument("--save-image", type=Path)
    parser.add_argument("--check", type=Path, help="capture a 2D image while projecting")
    args = parser.parse_args()

    if args.card:
        card = next(r for r in csv.DictReader(open(args.cards, newline="", encoding="utf-8")) if r["name"] == args.card)
        cells, pos = parse_cells(card["quarter_turns"]), {"L": card["plate_L"], "R": card["plate_R"]}
    else:
        cells, pos = parse_cells(args.cells), {"L": args.pos_L, "R": args.pos_R}
    plates = {side: tp.load_plates(args.calibration, position)[side] for side, position in pos.items()}
    meshes = {kind: wo.feature_edges(wo.load_stl(args.cad / file)) for kind, file in STEP_MESHES[args.step].items()}
    simple = args.step == "syrz"
    if simple:
        meshes = SIMPLE_SHAPES
    items = scene_edges(cells, plates, meshes, args.step)

    import zivid
    from zivid import projection

    app = zivid.Application()
    camera = app.connect_camera()
    try:
        shape = tuple(projection.projector_resolution(camera))
        pixel_fn = lambda pts: projection.pixels_from_3d_points(camera, np.asarray(pts, dtype=float))   # noqa: E731
        image = draw_scene(items, pixel_fn, shape, args.thickness, args.fill, SIMPLE_COLORS_BGR if simple else COLORS_BGR, simple)
        lit = int((image[..., :3].sum(axis=2) > 0).sum())
        print(f"{len(cells)} slot(s) {sorted(cells)}, plates {pos}, projector {shape[1]}x{shape[0]}, lit pixels {lit}")
        if args.save_image:
            import cv2

            cv2.imwrite(str(args.save_image), image)
            print("projector image ->", args.save_image)
        if args.no_project:
            return 0
        with projection.show_image_bgra(camera, image) as projected:
            if args.check:
                import cv2

                settings = zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition(exposure_time=datetime.timedelta(microseconds=8333), aperture=4.0, brightness=0.0)])
                rgb = projected.capture_2d(settings).image_rgb().copy_data()
                cv2.imwrite(str(args.check), cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR))
                print("capture with projection ->", args.check)
            if args.seconds > 0:
                time.sleep(args.seconds)
            else:
                input("projecting, press Enter to stop ")
    finally:
        camera.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
