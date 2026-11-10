"""Light up a tray or single pockets on the table with the Zivid projector.

Runs on the Jetson (needs the ``zivid`` package and the camera). The 3D pocket corners come from the
slot calibration (``docs/researcheffort/cell_calibration_2026-10-01.json``, ``T_cam_slot`` per slot),
``zivid.projection.pixels_from_3d_points`` maps them to projector pixels, and the outline is projected.

    python3 tk_project.py cell_calibration_2026-10-01.json --position P1 --slot L3 R5 [--fill] [--level top]
    python3 tk_project.py cell_calibration_2026-10-01.json --position P1 --tray L --seconds 0
    python3 tk_project.py cell_calibration_2026-10-01.json --position P1 --slot L3 --save-image proj.png --no-project
    python3 tk_project.py cell_calibration_2026-10-01.json --position P1 --slot L3 --check seen.png

``--position`` must be where the plates actually stand (P1 or P2 of the calibration). The projection
stops after ``--seconds`` (0 = until Enter). Any ``capture`` on the camera ends it; ``capture_2d`` on the
projected image (``--check``) does not. Stop the autocapture first: the camera can have one client only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_POCKET_MM = 50.0  # used when the calibration has no measured pocket size (NaN)
COLORS = {"green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 0, 0), "white": (255, 255, 255), "yellow": (0, 255, 255)}


def load_plates(path: Path, position: str) -> dict[str, dict]:
    """Return ``{side: {"slots": {name: T_cam_slot}, "pocket": (w, h), "floor_depth": mm}}`` for a position."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if position not in raw["positions"]:
        raise SystemExit(f"unknown position {position!r}; calibration has {sorted(raw['positions'])}")
    out = {}
    for side, plate in raw["positions"][position]["plates"].items():
        pocket = tuple(v if math.isfinite(v) else DEFAULT_POCKET_MM for v in plate["pocket_size_mm"])
        out[side] = {
            "slots": {n: np.asarray(T, dtype=float) for n, T in plate["slots"].items()},
            "pocket": pocket,
            "floor_depth": float(plate["floor_depth_mm"]),
        }
    return out


def pocket_corners(T_cam_slot: np.ndarray, pocket: tuple[float, float], height_mm: float) -> np.ndarray:
    """Four pocket corners in the camera frame (mm), ``height_mm`` above the pocket floor along the plate normal.

    The slot frame has its origin at the pocket centre on the floor, z towards the camera, x/y along the pocket.
    """
    hx, hy = pocket[0] / 2.0, pocket[1] / 2.0
    local = np.array([[-hx, -hy, height_mm, 1.0], [hx, -hy, height_mm, 1.0], [hx, hy, height_mm, 1.0], [-hx, hy, height_mm, 1.0]])
    return (T_cam_slot @ local.T).T[:, :3]


def select_slots(plates: dict[str, dict], slots: list[str], tray: str | None) -> list[str]:
    """Names of the slots to light: the given ones, or all nine of a tray ('L' or 'R')."""
    by_name = {n: s for p in plates.values() for s in [p] for n in p["slots"]}
    if tray is not None:
        if tray not in plates:
            raise SystemExit(f"unknown tray {tray!r}; this position has {sorted(plates)}")
        return sorted(plates[tray]["slots"])
    unknown = [s for s in slots if s not in by_name]
    if unknown:
        raise SystemExit(f"unknown slot(s) {unknown}; available: {sorted(by_name)}")
    return list(slots)


def polygons_for(plates: dict[str, dict], names: list[str], level: str) -> list[np.ndarray]:
    """3D corner polygons (4x3, camera mm) of the named slots, on the pocket floor or the plate top."""
    polys = []
    for name in names:
        plate = next(p for p in plates.values() if name in p["slots"])
        height = plate["floor_depth"] if level == "top" else 0.0
        polys.append(pocket_corners(plate["slots"][name], plate["pocket"], height))
    return polys


def draw_polygons(pixel_polys: list[np.ndarray], shape: tuple[int, int], color: tuple[int, int, int], fill: bool, thickness: int) -> np.ndarray:
    """BGRA image (height, width) with the polygons drawn; black pixels project no light."""
    import cv2

    image = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
    image[..., 3] = 255
    bgra = (*color, 255)
    for poly in pixel_polys:
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        if fill:
            cv2.fillPoly(image, [pts], bgra)
        else:
            cv2.polylines(image, [pts], True, bgra, thickness, cv2.LINE_AA)
    return image


def count_outside(pixel_polys: list[np.ndarray], shape: tuple[int, int]) -> int:
    """Number of polygon corners that fall outside the projector image (height, width)."""
    pts = np.concatenate(pixel_polys)
    return int(((pts[:, 0] < 0) | (pts[:, 0] >= shape[1]) | (pts[:, 1] < 0) | (pts[:, 1] >= shape[0])).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("calibration", type=Path, help="slot calibration JSON (cell_calibration_*.json)")
    parser.add_argument("--position", required=True, help="plate position of the calibration, P1 or P2")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--slot", nargs="+", default=[], help="pockets to light, e.g. L3 R5")
    target.add_argument("--tray", help="light all nine pockets of one tray, L or R")
    parser.add_argument("--level", choices=["floor", "top"], default="floor", help="project at the pocket floor or the plate top")
    parser.add_argument("--fill", action="store_true", help="fill the pockets instead of an outline")
    parser.add_argument("--color", choices=sorted(COLORS), default="green")
    parser.add_argument("--thickness", type=int, default=4, help="outline width in projector pixels")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to project; 0 waits for Enter")
    parser.add_argument("--no-project", action="store_true", help="only compute (and --save-image); do not light anything")
    parser.add_argument("--save-image", type=Path, help="write the projector image as PNG")
    parser.add_argument("--check", type=Path, help="while projecting, capture a 2D image and save it as PNG")
    args = parser.parse_args()

    plates = load_plates(args.calibration, args.position)
    names = select_slots(plates, args.slot, args.tray)
    polys = polygons_for(plates, names, args.level)

    import zivid
    from zivid import projection

    app = zivid.Application()
    camera = app.connect_camera()
    try:
        shape = tuple(projection.projector_resolution(camera))  # (height, width)
        pixel_polys = [np.asarray(projection.pixels_from_3d_points(camera, poly), dtype=float) for poly in polys]
        outside = count_outside(pixel_polys, shape)
        print(f"{len(names)} pocket(s) {names} at the {args.level}, projector {shape[1]}x{shape[0]}, corners outside: {outside}")
        if outside:
            print("warning: part of the outline lies outside the projector image", file=sys.stderr)
        image = draw_polygons(pixel_polys, shape, COLORS[args.color], args.fill, args.thickness)
        if args.save_image:
            import cv2

            cv2.imwrite(str(args.save_image), image)
            print("projector image ->", args.save_image)
        if args.no_project:
            return 0
        with projection.show_image_bgra(camera, image) as projected:
            if args.check:
                import cv2

                # rgb capture while projecting needs brightness 0 (see ProjectedImage.capture_2d); 8333 us = one projector period
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
