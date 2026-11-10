"""Pose of every part type in its slot frame, per scene-card code and quarter turn, from the accepted captures.

Runs on the workstation next to ``runs/``. For each accepted capture the card says which code (U, H, HO, UO) and
quarter turn (qN) stands in each slot; the ground-truth pose in the camera frame and the slot frame of the plate
position give ``T_slot_part``. The median over all captures is the pose the projector should draw.

    python3 tools/derive_slot_poses.py --runs runs/benchmark_2026-10-01_v2 runs/benchmark_2026-10-06 \\
        --cards scene_cards_poty_v1.csv --calibration runs/benchmark_2026-10-01_v2/calibration.json --out slot_part_poses_poty.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse_cells(text: str) -> dict[str, tuple[str, int]]:
    cells = {}
    for token in (text or "").split():
        slot, _, rest = token.partition(":")
        code, _, turn = rest.partition("/")
        cells[slot] = (code, int(turn[1:]) if turn else 0)
    return cells


def kind_of(object_id: str) -> str:
    return object_id.split("_")[-1]


def yaw_deg(rot: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))


def circular_median(angles_deg: list[float]) -> tuple[float, float]:
    """Mean direction and the largest deviation (deg) of angles on the circle."""
    a = np.radians(angles_deg)
    mean = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
    dev = np.abs((np.array(angles_deg) - mean + 180) % 360 - 180)
    return float(mean), float(dev.max())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    calibration = json.loads(args.calibration.read_text())["positions"]
    cards = {r["name"]: r for r in csv.DictReader(open(args.cards, newline="", encoding="utf-8"))}
    groups: dict[tuple, list] = {}
    for run in args.runs:
        for review_path in sorted(glob.glob(str(run / "review" / "*.json"))):
            review = json.loads(Path(review_path).read_text())
            cid = review["sample_id"]
            if review["capture"]["verdict"] != "accept" or cid not in cards:
                continue
            card = cards[cid]
            cells = parse_cells(card["quarter_turns"])
            dets = json.loads((run / "annotations" / f"{cid}_detections.json").read_text())["detections"]
            poses = json.loads((run / "annotations" / f"{cid}_poses.json").read_text())["poses"]
            for det, pose in zip(dets, poses):
                slot = det["slot"]
                if slot not in cells:
                    continue
                code, quarter = cells[slot]
                position = card["plate_" + slot[0]]
                t_cam_slot = np.array(calibration[position]["plates"][slot[0]]["slots"][slot])
                t_cam_part = np.eye(4)
                t_cam_part[:3, :3], t_cam_part[:3, 3] = quat_to_rot(pose["rotation_quat"]), pose["translation_mm"]
                t_slot_part = np.linalg.inv(t_cam_slot) @ t_cam_part
                groups.setdefault((code, kind_of(det["object_id"]), quarter), []).append(t_slot_part)
    out = {}
    for (code, kind, quarter), mats in sorted(groups.items()):
        pos = np.array([m[:3, 3] for m in mats])
        yaw, yaw_dev = circular_median([yaw_deg(m[:3, :3]) for m in mats])
        out.setdefault(code, {}).setdefault(f"q{quarter}", {})[kind] = {
            "x_mm": round(float(np.median(pos[:, 0])), 2), "y_mm": round(float(np.median(pos[:, 1])), 2),
            "z_mm": round(float(np.median(pos[:, 2])), 2), "yaw_deg": round(yaw, 1), "n": len(mats),
            "pos_spread_mm": round(float(np.abs(pos - np.median(pos, axis=0)).max()), 2), "yaw_spread_deg": round(yaw_dev, 1)}
    args.out.write_text(json.dumps({"frame": "slot frame (origin pocket centre on the floor, z towards the camera); pose of the CAD frame",
                                    "codes": out}, indent=2))
    for code, quarters in out.items():
        for q, kinds in quarters.items():
            for kind, v in kinds.items():
                print(f"{code:2s} {q} {kind:9s} n={v['n']:2d} xyz=({v['x_mm']:6.1f},{v['y_mm']:6.1f},{v['z_mm']:6.1f}) yaw={v['yaw_deg']:7.1f} spread pos {v['pos_spread_mm']:.1f} mm yaw {v['yaw_spread_deg']:.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
