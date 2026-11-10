"""Draw the CAD wireframe of every labelled part at its estimated pose onto a benchmark image.

    python3 tools/wireframe_overlay.py --image S03_x.png --poses S03_x_poses.json --cam-k cam_K.txt \\
        --cad-dir <folder with the STL files> --out overlay.jpg [--crop x0 y0 x1 y1] [--only Halter Oberteil]
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np

MESHES = {"Unterteil": "Einlegen_(PO_TX)_Unterteil_49,8mm.stl", "Halter": "Einlegen_(PO_TX)_Halter_49,8mm.stl",
          "Oberteil": "Einlegen_(PO_TX)_Oberteil_49,8mm.stl"}
COLORS = {"Unterteil": (60, 200, 60), "Halter": (230, 160, 40), "Oberteil": (60, 80, 240)}   # BGR
FEATURE_DEG = 25.0


def load_stl(path: Path) -> np.ndarray:
    """Triangles (N, 3, 3) of a binary STL, in mm."""
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + count * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3, 3).astype(float)


def feature_edges(tri: np.ndarray, angle_deg: float = FEATURE_DEG) -> np.ndarray:
    """Edges (M, 2, 3) between faces whose normals differ by more than ``angle_deg`` (plus boundary edges)."""
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    edges: dict[tuple, list] = {}
    for i, t in enumerate(tri):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((tuple(np.round(t[a], 3)), tuple(np.round(t[b], 3)))))
            edges.setdefault(key, []).append(i)
    keep = []
    for (p, q), faces in edges.items():
        if len(faces) == 1 or np.degrees(np.arccos(np.clip(normals[faces[0]] @ normals[faces[1]], -1, 1))) > angle_deg:
            keep.append((p, q))
    return np.array(keep, dtype=float) if keep else np.zeros((0, 2, 3))


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.stack([k[0, 0] * points[:, 0] / points[:, 2] + k[0, 2], k[1, 1] * points[:, 1] / points[:, 2] + k[1, 2]], axis=1)


def draw(image: np.ndarray, poses: list[dict], k: np.ndarray, cad_dir: Path, only=None) -> np.ndarray:
    out = image.copy()
    cache: dict[str, np.ndarray] = {}
    for pose in poses:
        kind = pose["object_id"].split("_")[-1]
        if kind not in MESHES or (only and kind not in only):
            continue
        if kind not in cache:
            cache[kind] = feature_edges(load_stl(cad_dir / MESHES[kind]))
        from scipy.spatial.transform import Rotation   # only needed here, the Jetson projector tool imports this module without scipy
        rot = Rotation.from_euler("xyz", pose["rotation_euler_xyz_deg"], degrees=True).as_matrix()
        edges = cache[kind].reshape(-1, 3) @ rot.T + np.array(pose["translation_mm"])
        uv = project(edges, k).reshape(-1, 2, 2)
        for a, b in uv:
            cv2.line(out, tuple(int(v) for v in a), tuple(int(v) for v in b), COLORS[kind], 2, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--cam-k", type=Path, required=True)
    parser.add_argument("--cad-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    image = cv2.imread(str(args.image))
    out = draw(image, json.loads(args.poses.read_text())["poses"], np.loadtxt(args.cam_k), args.cad_dir, args.only)
    if args.crop:
        x0, y0, x1, y1 = args.crop
        out = out[y0:y1, x0:x1]
    cv2.imwrite(str(args.out), cv2.resize(out, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
