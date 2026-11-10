"""Web tool to hand-verify the marker ground truth of benchmark captures.

It shows, per capture, the original image (with markers), the filled-in image,
the visible part masks, the CAD model projected at the ground-truth pose
(sharp-edge wireframe, silhouette, coordinate axes), the boxes and a heat map
of measured minus rendered depth. Per part the reviewer marks the ground truth
as correct, pose wrong, mask wrong or wrong label; per capture accept or
reject. Verdicts go to ``RUN_DIR/review/<sample_id>.json``; the ground-truth
files are never modified.

Inputs are the output folder of ``marker_ground_truth.py`` (RUN_DIR), the
sample folder it was made from, and the same part configuration. Every page
load rescans RUN_DIR, so a run that is still growing shows new captures on
reload.

The server binds to 127.0.0.1 only. Open it through an SSH tunnel::

    python review_gui.py RUN_DIR --samples SAMPLE_DIR --config CONFIG.json [--port 8765]
    ssh -N -L 8765:localhost:8765 Workstationlernfabrik   # then http://localhost:8765
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np

import marker_ground_truth as mgt

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
REVIEW_SCHEMA_VERSION = "1.0"
PART_VERDICTS = ("correct", "pose_wrong", "mask_wrong", "wrong_label")
CAPTURE_VERDICTS = ("accept", "reject")
LAYERS = ("masks", "wire", "hidden", "sil", "axes", "bbox", "resid")
ZOOM_LAYERS = ("wire", "bbox")  # the zoom window shows only these, and only for the selected part
SHARP_ANGLE_DEG = 30.0
AXIS_LENGTH_MM = 20.0
RESIDUAL_CLIP_MM = 5.0
EDGE_VISIBILITY_TOL_MM = 1.0
EDGE_STEP_PX = 2.0
CACHE_SIZE = 2
MAX_BODY_BYTES = 65536

# BGR colours. Masks and boxes use one colour per part; model layers are fixed.
PART_COLOURS = [
    (60, 180, 255), (255, 120, 60), (80, 220, 80), (200, 80, 255), (0, 200, 200),
    (255, 80, 180), (160, 160, 255), (80, 160, 120), (0, 120, 255), (255, 200, 0),
]
WIRE_COLOUR = (0, 255, 255)
HIDDEN_COLOUR = (0, 140, 140)
SILHOUETTE_COLOUR = (255, 255, 0)
AXIS_COLOURS = ((0, 0, 255), (0, 200, 0), (255, 0, 0))  # x red, y green, z blue


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def quaternion_wxyz_to_rotation(q: list[float]) -> np.ndarray:
    """Rotation matrix of a quaternion (w, x, y, z); normalises the input."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def sharp_edges(mesh: mgt.Mesh, angle_deg: float = SHARP_ANGLE_DEG) -> np.ndarray:
    """Mesh edges whose adjacent face normals differ by more than ``angle_deg``.

    Boundary edges (one face) and non-manifold edges (more than two faces) count
    as sharp. Degenerate faces are ignored.

    Returns:
        (E, 2) vertex indices.
    """
    v, f = mesh.vertices, mesh.faces
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    norm = np.linalg.norm(normals, axis=1)
    keep = norm > 1e-12
    f, normals = f[keep], normals[keep] / norm[keep, None]
    if len(f) == 0:
        return np.zeros((0, 2), np.int64)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    face_of_edge = np.tile(np.arange(len(f)), 3)
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    face_a = face_of_edge[order[starts]]
    face_b = face_of_edge[order[starts + counts - 1]]
    cos = np.sum(normals[face_a] * normals[face_b], axis=1)
    sharp = (counts != 2) | (cos < np.cos(np.radians(angle_deg)))
    return unique[sharp]


def edge_polylines(
    vertices_cam: np.ndarray,
    edges: np.ndarray,
    K: np.ndarray,
    zbuffer_max: np.ndarray,
    tol_mm: float = EDGE_VISIBILITY_TOL_MM,
    step_px: float = EDGE_STEP_PX,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Project edges and split them into visible and hidden pieces.

    An edge piece is visible when its depth is not behind the rendered scene:
    ``zbuffer_max`` is the z-buffer of all rendered parts after a 3x3 max
    filter, so edges on creases and silhouettes stay visible.

    Pixel coordinates follow ``marker_ground_truth.render_depth``: pixel
    (x, y) covers [x, x + 1) x [y, y + 1).

    Returns:
        (visible, hidden) lists of polylines, each (N, 2) float pixels.
    """
    height, width = zbuffer_max.shape
    visible: list[np.ndarray] = []
    hidden: list[np.ndarray] = []
    for a_idx, b_idx in edges:
        a, b = vertices_cam[a_idx], vertices_cam[b_idx]
        if a[2] <= 1.0 or b[2] <= 1.0:
            continue
        uv_ab = mgt.project_points(K, np.stack([a, b]))
        n = max(1, int(np.ceil(np.linalg.norm(uv_ab[1] - uv_ab[0]) / step_px)))
        ts = np.linspace(0.0, 1.0, n + 1)[:, None]
        pts = a + ts * (b - a)
        uv = mgt.project_points(K, pts)
        mid = (pts[:-1] + pts[1:]) / 2
        uv_mid = mgt.project_points(K, mid)
        ix, iy = np.floor(uv_mid[:, 0]).astype(int), np.floor(uv_mid[:, 1]).astype(int)
        inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        is_visible = np.ones(n, bool)
        is_visible[inside] = mid[inside, 2] <= zbuffer_max[iy[inside], ix[inside]] + tol_mm
        start = 0
        for k in range(1, n + 1):
            if k == n or is_visible[k] != is_visible[start]:
                (visible if is_visible[start] else hidden).append(uv[start : k + 1])
                start = k
    return visible, hidden


# ---------------------------------------------------------------------------
# Loading one capture
# ---------------------------------------------------------------------------


@dataclass
class PartView:
    """Everything the viewer needs about one ground-truth part."""

    index: int
    label: str
    marker_id: Optional[int]
    pose_method: Optional[str]
    bbox: Optional[tuple[int, int, int, int]]
    depth_residual_mm: Optional[float]
    visible_pixels: Optional[int]
    flags: list[str]
    T_cam_part: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None  # bool, full resolution
    mask_missing: bool = False
    residual_bias_mm: Optional[float] = None  # signed median of measured - rendered
    render_bbox: Optional[tuple[int, int, int, int]] = None
    wire_visible: list[np.ndarray] = field(default_factory=list)
    wire_hidden: list[np.ndarray] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    slot: Optional[str] = None  # cell-twin reports: slot, refinement and marker check
    refinement: Optional[dict] = None
    marker_check: Optional[dict] = None
    occluded_fraction: Optional[float] = None

    def summary(self) -> dict:
        """JSON-friendly description for the page."""
        pose = None
        if self.T_cam_part is not None:
            pose = {
                "translation_mm": [round(float(v), 2) for v in self.T_cam_part[:3, 3]],
                "euler_xyz_deg": [round(v, 2) for v in mgt.rotation_to_euler_xyz_deg(self.T_cam_part[:3, :3])],
            }
        return {
            "index": self.index,
            "label": self.label,
            "marker_id": self.marker_id,
            "pose_method": self.pose_method,
            "bbox": list(self.bbox) if self.bbox else None,
            "view_bbox": list(self.bbox or self.render_bbox or ()) or None,
            "depth_residual_mm": self.depth_residual_mm,
            "residual_bias_mm": self.residual_bias_mm,
            "visible_pixels": self.visible_pixels,
            "flags": self.flags,
            "pose": pose,
            "notes": self.notes,
            "slot": self.slot,
            "refinement": self.refinement,
            "marker_check": self.marker_check,
            "occluded_fraction": self.occluded_fraction,
            "colour": "#%02x%02x%02x" % part_colour(self.index)[::-1],
        }


@dataclass
class CaptureView:
    """One capture with its ground truth, rendered once and cached."""

    sample_id: str
    shape: tuple[int, int]
    K: Optional[np.ndarray]
    original: Optional[np.ndarray]
    filled: Optional[np.ndarray]
    measured_z: Optional[np.ndarray]  # float32 mm, NaN invalid
    rendered_z: np.ndarray  # float32 mm, inf where no part
    owner: np.ndarray  # int16 part index, -1 where no part
    parts: list[PartView]
    report: Optional[dict]
    notes: list[str]


def part_colour(index: int) -> tuple[int, int, int]:
    """BGR colour of part ``index``."""
    return PART_COLOURS[index % len(PART_COLOURS)]


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):  # missing, or still being written
        return None


def _mesh_for(config: mgt.Config, marker_id: Optional[int], label: str) -> Optional[mgt.Mesh]:
    if marker_id is not None and marker_id in config.parts:
        return config.parts[marker_id].mesh
    for spec in config.parts.values():
        if spec.label == label:
            return spec.mesh
    return None


def load_capture(run_dir: Path, sample_dir: Path, sample_id: str, config: mgt.Config) -> CaptureView:
    """Load one capture and render its ground-truth models.

    Missing pieces (no report yet, no mask file, no depth, no original image)
    are tolerated and listed in ``notes``.
    """
    run_dir, sample_dir = Path(run_dir), Path(sample_dir)
    notes: list[str] = []
    try:
        sample = mgt.load_sample(sample_dir, sample_id)
        original, K, xyz = sample.rgb, sample.K, sample.xyz
    except (FileNotFoundError, OSError, ValueError) as exc:
        notes.append(f"sample not loadable: {exc}")
        original, xyz = None, None
        k_path = sample_dir / f"{sample_id}_cam_K.txt"
        K = np.loadtxt(k_path).reshape(3, 3) if k_path.exists() else None
    filled = cv2.imread(str(run_dir / "images" / f"{sample_id}.png"), cv2.IMREAD_COLOR)
    if filled is None:
        notes.append("filled-in image missing")
    if original is None and filled is None:
        raise FileNotFoundError(f"{sample_id}: neither original nor filled-in image found")
    shape = (original if original is not None else filled).shape[:2]
    measured_z = xyz[..., 2].astype(np.float32) if xyz is not None else None
    if measured_z is None:
        notes.append("no depth: residual heat map unavailable")

    ann = run_dir / "annotations"
    report = _read_json(ann / f"{sample_id}_report.json")
    poses_doc = _read_json(ann / f"{sample_id}_poses.json") or {}
    poses = poses_doc.get("poses", [])
    if report is None:
        notes.append("no ground-truth report yet")
    if K is None:
        notes.append("no camera matrix: model layers unavailable")

    rendered_z = np.full(shape, np.inf, np.float32)
    owner = np.full(shape, -1, np.int16)
    parts: list[PartView] = []
    meshes: list[Optional[mgt.Mesh]] = []
    for i, entry in enumerate((report or {}).get("parts", [])):
        part = PartView(
            index=i,
            label=entry.get("label", "?"),
            marker_id=entry.get("marker_id"),
            pose_method=entry.get("pose_method"),
            bbox=tuple(entry["bbox_2d"]) if entry.get("bbox_2d") else None,
            depth_residual_mm=entry.get("depth_residual_mm"),
            visible_pixels=entry.get("visible_pixels"),
            flags=list(entry.get("flags", [])),
            slot=entry.get("slot"),
            refinement=entry.get("refinement"),
            marker_check=entry.get("marker_check"),
            occluded_fraction=entry.get("occluded_fraction"),
        )
        part.notes.extend(entry.get("notes", []))
        if i < len(poses) and poses[i].get("rotation_quat") and poses[i].get("translation_mm"):
            part.T_cam_part = mgt.make_transform(
                quaternion_wxyz_to_rotation(poses[i]["rotation_quat"]), np.asarray(poses[i]["translation_mm"], float)
            )
        else:
            part.notes.append("no pose")
        mask = cv2.imread(str(run_dir / "masks" / f"{sample_id}_det_{i}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == shape:
            part.mask = mask > 0
        else:
            part.mask_missing = True
            part.notes.append("mask file missing" + (" (part not visible)" if part.bbox is None else ""))
        mesh = _mesh_for(config, part.marker_id, part.label)
        if mesh is None:
            part.notes.append("no mesh for this label in the config")
        meshes.append(mesh)
        parts.append(part)

    if K is not None:
        for part, mesh in zip(parts, meshes):
            if mesh is None or part.T_cam_part is None:
                continue
            depth = mgt.render_depth(mesh, part.T_cam_part, K, shape)
            closer = depth < rendered_z
            rendered_z[closer] = depth[closer]
            owner[closer] = part.index
        zbuffer_max = cv2.dilate(rendered_z, np.ones((3, 3), np.uint8))
        for part, mesh in zip(parts, meshes):
            if mesh is None or part.T_cam_part is None:
                continue
            verts = (part.T_cam_part[:3, :3] @ mesh.vertices.T).T + part.T_cam_part[:3, 3]
            part.wire_visible, part.wire_hidden = edge_polylines(verts, _edges_for(mesh), K, zbuffer_max)
            ys, xs = np.nonzero(owner == part.index)
            if len(xs):
                part.render_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                if measured_z is not None:
                    diff = measured_z[ys, xs] - rendered_z[ys, xs]
                    diff = diff[np.isfinite(diff)]
                    if len(diff):
                        part.residual_bias_mm = float(np.median(diff))
            else:
                part.notes.append("model not visible in the image")
    return CaptureView(sample_id, shape, K, original, filled, measured_z, rendered_z, owner, parts, report, notes)


_EDGE_CACHE: dict[int, np.ndarray] = {}


def _edges_for(mesh: mgt.Mesh) -> np.ndarray:
    key = id(mesh)
    if key not in _EDGE_CACHE:
        _EDGE_CACHE[key] = sharp_edges(mesh)
    return _EDGE_CACHE[key]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def residual_colours(residual_mm: np.ndarray, clip_mm: float = RESIDUAL_CLIP_MM) -> np.ndarray:
    """Diverging colours (BGR uint8): blue = measured nearer than the model,
    white = agreement, red = measured farther than the model."""
    t = (np.clip(residual_mm, -clip_mm, clip_mm) + clip_mm) / (2 * clip_mm)
    low = np.clip(2 * t, 0, 1)
    high = np.clip(2 - 2 * t, 0, 1)
    b = 255 * high
    g = 255 * np.minimum(low, high)
    r = 255 * low
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


def _blend(img: np.ndarray, where: np.ndarray, colour, alpha: float) -> None:
    if where.any():
        img[where] = (img[where] * (1 - alpha) + np.asarray(colour, float) * alpha).astype(np.uint8)


def _crop_resize(arr: np.ndarray, roi: tuple[int, int, int, int], size: tuple[int, int], interp: int) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return cv2.resize(np.ascontiguousarray(arr[y0:y1, x0:x1]), size, interpolation=interp)


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(p[0]), int(p[1])


def part_roi(view: CaptureView, index: int, margin_px: int = 40) -> tuple[int, int, int, int]:
    """Region around a part (report box, else rendered model), with a margin."""
    part = view.parts[index]
    box = part.bbox or part.render_bbox
    height, width = view.shape
    if box is None:
        return (0, 0, width, height)
    x0, y0, x1, y1 = box
    return (max(0, x0 - margin_px), max(0, y0 - margin_px), min(width, x1 + margin_px), min(height, y1 + margin_px))


def render_view(
    view: CaptureView,
    layers: tuple[str, ...] | list[str] | set[str],
    base: str = "filled",
    scale: float = 0.5,
    roi: Optional[tuple[int, int, int, int]] = None,
    only_part: Optional[int] = None,
) -> np.ndarray:
    """Draw the chosen layers over the original or filled-in image.

    Args:
        view: The loaded capture.
        layers: Any of ``LAYERS``.
        base: ``"original"`` or ``"filled"``; falls back to the other if missing.
        scale: Output pixels per image pixel.
        roi: (x0, y0, x1, y1) crop in image pixels; the whole image if None.
        only_part: Index of a part; the per-part layers (masks, wireframes, axes, boxes and labels) then draw only that part.

    Returns:
        BGR uint8 image of size ``roi * scale``.
    """
    height, width = view.shape
    x0, y0, x1, y1 = roi or (0, 0, width, height)
    size = (max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale))))
    roi = (x0, y0, x1, y1)
    first, second = (view.original, view.filled) if base == "original" else (view.filled, view.original)
    image = first if first is not None else second
    if image is None:
        out = np.full((size[1], size[0], 3), 40, np.uint8)
    else:
        out = _crop_resize(image, roi, size, cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)
    thick = max(1, int(round(min(sx, 2.0))))
    drawn = view.parts if only_part is None else [p for p in view.parts if p.index == only_part]

    def to_px(uv: np.ndarray) -> np.ndarray:
        # Image coordinates (pixel x covers [x, x + 1)) to cv2 drawing coordinates, 4 fractional bits.
        p = np.stack([(uv[:, 0] - x0) * sx - 0.5, (uv[:, 1] - y0) * sy - 0.5], axis=1)
        return np.round(p * 16).astype(np.int32)

    owner = None
    if {"resid", "sil"} & set(layers):
        owner = _crop_resize(view.owner, roi, size, cv2.INTER_NEAREST)
    if "resid" in layers:
        rendered = _crop_resize(view.rendered_z, roi, size, cv2.INTER_NEAREST)
        has_model = np.isfinite(rendered)
        if view.measured_z is not None:
            measured = _crop_resize(view.measured_z, roi, size, cv2.INTER_NEAREST)
            both = has_model & np.isfinite(measured)
            if both.any():
                colours = residual_colours(measured[both] - rendered[both]).astype(float)
                out[both] = (out[both] * 0.25 + colours * 0.75).astype(np.uint8)
            _blend(out, has_model & ~np.isfinite(measured), (90, 90, 90), 0.75)
        else:
            _blend(out, has_model, (90, 90, 90), 0.75)
    if "sil" in layers:
        _blend(out, owner >= 0, SILHOUETTE_COLOUR, 0.35)
    if "masks" in layers:
        for part in drawn:
            if part.mask is None:
                continue
            m = _crop_resize(part.mask.astype(np.uint8), roi, size, cv2.INTER_NEAREST)
            _blend(out, m > 0, part_colour(part.index), 0.4)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, contours, -1, part_colour(part.index), 1)
    if "hidden" in layers:
        for part in drawn:
            lines = [to_px(p) for p in part.wire_hidden]
            if lines:
                cv2.polylines(out, lines, False, HIDDEN_COLOUR, 1, cv2.LINE_AA, shift=4)
    if "wire" in layers:
        for part in drawn:
            lines = [to_px(p) for p in part.wire_visible]
            if lines:
                cv2.polylines(out, lines, False, WIRE_COLOUR, thick, cv2.LINE_AA, shift=4)
    if "axes" in layers and view.K is not None:
        for part in drawn:
            T = part.T_cam_part
            if T is None:
                continue
            pts = np.vstack([T[:3, 3], (T[:3, :3] @ (np.eye(3) * AXIS_LENGTH_MM)).T + T[:3, 3]])
            if np.any(pts[:, 2] <= 1.0):
                continue
            px = to_px(mgt.project_points(view.K, pts))
            for k in range(3):
                cv2.arrowedLine(out, _pt(px[0]), _pt(px[k + 1]), AXIS_COLOURS[k], thick + 1, cv2.LINE_AA, shift=4, tipLength=0.2)
    if "bbox" in layers:
        for part in drawn:
            colour = part_colour(part.index)
            text = f"#{part.index} {part.label}" + (f" (ID {part.marker_id})" if part.marker_id is not None else "")
            if part.bbox is not None:
                bx0, by0, bx1, by1 = part.bbox
                p = to_px(np.array([[bx0, by0], [bx1, by1]], float))
                cv2.rectangle(out, _pt(p[0]), _pt(p[1]), colour, thick, cv2.LINE_AA, shift=4)
                anchor = p[0] // 16
            elif part.render_bbox is not None:
                anchor = to_px(np.array(part.render_bbox[:2], float)[None])[0] // 16
                text += " [not visible]"
            else:
                continue
            org = (int(anchor[0]), int(max(anchor[1] - 5, 12)))
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Run scanning and review store
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_run_samples(run_dir: Path) -> list[str]:
    """Sample IDs with any ground-truth output in the run (report, poses or image)."""
    run_dir = Path(run_dir)
    ids = set()
    for suffix in ("_report.json", "_poses.json", "_detections.json"):
        ids |= {p.name[: -len(suffix)] for p in (run_dir / "annotations").glob(f"*{suffix}")}
    ids |= {p.stem for p in (run_dir / "images").glob("*.png")}
    return sorted(ids)


def review_path(run_dir: Path, sample_id: str) -> Path:
    """Where the verdicts of one capture are stored."""
    return Path(run_dir) / "review" / f"{sample_id}.json"


def load_review(run_dir: Path, sample_id: str) -> dict:
    """Stored verdicts, or an empty review."""
    try:
        return json.loads(review_path(run_dir, sample_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"schema_version": REVIEW_SCHEMA_VERSION, "sample_id": sample_id, "capture": None, "parts": []}


_REVIEW_LOCK = threading.Lock()


def update_review(
    run_dir: Path,
    sample_id: str,
    reviewer: str,
    *,
    gt_created: Optional[str] = None,
    capture_verdict: Optional[str] = None,
    capture_note: Optional[str] = None,
    part_index: Optional[int] = None,
    part_label: Optional[str] = None,
    part_marker_id: Optional[int] = None,
    part_verdict: Optional[str] = None,
    part_note: Optional[str] = None,
) -> dict:
    """Merge one verdict into ``review/<sample_id>.json`` and return the review.

    A capture verdict is set when ``capture_verdict`` or ``capture_note`` is
    given, a part verdict when ``part_index`` is given. Verdict ``None`` with a
    note keeps the earlier verdict. Every record carries reviewer, timestamp
    and the ``created`` time of the ground truth it judged, so a later rerun of
    the ground truth makes the verdict visibly stale.

    Raises:
        ValueError: unknown verdict or empty reviewer.
    """
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise ValueError("reviewer name is required")
    if capture_verdict is not None and capture_verdict not in CAPTURE_VERDICTS:
        raise ValueError(f"capture verdict must be one of {CAPTURE_VERDICTS}")
    if part_verdict is not None and part_verdict not in PART_VERDICTS:
        raise ValueError(f"part verdict must be one of {PART_VERDICTS}")
    with _REVIEW_LOCK:
        review = load_review(run_dir, sample_id)
        stamp = {"reviewer": reviewer, "timestamp": _now(), "gt_created": gt_created}
        if capture_verdict is not None or capture_note is not None:
            old = review.get("capture") or {}
            review["capture"] = {
                "verdict": capture_verdict if capture_verdict is not None else old.get("verdict"),
                "note": capture_note if capture_note is not None else old.get("note", ""),
                **stamp,
            }
        if part_index is not None:
            parts = {p["index"]: p for p in review.get("parts", [])}
            old = parts.get(part_index, {})
            parts[part_index] = {
                "index": part_index,
                "label": part_label if part_label is not None else old.get("label"),
                "marker_id": part_marker_id if part_marker_id is not None else old.get("marker_id"),
                "verdict": part_verdict if part_verdict is not None else old.get("verdict"),
                "note": part_note if part_note is not None else old.get("note", ""),
                **stamp,
            }
            review["parts"] = [parts[k] for k in sorted(parts)]
        review.update(schema_version=REVIEW_SCHEMA_VERSION, sample_id=sample_id, updated=_now(), reviewer=reviewer)
        path = review_path(run_dir, sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(review, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    return review


def review_status(review: dict, n_parts: int, gt_created: Optional[str]) -> str:
    """Short review status for the capture list."""
    records = [r for r in [review.get("capture")] + review.get("parts", []) if r]
    if not records:
        return "unreviewed"
    if gt_created and any(r.get("gt_created") != gt_created for r in records):
        return "stale (GT rerun)"
    capture = review.get("capture") or {}
    if capture.get("verdict") == "accept":
        return "accepted"
    if capture.get("verdict") == "reject":
        return "rejected"
    judged = sum(1 for p in review.get("parts", []) if p.get("verdict"))
    return f"in progress {judged}/{n_parts}"


def scan_run(run_dir: Path) -> list[dict]:
    """One row per capture: script status, parts, worst residual, review status."""
    rows = []
    for sid in list_run_samples(run_dir):
        row = {"sample_id": sid, "gt": "no report yet", "n_parts": None, "open_parts": None, "worst_residual_mm": None, "flags": []}
        report = _read_json(Path(run_dir) / "annotations" / f"{sid}_report.json")
        if report is not None:
            parts = report.get("parts", [])
            residuals = [p["depth_residual_mm"] for p in parts if p.get("depth_residual_mm") is not None]
            row.update(
                gt="rejected" if report.get("rejected") else "accepted",
                n_parts=len(parts),
                worst_residual_mm=max(residuals) if residuals else None,
                flags=sorted({f for p in parts for f in p.get("flags", [])}),
            )
        review = load_review(run_dir, sid)
        row["review"] = review_status(review, row["n_parts"] or 0, (report or {}).get("created"))
        if row["n_parts"] is not None:  # parts without a verdict; none for a rejected capture
            judged = {p["index"] for p in review.get("parts", []) if p.get("verdict")}
            rejected = (review.get("capture") or {}).get("verdict") == "reject"
            row["open_parts"] = 0 if rejected else sum(1 for i in range(row["n_parts"]) if i not in judged)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

STYLE = """
body{font-family:system-ui,sans-serif;margin:0;background:#1b1d21;color:#e4e4e4;font-size:14px}
a{color:#7cc4ff}header{padding:8px 12px;background:#26292f;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
table{border-collapse:collapse;margin:12px}td,th{padding:4px 10px;border-bottom:1px solid #333;text-align:left}
.rejected{color:#ff8a80}.accepted{color:#9be29b}.muted{color:#999}
#wrap{display:flex;gap:10px;padding:10px}#left{width:210px;flex:none}#mid{flex:1;min-width:300px}
#right{width:400px;flex:none;max-height:calc(100vh - 60px);overflow:auto}
#main{max-width:100%;cursor:crosshair;display:block}#zoom{max-width:100%;display:block;margin-top:6px}
.card{background:#26292f;border-left:6px solid;padding:6px 8px;margin-bottom:8px}.card.sel{outline:2px solid #fff}
button{background:#3a3f47;color:#eee;border:1px solid #555;padding:3px 7px;margin:2px;cursor:pointer;border-radius:3px}
button.on{background:#2e7d32;border-color:#9be29b}button.on.bad{background:#b71c1c;border-color:#ff8a80}
input[type=text],textarea{width:100%;box-sizing:border-box;background:#111;color:#eee;border:1px solid #555}
.legend{height:12px;background:linear-gradient(90deg,#0000ff,#ffffff,#ff0000)}kbd{background:#444;padding:0 4px;border-radius:3px}
"""


def _link(sid: str, only_open: bool = False) -> str:
    return "/capture/" + quote(sid, safe="") + ("?open=1" if only_open else "")


def render_index(run_dir: Path, rows: list[dict], only_open: bool = False) -> str:
    """Capture list page; ``only_open`` lists only captures that still have parts without a verdict."""
    body = []
    n_all = len(rows)
    if only_open:
        rows = [r for r in rows if (r["open_parts"] or 0) > 0]
    for n, row in enumerate(rows):
        worst = f"{row['worst_residual_mm']:.2f}" if row["worst_residual_mm"] is not None else "-"
        body.append(
            f"<tr><td>{n + 1}</td><td><a href='{_link(row['sample_id'], only_open)}'>{html.escape(row['sample_id'])}</a></td>"
            f"<td class='{html.escape(row['gt'].split()[0])}'>{html.escape(row['gt'])}</td>"
            f"<td class='{html.escape(row['review'].split()[0])}'>{html.escape(row['review'])}</td>"
            f"<td>{row['n_parts'] if row['n_parts'] is not None else '-'}</td>"
            f"<td>{row['open_parts'] if row['open_parts'] is not None else '-'}</td><td>{worst}</td>"
            f"<td class='muted'>{html.escape(', '.join(row['flags']))}</td></tr>"
        )
    n_rev = sum(r["review"] in ("accepted", "rejected") for r in rows)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>GT review</title><style>{STYLE}</style></head><body>"
        f"<header><b>Ground-truth review</b><span class='muted'>{html.escape(str(run_dir))}</span>"
        f"<span>{len(rows)} of {n_all} captures{' with open parts' if only_open else ''}, {n_rev} reviewed</span>"
        f"<a href='{'/' if only_open else '/?open=1'}'>{'show all captures' if only_open else 'only captures with open parts'}</a>"
        "<span class='muted'>reload to rescan</span></header>"
        "<table><tr><th>#</th><th>capture</th><th>script</th><th>review</th><th>parts</th><th>open parts</th>"
        "<th>worst depth residual (mm)</th><th>flags</th></tr>" + "".join(body) + "</table></body></html>"
    )


CAPTURE_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const LAYERS = ['masks','wire','hidden','sil','axes','bbox','resid'];
const VERDICTS = {c:'correct', p:'pose_wrong', m:'mask_wrong', l:'wrong_label'};
function store(k, v){ try{ if(v===undefined) return localStorage.getItem(k); localStorage.setItem(k, v);}catch(e){ return null; } }
let st = {layers: new Set((store('layers') || 'masks,wire,axes,bbox').split(',').filter(x=>x)),
          base: store('base') || 'filled', sel: D.parts.length ? 0 : -1, review: D.review};
const $ = id => document.getElementById(id);
const OPEN = !!D.open;
function isOpen(i){ const r = (st.review.parts || []).find(p => p.index === i); return !(r && r.verdict); }
function openIdx(){ return D.parts.map(p => p.index).filter(isOpen); }
if (OPEN) { const o = openIdx(); st.sel = o.length ? o[0] : -1; }
$('reviewer').value = store('reviewer') || D.default_reviewer;
$('reviewer').onchange = () => store('reviewer', $('reviewer').value.trim());
function q(){ return 'base=' + st.base + '&layers=' + [...st.layers].join(','); }
function refresh(){
  const w = Math.max(400, Math.round($('mid').clientWidth / 100) * 100);
  $('main').src = D.img + '?' + q() + '&w=' + w;
  $('zoom').style.display = st.sel >= 0 ? 'block' : 'none';
  if (st.sel >= 0) $('zoom').src = D.img + '?' + q() + '&part=' + st.sel + '&w=' + Math.max(380, $('right').clientWidth - 20);
  LAYERS.forEach(l => $('L_' + l).checked = st.layers.has(l));
  $('B_' + st.base).checked = true;
  document.querySelectorAll('.card').forEach(c => { c.classList.toggle('sel', +c.dataset.i === st.sel); c.style.display = (OPEN && !isOpen(+c.dataset.i)) ? 'none' : ''; });
  $('noopen').style.display = (OPEN && !openIdx().length) ? 'block' : 'none';
  const rp = {}; (st.review.parts || []).forEach(p => rp[p.index] = p);
  document.querySelectorAll('[data-v]').forEach(b => {
    const r = rp[+b.dataset.i]; const on = r && r.verdict === b.dataset.v;
    b.classList.toggle('on', !!on); b.classList.toggle('bad', !!on && b.dataset.v !== 'correct'); });
  document.querySelectorAll('.pnote').forEach(n => { if (document.activeElement !== n) n.value = (rp[+n.dataset.i] || {}).note || ''; });
  const cap = st.review.capture || {};
  ['accept','reject'].forEach(v => { const b = $('C_' + v); b.classList.toggle('on', cap.verdict === v); b.classList.toggle('bad', cap.verdict === v && v === 'reject'); });
  if (document.activeElement !== $('cnote')) $('cnote').value = cap.note || '';
  $('rstatus').textContent = st.review.updated ? ('saved ' + st.review.updated + ' by ' + st.review.reviewer) : 'not reviewed';
}
function toggle(l){ st.layers.has(l) ? st.layers.delete(l) : st.layers.add(l); store('layers', [...st.layers].join(',')); refresh(); }
function setBase(b){ st.base = b; store('base', b); refresh(); }
async function save(body){
  let r = $('reviewer').value.trim();
  if (!r) { r = (prompt('Reviewer name') || '').trim(); if (!r) return false; $('reviewer').value = r; store('reviewer', r); }
  body.reviewer = r;
  const res = await fetch(D.api, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  if (!res.ok) { alert('save failed: ' + await res.text()); return false; }
  st.review = await res.json(); refresh(); return true;
}
function partVerdict(i, v){ if (i < 0) return; save({part_index:i, part_verdict:v}).then(ok => {
  if (!ok) return;
  if (OPEN) { const o = openIdx(); const n = o.find(k => k > st.sel); if (n !== undefined) st.sel = n; else if (o.length) st.sel = o[0]; else if (D.next) { location.href = D.next; return; } refresh(); }
  else if (st.sel < D.parts.length - 1) { st.sel++; refresh(); } }); }
function captureVerdict(v){ save({capture_verdict:v, capture_note:$('cnote').value}).then(ok => { if (ok && D.next) location.href = D.next; }); }
function select(i){ st.sel = i; refresh(); }
function go(u){ if (u) location.href = u; }
$('main').onclick = ev => {
  const img = $('main'), s = D.width / img.clientWidth;
  const x = ev.offsetX * s, y = ev.offsetY * s;
  let best = -1, area = Infinity;
  D.parts.forEach(p => { const b = p.view_bbox; if (b && x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3] && (b[2]-b[0])*(b[3]-b[1]) < area) { best = p.index; area = (b[2]-b[0])*(b[3]-b[1]); } });
  if (best >= 0) select(best);
};
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'TEXTAREA') { if (ev.key === 'Escape') ev.target.blur(); return; }
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = ev.key;
  if (k === 'ArrowRight' || k === 'j') go(D.next);
  else if (k === 'ArrowLeft' || k === 'k') go(D.prev);
  else if (k === ']' || k === 'ArrowDown') { if (OPEN) { const n = openIdx().find(x => x > st.sel); if (n !== undefined) select(n); } else if (st.sel < D.parts.length - 1) select(st.sel + 1); }
  else if (k === '[' || k === 'ArrowUp') { if (OPEN) { const n = openIdx().filter(x => x < st.sel).pop(); if (n !== undefined) select(n); } else if (st.sel > 0) select(st.sel - 1); }
  else if (VERDICTS[k]) partVerdict(st.sel, VERDICTS[k]);
  else if (k === 'a') captureVerdict('accept');
  else if (k === 'r') captureVerdict('reject');
  else if (k === 'n') { const n = document.querySelector('.pnote[data-i="' + st.sel + '"]'); if (n) n.focus(); }
  else if (k === '0') setBase(st.base === 'filled' ? 'original' : 'filled');
  else if (k >= '1' && k <= '7') toggle(LAYERS[+k - 1]);
  else if (k === 'u') go('/');
  else return;
  ev.preventDefault();
});
document.querySelectorAll('.pnote').forEach(n => n.addEventListener('change', () => save({part_index:+n.dataset.i, part_note:n.value})));
$('cnote').addEventListener('change', () => save({capture_note:$('cnote').value}));
window.addEventListener('resize', refresh);
refresh();
"""

LAYER_NAMES = {
    "masks": "masks (visible)",
    "wire": "CAD wireframe",
    "hidden": "hidden CAD edges",
    "sil": "CAD silhouette",
    "axes": "axes x/y/z 20 mm",
    "bbox": "boxes + labels",
    "resid": "depth residual",
}


def _fmt(v: Optional[float], digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _twin_lines(p: dict) -> str:
    """Extra card lines of a cell-twin report: refinement corrections and marker check."""
    out = ""
    ref = p.get("refinement") or {}
    corr = ref.get("correction")
    if corr:
        yaw = ref.get("yaw_hypothesis_deg", ref.get("halter_yaw_hypothesis_deg"))
        out += (
            f"<div class='muted'>fit correction: dx {corr['dx_mm']:+.2f}, dy {corr['dy_mm']:+.2f}, dz {corr['dz_mm']:+.2f} mm, "
            f"dyaw {corr['dyaw_deg']:+.2f}&deg; (yaw hypothesis {yaw}&deg;, stage {html.escape(str(ref.get('stage')))})</div>"
        )
    mc = p.get("marker_check")
    if mc and "dx_mm" in mc:
        out += (
            f"<div class='muted'>marker check: ({mc['dx_mm']:+.2f}, {mc['dy_mm']:+.2f}, {mc['dz_mm']:+.2f}) mm, "
            f"tilt {mc['tilt_deg']:.1f}&deg;, yaw {mc['yaw_deg']:+.1f}&deg; ({html.escape(str(mc.get('pose_method')))})</div>"
        )
    elif mc:
        out += f"<div class='muted'>marker check: {html.escape(str(mc.get('status')))}</div>"
    return out


def render_capture_page(view: CaptureView, review: dict, ids: list[str], default_reviewer: str, only_open: bool = False) -> str:
    """Page of one capture: image with layer toggles, part panel, verdicts.

    ``only_open``: show only parts without a verdict; ``ids`` should then list only captures with open parts."""
    sid = view.sample_id
    pos = ids.index(sid) if sid in ids else -1
    prev_id = ids[pos - 1] if pos > 0 else None
    next_id = ids[pos + 1] if 0 <= pos < len(ids) - 1 else None
    parts = [p.summary() for p in view.parts]
    report = view.report or {}
    gt = "no report yet" if view.report is None else ("rejected" if report.get("rejected") else "accepted")
    data = {
        "img": "/img/" + quote(sid, safe=""),
        "api": "/api/review/" + quote(sid, safe=""),
        "prev": _link(prev_id, only_open) if prev_id else None,
        "next": _link(next_id, only_open) if next_id else None,
        "open": only_open,
        "width": view.shape[1],
        "parts": parts,
        "review": review,
        "default_reviewer": default_reviewer,
    }
    data_json = json.dumps(data).replace("</", "<\\/")
    cards = []
    for p in parts:
        pose = p["pose"]
        pose_html = (
            f"t = ({', '.join(_fmt(v, 1) for v in pose['translation_mm'])}) mm<br>"
            f"Euler XYZ = ({', '.join(_fmt(v, 1) for v in pose['euler_xyz_deg'])})&deg;"
            if pose
            else "no pose"
        )
        buttons = "".join(
            f"<button data-i='{p['index']}' data-v='{v}' onclick='partVerdict({p['index']},\"{v}\")'>"
            f"{v.replace('_', ' ')} <kbd>{v[0] if v != 'wrong_label' else 'l'}</kbd></button>"
            for v in PART_VERDICTS
        )
        flags = ", ".join(p["flags"]) or "none"
        notes = "".join(f"<div class='muted'>{html.escape(n)}</div>" for n in p["notes"])
        notes += _twin_lines(p)
        cards.append(
            f"<div class='card' data-i='{p['index']}' style='border-color:{p['colour']}' onclick='select({p['index']})'>"
            f"<b>#{p['index']} {html.escape(p['label'])}</b> &middot; {('slot ' + html.escape(p['slot'])) if p['slot'] else ''} marker ID {p['marker_id']} &middot; {html.escape(str(p['pose_method']))}<br>"
            f"{pose_html}<br>depth residual (median |d|) {_fmt(p['depth_residual_mm'])} mm, bias {_fmt(p['residual_bias_mm'])} mm<br>"
            f"visible px {p['visible_pixels']} &middot; flags: <span class='{'rejected' if p['flags'] else ''}'>{html.escape(flags)}</span>"
            f"{notes}<div>{buttons}</div><input type='text' class='pnote' data-i='{p['index']}' placeholder='note (n)'></div>"
        )
    toggles = "".join(
        f"<label><input type='checkbox' id='L_{name}' onchange='toggle(\"{name}\")'> {LAYER_NAMES[name]} <kbd>{k + 1}</kbd></label><br>"
        for k, name in enumerate(LAYERS)
    )
    notes = "".join(f"<div class='muted'>{html.escape(n)}</div>" for n in view.notes)
    flags_line = ""
    if report.get("unknown_marker_ids"):
        flags_line += f"<div class='rejected'>unknown marker IDs: {report['unknown_marker_ids']}</div>"
    if report.get("plate_position"):
        flags_line += f"<div class='muted'>plate position {html.escape(str(report['plate_position']))}</div>"
    for reason in report.get("rejection_reasons", []):
        flags_line += f"<div class='rejected'>{html.escape(reason)}</div>"
    if report.get("expected_mismatch"):
        flags_line += f"<div class='rejected'>scene mismatch: {html.escape(json.dumps(report['expected_mismatch']))}</div>"
    nav = (
        f"<a href='{data['prev'] or '#'}'>&larr; prev</a> <b>{html.escape(sid)}</b> ({pos + 1}/{len(ids)}) "
        f"<a href='{data['next'] or '#'}'>next &rarr;</a> <a href='/{'?open=1' if only_open else ''}'>list</a> "
        f"<a href='{_link(sid, not only_open)}'>{'show all parts' if only_open else 'only open parts'}</a>"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(sid)} - GT review</title><style>{STYLE}</style></head><body>"
        f"<header>{nav}<span>script: <span class='{gt.split()[0]}'>{gt}</span></span>"
        f"<span id='rstatus' class='muted'></span><span style='margin-left:auto'>reviewer "
        f"<input type='text' id='reviewer' style='width:120px'></span></header>"
        f"<div id='wrap'><div id='left'><b>Base image</b> <kbd>0</kbd><br>"
        f"<label><input type='radio' name='base' id='B_filled' onchange='setBase(\"filled\")'> filled-in</label><br>"
        f"<label><input type='radio' name='base' id='B_original' onchange='setBase(\"original\")'> original</label><br><br>"
        f"<b>Layers</b><br>{toggles}<br><b>Depth residual</b><br>measured &minus; rendered<div class='legend'></div>"
        f"<div style='display:flex;justify-content:space-between'><span>&minus;5</span><span>0</span><span>+5 mm</span></div>"
        f"<div class='muted'>blue: surface nearer than model; red: farther; grey: no depth</div><br>"
        f"<b>Capture</b><br><button id='C_accept' onclick='captureVerdict(\"accept\")'>accept <kbd>a</kbd></button>"
        f"<button id='C_reject' onclick='captureVerdict(\"reject\")'>reject <kbd>r</kbd></button>"
        f"<textarea id='cnote' rows='3' placeholder='capture note'></textarea>{flags_line}{notes}<br>"
        f"<div class='muted'><kbd>&rarr;</kbd>/<kbd>j</kbd> next, <kbd>&larr;</kbd>/<kbd>k</kbd> prev, "
        f"<kbd>[</kbd><kbd>]</kbd> part, <kbd>c</kbd> <kbd>p</kbd> <kbd>m</kbd> <kbd>l</kbd> part verdict, "
        f"<kbd>n</kbd> note, <kbd>u</kbd> list. Click a part to zoom. a/r save and go to the next capture.</div></div>"
        f"<div id='mid'><img id='main' alt='capture'></div>"
        f"<div id='right'><div id='noopen' class='muted' style='display:none'>all parts of this capture have a verdict</div>"
        f"{''.join(cards) or '<div class=muted>no ground-truth parts</div>'}<img id='zoom' alt='zoom'></div></div>"
        f"<script id='data' type='application/json'>{data_json}</script>"
        f"<script>{CAPTURE_JS}</script></body></html>"
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class ReviewApp:
    """State shared by the request handlers: paths, config, capture cache."""

    def __init__(self, run_dir: Path, sample_dir: Path, config: mgt.Config, default_reviewer: str = ""):
        self.run_dir = Path(run_dir)
        self.sample_dir = Path(sample_dir)
        self.config = config
        self.default_reviewer = default_reviewer
        self._cache: OrderedDict[str, tuple[tuple, CaptureView]] = OrderedDict()
        self._lock = threading.Lock()

    def _stamp(self, sid: str) -> tuple:
        paths = [self.run_dir / "annotations" / f"{sid}_report.json", self.run_dir / "annotations" / f"{sid}_poses.json",
                 self.run_dir / "images" / f"{sid}.png", self.sample_dir / f"{sid}_rgb.png"]
        paths += sorted((self.run_dir / "masks").glob(f"{sid}_det_*.png"))
        return tuple((str(p), p.stat().st_mtime_ns) for p in paths if p.exists())

    def capture(self, sid: str) -> CaptureView:
        """Loaded capture, re-rendered when its files change."""
        with self._lock:
            stamp = self._stamp(sid)
            hit = self._cache.get(sid)
            if hit and hit[0] == stamp:
                self._cache.move_to_end(sid)
                return hit[1]
            view = load_capture(self.run_dir, self.sample_dir, sid, self.config)
            self._cache[sid] = (stamp, view)
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
            return view


def make_handler(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    """Request handler bound to ``app``."""

    class Handler(BaseHTTPRequestHandler):
        def log_request(self, code="-", size="-") -> None:  # log only failed requests
            if str(code)[:1] in ("4", "5"):
                super().log_request(code, size)

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _text(self, status: int, text: str) -> None:
            self._send(status, text.encode("utf-8"), "text/plain; charset=utf-8")

        def _sid(self, prefix: str, path: str) -> Optional[str]:
            sid = unquote(path[len(prefix):])
            return sid if sid in list_run_samples(app.run_dir) else None

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            url = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    page = render_index(app.run_dir, scan_run(app.run_dir), query.get("open") == "1")
                    self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                elif url.path.startswith("/capture/"):
                    sid = self._sid("/capture/", url.path)
                    if sid is None:
                        return self._text(404, "unknown capture")
                    only_open = query.get("open") == "1"
                    ids = list_run_samples(app.run_dir)
                    if only_open:  # prev/next skip captures without open parts
                        ids = [r["sample_id"] for r in scan_run(app.run_dir) if (r["open_parts"] or 0) > 0 or r["sample_id"] == sid]
                    page = render_capture_page(
                        app.capture(sid), load_review(app.run_dir, sid), ids, app.default_reviewer, only_open
                    )
                    self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                elif url.path.startswith("/img/"):
                    sid = self._sid("/img/", url.path)
                    if sid is None:
                        return self._text(404, "unknown capture")
                    self._send(200, self._image(app.capture(sid), query), "image/jpeg")
                elif url.path.startswith("/api/review/"):
                    sid = self._sid("/api/review/", url.path)
                    if sid is None:
                        return self._text(404, "unknown capture")
                    self._send(200, json.dumps(load_review(app.run_dir, sid)).encode(), "application/json")
                elif url.path == "/api/captures":
                    self._send(200, json.dumps(scan_run(app.run_dir)).encode(), "application/json")
                else:
                    self._text(404, "not found")
            except Exception as exc:  # keep serving; show the problem in the browser
                self._text(500, f"{exc.__class__.__name__}: {exc}")

        def _image(self, view: CaptureView, query: dict) -> bytes:
            layers = [l for l in query.get("layers", "").split(",") if l in LAYERS]
            base = query.get("base", "filled")
            width = min(max(int(query.get("w", 1200)), 100), 4000)
            only = None
            if "part" in query and 0 <= int(query["part"]) < len(view.parts):
                roi = part_roi(view, int(query["part"]))
                scale = min(width / (roi[2] - roi[0]), 6.0)
                # zoom window: only the wireframe and the label of the part under review
                layers, only = [l for l in layers if l in ZOOM_LAYERS], view.parts[int(query["part"])].index
            else:
                roi, scale = None, min(width / view.shape[1], 1.0)
            img = render_view(view, layers, base, scale, roi, only)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return buf.tobytes()

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if not url.path.startswith("/api/review/"):
                return self._text(404, "not found")
            sid = self._sid("/api/review/", url.path)
            if sid is None:
                return self._text(404, "unknown capture")
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                return self._text(413, "request too large")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                report = _read_json(app.run_dir / "annotations" / f"{sid}_report.json") or {}
                parts = report.get("parts", [])
                kwargs = {}
                if body.get("part_index") is not None:
                    i = int(body["part_index"])
                    if not 0 <= i < len(parts):
                        return self._text(400, "part index out of range")
                    kwargs = dict(part_index=i, part_label=parts[i].get("label"), part_marker_id=parts[i].get("marker_id"),
                                  part_verdict=body.get("part_verdict"), part_note=body.get("part_note"))
                if body.get("capture_verdict") is not None or body.get("capture_note") is not None:
                    kwargs.update(capture_verdict=body.get("capture_verdict"), capture_note=body.get("capture_note"))
                if not kwargs:
                    return self._text(400, "nothing to save")
                review = update_review(app.run_dir, sid, body.get("reviewer", ""), gt_created=report.get("created"), **kwargs)
            except (ValueError, TypeError) as exc:
                return self._text(400, str(exc))
            self._send(200, json.dumps(review).encode(), "application/json")

    return Handler


def make_server(app: ReviewApp, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """HTTP server on 127.0.0.1 only (port 0 picks a free port)."""
    server = ThreadingHTTPServer((HOST, port), make_handler(app))
    server.daemon_threads = True
    return server


def load_any_config(path: Path) -> mgt.Config:
    """Marker configuration, or the part configuration of ``cell_ground_truth.py`` (has ``steps``)."""
    if "steps" in json.loads(Path(path).read_text(encoding="utf-8")):
        import cell_ground_truth as cgt

        return cgt.load_cell_config(path).config
    return mgt.load_config(path)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="output folder of marker_ground_truth.py")
    parser.add_argument("--samples", type=Path, required=True, help="sample folder the run was made from")
    parser.add_argument("--config", type=Path, required=True, help="part configuration used for the run")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reviewer", default="", help="default reviewer name shown in the page")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    app = ReviewApp(args.run_dir, args.samples, load_any_config(args.config), args.reviewer)
    server = make_server(app, args.port)
    print(f"serving {args.run_dir} on http://{HOST}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
