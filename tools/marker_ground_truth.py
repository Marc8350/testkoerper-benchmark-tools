"""Create benchmark ground truth for test bodies from their ArUco markers.

Every marked test body carries a 15 mm ArUco marker (DICT_4X4_50) at the same
place on each copy. The marker gives the part type (marker ID) and a first
6D pose. This tool turns one real capture into benchmark ground truth:

1. Load a capture in the pipeline sample format (SCHNITTSTELLEN.md §5.1):
   ``<id>_rgb.png``, ``<id>_xyz.npy`` (mm, NaN = invalid), ``<id>_cam_K.txt``.
2. Detect all markers and estimate each marker pose in the camera frame
   (``CAM``). Tilt comes from a plane fitted to the measured points on the
   marker, position and yaw from the marker corners intersected with that
   plane. Without usable depth it falls back to PnP on the marker corners.
3. Chain the fixed marker-to-part offset from the configuration to get the
   part pose ``T_cam_part`` in the CAD part frame (``PART``, §3).
4. Render all ground-truth parts with a z-buffer. A pixel belongs to a part if
   the part is the nearest rendered surface there and, where depth is valid,
   the measured depth agrees with the rendered depth. This removes pixels
   hidden by objects that have no ground truth (robot arm, unmarked parts).
   The bounding box is the tight hull of that visible mask.
5. Fill in the marker areas in the colour image so the marker is not a cue.
   Depth is left as measured.
6. Write detections (§5.2), poses (§5.3, frame ``CAM``), YOLO labels for the
   ``objekterkennung/realbilder`` seam, masks and a quality report.

The report flags a sample when the parts found differ from the expected parts
of the scene, or when a part's depth residual is large (marker pose wrong or
part moved). Flagged samples must not enter the test set.

Usage::

    python marker_ground_truth.py SAMPLE_DIR --config marker_parts.json \\
        --out OUT_DIR [--scene scene.json] [--sample-id ID ...]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

SCHEMA_VERSION = "1.0"
DEFAULT_DEPTH_TOLERANCE_MM = 4.0
DEFAULT_RESIDUAL_LIMIT_MM = 2.0
MIN_PLANE_POINTS = 30

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Mesh:
    """Triangle mesh in millimetres, in the CAD part frame."""

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int64


@dataclass
class PartSpec:
    """Configuration of one marked part type."""

    marker_id: int
    label: str
    mesh: Mesh
    T_marker_part: np.ndarray  # (4, 4)


@dataclass
class Config:
    """Marker dictionary, marker size and all marked part types."""

    dictionary: str
    marker_size_mm: float
    parts: dict[int, PartSpec]
    class_names: list[str]


@dataclass
class Sample:
    """One capture in the pipeline sample format."""

    sample_id: str
    rgb: np.ndarray  # (H, W, 3) BGR uint8
    xyz: Optional[np.ndarray]  # (H, W, 3) float32 mm, NaN invalid
    K: np.ndarray  # (3, 3)


@dataclass
class MarkerPose:
    """Pose of one detected marker in the camera frame."""

    marker_id: int
    corners_px: np.ndarray  # (4, 2) TL, TR, BR, BL
    T_cam_marker: np.ndarray  # (4, 4)
    method: str  # "depth_plane" or "pnp"
    n_points: int


@dataclass
class PartGroundTruth:
    """Ground truth of one part instance."""

    marker: MarkerPose
    spec: PartSpec
    T_cam_part: np.ndarray
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
    bbox: Optional[tuple[int, int, int, int]] = None  # x_min, y_min, x_max, y_max
    depth_residual_mm: Optional[float] = None
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a homogeneous 4x4 transform from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def rotation_to_quaternion_wxyz(R: np.ndarray) -> list[float]:
    """Convert a rotation matrix to a unit quaternion (w, x, y, z), w >= 0."""
    m = R
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q) / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def rotation_to_euler_xyz_deg(R: np.ndarray) -> list[float]:
    """Extrinsic XYZ Euler angles in degrees, i.e. R = Rz(c) @ Ry(b) @ Rx(a)."""
    b = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(b)) > 1e-8:
        a = np.arctan2(R[2, 1], R[2, 2])
        c = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: put everything into a
        a = np.arctan2(-R[1, 2], R[1, 1])
        c = 0.0
    return [float(np.degrees(v)) for v in (a, b, c)]


def backproject_ray(K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Unit viewing rays for pixel coordinates (N, 2) in the camera frame."""
    uv = np.atleast_2d(uv).astype(float)
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    rays = np.stack([x, y, np.ones_like(x)], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def project_points(K: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    """Project camera-frame points (N, 3) to pixels (N, 2)."""
    z = points_cam[:, 2]
    return np.stack(
        [K[0, 0] * points_cam[:, 0] / z + K[0, 2], K[1, 1] * points_cam[:, 1] / z + K[1, 2]],
        axis=1,
    )


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through points. Returns (centroid, unit normal)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    return centroid, vt[-1]


# ---------------------------------------------------------------------------
# Mesh loading and rendering
# ---------------------------------------------------------------------------


def load_stl(path: Path, scale_to_mm: float = 1.0) -> Mesh:
    """Read a binary or ASCII STL file into a de-duplicated mesh."""
    data = Path(path).read_bytes()
    triangles: np.ndarray
    is_binary = len(data) >= 84 and 84 + 50 * struct.unpack("<I", data[80:84])[0] == len(data)
    if is_binary:
        n = struct.unpack("<I", data[80:84])[0]
        record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        triangles = np.frombuffer(data, dtype=record, count=n, offset=84)["v"].astype(float)
    else:
        text = data.decode("utf-8", errors="replace").split()
        coords = [float(text[i + 1 : i + 4][k]) for i, tok in enumerate(text) if tok == "vertex" for k in range(3)]
        triangles = np.asarray(coords, dtype=float).reshape(-1, 3, 3)
    flat = triangles.reshape(-1, 3) * scale_to_mm
    vertices, inverse = np.unique(np.round(flat, 6), axis=0, return_inverse=True)
    return Mesh(vertices=vertices, faces=inverse.reshape(-1, 3).astype(np.int64))


def render_depth(mesh: Mesh, T_cam_part: np.ndarray, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Z-buffer render of a mesh. Returns depth in mm, +inf where empty."""
    height, width = shape
    depth = np.full(shape, np.inf)
    verts = (T_cam_part[:3, :3] @ mesh.vertices.T).T + T_cam_part[:3, 3]
    if np.any(verts[:, 2] <= 1e-6):
        verts = verts.copy()
        verts[:, 2] = np.maximum(verts[:, 2], 1e-6)
    uv = project_points(K, verts)
    inv_z = 1.0 / verts[:, 2]
    for face in mesh.faces:
        p = uv[face]
        x0, y0 = np.floor(p.min(axis=0)).astype(int)
        x1, y1 = np.ceil(p.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width - 1), min(y1, height - 1)
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(area) < 1e-12:
            continue
        w0 = ((bx - xs) * (cy - ys) - (cx - xs) * (by - ys)) / area
        w1 = ((cx - xs) * (ay - ys) - (ax - xs) * (cy - ys)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = 1.0 / (w0 * inv_z[face[0]] + w1 * inv_z[face[1]] + w2 * inv_z[face[2]])
        region = depth[y0 : y1 + 1, x0 : x1 + 1]
        update = inside & (z < region)
        region[update] = z[update]
    return depth


def render_depth_roi(
    mesh: Mesh, T_cam_part: np.ndarray, K: np.ndarray, roi: tuple[int, int, int, int]
) -> np.ndarray:
    """Z-buffer render of a mesh into the crop ``roi`` = (x0, y0, x1, y1) only.

    The result equals ``render_depth(...)[y0:y1, x0:x1]`` but costs only the crop.
    """
    x0, y0, x1, y1 = roi
    K_crop = K.copy()
    K_crop[0, 2] -= x0
    K_crop[1, 2] -= y0
    return render_depth(mesh, T_cam_part, K_crop, (y1 - y0, x1 - x0))


def projected_bbox(
    mesh: Mesh, T_cam_part: np.ndarray, K: np.ndarray, shape: tuple[int, int], margin_px: int = 2
) -> Optional[tuple[int, int, int, int]]:
    """Pixel box (x0, y0, x1, y1) around the projected mesh, clipped to the image; None if outside."""
    verts = (T_cam_part[:3, :3] @ mesh.vertices.T).T + T_cam_part[:3, 3]
    if np.any(verts[:, 2] <= 1.0):
        return None
    uv = project_points(K, verts)
    x0, y0 = np.floor(uv.min(axis=0)).astype(int) - margin_px
    x1, y1 = np.ceil(uv.max(axis=0)).astype(int) + margin_px
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, shape[1]), min(y1, shape[0])
    return (int(x0), int(y0), int(x1), int(y1)) if x1 > x0 and y1 > y0 else None


def render_scene(
    items: list[tuple[Mesh, np.ndarray]], K: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer of several posed meshes sharing one depth buffer.

    Args:
        items: (mesh, T_cam_part) per part.
        K: Camera matrix.
        shape: (height, width).

    Returns:
        (nearest depth in mm, +inf where empty; index of the nearest item, -1 where empty).
    """
    depth = np.full(shape, np.inf)
    owner = np.full(shape, -1, np.int32)
    for i, (mesh, T) in enumerate(items):
        roi = projected_bbox(mesh, T, K, shape)
        if roi is None:
            continue
        x0, y0, x1, y1 = roi
        d = render_depth_roi(mesh, T, K, roi)
        region, owner_region = depth[y0:y1, x0:x1], owner[y0:y1, x0:x1]
        nearer = d < region
        region[nearer] = d[nearer]
        owner_region[nearer] = i
    return depth, owner


# ---------------------------------------------------------------------------
# Configuration and sample IO
# ---------------------------------------------------------------------------


def marker_offset_from_top_face(mesh: Mesh, yaw_deg: float = 0.0) -> np.ndarray:
    """Marker-to-part transform for a marker centred on the part's top face.

    Assumes the marker lies flat on the highest face (+Z in ``PART``), centred
    on the mesh's XY extent, with its x axis rotated by ``yaw_deg`` from the
    part's x axis. ``yaw_deg`` must be calibrated once per part type.

    Returns:
        T_marker_part (4x4): maps part coordinates to marker coordinates.
    """
    top = mesh.vertices[:, 2].max()
    centre = np.array(
        [
            (mesh.vertices[:, 0].min() + mesh.vertices[:, 0].max()) / 2,
            (mesh.vertices[:, 1].min() + mesh.vertices[:, 1].max()) / 2,
            top,
        ]
    )
    yaw = np.radians(yaw_deg)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return invert_transform(make_transform(R, centre))


def load_config(path: Path) -> Config:
    """Load the marker/part configuration (JSON).

    Each part entry needs ``label`` and ``mesh`` (STL, relative to the config
    file). It either gives ``T_marker_part`` (4x4, mm) or ``marker_on_top_face``
    with an optional ``yaw_deg``.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    parts: dict[int, PartSpec] = {}
    for marker_id, entry in raw["parts"].items():
        mesh = load_stl(path.parent / entry["mesh"], entry.get("mesh_scale_to_mm", 1.0))
        if "T_marker_part" in entry:
            T = np.asarray(entry["T_marker_part"], dtype=float)
        elif entry.get("marker_on_top_face"):
            T = marker_offset_from_top_face(mesh, entry.get("yaw_deg", 0.0))
        else:
            raise ValueError(f"part {marker_id}: need T_marker_part or marker_on_top_face")
        if T.shape != (4, 4):
            raise ValueError(f"part {marker_id}: T_marker_part must be 4x4")
        parts[int(marker_id)] = PartSpec(int(marker_id), entry["label"], mesh, T)
    class_names = raw.get("class_names") or sorted({p.label for p in parts.values()})
    missing = {p.label for p in parts.values()} - set(class_names)
    if missing:
        raise ValueError(f"labels missing from class_names: {sorted(missing)}")
    return Config(raw.get("marker_dictionary", "DICT_4X4_50"), float(raw["marker_size_mm"]), parts, list(class_names))


def load_sample(sample_dir: Path, sample_id: str) -> Sample:
    """Load ``<id>_rgb.png``, ``<id>_xyz.npy`` (optional) and ``<id>_cam_K.txt``."""
    sample_dir = Path(sample_dir)
    rgb = cv2.imread(str(sample_dir / f"{sample_id}_rgb.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(sample_dir / f"{sample_id}_rgb.png")
    xyz_path = sample_dir / f"{sample_id}_xyz.npy"
    xyz = np.load(xyz_path).astype(np.float32) if xyz_path.exists() else None
    if xyz is not None and xyz.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"{sample_id}: xyz {xyz.shape[:2]} and rgb {rgb.shape[:2]} are not pixel-aligned")
    K = np.loadtxt(sample_dir / f"{sample_id}_cam_K.txt").reshape(3, 3)
    return Sample(sample_id, rgb, xyz, K)


def list_sample_ids(sample_dir: Path) -> list[str]:
    """All sample IDs in a directory (every ``*_rgb.png`` with a ``_cam_K.txt``)."""
    ids = []
    for rgb in sorted(Path(sample_dir).glob("*_rgb.png")):
        sid = rgb.name[: -len("_rgb.png")]
        if (rgb.parent / f"{sid}_cam_K.txt").exists():
            ids.append(sid)
    return ids


# ---------------------------------------------------------------------------
# Marker detection and pose
# ---------------------------------------------------------------------------


def _detector_parameters() -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.01
    params.adaptiveThreshWinSizeMax = 35
    return params


def detect_markers(rgb: np.ndarray, dictionary: str) -> list[tuple[int, np.ndarray]]:
    """Detect all markers, including repeated IDs.

    All copies of one part type carry the same marker ID, so one ID can appear
    several times in an image.

    Returns:
        (marker_id, corners (4, 2) TL, TR, BR, BL), sorted by ID and then by
        image position (top to bottom, left to right).
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    detector = cv2.aruco.ArucoDetector(aruco_dict, _detector_parameters())
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    found = [(int(i), c.reshape(4, 2).astype(float)) for c, i in zip(corners, ids.ravel())]
    found.sort(key=lambda m: (m[0], m[1][:, 1].mean(), m[1][:, 0].mean()))
    return found


def _pose_from_depth(corners: np.ndarray, xyz: np.ndarray, K: np.ndarray) -> Optional[tuple[np.ndarray, int]]:
    mask = np.zeros(xyz.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 1)
    pts = xyz[mask.astype(bool)]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < MIN_PLANE_POINTS:
        return None
    centroid, normal = fit_plane(pts)
    rays = backproject_ray(K, corners)
    denom = rays @ normal
    if np.any(np.abs(denom) < 1e-6):
        return None
    corners3d = rays * ((centroid @ normal) / denom)[:, None]
    tl, tr, br, bl = corners3d
    x = (tr - tl) + (br - bl)
    if normal @ centroid > 0:  # z axis points out of the marker, towards the camera
        normal = -normal
    x -= (x @ normal) * normal
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    return make_transform(np.column_stack([x, y, normal]), corners3d.mean(axis=0)), len(pts)


def _pose_from_pnp(corners: np.ndarray, K: np.ndarray, size_mm: float) -> np.ndarray:
    h = size_mm / 2.0
    object_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=float)
    ok, rvec, tvec = cv2.solvePnP(object_points, corners, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return make_transform(R, tvec.ravel())


def estimate_marker_pose(marker_id: int, corners: np.ndarray, sample: Sample, size_mm: float) -> MarkerPose:
    """Marker pose in CAM. Marker frame: origin at the centre, x along the top
    edge (TL to TR), z out of the marker, y = z x x (OpenCV convention).
    """
    if sample.xyz is not None:
        result = _pose_from_depth(corners, sample.xyz, sample.K)
        if result is not None:
            return MarkerPose(marker_id, corners, result[0], "depth_plane", result[1])
    return MarkerPose(marker_id, corners, _pose_from_pnp(corners, sample.K, size_mm), "pnp", 0)


# ---------------------------------------------------------------------------
# Masks, residuals, inpainting
# ---------------------------------------------------------------------------


def compute_visible_masks(
    parts: list[PartGroundTruth], sample: Sample, tolerance_mm: float = DEFAULT_DEPTH_TOLERANCE_MM
) -> None:
    """Fill ``mask``, ``bbox`` and ``depth_residual_mm`` of every part in place."""
    shape = sample.rgb.shape[:2]
    if not parts:
        return
    depths = np.stack([render_depth(p.spec.mesh, p.T_cam_part, sample.K, shape) for p in parts])
    nearest = np.argmin(depths, axis=0)
    nearest_depth = np.min(depths, axis=0)
    measured = sample.xyz[..., 2] if sample.xyz is not None else np.full(shape, np.nan)
    valid = np.isfinite(measured)
    for i, part in enumerate(parts):
        rendered = (nearest == i) & np.isfinite(nearest_depth)
        agrees = ~valid | (np.abs(measured - nearest_depth) <= tolerance_mm)
        part.mask = rendered & agrees
        both = rendered & valid
        if both.any():
            part.depth_residual_mm = float(np.median(np.abs(measured[both] - nearest_depth[both])))
        ys, xs = np.nonzero(part.mask)
        if len(xs):
            part.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            part.flags.append("not_visible")


def marker_region(corners: np.ndarray, shape: tuple[int, int], margin_px: int) -> np.ndarray:
    """Binary mask of the marker quad, grown by ``margin_px``."""
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
    if margin_px > 0:
        mask = cv2.dilate(mask, np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8))
    return mask


def _side_px(corners: np.ndarray) -> float:
    return float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))


def quiet_zone_widths(rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Width in px of the bright sticker frame outside each side of the marker quad.

    Measured from the brightness profile along the outward normal of each side. A side
    whose profile never drops (a bright neighbour touches the sticker) gets the median of
    the other sides; all zero if there is no clearly brighter frame.
    """
    lum = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
    cell = _side_px(corners) / 6.0
    steps = np.arange(0.0, 2.0 * cell, 0.25)
    center = corners.mean(0)
    ts = np.linspace(0.25, 0.75, 9)
    widths = np.full(4, np.nan)
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        normal = (a + b) / 2 - center
        normal /= np.linalg.norm(normal)
        points = a[None] * (1 - ts[:, None]) + b[None] * ts[:, None]
        xs = (points[:, 0][None] + steps[:, None] * normal[0]).astype(np.float32)
        ys = (points[:, 1][None] + steps[:, None] * normal[1]).astype(np.float32)
        profile = cv2.remap(lum, xs, ys, cv2.INTER_LINEAR).mean(1)
        frame, background = float(profile[: int(cell * 2 * 4)].max()), float(np.percentile(profile, 10))
        if frame - background < 10.0:
            widths[k] = 0.0
            continue
        peak = int(np.argmax(profile))
        below = np.nonzero(profile[peak:] < background + 0.5 * (frame - background))[0]
        if len(below):
            widths[k] = steps[peak + below[0]]
    widths[np.isnan(widths)] = np.nanmedian(widths) if not np.isnan(widths).all() else 0.0
    return widths


def quiet_zone_px(rgb: np.ndarray, corners: np.ndarray) -> float:
    """Median width in px of the bright sticker frame (see ``quiet_zone_widths``)."""
    return float(np.median(quiet_zone_widths(rgb, corners)))


def sticker_region(rgb: np.ndarray, corners: np.ndarray, margin_px: int) -> np.ndarray:
    """Binary mask of the whole marker sticker: each quad side moved out by its own frame width + ``margin_px``."""
    center = corners.mean(0)
    offset = quiet_zone_widths(rgb, corners) + margin_px
    lines = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        normal = (a + b) / 2 - center
        normal /= np.linalg.norm(normal)
        shifted = a + normal * offset[k]
        direction = b - a
        lines.append(np.cross([*shifted, 1.0], [*(shifted + direction), 1.0]))
    grown = np.array([np.cross(lines[k - 1], lines[k]) for k in range(4)])
    grown = grown[:, :2] / grown[:, 2:3]
    return marker_region(grown, rgb.shape[:2], 0)


def _inpaint_opencv(method: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def run(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(rgb, mask, 7, method)

    return run


def _fill_surface_region(img: np.ndarray, region: np.ndarray, feather_px: float, rng: np.random.Generator) -> None:
    """Replace ``region`` of float image ``img`` by the surrounding surface, in place.

    Only border pixels as dark as the part itself feed the smooth fill, so a bright
    neighbour (aluminium plate, sticker frame) cannot bleed in. The sensor noise of
    the surroundings is added back and the edge is feathered.
    """
    region_u8 = region.astype(np.uint8)
    band = (cv2.dilate(region_u8, np.ones((25, 25), np.uint8)) > 0) & ~region
    lum = cv2.GaussianBlur(img.mean(2), (0, 0), 0.8)
    dark = np.sort(lum[band])[: max(1, int(band.sum()) // 2)]
    level = float(np.median(dark))
    spread = float(np.median(np.abs(dark - level)))
    surface = ~region & (lum < level + max(12.0, 4.0 * spread))
    valid = surface.astype(np.float32)
    fill = np.zeros_like(img)
    done = np.zeros(region.shape, bool)
    for sigma in (2, 4, 8, 16, 32, 64):
        weight = cv2.GaussianBlur(valid, (0, 0), sigma)
        colour = cv2.GaussianBlur(img * valid[..., None], (0, 0), sigma)
        ok = (weight > 0.02) & region & ~done
        fill[ok] = colour[ok] / weight[ok, None]
        done |= ok
    fill[region & ~done] = img[surface].mean(0) if surface.any() else img[~region].mean(0)
    near = (cv2.dilate(region_u8, np.ones((17, 17), np.uint8)) > 0) & surface
    detail = img - cv2.GaussianBlur(img, (0, 0), 1.5)
    sigma_noise = 1.4826 * np.median(np.abs(detail[near] - np.median(detail[near], 0)), 0) if near.sum() > 50 else np.zeros(3)
    shared = rng.normal(0, 1, region.shape).astype(np.float32)
    own = rng.normal(0, 1, img.shape).astype(np.float32)
    noise = cv2.GaussianBlur(0.8 * shared[..., None] + 0.6 * own, (0, 0), 0.6)
    noise /= max(float(noise.std()), 1e-6)
    alpha = np.clip(cv2.distanceTransform(region_u8, cv2.DIST_L2, 3) / feather_px, 0, 1)[..., None]
    img[:] = img * (1 - alpha) + (fill + noise * sigma_noise) * alpha


def _inpaint_surface(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = rgb.astype(np.float32)
    rng = np.random.default_rng(0)
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    for k in range(1, count):
        _fill_surface_region(img, labels == k, 1.0, rng)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


INPAINT_BACKENDS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "surface": _inpaint_surface,
    "telea": _inpaint_opencv(cv2.INPAINT_TELEA),
    "ns": _inpaint_opencv(cv2.INPAINT_NS),
}


def inpaint_markers(rgb: np.ndarray, corners_list: list[np.ndarray], backend: str, margin_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill all marker stickers. Returns (inpainted image, union mask)."""
    union = np.zeros(rgb.shape[:2], np.uint8)
    for corners in corners_list:
        union |= sticker_region(rgb, corners, margin_px)
    if not union.any():
        return rgb.copy(), union
    return INPAINT_BACKENDS[backend](rgb, union), union


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def detections_record(sample_id: str, parts: list[PartGroundTruth]) -> dict:
    """§5.2 detection file with ground-truth boxes and mask references."""
    detections = []
    for i, part in enumerate(parts):
        if part.bbox is None:
            continue
        detections.append(
            {
                "object_id": part.spec.label,
                "class_confidence": 1.0,
                "bbox_2d": list(part.bbox),
                "mask_ref": f"masks/{sample_id}_det_{i}.png",
                "marker_id": part.marker.marker_id,
            }
        )
    return {"schema_version": SCHEMA_VERSION, "timestamp": _timestamp(), "capture_id": sample_id, "detections": detections}


def poses_record(sample_id: str, parts: list[PartGroundTruth]) -> dict:
    """§5.3 pose file in frame CAM (WORK is not yet defined in §2)."""
    poses = []
    for part in parts:
        R, t = part.T_cam_part[:3, :3], part.T_cam_part[:3, 3]
        poses.append(
            {
                "object_id": part.spec.label,
                "translation_mm": [float(v) for v in t],
                "rotation_quat": rotation_to_quaternion_wxyz(R),
                "rotation_euler_xyz_deg": rotation_to_euler_xyz_deg(R),
                "uncertainty": None,
                "method": (
                    f"ArUco marker {part.marker.marker_id} ({part.marker.method}) + fixed marker-to-part offset; "
                    "uncertainty null: ground-truth accuracy not yet measured (repeatability study pending)"
                ),
                "n_points_used": part.marker.n_points,
                "fitness": None,
            }
        )
    return {"schema_version": SCHEMA_VERSION, "timestamp": _timestamp(), "capture_id": sample_id, "frame": "CAM", "poses": poses}


def yolo_lines(parts: list[PartGroundTruth], class_names: list[str], shape: tuple[int, int]) -> list[str]:
    """YOLO label lines (class cx cy w h, normalised) for visible parts."""
    height, width = shape
    lines = []
    for part in parts:
        if part.bbox is None:
            continue
        x0, y0, x1, y1 = part.bbox
        lines.append(
            f"{class_names.index(part.spec.label)} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"
        )
    return lines


def expected_mismatch(parts: list[PartGroundTruth], expected: Optional[dict[str, int]]) -> dict[str, dict[str, int]]:
    """Per-label expected vs found counts where they differ."""
    if expected is None:
        return {}
    found: dict[str, int] = {}
    for part in parts:
        found[part.spec.label] = found.get(part.spec.label, 0) + 1
    labels = set(expected) | set(found)
    return {
        label: {"expected": expected.get(label, 0), "found": found.get(label, 0)}
        for label in sorted(labels)
        if expected.get(label, 0) != found.get(label, 0)
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_sample(
    sample: Sample,
    config: Config,
    out_dir: Path,
    expected: Optional[dict[str, int]] = None,
    inpaint_backend: str = "surface",
    inpaint_margin_px: int = 1,
    depth_tolerance_mm: float = DEFAULT_DEPTH_TOLERANCE_MM,
    residual_limit_mm: float = DEFAULT_RESIDUAL_LIMIT_MM,
) -> dict:
    """Create all ground-truth files for one sample. Returns the report."""
    out_dir = Path(out_dir)
    markers = detect_markers(sample.rgb, config.dictionary)
    unknown_ids = sorted({marker_id for marker_id, _ in markers} - set(config.parts))
    parts = []
    for marker_id, corners in markers:
        if marker_id not in config.parts:
            continue
        spec = config.parts[marker_id]
        marker = estimate_marker_pose(marker_id, corners, sample, config.marker_size_mm)
        T_cam_part = marker.T_cam_marker @ spec.T_marker_part
        parts.append(PartGroundTruth(marker, spec, T_cam_part))
    compute_visible_masks(parts, sample, depth_tolerance_mm)
    for part in parts:
        if part.marker.method != "depth_plane":
            part.flags.append("pose_from_pnp")
        if part.depth_residual_mm is not None and part.depth_residual_mm > residual_limit_mm:
            part.flags.append("depth_residual_high")

    inpainted, marker_mask = inpaint_markers(
        sample.rgb, [corners for _, corners in markers], inpaint_backend, inpaint_margin_px
    )
    mismatch = expected_mismatch(parts, expected)
    rejected = bool(mismatch) or any(part.flags for part in parts)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "images" / f"{sample.sample_id}.png"), inpainted)
    cv2.imwrite(str(out_dir / "masks" / f"{sample.sample_id}_markers.png"), marker_mask)
    for i, part in enumerate(parts):
        if part.bbox is not None:
            cv2.imwrite(str(out_dir / "masks" / f"{sample.sample_id}_det_{i}.png"), part.mask.astype(np.uint8) * 255)
    (out_dir / "labels" / f"{sample.sample_id}.txt").write_text(
        "".join(line + "\n" for line in yolo_lines(parts, config.class_names, sample.rgb.shape[:2]))
    )
    (out_dir / "klassen.txt").write_text("".join(name + "\n" for name in config.class_names))
    ann = out_dir / "annotations"
    (ann / f"{sample.sample_id}_detections.json").write_text(json.dumps(detections_record(sample.sample_id, parts), indent=2))
    (ann / f"{sample.sample_id}_poses.json").write_text(json.dumps(poses_record(sample.sample_id, parts), indent=2))

    report = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "created": _timestamp(),
        "rejected": rejected,
        "unknown_marker_ids": unknown_ids,
        "expected_mismatch": mismatch,
        "inpaint_backend": inpaint_backend,
        "parts": [
            {
                "marker_id": part.marker.marker_id,
                "label": part.spec.label,
                "pose_method": part.marker.method,
                "bbox_2d": list(part.bbox) if part.bbox else None,
                "visible_pixels": int(part.mask.sum()),
                "depth_residual_mm": part.depth_residual_mm,
                "flags": part.flags,
            }
            for part in parts
        ],
    }
    (ann / f"{sample.sample_id}_report.json").write_text(json.dumps(report, indent=2))
    return report


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="JSON {sample_id: {label: expected_count}}")
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--inpaint", default="surface", choices=sorted(INPAINT_BACKENDS))
    parser.add_argument("--inpaint-margin-px", type=int, default=1)
    parser.add_argument("--depth-tolerance-mm", type=float, default=DEFAULT_DEPTH_TOLERANCE_MM)
    parser.add_argument("--residual-limit-mm", type=float, default=DEFAULT_RESIDUAL_LIMIT_MM)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    scene = json.loads(args.scene.read_text()) if args.scene else {}
    sample_ids = args.sample_ids or list_sample_ids(args.sample_dir)
    if not sample_ids:
        print(f"no samples in {args.sample_dir}", file=sys.stderr)
        return 1
    n_rejected = 0
    for sid in sample_ids:
        report = process_sample(
            load_sample(args.sample_dir, sid),
            config,
            args.out,
            expected=scene.get(sid),
            inpaint_backend=args.inpaint,
            inpaint_margin_px=args.inpaint_margin_px,
            depth_tolerance_mm=args.depth_tolerance_mm,
            residual_limit_mm=args.residual_limit_mm,
        )
        n_rejected += report["rejected"]
        status = "REJECTED" if report["rejected"] else "ok"
        print(f"{sid}: {len(report['parts'])} parts, {status}")
    print(f"{len(sample_ids) - n_rejected}/{len(sample_ids)} samples accepted")
    return 1 if n_rejected else 0


if __name__ == "__main__":
    sys.exit(main())
