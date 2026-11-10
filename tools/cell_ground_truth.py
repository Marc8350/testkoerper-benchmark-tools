"""Benchmark ground truth from the cell geometry, with markers as a check.

``marker_ground_truth.py`` derives part poses from ArUco markers. That chain is
limited by the marker-to-part offset and by noisy marker tilt, and it gives no
pose for the unmarked parts (Unterteil, Halter). This tool takes the cell
geometry instead:

1. **Slot calibration** (``calibrate``). Per plate position (the two 3x3
   aluminium tray plates stand at a few fixed places), the empty captures give
   the top plane of each plate, the pocket floor plane and the 3x3 lattice of
   pocket centres. Output: ``T_cam_slot`` for the 18 slots ``L1..L9``,
   ``R1..R9`` (row-major from the top left as seen in the camera image, ``L``
   is the left plate in the image). The slot frame sits at the pocket centre on
   the pocket *floor*, ``z`` is the plate normal towards the camera, ``x`` runs
   along the lattice columns (image right), ``y = z x x`` (image up).
2. **Layout to parts.** ``capture_log.csv`` says which slot holds what (``U``
   Unterteil, ``H`` empty Halter, ``HO`` Halter with Oberteil, ``E`` elastic).
3. **Initial pose** ``T_cam_slot @ T_slot_part``. The parts stand with their
   CAD bottom face on the pocket floor, so ``T_slot_part`` is a rotation about
   the slot ``z`` by a multiple of 90 degrees. The Oberteil sits at
   ``T_halter_oberteil`` in its Halter (CAD).
4. **Bounded refinement against the depth.** Four yaw hypotheses, then x, y
   within +-3 mm, yaw within +-3 degrees and z within +-1 mm. Cost: truncated
   depth difference to the model, where the model is the part (or, for a Halter
   with Oberteil, both rigidly linked) in front of the empty-plate depth. Every
   correction is logged; parts that hit a bound are flagged.
5. **Markers as an independent check.** The predicted marker pose from the
   chain is compared with the detected marker. The discrepancy measures the
   ground-truth error; it does not drive the pose.
6. **Masks** from one z-buffer of all parts with the capture's intrinsics, minus
   pixels where the measured depth is clearly closer than the rendered surface.
   Missing depth does not remove pixels (black parts have depth holes).

The output folder has the layout of ``marker_ground_truth.py`` and works with
``review_gui.py``. Usage::

    python cell_ground_truth.py calibrate SAMPLE_DIR --positions positions.json --out calib.json
    python cell_ground_truth.py run SAMPLE_DIR --config cell_parts.json \\
        --calibration calib.json --positions positions.json \\
        --capture-log capture_log.csv --out OUT_DIR [--sample-id ID ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

import marker_ground_truth as mgt

SCHEMA_VERSION = "1.0"
RANSAC_SEED = 12345
BAND_MM = (800.0, 1300.0)  # plausible depth of the plates
PLANE_INLIER_MM = 0.7
FLOOR_GAP_RANGE_MM = (8.0, 30.0)  # pocket depth window searched for the second plane
RIM_STEP_MM = 0.25
RIM_RUN_SAMPLES = 10
RIM_RUN_MIN_HITS = 8  # of RIM_RUN_SAMPLES consecutive samples on the top plane
RIM_SCAN_FROM_MM = 6.0
RIM_SCAN_TO_MM = 45.0
RIM_LINE_OFFSETS_MM = np.linspace(-10.0, 10.0, 9)
PITCH_NOMINAL_MM = 60.0
MIN_RIM_LINES = 5
GRID_OUTLIER_MM = 1.5
MIN_RIM_POCKETS = 6
POCKET_FLOOR_MIN_PX = 1500
SLOT_NAMES = [f"{side}{k}" for side in "LR" for k in range(1, 10)]

# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def rot_z(deg: float) -> np.ndarray:
    """3x3 rotation about z by ``deg`` degrees."""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def translate(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """4x4 pure translation."""
    return mgt.make_transform(np.eye(3), np.array([x, y, z], dtype=float))


def pose_from_rz(yaw_deg: float, t: Optional[np.ndarray] = None) -> np.ndarray:
    """4x4 transform with rotation about z by ``yaw_deg`` and translation ``t``."""
    return mgt.make_transform(rot_z(yaw_deg), np.zeros(3) if t is None else np.asarray(t, dtype=float))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Closest rotation matrix (SVD), with determinant +1."""
    u, _, vt = np.linalg.svd(R)
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1
        out = u @ vt
    return out


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    """Mean of rigid transforms: mean translation, orthonormalised mean rotation."""
    R = orthonormalize(np.mean([T[:3, :3] for T in transforms], axis=0))
    return mgt.make_transform(R, np.mean([T[:3, 3] for T in transforms], axis=0))


def wrap_deg(angle: float, period: float = 360.0) -> float:
    """Wrap an angle to ``[-period/2, period/2)``."""
    return float((angle + period / 2.0) % period - period / 2.0)


def yaw_of(R: np.ndarray) -> float:
    """Rotation about z of ``R`` in degrees (``atan2(R10, R00)``)."""
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


# ---------------------------------------------------------------------------
# Plane fitting
# ---------------------------------------------------------------------------


def fit_plane_toward_camera(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane. Returns (centroid, unit normal with ``n_z < 0``).

    The camera looks along +z, so a normal with negative z points towards it.
    """
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[2] > 0:
        normal = -normal
    return centroid, normal


def ransac_plane(
    points: np.ndarray, threshold_mm: float, rng: np.random.Generator, iterations: int = 400, max_points: int = 40000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC plane with a least-squares refit on the inliers.

    Args:
        points: (N, 3) float64 points.
        threshold_mm: Inlier distance.
        rng: Seeded generator, so the ground truth is reproducible.

    Returns:
        (centroid, normal towards the camera, boolean inlier mask over ``points``).

    Raises:
        ValueError: Fewer than 3 usable points.
    """
    if len(points) < 3:
        raise ValueError("not enough points for a plane")
    sample = points if len(points) <= max_points else points[rng.choice(len(points), max_points, replace=False)]
    best: Optional[np.ndarray] = None
    for _ in range(iterations):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        inliers = np.abs((sample - tri[0]) @ (normal / norm)) < threshold_mm
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None:
        raise ValueError("degenerate points")
    for _ in range(4):
        centroid, normal = fit_plane_toward_camera(sample[best])
        best = np.abs((sample - centroid) @ normal) < threshold_mm
    centroid, normal = fit_plane_toward_camera(sample[best])
    return centroid, normal, np.abs((points - centroid) @ normal) < threshold_mm


def plane_depth_along_rays(
    K: np.ndarray, centroid: np.ndarray, normal: np.ndarray, shape: tuple[int, int], roi: tuple[int, int, int, int]
) -> np.ndarray:
    """Depth (z, mm) of a plane at every pixel centre of ``roi`` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = roi
    u, v = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    denom = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        return (centroid @ normal) / denom


# ---------------------------------------------------------------------------
# Slot calibration
# ---------------------------------------------------------------------------


@dataclass
class PlateCalibration:
    """Geometry of one plate (``L`` or ``R``) at one plate position."""

    side: str
    top_point: np.ndarray  # a point on the top plane
    normal: np.ndarray  # unit, towards the camera
    floor_depth_mm: float  # distance of the pocket floor below the top plane
    slots: dict[str, np.ndarray]  # "L1".."L9" -> T_cam_slot (floor-centred)
    pitch_mm: tuple[float, float]  # column step, row step
    grid_angle_deg: float  # angle between column and row step, 90 for a square grid
    top_rms_mm: float
    floor_rms_mm: float
    lattice_rms_mm: float  # rms distance of the measured pocket centres to the lattice
    pocket_size_mm: tuple[float, float]  # rim to rim, along columns and rows
    n_pockets_measured: int
    measured: dict[str, bool] = field(default_factory=dict)  # slot -> centre measured (not extrapolated)
    deviation_mm: dict[str, float] = field(default_factory=dict)  # slot -> distance to lattice
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """JSON-friendly description."""
        return {
            "side": self.side,
            "top_point_mm": [round(float(v), 3) for v in self.top_point],
            "normal_to_camera": [round(float(v), 6) for v in self.normal],
            "tilt_deg": round(float(np.degrees(np.arccos(-self.normal[2]))), 3),
            "floor_depth_mm": round(self.floor_depth_mm, 3),
            "pitch_mm": [round(p, 3) for p in self.pitch_mm],
            "grid_angle_deg": round(self.grid_angle_deg, 3),
            "top_plane_rms_mm": round(self.top_rms_mm, 3),
            "floor_plane_rms_mm": round(self.floor_rms_mm, 3),
            "lattice_rms_mm": round(self.lattice_rms_mm, 3),
            "pocket_size_mm": [round(p, 2) for p in self.pocket_size_mm],
            "n_pockets_measured": self.n_pockets_measured,
            "measured": self.measured,
            "deviation_from_lattice_mm": {k: round(v, 3) for k, v in self.deviation_mm.items()},
            "warnings": self.warnings,
        }


def _plate_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane axes (image right, image down) from the plate normal."""
    e1 = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.array([0.0, 1.0, 0.0]) - normal[1] * normal
    e2 -= (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _lookup_mask(mask: np.ndarray, K: np.ndarray, points: np.ndarray, roi_origin: tuple[int, int]) -> np.ndarray:
    """Mask value at the pixels where camera-frame ``points`` project (False outside)."""
    uv = mgt.project_points(K, points)
    ix = np.floor(uv[:, 0]).astype(int) - roi_origin[0]
    iy = np.floor(uv[:, 1]).astype(int) - roi_origin[1]
    inside = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
    out = np.zeros(len(points), bool)
    out[inside] = mask[iy[inside], ix[inside]]
    return out


def _rim_position(
    top_mask: np.ndarray, K: np.ndarray, roi_origin: tuple[int, int], origin: np.ndarray, direction: np.ndarray, perp: np.ndarray
) -> Optional[float]:
    """Distance from ``origin`` along ``direction`` to the rim of the pocket.

    Casts parallel lines (offsets along ``perp``) in the top plane and takes the
    first place where the top-plane mask holds for a run of samples; the median
    over the lines is returned, or None when too few lines found a rim.
    """
    ts = np.arange(RIM_SCAN_FROM_MM, RIM_SCAN_TO_MM, RIM_STEP_MM)
    found = []
    for offset in RIM_LINE_OFFSETS_MM:
        pts = origin + ts[:, None] * direction + offset * perp
        hit = _lookup_mask(top_mask, K, pts, roi_origin)
        run = np.convolve(hit.astype(int), np.ones(RIM_RUN_SAMPLES, int), mode="valid") >= RIM_RUN_MIN_HITS
        if run.any():
            found.append(ts[int(np.argmax(run))])
    return float(np.median(found)) if len(found) >= MIN_RIM_LINES else None


def _fit_axis_lattice(
    ij: np.ndarray, values: np.ndarray, weights: np.ndarray, tolerance_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares ``value = o + i * a + j * b`` with outlier removal.

    Returns:
        (parameters (o, a, b), boolean mask of the observations that were used).

    Raises:
        ValueError: Fewer than 3 observations or a degenerate arrangement.
    """
    use = np.ones(len(values), bool)
    A = np.column_stack([np.ones(len(ij)), ij[:, 0], ij[:, 1]]).astype(float)
    for _ in range(3):
        if use.sum() < 3 or np.linalg.matrix_rank(A[use]) < 3:
            raise ValueError("lattice is under-determined")
        w = np.sqrt(weights[use])[:, None]
        sol, *_ = np.linalg.lstsq(A[use] * w, values[use] * w[:, 0], rcond=None)
        resid = np.abs(A @ sol - values)
        new_use = resid < tolerance_mm
        if new_use.sum() < 3 or np.array_equal(new_use, use):
            break
        use = new_use
    return sol, use


@dataclass
class PocketMeasurements:
    """Raw pocket measurements of one plate in one empty capture (plate coordinates, mm)."""

    side: str
    top_point: np.ndarray
    normal: np.ndarray
    floor_depth_mm: float
    top_rms_mm: float
    floor_rms_mm: float
    top_plane_found: bool
    axes: tuple[np.ndarray, np.ndarray]  # e1 (image right), e2 (image down), in the plane
    seed_uv: np.ndarray  # (N, 2) floor-region centroids
    floor_px: np.ndarray  # (N,) floor-region areas
    rim_uv: np.ndarray  # (N, 2) rim-based centres, NaN where a rim was not found
    rim_size: np.ndarray  # (N, 2) rim-to-rim sizes, NaN where not found
    warnings: list[str] = field(default_factory=list)


def measure_pockets(
    sample: mgt.Sample,
    side: str,
    roi: tuple[int, int, int, int],
    rng: Optional[np.random.Generator] = None,
    floor_depth_prior_mm: Optional[float] = None,
) -> PocketMeasurements:
    """Fit the plate planes and measure the pockets of one plate in an empty capture.

    The pocket floor is the largest plane in the ROI or the second one; the top
    plane is parallel to it, 17 mm nearer to the camera. When the top plane is
    not measurable (plate top dark or glossy so that depth is missing) it is
    taken from the floor plane and ``floor_depth_prior_mm``, no rims are
    measured, and a warning is added.

    Args:
        sample: Empty capture (colour, ``xyz`` in mm, intrinsics).
        side: ``"L"`` or ``"R"`` (slot name prefix).
        roi: (x0, y0, x1, y1) pixel box around the plate.
        rng: Seeded generator; the default is seeded with ``RANSAC_SEED``.
        floor_depth_prior_mm: Pocket depth to use when the top plane is missing.

    Raises:
        ValueError: Planes or pockets cannot be found.
    """
    if sample.xyz is None:
        raise ValueError("slot calibration needs depth")
    rng = rng or np.random.default_rng(RANSAC_SEED)
    x0, y0, x1, y1 = roi
    K = sample.K
    cloud = sample.xyz[y0:y1, x0:x1].astype(np.float64)
    finite = np.isfinite(cloud).all(axis=-1)
    in_band = finite & (cloud[..., 2] > BAND_MM[0]) & (cloud[..., 2] < BAND_MM[1])
    points = cloud[in_band]
    if len(points) < 5000:
        raise ValueError(f"plate {side}: only {len(points)} points in the depth band inside the ROI")
    warnings: list[str] = []

    # 1. The two parallel planes: plate top and pocket floor.
    centroid_a, normal_a, inl_a = ransac_plane(points, PLANE_INLIER_MM, rng)
    gap = np.abs((points - centroid_a) @ normal_a)
    second = (gap > FLOOR_GAP_RANGE_MM[0]) & (gap < FLOOR_GAP_RANGE_MM[1])
    plane_b = None
    if second.sum() >= 3000:
        centroid_b, normal_b, inl_b = ransac_plane(points[second], PLANE_INLIER_MM, rng)
        skew = float(np.degrees(np.arccos(np.clip(normal_a @ normal_b, -1, 1))))
        if skew < 0.3 and inl_b.sum() >= 0.4 * min(inl_a.sum(), second.sum()):
            plane_b = (centroid_b, normal_b, int(inl_b.sum()))
    if plane_b is not None and (plane_b[0] - centroid_a) @ normal_a >= 0:
        top_c, top_n = plane_b[0], plane_b[1]
        floor_c, floor_n = centroid_a, normal_a
    elif plane_b is not None:
        top_c, top_n = centroid_a, normal_a
        floor_c, floor_n = plane_b[0], plane_b[1]
    else:  # only one plane measured, and it is the floor: the top is derived from it
        if floor_depth_prior_mm is None:
            raise ValueError(f"plate {side}: top plane not measurable and no floor-depth prior given")
        floor_c, floor_n = centroid_a, normal_a
        top_n = floor_n
        top_c = floor_c + floor_depth_prior_mm * floor_n
        warnings.append(
            f"top plane not measurable (depth missing on the plate top); derived from the floor plane and a "
            f"pocket depth of {floor_depth_prior_mm:.2f} mm taken from the other plate"
        )
    floor_depth = float((top_c - floor_c) @ top_n)

    safe = np.where(finite[..., None], cloud, 0.0)  # NaN-free copy: matmul on NaN warns on some platforms
    top_res = np.where(finite, (safe - top_c) @ top_n, np.nan)
    floor_res = np.where(finite, (safe - floor_c) @ floor_n, np.nan)
    top_mask = np.nan_to_num(np.abs(top_res), nan=99.0) < 1.0
    floor_inl = np.nan_to_num(np.abs(floor_res), nan=99.0) < 0.6
    top_rms = float(np.sqrt(np.mean(top_res[top_mask] ** 2))) if plane_b is not None and top_mask.any() else float("nan")
    floor_rms = float(np.sqrt(np.mean(floor_res[floor_inl] ** 2)))

    # 2. Seeds: the pocket floors (centroids are biased by the projector-shadow strip on one wall).
    floor_img = cv2.morphologyEx(floor_inl.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    floor_img = cv2.morphologyEx(floor_img, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(floor_img, connectivity=8)
    e1, e2 = _plate_axes(top_n)
    seeds, areas = [], []
    for k in range(1, n_labels):
        if stats[k, cv2.CC_STAT_AREA] < POCKET_FLOOR_MIN_PX:
            continue
        member = (labels == k) & finite
        if member.sum() < POCKET_FLOOR_MIN_PX // 3:
            continue
        c = cloud[member].mean(axis=0)
        seeds.append(c - ((c - top_c) @ top_n) * top_n)  # onto the top plane
        areas.append(stats[k, cv2.CC_STAT_AREA])
    if len(seeds) < 4:
        raise ValueError(f"plate {side}: found only {len(seeds)} pocket floors")
    seeds = np.array(seeds)

    # 3. Pocket centres from the rim, where the top plane begins (not affected by shadow strips).
    rim_uv = np.full((len(seeds), 2), np.nan)
    rim_size = np.full((len(seeds), 2), np.nan)
    if plane_b is not None:
        for n, seed in enumerate(seeds):
            r1p = _rim_position(top_mask, K, (x0, y0), seed, e1, e2)
            r1m = _rim_position(top_mask, K, (x0, y0), seed, -e1, e2)
            r2p = _rim_position(top_mask, K, (x0, y0), seed, e2, e1)
            r2m = _rim_position(top_mask, K, (x0, y0), seed, -e2, e1)
            s_uv = np.array([(seed - top_c) @ e1, (seed - top_c) @ e2])
            if r1p is not None and r1m is not None:
                rim_uv[n, 0] = s_uv[0] + (r1p - r1m) / 2
                rim_size[n, 0] = r1p + r1m
            if r2p is not None and r2m is not None:
                rim_uv[n, 1] = s_uv[1] + (r2p - r2m) / 2
                rim_size[n, 1] = r2p + r2m
    seed_uv = np.column_stack([(seeds - top_c) @ e1, (seeds - top_c) @ e2])
    return PocketMeasurements(
        side, top_c, top_n, floor_depth, top_rms, floor_rms, plane_b is not None, (e1, e2),
        seed_uv, np.array(areas), rim_uv, rim_size, warnings,
    )


def _cluster_1d(values: np.ndarray, gap: float) -> np.ndarray:
    """Index of the group of each value, groups separated by gaps larger than ``gap``."""
    order = np.argsort(values)
    index = np.zeros(len(values), int)
    k = 0
    for a, b in zip(order[:-1], order[1:]):
        if values[b] - values[a] > gap:
            k += 1
        index[b] = k
    return index


def estimate_seed_bias(measurements: list[PocketMeasurements]) -> Optional[np.ndarray]:
    """Median offset (rim-based centre minus floor-centroid seed) in plate coordinates, or None.

    Used to correct the floor centroids on plates where the rim cannot be measured.
    """
    offsets = []
    for m in measurements:
        full = m.floor_px >= 0.85 * np.median(m.floor_px)
        both = np.isfinite(m.rim_uv).all(axis=1) & full
        if both.any():
            offsets.append(m.rim_uv[both] - m.seed_uv[both])
    return np.median(np.concatenate(offsets), axis=0) if offsets else None


def _fit_rigid_lattice(
    ij: np.ndarray, obs: np.ndarray, weights: np.ndarray, pitch_mm: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lattice with known, orthogonal pitch: fit origin and rotation (grid search plus refinement).

    Args:
        ij: (N, 2) column and row index of each pocket.
        obs: (N, 2) observed (u, v) of each pocket, NaN where missing.
        weights: (N, 2) weights, 0 where missing.
        pitch_mm: (column step, row step) lengths.

    Returns:
        (origin (2,), column step vector (2,), row step vector (2,)).
    """
    pu, pv = pitch_mm
    ok = weights > 0

    def solve(phi: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        c, s_ = np.cos(phi), np.sin(phi)
        a, b = np.array([c, s_]) * pu, np.array([-s_, c]) * pv
        pred = ij[:, :1] * a + ij[:, 1:] * b
        origin = np.array([np.sum(weights[:, k] * (obs[:, k] - pred[:, k])[ok[:, k]].sum() * 0 + 0) for k in range(2)])
        for k in range(2):
            w = weights[ok[:, k], k]
            origin[k] = np.sum(w * (obs[ok[:, k], k] - pred[ok[:, k], k])) / w.sum()
        resid = np.where(ok, obs - (origin + pred), 0.0)
        return float(np.sum(weights * resid**2)), origin, a, b

    phis = np.radians(np.arange(-6.0, 6.0001, 0.05))
    costs = [solve(p)[0] for p in phis]
    best = phis[int(np.argmin(costs))]
    for step in (np.radians(0.02), np.radians(0.005)):
        cand = best + np.linspace(-5, 5, 11) * step
        best = cand[int(np.argmin([solve(p)[0] for p in cand]))]
    _, origin, a, b = solve(best)
    return origin, a, b


def fit_plate(
    m: PocketMeasurements,
    seed_bias_uv: Optional[np.ndarray] = None,
    pitch_prior_mm: Optional[tuple[float, float]] = None,
) -> PlateCalibration:
    """Fit the 3x3 lattice to the pocket measurements and build the 9 slot frames.

    Lattice indices come from all floor regions (also ones whose rim is hidden).
    Each plate-plane coordinate (u along image-right, v along image-down) is
    observed from the rim-based centres (weight 1); complete floor regions
    without a rim-based coordinate contribute their bias-corrected centroid
    (weight 0.25). With at least ``MIN_RIM_POCKETS`` rim-measured pockets the
    lattice is fitted freely (pitch and shear are results). With fewer, and a
    ``pitch_prior_mm``, the pitch is fixed and orthogonal and only origin and
    rotation are fitted.

    Raises:
        ValueError: The pockets do not form a 3x3 lattice or too few are measured.
    """
    side, (e1, e2), top_c, top_n = m.side, m.axes, m.top_point, m.normal
    warnings = list(m.warnings)
    ij = np.column_stack(
        [_cluster_1d(m.seed_uv[:, 0], 0.4 * PITCH_NOMINAL_MM), _cluster_1d(m.seed_uv[:, 1], 0.4 * PITCH_NOMINAL_MM)]
    )
    if ij.max() > 2:
        raise ValueError(f"plate {side}: pockets form more than a 3x3 lattice ({ij.max() + 1} rows or columns)")
    full = m.floor_px >= 0.85 * np.median(m.floor_px)
    rim_ok = np.isfinite(m.rim_uv)
    seed_ok = full[:, None] & ~rim_ok & (seed_bias_uv is not None)
    obs = np.where(rim_ok, m.rim_uv, m.seed_uv + (0.0 if seed_bias_uv is None else seed_bias_uv))
    weights = np.where(rim_ok, 1.0, np.where(seed_ok, 0.25, 0.0))
    n_rim_pockets = int(rim_ok.all(axis=1).sum())
    rigid = n_rim_pockets < MIN_RIM_POCKETS and pitch_prior_mm is not None
    if rigid:
        warnings.append(
            f"only {n_rim_pockets} pockets measured at the rim: pitch fixed to {pitch_prior_mm[0]:.2f} x "
            f"{pitch_prior_mm[1]:.2f} mm (from the other plates), orthogonal grid"
        )
    used = weights > 0
    for _ in range(3):
        if rigid:
            if used.any(axis=1).sum() < 2:
                raise ValueError(f"plate {side}: too few pockets for the rigid lattice")
            o_uv, a_uv, b_uv = _fit_rigid_lattice(ij, obs, np.where(used, weights, 0.0), pitch_prior_mm)
        else:
            sols = []
            for axis in (0, 1):
                take = used[:, axis]
                if take.sum() < 3:
                    raise ValueError(f"plate {side}: fewer than 3 pockets measurable along axis {axis}")
                sol, _ = _fit_axis_lattice(ij[take], obs[take, axis], weights[take, axis], 1e9)
                sols.append(sol)
            o_uv = np.array([sols[0][0], sols[1][0]])
            a_uv = np.array([sols[0][1], sols[1][1]])
            b_uv = np.array([sols[0][2], sols[1][2]])
        pred = o_uv + ij[:, :1] * a_uv + ij[:, 1:] * b_uv
        resid = np.where(weights > 0, obs - pred, 0.0)
        limit = np.where(rim_ok, GRID_OUTLIER_MM, 3.0)
        new_used = (weights > 0) & (np.abs(resid) < limit)
        if np.array_equal(new_used, used):
            break
        used = new_used
    for axis in (0, 1):
        if (used[:, axis] & seed_ok[:, axis]).any():
            warnings.append(
                f"axis {'uv'[axis]}: {int((used[:, axis] & seed_ok[:, axis]).sum())} pocket(s) measured from the floor "
                f"region with a bias correction of {seed_bias_uv[axis]:+.2f} mm instead of the rim"
            )
    a3 = a_uv[0] * e1 + a_uv[1] * e2
    b3 = b_uv[0] * e1 + b_uv[1] * e2
    x_axis = a3 - (a3 @ top_n) * top_n
    x_axis /= np.linalg.norm(x_axis)
    R = np.column_stack([x_axis, np.cross(top_n, x_axis), top_n])
    if abs(np.linalg.det(R) - 1.0) > 1e-9:
        raise ValueError("slot frame is not a proper rotation")
    slots, measured, deviation = {}, {}, {}
    for row in range(3):
        for col in range(3):
            name = f"{side}{row * 3 + col + 1}"
            uv = o_uv + col * a_uv + row * b_uv
            slots[name] = mgt.make_transform(R, top_c + uv[0] * e1 + uv[1] * e2 - m.floor_depth_mm * top_n)
            hit = np.nonzero((ij[:, 0] == col) & (ij[:, 1] == row))[0]
            ok = [k for k in hit if (used[k] & rim_ok[k]).all()]
            measured[name] = bool(ok)
            deviation[name] = float(np.hypot(*resid[ok[0]])) if ok else float("nan")
    unmeasured = [n for n, ok in measured.items() if not ok]
    if unmeasured:
        warnings.append("slot centres not measured at the rim (floor region or lattice): " + ", ".join(unmeasured))
    dev = np.array([v for v in deviation.values() if np.isfinite(v)])
    sizes = m.rim_size[np.isfinite(m.rim_size).all(axis=1)]
    return PlateCalibration(
        side=side,
        top_point=top_c,
        normal=top_n,
        floor_depth_mm=m.floor_depth_mm,
        slots=slots,
        pitch_mm=(float(np.linalg.norm(a3)), float(np.linalg.norm(b3))),
        grid_angle_deg=float(np.degrees(np.arccos(np.clip(a3 @ b3 / np.linalg.norm(a3) / np.linalg.norm(b3), -1, 1)))),
        top_rms_mm=m.top_rms_mm,
        floor_rms_mm=m.floor_rms_mm,
        lattice_rms_mm=float(np.sqrt(np.mean(dev**2))) if len(dev) else float("nan"),
        pocket_size_mm=(float(np.median(sizes[:, 0])), float(np.median(sizes[:, 1]))) if len(sizes) else (float("nan"),) * 2,
        n_pockets_measured=int(sum(measured.values())),
        measured=measured,
        deviation_mm=deviation,
        warnings=warnings,
    )


def calibrate_sample(
    sample: mgt.Sample, rois: dict[str, tuple[int, int, int, int]], pitch_prior_mm: Optional[tuple[float, float]] = None
) -> dict[str, PlateCalibration]:
    """Calibrate both plates from one empty capture.

    Args:
        sample: Empty capture.
        rois: ``{"L": roi, "R": roi}`` pixel boxes (x0, y0, x1, y1).
        pitch_prior_mm: Lattice pitch (columns, rows) to use for a plate with too few
            measured pockets when no other plate of this capture is well measured.

    Returns:
        ``{"L": PlateCalibration, "R": PlateCalibration}``.
    """
    rng = np.random.default_rng(RANSAC_SEED)
    first, errors = {}, {}
    for side, roi in rois.items():
        try:
            first[side] = measure_pockets(sample, side, roi, rng, floor_depth_prior_mm=None)
        except ValueError as exc:
            errors[side] = exc
    depths = [m.floor_depth_mm for m in first.values() if m.top_plane_found]
    prior = float(np.median(depths)) if depths else None
    for side in errors:
        first[side] = measure_pockets(sample, side, rois[side], rng, floor_depth_prior_mm=prior)
    bias = estimate_seed_bias(list(first.values()))
    free = {}
    for side, m in first.items():
        try:
            free[side] = fit_plate(m, bias)
        except ValueError:
            pass
    good = [pc.pitch_mm for pc in free.values() if pc.n_pockets_measured >= MIN_RIM_POCKETS]
    prior = pitch_prior_mm or (tuple(float(v) for v in np.median(good, axis=0)) if good else None)
    return {side: fit_plate(m, bias, prior) for side, m in first.items()}


# ---------------------------------------------------------------------------
# Plate positions: configuration, calibration over all empty captures, JSON
# ---------------------------------------------------------------------------


@dataclass
class PositionSpec:
    """One plate position: its empty captures, plate ROIs and where it starts."""

    name: str
    empty: list[str]
    roi: dict[str, tuple[int, int, int, int]]
    from_sample: Optional[str] = None  # captures made at or after this one belong to the position
    keep: Optional[list[str]] = None  # plates of the ROIs that really stand at this position (the others only help the fit)


def load_positions(path: Path) -> list[PositionSpec]:
    """Load the plate-position configuration (JSON).

    ``{"positions": {"P1": {"empty": [ids], "roi": {"L": [x0, y0, x1, y1], "R": [...]}},
    "P2": {..., "from_sample": "S02_fill_poty-a_03"}}}``. ``"keep": ["L"]`` limits a position to the
    plates that really stand there (a plate of another position in the same empty capture still helps the pitch fit). The first position is the
    default; a position with ``from_sample`` applies to all captures made at or after
    that capture (time from ``capture_log.csv``), until a later position starts.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        PositionSpec(name, list(entry["empty"]), {k: tuple(v) for k, v in entry["roi"].items()}, entry.get("from_sample"), entry.get("keep"))
        for name, entry in raw["positions"].items()
    ]


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.strip().replace("Z", "+00:00"))


def resolve_position(sample_id: str, positions: list[PositionSpec], log: dict[str, dict]) -> str:
    """Plate position of a capture: the last position whose start capture is not later.

    Captures without a time in the log get the first (default) position.
    """
    chosen = positions[0].name
    when = log.get(sample_id, {}).get("time")
    if not when:
        return chosen
    for pos in positions[1:]:
        start = log.get(pos.from_sample or "", {}).get("time")
        if start and _parse_time(when) >= _parse_time(start):
            chosen = pos.name
    return chosen


@dataclass
class PositionCalibration:
    """Averaged slot calibration of one plate position."""

    name: str
    plates: dict[str, PlateCalibration]
    sources: list[str]
    spread: dict[str, dict[str, float]]  # side -> {"max_translation_mm", "max_rotation_deg"} over the empties

    @property
    def slots(self) -> dict[str, np.ndarray]:
        """``T_cam_slot`` of all slots of this position."""
        return {name: T for plate in self.plates.values() for name, T in plate.slots.items()}


def calibrate_position(
    spec: PositionSpec, samples: list[mgt.Sample], pitch_prior_mm: Optional[tuple[float, float]] = None
) -> PositionCalibration:
    """Calibrate one plate position from its empty captures (slot frames are averaged)."""
    per_sample = [calibrate_sample(s, spec.roi, pitch_prior_mm) for s in samples]
    plates, spread = {}, {}
    for side in spec.roi:
        group = [r[side] for r in per_sample if side in r]
        if not group:
            raise ValueError(f"position {spec.name}: plate {side} could not be calibrated in any empty capture")
        slots = {n: average_transforms([g.slots[n] for g in group]) for n in group[0].slots}
        dt = max(np.linalg.norm(g.slots[n][:3, 3] - slots[n][:3, 3]) for g in group for n in slots)
        dr = max(rotation_angle_deg(g.slots[n][:3, :3].T @ slots[n][:3, :3]) for g in group for n in slots)
        first = group[0]
        z = slots[f"{side}1"][:3, 2]
        plates[side] = PlateCalibration(
            side=side,
            top_point=slots[f"{side}1"][:3, 3] + float(np.mean([g.floor_depth_mm for g in group])) * z,
            normal=z,
            floor_depth_mm=float(np.mean([g.floor_depth_mm for g in group])),
            slots=slots,
            pitch_mm=tuple(float(v) for v in np.mean([g.pitch_mm for g in group], axis=0)),
            grid_angle_deg=float(np.mean([g.grid_angle_deg for g in group])),
            top_rms_mm=float(np.nanmean([g.top_rms_mm for g in group])) if not np.all(np.isnan([g.top_rms_mm for g in group])) else float("nan"),
            floor_rms_mm=float(np.mean([g.floor_rms_mm for g in group])),
            lattice_rms_mm=float(np.nanmean([g.lattice_rms_mm for g in group])) if not np.all(np.isnan([g.lattice_rms_mm for g in group])) else float("nan"),
            pocket_size_mm=first.pocket_size_mm,
            n_pockets_measured=int(np.min([g.n_pockets_measured for g in group])),
            measured={n: all(g.measured[n] for g in group) for n in first.measured},
            deviation_mm=first.deviation_mm,
            warnings=sorted({w for g in group for w in g.warnings}),
        )
        spread[side] = {"max_translation_mm": float(dt), "max_rotation_deg": float(dr)}
    if spec.keep is not None:
        plates, spread = {k: v for k, v in plates.items() if k in spec.keep}, {k: v for k, v in spread.items() if k in spec.keep}
    return PositionCalibration(spec.name, plates, [s.sample_id for s in samples], spread)


def calibrate_all(
    specs: list[PositionSpec], sample_dir: Path
) -> tuple[dict[str, PositionCalibration], tuple[float, float]]:
    """Calibrate every plate position from its empty captures.

    Two passes: the first fits every well-measured plate freely; the global median
    pitch of those plates then constrains plates with too few measurable pockets.

    Returns:
        (calibration per position name, global pitch (columns, rows) in mm).
    """
    loaded = {spec.name: [mgt.load_sample(sample_dir, sid) for sid in spec.empty] for spec in specs}
    first = {spec.name: calibrate_position(spec, loaded[spec.name]) for spec in specs}
    pitches = [
        plate.pitch_mm
        for pos in first.values()
        for plate in pos.plates.values()
        if plate.n_pockets_measured >= MIN_RIM_POCKETS
    ]
    pitch = tuple(float(v) for v in np.median(pitches, axis=0)) if pitches else (PITCH_NOMINAL_MM, PITCH_NOMINAL_MM)
    final = {spec.name: calibrate_position(spec, loaded[spec.name], pitch) for spec in specs}
    return final, pitch


def calibration_to_json(calibration: dict[str, PositionCalibration], specs: list[PositionSpec], pitch: tuple[float, float]) -> dict:
    """Serialisable calibration: slots as 4x4 lists, plate summaries, spreads."""
    out = {}
    for spec in specs:
        pos = calibration[spec.name]
        out[spec.name] = {
            "empty": pos.sources,
            "roi": {k: list(v) for k, v in spec.roi.items()},
            "from_sample": spec.from_sample,
            "plates": {
                side: {**plate.summary(), "slots": {n: T.tolist() for n, T in plate.slots.items()}}
                for side, plate in pos.plates.items()
            },
            "repeatability_between_empties": pos.spread,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "frame_convention": "T_cam_slot: origin at the pocket centre on the pocket floor, z = plate normal towards the "
        "camera, x along the lattice columns (image right), y = z x x (image up)",
        "pitch_mm_global_median": [round(v, 3) for v in pitch],
        "positions": out,
    }


@dataclass
class CalibratedPlate:
    """Plate geometry as loaded from the calibration file."""

    top_point: np.ndarray
    normal: np.ndarray
    floor_depth_mm: float


@dataclass
class Calibration:
    """Loaded slot calibration: ``slots[position][slot] = T_cam_slot`` and plate planes."""

    slots: dict[str, dict[str, np.ndarray]]
    plates: dict[str, dict[str, CalibratedPlate]]
    empty: dict[str, list[str]]
    roi: dict[str, dict[str, tuple[int, int, int, int]]]
    raw: dict
    rim_pockets: dict[str, dict[str, int]] = field(default_factory=dict)  # position -> plate -> pockets measured at the rim


def load_calibration(path: Path) -> Calibration:
    """Load a calibration written by ``calibrate``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    slots, plates, empty, roi, rim = {}, {}, {}, {}, {}
    for name, pos in raw["positions"].items():
        rim[name] = {side: int(p["n_pockets_measured"]) for side, p in pos["plates"].items()}
        slots[name] = {n: np.asarray(T, dtype=float) for plate in pos["plates"].values() for n, T in plate["slots"].items()}
        plates[name] = {
            side: CalibratedPlate(np.asarray(p["top_point_mm"], float), np.asarray(p["normal_to_camera"], float), p["floor_depth_mm"])
            for side, p in pos["plates"].items()
        }
        empty[name] = list(pos["empty"])
        roi[name] = {k: tuple(v) for k, v in pos["roi"].items()}
    return Calibration(slots, plates, empty, roi, raw, rim)


def position_fractions(sample: mgt.Sample, calibration: Calibration) -> dict[str, dict[str, float]]:
    """Per plate and position: share of the valid depth in the plate ROI on that position's planes.

    Returns ``{plate: {position: fraction}}``. The assigned position of a plate should
    score highest; a clear win for another position means the plate was assigned wrongly.
    """
    out: dict[str, dict[str, float]] = {}
    if sample.xyz is None:
        return out
    for name, plates in calibration.plates.items():
        for side, plate in plates.items():
            x0, y0, x1, y1 = calibration.roi[name][side]
            cloud = sample.xyz[y0:y1, x0:x1].astype(np.float64).reshape(-1, 3)
            cloud = cloud[np.isfinite(cloud).all(axis=1)]
            cloud = cloud[(cloud[:, 2] > BAND_MM[0]) & (cloud[:, 2] < BAND_MM[1])]
            d = (cloud - plate.top_point) @ plate.normal
            hit = int(((np.abs(d) < 1.0) | (np.abs(d + plate.floor_depth_mm) < 1.0)).sum())
            out.setdefault(side, {})[name] = hit / len(cloud) if len(cloud) else 0.0
    return out


# ---------------------------------------------------------------------------
# Layout of a capture (capture_log.csv)
# ---------------------------------------------------------------------------

LAYOUT_CODES = {
    "U": "Unterteil", "H": "Halter", "HO": "Halter with Oberteil", "UO": "Unterteil with Oberteil inserted (joined)",
    "E": "elastic part", "?": "unknown",
}
_SLOT = r"[LR][1-9](?:\s*-\s*[LR]?[1-9])?"
_LAYOUT_GROUP = re.compile(r"((?:" + _SLOT + r"\s*,\s*)*" + _SLOT + r")\s*(?:(HO|UO|H|U|E|empty)\b|(\?))")


def _expand_slots(spec: str) -> list[str]:
    side, first = spec.strip()[0], int(spec.strip()[1])
    if "-" not in spec:
        return [f"{side}{first}"]
    end = spec.split("-")[1].strip()
    if end[0] in "LR" and end[0] != side:
        raise ValueError(f"slot range across plates: {spec!r}")
    last = int(end[-1])
    if last < first:
        raise ValueError(f"descending slot range: {spec!r}")
    return [f"{side}{k}" for k in range(first, last + 1)]


def parse_layout(text: str) -> dict[str, str]:
    """Slot occupancy from the free-text layout of a capture (capture log or review log).

    Groups of slots (lists such as ``L2,L4,L6`` and ranges such as ``L1-L8``) followed
    by a code (``U`` Unterteil, ``H`` empty Halter, ``HO`` Halter with Oberteil, ``UO``
    Unterteil with Oberteil inserted, ``E`` elastic part, ``empty``, ``?`` unknown) are read; everything else (scene name, plate
    position, notes such as ``L3,L6,L9 partly behind robot arm``, parentheses) is
    ignored. Slots that are not listed are empty. ``empty`` slots are not part of the
    result; ``?`` slots are (code ``"?"``), so a caller can refuse captures with
    unknown slots.

    Returns:
        ``{slot: code}`` for occupied and unknown slots.

    Raises:
        ValueError: A slot appears with two different codes, or a range is invalid.
    """
    text = re.sub(r"\([^)]*\)", " ", text or "")
    seen: dict[str, str] = {}
    for match in _LAYOUT_GROUP.finditer(text):
        code = match.group(2) or match.group(3)
        for spec in re.split(r"\s*,\s*", match.group(1)):
            for slot in _expand_slots(spec):
                if seen.get(slot, code) != code:
                    raise ValueError(f"slot {slot} appears with codes {seen[slot]} and {code}")
                seen[slot] = code
    return {slot: code for slot, code in seen.items() if code != "empty"}


def read_capture_log(path: Path) -> dict[str, dict]:
    """Rows of ``capture_log.csv`` keyed by the capture name without ``.zdf``."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[re.sub(r"\.zdf$", "", row["file"].strip())] = row
    return rows


_NAME = re.compile(r"^S\d+_([a-z]+)_([A-Za-z0-9]+)(?:-[^_]*)?_\d+$")


def parse_step_key(sample_id: str) -> Optional[tuple[str, str]]:
    """``(block, step)`` of a capture name, e.g. ``S02_fill_poty-a_03`` -> ``("fill", "poty")``."""
    match = _NAME.match(sample_id)
    return (match.group(1), match.group(2)) if match else None


# ---------------------------------------------------------------------------
# Part configuration
# ---------------------------------------------------------------------------


@dataclass
class StepParts:
    """The parts of one assembly step and how they relate."""

    key: str
    specs: dict[str, mgt.PartSpec]  # "U", "H", "O" (those the step has)
    T_halter_oberteil: np.ndarray  # Oberteil seat in the Halter frame (CAD)
    aliases: list[str] = field(default_factory=list)  # step names used in the review log
    joint: Optional[dict] = None  # Oberteil inserted in the Unterteil: z_mm, y_mm, y_half_range_mm (CAD)
    halter_yaw_free: bool = False  # Halter without a yaw-defining feature (round recess): its yaw is not ambiguous, it is undefined


@dataclass
class CellConfig:
    """Parts per step, plus a ``marker_ground_truth.Config`` view for the review GUI."""

    dictionary: str
    marker_size_mm: float
    steps: dict[str, StepParts]
    class_names: list[str]
    config: mgt.Config


UNMARKED_ID_BASE = 1000


def load_cell_config(path: Path) -> CellConfig:
    """Load ``cell_parts.json``.

    ``{"marker_dictionary": "DICT_4X4_50", "marker_size_mm": 15,
    "steps": {"poty": {"U": {"label", "mesh"}, "H": {...}, "O": {"label", "mesh",
    "marker_id", "marker_on_top_face": true, "yaw_deg": 0, "halter_seat_z_mm": 25},
    "aliases": ["Einsetzen_(PO_TY)"],
    "UO": {"z_mm": 25, "y_mm": -3.5, "y_half_range_mm": 9.4, "y_physical_half_range_mm": 8.4}}}}``. ``aliases`` are the step names
    of the review log. ``UO`` describes the joined state (see ``fit_oberteil_in_unterteil``).
    Mesh paths are relative to the config file. The Oberteil seat in the Halter is
    ``halter_seat_z_mm`` along z; the default is the top of the Halter mesh, which puts
    the Oberteil's CAD origin (the bottom of its bar) on the Halter's top face.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    steps: dict[str, StepParts] = {}
    gui_parts: dict[int, mgt.PartSpec] = {}
    labels: list[str] = []
    next_id = UNMARKED_ID_BASE
    for key, entry in raw["steps"].items():
        specs: dict[str, mgt.PartSpec] = {}
        for kind in ("U", "H", "O"):
            if kind not in entry:
                continue
            e = entry[kind]
            mesh = mgt.load_stl(path.parent / e["mesh"], e.get("mesh_scale_to_mm", 1.0))
            if kind == "O":
                if "T_marker_part" in e:
                    T = np.asarray(e["T_marker_part"], dtype=float)
                elif e.get("marker_on_top_face"):
                    T = mgt.marker_offset_from_top_face(mesh, e.get("yaw_deg", 0.0))
                else:
                    raise ValueError(f"step {key}: Oberteil needs T_marker_part or marker_on_top_face")
                marker_id = int(e["marker_id"])
            else:
                T, marker_id = np.eye(4), next_id
                next_id += 1
            spec = mgt.PartSpec(marker_id, e["label"], mesh, T)
            specs[kind] = spec
            gui_parts[marker_id] = spec
            if e["label"] not in labels:
                labels.append(e["label"])
        seat_z = None
        if "O" in specs and "H" in specs:
            seat_z = entry["O"].get("halter_seat_z_mm", float(specs["H"].mesh.vertices[:, 2].max()))
        steps[key] = StepParts(key, specs, translate(0.0, 0.0, seat_z or 0.0), list(entry.get("aliases", [])), entry.get("UO"),
                                  bool(entry.get("H", {}).get("yaw_free", False)))
    config = mgt.Config(raw.get("marker_dictionary", "DICT_4X4_50"), float(raw["marker_size_mm"]), gui_parts, labels)
    return CellConfig(config.dictionary, config.marker_size_mm, steps, labels, config)


# ---------------------------------------------------------------------------
# Fitting parts to the depth
# ---------------------------------------------------------------------------

TRUNCATE_MM = 3.0
BOUND_XY_MM = 3.0
BOUND_Z_MM = 2.0  # SY_RZ seats sit 1.1-1.5 mm above the CAD seat (2026-10-28); PO_TY fits stayed within 1.0
BOUND_YAW_DEG = 3.0
BOUND_XY_OBERTEIL_MM = 1.5
BOUND_TOLERANCE = 0.05  # a parameter within this of its bound counts as at the bound
YAW_HYPOTHESES_DEG = (0.0, 90.0, 180.0, 270.0)
OBERTEIL_ROTATIONS_DEG = (0.0, 90.0, 180.0, 270.0)  # Oberteil in its Halter, about the Halter z axis
ROI_MARGIN_PX = 14
MIN_SUPPORT_PX = 300
RESIDUAL_LIMIT_MM = 2.0
OCCLUSION_TOLERANCE_MM = 4.0


@dataclass
class Fit:
    """Result of one bounded fit: parameters (dx, dy, dz in mm, dyaw in degrees), cost, evaluations."""

    params: np.ndarray
    cost: float
    evaluations: int
    at_bound: list[str]


def pattern_search(
    cost: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    steps: np.ndarray,
    min_steps: np.ndarray,
    max_evaluations: int = 400,
) -> tuple[np.ndarray, float, int]:
    """Bounded coordinate (compass) search, no external dependencies.

    Tries +-step on every coordinate, keeps improvements, halves all steps when none
    helps, and stops when every step is below its minimum.
    """
    x = np.clip(np.asarray(x0, dtype=float), lower, upper)
    best = cost(x)
    evaluations = 1
    steps = np.asarray(steps, dtype=float).copy()
    while np.any(steps > min_steps) and evaluations < max_evaluations:
        improved = False
        for k in range(len(x)):
            for sign in (1.0, -1.0):
                cand = x.copy()
                cand[k] = np.clip(cand[k] + sign * steps[k], lower[k], upper[k])
                if cand[k] == x[k]:
                    continue
                c = cost(cand)
                evaluations += 1
                if c < best - 1e-12:
                    x, best, improved = cand, c, True
                    break
        if not improved:
            steps = steps / 2.0
    return x, best, evaluations


def slot_pose(T_cam_slot: np.ndarray, params: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Part pose for slot-frame offsets ``params`` = (dx, dy, dz, dyaw) around the yaw hypothesis."""
    return T_cam_slot @ translate(params[0], params[1], params[2]) @ pose_from_rz(yaw_deg + params[3])


class DepthModel:
    """Truncated depth cost of posed meshes in front of the empty-plate background.

    The model depth at a pixel is the nearest of the posed meshes (fixed ones plus the
    moving ones) and the background (the empty-plate depth of the same plate position).
    Pixels without a measured depth are ignored; pixels where the model has no depth
    count with the truncation value.
    """

    def __init__(self, measured: np.ndarray, background: np.ndarray, K: np.ndarray, roi: tuple[int, int, int, int]):
        x0, y0, x1, y1 = roi
        self.K, self.roi = K, roi
        self.measured = measured[y0:y1, x0:x1].astype(np.float64)
        self.background = background[y0:y1, x0:x1].astype(np.float64)
        self.valid = np.isfinite(self.measured)
        self.fixed = np.full(self.measured.shape, np.inf)

    def set_fixed(self, items: list[tuple[mgt.Mesh, np.ndarray]]) -> None:
        """Meshes that do not move during the fit (e.g. the Halter while the Oberteil is fitted)."""
        self.fixed = np.full(self.measured.shape, np.inf)
        for mesh, T in items:
            self.fixed = np.minimum(self.fixed, mgt.render_depth_roi(mesh, T, self.K, self.roi))

    def cost(self, items: list[tuple[mgt.Mesh, np.ndarray]]) -> float:
        """Sum of squared truncated depth differences over the measured pixels."""
        z = self.fixed
        for mesh, T in items:
            z = np.minimum(z, mgt.render_depth_roi(mesh, T, self.K, self.roi))
        z = np.where(np.isfinite(self.background), np.minimum(z, np.nan_to_num(self.background, nan=np.inf)), z)
        diff = np.where(np.isfinite(z), np.abs(self.measured - z), TRUNCATE_MM)
        return float(np.sum(np.minimum(diff[self.valid], TRUNCATE_MM) ** 2))


def union_roi(
    meshes_T: list[list[tuple[mgt.Mesh, np.ndarray]]], K: np.ndarray, shape: tuple[int, int], margin_px: int = ROI_MARGIN_PX
) -> Optional[tuple[int, int, int, int]]:
    """Pixel box around several alternative posed models, grown by ``margin_px``."""
    boxes = [b for items in meshes_T for mesh, T in items if (b := mgt.projected_bbox(mesh, T, K, shape, margin_px)) is not None]
    if not boxes:
        return None
    b = np.array(boxes)
    return (int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max()))


def _at_bound(params: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[str]:
    names = ("x", "y", "z", "yaw")
    return [
        names[k]
        for k in range(4)
        if upper[k] > lower[k] and (params[k] - lower[k] < BOUND_TOLERANCE or upper[k] - params[k] < BOUND_TOLERANCE)
    ]


def _same_render(a: np.ndarray, b: np.ndarray, atol_mm: float = 0.05, max_fraction: float = 0.005) -> bool:
    """Whether two depth renders show the same surface (edge pixels may differ by rasterisation)."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    differing = int((fa != fb).sum()) + int((np.abs(a[both] - b[both]) > atol_mm).sum())
    return differing <= max_fraction * max(int(fa.sum()), 1) + 5


def fit_slot_part(
    model: DepthModel,
    T_cam_slot: np.ndarray,
    variants: list[list[tuple[mgt.Mesh, np.ndarray]]],
    seat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    hypotheses_deg: tuple[float, ...] = YAW_HYPOTHESES_DEG,
) -> tuple[Fit, float, dict]:
    """Fit a (compound) part standing in a slot: discrete hypotheses, then a bounded search.

    The hypotheses are every combination of a yaw about the slot ``z`` axis and a
    variant (alternative arrangement of the rigidly linked meshes, e.g. the Oberteil
    turned in its Halter). Hypotheses that render identically (symmetry) form one
    class; the one with the smallest variant index and yaw represents it. The best
    two classes are refined.

    Args:
        model: Depth cost over the part's ROI.
        T_cam_slot: Slot frame.
        variants: Alternatives of ``[(mesh, T_member_in_part_frame), ...]``.
        seat: Seat transform ``T_slot_part`` at yaw 0 (identity: bottom on the pocket floor).
        lower, upper: Bounds of (dx, dy, dz in mm, dyaw in degrees).
        hypotheses_deg: Yaw hypotheses about the slot z axis.

    Returns:
        (best fit, chosen yaw hypothesis in degrees, info: chosen ``variant`` index,
        hypothesis costs, symmetric equivalents, yaw margin).
    """

    def posed(combo: tuple[int, float], p: np.ndarray) -> list[tuple[mgt.Mesh, np.ndarray]]:
        T = slot_pose(T_cam_slot, p, combo[1]) @ seat
        return [(mesh, T @ T_m) for mesh, T_m in variants[combo[0]]]

    zero = np.zeros(4)
    combos = [(v, yaw) for v in range(len(variants)) for yaw in hypotheses_deg]
    renders = []
    for combo in combos:
        z = np.full(model.measured.shape, np.inf)
        for mesh, T in posed(combo, zero):
            z = np.minimum(z, mgt.render_depth_roi(mesh, T, model.K, model.roi))
        renders.append(z)
    classes: list[list[tuple[int, float]]] = []
    keep: list[int] = []
    for k, combo in enumerate(combos):
        for c, rep in enumerate(keep):
            if _same_render(renders[k], renders[rep]):
                classes[c].append(combo)
                break
        else:
            keep.append(k)
            classes.append([combo])
    coarse = []
    grid = np.array([-1.5, 0.0, 1.5])
    for k in keep:
        best = min((model.cost(posed(combos[k], np.array([dx, dy, 0.0, 0.0]))), dx, dy) for dx in grid for dy in grid)
        coarse.append((best[0], k, np.array([best[1], best[2], 0.0, 0.0])))
    coarse.sort(key=lambda t: t[0])
    results = []
    steps0, min_steps = np.array([1.0, 1.0, 0.5, 1.0]), np.array([0.05, 0.05, 0.05, 0.1])
    for _, k, x0 in coarse[:2]:
        combo = combos[k]
        x, c, n = pattern_search(lambda p: model.cost(posed(combo, p)), x0, lower, upper, steps0, min_steps)
        results.append((c, combo, x, n))
    results.sort(key=lambda t: t[0])
    c, combo, x, n = results[0]
    margin = (results[1][0] - c) / max(c, 1e-9) if len(results) > 1 else float("inf")
    info = {
        "yaw_hypothesis_deg": combo[1],
        "variant": combo[0],
        "symmetric_equivalents": [{"variant": v, "yaw_deg": y} for v, y in next(cls for cls in classes if combo in cls)],
        "hypothesis_costs": {f"v{r[1][0]}/{r[1][1]:.0f}": round(r[0], 1) for r in results},
        "yaw_margin": None if not np.isfinite(margin) else round(margin, 3),
        "n_yaw_classes": len(keep),
    }
    return Fit(x, c, n, _at_bound(x, lower, upper)), combo[1], info


# ---------------------------------------------------------------------------
# Ground truth of one capture
# ---------------------------------------------------------------------------

BLOCKING_FLAGS = {
    "refinement_bound_hit",
    "depth_residual_high",
    "marker_outlier",
    "yaw_ambiguous",
}
OCCUPIED_HEIGHT_MM = 12.0
YAW_AMBIGUOUS_MARGIN = 0.05


@dataclass
class TwinPart:
    """One ground-truth part from the cell twin (slot, seat, fit)."""

    slot: str
    kind: str  # "U", "H" or "O"
    spec: mgt.PartSpec
    T_cam_part: np.ndarray
    T_cam_slot: np.ndarray
    refinement: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
    bbox: Optional[tuple[int, int, int, int]] = None
    depth_residual_mm: Optional[float] = None
    support_px: int = 0
    occluded_fraction: float = 0.0
    marker_check: Optional[dict] = None


def background_depth(sample_dir: Path, empty_ids: list[str]) -> np.ndarray:
    """Per-pixel median depth of the empty captures of a plate position (NaN where none is valid)."""
    stack = np.stack([mgt.load_sample(sample_dir, sid).xyz[..., 2] for sid in empty_ids])
    with np.errstate(all="ignore"):
        return np.nanmedian(stack, axis=0).astype(np.float32)


def slot_occupancy(sample: mgt.Sample, slots: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per slot: 90th percentile height above the pocket floor and an occupied/empty verdict.

    Looks at the points within 20 mm of the slot centre in the slot frame; the
    percentile skips the V-groove, which is lower than the rest of a part.
    """
    out = {}
    if sample.xyz is None:
        return out
    height, width = sample.xyz.shape[:2]
    for name, T in slots.items():
        u, v = mgt.project_points(sample.K, T[None, :3, 3])[0]
        x0, x1 = max(int(u) - 120, 0), min(int(u) + 120, width)
        y0, y1 = max(int(v) - 120, 0), min(int(v) + 120, height)
        cloud = sample.xyz[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        local = (cloud - T[:3, 3]) @ T[:3, :3]
        local = local[(np.abs(local[:, 0]) < 20) & (np.abs(local[:, 1]) < 20)]
        if len(local) < 200:
            out[name] = {"p90_height_mm": None, "occupied": None, "n_points": int(len(local))}
        else:
            h = float(np.percentile(local[:, 2], 90))
            out[name] = {"p90_height_mm": round(h, 2), "occupied": h > OCCUPIED_HEIGHT_MM, "n_points": int(len(local))}
    return out


def _delta_pose(p: np.ndarray) -> np.ndarray:
    return translate(p[0], p[1], p[2]) @ pose_from_rz(p[3])


def _corr(params: np.ndarray) -> dict:
    return {"dx_mm": round(float(params[0]), 3), "dy_mm": round(float(params[1]), 3),
            "dz_mm": round(float(params[2]), 3), "dyaw_deg": round(float(params[3]), 3)}


def marker_for_slot(
    marker_poses: list[mgt.MarkerPose], marker_id: int, T_slot: np.ndarray, radius_mm: float = 45.0
) -> Optional[mgt.MarkerPose]:
    """The marker of a given ID closest to a slot centre (in the slot plane), within ``radius_mm``."""
    best, best_d = None, radius_mm
    for pose in marker_poses:
        if pose.marker_id != marker_id:
            continue
        local = mgt.invert_transform(T_slot) @ np.append(pose.T_cam_marker[:3, 3], 1.0)
        d = float(np.hypot(local[0], local[1]))
        if d < best_d:
            best, best_d = pose, d
    return best


def orient_from_marker(part: TwinPart, marker: Optional[mgt.MarkerPose]) -> None:
    """Resolve the 180-degree symmetry of an Oberteil with the detected marker yaw, in place.

    Depth cannot tell the Oberteil from itself turned by 180 degrees about its z axis.
    The pose is turned by 180 degrees when that brings the predicted marker orientation
    (``T_cam_part @ inv(T_marker_part)``) closer to the detected one. The marker only
    chooses between the two equivalent poses; position and the rest of the orientation
    come from the depth fit.
    """
    if marker is None:
        return
    pred = part.T_cam_part @ mgt.invert_transform(part.spec.T_marker_part)
    yaw = wrap_deg(yaw_of((mgt.invert_transform(pred) @ marker.T_cam_marker)[:3, :3]))
    part.refinement["marker_yaw_difference_deg"] = round(yaw, 2)
    if abs(yaw) > 90.0:
        part.T_cam_part = part.T_cam_part @ pose_from_rz(180.0)
        part.refinement["turned_180_by_marker"] = True
        part.notes.append("Oberteil turned 180 deg (symmetric in depth) to match the marker orientation")


def fit_oberteil_in_unterteil(
    model: DepthModel, T_unterteil: np.ndarray, O_mesh: mgt.Mesh, joint: dict
) -> tuple[np.ndarray, Fit, dict]:
    """Fit an Oberteil inserted in the V-groove of an Unterteil (state ``UO``).

    The CAD joining transform puts the Oberteil origin (bottom of its bar) on the top
    face of the Unterteil, ``joint["z_mm"]`` above the Unterteil origin, with the wedge
    flanks on the groove flanks. Along the groove (the Unterteil y axis) the Oberteil can
    sit anywhere in the free play of the groove, ``joint["y_mm"]`` +- ``y_half_range_mm`` (the
    physical play plus 1 mm tolerance),
    which is how the test body for a Y-error works. The play is scanned in steps, then
    refined with x within +-1.5 mm, z within +-1 mm and yaw within +-3 degrees.

    Args:
        model: Depth cost with the Unterteil as fixed model.
        T_unterteil: Fitted Unterteil pose.
        O_mesh: Oberteil mesh.
        joint: ``{"z_mm", "y_mm", "y_half_range_mm"}``.

    Returns:
        (Oberteil pose, fit of (dx, dy, dz, dyaw) around the nominal joint, info).
    """
    base = T_unterteil @ translate(0.0, joint["y_mm"], joint["z_mm"])
    half = float(joint["y_half_range_mm"])
    yaw_half = float(joint.get("yaw_half_range_deg", BOUND_YAW_DEG))
    lower = np.array([-BOUND_XY_OBERTEIL_MM, -half, -BOUND_Z_MM, -yaw_half])
    upper = np.array([BOUND_XY_OBERTEIL_MM, half, BOUND_Z_MM, yaw_half])

    def cost(p: np.ndarray) -> float:
        return model.cost([(O_mesh, base @ _delta_pose(p))])

    scan = [(cost(np.array([0.0, dy, 0.0, 0.0])), dy) for dy in np.arange(-half, half + 1e-9, 1.5)]
    start = min(scan)[1]
    start_yaw = 0.0
    if yaw_half > BOUND_YAW_DEG:  # the Oberteil turns in the cone seat (SY_RZ): scan the turn first
        start_yaw = min((cost(np.array([0.0, start, 0.0, a])), a) for a in np.arange(-yaw_half, yaw_half + 1e-9, 5.0))[1]
    x, c, n = pattern_search(
        cost, np.array([0.0, start, 0.0, start_yaw]), lower, upper, np.array([0.5, 1.0, 0.5, 1.0]), np.array([0.05, 0.05, 0.05, 0.1])
    )
    fit = Fit(x, c, n + len(scan), _at_bound(x, lower, upper))
    return base @ _delta_pose(x), fit, {"y_in_groove_mm": round(float(joint["y_mm"] + x[1]), 3), "scan_start_dy_mm": round(float(start), 2)}


def build_twin_parts(
    sample: mgt.Sample,
    step: StepParts,
    layout: dict[str, str],
    slots: dict[str, np.ndarray],
    background: np.ndarray,
    marker_poses: Optional[list[mgt.MarkerPose]] = None,
) -> tuple[list[TwinPart], dict[str, str]]:
    """Place and fit all parts of a capture.

    Per slot: ``U`` Unterteil alone, ``H`` Halter alone, ``HO`` Halter with Oberteil
    (fitted rigidly linked, the Oberteil may sit turned in the Halter), ``UO`` Unterteil
    with the Oberteil inserted (Unterteil fitted first, then the Oberteil in the groove).
    Every Unterteil/Halter gets the discrete yaw search 0/90/180/270 degrees followed by
    the bounded fit. The marker (when given) resolves the 180-degree symmetry of the
    Oberteil.

    Returns:
        (parts, unsupported ``{slot: reason}``). A slot is unsupported when its code
        has no mesh in the step (``E`` elastic parts, or a part the step does not have).
    """
    shape = sample.rgb.shape[:2]
    measured = sample.xyz[..., 2]
    parts: list[TwinPart] = []
    unsupported: dict[str, str] = {}
    lower = np.array([-BOUND_XY_MM, -BOUND_XY_MM, -BOUND_Z_MM, -BOUND_YAW_DEG])
    upper = -lower
    for slot in sorted(layout, key=SLOT_NAMES.index):
        code = layout[slot]
        kinds = {"U": ["U"], "H": ["H"], "HO": ["H", "O"], "UO": ["U", "O"]}.get(code)
        if kinds is None or any(k not in step.specs for k in kinds) or (code == "UO" and not step.joint):
            unsupported[slot] = f"code {code} has no mesh or joint definition in step {step.key}"
            continue
        T_slot = slots[slot]
        base_spec = step.specs[kinds[0]]
        O_spec = step.specs.get("O")
        if code == "HO":  # the Oberteil may sit turned in its Halter: variants 0/90/180/270 degrees
            variants = [
                [(base_spec.mesh, np.eye(4)), (O_spec.mesh, step.T_halter_oberteil @ pose_from_rz(rel))]
                for rel in OBERTEIL_ROTATIONS_DEG
            ]
        else:
            variants = [[(base_spec.mesh, np.eye(4))]]
        roi_items = [[(mesh, slot_pose(T_slot, np.zeros(4), yaw) @ T_m) for mesh, T_m in members] for members in variants for yaw in YAW_HYPOTHESES_DEG]
        if code == "UO":  # the Oberteil slides along the groove: cover its whole play
            for yaw in YAW_HYPOTHESES_DEG:
                for dy in (-step.joint["y_half_range_mm"], step.joint["y_half_range_mm"]):
                    roi_items.append([(O_spec.mesh, slot_pose(T_slot, np.zeros(4), yaw) @ translate(0.0, step.joint["y_mm"] + dy, step.joint["z_mm"]))])
        roi = union_roi(roi_items, sample.K, shape)
        if roi is None:
            unsupported[slot] = "slot outside the image"
            continue
        model = DepthModel(measured, background, sample.K, roi)
        fit, yaw, info = fit_slot_part(model, T_slot, variants, np.eye(4), lower, upper)
        T_base = slot_pose(T_slot, fit.params, yaw)
        base = TwinPart(slot, kinds[0], base_spec, T_base, T_slot)
        base.refinement = {**info, "stage": "compound" if code == "HO" else "single", "correction": _corr(fit.params),
                           "cost": round(fit.cost, 1), "evaluations": fit.evaluations,
                           "bounds": {"xy_mm": BOUND_XY_MM, "z_mm": BOUND_Z_MM, "yaw_deg": BOUND_YAW_DEG}}
        if code == "HO":
            base.refinement["oberteil_rotation_in_halter_deg"] = OBERTEIL_ROTATIONS_DEG[info["variant"]]
        if fit.at_bound:
            base.flags.append("refinement_bound_hit")
            base.notes.append("at bound: " + ", ".join(fit.at_bound))
        base.refinement["yaw_ambiguous"] = bool(
            info["yaw_margin"] is not None and info["yaw_margin"] < YAW_AMBIGUOUS_MARGIN and info["n_yaw_classes"] > 1
            and not (kinds[0] == "H" and step.halter_yaw_free)
        )
        parts.append(base)
        if code not in ("HO", "UO"):
            continue
        marker = marker_for_slot(marker_poses or [], O_spec.marker_id, T_slot)
        if code == "HO":
            rel_yaw = OBERTEIL_ROTATIONS_DEG[info["variant"]]
            model.set_fixed([(base_spec.mesh, T_base)])
            T_seat = T_base @ step.T_halter_oberteil @ pose_from_rz(rel_yaw)
            o_yaw = float((step.joint or {}).get("yaw_half_range_deg", BOUND_YAW_DEG))  # SY_RZ: the Oberteil turns freely in the cone seat
            o_lower = np.array([-BOUND_XY_OBERTEIL_MM, -BOUND_XY_OBERTEIL_MM, -BOUND_Z_MM, -o_yaw])

            def o_cost(p: np.ndarray) -> float:
                return model.cost([(O_spec.mesh, T_seat @ _delta_pose(p))])

            o_start = np.zeros(4)
            if o_yaw > BOUND_YAW_DEG:
                o_start[3] = min((o_cost(np.array([0.0, 0.0, 0.0, a])), a) for a in np.arange(-o_yaw, o_yaw + 1e-9, 5.0))[1]
            x, c, n = pattern_search(o_cost, o_start, o_lower, -o_lower, np.array([0.5, 0.5, 0.5, 1.0]), np.array([0.05, 0.05, 0.05, 0.1]))
            top = TwinPart(slot, "O", O_spec, T_seat @ _delta_pose(x), T_slot)
            top.refinement = {"stage": "oberteil_in_halter", "correction": _corr(x), "cost": round(c, 1), "evaluations": n,
                              "bounds": {"xy_mm": BOUND_XY_OBERTEIL_MM, "z_mm": BOUND_Z_MM, "yaw_deg": o_yaw},
                              "halter_yaw_hypothesis_deg": yaw, "rotation_in_halter_deg": rel_yaw}
            if rel_yaw % 180.0 != 0.0:
                top.notes.append(f"Oberteil turned {rel_yaw:.0f} deg in its Halter relative to the CAD seat")
            at = _at_bound(x, o_lower, -o_lower)
        else:
            model.set_fixed([(base_spec.mesh, T_base)])
            T_o, ofit, extra = fit_oberteil_in_unterteil(model, T_base, O_spec.mesh, step.joint)
            top = TwinPart(slot, "O", O_spec, T_o, T_slot)
            top.refinement = {"stage": "oberteil_in_unterteil", "correction": _corr(ofit.params), "cost": round(ofit.cost, 1),
                              "evaluations": ofit.evaluations, "unterteil_yaw_hypothesis_deg": yaw, **extra,
                              "bounds": {"x_mm": BOUND_XY_OBERTEIL_MM, "y_mm_half_range": step.joint["y_half_range_mm"],
                                         "z_mm": BOUND_Z_MM, "yaw_deg": step.joint.get("yaw_half_range_deg", BOUND_YAW_DEG)}}
            lo = np.array([-BOUND_XY_OBERTEIL_MM, -step.joint["y_half_range_mm"], -BOUND_Z_MM, -step.joint.get("yaw_half_range_deg", BOUND_YAW_DEG)])
            at = _at_bound(ofit.params, lo, -lo)
            top.notes.append(f"inserted in the Unterteil groove at y = {extra['y_in_groove_mm']:+.1f} mm (CAD joining pose, free play along the groove)")
            stop = step.joint.get("y_physical_half_range_mm")
            if stop is not None and abs(ofit.params[1]) >= stop - 0.6:
                top.notes.append("Oberteil is at the end of the groove (against the wall)")
        if at:
            top.flags.append("refinement_bound_hit")
            top.notes.append("at bound: " + ", ".join(at))
        orient_from_marker(top, marker)
        parts.append(top)
    return parts, unsupported


@dataclass
class PlateShift:
    """Slot-frame shift of a plate position whose slot centres were not measured at the rim."""

    dx_mm: float
    dy_mm: float
    n_parts: int
    n_captures: int
    reason: str


def weak_plates(calibration: Calibration) -> list[tuple[str, str]]:
    """(position, plate) pairs with fewer than ``MIN_RIM_POCKETS`` slot centres measured at the rim."""
    return [
        (pos, side)
        for pos, by_side in calibration.rim_pockets.items()
        for side, n in by_side.items()
        if n < MIN_RIM_POCKETS
    ]


def estimate_plate_shifts(
    items: list[tuple[mgt.Sample, StepParts, dict[str, str], dict[str, str]]],
    calibration: Calibration,
    background_for: Callable[[dict[str, str]], np.ndarray],
    max_captures: int = 4,
) -> dict[tuple[str, str], PlateShift]:
    """Centre error of weak plates, from the parts standing on them.

    On a plate with fewer than ``MIN_RIM_POCKETS`` rim-measured pockets the slot centres
    come from floor regions with a bias correction and can be off by 1-3 mm. Parts seat
    in the pockets within a fraction of a millimetre, so the median (dx, dy) of the fit
    corrections of the Unterteile and Halter on such a plate, in the slot frame, is the
    plate's centre error. It is estimated once per (position, plate) over up to
    ``max_captures`` captures with at least 3 such parts on the plate, and applied to all
    captures at that position. It is recorded in every report.

    Args:
        items: Per capture (sample, step, layout, plate positions).
        calibration: Slot calibration.
        background_for: Empty-plate depth for a capture's plate positions.
        max_captures: Captures used per (position, plate).

    Returns:
        ``{(position, plate): PlateShift}`` for the weak pairs that could be estimated.
    """
    shifts: dict[tuple[str, str], PlateShift] = {}
    for pos, side in weak_plates(calibration):
        per_capture, n_parts = [], 0
        for sample, step, layout, positions in items:
            sub = {k: v for k, v in layout.items() if k[0] == side}
            if positions.get(side) != pos or sum(v in ("U", "H", "HO", "UO") for v in sub.values()) < 3:
                continue
            slots = {n: T for n, T in calibration.slots[pos].items() if n[0] == side}
            parts, _ = build_twin_parts(sample, step, sub, slots, background_for(positions))
            corr = np.array(
                [[p.refinement["correction"]["dx_mm"], p.refinement["correction"]["dy_mm"]] for p in parts if p.kind in ("U", "H")]
            )
            if len(corr) >= 3:
                per_capture.append(np.median(corr, axis=0))
                n_parts += len(corr)
            if len(per_capture) >= max_captures:
                break
        if per_capture:
            m = np.median(per_capture, axis=0)
            shifts[(pos, side)] = PlateShift(
                round(float(m[0]), 3), round(float(m[1]), 3), n_parts, len(per_capture),
                f"only {calibration.rim_pockets[pos][side]} slot centres measured at the rim in the calibration",
            )
    return shifts


def slots_for(
    calibration: Calibration, positions: dict[str, str], shifts: Optional[dict[tuple[str, str], PlateShift]] = None
) -> tuple[dict[str, np.ndarray], dict]:
    """Slot frames of a capture whose plates may stand at different positions, with plate shifts applied.

    Returns:
        (``{slot: T_cam_slot}``, ``{plate: shift record}`` of the shifted plates).
    """
    slots, record = {}, {}
    for side, pos in positions.items():
        shift = (shifts or {}).get((pos, side))
        for name, T in calibration.slots[pos].items():
            if name[0] == side:
                slots[name] = T if shift is None else T @ translate(shift.dx_mm, shift.dy_mm, 0.0)
        if shift is not None:
            record[side] = {"position": pos, "dx_mm": shift.dx_mm, "dy_mm": shift.dy_mm, "n_parts": shift.n_parts,
                            "n_captures": shift.n_captures, "reason": shift.reason}
    return slots, record


def compute_twin_masks(
    parts: list[TwinPart], sample: mgt.Sample, tolerance_mm: float = OCCLUSION_TOLERANCE_MM
) -> None:
    """Visible masks, boxes, depth residuals and flags of all parts, in place.

    All parts are rendered into one z-buffer with the capture's intrinsics. A pixel
    belongs to a part when the part is the nearest rendered surface there and the
    measured depth is not clearly closer than the rendered one (something in front,
    e.g. the robot arm). Missing depth does not remove a pixel.
    """
    if not parts:
        return
    shape = sample.rgb.shape[:2]
    depth, owner = mgt.render_scene([(p.spec.mesh, p.T_cam_part) for p in parts], sample.K, shape)
    measured = sample.xyz[..., 2] if sample.xyz is not None else np.full(shape, np.nan)
    valid = np.isfinite(measured)
    in_front = valid & (measured < depth - tolerance_mm)
    for i, part in enumerate(parts):
        rendered = owner == i
        part.mask = rendered & ~in_front
        n_rendered = int(rendered.sum())
        part.occluded_fraction = float(in_front[rendered].sum() / n_rendered) if n_rendered else 0.0
        both = part.mask & valid
        part.support_px = int(both.sum())
        if both.any():
            part.depth_residual_mm = float(np.median(np.abs(measured[both] - depth[both])))
        ys, xs = np.nonzero(part.mask)
        if len(xs):
            part.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            part.notes.append("not visible: hidden behind other objects")
        if part.occluded_fraction > 0.2:
            part.notes.append(f"partly occluded ({part.occluded_fraction:.0%} of the model hidden behind nearer surfaces)")
        if part.depth_residual_mm is not None and part.depth_residual_mm > RESIDUAL_LIMIT_MM:
            part.flags.append("depth_residual_high")
        if part.bbox is not None and part.support_px < MIN_SUPPORT_PX:
            part.flags.append("low_depth_support")
        elif part.refinement.get("yaw_ambiguous"):
            part.flags.append("yaw_ambiguous")


# ---------------------------------------------------------------------------
# Marker check
# ---------------------------------------------------------------------------


def check_markers(
    parts: list[TwinPart], markers: list[tuple[int, np.ndarray]], sample: mgt.Sample, marker_size_mm: float,
    max_distance_px: float = 40.0,
) -> list[dict]:
    """Compare detected markers with the marker pose predicted by the twin chain.

    For every Oberteil the marker pose is predicted as ``T_cam_part @ inv(T_marker_part)``
    and matched to the detected marker with the nearest centre. The difference
    ``inv(T_predicted) @ T_measured`` is stored in ``part.marker_check``: in-plane
    offset (dx, dy), offset along the marker normal (dz), tilt and yaw. It measures the
    ground-truth error (plus the marker placement on the part); it does not change the pose.

    Returns:
        Detected markers that no part claimed (``{"marker_id", "centre_px"}``).
    """
    free = list(range(len(markers)))
    for part in parts:
        if part.kind != "O":
            continue
        if part.support_px < MIN_SUPPORT_PX:
            part.marker_check = {"status": "skipped: too little depth support"}
            continue
        T_pred = part.T_cam_part @ mgt.invert_transform(part.spec.T_marker_part)
        centre = mgt.project_points(sample.K, T_pred[None, :3, 3])[0]
        best, best_d = None, max_distance_px
        for k in free:
            d = float(np.linalg.norm(markers[k][1].mean(axis=0) - centre))
            if d < best_d:
                best, best_d = k, d
        if best is None:
            part.marker_check = {"status": "no marker found near the predicted position"}
            part.notes.append("marker not found near the predicted position")
            continue
        free.remove(best)
        marker_id, corners = markers[best]
        pose = mgt.estimate_marker_pose(marker_id, corners, sample, marker_size_mm)
        delta = mgt.invert_transform(T_pred) @ pose.T_cam_marker
        R = delta[:3, :3]
        part.marker_check = {
            "status": "ok" if marker_id == part.spec.marker_id else f"wrong marker id {marker_id}",
            "marker_id": marker_id,
            "pose_method": pose.method,
            "dx_mm": round(float(delta[0, 3]), 3),
            "dy_mm": round(float(delta[1, 3]), 3),
            "dz_mm": round(float(delta[2, 3]), 3),
            "distance_mm": round(float(np.linalg.norm(delta[:3, 3])), 3),
            "tilt_deg": round(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), 3),
            "yaw_deg": round(wrap_deg(yaw_of(R)), 3),
        }
    return [{"marker_id": markers[k][0], "centre_px": [round(float(v), 1) for v in markers[k][1].mean(axis=0)]} for k in free]


def marker_offsets(reports: list[dict]) -> dict[str, dict]:
    """Per Oberteil label: median marker offset over all checked parts, and the spread around it.

    The median is the marker placement on the part relative to the CAD-nominal offset
    (a calibration constant, to be written into the configuration). The spread around it
    is the ground-truth error, including the marker's placement tolerance. Yaw is
    reported modulo 360 (circular median).
    """
    by_label: dict[str, list[dict]] = {}
    for report in reports:
        for part in report.get("parts", []):
            check = part.get("marker_check")
            if check and "dx_mm" in check and check.get("pose_method") == "depth_plane":
                by_label.setdefault(part["label"], []).append(check)
    out = {}
    for label, checks in by_label.items():
        arr = {k: np.array([c[k] for c in checks]) for k in ("dx_mm", "dy_mm", "dz_mm", "tilt_deg")}
        yaw = np.radians([c["yaw_deg"] for c in checks])
        yaw_med = float(np.degrees(np.arctan2(np.median(np.sin(yaw)), np.median(np.cos(yaw)))))
        yaw_dev = np.array([wrap_deg(c["yaw_deg"] - yaw_med) for c in checks])
        out[label] = {
            "n": len(checks),
            "median_offset": {"dx_mm": round(float(np.median(arr["dx_mm"])), 3), "dy_mm": round(float(np.median(arr["dy_mm"])), 3),
                              "dz_mm": round(float(np.median(arr["dz_mm"])), 3), "yaw_deg": round(yaw_med, 3)},
            "spread_around_median": {
                "dx_mm_std": round(float(np.std(arr["dx_mm"])), 3), "dy_mm_std": round(float(np.std(arr["dy_mm"])), 3),
                "dz_mm_std": round(float(np.std(arr["dz_mm"])), 3), "yaw_deg_std": round(float(np.std(yaw_dev)), 3),
                "tilt_deg_median": round(float(np.median(arr["tilt_deg"])), 3),
                "tilt_deg_max": round(float(np.max(arr["tilt_deg"])), 3),
            },
        }
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_outputs(
    out_dir: Path, sample: mgt.Sample, parts: list[TwinPart], class_names: list[str],
    markers: list[tuple[int, np.ndarray]], report: dict, inpaint_backend: str, inpaint_margin_px: int,
) -> None:
    for sub in ("images", "labels", "masks", "annotations"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    sid = sample.sample_id
    inpainted, marker_mask = mgt.inpaint_markers(sample.rgb, [c for _, c in markers], inpaint_backend, inpaint_margin_px)
    cv2.imwrite(str(out_dir / "images" / f"{sid}.png"), inpainted)
    cv2.imwrite(str(out_dir / "masks" / f"{sid}_markers.png"), marker_mask)
    for old in (out_dir / "masks").glob(f"{sid}_det_*.png"):
        old.unlink()
    for i, part in enumerate(parts):
        if part.bbox is not None:
            cv2.imwrite(str(out_dir / "masks" / f"{sid}_det_{i}.png"), part.mask.astype(np.uint8) * 255)
    (out_dir / "labels" / f"{sid}.txt").write_text(
        "".join(line + "\n" for line in mgt.yolo_lines(parts, class_names, sample.rgb.shape[:2]))
    )
    (out_dir / "klassen.txt").write_text("".join(name + "\n" for name in class_names))
    stamp = report["created"]
    detections = [
        {
            "object_id": p.spec.label,
            "class_confidence": 1.0,
            "bbox_2d": list(p.bbox),
            "mask_ref": f"masks/{sid}_det_{i}.png",
            "slot": p.slot,
            "marker_id": p.spec.marker_id if p.kind == "O" else None,
        }
        for i, p in enumerate(parts)
        if p.bbox is not None
    ]
    poses = []
    for p in parts:
        R, t = p.T_cam_part[:3, :3], p.T_cam_part[:3, 3]
        poses.append(
            {
                "object_id": p.spec.label,
                "translation_mm": [float(v) for v in t],
                "rotation_quat": mgt.rotation_to_quaternion_wxyz(R),
                "rotation_euler_xyz_deg": mgt.rotation_to_euler_xyz_deg(R),
                "uncertainty": None,
                "method": "cell twin: slot calibration (empty captures) x seat, bounded depth fit; "
                "uncertainty null: ground-truth accuracy not yet measured",
                "n_points_used": p.support_px,
                "fitness": None,
            }
        )
    ann = out_dir / "annotations"
    ann.joinpath(f"{sid}_detections.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "timestamp": stamp, "capture_id": sid, "detections": detections}, indent=2)
    )
    ann.joinpath(f"{sid}_poses.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "timestamp": stamp, "capture_id": sid, "frame": "CAM", "poses": poses}, indent=2)
    )
    ann.joinpath(f"{sid}_report.json").write_text(json.dumps(report, indent=2))


def process_capture(
    sample: mgt.Sample,
    cfg: CellConfig,
    step: StepParts,
    layout: dict[str, str],
    positions: dict[str, str],
    calibration: Calibration,
    background: np.ndarray,
    out_dir: Path,
    log_layout: str = "",
    plate_shifts: Optional[dict[tuple[str, str], PlateShift]] = None,
    inpaint_backend: str = "surface",
    inpaint_margin_px: int = 1,
) -> dict:
    """Create all ground-truth files of one capture and return its report."""
    slots, adjustment = slots_for(calibration, positions, plate_shifts)
    markers = mgt.detect_markers(sample.rgb, cfg.dictionary)
    o_id = step.specs["O"].marker_id if "O" in step.specs else None
    marker_poses = [mgt.estimate_marker_pose(i, c, sample, cfg.marker_size_mm) for i, c in markers if i == o_id]
    weak = [(pos, side) for pos, side in weak_plates(calibration) if positions.get(side) == pos and (pos, side) not in (plate_shifts or {})]
    parts, unsupported = build_twin_parts(sample, step, layout, slots, background, marker_poses)
    compute_twin_masks(parts, sample)
    unclaimed = check_markers(parts, markers, sample, cfg.marker_size_mm)
    occupancy = slot_occupancy(sample, slots)
    mismatch = {}
    for slot, info in occupancy.items():
        expected = slot in layout
        if info["occupied"] is not None and info["occupied"] != expected:
            mismatch[slot] = {"layout": layout.get(slot, "empty"), "measured_p90_height_mm": info["p90_height_mm"]}
    fractions = position_fractions(sample, calibration)
    position_ok = all(max(by_pos, key=by_pos.get) == positions[side] for side, by_pos in fractions.items() if side in positions)
    report = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "method": "cell_twin",
        "step": step.key,
        "plate_position": positions,
        "plate_adjustment": adjustment,
        "plates_with_unmeasured_slot_centres": [f"{side} at {pos}" for pos, side in weak],
        "position_check": {
            "fraction_on_plate_planes": {side: {k: round(v, 3) for k, v in by_pos.items()} for side, by_pos in fractions.items()},
            "assigned_fits_best": position_ok,
        },
        "layout_text": log_layout,
        "layout": layout,
        "unsupported_slots": unsupported,
        "layout_mismatch": mismatch,
        "unclaimed_markers": unclaimed,
        "unknown_marker_ids": sorted({mid for mid, _ in markers} - {step.specs["O"].marker_id} if "O" in step.specs else set()),
        "inpaint_backend": inpaint_backend,
        "parts": [],
    }
    for i, p in enumerate(parts):
        report["parts"].append(
            {
                "marker_id": p.spec.marker_id if p.kind == "O" else None,
                "label": p.spec.label,
                "slot": p.slot,
                "part_type": p.kind,
                "pose_method": "cell_twin",
                "bbox_2d": list(p.bbox) if p.bbox else None,
                "visible_pixels": int(p.mask.sum()),
                "support_px": p.support_px,
                "occluded_fraction": round(p.occluded_fraction, 3),
                "depth_residual_mm": p.depth_residual_mm,
                "refinement": p.refinement,
                "marker_check": p.marker_check,
                "flags": p.flags,
                "notes": p.notes,
            }
        )
    _apply_rejection(report)
    _write_outputs(Path(out_dir), sample, parts, cfg.class_names, markers, report, inpaint_backend, inpaint_margin_px)
    return report


def _apply_rejection(report: dict) -> None:
    """Set ``rejected`` and ``rejection_reasons`` from part flags and capture-level problems."""
    reasons = []
    for part in report["parts"]:
        blocking = sorted(set(part["flags"]) & BLOCKING_FLAGS)
        if blocking:
            reasons.append(f"{part['slot']} {part['label']}: {', '.join(blocking)}")
    if report["layout_mismatch"]:
        reasons.append("layout differs from the measured slot occupancy: " + ", ".join(sorted(report["layout_mismatch"])))
    if report["unsupported_slots"]:
        reasons.append("unsupported slots: " + ", ".join(sorted(report["unsupported_slots"])))
    if not report["position_check"]["assigned_fits_best"]:
        reasons.append("plate position does not match the depth data")
    report["rejection_reasons"] = reasons
    report["rejected"] = bool(reasons)


# ---------------------------------------------------------------------------
# Run over several captures
# ---------------------------------------------------------------------------

MARKER_OUTLIER_MM = 2.0
MARKER_OUTLIER_YAW_DEG = 5.0
SKIP_BLOCKS = {"teach", "test", "empty", "tape"}


@dataclass
class CaptureRecord:
    """What the logs say about one capture."""

    sample_id: str
    step_key: str
    layout_text: str
    positions: dict[str, str]  # plate -> plate position
    source: str  # "capture_log" or "review_log"


def read_review_log(path: Path) -> dict[str, dict]:
    """Rows of ``review_log.csv`` (file, verdict, step, pos_L, pos_R, layout, notes, reviewed_at).

    Several rows may exist per file; the latest (by ``reviewed_at``, then file order) wins.
    """
    rows: dict[str, tuple[tuple[str, int], dict]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for n, row in enumerate(csv.DictReader(handle)):
            key = re.sub(r"\.zdf$", "", row["file"].strip())
            order = ((row.get("reviewed_at") or "").strip(), n)
            if key not in rows or order >= rows[key][0]:
                rows[key] = (order, row)
    return {k: v[1] for k, v in rows.items()}


def resolve_step_key(step_name: str, cfg: CellConfig) -> Optional[str]:
    """Configured step for a step name from the review log (alias, key, or the ``(XX_YY)`` token)."""
    name = step_name.strip()
    for key, step in cfg.steps.items():
        if name == key or name in step.aliases:
            return key
    token = re.search(r"\(([^)]*)\)", name)
    flat = re.sub(r"[^a-z0-9]", "", (token.group(1) if token else name).lower())
    matches = [key for key in cfg.steps if re.sub(r"[^a-z0-9]", "", key.lower()) == flat]
    return matches[0] if len(matches) == 1 else None


def select_captures(
    sample_ids: list[str], log: dict[str, dict], cfg: CellConfig, explicit: bool,
    positions: Optional[list[PositionSpec]] = None, review: Optional[dict[str, dict]] = None,
    default_step: Optional[str] = None,
) -> tuple[list[CaptureRecord], dict[str, str]]:
    """Captures to process and the reasons for skipping the others.

    A capture with a row in the review log (``review``) uses that row: verdict must be
    ``accept``; step, plate positions and layout come from it. Other captures use the
    capture log: only ``fill`` captures of configured steps (any block when ``explicit``),
    plate position from the capture time. Skipped with a message: layouts that start with
    ``REJECTED`` or ``PENDING_REVIEW``, layouts with ``?`` slots, ``teach`` captures,
    steps without mesh mapping, unreadable layouts. A review step ``unknown`` or empty is
    replaced by ``default_step`` when given.

    Returns:
        (records, ``{sample id: reason}``).
    """
    chosen, skipped = [], {}
    positions = positions or []
    for sid in sample_ids:
        row = (review or {}).get(sid)
        if row is not None:
            verdict = (row.get("verdict") or "").strip().lower()
            step_name, layout_text = row.get("step", ""), row.get("layout", "")
            if verdict != "accept":
                skipped[sid] = f"review verdict is '{verdict or 'empty'}'"
                continue
            step_key = resolve_step_key(step_name, cfg)
            if step_key is None and default_step and step_name.strip().lower() in ("", "unknown"):
                step_key = default_step  # the reviewer could not name the step; the caller knows it
            pos = {"L": (row.get("pos_L") or "").strip(), "R": (row.get("pos_R") or "").strip()}
            source = "review_log"
        else:
            parsed = parse_step_key(sid)
            row = log.get(sid)
            layout_text = (row or {}).get("layout", "")
            step_name = parsed[1] if parsed else ""
            if parsed is None:
                skipped[sid] = "no review-log row, and the name does not follow S<nn>_<block>_<scene>_<NN>"
                continue
            if parsed[0] == "teach" or (parsed[0] in SKIP_BLOCKS and not explicit):
                skipped[sid] = f"block '{parsed[0]}' is not a dataset capture"
                continue
            if parsed[0] != "fill" and not explicit:
                skipped[sid] = f"block '{parsed[0]}' is not a fill capture"
                continue
            if row is None:
                skipped[sid] = "not in the capture log (no layout)"
                continue
            if layout_text.strip().upper().startswith(("REJECTED", "PENDING_REVIEW")):
                skipped[sid] = f"layout says {layout_text.strip().split(':')[0].split()[0]}"
                continue
            step_key = parsed[1] if parsed[1] in cfg.steps else resolve_step_key(parsed[1], cfg)
            name = resolve_position(sid, positions, log) if positions else ""
            pos = {"L": name, "R": name}
            source = "capture_log"
        if step_key is None:
            skipped[sid] = f"no mesh mapping for body type '{step_name}' in the configuration"
            continue
        try:
            layout = parse_layout(layout_text)
        except ValueError as exc:
            skipped[sid] = f"layout not readable: {exc}"
            continue
        unknown = sorted(k for k, v in layout.items() if v == "?")
        if unknown:
            skipped[sid] = "layout has unknown slots (?): " + ", ".join(unknown)
            continue
        chosen.append(CaptureRecord(sid, step_key, layout_text, pos, source))
    return chosen, skipped


def finalize_run(out_dir: Path, sample_ids: list[str]) -> dict:
    """Flag marker outliers, rewrite the reports and write ``summary.json``.

    A marker is an outlier when its offset differs from the median offset of the same
    label (over the whole run) by more than ``MARKER_OUTLIER_MM`` (in 3D) or its yaw by
    more than ``MARKER_OUTLIER_YAW_DEG`` modulo 180 degrees (the Oberteil is
    180-degree symmetric). Needs at least 3 checked markers per label.
    """
    out_dir = Path(out_dir)
    paths = {sid: out_dir / "annotations" / f"{sid}_report.json" for sid in sample_ids}
    reports = {sid: json.loads(p.read_text()) for sid, p in paths.items()}
    offsets = marker_offsets(list(reports.values()))
    for sid, report in reports.items():
        for part in report["parts"]:
            check, off = part.get("marker_check"), offsets.get(part["label"])
            if not check or "dx_mm" not in check or not off or off["n"] < 3 or check.get("pose_method") != "depth_plane":
                continue
            m = off["median_offset"]
            lateral = float(np.linalg.norm([check["dx_mm"] - m["dx_mm"], check["dy_mm"] - m["dy_mm"], check["dz_mm"] - m["dz_mm"]]))
            yaw = abs(wrap_deg(check["yaw_deg"] - m["yaw_deg"], 180.0))
            check["deviation_from_label_median_mm"] = round(lateral, 3)
            check["yaw_deviation_mod180_deg"] = round(yaw, 3)
            if (lateral > MARKER_OUTLIER_MM or yaw > MARKER_OUTLIER_YAW_DEG) and "marker_outlier" not in part["flags"]:
                part["flags"].append("marker_outlier")
        _apply_rejection(report)
        paths[sid].write_text(json.dumps(report, indent=2))
    per_capture = {}
    for sid, report in reports.items():
        stats: dict[str, dict] = {}
        for kind in ("U", "H", "O"):
            sel = [p for p in report["parts"] if p["part_type"] == kind]
            if not sel:
                continue
            corr = np.array([[abs(p["refinement"]["correction"][k]) for k in ("dx_mm", "dy_mm", "dz_mm", "dyaw_deg")] for p in sel])
            res = [p["depth_residual_mm"] for p in sel if p["depth_residual_mm"] is not None]
            marks = [p["marker_check"]["distance_mm"] for p in sel if p.get("marker_check") and "distance_mm" in p["marker_check"]]
            stats[kind] = {
                "n": len(sel),
                "median_abs_correction": dict(zip(("dx_mm", "dy_mm", "dz_mm", "dyaw_deg"), [round(float(v), 3) for v in np.median(corr, axis=0)])),
                "max_abs_correction": dict(zip(("dx_mm", "dy_mm", "dz_mm", "dyaw_deg"), [round(float(v), 3) for v in corr.max(axis=0)])),
                "depth_residual_median_mm": round(float(np.median(res)), 3) if res else None,
                "marker_distance_median_mm": round(float(np.median(marks)), 3) if marks else None,
                "flagged": [f"{p['slot']}:{','.join(p['flags'])}" for p in sel if p["flags"]],
            }
        per_capture[sid] = {
            "plate_position": report["plate_position"],
            "rejected": report["rejected"],
            "rejection_reasons": report["rejection_reasons"],
            "parts_by_type": stats,
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "captures": per_capture,
        "marker_offsets": offsets,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def composite_background(
    positions: dict[str, str], sample_dir: Path, calibration: Calibration, cache: dict[str, np.ndarray]
) -> np.ndarray:
    """Empty-plate depth for a capture whose plates may stand at different positions.

    The left half of the image takes the empty depth of the left plate's position, the
    right half that of the right plate's position.
    """
    for name in set(positions.values()):
        if name not in cache:
            cache[name] = background_depth(sample_dir, calibration.empty[name])
    left, right = cache[positions["L"]], cache[positions["R"]]
    if positions["L"] == positions["R"]:
        return left
    out = left.copy()
    half = out.shape[1] // 2
    out[:, half:] = right[:, half:]
    return out


def run_captures(
    sample_dir: Path, cfg: CellConfig, positions: list[PositionSpec], calibration: Calibration, log: dict[str, dict],
    out_dir: Path, sample_ids: Optional[list[str]] = None, forced_position: Optional[str] = None, log_fn=print,
    review: Optional[dict[str, dict]] = None, default_step: Optional[str] = None,
) -> dict:
    """Process the selected captures of ``sample_dir`` and finalize the run."""
    explicit = sample_ids is not None
    ids = sample_ids if explicit else mgt.list_sample_ids(sample_dir)
    records, skipped = select_captures(ids, log, cfg, explicit, positions, review, default_step)
    for sid, reason in skipped.items():
        log_fn(f"skip {sid}: {reason}")
    cache: dict[str, np.ndarray] = {}
    usable = []
    for rec in records:
        if forced_position:
            rec.positions = {"L": forced_position, "R": forced_position}
        missing = sorted({v for v in rec.positions.values() if v not in calibration.slots})
        if missing:
            log_fn(f"skip {rec.sample_id}: no slot calibration for plate position {', '.join(missing)}")
            skipped[rec.sample_id] = f"no slot calibration for plate position {', '.join(missing)}"
        else:
            usable.append(rec)

    def background_for(positions: dict[str, str]) -> np.ndarray:
        return composite_background(positions, sample_dir, calibration, cache)

    items = [
        (mgt.load_sample(sample_dir, r.sample_id), cfg.steps[r.step_key], parse_layout(r.layout_text), r.positions)
        for r in usable
        if any(
            (pos, side) in weak_plates(calibration) and sum(v in ("U", "H", "HO", "UO") for k, v in parse_layout(r.layout_text).items() if k[0] == side) >= 3
            for side, pos in r.positions.items()
        )
    ]
    shifts = estimate_plate_shifts(items, calibration, background_for) if items else {}
    del items
    for (pos, side), sh in shifts.items():
        log_fn(f"plate {side} at {pos}: slot centres shifted by ({sh.dx_mm:+.2f}, {sh.dy_mm:+.2f}) mm from {sh.n_parts} parts in {sh.n_captures} captures")
    done = []
    for rec in usable:
        report = process_capture(
            mgt.load_sample(sample_dir, rec.sample_id), cfg, cfg.steps[rec.step_key], parse_layout(rec.layout_text),
            rec.positions, calibration, background_for(rec.positions), out_dir, rec.layout_text, plate_shifts=shifts,
        )
        done.append(rec.sample_id)
        log_fn(f"{rec.sample_id}: positions {rec.positions}, {len(report['parts'])} parts, {'REJECTED' if report['rejected'] else 'ok'}")
        for reason in report["rejection_reasons"]:
            log_fn(f"    {reason}")
    summary = finalize_run(out_dir, done) if done else {}
    summary["skipped"] = skipped
    summary["plate_shifts"] = {f"{side} at {pos}": vars(sh) for (pos, side), sh in shifts.items()}
    if done:
        (Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate", help="slot calibration from the empty captures")
    cal.add_argument("sample_dir", type=Path)
    cal.add_argument("--positions", type=Path, required=True)
    cal.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run", help="ground truth for fill captures")
    run.add_argument("sample_dir", type=Path)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--calibration", type=Path, required=True)
    run.add_argument("--positions", type=Path, required=True)
    run.add_argument("--capture-log", type=Path, required=True)
    run.add_argument("--review-log", type=Path, help="review_log.csv of the session (default: next to the capture log, if present)")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--sample-id", action="append", dest="sample_ids")
    run.add_argument("--default-step", help="step key to use when the review log says 'unknown' or leaves the step empty")
    run.add_argument("--position", help="force the plate position instead of resolving it from the log")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    positions = load_positions(args.positions)
    if args.command == "calibrate":
        calibration, pitch = calibrate_all(positions, args.sample_dir)
        doc = calibration_to_json(calibration, positions, pitch)
        args.out.write_text(json.dumps(doc, indent=2))
        print(f"pitch {pitch[0]:.3f} x {pitch[1]:.3f} mm")
        for name, pos in doc["positions"].items():
            for side, plate in pos["plates"].items():
                print(f"{name} {side}: pitch {plate['pitch_mm']}, floor {plate['floor_depth_mm']} mm, lattice rms "
                      f"{plate['lattice_rms_mm']} mm, {plate['n_pockets_measured']}/9 measured at the rim")
                for w in plate["warnings"]:
                    print(f"    {w}")
        return 0
    cfg = load_cell_config(args.config)
    review_path = args.review_log or args.capture_log.parent / "review_log.csv"
    summary = run_captures(
        args.sample_dir, cfg, positions, load_calibration(args.calibration), read_capture_log(args.capture_log),
        args.out, args.sample_ids, args.position, review=read_review_log(review_path) if review_path.exists() else None, default_step=args.default_step,
    )
    return 0 if summary.get("captures") else 1


if __name__ == "__main__":
    sys.exit(main())
