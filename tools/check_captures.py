"""Quick quality check for new Zivid captures of the test-body benchmark.

Run it after every capture block. It

1. converts new ``.zdf`` files in a folder to pipeline samples
   (``<stem>_rgb.png``, ``<stem>_xyz.npy``, ``<stem>_cam_K.txt``,
   ``<stem>_meta.json``, ``<stem>_settings.yml``); files already converted are
   skipped,
2. detects the ArUco markers with ``marker_ground_truth`` and prints per
   capture: valid-depth fraction, marker IDs found (with counts), marker side
   length in px, valid depth on the markers, and a short hash of the Zivid
   settings stored in the ``.zdf``,
3. warns when something looks wrong: an expected marker ID is missing, markers
   appear where none should be (empty trays, taped scenes), two captures of a
   repeatability block differ by more than 1 px in marker corners, a taped
   capture differs in depth from its marker capture (something moved), or the
   Zivid settings change within the folder.

File names follow ``S<nn>_<block>_<scene>_<NN>.zdf``; see
``docs/researcheffort/pilot_capture_guide.md``. Files with other names are
still checked, as if markers were expected.

Usage::

    python check_captures.py ZDF_DIR --samples-dir SAMPLES_DIR --expect 6 \\
        [--log capture_log.csv] [--pattern "S01_*.zdf"] [--no-convert]

The exit code is 1 when there is at least one warning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

import marker_ground_truth as mgt

REPEAT_LIMIT_PX = 1.0
MIN_VALID_FRACTION = 0.5
MIN_MARKER_SIDE_PX = 25.0
PAIR_DEPTH_TOL_MM = 2.0
PAIR_DEPTH_FRACTION = 0.01
SAMPLE_SUFFIXES = ("_rgb.png", "_xyz.npy", "_cam_K.txt", "_meta.json")

NAME_RE = re.compile(r"^(S\d{2})_([a-z]+)_(.+)_(\d{2,3})$")

Marker = tuple[int, np.ndarray]  # (marker id, corners (4, 2))


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------


@dataclass
class ParsedName:
    """Parts of a capture file name ``S01_calib_oberteil_s2-2_r090_03``."""

    session: str
    block: str  # test, empty, calib, rep, fill, tape
    scene: str  # everything between block and index
    index: int

    @property
    def group_key(self) -> str:
        """Captures with the same key differ only in their index."""
        return f"{self.session}_{self.block}_{self.scene}"


def parse_capture_name(stem: str) -> Optional[ParsedName]:
    """Parse a stem, or return None when it does not follow the convention."""
    m = NAME_RE.match(stem)
    if not m:
        return None
    return ParsedName(m.group(1), m.group(2), m.group(3), int(m.group(4)))


def marker_expectation(parsed: Optional[ParsedName]) -> str:
    """What markers a capture must show: ``"none"``, ``"some"`` or ``"any"``.

    ``"some"`` means at least one marker with an expected ID. Empty trays and
    taped scenes must show none. In a calibration block only the ``oberteil``
    captures show a marker (Halter and Unterteil carry none).
    """
    if parsed is None:
        return "some"
    if parsed.block in ("empty", "tape"):
        return "none"
    if parsed.block == "test":
        return "any"
    if parsed.block == "calib":
        return "some" if parsed.scene.split("_")[0] == "oberteil" else "none"
    return "some"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def sample_files(samples_dir: Path, stem: str) -> list[Path]:
    """The four files that make up one converted sample."""
    return [Path(samples_dir) / f"{stem}{suffix}" for suffix in SAMPLE_SUFFIXES]


def is_converted(samples_dir: Path, stem: str) -> bool:
    """True when the sample files of ``stem`` all exist."""
    return all(p.exists() for p in sample_files(samples_dir, stem))


def settings_path(samples_dir: Path, stem: str) -> Path:
    return Path(samples_dir) / f"{stem}_settings.yml"


def settings_hash(samples_dir: Path, stem: str) -> Optional[str]:
    """Short SHA-1 of the saved Zivid settings text, or None if not saved."""
    path = settings_path(samples_dir, stem)
    if not path.exists():
        return None
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8]


def make_zivid_converter() -> Callable[[Path, Path, bool], None]:
    """Build the ``.zdf`` converter. Imports zivid here, so tests never need it.

    The returned function converts one file. With ``settings_only`` it only
    writes the settings dump and leaves the sample untouched. Output format
    equals the earlier conversion of the thesis captures.
    """
    import zivid
    import zivid.experimental.calibration

    app = zivid.Application()  # noqa: F841  (must stay alive while frames are read)

    def convert(zdf_path: Path, samples_dir: Path, settings_only: bool = False) -> None:
        stem = Path(zdf_path).stem
        samples_dir = Path(samples_dir)
        samples_dir.mkdir(parents=True, exist_ok=True)
        frame = zivid.Frame(str(zdf_path))
        frame.settings.save(str(settings_path(samples_dir, stem)))
        if settings_only:
            return
        cloud = frame.point_cloud()
        xyz = cloud.copy_data("xyz")
        rgba = cloud.copy_data("rgba_srgb")  # sRGB like the Jetson API; "rgba" is linear and about 2.2x darker
        K = zivid.experimental.calibration.estimate_intrinsics(frame).camera_matrix
        K = np.array([[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]])
        rgb_bgr = np.ascontiguousarray(rgba[..., :3][..., ::-1])
        cv2.imwrite(str(samples_dir / f"{stem}_rgb.png"), rgb_bgr)
        np.save(samples_dir / f"{stem}_xyz.npy", xyz.astype(np.float32))
        np.savetxt(samples_dir / f"{stem}_cam_K.txt", K)
        meta = {
            "schema_version": "1.0",
            "sample_id": stem,
            "source": "real",
            "capture_backend": "zivid_zdf",
            "units": "mm",
            "has_snr": False,
            "xyz_from": "sdk_measurement",
            "zdf": Path(zdf_path).name,
            "valid_fraction": float(np.isfinite(xyz[..., 2]).mean()),
            "files": {
                "rgb": f"{stem}_rgb.png",
                "xyz": f"{stem}_xyz.npy",
                "cam_k": f"{stem}_cam_K.txt",
                "settings": f"{stem}_settings.yml",
            },
        }
        (samples_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=1))

    return convert


def convert_new(
    zdf_files: list[Path], samples_dir: Path, converter: Optional[Callable[..., None]]
) -> tuple[list[str], list[str]]:
    """Convert files that have no sample yet; add missing settings dumps.

    Returns (converted stems, stems that could not be converted).
    """
    converted, failed = [], []
    for zdf in zdf_files:
        stem = zdf.stem
        have_sample = is_converted(samples_dir, stem)
        have_settings = settings_path(samples_dir, stem).exists()
        if have_sample and have_settings:
            continue
        if converter is None:
            if not have_sample:
                failed.append(stem)
            continue
        try:
            if have_sample:
                converter(zdf, samples_dir, settings_only=True)
            else:
                converter(zdf, samples_dir)
                converted.append(stem)
        except Exception as exc:  # a broken ZDF must not stop the check
            print(f"  could not convert {zdf.name}: {exc}", file=sys.stderr)
            failed.append(stem)
    return converted, failed


# ---------------------------------------------------------------------------
# Marker measurements
# ---------------------------------------------------------------------------


def detect_all_markers(rgb: np.ndarray, dictionary: str) -> tuple[list[Marker], bool]:
    """Detect every marker. Copies of one part carry the same ID.

    Returns:
        (markers as (id, corners) sorted by id and position, duplicate_ids flag)
    """
    markers = mgt.detect_markers(rgb, dictionary)
    ids = [marker_id for marker_id, _ in markers]
    return markers, len(ids) != len(set(ids))


def side_length_px(corners: np.ndarray) -> float:
    """Mean edge length of the marker quadrilateral in px."""
    edges = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    return float(edges.mean())


def valid_depth_fraction(xyz: np.ndarray, corners: Optional[np.ndarray] = None) -> float:
    """Fraction of finite depth pixels, in the whole frame or inside a marker."""
    z = np.asarray(xyz[..., 2])
    if corners is None:
        return float(np.isfinite(z).mean())
    mask = np.zeros(z.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 1)
    inside = z[mask.astype(bool)]
    return float(np.isfinite(inside).mean()) if inside.size else 0.0


def corner_spread_px(corner_sets: list[np.ndarray]) -> float:
    """Largest distance between any two captures of the same marker corner."""
    arr = np.stack(corner_sets)  # (n, 4, 2)
    diff = arr[:, None] - arr[None]
    return float(np.linalg.norm(diff, axis=-1).max())


# ---------------------------------------------------------------------------
# Per-capture result
# ---------------------------------------------------------------------------


@dataclass
class CaptureResult:
    stem: str
    parsed: Optional[ParsedName]
    valid_fraction: float
    markers: list[Marker]
    duplicate_ids: bool
    marker_depth_valid: list[float]
    settings: Optional[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def id_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for marker_id, _ in self.markers:
            counts[marker_id] = counts.get(marker_id, 0) + 1
        return counts

    @property
    def sides(self) -> list[float]:
        return [side_length_px(c) for _, c in self.markers]


def analyse_capture(
    samples_dir: Path,
    stem: str,
    expected_ids: list[int],
    dictionary: str = "DICT_4X4_50",
    min_valid: float = MIN_VALID_FRACTION,
    min_side_px: float = MIN_MARKER_SIDE_PX,
) -> CaptureResult:
    """Measure one converted capture and collect its warnings."""
    sample = mgt.load_sample(samples_dir, stem)
    parsed = parse_capture_name(stem)
    markers, duplicates = detect_all_markers(sample.rgb, dictionary)
    valid = valid_depth_fraction(sample.xyz) if sample.xyz is not None else 0.0
    depth_valid = [valid_depth_fraction(sample.xyz, c) for _, c in markers] if sample.xyz is not None else []
    result = CaptureResult(stem, parsed, valid, markers, duplicates, depth_valid, settings_hash(samples_dir, stem))
    warn = result.warnings

    if valid < min_valid:
        warn.append(f"low valid depth ({valid:.0%} < {min_valid:.0%})")
    expectation = marker_expectation(parsed)
    found_ids = set(result.id_counts)
    if expectation == "none" and markers:
        warn.append(f"markers visible but none expected: {_fmt_ids(result.id_counts)}")
    if expectation == "some" and expected_ids:
        missing = [i for i in expected_ids if i not in found_ids]
        if missing:
            warn.append(f"expected marker ID {missing} missing (found: {_fmt_ids(result.id_counts) or 'none'})")
    if expectation == "some" and not expected_ids and not markers:
        warn.append("no marker found")
    small = [s for s in result.sides if s < min_side_px]
    if small:
        warn.append(f"marker side {min(small):.1f} px < {min_side_px:.0f} px (too far, or out of focus)")
    if any(f < 0.5 for f in depth_valid):
        warn.append("depth valid on less than 50% of a marker")
    return result


def _fmt_ids(counts: dict[int, int]) -> str:
    return " ".join(f"{i}x{n}" for i, n in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Block checks
# ---------------------------------------------------------------------------


def _match_to_reference(ref: list[Marker], other: list[Marker]) -> dict[int, np.ndarray]:
    """For each reference marker, the corners of the nearest same-ID marker."""
    matched: dict[int, np.ndarray] = {}
    for k, (ref_id, ref_corners) in enumerate(ref):
        candidates = [c for i, c in other if i == ref_id]
        if not candidates:
            continue
        dist = [np.linalg.norm(c.mean(axis=0) - ref_corners.mean(axis=0)) for c in candidates]
        j = int(np.argmin(dist))
        if dist[j] < 20.0:
            matched[k] = candidates[j]
    return matched


def check_repeatability(results: list[CaptureResult], limit_px: float = REPEAT_LIMIT_PX) -> tuple[str, list[str]]:
    """Compare marker corners across the captures of one repeatability group.

    Markers are matched to the first capture by ID and nearest centre, because
    several copies share one ID. Returns (summary line, warnings).
    """
    results = sorted(results, key=lambda r: r.parsed.index if r.parsed else 0)
    ref = results[0]
    warnings: list[str] = []
    tracks: dict[int, list[np.ndarray]] = {k: [c] for k, (_, c) in enumerate(ref.markers)}
    for res in results[1:]:
        matched = _match_to_reference(ref.markers, res.markers)
        if len(matched) != len(ref.markers) or len(res.markers) != len(ref.markers):
            warnings.append(
                f"{res.stem}: {len(res.markers)} markers found, reference {ref.stem} has {len(ref.markers)}"
            )
        for k, corners in matched.items():
            tracks[k].append(corners)
    if not ref.markers:
        return f"{len(results)} captures, no markers in the reference capture", warnings + [
            f"{ref.stem}: no marker found, repeatability cannot be checked"
        ]
    spreads = [corner_spread_px(t) for t in tracks.values() if len(t) > 1]
    worst = max(spreads) if spreads else 0.0
    summary = (
        f"{len(results)} captures, {len(ref.markers)} markers, "
        f"largest corner difference between any two captures {worst:.2f} px (limit {limit_px:.1f})"
    )
    if worst > limit_px:
        warnings.append(f"marker corners move by up to {worst:.2f} px between captures (limit {limit_px:.1f} px)")
    return summary, warnings


def check_tape_pair(
    samples_dir: Path,
    marker_stem: str,
    taped_stem: str,
    tol_mm: float = PAIR_DEPTH_TOL_MM,
    max_fraction: float = PAIR_DEPTH_FRACTION,
) -> Optional[str]:
    """Warn when depth differs between a marker capture and its taped twin."""
    a = np.load(Path(samples_dir) / f"{marker_stem}_xyz.npy", mmap_mode="r")[..., 2]
    b = np.load(Path(samples_dir) / f"{taped_stem}_xyz.npy", mmap_mode="r")[..., 2]
    both = np.isfinite(a) & np.isfinite(b)
    if not both.any():
        return f"{taped_stem}: no common valid depth with {marker_stem}"
    changed = float((np.abs(a[both] - b[both]) > tol_mm).mean())
    if changed > max_fraction:
        return (
            f"{taped_stem}: depth differs by more than {tol_mm:.0f} mm on {changed:.1%} of the pixels "
            f"compared with {marker_stem}; something moved between the two captures"
        )
    return None


def check_log(zdf_stems: list[str], log_path: Path, all_stems: Optional[list[str]] = None) -> list[str]:
    """Compare the ``file`` column of the capture log with the ZDF files.

    ``zdf_stems`` are the files being checked and must each have a row.
    ``all_stems`` (default: the same) are all files in the folder; a row
    without a file in it is reported.
    """
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "file" not in rows[0]:
        return [f"{log_path.name}: no 'file' column"]
    logged = {Path(r["file"].strip()).stem for r in rows if r.get("file", "").strip()}
    warnings = [f"{s}: no row in {log_path.name}" for s in sorted(set(zdf_stems) - logged)]
    warnings += [f"{s}: in {log_path.name} but no .zdf file" for s in sorted(logged - set(all_stems or zdf_stems))]
    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def format_line(res: CaptureResult) -> str:
    ids = _fmt_ids(res.id_counts) or "-"
    if res.sides:
        side = f"{np.mean(res.sides):.1f}px" if len(res.sides) == 1 else (
            f"{np.mean(res.sides):.1f}px ({min(res.sides):.1f}-{max(res.sides):.1f})"
        )
    else:
        side = "-"
    mdepth = f"{np.mean(res.marker_depth_valid):.0%}" if res.marker_depth_valid else "-"
    dup = " dup-ID" if res.duplicate_ids else ""
    return (
        f"{res.stem}: valid {res.valid_fraction:.1%} | ids {ids}{dup} | side {side} | "
        f"marker depth {mdepth} | settings {res.settings or 'n/a'}"
    )


def run(
    zdf_dir: Path,
    samples_dir: Path,
    expected_ids: list[int],
    pattern: str = "*.zdf",
    converter: Optional[Callable[..., None]] = None,
    log_path: Optional[Path] = None,
    dictionary: str = "DICT_4X4_50",
    out: Callable[[str], None] = print,
) -> int:
    """Convert, measure and print. Returns the number of warnings."""
    zdf_files = sorted(Path(zdf_dir).glob(pattern))
    stems = [p.stem for p in zdf_files]
    if not zdf_files:
        out(f"no files matching {pattern} in {zdf_dir}")
        return 1
    converted, failed = convert_new(zdf_files, Path(samples_dir), converter)
    out(f"{len(zdf_files)} zdf files; converted {len(converted)} new, {len(stems) - len(converted) - len(failed)} already there")
    warnings: list[str] = [f"{s}: not converted" for s in failed]

    results: dict[str, CaptureResult] = {}
    for stem in stems:
        if stem in failed:
            continue
        res = analyse_capture(Path(samples_dir), stem, expected_ids, dictionary)
        results[stem] = res
        out(format_line(res) + ("" if not res.warnings else "  <-- " + "; ".join(res.warnings)))
        warnings += [f"{stem}: {w}" for w in res.warnings]

    groups: dict[str, list[CaptureResult]] = {}
    for res in results.values():
        if res.parsed and res.parsed.block == "rep":
            groups.setdefault(res.parsed.group_key, []).append(res)
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            warnings.append(f"{key}: only {len(members)} capture in the repeatability block")
            continue
        summary, block_warnings = check_repeatability(members)
        out(f"repeatability {key}: {summary}")
        warnings += [f"{key}: {w}" for w in block_warnings]

    for res in results.values():
        if res.parsed and res.parsed.block == "tape":
            partner = f"{res.parsed.session}_fill_{res.parsed.scene}_{res.parsed.index:02d}"
            if partner not in results:
                warnings.append(f"{res.stem}: marker capture {partner} not found")
                continue
            problem = check_tape_pair(Path(samples_dir), partner, res.stem)
            out(f"tape pair {partner} / {res.stem}: {'ok' if problem is None else 'CHECK'}")
            if problem:
                warnings.append(problem)

    hashes: dict[str, int] = {}
    for res in results.values():
        hashes[res.settings or "n/a"] = hashes.get(res.settings or "n/a", 0) + 1
    if len(hashes) > 1:
        summary = ", ".join(f"{h} x{n}" for h, n in sorted(hashes.items()))
        warnings.append(f"Zivid settings differ within this folder: {summary}")
    else:
        out(f"Zivid settings: {next(iter(hashes), 'n/a')} for all captures")

    if log_path is not None:
        all_stems = [p.stem for p in Path(zdf_dir).glob("*.zdf")]
        warnings += check_log(stems, log_path, all_stems)

    out(f"\n{len(warnings)} warning(s)")
    for w in warnings:
        out(f"  WARN {w}")
    return len(warnings)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("zdf_dir", type=Path)
    parser.add_argument("--samples-dir", type=Path, required=True, help="where converted samples go")
    parser.add_argument("--expect", default="", help="expected marker IDs, comma separated, e.g. 6")
    parser.add_argument("--pattern", default="*.zdf", help="glob for the .zdf files (default *.zdf)")
    parser.add_argument("--log", type=Path, help="capture_log.csv to compare with the files")
    parser.add_argument("--no-convert", action="store_true", help="only analyse samples that already exist")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    expected = [int(v) for v in args.expect.split(",") if v.strip()]
    converter = None if args.no_convert else make_zivid_converter()
    n_warnings = run(args.zdf_dir, args.samples_dir, expected, args.pattern, converter, args.log, args.dictionary)
    return 1 if n_warnings else 0


if __name__ == "__main__":
    sys.exit(main())
