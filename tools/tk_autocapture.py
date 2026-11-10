"""Capture automatically whenever the scene has changed and then stays still.

Runs on the Jetson `lara5roboter`. Polls fast 2D images (~1 per second). When the
scene differs from the last saved capture AND has been still for STILL_S seconds,
it takes a full 2D+3D capture (thesis settings) and saves <out>/<prefix>_NNN.zdf
plus a preview; a `.done` file marks it complete for the transfer loop.

Usage: python3 tk_autocapture.py PREFIX [--minutes 120] [--out DIR] [--still-s 3]
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import zivid

HERE = Path(__file__).parent
SETTINGS = HERE / "settings_thesis_einlegen.yml"
SCALE = 0.125          # analyse 306x256 thumbnails
PIX_DIFF = 25          # grey-level change that counts as "changed pixel"
STILL_FRAC = 0.0005    # max changed fraction between consecutive polls to count as still
CHANGED_FRAC = 0.0015  # min changed fraction vs. the last saved capture
STILL_S = 3.0
MIN_FREE_GB = 1.0


def thumb(bgra: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (5, 5), 0).astype(np.int16)


def changed(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(a - b) > PIX_DIFF).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument("--still-s", type=float, default=STILL_S,
                        help="seconds without change before a capture (longer = fewer captures while setting up)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    app = zivid.Application()  # must exist before Settings.load
    settings = zivid.Settings.load(str(SETTINGS))
    camera = app.connect_camera()
    n = 1 + max([int(p.stem.rsplit("_", 1)[1]) for p in out.glob(f"{args.prefix}_*.done")] or [0])
    saved, prev, still_since = None, None, None
    end = time.time() + 60 * args.minutes
    print(f"autocapture {args.prefix} from #{n:03d}", flush=True)
    try:
        while time.time() < end:
            cur = thumb(np.asarray(camera.capture_2d(settings.color).image_bgra().copy_data()))
            if prev is not None and changed(cur, prev) <= STILL_FRAC:
                still_since = still_since or time.time()
            else:
                still_since = None
            prev = cur
            if saved is None:  # reference = scene at start, never captured
                saved = cur
                continue
            if still_since and time.time() - still_since >= args.still_s and changed(cur, saved) >= CHANGED_FRAC:
                free_gb = __import__("shutil").disk_usage(out).free / 1e9
                if free_gb < MIN_FREE_GB:
                    print(f"disk low ({free_gb:.1f} GB), waiting for transfer", flush=True)
                    time.sleep(5)
                    continue
                name = f"{args.prefix}_{n:03d}"
                frame = camera.capture_2d_3d(settings)
                frame.save(str(out / f"{name}.zdf"))
                rgba = frame.point_cloud().copy_data("rgba_srgb")
                cv2.imwrite(str(out / f"{name}_preview.jpg"),
                            cv2.resize(cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR), (1224, 1024)),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                (out / f"{name}.done").touch()
                print(f"{time.strftime('%H:%M:%S')} saved {name} (changed {changed(cur, saved):.4f})", flush=True)
                saved, still_since, n = cur, None, n + 1
            time.sleep(0.5)
    finally:
        camera.disconnect()


if __name__ == "__main__":
    main()
