"""Stream live Zivid 2D colour frames from the Jetson as length-prefixed JPEGs on stdout.

Runs on the Jetson (holds the camera while it runs; stop it before tk_shot.sh, the projector tools, or Zivid Studio).
Each frame: 8-byte big-endian length, then the JPEG (half resolution, 1224 x 1024, like the training renders).

    ssh lara5roboter 'python3 ~/testkoerper_benchmark_capture/tk_live_capture.py' | <consumer>

Options: --interval SECONDS between captures (default 0.5), --settings FILE (default: the thesis settings, 2D part only).
"""
import argparse
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zivid

parser = argparse.ArgumentParser()
parser.add_argument("--settings", default=str(Path(__file__).with_name("settings_thesis_einlegen.yml")))
parser.add_argument("--interval", type=float, default=0.5)
parser.add_argument("--quality", type=int, default=90)
args = parser.parse_args()

app = zivid.Application()
camera = app.connect_camera()
settings_2d = zivid.Settings.load(args.settings).color
out = sys.stdout.buffer
try:
    while True:
        t0 = time.time()
        frame = camera.capture_2d(settings_2d)
        rgba = frame.image_rgba_srgb().copy_data()
        bgr = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2BGR)
        small = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        out.write(struct.pack(">Q", len(jpg)) + jpg.tobytes())
        out.flush()
        sys.stderr.write(f"frame {time.time() - t0:.2f}s {len(jpg) // 1000} kB\n")
        time.sleep(max(0.0, args.interval - (time.time() - t0)))
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    camera.disconnect()
