"""Scene-card driven capture loop: autocapture -> workstation -> ground truth -> review tool.

Runs on the Mac and drives the existing tools. The operator sets up the scene shown by
``next_scene`` and steps back; ``tk_autocapture.py`` on the Jetson takes the capture once
the scene has changed and stayed still. ``next`` then processes **one** capture and exits:

1. streams the ZDF to the workstation under the name of the current scene card and
   verifies the checksum (the Jetson disk is nearly full, the copy is deleted),
2. converts it to a sample (``check_captures.py``),
3. appends the card's layout to ``capture_log.csv`` and a row to ``review_log.csv``,
4. runs ``cell_ground_truth.py run`` for that capture only,
5. prints the tool verdict and the next scene.

A capture the tool rejects does not advance the card; the next capture counts as a new
attempt of the same card (named ``<scene> + 100 * (attempt - 1)``). The human verdicts
stay in the review tool.

Usage::

    python3 tools/capture_feed.py start  --session S03 --sess-dir benchmark_2026-10-06
    python3 tools/capture_feed.py next   --session S03 --sess-dir benchmark_2026-10-06
    python3 tools/capture_feed.py status --session S03 --sess-dir benchmark_2026-10-06
    python3 tools/capture_feed.py skip | stop ...
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs/researcheffort"
STATE_DIR = Path.home() / ".capture_feed"
BENCH = "Desktop/Gruppe3/testkoerper_benchmark"
JETSON_DIR = "testkoerper_benchmark_capture"
CALIBRATION = "runs/benchmark_2026-10-01_v2/calibration.json"
POSITIONS = "data/plate_positions_2026-10-01.json"
# Step-specific settings. ``projection`` = the scene wireframes of ``tk_project_scene.py`` are measured for this step.
STEPS = {
    "poty": {"label": "Einlegen_PO_TY", "name": "Einsetzen_(PO_TY)", "cards": "scene_cards_poty_v1.csv",
             "config": "data/cell_parts_poty.json", "projection": True, "reference": "S02 fill_poty-a_01"},
    "syrz": {"label": "Einlegen_SY_RZ", "name": "Einsetzen_(SY_RZ)", "cards": "scene_cards_syrz_v1.csv",
             "config": "data/cell_parts_syrz.json", "projection": True,
             "calibration": "runs/benchmark_2026-10-28/calibration.json", "positions": "data/plate_positions_2026-10-28.json",
             "reference": "Unterteil: Wand-zu-Wand-Richtung waagerecht im Kamerabild"},
}
STEP_KEY = "poty"
CARDS = DOCS / STEPS[STEP_KEY]["cards"]
CELL_CONFIG = STEPS[STEP_KEY]["config"]
STEP_NAME = STEPS[STEP_KEY]["name"]
SLOT_CODES = ("U", "H", "HO", "UO")


# --- pure helpers (tested) ---------------------------------------------------------

def set_step(key: str) -> dict:
    """Select the step this process works on (cards file, cell config, step name); returns its settings."""
    global STEP_KEY, CARDS, CELL_CONFIG, STEP_NAME, CALIBRATION, POSITIONS
    STEP_KEY, spec = key, STEPS[key]
    CARDS, CELL_CONFIG, STEP_NAME = DOCS / spec["cards"], spec["config"], spec["name"]
    CALIBRATION, POSITIONS = spec.get("calibration", CALIBRATION), spec.get("positions", POSITIONS)
    return spec


def load_cards(path: Path, session: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["session"] == session]
    return sorted(rows, key=lambda r: int(r["scene"]))


def attempt_name(card: dict, attempt: int) -> str:
    """Capture name of attempt ``attempt`` (1-based) of a card: scene number + 100 per retry."""
    if attempt <= 1:
        return card["name"]
    head, _ = card["name"].rsplit("_", 1)
    return f"{head}_{int(card['scene']) + 100 * (attempt - 1):03d}"


def next_attempt(card: dict, history: list[dict]) -> int:
    """First attempt number of ``card`` that has no capture yet (names must never collide)."""
    used = [h.get("attempt", 1) for h in history if h.get("card") == card["name"] and h.get("name")]
    return max(used, default=0) + 1


def first_open(cards: list[dict], history: list[dict]) -> int:
    """Index of the first card without an accepted, stored or skipped capture (``len(cards)`` if none)."""
    done = {h["card"] for h in history if h.get("outcome") in ("accepted", "stored", "skipped")}
    return next((i for i, c in enumerate(cards) if c["name"] not in done), len(cards))


def parse_quarters(text: str) -> dict[str, tuple[str, str]]:
    """``'L1:UO/q3 R2:H'`` -> ``{'L1': ('UO', 'q3'), 'R2': ('H', '')}``."""
    cells = {}
    for token in (text or "").split():
        slot, _, rest = token.partition(":")
        code, _, turn = rest.partition("/")
        cells[slot] = (code, turn)
    return cells


def render_card(card: dict, attempt: int = 1) -> str:
    """Operator view of a scene card: two 3x3 grids (as seen in the camera image)."""
    cells = parse_quarters(card["quarter_turns"])
    lines = [f"Szene {card['name']}  ({card['fill']}, {card['n_occupied']} Teile)"
             + (f"  [Versuch {attempt}]" if attempt > 1 else ""),
             f"Platten: links {card['plate_L']}, rechts {card['plate_R']}   "
             f"Session {card['session']}, Tag {card['day']}, {card['split']}",
             "Raster wie im Kamerabild (oben links = Slot 1):  U=Unterteil  H=Halter  "
             "HO=Oberteil auf Halter  UO=Oberteil im Unterteil  qN=N Vierteldrehungen gegen den Uhrzeigersinn", ""]
    for row in range(3):
        parts = []
        for side in ("L", "R"):
            cols = []
            for col in range(3):
                code, turn = cells.get(f"{side}{row * 3 + col + 1}", ("", ""))
                cols.append(f"{(code + ' ' + turn).strip() or '.':>7}")
            parts.append(" ".join(cols))
        lines.append(f"  {parts[0]}      |      {parts[1]}")
    lines += ["", f"Layout: {card['layout']}"]
    return "\n".join(lines)


def review_row(name: str, card: dict, now: str) -> str:
    """One ``review_log.csv`` row (layout from the card, not checked by a reviewer)."""
    layout = card["layout"].replace('"', "'")
    return (f'{name}.zdf,accept,{STEP_NAME},{card["plate_L"]},{card["plate_R"]},"{layout}",'
            f'"scene card {card["name"]}; layout not checked by a reviewer",{now}')


def reject_row(name: str, card: dict, now: str, note: str) -> str:
    """``review_log.csv`` row that withdraws a capture (the latest row of a file wins; verdict != accept = skipped)."""
    layout = card["layout"].replace('"', "'")
    return f'{name}.zdf,reject,{STEP_NAME},{card["plate_L"]},{card["plate_R"]},"{layout}","{note}",{now}'


def capture_row(name: str, card: dict, now: str) -> str:
    return f'{name}.zdf,fill,"{card["layout"].replace(chr(34), chr(39))}",{now}'


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"index": 0, "attempt": 1, "history": []}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def classify(verdict: dict) -> str:
    """``stored`` (ground truth skipped), ``accepted`` (tool ok), ``rejected`` (tool says the scene is wrong)
    or ``error`` (no ground truth produced)."""
    if verdict.get("gt_skipped"):
        return "stored"
    if not verdict.get("found"):
        return "error"
    return "rejected" if verdict["rejected"] or verdict["layout_mismatch"] else "accepted"


def summarize(report: dict | None, summary: dict | None, name: str) -> dict:
    """Condensed tool verdict from the per-capture report and the run summary."""
    entry = ((summary or {}).get("captures") or {}).get(name, {})
    parts = entry.get("parts_by_type") or {}
    return {
        "rejected": bool(entry.get("rejected", False)),
        "reasons": entry.get("rejection_reasons") or [],
        "layout_mismatch": (report or {}).get("layout_mismatch") or {},
        "unsupported_slots": (report or {}).get("unsupported_slots") or {},
        "plate_adjustment": (report or {}).get("plate_adjustment") or {},
        "depth_residual_mm": {k: v.get("depth_residual_median_mm") for k, v in parts.items()},
        "flagged": {k: v.get("flagged") for k, v in parts.items() if v.get("flagged")},
    }


# --- ssh plumbing -------------------------------------------------------------------

def _ssh(host: str, command: str, check: bool = True, timeout: int = 600) -> str:
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, command],
                            capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"ssh {host} failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def _stream(src_cmd: str, dst_cmd: str) -> None:
    producer = subprocess.Popen(["ssh", "lara5roboter", src_cmd], stdout=subprocess.PIPE)
    consumer = subprocess.run(["ssh", "Workstationlernfabrik", dst_cmd], stdin=producer.stdout)
    producer.stdout.close()
    if producer.wait() != 0 or consumer.returncode != 0:
        raise RuntimeError("streaming the ZDF failed")


def _append_line(path_on_ws: str, header: str, line: str) -> None:
    quoted = shlex.quote(line)
    _ssh("Workstationlernfabrik",
         f"cd ~ && f={shlex.quote(path_on_ws)} && {{ test -e \"$f\" || echo {shlex.quote(header)} > \"$f\"; }} && echo {quoted} >> \"$f\"")


def projector_cmd() -> str:
    return (f"python3 tk_project_scene.py --cards {CARDS.name} "
            f"--calibration cell_calibration_2026-10-01.json --cad cad --step {STEP_KEY}")


def stop_projection() -> None:
    """End the scene projection on the Jetson (the camera allows one client, so a capture needs it gone)."""
    _ssh("lara5roboter", "pkill -f 'tk_project_scene[.]py'; sleep 1.5; true", check=False, timeout=30)


def project_card(card_name: str) -> None:
    """Project the wireframes of a scene card onto the table (replaces a running projection)."""
    stop_projection()
    _ssh("lara5roboter",
         f"cd ~/{JETSON_DIR} && (setsid nohup {projector_cmd()} --card {shlex.quote(card_name)} --seconds 7200 "
         f"> /tmp/tk_project_scene.log 2>&1 < /dev/null &); true", check=False, timeout=30)


def direct_capture(name: str) -> None:
    """Take one capture on the Jetson (3 tries, like ``tk_shot.sh``); the camera must not be in use."""
    stop_projection()
    out = _ssh("lara5roboter",
               f"cd ~/{JETSON_DIR} && for i in 1 2 3; do python3 tk_capture.py {shlex.quote(name)} 2>&1 | grep -E '^{name}:' && break; "
               f"rm -f out/{name}.zdf; sleep 2; done; test -e out/{name}.zdf && echo OK", check=False, timeout=300)
    if "OK" not in out:
        raise RuntimeError("capture on the Jetson failed (is the autocapture still running? use `capture_feed.py stop`)")


def fetch_preview(name: str, sess_dir: str) -> Path | None:
    """Copy the capture preview from the workstation to the Mac; ``None`` if it does not exist."""
    target = STATE_DIR / "previews" / sess_dir / f"{name}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(["scp", "-q", f"Workstationlernfabrik:{BENCH}/data/zdf/{sess_dir}/previews/{name}_preview.jpg", str(target)],
                          capture_output=True)
    return target if done.returncode == 0 and target.is_file() else None


def capture_and_process(card: dict, attempt: int, sess_dir: str, progress=lambda message: None,
                        run_gt: bool = True) -> tuple[str, dict]:
    """The manual trigger: capture now, then the same path as for an autocapture."""
    name = attempt_name(card, attempt)
    progress("Aufnahme auf dem Jetson")
    direct_capture(name)
    return process_capture(name, card, attempt, sess_dir, progress, run_gt=run_gt)


def next_jetson_capture() -> str | None:
    out = _ssh("lara5roboter", f"cd ~/{JETSON_DIR}/out && ls *.done 2>/dev/null | sed 's/\\.done$//' | sort | head -1",
               check=False)
    return out.strip() or None


GT_SKIPPED = {"found": None, "rejected": False, "reasons": [], "layout_mismatch": {}, "unsupported_slots": {},
              "plate_adjustment": {}, "depth_residual_mm": {}, "flagged": {}, "gt_skipped": True}


def process_capture(jetson_name: str, card: dict, attempt: int, sess_dir: str,
                    progress=lambda message: None, run_gt: bool = True) -> tuple[str, dict]:
    name = attempt_name(card, attempt)
    progress("Übertragung zur Workstation")
    d = f"{BENCH}/data/zdf/{sess_dir}"
    j = f"{JETSON_DIR}/out"
    _ssh("Workstationlernfabrik", f"mkdir -p ~/{d}/previews && test ! -e ~/{d}/{shlex.quote(name)}.zdf")
    _stream(f"cat ~/{j}/{jetson_name}.zdf", f"cat > ~/{d}/{name}.zdf")
    a = _ssh("lara5roboter", f"sha256sum ~/{j}/{jetson_name}.zdf")[:64]
    b = _ssh("Workstationlernfabrik", f"sha256sum ~/{d}/{name}.zdf")[:64]
    if a != b:
        raise RuntimeError(f"checksum mismatch for {jetson_name}; Jetson copy kept")
    try:
        _stream(f"cat ~/{j}/{jetson_name}_preview.jpg", f"cat > ~/{d}/previews/{name}_preview.jpg")
    except RuntimeError:
        pass                                    # the preview is a convenience, not part of the data
    _ssh("lara5roboter", f"rm -f ~/{j}/{jetson_name}.zdf ~/{j}/{jetson_name}_preview.jpg ~/{j}/{jetson_name}.done")
    progress("Umwandlung in ein Sample")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _ssh("Workstationlernfabrik",
         f"cd ~/{BENCH} && .venv/bin/python tools/check_captures.py data/zdf/{sess_dir} "
         f"--samples-dir data/samples/{sess_dir} --pattern {shlex.quote(name + '.zdf')} >/dev/null 2>&1", check=False)
    # Show the capture in the review tool right away (it lists every run/images/<id>.png). A copy, not a
    # link: the ground truth later writes the marker-free image to the same path. `cp -n` keeps that one.
    progress("In das Review-Tool übernehmen")
    _ssh("Workstationlernfabrik",
         f"cd ~/{BENCH} && mkdir -p runs/{sess_dir}/images && test -e data/samples/{sess_dir}/{shlex.quote(name)}_rgb.png && "
         f"cp -n data/samples/{sess_dir}/{shlex.quote(name)}_rgb.png runs/{sess_dir}/images/{shlex.quote(name)}.png", check=False)
    _append_line(f"{d}/capture_log.csv", "file,block,layout,time", capture_row(name, card, now))
    _append_line(f"{d}/review_log.csv", "file,verdict,step,pos_L,pos_R,layout,notes,reviewed_at", review_row(name, card, now))
    if not run_gt:
        return name, dict(GT_SKIPPED)
    progress("Ground Truth wird berechnet")
    t0 = time.time()
    gt = _ssh("Workstationlernfabrik",
              f"cd ~/{BENCH} && .venv/bin/python tools/cell_ground_truth.py run data/samples/{sess_dir} "
              f"--config {CELL_CONFIG} --calibration {CALIBRATION} --positions {POSITIONS} "
              f"--capture-log data/zdf/{sess_dir}/capture_log.csv --default-step {STEP_KEY} "
              f"--out runs/{sess_dir} --sample-id {shlex.quote(name)} 2>&1 | tail -15", check=False, timeout=1800)
    seconds = round(time.time() - t0)
    report = _read_json(f"{BENCH}/runs/{sess_dir}/annotations/{name}_report.json")
    summary = _read_json(f"{BENCH}/runs/{sess_dir}/summary.json")
    verdict = summarize(report, summary, name)
    verdict["gt_seconds"] = seconds
    verdict["gt_output_tail"] = gt.strip().splitlines()[-4:]
    verdict["found"] = bool(report)
    return name, verdict


def reject_capture(name: str, card: dict, sess_dir: str) -> None:
    """Withdraw a stored capture in the session's review log (operator retake: hand in view, crooked part, ...)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _append_line(f"{BENCH}/data/zdf/{sess_dir}/review_log.csv", "file,verdict,step,pos_L,pos_R,layout,notes,reviewed_at",
                 reject_row(name, card, now, "retaken by the operator"))


def _read_json(path_on_ws: str):
    out = _ssh("Workstationlernfabrik", f"cat ~/{shlex.quote(path_on_ws)} 2>/dev/null", check=False)
    try:
        return json.loads(out) if out.strip() else None
    except json.JSONDecodeError:
        return None


# --- commands -----------------------------------------------------------------------

def _paths(args):
    return STATE_DIR / f"{args.sess_dir}_{args.session}.json", load_cards(CARDS, args.session)


def cmd_start(args) -> int:
    state_path, cards = _paths(args)
    if not cards:
        print(f"no cards for session {args.session}", file=sys.stderr)
        return 1
    running = _ssh("lara5roboter", "pgrep -f '[t]k_autocapture.py' || true", check=False).strip()
    if running:
        print(f"autocapture already running on the Jetson (pid {running})")
    else:
        _ssh("lara5roboter",
             f"cd ~/{JETSON_DIR}; nohup python3 tk_autocapture.py {args.session}_auto --minutes {args.minutes} "
             f"--still-s {args.still_s} > autocapture.log 2>&1 < /dev/null &", check=False)
        time.sleep(8)
        print(_ssh("lara5roboter", f"tail -2 ~/{JETSON_DIR}/autocapture.log", check=False).strip())
    state = load_state(state_path)
    save_state(state_path, state)
    print(f"\n{len(cards)} cards, next: {state['index'] + 1}/{len(cards)}\n")
    print(render_card(cards[state["index"]], state["attempt"]))
    return 0


def cmd_next(args) -> int:
    state_path, cards = _paths(args)
    state = load_state(state_path)
    if state["index"] >= len(cards):
        print("all scenes of this session are done")
        return 0
    card, deadline = cards[state["index"]], time.time() + args.timeout
    jetson_name = None
    while time.time() < deadline:
        jetson_name = next_jetson_capture()
        if jetson_name:
            break
        time.sleep(3)
    if not jetson_name:
        print("TIMEOUT: no new capture")
        return 2
    name, verdict = process_capture(jetson_name, card, state["attempt"], args.sess_dir, run_gt=not args.no_gt)
    outcome = classify(verdict)
    ok = outcome in ("accepted", "stored")
    state["history"].append({"name": name, "card": card["name"], "attempt": state["attempt"], "outcome": outcome})
    if ok:
        state["index"], state["attempt"] = state["index"] + 1, 1
    else:
        state["attempt"] += 1                      # the name is taken, the next capture gets a new one
    save_state(state_path, state)
    print({"accepted": "CAPTURED ", "stored": "STORED (ground truth skipped) ", "rejected": "REJECTED_BY_TOOL ", "error": "PROCESSING_ERROR (capture is stored, repeat not needed) "}[outcome] + name)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\nProgress: {state['index']}/{len(cards)} scenes accepted by the tool\n")
    if state["index"] < len(cards):
        print(("NEXT SCENE:\n" if ok else "REPEAT THE SAME SCENE (the tool did not accept the capture):\n")
              + render_card(cards[state["index"]], state["attempt"]))
        if outcome == "error":
            print("\n(Processing error: fix it and run `reprocess`; the scene itself does not need to be repeated.)")
    else:
        print("SESSION COMPLETE")
    return 0


def cmd_status(args) -> int:
    state_path, cards = _paths(args)
    state = load_state(state_path)
    print(f"{state['index']}/{len(cards)} accepted, attempt {state['attempt']}, history {len(state['history'])}")
    if state["index"] < len(cards):
        print(render_card(cards[state["index"]], state["attempt"]))
    return 0


def cmd_skip(args) -> int:
    state_path, cards = _paths(args)
    state = load_state(state_path)
    state["history"].append({"card": cards[state["index"]]["name"], "skipped": True})
    state["index"], state["attempt"] = state["index"] + 1, 1
    save_state(state_path, state)
    return cmd_status(args)


def cmd_reprocess(args) -> int:
    """Rerun the ground truth for the stored captures of the last attempt (after a processing error)."""
    state_path, cards = _paths(args)
    state = load_state(state_path)
    errors = [h for h in state["history"] if h.get("outcome") == "error"]
    if not errors:
        print("nothing to reprocess")
        return 0
    entry = errors[-1]
    name = entry["name"]
    sess = args.sess_dir
    _ssh("Workstationlernfabrik",
         f"cd ~/{BENCH} && .venv/bin/python tools/cell_ground_truth.py run data/samples/{sess} --config {CELL_CONFIG} "
         f"--calibration {CALIBRATION} --positions {POSITIONS} --capture-log data/zdf/{sess}/capture_log.csv "
         f"--default-step {STEP_KEY} --out runs/{sess} --sample-id {shlex.quote(name)} >/dev/null 2>&1", check=False, timeout=1800)
    verdict = summarize(_read_json(f"{BENCH}/runs/{sess}/annotations/{name}_report.json"),
                        _read_json(f"{BENCH}/runs/{sess}/summary.json"), name)
    verdict["found"] = verdict["found"] if "found" in verdict else True
    outcome = "accepted" if (not verdict["rejected"] and not verdict["layout_mismatch"]) else "rejected"
    entry["outcome"] = outcome
    if outcome == "accepted" and entry["card"] == cards[state["index"]]["name"]:
        state["index"], state["attempt"] = state["index"] + 1, 1
    save_state(state_path, state)
    print(outcome.upper(), name, json.dumps(verdict, ensure_ascii=False))
    return 0


def run_session_gt(sess_dir: str) -> str:
    """Ground truth for every capture of the session in one go; returns the tail of the tool output."""
    return _ssh("Workstationlernfabrik",
                f"cd ~/{BENCH} && .venv/bin/python tools/cell_ground_truth.py run data/samples/{sess_dir} --config {CELL_CONFIG} "
                f"--calibration {CALIBRATION} --positions {POSITIONS} --capture-log data/zdf/{sess_dir}/capture_log.csv "
                f"--default-step {STEP_KEY} --out runs/{sess_dir} 2>&1 | grep -vE 'RuntimeWarning|nanmedian' | tail -80",
                check=False, timeout=7200).strip()


_GT_LINE = re.compile(r"^(S\d+_[\w\-]+): positions (\{[^}]*\}), (\d+) parts, (ok|REJECTED)")


def summarize_gt_output(text: str) -> list[dict]:
    """One entry per capture line of the tool output: ``{name, parts, verdict}``."""
    entries = []
    for line in text.splitlines():
        match = _GT_LINE.match(line.strip())
        if match:
            entries.append({"name": match.group(1), "parts": int(match.group(3)),
                            "verdict": "accepted" if match.group(4) == "ok" else "rejected"})
    return entries


def cmd_gt(args) -> int:
    """Ground truth for every capture of the session in one go (after captures made without it)."""
    output = run_session_gt(args.sess_dir)
    for entry in summarize_gt_output(output):
        print(f"{entry['name']:24s} {entry['parts']:3d} Teile  {entry['verdict']}")
    return 0


def cmd_stop(args) -> int:
    _ssh("lara5roboter", "pkill -f '[t]k_autocapture.py' || true", check=False)
    print("autocapture stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["start", "next", "status", "skip", "stop", "reprocess", "gt"])
    parser.add_argument("--session", default="S03")
    parser.add_argument("--sess-dir", default=f"benchmark_{datetime.now().strftime('%Y-%m-%d')}")
    parser.add_argument("--timeout", type=int, default=3000, help="seconds to wait for a capture (next)")
    parser.add_argument("--minutes", type=int, default=240, help="autocapture duration (start)")
    parser.add_argument("--no-gt", action="store_true", help="skip the ground truth after each capture (next)")
    parser.add_argument("--still-s", type=float, default=20.0, help="seconds of stillness before a capture (start)")
    parser.add_argument("--step", default="poty", choices=sorted(STEPS))
    args = parser.parse_args(argv)
    set_step(args.step)
    return {"start": cmd_start, "next": cmd_next, "status": cmd_status, "skip": cmd_skip, "stop": cmd_stop,
            "reprocess": cmd_reprocess, "gt": cmd_gt}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
