"""Counterbalanced, seeded scene cards for benchmark capture sessions.

A scene card says which part code (``U``, ``H``, ``HO``, ``UO``) stands in which of the
18 slots (``L1..L9`` left plate, ``R1..R9`` right plate, row-major in the camera
image) and in which quarter-turn orientation. Cards are generated in advance so that
part copies, slots, orientations and fill levels are not correlated with the operator's
habits (``benchmark_testkoerper.md`` §4.2, ``ground_truth_capture.md`` §1).

Balancing target: equal instance counts of the three classes Unterteil, Halter,
Oberteil over all scenes, similar use of every slot, and equal use of the four
orientations per class. Each scene is the best of ``CANDIDATES`` random candidates
under that cost, so the result is deterministic for a given seed.

Usage::

    python3 tools/make_scene_cards.py --out docs/researcheffort/scene_cards_poty_v1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SLOTS = {"L": [f"L{i}" for i in range(1, 10)], "R": [f"R{i}" for i in range(1, 10)]}
ALL_SLOTS = SLOTS["L"] + SLOTS["R"]
CLASSES = ("Unterteil", "Halter", "Oberteil")

STEPS = {
    "poty": {
        "label": "Einlegen_PO_TY",
        "marker_id": 1,
        "codes": {"U": ("Unterteil",), "H": ("Halter",),
                  "HO": ("Halter", "Oberteil"), "UO": ("Unterteil", "Oberteil")},
        "reference": "the orientation in S02 `fill_poty-a_01`",
    },
    "syrz": {
        "label": "Einlegen_SY_RZ",
        "marker_id": 5,
        "codes": {"U": ("Unterteil",), "H": ("Halter",),
                  "HO": ("Halter", "Oberteil"), "UO": ("Unterteil", "Oberteil")},
        "reference": "q0 = the Unterteil's wall-to-wall direction horizontal in the camera image",
    },
}
COPIES = {"Unterteil": 8, "Halter": 8, "Oberteil": 8}    # physical copies of each part (Marc, 2026-10-06)
# At most Unterteil + Halter copies can be filled: every slot needs an Unterteil or a Halter.
FILL_RANGE = {"full": (14, 16), "medium": (6, 13), "sparse": (1, 5), "empty": (0, 0)}
FILL_SHARE = (("full", 0.40), ("medium", 0.40), ("sparse", 0.16), ("empty", 0.04))
PLATE_CONFIGS = [("P1", "P2"), ("P2", "P1"), ("P1", "P1"), ("P2", "P2"), ("P1", "P2")]
CANDIDATES = 300
PRIMARY_SHARE = 0.7        # share of parts on a plate that use the plate's primary code


def fill_plan(n: int, sessions: int, rng: np.random.Generator) -> list[list[str]]:
    """Fill class of every scene, dealt so that each session gets a full and a sparse scene."""
    counts = {name: int(round(share * n)) for name, share in FILL_SHARE}
    counts["medium"] += n - sum(counts.values())
    ordered = [name for name, _ in FILL_SHARE for _ in range(counts[name])]
    per_session: list[list[str]] = [[] for _ in range(sessions)]
    for i, name in enumerate(ordered):
        per_session[i % sessions].append(name)
    for scenes in per_session:
        rng.shuffle(scenes)
    return per_session


def _composition(rng, n_occupied: int, copies: dict[str, int]) -> dict[str, int]:
    """Counts of the codes U, H, HO, UO within the physical copies of Unterteil, Halter and Oberteil."""
    max_slots = copies["Unterteil"] + copies["Halter"]
    if n_occupied > max_slots:
        raise ValueError(f"at most {max_slots} slots can be filled with {copies}")
    unterteil = int(rng.integers(max(0, n_occupied - copies["Halter"]), min(copies["Unterteil"], n_occupied) + 1))
    halter = n_occupied - unterteil
    uo = int(rng.integers(0, min(unterteil, copies["Oberteil"]) + 1))
    ho = int(rng.integers(0, min(halter, copies["Oberteil"] - uo) + 1))
    return {"U": unterteil - uo, "H": halter - ho, "HO": ho, "UO": uo}


def _candidate(rng, n_occupied: int, copies: dict[str, int] = COPIES):
    counts = _composition(rng, n_occupied, copies)
    groups = [code for code in rng.permutation(list(counts)) for _ in range(counts[str(code)])]
    lo, hi = max(0, n_occupied - 9), min(9, n_occupied)
    on_left = int(rng.integers(lo, hi + 1))
    cells: dict[str, str] = {}
    quarter: dict[str, int] = {}
    for side, codes_on_plate in (("L", groups[:on_left]), ("R", groups[on_left:])):
        slots = rng.choice(SLOTS[side], size=len(codes_on_plate), replace=False) if codes_on_plate else []
        for slot, code in zip(slots, rng.permutation(codes_on_plate) if codes_on_plate else []):
            cells[str(slot)] = str(code)      # same codes are grouped, so a plate is mostly one or two codes
            quarter[str(slot)] = int(rng.integers(0, 4))
    return cells, quarter


def _spread(values) -> float:
    values = list(values)
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values)


def _tally(totals, code_classes, cells, quarter):
    cls, slot, ori, state = (dict(totals[k]) for k in ("cls", "slot", "ori", "code"))
    for s, code in cells.items():
        slot[s] += 1
        state[code] += 1
        for c in code_classes[code]:
            cls[c] += 1
            if c != "Halter":                   # the Halter is symmetric: no orientation
                ori[(c, quarter[s])] += 1
    return {"cls": cls, "slot": slot, "ori": ori, "code": state}


def _cost(t) -> float:
    ori_cost = sum(_spread(t["ori"][(c, q)] for q in range(4)) for c in ("Unterteil", "Oberteil"))
    return (3.0 * _spread(t["cls"].values()) + 2.0 * _spread(t["code"].values())
            + _spread(t["slot"].values()) + ori_cost)


def make_cards(step: str = "poty", n: int = 50, sessions: int = 5, seed: int = 2026,
               first_session: int = 3, dev_sessions: tuple[int, ...] = (0, 3),
               copies: dict[str, int] = COPIES) -> list[dict]:
    """Generate ``n`` scene cards in ``sessions`` sessions (``n`` must divide evenly)."""
    if n % sessions:
        raise ValueError("n must be a multiple of sessions")
    spec = STEPS[step]
    code_classes = spec["codes"]
    rng = np.random.default_rng(seed)
    plan = fill_plan(n, sessions, rng)
    totals = {"cls": {c: 0 for c in CLASSES}, "slot": {s: 0 for s in ALL_SLOTS},
              "ori": {(c, q): 0 for c in ("Unterteil", "Oberteil") for q in range(4)},
              "code": {code: 0 for code in code_classes}}
    cards = []
    for si in range(sessions):
        left, right = PLATE_CONFIGS[si % len(PLATE_CONFIGS)]
        for scene_no, fill in enumerate(plan[si], start=1):
            lo, hi = FILL_RANGE[fill]
            n_occ = int(rng.integers(lo, hi + 1))
            best = min((_candidate(rng, n_occ, copies) for _ in range(CANDIDATES)),
                       key=lambda cq: _cost(_tally(totals, code_classes, *cq)))
            cells, quarter = best
            totals = _tally(totals, code_classes, cells, quarter)
            session = first_session + si
            name = f"S{session:02d}_fill_{step}-c_{scene_no:02d}"
            has_oberteil = any("Oberteil" in code_classes[c] for c in cells.values())
            cards.append({
                "name": name, "session": f"S{session:02d}", "scene": scene_no,
                "day": "A" if si < (sessions * 3 + 4) // 5 else "B",
                "split": "dev" if si in dev_sessions else "test",
                "plate_L": left, "plate_R": right, "fill": fill, "n_occupied": len(cells),
                "layout": layout_text(cells), "quarter_turns": quarter_text(cells, quarter),
                "marker_id": spec["marker_id"] if has_oberteil else "",
                "cells": cells, "quarter": quarter,
            })
    return cards


def _runs(slots: list[str]) -> str:
    """``['L1','L2','L3','L5']`` -> ``'L1-L3,L5'`` (slots of one plate, sorted)."""
    nums = sorted(int(s[1:]) for s in slots)
    side, out, i = slots[0][0], [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{side}{nums[i]}" if i == j else f"{side}{nums[i]}-{side}{nums[j]}")
        i = j + 1
    return ",".join(out)


def layout_text(cells: dict[str, str]) -> str:
    """Layout in the syntax of ``cell_ground_truth.parse_layout``."""
    if not cells:
        return "L1-L9,R1-R9 empty"
    groups = []
    for code in ("U", "H", "HO", "UO"):
        parts = [_runs([s for s in SLOTS[side] if cells.get(s) == code])
                 for side in ("L", "R") if any(cells.get(s) == code for s in SLOTS[side])]
        if parts:
            groups.append(f"{','.join(parts)} {code}")
    return "; ".join(groups) + "; all other slots empty"


def quarter_text(cells: dict[str, str], quarter: dict[str, int]) -> str:
    """Operator hint ``slot:code/q`` (quarter turns counter-clockwise from the S02 reference).

    The Halter is symmetric, so ``H`` carries no turn. The ground truth finds the real
    orientation itself; this column only drives the balance.
    """
    return " ".join(f"{s}:{c}" if c == "H" else f"{s}:{c}/q{quarter[s]}"
                    for s, c in sorted(cells.items(), key=lambda kv: ALL_SLOTS.index(kv[0])))


def balance_report(cards: list[dict], step: str = "poty") -> dict:
    """Instance counts per class, per slot and per class orientation over all cards."""
    code_classes = STEPS[step]["codes"]
    cls = {c: 0 for c in CLASSES}
    slot = {s: 0 for s in ALL_SLOTS}
    ori = {(c, q): 0 for c in ("Unterteil", "Oberteil") for q in range(4)}
    codes = {code: 0 for code in code_classes}
    for card in cards:
        for s, code in card["cells"].items():
            slot[s] += 1
            codes[code] += 1
            for c in code_classes[code]:
                cls[c] += 1
                if c != "Halter":
                    ori[(c, card["quarter"][s])] += 1
    return {"classes": cls, "slots": slot, "orientations": ori, "codes": dict(codes)}


FIELDS = ["name", "session", "scene", "day", "split", "plate_L", "plate_R", "fill",
          "n_occupied", "layout", "quarter_turns", "marker_id"]


def write_csv(cards: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cards)


def write_markdown(cards: list[dict], path: Path, seed: int, step: str = "poty") -> None:
    rep = balance_report(cards, step)
    lines = [
        f"# Scene cards {STEPS[step]['label']} (seed {seed})", "",
        "Generated by `tools/make_scene_cards.py`. Rules: the camera never moves; parts sit fully "
        "seated in the pockets; an empty capture exists for every plate position and the plates "
        "are not moved afterwards; robot arm, hands and cables out of view; **no tape**. "
        f"At most {COPIES['Unterteil']} Unterteile, {COPIES['Halter']} Halter and {COPIES['Oberteil']} Oberteile per scene (the physical copies). "
        "Layout syntax as in `ground_truth_capture.md` §2.3. `q` = quarter turns counter-clockwise "
        f"from {STEPS[step]['reference']}; the Halter needs none.", "",
        "Per session: set the plates to the listed positions, take **two empty captures** "
        "(`S<nn>_empty_plates_01/02`) and register them in `plate_positions_<date>.json`, then "
        "the scenes. Set `SESS=benchmark_<date>` first.", "",
        f"Instances over all cards: {rep['classes']}; slots per code: {rep['codes']}.", ""]
    for session in sorted({c["session"] for c in cards}):
        scenes = [c for c in cards if c["session"] == session]
        first = scenes[0]
        lines += [f"## {session}: plates L={first['plate_L']}, R={first['plate_R']}, "
                  f"day {first['day']}, **{first['split']}**", ""]
        for c in scenes:
            lines += [f"- [ ] `{c['name']}` ({c['fill']}, {c['n_occupied']} parts): {c['layout']}",
                      f"  - `{c['quarter_turns'] or 'empty plates'}`"]
        lines += ["", "```bash"]
        for c in scenes:
            expect = f" {c['marker_id']}" if c["marker_id"] != "" else ""
            lines.append(f'tools/tk_shot.sh "$SESS" {c["name"]} "{c["layout"]}"{expect}')
        lines += ["```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="path without extension")
    parser.add_argument("--step", default="poty", choices=sorted(STEPS))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--first-session", type=int, default=3)
    parser.add_argument("--dev-sessions", default="0,3", help="0-based session indexes used as dev")
    parser.add_argument("--copies", default="8,8,8", help="physical copies: Unterteil,Halter,Oberteil")
    args = parser.parse_args(argv)
    dev = tuple(int(x) for x in args.dev_sessions.split(",") if x != "")
    copies = dict(zip(CLASSES, (int(x) for x in args.copies.split(","))))
    cards = make_cards(args.step, args.n, args.sessions, args.seed, args.first_session, dev, copies)
    write_csv(cards, args.out.with_suffix(".csv"))
    write_markdown(cards, args.out.with_suffix(".md"), args.seed, args.step)
    rep = balance_report(cards, args.step)
    print(f"{len(cards)} cards -> {args.out.with_suffix('.csv')} and .md")
    print("classes:", rep["classes"])
    print("codes:", rep["codes"])
    print("slots min/max:", min(rep["slots"].values()), max(rep["slots"].values()))
    print("orientations:", {f"{c}/q{q}": v for (c, q), v in rep["orientations"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
