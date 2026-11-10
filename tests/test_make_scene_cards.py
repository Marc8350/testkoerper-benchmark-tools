import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import cell_ground_truth as cgt  # noqa: E402
import make_scene_cards as msc  # noqa: E402


def test_cards_are_deterministic_and_seed_dependent():
    a = msc.make_cards(seed=1)
    assert [c["layout"] for c in a] == [c["layout"] for c in msc.make_cards(seed=1)]
    assert [c["layout"] for c in a] != [c["layout"] for c in msc.make_cards(seed=2)]


def test_counts_sessions_splits_and_fill_classes():
    cards = msc.make_cards()
    assert len(cards) == 50
    assert Counter(c["session"] for c in cards) == {f"S{n:02d}": 10 for n in range(3, 8)}
    assert Counter(c["split"] for c in cards) == {"test": 30, "dev": 20}
    assert Counter(c["fill"] for c in cards) == {"full": 20, "medium": 20, "sparse": 8, "empty": 2}
    for s in {c["session"] for c in cards}:
        fills = {c["fill"] for c in cards if c["session"] == s}
        assert {"full", "sparse"} <= fills
    assert len({c["name"] for c in cards}) == 50


def test_layouts_parse_back_to_the_intended_occupancy():
    for card in msc.make_cards():
        assert cgt.parse_layout(card["layout"]) == card["cells"]
        assert cgt.parse_step_key(card["name"]) is not None
        assert len(card["cells"]) == card["n_occupied"]
        assert all(slot in msc.ALL_SLOTS for slot in card["cells"])


def test_marker_id_only_when_an_oberteil_is_present():
    for card in msc.make_cards():
        has_oberteil = any(code in ("HO", "UO") for code in card["cells"].values())
        assert (card["marker_id"] == 1) == has_oberteil


def test_balance_is_close_to_uniform():
    rep = msc.balance_report(msc.make_cards())
    classes = list(rep["classes"].values())
    assert max(classes) - min(classes) <= 0.1 * (sum(classes) / 3)
    slots = list(rep["slots"].values())
    assert max(slots) - min(slots) <= 0.35 * (sum(slots) / 18)
    for cls in ("Unterteil", "Oberteil"):
        per_turn = [rep["orientations"][(cls, q)] for q in range(4)]
        assert max(per_turn) - min(per_turn) <= 0.25 * (sum(per_turn) / 4) + 1


def test_runs_compress_consecutive_slots():
    assert msc._runs(["L1", "L2", "L3", "L5"]) == "L1-L3,L5"
    assert msc._runs(["R9"]) == "R9"


def test_no_scene_needs_more_copies_than_exist():
    code_classes = msc.STEPS["poty"]["codes"]
    for card in msc.make_cards():
        used = Counter(c for code in card["cells"].values() for c in code_classes[code])
        assert all(n <= msc.COPIES[c] for c, n in used.items()), (card["name"], used)
        assert card["n_occupied"] <= msc.COPIES["Unterteil"] + msc.COPIES["Halter"]


def test_states_are_balanced_too():
    codes = list(msc.balance_report(msc.make_cards())["codes"].values())
    assert max(codes) - min(codes) <= 0.15 * (sum(codes) / len(codes))
