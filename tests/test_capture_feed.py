import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import capture_feed as cf  # noqa: E402


def _cards():
    return cf.load_cards(cf.CARDS, "S03")


def test_cards_of_a_session_are_ordered():
    cards = _cards()
    assert [int(c["scene"]) for c in cards] == list(range(1, 11))
    assert cf.load_cards(cf.CARDS, "S99") == []


def test_attempt_names_stay_parsable_and_unique():
    card = _cards()[0]
    assert cf.attempt_name(card, 1) == "S03_fill_poty-c_01"
    assert cf.attempt_name(card, 2) == "S03_fill_poty-c_101"
    assert cf.attempt_name(_cards()[9], 3) == "S03_fill_poty-c_210"
    import cell_ground_truth as cgt
    assert cgt.parse_step_key(cf.attempt_name(card, 2)) is not None


def test_render_card_shows_every_part_once():
    for card in _cards():
        text = cf.render_card(card)
        assert card["name"] in text and card["layout"] in text
        occupied = sum(1 for _ in cf.parse_quarters(card["quarter_turns"]))
        assert occupied == int(card["n_occupied"])


def test_parse_quarters():
    assert cf.parse_quarters("L1:UO/q3 R2:H") == {"L1": ("UO", "q3"), "R2": ("H", "")}
    assert cf.parse_quarters("") == {}


def test_log_rows_use_the_card_layout_and_plate_positions():
    card = _cards()[0]
    row = cf.review_row("S03_fill_poty-c_01", card, "2026-10-06T12:00:00Z")
    assert row.startswith("S03_fill_poty-c_01.zdf,accept,Einsetzen_(PO_TY),P1,P2,")
    assert card["layout"] in row
    assert cf.capture_row("S03_fill_poty-c_01", card, "t").startswith("S03_fill_poty-c_01.zdf,fill,")


def test_summarize_reads_rejection_and_residuals():
    summary = {"captures": {"x": {"rejected": True, "rejection_reasons": ["depth residual"],
                                  "parts_by_type": {"U": {"depth_residual_median_mm": 0.4, "flagged": []},
                                                    "H": {"depth_residual_median_mm": 3.1, "flagged": ["residual"]}}}}}
    out = cf.summarize({"layout_mismatch": {"L1": "found H, card U"}}, summary, "x")
    assert out["rejected"] and out["reasons"] == ["depth residual"]
    assert out["layout_mismatch"] == {"L1": "found H, card U"}
    assert out["depth_residual_mm"] == {"U": 0.4, "H": 3.1} and out["flagged"] == {"H": ["residual"]}
    assert cf.summarize(None, None, "x")["rejected"] is False


def test_state_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    assert cf.load_state(path) == {"index": 0, "attempt": 1, "history": []}
    cf.save_state(path, {"index": 3, "attempt": 2, "history": []})
    assert cf.load_state(path)["index"] == 3


def test_classify_separates_errors_from_rejections():
    ok = {"found": True, "rejected": False, "layout_mismatch": {}}
    assert cf.classify(ok) == "accepted"
    assert cf.classify({**ok, "rejected": True}) == "rejected"
    assert cf.classify({**ok, "layout_mismatch": {"L1": "x"}}) == "rejected"
    assert cf.classify({**ok, "found": False}) == "error"


def test_skipped_ground_truth_counts_as_stored():
    assert cf.classify(dict(cf.GT_SKIPPED)) == "stored"


def test_capture_is_copied_into_the_review_tool_even_without_ground_truth(monkeypatch):
    commands = []

    def fake_ssh(host, command, check=True, timeout=600):
        commands.append((host, command))
        return "a" * 64 + "  file"

    monkeypatch.setattr(cf, "_ssh", fake_ssh)
    monkeypatch.setattr(cf, "_stream", lambda *args: None)
    card = cf.load_cards(cf.CARDS, "S03")[0]
    name, verdict = cf.process_capture("S03_auto_001", card, 1, "benchmark_test", run_gt=False)
    assert name == "S03_fill_poty-c_01" and verdict["gt_skipped"] is True
    copy = [c for _, c in commands if "runs/benchmark_test/images" in c and "cp -n" in c]
    assert copy and "data/samples/benchmark_test/S03_fill_poty-c_01_rgb.png" in copy[0]
    assert not any("cell_ground_truth.py run" in c for _, c in commands)


def test_summarize_gt_output_reads_capture_lines():
    text = ("some warning\nS03_fill_poty-c_02: positions {'L': 'P1', 'R': 'P2'}, 4 parts, ok\n"
            "S03_fill_poty-c_101: positions {'L': 'P1', 'R': 'P2'}, 23 parts, REJECTED\n    R1 Einlegen: flag\n")
    assert cf.summarize_gt_output(text) == [{"name": "S03_fill_poty-c_02", "parts": 4, "verdict": "accepted"},
                                            {"name": "S03_fill_poty-c_101", "parts": 23, "verdict": "rejected"}]


def test_redo_uses_unused_attempt_and_first_open_card():
    cards = _cards()
    history = [{"name": "S03_fill_poty-c_01", "card": "S03_fill_poty-c_01", "attempt": 1, "outcome": "accepted"},
               {"name": "S03_fill_poty-c_02", "card": "S03_fill_poty-c_02", "attempt": 1, "outcome": "rejected"},
               {"name": "S03_fill_poty-c_102", "card": "S03_fill_poty-c_02", "attempt": 2, "outcome": "rejected"}]
    assert cf.next_attempt(cards[0], history) == 2 and cf.next_attempt(cards[2], history) == 1
    assert cf.next_attempt(cards[1], history) == 3
    assert cf.first_open(cards, history) == 1
    assert cf.first_open(cards, history + [{"card": cards[1]["name"], "outcome": "stored"}]) == 2


def test_reject_row_withdraws_the_capture():
    card = cf.load_cards(cf.CARDS, "S03")[0]
    row = cf.reject_row("S03_fill_poty-c_01", card, "2026-10-28T10:00:00Z", "retaken by the operator")
    assert row.startswith("S03_fill_poty-c_01.zdf,reject,") and row.endswith(",2026-10-28T10:00:00Z")
