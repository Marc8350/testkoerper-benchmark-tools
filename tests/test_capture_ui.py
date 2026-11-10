import json
import sys
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import capture_feed as cf  # noqa: E402
import capture_ui as ui  # noqa: E402


def _server(tmp_path, capture_fn, reject_fn=lambda name, card: None):
    cards = cf.load_cards(cf.CARDS, "S03")
    app = ui.App(cards, tmp_path / "state.json", "benchmark_test", capture_fn=capture_fn, preview_fn=lambda name: None, reject_fn=reject_fn)
    guide = tmp_path / "guide"
    guide.mkdir()
    (guide / f"{cards[0]['name']}.png").write_bytes(b"png")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler(app, guide))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return app, httpd


def _call(httpd, method, path):
    conn = HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    conn.request(method, path)
    response = conn.getresponse()
    return response.status, response.read()


def _wait_idle(app, seconds=5):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not app.busy:
            return
        time.sleep(0.02)
    raise AssertionError("job still busy")


def test_page_state_and_guide_image(tmp_path):
    app, httpd = _server(tmp_path, lambda *a: None)
    try:
        status, body = _call(httpd, "GET", "/")
        assert status == 200 and b"Aufnahme ausl" in body
        state = json.loads(_call(httpd, "GET", "/api/state")[1])
        assert state["total"] == 10 and state["index"] == 0 and state["card"]["name"] == "S03_fill_poty-c_01"
        assert _call(httpd, "GET", "/guide/S03_fill_poty-c_01.png") == (200, b"png")
        assert _call(httpd, "GET", "/guide/../../etc/passwd")[0] == 404
    finally:
        httpd.shutdown()


def test_accepted_capture_advances_and_rejected_does_not(tmp_path):
    verdicts = iter([
        {"found": True, "rejected": True, "reasons": ["x"], "layout_mismatch": {"L1": "?"}},
        {"found": True, "rejected": False, "reasons": [], "layout_mismatch": {}},
    ])
    seen = []

    def fake_capture(card, attempt, sess_dir, progress):
        progress("Aufnahme auf dem Jetson")
        seen.append((card["name"], attempt))
        return cf.attempt_name(card, attempt), next(verdicts)

    app, httpd = _server(tmp_path, fake_capture)
    try:
        assert json.loads(_call(httpd, "POST", "/api/capture")[1]) == {"started": True}
        _wait_idle(app)
        state = json.loads(_call(httpd, "GET", "/api/state")[1])
        assert state["index"] == 0 and state["attempt"] == 2 and state["last"]["outcome"] == "rejected"
        _call(httpd, "POST", "/api/capture")
        _wait_idle(app)
        state = json.loads(_call(httpd, "GET", "/api/state")[1])
        assert state["index"] == 1 and state["attempt"] == 1 and state["last"]["outcome"] == "accepted"
        assert seen == [("S03_fill_poty-c_01", 1), ("S03_fill_poty-c_01", 2)]
    finally:
        httpd.shutdown()


def test_errors_are_reported_and_nothing_advances(tmp_path):
    def broken(card, attempt, sess_dir, progress):
        raise RuntimeError("camera busy")

    app, httpd = _server(tmp_path, broken)
    try:
        _call(httpd, "POST", "/api/capture")
        _wait_idle(app)
        state = json.loads(_call(httpd, "GET", "/api/state")[1])
        assert state["error"] == "camera busy" and state["index"] == 0 and state["attempt"] == 1
    finally:
        httpd.shutdown()


def test_skip_and_select(tmp_path):
    app, httpd = _server(tmp_path, lambda *a: None)
    try:
        _call(httpd, "POST", "/api/skip")
        assert json.loads(_call(httpd, "GET", "/api/state")[1])["index"] == 1
        _call(httpd, "POST", "/api/select?index=4")
        assert json.loads(_call(httpd, "GET", "/api/state")[1])["index"] == 4
    finally:
        httpd.shutdown()


def test_stored_capture_advances(tmp_path):
    def fake_capture(card, attempt, sess_dir, progress):
        return cf.attempt_name(card, attempt), dict(cf.GT_SKIPPED)

    app, httpd = _server(tmp_path, fake_capture)
    try:
        _call(httpd, "POST", "/api/capture")
        _wait_idle(app)
        state = json.loads(_call(httpd, "GET", "/api/state")[1])
        assert state["index"] == 1 and state["last"]["outcome"] == "stored" and state["run_gt"] is False
    finally:
        httpd.shutdown()


def test_session_ground_truth_runs_in_the_background(tmp_path):
    cards = cf.load_cards(cf.CARDS, "S03")
    app = ui.App(cards, tmp_path / "state.json", "benchmark_test",
                 gt_fn=lambda: "S03_fill_poty-c_02: positions {'L': 'P1'}, 4 parts, ok")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler(app, tmp_path))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert json.loads(_call(httpd, "POST", "/api/gt")[1]) == {"started": True}
        _wait_idle(app)
        state = json.loads(_call(httpd, "GET", "/api/state")[1])
        assert state["gt_result"] == [{"name": "S03_fill_poty-c_02", "parts": 4, "verdict": "accepted"}]
    finally:
        httpd.shutdown()


def test_redo_gets_a_new_name_and_gt_does_not_reopen(tmp_path):
    def fake_capture(card, attempt, sess_dir, progress):
        return cf.attempt_name(card, attempt), dict(cf.GT_SKIPPED)

    cards = cf.load_cards(cf.CARDS, "S03")
    app = ui.App(cards, tmp_path / "state.json", "benchmark_test", capture_fn=fake_capture, preview_fn=lambda n: None,
                 gt_fn=lambda: "S03_fill_poty-c_01: positions {'L': 'P1'}, 4 parts, REJECTED\n")
    for _ in range(2):
        app.start_capture()
        _wait_idle(app)
    app.start_gt()
    _wait_idle(app)
    state = cf.load_state(tmp_path / "state.json")
    assert state["index"] == 2                                  # tool flags are for the human review, no auto redo
    app.select(0)
    state = cf.load_state(tmp_path / "state.json")
    assert state["index"] == 0 and state["attempt"] == 2
    app.start_capture()
    _wait_idle(app)
    state = cf.load_state(tmp_path / "state.json")
    assert state["history"][-1]["name"] == "S03_fill_poty-c_101" and state["index"] == 2


def test_scene_projection_follows_the_current_card_and_never_blocks(tmp_path):
    shown = []

    def fake_capture(card, attempt, sess_dir, progress):
        return cf.attempt_name(card, attempt), dict(cf.GT_SKIPPED)

    def project(name):
        shown.append(name)
        if name.endswith("_03"):
            raise RuntimeError("camera busy")

    cards = cf.load_cards(cf.CARDS, "S03")
    app = ui.App(cards, tmp_path / "state.json", "benchmark_test", capture_fn=fake_capture, preview_fn=lambda n: None,
                 project_fn=project)
    app.project_current()
    app.start_capture()
    _wait_idle(app)
    app.skip()
    app.select(5)
    deadline = time.time() + 3
    while len(shown) < 4 and time.time() < deadline:
        time.sleep(0.02)
    assert sorted(shown) == ["S03_fill_poty-c_01", "S03_fill_poty-c_02", "S03_fill_poty-c_03", "S03_fill_poty-c_06"]
    state = app.snapshot()
    assert state["project"] is True and isinstance(state["project_note"], str)


def test_retake_withdraws_the_last_capture_and_reopens_its_scene(tmp_path):
    rejected = []
    stored = {"found": True, "rejected": False, "reasons": [], "layout_mismatch": {}}
    app, httpd = _server(tmp_path, lambda card, attempt, sess, progress: (cf.attempt_name(card, attempt), stored),
                         reject_fn=lambda name, card: rejected.append(name))
    try:
        assert json.loads(_call(httpd, "GET", "/api/state")[1])["can_retake"] is False
        assert json.loads(_call(httpd, "POST", "/api/retake")[1]) == {"started": False}
        _call(httpd, "POST", "/api/capture")
        _wait_idle(app)
        state = json.loads(_call(httpd, "GET", "/api/state")[1])
        assert state["index"] == 1 and state["can_retake"] and state["retake_name"] == "S03_fill_poty-c_01"
        assert json.loads(_call(httpd, "POST", "/api/retake")[1]) == {"started": True}
        state = json.loads(_call(httpd, "GET", "/api/state")[1])
        assert state["index"] == 0 and state["attempt"] == 2 and not state["can_retake"] and state["last"] is None
        assert rejected == ["S03_fill_poty-c_01"]
        _call(httpd, "POST", "/api/capture")
        _wait_idle(app)
        state = json.loads(_call(httpd, "GET", "/api/state")[1])
        assert state["index"] == 1 and state["retake_name"] == "S03_fill_poty-c_101"
    finally:
        httpd.shutdown()


def test_failed_reject_is_reported(tmp_path):
    def broken(name, card):
        raise RuntimeError("ssh down")

    stored = {"found": True, "rejected": False, "reasons": [], "layout_mismatch": {}}
    app, httpd = _server(tmp_path, lambda card, attempt, sess, progress: (cf.attempt_name(card, attempt), stored), reject_fn=broken)
    try:
        _call(httpd, "POST", "/api/capture")
        _wait_idle(app)
        _call(httpd, "POST", "/api/retake")
        assert "ssh down" in json.loads(_call(httpd, "GET", "/api/state")[1])["error"]
    finally:
        httpd.shutdown()
