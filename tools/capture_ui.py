"""Capture tool with a manual trigger: shows the scene card as an Isaac render, you press the button.

Runs on the Mac and serves a page on http://localhost:8770::

    python3 tools/capture_ui.py --session S03 --sess-dir benchmark_2026-10-06

The page shows the current scene (Isaac Sim render from ``scene_card_renders/``, the two plate
grids, the live status) and a button. A press captures on the Jetson (the autocapture must be
stopped), streams the ZDF to the workstation, converts it, computes the ground truth and shows
the tool verdict next to the capture preview. An accepted scene advances the card; a rejected one
stays and the next press counts as a new attempt. The human verdicts stay in the review tool.
"""
from __future__ import annotations

import argparse
import functools
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import capture_feed as cf

GUIDE_DIR = Path(__file__).resolve().parent.parent / "docs/researcheffort/scene_card_renders"


class App:
    """Session state plus the single background capture job (thread-safe)."""

    def __init__(self, cards: list[dict], state_path: Path, sess_dir: str, capture_fn=None, preview_fn=None,
                 run_gt: bool = False, gt_fn=None, project_fn=None, reject_fn=None):
        self.reject_fn = reject_fn or (lambda name, card: cf.reject_capture(name, card, sess_dir))
        self.cards, self.state_path, self.sess_dir, self.run_gt = cards, state_path, sess_dir, run_gt
        self.project_fn = project_fn                      # called with the card name; None = no projection
        self.project_note = ""
        self.gt_fn = gt_fn or (lambda: cf.run_session_gt(sess_dir))
        self.gt_result: list[dict] | None = None
        self.capture_fn = capture_fn or functools.partial(cf.capture_and_process, run_gt=run_gt)
        self.preview_fn = preview_fn or (lambda name: cf.fetch_preview(name, sess_dir))
        self.lock = threading.Lock()
        self.busy = False
        self.step = ""
        self.last: dict | None = None
        self.error = ""

    # -- views --
    def snapshot(self) -> dict:
        with self.lock:
            state = cf.load_state(self.state_path)
            index = min(state["index"], len(self.cards) - 1)
            card = self.cards[index]
            return {
                "session": card["session"], "sess_dir": self.sess_dir, "total": len(self.cards),
                "index": state["index"], "attempt": state["attempt"], "done": state["index"] >= len(self.cards),
                "card": {**card, "cells": {s: list(v) for s, v in cf.parse_quarters(card["quarter_turns"]).items()}},
                "busy": self.busy, "step": self.step, "last": self.last, "error": self.error, "run_gt": self.run_gt,
                "gt_result": self.gt_result, "project": self.project_fn is not None, "project_note": self.project_note,
                "history": state["history"][-12:], "can_retake": self._last_stored(state) is not None,
                "retake_name": (self._last_stored(state) or {}).get("name", ""),
                "names": [{"name": c["name"], "scene": c["scene"], "fill": c["fill"]} for c in self.cards],
            }

    @staticmethod
    def _last_stored(state: dict) -> dict | None:
        return next((h for h in reversed(state["history"]) if h.get("name") and h.get("outcome") in ("accepted", "stored")), None)

    def retake(self) -> bool:
        """Withdraw the last stored capture (review log) and make its scene the current one again, as a new attempt."""
        with self.lock:
            state = cf.load_state(self.state_path)
            entry = self._last_stored(state)
            if self.busy or entry is None:
                return False
            card = next(c for c in self.cards if c["name"] == entry["card"])
            entry["outcome"] = "retaken"
            state["index"] = self.cards.index(card)
            state["attempt"] = cf.next_attempt(card, state["history"])
            cf.save_state(self.state_path, state)
            self.last, self.error = None, ""
        try:
            self.reject_fn(entry["name"], card)
        except Exception as exc:                          # the capture would still count as accepted in the log
            with self.lock:
                self.error = f"{entry['name']} konnte nicht im review_log verworfen werden: {exc}"
        self.project_current()
        return True

    # -- projection --
    def project_current(self) -> None:
        """Project the current scene card in the background (a failed projection never blocks the capture)."""
        if self.project_fn is None:
            return
        state = cf.load_state(self.state_path)
        if state["index"] >= len(self.cards):
            return
        name = self.cards[state["index"]]["name"]
        threading.Thread(target=self._project, args=(name,), daemon=True).start()

    def _project(self, name: str) -> None:
        try:
            self.project_fn(name)
            note = f"projiziert: {name}"
        except Exception as exc:
            note = f"Projektion fehlgeschlagen: {exc}"
        with self.lock:
            self.project_note = note

    # -- actions --
    def start_capture(self) -> bool:
        with self.lock:
            state = cf.load_state(self.state_path)
            if self.busy or state["index"] >= len(self.cards):
                return False
            self.busy, self.step, self.error = True, "Start", ""
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _progress(self, message: str) -> None:
        with self.lock:
            self.step = message

    def _run(self) -> None:
        try:
            state = cf.load_state(self.state_path)
            card, attempt = self.cards[state["index"]], state["attempt"]
            name, verdict = self.capture_fn(card, attempt, self.sess_dir, self._progress)
            outcome = cf.classify(verdict)
            preview = self.preview_fn(name)
            with self.lock:
                state = cf.load_state(self.state_path)
                state["history"].append({"name": name, "card": card["name"], "attempt": attempt, "outcome": outcome})
                if outcome in ("accepted", "stored"):
                    state["index"] = cf.first_open(self.cards, state["history"])
                    state["attempt"] = (cf.next_attempt(self.cards[state["index"]], state["history"])
                                        if state["index"] < len(self.cards) else 1)
                else:
                    state["attempt"] += 1
                cf.save_state(self.state_path, state)
                self.last = {"name": name, "card": card["name"], "outcome": outcome, "verdict": verdict,
                             "preview": bool(preview)}
        except Exception as exc:                          # shown in the page, nothing is advanced
            with self.lock:
                self.error = str(exc)
        finally:
            with self.lock:
                self.busy, self.step = False, ""
            self.project_current()                        # the capture ended the projection: show the (next) scene again

    def start_gt(self) -> bool:
        """Ground truth for the whole session in the background (one job at a time)."""
        with self.lock:
            if self.busy:
                return False
            self.busy, self.step, self.error, self.gt_result = True, "Ground Truth für die Session läuft", "", None
        threading.Thread(target=self._run_gt, daemon=True).start()
        return True

    def _run_gt(self) -> None:
        try:
            entries = cf.summarize_gt_output(self.gt_fn())
            with self.lock:
                self.gt_result = entries
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            with self.lock:
                self.busy, self.step = False, ""

    def skip(self) -> None:
        with self.lock:
            state = cf.load_state(self.state_path)
            if state["index"] < len(self.cards):
                state["history"].append({"card": self.cards[state["index"]]["name"], "outcome": "skipped"})
                state["index"], state["attempt"] = state["index"] + 1, 1
                cf.save_state(self.state_path, state)
        self.project_current()

    def select(self, index: int) -> None:
        with self.lock:
            state = cf.load_state(self.state_path)
            if not self.busy and 0 <= index < len(self.cards):
                state["index"], state["attempt"] = index, cf.next_attempt(self.cards[index], state["history"])
                cf.save_state(self.state_path, state)
        self.project_current()

    def dismiss(self) -> None:
        with self.lock:
            self.last, self.error = None, ""


PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Aufnahme @LABEL@</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f4f5f2;--card:#fff;--ink:#1b1f1a;--mute:#5d665b;--line:#d9ddd5;--ok:#1f7a3d;--bad:#b3261e;--warn:#9a6b00;--acc:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#161815;--card:#20231f;--ink:#e9ece6;--mute:#a1aa9e;--line:#363b34;--ok:#5ac27a;--bad:#ff8a80;--warn:#e6b84d;--acc:#8ab4ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:12px 20px;border-bottom:1px solid var(--line);background:var(--card);flex-wrap:wrap}
header h1{font-size:18px;margin:0}.pill{padding:2px 10px;border-radius:99px;border:1px solid var(--line);color:var(--mute);font-size:13px}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;padding:16px 20px}
@media (max-width:1100px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
img{max-width:100%;border-radius:6px;display:block}img.rot{transform:rotate(180deg)}h2{font-size:15px;margin:0 0 8px}
.grids{display:grid;grid-template-columns:1fr 1fr;gap:12px}table{border-collapse:collapse;width:100%}
td{border:1px solid var(--line);text-align:center;height:46px;font-size:13px;font-weight:600}
td.e{color:var(--mute);font-weight:400}.U{background:#d7ecff}.H{background:#ffe7c2}.HO{background:#f6d4f6}.UO{background:#d4f3dd}
@media (prefers-color-scheme:dark){.U{background:#1f3a55}.H{background:#55401c}.HO{background:#4f2c4f}.UO{background:#1f4a2c}}
button{font:inherit;padding:14px 22px;border-radius:10px;border:0;background:var(--acc);color:#fff;font-weight:600;cursor:pointer;width:100%;font-size:17px}
button:disabled{opacity:.5;cursor:wait}button.s{background:transparent;color:var(--ink);border:1px solid var(--line);width:auto;padding:6px 12px;font-size:13px;font-weight:500}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.mute{color:var(--mute)}code{font-size:12px}
.hist span{display:inline-block;margin:2px 6px 2px 0;font-size:12px}
</style></head><body>
<header><h1>Aufnahme @LABEL@</h1><span class="pill" id="sess"></span><span class="pill" id="prog"></span><span class="pill" id="plates"></span>
<select id="jump" class="pill"></select><a class="pill" href="http://localhost:8767" target="_blank">Annotationstool</a></header>
<main>
<section class="card"><h2 id="title"></h2><img id="render" alt="Isaac-Sim-Render der Szene"><p class="mute" id="layout"></p></section>
<section>
 <div class="card"><h2>Raster aus deiner Sicht vom Roboter (um 180° gedreht, Slot 1 ist unten rechts)</h2><div class="grids"><div><b id="tr">Platte R</b><table id="gr"></table></div><div><b id="tl">Platte L</b><table id="gl"></table></div></div>
 <p class="mute" style="margin:8px 0 0">U Unterteil · H Halter · HO Oberteil auf Halter · UO Oberteil im Unterteil · qN = N Vierteldrehungen gegen den Uhrzeigersinn, Referenz: @REFERENCE@ (im Kamerabild; der Render ist wie deine Ansicht gedreht)</p></div>
 <div class="card" style="margin-top:12px"><button id="go">Aufnahme auslösen</button><p id="status" class="mute" style="margin:10px 0 0"></p>
  <p style="margin:10px 0 0"><button class="s" id="skip">Szene überspringen</button> <button class="s" id="retake" hidden>Letzte Aufnahme wiederholen</button> <button class="s" id="proj">Szene projizieren</button> <span class="mute" id="projnote"></span></p></div>
 <div class="card" style="margin-top:12px" id="resultbox" hidden><h2 id="rtitle"></h2><div id="rbody"></div><img id="preview" class="rot" alt="Aufnahme (um 180° gedreht)" style="margin-top:8px"><p><button class="s" id="dismiss">Ausblenden</button></p></div>
 <div class="card" style="margin-top:12px" id="gtbox" hidden><h2>Ground Truth der Session</h2><p class="mute" id="gtintro"></p><button id="gtgo">Ground Truth berechnen</button><div id="gtres" style="margin-top:10px"></div></div>
 <div class="card hist" style="margin-top:12px"><h2>Verlauf</h2><div id="hist" class="mute">noch keine Aufnahme</div></div>
</section></main>
<script>
const $=s=>document.querySelector(s);let cur=null,jumpBuilt=false;
function grid(side,cells){let h='';for(let r=2;r>=0;r--){h+='<tr>';for(let c=2;c>=0;c--){const k=side+(r*3+c+1),v=cells[k];
 h+=v?`<td class="${v[0]}">${v[0]}${v[1]?'<br>'+v[1]:''}</td>`:'<td class="e">·</td>';}h+='</tr>';}return h;}
async function api(p,m){const r=await fetch(p,{method:m||'GET'});return r.json();}
function render(s){cur=s;$('#retake').hidden=!s.can_retake||s.busy;$('#retake').textContent='Letzte Aufnahme wiederholen ('+s.retake_name+')';$('#proj').hidden=!s.project;$('#projnote').textContent=s.project?(s.project_note||''):'';const c=s.card;$('#sess').textContent=s.session+' · '+c.day+' · '+c.split;$('#prog').textContent=Math.min(s.index+1,s.total)+' / '+s.total+(s.attempt>1?'  (Versuch '+s.attempt+')':'');
 $('#plates').textContent='links '+c.plate_L+' · rechts '+c.plate_R;$('#title').textContent=s.done?'Session fertig':c.name+' ('+c.fill+', '+c.n_occupied+' Teile)';
 const img=$('#render');img.onerror=()=>{img.hidden=true};img.onload=()=>{img.hidden=false};const want='/guide/'+c.name+'.png';if(!img.src.endsWith(want))img.src=want;$('#layout').textContent='Layout: '+c.layout;
 $('#gl').innerHTML=grid('L',c.cells);$('#gr').innerHTML=grid('R',c.cells);$('#tl').textContent='Platte L ('+c.plate_L+', P1 = unten)';$('#tr').textContent='Platte R ('+c.plate_R+')';
 $('#go').disabled=s.busy||s.done;$('#go').textContent=s.busy?'… '+(s.step||'läuft'):(s.done?'Alle Szenen aufgenommen':'Aufnahme auslösen');
 $('#status').innerHTML=s.error?'<span class="bad">Fehler: '+s.error+'</span>':(s.busy?'<span class="warn">'+(s.done?s.step:'Bitte aus dem Bild treten und nichts bewegen.')+'</span>':(s.done?'Session fertig.':'Szene aufbauen, Roboterarm und Hände aus dem Bild, dann auslösen.')+(s.done?'':(s.run_gt?'':' (Ground Truth aus, wird später berechnet)')));
 if(!jumpBuilt){$('#jump').innerHTML=s.names.map((n,i)=>`<option value="${i}">${n.scene}. ${n.fill}</option>`).join('');jumpBuilt=true;}$('#jump').value=Math.min(s.index,s.total-1);
 $('#hist').innerHTML=s.history.length?s.history.map(h=>`<span class="${(h.outcome=='accepted'||h.outcome=='stored')?'ok':(h.outcome=='skipped'?'mute':'bad')}">${h.name||h.card}: ${h.outcome}</span>`).join(''):'noch keine Aufnahme';
 const l=s.last;$('#resultbox').hidden=!l;if(l){const v=l.verdict,oc=l.outcome;const good=oc=='accepted'||oc=='stored';$('#rtitle').innerHTML=`<span class="${good?'ok':'bad'}">${oc=='accepted'?'Akzeptiert':(oc=='stored'?'Gespeichert':(oc=='rejected'?'Vom Werkzeug abgelehnt':'Verarbeitungsfehler'))}</span> · ${l.name}`;
  let b='';const mm=Object.entries(v.layout_mismatch||{});if(mm.length)b+='<p class="bad">Layout weicht ab: '+mm.map(([k,x])=>k).join(', ')+'</p>';
  if((v.reasons||[]).length)b+='<p class="bad"><code>'+v.reasons.slice(0,5).join('<br>')+'</code></p>';
  const pa=v.plate_adjustment||{};if(Object.keys(pa).length)b+='<p class="warn">Plattenverschiebung: <code>'+JSON.stringify(pa)+'</code></p>';
  if(oc=='accepted')b+='<p class="mute">Das Annotationstool zeigt die Aufnahme jetzt. Nächste Szene ist geladen.</p>';if(oc=='stored')b+='<p class="mute">Ground Truth wird später für die ganze Session berechnet. Nächste Szene ist geladen. Prüfe in der Vorschau, ob der Aufbau stimmt.</p>';if(oc=='rejected')b+='<p>Szene prüfen und noch einmal auslösen (nächster Versuch).</p>';
  $('#rbody').innerHTML=b;const p=$('#preview');if(l.preview){p.hidden=false;if(p.dataset.name!==l.name){p.dataset.name=l.name;p.src='/preview/'+l.name+'.jpg';}}else p.hidden=true;}
 const gb=$('#gtbox');gb.hidden=!(s.done||s.gt_result);$('#gtintro').textContent=s.done?'Alle '+s.total+' Szenen aufgenommen. Jetzt die Ground Truth für alle Aufnahmen berechnen (etwa 3 s je Aufnahme), dann im Annotationstool prüfen.':'';
 $('#gtgo').disabled=s.busy;$('#gtgo').textContent=s.busy&&!s.gt_result?'… läuft':'Ground Truth berechnen';
 $('#gtres').innerHTML=s.gt_result?('<table>'+s.gt_result.map(e=>`<tr><td style="text-align:left">${e.name}</td><td>${e.parts} Teile</td><td class="${e.verdict=='accepted'?'ok':'bad'}">${e.verdict=='accepted'?'ok':'abgelehnt'}</td></tr>`).join('')+'</table><p><a href="http://localhost:8767" target="_blank">Im Annotationstool prüfen</a></p>'):'';}
async function tick(){try{render(await api('/api/state'));}catch(e){}setTimeout(tick,cur&&cur.busy?800:2500);}
$('#go').onclick=async()=>{$('#go').disabled=true;await api('/api/capture','POST');tick();};
$('#gtgo').onclick=async()=>{$('#gtgo').disabled=true;await api('/api/gt','POST');tick();};
$('#proj').onclick=async()=>{await api('/api/project','POST');};
$('#retake').onclick=async()=>{if(confirm('Aufnahme '+cur.retake_name+' verwerfen und die Szene noch einmal aufnehmen?')){await api('/api/retake','POST');tick();}};
$('#skip').onclick=async()=>{if(confirm('Diese Szene überspringen?')){await api('/api/skip','POST');tick();}};
$('#dismiss').onclick=async()=>{await api('/api/dismiss','POST');tick();};
$('#jump').onchange=async e=>{await api('/api/select?index='+e.target.value,'POST');tick();};
tick();
</script></body></html>"""


def make_handler(app: App, guide_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):                      # keep the terminal quiet
            return

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, ctype: str):
            if path.is_file():
                self._send(path.read_bytes(), ctype)
            else:
                self._send(b"not found", "text/plain", 404)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                spec = cf.STEPS[cf.STEP_KEY]
                page = PAGE.replace("@LABEL@", spec["label"]).replace("@REFERENCE@", spec["reference"])
                self._send(page.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._send(json.dumps(app.snapshot()).encode("utf-8"), "application/json")
            elif url.path.startswith("/guide/"):
                self._file(guide_dir / Path(unquote(url.path[len("/guide/"):])).name, "image/png")
            elif url.path.startswith("/preview/"):
                name = Path(unquote(url.path[len("/preview/"):])).stem
                self._file(cf.STATE_DIR / "previews" / app.sess_dir / f"{name}.jpg", "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path == "/api/capture":
                result = {"started": app.start_capture()}
            elif url.path == "/api/gt":
                result = {"started": app.start_gt()}
            elif url.path == "/api/project":
                app.project_current()
                result = {"ok": True}
            elif url.path == "/api/retake":
                result = {"started": app.retake()}
            elif url.path == "/api/skip":
                app.skip()
                result = {"ok": True}
            elif url.path == "/api/dismiss":
                app.dismiss()
                result = {"ok": True}
            elif url.path == "/api/select":
                query = dict(p.split("=") for p in url.query.split("&") if "=" in p)
                app.select(int(query.get("index", "0")))
                result = {"ok": True}
            else:
                return self._send(b"not found", "text/plain", 404)
            self._send(json.dumps(result).encode("utf-8"), "application/json")

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--session", default="S03")
    parser.add_argument("--sess-dir", required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--gt", action="store_true", help="compute the ground truth after every capture (slower)")
    parser.add_argument("--no-project", action="store_true", help="do not project the scene wireframes onto the table")
    parser.add_argument("--step", default="poty", choices=sorted(cf.STEPS))
    parser.add_argument("--calibration", help="slot calibration for the ground truth (default: the step's)")
    parser.add_argument("--positions", help="plate positions file for the ground truth (default: the step's)")
    args = parser.parse_args(argv)
    spec = cf.set_step(args.step)
    cf.CALIBRATION, cf.POSITIONS = args.calibration or cf.CALIBRATION, args.positions or cf.POSITIONS
    cards = cf.load_cards(cf.CARDS, args.session)
    if not cards:
        print(f"no cards for session {args.session}")
        return 1
    app = App(cards, cf.STATE_DIR / f"{args.sess_dir}_{args.session}.json", args.sess_dir, run_gt=args.gt,
              project_fn=None if args.no_project or not spec["projection"] else cf.project_card)
    app.project_current()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app, GUIDE_DIR))
    url = f"http://localhost:{args.port}"
    print(f"capture tool on {url} (session {args.session}, {len(cards)} scenes); Ctrl+C to stop")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
