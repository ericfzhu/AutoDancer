"""Small local HTTP dashboard for live symbolic worker observations."""

# ruff: noqa: E501

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import numpy as np

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoDancer Live Workers</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #090b10; color: #e8ecf2; }
header { position: sticky; top: 0; z-index: 2; padding: 14px 18px; background: #11151ddd;
  backdrop-filter: blur(8px); border-bottom: 1px solid #283142; }
h1 { margin: 0 0 8px; font-size: 19px; font-weight: 700; }
#summary { display: flex; flex-wrap: wrap; gap: 8px 18px; color: #aab4c3; font-size: 13px; }
#workers { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
  gap: 12px; padding: 12px; }
.worker { background: #121722; border: 1px solid #273044; border-radius: 9px; overflow: hidden; }
.worker.unhealthy { border-color: #d05252; }
.worker-head { padding: 9px 11px; display: flex; justify-content: space-between; gap: 8px;
  border-bottom: 1px solid #273044; font-size: 13px; }
.worker-id { font-weight: 700; }
.running { color: #67d391; } .dead, .aborted { color: #ef7676; } .won { color: #ffd166; }
.body { display: grid; grid-template-columns: minmax(252px, 1fr) 132px; gap: 9px; padding: 9px; }
canvas { width: 100%; aspect-ratio: 1; background: #07090d; image-rendering: pixelated;
  border: 1px solid #202838; border-radius: 4px; }
.stats { font-size: 12px; color: #aeb8c8; line-height: 1.55; overflow-wrap: anywhere; }
.stats b { color: #eef2f8; font-weight: 600; }
.events { color: #f2c879; margin-top: 7px; }
.legend { padding: 0 10px 10px; color: #7f8a9b; font-size: 11px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin: 0 3px 0 9px; }
.dot:first-child { margin-left: 0; }
@media (max-width: 480px) { .body { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header><h1>AutoDancer — live Bard workers</h1><div id="summary">Waiting for telemetry…</div></header>
<main id="workers"></main>
<script>
const ACTIONS = ["up","right","down","left","wait","bomb","item 1","item 2","throw","spell 1","spell 2"];
const TERRAIN = ["#080a0e", "#252b34", "#687382", "#8a5dd1"];
const workersNode = document.getElementById("workers");
const summaryNode = document.getElementById("summary");
let cards = new Map();

function fmt(value, digits=2) {
  if (value === null || value === undefined) return "—";
  return typeof value === "number" ? value.toFixed(digits).replace(/\.00$/, "") : value;
}
function ensureCard(worker) {
  if (cards.has(worker.instance_id)) return cards.get(worker.instance_id);
  const node = document.createElement("section"); node.className = "worker";
  node.innerHTML = `<div class="worker-head"><span class="worker-id"></span><span class="status"></span></div>
    <div class="body"><canvas width="420" height="420"></canvas><div class="stats"></div></div>
    <div class="legend"><span class="dot" style="background:#4ee4e4"></span>player
    <span class="dot" style="background:#ef6464"></span>enemy
    <span class="dot" style="background:#f7d154"></span>item
    <span class="dot" style="background:#ef922f"></span>trap
    <span class="dot" style="background:#8a5dd1"></span>stairs</div>`;
  workersNode.appendChild(node); cards.set(worker.instance_id, node); return node;
}
function drawGrid(canvas, grid) {
  if (!grid || !grid.length) return;
  const ctx = canvas.getContext("2d"); const size = grid.length; const cell = canvas.width / size;
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (let y=0; y<size; y++) for (let x=0; x<size; x++) {
    const v = grid[y][x], terrain=v[0]||0, actor=v[2]||0, health=v[4]||0,
      item=v[6]||0, trap=v[8]||0, visibility=v[9]||0;
    ctx.fillStyle = TERRAIN[terrain] || "#38404a"; ctx.fillRect(x*cell,y*cell,cell+0.5,cell+0.5);
    if (visibility === 0) { ctx.fillStyle="#050609e8"; ctx.fillRect(x*cell,y*cell,cell+0.5,cell+0.5); continue; }
    if (visibility === 1) { ctx.fillStyle="#0506098c"; ctx.fillRect(x*cell,y*cell,cell+0.5,cell+0.5); }
    if (trap) { ctx.fillStyle="#ef922f"; ctx.fillRect((x+.28)*cell,(y+.28)*cell,.44*cell,.44*cell); }
    if (item) { ctx.fillStyle="#f7d154"; ctx.beginPath(); ctx.arc((x+.5)*cell,(y+.5)*cell,.23*cell,0,Math.PI*2); ctx.fill(); }
    if (actor) {
      ctx.fillStyle = actor === 1 ? "#4ee4e4" : actor === 9 ? "#ed62d6" : "#ef6464";
      ctx.beginPath(); ctx.arc((x+.5)*cell,(y+.5)*cell,.35*cell,0,Math.PI*2); ctx.fill();
      if (health && cell >= 15) { ctx.fillStyle="#090b10"; ctx.font=`bold ${Math.max(8,cell*.43)}px monospace`; ctx.fillText(health,(x+.5)*cell,(y+.52)*cell); }
    }
  }
  ctx.strokeStyle="#b9c4d733"; ctx.lineWidth=1;
  ctx.strokeRect(10*cell,10*cell,cell,cell);
}
function updateCard(worker) {
  const node=ensureCard(worker), info=worker.info||{}, p=worker.player||[], h=worker.health||{};
  node.classList.toggle("unhealthy", h.healthy === false);
  node.querySelector(".worker-id").textContent = `${worker.instance_id} · PID ${h.pid ?? "—"}`;
  const status=info.episode_status||"starting", statusNode=node.querySelector(".status");
  statusNode.className=`status ${status}`; statusNode.textContent=status;
  drawGrid(node.querySelector("canvas"), worker.grid);
  const action = worker.action === null ? "—" : `${worker.action}: ${ACTIONS[worker.action] ?? "?"}`;
  const events=(info.raw_events||[]).map(e=>e.kind).join(", ") || "none";
  const rewardParts=Object.entries(info.reward_components||{}).filter(([,v])=>v!==0)
    .map(([k,v])=>`${k} ${Number(v).toFixed(3)}`).join(", ") || "turn pending";
  const inventory=(worker.inventory||[]).map((slot,index)=>({slot,index}))
    .filter(item=>item.slot.some(value=>value)).map(item=>`${item.index}:${item.slot.join("/")}`).join(", ") || "empty";
  node.querySelector(".stats").innerHTML = `<b>Zone/floor</b> ${info.zone??0}-${info.floor??0}<br>
    <b>HP</b> ${p[0]??0}/${p[1]??0}<br><b>Gold</b> ${p[2]??0} · <b>Bombs</b> ${p[9]??0}<br>
    <b>Inventory</b> ${inventory}<br>
    <b>Turn</b> ${info.turns??p[8]??0}<br><b>Action</b> ${action}<br><b>Reward</b> ${fmt(worker.reward,3)}<br>
    <b>Seed</b> ${info.seed??"—"}<br><b>Run</b> ${info.run_id??"—"}<br>
    <b>Latency</b> ${fmt(h.last_latency,3)}s<br><b>Restarts</b> ${h.restart_count??0}<br>
    <div class="events"><b>Reward parts</b> ${rewardParts}</div>
    <div class="events"><b>Events</b> ${events}</div>`;
}
async function refresh() {
  try {
    const response=await fetch(`/api/state?t=${Date.now()}`, {cache:"no-store"});
    const state=await response.json(), t=state.training||{};
    summaryNode.innerHTML = `<span>Status <b>${state.status}</b></span><span>Step <b>${t.global_step??0}</b></span>
      <span>Update <b>${t.updates??0}</b></span><span>Throughput <b>${fmt(t.steps_per_second)} steps/s</b></span>
      <span>Episodes <b>${t.episodes??0}</b></span><span>Kills <b>${t.enemy_kills??0}</b></span>
      <span>Pickups <b>${t.items_collected??0}</b></span><span>Restarts <b>${t.worker_restarts??0}</b></span>`;
    (state.workers||[]).forEach(updateCard);
  } catch (error) { summaryNode.textContent=`Dashboard disconnected: ${error}`; }
}
setInterval(refresh, 500); refresh();
</script>
</body></html>"""


@dataclass(slots=True)
class DashboardState:
    """Thread-safe, JSON-ready snapshot shared by trainer and HTTP handlers."""

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _status: str = field(default="starting", init=False)
    _training: dict[str, Any] = field(default_factory=dict, init=False)
    _workers: list[dict[str, Any]] = field(default_factory=list, init=False)
    _revision: int = field(default=0, init=False)

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._revision += 1

    def update_training(self, metrics: dict[str, Any]) -> None:
        with self._lock:
            self._training = dict(metrics)
            self._revision += 1

    def update_workers(
        self,
        worker_ids: list[str],
        observation: dict[str, np.ndarray],
        infos: list[dict[str, Any]],
        *,
        actions: np.ndarray | list[int] | None = None,
        rewards: np.ndarray | list[float] | None = None,
        health: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        workers = []
        for index, worker_id in enumerate(worker_ids):
            info = dict(infos[index])
            workers.append(
                {
                    "instance_id": worker_id,
                    "grid": observation["grid"][index].tolist(),
                    "player": observation["player"][index].tolist(),
                    "inventory": observation["inventory"][index].tolist(),
                    "action_mask": observation["action_mask"][index].tolist(),
                    "action": None if actions is None else int(actions[index]),
                    "reward": None if rewards is None else float(rewards[index]),
                    "info": info,
                    "health": dict((health or {}).get(worker_id, {})),
                }
            )
        with self._lock:
            self._workers = workers
            self._revision += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "revision": self._revision,
                "updated_at": time.time(),
                "status": self._status,
                "training": self._training,
                "workers": self._workers,
            }


class DashboardServer:
    def __init__(self, state: DashboardState, *, host: str = "127.0.0.1", port: int = 8765):
        self.state = state
        self.host = host
        self.requested_port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self.requested_port if self._server is None else int(self._server.server_port)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> DashboardServer:
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path
                if path == "/":
                    payload = DASHBOARD_HTML.encode("utf-8")
                    content_type = "text/html; charset=utf-8"
                elif path == "/api/state":
                    payload = json.dumps(state.snapshot(), separators=(",", ":")).encode()
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format: str, *args: object) -> None:
                del args

        self._server = ThreadingHTTPServer((self.host, self.requested_port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="autodancer-dashboard",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def __enter__(self) -> DashboardServer:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
