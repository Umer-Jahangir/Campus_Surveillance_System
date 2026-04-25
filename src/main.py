"""
dashboard_server.py
===================
Flask + Socket.IO bridge that runs the OpenVINO multi-stream pipeline and
streams annotated JPEG frames, detection events, and performance stats to
the browser dashboard in real time.

No cv2.imshow — all video output is in the browser.

Usage:
    python main.py
Then open http://localhost:5000
"""

import base64
import ctypes
import json
import multiprocessing
import multiprocessing.shared_memory
import os
import sys
import time
import threading
from queue import Empty

import cv2
import numpy as np
import psutil
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO

# ── project imports ────────────────────────────────────────────────────────
from worker import decoder_worker, inference_worker
from monitor import monitor_worker
from utils import compute_fps, summarize_latency

# ── app setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "ov-dashboard-secret-2025"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=False, engineio_logger=False)

# ── frame / pipeline constants ─────────────────────────────────────────────
FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS = 256, 320, 3
FRAME_SHAPE  = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
FRAME_SIZE   = FRAME_HEIGHT * FRAME_WIDTH * FRAME_CHANNELS
JPEG_QUALITY = 70         # balance: quality vs bandwidth
MAX_ALERTS   = 200         # rolling alert history kept in memory

# ── global state ───────────────────────────────────────────────────────────
_lock = threading.Lock()

state = {
    "running":           False,
    "restarting":        False,   # True while pipeline hot-restarts
    "streams":           [],      # [{"url": str, "label": str}, ...]
    # live process handles
    "decoder_procs":     [],
    "inference_proc":    None,
    "monitor_proc":      None,
    # IPC
    "stop_event":        None,
    "inference_alive":   None,
    "meta_queues":       [],
    "result_queue":      None,
    "shared_cpu":        None,
    "shared_ram":        None,
    # shared memory
    "shm_objects":       [],
    "shm_names":         [],
    # per-stream statistics (indexed by stream id)
    "stats":             {},
}

# rolling alert list  [{stream_id, label, conf, track_id, ts}, ...]
alerts: list = []
alert_lock = threading.Lock()

# ── helpers ────────────────────────────────────────────────────────────────

def compute_cores_per_decoder(num_streams: int) -> int:
    total = psutil.cpu_count(logical=False) or 4
    reserve = max(2, total // 4)
    pool = total - reserve
    per = max(1, pool // num_streams)
    while (per * num_streams) > (total - 2) and per > 1:
        per -= 1
    return per


def annotate_frame(frame: np.ndarray, boxes, track_ids, labels, confs) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of frame."""
    out = frame.copy()
    if boxes is None:
        return out
    for bi, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        tid  = int(track_ids[bi]) if track_ids is not None else -1
        lbl  = labels[bi] if labels else "?"
        conf = float(confs[bi]) if confs is not None else 0.0
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 80), 2)
        tag = f"{lbl} {conf:.2f} #{tid}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (x1, max(y1-th-6, 0)), (x1+tw+4, y1), (0, 255, 80), -1)
        cv2.putText(out, tag, (x1+2, max(y1-4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return out


def frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode("utf-8")


def _stats_snapshot(s: dict) -> dict:
    m_avg, m_p95, m_max = summarize_latency(s["model_latencies"])
    p_avg, p_p95, p_max = summarize_latency(s["pipeline_latencies"])
    dur = 0.0
    if s["start_time"] and s["end_time"]:
        dur = s["end_time"] - s["start_time"]
    return {
        "fps":     round(s["live_fps"], 1),
        "frames":  s["frames"],
        "dropped": s.get("dropped", 0),
        "duration": round(dur, 1),
        "model": {
            "avg": round(m_avg, 1),
            "p95": round(m_p95, 1),
            "max": round(m_max, 1),
        },
        "e2e": {
            "avg": round(p_avg, 1),
            "p95": round(p_p95, 1),
            "max": round(p_max, 1),
        },
    }


# ── emit loop ──────────────────────────────────────────────────────────────

def _emit_loop():
    """
    Runs in a daemon thread.  Drains result_queue, encodes frames as JPEG,
    and emits Socket.IO events to all connected browsers.
    """
    last_sys_emit = time.monotonic()

    while True:
        with _lock:
            running = state["running"]
            rq      = state["result_queue"]
            stop_ev = state["stop_event"]

        if not running or rq is None:
            time.sleep(0.05)
            continue

        if stop_ev and stop_ev.is_set():
            time.sleep(0.1)
            continue

        # ── drain result queue ────────────────────────────────────────────
        try:
            for _ in range(48):
                msg = rq.get_nowait()
                if msg[0] != "frame":
                    continue

                (_, idx, frame, boxes, track_ids, classes, confs, labels,
                 capture_time, model_lat, e2e_lat) = msg

                with _lock:
                    s = state["stats"].get(idx)
                if s is None:
                    continue

                # update stats
                now_t = time.perf_counter()
                if s["start_time"] is None:
                    s["start_time"] = capture_time
                s["end_time"] = capture_time
                s["frames"] += 1

                if len(s["model_latencies"]) >= 1000:
                    s["model_latencies"].pop(0)
                s["model_latencies"].append(model_lat)

                if len(s["pipeline_latencies"]) >= 1000:
                    s["pipeline_latencies"].pop(0)
                s["pipeline_latencies"].append(e2e_lat)

                elapsed = now_t - s["last_fps_time"]
                if elapsed >= 1.0:
                    s["live_fps"] = (s["frames"] - s["last_fps_frames"]) / elapsed
                    s["last_fps_time"]   = now_t
                    s["last_fps_frames"] = s["frames"]

                # build detection payload + alert entries
                det_list = []
                if boxes is not None and len(boxes):
                    for bi in range(len(boxes)):
                        det_list.append({
                            "label":    labels[bi] if labels else "object",
                            "conf":     round(float(confs[bi]), 3) if confs is not None else 0.0,
                            "track_id": int(track_ids[bi]) if track_ids is not None else -1,
                            "box":      [int(v) for v in boxes[bi]],
                        })

                    with alert_lock:
                        stream_label = (state["streams"][idx]["label"]
                                        if idx < len(state["streams"]) else f"S{idx}")
                        for d in det_list:
                            alerts.append({
                                "stream_id":    idx,
                                "stream_label": stream_label,
                                "label":        d["label"],
                                "conf":         d["conf"],
                                "track_id":     d["track_id"],
                                "ts":           time.time(),
                            })
                        # trim
                        del alerts[:max(0, len(alerts) - MAX_ALERTS)]

                # annotate + encode
                annotated = annotate_frame(frame, boxes, track_ids, labels, confs)
                b64 = frame_to_b64(annotated)

                socketio.emit("frame_update", {
                    "stream_id":  idx,
                    "frame":      b64,
                    "detections": det_list,
                    "stats":      _stats_snapshot(s),
                })

                # dedicated alert event (only when objects detected)
                if det_list:
                    socketio.emit("detection_alert", {
                        "stream_id":    idx,
                        "stream_label": (state["streams"][idx]["label"]
                                         if idx < len(state["streams"]) else f"S{idx}"),
                        "detections":   det_list,
                        "ts":           time.time(),
                    })

        except Empty:
            pass

        # ── system stats (every 0.5 s) ────────────────────────────────────
        now_mono = time.monotonic()
        if now_mono - last_sys_emit >= 0.5:
            with _lock:
                cpu = state["shared_cpu"].value if state["shared_cpu"] else 0.0
                ram = state["shared_ram"].value if state["shared_ram"] else 0.0
            socketio.emit("system_stats", {
                "cpu": round(cpu, 1),
                "ram": round(ram, 1),
            })
            last_sys_emit = now_mono

        time.sleep(0.012)   # ~83 Hz poll; actual frame rate limited by source


threading.Thread(target=_emit_loop, daemon=True, name="emit-loop").start()


# ── pipeline lifecycle ─────────────────────────────────────────────────────

def _start_pipeline(streams: list) -> None:
    """Must be called while holding _lock."""
    num_streams = len(streams)
    stream_urls = [s["url"] for s in streams]
    cpp = compute_cores_per_decoder(num_streams)

    stop_event      = multiprocessing.Event()
    inference_alive = multiprocessing.Event()

    shm_objects, shm_names = [], []
    for _ in range(num_streams):
        sa = multiprocessing.shared_memory.SharedMemory(create=True, size=FRAME_SIZE)
        sb = multiprocessing.shared_memory.SharedMemory(create=True, size=FRAME_SIZE)
        shm_objects.append((sa, sb))
        shm_names.append((sa.name, sb.name))

    meta_queues  = [multiprocessing.Queue(maxsize=8) for _ in range(num_streams)]
    result_queue = multiprocessing.Queue(maxsize=512)
    shared_cpu   = multiprocessing.Value('d', 0.0)
    shared_ram   = multiprocessing.Value('d', 0.0)

    inf_proc = multiprocessing.Process(
        target=inference_worker,
        args=(num_streams, FRAME_SHAPE, shm_names,
              meta_queues, result_queue, stop_event,
              inference_alive, num_streams, cpp),
    )
    inf_proc.start()

    dec_procs = []
    for i, url in enumerate(stream_urls):
        p = multiprocessing.Process(
            target=decoder_worker,
            args=(i, url, shm_names[i], FRAME_SHAPE,
                  meta_queues[i], stop_event, inference_alive,
                  num_streams, cpp),
        )
        p.start()
        dec_procs.append(p)

    mon_proc = multiprocessing.Process(
        target=monitor_worker, args=(shared_cpu, shared_ram, stop_event)
    )
    mon_proc.start()

    per_stream_stats = {}
    for i in range(num_streams):
        per_stream_stats[i] = {
            "frames": 0, "dropped": 0,
            "model_latencies": [], "pipeline_latencies": [],
            "start_time": None, "end_time": None,
            "last_fps_time": time.perf_counter(),
            "last_fps_frames": 0, "live_fps": 0.0,
        }

    state.update({
        "running":        True,
        "restarting":     False,
        "decoder_procs":  dec_procs,
        "inference_proc": inf_proc,
        "monitor_proc":   mon_proc,
        "stop_event":     stop_event,
        "inference_alive":inference_alive,
        "meta_queues":    meta_queues,
        "result_queue":   result_queue,
        "shared_cpu":     shared_cpu,
        "shared_ram":     shared_ram,
        "shm_objects":    shm_objects,
        "shm_names":      shm_names,
        "stats":          per_stream_stats,
    })


def _stop_pipeline() -> None:
    """Gracefully shuts down all processes and frees shared memory."""
    if not state["running"]:
        return

    state["stop_event"].set()

    all_procs = (state["decoder_procs"]
                 + [state["inference_proc"], state["monitor_proc"]])
    for p in all_procs:
        if not p:
            continue
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=3)
        if p.is_alive():
            p.kill()
            p.join(timeout=2)

    for sa, sb in state["shm_objects"]:
        for shm in (sa, sb):
            try: shm.close()
            except Exception: pass
            try: shm.unlink()
            except Exception: pass

    state.update({
        "running": False,
        "restarting": False,
        "decoder_procs": [], "inference_proc": None, "monitor_proc": None,
        "stop_event": None, "inference_alive": None,
        "meta_queues": [], "result_queue": None,
        "shared_cpu": None, "shared_ram": None,
        "shm_objects": [], "shm_names": [],
        "stats": {},
    })


def _hot_restart(streams: list) -> None:
    """
    Stop the running pipeline and restart with a new stream list.
    Must be called while holding _lock.
    """
    state["restarting"] = True
    _stop_pipeline()
    if streams:
        _start_pipeline(streams)


# ── REST endpoints ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "dashboard.html")


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "running":     state["running"],
            "restarting":  state["restarting"],
            "streams":     state["streams"],
            "num_streams": len(state["streams"]),
        })


@app.route("/api/streams", methods=["GET"])
def api_get_streams():
    with _lock:
        return jsonify({"streams": state["streams"],
                        "running": state["running"]})


@app.route("/api/streams", methods=["POST"])
def api_add_stream():
    data  = request.get_json(silent=True) or {}
    url   = (data.get("url") or "").strip()
    label = (data.get("label") or f"Stream {len(state['streams'])+1}").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    with _lock:
        was_running = state["running"]
        state["streams"].append({"url": url, "label": label})
        snap = list(state["streams"])
        if was_running:
            # Hot-restart: stop then restart with updated stream list
            socketio.emit("pipeline_status", {
                "running": False, "restarting": True,
                "num_streams": len(snap)
            })
            _hot_restart(snap)

    socketio.emit("streams_changed", {"streams": snap})
    if was_running:
        socketio.emit("pipeline_status", {
            "running": True, "restarting": False,
            "num_streams": len(snap)
        })
    return jsonify({"ok": True, "streams": snap, "restarted": was_running}), 201


@app.route("/api/streams/<int:idx>", methods=["DELETE"])
def api_remove_stream(idx: int):
    with _lock:
        if not (0 <= idx < len(state["streams"])):
            return jsonify({"error": "Index out of range"}), 404
        was_running = state["running"]
        state["streams"].pop(idx)
        snap = list(state["streams"])
        if was_running:
            socketio.emit("pipeline_status", {
                "running": False, "restarting": True,
                "num_streams": len(snap)
            })
            if snap:
                _hot_restart(snap)
            else:
                _stop_pipeline()

    socketio.emit("streams_changed", {"streams": snap})
    if was_running:
        socketio.emit("pipeline_status", {
            "running": bool(snap), "restarting": False,
            "num_streams": len(snap)
        })
    return jsonify({"ok": True, "streams": snap, "restarted": was_running})


@app.route("/api/streams/<int:idx>", methods=["PATCH"])
def api_update_stream(idx: int):
    data = request.get_json(silent=True) or {}
    with _lock:
        if not (0 <= idx < len(state["streams"])):
            return jsonify({"error": "Index out of range"}), 404
        was_running = state["running"]
        if "url"   in data: state["streams"][idx]["url"]   = data["url"].strip()
        if "label" in data: state["streams"][idx]["label"] = data["label"].strip()
        snap = list(state["streams"])
        if was_running and "url" in data:
            # URL changed — hot-restart needed
            socketio.emit("pipeline_status", {
                "running": False, "restarting": True,
                "num_streams": len(snap)
            })
            _hot_restart(snap)

    socketio.emit("streams_changed", {"streams": snap})
    if was_running and "url" in data:
        socketio.emit("pipeline_status", {
            "running": True, "restarting": False,
            "num_streams": len(snap)
        })
    return jsonify({"ok": True, "streams": snap})


@app.route("/api/start", methods=["POST"])
def api_start():
    with _lock:
        if state["running"]:
            return jsonify({"error": "Already running"}), 409
        if not state["streams"]:
            return jsonify({"error": "Add at least one stream first"}), 400
        _start_pipeline(state["streams"])
    socketio.emit("pipeline_status", {
        "running": True, "restarting": False,
        "num_streams": len(state["streams"])
    })
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with _lock:
        _stop_pipeline()
    socketio.emit("pipeline_status", {"running": False, "restarting": False})
    return jsonify({"ok": True})


@app.route("/api/alerts")
def api_alerts():
    limit = int(request.args.get("limit", 100))
    stream_id = request.args.get("stream_id")
    with alert_lock:
        if stream_id is not None:
            filtered = [a for a in alerts if str(a["stream_id"]) == str(stream_id)]
        else:
            filtered = list(alerts)
        recent = list(reversed(filtered[-limit:]))
    return jsonify({"alerts": recent, "total": len(filtered)})


@app.route("/api/stats")
def api_stats():
    with _lock:
        out = {}
        for i, s in state["stats"].items():
            out[i] = _stats_snapshot(s)
        return jsonify({"stats": out, "streams": state["streams"]})


@app.route("/api/alerts/clear", methods=["POST"])
def api_clear_alerts():
    with alert_lock:
        alerts.clear()
    socketio.emit("alerts_cleared", {})
    return jsonify({"ok": True})


# ── Socket.IO events ───────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    with _lock:
        emit_initial = {
            "running":    state["running"],
            "restarting": state["restarting"],
            "streams":    state["streams"],
        }
    socketio.emit("initial_state", emit_initial)


# ── entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass  # non-Windows no-op

    print("=" * 60)
    print("  OpenVINO Multi-Stream Dashboard")
    print("  http://localhost:5000")
    print("=" * 60)

    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                     allow_unsafe_werkzeug=True)
    finally:
        with _lock:
            _stop_pipeline()
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        print("[Server] Shutdown complete.")