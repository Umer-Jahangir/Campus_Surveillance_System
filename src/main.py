"""
dashboard_server.py  (v2 – production-grade)
=============================================
Changes from optimised v1
--------------------------
P1  Frames are NEVER passed through multiprocessing.Queue.
    The queue carries only lightweight metadata tuples.
    Frames are read directly from shared memory inside the dashboard process.

P2  Base64 frame transport over Socket.IO is eliminated entirely.
    Socket.IO now emits ONLY metadata (boxes, track_ids, activities, alerts).

P3  Server-side frame annotation (annotate_frame / cv2.putText) is removed.
    The browser overlays bounding boxes on a <canvas> element instead.

P4  Video is delivered via MJPEG HTTP streaming (/video_feed/<stream_id>).
    One persistent HTTP connection per stream; no WebSocket frame overhead.
    Browser: <img src="/video_feed/0"> + <canvas> overlay.

P5  Shared-memory double-buffering is synchronised with a per-stream
    multiprocessing.Value('i') that records the active buffer index.
    The dashboard reads whichever buffer the writer last committed to,
    eliminating frame tearing from a partially-written buffer.
    Worker contract: write frame → then atomically set active_buf_idx.value.

P6  Queue drain uses a bounded try/except Empty loop instead of qsize().
    qsize() is unreliable on macOS and some Linux configurations.

P7  Workers should check result_queue.full() and DROP the frame rather than
    block, keeping the queue shallow.  Enforced in worker.py (not here).
"""

import collections
import ctypes
import multiprocessing
import multiprocessing.shared_memory
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty

import cv2
import numpy as np
import psutil
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO

from worker import decoder_worker, inference_worker
from monitor import monitor_worker
from utils import compute_fps, summarize_latency

# ── app setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "ov-dashboard-secret-2025"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=False, engineio_logger=False)

# ── constants ──────────────────────────────────────────────────────────────
FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS = 256, 320, 3
FRAME_SHAPE  = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
FRAME_SIZE   = FRAME_HEIGHT * FRAME_WIDTH * FRAME_CHANNELS
JPEG_QUALITY = 70
MAX_ALERTS          = 200
MAX_BEHAVIOR_ALERTS = 200
DRAIN_LIMIT         = 256    # upper cap on metadata msgs pulled per tick
SYS_EMIT_INTERVAL   = 0.5   # seconds between system_stats broadcasts
LOOP_SLEEP          = 0.012  # seconds
MJPEG_BOUNDARY      = b"--mjpegframe"

# ── module-level thread pool for parallel JPEG encoding ───────────────────
# Used exclusively for the MJPEG path — NOT for Socket.IO (P2).
_encode_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="jpeg-enc")

# ── MJPEG frame store ─────────────────────────────────────────────────────
# Latest encoded JPEG bytes per stream_id.  Protected by _mjpeg_lock.
# Emit-loop writes; _mjpeg_generator threads read.
_mjpeg_frames: dict[int, bytes]           = {}
_mjpeg_events: dict[int, threading.Event] = {}  # signalled on new frame
_mjpeg_lock = threading.Lock()

# ── global state ───────────────────────────────────────────────────────────
_lock = threading.Lock()

state = {
    "running":           False,
    "restarting":        False,
    "streams":           [],
    "decoder_procs":     [],
    "inference_proc":    None,
    "monitor_proc":      None,
    "stop_event":        None,
    "inference_alive":   None,
    "meta_queues":       [],
    "result_queue":      None,
    "shared_cpu":        None,
    "shared_ram":        None,
    "shm_objects":       [],
    "shm_names":         [],
    "active_buf_idxs":   [],   # P5: per-stream multiprocessing.Value('i')
    "stats":             {},
}

alerts:          list = []
behavior_alerts: list = []
alert_lock          = threading.Lock()
behavior_alert_lock = threading.Lock()

# ── helpers ────────────────────────────────────────────────────────────────

def compute_cores_per_decoder(num_streams: int) -> int:
    total   = psutil.cpu_count(logical=False) or 4
    reserve = max(2, total // 4)
    pool    = total - reserve
    per     = max(1, pool // num_streams)
    while (per * num_streams) > (total - 2) and per > 1:
        per -= 1
    return per


def _read_frame_from_shm(stream_idx: int) -> "np.ndarray | None":
    """
    P1 / P5: Read the latest committed frame directly from shared memory.

    Uses active_buf_idxs[stream_idx].value (0 or 1) to pick the buffer
    the writer last finished writing, preventing half-written frame reads.
    Returns a *copy* so the shared buffer is immediately reusable.
    """
    with _lock:
        shm_objects = state["shm_objects"]
        active_bufs = state["active_buf_idxs"]
        if stream_idx >= len(shm_objects) or stream_idx >= len(active_bufs):
            return None
        sa, sb  = shm_objects[stream_idx]
        buf_idx = active_bufs[stream_idx].value   # atomic read
        shm     = sa if buf_idx == 0 else sb

    arr = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
    return arr.copy()   # copy before lock / shm reference is released


def _encode_stream_jpeg(stream_idx: int) -> "bytes | None":
    """
    P4: Grab the latest shared-memory frame for stream_idx and JPEG-encode
    it.  Runs inside _encode_pool — imencode releases the GIL, so multiple
    streams encode concurrently.
    """
    frame = _read_frame_from_shm(stream_idx)
    if frame is None:
        return None
    _, buf = cv2.imencode(".jpg", frame,
                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return bytes(buf)


def _stats_snapshot(s: dict) -> dict:
    m_avg, m_p95, m_max = summarize_latency(s["model_latencies"])
    p_avg, p_p95, p_max = summarize_latency(s["pipeline_latencies"])
    dur = 0.0
    if s["start_time"] and s["end_time"]:
        dur = s["end_time"] - s["start_time"]
    return {
        "fps":      round(s["live_fps"], 1),
        "frames":   s["frames"],
        "dropped":  s.get("dropped", 0),
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


# ── MJPEG streaming ────────────────────────────────────────────────────────

def _mjpeg_generator(stream_id: int):
    """
    P4: Infinite multipart generator for /video_feed/<stream_id>.

    Blocks on a per-stream threading.Event until the emit-loop deposits a
    new JPEG, then yields the multipart chunk.  One persistent HTTP
    connection per browser tab; zero WebSocket overhead for raw video.
    """
    # Ensure an event exists before entering the loop.
    with _mjpeg_lock:
        if stream_id not in _mjpeg_events:
            _mjpeg_events[stream_id] = threading.Event()
    evt = _mjpeg_events[stream_id]

    while True:
        with _lock:
            running = state["running"]
        if not running:
            break

        signalled = evt.wait(timeout=1.0)
        if not signalled:
            continue   # pipeline paused or not yet started
        evt.clear()

        with _mjpeg_lock:
            jpeg = _mjpeg_frames.get(stream_id)
        if jpeg is None:
            continue

        yield (
            MJPEG_BOUNDARY + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg + b"\r\n"
        )


@app.route("/video_feed/<int:stream_id>")
def video_feed(stream_id: int):
    """
    P4: MJPEG HTTP endpoint.
    Browser usage: <img src="/video_feed/0">
    Overlay bounding boxes on a <canvas> element using Socket.IO metadata.
    """
    return Response(
        _mjpeg_generator(stream_id),
        mimetype="multipart/x-mixed-replace; boundary=mjpegframe",
    )


# ── emit loop ──────────────────────────────────────────────────────────────

def _emit_loop() -> None:
    """
    Daemon thread: drains result_queue for METADATA ONLY (no frame arrays),
    reads frames from shared memory, encodes JPEG for MJPEG clients, and
    emits lightweight Socket.IO metadata events.

    Key differences from v1
    -----------------------
    * No numpy frame array is unpickled from the queue (P1).
      Frame arrays in messages are accepted for backward compat but discarded.
    * No JPEG encoding on the Socket.IO path — only MJPEG path encodes (P2).
    * annotate_frame() removed entirely — browser canvas does it (P3).
    * Queue drain uses bounded try/except Empty, not qsize() (P6).
    * Socket.IO 'frame_meta' payload: boxes + track_ids + activities, no image.
    """
    last_sys_emit = time.monotonic()

    while True:
        # ── snapshot shared state ONCE per tick ───────────────────────────
        with _lock:
            running   = state["running"]
            rq        = state["result_queue"]
            stop_ev   = state["stop_event"]
            stats_ref = state["stats"]
            streams   = state["streams"]

        if not running or rq is None or (stop_ev and stop_ev.is_set()):
            time.sleep(0.05)
            continue

        # Streams that received new metadata this tick → need a fresh JPEG.
        updated_streams: set[int]     = set()
        latest_meta: dict[int, dict]  = {}
        pending_behavior: list[tuple] = []

        # ── P6: bounded drain with try/except — portable, no qsize() ─────
        for _ in range(DRAIN_LIMIT):
            try:
                msg = rq.get_nowait()
            except Empty:
                break

            if msg[0] != "frame":
                continue

            # ── Unpack result message ──────────────────────────────────────
            #
            # worker.py (current, frameless) sends a 12-element tuple:
            #   ("frame", stream_id, boxes, track_ids, classes, confs,
            #    labels, timestamp, per_frame_ms, e2e_lat,
            #    behavior_alerts, activities)
            #
            # Legacy workers (with _frame at index 2) sent 13-14 elements.
            # Distinguish purely by length.
            #
            if len(msg) >= 14:
                # Legacy 14+: _frame at index 2, activities at index 13
                (_, idx, _frame, boxes, track_ids, classes, confs, labels,
                 capture_time, model_lat, e2e_lat, b_alerts,
                 per_person_activities) = msg[:14]

            elif len(msg) == 13:
                # Legacy 13: _frame at index 2, no activities
                (_, idx, _frame, boxes, track_ids, classes, confs, labels,
                 capture_time, model_lat, e2e_lat, b_alerts, _act) = msg
                per_person_activities = {}

            elif len(msg) == 12:
                # Current worker.py — NO _frame field:
                # ("frame", stream_id, boxes, track_ids, classes, confs,
                #  labels, timestamp, per_frame_ms, e2e_lat,
                #  behavior_alerts, activities)
                (_, idx, boxes, track_ids, classes, confs, labels,
                 capture_time, model_lat, e2e_lat,
                 b_alerts, per_person_activities) = msg

            else:
                # Minimal legacy 11: _frame at index 2, no alerts
                (_, idx, _frame, boxes, track_ids, classes, confs, labels,
                 capture_time, model_lat, e2e_lat) = msg
                b_alerts, per_person_activities = [], {}

            s = stats_ref.get(idx)
            if s is None:
                continue

            # ── update stats (emit-loop is sole writer — no lock needed) ──
            now_t = time.perf_counter()
            if s["start_time"] is None:
                s["start_time"] = capture_time
            s["end_time"] = capture_time
            s["frames"]  += 1
            s["model_latencies"].append(model_lat)
            s["pipeline_latencies"].append(e2e_lat)

            elapsed = now_t - s["last_fps_time"]
            if elapsed >= 1.0:
                s["live_fps"]        = (s["frames"] - s["last_fps_frames"]) / elapsed
                s["last_fps_time"]   = now_t
                s["last_fps_frames"] = s["frames"]

                if now_t - s["last_log_time"] >= 5.0:
                    m_avg, m_p95, m_max = summarize_latency(s["model_latencies"])
                    p_avg, p_p95, p_max = summarize_latency(s["pipeline_latencies"])
                    print(
                        f"[Benchmark][Stream {idx}] "
                        f"fps={s['live_fps']:.1f} frames={s['frames']} "
                        f"dropped={s.get('dropped', 0)} "
                        f"model_avg={m_avg:.1f} p95={m_p95:.1f} max={m_max:.1f} "
                        f"e2e_avg={p_avg:.1f} p95={p_p95:.1f} max={p_max:.1f}"
                    )
                    s["last_log_time"] = now_t

            updated_streams.add(idx)

            # P2 / P3: metadata-only payload — NO image bytes, NO annotation.
            latest_meta[idx] = {
                "stream_id":    idx,
                "boxes":        boxes.tolist() if hasattr(boxes, "tolist") else (boxes or []),
                "track_ids":    track_ids.tolist() if hasattr(track_ids, "tolist") else (track_ids or []),
                "activities":   per_person_activities or {},
                "person_count": len(per_person_activities) if per_person_activities else 0,
                "stats":        _stats_snapshot(s),
            }

            if b_alerts:
                fight_alerts = [
                    ba for ba in b_alerts
                    if ba.get("label") == "fight" or ba.get("type") == "fight"
                ]
                if fight_alerts:
                    stream_label = (
                        streams[idx]["label"]
                        if idx < len(streams) else f"S{idx}"
                    )
                    pending_behavior.append((idx, stream_label, fight_alerts))

        # ── P1 / P4: encode fresh JPEG from shared memory for MJPEG store ─
        # Submitted to the thread pool so N streams encode concurrently.
        # imencode releases the GIL → true parallelism even under CPython.
        if updated_streams:
            futures = {
                _encode_pool.submit(_encode_stream_jpeg, idx): idx
                for idx in updated_streams
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    jpeg = fut.result()
                    if jpeg:
                        with _mjpeg_lock:
                            _mjpeg_frames[idx] = jpeg
                            if idx not in _mjpeg_events:
                                _mjpeg_events[idx] = threading.Event()
                            _mjpeg_events[idx].set()   # wake MJPEG generator
                except Exception as exc:
                    print(f"[emit-loop] shm encode error stream {idx}: {exc}")

        # ── P2 / P3: emit metadata-only Socket.IO events ──────────────────
        # 'frame_meta' replaces the old 'frame_update' that carried b64 JPEG.
        # Frontend listens on 'frame_meta' and draws boxes on <canvas>.
        for meta_payload in latest_meta.values():
            socketio.emit("frame_meta", meta_payload)

        # ── behavior alerts ────────────────────────────────────────────────
        for idx, stream_label, fight_alerts in pending_behavior:
            with behavior_alert_lock:
                for ba in fight_alerts:
                    behavior_alerts.append({
                        "stream_id":    idx,
                        "stream_label": stream_label,
                        **ba,
                    })
                del behavior_alerts[
                    :max(0, len(behavior_alerts) - MAX_BEHAVIOR_ALERTS)
                ]
            socketio.emit("behavior_alert", {
                "stream_id":    idx,
                "stream_label": stream_label,
                "alerts":       fight_alerts,
                "ts":           time.time(),
            })

        # ── system stats (throttled) ───────────────────────────────────────
        now_mono = time.monotonic()
        if now_mono - last_sys_emit >= SYS_EMIT_INTERVAL:
            with _lock:
                cpu = state["shared_cpu"].value if state["shared_cpu"] else 0.0
                ram = state["shared_ram"].value if state["shared_ram"] else 0.0
            socketio.emit("system_stats", {
                "cpu": round(cpu, 1),
                "ram": round(ram, 1),
            })
            last_sys_emit = now_mono

        time.sleep(LOOP_SLEEP)


threading.Thread(target=_emit_loop, daemon=True, name="emit-loop").start()


# ── pipeline lifecycle ─────────────────────────────────────────────────────

def _start_pipeline(streams: list) -> None:
    """Allocate shared memory, spawn workers, initialise per-stream stats."""
    num_streams = len(streams)
    stream_urls = [s["url"] for s in streams]
    cpp         = compute_cores_per_decoder(num_streams)

    stop_event      = multiprocessing.Event()
    inference_alive = multiprocessing.Event()

    shm_objects, shm_names = [], []
    # P5: one active-buffer-index Value per stream.
    # Worker writes frame to the inactive buffer, then sets value → the
    # dashboard always reads the last fully committed buffer.
    active_buf_idxs: list[multiprocessing.Value] = []

    for _ in range(num_streams):
        sa = multiprocessing.shared_memory.SharedMemory(create=True, size=FRAME_SIZE)
        sb = multiprocessing.shared_memory.SharedMemory(create=True, size=FRAME_SIZE)
        shm_objects.append((sa, sb))
        shm_names.append((sa.name, sb.name))
        active_buf_idxs.append(multiprocessing.Value('i', 0))   # P5

    meta_queues  = [multiprocessing.Queue(maxsize=8)   for _ in range(num_streams)]
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
            args=(i, url, shm_names[i], active_buf_idxs[i], FRAME_SHAPE,
                  meta_queues[i], stop_event, inference_alive,
                  num_streams, cpp),
        )
        p.start()
        dec_procs.append(p)

    mon_proc = multiprocessing.Process(
        target=monitor_worker,
        args=(shared_cpu, shared_ram, stop_event),
    )
    mon_proc.start()

    per_stream_stats = {}
    for i in range(num_streams):
        per_stream_stats[i] = {
            "frames":  0,
            "dropped": 0,
            "model_latencies":    collections.deque(maxlen=1000),
            "pipeline_latencies": collections.deque(maxlen=1000),
            "start_time":      None,
            "end_time":        None,
            "last_fps_time":   time.perf_counter(),
            "last_fps_frames": 0,
            "live_fps":        0.0,
            "last_log_time":   time.perf_counter(),
            "activity":        "no-person",
        }

    # Clear MJPEG store for the new session.
    with _mjpeg_lock:
        _mjpeg_frames.clear()
        _mjpeg_events.clear()

    state.update({
        "running":         True,
        "restarting":      False,
        "decoder_procs":   dec_procs,
        "inference_proc":  inf_proc,
        "monitor_proc":    mon_proc,
        "stop_event":      stop_event,
        "inference_alive": inference_alive,
        "meta_queues":     meta_queues,
        "result_queue":    result_queue,
        "shared_cpu":      shared_cpu,
        "shared_ram":      shared_ram,
        "shm_objects":     shm_objects,
        "shm_names":       shm_names,
        "active_buf_idxs": active_buf_idxs,   # P5
        "stats":           per_stream_stats,
    })


def _stop_pipeline() -> None:
    """Terminate all worker processes and release shared memory."""
    if not state["running"]:
        return

    state["stop_event"].set()

    all_procs = (
        state["decoder_procs"]
        + [state["inference_proc"], state["monitor_proc"]]
    )
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
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass

    # Wake any blocked MJPEG generators so their threads can exit cleanly.
    with _mjpeg_lock:
        _mjpeg_frames.clear()
        for evt in _mjpeg_events.values():
            evt.set()
        _mjpeg_events.clear()

    state.update({
        "running": False, "restarting": False,
        "decoder_procs": [], "inference_proc": None, "monitor_proc": None,
        "stop_event": None, "inference_alive": None,
        "meta_queues": [], "result_queue": None,
        "shared_cpu": None, "shared_ram": None,
        "shm_objects": [], "shm_names": [],
        "active_buf_idxs": [],
        "stats": {},
    })


def _hot_restart_bg(streams: list) -> None:
    """
    Stop the current pipeline and start a new one with the given stream list.
    Runs in a background daemon thread so the calling REST handler returns
    immediately instead of blocking for the full join-timeout chain.
    """
    with _lock:
        state["restarting"] = True

    socketio.emit("pipeline_status", {
        "running": False, "restarting": True, "num_streams": len(streams)
    })

    with _lock:
        _stop_pipeline()
        if streams:
            _start_pipeline(streams)
        state["restarting"] = False

    socketio.emit("pipeline_status", {
        "running":     bool(streams),
        "restarting":  False,
        "num_streams": len(streams),
    })


def _hot_restart(streams: list) -> None:
    """Spawn a daemon thread to handle stop+restart without blocking callers."""
    threading.Thread(
        target=_hot_restart_bg,
        args=(streams,),
        daemon=True,
        name="hot-restart",
    ).start()


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
        return jsonify({
            "streams": state["streams"],
            "running": state["running"],
        })


@app.route("/api/streams", methods=["POST"])
def api_add_stream():
    data  = request.get_json(silent=True) or {}
    url   = (data.get("url") or "").strip()
    label = (data.get("label") or
             f"Stream {len(state['streams']) + 1}").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    with _lock:
        was_running = state["running"]
        state["streams"].append({"url": url, "label": label})
        snap = list(state["streams"])

    socketio.emit("streams_changed", {"streams": snap})
    if was_running:
        _hot_restart(snap)

    return jsonify({"ok": True, "streams": snap, "restarted": was_running}), 201


@app.route("/api/streams/<int:idx>", methods=["DELETE"])
def api_remove_stream(idx: int):
    with _lock:
        if not (0 <= idx < len(state["streams"])):
            return jsonify({"error": "Index out of range"}), 404
        was_running = state["running"]
        state["streams"].pop(idx)
        snap = list(state["streams"])

    socketio.emit("streams_changed", {"streams": snap})
    if was_running:
        _hot_restart(snap)

    return jsonify({"ok": True, "streams": snap, "restarted": was_running})


@app.route("/api/streams/<int:idx>", methods=["PATCH"])
def api_update_stream(idx: int):
    data = request.get_json(silent=True) or {}
    with _lock:
        if not (0 <= idx < len(state["streams"])):
            return jsonify({"error": "Index out of range"}), 404
        was_running = state["running"]
        if "url"   in data:
            state["streams"][idx]["url"]   = data["url"].strip()
        if "label" in data:
            state["streams"][idx]["label"] = data["label"].strip()
        snap = list(state["streams"])

    socketio.emit("streams_changed", {"streams": snap})
    if was_running and "url" in data:
        _hot_restart(snap)

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
        "running":     True,
        "restarting":  False,
        "num_streams": len(state["streams"]),
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
    limit     = int(request.args.get("limit", 100))
    stream_id = request.args.get("stream_id")
    with alert_lock:
        filtered = (
            [a for a in alerts if str(a["stream_id"]) == str(stream_id)]
            if stream_id is not None else list(alerts)
        )
        recent = list(reversed(filtered[-limit:]))
    return jsonify({"alerts": recent, "total": len(filtered)})


@app.route("/api/behavior_alerts")
def api_behavior_alerts():
    limit     = int(request.args.get("limit", 100))
    stream_id = request.args.get("stream_id")
    with behavior_alert_lock:
        filtered = (
            [a for a in behavior_alerts
             if str(a["stream_id"]) == str(stream_id)]
            if stream_id is not None else list(behavior_alerts)
        )
        recent = list(reversed(filtered[-limit:]))
    return jsonify({"alerts": recent, "total": len(filtered)})


@app.route("/api/stats")
def api_stats():
    with _lock:
        out = {i: _stats_snapshot(s) for i, s in state["stats"].items()}
        return jsonify({"stats": out, "streams": state["streams"]})


@app.route("/api/alerts/clear", methods=["POST"])
def api_clear_alerts():
    with alert_lock:
        alerts.clear()
    with behavior_alert_lock:
        behavior_alerts.clear()
    socketio.emit("alerts_cleared", {})
    return jsonify({"ok": True})


# ── Socket.IO events ───────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    with _lock:
        initial = {
            "running":    state["running"],
            "restarting": state["restarting"],
            "streams":    state["streams"],
        }
    socketio.emit("initial_state", initial)


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    print("=" * 60)
    print("  Campus Surveillance Dashboard")
    print("  http://localhost:5000")
    print("=" * 60)

    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                     allow_unsafe_werkzeug=True)
    finally:
        with _lock:
            _stop_pipeline()
        _encode_pool.shutdown(wait=False)
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        print("[Server] Shutdown complete.")