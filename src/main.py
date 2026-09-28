"""Flask dashboard: bounded immutable frame packets and synchronized MJPEG overlays."""

import collections
import ctypes
import multiprocessing
import time
import threading
import uuid
from queue import Empty

import cv2
import psutil
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO

from worker import decoder_worker, inference_worker
from monitor import monitor_worker
from utils import summarize_latency

# ── app setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "ov-dashboard-secret-2025"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# ── constants ──────────────────────────────────────────────────────────────
FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS = 256, 320, 3
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
FRAME_SIZE = FRAME_HEIGHT * FRAME_WIDTH * FRAME_CHANNELS
JPEG_QUALITY = 70
MAX_ALERTS = 200
MAX_BEHAVIOR_ALERTS = 200
DRAIN_LIMIT = 256  # upper cap on metadata msgs pulled per tick
SYS_EMIT_INTERVAL = 0.5  # seconds between system_stats broadcasts
LOOP_SLEEP = 0.012  # seconds
MJPEG_BOUNDARY = b"--mjpegframe"

# ── MJPEG frame store ─────────────────────────────────────────────────────
# Latest encoded JPEG bytes per stream_id.  Protected by _mjpeg_lock.
# Emit-loop writes; _mjpeg_generator threads read.
_mjpeg_frames: dict[int, bytes] = {}
_mjpeg_events: dict[int, threading.Event] = {}  # signalled on new frame
_mjpeg_lock = threading.Lock()

# ── global state ───────────────────────────────────────────────────────────
_lock = threading.Lock()

state = {
    "running": False,
    "error": None,
    "latest_meta": {},
    "session_id": None,
    "restarting": False,
    "streams": [],
    "decoder_procs": [],
    "inference_proc": None,
    "monitor_proc": None,
    "stop_event": None,
    "inference_alive": None,
    "meta_queues": [],
    "result_queue": None,
    "shared_cpu": None,
    "shared_ram": None,
    "stats": {},
}

alerts: list = []
behavior_alerts: list = []
alert_lock = threading.Lock()
behavior_alert_lock = threading.Lock()

# ── helpers ────────────────────────────────────────────────────────────────


def compute_cores_per_decoder(num_streams: int) -> int:
    total = psutil.cpu_count(logical=False) or 4
    reserve = max(2, total // 4)
    pool = total - reserve
    per = max(1, pool // num_streams)
    while (per * num_streams) > (total - 2) and per > 1:
        per -= 1
    return per


def _stats_snapshot(s: dict) -> dict:
    m_avg, m_p95, m_max = summarize_latency(s["model_latencies"])
    p_avg, p_p95, p_max = summarize_latency(s["pipeline_latencies"])
    dur = 0.0
    if s["start_time"] and s["end_time"]:
        dur = s["end_time"] - s["start_time"]
    return {
        "fps": round(s["live_fps"], 1),
        "frames": s["frames"],
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


# ── MJPEG streaming ────────────────────────────────────────────────────────


def _mjpeg_generator(stream_id: int):
    """
    Yield multipart JPEG frames for /video_feed/<stream_id>.

    Blocks on a per-stream threading.Event until the emit-loop deposits a
    new JPEG, then yields the multipart chunk.  One persistent HTTP
    connection per browser tab; zero WebSocket overhead for raw video.
    """
    # Ensure an event exists before entering the loop.
    with _mjpeg_lock:
        if stream_id not in _mjpeg_events:
            _mjpeg_events[stream_id] = threading.Event()
    evt = _mjpeg_events[stream_id]

    generation = state["stop_event"]
    last_jpeg = None
    last_sent = 0.0
    prefix = MJPEG_BOUNDARY + b"\r\n"
    while True:
        with _lock:
            running = state["running"] and state["stop_event"] is generation
        if not running:
            break

        with _mjpeg_lock:
            jpeg = _mjpeg_frames.get(stream_id)
        if jpeg is None or (jpeg is last_jpeg and time.monotonic() - last_sent < 1.0):
            evt.wait(timeout=0.1)
            evt.clear()
            continue
        with _lock:
            if state["stop_event"] is not generation or not state["running"]:
                break
        last_jpeg = jpeg
        # Repeat cached pixels so multipart clients complete decoding at EOF.
        # Metadata timestamps remain unchanged; this is not a new video frame.
        last_sent = time.monotonic()

        yield (
            prefix + b"Content-Type: image/jpeg\r\n"
            b"Content-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg
            + b"\r\n"
            + MJPEG_BOUNDARY
            + b"\r\n"
        )
        prefix = b""


@app.route("/video_feed/<int:stream_id>")
def video_feed(stream_id: int):
    """
    MJPEG HTTP endpoint.
    Browser usage: <img src="/video_feed/0">
    The JPEG includes boxes from that exact inference frame.
    """
    return Response(
        _mjpeg_generator(stream_id),
        mimetype="multipart/x-mixed-replace; boundary=mjpegframe",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# ── emit loop ──────────────────────────────────────────────────────────────


def _emit_loop() -> None:
    """Publish metadata and JPEG already paired by the inference worker."""
    last_sys_emit = time.monotonic()

    while True:
        # ── snapshot shared state ONCE per tick ───────────────────────────
        with _lock:
            running = state["running"]
            rq = state["result_queue"]
            stop_ev = state["stop_event"]
            stats_ref = state["stats"]
            streams = state["streams"]

        if not running or rq is None or (stop_ev and stop_ev.is_set()):
            time.sleep(0.05)
            continue

        # Streams that received new metadata this tick → need a fresh JPEG.
        updated_streams: set[int] = set()
        latest_meta: dict[int, dict] = {}
        pending_behavior: list[tuple] = []

        # ── P6: bounded drain with try/except — portable, no qsize() ─────
        for _ in range(DRAIN_LIMIT):
            try:
                msg = rq.get_nowait()
            except Empty:
                break

            if isinstance(msg, dict) and msg.get("type") == "error":
                with _lock:
                    if rq is state["result_queue"]:
                        state["error"] = msg["error"]
                        _stop_pipeline()
                socketio.emit(
                    "pipeline_status", {"running": False, "error": msg["error"]}
                )
                break
            if not isinstance(msg, dict) or msg.get("type") != "frame":
                continue
            idx = msg["stream_id"]
            boxes, track_ids = msg["boxes"], msg["track_ids"]
            capture_time = msg["timestamp"]
            model_lat, e2e_lat = msg["model_lat"], msg["e2e_lat"]
            b_alerts = msg["behavior_alerts"]
            per_person_activities = msg["activities"]
            with _lock:
                if rq is not state["result_queue"] or not state["running"]:
                    break
            with _mjpeg_lock:
                _mjpeg_frames[idx] = msg["jpeg"]
                _mjpeg_events.setdefault(idx, threading.Event()).set()

            s = stats_ref.get(idx)
            if s is None:
                continue

            # ── update stats (emit-loop is sole writer — no lock needed) ──
            now_t = time.perf_counter()
            if s["start_time"] is None:
                s["start_time"] = capture_time
            s["end_time"] = capture_time
            s["frames"] += 1
            s["model_latencies"].append(model_lat)
            s["pipeline_latencies"].append(e2e_lat)

            elapsed = now_t - s["last_fps_time"]
            if elapsed >= 1.0:
                s["live_fps"] = (s["frames"] - s["last_fps_frames"]) / elapsed
                s["last_fps_time"] = now_t
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

            s["dropped"] += max(0, msg["frame_id"] - s.get("last_frame_id", -1) - 1)
            s["last_frame_id"] = msg["frame_id"]
            updated_streams.add(idx)

            # Image and overlays travel together in MJPEG; this is metadata only.
            latest_meta[idx] = {
                "session_id": state["session_id"],
                "published_at": time.time(),
                "frame_id": msg["frame_id"],
                "source_time": msg["source_time"],
                "coordinate_space": "inference_pixels_xyxy",
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "original_boxes": msg["original_boxes"].tolist(),
                "geometry": msg["geometry"],
                "scores": msg["scores"],
                "action_status": msg["action_status"],
                "action_diagnostics": msg.get("action_diagnostics", {}),
                "scene_action": msg.get("scene_action", {}),
                "source_fps": msg.get("source_fps"),
                "mode": msg.get("mode", "combined"),
                "source_kind": msg.get("source_kind"),
                "total_frames": msg.get("total_frames"),
                "profile": msg.get("profile", {}),
                "delivery_ms": (time.perf_counter() - msg["timestamp"]) * 1000,
                "queue_ms": msg.get("queue_ms"),
                "pipeline_ms": msg.get("e2e_lat"),
                "stream_id": idx,
                "boxes": boxes.tolist() if hasattr(boxes, "tolist") else (boxes or []),
                "track_ids": (
                    track_ids.tolist()
                    if hasattr(track_ids, "tolist")
                    else (track_ids or [])
                ),
                "activities": per_person_activities or {},
                "person_count": (
                    len(per_person_activities) if per_person_activities else 0
                ),
                "stats": _stats_snapshot(s),
            }

            if msg.get("scene_event"):
                scene_event = {
                    **msg["scene_event"],
                    "stream_id": idx,
                    "session_id": state["session_id"],
                    "frame_id": msg["frame_id"],
                    "source_time": msg["source_time"],
                    "scope": "scene",
                }
                with behavior_alert_lock:
                    behavior_alerts.append(scene_event)
                    del behavior_alerts[:-MAX_BEHAVIOR_ALERTS]
                socketio.emit("scene_alert", scene_event)

            if b_alerts:
                fight_alerts = [
                    ba
                    for ba in b_alerts
                    if ba.get("label") == "fight" or ba.get("type") == "fight"
                ]
                if fight_alerts:
                    stream_label = (
                        streams[idx]["label"] if idx < len(streams) else f"S{idx}"
                    )
                    pending_behavior.append((idx, stream_label, fight_alerts))

        # Emit metadata-only Socket.IO events.
        # 'frame_meta' replaces the old 'frame_update' that carried b64 JPEG.
        # Frontend listens on 'frame_meta' and draws boxes on <canvas>.
        for meta_payload in latest_meta.values():
            with _lock:
                if rq is state["result_queue"] and state["running"]:
                    state["latest_meta"][meta_payload["stream_id"]] = meta_payload
                    socketio.emit("frame_meta", meta_payload)

        # ── behavior alerts ────────────────────────────────────────────────
        for idx, stream_label, fight_alerts in pending_behavior:
            with _lock:
                if rq is not state["result_queue"] or not state["running"]:
                    break
            with behavior_alert_lock:
                for ba in fight_alerts:
                    behavior_alerts.append(
                        {
                            "stream_id": idx,
                            "stream_label": stream_label,
                            "session_id": state["session_id"],
                            **ba,
                        }
                    )
                del behavior_alerts[
                    : max(0, len(behavior_alerts) - MAX_BEHAVIOR_ALERTS)
                ]
            socketio.emit(
                "behavior_alert",
                {
                    "stream_id": idx,
                    "stream_label": stream_label,
                    "alerts": fight_alerts,
                    "ts": time.time(),
                },
            )

        # ── system stats (throttled) ───────────────────────────────────────
        now_mono = time.monotonic()
        if now_mono - last_sys_emit >= SYS_EMIT_INTERVAL:
            with _lock:
                cpu = state["shared_cpu"].value if state["shared_cpu"] else 0.0
                ram = state["shared_ram"].value if state["shared_ram"] else 0.0
            socketio.emit(
                "system_stats",
                {
                    "cpu": round(cpu, 1),
                    "ram": round(ram, 1),
                },
            )
            # Loss of observation is not evidence that the fight has ended. Cached
            # MJPEG repetitions do not refresh published_at or create events.
            for idx, cached in list(state["latest_meta"].items()):
                action = cached.get("scene_action", {})
                if (
                    time.time() - cached["published_at"] > 3
                    and action.get("state") == "suspected_fight"
                ):
                    action["state"] = "stale"
                    event = {
                        "stream_id": idx,
                        "session_id": cached["session_id"],
                        "phase": "observation_lost",
                        "reason": "no fresh frames; fight outcome unknown",
                        "scope": "scene",
                        "source_time": cached["source_time"],
                        "frame_id": cached["frame_id"],
                    }
                    with behavior_alert_lock:
                        behavior_alerts.append(event)
                        del behavior_alerts[:-MAX_BEHAVIOR_ALERTS]
                    socketio.emit("scene_alert", event)
            last_sys_emit = now_mono

        time.sleep(LOOP_SLEEP)


threading.Thread(target=_emit_loop, daemon=True, name="emit-loop").start()


# ── pipeline lifecycle ─────────────────────────────────────────────────────


def _start_pipeline(streams: list) -> None:
    """Spawn fresh per-stream workers and bounded frame queues."""
    num_streams = len(streams)
    stream_urls = [s["url"] for s in streams]
    cpp = compute_cores_per_decoder(num_streams)

    stop_event = multiprocessing.Event()
    inference_alive = multiprocessing.Event()

    meta_queues = [multiprocessing.Queue(maxsize=8) for _ in range(num_streams)]
    scene_queues = [multiprocessing.Queue(maxsize=2) for _ in range(num_streams)]
    result_queue = multiprocessing.Queue(maxsize=8)
    shared_cpu = multiprocessing.Value("d", 0.0)
    shared_ram = multiprocessing.Value("d", 0.0)

    inf_proc = multiprocessing.Process(
        target=inference_worker,
        args=(
            num_streams,
            FRAME_SHAPE,
            meta_queues,
            result_queue,
            stop_event,
            inference_alive,
            num_streams,
            cpp,
            scene_queues,
        ),
    )
    inf_proc.start()

    dec_procs = []
    for i, url in enumerate(stream_urls):
        p = multiprocessing.Process(
            target=decoder_worker,
            args=(
                i,
                url,
                FRAME_SHAPE,
                meta_queues[i],
                stop_event,
                inference_alive,
                num_streams,
                cpp,
                scene_queues[i],
            ),
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
            "frames": 0,
            "dropped": 0,
            "model_latencies": collections.deque(maxlen=1000),
            "pipeline_latencies": collections.deque(maxlen=1000),
            "start_time": None,
            "end_time": None,
            "last_fps_time": time.perf_counter(),
            "last_fps_frames": 0,
            "live_fps": 0.0,
            "last_log_time": time.perf_counter(),
            "activity": "no-person",
        }

    # Clear MJPEG store for the new session.
    with _mjpeg_lock:
        _mjpeg_frames.clear()
        _mjpeg_events.clear()

    state.update(
        {
            "running": True,
            "error": None,
            "latest_meta": {},
            "session_id": uuid.uuid4().hex,
            "restarting": False,
            "decoder_procs": dec_procs,
            "inference_proc": inf_proc,
            "monitor_proc": mon_proc,
            "stop_event": stop_event,
            "inference_alive": inference_alive,
            "meta_queues": meta_queues,
            "scene_queues": scene_queues,
            "result_queue": result_queue,
            "shared_cpu": shared_cpu,
            "shared_ram": shared_ram,
            "stats": per_stream_stats,
        }
    )


def _stop_pipeline() -> None:
    """Terminate workers and discard this run's queues and frame cache."""
    if not state["running"]:
        return

    for idx, cached in state["latest_meta"].items():
        if cached.get("scene_action", {}).get("state") == "suspected_fight":
            event = {
                "phase": "observation_lost",
                "reason": "pipeline stopped; fight outcome unknown",
                "scope": "scene",
                "stream_id": idx,
                "session_id": cached["session_id"],
                "source_time": cached["source_time"],
            }
            with behavior_alert_lock:
                behavior_alerts.append(event)
                del behavior_alerts[:-MAX_BEHAVIOR_ALERTS]
            socketio.emit("scene_alert", event)

    state["stop_event"].set()

    all_procs = state["decoder_procs"] + [
        state["inference_proc"],
        state["monitor_proc"],
    ]
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

    # Wake any blocked MJPEG generators so their threads can exit cleanly.
    for q in (
        state.get("meta_queues", [])
        + state.get("scene_queues", [])
        + ([state["result_queue"]] if state.get("result_queue") else [])
    ):
        q.cancel_join_thread()
        q.close()
    with _mjpeg_lock:
        _mjpeg_frames.clear()
        for evt in _mjpeg_events.values():
            evt.set()
        _mjpeg_events.clear()

    state.update(
        {
            "running": False,
            "restarting": False,
            "latest_meta": {},
            "decoder_procs": [],
            "inference_proc": None,
            "monitor_proc": None,
            "stop_event": None,
            "inference_alive": None,
            "meta_queues": [],
            "result_queue": None,
            "scene_queues": [],
            "shared_cpu": None,
            "shared_ram": None,
            "stats": {},
        }
    )


def _hot_restart_bg(streams: list) -> None:
    """
    Stop the current pipeline and start a new one with the given stream list.
    Runs in a background daemon thread so the calling REST handler returns
    immediately instead of blocking for the full join-timeout chain.
    """
    with _lock:
        state["restarting"] = True

    socketio.emit(
        "pipeline_status",
        {"running": False, "restarting": True, "num_streams": len(streams)},
    )

    with _lock:
        _stop_pipeline()
        try:
            if streams:
                for stream in streams:
                    cap = cv2.VideoCapture(stream["url"])
                    opened = cap.isOpened()
                    cap.release()
                    if not opened:
                        raise ValueError(f"Cannot open source: {stream['label']}")
                _start_pipeline(streams)
        except Exception as exc:
            state["error"] = str(exc)
        state["restarting"] = False

    socketio.emit(
        "pipeline_status",
        {
            "running": state["running"],
            "error": state["error"],
            "restarting": False,
            "num_streams": len(streams),
        },
    )


def _hot_restart(streams: list) -> None:
    """Spawn a daemon thread to handle stop+restart without blocking callers."""
    with _lock:
        state["restarting"] = True
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
        return jsonify(
            {
                "running": state["running"],
                "error": state["error"],
                "inference_alive": bool(
                    state["inference_proc"] and state["inference_proc"].is_alive()
                ),
                "restarting": state["restarting"],
                "streams": state["streams"],
                "num_streams": len(state["streams"]),
            }
        )


@app.route("/api/streams", methods=["GET"])
def api_get_streams():
    with _lock:
        return jsonify(
            {
                "streams": state["streams"],
                "running": state["running"],
            }
        )


@app.route("/api/streams", methods=["POST"])
def api_add_stream():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    label = (data.get("label") or f"Stream {len(state['streams']) + 1}").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    with _lock:
        if state["restarting"]:
            return jsonify({"error": "Pipeline restart in progress"}), 409
        was_running = state["running"]
        if was_running:
            state["restarting"] = True
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
        if state["restarting"]:
            return jsonify({"error": "Pipeline restart in progress"}), 409
        was_running = state["running"]
        if was_running:
            state["restarting"] = True
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
        if state["restarting"]:
            return jsonify({"error": "Pipeline restart in progress"}), 409
        was_running = state["running"]
        if was_running and "url" in data:
            state["restarting"] = True
        if "url" in data:
            state["streams"][idx]["url"] = data["url"].strip()
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
        if state["running"] or state["restarting"]:
            return jsonify({"error": "Already running or restarting"}), 409
        if not state["streams"]:
            return jsonify({"error": "Add at least one stream first"}), 400
        try:
            import os

            for stream in state["streams"]:
                cap = cv2.VideoCapture(stream["url"])
                opened = cap.isOpened()
                cap.release()
                if not opened:
                    raise ValueError(f"Cannot open source: {stream['label']}")
            mode = os.environ.get("DETECTION_MODE", "combined")
            if mode not in ("combined", "scene_only"):
                raise ValueError("DETECTION_MODE must be combined or scene_only")
            # Scene model loads once in the inference process. Its exact load
            # failure is published as unavailable while person tracking runs.
            _start_pipeline(state["streams"])
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
    socketio.emit(
        "pipeline_status",
        {
            "running": True,
            "restarting": False,
            "num_streams": len(state["streams"]),
        },
    )
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with _lock:
        if state["restarting"]:
            return jsonify({"error": "Pipeline restart in progress"}), 409
        _stop_pipeline()
    socketio.emit("pipeline_status", {"running": False, "restarting": False})
    return jsonify({"ok": True})


@app.route("/api/alerts")
def api_alerts():
    limit = int(request.args.get("limit", 100))
    stream_id = request.args.get("stream_id")
    with alert_lock:
        filtered = (
            [a for a in alerts if str(a["stream_id"]) == str(stream_id)]
            if stream_id is not None
            else list(alerts)
        )
        recent = list(reversed(filtered[-limit:]))
    return jsonify({"alerts": recent, "total": len(filtered)})


@app.route("/api/behavior_alerts")
def api_behavior_alerts():
    limit = int(request.args.get("limit", 100))
    stream_id = request.args.get("stream_id")
    with behavior_alert_lock:
        filtered = (
            [a for a in behavior_alerts if str(a["stream_id"]) == str(stream_id)]
            if stream_id is not None
            else list(behavior_alerts)
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
            "running": state["running"],
            "restarting": state["restarting"],
            "streams": state["streams"],
        }
        cached = list(state["latest_meta"].values())
    socketio.emit("initial_state", initial, to=request.sid)
    for payload in cached:
        socketio.emit("frame_meta", payload, to=request.sid)
    with behavior_alert_lock:
        scene_history = [a for a in behavior_alerts if a.get("scope") == "scene"]
    if scene_history:
        socketio.emit("scene_history", scene_history, to=request.sid)


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
        socketio.run(
            app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
        )
    finally:
        with _lock:
            _stop_pipeline()
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        print("[Server] Shutdown complete.")
