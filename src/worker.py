import cv2
import time
import numpy as np
import os
import psutil
import queue
import supervision as sv
from multiprocessing import shared_memory

from detector import Detector
from basic_behavior import BehaviorAnalyzer
from utils import now, latency_ms


# ============================================================
# CPU CORE ASSIGNMENT
# ============================================================

def get_pcore_logical_ids():
    """Return list of logical processor IDs that belong to P-cores (heuristic)."""
    freqs = psutil.cpu_freq(percpu=True)
    if not freqs or len(freqs) < 2:
        total_logical = psutil.cpu_count(logical=True)
        return list(range(4)) if total_logical > 8 else list(range(total_logical))

    max_freq  = max(f.max for f in freqs if f is not None)
    threshold = max_freq * 0.9
    pcores    = [i for i, f in enumerate(freqs)
                 if f is not None and f.max >= threshold]
    return pcores


def assign_cores_hybrid(worker_index, num_decoders, cores_per_decoder,
                        is_inference=False):
    """
    Assign logical cores intelligently:
    - Inference gets all P-cores.
    - Decoders get E-cores, round-robin.
    """
    total_logical = psutil.cpu_count(logical=True)
    pcore_ids     = get_pcore_logical_ids()
    ecore_ids     = [i for i in range(total_logical) if i not in pcore_ids]

    if is_inference:
        return pcore_ids

    start = worker_index * cores_per_decoder
    end   = start + cores_per_decoder
    if end > len(ecore_ids):
        print(f"Warning: insufficient E-cores for decoder {worker_index}, "
              f"using any cores.")
        all_cores = list(range(total_logical))
        return all_cores[start:end]
    return ecore_ids[start:end]


# ============================================================
# DECODER WORKER
# ============================================================

def decoder_worker(index, stream_url, shm_names, frame_shape,
                   meta_queue, stop_event, inference_alive,
                   num_decoders, cores_per_decoder):
    """
    shm_names: (name_a, name_b) — two shared memory slots for double-buffering.
    The decoder alternates between them each frame so inference always reads
    a stable, fully-written buffer.
    """

    p = psutil.Process()

    try:
        core_ids = assign_cores_hybrid(index, num_decoders, cores_per_decoder,
                                       is_inference=False)
        p.cpu_affinity(core_ids)
        print(f"[Decoder {index}] Pinned to cores {core_ids}")
    except Exception as e:
        print(f"[Decoder {index}] Affinity error: {e}")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    os.environ["OPENCV_LOG_LEVEL"]              = "ERROR"

    def open_stream(url):
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    cap = open_stream(stream_url)

    if not cap.isOpened():
        print(f"[Decoder {index}] Failed to open stream")
        return

    try:
        shm_slots     = [shared_memory.SharedMemory(name=n) for n in shm_names]
        frame_buffers = [
            np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
            for shm in shm_slots
        ]
    except Exception as e:
        print(f"[Decoder {index}] Shared memory error: {e}")
        cap.release()
        return

    print(f"[Decoder {index}] Waiting for inference...")

    while not stop_event.is_set() and not inference_alive.is_set():
        time.sleep(0.1)

    if stop_event.is_set():
        cap.release()
        for shm in shm_slots:
            shm.close()
        return

    print(f"[Decoder {index}] Started for {stream_url}")

    drop_count      = 0
    stall_count     = 0
    reconnect_delay = 2

    pace_sleep = 0.033
    PACE_MIN   = 0.02
    PACE_MAX   = 0.08
    ALPHA_UP   = 0.05
    ALPHA_DOWN = 0.15

    READ_STALL_THRESHOLD_MS = 80.0

    if os.path.isfile(stream_url):
        source_fps         = cap.get(cv2.CAP_PROP_FPS)
        file_frame_interval = (1.0 / source_fps) if source_fps > 0 else 0.033
        print(f"[Decoder {index}] Local file — pacing to {source_fps:.1f} FPS "
              f"({file_frame_interval*1000:.1f}ms/frame)")
    else:
        file_frame_interval = 0

    buf_idx = 0

    while not stop_event.is_set():

        if not inference_alive.is_set():
            time.sleep(0.5)
            continue

        read_start = time.perf_counter()
        ret, frame = cap.read()
        read_ms    = (time.perf_counter() - read_start) * 1000

        if not ret:
            if os.path.isfile(stream_url):
                print(f"[Decoder {index}] End of file")
                break

            print(f"[Decoder {index}] Reconnecting in {reconnect_delay}s")
            cap.release()
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 10)
            cap = open_stream(stream_url)
            if cap.isOpened():
                reconnect_delay = 2
            continue

        if read_ms > READ_STALL_THRESHOLD_MS:
            stall_count += 1
            if stall_count % 20 == 0:
                print(f"[Decoder {index}] stalled reads={stall_count} "
                      f"last={read_ms:.0f}ms")
            continue

        frame_resized = cv2.resize(
            frame,
            (frame_shape[1], frame_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        np.copyto(frame_buffers[buf_idx], frame_resized)

        sent = False
        try:
            meta_queue.put_nowait({
                "stream_id": index,
                "buf_idx":   buf_idx,
                "timestamp": now()
            })
            sent = True
        except queue.Full:
            drop_count += 1
            if drop_count % 100 == 0:
                print(f"[Decoder {index}] dropped {drop_count}")

        buf_idx ^= 1

        sample_interval = 5
        if (sent and not stop_event.is_set()) or (not sent):
            if file_frame_interval > 0:
                time.sleep(file_frame_interval)
            else:
                if drop_count % sample_interval == 0:
                    try:
                        q_size = meta_queue.qsize()
                    except Exception:
                        q_size = 0

                    if not sent or q_size > 2:
                        pace_sleep = min(pace_sleep * (1 + ALPHA_DOWN), PACE_MAX)
                    elif q_size == 0:
                        pace_sleep = max(pace_sleep * (1 - ALPHA_UP), PACE_MIN)

                time.sleep(pace_sleep)

    cap.release()
    for shm in shm_slots:
        shm.close()

    print(f"[Decoder {index}] Stopped | "
          f"Dropped={drop_count} | Stalled={stall_count}")


# ============================================================
# KEYPOINT HELPERS
# ============================================================

def _extract_keypoints(result) -> np.ndarray | None:
    """
    Return keypoints as np.ndarray shape (N, 17, 3) [x, y, conf]
    or None when the model didn't produce keypoints.
    """
    if result is None or result.keypoints is None:
        return None

    kps = result.keypoints
    xy  = kps.xy.cpu().numpy()   # (N, 17, 2)

    if kps.conf is not None:
        conf = kps.conf.cpu().numpy()[..., np.newaxis]   # (N, 17, 1)
        return np.concatenate([xy, conf], axis=-1).astype(np.float32)  # (N,17,3)

    # No confidence channel — pad with 1.0
    ones = np.ones((*xy.shape[:2], 1), dtype=np.float32)
    return np.concatenate([xy, ones], axis=-1)


def _match_keypoints_to_tracked(orig_boxes: np.ndarray,
                                 keypoints:   np.ndarray | None,
                                 tracked_boxes: np.ndarray) -> list[np.ndarray | None]:
    """
    Match tracked bounding boxes back to the original detection order
    (supervision may reorder / drop rows during ByteTrack update).

    Returns a list of length len(tracked_boxes), each element is a
    (17, 3) keypoint array or None.
    """
    if keypoints is None or len(orig_boxes) == 0:
        return [None] * len(tracked_boxes)

    matched = []
    for tb in tracked_boxes:
        tx1, ty1, tx2, ty2 = tb
        tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0

        best_idx  = -1
        best_dist = float("inf")
        for i, ob in enumerate(orig_boxes):
            ox1, oy1, ox2, oy2 = ob
            ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
            d = (tcx - ocx) ** 2 + (tcy - ocy) ** 2
            if d < best_dist:
                best_dist = d
                best_idx  = i

        if best_idx >= 0 and best_idx < len(keypoints):
            matched.append(keypoints[best_idx])
        else:
            matched.append(None)

    return matched


# ============================================================
# INFERENCE WORKER
# ============================================================

def inference_worker(num_streams, frame_shape, shm_names,
                     meta_queues, result_queue, stop_event,
                     inference_alive,
                     num_decoders, cores_per_decoder):
    """
    shm_names: list of (name_a, name_b) tuples, one per stream.
    Uses buf_idx from metadata to read the correct double-buffer slot.
    """

    p = psutil.Process()

    try:
        core_ids = assign_cores_hybrid(0, num_decoders, cores_per_decoder,
                                       is_inference=True)
        if not core_ids:
            core_ids = [psutil.cpu_count(logical=False) - 1]
        p.cpu_affinity(core_ids)
        print(f"[Inference] Pinned to cores {core_ids}")
    except Exception as e:
        print(f"[Inference] Affinity error: {e}")

    detector    = Detector()
    class_names = detector.model.names

    # Open 2 shm handles per stream
    shm_handles   = []
    frame_buffers = []
    for name_pair in shm_names:
        slots = [shared_memory.SharedMemory(name=n) for n in name_pair]
        shm_handles.append(slots)
        frame_buffers.append([
            np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
            for shm in slots
        ])

    detector.warmup(3)

    for mq in meta_queues:
        while True:
            try:    mq.get_nowait()
            except queue.Empty: break

    # Per-stream trackers and behavior analyzers
    trackers          = [sv.ByteTrack()       for _ in range(num_streams)]
    behavior_analyzers = [BehaviorAnalyzer()  for _ in range(num_streams)]

    inference_alive.set()
    print("[Inference] Ready — decoders can start")

    result_drop_count = 0

    while not stop_event.is_set():

        batch_frames   = []
        batch_metadata = []

        for i, mq in enumerate(meta_queues):
            try:
                meta = mq.get_nowait()
                age  = latency_ms(meta["timestamp"], now())
                if age > 500:
                    continue

                buf_idx = meta["buf_idx"]
                frame   = frame_buffers[i][buf_idx].copy()
                batch_frames.append(frame)
                batch_metadata.append(meta)
            except queue.Empty:
                pass

        if not batch_frames:
            stop_event.wait(timeout=0.001)
            continue

        start   = now()
        results = detector.detect_raw(batch_frames)
        total_ms      = latency_ms(start, now())
        per_frame_ms  = total_ms / len(batch_frames)

        for i, res in enumerate(results):
            meta      = batch_metadata[i]
            stream_id = meta["stream_id"]

            boxes     = None
            classes   = None
            confs     = None
            labels    = None
            track_ids = None
            behavior_alerts = []

            # res can be None when single-frame fallback also fails
            if res is None or res.boxes is None:
                pass   # boxes/track_ids remain None; emit frame with no detections
            else:
                # ── raw detection arrays ───────────────────────────────
                orig_boxes = res.boxes.xyxy.cpu().numpy()
                classes    = res.boxes.cls.cpu().numpy().astype(int)
                confs      = res.boxes.conf.cpu().numpy()
                labels     = [class_names[int(c)] for c in classes]

                # ── extract keypoints (pose model) ─────────────────────
                keypoints = _extract_keypoints(res)   # (N, 17, 3) or None

                # ── ByteTrack ──────────────────────────────────────────
                detections = sv.Detections(
                    xyxy=orig_boxes,
                    confidence=confs,
                    class_id=classes,
                )
                tracked   = trackers[stream_id].update_with_detections(detections)

                boxes     = tracked.xyxy
                classes   = tracked.class_id
                confs     = tracked.confidence
                track_ids = tracked.tracker_id

                # ── match keypoints to tracked boxes ───────────────────
                kp_per_track = _match_keypoints_to_tracked(
                    orig_boxes, keypoints, boxes
                )

                # ── behavior analysis ──────────────────────────────────
                analyzer = behavior_analyzers[stream_id]

                for bi, tid in enumerate(track_ids):
                    kp = kp_per_track[bi]
                    analyzer.update(int(tid), boxes[bi].tolist(), keypoints=kp)

                    alert = analyzer.get_individual_alert(int(tid))
                    if alert:
                        behavior_alerts.append(alert)

                # crowd alert (stream-level)
                crowd = analyzer.detect_crowd()
                if crowd:
                    behavior_alerts.append(crowd)

                # clean up ghost tracks
                analyzer.cleanup(timeout=5.0)

            # ── build result message ───────────────────────────────────
            e2e_lat = latency_ms(meta["timestamp"], now())
            msg = (
                "frame",
                stream_id,
                batch_frames[i],
                boxes,
                track_ids,
                classes,
                confs,
                labels,
                meta["timestamp"],
                per_frame_ms,
                e2e_lat,
                behavior_alerts,     # ← NEW: list of behavior alert dicts
            )

            try:
                result_queue.put_nowait(msg)
            except queue.Full:
                result_drop_count += 1
                if result_drop_count % 50 == 0:
                    print(f"[Inference] dropped results={result_drop_count}")

    for slot_pair in shm_handles:
        for shm in slot_pair:
            shm.close()

    print(f"[Inference] Stopped | Drops={result_drop_count}")