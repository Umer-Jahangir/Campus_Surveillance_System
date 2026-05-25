import cv2
import time
import numpy as np
import os
import psutil
import queue
import supervision as sv

from multiprocessing import shared_memory

from detector import Detector
from utils import now, latency_ms
from fight_detector_onnx import FightDetectorONNX


# ============================================================
# CPU CORE ASSIGNMENT
# ============================================================

def get_pcore_logical_ids():
    freqs = psutil.cpu_freq(percpu=True)
    if not freqs:
        total = psutil.cpu_count(logical=True)
        return list(range(total))
    max_freq = max(f.max for f in freqs if f is not None)
    threshold = max_freq * 0.90
    pcores = [i for i, f in enumerate(freqs) if f is not None and f.max >= threshold]
    return pcores


def assign_cores_hybrid(worker_index, num_decoders, cores_per_decoder, is_inference=False):
    total_logical = psutil.cpu_count(logical=True)
    pcores = get_pcore_logical_ids()
    ecores = [i for i in range(total_logical) if i not in pcores]
    if is_inference:
        return pcores if pcores else list(range(total_logical))
    start = worker_index * cores_per_decoder
    end = start + cores_per_decoder
    if end > len(ecores):
        return list(range(total_logical))
    return ecores[start:end]


# ============================================================
# DECODER WORKER
# ============================================================

def decoder_worker(
    index,
    stream_url,
    shm_names,
    active_buf_idx,
    frame_shape,
    meta_queue,
    stop_event,
    inference_alive,
    num_decoders,
    cores_per_decoder,
):
    process = psutil.Process()
    try:
        core_ids = assign_cores_hybrid(index, num_decoders, cores_per_decoder, is_inference=False)
        process.cpu_affinity(core_ids)
        print(f"[Decoder {index}] cores={core_ids}")
    except Exception as e:
        print(f"[Decoder {index}] affinity error: {e}")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

    def open_stream(url):
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    cap = open_stream(stream_url)
    if not cap.isOpened():
        print(f"[Decoder {index}] failed to open stream")
        return

    try:
        shm_slots = [shared_memory.SharedMemory(name=n) for n in shm_names]
        frame_buffers = [
            np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
            for shm in shm_slots
        ]
    except Exception as e:
        print(f"[Decoder {index}] shared memory error: {e}")
        cap.release()
        return

    print(f"[Decoder {index}] waiting for inference")
    while not stop_event.is_set() and not inference_alive.is_set():
        time.sleep(0.05)

    reconnect_delay = 2
    buf_idx = 0
    dropped = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if os.path.isfile(stream_url):
                print(f"[Decoder {index}] EOF")
                break
            print(f"[Decoder {index}] reconnecting...")
            cap.release()
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 10)
            cap = open_stream(stream_url)
            continue

        reconnect_delay = 2
        frame = cv2.resize(
            frame, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_LINEAR
        )
        np.copyto(frame_buffers[buf_idx], frame)
        active_buf_idx.value = buf_idx

        meta = {"stream_id": index, "buf_idx": buf_idx, "timestamp": now()}
        try:
            if meta_queue.full():
                try:
                    meta_queue.get_nowait()
                except Exception:
                    pass
            meta_queue.put_nowait(meta)
        except queue.Full:
            dropped += 1

        buf_idx ^= 1

    cap.release()
    for shm in shm_slots:
        shm.close()
    print(f"[Decoder {index}] stopped dropped={dropped}")


# ============================================================
# KEYPOINT HELPERS
# ============================================================

def extract_keypoints(result):
    """Return keypoints array shape (N, 17, 3) or (N, 17, 2)."""
    if result is None or result.keypoints is None:
        return None
    kps = result.keypoints
    xy = kps.xy.cpu().numpy()           # (N, 17, 2)
    if kps.conf is not None:
        conf = kps.conf.cpu().numpy()[..., np.newaxis]   # (N, 17, 1)
        return np.concatenate([xy, conf], axis=-1).astype(np.float32)
    ones = np.ones((*xy.shape[:2], 1), dtype=np.float32)
    return np.concatenate([xy, ones], axis=-1)


def _iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def iou_match_keypoints(tracked_boxes, orig_boxes, keypoints, iou_threshold=0.1):
    """Match each tracked box to the detection box with highest IoU."""
    if keypoints is None or len(orig_boxes) == 0:
        return [None] * len(tracked_boxes)
    matched = []
    for t_box in tracked_boxes:
        best_iou, best_kp = iou_threshold, None
        for oi, o_box in enumerate(orig_boxes):
            score = _iou(t_box, o_box)
            if score > best_iou:
                best_iou, best_kp = score, keypoints[oi]
        matched.append(best_kp)
    return matched


# ============================================================
# MULTI-PERSON FIGHT GROUPING  ← key new logic
# ============================================================

def _box_center(box):
    """Return (cx, cy) of a xyxy bounding box."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _box_diag(box) -> float:
    """Diagonal length of a bounding box – used as a person-size proxy."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return np.sqrt(w * w + h * h)


def _persons_proximate(box_a, box_b, proximity_factor: float = 1.2) -> bool:
    """
    True when two persons are close enough to be physically interacting.

    Proximity criterion
    -------------------
    The Euclidean distance between bounding-box centres must be less than
    proximity_factor × (average diagonal of the two boxes).

    A factor of 1.2 means the centres must be within ~1.2 body-lengths of
    each other – tight enough to catch genuine contact without firing on
    people merely standing in the same room.

    Alternatively we also accept boxes that directly overlap (IoU > 0).
    """
    if _iou(box_a, box_b) > 0.0:          # boxes already touching/overlapping
        return True
    cx_a, cy_a = _box_center(box_a)
    cx_b, cy_b = _box_center(box_b)
    dist = np.hypot(cx_a - cx_b, cy_a - cy_b)
    avg_diag = (_box_diag(box_a) + _box_diag(box_b)) / 2.0
    return dist < avg_diag * proximity_factor


def detect_fight_groups(
    track_ids,
    boxes,
    fight_detector: FightDetectorONNX,
    proximity_factor: float = 1.2,
) -> list[dict]:
    """
    Detect fight groups using an ASYMMETRIC threshold rule.

    Why asymmetric?
    ---------------
    In a real fight the aggressor/attacker typically drives the violent
    pose sequence, so their LSTM score surpasses the threshold.  The
    victim or bystander may be reacting or retreating — their pose
    sequence looks different and their score stays lower.  Requiring BOTH
    persons to be above threshold misses most real incidents (as seen in
    the logs: track 19 ≈ 0.95, track 20 ≈ 0.07-0.19).

    Rule
    ----
    A fight group is formed when:
      1. At least ONE person's LSTM score ≥ threshold  (the "initiator"), AND
      2. At least ONE OTHER person is spatially proximate to them
         (any score — they are the "other party").

    The alert carries ALL members of the group so the dashboard can
    highlight every involved bounding box in red.

    Parameters
    ----------
    track_ids        : array-like of integer track IDs (aligned with boxes)
    boxes            : (N, 4) xyxy bounding boxes
    fight_detector   : FightDetectorONNX instance for this stream
    proximity_factor : passed to _persons_proximate (default 1.2)

    Returns
    -------
    List of alert dicts (one per distinct group):
      {"type": "fight", "confidence": float,
       "track_ids": [int, ...], "num_persons": int}
    """
    if track_ids is None or len(track_ids) == 0:
        return []

    tids = [int(t) for t in track_ids]

    # Need ≥ 2 persons in the frame at all
    if len(tids) < 2:
        return []

    # Build full lookup: tid → box, tid → score
    tid_to_box   = {int(tid): boxes[bi] for bi, tid in enumerate(tids)}
    tid_to_score = {tid: fight_detector.get_score(tid) for tid in tids}

    # Persons whose score is above the detection threshold
    initiators = [tid for tid in tids if fight_detector.above_threshold(tid)]
    if not initiators:
        return []   # nobody is showing fight behaviour at all

    # For each initiator, collect all proximate persons (any score)
    # Use frozenset keys to deduplicate groups that share the same members
    seen_groups: dict[frozenset, list[int]] = {}

    for initiator in initiators:
        if initiator not in tid_to_box:
            continue
        group = [initiator]
        for other in tids:
            if other == initiator or other not in tid_to_box:
                continue
            if _persons_proximate(
                tid_to_box[initiator], tid_to_box[other], proximity_factor
            ):
                group.append(other)

        if len(group) >= 2:
            key = frozenset(group)
            if key not in seen_groups:
                seen_groups[key] = group

    alerts = []
    for members in seen_groups.values():
        # Confidence = highest individual score within the group
        conf = max(tid_to_score[tid] for tid in members)
        alerts.append(
            {
                "type":       "fight",
                "confidence": conf,
                "track_ids":  members,
                "num_persons": len(members),
            }
        )
        print(
            f"🚨 FIGHT DETECTED  "
            f"initiator(s)={[t for t in members if fight_detector.above_threshold(t)]}  "
            f"all_persons={members}  conf={conf:.4f}"
        )

    return alerts


# ============================================================
# INFERENCE WORKER
# ============================================================

def inference_worker(
    num_streams,
    frame_shape,
    shm_names,
    meta_queues,
    result_queue,
    stop_event,
    inference_alive,
    num_decoders,
    cores_per_decoder,
):
    process = psutil.Process()
    try:
        core_ids = assign_cores_hybrid(0, num_decoders, cores_per_decoder, is_inference=True)
        process.cpu_affinity(core_ids)
        print(f"[Inference] cores={core_ids}")
    except Exception as e:
        print(f"[Inference] affinity error: {e}")

    detector = Detector()
    class_names = detector.model.names

    # Shared memory
    shm_handles = []
    frame_buffers = []
    for pair in shm_names:
        slots = [shared_memory.SharedMemory(name=n) for n in pair]
        shm_handles.append(slots)
        frame_buffers.append(
            [np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf) for shm in slots]
        )

    detector.warmup(3)

    # Flush stale metadata
    for mq in meta_queues:
        while True:
            try:
                mq.get_nowait()
            except queue.Empty:
                break

    trackers = [
        sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        for _ in range(num_streams)
    ]

    # One LSTM fight detector per stream
    onnx_model_path = "D:/Projects/real_time_vedio/src/models/lstm-violence-detection.onnx"
    fight_detectors = [
        FightDetectorONNX(onnx_model_path, seq_len=20, threshold=0.7)
        for _ in range(num_streams)
    ]

    cleanup_counter = 0
    inference_alive.set()
    print("[Inference] ready (multi-person fight detection)")
    dropped_results = 0

    while not stop_event.is_set():
        batch_frames = []
        batch_meta = []

        # Collect latest frame from each stream
        for i, mq in enumerate(meta_queues):
            latest = None
            while True:
                try:
                    latest = mq.get_nowait()
                except queue.Empty:
                    break
            if latest is None:
                continue
            if latency_ms(latest["timestamp"], now()) > 1000:
                continue
            buf_idx = latest["buf_idx"]
            batch_frames.append(frame_buffers[i][buf_idx])
            batch_meta.append(latest)

        if not batch_frames:
            time.sleep(0.001)
            continue

        # YOLOv8-pose inference
        infer_start = now()
        try:
            results = detector.detect_raw(batch_frames)
        except Exception as e:
            print(f"[Inference] detect error: {e}")
            continue
        total_ms = latency_ms(infer_start, now())
        per_frame_ms = total_ms / max(len(batch_frames), 1)

        for idx, result in enumerate(results):
            meta = batch_meta[idx]
            stream_id = meta["stream_id"]
            fd = fight_detectors[stream_id]

            boxes = np.empty((0, 4), dtype=np.float32)
            track_ids = None
            classes = np.array([], dtype=int)
            confs = np.array([], dtype=np.float32)
            labels = []
            behavior_alerts = []

            if result is not None and result.boxes is not None:
                rb = result.boxes
                orig_boxes = rb.xyxy.cpu().numpy()
                orig_classes = rb.cls.cpu().numpy().astype(int)
                orig_confs = rb.conf.cpu().numpy()
                keypoints = extract_keypoints(result)   # (N, 17, 2or3)

                # ── filter: persons only ─────────────────────────────
                person_mask = orig_classes == 0
                orig_boxes = orig_boxes[person_mask]
                orig_classes = orig_classes[person_mask]
                orig_confs = orig_confs[person_mask]
                if keypoints is not None:
                    keypoints = keypoints[person_mask]

                # ── filter: confidence ───────────────────────────────
                conf_mask = orig_confs > 0.35
                orig_boxes = orig_boxes[conf_mask]
                orig_classes = orig_classes[conf_mask]
                orig_confs = orig_confs[conf_mask]
                if keypoints is not None:
                    keypoints = keypoints[conf_mask]

                if len(orig_boxes) == 0:
                    # no persons in this frame
                    pass
                else:
                    labels = [class_names[int(c)] for c in orig_classes]

                    detections = sv.Detections(
                        xyxy=orig_boxes,
                        confidence=orig_confs,
                        class_id=orig_classes,
                    )
                    tracked = trackers[stream_id].update_with_detections(detections)

                    boxes = tracked.xyxy
                    track_ids = tracked.tracker_id
                    classes = tracked.class_id
                    confs = tracked.confidence

                    # Match pose keypoints from detections → tracked boxes
                    matched_kps = iou_match_keypoints(boxes, orig_boxes, keypoints)

                    # ── Step A: update LSTM buffers per person ────────
                    if track_ids is not None:
                        for bi, tid in enumerate(track_ids):
                            tid = int(tid)
                            kp = matched_kps[bi]
                            # kp shape: (17, 2or3); drop conf column → (17,2) → (34,)
                            if kp is not None and kp.shape[1] >= 2:
                                kp_xy = kp[:, :2]    # (17, 2)
                                fd.update(tid, kp_xy)
                                # Note: return value (score) is stored inside fd;
                                # we query fd.get_score() in the grouping step.

                    # ── Step B: group proximate persons around any high-scorer
                    #
                    # Asymmetric rule: ONE person above threshold + ONE
                    # proximate person (any score) = fight.  The victim /
                    # bystander does not need to show the same LSTM signature.
                    behavior_alerts = detect_fight_groups(
                        track_ids=track_ids,
                        boxes=boxes,
                        fight_detector=fd,
                        proximity_factor=1.2,   # ← tweak if needed
                    )

            # ── Housekeeping every 30 frames ──────────────────────────
            cleanup_counter += 1
            if cleanup_counter % 30 == 0 and track_ids is not None:
                active_ids = [int(t) for t in track_ids if t is not None]
                fd.cleanup(active_ids)

            e2e_lat = latency_ms(meta["timestamp"], now())

            # ── Build per-track activity dict for the overlay ─────────
            activities: dict[int, str] = {}
            if track_ids is not None:
                # Collect ALL track IDs that appear in ANY fight group
                fighting_ids: set[int] = set()
                for alert in behavior_alerts:
                    fighting_ids.update(int(t) for t in alert["track_ids"])

                for tid in track_ids:
                    if tid is not None:
                        activities[int(tid)] = (
                            "fight" if int(tid) in fighting_ids else ""
                        )

            msg = (
                "frame",
                stream_id,
                boxes,
                track_ids,
                classes,
                confs,
                labels,
                meta["timestamp"],
                per_frame_ms,
                e2e_lat,
                behavior_alerts,
                activities,
            )

            try:
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except Exception:
                        pass
                result_queue.put_nowait(msg)
            except queue.Full:
                dropped_results += 1

        time.sleep(0.001)

    for pair in shm_handles:
        for shm in pair:
            shm.close()
    print(f"[Inference] stopped  dropped={dropped_results}")