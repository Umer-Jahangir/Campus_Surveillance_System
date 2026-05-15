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
    """
    Heuristic:
    Highest-frequency cores are assumed to be P-cores.
    """

    freqs = psutil.cpu_freq(percpu=True)

    if not freqs:
        total = psutil.cpu_count(logical=True)
        return list(range(total))

    max_freq = max(f.max for f in freqs if f is not None)
    threshold = max_freq * 0.90

    pcores = [
        i for i, f in enumerate(freqs)
        if f is not None and f.max >= threshold
    ]

    return pcores


def assign_cores_hybrid(
    worker_index,
    num_decoders,
    cores_per_decoder,
    is_inference=False,
):
    """
    Inference:
        → P-cores

    Decoders:
        → E-cores
    """

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
    """
    Responsibilities ONLY:
    - Decode frames
    - Resize
    - Write to shared memory
    - Commit active buffer index
    - Push lightweight metadata token

    NO:
    - pacing logic
    - queue buildup
    - frame transfer
    """

    process = psutil.Process()

    try:
        core_ids = assign_cores_hybrid(
            index,
            num_decoders,
            cores_per_decoder,
            is_inference=False,
        )

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
        shm_slots = [
            shared_memory.SharedMemory(name=n)
            for n in shm_names
        ]

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
            frame,
            (frame_shape[1], frame_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        np.copyto(frame_buffers[buf_idx], frame)

        # IMPORTANT:
        # Commit buffer ONLY AFTER write finishes
        active_buf_idx.value = buf_idx

        meta = {
            "stream_id": index,
            "buf_idx": buf_idx,
            "timestamp": now(),
        }

        try:

            # Latest-frame semantics
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

    if result is None:
        return None

    if result.keypoints is None:
        return None

    kps = result.keypoints

    xy = kps.xy.cpu().numpy()

    if kps.conf is not None:

        conf = kps.conf.cpu().numpy()[..., np.newaxis]

        return np.concatenate(
            [xy, conf],
            axis=-1,
        ).astype(np.float32)

    ones = np.ones((*xy.shape[:2], 1), dtype=np.float32)

    return np.concatenate([xy, ones], axis=-1)


def _iou(box_a, box_b):
    """
    Compute IoU between two boxes in xyxy format.
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return 0.0

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def iou_match_keypoints(tracked_boxes, orig_boxes, keypoints, iou_threshold=0.1):
    """
    For every tracked box, find the best-matching original YOLO detection
    by IoU and return the corresponding keypoint.

    Parameters
    ----------
    tracked_boxes : np.ndarray, shape (N, 4)   — output of ByteTrack (xyxy)
    orig_boxes    : np.ndarray, shape (M, 4)   — raw YOLO detections (xyxy)
    keypoints     : np.ndarray, shape (M, K, 3) or None
    iou_threshold : float — minimum IoU to accept a match

    Returns
    -------
    matched_kps : list of length N
        matched_kps[i] is the keypoint array for tracked_boxes[i],
        or None if no match was found above the threshold.
    """

    if keypoints is None or len(orig_boxes) == 0:
        return [None] * len(tracked_boxes)

    matched_kps = []

    for t_box in tracked_boxes:

        best_iou = iou_threshold
        best_kp = None

        for oi, o_box in enumerate(orig_boxes):

            score = _iou(t_box, o_box)

            if score > best_iou:
                best_iou = score
                best_kp = keypoints[oi]

        matched_kps.append(best_kp)

    return matched_kps


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
    """
    Responsibilities ONLY:
    - Read latest frames
    - Inference
    - Tracking
    - Behavior analysis
    - Emit lightweight metadata

    NO:
    - image transfer
    - visualization
    - annotation
    """

    process = psutil.Process()

    try:

        core_ids = assign_cores_hybrid(
            0,
            num_decoders,
            cores_per_decoder,
            is_inference=True,
        )

        process.cpu_affinity(core_ids)

        print(f"[Inference] cores={core_ids}")

    except Exception as e:
        print(f"[Inference] affinity error: {e}")

    detector = Detector()

    class_names = detector.model.names

    # ========================================================
    # Shared Memory
    # ========================================================

    shm_handles = []
    frame_buffers = []

    for pair in shm_names:

        slots = [
            shared_memory.SharedMemory(name=n)
            for n in pair
        ]

        shm_handles.append(slots)

        frame_buffers.append([
            np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
            for shm in slots
        ])

    detector.warmup(3)

    # Flush stale metadata
    for mq in meta_queues:
        while True:
            try:
                mq.get_nowait()
            except queue.Empty:
                break

    trackers = [
        sv.ByteTrack()
        for _ in range(num_streams)
    ]

    analyzers = [
        BehaviorAnalyzer()
        for _ in range(num_streams)
    ]

    cleanup_counter = 0

    inference_alive.set()

    print("[Inference] ready")

    dropped_results = 0

    while not stop_event.is_set():

        batch_frames = []
        batch_meta = []

        # ====================================================
        # Collect latest frames
        # ====================================================

        for i, mq in enumerate(meta_queues):

            latest = None

            while True:

                try:
                    latest = mq.get_nowait()
                except queue.Empty:
                    break

            if latest is None:
                continue

            age = latency_ms(latest["timestamp"], now())

            if age > 1000:
                continue

            buf_idx = latest["buf_idx"]

            frame = frame_buffers[i][buf_idx]

            batch_frames.append(frame)
            batch_meta.append(latest)

        if not batch_frames:
            time.sleep(0.001)
            continue

        # ====================================================
        # Inference
        # ====================================================

        infer_start = now()

        results = detector.detect_raw(batch_frames)

        total_ms = latency_ms(infer_start, now())

        per_frame_ms = total_ms / max(len(batch_frames), 1)

        # ====================================================
        # Process Results
        # ====================================================

        for idx, result in enumerate(results):

            meta = batch_meta[idx]

            stream_id = meta["stream_id"]

            boxes = []
            classes = []
            confs = []
            labels = []
            track_ids = []

            activities = {}

            behavior_alerts = []

            if result is not None and result.boxes is not None:

                rb = result.boxes

                orig_boxes = rb.xyxy.cpu().numpy()

                classes = rb.cls.cpu().numpy().astype(int)

                confs = rb.conf.cpu().numpy()

                labels = [
                    class_names[int(c)]
                    for c in classes
                ]

                keypoints = extract_keypoints(result)

                detections = sv.Detections(
                    xyxy=orig_boxes,
                    confidence=confs,
                    class_id=classes,
                )

                tracked = trackers[stream_id].update_with_detections(
                    detections
                )

                boxes = tracked.xyxy
                track_ids = tracked.tracker_id
                classes = tracked.class_id
                confs = tracked.confidence

                # IoU-match tracked boxes → original YOLO pose detections.
                # ByteTrack can reorder, drop, or interpolate detections, so
                # the tracked index `bi` no longer maps 1-to-1 with the YOLO
                # output index. Using `bi` directly would assign the wrong
                # keypoints to the wrong person IDs and silently corrupt every
                # downstream history and classifier. We resolve the ambiguity
                # with IoU matching instead.
                matched_keypoints = iou_match_keypoints(
                    boxes,
                    orig_boxes,
                    keypoints,
                )

                analyzer = analyzers[stream_id]

                if track_ids is not None:

                    for bi, tid in enumerate(track_ids):

                        tid = int(tid)

                        kp = matched_keypoints[bi]

                        analyzer.update(
                            tid,
                            boxes[bi].tolist(),
                            keypoints=kp,
                        )

                        state = analyzer.classify(tid)

                        if state:
                            activities[tid] = state

                        alert = analyzer.get_individual_alert(tid)

                        if alert:
                            behavior_alerts.append(alert)

                crowd = analyzer.detect_crowd()

                if crowd:
                    behavior_alerts.append(crowd)

            # Cleanup less frequently
            cleanup_counter += 1

            if cleanup_counter % 30 == 0:
                analyzers[stream_id].cleanup(timeout=5.0)

            e2e_lat = latency_ms(
                meta["timestamp"],
                now(),
            )

            # ====================================================
            # LIGHTWEIGHT RESULT MESSAGE
            # ====================================================

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

    for pair in shm_handles:
        for shm in pair:
            shm.close()

    print(f"[Inference] stopped dropped={dropped_results}")