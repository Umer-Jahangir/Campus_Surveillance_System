from __future__ import annotations
import cv2
import time
import numpy as np
import os
import psutil
import queue


from utils import now, latency_ms
from pipeline_frames import prepare_frame, original_boxes, annotated_jpeg


# ============================================================
# CPU CORE ASSIGNMENT
# ============================================================


def get_cpu_topology() -> dict:
    """Detect logical/physical core counts and a best-effort P/E-core split."""
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical

    freqs = psutil.cpu_freq(percpu=True) or []
    pcores = []
    if freqs and len(freqs) >= 2:
        max_freq = max((f.max for f in freqs if f is not None), default=0.0)
        if max_freq > 0:
            threshold = max_freq * 0.90
            pcores = [
                i for i, f in enumerate(freqs) if f is not None and f.max >= threshold
            ]

    if not pcores:
        # Fallback for hybrid CPUs where per-core frequency data is unavailable.
        # Treat the first physical-core slots as the high-performance pool and
        # the remaining logical SMT slots as the efficiency pool.
        pcores = list(range(min(physical, logical)))

    ecores = [i for i in range(logical) if i not in pcores]
    return {
        "logical": logical,
        "physical": physical,
        "pcore_ids": pcores,
        "ecore_ids": ecores,
    }


def get_pcore_logical_ids():
    return get_cpu_topology()["pcore_ids"]


def assign_cores_hybrid(
    worker_index, num_decoders, cores_per_decoder, is_inference=False
):
    """
    Simple non-overlapping core plan for decoder + inference workers.

    - inference gets a dedicated P-core slice first
    - decoders use only the remaining cores, so they do not overlap
    - for 3 streams, each decoder gets about len(remaining) // 3 cores
    """
    topo = get_cpu_topology()
    logical = topo["logical"]
    pcores = topo["pcore_ids"]
    ecores = topo["ecore_ids"]

    # Reserve a small, stable inference pool from the fast cores.
    inference_pool = pcores[: max(2, min(len(pcores), logical // 2))]
    if not inference_pool:
        inference_pool = list(range(min(logical, 6)))

    decoder_pool = [i for i in range(logical) if i not in inference_pool]
    if not decoder_pool:
        decoder_pool = list(range(logical))

    if is_inference:
        selected = inference_pool
        print(
            f"[Topology] logical={logical} physical={topo['physical']} "
            f"pcores={pcores} ecores={ecores} inference_affinity={selected}"
        )
        return selected

    # Use the remaining cores only; this avoids overlap with inference.
    per_decoder = max(1, int(cores_per_decoder or 1))
    if num_decoders > 1:
        per_decoder = max(1, len(decoder_pool) // num_decoders)

    start = worker_index * per_decoder
    end = start + per_decoder
    selected = decoder_pool[start:end]

    if len(selected) < per_decoder:
        used = set(selected)
        fallback = [i for i in decoder_pool if i not in used]
        selected += fallback[: per_decoder - len(selected)]

    print(
        f"[Topology] logical={logical} physical={topo['physical']} "
        f"pcores={pcores} ecores={ecores} decoder_affinity={selected}"
    )
    return selected or decoder_pool


# ============================================================
# DECODER WORKER
# ============================================================


def decoder_worker(
    index,
    stream_url,
    frame_shape,
    meta_queue,
    stop_event,
    inference_alive,
    num_decoders,
    cores_per_decoder,
    scene_queue=None,
):
    process = psutil.Process()
    try:
        core_ids = assign_cores_hybrid(
            index, num_decoders, cores_per_decoder, is_inference=False
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

    print(f"[Decoder {index}] waiting for inference")
    while not stop_event.is_set() and not inference_alive.is_set():
        time.sleep(0.05)

    # Opening RTSP before model startup accumulated undecoded network frames.
    cap = open_stream(stream_url)
    if not cap.isOpened():
        print(f"[Decoder {index}] failed to open stream")
        return

    reconnect_delay = 2
    dropped = 0
    frame_id = 0
    epoch = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    last_source_time = None
    from scene_model import SceneSampler

    scene_sampler = SceneSampler()
    last_clip = None
    is_file = os.path.isfile(stream_url)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else None
    decoded_started = now()
    started = time.monotonic()

    while not stop_event.is_set():
        read_started = now()
        ret, frame = cap.read()
        decoded = now()
        if not ret:
            if os.path.isfile(stream_url):
                print(f"[Decoder {index}] EOF")
                break
            print(f"[Decoder {index}] reconnecting...")
            cap.release()
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 10)
            cap = open_stream(stream_url)
            epoch += 1
            continue

        reconnect_delay = 2
        source_time = (
            cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if os.path.isfile(stream_url)
            else time.monotonic()
        )
        if os.path.isfile(stream_url):
            # Prefer decoder presentation timestamps, including variable-frame-rate files.
            if (
                not np.isfinite(source_time)
                or source_time < 0
                or (last_source_time is not None and source_time <= last_source_time)
            ):
                source_time = max(frame_id / fps, (last_source_time or 0) + 1 / fps)
            last_source_time = source_time
            if os.environ.get("OFFLINE_REALTIME_PACING", "1") == "1":
                stop_event.wait(max(0, started + source_time - time.monotonic()))
        prep_started = now()
        scene_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        scene_clip = scene_sampler.update(scene_frame, source_time, epoch)
        if scene_queue is not None and not is_file:
            key = (epoch, scene_clip["end"] if scene_clip is not None else None)
            if key != last_clip:
                # Separate bounded immutable temporal packets: once per window,
                # never repeat the same 2.3 MiB clip on every display frame.
                packet = {
                    "epoch": epoch,
                    "clip": scene_clip,
                    "source_time": source_time,
                }
                try:
                    scene_queue.put_nowait(packet)
                    last_clip = key
                except queue.Full:
                    try:
                        scene_queue.get_nowait()
                    except queue.Empty:
                        pass
                    # Retry on the next decoded frame; do not silently mark sent.
        frame, geometry = prepare_frame(frame, frame_shape)
        # Queue owns this newly allocated array; decoder never mutates it again.
        meta = {
            "stream_id": index,
            "frame": frame,
            "timestamp": now(),
            "frame_id": frame_id,
            "source_time": source_time,
            "epoch": epoch,
            "geometry": geometry,
            "scene_clip": scene_clip,
            "source_fps": fps,
            "is_file": is_file,
            "total_frames": total_frames,
            "decode_ms": latency_ms(read_started, decoded),
            "prepare_ms": latency_ms(prep_started, now()),
            "decoded_fps": (frame_id + 1) / max(0.001, now() - decoded_started),
        }
        if scene_queue is not None and not is_file:
            meta["scene_clip"] = None
        frame_id += 1
        if meta["is_file"]:
            while not stop_event.is_set():
                try:
                    meta_queue.put(meta, timeout=0.1)
                    break
                except queue.Full:
                    continue
            continue
        try:
            if meta_queue.full():
                try:
                    meta_queue.get_nowait()
                except Exception:
                    pass
            meta_queue.put_nowait(meta)
        except queue.Full:
            dropped += 1

    cap.release()
    print(f"[Decoder {index}] stopped dropped={dropped}")


# ============================================================
# KEYPOINT HELPERS
# ============================================================


def extract_keypoints(result):
    """Return keypoints array shape (N, 17, 3) or (N, 17, 2)."""
    if result is None or result.keypoints is None:
        return None
    kps = result.keypoints
    xy = kps.xy.cpu().numpy()  # (N, 17, 2)
    if kps.conf is not None:
        conf = kps.conf.cpu().numpy()[..., np.newaxis]  # (N, 17, 1)
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
# OPTIONAL LEGACY POSE DIAGNOSTICS
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

    This geometric heuristic provides nearby-person context only. Proximity
    does not establish fighting or identify a participant.

    Alternatively we also accept boxes that directly overlap (IoU > 0).
    """
    if _iou(box_a, box_b) > 0.0:  # boxes already touching/overlapping
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
    """Report verified high scorers; nearby IDs are context, never labelled by proximity."""
    if track_ids is None or len(track_ids) == 0:
        return []

    tids = [int(t) for t in track_ids]

    # Need ≥ 2 persons in the frame at all
    if len(tids) < 2:
        return []

    # Build full lookup: tid → box, tid → score
    tid_to_box = {int(tid): boxes[bi] for bi, tid in enumerate(tids)}
    tid_to_score = {tid: fight_detector.get_score(tid) for tid in tids}

    # Persons whose score is above the detection threshold
    initiators = [tid for tid in tids if fight_detector.above_threshold(tid)]
    if not initiators:
        return []  # nobody is showing fight behaviour at all

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
                "type": "fight",
                "confidence": conf,
                "track_ids": [t for t in members if fight_detector.above_threshold(t)],
                "nearby_track_ids": [
                    t for t in members if not fight_detector.above_threshold(t)
                ],
                "num_persons": len(members),
            }
        )
        print(
            f"FIGHT DETECTED  "
            f"initiator(s)={[t for t in members if fight_detector.above_threshold(t)]}  "
            f"all_persons={members}  conf={conf:.4f}"
        )

    return alerts


# ============================================================
# INFERENCE WORKER
# ============================================================


def _run_inference(
    num_streams,
    frame_shape,
    meta_queues,
    result_queue,
    stop_event,
    inference_alive,
    num_decoders,
    cores_per_decoder,
    scene_queues=None,
):
    from fight_detector_onnx import FightDetectorONNX, DisabledPoseClassifier

    mode = os.environ.get("DETECTION_MODE", "combined")
    if mode not in ("combined", "scene_only"):
        raise ValueError("DETECTION_MODE must be combined or scene_only")
    detector = None
    if mode == "combined":
        import supervision as sv
        from detector import Detector
    process = psutil.Process()
    try:
        core_ids = assign_cores_hybrid(
            0, num_decoders, cores_per_decoder, is_inference=True
        )
        process.cpu_affinity(core_ids)
        print(f"[Inference] cores={core_ids}")
    except Exception as e:
        print(f"[Inference] affinity error: {e}")

    if mode == "combined":
        detector = Detector()
        detector.warmup(3)

    # Flush stale metadata
    for mq in meta_queues:
        while True:
            try:
                mq.get_nowait()
            except queue.Empty:
                break

    trackers = (
        [
            sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
            for _ in range(num_streams)
        ]
        if detector is not None
        else [None] * num_streams
    )

    # One LSTM fight detector per stream
    from pathlib import Path

    onnx_model_path = os.environ.get(
        "FIGHT_MODEL_PATH",
        str(Path(__file__).resolve().parent / "models/lstm-violence-detection.onnx"),
    )
    fight_detectors = [
        (
            FightDetectorONNX(
                onnx_model_path,
                seq_len=20,
                threshold=float(os.environ.get("FIGHT_THRESHOLD", "0.7")),
            )
            if detector is not None and os.environ.get("LEGACY_POSE_DIAGNOSTICS") == "1"
            else DisabledPoseClassifier()
        )
        for _ in range(num_streams)
    ]

    from scene_model import SceneModel, SceneWindow, configured_scene_path

    scene_model, scene_reason = None, None
    try:
        scene_model = SceneModel(configured_scene_path())
    except Exception as exc:
        scene_reason = str(exc)
        print(f"[Action unavailable] {scene_reason}")
    scene_windows = [SceneWindow(scene_model, scene_reason) for _ in range(num_streams)]

    last_meta = {}
    clip_cache = {}
    inference_alive.set()
    print(
        f"[Inference] ready mode={mode}; pose_instances={int(detector is not None)} scene_instances={int(scene_model is not None)}",
        flush=True,
    )
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
                    if latest.get("is_file"):
                        break
                except queue.Empty:
                    break
            if latest is None:
                continue
            if scene_queues is not None and not latest.get("is_file"):
                while True:
                    try:
                        clip_cache[i] = scene_queues[i].get_nowait()
                    except queue.Empty:
                        break
                cached = clip_cache.get(i)
                clip = (
                    cached["clip"]
                    if cached and cached["epoch"] == latest["epoch"]
                    else None
                )
                if clip is not None and clip["end"] > latest["source_time"]:
                    continue  # Wait for its matching-or-newer frame; no future overlay.
                latest["scene_clip"] = clip
            if (
                not latest.get("is_file")
                and latency_ms(latest["timestamp"], now()) > 1000
            ):
                continue
            batch_frames.append(latest["frame"])
            latest["queue_ms"] = latency_ms(latest["timestamp"], now())
            batch_meta.append(latest)

        if not batch_frames:
            time.sleep(0.001)
            continue

        # YOLOv8-pose inference
        infer_start = now()
        try:
            results = (
                detector.detect_raw(batch_frames)
                if detector is not None
                else [None] * len(batch_frames)
            )
        except Exception as e:
            raise RuntimeError("Pose inference failed; pipeline stopped") from e
        class_names = detector.model.names if detector is not None else {}
        total_ms = latency_ms(infer_start, now())
        per_frame_ms = total_ms / max(len(batch_frames), 1)

        for idx, result in enumerate(results):
            meta = batch_meta[idx]
            stream_id = meta["stream_id"]
            fd = fight_detectors[stream_id]
            action, scene_event = scene_windows[stream_id].update_clip(
                meta["scene_clip"], meta["source_time"], meta["epoch"]
            )
            previous = last_meta.get(stream_id)
            if previous and (
                meta["epoch"] != previous["epoch"]
                or meta["source_time"] <= previous["source_time"]
                or meta["source_time"] - previous["source_time"] > fd.max_gap
            ):
                if trackers[stream_id] is not None:
                    trackers[stream_id].reset()
                fd.cleanup([])
            last_meta[stream_id] = meta

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
                keypoints = extract_keypoints(result)  # (N, 17, 2or3)

                # ── filter: persons only ─────────────────────────────
                person_mask = orig_classes == 0
                orig_boxes = orig_boxes[person_mask]
                orig_classes = orig_classes[person_mask]
                orig_confs = orig_confs[person_mask]
                if keypoints is not None:
                    keypoints = keypoints[person_mask]

                # ── filter: confidence ───────────────────────────────
                conf_mask = orig_confs >= detector._confidence
                orig_boxes = orig_boxes[conf_mask]
                orig_classes = orig_classes[conf_mask]
                orig_confs = orig_confs[conf_mask]
                if keypoints is not None:
                    keypoints = keypoints[conf_mask]

                if len(orig_boxes) == 0:
                    trackers[stream_id].update_with_detections(sv.Detections.empty())
                else:
                    labels = [class_names[int(c)] for c in orig_classes]

                    detections = sv.Detections(
                        xyxy=orig_boxes,
                        confidence=orig_confs,
                        class_id=orig_classes,
                        data={"pose": keypoints} if keypoints is not None else {},
                    )
                    tracked = trackers[stream_id].update_with_detections(detections)

                    boxes = tracked.xyxy
                    track_ids = tracked.tracker_id
                    classes = tracked.class_id
                    confs = tracked.confidence

                    # Match pose keypoints from detections → tracked boxes
                    matched_kps = tracked.data.get("pose", [None] * len(boxes))

                    # ── Step A: update LSTM buffers per person ────────
                    if track_ids is not None:
                        for bi, tid in enumerate(track_ids):
                            tid = int(tid)
                            kp = matched_kps[bi]
                            # kp shape: (17, 2or3); drop conf column → (17,2) → (34,)
                            if kp is not None and kp.shape[1] >= 2:
                                kp_xy = kp[:, :2]  # (17, 2)
                                fd.update(
                                    tid,
                                    kp_xy,
                                    timestamp=meta["source_time"],
                                    geometry=meta["geometry"],
                                )
                            else:
                                fd._drop(tid)
                                # Note: return value (score) is stored inside fd;
                                # we query fd.get_score() in the grouping step.

                    # Optional legacy scores keep nearby IDs as context only.
                    # This path is separate from scene-level X3D decisions.
                    behavior_alerts = detect_fight_groups(
                        track_ids=track_ids,
                        boxes=boxes,
                        fight_detector=fd,
                        proximity_factor=1.2,
                    )

            # Expire missing tracks on every frame, including an empty scene.
            fd.cleanup([] if track_ids is None else [int(t) for t in track_ids])

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

            jpeg_start = now()
            jpeg = annotated_jpeg(
                batch_frames[idx], boxes, track_ids, activities, action
            )
            jpeg_ms = latency_ms(jpeg_start, now())
            msg = dict(
                type="frame",
                stream_id=stream_id,
                boxes=boxes,
                original_boxes=original_boxes(boxes, meta["geometry"]),
                track_ids=track_ids,
                classes=classes,
                confs=confs,
                labels=[class_names[int(c)] for c in classes],
                timestamp=meta["timestamp"],
                source_time=meta["source_time"],
                frame_id=meta["frame_id"],
                geometry=meta["geometry"],
                model_lat=per_frame_ms,
                e2e_lat=e2e_lat,
                queue_ms=meta["queue_ms"],
                behavior_alerts=behavior_alerts,
                activities=activities,
                scores={
                    int(t): fd.get_score(int(t))
                    for t in ([] if track_ids is None else track_ids)
                },
                action_diagnostics=fd.diagnostics(),
                scene_action=action,
                scene_event=scene_event,
                source_fps=meta["source_fps"],
                mode=mode,
                source_kind="offline" if meta["is_file"] else "live",
                total_frames=meta.get("total_frames"),
                profile={
                    "decode_ms": meta.get("decode_ms"),
                    "prepare_ms": meta.get("prepare_ms"),
                    "decoded_fps": meta.get("decoded_fps"),
                    "pose_ms": per_frame_ms if detector is not None else 0,
                    "jpeg_ms": jpeg_ms,
                    "pose_instances": int(detector is not None),
                    "scene_instances": int(scene_model is not None),
                },
                action_status=(
                    "verified" if fd.verified else "unverified model contract"
                ),
                jpeg=jpeg,
            )

            try:
                if meta["is_file"] or scene_event is not None:
                    while not stop_event.is_set():
                        try:
                            result_queue.put(msg, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    continue
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except Exception:
                        pass
                result_queue.put_nowait(msg)
            except queue.Full:
                dropped_results += 1

        time.sleep(0.001)

    print(f"[Inference] stopped dropped={dropped_results}")


def inference_worker(
    num_streams,
    frame_shape,
    meta_queues,
    result_queue,
    stop_event,
    inference_alive,
    num_decoders,
    cores_per_decoder,
    scene_queues=None,
):
    try:
        _run_inference(
            num_streams,
            frame_shape,
            meta_queues,
            result_queue,
            stop_event,
            inference_alive,
            num_decoders,
            cores_per_decoder,
            scene_queues,
        )
    except Exception as exc:
        result_queue.put({"type": "error", "error": str(exc)}, timeout=2)
    finally:
        inference_alive.clear()
