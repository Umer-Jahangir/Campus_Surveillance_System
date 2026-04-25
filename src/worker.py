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


# ============================================================
# CPU CORE ASSIGNMENT
# ============================================================

def get_pcore_logical_ids():
    """Return list of logical processor IDs that belong to P-cores (heuristic)."""
    freqs = psutil.cpu_freq(percpu=True)
    if not freqs or len(freqs) < 2:
        total_logical = psutil.cpu_count(logical=True)
        return list(range(4)) if total_logical > 8 else list(range(total_logical))

    max_freq = max(f.max for f in freqs if f is not None)
    threshold = max_freq * 0.9
    pcores = [i for i, f in enumerate(freqs) if f is not None and f.max >= threshold]
    return pcores


def assign_cores_hybrid(worker_index, num_decoders, cores_per_decoder, is_inference=False):
    """
    Assign logical cores intelligently:
    - Inference gets all P-cores.
    - Decoders get E-cores, round-robin.
    """
    total_logical = psutil.cpu_count(logical=True)
    pcore_ids = get_pcore_logical_ids()
    ecore_ids = [i for i in range(total_logical) if i not in pcore_ids]

    if is_inference:
        return pcore_ids

    start = worker_index * cores_per_decoder
    end = start + cores_per_decoder
    if end > len(ecore_ids):
        print(f"Warning: insufficient E-cores for decoder {worker_index}, using any cores.")
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
        core_ids = assign_cores_hybrid(index, num_decoders, cores_per_decoder, is_inference=False)
        p.cpu_affinity(core_ids)
        print(f"[Decoder {index}] Pinned to cores {core_ids}")
    except Exception as e:
        print(f"[Decoder {index}] Affinity error: {e}")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

    def open_stream(url):
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    cap = open_stream(stream_url)

    if not cap.isOpened():
        print(f"[Decoder {index}] Failed to open stream")
        return

    # Open both shared memory slots
    try:
        shm_slots = [shared_memory.SharedMemory(name=n) for n in shm_names]
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

    drop_count = 0
    stall_count = 0
    reconnect_delay = 2

    pace_sleep = 0.033
    PACE_MIN = 0.02
    PACE_MAX = 0.08
    ALPHA_UP = 0.05
    ALPHA_DOWN = 0.15

    READ_STALL_THRESHOLD_MS = 80.0

    # For local files pace to the source FPS so the decoder does not spin at
    # maximum speed and back off to PACE_MAX (80ms = 12.5 FPS).
    # For RTSP streams cap.read() naturally blocks on the network so adaptive
    # pacing still applies.
    if os.path.isfile(stream_url):
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        file_frame_interval = (1.0 / source_fps) if source_fps > 0 else 0.033
        print(f"[Decoder {index}] Local file — pacing to {source_fps:.1f} FPS "
              f"({file_frame_interval*1000:.1f}ms/frame)")
    else:
        file_frame_interval = 0  # use adaptive pacing for streams

    buf_idx = 0  # alternates between 0 and 1 each frame

    while not stop_event.is_set():

        if not inference_alive.is_set():
            time.sleep(0.5)
            continue

        read_start = time.perf_counter()
        ret, frame = cap.read()
        read_ms = (time.perf_counter() - read_start) * 1000

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
                print(
                    f"[Decoder {index}] stalled reads={stall_count} "
                    f"last={read_ms:.0f}ms"
                )

            continue

        frame_resized = cv2.resize(
            frame,
            (frame_shape[1], frame_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Write into the current buffer slot
        np.copyto(frame_buffers[buf_idx], frame_resized)

        sent = False

        try:
            meta_queue.put_nowait({
                "stream_id": index,
                "buf_idx": buf_idx,
                "timestamp": now()
            })
            sent = True

        except queue.Full:

            drop_count += 1

            if drop_count % 100 == 0:
                print(f"[Decoder {index}] dropped {drop_count}")

        # Flip to the other slot for the next frame
        buf_idx ^= 1

        # Optimized pacing: sample queue every 5th frame instead of every frame
        sample_interval = 5
        if (sent and not stop_event.is_set()) or (sent is False):
            if file_frame_interval > 0:
                # Local files: fixed interval matching source FPS
                time.sleep(file_frame_interval)
            else:
                # RTSP: adaptive pacing (simplified — check queue less often)
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

    print(
        f"[Decoder {index}] Stopped | "
        f"Dropped={drop_count} | "
        f"Stalled={stall_count}"
    )


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
        core_ids = assign_cores_hybrid(0, num_decoders, cores_per_decoder, is_inference=True)

        if not core_ids:
            core_ids = [psutil.cpu_count(logical=False) - 1]

        p.cpu_affinity(core_ids)
        print(f"[Inference] Pinned to cores {core_ids}")

    except Exception as e:
        print(f"[Inference] Affinity error: {e}")

    detector = Detector()

    class_names = detector.model.names

    # Open 2 shm handles per stream — index as [stream][buf_idx]
    shm_handles = []
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
            try:
                mq.get_nowait()
            except queue.Empty:
                break

    # Initialize ByteTrack trackers
    trackers = [sv.ByteTrack() for _ in range(num_streams)]

    inference_alive.set()

    print("[Inference] Ready — decoders can start")

    result_drop_count = 0

    while not stop_event.is_set():

        batch_frames = []
        batch_metadata = []

        for i, mq in enumerate(meta_queues):

            try:

                meta = mq.get_nowait()

                age = latency_ms(meta["timestamp"], now())

                if age > 500:
                    continue

                # .copy() is the read fence that makes double-buffering safe —
                # without it the decoder can overwrite the buffer while the
                # result queue still holds a reference to it.
                buf_idx = meta["buf_idx"]
                frame = frame_buffers[i][buf_idx].copy()

                batch_frames.append(frame)
                batch_metadata.append(meta)

            except queue.Empty:
                pass

        if not batch_frames:
            stop_event.wait(timeout=0.001)
            continue

        start = now()

        results = detector.detect_raw(batch_frames)

        total_ms = latency_ms(start, now())
        per_frame_ms = total_ms / len(batch_frames)

        for i, res in enumerate(results):

            meta = batch_metadata[i]

            boxes = None
            classes = None
            confs = None
            labels = None
            track_ids = None

            if res.boxes is not None:

                boxes = res.boxes.xyxy.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                labels = [class_names[int(c)] for c in classes]

                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=confs,
                    class_id=classes
                )

                tracked = trackers[meta["stream_id"]].update_with_detections(detections)

                boxes = tracked.xyxy
                classes = tracked.class_id
                confs = tracked.confidence
                track_ids = tracked.tracker_id

            e2e_lat = latency_ms(meta["timestamp"], now())

            msg = (
                "frame",
                meta["stream_id"],
                batch_frames[i],
                boxes,
                track_ids,
                classes,
                confs,
                labels,
                meta["timestamp"],
                per_frame_ms,
                e2e_lat
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