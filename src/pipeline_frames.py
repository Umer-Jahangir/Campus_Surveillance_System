"""Frame ownership and coordinates shared by runtime and regression tests."""

import cv2
import numpy as np


def prepare_frame(frame, shape):
    h, w = frame.shape[:2]
    target_h, target_w = shape[:2]
    scale = min(target_w / w, target_h / h)
    rw, rh = round(w * scale), round(h * scale)
    x, y = (target_w - rw) // 2, (target_h - rh) // 2
    result = np.zeros(shape, dtype=np.uint8)
    result[y : y + rh, x : x + rw] = cv2.resize(frame, (rw, rh))
    return result, dict(
        original_width=w,
        original_height=h,
        scale_x=rw / w,
        scale_y=rh / h,
        pad_x=x,
        pad_y=y,
    )


def original_boxes(boxes, geometry):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry["pad_x"]) / geometry["scale_x"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry["pad_y"]) / geometry["scale_y"]
    return boxes


def annotated_jpeg(frame, boxes, track_ids, activities, scene=None):
    """Bake overlays into the exact inferred frame, never a newer decoder frame."""
    canvas = frame.copy()
    for box, tid in zip(boxes, track_ids if track_ids is not None else []):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        activity = activities.get(int(tid), "")
        color = (85, 51, 255) if activity else (247, 85, 168)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            canvas,
            f"#{tid} {activity}".strip(),
            (x1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    if scene:
        label = {
            "suspected_fight": "Suspected fight",
            "warming_up": "Action warming up",
            "unavailable": "Action unavailable",
            "active": "Action active",
        }.get(scene["state"], scene["state"])
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            label,
            (4, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (40, 70, 255) if scene["state"] == "suspected_fight" else (240, 240, 240),
            1,
        )
        score = scene.get("score")
        text = f"t={scene.get('source_time',0):.2f}s"
        if score is not None:
            text += f" p={score:.3f} [{scene['window_start']:.1f},{scene['window_end']:.1f}]"
        cv2.putText(
            canvas, text, (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 230), 1
        )
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()
