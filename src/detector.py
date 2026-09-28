import os
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Thread configuration (avoids oversubscription on CPU)
# ----------------------------------------------------------------------
import psutil as _det_psutil

_det_threads = max(1, (_det_psutil.cpu_count(logical=False) or 2) // 2)
os.environ.setdefault("OMP_NUM_THREADS", str(_det_threads))
os.environ.setdefault("OPENVINO_NUM_THREADS", str(_det_threads))
del _det_psutil, _det_threads


# ----------------------------------------------------------------------
class Detector:
    """YOLOv8 pose inference wrapper (per‑frame inference)."""

    def __init__(
        self,
        model_path: str = "models/yolov8n-pose_openvino_model",
        imgsz: int = 320,
        confidence: float = 0.35,
        warmup_iters: int = 1,
    ):
        """
        Args:
            model_path: Path to .pt or OpenVINO export directory.
            imgsz: Inference image size (square, will be letterboxed).
            confidence: Confidence threshold for detections.
            warmup_iters: Number of warmup iterations.
        """
        self._imgsz = imgsz
        self._confidence = confidence
        self._warmup_iters = warmup_iters

        from pathlib import Path

        model_path = os.environ.get("POSE_MODEL_PATH", model_path)
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            raise FileNotFoundError(
                f"Pose weights not found: {path}. Set POSE_MODEL_PATH."
            )
        self._imgsz = int(os.environ.get("POSE_IMAGE_SIZE", imgsz))
        self._confidence = float(os.environ.get("PERSON_THRESHOLD", confidence))
        if not 0 <= self._confidence <= 1:
            raise ValueError("PERSON_THRESHOLD must be in [0, 1]")
        self.model = YOLO(str(path), task="pose")

    # ------------------------------------------------------------------
    def warmup(self, num_iters: int = None):
        """Warm up with the correct frame shape (256x320) for real decoder output."""
        if num_iters is None:
            num_iters = self._warmup_iters
        dummy = np.zeros((256, self._imgsz, 3), dtype=np.uint8)
        print(f"[Detector] Warming up ({num_iters} passes)...")
        for i in range(num_iters):
            try:
                _ = self.model.predict(
                    dummy, imgsz=self._imgsz, verbose=False, device="cpu"
                )
            except Exception as e:
                raise RuntimeError("Pose model warmup failed") from e
        print("[Detector] Warm-up complete.")

    # ------------------------------------------------------------------
    def detect_raw(self, frames):
        """
        Perform inference on one or more frames.

        Args:
            frames: Single frame (H,W,3) or list of frames.

        Returns:
            Single YOLO Result object if input was a single frame,
            otherwise a list of Result objects (one per input frame).
        """
        single = not isinstance(frames, list)
        if single:
            frames = [frames]

        results = []
        for frame in frames:
            # Ensure contiguous uint8 layout
            if not (frame.flags["C_CONTIGUOUS"] and frame.dtype == np.uint8):
                frame = np.ascontiguousarray(frame, dtype=np.uint8)

            try:
                res = self.model.predict(
                    frame,
                    imgsz=self._imgsz,
                    conf=self._confidence,
                    rect=False,
                    verbose=False,
                    device="cpu",
                )
                # predict always returns a list for multiple inputs,
                # but we give a single frame → take the first element
                results.append(res[0] if isinstance(res, list) else res)
            except Exception as e:
                raise RuntimeError("Pose inference failed") from e

        return results[0] if single else results
