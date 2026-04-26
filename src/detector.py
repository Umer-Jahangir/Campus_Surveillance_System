import os
import shutil
import logging
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Thread configuration (avoids oversubscription on CPU)
# ----------------------------------------------------------------------
import psutil as _det_psutil
_det_threads = max(1, (_det_psutil.cpu_count(logical=False) or 2) // 2)
os.environ.setdefault("OMP_NUM_THREADS",      str(_det_threads))
os.environ.setdefault("OPENVINO_NUM_THREADS", str(_det_threads))
del _det_psutil, _det_threads

# ----------------------------------------------------------------------
class Detector:
    """YOLOv8 pose + OpenVINO inference wrapper (per‑frame inference)."""

    def __init__(self, model_path: str = "models/yolov8n-pose.pt",
                 imgsz: int = 320, confidence: float = 0.3,
                 warmup_iters: int = 1):
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

        # Prepare OpenVINO model path
        if model_path.endswith(".pt"):
            ov_path = model_path.replace(".pt", "_openvino_model")
        else:
            ov_path = model_path

        # Export to OpenVINO if not already present
        if not os.path.exists(ov_path):
            print(f"[Detector] Exporting {model_path} to OpenVINO FP16...")
            base_model = YOLO(model_path)
            base_model.export(format="openvino", half=True,
                              imgsz=self._imgsz, task="pose")

            raw = model_path.replace(".pt", "_openvino_model")
            if os.path.exists(raw) and raw != ov_path:
                if os.path.exists(ov_path):
                    shutil.rmtree(ov_path)
                shutil.move(raw, ov_path)
            print(f"[Detector] Export complete → {ov_path}")
        else:
            print(f"[Detector] Using existing model at {ov_path}")

        # Load model
        print(f"[Detector] Loading {ov_path}...")
        self.model = YOLO(ov_path, task="pose")

    # ------------------------------------------------------------------
    def warmup(self, num_iters: int = None):
        """Warm up with the correct frame shape (256x320) for real decoder output."""
        if num_iters is None:
            num_iters = self._warmup_iters
        dummy = np.zeros((256, self._imgsz, 3), dtype=np.uint8)
        print(f"[Detector] Warming up ({num_iters} passes)...")
        for i in range(num_iters):
            try:
                _ = self.model.predict(dummy, imgsz=self._imgsz,
                                       verbose=False, device="cpu")
            except Exception as e:
                print(f"[Detector] Warmup pass {i+1} failed (non‑fatal): {e}")
        print("[Detector] Warm‑up complete.")

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
                    verbose=False,
                    device="cpu",
                )
                # predict always returns a list for multiple inputs,
                # but we give a single frame → take the first element
                results.append(res[0] if isinstance(res, list) else res)
            except Exception as e:
                print(f"[Detector] Inference failed for a frame: {e}")
                results.append(None)

        return results[0] if single else results