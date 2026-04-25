import os

# Auto-detect available cores, reserve some for other processes
available_cores = os.cpu_count() or 8  # Fallback to 8
reserved_cores = max(2, available_cores // 4)  # Reserve 25%
inference_threads = max(2, available_cores - reserved_cores)

os.environ["OMP_NUM_THREADS"] = str(inference_threads)
os.environ["OPENVINO_NUM_THREADS"] = str(inference_threads)

print(f"[Detector] Auto-configured {inference_threads} threads "
      f"(from {available_cores} available cores, reserved {reserved_cores})")

import shutil
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path='models/yolov8n.pt'):
        # NO batch_size parameter — removed completely

        if model_path.endswith('.pt'):
            ov_path = model_path.replace('.pt', '_openvino_model')
        else:
            ov_path = model_path

        if not os.path.exists(ov_path):
            print(f"[Detector] Exporting {model_path} to OpenVINO FP16...")
            base_model = YOLO(model_path)
            base_model.export(format='openvino', half=True, imgsz=320)
            # Move exported folder to correct location if needed
            raw = model_path.replace('.pt', '_openvino_model')
            if os.path.exists(raw) and raw != ov_path:
                if os.path.exists(ov_path):
                    shutil.rmtree(ov_path)
                shutil.move(raw, ov_path)
            print(f"[Detector] Export complete -> {ov_path}")
        else:
            print(f"[Detector] Using existing model at {ov_path}")

        print(f"[Detector] Loading {ov_path}...")
        self.model = YOLO(ov_path, task='detect')

    def warmup(self, num_iters=1):
        """
        Warm up OpenVINO kernels before real inference.
        Single pass is enough for OpenVINO; skipping extra passes saves 1-2 seconds startup.
        """
        dummy = np.zeros((256, 320, 3), dtype=np.uint8)
        print(f"[Detector] Warming up ({num_iters} pass)...")
        try:
            _ = self.model.predict(dummy, imgsz=320, verbose=False, device='cpu')
            print("[Detector] Warm-up complete.")
        except Exception as e:
            print(f"[Detector] Warmup failed (non-fatal): {e}")

    def detect_raw(self, frames):
        """
        Accepts a single frame (np.ndarray) or list of frames.
        Returns single result or list of results (deterministic order).

        Optimized: vectorized inference via stacked array when possible.
        Falls back to per-frame if stacking fails.
        """
        single = not isinstance(frames, list)
        if single:
            frames = [frames]

        try:
            # Stack all frames into single batch
            batch = np.stack(frames, axis=0)
            results_batch = self.model.predict(batch, imgsz=320, verbose=False, device='cpu', conf=0.3)
            # Ensure list order matches input frame order
            results = [results_batch[i] if isinstance(results_batch, list) else results_batch for i in range(len(frames))]
            if not isinstance(results, list):
                results = [results]
        except Exception:
            # Fallback: per-frame inference (slower but robust)
            results = []
            for frame in frames:
                res = self.model.predict(frame, imgsz=320, verbose=False, device='cpu', conf=0.3)
                results.append(res[0])

        return results[0] if single else results