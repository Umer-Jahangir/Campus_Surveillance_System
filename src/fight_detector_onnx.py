"""Strict temporal pose scorer. Unknown training contracts are diagnostic-only."""

import json
import hashlib
import os
import warnings
from pathlib import Path
from collections import deque
import numpy as np


class DisabledPoseClassifier:
    """Person tracking stays available without the unsupported legacy scorer."""

    verified = False
    max_gap = 0.5
    threshold = 1.0

    def update(self, *args, **kwargs):
        return None

    def cleanup(self, *args):
        pass

    def _drop(self, *args):
        pass

    def get_score(self, *args):
        return 0.0

    def above_threshold(self, *args):
        return False

    def diagnostics(self):
        return {
            "state": "disabled",
            "reason": "Use scene-level X3D; legacy person semantics unverified",
            "tracks": {},
        }


class FightDetectorONNX:
    def __init__(self, model_path, seq_len=20, keypoint_dim=34, threshold=0.7):
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Fight weights not found: {path}. Set FIGHT_MODEL_PATH."
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(path), opts, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].shape[1:] != [seq_len, keypoint_dim]
            or inputs[0].type != "tensor(float)"
        ):
            raise ValueError(
                "Fight model must accept float32 [batch,20,34] COCO pose sequences"
            )
        if len(outputs) != 1 or len(outputs[0].shape) != 2 or outputs[0].shape[1] != 1:
            raise ValueError(
                "Fight model must output [batch,1]; video/scene models need a separate adapter"
            )
        self.input_name = inputs[0].name
        self.seq_len, self.keypoint_dim = seq_len, keypoint_dim
        self.threshold = float(os.environ.get("FIGHT_THRESHOLD", threshold))
        if not 0 <= self.threshold <= 1:
            raise ValueError("FIGHT_THRESHOLD must be in [0,1]")
        contract_path = os.environ.get("FIGHT_CONTRACT")
        self.contract = (
            json.loads(Path(contract_path).read_text()) if contract_path else {}
        )
        self.verified = bool(self.contract.get("verified", False))
        if self.verified:
            if (
                self.contract.get("model_sha256")
                != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError(
                    "Contract model_sha256 does not match selected fight weights"
                )
            required = {
                "keypoints": "coco17_xy",
                "scope": "person",
                "positive_label": "fight",
            }
            if any(self.contract.get(k) != v for k, v in required.items()):
                raise ValueError(
                    "Contract must declare coco17_xy, person scope and fight positive label"
                )
            if self.contract.get("coordinates") not in (
                "inference_pixels",
                "original_pixels",
                "normalized",
            ):
                raise ValueError(
                    "Contract coordinates must be inference_pixels, original_pixels or normalized"
                )
            if not self.contract.get("sample_fps", 0) > 0:
                raise ValueError("Contract sample_fps must be positive")
        else:
            warnings.warn(
                "Fight model training preprocessing, sampling and label mapping are unverified. Raw scores only; fight alerts unavailable.",
                RuntimeWarning,
            )
        self.max_gap = float(self.contract.get("max_gap_seconds", 0.5))
        sample_fps = float(self.contract.get("sample_fps", 30))
        if (
            not np.isfinite(sample_fps)
            or sample_fps <= 0
            or not np.isfinite(self.max_gap)
            or self.max_gap <= 0
        ):
            raise ValueError(
                "sample_fps and max_gap_seconds must be finite and positive"
            )
        self.interval = 1.0 / sample_fps
        self.buffers, self.scores, self.timestamps = {}, {}, {}

    def update(self, track_id, keypoints_xy, timestamp=None, geometry=None):
        arr = np.asarray(keypoints_xy, dtype=np.float32)
        if arr.shape not in ((17, 2), (34,)) or not np.isfinite(arr).all():
            self._drop(track_id)
            return None
        arr = arr.reshape(17, 2).copy()
        timestamp = float(
            timestamp
            if timestamp is not None
            else self.timestamps.get(track_id, -self.interval) + self.interval
        )
        if not np.isfinite(timestamp):
            self._drop(track_id)
            raise ValueError("Pose timestamp must be finite")
        previous = self.timestamps.get(track_id)
        if previous is not None:
            delta = timestamp - previous
            if (
                delta <= 0
                or delta > self.max_gap
                or (self.verified and delta > self.interval * 1.5)
            ):
                self._drop(track_id)
            elif delta < self.interval * 0.95:
                return self.scores.get(track_id)
        coordinates = self.contract.get("coordinates", "inference_pixels")
        if coordinates != "inference_pixels":
            if geometry is None:
                raise ValueError("Source geometry required by action model contract")
            arr[:, 0] = (arr[:, 0] - geometry["pad_x"]) / geometry["scale_x"]
            arr[:, 1] = (arr[:, 1] - geometry["pad_y"]) / geometry["scale_y"]
            if coordinates == "normalized":
                arr /= [geometry["original_width"], geometry["original_height"]]
        vector = arr.flatten()
        if "mean" in self.contract or "std" in self.contract:
            mean = np.asarray(self.contract["mean"], dtype=np.float32)
            std = np.asarray(self.contract["std"], dtype=np.float32)
            if (
                mean.shape != (34,)
                or std.shape != (34,)
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or np.any(std <= 0)
            ):
                raise ValueError(
                    "Contract mean/std must have 34 finite values and positive std"
                )
            vector = (vector - mean) / std
        self.timestamps[track_id] = timestamp
        self.buffers.setdefault(track_id, deque(maxlen=self.seq_len)).append(vector)
        if len(self.buffers[track_id]) < self.seq_len:
            return None
        inp = np.asarray([self.buffers[track_id]], dtype=np.float32)
        score = float(self.session.run(None, {self.input_name: inp})[0][0][0])
        if not np.isfinite(score) or not 0 <= score <= 1:
            self._drop(track_id)
            raise ValueError("Fight model output is not a finite probability")
        self.scores[track_id] = score
        return score

    def get_score(self, track_id):
        return self.scores.get(track_id, 0.0)

    def above_threshold(self, track_id):
        return (
            self.verified
            and track_id in self.scores
            and self.get_score(track_id) >= self.threshold
        )

    def _drop(self, tid):
        for mapping in (self.buffers, self.scores, self.timestamps):
            mapping.pop(tid, None)

    def cleanup(self, active_ids):
        active = set(active_ids)
        for tid in list(self.buffers):
            if tid not in active:
                self._drop(tid)

    def diagnostics(self):
        """Return compact per-stream readiness state for troubleshooting."""
        return {
            "verified": self.verified,
            "sequence_length": self.seq_len,
            "sample_interval_seconds": self.interval,
            "max_gap_seconds": self.max_gap,
            "tracks": {
                int(tid): {
                    "samples": len(buf),
                    "ready": len(buf) >= self.seq_len,
                    "score": self.scores.get(tid),
                    "last_timestamp": self.timestamps.get(tid),
                }
                for tid, buf in self.buffers.items()
            },
        }
