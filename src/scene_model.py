"""Scene-only X3D adapter. No participant attribution is inferred from its score.

Reference: visionlab-ai/school-violence-detection-app video_utils.py and
sv_model/x3d_model.py; pinned artifact/evidence is recorded in config.
"""

from pathlib import Path
import hashlib
import os
import time
import cv2
import numpy as np

X3D_SHA256 = "e833f69d110f167cad4a6c38d385564bdb2f6de63d246e45cb03ff9aa17f0349"


def preprocess_scene(frames, count=16):
    """Uniform indices, BGR->RGB, bilinear square resize, /255 then mean/std."""
    if not len(frames):
        raise ValueError("A scene window must contain frames")
    indices = np.linspace(0, len(frames) - 1, count).astype(int)
    rgb = np.stack(
        [
            cv2.resize(frames[i], (224, 224), interpolation=cv2.INTER_LINEAR)[..., ::-1]
            for i in indices
        ]
    ).astype(np.float32)
    return np.ascontiguousarray(
        ((rgb / 255.0 - 0.45) / 0.225).transpose(3, 0, 1, 2)[None]
    )


class SceneModel:
    """One shared CPU model; callers own their stream windows."""

    def __init__(self, path):
        import torch
        from pytorchvideo.models.hub import x3d_m

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Scene model missing: {path}. Set SCENE_MODEL_PATH."
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != X3D_SHA256:
            raise ValueError(
                "Scene checkpoint hash does not match the supported X3D contract"
            )
        torch.set_num_threads(int(os.environ.get("ACTION_THREADS", "4")))
        # Permit only NumPy scalar metadata used by this inspected checkpoint.
        with torch.serialization.safe_globals(
            [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
        ):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model = x3d_m(pretrained=False)
        features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = torch.nn.Sequential(
            torch.nn.Dropout(0.3), torch.nn.Linear(features, 2)
        )
        state = {k.removeprefix("backbone."): v for k, v in checkpoint["model"].items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.sha256 = X3D_SHA256

    def predict(self, frames):
        import torch

        start = time.perf_counter()
        tensor = torch.from_numpy(preprocess_scene(frames))
        prepared = time.perf_counter()
        with torch.inference_mode():
            output = self.model(tensor)
            # PyTorchVideo's X3D head already applies Softmax. Applying it
            # twice compresses probabilities into ~[.269,.731].
            score = float(output[0, 1])
        if output.shape != (1, 2) or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Scene model returned a nonfinite score")
        return {
            "score": score,
            "preprocess_ms": (prepared - start) * 1000,
            "action_ms": (time.perf_counter() - prepared) * 1000,
            "scope": "scene",
        }


def configured_scene_path():
    return Path(
        os.environ.get(
            "SCENE_MODEL_PATH",
            str(Path(__file__).parent / "models/final_x3d_realtime.pt"),
        )
    )


class SceneWindow:
    """Bounded causal windows; scene decisions never attribute a person.

    Five seconds is the RWF training clip duration. Window length and stride
    are deployment settings, not a claim of live-stream training provenance.
    """

    def __init__(self, model, reason=None):
        from collections import deque

        self.model, self.reason = model, reason
        self.duration = float(os.environ.get("SCENE_WINDOW_SECONDS", "5"))
        self.stride = float(os.environ.get("SCENE_STRIDE_SECONDS", "1"))
        self.threshold = float(os.environ.get("SCENE_THRESHOLD", "0.5"))
        self.required = int(os.environ.get("SCENE_CONFIRM_WINDOWS", "2"))
        if not (
            np.isfinite([self.duration, self.stride, self.threshold]).all()
            and self.duration > 0
            and self.stride > 0
            and 0 <= self.threshold <= 1
            and self.required >= 1
        ):
            raise ValueError(
                "Invalid scene window, stride, threshold or confirmation count"
            )
        self.frames = deque(maxlen=512)
        self.last = None
        self.last_prediction = None
        self.epoch = None
        self.positive = 0
        self.active = False
        self.event_id = 0
        self.decision_count = 0
        self.status = {
            "state": "unavailable" if model is None else "warming_up",
            "reason": reason,
            "model": "X3D-M / final_x3d_realtime",
            "scope": "scene",
            "threshold": self.threshold,
            "confirm_windows": self.required,
            "score": None,
        }

    def update(self, frame, timestamp, epoch=0):
        event = None
        if self.model is None:
            return dict(self.status), event
        if not np.isfinite(timestamp):
            raise ValueError("Scene timestamp must be finite")
        if timestamp == self.last and epoch == self.epoch:
            return dict(self.status), event
        if self.last is not None and (
            epoch != self.epoch or timestamp < self.last or timestamp - self.last > 1
        ):
            if self.active:
                event = {
                    "phase": "observation_lost",
                    "reason": "source discontinuity",
                    "event_id": self.event_id,
                }
            self.frames.clear()
            self.last_prediction = None
            self.positive = 0
            self.active = False
            self.status.update(state="warming_up", score=None)
        self.last = timestamp
        self.epoch = epoch
        self.frames.append((timestamp, frame.copy()))
        while len(self.frames) > 1 and self.frames[1][0] < timestamp - self.duration:
            self.frames.popleft()
        self.status.update(source_time=timestamp, samples=len(self.frames))
        if timestamp - self.frames[0][0] < self.duration - 0.05:
            return dict(self.status), event
        if (
            self.last_prediction is not None
            and timestamp - self.last_prediction < self.stride
        ):
            return dict(self.status), event
        return self._decide([f for _, f in self.frames], self.frames[0][0], timestamp)

    def update_clip(self, clip, timestamp, epoch=0):
        """Decode-time samples survive skipped inference frames on live inputs."""
        if self.model is None:
            return dict(self.status), None
        event = None
        if self.epoch is not None and (epoch != self.epoch or timestamp < self.last):
            if self.active:
                event = {
                    "phase": "observation_lost",
                    "reason": "source discontinuity",
                    "event_id": self.event_id,
                }
            self.active = False
            self.positive = 0
            self.last_prediction = None
            self.status.update(state="warming_up", score=None)
        self.epoch = epoch
        self.last = timestamp
        self.status["source_time"] = timestamp
        if clip is None:
            if self.last_prediction is not None:
                if self.active:
                    event = {
                        "phase": "observation_lost",
                        "reason": "source window reset",
                        "event_id": self.event_id,
                    }
                self.active = False
                self.positive = 0
                self.last_prediction = None
                self.status.update(state="warming_up", score=None)
            return dict(self.status), event
        if self.last_prediction is not None and clip["end"] <= self.last_prediction:
            return dict(self.status), event
        status, decision_event = self._decide(
            clip["frames"], clip["start"], clip["end"]
        )
        return status, decision_event or event

    def _decide(self, frames, window_start, timestamp):
        event = None
        prediction = self.model.predict(frames)
        gap = None if self.last_prediction is None else timestamp - self.last_prediction
        self.decision_count += 1
        self.last_prediction = timestamp
        self.positive = (
            self.positive + 1 if prediction["score"] >= self.threshold else 0
        )
        was_active = self.active
        self.active = (
            self.positive >= self.required
            if not self.active
            else prediction["score"] >= self.threshold
        )
        self.status.update(
            prediction,
            state="suspected_fight" if self.active else "active",
            decision_count=self.decision_count,
            decision_gap_seconds=gap,
            configured_stride_seconds=self.stride,
            window_start=window_start,
            window_end=timestamp,
            consecutive_positive=self.positive,
        )
        if self.active != was_active:
            if self.active:
                self.event_id += 1
            event = {
                "phase": "start" if self.active else "end",
                "event_id": self.event_id,
                **self.status,
            }
            if not self.active:
                event["reason"] = "fight no longer detected in analyzed window"
        return dict(self.status), event


class SceneSampler:
    """Decoder-side bounded sampling; packet arrays are immutable after creation."""

    def __init__(self):
        from collections import deque

        self.frames = deque(maxlen=512)
        self.duration = float(os.environ.get("SCENE_WINDOW_SECONDS", "5"))
        self.stride = float(os.environ.get("SCENE_STRIDE_SECONDS", "1"))
        self.cached = None
        self.epoch = None
        self.last = None

    def update(self, frame, timestamp, epoch):
        if self.last is not None and (
            epoch != self.epoch or timestamp <= self.last or timestamp - self.last > 1
        ):
            self.frames.clear()
            self.cached = None
        self.epoch = epoch
        self.last = timestamp
        self.frames.append((timestamp, frame))
        while len(self.frames) > 1 and self.frames[1][0] < timestamp - self.duration:
            self.frames.popleft()
        if timestamp - self.frames[0][0] >= self.duration - 0.05 and (
            self.cached is None or timestamp - self.cached["end"] >= self.stride
        ):
            indices = np.linspace(0, len(self.frames) - 1, 16).astype(int)
            self.cached = {
                "start": self.frames[0][0],
                "end": timestamp,
                "frames": np.stack([self.frames[i][1] for i in indices]),
            }
        return self.cached
