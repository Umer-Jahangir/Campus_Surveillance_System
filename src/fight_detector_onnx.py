import numpy as np
from collections import deque
import onnxruntime as ort


class FightDetectorONNX:
    """
    Per-stream LSTM fight scorer.

    Design change vs. the old version
    -----------------------------------
    The old code fired an alert as soon as ONE person's LSTM score exceeded
    the threshold.  That is wrong: a fight requires ≥2 people.

    This class now does only ONE thing: maintain a rolling 20-frame keypoint
    buffer per tracked person and return that person's fight-probability score
    when the buffer is full.  The multi-person grouping logic lives in
    inference_worker (worker.py), where bounding boxes are available.

    Public API
    ----------
    update(track_id, keypoints_xy) -> float | None
        Feed one frame of keypoints for a person.
        Returns the LSTM output (0-1) once the buffer is full, else None.

    get_score(track_id) -> float
        Return the most recent score for a track (0.0 if never scored yet).

    cleanup(active_ids)
        Drop buffers/scores for tracks that are no longer alive.
    """

    def __init__(self, model_path: str, seq_len: int = 20,
                 keypoint_dim: int = 34, threshold: float = 0.6):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name  # e.g. "input_layer"
        self.seq_len = seq_len
        self.keypoint_dim = keypoint_dim
        self.threshold = threshold

        # track_id -> deque[np.ndarray shape (34,)]
        self.buffers: dict[int, deque] = {}
        # track_id -> latest LSTM score (float 0-1)
        self.scores: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Core update – returns raw score, NO alert logic here
    # ------------------------------------------------------------------

    def update(self, track_id: int, keypoints_xy) -> float | None:
        """
        Parameters
        ----------
        track_id     : ByteTrack integer ID
        keypoints_xy : array-like, shape (17, 2) or (34,)
                       x/y coordinates only – confidence column must be
                       stripped by the caller before passing in.

        Returns
        -------
        float   – LSTM fight-probability score (0-1) when buffer is full
        None    – buffer not yet full (keep accumulating frames)
        """
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.seq_len)

        # Normalise to flat float32 vector of length 34
        arr = np.asarray(keypoints_xy, dtype=np.float32)
        if arr.shape == (17, 2):
            pose_vec = arr.flatten()           # (34,)
        elif arr.shape == (34,):
            pose_vec = arr
        else:
            return None                        # unexpected shape – skip silently

        self.buffers[track_id].append(pose_vec)

        if len(self.buffers[track_id]) < self.seq_len:
            return None                        # not enough history yet

        # Run LSTM inference
        inp = np.stack(list(self.buffers[track_id]), axis=0).reshape(
            1, self.seq_len, self.keypoint_dim
        )  # (1, 20, 34)
        score = float(self.session.run(None, {self.input_name: inp})[0][0][0])
        self.scores[track_id] = score

        print(
            f"[ONNX] track={track_id:3d}  score={score:.4f}"
            f"  {'⚡ HIGH' if score >= self.threshold else '      '}"
        )
        return score

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_score(self, track_id: int) -> float:
        """Last known score for *track_id* (0.0 if never scored)."""
        return self.scores.get(track_id, 0.0)

    def above_threshold(self, track_id: int) -> bool:
        return self.get_score(track_id) >= self.threshold

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def cleanup(self, active_ids) -> None:
        """Remove state for tracks that have left the scene."""
        active = set(active_ids)
        for tid in list(self.buffers):
            if tid not in active:
                del self.buffers[tid]
        for tid in list(self.scores):
            if tid not in active:
                del self.scores[tid]