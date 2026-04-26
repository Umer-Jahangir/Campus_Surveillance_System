"""
basic_behavior.py
===================
Pose-aware behavior classifier for yolov8n-pose.

movement (speed-based):
    - standing   (idle)
    - walking
    - running

pose-based:
    - sitting    (hip‑knee‑nose angle + vertical torso angle + knee‑hip ratio)

time-based:
    - loitering

context-based:
    - crowd

COCO keypoint indices (17 landmarks)
--------------------------------------
0  nose          1  left_eye      2  right_eye
3  left_ear      4  right_ear
5  left_shoulder 6  right_shoulder
7  left_elbow    8  right_elbow
9  left_wrist    10 right_wrist
11 left_hip      12 right_hip
13 left_knee     14 right_knee
15 left_ankle    16 right_ankle
"""

import time
import math
import numpy as np

# ── COCO landmark indices ──────────────────────────────────────────────────
KP_NOSE        = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP      = 11
KP_R_HIP      = 12
KP_L_KNEE     = 13
KP_R_KNEE     = 14
KP_L_ANKLE    = 15
KP_R_ANKLE    = 16


class BehaviorAnalyzer:
    def __init__(self):
        self.tracks = {}

        # =========================
        # Tunable Parameters
        # =========================
        self.MAX_HISTORY = 20

        # Movement thresholds (pixels / frame)
        self.WALK_SPEED = 5
        self.RUN_SPEED  = 15

        # Loitering
        self.LOITER_RADIUS = 25   # px – spatial spread
        self.LOITER_TIME   = 8    # s  – dwell time

        # Sitting — more sensitive thresholds
        self.SIT_ANGLE_THRESH   = 105   # degrees (<105° → strong sitting)
        self.SIT_RATIO_THRESH   = 0.28  # (knee_y - hip_y)/bbox_h
        self.SIT_CONF_THRESH    = 0.12  # minimal keypoint confidence
        self.SIT_FRAMES_NEEDED  = 3     # consecutive frames to confirm (was 5)
        self.SIT_EVIDENCE_MIN   = 1     # one strong signal is enough

        # Crowd
        self.CROWD_MIN_PEOPLE = 4
        self.CROWD_RADIUS     = 80
        self.CROWD_TIME       = 3

        # Alert cooldown
        self.ALERT_COOLDOWN = 5

        # Crowd state
        self.crowd_state = {"active": False, "start_time": 0}

    # =========================================================
    # Helper: angle between three points (a‑b‑c)
    # =========================================================
    @staticmethod
    def _angle_between_points(a, b, c):
        """
        Returns angle in degrees at point b.
        a, b, c : (x, y) or numpy arrays with (x, y, conf)
        """
        try:
            a = np.array(a[:2]) if len(a) > 1 else np.array(a)
            b = np.array(b[:2]) if len(b) > 1 else np.array(b)
            c = np.array(c[:2]) if len(c) > 1 else np.array(c)
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi
            return angle
        except Exception:
            return None

    # =========================================================
    # UPDATE TRACK
    # =========================================================
    def update(self, track_id: int, bbox: tuple, keypoints=None):
        """
        Parameters
        ----------
        track_id  : tracker ID (int)
        bbox      : (x1, y1, x2, y2) in pixels
        keypoints : np.ndarray shape (17, 2) or (17, 3) [x, y, conf]
                    Pass None when not available.
        """
        x1, y1, x2, y2 = bbox
        cx     = (x1 + x2) / 2.0
        cy     = (y1 + y2) / 2.0
        bbox_h = max(y2 - y1, 1)
        now    = time.time()

        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "positions":        [],
                "kp_history":       [],   # list of np arrays (17,2) or (17,3)
                "sit_count":        0,    # consecutive sitting frames
                "bbox_h":           bbox_h,
                "first_seen":       now,
                "last_seen":        now,
                "state":            "unknown",
                "last_alert":       0,
            }

        t = self.tracks[track_id]
        t["last_seen"] = now
        t["bbox_h"]    = bbox_h

        t["positions"].append((cx, cy))
        if len(t["positions"]) > self.MAX_HISTORY:
            t["positions"].pop(0)

        if keypoints is not None:
            kp = np.asarray(keypoints, dtype=np.float32)
            t["kp_history"].append(kp)
            if len(t["kp_history"]) > self.MAX_HISTORY:
                t["kp_history"].pop(0)

    # =========================================================
    # SPEED
    # =========================================================
    def _speed(self, positions: list) -> float:
        if len(positions) < 2:
            return 0.0
        (x1, y1), (x2, y2) = positions[-2], positions[-1]
        return math.hypot(x2 - x1, y2 - y1)

    # =========================================================
    # LOITERING
    # =========================================================
    def _is_loitering(self, track: dict) -> bool:
        positions = track["positions"]
        if len(positions) < 10:
            return False
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        if (max(xs) - min(xs)) < self.LOITER_RADIUS and \
           (max(ys) - min(ys)) < self.LOITER_RADIUS:
            return (time.time() - track["first_seen"]) > self.LOITER_TIME
        return False

    # =========================================================
    # KEYPOINT VALIDITY
    # =========================================================
    def _kp_valid(self, pt: np.ndarray) -> bool:
        if pt is None:
            return False
        if len(pt) >= 3:
            return float(pt[2]) >= self.SIT_CONF_THRESH
        return float(pt[0]) > 1.0 or float(pt[1]) > 1.0

    # =========================================================
    # SITTING (Enhanced: angle + torso + ratio + speed)
    # =========================================================
    def _is_sitting(self, track: dict) -> bool:
        kp_history = track.get("kp_history", [])
        bbox_h     = max(track.get("bbox_h", 100), 1)

        if not kp_history:
            return False

        kp = kp_history[-1]
        if kp is None or len(kp) < 17:
            track["sit_count"] = 0
            return False

        try:
            nose       = kp[KP_NOSE]
            l_shoulder = kp[KP_L_SHOULDER]
            r_shoulder = kp[KP_R_SHOULDER]
            l_hip      = kp[KP_L_HIP]
            r_hip      = kp[KP_R_HIP]
            l_knee     = kp[KP_L_KNEE]
            r_knee     = kp[KP_R_KNEE]
            l_ankle    = kp[KP_L_ANKLE]
            r_ankle    = kp[KP_R_ANKLE]

            shoulder_pts = [p for p in [l_shoulder, r_shoulder] if self._kp_valid(p)]
            hip_pts      = [p for p in [l_hip,      r_hip]      if self._kp_valid(p)]
            knee_pts     = [p for p in [l_knee,     r_knee]     if self._kp_valid(p)]
            ankle_pts    = [p for p in [l_ankle,    r_ankle]    if self._kp_valid(p)]

            if not hip_pts:
                track["sit_count"] = 0
                return False

            hip_y = float(np.mean([p[1] for p in hip_pts]))
            evidence = 0

            # ---- SIGNAL 1: Nose–Hip–Knee angle (strong) ----
            if self._kp_valid(nose) and knee_pts:
                knee_mid = np.mean([p[:2] for p in knee_pts], axis=0)
                hip_mid  = np.mean([p[:2] for p in hip_pts], axis=0)
                angle = self._angle_between_points(nose[:2], hip_mid, knee_mid)
                if angle is not None and angle < self.SIT_ANGLE_THRESH:
                    evidence += 3                     # high weight
                elif angle is not None and angle < 120:
                    evidence += 1

            # ---- SIGNAL 2: Vertical torso angle (shoulder–hip–down) ----
            # Standing: ~170–180°, Sitting: ~90–140°
            if shoulder_pts and hip_pts:
                shoulder_mid = np.mean([p[:2] for p in shoulder_pts], axis=0)
                hip_mid      = np.mean([p[:2] for p in hip_pts], axis=0)
                vertical_ref = (hip_mid[0], hip_mid[1] + 100)
                torso_angle = self._angle_between_points(shoulder_mid, hip_mid, vertical_ref)
                if torso_angle is not None and torso_angle < 140:
                    evidence += 2

            # ---- SIGNAL 3: Knee close to hip (vertical ratio) ----
            if knee_pts:
                knee_y = float(np.mean([p[1] for p in knee_pts]))
                ratio  = (knee_y - hip_y) / bbox_h
                if ratio < self.SIT_RATIO_THRESH:
                    evidence += 1

            # ---- SIGNAL 4: Ankles not visible (if knees visible) ----
            if knee_pts and not ankle_pts:
                evidence += 1

            # ---- SIGNAL 5: Large torso fraction (compressed posture) ----
            torso_frac = 0.0
            if shoulder_pts:
                shoulder_y = float(np.mean([p[1] for p in shoulder_pts]))
                torso_frac = (hip_y - shoulder_y) / bbox_h
                if torso_frac > 0.38:
                    evidence += 1

            # ---- SIGNAL 6: Low speed + high torso fraction (still sitting) ----
            speed = self._speed(track["positions"])
            if speed < 2 and torso_frac > 0.4:
                evidence += 2

            # ---- update frame counter ----
            if evidence >= self.SIT_EVIDENCE_MIN:
                track["sit_count"] = min(track["sit_count"] + 1, 8)
            else:
                track["sit_count"] = max(track["sit_count"] - 1, 0)

            return track["sit_count"] >= self.SIT_FRAMES_NEEDED

        except (IndexError, TypeError, ValueError):
            track["sit_count"] = 0
            return False

    # =========================================================
    # CLASSIFY INDIVIDUAL
    # =========================================================
    def classify(self, track_id: int) -> str | None:
        track = self.tracks.get(track_id)
        if not track:
            return None

        # Priority order: loitering > sitting > running > walking > standing
        if self._is_loitering(track):
            return "loitering"

        if self._is_sitting(track):
            return "sitting"

        speed = self._speed(track["positions"])

        if speed > self.RUN_SPEED:
            return "running"
        elif speed > self.WALK_SPEED:
            return "walking"
        else:
            return "standing"

    # =========================================================
    # INDIVIDUAL ALERT
    # =========================================================
    def get_individual_alert(self, track_id: int) -> dict | None:
        track = self.tracks.get(track_id)
        if not track:
            return None

        state = self.classify(track_id)
        now   = time.time()

        if state != track["state"]:
            if now - track["last_alert"] > self.ALERT_COOLDOWN:
                track["state"]      = state
                track["last_alert"] = now
                return {
                    "type":     "individual",
                    "track_id": track_id,
                    "label":    state,
                    "ts":       now,
                }

        return None

    # =========================================================
    # CROWD DETECTION
    # =========================================================
    def detect_crowd(self) -> dict | None:
        now       = time.time()
        positions = []

        for track in self.tracks.values():
            if now - track["last_seen"] < 1.0 and track["positions"]:
                positions.append(track["positions"][-1])

        if len(positions) < self.CROWD_MIN_PEOPLE:
            self.crowd_state["active"] = False
            return None

        close_pairs = sum(
            1
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
            if math.hypot(
                positions[i][0] - positions[j][0],
                positions[i][1] - positions[j][1]
            ) < self.CROWD_RADIUS
        )

        if close_pairs >= len(positions):
            if not self.crowd_state["active"]:
                self.crowd_state["start_time"] = now
            self.crowd_state["active"] = True

            if now - self.crowd_state["start_time"] > self.CROWD_TIME:
                return {
                    "type":  "crowd",
                    "label": "crowd",
                    "count": len(positions),
                    "ts":    now,
                }
        else:
            self.crowd_state["active"] = False

        return None

    # =========================================================
    # CLEANUP
    # =========================================================
    def cleanup(self, timeout: float = 5.0):
        now        = time.time()
        remove_ids = [tid for tid, t in self.tracks.items()
                      if now - t["last_seen"] > timeout]
        for tid in remove_ids:
            del self.tracks[tid]