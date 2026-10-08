"""
StudyGuard - FocusEngine
========================
Real-time student focus monitoring for the StudyGuard Edge-AIoT platform.

Signals
-------
1. Active phone usage     : YOLOv8n 'cell phone' (cls 67) + spatial context
                            (chest/face zone) OR head pitch-down > 20 deg.
2. Timepass / slacking    : head roll > 28 deg, or head pitch > 40 deg with no
                            reading motion for 4 s, or absence > 4 s.
3. Drowsiness             : EAR < 0.20 for 1 s.
4. Gaze wandering         : |yaw| > 30 deg for 3 s.

Models
------
- Ultralytics YOLOv8-nano (`yolov8n.pt`) -> person / cell phone boxes.
- MediaPipe FaceLandmarker (Tasks API)   -> 478 face landmarks (EAR) and the
  4x4 facial transformation matrix (head pose: pitch / yaw / roll).

Head pose convention (validated against MediaPipe "metric space" docs):
the task's metric space is right-handed with the virtual camera at the origin
looking down -Z, therefore +X = image right, +Y = image up. The canonical face
+Z axis points out of the face, so:

    normal  n = R[:3, 2]
    pitch   = asin(-n.y)      # positive = looking down
    yaw     = atan2(n.x, n.z) # positive = facing image right
    roll    = asin(R[1,0] / cos(pitch))

Usage
-----
    from engine.focus_engine import FocusEngine
    engine = FocusEngine()
    out = engine.process_frame(bgr_frame)
    engine.close()

CLI demo
--------
    python engine/focus_engine.py            # webcam 0
    python engine/focus_engine.py image.jpg  # still image
    python engine/focus_engine.py clip.mp4   # video file
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from ultralytics import YOLO

__all__ = ["FocusConfig", "FocusEngine"]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

PERSON_CLASS = 0
PHONE_CLASS = 67

# MediaPipe face-mesh landmark indices (validated on real faces).
LM_NOSE_TIP = 1
LM_CHIN = 152
LM_EYE_L_OUTER = 33
LM_EYE_L_INNER = 133
LM_EYE_R_INNER = 362
LM_EYE_R_OUTER = 263
LM_EYE_L_UP, LM_EYE_L_LOW = 159, 145
LM_EYE_L_UP2, LM_EYE_L_LOW2 = 160, 144
LM_EYE_R_UP, LM_EYE_R_LOW = 386, 374
LM_EYE_R_UP2, LM_EYE_R_LOW2 = 385, 373

EYE_POLY_L = (LM_EYE_L_OUTER, LM_EYE_L_UP, LM_EYE_L_INNER, LM_EYE_L_LOW)
EYE_POLY_R = (LM_EYE_R_INNER, LM_EYE_R_UP, LM_EYE_R_OUTER, LM_EYE_R_LOW)
EYE_DOTS = (LM_EYE_L_UP2, LM_EYE_L_LOW2, LM_EYE_R_UP2, LM_EYE_R_LOW2)

Box = Tuple[float, float, float, float]

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (76, 214, 92)
COLOR_AMBER = (0, 182, 255)
COLOR_RED = (66, 66, 245)
COLOR_CYAN = (255, 245, 60)
COLOR_TEXT = (240, 240, 240)
COLOR_MUTED = (148, 152, 162)
COLOR_PANEL = (24, 26, 30)
COLOR_BORDER = (70, 74, 82)
COLOR_BODY = (150, 165, 175)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

class _Hold:
    """Turns a raw boolean condition into a *sustained* flag.

    The flag only reads True once the condition stayed True for `hold_s`
    consecutive seconds; any flicker resets the timer.
    """

    __slots__ = ("hold_s", "_since")

    def __init__(self, hold_s: float) -> None:
        self.hold_s = float(hold_s)
        self._since: Optional[float] = None

    def update(self, active: bool, now: float) -> bool:
        if not active:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        if self.hold_s <= 0.0:
            return True
        return (now - self._since) >= self.hold_s

    def held_for(self, now: float) -> float:
        return 0.0 if self._since is None else now - self._since


class _MotionWindow:
    """Sliding window of head pose used to detect "reading motion".

    Uses the *range* (max-min) of pitch/yaw inside the window instead of the
    sum of deltas, so landmark jitter does not look like motion.
    """

    def __init__(self, window_s: float) -> None:
        self.window_s = float(window_s)
        self._s: Deque[Tuple[float, float, float]] = deque()

    def add(self, now: float, pitch: float, yaw: float) -> None:
        self._s.append((now, pitch, yaw))
        while self._s and now - self._s[0][0] > self.window_s:
            self._s.popleft()

    def has_motion(self, pitch_range_deg: float, yaw_range_deg: float) -> bool:
        if len(self._s) < 2:
            return False
        pitches = [s[1] for s in self._s]
        yaws = [s[2] for s in self._s]
        return (
            (max(pitches) - min(pitches)) >= pitch_range_deg
            or (max(yaws) - min(yaws)) >= yaw_range_deg
        )


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _sgn(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:+.0f}"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FocusConfig:
    """All StudyGuard thresholds in one place (defaults match the spec)."""

    yolo_weights: str = str(_PROJECT_ROOT / "yolov8n.pt")
    face_landmarker_path: str = str(_PROJECT_ROOT / "models" / "face_landmarker.task")

    person_conf: float = 0.30
    phone_conf: float = 0.35
    face_confidence: float = 0.50

    # YOLO speed knobs: 416px is ~3x faster than 640px on CPU with no
    # meaningful loss for close-up person/phone boxes. The detector runs
    # every `yolo_every_n`-th frame (cached in between) while the cheap
    # face/EAR path still runs on every frame, so eye closure stays instant.
    yolo_imgsz: int = 416
    yolo_every_n: int = 2

    phone_pitch_deg: float = 20.0
    phone_hold_s: float = 0.30
    chest_zone_ratio: float = 0.60

    slump_roll_deg: float = 28.0
    slump_pitch_deg: float = 40.0
    slump_hold_s: float = 4.0
    reading_window_s: float = 1.5
    reading_pitch_range_deg: float = 8.0
    reading_yaw_range_deg: float = 10.0

    gaze_yaw_deg: float = 30.0
    gaze_hold_s: float = 3.0

    absence_hold_s: float = 4.0

    drowsy_ear: float = 0.20
    drowsy_hold_s: float = 1.0

    ema_alpha: float = 0.25
    frame_loss_grace_s: float = 2.0


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class FocusEngine:
    """StudyGuard focus detection engine.

    Feed BGR frames to :meth:`process_frame`; it returns the annotated frame
    plus the focus score, status flags and an alert message.
    """

    STATUS_FOCUSED = "Focused"
    STATUS_PHONE = "Using Phone"
    STATUS_SLACK = "Slacking"
    STATUS_DISTRACTED = "Distracted"
    STATUS_DROWSY = "Drowsy"
    STATUS_AWAY = "Away"

    def __init__(self, config: Optional[FocusConfig] = None) -> None:
        self.cfg = config or FocusConfig()

        self.yolo = YOLO(self.cfg.yolo_weights)

        model_path = self._ensure_face_model(Path(self.cfg.face_landmarker_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.cfg.face_confidence,
            min_face_presence_confidence=self.cfg.face_confidence,
            min_tracking_confidence=self.cfg.face_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._face = mp_vision.FaceLandmarker.create_from_options(options)

        self._hold_phone = _Hold(self.cfg.phone_hold_s)
        self._hold_slump = _Hold(self.cfg.slump_hold_s)
        self._hold_gaze = _Hold(self.cfg.gaze_hold_s)
        self._hold_absence = _Hold(self.cfg.absence_hold_s)
        self._hold_drowsy = _Hold(self.cfg.drowsy_hold_s)
        self._pose_window = _MotionWindow(self.cfg.reading_window_s)

        self.phone_detected = False
        self.is_slacking = False
        self.is_distracted = False
        self.is_away = False
        self.is_drowsy = False

        self._ema = 100.0
        self._last_ts_ms = 0
        self._frame_lost_since: Optional[float] = None
        self._last_payload: Optional[Dict[str, object]] = None
        self._last_annotated: Optional[np.ndarray] = None

        self._last_person_box: Optional[Box] = None
        self._last_phone_box: Optional[Box] = None
        self._last_pts: Optional[np.ndarray] = None
        self._last_pose: Optional[Tuple[float, float, float]] = None
        self._last_ear: Optional[float] = None

        # YOLO cadence state: the heavy detector runs every N-th frame and its
        # boxes are reused in between. Face/EAR/pose always run per frame.
        self._frame_index = 0
        self._cached_person_boxes: List[Box] = []
        self._cached_phone_boxes: List[Box] = []
        self._last_frame_shape: Optional[Tuple[int, int]] = None

    # ---------------------------------------------------------------- setup #

    @staticmethod
    def _ensure_face_model(path: Path) -> Path:
        if path.exists() and path.stat().st_size > 1_000_000:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            with requests.get(FACE_LANDMARKER_URL, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
            if tmp.stat().st_size < 1_000_000:
                raise RuntimeError(
                    "Downloaded face_landmarker.task is too small - network or "
                    "mirror problem."
                )
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return path

    def close(self) -> None:
        try:
            self._face.close()
        except Exception:
            pass

    def __enter__(self) -> "FocusEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ detection #

    def _detect_objects(self, frame: np.ndarray) -> Tuple[List[Box], List[Box]]:
        persons: List[Box] = []
        phones: List[Box] = []
        try:
            conf = min(self.cfg.person_conf, self.cfg.phone_conf)
            results = self.yolo.predict(
                frame,
                conf=conf,
                classes=[PERSON_CLASS, PHONE_CLASS],
                imgsz=self.cfg.yolo_imgsz,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return persons, phones
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cls_id, score in zip(xyxy, clss, confs):
                box = (float(x1), float(y1), float(x2), float(y2))
                if cls_id == PHONE_CLASS and score >= self.cfg.phone_conf:
                    phones.append(box)
                elif cls_id == PERSON_CLASS and score >= self.cfg.person_conf:
                    persons.append(box)
        except Exception:
            return [], []
        persons.sort(key=_area, reverse=True)
        phones.sort(key=_area, reverse=True)
        return persons, phones

    def _detect_face(
        self, frame: np.ndarray, now: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
            )
            ts = int(now * 1000.0)
            if ts <= self._last_ts_ms:
                ts = self._last_ts_ms + 1
            self._last_ts_ms = ts
            result = self._face.detect_for_video(image, ts)
            if not result.face_landmarks:
                return None, None
            marks = np.array(
                [(p.x, p.y) for p in result.face_landmarks[0]], dtype=np.float64
            )
            matrix = None
            if result.facial_transformation_matrixes:
                matrix = np.asarray(result.facial_transformation_matrixes[0])
            return marks, matrix
        except Exception:
            return None, None

    # -------------------------------------------------------- measurements #

    @staticmethod
    def _head_pose(
        matrix: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float, float]]:
        """Return (pitch_down, yaw, roll) in degrees, or None if unusable."""
        if matrix is None:
            return None
        try:
            m = np.asarray(matrix, dtype=np.float64)
        except Exception:
            return None
        if m.shape != (4, 4) or not np.isfinite(m).all():
            return None
        rot = m[:3, :3]
        if not 0.75 <= float(np.linalg.det(rot)) <= 1.25:
            return None
        nx, ny, nz = float(rot[0, 2]), float(rot[1, 2]), float(rot[2, 2])
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm < 1e-6:
            return None
        nx, ny, nz = nx / norm, ny / norm, nz / norm
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -ny))))
        yaw = math.degrees(math.atan2(nx, nz))
        cos_p = max(1e-6, abs(math.cos(math.radians(pitch))))
        roll = math.degrees(
            math.asin(max(-1.0, min(1.0, float(rot[1, 0]) / cos_p)))
        )
        return pitch, yaw, roll

    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray) -> Optional[float]:
        """Standard 6-landmark EAR, averaged over the eyes that are visible."""

        def one(outer: int, inner: int, up1: int, lo1: int, up2: int, lo2: int) -> float:
            horizontal = float(np.linalg.norm(pts[outer] - pts[inner])) + 1e-6
            v1 = float(np.linalg.norm(pts[up1] - pts[lo1]))
            v2 = float(np.linalg.norm(pts[up2] - pts[lo2]))
            return (v1 + v2) / (2.0 * horizontal)

        values = [
            one(
                LM_EYE_L_OUTER, LM_EYE_L_INNER,
                LM_EYE_L_UP, LM_EYE_L_LOW, LM_EYE_L_UP2, LM_EYE_L_LOW2,
            ),
            one(
                LM_EYE_R_OUTER, LM_EYE_R_INNER,
                LM_EYE_R_UP, LM_EYE_R_LOW, LM_EYE_R_UP2, LM_EYE_R_LOW2,
            ),
        ]
        return float(sum(values) / len(values))

    # ------------------------------------------------------------- rules --- #

    def _phone_in_chest_zone(self, phone: Box, body: Box) -> bool:
        """True when the phone sits in the student's chest/face region."""
        bx1, by1, bx2, by2 = body
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        mx = 0.10 * bw
        zone = (bx1 - mx, by1 - 0.05 * bh, bx2 + mx, by1 + self.cfg.chest_zone_ratio * bh)
        px1, py1, px2, py2 = phone
        cx = (px1 + px2) * 0.5
        cy = (py1 + py2) * 0.5
        if zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]:
            return True
        ox = max(0.0, min(px2, zone[2]) - max(px1, zone[0]))
        oy = max(0.0, min(py2, zone[3]) - max(py1, zone[1]))
        return (ox * oy) / max(1.0, _area(phone)) >= 0.35

    @staticmethod
    def _face_body_box(face: Box) -> Box:
        """Expand a face box into a head+chest proxy (when YOLO misses)."""
        x1, y1, x2, y2 = face
        fw = max(1.0, x2 - x1)
        fh = max(1.0, y2 - y1)
        return (x1 - 0.8 * fw, y1 - 0.35 * fh, x2 + 0.8 * fw, y1 + 2.6 * fh)

    def _update_state(
        self,
        now: float,
        present: bool,
        body_box: Optional[Box],
        phone_box: Optional[Box],
        pose: Optional[Tuple[float, float, float]],
        ear: Optional[float],
    ) -> None:
        cfg = self.cfg
        self.is_away = self._hold_absence.update(not present, now)

        pitch = yaw = roll = None
        if pose is not None:
            pitch, yaw, roll = pose
            self._pose_window.add(now, pitch, yaw)

        phone_active = False
        if phone_box is not None:
            head_down = pitch is not None and pitch > cfg.phone_pitch_deg
            near_body = (
                body_box is not None and self._phone_in_chest_zone(phone_box, body_box)
            )
            phone_active = near_body or head_down
        self.phone_detected = self._hold_phone.update(phone_active, now)

        slump_active = False
        if pose is not None:
            reading = self._pose_window.has_motion(
                cfg.reading_pitch_range_deg, cfg.reading_yaw_range_deg
            )
            slump_active = abs(roll) > cfg.slump_roll_deg or (
                pitch > cfg.slump_pitch_deg and not reading
            )
        self.is_slacking = self._hold_slump.update(slump_active, now)

        gaze_active = pose is not None and abs(yaw) > cfg.gaze_yaw_deg
        self.is_distracted = self._hold_gaze.update(gaze_active, now)

        drowsy_active = ear is not None and ear < cfg.drowsy_ear
        self.is_drowsy = self._hold_drowsy.update(drowsy_active, now)

    # -------------------------------------------------------------- score -- #

    def _raw_score(self) -> int:
        """Instantaneous penalty-based score (no smoothing). Pure - no state."""
        raw = 100.0
        if self.phone_detected:
            raw -= 40.0
        if self.is_slacking:
            raw -= 25.0
        if self.is_distracted:
            raw -= 20.0
        if self.is_away:
            raw -= 50.0
        if self.is_drowsy:
            raw -= 30.0
        return int(round(max(0.0, raw)))

    def _apply_score(self) -> int:
        """Advance the EMA toward the raw score and return the smoothed value."""
        raw = float(self._raw_score())
        self._ema = self.cfg.ema_alpha * raw + (1.0 - self.cfg.ema_alpha) * self._ema
        return int(round(min(100.0, max(0.0, self._ema))))

    def _status_and_alert(self) -> Tuple[str, str, Tuple[int, int, int], Optional[str]]:
        """Priority ladder: (status, HUD text, colour, alert message). Pure."""
        if self.phone_detected:
            return self.STATUS_PHONE, "Using Phone", COLOR_RED, "Put your phone away!"
        if self.is_away:
            return self.STATUS_AWAY, "Away From Desk", COLOR_RED, "Return to your chair!"
        if self.is_slacking:
            return (
                self.STATUS_SLACK, "Slacking / Head Down", COLOR_AMBER, "Sit upright!"
            )
        if self.is_drowsy:
            return (
                self.STATUS_DROWSY, "Drowsy", COLOR_AMBER,
                "Stay awake - take a short break!",
            )
        if self.is_distracted:
            return (
                self.STATUS_DISTRACTED, "Looking Away", COLOR_AMBER,
                "Eyes on your books!",
            )
        return self.STATUS_FOCUSED, "Focused", COLOR_GREEN, None

    # ------------------------------------------------------- public entry -- #

    @staticmethod
    def _valid_frame(frame: object) -> bool:
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        )

    def process_frame(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Accepts: BGR OpenCV frame (from webcam or phone stream)
        Returns:
        {
            "processed_frame": np.ndarray,      # BGR frame with visual HUD overlays
            "focus_score": int,                 # 0 to 100
            "status": str,                      # "Focused" | "Using Phone" | "Slacking"
                                                # | "Distracted" | "Drowsy" | "Away"
            "is_drowsy": bool,
            "is_distracted": bool,
            "phone_detected": bool,
            "is_slacking": bool,
            "is_away": bool,
            "alert_message": str or None
        }
        """
        now = time.monotonic()
        if not self._valid_frame(frame):
            return self._handle_frame_loss(now)
        self._frame_lost_since = None

        canvas = frame
        if canvas.dtype != np.uint8:
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        canvas = canvas.copy()
        height, width = canvas.shape[:2]

        # Heavy YOLO pass runs every N-th frame (or when the frame size
        # changes); the cheap face/EAR/pose pass below runs every frame.
        self._frame_index += 1
        cadence = max(1, int(self.cfg.yolo_every_n))
        if (
            self._frame_index == 1
            or (height, width) != self._last_frame_shape
            or (self._frame_index % cadence) == 1
        ):
            person_boxes, phone_boxes = self._detect_objects(canvas)
            self._cached_person_boxes = person_boxes
            self._cached_phone_boxes = phone_boxes
            self._last_frame_shape = (height, width)
        else:
            person_boxes = self._cached_person_boxes
            phone_boxes = self._cached_phone_boxes
        person_box = person_boxes[0] if person_boxes else None
        phone_box = phone_boxes[0] if phone_boxes else None

        marks, matrix = self._detect_face(canvas, now)
        pts: Optional[np.ndarray] = None
        face_box: Optional[Box] = None
        ear: Optional[float] = None
        pose = self._head_pose(matrix)
        if marks is not None:
            pts = np.stack([marks[:, 0] * width, marks[:, 1] * height], axis=1)
            face_box = (
                float(pts[:, 0].min()),
                float(pts[:, 1].min()),
                float(pts[:, 0].max()),
                float(pts[:, 1].max()),
            )
            ear = self._eye_aspect_ratio(pts)

        body_box = person_box
        if body_box is None and face_box is not None:
            body_box = self._face_body_box(face_box)

        present = person_box is not None or marks is not None
        self._update_state(now, present, body_box, phone_box, pose, ear)

        self._last_person_box = person_box if person_box is not None else face_box
        self._last_phone_box = phone_box
        self._last_pts = pts
        self._last_pose = pose
        self._last_ear = ear

        payload = self._finalize(canvas)
        self._last_payload = payload
        self._last_annotated = canvas
        return payload

    def _handle_frame_loss(self, now: float) -> Dict[str, object]:
        """Camera dropped / returned a corrupt frame: hold state, then degrade."""
        if self._frame_lost_since is None:
            self._frame_lost_since = now
        lost_for = now - self._frame_lost_since

        if lost_for <= self.cfg.frame_loss_grace_s and self._last_payload is not None:
            payload = dict(self._last_payload)
            last = self._last_annotated
            payload["processed_frame"] = last.copy() if last is not None else np.zeros(
                (480, 640, 3), dtype=np.uint8
            )
            if payload.get("alert_message") is None:
                payload["alert_message"] = "Camera frame lost"
            return payload

        if self._last_annotated is not None:
            canvas = self._last_annotated.copy()
        else:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self._update_state(now, False, None, None, None, None)
        return self._finalize(canvas)

    def _finalize(self, canvas: np.ndarray) -> Dict[str, object]:
        score = self._apply_score()
        status, hud_text, color, alert = self._status_and_alert()
        self._annotate(canvas, score, hud_text, color, alert)
        return {
            "processed_frame": canvas,
            "focus_score": int(score),
            "status": status,
            "is_drowsy": bool(self.is_drowsy),
            "is_distracted": bool(self.is_distracted),
            "phone_detected": bool(self.phone_detected),
            "is_slacking": bool(self.is_slacking),
            "is_away": bool(self.is_away),
            "alert_message": alert,
        }

    # ---------------------------------------------------------- rendering -- #

    @staticmethod
    def _draw_box(
        canvas: np.ndarray,
        box: Box,
        color: Tuple[int, int, int],
        label: str,
        thickness: int,
    ) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        (tw, th) = cv2.getTextSize(label, FONT, 0.58, 2)[0]
        ly = y1 - 8
        if ly - th - 10 < 0:
            ly = y1 + th + 12
        cv2.rectangle(
            canvas, (x1, ly - th - 6), (x1 + tw + 10, ly + 6), color, -1
        )
        cv2.putText(
            canvas, label, (x1 + 5, ly), FONT, 0.58, (255, 255, 255), 2, cv2.LINE_AA
        )

    @staticmethod
    def _draw_eyes(canvas: np.ndarray, pts: np.ndarray) -> None:
        for poly in (EYE_POLY_L, EYE_POLY_R):
            shape = np.array([pts[i] for i in poly], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [shape], True, COLOR_CYAN, 1, cv2.LINE_AA)
        for idx in EYE_DOTS:
            cv2.circle(
                canvas,
                (int(pts[idx][0]), int(pts[idx][1])),
                2,
                COLOR_CYAN,
                -1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_hud(
        canvas: np.ndarray,
        score: int,
        hud_text: str,
        color: Tuple[int, int, int],
        ear: Optional[float],
        pose: Optional[Tuple[float, float, float]],
    ) -> None:
        x0, y0, cw, ch = 16, 16, 340, 154
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + cw, y0 + ch), COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.74, canvas, 0.26, 0.0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y0 + ch), COLOR_BORDER, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + 5, y0 + ch), color, -1)

        cv2.putText(
            canvas, "FOCUS SCORE", (x0 + 22, y0 + 32),
            FONT, 0.47, COLOR_MUTED, 1, cv2.LINE_AA,
        )
        score_color = COLOR_GREEN if score >= 75 else (COLOR_AMBER if score >= 50 else COLOR_RED)
        score_text = str(score)
        cv2.putText(
            canvas, score_text, (x0 + 22, y0 + 92),
            FONT, 1.5, score_color, 3, cv2.LINE_AA,
        )
        (tw, _) = cv2.getTextSize(score_text, FONT, 1.5, 3)[0]
        cv2.putText(
            canvas, "/100", (x0 + 30 + tw, y0 + 92),
            FONT, 0.55, COLOR_MUTED, 1, cv2.LINE_AA,
        )

        cv2.circle(canvas, (x0 + 29, y0 + 119), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas, hud_text, (x0 + 44, y0 + 124),
            FONT, 0.6, COLOR_TEXT, 2, cv2.LINE_AA,
        )

        pitch = yaw = roll = None
        if pose is not None:
            pitch, yaw, roll = pose
        metrics = (
            f"EAR {_fmt(ear)}   PITCH {_sgn(pitch)}   "
            f"YAW {_sgn(yaw)}   ROLL {_sgn(roll)}"
        )
        cv2.putText(
            canvas, metrics, (x0 + 22, y0 + 145),
            FONT, 0.45, COLOR_MUTED, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_alert(
        canvas: np.ndarray, alert: str, color: Tuple[int, int, int]
    ) -> None:
        height, width = canvas.shape[:2]
        (tw, th) = cv2.getTextSize(alert, FONT, 0.7, 2)[0]
        bw = min(width - 24, tw + 36)
        bh = th + 24
        bx = max(12, (width - bw) // 2)
        by = height - bh - 18
        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (20, 20, 24), -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0.0, canvas)
        cv2.rectangle(canvas, (bx, by), (bx + 4, by + bh), color, -1)
        cv2.putText(
            canvas, alert, (bx + 18, by + th + 8),
            FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _annotate(
        self,
        canvas: np.ndarray,
        score: int,
        hud_text: str,
        color: Tuple[int, int, int],
        alert: Optional[str],
    ) -> None:
        if self._last_phone_box is not None:
            active = self.phone_detected
            self._draw_box(
                canvas,
                self._last_phone_box,
                (0, 0, 255) if active else (0, 170, 255),
                "PHONE [ALERT]" if active else "PHONE",
                3 if active else 2,
            )
        if self._last_person_box is not None:
            self._draw_box(canvas, self._last_person_box, COLOR_BODY, "STUDENT", 1)
        if self._last_pts is not None:
            self._draw_eyes(canvas, self._last_pts)
        self._draw_hud(canvas, score, hud_text, color, self._last_ear, self._last_pose)
        if alert:
            self._draw_alert(canvas, alert, color)


# --------------------------------------------------------------------------- #
# Demo entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    engine = FocusEngine()
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    last_status = None
    try:
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
            if not cap.isOpened():
                raise SystemExit(f"Cannot open camera {source}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out = engine.process_frame(frame)
                if out["status"] != last_status:
                    last_status = out["status"]
                    print(
                        f"[{out['status']}] score={out['focus_score']} "
                        f"alert={out['alert_message']}"
                    )
                cv2.imshow("StudyGuard - Focus Engine", out["processed_frame"])
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            cap.release()
        elif Path(source).suffix.lower() in image_ext:
            frame = cv2.imread(source)
            if frame is None:
                raise SystemExit(f"Cannot read image {source}")
            out = engine.process_frame(frame)
            print(out["status"], out["focus_score"], out["alert_message"])
            cv2.imshow("StudyGuard - Focus Engine", out["processed_frame"])
            cv2.waitKey(0)
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise SystemExit(f"Cannot open video {source}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out = engine.process_frame(frame)
                cv2.imshow("StudyGuard - Focus Engine", out["processed_frame"])
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            cap.release()
    finally:
        engine.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
