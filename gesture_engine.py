from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import pyautogui

from gesture_keyboard_v3 import (
    BaseOptions,
    BrightnessController,
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RedPenTracker,
    VisionRunningMode,
    classify_right_hand,
    clamp,
    draw_landmarks,
    ensure_models,
    fingers_up,
    hand_openness_ratio,
    mouth_open_ratio,
    pinch_ratio,
    VolumeController,
)

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

GESTURE_MODEL_PATH = "gesture_recognizer.task"
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)


@dataclass
class EngineSettings:
    camera: Optional[int]
    cooldown: float
    hold_threshold: float
    pen_sensitivity: float
    key_output_enabled: bool
    mirror_preview: bool
    click_speed_profile: str


@dataclass
class PinchCalibration:
    close_world: float = 0.22
    open_world: float = 0.28
    close_image: float = 0.30
    open_image: float = 0.36
    source_preference: str = "auto"


@dataclass
class PointerCalibration:
    deadzone_base_px: float = 2.5
    smoothness_scalar: float = 1.0


@dataclass
class CalibrationQuality:
    ready: bool
    score: float
    warnings: list[str] = field(default_factory=list)
    sample_counts: dict[str, int] = field(default_factory=dict)
    separation_world: Optional[float] = None
    separation_image: Optional[float] = None
    used_source: str = "image"


@dataclass
class CalibrationSession:
    active: bool = False
    safe_mode: bool = True
    current_step: Optional[str] = None
    capture_started_at: Optional[float] = None
    capture_duration_ms: int = 0
    capture_completed: bool = False
    derived_ready: bool = False
    samples: dict[str, list[dict[str, float]]] = field(
        default_factory=lambda: {
            "open_pinch": [],
            "closed_pinch": [],
            "pointer_still": [],
            "pointer_move": [],
        }
    )
    derived_pinch: Optional[PinchCalibration] = None
    derived_pointer: Optional[PointerCalibration] = None
    quality: Optional[CalibrationQuality] = None
    previous_snapshot: Optional[dict[str, Any]] = None


@dataclass
class CalibrationProfile:
    profile_id: str
    name: str
    created_at: float
    updated_at: float
    settings: EngineSettings
    pinch: PinchCalibration
    pointer: PointerCalibration


_UNSET = object()


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    index = clamp(q, 0.0, 1.0) * (len(ordered) - 1)
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    frac = index - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


class OneEuroFilter1D:
    def __init__(self) -> None:
        self.value_prev: Optional[float] = None
        self.derivative_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        cutoff = max(cutoff, 1e-6)
        dt = max(dt, 1e-6)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.value_prev = None
        self.derivative_prev = 0.0
        self.t_prev = None

    def filter(self, value: float, timestamp: float, min_cutoff: float, beta: float, d_cutoff: float) -> float:
        if self.value_prev is None or self.t_prev is None:
            self.value_prev = value
            self.t_prev = timestamp
            self.derivative_prev = 0.0
            return value

        dt = max(timestamp - self.t_prev, 1e-6)
        raw_derivative = (value - self.value_prev) / dt
        alpha_d = self._alpha(dt, d_cutoff)
        derivative = alpha_d * raw_derivative + (1.0 - alpha_d) * self.derivative_prev
        cutoff = min_cutoff + beta * abs(derivative)
        alpha = self._alpha(dt, cutoff)
        filtered = alpha * value + (1.0 - alpha) * self.value_prev

        self.derivative_prev = derivative
        self.value_prev = filtered
        self.t_prev = timestamp
        return filtered


class OneEuro2DFilter:
    def __init__(self) -> None:
        self.fx = OneEuroFilter1D()
        self.fy = OneEuroFilter1D()

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def filter(self, point: tuple[float, float], timestamp: float, profile: dict[str, float]) -> tuple[float, float]:
        x = self.fx.filter(
            point[0],
            timestamp,
            min_cutoff=profile["min_cutoff"],
            beta=profile["beta"],
            d_cutoff=profile["d_cutoff"],
        )
        y = self.fy.filter(
            point[1],
            timestamp,
            min_cutoff=profile["min_cutoff"],
            beta=profile["beta"],
            d_cutoff=profile["d_cutoff"],
        )
        return x, y


class GestureEngine:
    def __init__(self, default_camera: Optional[int], key_output_enabled: bool) -> None:
        self.running = False
        self.cap = None
        self.capture_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.latest_result = None
        self.result_lock = threading.Lock()
        self.latest_face_result = None
        self.face_result_lock = threading.Lock()
        self.latest_gesture_result = None
        self.gesture_result_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.last_frame_jpeg: Optional[bytes] = None
        self._last_timestamp_ms = int(time.monotonic() * 1000)
        self.latest_frame_lock = threading.Lock()
        self.latest_frame_ready = threading.Event()
        self.latest_captured_frame: Optional[tuple[int, float, object]] = None
        self.capture_sequence = 0

        self.capture_fps = 0.0
        self.inference_fps = 0.0
        self.dropped_frames = 0
        self.end_to_end_latency_ms = 0.0
        self._capture_window_start = time.monotonic()
        self._capture_window_count = 0
        self._inference_window_start = time.monotonic()
        self._inference_window_count = 0

        self.landmarker = None
        self.face_landmarker = None
        self.gesture_recognizer = None

        self.cooldown = 0.8
        self.hold_threshold = 0.5
        self.last_trigger_time = 0.0
        self.right_hold_start: Optional[float] = None
        self.prev_right_gesture: Optional[str] = None
        self.right_hold_progress = 0.0
        self.right_gesture_source = "none"
        self.right_gesture_confidence = 0.0
        self.right_confirm_frames_required = 2
        self.right_pending_candidate_id: Optional[str] = None
        self.right_pending_frames = 0
        self.right_recognizer_thresholds = {
            "Closed_Fist": 0.72,
            "Pointing_Up": 0.68,
            "Victory": 0.68,
            "Open_Palm": 0.70,
            "Thumb_Up": 0.70,
            "ILoveYou": 0.72,
        }
        self.right_landmark_thresholds = {
            "space": 0.65,
            "up": 0.65,
            "down": 0.65,
            "left": 0.65,
            "right": 0.68,
            "enter": 0.65,
            "ctrl+v": 0.68,
            "escape": 0.68,
            "ctrl+c": 0.68,
        }

        self.key_count = 0
        self.last_key_text = "-"
        self.pending_status_message: Optional[str] = None
        self.enabled = key_output_enabled
        self.mirror_preview = True
        self.default_camera = default_camera
        self.active_camera_index: Optional[int] = None

        self.left_mode = "idle"
        self.left_dragging = False
        self.left_anchor_y: Optional[float] = None
        self.left_base_value: Optional[int] = None
        self.left_last_value: Optional[int] = None
        self.left_pinch_start: Optional[float] = None
        self.left_pinch_mode: Optional[str] = None
        self.left_pinch_target: Optional[tuple[int, int]] = None
        self.left_right_click_fired = False
        self.brightness_arm_start: Optional[float] = None
        self.brightness_exit_gesture_start: Optional[float] = None
        self.brightness_arm_hold = 2
        self.brightness_exit_hold = 0.25

        # Pinch interaction FSM: open -> arming -> pinched -> dragging -> release -> cooldown
        self.pinch_state = "open"
        self.pinch_state_since = time.monotonic()
        self.pinch_mode_pen = False
        self.pinch_started_at: Optional[float] = None
        self.pinch_arming_started_at: Optional[float] = None
        self.pinch_close_frames = 0
        self.pinch_anchor_point: Optional[tuple[float, float]] = None
        self.pinch_last_point: Optional[tuple[int, int]] = None
        self.pinch_max_move_px = 0.0
        self.pinch_release_handled = False
        self.pinch_ratio_ema: Optional[float] = None
        self.pinch_ratio_ema_alpha = 0.45
        self.pinch_last_ratio: Optional[float] = None
        self.pinch_ratio_source = "none"
        self.middle_pinch_ratio_ema: Optional[float] = None
        self.pinch_intent: Optional[str] = None
        self.latest_left_image_ratio: Optional[float] = None
        self.latest_left_world_ratio: Optional[float] = None
        self.pinch_cooldown_started_at: Optional[float] = None
        self.click_speed_profile = "turbo"
        # Defaults are overwritten by _apply_click_speed_profile().
        self.pinch_arming_min_seconds = 0.07
        self.pinch_arming_min_frames = 3
        self.pinch_right_click_confirm_seconds = 0.12
        self.pinch_drag_hold_seconds = 0.22
        self.pinch_click_max_seconds = 0.22
        self.pinch_drag_move_threshold_px = 20.0
        self.pinch_cooldown_seconds = 0.12
        self.pinch_release_linger_seconds = 0.02
        self.pinch_calibration = PinchCalibration()
        self.pinch_close_world = self.pinch_calibration.close_world
        self.pinch_open_world = self.pinch_calibration.open_world
        self.pinch_close_image = self.pinch_calibration.close_image
        self.pinch_open_image = self.pinch_calibration.open_image
        self._apply_click_speed_profile(self.click_speed_profile)

        self.screen_width, self.screen_height = pyautogui.size()
        self.volume = VolumeController()
        self.brightness = BrightnessController()
        # Brightness I/O can block on many systems; keep reads/writes throttled.
        self._brightness_cached_percent: Optional[int] = None
        self._brightness_cache_time = 0.0
        self._brightness_read_interval = 0.12
        self._brightness_last_write_time = 0.0
        self._brightness_write_interval = 0.08
        self._brightness_pending_target: Optional[int] = None
        self.red_pen = RedPenTracker(screen_width=self.screen_width, screen_height=self.screen_height)
        self.pen_sensitivity = self.red_pen.sensitivity_percent
        self.pen_state = "no-right-hand"
        self.pen_target: Optional[tuple[int, int]] = None
        self.pen_frame_point: Optional[tuple[int, int]] = None
        self.pen_roi: Optional[tuple[int, int, int, int]] = None
        self.warned_volume_unavailable = False
        self.warned_brightness_unavailable = False
        self.pointer_mode = "idle"
        self.pointer_filter = OneEuro2DFilter()
        self._pointer_profile_bases = {
            "move": {"deadzone_px": 2.5, "min_cutoff": 1.5, "beta": 0.08, "d_cutoff": 1.0},
            "precision": {"deadzone_px": 1.5, "min_cutoff": 1.1, "beta": 0.04, "d_cutoff": 1.0},
            "drag": {"deadzone_px": 1.0, "min_cutoff": 0.9, "beta": 0.03, "d_cutoff": 1.0},
        }
        self.pointer_calibration = PointerCalibration()
        self.pointer_profiles = self._build_pointer_profiles(
            self.pointer_calibration.deadzone_base_px,
            self.pointer_calibration.smoothness_scalar,
        )
        self.pointer_last_raw: Optional[tuple[float, float]] = None
        self.pointer_last_filtered: Optional[tuple[float, float]] = None
        self.pointer_was_active = False

        self.mouth_threshold = 0.08
        self.mouth_hold_seconds = 5.0
        self.mouth_enabled = True
        self.mouth_open_start: Optional[float] = None
        self.mouth_triggered = False
        self.mouth_open_active = False
        self.mouth_signal_mode = "ratio_fallback"
        self.mouth_blend_score: Optional[float] = None
        self.mouth_ratio_norm: Optional[float] = None
        self.mouth_signal_ema: Optional[float] = None
        self.mouth_ratio_min = 0.045
        self.mouth_ratio_max = 0.11
        self.mouth_hybrid_open_threshold = 0.62
        self.mouth_hybrid_close_threshold = 0.48
        self.mouth_ratio_ema_alpha = 0.35

        self.warning_history: deque[str] = deque(maxlen=8)
        self.error_history: deque[str] = deque(maxlen=8)
        self.calibration_lock = threading.Lock()
        self.calibration_session = CalibrationSession()
        self.calibration_profiles: dict[str, CalibrationProfile] = {}
        self.active_profile_id: Optional[str] = None
        self.active_profile_name = "Default"
        self.calibration_safe_mode = False
        self._profile_sync_suspended = False
        self.calibration_last_snapshot: Optional[dict[str, Any]] = None
        self.calibration_store_path = self._resolve_calibration_store_path()
        self.calibration_sampling_defaults = {
            "open_pinch": 2200,
            "closed_pinch": 2200,
            "pointer_still": 2500,
            "pointer_move": 3000,
        }
        self.calibration_min_samples = {
            "open_pinch": 20,
            "closed_pinch": 20,
            "pointer_still": 18,
            "pointer_move": 24,
        }
        self.calibration_quality_min_score = 0.55

        self.state = {
            "session_status": "initializing",
            "running": False,
            "camera_index": None,
            "gestureText": "R: -\nL: -\nM: -\nP: -",
            "rightText": "idle",
            "leftText": "idle",
            "mouthText": "disabled",
            "penState": "no-right-hand",
            "lastKey": "-",
            "keyCount": 0,
            "status": "Loading gesture model...",
            "holdProgress": 0.0,
            "warnings": [],
            "errors": [],
            "metrics": {
                "capture_fps": 0.0,
                "inference_fps": 0.0,
                "dropped_frames": 0,
                "end_to_end_latency_ms": 0.0,
            },
            "interaction": {
                "pinch_state": self.pinch_state,
                "pinch_ratio": None,
                "pinch_source": "none",
                "pointer_mode": self.pointer_mode,
                "right_gesture_source": self.right_gesture_source,
                "right_gesture_confidence": self.right_gesture_confidence,
                "mouth_blend_score": self.mouth_blend_score,
                "mouth_signal_mode": self.mouth_signal_mode,
            },
            "calibration": self._calibration_state_snapshot(),
            "updatedAt": time.time(),
        }

        self._load_calibration_profiles()
        self._apply_active_profile_to_runtime()
        self._init_landmarker()
        with self.state_lock:
            self.state["session_status"] = "idle"
            self.state["warnings"] = list(self.warning_history)
            self.state["errors"] = list(self.error_history)
            self.state["calibration"] = self._calibration_state_snapshot()
            if self.face_landmarker is None:
                self.state["status"] = "Face model unavailable. Mouth-close is disabled."
            elif self.gesture_recognizer is None:
                self.state["status"] = "Model ready. Gesture recognizer unavailable; using landmark fallback."
            else:
                self.state["status"] = "Model ready. Start session to open the webcam."

    def close(self) -> None:
        self.stop()
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        if self.face_landmarker is not None:
            self.face_landmarker.close()
            self.face_landmarker = None
        if self.gesture_recognizer is not None:
            self.gesture_recognizer.close()
            self.gesture_recognizer = None

    def _ensure_gesture_model(self) -> bool:
        if os.path.exists(GESTURE_MODEL_PATH):
            return True
        try:
            urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)
            return True
        except Exception as exc:
            self._record_warning(f"Gesture recognizer model unavailable: {exc}")
            return False

    def _init_landmarker(self) -> None:
        def callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
            with self.result_lock:
                self.latest_result = result

        def face_callback(result: Any, output_image: mp.Image, timestamp_ms: int) -> None:
            with self.face_result_lock:
                self.latest_face_result = result

        def gesture_callback(result: Any, output_image: mp.Image, timestamp_ms: int) -> None:
            with self.gesture_result_lock:
                self.latest_gesture_result = result

        face_model_ready = ensure_models()
        self.landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                result_callback=callback,
            )
        )
        if face_model_ready:
            self.face_landmarker = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
                    running_mode=VisionRunningMode.LIVE_STREAM,
                    num_faces=1,
                    min_face_detection_confidence=0.6,
                    min_face_presence_confidence=0.6,
                    min_tracking_confidence=0.6,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                    result_callback=face_callback,
                )
            )
        else:
            self.face_landmarker = None

        self.gesture_recognizer = None
        if self._ensure_gesture_model():
            try:
                self.gesture_recognizer = GestureRecognizer.create_from_options(
                    GestureRecognizerOptions(
                        base_options=BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
                        running_mode=VisionRunningMode.LIVE_STREAM,
                        num_hands=2,
                        min_hand_detection_confidence=0.65,
                        min_hand_presence_confidence=0.6,
                        min_tracking_confidence=0.6,
                        result_callback=gesture_callback,
                    )
                )
            except Exception as exc:
                self.gesture_recognizer = None
                self._record_warning(f"Gesture recognizer init failed, using landmark fallback: {exc}")

    def _record_warning(self, message: str) -> None:
        self.warning_history.appendleft(message)

    def _record_error(self, message: str) -> None:
        self.error_history.appendleft(message)

    def _apply_click_speed_profile(self, profile: str) -> None:
        presets = {
            "balanced": {
                "arming_seconds": 0.07,
                "arming_frames": 3,
                "right_click_seconds": 0.12,
                "drag_hold_seconds": 0.22,
                "click_max_seconds": 0.22,
                "cooldown_seconds": 0.12,
            },
            "fast": {
                "arming_seconds": 0.04,
                "arming_frames": 2,
                "right_click_seconds": 0.07,
                "drag_hold_seconds": 0.16,
                "click_max_seconds": 0.16,
                "cooldown_seconds": 0.06,
            },
            "turbo": {
                "arming_seconds": 0.025,
                "arming_frames": 1,
                "right_click_seconds": 0.04,
                "drag_hold_seconds": 0.13,
                "click_max_seconds": 0.14,
                "cooldown_seconds": 0.03,
            },
            "ultra": {
                "arming_seconds": 0.015,
                "arming_frames": 1,
                "right_click_seconds": 0.03,
                "drag_hold_seconds": 0.11,
                "click_max_seconds": 0.12,
                "cooldown_seconds": 0.02,
            },
        }
        normalized = (profile or "").strip().lower()
        chosen = presets.get(normalized, presets["fast"])
        self.click_speed_profile = normalized if normalized in presets else "fast"
        self.pinch_arming_min_seconds = chosen["arming_seconds"]
        self.pinch_arming_min_frames = chosen["arming_frames"]
        self.pinch_right_click_confirm_seconds = chosen["right_click_seconds"]
        self.pinch_drag_hold_seconds = chosen["drag_hold_seconds"]
        self.pinch_click_max_seconds = chosen["click_max_seconds"]
        self.pinch_cooldown_seconds = chosen["cooldown_seconds"]

    def _resolve_calibration_store_path(self) -> Path:
        appdata_dir = os.getenv("APPDATA")
        if appdata_dir:
            return Path(appdata_dir) / "GestureKeys" / "calibration_profiles_v1.json"
        return Path(__file__).resolve().parent / "calibration_profiles_v1.json"

    def _build_pointer_profiles(self, deadzone_base_px: float, smoothness_scalar: float) -> dict[str, dict[str, float]]:
        deadzone_base = clamp(float(deadzone_base_px), 0.8, 5.5)
        scalar = clamp(float(smoothness_scalar), 0.65, 1.45)
        deadzone_factor = {"move": 1.0, "precision": 0.6, "drag": 0.4}

        profiles: dict[str, dict[str, float]] = {}
        for mode, base in self._pointer_profile_bases.items():
            factor = deadzone_factor.get(mode, 1.0)
            profiles[mode] = {
                "deadzone_px": round(clamp(deadzone_base * factor, 0.6, 8.0), 3),
                "min_cutoff": round(clamp(base["min_cutoff"] / scalar, 0.45, 2.2), 3),
                "beta": round(clamp(base["beta"] / scalar, 0.01, 0.2), 4),
                "d_cutoff": float(base["d_cutoff"]),
            }
        return profiles

    def _apply_pointer_calibration(self, pointer: PointerCalibration) -> None:
        self.pointer_calibration = PointerCalibration(
            deadzone_base_px=clamp(float(pointer.deadzone_base_px), 0.8, 5.5),
            smoothness_scalar=clamp(float(pointer.smoothness_scalar), 0.65, 1.45),
        )
        self.pointer_profiles = self._build_pointer_profiles(
            self.pointer_calibration.deadzone_base_px,
            self.pointer_calibration.smoothness_scalar,
        )
        self._reset_pointer_filter()

    def _apply_pinch_calibration(self, pinch: PinchCalibration) -> None:
        self.pinch_calibration = PinchCalibration(
            close_world=clamp(float(pinch.close_world), 0.08, 0.55),
            open_world=clamp(float(pinch.open_world), 0.12, 0.70),
            close_image=clamp(float(pinch.close_image), 0.12, 0.65),
            open_image=clamp(float(pinch.open_image), 0.16, 0.78),
            source_preference=str(pinch.source_preference or "auto"),
        )
        min_world_gap = 0.035
        if self.pinch_calibration.open_world - self.pinch_calibration.close_world < min_world_gap:
            self.pinch_calibration.open_world = clamp(
                self.pinch_calibration.close_world + min_world_gap,
                0.14,
                0.75,
            )
        min_image_gap = 0.05
        if self.pinch_calibration.open_image - self.pinch_calibration.close_image < min_image_gap:
            self.pinch_calibration.open_image = clamp(
                self.pinch_calibration.close_image + min_image_gap,
                0.2,
                0.80,
            )
        self.pinch_close_world = self.pinch_calibration.close_world
        self.pinch_open_world = self.pinch_calibration.open_world
        self.pinch_close_image = self.pinch_calibration.close_image
        self.pinch_open_image = self.pinch_calibration.open_image

    def _current_engine_settings(self) -> EngineSettings:
        return EngineSettings(
            camera=self.default_camera,
            cooldown=float(self.cooldown),
            hold_threshold=float(self.hold_threshold),
            pen_sensitivity=float(self.pen_sensitivity),
            key_output_enabled=bool(self.enabled),
            mirror_preview=bool(self.mirror_preview),
            click_speed_profile=str(self.click_speed_profile),
        )

    def _create_default_profile(self) -> CalibrationProfile:
        now = time.time()
        return CalibrationProfile(
            profile_id="default",
            name="Default",
            created_at=now,
            updated_at=now,
            settings=self._current_engine_settings(),
            pinch=PinchCalibration(**asdict(self.pinch_calibration)),
            pointer=PointerCalibration(**asdict(self.pointer_calibration)),
        )

    def _profile_to_dict(self, profile: CalibrationProfile) -> dict[str, Any]:
        return {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "settings": asdict(profile.settings),
            "pinch": asdict(profile.pinch),
            "pointer": asdict(profile.pointer),
        }

    def _profile_from_dict(self, payload: dict[str, Any]) -> Optional[CalibrationProfile]:
        try:
            profile_id = str(payload.get("profile_id") or payload.get("id") or "").strip()
            if not profile_id:
                profile_id = f"profile-{uuid.uuid4().hex[:8]}"
            name = str(payload.get("name") or "Profile").strip() or "Profile"
            created_at = float(payload.get("created_at") or time.time())
            updated_at = float(payload.get("updated_at") or created_at)
            settings_raw = payload.get("settings") or {}
            pinch_raw = payload.get("pinch") or {}
            pointer_raw = payload.get("pointer") or {}
            settings = EngineSettings(
                camera=None if settings_raw.get("camera") is None else int(settings_raw.get("camera")),
                cooldown=clamp(float(settings_raw.get("cooldown", self.cooldown)), 0.2, 2.0),
                hold_threshold=clamp(float(settings_raw.get("hold_threshold", self.hold_threshold)), 0.1, 1.5),
                pen_sensitivity=clamp(float(settings_raw.get("pen_sensitivity", self.pen_sensitivity)), 0.0, 100.0),
                key_output_enabled=bool(settings_raw.get("key_output_enabled", self.enabled)),
                mirror_preview=bool(settings_raw.get("mirror_preview", self.mirror_preview)),
                click_speed_profile=str(settings_raw.get("click_speed_profile", self.click_speed_profile)),
            )
            pinch = PinchCalibration(
                close_world=float(pinch_raw.get("close_world", self.pinch_close_world)),
                open_world=float(pinch_raw.get("open_world", self.pinch_open_world)),
                close_image=float(pinch_raw.get("close_image", self.pinch_close_image)),
                open_image=float(pinch_raw.get("open_image", self.pinch_open_image)),
                source_preference=str(pinch_raw.get("source_preference", "auto")),
            )
            pointer = PointerCalibration(
                deadzone_base_px=float(pointer_raw.get("deadzone_base_px", self.pointer_calibration.deadzone_base_px)),
                smoothness_scalar=float(pointer_raw.get("smoothness_scalar", self.pointer_calibration.smoothness_scalar)),
            )
            return CalibrationProfile(
                profile_id=profile_id,
                name=name,
                created_at=created_at,
                updated_at=updated_at,
                settings=settings,
                pinch=pinch,
                pointer=pointer,
            )
        except Exception:
            return None

    def _save_calibration_profiles(self) -> None:
        try:
            self.calibration_store_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "active_profile_id": self.active_profile_id,
                "profiles": [self._profile_to_dict(profile) for profile in self.calibration_profiles.values()],
            }
            self.calibration_store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self._record_warning(f"Calibration profiles were not saved: {exc}")

    def _load_calibration_profiles(self) -> None:
        default_profile = self._create_default_profile()
        self.calibration_profiles = {default_profile.profile_id: default_profile}
        self.active_profile_id = default_profile.profile_id
        self.active_profile_name = default_profile.name

        if not self.calibration_store_path.exists():
            self._save_calibration_profiles()
            return
        try:
            payload = json.loads(self.calibration_store_path.read_text(encoding="utf-8"))
            parsed: dict[str, CalibrationProfile] = {}
            for item in payload.get("profiles", []):
                profile = self._profile_from_dict(item)
                if profile is not None:
                    parsed[profile.profile_id] = profile
            if parsed:
                self.calibration_profiles = parsed
            requested_active = str(payload.get("active_profile_id") or "").strip()
            if requested_active and requested_active in self.calibration_profiles:
                self.active_profile_id = requested_active
            elif self.active_profile_id not in self.calibration_profiles:
                self.active_profile_id = next(iter(self.calibration_profiles))
            self.active_profile_name = self.calibration_profiles[self.active_profile_id].name
        except Exception as exc:
            self._record_warning(f"Calibration profiles file could not be loaded, using defaults: {exc}")
            self._save_calibration_profiles()

    def _get_active_profile(self) -> Optional[CalibrationProfile]:
        if self.active_profile_id and self.active_profile_id in self.calibration_profiles:
            return self.calibration_profiles[self.active_profile_id]
        if not self.calibration_profiles:
            return None
        self.active_profile_id = next(iter(self.calibration_profiles))
        return self.calibration_profiles[self.active_profile_id]

    def _apply_active_profile_to_runtime(self) -> None:
        active = self._get_active_profile()
        if active is None:
            return
        self.active_profile_name = active.name
        self._profile_sync_suspended = True
        try:
            self.update_settings(
                camera=active.settings.camera,
                cooldown=active.settings.cooldown,
                hold_threshold=active.settings.hold_threshold,
                pen_sensitivity=active.settings.pen_sensitivity,
                key_output_enabled=active.settings.key_output_enabled,
                mirror_preview=active.settings.mirror_preview,
                click_speed_profile=active.settings.click_speed_profile,
                _autosync_profile=False,
            )
            self._apply_pinch_calibration(active.pinch)
            self._apply_pointer_calibration(active.pointer)
        finally:
            self._profile_sync_suspended = False

    def _sync_active_profile_from_runtime(self) -> None:
        if self._profile_sync_suspended:
            return
        active = self._get_active_profile()
        if active is None:
            return
        active.settings = self._current_engine_settings()
        active.pinch = PinchCalibration(
            close_world=self.pinch_calibration.close_world,
            open_world=self.pinch_calibration.open_world,
            close_image=self.pinch_calibration.close_image,
            open_image=self.pinch_calibration.open_image,
            source_preference=self.pinch_calibration.source_preference,
        )
        active.pointer = PointerCalibration(
            deadzone_base_px=self.pointer_calibration.deadzone_base_px,
            smoothness_scalar=self.pointer_calibration.smoothness_scalar,
        )
        active.updated_at = time.time()
        self.active_profile_name = active.name
        self._save_calibration_profiles()

    def _calibration_state_snapshot(self) -> dict[str, Any]:
        with self.calibration_lock:
            session = self.calibration_session
            now = time.monotonic()
            remaining_ms = 0
            if session.capture_started_at is not None and session.current_step:
                elapsed = int((now - session.capture_started_at) * 1000.0)
                remaining_ms = max(session.capture_duration_ms - elapsed, 0)
                if remaining_ms <= 0:
                    session.capture_completed = True
                    session.current_step = None
                    session.capture_started_at = None
                    session.capture_duration_ms = 0
            sample_counts = {step: len(values) for step, values in session.samples.items()}
            quality = session.quality
            quality_payload = None
            if quality is not None:
                quality_payload = {
                    "ready": bool(quality.ready),
                    "score": round(float(quality.score), 3),
                    "warnings": list(quality.warnings),
                    "sample_counts": dict(quality.sample_counts),
                    "separation_world": quality.separation_world,
                    "separation_image": quality.separation_image,
                    "used_source": quality.used_source,
                }
            derived_payload = None
            if session.derived_pinch is not None and session.derived_pointer is not None:
                derived_payload = {
                    "pinch": asdict(session.derived_pinch),
                    "pointer": asdict(session.derived_pointer),
                    "profiles": self._build_pointer_profiles(
                        session.derived_pointer.deadzone_base_px,
                        session.derived_pointer.smoothness_scalar,
                    ),
                }
            return {
                "safe_mode": bool(self.calibration_safe_mode),
                "active_profile_id": self.active_profile_id,
                "active_profile_name": self.active_profile_name,
                "session_active": bool(session.active),
                "capture_step": session.current_step,
                "capture_completed": bool(session.capture_completed),
                "capture_remaining_ms": remaining_ms,
                "sample_counts": sample_counts,
                "derived_ready": bool(session.derived_ready),
                "quality": quality_payload,
                "current": {
                    "pinch": asdict(self.pinch_calibration),
                    "pointer": asdict(self.pointer_calibration),
                    "profiles": self.pointer_profiles,
                },
                "derived": derived_payload,
            }

    def list_calibration_profiles(self) -> dict[str, Any]:
        profiles = [
            {
                "profile_id": profile.profile_id,
                "name": profile.name,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
                "is_active": profile.profile_id == self.active_profile_id,
            }
            for profile in self.calibration_profiles.values()
        ]
        profiles.sort(key=lambda item: (not item["is_active"], item["name"].lower()))
        return {
            "active_profile_id": self.active_profile_id,
            "active_profile_name": self.active_profile_name,
            "profiles": profiles,
        }

    def create_calibration_profile(self, name: str, clone_active: bool = True) -> dict[str, Any]:
        trimmed = (name or "").strip() or "New Profile"
        now = time.time()
        source = self._get_active_profile() if clone_active else None
        if source is not None:
            settings = EngineSettings(**asdict(source.settings))
            pinch = PinchCalibration(**asdict(source.pinch))
            pointer = PointerCalibration(**asdict(source.pointer))
        else:
            settings = self._current_engine_settings()
            pinch = PinchCalibration(**asdict(self.pinch_calibration))
            pointer = PointerCalibration(**asdict(self.pointer_calibration))
        profile = CalibrationProfile(
            profile_id=f"profile-{uuid.uuid4().hex[:8]}",
            name=trimmed,
            created_at=now,
            updated_at=now,
            settings=settings,
            pinch=pinch,
            pointer=pointer,
        )
        self.calibration_profiles[profile.profile_id] = profile
        self.active_profile_id = profile.profile_id
        self.active_profile_name = profile.name
        self._apply_active_profile_to_runtime()
        self._save_calibration_profiles()
        return self.list_calibration_profiles()

    def activate_calibration_profile(self, profile_id: str) -> dict[str, Any]:
        if profile_id not in self.calibration_profiles:
            return {"ok": False, "message": "Profile not found.", "profiles": self.list_calibration_profiles()}
        self.active_profile_id = profile_id
        self.active_profile_name = self.calibration_profiles[profile_id].name
        self._apply_active_profile_to_runtime()
        self._save_calibration_profiles()
        return {"ok": True, "message": "Profile activated.", "profiles": self.list_calibration_profiles()}

    def delete_calibration_profile(self, profile_id: str) -> dict[str, Any]:
        if profile_id not in self.calibration_profiles:
            return {"ok": False, "message": "Profile not found.", "profiles": self.list_calibration_profiles()}
        if len(self.calibration_profiles) <= 1:
            return {
                "ok": False,
                "message": "At least one calibration profile must remain.",
                "profiles": self.list_calibration_profiles(),
            }
        del self.calibration_profiles[profile_id]
        if self.active_profile_id == profile_id:
            self.active_profile_id = next(iter(self.calibration_profiles))
            self.active_profile_name = self.calibration_profiles[self.active_profile_id].name
            self._apply_active_profile_to_runtime()
        self._save_calibration_profiles()
        return {"ok": True, "message": "Profile deleted.", "profiles": self.list_calibration_profiles()}

    def _start_capture_step(self, session: CalibrationSession, step: str, duration_ms: int, now: float) -> None:
        session.samples[step] = []
        session.current_step = step
        session.capture_started_at = now
        session.capture_duration_ms = int(clamp(float(duration_ms), 450.0, 9000.0))
        session.capture_completed = False
        session.derived_ready = False
        session.derived_pinch = None
        session.derived_pointer = None
        session.quality = None

    def start_calibration_session(self, safe_mode: bool = True) -> dict[str, Any]:
        with self.calibration_lock:
            previous_snapshot = {
                "settings": self._current_engine_settings(),
                "pinch": PinchCalibration(**asdict(self.pinch_calibration)),
                "pointer": PointerCalibration(**asdict(self.pointer_calibration)),
            }
            self.calibration_session = CalibrationSession(
                active=True,
                safe_mode=bool(safe_mode),
                previous_snapshot=previous_snapshot,
            )
            self.calibration_safe_mode = bool(safe_mode)
        return {"ok": True, "state": self._calibration_state_snapshot()}

    def cancel_calibration_session(self) -> dict[str, Any]:
        with self.calibration_lock:
            self.calibration_session = CalibrationSession()
            self.calibration_safe_mode = False
        return {"ok": True, "state": self._calibration_state_snapshot()}

    def start_calibration_capture(self, step: str, duration_ms: Optional[int] = None) -> dict[str, Any]:
        normalized = (step or "").strip().lower()
        if normalized not in self.calibration_sampling_defaults:
            return {"ok": False, "message": "Invalid calibration step.", "state": self._calibration_state_snapshot()}
        ok = True
        message = f"Capture started for {normalized}."
        with self.calibration_lock:
            if not self.calibration_session.active:
                ok = False
                message = "Calibration session is not active."
            else:
                sample_ms = duration_ms if duration_ms is not None else self.calibration_sampling_defaults[normalized]
                self._start_capture_step(self.calibration_session, normalized, int(sample_ms), time.monotonic())
        return {"ok": ok, "message": message, "state": self._calibration_state_snapshot()}

    def _collect_calibration_samples(
        self,
        now: float,
        image_ratio: Optional[float],
        world_ratio: Optional[float],
        pen_target: Optional[tuple[int, int]],
    ) -> None:
        with self.calibration_lock:
            session = self.calibration_session
            if not session.active or session.current_step is None or session.capture_started_at is None:
                return
            if ((now - session.capture_started_at) * 1000.0) >= session.capture_duration_ms:
                session.capture_completed = True
                session.current_step = None
                session.capture_started_at = None
                session.capture_duration_ms = 0
                return

            step = session.current_step
            if step in ("open_pinch", "closed_pinch"):
                sample: dict[str, float] = {}
                if image_ratio is not None:
                    sample["image"] = float(image_ratio)
                if world_ratio is not None:
                    sample["world"] = float(world_ratio)
                if sample:
                    session.samples[step].append(sample)
                return

            if step in ("pointer_still", "pointer_move") and pen_target is not None:
                session.samples[step].append({"x": float(pen_target[0]), "y": float(pen_target[1]), "t": now})

    def _derive_pinch_thresholds(
        self,
        open_values: list[float],
        closed_values: list[float],
        *,
        source: str,
    ) -> tuple[Optional[tuple[float, float]], float, list[str]]:
        warnings: list[str] = []
        if len(open_values) < self.calibration_min_samples["open_pinch"] or len(closed_values) < self.calibration_min_samples["closed_pinch"]:
            warnings.append(f"{source} pinch samples are low.")
            return None, 0.0, warnings

        closed_q90 = _quantile(closed_values, 0.90)
        open_q10 = _quantile(open_values, 0.10)
        separation = max(open_q10 - closed_q90, 0.0)
        min_gap = 0.035 if source == "world" else 0.05
        if separation < min_gap:
            warnings.append(f"{source} pinch separation is weak.")

        effective = max(separation, min_gap)
        close_t = closed_q90 + effective * 0.22
        open_t = open_q10 - effective * 0.20

        if source == "world":
            close_t = clamp(close_t, 0.08, 0.56)
            open_t = clamp(open_t, 0.13, 0.74)
            min_hyst = 0.035
        else:
            close_t = clamp(close_t, 0.12, 0.66)
            open_t = clamp(open_t, 0.18, 0.80)
            min_hyst = 0.05
        if open_t - close_t < min_hyst:
            midpoint = (open_t + close_t) * 0.5
            close_t = midpoint - min_hyst * 0.5
            open_t = midpoint + min_hyst * 0.5

        score = clamp((separation / max(min_gap, 1e-6)) * 0.6 + 0.4, 0.0, 1.0)
        return (close_t, open_t), separation, warnings

    def _derive_pointer_calibration(self, still_points: list[dict[str, float]], move_points: list[dict[str, float]]) -> tuple[PointerCalibration, float, list[str]]:
        warnings: list[str] = []
        if len(still_points) < self.calibration_min_samples["pointer_still"]:
            warnings.append("Pointer still samples are low.")
        if len(move_points) < self.calibration_min_samples["pointer_move"]:
            warnings.append("Pointer move samples are low.")

        jitter_distances: list[float] = []
        if still_points:
            xs = [point["x"] for point in still_points]
            ys = [point["y"] for point in still_points]
            center = (statistics.median(xs), statistics.median(ys))
            jitter_distances = [math.hypot(point["x"] - center[0], point["y"] - center[1]) for point in still_points]
        jitter95 = _quantile(jitter_distances, 0.95) if jitter_distances else 1.8

        speeds: list[float] = []
        for idx in range(1, len(move_points)):
            prev = move_points[idx - 1]
            curr = move_points[idx]
            dt = max(curr["t"] - prev["t"], 1e-6)
            dist = math.hypot(curr["x"] - prev["x"], curr["y"] - prev["y"])
            speeds.append(dist / dt)
        median_speed = statistics.median(speeds) if speeds else 240.0

        deadzone = clamp((jitter95 * 1.35) + 0.8, 0.8, 5.5)
        jitter_norm = clamp(jitter95 / 4.0, 0.0, 1.0)
        speed_norm = clamp(median_speed / 950.0, 0.0, 1.0)
        smoothness = clamp(0.90 + (jitter_norm * 0.40) - (speed_norm * 0.20), 0.70, 1.35)
        score = clamp((1.0 - jitter_norm) * 0.4 + speed_norm * 0.3 + 0.3, 0.0, 1.0)
        return PointerCalibration(deadzone_base_px=deadzone, smoothness_scalar=smoothness), score, warnings

    def derive_calibration(self) -> dict[str, Any]:
        failure_message: Optional[str] = None
        quality: Optional[CalibrationQuality] = None
        derived_pinch: Optional[PinchCalibration] = None
        derived_pointer: Optional[PointerCalibration] = None
        with self.calibration_lock:
            session = self.calibration_session
            if not session.active:
                failure_message = "Calibration session is not active."
            else:
                open_samples = session.samples["open_pinch"]
                closed_samples = session.samples["closed_pinch"]
                still_samples = session.samples["pointer_still"]
                move_samples = session.samples["pointer_move"]

                open_world = [item["world"] for item in open_samples if "world" in item]
                closed_world = [item["world"] for item in closed_samples if "world" in item]
                open_image = [item["image"] for item in open_samples if "image" in item]
                closed_image = [item["image"] for item in closed_samples if "image" in item]

                world_thresholds, world_sep, world_warnings = self._derive_pinch_thresholds(
                    open_world,
                    closed_world,
                    source="world",
                )
                image_thresholds, image_sep, image_warnings = self._derive_pinch_thresholds(
                    open_image,
                    closed_image,
                    source="image",
                )

                used_source = "world" if world_thresholds is not None else "image"
                if image_thresholds is None and world_thresholds is None:
                    failure_message = "Not enough pinch samples to derive thresholds."
                else:
                    world_close, world_open = (
                        world_thresholds if world_thresholds is not None else (self.pinch_close_world, self.pinch_open_world)
                    )
                    image_close, image_open = (
                        image_thresholds if image_thresholds is not None else (self.pinch_close_image, self.pinch_open_image)
                    )
                    derived_pinch = PinchCalibration(
                        close_world=world_close,
                        open_world=world_open,
                        close_image=image_close,
                        open_image=image_open,
                        source_preference=used_source,
                    )

                    derived_pointer, pointer_score, pointer_warnings = self._derive_pointer_calibration(
                        still_samples,
                        move_samples,
                    )
                    sample_counts = {step: len(values) for step, values in session.samples.items()}
                    count_score = clamp(
                        (
                            min(sample_counts["open_pinch"], self.calibration_min_samples["open_pinch"])
                            + min(sample_counts["closed_pinch"], self.calibration_min_samples["closed_pinch"])
                            + min(sample_counts["pointer_still"], self.calibration_min_samples["pointer_still"])
                            + min(sample_counts["pointer_move"], self.calibration_min_samples["pointer_move"])
                        )
                        / (
                            self.calibration_min_samples["open_pinch"]
                            + self.calibration_min_samples["closed_pinch"]
                            + self.calibration_min_samples["pointer_still"]
                            + self.calibration_min_samples["pointer_move"]
                        ),
                        0.0,
                        1.0,
                    )
                    sep_ref = max(world_sep, image_sep)
                    sep_goal = 0.06
                    separation_score = clamp(sep_ref / sep_goal, 0.0, 1.0)
                    quality_score = clamp(
                        (count_score * 0.45) + (separation_score * 0.35) + (pointer_score * 0.20),
                        0.0,
                        1.0,
                    )
                    warnings = world_warnings + image_warnings + pointer_warnings
                    if quality_score < self.calibration_quality_min_score:
                        warnings.append("Calibration confidence is low; consider retaking samples.")

                    quality = CalibrationQuality(
                        ready=quality_score >= self.calibration_quality_min_score,
                        score=quality_score,
                        warnings=warnings,
                        sample_counts=sample_counts,
                        separation_world=round(world_sep, 4) if world_sep > 0 else None,
                        separation_image=round(image_sep, 4) if image_sep > 0 else None,
                        used_source=used_source,
                    )
                    session.quality = quality
                    session.derived_pinch = derived_pinch
                    session.derived_pointer = derived_pointer
                    session.derived_ready = quality.ready

        if failure_message:
            return {"ok": False, "message": failure_message, "state": self._calibration_state_snapshot()}
        assert quality is not None and derived_pinch is not None and derived_pointer is not None

        return {
            "ok": True,
            "message": "Calibration derived." if quality.ready else "Calibration derived with warnings.",
            "derived": {
                "pinch": asdict(derived_pinch),
                "pointer": asdict(derived_pointer),
                "quality": asdict(quality),
                "profiles": self._build_pointer_profiles(
                    derived_pointer.deadzone_base_px,
                    derived_pointer.smoothness_scalar,
                ),
            },
            "state": self._calibration_state_snapshot(),
        }

    def apply_calibration(self, save_profile: bool = True) -> dict[str, Any]:
        failure_message: Optional[str] = None
        pinch: Optional[PinchCalibration] = None
        pointer: Optional[PointerCalibration] = None
        with self.calibration_lock:
            session = self.calibration_session
            if not session.active:
                failure_message = "Calibration session is not active."
            elif not session.derived_ready or session.derived_pinch is None or session.derived_pointer is None:
                failure_message = "No derived calibration to apply."
            else:
                if session.previous_snapshot is not None:
                    self.calibration_last_snapshot = session.previous_snapshot
                pinch = session.derived_pinch
                pointer = session.derived_pointer
                self.calibration_safe_mode = False
                self.calibration_session = CalibrationSession()

        if failure_message is not None:
            return {"ok": False, "message": failure_message, "state": self._calibration_state_snapshot()}
        assert pinch is not None and pointer is not None

        self._apply_pinch_calibration(pinch)
        self._apply_pointer_calibration(pointer)
        if save_profile:
            self._sync_active_profile_from_runtime()
        return {"ok": True, "message": "Calibration applied.", "state": self._calibration_state_snapshot()}

    def rollback_last_calibration(self) -> dict[str, Any]:
        snapshot = self.calibration_last_snapshot
        if snapshot is None:
            return {"ok": False, "message": "No calibration snapshot available to rollback.", "state": self._calibration_state_snapshot()}
        self.update_settings(
            camera=snapshot["settings"].camera,
            cooldown=snapshot["settings"].cooldown,
            hold_threshold=snapshot["settings"].hold_threshold,
            pen_sensitivity=snapshot["settings"].pen_sensitivity,
            key_output_enabled=snapshot["settings"].key_output_enabled,
            mirror_preview=snapshot["settings"].mirror_preview,
            click_speed_profile=snapshot["settings"].click_speed_profile,
            _autosync_profile=False,
        )
        self._apply_pinch_calibration(snapshot["pinch"])
        self._apply_pointer_calibration(snapshot["pointer"])
        self._sync_active_profile_from_runtime()
        self.calibration_last_snapshot = None
        return {"ok": True, "message": "Calibration rolled back.", "state": self._calibration_state_snapshot()}

    def get_calibration_session_state(self) -> dict[str, Any]:
        return {"ok": True, "state": self._calibration_state_snapshot()}

    def _allow_side_effects(self) -> bool:
        return not self.calibration_safe_mode

    def _reset_right_gesture_state(self) -> None:
        self.right_hold_start = None
        self.prev_right_gesture = None
        self.right_hold_progress = 0.0
        self.right_gesture_source = "none"
        self.right_gesture_confidence = 0.0
        self.right_pending_candidate_id = None
        self.right_pending_frames = 0

    def _extract_right_recognizer_candidate(self, gesture_result) -> Optional[tuple[str, str, float]]:
        if not gesture_result or not getattr(gesture_result, "gestures", None):
            return None
        if not gesture_result.gestures:
            return None

        mapping = {
            "Closed_Fist": ("Fist - Space", "space"),
            "Pointing_Up": ("Index - Up", "up"),
            "Victory": ("Victory - Down", "down"),
            "Open_Palm": ("Open Hand - Right", "right"),
            "Thumb_Up": ("Thumb Up - Enter", "enter"),
            "ILoveYou": ("Shaka - Escape", "escape"),
        }

        best_idx = None
        best_handedness_score = -1.0
        for i, handed_list in enumerate(getattr(gesture_result, "handedness", []) or []):
            if not handed_list:
                continue
            top = handed_list[0]
            name = (getattr(top, "category_name", "") or "").lower()
            score = float(getattr(top, "score", 0.0) or 0.0)
            if name == "right" and score > best_handedness_score:
                best_handedness_score = score
                best_idx = i
        if best_idx is None:
            best_idx = 0
        if best_idx >= len(gesture_result.gestures):
            return None

        categories = gesture_result.gestures[best_idx]
        if not categories:
            return None
        category = categories[0]
        category_name = getattr(category, "category_name", None)
        confidence = float(getattr(category, "score", 0.0) or 0.0)
        if category_name not in mapping:
            return None
        threshold = self.right_recognizer_thresholds.get(category_name, 0.7)
        if confidence < threshold:
            return None
        gesture_text, key_action = mapping[category_name]
        return gesture_text, key_action, confidence

    def _estimate_landmark_right_confidence(self, landmarks, key_action: str) -> float:
        thumb, index, middle, ring, pinky = fingers_up(landmarks)
        openness = hand_openness_ratio(landmarks)
        p_ratio = pinch_ratio(landmarks)

        if key_action == "ctrl+c":
            return clamp((0.40 - p_ratio) / 0.14, 0.0, 1.0)
        if key_action == "space":
            return clamp((0.30 - openness) / 0.22, 0.0, 1.0)
        if key_action == "right":
            return clamp((openness - 0.50) / 0.30, 0.0, 1.0)

        expected = {
            "enter": (True, False, False, False, False),
            "up": (False, True, False, False, False),
            "down": (False, True, True, False, False),
            "left": (False, True, True, True, False),
            "ctrl+v": (True, True, True, False, False),
            "escape": (True, False, False, False, True),
        }.get(key_action)
        if expected is None:
            return 0.0
        observed = (thumb, index, middle, ring, pinky)
        matched = sum(1 for o, e in zip(observed, expected) if o == e)
        return matched / 5.0

    def _extract_landmark_right_candidate(self, landmarks) -> Optional[tuple[str, str, float]]:
        if landmarks is None:
            return None
        gesture_text, key_action = classify_right_hand(landmarks)
        if not gesture_text or not key_action:
            return None
        confidence = self._estimate_landmark_right_confidence(landmarks, key_action)
        threshold = self.right_landmark_thresholds.get(key_action, 0.7)
        if confidence < threshold:
            return None
        return gesture_text, key_action, confidence

    def _resolve_right_candidate(self, landmarks, gesture_result) -> Optional[tuple[str, str, str, float]]:
        recognizer_candidate = self._extract_right_recognizer_candidate(gesture_result)
        if recognizer_candidate is not None:
            gesture_text, key_action, confidence = recognizer_candidate
            return gesture_text, key_action, "recognizer", confidence

        landmark_candidate = self._extract_landmark_right_candidate(landmarks)
        if landmark_candidate is not None:
            gesture_text, key_action, confidence = landmark_candidate
            return gesture_text, key_action, "landmark", confidence
        return None

    def _extract_mouth_blendshape_score(self, face_results) -> Optional[float]:
        blend_list = getattr(face_results, "face_blendshapes", None) if face_results is not None else None
        if not blend_list:
            return None
        if not blend_list[0]:
            return None
        scores = {}
        for cat in blend_list[0]:
            name = (getattr(cat, "category_name", "") or "").lower()
            scores[name] = float(getattr(cat, "score", 0.0) or 0.0)
        jaw_open = scores.get("jawopen")
        mouth_open = scores.get("mouthopen")
        if jaw_open is None and mouth_open is None:
            return None
        if jaw_open is not None and mouth_open is not None:
            return clamp((0.6 * jaw_open) + (0.4 * mouth_open), 0.0, 1.0)
        return clamp(jaw_open if jaw_open is not None else mouth_open, 0.0, 1.0)

    def get_state(self) -> dict:
        with self.state_lock:
            return dict(self.state)

    def get_settings(self) -> dict:
        settings = EngineSettings(
            camera=self.default_camera,
            cooldown=float(self.cooldown),
            hold_threshold=float(self.hold_threshold),
            pen_sensitivity=float(self.pen_sensitivity),
            key_output_enabled=bool(self.enabled),
            mirror_preview=bool(self.mirror_preview),
            click_speed_profile=str(self.click_speed_profile),
        )
        return {
            "camera": settings.camera,
            "cooldown": settings.cooldown,
            "hold_threshold": settings.hold_threshold,
            "pen_sensitivity": settings.pen_sensitivity,
            "key_output_enabled": settings.key_output_enabled,
            "mirror_preview": settings.mirror_preview,
            "click_speed_profile": settings.click_speed_profile,
            "mouth_enabled": bool(self.mouth_enabled),
        }

    def update_settings(
        self,
        *,
        camera=_UNSET,
        cooldown: Optional[float] = None,
        hold_threshold: Optional[float] = None,
        pen_sensitivity: Optional[float] = None,
        key_output_enabled: Optional[bool] = None,
        mirror_preview: Optional[bool] = None,
        click_speed_profile: Optional[str] = None,
        mouth_enabled: Optional[bool] = None,
        _autosync_profile: bool = True,
    ) -> dict:
        if camera is not _UNSET:
            self.default_camera = None if camera is None else int(camera)
        if cooldown is not None:
            self.cooldown = clamp(float(cooldown), 0.2, 2.0)
        if hold_threshold is not None:
            self.hold_threshold = clamp(float(hold_threshold), 0.1, 1.5)
        if pen_sensitivity is not None:
            self.pen_sensitivity = clamp(float(pen_sensitivity), 0.0, 100.0)
            self.red_pen.set_sensitivity(self.pen_sensitivity)
        if key_output_enabled is not None:
            self.enabled = bool(key_output_enabled)
        if mirror_preview is not None:
            self.mirror_preview = bool(mirror_preview)
        if click_speed_profile is not None:
            self._apply_click_speed_profile(str(click_speed_profile))
        if mouth_enabled is not None:
            self.mouth_enabled = bool(mouth_enabled)
        if _autosync_profile:
            self._sync_active_profile_from_runtime()
        return self.get_settings()

    def start(self) -> dict:
        if self.running:
            return {"ok": True, "message": "Session already running.", "camera_index": self.active_camera_index}
        if self.landmarker is None:
            with self.state_lock:
                self.state["session_status"] = "error"
                self.state["status"] = "Model failed to load."
            return {"ok": False, "message": "Model failed to load.", "camera_index": None}

        # Ensure any previous capture thread is fully down before reopening camera.
        if (self.capture_thread and self.capture_thread.is_alive()) or (
            self.processing_thread and self.processing_thread.is_alive()
        ):
            self.stop_event.set()
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.5)
            if self.processing_thread and self.processing_thread.is_alive():
                self.latest_frame_ready.set()
                self.processing_thread.join(timeout=2.5)
            if (self.capture_thread and self.capture_thread.is_alive()) or (
                self.processing_thread and self.processing_thread.is_alive()
            ):
                message = "Previous camera session is still shutting down. Please try start again."
                self._record_warning(message)
                with self.state_lock:
                    self.state["session_status"] = "error"
                    self.state["status"] = message
                    self.state["warnings"] = list(self.warning_history)
                return {"ok": False, "message": message, "camera_index": None}
        self.capture_thread = None
        self.processing_thread = None

        cap, idx = self._open_camera(self.default_camera)
        if cap is None:
            message = "Camera error: unable to read frames from any camera."
            self._record_error(message)
            with self.state_lock:
                self.state["session_status"] = "error"
                self.state["status"] = message
                self.state["errors"] = list(self.error_history)
            return {"ok": False, "message": message, "camera_index": None}

        self.cap = cap
        self.active_camera_index = idx
        self.stop_event.clear()
        self.running = True
        self.capture_fps = 0.0
        self.inference_fps = 0.0
        self.dropped_frames = 0
        self.end_to_end_latency_ms = 0.0
        self._capture_window_start = time.monotonic()
        self._capture_window_count = 0
        self._inference_window_start = time.monotonic()
        self._inference_window_count = 0
        self.capture_sequence = 0
        with self.latest_frame_lock:
            self.latest_captured_frame = None
            self.latest_frame_ready.clear()
        with self.result_lock:
            self.latest_result = None
        with self.face_result_lock:
            self.latest_face_result = None
        with self.gesture_result_lock:
            self.latest_gesture_result = None

        self.last_trigger_time = 0.0
        self._reset_right_gesture_state()
        self._stop_left_controls()
        self._reset_pointer_filter()
        self.pointer_mode = "idle"
        self.red_pen.reset()
        self.pen_state = "no-right-hand"
        self.pen_target = None
        self.pen_frame_point = None
        self.pen_roi = None

        self.mouth_open_start = None
        self.mouth_triggered = False
        self.mouth_open_active = False
        self.mouth_blend_score = None
        self.mouth_ratio_norm = None
        self.mouth_signal_ema = None
        self.mouth_signal_mode = "ratio_fallback"
        self.pending_status_message = None

        with self.state_lock:
            self.state["session_status"] = "running"
            self.state["running"] = True
            self.state["camera_index"] = idx
            self.state["status"] = f"Camera running on index {idx}. Show your hands and face."
            self.state["metrics"] = {
                "capture_fps": 0.0,
                "inference_fps": 0.0,
                "dropped_frames": 0,
                "end_to_end_latency_ms": 0.0,
            }
            self.state["interaction"] = {
                "pinch_state": self.pinch_state,
                "pinch_ratio": self.pinch_last_ratio,
                "pinch_source": self.pinch_ratio_source,
                "pointer_mode": self.pointer_mode,
                "right_gesture_source": self.right_gesture_source,
                "right_gesture_confidence": self.right_gesture_confidence,
                "mouth_blend_score": self.mouth_blend_score,
                "mouth_signal_mode": self.mouth_signal_mode,
            }
            self.state["calibration"] = self._calibration_state_snapshot()
            self.state["updatedAt"] = time.time()

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="gesture-capture")
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="gesture-processing"
        )
        self.capture_thread.start()
        self.processing_thread.start()
        return {"ok": True, "message": "Session started.", "camera_index": idx}

    def stop(self) -> dict:
        if not self.running:
            self._reset_pointer_filter()
            self.pointer_mode = "idle"
            self._reset_right_gesture_state()
            self.mouth_blend_score = None
            self.mouth_ratio_norm = None
            self.mouth_signal_ema = None
            self.mouth_signal_mode = "ratio_fallback"
            self.mouth_open_active = False
            with self.state_lock:
                self.state["session_status"] = "idle"
                self.state["running"] = False
                self.state["interaction"] = {
                    "pinch_state": self.pinch_state,
                    "pinch_ratio": self.pinch_last_ratio,
                    "pinch_source": self.pinch_ratio_source,
                    "pointer_mode": self.pointer_mode,
                    "right_gesture_source": self.right_gesture_source,
                    "right_gesture_confidence": self.right_gesture_confidence,
                    "mouth_blend_score": self.mouth_blend_score,
                    "mouth_signal_mode": self.mouth_signal_mode,
                }
                self.state["calibration"] = self._calibration_state_snapshot()
            return {"ok": True, "message": "Session already stopped.", "camera_index": None}

        self.running = False
        self.stop_event.set()
        self.latest_frame_ready.set()
        # Release camera first to unblock any pending read() call.
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.5)
            if self.capture_thread.is_alive():
                self._record_warning("Camera thread took longer than expected to stop.")
        self.capture_thread = None
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.5)
            if self.processing_thread.is_alive():
                self._record_warning("Processing thread took longer than expected to stop.")
        self.processing_thread = None

        self._stop_left_controls()
        self._reset_pointer_filter()
        self.pointer_mode = "idle"
        self._reset_right_gesture_state()
        self.red_pen.reset()
        self.pen_state = "no-right-hand"
        self.pen_target = None
        self.pen_frame_point = None
        self.pen_roi = None
        self.active_camera_index = None
        with self.frame_lock:
            self.last_frame_jpeg = None
        with self.latest_frame_lock:
            self.latest_captured_frame = None
            self.latest_frame_ready.clear()
        with self.gesture_result_lock:
            self.latest_gesture_result = None

        self.mouth_blend_score = None
        self.mouth_ratio_norm = None
        self.mouth_signal_ema = None
        self.mouth_signal_mode = "ratio_fallback"
        self.mouth_open_active = False

        with self.state_lock:
            self.state["session_status"] = "idle"
            self.state["running"] = False
            self.state["camera_index"] = None
            self.state["gestureText"] = "R: -\nL: -\nM: -\nP: -"
            self.state["rightText"] = "idle"
            self.state["leftText"] = "idle"
            self.state["mouthText"] = "closed" if self.face_landmarker and self.mouth_enabled else "disabled"
            self.state["penState"] = "no-right-hand"
            self.state["status"] = "Stopped."
            self.state["holdProgress"] = 0.0
            self.state["metrics"] = {
                "capture_fps": self.capture_fps,
                "inference_fps": self.inference_fps,
                "dropped_frames": self.dropped_frames,
                "end_to_end_latency_ms": self.end_to_end_latency_ms,
            }
            self.state["interaction"] = {
                "pinch_state": self.pinch_state,
                "pinch_ratio": self.pinch_last_ratio,
                "pinch_source": self.pinch_ratio_source,
                "pointer_mode": self.pointer_mode,
                "right_gesture_source": self.right_gesture_source,
                "right_gesture_confidence": self.right_gesture_confidence,
                "mouth_blend_score": self.mouth_blend_score,
                "mouth_signal_mode": self.mouth_signal_mode,
            }
            self.state["calibration"] = self._calibration_state_snapshot()
            self.state["updatedAt"] = time.time()

        return {"ok": True, "message": "Session stopped.", "camera_index": None}

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.last_frame_jpeg

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms
        return now_ms

    def _update_capture_metrics(self, now: float) -> None:
        self._capture_window_count += 1
        elapsed = now - self._capture_window_start
        if elapsed >= 1.0:
            self.capture_fps = self._capture_window_count / elapsed
            self._capture_window_count = 0
            self._capture_window_start = now

    def _update_inference_metrics(self, now: float, captured_at: float) -> None:
        self._inference_window_count += 1
        elapsed = now - self._inference_window_start
        if elapsed >= 1.0:
            self.inference_fps = self._inference_window_count / elapsed
            self._inference_window_count = 0
            self._inference_window_start = now
        latency_ms = max((now - captured_at) * 1000.0, 0.0)
        if self.end_to_end_latency_ms <= 0:
            self.end_to_end_latency_ms = latency_ms
        else:
            self.end_to_end_latency_ms = (self.end_to_end_latency_ms * 0.85) + (latency_ms * 0.15)

    def _open_camera(self, preferred: Optional[int]) -> tuple[Optional[object], Optional[int]]:
        # Auto mode: prefer laptop/default camera first (index 0), then fallbacks.
        indexes = [preferred] if preferred is not None else [0, 1, 2, 3, 4]
        # CAP_DSHOW is generally the most reliable for quick stop/start cycles on Windows.
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        backend_names = {
            cv2.CAP_ANY: "CAP_ANY",
            cv2.CAP_MSMF: "CAP_MSMF",
            cv2.CAP_DSHOW: "CAP_DSHOW",
        }
        for idx in indexes:
            for backend in backends:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                if backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(30):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        self.pending_status_message = (
                            f"Camera running on index {idx} ({backend_names.get(backend, backend)}). "
                            "Show your hands and face."
                        )
                        return cap, idx
                    time.sleep(0.025)
                cap.release()
            time.sleep(0.12)
        return None, None

    def _extract_hands(
        self, result
    ) -> tuple[Optional[object], Optional[object], Optional[object], Optional[object]]:
        right, left = None, None
        right_world, left_world = None, None
        unknown: list[tuple[object, Optional[object]]] = []
        if not result or not result.hand_landmarks:
            return None, None, None, None

        world_list = getattr(result, "hand_world_landmarks", None)
        for index, landmarks in enumerate(result.hand_landmarks):
            label = None
            world_landmarks = None
            if world_list and index < len(world_list):
                world_landmarks = world_list[index]
            if result.handedness and index < len(result.handedness) and result.handedness[index]:
                label = getattr(result.handedness[index][0], "category_name", None)
                if label:
                    label = label.lower()
            if label == "right" and right is None:
                right = landmarks
                right_world = world_landmarks
            elif label == "left" and left is None:
                left = landmarks
                left_world = world_landmarks
            else:
                unknown.append((landmarks, world_landmarks))
        for landmarks, world_landmarks in unknown:
            if right is None:
                right = landmarks
                right_world = world_landmarks
            elif left is None:
                left = landmarks
                left_world = world_landmarks
        return right, left, right_world, left_world

    @staticmethod
    def _distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _distance_xyz(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    @staticmethod
    def _distance_landmark_xy(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _compute_pinch_ratio(
        self, landmarks, world_landmarks
    ) -> tuple[Optional[float], str, Optional[float], Optional[float]]:
        if landmarks is None:
            return None, "none", None, None

        image_ratio = pinch_ratio(landmarks)
        world_ratio = None
        if world_landmarks and len(world_landmarks) > 17:
            palm_span = self._distance_xyz(world_landmarks[5], world_landmarks[17])
            if palm_span > 1e-6:
                world_ratio = self._distance_xyz(world_landmarks[4], world_landmarks[8]) / palm_span

        if world_ratio is not None:
            source = "world+image" if image_ratio is not None else "world"
            return world_ratio, source, image_ratio, world_ratio
        return image_ratio, "image", image_ratio, None

    def _compute_middle_pinch_ratio(
        self, landmarks, world_landmarks
    ) -> tuple[Optional[float], str, Optional[float], Optional[float]]:
        if landmarks is None:
            return None, "none", None, None

        image_ratio = self._distance_landmark_xy(landmarks[4], landmarks[12]) / max(
            self._distance_landmark_xy(landmarks[5], landmarks[17]), 1e-6
        )
        world_ratio = None
        if world_landmarks and len(world_landmarks) > 17:
            palm_span = self._distance_xyz(world_landmarks[5], world_landmarks[17])
            if palm_span > 1e-6:
                world_ratio = self._distance_xyz(world_landmarks[4], world_landmarks[12]) / palm_span

        if world_ratio is not None:
            source = "world+image" if image_ratio is not None else "world"
            return world_ratio, source, image_ratio, world_ratio
        return image_ratio, "image", image_ratio, None

    def _set_pinch_state(self, state: str, now: float) -> None:
        self.pinch_state = state
        self.pinch_state_since = now
        if state == "open":
            if self.left_mode.startswith("pinch"):
                self.left_mode = "idle"
        elif state == "arming":
            self.left_mode = "pinch_arm"
        elif state == "pinched":
            self.left_mode = "pinch_ready" if self.pinch_mode_pen else "pinch_right"
        elif state == "dragging":
            self.left_mode = "pinch_drag"
        elif state == "release":
            self.left_mode = "pinch_release"
        elif state == "cooldown":
            self.left_mode = "pinch_cooldown"

    def _reset_pointer_filter(self) -> None:
        self.pointer_filter.reset()
        self.pointer_last_raw = None
        self.pointer_last_filtered = None

    def _reset_pinch_interaction(self, now: float, *, reset_cursor: bool) -> None:
        self._release_drag()
        self.left_pinch_start = None
        self.left_pinch_mode = None
        self.left_pinch_target = None
        self.left_right_click_fired = False
        self.pinch_mode_pen = False
        self.pinch_intent = None
        self.pinch_started_at = None
        self.pinch_arming_started_at = None
        self.pinch_close_frames = 0
        self.pinch_anchor_point = None
        self.pinch_last_point = None
        self.pinch_max_move_px = 0.0
        self.pinch_release_handled = False
        self.pinch_ratio_ema = None
        self.middle_pinch_ratio_ema = None
        self.pinch_last_ratio = None
        self.pinch_ratio_source = "none"
        self.latest_left_image_ratio = None
        self.latest_left_world_ratio = None
        self.pinch_cooldown_started_at = None
        self._set_pinch_state("open", now)
        if reset_cursor:
            self._reset_pointer_filter()

    def _smooth_pointer(self, point: tuple[int, int], mode: str, now: float) -> tuple[int, int]:
        profile = self.pointer_profiles.get(mode, self.pointer_profiles["move"])
        raw = (float(point[0]), float(point[1]))

        if self.pointer_last_raw is not None and self.pointer_last_filtered is not None:
            if self._distance_xy(raw, self.pointer_last_raw) < profile["deadzone_px"]:
                return (int(round(self.pointer_last_filtered[0])), int(round(self.pointer_last_filtered[1])))

        filtered = self.pointer_filter.filter(raw, now, profile)
        filtered = (
            clamp(filtered[0], 0.0, max(self.screen_width - 1, 0)),
            clamp(filtered[1], 0.0, max(self.screen_height - 1, 0)),
        )
        self.pointer_last_raw = raw
        self.pointer_last_filtered = filtered
        return (int(round(filtered[0])), int(round(filtered[1])))

    def _move_cursor(self, point: tuple[int, int], mode: str = "move", now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        target = self._smooth_pointer(point, mode, now)
        if not self._allow_side_effects():
            self.pointer_mode = mode
            return True
        try:
            pyautogui.moveTo(target[0], target[1], duration=0)
            self.pointer_mode = mode
            return True
        except Exception as exc:
            self.pending_status_message = f"Cursor move error: {exc}"
            self._record_error(self.pending_status_message)
            return False

    def _send_key(self, key_action: str) -> None:
        if not self.enabled or not self._allow_side_effects():
            return
        try:
            if "+" in key_action:
                pyautogui.hotkey(*key_action.split("+"))
            else:
                pyautogui.press(key_action)
            self.key_count += 1
            self.last_key_text = key_action.upper()
        except Exception as exc:
            self.pending_status_message = f"Key output error: {exc}"
            self._record_error(self.pending_status_message)

    def _release_drag(self) -> None:
        if self.left_dragging:
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
        self.left_dragging = False

    def _stop_left_controls(self) -> None:
        now = time.monotonic()
        self._reset_pinch_interaction(now, reset_cursor=False)
        self.left_mode = "idle"
        self.left_anchor_y = None
        self.left_base_value = None
        self.left_last_value = None
        self.brightness_arm_start = None
        self.brightness_exit_gesture_start = None
        self._brightness_pending_target = None

    def _get_brightness_cached(self, now: float) -> Optional[int]:
        if (
            self._brightness_cached_percent is not None
            and now - self._brightness_cache_time < self._brightness_read_interval
        ):
            return self._brightness_cached_percent
        current = self.brightness.get_percent()
        if current is not None:
            self._brightness_cached_percent = current
            self._brightness_cache_time = now
        return current

    def _apply_brightness_target(self, target: int, now: float) -> bool:
        target = int(clamp(target, 0, 100))
        if now - self._brightness_last_write_time < self._brightness_write_interval:
            self._brightness_pending_target = target
            return False
        if not self._allow_side_effects():
            self._brightness_cached_percent = target
            self._brightness_cache_time = now
            self._brightness_pending_target = None
            return True
        if not self.brightness.set_percent(target):
            return False
        self._brightness_last_write_time = now
        self._brightness_cached_percent = target
        self._brightness_cache_time = now
        self._brightness_pending_target = None
        return True

    def _handle_right(self, landmarks, world_landmarks, gesture_result, now: float) -> str:
        if landmarks is None:
            self._reset_right_gesture_state()
            return "idle"

        candidate = self._resolve_right_candidate(landmarks, gesture_result)
        if candidate is None:
            self._reset_right_gesture_state()
            return "none"

        gesture, key_action, source, confidence = candidate
        candidate_id = f"{source}:{gesture}:{key_action}"
        if candidate_id == self.right_pending_candidate_id:
            self.right_pending_frames += 1
        else:
            self.right_pending_candidate_id = candidate_id
            self.right_pending_frames = 1

        self.right_gesture_source = source
        self.right_gesture_confidence = confidence

        if self.right_pending_frames < self.right_confirm_frames_required:
            self.right_hold_start = None
            self.prev_right_gesture = None
            self.right_hold_progress = 0.0
            return f"confirm {gesture}"

        if gesture == self.prev_right_gesture:
            if self.right_hold_start is None:
                self.right_hold_start = now
                self.right_hold_progress = 0.0
            else:
                elapsed = now - self.right_hold_start
                self.right_hold_progress = clamp(elapsed / self.hold_threshold, 0.0, 1.0)
                if elapsed >= self.hold_threshold and now - self.last_trigger_time >= self.cooldown:
                    self._send_key(key_action)
                    self.last_trigger_time = now
                    self.right_hold_start = None
                    self.right_hold_progress = 0.0
        else:
            self.right_hold_start = None
            self.right_hold_progress = 0.0
        self.prev_right_gesture = gesture
        return gesture

    def _handle_slide(self, mode: str, y_now: float, getter, setter) -> str:
        current = getter()
        if current is None:
            return f"{mode} unavailable"
        if self.left_mode != mode:
            self._release_drag()
            self.left_mode = mode
            self.left_anchor_y = y_now
            self.left_base_value = current
            self.left_last_value = current
            self.left_pinch_start = None
            self.brightness_arm_start = None
            self.brightness_exit_gesture_start = None
        if self.left_anchor_y is None or self.left_base_value is None:
            return f"{mode} {current}%"
        delta = (self.left_anchor_y - y_now) * 180
        target = int(round(clamp(self.left_base_value + delta, 0, 100)))
        step = 2 if mode == "brightness" else 1
        if self.left_last_value is None or abs(target - self.left_last_value) >= step:
            if not self._allow_side_effects():
                self.left_last_value = target
            elif setter(target):
                self.left_last_value = target
        shown = self.left_last_value if self.left_last_value is not None else current
        return f"{mode} {shown}%"

    def _handle_fist_brightness(self, landmarks, now: float) -> str:
        current = self._get_brightness_cached(now)
        if current is None:
            if not self.warned_brightness_unavailable:
                self.pending_status_message = "Brightness control unavailable on this monitor setup."
                self.warned_brightness_unavailable = True
                self._record_warning(self.pending_status_message)
            return "brightness unavailable"
        if self.left_mode != "brightness_fist":
            self._release_drag()
            self.left_mode = "brightness_fist"
            self.left_last_value = current
            self.left_pinch_start = None
            self._brightness_pending_target = None
        self.brightness_arm_start = None
        openness = hand_openness_ratio(landmarks)
        target = int(round(1 + openness * 99))
        if self.left_last_value is None or abs(target - self.left_last_value) >= 1:
            self._brightness_pending_target = target
        if self._brightness_pending_target is not None:
            if self._apply_brightness_target(self._brightness_pending_target, now):
                self.left_last_value = self._brightness_cached_percent
        shown = self.left_last_value if self.left_last_value is not None else target
        return f"brightness {shown}%"

    def _finalize_pinch_release(self, now: float) -> None:
        if self.pinch_release_handled:
            return

        duration = max(now - self.pinch_started_at, 0.0) if self.pinch_started_at is not None else 0.0
        intent = self.pinch_intent or ("left_click_drag" if self.pinch_mode_pen else "right_click")
        try:
            if intent == "left_click_drag":
                if self.left_dragging:
                    self._release_drag()
                elif duration < self.pinch_click_max_seconds and self.pinch_max_move_px < self.pinch_drag_move_threshold_px:
                    if self.pinch_last_point is not None:
                        self._move_cursor(self.pinch_last_point, mode="precision", now=now)
                    if self._allow_side_effects():
                        pyautogui.click(button="left")
            elif intent == "right_click" and duration >= self.pinch_right_click_confirm_seconds and self._allow_side_effects():
                pyautogui.click(button="right")
        except Exception as exc:
            side = "left" if intent == "left_click_drag" else "right"
            self.pending_status_message = f"{side.title()} click error: {exc}"
            self._record_error(self.pending_status_message)
        self.pinch_release_handled = True

    def _handle_left(
        self,
        landmarks,
        world_landmarks,
        pen_target: Optional[tuple[int, int]],
        pen_active: bool,
        now: float,
    ) -> tuple[str, str]:
        pointer_mode = "move" if pen_active else "idle"

        if landmarks is None:
            self.latest_left_image_ratio = None
            self.latest_left_world_ratio = None
            was_active_pinch = self.pinch_state != "open" or self.left_dragging
            self._stop_left_controls()
            if was_active_pinch:
                self._reset_pointer_filter()
            return "idle", pointer_mode

        thumb, index, middle, ring, pinky = fingers_up(landmarks)
        count = sum((thumb, index, middle, ring, pinky))
        fist = count == 0
        pinky_only = (not thumb) and (not index) and (not middle) and (not ring) and pinky

        if pinky_only:
            self._reset_pinch_interaction(now, reset_cursor=False)
            if self.left_mode == "brightness_fist":
                self.left_mode = "idle"
            self.brightness_arm_start = None
            self.brightness_exit_gesture_start = None
            text = self._handle_slide("volume", landmarks[0].y, self.volume.get_percent, self.volume.set_percent)
            if text == "volume unavailable" and not self.warned_volume_unavailable:
                detail = f" ({self.volume.error})" if self.volume.error else ""
                self.pending_status_message = f"Volume control unavailable on this device/environment{detail}."
                self.warned_volume_unavailable = True
                self._record_warning(self.pending_status_message)
            return text, pointer_mode

        if self.left_mode == "brightness_fist":
            self._reset_pinch_interaction(now, reset_cursor=False)
            exit_gesture = thumb and (not index) and (not middle) and (not ring) and pinky
            if exit_gesture:
                if self.brightness_exit_gesture_start is None:
                    self.brightness_exit_gesture_start = now
                elapsed = now - self.brightness_exit_gesture_start
                if elapsed >= self.brightness_exit_hold:
                    self.left_mode = "idle"
                    self.brightness_exit_gesture_start = None
                    self.brightness_arm_start = None
                    self.left_last_value = None
                    return "brightness released", pointer_mode
                return f"brightness exit arm {elapsed:.2f}/{self.brightness_exit_hold:.2f}s", pointer_mode
            self.brightness_exit_gesture_start = None
            return self._handle_fist_brightness(landmarks, now), pointer_mode

        ratio, source, image_ratio, world_ratio = self._compute_pinch_ratio(landmarks, world_landmarks)
        middle_ratio, _, middle_image_ratio, middle_world_ratio = self._compute_middle_pinch_ratio(
            landmarks, world_landmarks
        )
        self.latest_left_image_ratio = image_ratio
        self.latest_left_world_ratio = world_ratio
        if ratio is None or middle_ratio is None:
            self._reset_pinch_interaction(now, reset_cursor=False)
            self._stop_left_controls()
            return "idle", pointer_mode

        if self.pinch_ratio_ema is None:
            self.pinch_ratio_ema = ratio
        else:
            alpha = self.pinch_ratio_ema_alpha
            self.pinch_ratio_ema = alpha * ratio + (1.0 - alpha) * self.pinch_ratio_ema
        if self.middle_pinch_ratio_ema is None:
            self.middle_pinch_ratio_ema = middle_ratio
        else:
            alpha = self.pinch_ratio_ema_alpha
            self.middle_pinch_ratio_ema = alpha * middle_ratio + (1.0 - alpha) * self.middle_pinch_ratio_ema
        smoothed_ratio = self.pinch_ratio_ema
        smoothed_middle_ratio = self.middle_pinch_ratio_ema
        self.pinch_last_ratio = smoothed_ratio
        self.pinch_ratio_source = source

        def _is_close(smoothed: float, image_val: Optional[float], world_val: Optional[float]) -> bool:
            close_threshold = self.pinch_close_world if world_val is not None else self.pinch_close_image
            closed = smoothed <= close_threshold
            if image_val is not None and image_val <= self.pinch_close_image:
                closed = True
            if world_val is not None and world_val <= self.pinch_close_world:
                closed = True
            return closed

        def _is_open(image_val: Optional[float], world_val: Optional[float]) -> bool:
            if world_val is not None and image_val is not None:
                return world_val >= self.pinch_open_world and image_val >= self.pinch_open_image
            if world_val is not None:
                return world_val >= self.pinch_open_world
            return image_val is not None and image_val >= self.pinch_open_image

        index_is_close = _is_close(smoothed_ratio, image_ratio, world_ratio)
        index_is_open = _is_open(image_ratio, world_ratio)
        middle_is_close = _is_close(smoothed_middle_ratio, middle_image_ratio, middle_world_ratio)
        middle_is_open = _is_open(middle_image_ratio, middle_world_ratio)
        right_click_close = middle_is_close and not index_is_close
        right_click_open = middle_is_open or index_is_close

        if fist and not index_is_close and not middle_is_close:
            self._reset_pinch_interaction(now, reset_cursor=False)
            if self.brightness_arm_start is None:
                self.brightness_arm_start = now
            elapsed = now - self.brightness_arm_start
            if elapsed >= self.brightness_arm_hold:
                return self._handle_fist_brightness(landmarks, now), pointer_mode
            return f"brightness arm {elapsed:.2f}/{self.brightness_arm_hold:.2f}s", pointer_mode

        self.brightness_arm_start = None
        self.brightness_exit_gesture_start = None

        if self.pinch_state == "cooldown":
            if self.pinch_cooldown_started_at is None:
                self.pinch_cooldown_started_at = now
            if now - self.pinch_cooldown_started_at >= self.pinch_cooldown_seconds:
                self._set_pinch_state("open", now)
                self.pinch_cooldown_started_at = None
            else:
                remaining = max(self.pinch_cooldown_seconds - (now - self.pinch_cooldown_started_at), 0.0)
                return f"cooldown {remaining:.2f}s", pointer_mode

        if self.pinch_state == "release":
            self._finalize_pinch_release(now)
            if now - self.pinch_state_since >= self.pinch_release_linger_seconds:
                self._set_pinch_state("cooldown", now)
                self.pinch_cooldown_started_at = now
            release_mode = (
                "precision" if self.pinch_intent == "left_click_drag" and pen_active else pointer_mode
            )
            return "release", release_mode

        if self.pinch_state == "dragging":
            if self.pinch_intent != "left_click_drag":
                self._set_pinch_state("release", now)
                self.pinch_release_handled = False
                return "release", pointer_mode
            if pen_target is not None and self.pinch_anchor_point is not None:
                self.pinch_last_point = pen_target
                self.pinch_max_move_px = max(
                    self.pinch_max_move_px,
                    self._distance_xy((float(pen_target[0]), float(pen_target[1])), self.pinch_anchor_point),
                )
            if not pen_active or index_is_open:
                self._set_pinch_state("release", now)
                self.pinch_release_handled = False
                return "release", pointer_mode
            return "drag", "drag" if pen_active else pointer_mode

        if self.pinch_state == "pinched":
            if self.pinch_started_at is None:
                self.pinch_started_at = now
                self.left_pinch_start = now
            elapsed = now - self.pinch_started_at
            if self.pinch_intent == "left_click_drag" and pen_target is not None:
                self.pinch_last_point = pen_target
                if self.pinch_anchor_point is None:
                    self.pinch_anchor_point = (float(pen_target[0]), float(pen_target[1]))
                self.pinch_max_move_px = max(
                    self.pinch_max_move_px,
                    self._distance_xy((float(pen_target[0]), float(pen_target[1])), self.pinch_anchor_point),
                )
                should_drag = (
                    elapsed >= self.pinch_drag_hold_seconds or self.pinch_max_move_px >= self.pinch_drag_move_threshold_px
                )
                if should_drag and not self.left_dragging:
                    if self._allow_side_effects():
                        try:
                            pyautogui.mouseDown(button="left")
                            self.left_dragging = True
                            self._set_pinch_state("dragging", now)
                        except Exception as exc:
                            self.pending_status_message = f"Drag start error: {exc}"
                            self._record_error(self.pending_status_message)
                    else:
                        self._set_pinch_state("dragging", now)
            should_release = False
            if self.pinch_intent == "left_click_drag":
                should_release = index_is_open or not pen_active
            elif self.pinch_intent == "right_click":
                should_release = right_click_open or not pen_active
            if should_release:
                self._set_pinch_state("release", now)
                self.pinch_release_handled = False
                return "release", pointer_mode
            mode = "precision" if self.pinch_intent == "left_click_drag" and pen_active else pointer_mode
            label = "left pinch" if self.pinch_intent == "left_click_drag" else "right pinch"
            return f"{label} {elapsed:.2f}s", mode

        if self.pinch_state == "arming":
            if not pen_active:
                self._set_pinch_state("open", now)
                self.pinch_arming_started_at = None
                self.pinch_close_frames = 0
                self.pinch_intent = None
                return "idle", pointer_mode
            if self.pinch_intent is None:
                self.pinch_intent = "right_click" if right_click_close else "left_click_drag"
            active_close = right_click_close if self.pinch_intent == "right_click" else index_is_close
            active_open = right_click_open if self.pinch_intent == "right_click" else index_is_open
            if active_close:
                if self.pinch_arming_started_at is None:
                    self.pinch_arming_started_at = now
                    self.pinch_close_frames = 1
                else:
                    self.pinch_close_frames += 1
                arm_elapsed = now - self.pinch_arming_started_at
                if arm_elapsed >= self.pinch_arming_min_seconds and self.pinch_close_frames >= self.pinch_arming_min_frames:
                    self.pinch_mode_pen = self.pinch_intent == "left_click_drag"
                    self.pinch_started_at = now
                    self.left_pinch_start = now
                    self.left_pinch_mode = "left" if self.pinch_mode_pen else "right"
                    self.pinch_anchor_point = (
                        (float(pen_target[0]), float(pen_target[1])) if pen_target is not None else None
                    )
                    self.pinch_last_point = pen_target
                    self.left_pinch_target = pen_target
                    self.pinch_max_move_px = 0.0
                    self.pinch_release_handled = False
                    self._set_pinch_state("pinched", now)
                    return ("left pinch 0.00s" if self.pinch_mode_pen else "right pinch 0.00s"), (
                        "precision" if self.pinch_mode_pen and pen_active else pointer_mode
                    )
                return f"pinch arm {arm_elapsed:.2f}/{self.pinch_arming_min_seconds:.2f}s", (
                    "precision" if pen_active else pointer_mode
                )
            alternate_intent = "left_click_drag" if self.pinch_intent == "right_click" else "right_click"
            alternate_close = index_is_close if alternate_intent == "left_click_drag" else right_click_close
            if alternate_close:
                self.pinch_intent = alternate_intent
                self.pinch_arming_started_at = now
                self.pinch_close_frames = 1
                return f"pinch arm 0.00/{self.pinch_arming_min_seconds:.2f}s", (
                    "precision" if pen_active else pointer_mode
                )
            if active_open:
                self._set_pinch_state("open", now)
                self.pinch_arming_started_at = None
                self.pinch_close_frames = 0
                self.pinch_intent = None
                return "idle", pointer_mode
            arm_elapsed = 0.0 if self.pinch_arming_started_at is None else now - self.pinch_arming_started_at
            return f"pinch arm {arm_elapsed:.2f}/{self.pinch_arming_min_seconds:.2f}s", (
                "precision" if pen_active else pointer_mode
            )

        start_intent: Optional[str] = None
        if pen_active and index_is_close:
            start_intent = "left_click_drag"
        elif pen_active and right_click_close:
            start_intent = "right_click"

        if start_intent is not None:
            self.pinch_arming_started_at = now
            self.pinch_close_frames = 1
            self.pinch_intent = start_intent
            self._set_pinch_state("arming", now)
            return f"pinch arm 0.00/{self.pinch_arming_min_seconds:.2f}s", (
                "precision" if pen_active else pointer_mode
            )

        return "idle", pointer_mode

    def _handle_mouth(self, face_results, now: float) -> str:
        if not self.mouth_enabled:
            self.mouth_open_start = None
            self.mouth_triggered = False
            self.mouth_open_active = False
            self.mouth_blend_score = None
            self.mouth_ratio_norm = None
            self.mouth_signal_ema = None
            self.mouth_signal_mode = "ratio_fallback"
            return "disabled"

        if self.face_landmarker is None:
            self.mouth_signal_mode = "ratio_fallback"
            self.mouth_blend_score = None
            return "disabled"

        if not face_results or not face_results.face_landmarks:
            self.mouth_open_start = None
            self.mouth_triggered = False
            self.mouth_open_active = False
            self.mouth_blend_score = None
            self.mouth_ratio_norm = None
            self.mouth_signal_ema = None
            self.mouth_signal_mode = "ratio_fallback"
            return "closed"

        ratio_raw = mouth_open_ratio(face_results.face_landmarks[0])
        ratio_norm = clamp(
            (ratio_raw - self.mouth_ratio_min) / max(self.mouth_ratio_max - self.mouth_ratio_min, 1e-6),
            0.0,
            1.0,
        )
        blend_score = self._extract_mouth_blendshape_score(face_results)

        self.mouth_ratio_norm = ratio_norm
        self.mouth_blend_score = blend_score
        if blend_score is None:
            blended_signal = ratio_norm
            self.mouth_signal_mode = "ratio_fallback"
        else:
            blended_signal = clamp((0.55 * ratio_norm) + (0.45 * blend_score), 0.0, 1.0)
            self.mouth_signal_mode = "hybrid"

        if self.mouth_signal_ema is None:
            self.mouth_signal_ema = blended_signal
        else:
            alpha = self.mouth_ratio_ema_alpha
            self.mouth_signal_ema = alpha * blended_signal + (1.0 - alpha) * self.mouth_signal_ema

        if self.mouth_open_active:
            open_now = self.mouth_signal_ema >= self.mouth_hybrid_close_threshold
        else:
            open_now = self.mouth_signal_ema >= self.mouth_hybrid_open_threshold
        self.mouth_open_active = open_now

        if open_now:
            if self.mouth_open_start is None:
                self.mouth_open_start = now
                self.mouth_triggered = False
            elapsed = now - self.mouth_open_start
            if elapsed >= self.mouth_hold_seconds and not self.mouth_triggered:
                try:
                    if self._allow_side_effects():
                        pyautogui.hotkey("alt", "f4")
                        self.pending_status_message = "Mouth open 5s detected: Alt+F4 sent."
                    else:
                        self.pending_status_message = "Calibration safe mode: mouth close action suppressed."
                except Exception as exc:
                    self.pending_status_message = f"Window close error: {exc}"
                    self._record_error(self.pending_status_message)
                self.mouth_triggered = True
            return "triggered" if self.mouth_triggered else f"open {elapsed:.1f}/{self.mouth_hold_seconds:.1f}s"
        self.mouth_open_start = None
        self.mouth_triggered = False
        self.mouth_open_active = False
        return "closed"

    def _capture_loop(self) -> None:
        read_failures = 0
        try:
            while not self.stop_event.is_set() and self.cap is not None:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures >= 40:
                        self.pending_status_message = "Camera feed stalled. Stop and start session to recover."
                        self._record_warning(self.pending_status_message)
                        read_failures = 0
                        time.sleep(0.1)
                    else:
                        time.sleep(0.02)
                    continue
                read_failures = 0
                if self.mirror_preview:
                    frame = cv2.flip(frame, 1)
                captured_at = time.monotonic()
                self._update_capture_metrics(captured_at)

                with self.latest_frame_lock:
                    if self.latest_captured_frame is not None:
                        self.dropped_frames += 1
                    self.capture_sequence += 1
                    self.latest_captured_frame = (self.capture_sequence, captured_at, frame)
                    self.latest_frame_ready.set()
        except Exception as exc:
            message = f"Capture loop error: {exc}"
            self._record_error(message)
            with self.state_lock:
                self.state["session_status"] = "error"
                self.state["status"] = message
                self.state["errors"] = list(self.error_history)
                self.state["running"] = False
                self.state["updatedAt"] = time.time()
            self.running = False
            self.stop_event.set()
        finally:
            self.latest_frame_ready.set()
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _processing_loop(self) -> None:
        try:
            while True:
                if self.stop_event.is_set():
                    with self.latest_frame_lock:
                        if self.latest_captured_frame is None:
                            break
                if not self.latest_frame_ready.wait(timeout=0.2):
                    continue
                with self.latest_frame_lock:
                    item = self.latest_captured_frame
                    self.latest_captured_frame = None
                    self.latest_frame_ready.clear()
                if item is None:
                    continue
                _, captured_at, frame = item

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = self._next_timestamp_ms()
                self.landmarker.detect_async(mp_image, timestamp_ms)
                if self.face_landmarker is not None:
                    self.face_landmarker.detect_async(mp_image, timestamp_ms)
                if self.gesture_recognizer is not None:
                    self.gesture_recognizer.recognize_async(mp_image, timestamp_ms)

                with self.result_lock:
                    result = self.latest_result
                with self.face_result_lock:
                    face_results = self.latest_face_result
                with self.gesture_result_lock:
                    gesture_results = self.latest_gesture_result

                right_hand, left_hand, right_hand_world, left_hand_world = self._extract_hands(result)

                if right_hand is not None:
                    draw_landmarks(frame, right_hand, (0, 255, 224))
                if left_hand is not None:
                    draw_landmarks(frame, left_hand, (255, 180, 0))

                now = time.monotonic()
                pen_target, pen_state, pen_frame_point, pen_roi = self.red_pen.update(frame, right_hand, now)
                self.pen_target = pen_target
                self.pen_state = pen_state
                self.pen_frame_point = pen_frame_point
                self.pen_roi = pen_roi

                pen_active = pen_target is not None
                if not pen_active and self.pointer_was_active:
                    self._reset_pointer_filter()
                self.pointer_was_active = pen_active

                if pen_roi is not None:
                    x1, y1, x2, y2 = pen_roi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 200), 1)
                if pen_frame_point is not None:
                    cv2.circle(frame, pen_frame_point, 12, (0, 0, 255), 2)
                    cv2.circle(frame, pen_frame_point, 4, (0, 0, 255), -1)

                left_text, pointer_mode = self._handle_left(left_hand, left_hand_world, pen_target, pen_active, now)
                self._collect_calibration_samples(
                    now,
                    self.latest_left_image_ratio,
                    self.latest_left_world_ratio,
                    pen_target,
                )
                if pen_active and pen_target is not None:
                    self.pointer_mode = pointer_mode
                    self._move_cursor(pen_target, mode=pointer_mode, now=now)
                else:
                    self.pointer_mode = "idle"
                left_priority = left_text != "idle"

                if pen_active or left_priority:
                    self._handle_right(None, None, None, now)
                    right_text = "suppressed" if right_hand is not None else "idle"
                else:
                    right_text = self._handle_right(right_hand, right_hand_world, gesture_results, now)

                mouth_text = self._handle_mouth(face_results, now)

                cv2.putText(frame, f"R: {right_text}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 224), 2)
                cv2.putText(frame, f"L: {left_text}", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
                cv2.putText(frame, f"M: {mouth_text}", (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 2)
                cv2.putText(frame, f"P: {pen_state}", (14, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 160), 2)

                if self.right_hold_progress > 0 and self.prev_right_gesture:
                    width = int(frame.shape[1] * self.right_hold_progress)
                    cv2.rectangle(frame, (0, frame.shape[0] - 8), (width, frame.shape[0]), (0, 255, 180), -1)

                encode_ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
                if encode_ok:
                    with self.frame_lock:
                        self.last_frame_jpeg = jpg.tobytes()

                status = self.pending_status_message
                self.pending_status_message = None
                if status:
                    status_lower = status.lower()
                    if "error" in status_lower:
                        self._record_error(status)
                    elif "unavailable" in status_lower or "disabled" in status_lower:
                        self._record_warning(status)
                self._update_inference_metrics(time.monotonic(), captured_at)

                with self.state_lock:
                    if status:
                        self.state["status"] = status
                    self.state["running"] = bool(self.running)
                    self.state["session_status"] = "running" if self.running else "idle"
                    self.state["camera_index"] = self.active_camera_index
                    self.state["gestureText"] = f"R: {right_text}\nL: {left_text}\nM: {mouth_text}\nP: {pen_state}"
                    self.state["rightText"] = right_text
                    self.state["leftText"] = left_text
                    self.state["mouthText"] = mouth_text
                    self.state["penState"] = pen_state
                    self.state["lastKey"] = self.last_key_text
                    self.state["keyCount"] = self.key_count
                    self.state["holdProgress"] = float(self.right_hold_progress)
                    self.state["warnings"] = list(self.warning_history)
                    self.state["errors"] = list(self.error_history)
                    self.state["metrics"] = {
                        "capture_fps": round(self.capture_fps, 2),
                        "inference_fps": round(self.inference_fps, 2),
                        "dropped_frames": int(self.dropped_frames),
                        "end_to_end_latency_ms": round(self.end_to_end_latency_ms, 2),
                    }
                    self.state["interaction"] = {
                        "pinch_state": self.pinch_state,
                        "pinch_ratio": None if self.pinch_last_ratio is None else round(self.pinch_last_ratio, 4),
                        "pinch_source": self.pinch_ratio_source,
                        "pointer_mode": self.pointer_mode,
                        "right_gesture_source": self.right_gesture_source,
                        "right_gesture_confidence": round(self.right_gesture_confidence, 3),
                        "mouth_blend_score": (
                            None if self.mouth_blend_score is None else round(self.mouth_blend_score, 3)
                        ),
                        "mouth_signal_mode": self.mouth_signal_mode,
                    }
                    self.state["calibration"] = self._calibration_state_snapshot()
                    self.state["updatedAt"] = time.time()
        except Exception as exc:
            message = f"Processing loop error: {exc}"
            self._record_error(message)
            with self.state_lock:
                self.state["session_status"] = "error"
                self.state["status"] = message
                self.state["errors"] = list(self.error_history)
                self.state["running"] = False
                self.state["updatedAt"] = time.time()
            self.running = False
            self.stop_event.set()
        finally:
            self._stop_left_controls()
