"""
GestureKeys v2 - dual-hand control and mouth-triggered window close.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
import urllib.request
from ctypes import POINTER, cast
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import font, ttk

try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
except Exception:  # pragma: no cover
    CLSCTX_ALL = None
    AudioUtilities = None
    IAudioEndpointVolume = None

try:
    import screen_brightness_control as sbc
except Exception:  # pragma: no cover
    sbc = None


MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0


@dataclass
class UiState:
    frame_rgb: Optional[object]
    gesture_text: str
    status_text: Optional[str] = None
    key_text: Optional[str] = None
    count_text: Optional[str] = None


def ensure_models() -> bool:
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if os.path.exists(FACE_MODEL_PATH):
        return True
    if not os.path.exists(FACE_MODEL_PATH):
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        except Exception:
            return False
    return True


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def landmark_distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def draw_landmarks(frame, landmarks, color) -> None:
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], color, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, color, -1)
        cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)


def fingers_up(landmarks) -> list[bool]:
    center_x = sum(landmarks[i].x for i in (0, 5, 9, 13, 17)) / 5
    thumb_tip_offset = abs(landmarks[4].x - center_x)
    thumb_joint_offset = abs(landmarks[3].x - center_x)
    thumb_up = thumb_tip_offset > thumb_joint_offset + 0.02
    up = [thumb_up]
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        up.append(landmarks[tip].y < landmarks[pip].y)
    return up


def pinch_ratio(landmarks) -> float:
    return landmark_distance(landmarks[4], landmarks[8]) / max(
        landmark_distance(landmarks[5], landmarks[17]), 1e-6
    )


def classify_right_hand(landmarks) -> tuple[Optional[str], Optional[str]]:
    thumb, index, middle, ring, pinky = fingers_up(landmarks)
    count = sum((thumb, index, middle, ring, pinky))
    pinch = pinch_ratio(landmarks) < 0.32
    if pinch and not middle and not ring and not pinky:
        return ("Pinch - Copy", "ctrl+c")
    if count == 0:
        return ("Fist - Space", "space")
    if thumb and not index and not middle and not ring and not pinky:
        return ("Thumb Up - Enter", "enter")
    if not thumb and index and not middle and not ring and not pinky:
        return ("Index - Up", "up")
    if not thumb and index and middle and not ring and not pinky:
        return ("Victory - Down", "down")
    if not thumb and index and middle and ring and not pinky:
        return ("Three - Left", "left")
    if count == 5:
        return ("Open Hand - Right", "right")
    if not thumb and not index and middle and not ring and not pinky:
        return ("Middle - Paste", "ctrl+v")
    if thumb and not index and not middle and not ring and pinky:
        return ("Shaka - Escape", "escape")
    return (None, None)


def mouth_open_ratio(face_landmarks) -> float:
    up = face_landmarks[13]
    low = face_landmarks[14]
    left = face_landmarks[78]
    right = face_landmarks[308]
    return landmark_distance(up, low) / max(landmark_distance(left, right), 1e-6)


def hand_openness_ratio(landmarks) -> float:
    wrist = landmarks[0]
    palm_span = landmark_distance(landmarks[5], landmarks[17])
    ratios = []
    for tip, mcp in ((4, 2), (8, 5), (12, 9), (16, 13), (20, 17)):
        tip_dist = landmark_distance(landmarks[tip], wrist)
        mcp_dist = landmark_distance(landmarks[mcp], wrist)
        ratios.append((tip_dist - mcp_dist) / max(palm_span, 1e-6))

    raw = sum(ratios) / len(ratios)
    return clamp((raw - 0.35) / 1.55, 0.0, 1.0)


class VolumeController:
    def __init__(self) -> None:
        self.endpoint = None
        self.available = False
        self.error: Optional[str] = None
        if not AudioUtilities:
            self.error = "pycaw not available"
            return
        try:
            speakers = AudioUtilities.GetSpeakers()
            endpoint = getattr(speakers, "EndpointVolume", None)
            if endpoint is None and IAudioEndpointVolume and CLSCTX_ALL is not None and hasattr(speakers, "Activate"):
                interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint = cast(interface, POINTER(IAudioEndpointVolume))
            self.endpoint = endpoint
            self.available = endpoint is not None
            if not self.available:
                self.error = "no audio endpoint"
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def get_percent(self) -> Optional[int]:
        if not self.available or self.endpoint is None:
            return None
        try:
            return int(round(self.endpoint.GetMasterVolumeLevelScalar() * 100))
        except Exception:
            return None

    def set_percent(self, value: float) -> bool:
        if not self.available or self.endpoint is None:
            return False
        try:
            self.endpoint.SetMasterVolumeLevelScalar(clamp(value, 0, 100) / 100.0, None)
            return True
        except Exception:
            return False


class BrightnessController:
    def __init__(self) -> None:
        self.available = sbc is not None

    def get_percent(self) -> Optional[int]:
        if not self.available:
            return None
        try:
            value = sbc.get_brightness(display=0)
            if isinstance(value, list):
                value = value[0]
            return int(value)
        except Exception:
            return None

    def set_percent(self, value: float) -> bool:
        if not self.available:
            return False
        try:
            sbc.set_brightness(int(clamp(value, 0, 100)), display=0)
            return True
        except Exception:
            return False


class GestureKeyApp:
    def __init__(self, root: tk.Tk, default_camera: Optional[int], key_output_enabled: bool) -> None:
        self.root = root
        self.root.title("GestureKeys")
        self.root.geometry("980x680")
        self.root.configure(bg="#0d0d0f")
        self.root.resizable(True, True)

        self.running = False
        self.cap = None
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.latest_result = None
        self.result_lock = threading.Lock()
        self.ui_state: Optional[UiState] = None
        self.ui_state_lock = threading.Lock()

        self.landmarker = None
        self.face_landmarker = None

        self.cooldown = 0.8
        self.hold_threshold = 0.5
        self.last_trigger_time = 0.0
        self.right_hold_start: Optional[float] = None
        self.prev_right_gesture: Optional[str] = None
        self.right_hold_progress = 0.0

        self.key_count = 0
        self.last_key_text = "-"
        self.pending_status_message: Optional[str] = None
        self.enabled = key_output_enabled
        self.mirror_preview = True
        self.default_camera = default_camera

        self.left_mode = "idle"
        self.left_dragging = False
        self.left_anchor_y: Optional[float] = None
        self.left_base_value: Optional[int] = None
        self.left_last_value: Optional[int] = None
        self.cursor_x: Optional[float] = None
        self.cursor_y: Optional[float] = None
        self.screen_width, self.screen_height = pyautogui.size()

        self.volume = VolumeController()
        self.brightness = BrightnessController()
        self.warned_volume_unavailable = False
        self.warned_brightness_unavailable = False

        self.mouth_threshold = 0.08
        self.mouth_hold_seconds = 5.0
        self.mouth_open_start: Optional[float] = None
        self.mouth_triggered = False

        self._build_ui()
        self._init_landmarker()
        self._schedule_ui_refresh()

    def _build_ui(self) -> None:
        title_font = font.Font(family="Consolas", size=18, weight="bold")
        label_font = font.Font(family="Consolas", size=11)
        mono_font = font.Font(family="Consolas", size=13, weight="bold")
        small_font = font.Font(family="Consolas", size=9)
        bg, panel, accent, dim, text = "#0d0d0f", "#16161a", "#00ffe0", "#3a3a4a", "#e0e0e8"

        header = tk.Frame(self.root, bg=bg, pady=10)
        header.pack(fill="x", padx=20)
        tk.Label(header, text="GESTURE", font=title_font, bg=bg, fg=accent).pack(side="left")
        tk.Label(header, text="KEYS", font=title_font, bg=bg, fg=text).pack(side="left")
        tk.Label(header, text="  camera-to-keyboard bridge", font=small_font, bg=bg, fg=dim).pack(
            side="left", padx=10
        )

        cols = tk.Frame(self.root, bg=bg)
        cols.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        right = tk.Frame(cols, bg=bg, width=290)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        stat = tk.Frame(right, bg=panel, pady=14, padx=14)
        stat.pack(fill="x", pady=(0, 10))
        tk.Label(stat, text="DETECTED STATE", font=small_font, bg=panel, fg=dim).pack(anchor="w")
        self.gesture_var = tk.StringVar(value="R: -\nL: -\nM: -")
        tk.Label(
            stat,
            textvariable=self.gesture_var,
            font=small_font,
            bg=panel,
            fg=accent,
            wraplength=240,
            justify="left",
        ).pack(anchor="w", pady=(2, 10))
        tk.Label(stat, text="LAST KEY SENT", font=small_font, bg=panel, fg=dim).pack(anchor="w")
        self.key_var = tk.StringVar(value="-")
        tk.Label(stat, textvariable=self.key_var, font=mono_font, bg=panel, fg=text).pack(
            anchor="w", pady=(2, 10)
        )
        tk.Label(stat, text="KEYS TRIGGERED", font=small_font, bg=panel, fg=dim).pack(anchor="w")
        self.count_var = tk.StringVar(value="0")
        tk.Label(stat, textvariable=self.count_var, font=mono_font, bg=panel, fg=text).pack(anchor="w")

        ctrl = tk.Frame(right, bg=panel, pady=14, padx=14)
        ctrl.pack(fill="x", pady=(0, 10))
        tk.Label(ctrl, text="SETTINGS", font=small_font, bg=panel, fg=dim).pack(anchor="w", pady=(0, 8))
        tk.Label(ctrl, text="Camera", font=small_font, bg=panel, fg=text).pack(anchor="w")
        self.camera_var = tk.StringVar(value="Auto" if self.default_camera is None else str(self.default_camera))
        ttk.Combobox(
            ctrl,
            textvariable=self.camera_var,
            state="readonly",
            values=["Auto", "0", "1", "2", "3", "4"],
        ).pack(fill="x", pady=(2, 8))

        tk.Label(ctrl, text="Right-hand cooldown (s)", font=small_font, bg=panel, fg=text).pack(anchor="w")
        self.cooldown_var = tk.DoubleVar(value=self.cooldown)
        ttk.Scale(
            ctrl,
            from_=0.2,
            to=2.0,
            variable=self.cooldown_var,
            orient="horizontal",
            command=lambda v: setattr(self, "cooldown", float(v)),
        ).pack(fill="x", pady=(2, 8))
        tk.Label(ctrl, text="Right-hand hold to fire (s)", font=small_font, bg=panel, fg=text).pack(anchor="w")
        self.hold_var = tk.DoubleVar(value=self.hold_threshold)
        ttk.Scale(
            ctrl,
            from_=0.1,
            to=1.5,
            variable=self.hold_var,
            orient="horizontal",
            command=lambda v: setattr(self, "hold_threshold", float(v)),
        ).pack(fill="x", pady=(2, 8))

        self.enabled_var = tk.BooleanVar(value=self.enabled)
        tk.Checkbutton(
            ctrl,
            text="Enable right-hand key output",
            variable=self.enabled_var,
            bg=panel,
            fg=text,
            selectcolor=panel,
            activebackground=panel,
            font=small_font,
            command=lambda: setattr(self, "enabled", self.enabled_var.get()),
        ).pack(anchor="w")
        self.mirror_var = tk.BooleanVar(value=self.mirror_preview)
        tk.Checkbutton(
            ctrl,
            text="Mirror preview",
            variable=self.mirror_var,
            bg=panel,
            fg=text,
            selectcolor=panel,
            activebackground=panel,
            font=small_font,
            command=lambda: setattr(self, "mirror_preview", self.mirror_var.get()),
        ).pack(anchor="w")

        btn = tk.Frame(right, bg=bg)
        btn.pack(fill="x")
        self.start_btn = tk.Button(
            btn,
            text="START",
            font=label_font,
            bg=accent,
            fg="#000000",
            relief="flat",
            padx=14,
            pady=8,
            cursor="hand2",
            command=self.start_camera,
        )
        self.start_btn.pack(fill="x", pady=(0, 6))
        self.stop_btn = tk.Button(
            btn,
            text="STOP",
            font=label_font,
            bg=dim,
            fg=text,
            relief="flat",
            padx=14,
            pady=8,
            cursor="hand2",
            state="disabled",
            command=self.stop_camera,
        )
        self.stop_btn.pack(fill="x")

        ref = tk.Frame(right, bg=panel, padx=14, pady=10)
        ref.pack(fill="x", pady=(10, 0))
        tk.Label(ref, text="GESTURE MAP", font=small_font, bg=panel, fg=dim).pack(anchor="w", pady=(0, 6))
        for name, action in [
            ("Right hand", "Keyboard map"),
            ("Left pinch", "Mouse drag"),
            ("Left pinky", "Volume"),
            ("Left fist/open", "Brightness"),
            ("Mouth 5s", "Alt+F4"),
        ]:
            row = tk.Frame(ref, bg=panel)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, font=small_font, bg=panel, fg=text, width=12, anchor="w").pack(side="left")
            tk.Label(row, text=action, font=small_font, bg=panel, fg=accent, anchor="w").pack(side="left")

        self.status_var = tk.StringVar(value="Loading gesture model...")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=small_font,
            bg="#0a0a0c",
            fg=dim,
            anchor="w",
            pady=4,
            padx=14,
        ).pack(fill="x", side="bottom")

        left = tk.Frame(cols, bg=panel)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        tk.Label(left, text="CAMERA FEED", font=small_font, bg=panel, fg=dim).pack(anchor="w", padx=10, pady=(10, 4))
        self.preview_font = font.Font(family="Consolas", size=15, weight="bold")
        self.cam_canvas = tk.Label(
            left,
            bg="#000000",
            fg="#8b93a7",
            width=640,
            height=420,
            justify="center",
            font=self.preview_font,
            compound="center",
        )
        self.cam_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._set_preview_message("Loading model...", "GestureKeys is preparing the hand detector.")

    def _init_landmarker(self) -> None:
        def callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
            with self.result_lock:
                self.latest_result = result

        face_model_ready = ensure_models()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            result_callback=callback,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        if face_model_ready:
            self.face_landmarker = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.6,
                    min_face_presence_confidence=0.6,
                    min_tracking_confidence=0.6,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
            )
        else:
            self.face_landmarker = None
            self.pending_status_message = (
                "Mouth-close disabled: face model could not be downloaded."
            )
        self.status_var.set("Model ready. Click START to begin.")
        self._set_preview_message("Camera idle", "Click START to open your webcam.")

    def _set_preview_message(self, title: str, detail: Optional[str] = None) -> None:
        message = title if not detail else f"{title}\n\n{detail}"
        self.cam_canvas.config(image="", text=message)
        self.cam_canvas.image = None

    def _schedule_ui_refresh(self) -> None:
        self._consume_ui_state()
        self.root.after(33, self._schedule_ui_refresh)

    def _consume_ui_state(self) -> None:
        with self.ui_state_lock:
            state = self.ui_state
            self.ui_state = None
        if not state:
            return
        if state.frame_rgb is not None:
            image = Image.fromarray(state.frame_rgb).resize((640, 420), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=image)
            self.cam_canvas.config(image=photo, text="")
            self.cam_canvas.image = photo
        self.gesture_var.set(state.gesture_text)
        if state.status_text:
            self.status_var.set(state.status_text)
        if state.key_text is not None:
            self.key_var.set(state.key_text)
        if state.count_text is not None:
            self.count_var.set(state.count_text)

    def _publish_ui_state(self, frame_rgb, gesture_text: str, status_text: Optional[str] = None) -> None:
        with self.ui_state_lock:
            self.ui_state = UiState(
                frame_rgb=frame_rgb,
                gesture_text=gesture_text,
                status_text=status_text,
                key_text=self.last_key_text,
                count_text=str(self.key_count),
            )

    def _parse_camera_choice(self) -> Optional[int]:
        selected = self.camera_var.get()
        return None if selected == "Auto" else int(selected)

    def _open_camera(self, preferred: Optional[int]) -> tuple[Optional[object], Optional[int]]:
        candidates = [preferred] if preferred is not None else [0, 1, 2]
        for idx in candidates:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue
            for _ in range(12):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, idx
            cap.release()
        return None, None

    def _extract_hands(self, result) -> tuple[Optional[object], Optional[object]]:
        right, left = None, None
        unknown = []
        if not result or not result.hand_landmarks:
            return None, None
        for i, lms in enumerate(result.hand_landmarks):
            label = None
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                label = getattr(result.handedness[i][0], "category_name", None)
                if label:
                    label = label.lower()
            if label == "right" and right is None:
                right = lms
            elif label == "left" and left is None:
                left = lms
            else:
                unknown.append(lms)
        for lms in unknown:
            if right is None:
                right = lms
            elif left is None:
                left = lms
        return right, left

    def _send_key(self, key_action: str) -> None:
        if not self.enabled:
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

    def _release_drag(self) -> None:
        if self.left_dragging:
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
        self.left_dragging = False
        self.cursor_x = None
        self.cursor_y = None

    def _stop_left_controls(self) -> None:
        self._release_drag()
        self.left_mode = "idle"
        self.left_anchor_y = None
        self.left_base_value = None
        self.left_last_value = None

    def _handle_right(self, landmarks, now: float) -> str:
        if landmarks is None:
            self.right_hold_start = None
            self.prev_right_gesture = None
            self.right_hold_progress = 0.0
            return "idle"
        gesture, key = classify_right_hand(landmarks)
        if gesture and gesture == self.prev_right_gesture:
            if self.right_hold_start is None:
                self.right_hold_start = now
                self.right_hold_progress = 0.0
            else:
                elapsed = now - self.right_hold_start
                self.right_hold_progress = clamp(elapsed / self.hold_threshold, 0.0, 1.0)
                if elapsed >= self.hold_threshold and now - self.last_trigger_time >= self.cooldown:
                    if key:
                        self._send_key(key)
                    self.last_trigger_time = now
                    self.right_hold_start = None
                    self.right_hold_progress = 0.0
        else:
            self.right_hold_start = None
            self.right_hold_progress = 0.0
        self.prev_right_gesture = gesture
        return gesture or "none"

    def _handle_drag(self, landmarks) -> str:
        tip = landmarks[8]
        self.left_mode = "drag"
        tx = int(clamp(tip.x, 0.0, 1.0) * (self.screen_width - 1))
        ty = int(clamp(tip.y, 0.0, 1.0) * (self.screen_height - 1))
        if self.cursor_x is None or self.cursor_y is None:
            self.cursor_x, self.cursor_y = float(tx), float(ty)
        else:
            self.cursor_x += (tx - self.cursor_x) * 0.35
            self.cursor_y += (ty - self.cursor_y) * 0.35
        if not self.left_dragging:
            pyautogui.mouseDown(button="left")
            self.left_dragging = True
        pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y), duration=0)
        return "drag"

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
        if self.left_anchor_y is None or self.left_base_value is None:
            return f"{mode} {current}%"
        delta = (self.left_anchor_y - y_now) * 180
        target = int(round(clamp(self.left_base_value + delta, 0, 100)))
        step = 2 if mode == "brightness" else 1
        if self.left_last_value is None or abs(target - self.left_last_value) >= step:
            if setter(target):
                self.left_last_value = target
        shown = self.left_last_value if self.left_last_value is not None else current
        return f"{mode} {shown}%"

    def _handle_fist_brightness(self, landmarks) -> str:
        current = self.brightness.get_percent()
        if current is None:
            if not self.warned_brightness_unavailable:
                self.pending_status_message = "Brightness control unavailable on this monitor setup."
                self.warned_brightness_unavailable = True
            return "brightness unavailable"

        if self.left_mode != "brightness_fist":
            self._release_drag()
            self.left_mode = "brightness_fist"
            self.left_last_value = current

        openness = hand_openness_ratio(landmarks)
        target = int(round(1 + openness * 99))
        if self.left_last_value is None or abs(target - self.left_last_value) >= 1:
            if self.brightness.set_percent(target):
                self.left_last_value = target

        shown = self.left_last_value if self.left_last_value is not None else target
        return f"brightness {shown}%"

    def _handle_left(self, landmarks) -> str:
        if landmarks is None:
            self._stop_left_controls()
            return "idle"
        thumb, index, middle, ring, pinky = fingers_up(landmarks)
        finger_count = sum((thumb, index, middle, ring, pinky))
        fist = finger_count == 0
        pinch = pinch_ratio(landmarks) < 0.32
        pinky_only = (not thumb) and (not index) and (not middle) and (not ring) and pinky
        if pinky_only:
            text = self._handle_slide("volume", landmarks[0].y, self.volume.get_percent, self.volume.set_percent)
            if text == "volume unavailable" and not self.warned_volume_unavailable:
                detail = f" ({self.volume.error})" if self.volume.error else ""
                self.pending_status_message = f"Volume control unavailable on this device/environment{detail}."
                self.warned_volume_unavailable = True
            return text
        if pinch:
            return self._handle_drag(landmarks)
        if fist or self.left_mode == "brightness_fist":
            return self._handle_fist_brightness(landmarks)
        self._stop_left_controls()
        return "idle"

    def _handle_mouth(self, face_results, now: float) -> str:
        if self.face_landmarker is None:
            return "disabled"
        open_now = False
        if face_results and face_results.face_landmarks:
            open_now = mouth_open_ratio(face_results.face_landmarks[0]) >= self.mouth_threshold
        if open_now:
            if self.mouth_open_start is None:
                self.mouth_open_start = now
                self.mouth_triggered = False
            elapsed = now - self.mouth_open_start
            if elapsed >= self.mouth_hold_seconds and not self.mouth_triggered:
                pyautogui.hotkey("alt", "f4")
                self.pending_status_message = "Mouth open 5s detected: Alt+F4 sent."
                self.mouth_triggered = True
            return "triggered" if self.mouth_triggered else f"open {elapsed:.1f}/{self.mouth_hold_seconds:.1f}s"
        self.mouth_open_start = None
        self.mouth_triggered = False
        return "closed"

    def start_camera(self) -> None:
        if self.running:
            return
        if self.landmarker is None:
            self._set_preview_message("Model failed to load", "hand_landmarker.task is missing.")
            return
        self._set_preview_message("Opening camera...", "Trying the selected webcam.")
        cap, idx = self._open_camera(self._parse_camera_choice())
        if cap is None:
            self._set_preview_message("Camera unavailable", "Try Camera 1 or 2 in settings.")
            self.status_var.set("Camera error: unable to read frames from any camera.")
            return
        self.cap = cap
        self.stop_event.clear()
        self.running = True
        self.last_trigger_time = 0.0
        self.right_hold_start = None
        self.prev_right_gesture = None
        self.right_hold_progress = 0.0
        self._stop_left_controls()
        self.mouth_open_start = None
        self.mouth_triggered = False
        self.pending_status_message = None
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set(f"Camera running on index {idx}. Show your hands and face.")
        self.capture_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.capture_thread.start()

    def stop_camera(self) -> None:
        if not self.running:
            return
        self.running = False
        self.stop_event.set()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        self.capture_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._stop_left_controls()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._set_preview_message("Camera stopped", "Click START to open the webcam again.")
        self.gesture_var.set("R: -\nL: -\nM: -")
        self.status_var.set("Stopped.")

    def _camera_loop(self) -> None:
        ts = 0
        while not self.stop_event.is_set() and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            if self.mirror_preview:
                frame = cv2.flip(frame, 1)
            ts += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.landmarker.detect_async(mp_image, ts)
            face_results = self.face_landmarker.detect(mp_image) if self.face_landmarker else None
            with self.result_lock:
                result = self.latest_result
            right_hand, left_hand = self._extract_hands(result)
            if right_hand is not None:
                draw_landmarks(frame, right_hand, (0, 255, 224))
            if left_hand is not None:
                draw_landmarks(frame, left_hand, (255, 180, 0))
            now = time.monotonic()
            right_text = self._handle_right(right_hand, now)
            left_text = self._handle_left(left_hand)
            mouth_text = self._handle_mouth(face_results, now)

            cv2.putText(frame, f"R: {right_text}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 224), 2)
            cv2.putText(frame, f"L: {left_text}", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
            cv2.putText(frame, f"M: {mouth_text}", (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 2)
            if self.right_hold_progress > 0 and self.prev_right_gesture:
                w = int(frame.shape[1] * self.right_hold_progress)
                cv2.rectangle(frame, (0, frame.shape[0] - 8), (w, frame.shape[0]), (0, 255, 180), -1)

            ui_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            status = self.pending_status_message
            self.pending_status_message = None
            self._publish_ui_state(ui_frame, f"R: {right_text}\nL: {left_text}\nM: {mouth_text}", status)

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._stop_left_controls()

    def on_close(self) -> None:
        self.stop_camera()
        if self.landmarker is not None:
            self.landmarker.close()
        if self.face_landmarker is not None:
            self.face_landmarker.close()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map webcam hand gestures to keyboard input.")
    parser.add_argument("--camera", type=int, default=None, help="Preferred camera index.")
    parser.add_argument(
        "--disable-keys",
        action="store_true",
        help="Detect gestures without sending right-hand keyboard input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = GestureKeyApp(root, default_camera=args.camera, key_output_enabled=not args.disable_keys)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
