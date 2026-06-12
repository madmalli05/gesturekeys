"""
GestureKeys v4 - dual-hand control with green-pen pointer assist.
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
import numpy as np
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


@dataclass(frozen=True)
class ThemeTokens:
    bg_base: str = "#0b1118"
    bg_shell: str = "#0f1621"
    bg_card: str = "#141e2d"
    bg_card_alt: str = "#1a2637"
    bg_preview: str = "#0c121b"
    bg_status: str = "#0b131d"
    border_soft: str = "#283449"
    border_strong: str = "#33465f"
    text_primary: str = "#f2f6fd"
    text_secondary: str = "#c8d3e4"
    text_muted: str = "#8f9cb0"
    accent: str = "#7de9ff"
    accent_soft: str = "#4cc8e0"
    accent_deep: str = "#1f6d80"
    positive: str = "#89f2c7"
    danger: str = "#ff97a1"
    button_primary: str = "#5fdaf3"
    button_primary_hover: str = "#8deeff"
    button_secondary: str = "#223146"
    button_secondary_hover: str = "#2f4360"
    button_disabled: str = "#1a2432"


def ensure_models() -> bool:
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if os.path.exists(FACE_MODEL_PATH):
        return True
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        return True
    except Exception:
        return False


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
    if thumb and index and middle and not ring and not pinky:
        return ("Thumb+Index+Middle - Paste", "ctrl+v")
    if thumb and not index and not middle and not ring and pinky:
        return ("Shaka - Escape", "escape")
    return (None, None)


def mouth_open_ratio(face_landmarks) -> float:
    upper = face_landmarks[13]
    lower = face_landmarks[14]
    left = face_landmarks[78]
    right = face_landmarks[308]
    return landmark_distance(upper, lower) / max(landmark_distance(left, right), 1e-6)


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


class RedPenTracker:
    def __init__(self, screen_width: int, screen_height: int) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Green range in HSV (OpenCV hue scale 0-179).
        self.hsv_lower = np.array([35, 90, 60], dtype=np.uint8)
        self.hsv_upper = np.array([85, 255, 255], dtype=np.uint8)
        self.sensitivity_percent = 50.0
        self.kernel = np.ones((5, 5), dtype=np.uint8)

        self.roi_padding = 28
        self.min_area_px = 220.0
        self.virtual_inset_ratio = 0.15
        self.pen_lost_grace = 0.5
        self.deadzone_px = 3.5
        self.base_alpha = 0.10
        self.speed_gain = 0.0045
        self.min_alpha = 0.07
        self.max_alpha = 0.72
        self.filtered: Optional[np.ndarray] = None
        self.last_raw: Optional[np.ndarray] = None
        self.last_time: Optional[float] = None
        self.last_seen_time: Optional[float] = None
        self.last_screen_point: Optional[tuple[int, int]] = None
        self.last_frame_point: Optional[tuple[int, int]] = None
        self.last_roi: Optional[tuple[int, int, int, int]] = None
        self.set_sensitivity(self.sensitivity_percent)

    def reset(self) -> None:
        self.filtered = None
        self.last_raw = None
        self.last_time = None
        self.last_seen_time = None
        self.last_screen_point = None
        self.last_frame_point = None
        self.last_roi = None

    def set_sensitivity(self, percent: float) -> None:
        value = clamp(percent, 0.0, 100.0)
        t = value / 100.0
        low_h = int(round(45 - 15 * t))
        high_h = int(round(75 + 20 * t))
        min_s = int(round(140 - 80 * t))
        min_v = int(round(110 - 60 * t))
        self.hsv_lower = np.array([low_h, min_s, min_v], dtype=np.uint8)
        self.hsv_upper = np.array([high_h, 255, 255], dtype=np.uint8)
        self.sensitivity_percent = value

    def _hand_roi(self, hand_landmarks, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
        if hand_landmarks is None:
            return None
        xs = [int(clamp(lm.x, 0.0, 1.0) * (width - 1)) for lm in hand_landmarks]
        ys = [int(clamp(lm.y, 0.0, 1.0) * (height - 1)) for lm in hand_landmarks]
        if not xs or not ys:
            return None
        x1 = max(min(xs) - self.roi_padding, 0)
        y1 = max(min(ys) - self.roi_padding, 0)
        x2 = min(max(xs) + self.roi_padding, width - 1)
        y2 = min(max(ys) + self.roi_padding, height - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _smooth(self, point: tuple[int, int], now: float) -> tuple[int, int]:
        current = np.array([float(point[0]), float(point[1])], dtype=np.float64)
        if self.filtered is None:
            self.filtered = current
            self.last_raw = current
            self.last_time = now
            return int(current[0]), int(current[1])

        dt = max(now - (self.last_time if self.last_time is not None else now), 1e-3)
        previous_raw = self.last_raw if self.last_raw is not None else current
        speed = float(np.linalg.norm(current - previous_raw) / dt)
        alpha = clamp(self.base_alpha + speed * self.speed_gain, self.min_alpha, self.max_alpha)
        raw = current.copy()
        if float(np.linalg.norm(current - self.filtered)) < self.deadzone_px:
            current = self.filtered.copy()
        self.filtered = self.filtered + alpha * (current - self.filtered)
        self.last_raw = raw
        self.last_time = now
        return int(self.filtered[0]), int(self.filtered[1])

    def _frame_to_screen(self, point: tuple[int, int], width: int, height: int) -> tuple[int, int]:
        inset_x = int(round(max(width - 1, 1) * self.virtual_inset_ratio))
        inset_y = int(round(max(height - 1, 1) * self.virtual_inset_ratio))
        min_x = inset_x
        min_y = inset_y
        max_x = max((width - 1) - inset_x, min_x + 1)
        max_y = max((height - 1) - inset_y, min_y + 1)
        norm_x = clamp((point[0] - min_x) / max(max_x - min_x, 1), 0.0, 1.0)
        norm_y = clamp((point[1] - min_y) / max(max_y - min_y, 1), 0.0, 1.0)
        sx = int(norm_x * (self.screen_width - 1))
        sy = int(norm_y * (self.screen_height - 1))
        return sx, sy

    def update(
        self, frame_bgr, right_hand_landmarks, now: float
    ) -> tuple[Optional[tuple[int, int]], str, Optional[tuple[int, int]], Optional[tuple[int, int, int, int]]]:
        height, width = frame_bgr.shape[:2]
        roi = self._hand_roi(right_hand_landmarks, width, height)
        if roi is None:
            self.reset()
            return None, "no-right-hand", None, None

        x1, y1, x2, y2 = roi
        crop = frame_bgr[y1:y2, x1:x2]
        detection_failed = crop.size == 0

        frame_point: Optional[tuple[int, int]] = None
        if not detection_failed:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 90, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(best)
                if area >= self.min_area_px:
                    moments = cv2.moments(best)
                    if moments["m00"] > 1e-6:
                        frame_point = (
                            int(moments["m10"] / moments["m00"]) + x1,
                            int(moments["m01"] / moments["m00"]) + y1,
                        )
            detection_failed = frame_point is None

        if detection_failed:
            if self.last_seen_time is not None and self.last_screen_point is not None:
                elapsed = now - self.last_seen_time
                if elapsed < self.pen_lost_grace:
                    hold_text = f"hold {elapsed:.1f}/{self.pen_lost_grace:.1f}s"
                    return self.last_screen_point, hold_text, self.last_frame_point, self.last_roi
            self.filtered = None
            self.last_raw = None
            self.last_time = None
            self.last_seen_time = None
            self.last_screen_point = None
            self.last_frame_point = None
            self.last_roi = None
            return None, "no-pen", None, roi

        screen_point = self._smooth(self._frame_to_screen(frame_point, width, height), now)
        self.last_seen_time = now
        self.last_screen_point = screen_point
        self.last_frame_point = frame_point
        self.last_roi = roi
        return screen_point, "ready", frame_point, roi


class GestureKeyApp:
    def __init__(self, root: tk.Tk, default_camera: Optional[int], key_output_enabled: bool) -> None:
        self.root = root
        self.theme = ThemeTokens()
        self.spacing = {
            "outer_x": 24,
            "outer_y": 18,
            "section_gap": 16,
            "column_gap": 14,
            "card_gap": 12,
            "card_pad_x": 16,
            "card_pad_y": 14,
        }
        self._button_palettes: dict[str, dict[str, str]] = {}
        self._init_fonts()
        self.root.title("GestureKeys")
        self.root.geometry("1240x780")
        self.root.minsize(1080, 700)
        self.root.configure(bg=self.theme.bg_base)
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
        self.left_pinch_start: Optional[float] = None
        self.left_pinch_mode: Optional[str] = None
        self.left_pinch_target: Optional[tuple[int, int]] = None
        self.left_right_click_fired = False
        self.left_click_drag_threshold = 0.22
        self.brightness_arm_start: Optional[float] = None
        self.brightness_exit_gesture_start: Optional[float] = None
        self.brightness_arm_hold = 0.22
        self.brightness_exit_hold = 0.25

        self.screen_width, self.screen_height = pyautogui.size()
        self.volume = VolumeController()
        self.brightness = BrightnessController()
        self.red_pen = RedPenTracker(screen_width=self.screen_width, screen_height=self.screen_height)
        self.pen_sensitivity = self.red_pen.sensitivity_percent
        self.pen_state = "no-right-hand"
        self.pen_target: Optional[tuple[int, int]] = None
        self.pen_frame_point: Optional[tuple[int, int]] = None
        self.pen_roi: Optional[tuple[int, int, int, int]] = None
        self.warned_volume_unavailable = False
        self.warned_brightness_unavailable = False

        self.mouth_threshold = 0.08
        self.mouth_hold_seconds = 5.0
        self.mouth_open_start: Optional[float] = None
        self.mouth_triggered = False

        self._configure_ttk_styles()
        self._build_ui()
        self._init_landmarker()
        self._schedule_ui_refresh()

    def _pick_font_family(self, *candidates: str) -> str:
        try:
            available = set(font.families(self.root))
        except Exception:
            available = set()
        for family in candidates:
            if family in available:
                return family
        return candidates[-1] if candidates else "Segoe UI"

    def _init_fonts(self) -> None:
        ui_family = self._pick_font_family("Segoe UI Variable Text", "Segoe UI", "Calibri")
        display_family = self._pick_font_family("Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI")
        mono_family = self._pick_font_family("Cascadia Mono", "Consolas", "Courier New")
        self.fonts = {
            "display": font.Font(family=display_family, size=24, weight="bold"),
            "title": font.Font(family=display_family, size=15, weight="bold"),
            "section": font.Font(family=ui_family, size=10, weight="bold"),
            "body": font.Font(family=ui_family, size=10),
            "meta": font.Font(family=ui_family, size=9),
            "metric": font.Font(family=display_family, size=17, weight="bold"),
            "mono": font.Font(family=mono_family, size=11),
            "button": font.Font(family=ui_family, size=10, weight="bold"),
            "preview": font.Font(family=display_family, size=15, weight="bold"),
            "status": font.Font(family=ui_family, size=9),
        }

    def _configure_ttk_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        def safe_config(style_name: str, **kwargs) -> None:
            try:
                style.configure(style_name, **kwargs)
            except tk.TclError:
                pass

        def safe_map(style_name: str, **kwargs) -> None:
            try:
                style.map(style_name, **kwargs)
            except tk.TclError:
                pass

        t = self.theme
        safe_config(
            "GK.TCombobox",
            foreground=t.text_primary,
            fieldbackground=t.bg_card_alt,
            background=t.bg_card_alt,
            bordercolor=t.border_soft,
            arrowcolor=t.text_secondary,
            padding=(10, 6),
            insertcolor=t.text_primary,
        )
        safe_map(
            "GK.TCombobox",
            fieldbackground=[("readonly", t.bg_card_alt), ("disabled", t.button_disabled)],
            foreground=[("readonly", t.text_primary), ("disabled", t.text_muted)],
            arrowcolor=[("disabled", t.text_muted), ("readonly", t.text_secondary)],
            bordercolor=[("focus", t.accent_soft), ("readonly", t.border_soft)],
        )

        safe_config(
            "GK.Horizontal.TScale",
            background=t.bg_card,
            troughcolor=t.bg_shell,
            bordercolor=t.border_soft,
        )
        safe_map(
            "GK.Horizontal.TScale",
            background=[("disabled", t.button_disabled), ("active", t.bg_card_alt)],
            troughcolor=[("disabled", t.button_disabled), ("active", t.bg_shell)],
        )

        safe_config(
            "GK.TCheckbutton",
            background=t.bg_card,
            foreground=t.text_secondary,
            font=self.fonts["body"],
            indicatorcolor=t.bg_shell,
            indicatormargin=2,
            padding=2,
        )
        safe_map(
            "GK.TCheckbutton",
            background=[("active", t.bg_card), ("selected", t.bg_card)],
            foreground=[("selected", t.text_primary), ("!selected", t.text_secondary), ("disabled", t.text_muted)],
            indicatorcolor=[("selected", t.accent_soft), ("!selected", t.bg_shell), ("disabled", t.button_disabled)],
        )

        self.root.option_add("*TCombobox*Listbox*Background", t.bg_card_alt)
        self.root.option_add("*TCombobox*Listbox*Foreground", t.text_primary)
        self.root.option_add("*TCombobox*Listbox*selectBackground", t.accent_deep)
        self.root.option_add("*TCombobox*Listbox*selectForeground", t.text_primary)

    def _create_card(
        self, parent, *, background: Optional[str] = None, padding: Optional[tuple[int, int]] = None
    ) -> tuple[tk.Frame, tk.Frame]:
        bg = background or self.theme.bg_card
        pad_x, pad_y = padding or (self.spacing["card_pad_x"], self.spacing["card_pad_y"])
        card = tk.Frame(parent, bg=bg, highlightthickness=1, highlightbackground=self.theme.border_soft, bd=0)
        inner = tk.Frame(card, bg=bg, padx=pad_x, pady=pad_y)
        inner.pack(fill="both", expand=True)
        return card, inner

    def _add_card_header(self, parent: tk.Frame, title: str, subtitle: str) -> None:
        tk.Label(
            parent,
            text=title.upper(),
            font=self.fonts["section"],
            bg=parent.cget("bg"),
            fg=self.theme.text_muted,
        ).pack(anchor="w")
        tk.Label(
            parent,
            text=subtitle,
            font=self.fonts["body"],
            bg=parent.cget("bg"),
            fg=self.theme.text_secondary,
            wraplength=330,
            justify="left",
        ).pack(anchor="w", pady=(3, 10))

    def _create_metric_tile(self, parent: tk.Frame, title: str, value_var: tk.StringVar) -> tk.Frame:
        tile = tk.Frame(
            parent,
            bg=self.theme.bg_card_alt,
            highlightthickness=1,
            highlightbackground=self.theme.border_soft,
            padx=11,
            pady=9,
        )
        tk.Label(
            tile,
            text=title.upper(),
            font=self.fonts["meta"],
            bg=self.theme.bg_card_alt,
            fg=self.theme.text_muted,
        ).pack(anchor="w")
        tk.Label(
            tile,
            textvariable=value_var,
            font=self.fonts["metric"],
            bg=self.theme.bg_card_alt,
            fg=self.theme.text_primary,
        ).pack(anchor="w", pady=(3, 0))
        return tile

    def _register_action_button(self, button: tk.Button, role: str) -> None:
        t = self.theme
        if role == "primary":
            palette = {
                "bg": t.button_primary,
                "hover": t.button_primary_hover,
                "fg": "#041018",
                "disabled_bg": t.button_disabled,
                "disabled_fg": t.text_muted,
            }
        else:
            palette = {
                "bg": t.button_secondary,
                "hover": t.button_secondary_hover,
                "fg": t.text_secondary,
                "disabled_bg": t.button_disabled,
                "disabled_fg": t.text_muted,
            }
        key = str(button)
        self._button_palettes[key] = palette
        button.configure(
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=t.border_soft,
            activeforeground=palette["fg"],
            font=self.fonts["button"],
            padx=14,
            pady=10,
            cursor="hand2",
        )

        def on_enter(_event) -> None:
            if str(button.cget("state")) != "disabled":
                button.configure(bg=palette["hover"])

        def on_leave(_event) -> None:
            self._refresh_action_button(button)

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        self._refresh_action_button(button)

    def _refresh_action_button(self, button: tk.Button) -> None:
        palette = self._button_palettes.get(str(button))
        if not palette:
            return
        disabled = str(button.cget("state")) == "disabled"
        button.configure(
            bg=palette["disabled_bg"] if disabled else palette["bg"],
            fg=palette["disabled_fg"] if disabled else palette["fg"],
            activebackground=palette["disabled_bg"] if disabled else palette["hover"],
            activeforeground=palette["disabled_fg"] if disabled else palette["fg"],
            cursor="arrow" if disabled else "hand2",
        )

    def _set_action_button_enabled(self, button: tk.Button, enabled: bool) -> None:
        button.configure(state="normal" if enabled else "disabled")
        self._refresh_action_button(button)

    def _build_ui(self) -> None:
        t = self.theme
        s = self.spacing

        shell = tk.Frame(self.root, bg=t.bg_base, padx=s["outer_x"], pady=s["outer_y"])
        shell.pack(fill="both", expand=True)

        header = tk.Frame(shell, bg=t.bg_base)
        header.pack(fill="x", pady=(0, s["section_gap"]))
        brand = tk.Frame(header, bg=t.bg_base)
        brand.pack(side="left", fill="x", expand=True)
        tk.Label(brand, text="GestureKeys", font=self.fonts["display"], bg=t.bg_base, fg=t.text_primary).pack(anchor="w")
        tk.Label(
            brand,
            text="Precision gesture control with premium realtime telemetry",
            font=self.fonts["body"],
            bg=t.bg_base,
            fg=t.text_muted,
        ).pack(anchor="w", pady=(2, 0))
        chip = tk.Frame(
            header,
            bg=t.bg_card,
            highlightthickness=1,
            highlightbackground=t.border_soft,
            padx=14,
            pady=8,
        )
        chip.pack(side="right")
        tk.Label(chip, text="REALTIME SUITE", font=self.fonts["meta"], bg=t.bg_card, fg=t.accent).pack(anchor="center")
        tk.Label(chip, text="Desktop 2026 UI", font=self.fonts["meta"], bg=t.bg_card, fg=t.text_muted).pack(anchor="center")

        cols = tk.Frame(shell, bg=t.bg_base)
        cols.pack(fill="both", expand=True)

        right = tk.Frame(cols, bg=t.bg_base, width=372)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        left_card, left_body = self._create_card(cols, background=t.bg_shell, padding=(20, 18))
        left_card.pack(side="left", fill="both", expand=True, padx=(0, s["column_gap"]))
        preview_header = tk.Frame(left_body, bg=t.bg_shell)
        preview_header.pack(fill="x", pady=(0, 12))
        preview_header_text = tk.Frame(preview_header, bg=t.bg_shell)
        preview_header_text.pack(side="left", fill="x", expand=True)
        tk.Label(
            preview_header_text,
            text="Live Camera Preview",
            font=self.fonts["title"],
            bg=t.bg_shell,
            fg=t.text_primary,
        ).pack(anchor="w")
        tk.Label(
            preview_header_text,
            text="Hero view for hand, face, and pen tracking feedback",
            font=self.fonts["body"],
            bg=t.bg_shell,
            fg=t.text_muted,
        ).pack(anchor="w", pady=(2, 0))
        tk.Label(
            preview_header,
            text="33ms refresh",
            font=self.fonts["meta"],
            bg=t.bg_card_alt,
            fg=t.accent,
            padx=10,
            pady=5,
        ).pack(side="right")

        preview_shell = tk.Frame(
            left_body,
            bg=t.bg_preview,
            highlightthickness=1,
            highlightbackground=t.border_strong,
            bd=0,
        )
        preview_shell.pack(fill="both", expand=True)
        preview_meta = tk.Frame(preview_shell, bg=t.bg_preview, padx=12, pady=10)
        preview_meta.pack(fill="x")
        tk.Label(
            preview_meta,
            text="INPUT: RGB stream + MediaPipe landmarks",
            font=self.fonts["meta"],
            bg=t.bg_preview,
            fg=t.text_muted,
        ).pack(side="left")
        tk.Label(
            preview_meta,
            text="Mirror toggle available",
            font=self.fonts["meta"],
            bg=t.bg_preview,
            fg=t.text_secondary,
        ).pack(side="right")

        preview_stage = tk.Frame(
            preview_shell,
            bg="#040911",
            highlightthickness=1,
            highlightbackground=t.border_soft,
            padx=6,
            pady=6,
        )
        preview_stage.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.preview_font = self.fonts["preview"]
        self.cam_canvas = tk.Label(
            preview_stage,
            bg="#02060b",
            fg=t.text_muted,
            justify="center",
            font=self.preview_font,
            compound="center",
        )
        self.cam_canvas.pack(fill="both", expand=True)

        stat_card, stat = self._create_card(right)
        stat_card.pack(fill="x", pady=(0, s["card_gap"]))
        self._add_card_header(stat, "Detected state", "Live gesture and action telemetry")
        self.gesture_var = tk.StringVar(value="R: -\nL: -\nM: -\nP: -")
        gesture_frame = tk.Frame(
            stat,
            bg=t.bg_card_alt,
            highlightthickness=1,
            highlightbackground=t.border_soft,
            padx=12,
            pady=10,
        )
        gesture_frame.pack(fill="x")
        tk.Label(
            gesture_frame,
            textvariable=self.gesture_var,
            font=self.fonts["mono"],
            bg=t.bg_card_alt,
            fg=t.accent,
            justify="left",
            wraplength=320,
        ).pack(anchor="w")
        metric_row = tk.Frame(stat, bg=t.bg_card)
        metric_row.pack(fill="x", pady=(10, 0))
        self.key_var = tk.StringVar(value="-")
        key_tile = self._create_metric_tile(metric_row, "Last key", self.key_var)
        key_tile.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.count_var = tk.StringVar(value="0")
        count_tile = self._create_metric_tile(metric_row, "Keys triggered", self.count_var)
        count_tile.pack(side="left", fill="both", expand=True)

        ctrl_card, ctrl = self._create_card(right)
        ctrl_card.pack(fill="x", pady=(0, s["card_gap"]))
        self._add_card_header(ctrl, "Settings", "Tune responsiveness and pointer behavior")

        tk.Label(ctrl, text="Camera source", font=self.fonts["meta"], bg=t.bg_card, fg=t.text_muted).pack(anchor="w")
        self.camera_var = tk.StringVar(value="Auto" if self.default_camera is None else str(self.default_camera))
        ttk.Combobox(
            ctrl,
            textvariable=self.camera_var,
            state="readonly",
            values=["Auto", "0", "1", "2", "3", "4"],
            style="GK.TCombobox",
        ).pack(fill="x", pady=(3, 10))

        cooldown_header = tk.Frame(ctrl, bg=t.bg_card)
        cooldown_header.pack(fill="x")
        tk.Label(
            cooldown_header,
            text="Right-hand cooldown",
            font=self.fonts["meta"],
            bg=t.bg_card,
            fg=t.text_muted,
        ).pack(side="left")
        self.cooldown_text = tk.StringVar(value=f"{self.cooldown:.2f}s")
        tk.Label(
            cooldown_header,
            textvariable=self.cooldown_text,
            font=self.fonts["meta"],
            bg=t.bg_card,
            fg=t.text_secondary,
        ).pack(side="right")
        self.cooldown_var = tk.DoubleVar(value=self.cooldown)
        ttk.Scale(
            ctrl,
            from_=0.2,
            to=2.0,
            variable=self.cooldown_var,
            orient="horizontal",
            style="GK.Horizontal.TScale",
            command=self._update_cooldown,
        ).pack(fill="x", pady=(3, 10))

        hold_header = tk.Frame(ctrl, bg=t.bg_card)
        hold_header.pack(fill="x")
        tk.Label(
            hold_header,
            text="Right-hand hold to fire",
            font=self.fonts["meta"],
            bg=t.bg_card,
            fg=t.text_muted,
        ).pack(side="left")
        self.hold_text = tk.StringVar(value=f"{self.hold_threshold:.2f}s")
        tk.Label(
            hold_header,
            textvariable=self.hold_text,
            font=self.fonts["meta"],
            bg=t.bg_card,
            fg=t.text_secondary,
        ).pack(side="right")
        self.hold_var = tk.DoubleVar(value=self.hold_threshold)
        ttk.Scale(
            ctrl,
            from_=0.1,
            to=1.5,
            variable=self.hold_var,
            orient="horizontal",
            style="GK.Horizontal.TScale",
            command=self._update_hold_threshold,
        ).pack(fill="x", pady=(3, 10))

        pen_header = tk.Frame(ctrl, bg=t.bg_card)
        pen_header.pack(fill="x")
        tk.Label(
            pen_header,
            text="Green-pen sensitivity",
            font=self.fonts["meta"],
            bg=t.bg_card,
            fg=t.text_muted,
        ).pack(side="left")
        self.pen_sensitivity_text = tk.StringVar(value=f"{int(round(self.pen_sensitivity))}%")
        tk.Label(
            pen_header,
            textvariable=self.pen_sensitivity_text,
            font=self.fonts["meta"],
            bg=t.bg_card,
            fg=t.text_secondary,
        ).pack(side="right")
        self.pen_sensitivity_var = tk.DoubleVar(value=self.pen_sensitivity)
        ttk.Scale(
            ctrl,
            from_=0,
            to=100,
            variable=self.pen_sensitivity_var,
            orient="horizontal",
            style="GK.Horizontal.TScale",
            command=self._update_pen_sensitivity,
        ).pack(fill="x", pady=(3, 10))

        self.enabled_var = tk.BooleanVar(value=self.enabled)
        ttk.Checkbutton(
            ctrl,
            text="Enable right-hand key output",
            variable=self.enabled_var,
            style="GK.TCheckbutton",
            command=lambda: setattr(self, "enabled", self.enabled_var.get()),
        ).pack(anchor="w", pady=(0, 2))
        self.mirror_var = tk.BooleanVar(value=self.mirror_preview)
        ttk.Checkbutton(
            ctrl,
            text="Mirror preview",
            variable=self.mirror_var,
            style="GK.TCheckbutton",
            command=lambda: setattr(self, "mirror_preview", self.mirror_var.get()),
        ).pack(anchor="w")

        actions_card, actions = self._create_card(right)
        actions_card.pack(fill="x", pady=(0, s["card_gap"]))
        self._add_card_header(actions, "Session controls", "Operate the camera loop safely")
        self.start_btn = tk.Button(actions, text="START SESSION", command=self.start_camera)
        self.start_btn.pack(fill="x", pady=(0, 7))
        self.stop_btn = tk.Button(actions, text="STOP SESSION", command=self.stop_camera)
        self.stop_btn.pack(fill="x")
        self._register_action_button(self.start_btn, role="primary")
        self._register_action_button(self.stop_btn, role="secondary")
        self._set_action_button_enabled(self.start_btn, True)
        self._set_action_button_enabled(self.stop_btn, False)

        ref_card, ref = self._create_card(right)
        ref_card.pack(fill="x")
        self._add_card_header(ref, "Gesture map", "Control surfaces active in this build")
        for name, action in [
            ("Right hand", "Keyboard map"),
            ("Right green pen", "Mouse pointer"),
            ("Left pinch + pen", "Left click / drag"),
            ("Left pinch no pen", "Right click"),
            ("Left pinky", "Volume"),
            ("Left fist/open", "Brightness"),
            ("Mouth (5s)", "Alt+F4"),
        ]:
            row = tk.Frame(ref, bg=t.bg_card)
            row.pack(fill="x", pady=1)
            tk.Label(
                row,
                text=name,
                font=self.fonts["meta"],
                bg=t.bg_card,
                fg=t.text_secondary,
                anchor="w",
            ).pack(side="left")
            tk.Label(
                row,
                text=action,
                font=self.fonts["meta"],
                bg=t.bg_card,
                fg=t.accent_soft,
                anchor="e",
            ).pack(side="right")

        self.status_var = tk.StringVar(value="Loading gesture model...")
        status_shell = tk.Frame(
            self.root,
            bg=t.bg_status,
            highlightthickness=1,
            highlightbackground=t.border_soft,
            bd=0,
            padx=12,
            pady=7,
        )
        status_shell.pack(fill="x", side="bottom", padx=s["outer_x"], pady=(0, s["outer_y"]))
        tk.Label(
            status_shell,
            text="SYSTEM",
            font=self.fonts["meta"],
            bg=t.bg_status,
            fg=t.text_muted,
        ).pack(side="left", padx=(2, 10))
        tk.Label(
            status_shell,
            textvariable=self.status_var,
            font=self.fonts["status"],
            bg=t.bg_status,
            fg=t.text_secondary,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        self._set_preview_message("Loading model...", "GestureKeys is preparing detectors.")

    def _init_landmarker(self) -> None:
        def callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
            with self.result_lock:
                self.latest_result = result

        face_model_ready = ensure_models()
        self.landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
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
            self.pending_status_message = "Face model unavailable. Mouth-close is disabled."
        self.status_var.set("Model ready. Click START to begin.")
        self._set_preview_message("Camera idle", "Click START to open your webcam.")

    def _set_preview_message(self, title: str, detail: Optional[str] = None) -> None:
        message = title if not detail else f"{title}\n\n{detail}"
        self.cam_canvas.config(
            image="",
            text=message,
            anchor="center",
            padx=24,
            pady=24,
            wraplength=760,
            justify="center",
        )
        self.cam_canvas.image = None

    def _update_cooldown(self, value: str) -> None:
        try:
            numeric = float(value)
        except Exception:
            return
        self.cooldown = clamp(numeric, 0.2, 2.0)
        if hasattr(self, "cooldown_text"):
            self.cooldown_text.set(f"{self.cooldown:.2f}s")

    def _update_hold_threshold(self, value: str) -> None:
        try:
            numeric = float(value)
        except Exception:
            return
        self.hold_threshold = clamp(numeric, 0.1, 1.5)
        if hasattr(self, "hold_text"):
            self.hold_text.set(f"{self.hold_threshold:.2f}s")

    def _update_pen_sensitivity(self, value: str) -> None:
        try:
            numeric = float(value)
        except Exception:
            return
        self.pen_sensitivity = clamp(numeric, 0.0, 100.0)
        self.red_pen.set_sensitivity(self.pen_sensitivity)
        if hasattr(self, "pen_sensitivity_text"):
            self.pen_sensitivity_text.set(f"{int(round(self.pen_sensitivity))}%")

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
            width = max(self.cam_canvas.winfo_width(), 640)
            height = max(self.cam_canvas.winfo_height(), 420)
            width = min(width, 1440)
            height = min(height, 900)
            image = Image.fromarray(state.frame_rgb).resize((width, height), Image.LANCZOS)
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
        for index, landmarks in enumerate(result.hand_landmarks):
            label = None
            if result.handedness and index < len(result.handedness) and result.handedness[index]:
                label = getattr(result.handedness[index][0], "category_name", None)
                if label:
                    label = label.lower()
            if label == "right" and right is None:
                right = landmarks
            elif label == "left" and left is None:
                left = landmarks
            else:
                unknown.append(landmarks)
        for landmarks in unknown:
            if right is None:
                right = landmarks
            elif left is None:
                left = landmarks
        return right, left

    def _move_cursor(self, point: tuple[int, int]) -> bool:
        try:
            pyautogui.moveTo(point[0], point[1], duration=0)
            return True
        except Exception as exc:
            self.pending_status_message = f"Cursor move error: {exc}"
            return False

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

    def _stop_left_controls(self) -> None:
        self._release_drag()
        self.left_mode = "idle"
        self.left_anchor_y = None
        self.left_base_value = None
        self.left_last_value = None
        self.left_pinch_start = None
        self.left_pinch_mode = None
        self.left_pinch_target = None
        self.left_right_click_fired = False
        self.brightness_arm_start = None
        self.brightness_exit_gesture_start = None

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
            self.left_pinch_start = None
        self.brightness_arm_start = None
        openness = hand_openness_ratio(landmarks)
        target = int(round(1 + openness * 99))
        if self.left_last_value is None or abs(target - self.left_last_value) >= 1:
            if self.brightness.set_percent(target):
                self.left_last_value = target
        shown = self.left_last_value if self.left_last_value is not None else target
        return f"brightness {shown}%"

    def _finalize_left_pinch(self, now: float) -> None:
        if self.left_pinch_start is None:
            return
        elapsed = now - self.left_pinch_start
        if self.left_dragging:
            self._release_drag()
        elif self.left_pinch_mode == "left" and elapsed < self.left_click_drag_threshold:
            try:
                if self.left_pinch_target is not None:
                    self._move_cursor(self.left_pinch_target)
                pyautogui.click(button="left")
            except Exception as exc:
                self.pending_status_message = f"Left click error: {exc}"

        self.left_pinch_start = None
        self.left_pinch_mode = None
        self.left_pinch_target = None
        self.left_right_click_fired = False
        if self.left_mode.startswith("pinch"):
            self.left_mode = "idle"

    def _handle_left_pinch_with_pen(self, pen_target: Optional[tuple[int, int]], now: float) -> str:
        if self.left_pinch_start is None:
            self.left_pinch_start = now
            self.left_pinch_mode = "left"
            self.left_mode = "pinch_ready"
            self.left_right_click_fired = False
        if pen_target is not None:
            self.left_pinch_target = pen_target
            self._move_cursor(pen_target)
        elapsed = now - self.left_pinch_start
        if elapsed >= self.left_click_drag_threshold:
            if not self.left_dragging:
                try:
                    pyautogui.mouseDown(button="left")
                    self.left_dragging = True
                    self.left_mode = "pinch_drag"
                except Exception as exc:
                    self.pending_status_message = f"Drag start error: {exc}"
            if self.left_dragging and pen_target is not None:
                self._move_cursor(pen_target)
            return "drag"
        return f"left pinch {elapsed:.2f}s"

    def _handle_left_pinch_without_pen(self, now: float) -> str:
        if self.left_pinch_start is None:
            self.left_pinch_start = now
            self.left_pinch_mode = "right"
            self.left_mode = "pinch_right"
            self.left_right_click_fired = False
        if not self.left_right_click_fired:
            try:
                pyautogui.click(button="right")
                self.left_right_click_fired = True
            except Exception as exc:
                self.pending_status_message = f"Right click error: {exc}"
        elapsed = now - self.left_pinch_start
        return f"right click {elapsed:.2f}s"

    def _handle_left(self, landmarks, pen_target: Optional[tuple[int, int]], pen_active: bool, now: float) -> str:
        if landmarks is None:
            self._finalize_left_pinch(now)
            self._stop_left_controls()
            return "idle"

        thumb, index, middle, ring, pinky = fingers_up(landmarks)
        count = sum((thumb, index, middle, ring, pinky))
        fist = count == 0
        pinch = pinch_ratio(landmarks) < 0.32
        pinky_only = (not thumb) and (not index) and (not middle) and (not ring) and pinky

        if pinky_only:
            self._finalize_left_pinch(now)
            if self.left_mode == "brightness_fist":
                self.left_mode = "idle"
            self.brightness_arm_start = None
            self.brightness_exit_gesture_start = None
            text = self._handle_slide("volume", landmarks[0].y, self.volume.get_percent, self.volume.set_percent)
            if text == "volume unavailable" and not self.warned_volume_unavailable:
                detail = f" ({self.volume.error})" if self.volume.error else ""
                self.pending_status_message = f"Volume control unavailable on this device/environment{detail}."
                self.warned_volume_unavailable = True
            return text

        if self.left_mode == "brightness_fist":
            self._finalize_left_pinch(now)
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
                    return "brightness released"
                return f"brightness exit arm {elapsed:.2f}/{self.brightness_exit_hold:.2f}s"
            self.brightness_exit_gesture_start = None
            return self._handle_fist_brightness(landmarks)

        if fist:
            self._finalize_left_pinch(now)
            if self.brightness_arm_start is None:
                self.brightness_arm_start = now
            elapsed = now - self.brightness_arm_start
            if elapsed >= self.brightness_arm_hold:
                return self._handle_fist_brightness(landmarks)
            return f"brightness arm {elapsed:.2f}/{self.brightness_arm_hold:.2f}s"

        self.brightness_arm_start = None
        self.brightness_exit_gesture_start = None

        if pinch:
            if self.left_pinch_start is None:
                self.left_pinch_start = now
                self.left_pinch_mode = "left" if pen_active else "right"
                self.left_right_click_fired = False
                self.left_mode = "pinch_ready" if self.left_pinch_mode == "left" else "pinch_right"
            if self.left_pinch_mode == "left":
                return self._handle_left_pinch_with_pen(pen_target, now)
            return self._handle_left_pinch_without_pen(now)

        self._finalize_left_pinch(now)
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
                try:
                    pyautogui.hotkey("alt", "f4")
                    self.pending_status_message = "Mouth open 5s detected: Alt+F4 sent."
                except Exception as exc:
                    self.pending_status_message = f"Window close error: {exc}"
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
        self.red_pen.reset()
        self.pen_state = "no-right-hand"
        self.pen_target = None
        self.pen_frame_point = None
        self.pen_roi = None

        self.mouth_open_start = None
        self.mouth_triggered = False
        self.pending_status_message = None

        self._set_action_button_enabled(self.start_btn, False)
        self._set_action_button_enabled(self.stop_btn, True)
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
        self.red_pen.reset()
        self.pen_state = "no-right-hand"
        self.pen_target = None
        self.pen_frame_point = None
        self.pen_roi = None
        self._set_action_button_enabled(self.start_btn, True)
        self._set_action_button_enabled(self.stop_btn, False)
        self._set_preview_message("Camera stopped", "Click START to open the webcam again.")
        self.gesture_var.set("R: -\nL: -\nM: -\nP: -")
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
            pen_target, pen_state, pen_frame_point, pen_roi = self.red_pen.update(frame, right_hand, now)
            self.pen_target = pen_target
            self.pen_state = pen_state
            self.pen_frame_point = pen_frame_point
            self.pen_roi = pen_roi

            pen_active = pen_target is not None
            if pen_active:
                self._move_cursor(pen_target)

            if pen_roi is not None:
                x1, y1, x2, y2 = pen_roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 200), 1)
            if pen_frame_point is not None:
                cv2.circle(frame, pen_frame_point, 12, (0, 0, 255), 2)
                cv2.circle(frame, pen_frame_point, 4, (0, 0, 255), -1)

            left_text = self._handle_left(left_hand, pen_target, pen_active, now)
            left_priority = left_text != "idle"

            if pen_active or left_priority:
                self._handle_right(None, now)
                right_text = "suppressed" if right_hand is not None else "idle"
            else:
                right_text = self._handle_right(right_hand, now)

            mouth_text = self._handle_mouth(face_results, now)

            cv2.putText(frame, f"R: {right_text}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 224), 2)
            cv2.putText(frame, f"L: {left_text}", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
            cv2.putText(frame, f"M: {mouth_text}", (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 2)
            cv2.putText(frame, f"P: {pen_state}", (14, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 160), 2)

            if self.right_hold_progress > 0 and self.prev_right_gesture:
                width = int(frame.shape[1] * self.right_hold_progress)
                cv2.rectangle(frame, (0, frame.shape[0] - 8), (width, frame.shape[0]), (0, 255, 180), -1)

            ui_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            status = self.pending_status_message
            self.pending_status_message = None
            self._publish_ui_state(
                ui_frame,
                f"R: {right_text}\nL: {left_text}\nM: {mouth_text}\nP: {pen_state}",
                status,
            )

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
    parser = argparse.ArgumentParser(description="Map webcam gestures and green-pen tracking to keyboard/mouse control.")
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
