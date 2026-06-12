"""Hardware-free smoke tests for GestureKeys.

These tests intentionally avoid importing the heavy vision modules
(``gesture_engine``, ``gesture_keyboard_v3``) because those pull in
``mediapipe``/``pyautogui`` and expect a webcam/display. They verify
repository invariants and that the lightweight launcher is importable, so
they can run in CI on any OS without project runtime dependencies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_launcher_imports_and_parses_args(monkeypatch):
    """The launcher has no third-party imports and should parse defaults."""
    monkeypatch.setattr(sys, "argv", ["gesture_keyboard.py"])
    import gesture_keyboard

    args = gesture_keyboard.parse_args()
    assert args.mode == "desktop"
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_launcher_accepts_known_modes(monkeypatch):
    import gesture_keyboard

    for mode in ("desktop", "api", "tk"):
        monkeypatch.setattr(sys, "argv", ["gesture_keyboard.py", "--mode", mode])
        assert gesture_keyboard.parse_args().mode == mode


def test_requirements_are_listed():
    reqs = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    listed = {line.strip() for line in reqs.splitlines() if line.strip()}
    for pkg in ("mediapipe", "opencv-python", "fastapi", "uvicorn"):
        assert pkg in listed, f"expected {pkg} in requirements.txt"


def test_example_calibration_is_valid_json():
    data = json.loads((REPO_ROOT / "gaze_calibration.example.json").read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert len(data["coeff_x"]) == 7
    assert len(data["coeff_y"]) == 7


def test_model_download_urls_use_https():
    """Model files auto-download; make sure the URLs stay HTTPS."""
    for source in ("gesture_engine.py", "gesture_keyboard_v3.py"):
        text = (REPO_ROOT / source).read_text(encoding="utf-8")
        if "storage.googleapis.com" in text:
            assert "https://storage.googleapis.com" in text
            assert "http://storage.googleapis.com" not in text
