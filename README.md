# GestureKeys

> Control your Windows PC with your hands. GestureKeys is a webcam-based,
> gesture-driven input system that turns hand poses, face cues, and a simple
> green marker into keyboard keys, mouse movement, clicks, volume, brightness,
> and window controls — all processed locally, in real time.

GestureKeys pairs a Python computer-vision engine (built on Google
[MediaPipe](https://ai.google.dev/edge/mediapipe)) with a polished
React + [Tauri](https://tauri.app/) desktop app. Hold a green object to drive
the pointer, pinch to click, and use right-hand poses for keyboard actions.
A built-in Calibration Wizard tunes the experience to your camera and hands.

![CI](https://github.com/madmalli05/gesturekeys/actions/workflows/ci.yml/badge.svg)

> ⚠️ **Status:** Experimental, Windows-first hobby project under active
> development. Expect rough edges and changing gestures.

---

## Table of contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Screenshots](#screenshots)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Setup (Windows)](#setup-windows)
- [Running the app](#running-the-app)
- [Gesture reference](#gesture-reference)
- [Configuration & calibration](#configuration--calibration)
- [Backend API surface](#backend-api-surface)
- [Troubleshooting](#troubleshooting)
- [Development roadmap](#development-roadmap)
- [Known limitations](#known-limitations)
- [Privacy & safety](#privacy--safety)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- 🖐️ **Hand-gesture keyboard** — right-hand poses map to keys (Space, arrows,
  Enter, Esc, copy/paste, …).
- 🟢 **Green-marker pointer** — hold a green object to move the mouse; left-hand
  pinches perform click, click-drag, and right-click.
- 🔊 **System control** — adjust master volume and screen brightness with
  dedicated gestures.
- 👄 **Optional face action** — hold mouth open to trigger `Alt+F4` (toggleable).
- 🎛️ **Calibration Wizard** — named profiles with auto-derived pinch and
  smoothing values for your camera/hands.
- ⚡ **Click speed tuner** — `balanced`, `fast`, `turbo`, and `ultra` profiles.
- 🖥️ **Premium desktop UI** — React + Tauri app with a live preview, settings,
  and session control; a legacy Tkinter fallback is included.
- 🔌 **Local FastAPI backend** — WebSocket state + MJPEG preview, usable
  headlessly via `--mode api`.
- 🔒 **Fully local** — all video is processed on-device; nothing is uploaded.

## Tech stack

| Layer        | Technology |
| ------------ | ---------- |
| Vision/ML    | MediaPipe (hand, face, gesture recognizer), OpenCV |
| Backend      | Python 3.10+, FastAPI, Uvicorn, WebSockets |
| System I/O   | PyAutoGUI, pycaw (audio), screen-brightness-control |
| Desktop UI   | React 18, TypeScript, Vite, Tauri 2 (Rust) |
| Legacy UI    | Tkinter (`gesture_keyboard_v3.py`) |
| Tooling      | Ruff, pytest, GitHub Actions |

## Screenshots

> _Add screenshots/GIFs here once you have them._ Place images under
> `docs/screenshots/` and reference them like the example below.

| Desktop UI | Live tracking |
| ---------- | ------------- |
| _coming soon_ | _coming soon_ |

<!--
![GestureKeys desktop UI](docs/screenshots/ui.png)
![Live tracking](docs/screenshots/tracking.gif)
-->

## Repository structure

```
gesturekeys/
├─ gesture_keyboard.py        # Launcher / entry point (--mode desktop|api|tk)
├─ gesture_api_server.py      # Uvicorn entry that serves the FastAPI app
├─ gesture_backend_api.py     # FastAPI app: sessions, settings, calibration, WS
├─ gesture_engine.py          # Core vision + control engine
├─ gesture_keyboard_v3.py     # Legacy Tkinter UI + shared helpers (imported!)
├─ gesture_keyboard_v2.py     # Older iteration, kept for reference (unused)
├─ requirements.txt           # Runtime dependencies
├─ requirements-dev.txt       # Lint + test tooling
├─ run_gesturekeys.ps1        # Convenience launcher (PowerShell)
├─ gaze_calibration.example.json  # Template; real file is git-ignored
├─ pyproject.toml             # Ruff + pytest configuration
├─ tests/                     # Hardware-free smoke tests
├─ ui-tauri/                  # React + Tauri desktop app
│  ├─ src/                    # React/TypeScript frontend
│  └─ src-tauri/              # Rust/Tauri shell
└─ .github/                   # CI workflow, issue/PR templates
```

> **Note:** MediaPipe model files (`hand_landmarker.task`,
> `face_landmarker.task`, `gesture_recognizer.task`) are **not** committed —
> they download automatically on first run.

## Requirements

- **Windows 10/11** (system control uses Windows APIs)
- **Python 3.10+**
- **Node.js 20+** (for the Tauri desktop UI)
- **Rust toolchain** (for Tauri dev/build — see <https://rustup.rs>)
- A **webcam**
- A **green object/marker** for pointer mode (a green pen, cap, sticky note, …)

## Setup (Windows)

Run these once from the `gesturekeys` folder.

### 1. Backend (Python)

```powershell
# Create a virtual environment
python -m venv .venv

# Activate it (PowerShell)
.\.venv\Scripts\Activate.ps1

# Install runtime dependencies
python -m pip install -r requirements.txt
```

> Prefer not to activate? Call the venv Python directly, e.g.
> `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

On first launch, the MediaPipe `*.task` models download automatically.

### 2. Frontend (Tauri desktop UI)

```powershell
cd .\ui-tauri
npm install
cd ..
```

## Running the app

### Full desktop app (recommended)

The launcher starts the backend and the Tauri UI together:

```powershell
# First run (also installs Python deps)
.\run_gesturekeys.ps1 -Mode desktop

# Subsequent runs (skip the pip install step)
.\run_gesturekeys.ps1 -Mode desktop -SkipInstall
```

Then click **Start Session** in the app window.

### Backend only (headless API)

```powershell
.\.venv\Scripts\python.exe .\gesture_keyboard.py --mode api
```

### Legacy Tkinter UI (fallback)

```powershell
.\.venv\Scripts\python.exe .\gesture_keyboard.py --mode tk
```

### Useful flags

```powershell
# Choose a specific camera index
.\.venv\Scripts\python.exe .\gesture_keyboard.py --mode desktop --camera 0

# Observe tracking without sending any keyboard/mouse input
.\.venv\Scripts\python.exe .\gesture_keyboard.py --mode desktop --disable-keys
```

Run `python gesture_keyboard.py --help` for the full list.

## Gesture reference

### Right hand → keyboard

| Gesture | Action |
| --- | --- |
| Fist | `Space` |
| Index finger up | `Up` |
| Victory | `Down` |
| Three fingers up | `Left` |
| Open hand | `Right` |
| Thumb up | `Enter` |
| Shaka | `Escape` |
| Pinch | `Ctrl+C` |
| Thumb + index + middle up | `Ctrl+V` |

### Pointer + left hand

| Gesture | Action |
| --- | --- |
| Green marker near right hand | Move mouse pointer |
| Green marker + left thumb-index pinch (short) | Left click |
| Green marker + left thumb-index pinch (hold/move) | Left click-drag |
| **Green marker + left thumb-middle pinch (release)** | **Right click** |
| Left pinky only + slide up/down | Master volume |
| Hold fist ~0.22s to arm, then open/close hand | Brightness |
| In brightness mode, hold thumb+pinky ~0.25s | Exit brightness mode |
| Mouth open for 5s (if enabled) | `Alt+F4` (one-shot until closed) |

## Configuration & calibration

- **Settings** (in the UI): toggle right-hand key output, mirror the preview,
  enable/disable the mouth action, and pick a click-speed profile.
- **Calibration Wizard:** create named profiles with auto-derived pinch and
  smoothing values. Profiles persist at:
  - `%APPDATA%\GestureKeys\calibration_profiles_v1.json`
  - fallback: a local `calibration_profiles_v1.json` if `%APPDATA%` is
    unavailable
- **Gaze calibration:** `gaze_calibration.json` is machine-specific and
  git-ignored. Copy `gaze_calibration.example.json` to `gaze_calibration.json`
  if you need a starting point.

Environment variables are **not** required today — runtime options are passed
as CLI flags. See `.env.example` for the convention if you add config later.

## Backend API surface

When running in `api`/`desktop` mode the FastAPI server (default
`http://127.0.0.1:8000`) exposes:

```
POST   /api/session/start
POST   /api/session/stop
GET    /api/session/state
GET    /api/settings
PATCH  /api/settings
GET    /api/calibration/profiles
POST   /api/calibration/profiles
POST   /api/calibration/profiles/{id}/activate
DELETE /api/calibration/profiles/{id}
POST   /api/calibration/session/start
GET    /api/calibration/session/state
POST   /api/calibration/session/capture
POST   /api/calibration/session/derive
POST   /api/calibration/session/apply
POST   /api/calibration/session/cancel
POST   /api/calibration/session/rollback
WS     /api/ws/state
GET    /api/preview.mjpg
```

## Troubleshooting

- **Camera light is on but tracking doesn't work:** click **Stop Session**,
  wait 1–2 seconds, then **Start Session**. Keep `Camera source` on
  `Auto (Laptop Default)` or try `Camera 0`. Close other apps that may own the
  camera.
- **WebSocket warnings/errors:** reinstall Python deps with
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- **PowerShell blocks `npm`:** use `npm.cmd` instead of `npm`.
- **Brightness control unavailable:** some monitors/drivers don't expose a
  compatible brightness API.
- **Models won't download:** check your internet connection, or manually place
  the `*.task` files in the repo root (URLs are in `gesture_engine.py` and
  `gesture_keyboard_v3.py`).
- **`Activate.ps1` is blocked:** run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or just call the venv
  Python directly (`.\.venv\Scripts\python.exe …`).

## Development roadmap

These are directional ideas, not commitments:

- [ ] Add screenshots/GIFs and a short demo video.
- [ ] User-configurable gesture → action mapping.
- [ ] Reduce pointer jitter and improve pinch reliability across lighting.
- [ ] Persist the mouth-action toggle across launches.
- [ ] Package a signed Windows installer via Tauri bundling.
- [ ] Incrementally clean up legacy modules and remove the lint baseline
      ignores in `pyproject.toml`.
- [ ] Optional cross-platform support (macOS/Linux input backends).

See [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.

## Known limitations

- **Windows only.** System control relies on Windows-specific APIs (`pycaw`,
  `screen-brightness-control`).
- **Lighting/background sensitive.** Green-marker tracking and hand detection
  degrade in poor lighting or with green clutter in frame.
- **One camera at a time.** The camera must not be held by another app.
- **No API authentication.** The backend is meant for localhost use only.
- **Brightness support varies** by monitor/driver.

## Privacy & safety

GestureKeys works with a **live webcam feed** and can **control your input
devices**, so privacy and safety matter:

- **Everything runs locally.** Frames are processed on-device by MediaPipe; no
  video, images, or landmarks leave your machine.
- **The backend binds to `127.0.0.1`** by default — not reachable from the
  network unless you change it.
- **Calibration files stay local** and are git-ignored.
- The app sends real keyboard/mouse/volume/brightness input and can close
  windows. Run it only from sources you trust, and use `--disable-keys` to
  watch tracking without it acting on your system.

If you ever expose the backend beyond localhost, you are streaming your webcam
over the network with no authentication — see [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for
setup, the lint/test workflow, and project gotchas (notably: don't move
`gesture_keyboard_v3.py`, and don't commit `*.task` models). Bug reports and
feature ideas can use the issue templates.

## License

Released under the [MIT License](LICENSE). © 2026 Ahmad Malli.
