# GestureKeys: notes for Claude Code sessions

Windows-first webcam system that turns hand gestures into keyboard, mouse, volume, and brightness input. Python MediaPipe backend plus a React/Tauri desktop UI. User-facing docs are in `README.md`; dev setup and PR checks are in `CONTRIBUTING.md`.

## Running

- Desktop app (backend + Tauri dev window): `.\run_gesturekeys.ps1 -Mode desktop` (add `-SkipInstall` after the first run)
- Headless backend on 127.0.0.1:8000: `.\.venv\Scripts\python.exe .\gesture_keyboard.py --mode api`
- Legacy Tkinter UI: `--mode tk`

Running the engine sends real keyboard, mouse, and system input to the Windows desktop. In an agent session, pass `--disable-keys` unless the user explicitly wants live input.

## Setup (local only, not in git)

- Python 3.10+ venv at `.venv`, then `.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt`
- `cd ui-tauri` then `npm install`
- Rust toolchain for Tauri builds
- MediaPipe `*.task` models download into the repo root on first run

## Checks before committing

- `.\.venv\Scripts\python.exe -m ruff check .`
- `.\.venv\Scripts\python.exe -m pytest` (hardware-free smoke tests only)
- Frontend changes: `cd ui-tauri` then `npm run build`

Vision code needs a webcam and cannot run in CI or in a sandbox.

## Layout

- `gesture_keyboard.py`: launcher, entry point for all modes
- `gesture_api_server.py`, `gesture_backend_api.py`: FastAPI app with sessions, settings, calibration, WebSocket state, MJPEG preview
- `gesture_engine.py`: core vision and control engine (largest file)
- `gesture_keyboard_v3.py`: legacy Tkinter UI, but `gesture_engine.py` imports shared helpers from it (`ensure_models`, `RedPenTracker`, `classify_right_hand`). Do not move or delete it without updating those imports.
- `gesture_keyboard_v2.py`: unused, excluded from lint. Ignore it.
- `ui-tauri/src`: React and TypeScript frontend. `ui-tauri/src-tauri`: Rust shell.

## Gotchas

- Never commit: `*.task` models, `gaze_calibration.json` and `calibration_profiles_v1.json` (personal calibration), `.env`, `.venv`, `node_modules`, `src-tauri/target`.
- The Ruff ignore list in `pyproject.toml` is an intentional baseline. Do not remove entries in unrelated changes.
- Keep formatting-only changes in separate commits from logic changes. Use short imperative commit subjects.
- Line endings are set in `.gitattributes`: `.ps1` is CRLF, `.sh` is LF.
