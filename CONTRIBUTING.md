# Contributing to GestureKeys

Thanks for your interest in improving GestureKeys! This is a Windows-first
hobby/experimental project, so most testing happens locally on a machine with
a webcam. The notes below keep contributions smooth.

## Development setup

GestureKeys has a Python backend and a React + Tauri frontend. From the repo
root (the `gesturekeys` folder):

```powershell
# 1. Python backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # lint + tests

# 2. Frontend (Tauri desktop UI)
cd .\ui-tauri
npm install
cd ..
```

Run the full app:

```powershell
.\run_gesturekeys.ps1 -Mode desktop          # first run installs deps
.\run_gesturekeys.ps1 -Mode desktop -SkipInstall   # later runs
```

## Before you open a PR

Run these from the repo root:

```powershell
.\.venv\Scripts\python.exe -m ruff check .     # lint
.\.venv\Scripts\python.exe -m pytest           # hardware-free smoke tests
```

If you changed the frontend:

```powershell
cd .\ui-tauri
npm run build
```

CI runs the same Python lint/test and the frontend build on every push and PR.

## Code style

- **Python** is linted with [Ruff](https://docs.astral.sh/ruff/) (config in
  [`pyproject.toml`](pyproject.toml)). A handful of rules are temporarily
  disabled as a *baseline* for the existing code — they're listed with
  comments in `pyproject.toml`. When you're cleaning a module, feel free to
  remove one of those ignores and apply the fixes:

  ```powershell
  .\.venv\Scripts\python.exe -m ruff check . --fix
  .\.venv\Scripts\python.exe -m ruff format .
  ```

  Keep formatting changes in separate commits from logic changes so diffs
  stay readable.
- **TypeScript/React** lives in `ui-tauri/src`. Match the existing style; the
  build runs `tsc` in strict mode.

## Project layout & gotchas

- `gesture_keyboard_v3.py` is **not** dead code. Besides being the legacy
  Tkinter UI, it exports shared helpers (`ensure_models`, `RedPenTracker`,
  `classify_right_hand`, …) that `gesture_engine.py` imports. Don't move or
  delete it without updating those imports.
- `gesture_keyboard_v2.py` is an older iteration with no importers. It's kept
  for reference only and is excluded from linting.
- MediaPipe model files (`*.task`) are **git-ignored** and download
  automatically on first run from Google's public model storage. Don't commit
  them. If you ever need to vendor them, set up Git LFS first
  (`git lfs install && git lfs track "*.task"`).
- Vision modules import `mediapipe`/`pyautogui` and expect a webcam, so they
  can't run in CI. Keep automated tests hardware-free (see `tests/`).

## Adding full Tauri CI later

CI currently builds only the web frontend (`npm run build`). Building the
Tauri Rust shell needs a Rust toolchain plus platform webkit/webcam libraries
and is best done locally on Windows. If you want it in CI, add a job using
[`tauri-apps/tauri-action`](https://github.com/tauri-apps/tauri-action) on a
`windows-latest` runner — expect longer, occasionally flaky builds.

## Commit messages

Short, imperative subject lines (e.g. "Add pinch sensitivity slider"). Group
unrelated changes into separate commits.
