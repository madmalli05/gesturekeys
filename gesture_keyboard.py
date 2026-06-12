from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
UI_TAURI_DIR = ROOT / "ui-tauri"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GestureKeys launcher.")
    parser.add_argument(
        "--mode",
        choices=["desktop", "api", "tk"],
        default="desktop",
        help="Launch mode: desktop (Tauri+API), api (backend only), or tk (legacy Tkinter UI).",
    )
    parser.add_argument("--camera", type=int, default=None, help="Preferred camera index.")
    parser.add_argument(
        "--disable-keys",
        action="store_true",
        help="Disable right-hand keyboard output in the gesture engine.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Backend API host (api mode).")
    parser.add_argument("--port", type=int, default=8000, help="Backend API port (api mode).")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="In desktop mode, do not fall back to legacy Tkinter if npm/Tauri is unavailable.",
    )
    return parser.parse_args()


def _legacy_tk_command(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(ROOT / "gesture_keyboard_v3.py")]
    if args.camera is not None:
        cmd += ["--camera", str(args.camera)]
    if args.disable_keys:
        cmd += ["--disable-keys"]
    return cmd


def run_tk(args: argparse.Namespace) -> int:
    return subprocess.call(_legacy_tk_command(args))


def run_api(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "gesture_api_server.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.camera is not None:
        cmd += ["--camera", str(args.camera)]
    if args.disable_keys:
        cmd += ["--disable-keys"]
    return subprocess.call(cmd)


def run_desktop(args: argparse.Namespace) -> int:
    backend_cmd = [
        sys.executable,
        str(ROOT / "gesture_keyboard.py"),
        "--mode",
        "api",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.camera is not None:
        backend_cmd += ["--camera", str(args.camera)]
    if args.disable_keys:
        backend_cmd += ["--disable-keys"]

    npm = shutil.which("npm")
    if npm is None or not UI_TAURI_DIR.exists():
        if args.no_fallback:
            print("Desktop mode requires npm and ./ui-tauri. Use --mode api or --mode tk.")
            return 1
        print("Desktop mode prerequisites missing. Falling back to legacy Tkinter mode.")
        return run_tk(args)

    backend = subprocess.Popen(backend_cmd, cwd=str(ROOT))
    tauri_process: Optional[subprocess.Popen] = None
    try:
        time.sleep(1.2)
        tauri_process = subprocess.Popen([npm, "run", "tauri", "dev"], cwd=str(UI_TAURI_DIR))
        return tauri_process.wait()
    except FileNotFoundError:
        if args.no_fallback:
            print("Tauri launch failed. Ensure Node/npm and Rust toolchain are installed.")
            return 1
        print("Tauri launch failed. Falling back to legacy Tkinter mode.")
        if tauri_process is not None:
            try:
                tauri_process.terminate()
            except Exception:
                pass
        return run_tk(args)
    finally:
        if backend.poll() is None:
            backend.terminate()
            try:
                backend.wait(timeout=4)
            except Exception:
                backend.kill()


def main() -> None:
    args = parse_args()
    if args.mode == "tk":
        code = run_tk(args)
    elif args.mode == "api":
        code = run_api(args)
    else:
        code = run_desktop(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
