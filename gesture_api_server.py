from __future__ import annotations

import argparse

import uvicorn

from gesture_backend_api import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GestureKeys backend API.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server.")
    parser.add_argument("--camera", type=int, default=None, help="Preferred camera index.")
    parser.add_argument(
        "--disable-keys",
        action="store_true",
        help="Disable right-hand keyboard output in the engine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(default_camera=args.camera, key_output_enabled=not args.disable_keys)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

