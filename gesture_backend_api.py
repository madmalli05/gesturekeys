from __future__ import annotations

import asyncio
import time
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from gesture_engine import GestureEngine


class SettingsPatch(BaseModel):
    camera: Optional[int] = Field(default=None, ge=0, le=8)
    cooldown: Optional[float] = Field(default=None, ge=0.2, le=2.0)
    hold_threshold: Optional[float] = Field(default=None, ge=0.1, le=1.5)
    pen_sensitivity: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    key_output_enabled: Optional[bool] = None
    mirror_preview: Optional[bool] = None
    click_speed_profile: Optional[Literal["balanced", "fast", "turbo", "ultra"]] = None
    mouth_enabled: Optional[bool] = None


class ProfileCreatePayload(BaseModel):
    name: str = Field(min_length=1, max_length=48)
    clone_active: bool = True


class CalibrationSessionStartPayload(BaseModel):
    safe_mode: bool = True


class CalibrationCapturePayload(BaseModel):
    step: Literal["open_pinch", "closed_pinch", "pointer_still", "pointer_move"]
    duration_ms: Optional[int] = Field(default=None, ge=450, le=9000)


class CalibrationApplyPayload(BaseModel):
    save_profile: bool = True


def _placeholder_frame(text: str) -> bytes:
    canvas = np.zeros((420, 720, 3), dtype=np.uint8)
    canvas[:] = (18, 20, 25)
    cv2.rectangle(canvas, (24, 24), (696, 396), (55, 68, 92), 1)
    cv2.putText(canvas, "GestureKeys API Preview", (42, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (190, 220, 245), 2)
    cv2.putText(canvas, text[:64], (42, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 175, 205), 1)
    ok, jpg = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return b""
    return jpg.tobytes()


def create_app(default_camera: Optional[int] = None, key_output_enabled: bool = True) -> FastAPI:
    app = FastAPI(title="GestureKeys Backend API", version="1.0.0")
    engine = GestureEngine(default_camera=default_camera, key_output_enabled=key_output_enabled)
    app.state.engine = engine

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        app.state.engine.close()

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "timestamp": time.time()}

    @app.get("/api/settings")
    async def get_settings() -> dict:
        return app.state.engine.get_settings()

    @app.patch("/api/settings")
    async def patch_settings(payload: SettingsPatch) -> dict:
        data = payload.model_dump(exclude_unset=True)
        updated = app.state.engine.update_settings(**data)
        return {"ok": True, "settings": updated, "state": app.state.engine.get_state()}

    @app.get("/api/calibration/profiles")
    async def calibration_profiles() -> dict:
        return {"ok": True, **app.state.engine.list_calibration_profiles()}

    @app.post("/api/calibration/profiles")
    async def calibration_profiles_create(payload: ProfileCreatePayload) -> dict:
        data = app.state.engine.create_calibration_profile(
            name=payload.name,
            clone_active=payload.clone_active,
        )
        return {"ok": True, **data}

    @app.post("/api/calibration/profiles/{profile_id}/activate")
    async def calibration_profiles_activate(profile_id: str) -> JSONResponse:
        result = app.state.engine.activate_calibration_profile(profile_id)
        payload = result
        if isinstance(result.get("profiles"), dict):
            payload = {"ok": result.get("ok", False), "message": result.get("message", ""), **result["profiles"]}
        status_code = 200 if result.get("ok") else 404
        return JSONResponse(content=payload, status_code=status_code)

    @app.delete("/api/calibration/profiles/{profile_id}")
    async def calibration_profiles_delete(profile_id: str) -> JSONResponse:
        result = app.state.engine.delete_calibration_profile(profile_id)
        payload = result
        if isinstance(result.get("profiles"), dict):
            payload = {"ok": result.get("ok", False), "message": result.get("message", ""), **result["profiles"]}
        status_code = 200 if result.get("ok") else 400
        return JSONResponse(content=payload, status_code=status_code)

    @app.post("/api/calibration/session/start")
    async def calibration_start(payload: CalibrationSessionStartPayload) -> dict:
        return app.state.engine.start_calibration_session(safe_mode=payload.safe_mode)

    @app.get("/api/calibration/session/state")
    async def calibration_state() -> dict:
        return app.state.engine.get_calibration_session_state()

    @app.post("/api/calibration/session/capture")
    async def calibration_capture(payload: CalibrationCapturePayload) -> JSONResponse:
        result = app.state.engine.start_calibration_capture(payload.step, payload.duration_ms)
        status_code = 200 if result.get("ok") else 400
        return JSONResponse(content=result, status_code=status_code)

    @app.post("/api/calibration/session/derive")
    async def calibration_derive() -> JSONResponse:
        result = app.state.engine.derive_calibration()
        status_code = 200 if result.get("ok") else 400
        return JSONResponse(content=result, status_code=status_code)

    @app.post("/api/calibration/session/apply")
    async def calibration_apply(payload: CalibrationApplyPayload) -> JSONResponse:
        result = app.state.engine.apply_calibration(save_profile=payload.save_profile)
        status_code = 200 if result.get("ok") else 400
        return JSONResponse(content=result, status_code=status_code)

    @app.post("/api/calibration/session/cancel")
    async def calibration_cancel() -> dict:
        return app.state.engine.cancel_calibration_session()

    @app.post("/api/calibration/session/rollback")
    async def calibration_rollback() -> JSONResponse:
        result = app.state.engine.rollback_last_calibration()
        status_code = 200 if result.get("ok") else 400
        return JSONResponse(content=result, status_code=status_code)

    @app.post("/api/session/start")
    async def start_session() -> JSONResponse:
        result = app.state.engine.start()
        status_code = 200 if result.get("ok") else 400
        return JSONResponse(
            content={"ok": result.get("ok"), "result": result, "state": app.state.engine.get_state()},
            status_code=status_code,
        )

    @app.post("/api/session/stop")
    async def stop_session() -> dict:
        result = app.state.engine.stop()
        return {"ok": result.get("ok"), "result": result, "state": app.state.engine.get_state()}

    @app.get("/api/session/state")
    async def session_state() -> dict:
        return app.state.engine.get_state()

    @app.websocket("/api/ws/state")
    async def ws_state(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.send_json(app.state.engine.get_state())
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return
        except Exception:
            try:
                await ws.close()
            except Exception:
                pass

    @app.get("/api/preview.mjpg")
    async def preview_mjpeg() -> StreamingResponse:
        boundary = "frame"

        def stream():
            while True:
                frame = app.state.engine.get_latest_frame_jpeg()
                if frame is None:
                    state = app.state.engine.get_state()
                    frame = _placeholder_frame(state.get("status", "Camera idle"))
                payload = (
                    b"--" + boundary.encode("ascii") + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    + frame
                    + b"\r\n"
                )
                yield payload
                time.sleep(1 / 24.0)

        return StreamingResponse(stream(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    return app
