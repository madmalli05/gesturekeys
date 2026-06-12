import type {
  CalibrationProfilesResponse,
  CalibrationSessionResponse,
  EngineSettings,
  EngineState
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      "Content-Type": "application/json"
    },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const apiBaseUrl = API_BASE;

export async function getState(): Promise<EngineState> {
  return requestJson<EngineState>("/api/session/state");
}

export async function getSettings(): Promise<EngineSettings> {
  return requestJson<EngineSettings>("/api/settings");
}

export async function patchSettings(patch: Partial<EngineSettings>): Promise<void> {
  await requestJson("/api/settings", {
    method: "PATCH",
    body: JSON.stringify(patch)
  });
}

export async function startSession(): Promise<void> {
  await requestJson("/api/session/start", { method: "POST" });
}

export async function stopSession(): Promise<void> {
  await requestJson("/api/session/stop", { method: "POST" });
}

export async function getCalibrationProfiles(): Promise<CalibrationProfilesResponse> {
  return requestJson<CalibrationProfilesResponse>("/api/calibration/profiles");
}

export async function createCalibrationProfile(name: string, cloneActive = true): Promise<CalibrationProfilesResponse> {
  return requestJson<CalibrationProfilesResponse>("/api/calibration/profiles", {
    method: "POST",
    body: JSON.stringify({ name, clone_active: cloneActive })
  });
}

export async function activateCalibrationProfile(profileId: string): Promise<CalibrationProfilesResponse> {
  return requestJson<CalibrationProfilesResponse>(`/api/calibration/profiles/${profileId}/activate`, {
    method: "POST"
  });
}

export async function deleteCalibrationProfile(profileId: string): Promise<CalibrationProfilesResponse> {
  return requestJson<CalibrationProfilesResponse>(`/api/calibration/profiles/${profileId}`, {
    method: "DELETE"
  });
}

export async function startCalibrationSession(safeMode = true): Promise<CalibrationSessionResponse> {
  return requestJson<CalibrationSessionResponse>("/api/calibration/session/start", {
    method: "POST",
    body: JSON.stringify({ safe_mode: safeMode })
  });
}

export async function getCalibrationSessionState(): Promise<CalibrationSessionResponse> {
  return requestJson<CalibrationSessionResponse>("/api/calibration/session/state");
}

export async function startCalibrationCapture(
  step: "open_pinch" | "closed_pinch" | "pointer_still" | "pointer_move",
  durationMs?: number
): Promise<CalibrationSessionResponse> {
  return requestJson<CalibrationSessionResponse>("/api/calibration/session/capture", {
    method: "POST",
    body: JSON.stringify({ step, duration_ms: durationMs ?? null })
  });
}

export async function deriveCalibrationSession(): Promise<CalibrationSessionResponse> {
  return requestJson<CalibrationSessionResponse>("/api/calibration/session/derive", {
    method: "POST"
  });
}

export async function applyCalibrationSession(saveProfile = true): Promise<CalibrationSessionResponse> {
  return requestJson<CalibrationSessionResponse>("/api/calibration/session/apply", {
    method: "POST",
    body: JSON.stringify({ save_profile: saveProfile })
  });
}

export async function cancelCalibrationSession(): Promise<CalibrationSessionResponse> {
  return requestJson<CalibrationSessionResponse>("/api/calibration/session/cancel", {
    method: "POST"
  });
}

export async function rollbackCalibrationSession(): Promise<CalibrationSessionResponse> {
  return requestJson<CalibrationSessionResponse>("/api/calibration/session/rollback", {
    method: "POST"
  });
}

export function createStateWebSocket(onMessage: (state: EngineState) => void): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  const ws = new WebSocket(`${wsBase}/api/ws/state`);
  ws.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data) as EngineState;
      onMessage(parsed);
    } catch {
      // Ignore malformed packets to keep stream resilient.
    }
  };
  return ws;
}
