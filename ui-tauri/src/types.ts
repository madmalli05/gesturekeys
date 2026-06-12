export type SessionStatus = "idle" | "starting" | "running" | "stopping" | "error";
export type ClickSpeedProfile = "balanced" | "fast" | "turbo" | "ultra";

export interface EngineMetrics {
  capture_fps: number;
  inference_fps: number;
  dropped_frames: number;
  end_to_end_latency_ms: number;
}

export interface InteractionDiagnostics {
  pinch_state: string;
  pinch_ratio: number | null;
  pinch_source: string;
  pointer_mode: string;
  right_gesture_source: string;
  right_gesture_confidence: number;
  mouth_blend_score: number | null;
  mouth_signal_mode: string;
}

export interface PinchCalibration {
  close_world: number;
  open_world: number;
  close_image: number;
  open_image: number;
  source_preference: string;
}

export interface PointerCalibration {
  deadzone_base_px: number;
  smoothness_scalar: number;
}

export interface CalibrationQuality {
  ready: boolean;
  score: number;
  warnings: string[];
  sample_counts: Record<string, number>;
  separation_world: number | null;
  separation_image: number | null;
  used_source: string;
}

export interface CalibrationDerived {
  pinch: PinchCalibration;
  pointer: PointerCalibration;
  profiles: Record<string, Record<string, number>>;
}

export interface CalibrationState {
  safe_mode: boolean;
  active_profile_id: string | null;
  active_profile_name: string;
  session_active: boolean;
  capture_step: string | null;
  capture_completed: boolean;
  capture_remaining_ms: number;
  sample_counts: Record<string, number>;
  derived_ready: boolean;
  quality: CalibrationQuality | null;
  current: CalibrationDerived;
  derived: CalibrationDerived | null;
}

export interface EngineState {
  session_status: SessionStatus | string;
  running: boolean;
  camera_index: number | null;
  gestureText: string;
  rightText: string;
  leftText: string;
  mouthText: string;
  penState: string;
  lastKey: string;
  keyCount: number;
  status: string;
  holdProgress: number;
  warnings: string[];
  errors: string[];
  metrics: EngineMetrics;
  interaction?: InteractionDiagnostics;
  calibration?: CalibrationState;
  updatedAt: number;
}

export interface EngineSettings {
  camera: number | null;
  cooldown: number;
  hold_threshold: number;
  pen_sensitivity: number;
  key_output_enabled: boolean;
  mirror_preview: boolean;
  click_speed_profile: ClickSpeedProfile;
  mouth_enabled: boolean;
}

export interface CalibrationProfileSummary {
  profile_id: string;
  name: string;
  created_at: number;
  updated_at: number;
  is_active: boolean;
}

export interface CalibrationProfilesResponse {
  ok: boolean;
  active_profile_id: string | null;
  active_profile_name: string;
  profiles: CalibrationProfileSummary[];
}

export interface CalibrationSessionResponse {
  ok: boolean;
  message?: string;
  state: CalibrationState;
  derived?: {
    pinch: PinchCalibration;
    pointer: PointerCalibration;
    quality: CalibrationQuality;
    profiles: Record<string, Record<string, number>>;
  };
}
