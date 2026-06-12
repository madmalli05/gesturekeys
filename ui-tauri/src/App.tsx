import { useEffect, useMemo, useRef, useState } from "react";
import {
  activateCalibrationProfile,
  apiBaseUrl,
  applyCalibrationSession,
  cancelCalibrationSession,
  createCalibrationProfile,
  createStateWebSocket,
  deleteCalibrationProfile,
  deriveCalibrationSession,
  getCalibrationProfiles,
  getCalibrationSessionState,
  getSettings,
  getState,
  patchSettings,
  rollbackCalibrationSession,
  startCalibrationCapture,
  startCalibrationSession,
  startSession,
  stopSession
} from "./api";
import type {
  CalibrationProfilesResponse,
  CalibrationState,
  EngineSettings,
  EngineState,
  SessionStatus
} from "./types";

const DEFAULT_STATE: EngineState = {
  session_status: "idle",
  running: false,
  camera_index: null,
  gestureText: "R: -\nL: -\nM: -\nP: -",
  rightText: "idle",
  leftText: "idle",
  mouthText: "disabled",
  penState: "no-right-hand",
  lastKey: "-",
  keyCount: 0,
  status: "Connecting to backend...",
  holdProgress: 0,
  warnings: [],
  errors: [],
  metrics: {
    capture_fps: 0,
    inference_fps: 0,
    dropped_frames: 0,
    end_to_end_latency_ms: 0
  },
  updatedAt: Date.now() / 1000
};

const DEFAULT_SETTINGS: EngineSettings = {
  camera: null,
  cooldown: 0.8,
  hold_threshold: 0.5,
  pen_sensitivity: 50,
  key_output_enabled: true,
  mirror_preview: true,
  click_speed_profile: "turbo",
  mouth_enabled: true
};

const EMPTY_PROFILES: CalibrationProfilesResponse = {
  ok: true,
  active_profile_id: null,
  active_profile_name: "Default",
  profiles: []
};

const CAPTURE_DURATIONS: Record<"open_pinch" | "closed_pinch" | "pointer_still" | "pointer_move", number> = {
  open_pinch: 2200,
  closed_pinch: 2200,
  pointer_still: 2500,
  pointer_move: 3000
};

const GESTURE_MAP = [
  ["Right hand", "Keyboard map"],
  ["Right green pen", "Mouse pointer"],
  ["Left pinch + pen", "Left click / drag"],
  ["Green pen + thumb-middle pinch", "Right click"],
  ["Left pinky", "Volume"],
  ["Left fist/open", "Brightness"],
  ["Mouth 5s", "Alt+F4"]
];

const WIZARD_STEPS = ["device", "open_pinch", "closed_pinch", "pointer_still", "pointer_move", "review"] as const;
type WizardStep = (typeof WIZARD_STEPS)[number];

type CaptureStep = "open_pinch" | "closed_pinch" | "pointer_still" | "pointer_move";

const STEP_LABELS: Record<WizardStep, string> = {
  device: "Device Setup",
  open_pinch: "Open Pinch Sample",
  closed_pinch: "Closed Pinch Sample",
  pointer_still: "Pointer Stillness",
  pointer_move: "Pointer Motion",
  review: "Review & Apply"
};

function statusTone(status: SessionStatus | string): string {
  if (status === "running") return "tone-running";
  if (status === "starting" || status === "stopping") return "tone-pending";
  if (status === "error") return "tone-error";
  return "tone-idle";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function nextStep(step: WizardStep): WizardStep {
  const index = WIZARD_STEPS.indexOf(step);
  return WIZARD_STEPS[Math.min(index + 1, WIZARD_STEPS.length - 1)];
}

function previousStep(step: WizardStep): WizardStep {
  const index = WIZARD_STEPS.indexOf(step);
  return WIZARD_STEPS[Math.max(index - 1, 0)];
}

export default function App() {
  const [state, setState] = useState<EngineState>(DEFAULT_STATE);
  const [settings, setSettings] = useState<EngineSettings>(DEFAULT_SETTINGS);
  const [connection, setConnection] = useState<"connecting" | "online" | "offline">("connecting");
  const [requestError, setRequestError] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [streamKey, setStreamKey] = useState<number>(Date.now());
  const [profiles, setProfiles] = useState<CalibrationProfilesResponse>(EMPTY_PROFILES);
  const [calibration, setCalibration] = useState<CalibrationState | null>(null);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [wizardStep, setWizardStep] = useState<WizardStep>("device");
  const [wizardBusy, setWizardBusy] = useState(false);
  const [wizardMessage, setWizardMessage] = useState("Calibration idle.");
  const [newProfileName, setNewProfileName] = useState("");
  const skipFirstPatch = useRef(true);

  const refreshCalibration = async (): Promise<void> => {
    const [profilesRes, sessionRes] = await Promise.all([getCalibrationProfiles(), getCalibrationSessionState()]);
    setProfiles(profilesRes);
    setCalibration(sessionRes.state);
  };

  useEffect(() => {
    let active = true;
    const hydrate = async () => {
      try {
        const [initialState, initialSettings, profilesRes, sessionRes] = await Promise.all([
          getState(),
          getSettings(),
          getCalibrationProfiles(),
          getCalibrationSessionState()
        ]);
        if (!active) return;
        setState(initialState);
        setSettings({ ...DEFAULT_SETTINGS, ...initialSettings });
        setProfiles(profilesRes);
        setCalibration(sessionRes.state);
        setRequestError("");
      } catch (error) {
        if (!active) return;
        setRequestError((error as Error).message);
      }
    };
    hydrate();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (skipFirstPatch.current) {
      skipFirstPatch.current = false;
      return;
    }
    setSaving(true);
    setRequestError("");
    const handle = window.setTimeout(async () => {
      try {
        await patchSettings(settings);
      } catch (error) {
        setRequestError((error as Error).message);
      } finally {
        setSaving(false);
      }
    }, 220);
    return () => {
      window.clearTimeout(handle);
    };
  }, [settings]);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnect: number | null = null;
    let active = true;

    const connect = () => {
      if (!active) return;
      setConnection("connecting");
      ws = createStateWebSocket((payload) => {
        setState(payload);
        if (payload.calibration) {
          setCalibration(payload.calibration);
        }
      });
      ws.onopen = () => setConnection("online");
      ws.onerror = () => setConnection("offline");
      ws.onclose = () => {
        if (!active) return;
        setConnection("offline");
        reconnect = window.setTimeout(connect, 1500);
      };
    };

    connect();
    return () => {
      active = false;
      if (reconnect !== null) {
        window.clearTimeout(reconnect);
      }
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const previewUrl = useMemo(() => `${apiBaseUrl}/api/preview.mjpg?k=${streamKey}`, [streamKey]);

  const applySetting = <K extends keyof EngineSettings>(key: K, value: EngineSettings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleStart = async () => {
    try {
      setRequestError("");
      setState((prev) => ({ ...prev, session_status: "starting" }));
      await startSession();
      setStreamKey(Date.now());
    } catch (error) {
      setRequestError((error as Error).message);
      setState((prev) => ({ ...prev, session_status: "error" }));
    }
  };

  const handleStop = async () => {
    try {
      setRequestError("");
      setState((prev) => ({ ...prev, session_status: "stopping" }));
      await stopSession();
      setStreamKey(Date.now());
    } catch (error) {
      setRequestError((error as Error).message);
      setState((prev) => ({ ...prev, session_status: "error" }));
    }
  };

  const handleCreateProfile = async () => {
    const name = newProfileName.trim();
    if (!name) return;
    try {
      setRequestError("");
      const response = await createCalibrationProfile(name, true);
      setProfiles(response);
      setNewProfileName("");
      const refreshedSettings = await getSettings();
      setSettings({ ...DEFAULT_SETTINGS, ...refreshedSettings });
      await refreshCalibration();
    } catch (error) {
      setRequestError((error as Error).message);
    }
  };

  const handleActivateProfile = async (profileId: string) => {
    try {
      setRequestError("");
      const response = await activateCalibrationProfile(profileId);
      setProfiles(response);
      const refreshedSettings = await getSettings();
      setSettings({ ...DEFAULT_SETTINGS, ...refreshedSettings });
      await refreshCalibration();
    } catch (error) {
      setRequestError((error as Error).message);
    }
  };

  const handleDeleteProfile = async (profileId: string) => {
    try {
      setRequestError("");
      const response = await deleteCalibrationProfile(profileId);
      setProfiles(response);
      await refreshCalibration();
    } catch (error) {
      setRequestError((error as Error).message);
    }
  };

  const openCalibrationWizard = async () => {
    if (!state.running) {
      setRequestError("Start session before running calibration wizard.");
      return;
    }
    try {
      setRequestError("");
      const response = await startCalibrationSession(true);
      setCalibration(response.state);
      setWizardStep("device");
      setWizardOpen(true);
      setWizardMessage("Calibration safe mode is active. OS actions are suppressed.");
    } catch (error) {
      setRequestError((error as Error).message);
    }
  };

  const closeCalibrationWizard = async () => {
    try {
      await cancelCalibrationSession();
      await refreshCalibration();
    } catch {
      // Keep close resilient.
    }
    setWizardOpen(false);
    setWizardStep("device");
    setWizardBusy(false);
    setWizardMessage("Calibration cancelled.");
  };

  const runCaptureStep = async (step: CaptureStep) => {
    setWizardBusy(true);
    setWizardMessage(`Capturing ${STEP_LABELS[step]}...`);
    try {
      const startResult = await startCalibrationCapture(step, CAPTURE_DURATIONS[step]);
      setCalibration(startResult.state);
      const timeoutAt = Date.now() + CAPTURE_DURATIONS[step] + 3600;
      while (Date.now() < timeoutAt) {
        await sleep(160);
        const snapshot = await getCalibrationSessionState();
        setCalibration(snapshot.state);
        if (!snapshot.state.capture_step && snapshot.state.capture_completed) {
          break;
        }
      }
      setWizardMessage(`${STEP_LABELS[step]} captured.`);
    } catch (error) {
      setRequestError((error as Error).message);
      setWizardMessage(`Capture failed for ${STEP_LABELS[step]}.`);
    } finally {
      setWizardBusy(false);
    }
  };

  const handleWizardNext = async () => {
    if (wizardBusy) return;
    if (wizardStep === "device") {
      setWizardStep(nextStep(wizardStep));
      return;
    }
    if (wizardStep === "review") {
      return;
    }
    await runCaptureStep(wizardStep as CaptureStep);
    setWizardStep(nextStep(wizardStep));
  };

  const handleWizardRetake = async () => {
    if (wizardBusy || wizardStep === "device" || wizardStep === "review") return;
    await runCaptureStep(wizardStep as CaptureStep);
  };

  const handleWizardDerive = async () => {
    setWizardBusy(true);
    setWizardMessage("Deriving calibration from samples...");
    try {
      const response = await deriveCalibrationSession();
      setCalibration(response.state);
      setWizardMessage(response.message ?? "Derivation complete.");
    } catch (error) {
      setRequestError((error as Error).message);
      setWizardMessage("Derivation failed.");
    } finally {
      setWizardBusy(false);
    }
  };

  const handleWizardApply = async () => {
    setWizardBusy(true);
    setWizardMessage("Applying calibration...");
    try {
      const response = await applyCalibrationSession(true);
      setCalibration(response.state);
      await refreshCalibration();
      setWizardMessage("Calibration applied and saved to active profile.");
      setWizardOpen(false);
      setWizardStep("device");
    } catch (error) {
      setRequestError((error as Error).message);
      setWizardMessage("Apply failed.");
    } finally {
      setWizardBusy(false);
    }
  };

  const handleRollback = async () => {
    try {
      setRequestError("");
      const response = await rollbackCalibrationSession();
      setCalibration(response.state);
      const refreshed = await getSettings();
      setSettings({ ...DEFAULT_SETTINGS, ...refreshed });
      await refreshCalibration();
      setWizardMessage("Calibration rollback applied.");
    } catch (error) {
      setRequestError((error as Error).message);
    }
  };

  const warningText = state.warnings.length ? state.warnings[0] : "";
  const errorText = requestError || (state.errors.length ? state.errors[0] : "");
  const stateLines = state.gestureText.split("\n");
  const qualityWarnings = calibration?.quality?.warnings ?? [];

  return (
    <div className="app-shell">
      <div className="bg-bloom bg-bloom-one" />
      <div className="bg-bloom bg-bloom-two" />

      <header className="floating-card top-bar">
        <div>
          <div className="eyebrow">GestureKeys</div>
          <h1>Premium Desktop Control Suite</h1>
          <p>React + Tauri surface powered by the original Python gesture engine.</p>
        </div>
        <div className={`status-pill ${statusTone(state.session_status)}`}>
          <span>{String(state.session_status).toUpperCase()}</span>
          <small>{connection.toUpperCase()}</small>
        </div>
      </header>

      <main className="main-grid">
        <section className="floating-card hero-card">
          <div className="card-header">
            <div>
              <h2>Live Camera Preview</h2>
              <p>Realtime landmarks, gesture overlays, and pointer tracking feed.</p>
            </div>
            <div className="meta-chip">API stream: /api/preview.mjpg</div>
          </div>

          <div className="preview-shell">
            <img src={previewUrl} alt="Gesture preview stream" />
          </div>

          <div className="hero-stats">
            <div className="mini-stat">
              <span>Last key</span>
              <strong>{state.lastKey || "-"}</strong>
            </div>
            <div className="mini-stat">
              <span>Keys triggered</span>
              <strong>{state.keyCount}</strong>
            </div>
            <div className="mini-stat">
              <span>Hold progress</span>
              <strong>{Math.round((state.holdProgress || 0) * 100)}%</strong>
            </div>
            <div className="mini-stat">
              <span>Camera index</span>
              <strong>{state.camera_index ?? "-"}</strong>
            </div>
            <div className="mini-stat">
              <span>Capture FPS</span>
              <strong>{Number(state.metrics?.capture_fps ?? 0).toFixed(1)}</strong>
            </div>
            <div className="mini-stat">
              <span>Inference FPS</span>
              <strong>{Number(state.metrics?.inference_fps ?? 0).toFixed(1)}</strong>
            </div>
            <div className="mini-stat">
              <span>Dropped frames</span>
              <strong>{state.metrics?.dropped_frames ?? 0}</strong>
            </div>
            <div className="mini-stat">
              <span>Latency (ms)</span>
              <strong>{Number(state.metrics?.end_to_end_latency_ms ?? 0).toFixed(1)}</strong>
            </div>
          </div>
        </section>

        <aside className="side-column">
          <section className="floating-card">
            <div className="card-header compact">
              <div>
                <h3>Detected State</h3>
                <p>Fast scan telemetry for active gestures and modalities.</p>
              </div>
            </div>
            <div className="state-list">
              {stateLines.map((line) => (
                <div key={line} className="state-line">
                  {line}
                </div>
              ))}
            </div>
          </section>

          <section className="floating-card">
            <div className="card-header compact">
              <div>
                <h3>Session Controls</h3>
                <p>Backend-linked controls with smooth live updates.</p>
              </div>
            </div>
            <div className="button-row">
              <button type="button" className="action-btn primary" onClick={handleStart}>
                Start Session
              </button>
              <button type="button" className="action-btn secondary" onClick={handleStop}>
                Stop Session
              </button>
            </div>

            <label className="field">
              <span>Camera source</span>
              <select
                value={settings.camera ?? -1}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  applySetting("camera", value < 0 ? null : value);
                }}
              >
                <option value={-1}>Auto (Laptop Default)</option>
                <option value={0}>Camera 0</option>
                <option value={1}>Camera 1</option>
                <option value={2}>Camera 2</option>
                <option value={3}>Camera 3</option>
                <option value={4}>Camera 4</option>
              </select>
            </label>

            <label className="field">
              <span>Right-hand cooldown ({settings.cooldown.toFixed(2)}s)</span>
              <input
                type="range"
                min={0.2}
                max={2.0}
                step={0.01}
                value={settings.cooldown}
                onChange={(event) => applySetting("cooldown", Number(event.target.value))}
              />
            </label>

            <label className="field">
              <span>Hold to fire ({settings.hold_threshold.toFixed(2)}s)</span>
              <input
                type="range"
                min={0.1}
                max={1.5}
                step={0.01}
                value={settings.hold_threshold}
                onChange={(event) => applySetting("hold_threshold", Number(event.target.value))}
              />
            </label>

            <label className="field">
              <span>Green-pen sensitivity ({Math.round(settings.pen_sensitivity)}%)</span>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={settings.pen_sensitivity}
                onChange={(event) => applySetting("pen_sensitivity", Number(event.target.value))}
              />
            </label>

            <label className="field">
              <span>Click speed tuner</span>
              <select
                value={settings.click_speed_profile}
                onChange={(event) =>
                  applySetting("click_speed_profile", event.target.value as EngineSettings["click_speed_profile"])
                }
              >
                <option value="balanced">Balanced (most conservative)</option>
                <option value="fast">Fast</option>
                <option value="turbo">Turbo (double-click friendly)</option>
                <option value="ultra">Ultra (0.015 arm, max speed)</option>
              </select>
            </label>

            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.key_output_enabled}
                onChange={(event) => applySetting("key_output_enabled", event.target.checked)}
              />
              <span>Enable right-hand key output</span>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.mirror_preview}
                onChange={(event) => applySetting("mirror_preview", event.target.checked)}
              />
              <span>Mirror preview</span>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.mouth_enabled}
                onChange={(event) => applySetting("mouth_enabled", event.target.checked)}
              />
              <span>Enable mouth close action</span>
            </label>
          </section>

          <section className="floating-card">
            <div className="card-header compact">
              <div>
                <h3>Calibration</h3>
                <p>Named profiles + guided calibration wizard.</p>
              </div>
            </div>

            <label className="field">
              <span>Active profile</span>
              <select
                value={profiles.active_profile_id ?? ""}
                onChange={(event) => {
                  const next = event.target.value;
                  if (next) {
                    void handleActivateProfile(next);
                  }
                }}
              >
                {profiles.profiles.length === 0 ? <option value="">No profiles</option> : null}
                {profiles.profiles.map((profile) => (
                  <option key={profile.profile_id} value={profile.profile_id}>
                    {profile.name}
                  </option>
                ))}
              </select>
            </label>

            <div className="field-row">
              <input
                className="text-input"
                placeholder="New profile name"
                value={newProfileName}
                onChange={(event) => setNewProfileName(event.target.value)}
              />
              <button type="button" className="action-btn secondary mini" onClick={handleCreateProfile}>
                Create
              </button>
            </div>

            <div className="button-row">
              <button type="button" className="action-btn primary" onClick={openCalibrationWizard}>
                Run Wizard
              </button>
              <button
                type="button"
                className="action-btn secondary"
                onClick={() => void openCalibrationWizard()}
              >
                Recalibrate
              </button>
            </div>

            <div className="button-row">
              <button type="button" className="action-btn secondary" onClick={handleRollback}>
                Rollback Last
              </button>
              <button
                type="button"
                className="action-btn secondary"
                onClick={() => {
                  const active = profiles.profiles.find((item) => item.is_active);
                  if (active && active.profile_id !== "default") {
                    void handleDeleteProfile(active.profile_id);
                  }
                }}
              >
                Delete Active
              </button>
            </div>

            <div className="calibration-meta">
              <div>
                <span>Safe mode</span>
                <strong>{calibration?.safe_mode ? "ON" : "OFF"}</strong>
              </div>
              <div>
                <span>Derived</span>
                <strong>{calibration?.derived_ready ? "READY" : "PENDING"}</strong>
              </div>
              <div>
                <span>Score</span>
                <strong>{calibration?.quality ? `${Math.round(calibration.quality.score * 100)}%` : "-"}</strong>
              </div>
            </div>
          </section>

          <section className="floating-card">
            <div className="card-header compact">
              <div>
                <h3>Gesture Map</h3>
                <p>Reference for active controls in this build.</p>
              </div>
            </div>
            <div className="gesture-map">
              {GESTURE_MAP.map(([gesture, action]) => (
                <div className="gesture-row" key={gesture}>
                  <span>{gesture}</span>
                  <strong>{action}</strong>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </main>

      <footer className="floating-card status-rail">
        <div className="status-main">{state.status || "Waiting for backend status..."}</div>
        <div className="status-meta">{saving ? "Saving settings..." : "Settings synced"}</div>
      </footer>

      {wizardOpen ? (
        <div className="wizard-overlay">
          <div className="floating-card wizard-modal">
            <div className="wizard-header">
              <div>
                <div className="eyebrow">Calibration Wizard</div>
                <h3>{STEP_LABELS[wizardStep]}</h3>
                <p>{wizardMessage}</p>
              </div>
              <button type="button" className="action-btn secondary mini" onClick={() => void closeCalibrationWizard()}>
                Close
              </button>
            </div>

            <div className="wizard-progress">
              {WIZARD_STEPS.map((step) => (
                <div key={step} className={`wizard-step ${step === wizardStep ? "active" : ""}`}>
                  {STEP_LABELS[step]}
                </div>
              ))}
            </div>

            <div className="wizard-body">
              {wizardStep === "device" ? (
                <div className="wizard-panel">
                  <p>
                    Confirm camera source, mirror mode, and pen sensitivity from Session Controls, then continue. Calibration
                    runs in safe mode so no keys, clicks, volume, or brightness actions will fire.
                  </p>
                  <div className="sample-grid">
                    <div>
                      <span>Camera</span>
                      <strong>{settings.camera ?? "Auto"}</strong>
                    </div>
                    <div>
                      <span>Mirror</span>
                      <strong>{settings.mirror_preview ? "On" : "Off"}</strong>
                    </div>
                    <div>
                      <span>Pen sensitivity</span>
                      <strong>{Math.round(settings.pen_sensitivity)}%</strong>
                    </div>
                  </div>
                </div>
              ) : null}

              {wizardStep !== "device" && wizardStep !== "review" ? (
                <div className="wizard-panel">
                  <p>
                    Hold the requested pose naturally while sampling. You can retake this step if sample counts or quality are
                    low.
                  </p>
                  <div className="sample-grid">
                    <div>
                      <span>Capture step</span>
                      <strong>{calibration?.capture_step ?? "idle"}</strong>
                    </div>
                    <div>
                      <span>Remaining</span>
                      <strong>{Math.ceil((calibration?.capture_remaining_ms ?? 0) / 100) / 10}s</strong>
                    </div>
                    <div>
                      <span>Samples</span>
                      <strong>{calibration?.sample_counts[wizardStep] ?? 0}</strong>
                    </div>
                  </div>
                </div>
              ) : null}

              {wizardStep === "review" ? (
                <div className="wizard-panel">
                  <div className="review-grid">
                    <div>
                      <h4>Current Pinch</h4>
                      <p>
                        W close/open: {calibration?.current.pinch.close_world.toFixed(3)} / {calibration?.current.pinch.open_world.toFixed(3)}
                      </p>
                      <p>
                        I close/open: {calibration?.current.pinch.close_image.toFixed(3)} / {calibration?.current.pinch.open_image.toFixed(3)}
                      </p>
                      <p>
                        Deadzone: {calibration?.current.pointer.deadzone_base_px.toFixed(2)} px | Smoothness: {calibration?.current.pointer.smoothness_scalar.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <h4>Derived</h4>
                      <p>
                        W close/open: {calibration?.derived?.pinch.close_world.toFixed(3) ?? "-"} / {calibration?.derived?.pinch.open_world.toFixed(3) ?? "-"}
                      </p>
                      <p>
                        I close/open: {calibration?.derived?.pinch.close_image.toFixed(3) ?? "-"} / {calibration?.derived?.pinch.open_image.toFixed(3) ?? "-"}
                      </p>
                      <p>
                        Deadzone: {calibration?.derived?.pointer.deadzone_base_px.toFixed(2) ?? "-"} px | Smoothness: {calibration?.derived?.pointer.smoothness_scalar.toFixed(2) ?? "-"}
                      </p>
                    </div>
                  </div>

                  <div className="sample-grid">
                    <div>
                      <span>Quality score</span>
                      <strong>{calibration?.quality ? `${Math.round(calibration.quality.score * 100)}%` : "-"}</strong>
                    </div>
                    <div>
                      <span>Used source</span>
                      <strong>{calibration?.quality?.used_source ?? "-"}</strong>
                    </div>
                    <div>
                      <span>Ready</span>
                      <strong>{calibration?.derived_ready ? "Yes" : "No"}</strong>
                    </div>
                  </div>

                  {qualityWarnings.length ? (
                    <div className="quality-warnings">
                      {qualityWarnings.map((warning) => (
                        <div key={warning}>{warning}</div>
                      ))}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>

            <div className="wizard-actions">
              <button
                type="button"
                className="action-btn secondary"
                disabled={wizardBusy || wizardStep === "device"}
                onClick={() => setWizardStep(previousStep(wizardStep))}
              >
                Back
              </button>

              {wizardStep !== "review" ? (
                <>
                  {wizardStep !== "device" ? (
                    <button type="button" className="action-btn secondary" disabled={wizardBusy} onClick={() => void handleWizardRetake()}>
                      Retake
                    </button>
                  ) : null}
                  <button type="button" className="action-btn primary" disabled={wizardBusy} onClick={() => void handleWizardNext()}>
                    {wizardStep === "device" ? "Start Sampling" : "Capture & Continue"}
                  </button>
                </>
              ) : (
                <>
                  <button type="button" className="action-btn secondary" disabled={wizardBusy} onClick={() => void handleWizardDerive()}>
                    Derive
                  </button>
                  <button
                    type="button"
                    className="action-btn primary"
                    disabled={wizardBusy || !calibration?.derived_ready}
                    onClick={() => void handleWizardApply()}
                  >
                    Apply & Save
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {warningText ? <div className="toast warning">{warningText}</div> : null}
      {errorText ? <div className="toast error">{errorText}</div> : null}
    </div>
  );
}
