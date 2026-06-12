# Security Policy

## Reporting a vulnerability

GestureKeys is an experimental desktop app, but security reports are welcome.
If you find a vulnerability, please **do not open a public issue**. Instead,
report it privately via GitHub's
[security advisories](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
("Report a vulnerability" on the repository's **Security** tab), or contact the
maintainer directly.

Please include:

- A description of the issue and its impact.
- Steps to reproduce.
- Affected version/commit.

You can expect an initial response within a reasonable time frame. Thanks for
helping keep users safe.

## Privacy & data handling

GestureKeys processes a **live webcam feed** to track hands, face, and a green
marker. This is sensitive data, so it's important to understand how it's used:

- **All processing is local.** Frames are analyzed on-device by MediaPipe.
  The app does not upload video, images, or landmarks to any server.
- **The backend binds to `127.0.0.1` (localhost) by default.** The FastAPI
  server and its MJPEG preview/WebSocket endpoints are reachable only from your
  own machine unless you deliberately change the host.
- **Calibration data stays local.** `gaze_calibration.json` and
  `%APPDATA%\GestureKeys\calibration_profiles_v1.json` are machine-specific and
  are git-ignored so they're never committed.
- **The app can control your keyboard, mouse, volume, brightness, and close
  windows** (via `pyautogui` and system APIs). Treat it like any input
  automation tool: run it only from sources you trust, and use
  `--disable-keys` if you want to observe tracking without it sending input.

### If you change the network binding

If you expose the backend beyond localhost (e.g. binding to `0.0.0.0` to view
the preview from another device), you are sending your webcam feed over the
network. Only do this on a trusted network, and consider that there is
currently **no authentication** on the API. Do not expose it to the public
internet.
