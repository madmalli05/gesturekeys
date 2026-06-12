# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project scaffolding for a clean, GitHub-ready repository: `.gitignore`,
  `.gitattributes`, MIT `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, this
  changelog, issue/PR templates, and a GitHub Actions CI workflow.
- Ruff lint configuration and a hardware-free `pytest` smoke-test suite.
- `requirements-dev.txt`, `.env.example`, and `gaze_calibration.example.json`.

### Changed
- MediaPipe model files (`*.task`) and personal calibration files are now
  git-ignored; models download automatically on first run.

## [0.1.0] — Premium desktop UI

Baseline captured when the repository was initialized. Highlights from prior
development:

### Added
- React + Tauri desktop UI as the primary interface.
- Calibration Wizard with named profiles and auto-derived pinch/smoothing.
- Click speed tuner with `balanced`, `fast`, `turbo`, and `ultra` profiles.
- Runtime toggle for the mouth-close action (defaults ON each launch).
- Paste gesture (`thumb + index + middle up` → `Ctrl+V`).

### Changed
- Right-click remapped to **green pen active + thumb-middle pinch**.

[Unreleased]: https://github.com/ahmadmalli/gesturekeys/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ahmadmalli/gesturekeys/releases/tag/v0.1.0
