# Copilot Usage Guide for `a4_measure`

## Big Picture

- `main.py` runs the real-time scanner: capture → detect A4 → warp to calibrated plane → segment objects → measure/inspect.
- All downstream logic assumes frames are warped to A4 scale using `warp_a4` and `a4_scale_mm_per_px`; keep work in this coordinate space unless you explicitly reproject.

## Core Modules

- `camera.open_capture` negotiates IP stream vs webcam. Prefer reusing it so resolution and buffer tweaks stay consistent.
- `detection.py` handles A4 discovery. `find_a4_quad` expects raw BGR frames; call `warp_a4` immediately afterward to guarantee `PX_PER_MM` resolution.
- `measure.py` owns segmentation, classification, and manual selection bridges. `classify_and_measure` (and helpers) must round physical sizes to nearest mm and attach `hit_contour`, `inner`, and `detection_method` fields.
- Manual selection is routed through `process_manual_selection` → `ShapeSnappingEngine` → `EnhancedContourAnalyzer`; submit selection rects as `(x, y, w, h)` in warped coordinates.

## Configuration & Conventions

- `config.py` is single source of truth for thresholds (Canny, morphology, contour area) and UI constants. Touching `PX_PER_MM` or `MIN_OBJECT_AREA_MM2` cascades into area filters throughout `measure.py`.
- Rectangle metrics are normalized so width ≤ height; keep that ordering when adding fields or tests.
- Always validate `mm_per_px_x` and `mm_per_px_y` > 0 before converting pixels to millimeters; reuse `_validate_scaling_factors` where possible.

## Developer Workflow

- Create a venv (`python -m venv .venv`) and `pip install -r requirements.txt` (OpenCV, NumPy, optional CUDA build).
- Run `python main.py`; press Enter to fall back to the local webcam, or provide the IP cam base URL (the code appends `/video` automatically).
- Use ESC to quit any window; after locking onto a frame, press any key to resume. Hit `M` in Inspect Mode to cycle AUTO → MANUAL RECT → MANUAL CIRCLE.
- Demo scripts (`demo_*.py`) exercise specific subsystems (manual selection, snapping, extended interactions); run them when debugging isolated features.

## Implementation Notes

- Filtering small contours: convert `MIN_OBJECT_AREA_MM2` to pixels via `PX_PER_MM**2` just like `main.py` does; keep the same check if you add new contour filters.
- Inner-shape detection relies on `detect_inner_circles` / `detect_inner_rectangles`; extend them instead of duplicating ROI logic.
- When emitting new measurement records, populate `hit_contour` with an `(N,1,2)` int32 array for interaction hit-testing, and set `inner=True` for holes/pockets.
- CUDA usage is auto-detected (`USE_CUDA` flag). Guard GPU-only code with that flag so CPU-only environments stay functional.

## When Extending Inspect Mode

- Favor the `ManualSelectionEngine` for mouse handling— it already handles scaling and validation. Manual callbacks should work in warped space, then resize for display via the scale set in `main.py`.
- Any new feedback to users should go through `UserFeedbackManager` / logging helpers in `manual_selection_errors.py` to keep console messaging uniform.

🤖 Ping the maintainer if something here seems off; otherwise keep these conventions to stay aligned with the existing tooling.
