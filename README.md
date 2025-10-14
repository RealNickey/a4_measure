
# A4 Object Dimension Scanner (Real‑Time)

Measure the dimensions of simple objects (rectangles and circles) placed on an A4 sheet using your IP camera stream (e.g., an Android IP Webcam app at `/video`) or your laptop's webcam as fallback.

## Features
- Real‑time detection of an A4 sheet as the size reference (210mm × 297mm).
- Automatically stops the live stream once both A4 and an object are stably detected, then processes a single **good frame** for precise measurements.
- Supports **rectangles** and **circles** (diameter/width/height in millimeters).
- Robust perspective correction and mm-per-pixel accuracy using A4 true dimensions.
- Fast path with optional **GPU acceleration** if OpenCV CUDA is available.
- Keyboard workflow: After showing results, **press any key** to resume scanning.

## Quick Start

### 1) Install dependencies (Python 3.9+ recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run
```bash
python main.py
```
- The app will ask for your IP camera base URL (e.g. `http://192.168.1.7:8080`), then it will automatically try `/video`.
- Leave it empty to use the default webcam.

### 3) How to use
1. Place an **A4 sheet** flat in view.
2. Put a **single object** (rectangle-like or circle-like) fully inside the sheet boundary.
3. The program will lock onto a stable frame, stop the stream, and print/draw the measurements.
4. Press **any key** in the window to continue scanning again.

### Notes & Tips
- Good lighting and clear contrast between the object and the A4 background improve accuracy.
- The object must lie flat on the sheet.
- If your IP cam uses a different path than `/video`, type the full stream URL when prompted.

## Output
- On-screen annotated image with dimensions (mm).
- Console prints with numeric results.

## Project Structure
```
a4_measure/
├── main.py            # Orchestrates real-time detection and measurement cycle
├── camera.py          # IP camera + webcam capture helper
├── detection.py       # A4 detection, homography estimation, warping
├── measure.py         # Object segmentation & measurement (rect/ellipse/circle)
├── utils.py           # Geometry helpers and small utilities
├── config.py          # Tunable parameters (thresholds, sizes, tolerances)
├── requirements.txt   # Python dependencies
└── README.md
```

## Accuracy
- After finding the A4, we **warp** the sheet to its true aspect (210×297 mm) at a fixed pixel density (`PX_PER_MM`), so physical size per pixel is known by construction.
- Measurements use contour analysis, minAreaRect for rectangles, and circle fit for round objects.

### Detection Accuracy Improvements
- **Sub-pixel corner refinement**: Achieves 0.1 pixel accuracy using OpenCV's `cornerSubPix`
- **Perspective quality scoring**: Real-time feedback on detection quality (0-100%)
- **Multi-frame calibration**: Averages multiple frames to reduce noise and jitter
- **Measurement confidence**: Each measurement includes a confidence score
- **Quality warnings**: Visual indicators when detection quality is low

See [DETECTION_IMPROVEMENTS.md](DETECTION_IMPROVEMENTS.md) for detailed documentation.

## License
MIT
