# Manual Distance Measurement Mode

## Overview
The Manual Distance Measurement mode allows users to measure distances between any two points on the A4 sheet, with zoom support for precise measurements.

## How to Access
1. Run the application:
  - Live feed: `python main.py`
  - Photo mode: `python main.py --photo path/to/image.png`
2. Wait for A4 detection to complete
3. Press 'M' key to cycle through modes until you reach "MANUAL_DISTANCE"

## Features
- **Point-to-point distance measurement**: Click two points to measure the distance between them
- **Zoom support**: Use mouse wheel to zoom in (up to 5x) or out for precise point selection
- **Multiple measurements**: Create multiple distance measurements on the same image
- **Visual feedback**: 
  - Cyan lines and points for completed measurements
  - Magenta lines and points for measurements in progress
  - Distance values displayed at the midpoint of each line

## Controls

### Mouse Controls
- **Left Click**: 
  - First click: Set the first point
  - Second click: Set the second point and complete the measurement
- **Mouse Move**: Preview the distance measurement while selecting the second point
- **Right Click**: Cancel the current measurement (before completing the second point)
- **Mouse Wheel**: 
  - Scroll up: Zoom in (max 5x)
  - Scroll down: Zoom out (min 1x)

### Keyboard Controls
- **'M' key**: Switch to next mode (cycle through AUTO → MANUAL_RECT → MANUAL_CIRCLE → MANUAL_DISTANCE)
- **'C' key**: Clear all distance measurements
- **ESC key**: Exit the application
- **Any other key**: Resume scanning (in live mode)

## Usage Example

### Basic Distance Measurement
1. Press 'M' until you see "MODE: MANUAL_DISTANCE" in the top-right corner
2. Click on the first point (e.g., corner of an object)
3. Move your mouse to the second point (you'll see a preview line)
4. Click on the second point to complete the measurement
5. The distance in millimeters will be displayed

### Precise Measurement with Zoom
1. Enter MANUAL_DISTANCE mode
2. Scroll up with mouse wheel to zoom in (zoom level shown in mode indicator)
3. Click the first point
4. Click the second point
5. The measurement is calibrated to the A4 reference, so zoom doesn't affect accuracy

### Multiple Measurements
1. Complete a measurement as described above
2. Click another first point to start a new measurement
3. Continue adding measurements as needed
4. All measurements remain visible on screen
5. Press 'C' to clear all measurements when done

## Technical Details
- Distances are calculated using the A4 sheet calibration (mm_per_px_x and mm_per_px_y)
- Measurements account for perspective correction
- Distance formula: `sqrt((x2-x1)² * mm_per_px_x² + (y2-y1)² * mm_per_px_y²)`
- All measurements are displayed with 0.1mm precision
- **Note**: Zoom level is tracked and displayed, providing a visual indicator for precision work. The underlying measurements remain accurate regardless of zoom level since they are based on the calibrated A4 reference.

## Tips
- Use zoom for measuring small features or when precise point placement is needed
- The A4 sheet must be properly detected before using manual distance mode
- Measurements are relative to the warped/corrected A4 image, ensuring accuracy
- Cancel a measurement with right-click if you placed the first point incorrectly
- Clear all measurements with 'C' key to start fresh

## Color Coding
- **Cyan (0, 255, 255)**: Completed measurements
- **Magenta (255, 0, 255)**: Measurement in progress (before second click)
- **White text**: Distance labels on black background for visibility
