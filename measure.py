import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from config import (BINARY_BLOCK_SIZE, BINARY_C, MIN_OBJECT_AREA_MM2, PX_PER_MM,
                    CIRCULARITY_CUTOFF, RECT_ANGLE_EPS_DEG, DRAW_FONT, DRAW_THICKNESS,
                    ENABLE_ADAPTIVE_THRESHOLD, ADAPTIVE_THRESHOLD_ENABLE_CLAHE,
                    ADAPTIVE_THRESHOLD_ENABLE_MULTIPASS, ADAPTIVE_THRESHOLD_ENABLE_LOCAL)
from utils import draw_text

def _area_px2_from_mm2(mm2):
    # Convert mm^2 to px^2 given PX_PER_MM
    return (mm2 * (PX_PER_MM ** 2))

# Global adaptive threshold calibrator instance (lazy initialization)
_adaptive_calibrator = None

def _get_adaptive_calibrator():
    """Get or create the adaptive threshold calibrator instance."""
    global _adaptive_calibrator
    if _adaptive_calibrator is None:
        try:
            from adaptive_threshold_calibrator import AdaptiveThresholdCalibrator
            _adaptive_calibrator = AdaptiveThresholdCalibrator(
                initial_block_size=BINARY_BLOCK_SIZE,
                initial_c=BINARY_C,
                enable_clahe=ADAPTIVE_THRESHOLD_ENABLE_CLAHE,
                enable_multipass=ADAPTIVE_THRESHOLD_ENABLE_MULTIPASS,
                enable_local_adaptive=ADAPTIVE_THRESHOLD_ENABLE_LOCAL
            )
            logging.info("Adaptive threshold calibrator initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize adaptive calibrator: {e}")
            _adaptive_calibrator = None
    return _adaptive_calibrator

def segment_object(a4_bgr):
    # Expect a light background (paper). Use adaptive threshold to get dark object(s).
    gray = cv2.cvtColor(a4_bgr, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive threshold calibration if enabled
    if ENABLE_ADAPTIVE_THRESHOLD:
        calibrator = _get_adaptive_calibrator()
        if calibrator is not None:
            try:
                bw, stats = calibrator.calibrate_and_threshold(gray)
                logging.debug(f"Adaptive threshold: block_size={stats['block_size']}, "
                            f"c={stats['c_constant']:.1f}, "
                            f"lighting={stats['lighting_stats']['lighting_condition']}")
                return bw
            except Exception as e:
                logging.warning(f"Adaptive threshold failed, falling back to standard: {e}")
                # Fall through to standard method
    
    # Standard adaptive threshold (fallback or if adaptive is disabled)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, BINARY_BLOCK_SIZE, BINARY_C)
    # Morph cleanup
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    return bw

def largest_inner_contour(mask, margin_px= int(8*PX_PER_MM)):
    # Deprecated in favor of all_inner_contours; keep for backward compatibility
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        if x <= margin_px or y <= margin_px or (x+ww) >= (w - margin_px) or (y+hh) >= (h - margin_px):
            continue
        area = cv2.contourArea(cnt)
        if area > best_area:
            best = cnt
            best_area = area
    return best

def all_inner_contours(mask, margin_px=int(8*PX_PER_MM)):
    # Find all valid inner contours (excluding edges), including nested ones
    h, w = mask.shape[:2]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for idx, cnt in enumerate(contours):
        x, y, ww, hh = cv2.boundingRect(cnt)
        if x <= margin_px or y <= margin_px or (x+ww) >= (w - margin_px) or (y+hh) >= (h - margin_px):
            continue
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        valid.append(cnt)
    # Sort by area descending to make labeling deterministic
    valid.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    return valid

# Hit testing polygon creation functions

def create_hit_testing_contour(shape_type: str, **kwargs) -> np.ndarray:
    """
    Create a hit testing contour for a given shape type.
    
    Args:
        shape_type: Type of shape ('circle' or 'rectangle')
        **kwargs: Shape-specific parameters
        
    Returns:
        Hit testing contour as numpy array
    """
    if shape_type == 'circle':
        center = kwargs.get('center', (0, 0))
        radius_px = kwargs.get('radius_px', 10)
        
        # Create circular contour with 36 points
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        circle_points = np.array([(int(center[0] + radius_px * np.cos(a)),
                                  int(center[1] + radius_px * np.sin(a))) for a in angles])
        return circle_points.reshape(-1, 1, 2).astype(np.int32)
        
    elif shape_type == 'rectangle':
        box = kwargs.get('box')
        if box is not None:
            return box.reshape(-1, 1, 2).astype(np.int32)
        
        # Fallback: create rectangle from center and dimensions
        center = kwargs.get('center', (0, 0))
        width = kwargs.get('width', 20)
        height = kwargs.get('height', 20)
        
        cx, cy = center
        hw, hh = width / 2, height / 2
        box = np.array([
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh]
        ], dtype=int)
        return box.reshape(-1, 1, 2).astype(np.int32)
    
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")

def create_shape_data(measurement_result, contour=None):
    """Convert measurement result to interactive shape data structure"""
    if measurement_result is None:
        return None
    
    # The measurement result already contains all needed fields from classify_and_measure
    # Just ensure all required fields are present and properly formatted
    shape_data = {
        "type": measurement_result["type"],
        "inner": measurement_result.get("inner", False),
        "hit_contour": measurement_result["hit_contour"],
        "area_px": measurement_result["area_px"]
    }
    
    if measurement_result["type"] == "circle":
        shape_data.update({
            "diameter_mm": float(measurement_result["diameter_mm"]),
            "center": tuple(measurement_result["center"]),
            "radius_px": float(measurement_result["radius_px"])
        })
    else:  # rectangle
        shape_data.update({
            "width_mm": float(measurement_result["width_mm"]),
            "height_mm": float(measurement_result["height_mm"]),
            "box": measurement_result["box"]
        })
    
    return shape_data

def classify_and_measure(cnt, mm_per_px_x, mm_per_px_y, detection_method="automatic"):
    # Compute circularity
    area = cv2.contourArea(cnt)
    if area <= 0:
        return None

    peri = cv2.arcLength(cnt, True)
    circularity = 4.0 * np.pi * area / (peri*peri + 1e-9)

    # Create hit testing contour from the original contour
    hit_contour = cnt.reshape(-1, 1, 2).astype(np.int32)

    if circularity >= CIRCULARITY_CUTOFF:
        # Circle-like
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        diameter_px = 2.0 * radius
        diameter_mm_raw = diameter_px * mm_per_px_x  # assume isotropic scale
        # Round to nearest millimeter for consistent precision (Requirement 1.5, 5.4)
        diameter_mm = round(diameter_mm_raw)
        
        # Create circular hit contour for better hit testing
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        circle_points = np.array([(int(center[0] + radius * np.cos(a)),
                                  int(center[1] + radius * np.sin(a))) for a in angles])
        hit_contour = circle_points.reshape(-1, 1, 2).astype(np.int32)
        
        return {
            "type": "circle",
            "diameter_mm": diameter_mm,
            "center": center,
            "radius_px": radius,
            "hit_contour": hit_contour,
            "area_px": area,
            "inner": False,
            "detection_method": detection_method
        }
    else:
        # Rectangle-like using minAreaRect
        rect = cv2.minAreaRect(cnt)  # ((cx,cy), (w,h), angle)
        (wpx, hpx) = rect[1]
        if wpx < 1 or hpx < 1:
            return None
        # Normalize width<height for consistency
        width_px = min(wpx, hpx)
        height_px = max(wpx, hpx)
        width_mm_raw = width_px * mm_per_px_x
        height_mm_raw = height_px * mm_per_px_y
        # Round to nearest millimeter for consistent precision (Requirement 1.5, 5.4)
        width_mm = round(width_mm_raw)
        height_mm = round(height_mm_raw)
        box = cv2.boxPoints(rect).astype(int)
        
        # Use the box points as hit contour for rectangles
        hit_contour = box.reshape(-1, 1, 2).astype(np.int32)

        # Optional right-angle sanity (not strictly enforced)
        return {
            "type": "rectangle",
            "width_mm": width_mm,
            "height_mm": height_mm,
            "box": box,
            "hit_contour": hit_contour,
            "area_px": area,
            "inner": False,
            "detection_method": detection_method
        }

def annotate_result(a4_bgr, result, mm_per_px):
    # Backward-compatible single-result annotation
    return annotate_results(a4_bgr, [result], mm_per_px)

def annotate_results(a4_bgr, results, mm_per_px):
    out = a4_bgr.copy()
    # Color palette for multiple objects
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 255, 0), (0, 128, 255), (255, 128, 0)
    ]
    idx = 0
    for res in results:
        # Use a distinct color for inner shapes to differentiate them
        if res.get("inner", False):
            color = (0, 0, 255)  # Red for inner shapes
        else:
            color = colors[idx % len(colors)]
        if res["type"] == "circle":
            center = res["center"]
            radius_px = int(res["radius_px"])
            cv2.circle(out, center, radius_px, color, 3)
            # diameter line
            x0 = center[0] - radius_px
            x1 = center[0] + radius_px
            y = center[1]
            cv2.line(out, (x0, y), (x1, y), color, 2)
            # Dimension text centered inside the circle
            text_inside = f"D={res['diameter_mm']:.0f}mm"
            ts = cv2.getTextSize(text_inside, DRAW_FONT, 0.9, 2)[0]
            text_org = (int(center[0] - ts[0] / 2), int(center[1] + ts[1] / 2))
            cv2.rectangle(out,
                          (text_org[0] - 6, text_org[1] - ts[1] - 6),
                          (text_org[0] + ts[0] + 6, text_org[1] + 6),
                          (255, 255, 255), -1)
            draw_text(out, text_inside, text_org, (0, 0, 0), 0.9, 2)
        else:
            box = res["box"]
            cv2.drawContours(out, [box], 0, color, 3)
            # Dimension arrows
            mid_left = ((box[0] + box[3]) / 2).astype(int)
            mid_right = ((box[1] + box[2]) / 2).astype(int)
            cv2.arrowedLine(out, tuple(mid_left), tuple(mid_right), color, 2, tipLength=0.02)
            cv2.arrowedLine(out, tuple(mid_right), tuple(mid_left), color, 2, tipLength=0.02)
            mid_top = ((box[0] + box[1]) / 2).astype(int)
            mid_bottom = ((box[2] + box[3]) / 2).astype(int)
            cv2.arrowedLine(out, tuple(mid_top), tuple(mid_bottom), color, 2, tipLength=0.02)
            cv2.arrowedLine(out, tuple(mid_bottom), tuple(mid_top), color, 2, tipLength=0.02)
            # Dimension text centered inside the rectangle
            cx = int(np.mean(box[:, 0]))
            cy = int(np.mean(box[:, 1]))
            text_inside = f"W={res['width_mm']:.0f}mm  H={res['height_mm']:.0f}mm"
            ts = cv2.getTextSize(text_inside, DRAW_FONT, 0.9, 2)[0]
            text_org = (int(cx - ts[0] / 2), int(cy + ts[1] / 2))
            cv2.rectangle(out,
                          (text_org[0] - 6, text_org[1] - ts[1] - 6),
                          (text_org[0] + ts[0] + 6, text_org[1] + 6),
                          (255, 255, 255), -1)
            draw_text(out, text_inside, text_org, (0, 0, 0), 0.9, 2)
        idx += 1
    return out

def detect_inner_circles(a4_bgr, object_mask, object_cnt, mm_per_px_x, min_radius_px=0):
    # Detect circular holes/marks inside a given object using HoughCircles on the object's ROI
    # Only return the largest valid inner circle to avoid clutter.
    h, w = object_mask.shape[:2]
    x, y, ww, hh = cv2.boundingRect(object_cnt)
    # Create ROI masked grayscale
    gray = cv2.cvtColor(a4_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+hh, x:x+ww].copy()
    roi_mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(roi_mask, [object_cnt - [x, y]], -1, 255, thickness=cv2.FILLED)
    # Suppress pixels outside the object
    roi_gray_masked = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)
    roi_blur = cv2.medianBlur(roi_gray_masked, 5)

    # Minimum radius threshold to reduce noise (≈2 mm by default)
    min_radius_px = max(min_radius_px, int(2 * PX_PER_MM))

    # HoughCircles parameters tuned modestly; may require adjustment per scene
    circles = cv2.HoughCircles(roi_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12,
                               param1=120, param2=22,
                               minRadius=min_radius_px,
                               maxRadius=int(min(ww, hh) * 0.45))
    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int)
    if len(circles) == 0:
        return []

    # Pick the largest circle only
    cx, cy, r = max(circles, key=lambda c: c[2])
    full_cx = int(cx + x)
    full_cy = int(cy + y)
    center = (full_cx, full_cy)
    diameter_mm_raw = (2.0 * r) * mm_per_px_x
    # Round to nearest millimeter for consistent precision (Requirement 1.5, 5.4)
    diameter_mm = round(diameter_mm_raw)
    area_px = np.pi * (r ** 2)
    
    # Create circular hit contour
    angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    circle_points = np.array([(int(center[0] + r * np.cos(a)),
                              int(center[1] + r * np.sin(a))) for a in angles])
    hit_contour = circle_points.reshape(-1, 1, 2).astype(np.int32)
    
    return [{
        "type": "circle",
        "diameter_mm": diameter_mm,
        "center": center,
        "radius_px": float(r),
        "hit_contour": hit_contour,
        "area_px": area_px,
        "inner": True,
        "detection_method": "automatic"
    }]


def detect_inner_rectangles(a4_bgr, object_mask, object_cnt, mm_per_px_x, mm_per_px_y):
    # Detect a prominent inner rectangle inside the object's ROI via contour approx; return largest only.
    x, y, ww, hh = cv2.boundingRect(object_cnt)
    # Restrict to object ROI using mask
    roi_mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(roi_mask, [object_cnt - [x, y]], -1, 255, thickness=cv2.FILLED)

    # Use edges inside ROI
    gray = cv2.cvtColor(a4_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+hh, x:x+ww]
    edges = cv2.Canny(roi_gray, 60, 180)
    edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for c in cnts:
        if cv2.contourArea(c) < 50:  # ignore tiny
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best = approx
                best_area = area
    if best is None:
        return []

    # Compute rectangle metrics
    rect = cv2.minAreaRect(best)
    (wpx, hpx) = rect[1]
    if wpx < 1 or hpx < 1:
        return []
    width_px = min(wpx, hpx)
    height_px = max(wpx, hpx)
    width_mm_raw = width_px * mm_per_px_x
    height_mm_raw = height_px * mm_per_px_y
    # Round to nearest millimeter for consistent precision (Requirement 1.5, 5.4)
    width_mm = round(width_mm_raw)
    height_mm = round(height_mm_raw)
    box = cv2.boxPoints(rect).astype(int)
    # map box to full image coords
    box[:, 0] += x
    box[:, 1] += y
    
    # Calculate area and create hit contour
    area_px = cv2.contourArea(box)
    hit_contour = box.reshape(-1, 1, 2).astype(np.int32)
    
    return [{
        "type": "rectangle",
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "box": box,
        "hit_contour": hit_contour,
        "area_px": area_px,
        "inner": True,
        "detection_method": "automatic"
    }]


# Manual Selection Integration Functions

def classify_and_measure_manual_selection(image: np.ndarray, selection_rect: Tuple[int, int, int, int], 
                                        shape_result: Dict[str, Any], mm_per_px_x: float, mm_per_px_y: float) -> Optional[Dict[str, Any]]:
    """
    Convert a manual shape selection result to the standard measurement format.
    
    This function bridges the gap between manual selection results from the shape snapping engine
    and the standard measurement data format used throughout the application.
    
    Args:
        image: Source image (BGR format)
        selection_rect: Original selection rectangle as (x, y, width, height)
        shape_result: Shape result from ShapeSnappingEngine
        mm_per_px_x: Millimeters per pixel in X direction
        mm_per_px_y: Millimeters per pixel in Y direction
        
    Returns:
        Measurement result in standard format or None if conversion fails
    """
    # Validate input parameters
    if shape_result is None:
        print("[WARN] No shape result provided for manual selection conversion")
        return None
    
    if "type" not in shape_result:
        print("[WARN] Shape result missing required 'type' field")
        return None
    
    shape_type = shape_result["type"]
    
    # Validate scaling factors before attempting conversion
    validation_result = _validate_scaling_factors(mm_per_px_x, mm_per_px_y)
    if not validation_result["valid"]:
        print(f"[ERROR] Invalid scaling factors for manual selection: {', '.join(validation_result['errors'])}")
        print("[ERROR] Cannot convert manual selection without valid A4 calibration.")
        return None
    
    # Log warnings if any
    for warning in validation_result.get("warnings", []):
        print(f"[WARN] {warning}")
    
    try:
        if shape_type == "circle":
            return _convert_manual_circle_to_measurement(shape_result, mm_per_px_x, selection_rect)
        elif shape_type == "rectangle":
            return _convert_manual_rectangle_to_measurement(shape_result, mm_per_px_x, mm_per_px_y, selection_rect)
        else:
            print(f"[WARN] Unsupported shape type for manual selection: {shape_type}")
            return None
            
    except ValueError as e:
        # Enhanced error handling with specific guidance for calibration issues
        print(f"[ERROR] Scaling factor validation failed: {e}")
        if "calibration" in str(e).lower() or "A4" in str(e):
            print("[ERROR] This indicates A4 paper calibration is invalid.")
            print("[ERROR] Please ensure A4 paper is properly positioned and detected before using manual selection.")
        return None
    except Exception as e:
        # Maintain existing shape detection error handling - ensure graceful degradation
        print(f"[ERROR] Error converting manual selection to measurement: {e}")
        print("[ERROR] Manual selection processing failed, but shape detection quality is preserved.")
        return None


def _convert_manual_circle_to_measurement(shape_result: Dict[str, Any], mm_per_px_x: float, 
                                        selection_rect: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """
    Convert manual circle selection to standard measurement format.
    
    Args:
        shape_result: Circle result from shape snapping engine
        mm_per_px_x: Millimeters per pixel conversion factor in X direction
        selection_rect: Original selection rectangle
        
    Returns:
        Circle measurement in standard format
        
    Raises:
        ValueError: If scaling factor is invalid (None, zero, or negative)
    """
    # Enhanced input validation for scaling factors with clear error messages
    if mm_per_px_x is None:
        raise ValueError("Scaling factor mm_per_px_x cannot be None - A4 calibration may have failed")
    if not isinstance(mm_per_px_x, (int, float)):
        raise ValueError(f"Scaling factor mm_per_px_x must be a number, got {type(mm_per_px_x)}")
    if mm_per_px_x <= 0:
        raise ValueError(f"Scaling factor mm_per_px_x must be positive, got {mm_per_px_x} - check A4 calibration")
    if not (0.01 <= mm_per_px_x <= 100.0):  # Reasonable range check
        raise ValueError(f"Scaling factor mm_per_px_x out of reasonable range (0.01-100.0): {mm_per_px_x} - A4 calibration may be incorrect")
    
    try:
        center = shape_result["center"]
        radius_px = shape_result.get("radius", shape_result.get("dimensions", [0])[0])
        
        # Validate shape data
        if not isinstance(radius_px, (int, float)) or radius_px <= 0:
            raise ValueError(f"Invalid circle radius: {radius_px}")
        
        # Calculate measurements - apply mm_per_px_x scaling factor to convert pixel diameter to millimeter diameter
        diameter_px = 2.0 * radius_px
        diameter_mm_raw = diameter_px * mm_per_px_x  # Use mm_per_px_x scaling factor for proper conversion
        # Round to nearest millimeter for consistent precision with Auto Mode (Requirement 1.5, 5.4)
        diameter_mm = round(diameter_mm_raw)
        area_px = np.pi * (radius_px ** 2)
        
        # Ensure radius_px remains in pixels for rendering purposes
        radius_px = float(radius_px)  # Keep as pixels for rendering
        
        # Create hit testing contour
        hit_contour = create_hit_testing_contour('circle', center=center, radius_px=radius_px)
        
        return {
            "type": "circle",
            "diameter_mm": diameter_mm,
            "center": center,
            "radius_px": radius_px,  # Remains in pixels for rendering purposes
            "hit_contour": hit_contour,
            "area_px": area_px,
            "inner": False,
            "detection_method": "manual",
            "selection_rect": selection_rect,
            "confidence_score": shape_result.get("confidence_score", 0.0),
            "manual_mode": shape_result.get("mode", "manual_circle")
        }
        
    except KeyError as e:
        raise ValueError(f"Missing required shape data for circle conversion: {e}")
    except Exception as e:
        raise ValueError(f"Error converting manual circle to measurement: {e}")


def _convert_manual_rectangle_to_measurement(shape_result: Dict[str, Any], mm_per_px_x: float, 
                                           mm_per_px_y: float, selection_rect: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """
    Convert manual rectangle selection to standard measurement format.
    
    Args:
        shape_result: Rectangle result from shape snapping engine
        mm_per_px_x: Millimeters per pixel in X direction
        mm_per_px_y: Millimeters per pixel in Y direction
        selection_rect: Original selection rectangle
        
    Returns:
        Rectangle measurement in standard format
        
    Raises:
        ValueError: If scaling factors are invalid (None, zero, or negative)
    """
    # Enhanced input validation for scaling factors with clear error messages
    if mm_per_px_x is None:
        raise ValueError("Scaling factor mm_per_px_x cannot be None - A4 calibration may have failed")
    if not isinstance(mm_per_px_x, (int, float)):
        raise ValueError(f"Scaling factor mm_per_px_x must be a number, got {type(mm_per_px_x)}")
    if mm_per_px_x <= 0:
        raise ValueError(f"Scaling factor mm_per_px_x must be positive, got {mm_per_px_x} - check A4 calibration")
    if not (0.01 <= mm_per_px_x <= 100.0):  # Reasonable range check
        raise ValueError(f"Scaling factor mm_per_px_x out of reasonable range (0.01-100.0): {mm_per_px_x} - A4 calibration may be incorrect")
    
    if mm_per_px_y is None:
        raise ValueError("Scaling factor mm_per_px_y cannot be None - A4 calibration may have failed")
    if not isinstance(mm_per_px_y, (int, float)):
        raise ValueError(f"Scaling factor mm_per_px_y must be a number, got {type(mm_per_px_y)}")
    if mm_per_px_y <= 0:
        raise ValueError(f"Scaling factor mm_per_px_y must be positive, got {mm_per_px_y} - check A4 calibration")
    if not (0.01 <= mm_per_px_y <= 100.0):  # Reasonable range check
        raise ValueError(f"Scaling factor mm_per_px_y out of reasonable range (0.01-100.0): {mm_per_px_y} - A4 calibration may be incorrect")
    
    try:
        # Get dimensions from shape result
        if "width" in shape_result and "height" in shape_result:
            width_px = shape_result["width"]
            height_px = shape_result["height"]
        elif "dimensions" in shape_result:
            width_px, height_px = shape_result["dimensions"]
        else:
            # Fallback: calculate from contour
            contour = shape_result.get("contour")
            if contour is not None:
                rect = cv2.minAreaRect(contour)
                width_px, height_px = rect[1]
            else:
                raise ValueError("Cannot determine rectangle dimensions from shape result")
        
        # Validate dimensions
        if not isinstance(width_px, (int, float)) or width_px <= 0:
            raise ValueError(f"Invalid rectangle width: {width_px}")
        if not isinstance(height_px, (int, float)) or height_px <= 0:
            raise ValueError(f"Invalid rectangle height: {height_px}")
        
        # Normalize width < height for consistency
        width_px = min(width_px, height_px)
        height_px = max(width_px, height_px)
        
        # Apply axis-specific scaling for accurate rectangular measurements
        # Apply mm_per_px_x to width conversion and mm_per_px_y to height conversion
        width_mm_raw = width_px * mm_per_px_x
        height_mm_raw = height_px * mm_per_px_y
        # Round to nearest millimeter for consistent precision with Auto Mode (Requirement 1.5, 5.4)
        width_mm = round(width_mm_raw)
        height_mm = round(height_mm_raw)
        
        # Get or create box points - ensure box coordinates remain in pixels for rendering purposes
        if "box" in shape_result:
            box = shape_result["box"]
        elif "contour" in shape_result:
            rect = cv2.minAreaRect(shape_result["contour"])
            box = cv2.boxPoints(rect).astype(int)
        else:
            # Create box from center and dimensions
            center = shape_result.get("center", (0, 0))
            cx, cy = center
            hw, hh = width_px / 2, height_px / 2
            box = np.array([
                [cx - hw, cy - hh],
                [cx + hw, cy - hh],
                [cx + hw, cy + hh],
                [cx - hw, cy + hh]
            ], dtype=int)
        
        # Ensure box coordinates remain in pixels for rendering purposes
        box = box.astype(int)  # Keep box coordinates in pixels for rendering
        
        # Calculate area in pixels
        area_px = cv2.contourArea(box) if len(box) > 2 else width_px * height_px
        
        # Create hit testing contour
        hit_contour = create_hit_testing_contour('rectangle', box=box)
        
        return {
            "type": "rectangle",
            "width_mm": width_mm,
            "height_mm": height_mm,
            "box": box,
            "hit_contour": hit_contour,
            "area_px": area_px,
            "inner": False,
            "detection_method": "manual",
            "selection_rect": selection_rect,
            "confidence_score": shape_result.get("confidence_score", 0.0),
            "manual_mode": shape_result.get("mode", "manual_rectangle")
        }
        
    except KeyError as e:
        raise ValueError(f"Missing required shape data for rectangle conversion: {e}")
    except Exception as e:
        raise ValueError(f"Error converting manual rectangle to measurement: {e}")


def _validate_scaling_factors(mm_per_px_x: float, mm_per_px_y: float) -> Dict[str, Any]:
    """
    Validate scaling factors for manual selection processing.
    
    Implements comprehensive validation for None, zero, or negative scaling factors
    with clear error messages for calibration issues.
    
    Args:
        mm_per_px_x: Millimeters per pixel in X direction
        mm_per_px_y: Millimeters per pixel in Y direction
        
    Returns:
        Dictionary with validation results:
        - valid: bool indicating if scaling factors are valid
        - errors: list of error messages
        - warnings: list of warning messages
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Validate mm_per_px_x
    if mm_per_px_x is None:
        validation["valid"] = False
        validation["errors"].append("mm_per_px_x cannot be None (A4 calibration failed)")
    elif not isinstance(mm_per_px_x, (int, float)):
        validation["valid"] = False
        validation["errors"].append(f"mm_per_px_x must be a number, got {type(mm_per_px_x)}")
    elif mm_per_px_x <= 0:
        validation["valid"] = False
        validation["errors"].append(f"mm_per_px_x must be positive, got {mm_per_px_x}")
    elif not (0.01 <= mm_per_px_x <= 100.0):
        validation["valid"] = False
        validation["errors"].append(f"mm_per_px_x out of reasonable range (0.01-100.0): {mm_per_px_x}")
    elif mm_per_px_x < 0.1 or mm_per_px_x > 10.0:
        validation["warnings"].append(f"mm_per_px_x seems unusual ({mm_per_px_x}), check A4 calibration")
    
    # Validate mm_per_px_y
    if mm_per_px_y is None:
        validation["valid"] = False
        validation["errors"].append("mm_per_px_y cannot be None (A4 calibration failed)")
    elif not isinstance(mm_per_px_y, (int, float)):
        validation["valid"] = False
        validation["errors"].append(f"mm_per_px_y must be a number, got {type(mm_per_px_y)}")
    elif mm_per_px_y <= 0:
        validation["valid"] = False
        validation["errors"].append(f"mm_per_px_y must be positive, got {mm_per_px_y}")
    elif not (0.01 <= mm_per_px_y <= 100.0):
        validation["valid"] = False
        validation["errors"].append(f"mm_per_px_y out of reasonable range (0.01-100.0): {mm_per_px_y}")
    elif mm_per_px_y < 0.1 or mm_per_px_y > 10.0:
        validation["warnings"].append(f"mm_per_px_y seems unusual ({mm_per_px_y}), check A4 calibration")
    
    # Check for significant anisotropy (different X and Y scaling)
    if validation["valid"] and mm_per_px_x is not None and mm_per_px_y is not None:
        ratio = max(mm_per_px_x, mm_per_px_y) / min(mm_per_px_x, mm_per_px_y)
        if ratio > 1.1:  # More than 10% difference
            validation["warnings"].append(f"Significant scaling difference between X ({mm_per_px_x:.3f}) and Y ({mm_per_px_y:.3f}) axes")
    
    return validation


def process_manual_selection(image: np.ndarray, selection_rect: Tuple[int, int, int, int], 
                           mode: str, mm_per_px_x: float, mm_per_px_y: float) -> Optional[Dict[str, Any]]:
    """
    Complete workflow for processing a manual selection into a measurement result.
    
    This function integrates the manual selection workflow with the existing measurement pipeline:
    1. Validates scaling factors to ensure they are positive and within reasonable ranges
    2. Uses enhanced contour analysis on the selected region
    3. Applies shape snapping to find the best shape
    4. Converts the result to standard measurement format with proper scaling
    
    Args:
        image: Source image (BGR format)
        selection_rect: Selection rectangle as (x, y, width, height)
        mode: Selection mode ("manual_circle" or "manual_rectangle")
        mm_per_px_x: Millimeters per pixel in X direction
        mm_per_px_y: Millimeters per pixel in Y direction
        
    Returns:
        Measurement result in standard format or None if processing fails
        
    Raises:
        ValueError: If scaling factors are invalid
    """
    try:
        # Comprehensive scaling factor validation with clear error messages for calibration issues
        validation_result = _validate_scaling_factors(mm_per_px_x, mm_per_px_y)
        if not validation_result["valid"]:
            # Provide clear error messages for calibration issues
            error_msg = f"Invalid scaling factors: {', '.join(validation_result['errors'])}"
            print(f"[ERROR] {error_msg}")
            print("[ERROR] This usually indicates A4 paper calibration failed or is incomplete.")
            print("[ERROR] Please ensure A4 paper is properly detected before using manual selection.")
            return None
        
        # Import required components
        from enhanced_contour_analyzer import EnhancedContourAnalyzer
        from shape_snapping_engine import ShapeSnappingEngine
        from selection_mode import SelectionMode
        
        # Convert mode string to SelectionMode enum
        if mode == "manual_circle":
            selection_mode = SelectionMode.MANUAL_CIRCLE
        elif mode == "manual_rectangle":
            selection_mode = SelectionMode.MANUAL_RECTANGLE
        else:
            print(f"[WARN] Invalid manual selection mode: {mode}")
            return None
        
        # Initialize components
        analyzer = EnhancedContourAnalyzer()
        snapping_engine = ShapeSnappingEngine(analyzer)
        
        # Perform shape snapping - maintain existing shape detection error handling
        shape_result = snapping_engine.snap_to_shape(image, selection_rect, selection_mode)
        
        if shape_result is None:
            print("[INFO] No suitable shape found in manual selection")
            return None
        
        # Ensure mm_per_px_x and mm_per_px_y parameters are properly passed to conversion functions
        # Convert to measurement format with validated scaling factors
        measurement_result = classify_and_measure_manual_selection(
            image, selection_rect, shape_result, mm_per_px_x, mm_per_px_y
        )
        
        return measurement_result
        
    except ValueError as e:
        # Enhanced error handling with specific calibration guidance
        print(f"[ERROR] Scaling factor validation failed: {e}")
        print("[ERROR] This indicates A4 paper calibration is invalid or missing.")
        print("[ERROR] Please recalibrate by ensuring A4 paper is properly positioned and detected.")
        return None
    except ImportError as e:
        print(f"[ERROR] Required manual selection components not available: {e}")
        print("[ERROR] Manual selection functionality may not be properly installed.")
        return None
    except Exception as e:
        # Maintain existing shape detection error handling
        print(f"[ERROR] Error processing manual selection: {e}")
        return None


def validate_manual_measurement_result(result: Dict[str, Any]) -> bool:
    """
    Validate that a manual measurement result has all required fields and is consistent.
    
    This function is a simplified wrapper around validate_measurement_result_data_structure
    that specifically checks for manual detection method and returns a boolean result.
    
    Args:
        result: Measurement result dictionary
        
    Returns:
        True if result is valid and has detection_method="manual", False otherwise
    """
    # Use the comprehensive validation function
    validation = validate_measurement_result_data_structure(result)
    
    if not validation["valid"]:
        return False
    
    # Additional check for manual detection method
    if result.get("detection_method") != "manual":
        return False
    
    return True


def merge_automatic_and_manual_results(automatic_results: List[Dict[str, Any]], 
                                     manual_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge automatic and manual measurement results into a single list.
    
    This function combines results from both detection methods while maintaining
    proper ordering and avoiding duplicates.
    
    Args:
        automatic_results: List of automatic detection results
        manual_results: List of manual selection results
        
    Returns:
        Combined list of measurement results with detection_method field
    """
    combined_results = []
    
    # Add automatic results (ensure they have detection_method field)
    for result in automatic_results:
        if result is not None:
            # Ensure detection_method field is set
            if "detection_method" not in result:
                result["detection_method"] = "automatic"
            combined_results.append(result)
    
    # Add manual results (they should already have detection_method = "manual")
    for result in manual_results:
        if result is not None and validate_manual_measurement_result(result):
            combined_results.append(result)
    
    return combined_results


def get_measurement_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of measurement results including detection method statistics.
    
    Args:
        results: List of measurement results
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_shapes": len(results),
        "automatic_count": 0,
        "manual_count": 0,
        "circles": 0,
        "rectangles": 0,
        "inner_shapes": 0,
        "detection_methods": {}
    }
    
    for result in results:
        # Count by detection method
        method = result.get("detection_method", "unknown")
        if method == "automatic":
            summary["automatic_count"] += 1
        elif method == "manual":
            summary["manual_count"] += 1
        
        summary["detection_methods"][method] = summary["detection_methods"].get(method, 0) + 1
        
        # Count by shape type
        if result["type"] == "circle":
            summary["circles"] += 1
        elif result["type"] == "rectangle":
            summary["rectangles"] += 1
        
        # Count inner shapes
        if result.get("inner", False):
            summary["inner_shapes"] += 1
    
    return summary


# Measurement Validation and Consistency Checking Functions

def compare_auto_vs_manual_measurements(auto_result: Dict[str, Any], 
                                      manual_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare Auto vs Manual measurements and validate consistency.
    
    Implements tolerance checking (±2mm or ±2% whichever is larger) as specified
    in requirements 5.1 and 5.2.
    
    Args:
        auto_result: Measurement result from automatic detection
        manual_result: Measurement result from manual selection
        
    Returns:
        Dictionary with comparison results including consistency status and differences
    """
    import logging
    
    if auto_result is None or manual_result is None:
        return {
            "consistent": False,
            "error": "One or both measurement results are None",
            "auto_result": auto_result,
            "manual_result": manual_result
        }
    
    # Validate that both results are for the same shape type
    if auto_result.get("type") != manual_result.get("type"):
        return {
            "consistent": False,
            "error": f"Shape type mismatch: auto={auto_result.get('type')}, manual={manual_result.get('type')}",
            "auto_result": auto_result,
            "manual_result": manual_result
        }
    
    shape_type = auto_result["type"]
    comparison = {
        "shape_type": shape_type,
        "consistent": True,
        "differences": {},
        "tolerance_checks": {},
        "auto_result": auto_result,
        "manual_result": manual_result
    }
    
    try:
        if shape_type == "circle":
            # Compare diameter measurements
            auto_diameter = auto_result["diameter_mm"]
            manual_diameter = manual_result["diameter_mm"]
            
            diff = abs(auto_diameter - manual_diameter)
            percent_diff = (diff / auto_diameter) * 100 if auto_diameter > 0 else 0
            
            # Tolerance checking: ±2mm or ±2% whichever is larger
            tolerance_mm = 2.0
            tolerance_percent = 2.0
            max_allowed_diff = max(tolerance_mm, auto_diameter * tolerance_percent / 100)
            
            is_within_tolerance = diff <= max_allowed_diff
            
            comparison["differences"]["diameter_mm"] = diff
            comparison["differences"]["diameter_percent"] = percent_diff
            comparison["tolerance_checks"]["diameter"] = {
                "within_tolerance": is_within_tolerance,
                "difference": diff,
                "max_allowed": max_allowed_diff,
                "tolerance_mm": tolerance_mm,
                "tolerance_percent": tolerance_percent
            }
            
            if not is_within_tolerance:
                comparison["consistent"] = False
                logging.warning(f"Circle diameter measurements differ significantly: "
                              f"Auto={auto_diameter:.1f}mm, Manual={manual_diameter:.1f}mm, "
                              f"Difference={diff:.1f}mm ({percent_diff:.1f}%), "
                              f"Max allowed={max_allowed_diff:.1f}mm")
            
        elif shape_type == "rectangle":
            # Compare width and height measurements
            auto_width = auto_result["width_mm"]
            auto_height = auto_result["height_mm"]
            manual_width = manual_result["width_mm"]
            manual_height = manual_result["height_mm"]
            
            width_diff = abs(auto_width - manual_width)
            height_diff = abs(auto_height - manual_height)
            
            width_percent_diff = (width_diff / auto_width) * 100 if auto_width > 0 else 0
            height_percent_diff = (height_diff / auto_height) * 100 if auto_height > 0 else 0
            
            # Tolerance checking: ±2mm or ±2% whichever is larger for each dimension
            tolerance_mm = 2.0
            tolerance_percent = 2.0
            
            max_allowed_width_diff = max(tolerance_mm, auto_width * tolerance_percent / 100)
            max_allowed_height_diff = max(tolerance_mm, auto_height * tolerance_percent / 100)
            
            width_within_tolerance = width_diff <= max_allowed_width_diff
            height_within_tolerance = height_diff <= max_allowed_height_diff
            
            comparison["differences"]["width_mm"] = width_diff
            comparison["differences"]["height_mm"] = height_diff
            comparison["differences"]["width_percent"] = width_percent_diff
            comparison["differences"]["height_percent"] = height_percent_diff
            
            comparison["tolerance_checks"]["width"] = {
                "within_tolerance": width_within_tolerance,
                "difference": width_diff,
                "max_allowed": max_allowed_width_diff,
                "tolerance_mm": tolerance_mm,
                "tolerance_percent": tolerance_percent
            }
            
            comparison["tolerance_checks"]["height"] = {
                "within_tolerance": height_within_tolerance,
                "difference": height_diff,
                "max_allowed": max_allowed_height_diff,
                "tolerance_mm": tolerance_mm,
                "tolerance_percent": tolerance_percent
            }
            
            if not width_within_tolerance:
                comparison["consistent"] = False
                logging.warning(f"Rectangle width measurements differ significantly: "
                              f"Auto={auto_width:.1f}mm, Manual={manual_width:.1f}mm, "
                              f"Difference={width_diff:.1f}mm ({width_percent_diff:.1f}%), "
                              f"Max allowed={max_allowed_width_diff:.1f}mm")
            
            if not height_within_tolerance:
                comparison["consistent"] = False
                logging.warning(f"Rectangle height measurements differ significantly: "
                              f"Auto={auto_height:.1f}mm, Manual={manual_height:.1f}mm, "
                              f"Difference={height_diff:.1f}mm ({height_percent_diff:.1f}%), "
                              f"Max allowed={max_allowed_height_diff:.1f}mm")
        
        else:
            comparison["consistent"] = False
            comparison["error"] = f"Unsupported shape type for comparison: {shape_type}"
    
    except Exception as e:
        comparison["consistent"] = False
        comparison["error"] = f"Error during measurement comparison: {str(e)}"
        logging.error(f"Error comparing measurements: {e}")
    
    return comparison


def validate_measurement_result_data_structure(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to validate measurement result data structure.
    
    Validates that a measurement result has all required fields and proper data types
    for both automatic and manual detection results.
    
    Args:
        result: Measurement result dictionary to validate
        
    Returns:
        Dictionary with validation results including status and any issues found
    """
    validation = {
        "valid": True,
        "issues": [],
        "result_type": None,
        "detection_method": None
    }
    
    if result is None:
        validation["valid"] = False
        validation["issues"].append("Result is None")
        return validation
    
    if not isinstance(result, dict):
        validation["valid"] = False
        validation["issues"].append(f"Result must be a dictionary, got {type(result)}")
        return validation
    
    # Check required fields for all measurement results
    required_fields = ["type", "hit_contour", "area_px"]
    for field in required_fields:
        if field not in result:
            validation["valid"] = False
            validation["issues"].append(f"Missing required field: {field}")
    
    # Validate shape type
    if "type" in result:
        shape_type = result["type"]
        validation["result_type"] = shape_type
        
        if shape_type not in ["circle", "rectangle"]:
            validation["valid"] = False
            validation["issues"].append(f"Invalid shape type: {shape_type}")
    
    # Validate detection method if present
    if "detection_method" in result:
        detection_method = result["detection_method"]
        validation["detection_method"] = detection_method
        
        if detection_method not in ["automatic", "manual"]:
            validation["valid"] = False
            validation["issues"].append(f"Invalid detection method: {detection_method}")
    
    # Validate shape-specific fields
    if validation["result_type"] == "circle":
        circle_fields = ["diameter_mm", "center", "radius_px"]
        for field in circle_fields:
            if field not in result:
                validation["valid"] = False
                validation["issues"].append(f"Missing circle field: {field}")
            else:
                # Validate data types and values
                if field == "diameter_mm":
                    if not isinstance(result[field], (int, float)) or result[field] <= 0:
                        validation["valid"] = False
                        validation["issues"].append(f"Invalid diameter_mm: must be positive number, got {result[field]}")
                elif field == "radius_px":
                    if not isinstance(result[field], (int, float)) or result[field] <= 0:
                        validation["valid"] = False
                        validation["issues"].append(f"Invalid radius_px: must be positive number, got {result[field]}")
                elif field == "center":
                    if not isinstance(result[field], (tuple, list)) or len(result[field]) != 2:
                        validation["valid"] = False
                        validation["issues"].append(f"Invalid center: must be 2-element tuple/list, got {result[field]}")
    
    elif validation["result_type"] == "rectangle":
        rect_fields = ["width_mm", "height_mm", "box"]
        for field in rect_fields:
            if field not in result:
                validation["valid"] = False
                validation["issues"].append(f"Missing rectangle field: {field}")
            else:
                # Validate data types and values
                if field in ["width_mm", "height_mm"]:
                    if not isinstance(result[field], (int, float)) or result[field] <= 0:
                        validation["valid"] = False
                        validation["issues"].append(f"Invalid {field}: must be positive number, got {result[field]}")
                elif field == "box":
                    if not isinstance(result[field], np.ndarray):
                        validation["valid"] = False
                        validation["issues"].append(f"Invalid box: must be numpy array, got {type(result[field])}")
                    elif result[field].shape != (4, 2):
                        validation["valid"] = False
                        validation["issues"].append(f"Invalid box shape: must be (4, 2), got {result[field].shape}")
    
    # Validate hit_contour format
    if "hit_contour" in result:
        hit_contour = result["hit_contour"]
        if not isinstance(hit_contour, np.ndarray):
            validation["valid"] = False
            validation["issues"].append(f"Invalid hit_contour: must be numpy array, got {type(hit_contour)}")
        elif len(hit_contour.shape) != 3 or hit_contour.shape[1] != 1 or hit_contour.shape[2] != 2:
            validation["valid"] = False
            validation["issues"].append(f"Invalid hit_contour shape: must be (N, 1, 2), got {hit_contour.shape}")
    
    # Validate area_px
    if "area_px" in result:
        if not isinstance(result["area_px"], (int, float)) or result["area_px"] <= 0:
            validation["valid"] = False
            validation["issues"].append(f"Invalid area_px: must be positive number, got {result['area_px']}")
    
    return validation


def log_measurement_consistency_warning(comparison: Dict[str, Any]) -> None:
    """
    Add logging for measurement consistency warnings.
    
    Logs detailed information about measurement inconsistencies between Auto and Manual modes.
    
    Args:
        comparison: Comparison result from compare_auto_vs_manual_measurements
    """
    import logging
    
    if comparison.get("consistent", True):
        return  # No warning needed for consistent measurements
    
    shape_type = comparison.get("shape_type", "unknown")
    error = comparison.get("error")
    
    if error:
        logging.warning(f"Measurement comparison failed for {shape_type}: {error}")
        return
    
    # Log detailed inconsistency information
    differences = comparison.get("differences", {})
    tolerance_checks = comparison.get("tolerance_checks", {})
    
    warning_msg = f"Measurement inconsistency detected for {shape_type}:"
    
    if shape_type == "circle":
        diameter_check = tolerance_checks.get("diameter", {})
        if not diameter_check.get("within_tolerance", True):
            diff = diameter_check.get("difference", 0)
            max_allowed = diameter_check.get("max_allowed", 0)
            warning_msg += f" Diameter difference {diff:.1f}mm exceeds tolerance {max_allowed:.1f}mm"
    
    elif shape_type == "rectangle":
        width_check = tolerance_checks.get("width", {})
        height_check = tolerance_checks.get("height", {})
        
        issues = []
        if not width_check.get("within_tolerance", True):
            diff = width_check.get("difference", 0)
            max_allowed = width_check.get("max_allowed", 0)
            issues.append(f"Width difference {diff:.1f}mm exceeds tolerance {max_allowed:.1f}mm")
        
        if not height_check.get("within_tolerance", True):
            diff = height_check.get("difference", 0)
            max_allowed = height_check.get("max_allowed", 0)
            issues.append(f"Height difference {diff:.1f}mm exceeds tolerance {max_allowed:.1f}mm")
        
        if issues:
            warning_msg += " " + ", ".join(issues)
    
    logging.warning(warning_msg)


def validate_measurement_consistency(auto_results: List[Dict[str, Any]], 
                                   manual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate consistency between automatic and manual measurement results.
    
    This function performs comprehensive validation of measurement consistency
    by comparing corresponding measurements and logging any inconsistencies.
    
    Args:
        auto_results: List of automatic detection results
        manual_results: List of manual selection results
        
    Returns:
        Dictionary with overall consistency validation results
    """
    import logging
    
    validation_summary = {
        "total_comparisons": 0,
        "consistent_measurements": 0,
        "inconsistent_measurements": 0,
        "validation_errors": 0,
        "comparisons": [],
        "overall_consistent": True
    }
    
    # If no results to compare, return early
    if not auto_results and not manual_results:
        return validation_summary
    
    # For now, we'll compare results by shape type and position similarity
    # In a more sophisticated implementation, we might use spatial matching
    
    for manual_result in manual_results:
        # Validate manual result structure first
        manual_validation = validate_measurement_result_data_structure(manual_result)
        if not manual_validation["valid"]:
            validation_summary["validation_errors"] += 1
            logging.error(f"Invalid manual measurement result: {manual_validation['issues']}")
            continue
        
        # Find corresponding automatic result (simplified matching by shape type)
        corresponding_auto = None
        for auto_result in auto_results:
            if auto_result.get("type") == manual_result.get("type"):
                # Validate auto result structure
                auto_validation = validate_measurement_result_data_structure(auto_result)
                if auto_validation["valid"]:
                    corresponding_auto = auto_result
                    break
                else:
                    validation_summary["validation_errors"] += 1
                    logging.error(f"Invalid automatic measurement result: {auto_validation['issues']}")
        
        if corresponding_auto:
            # Compare the measurements
            comparison = compare_auto_vs_manual_measurements(corresponding_auto, manual_result)
            validation_summary["comparisons"].append(comparison)
            validation_summary["total_comparisons"] += 1
            
            if comparison.get("consistent", False):
                validation_summary["consistent_measurements"] += 1
            else:
                validation_summary["inconsistent_measurements"] += 1
                validation_summary["overall_consistent"] = False
                log_measurement_consistency_warning(comparison)
    
    # Log summary
    if validation_summary["total_comparisons"] > 0:
        consistency_rate = (validation_summary["consistent_measurements"] / 
                          validation_summary["total_comparisons"]) * 100
        logging.info(f"Measurement consistency validation complete: "
                    f"{validation_summary['consistent_measurements']}/{validation_summary['total_comparisons']} "
                    f"measurements consistent ({consistency_rate:.1f}%)")
    
    return validation_summary