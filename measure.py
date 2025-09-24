import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from config import (BINARY_BLOCK_SIZE, BINARY_C, MIN_OBJECT_AREA_MM2, PX_PER_MM,
                    CIRCULARITY_CUTOFF, RECT_ANGLE_EPS_DEG, DRAW_FONT, DRAW_THICKNESS)
from utils import draw_text
from hit_testing import create_hit_testing_contour

def _area_px2_from_mm2(mm2):
    # Convert mm^2 to px^2 given PX_PER_MM
    return (mm2 * (PX_PER_MM ** 2))

def segment_object(a4_bgr):
    # Expect a light background (paper). Use adaptive threshold to get dark object(s).
    gray = cv2.cvtColor(a4_bgr, cv2.COLOR_BGR2GRAY)
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

# Hit testing polygon creation functions moved to hit_testing.py module

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

    if circularity >= CIRCULARITY_CUTOFF:
        # Circle-like
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        diameter_px = 2.0 * radius
        diameter_mm = diameter_px * mm_per_px_x  # assume isotropic scale
        center = (int(x), int(y))
        hit_contour = create_hit_testing_contour('circle', center=center, radius_px=radius)
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
        width_mm = width_px * mm_per_px_x
        height_mm = height_px * mm_per_px_y
        box = cv2.boxPoints(rect).astype(int)
        hit_contour = create_hit_testing_contour('rectangle', box=box)

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
    diameter_mm = (2.0 * r) * mm_per_px_x
    hit_contour = create_hit_testing_contour('circle', center=center, radius_px=float(r))
    area_px = np.pi * (r ** 2)
    
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
    width_mm = width_px * mm_per_px_x
    height_mm = height_px * mm_per_px_y
    box = cv2.boxPoints(rect).astype(int)
    # map box to full image coords
    box[:, 0] += x
    box[:, 1] += y
    hit_contour = create_hit_testing_contour('rectangle', box=box)
    area_px = cv2.contourArea(box)
    
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
    if shape_result is None or "type" not in shape_result:
        return None
    
    shape_type = shape_result["type"]
    
    try:
        if shape_type == "circle":
            return _convert_manual_circle_to_measurement(shape_result, mm_per_px_x, selection_rect)
        elif shape_type == "rectangle":
            return _convert_manual_rectangle_to_measurement(shape_result, mm_per_px_x, mm_per_px_y, selection_rect)
        else:
            return None
    except Exception as e:
        print(f"[WARN] Error converting manual selection to measurement: {e}")
        return None


def _convert_manual_circle_to_measurement(shape_result: Dict[str, Any], mm_per_px: float, 
                                        selection_rect: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """
    Convert manual circle selection to standard measurement format.
    
    Args:
        shape_result: Circle result from shape snapping engine
        mm_per_px: Millimeters per pixel conversion factor
        selection_rect: Original selection rectangle
        
    Returns:
        Circle measurement in standard format
    """
    center = shape_result["center"]
    radius_px = shape_result.get("radius", shape_result.get("dimensions", [0])[0])
    
    # Calculate measurements
    diameter_px = 2.0 * radius_px
    diameter_mm = diameter_px * mm_per_px
    area_px = np.pi * (radius_px ** 2)
    
    # Create hit testing contour
    hit_contour = create_hit_testing_contour('circle', center=center, radius_px=radius_px)
    
    return {
        "type": "circle",
        "diameter_mm": diameter_mm,
        "center": center,
        "radius_px": radius_px,
        "hit_contour": hit_contour,
        "area_px": area_px,
        "inner": False,
        "detection_method": "manual",
        "selection_rect": selection_rect,
        "confidence_score": shape_result.get("confidence_score", 0.0),
        "manual_mode": shape_result.get("mode", "manual_circle")
    }


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
    """
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
    
    # Normalize width < height for consistency
    width_px = min(width_px, height_px)
    height_px = max(width_px, height_px)
    
    # Convert to millimeters
    width_mm = width_px * mm_per_px_x
    height_mm = height_px * mm_per_px_y
    
    # Get or create box points
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
    
    # Calculate area
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


def process_manual_selection(image: np.ndarray, selection_rect: Tuple[int, int, int, int], 
                           mode: str, mm_per_px_x: float, mm_per_px_y: float) -> Optional[Dict[str, Any]]:
    """
    Complete workflow for processing a manual selection into a measurement result.
    
    This function integrates the manual selection workflow with the existing measurement pipeline:
    1. Uses enhanced contour analysis on the selected region
    2. Applies shape snapping to find the best shape
    3. Converts the result to standard measurement format
    
    Args:
        image: Source image (BGR format)
        selection_rect: Selection rectangle as (x, y, width, height)
        mode: Selection mode ("manual_circle" or "manual_rectangle")
        mm_per_px_x: Millimeters per pixel in X direction
        mm_per_px_y: Millimeters per pixel in Y direction
        
    Returns:
        Measurement result in standard format or None if processing fails
    """
    try:
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
        
        # Perform shape snapping
        shape_result = snapping_engine.snap_to_shape(image, selection_rect, selection_mode)
        
        if shape_result is None:
            print("[INFO] No suitable shape found in manual selection")
            return None
        
        # Convert to measurement format
        measurement_result = classify_and_measure_manual_selection(
            image, selection_rect, shape_result, mm_per_px_x, mm_per_px_y
        )
        
        return measurement_result
        
    except ImportError as e:
        print(f"[ERROR] Required manual selection components not available: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Error processing manual selection: {e}")
        return None


def validate_manual_measurement_result(result: Dict[str, Any]) -> bool:
    """
    Validate that a manual measurement result has all required fields and is consistent.
    
    Args:
        result: Measurement result dictionary
        
    Returns:
        True if result is valid, False otherwise
    """
    if result is None:
        return False
    
    # Check required fields
    required_fields = [
        "type", "detection_method", "hit_contour", "area_px"
    ]
    
    for field in required_fields:
        if field not in result:
            return False
    
    # Validate detection method
    if result["detection_method"] != "manual":
        return False
    
    # Validate shape-specific fields
    if result["type"] == "circle":
        circle_fields = ["diameter_mm", "center", "radius_px"]
        for field in circle_fields:
            if field not in result:
                return False
        
        # Validate circle measurements
        if result["diameter_mm"] <= 0 or result["radius_px"] <= 0:
            return False
            
    elif result["type"] == "rectangle":
        rect_fields = ["width_mm", "height_mm", "box"]
        for field in rect_fields:
            if field not in result:
                return False
        
        # Validate rectangle measurements
        if result["width_mm"] <= 0 or result["height_mm"] <= 0:
            return False
            
    else:
        return False
    
    # Validate hit contour format
    hit_contour = result["hit_contour"]
    if not isinstance(hit_contour, np.ndarray):
        return False
    
    if len(hit_contour.shape) != 3 or hit_contour.shape[1] != 1 or hit_contour.shape[2] != 2:
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