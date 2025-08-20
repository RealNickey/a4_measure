
import cv2
import numpy as np
from config import (BINARY_BLOCK_SIZE, BINARY_C, MIN_OBJECT_AREA_MM2, PX_PER_MM,
                    CIRCULARITY_CUTOFF, RECT_ANGLE_EPS_DEG, DRAW_FONT, DRAW_THICKNESS)
from utils import draw_text

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
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        # Ignore contours touching the page edges (within margin)
        if x <= margin_px or y <= margin_px or (x+ww) >= (w - margin_px) or (y+hh) >= (h - margin_px):
            continue
        area = cv2.contourArea(cnt)
        if area > best_area:
            best = cnt
            best_area = area
    return best

def classify_and_measure(cnt, mm_per_px_x, mm_per_px_y):
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
        return {
            "type": "circle",
            "diameter_mm": diameter_mm,
            "center": (int(x), int(y)),
            "radius_px": radius
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

        # Optional right-angle sanity (not strictly enforced)
        return {
            "type": "rectangle",
            "width_mm": width_mm,
            "height_mm": height_mm,
            "box": box
        }

def annotate_result(a4_bgr, result, mm_per_px):
    out = a4_bgr.copy()
    if result["type"] == "circle":
        center = result["center"]
        radius_px = int(result["radius_px"])
        # Draw circle with thicker line
        cv2.circle(out, center, radius_px, (0, 255, 0), 3)
        # Draw diameter line horizontally
        x0 = center[0] - radius_px
        x1 = center[0] + radius_px
        y = center[1]
        cv2.line(out, (x0, y), (x1, y), (255, 0, 0), 3)
        
        # Draw dimension text with background for better visibility
        text = f"Diameter: {result['diameter_mm']:.1f} mm"
        # Add background rectangle for text
        text_size = cv2.getTextSize(text, DRAW_FONT, 1.2, 3)[0]
        cv2.rectangle(out, (10, 10), (20 + text_size[0], 50), (255, 255, 255), -1)
        draw_text(out, text, (15, 35), (0, 0, 255), 1.2, 3)
        
        # Also draw text near the object
        center_text = f"{result['diameter_mm']:.1f}mm"
        text_pos = (center[0] - 40, center[1] - radius_px - 10)
        cv2.rectangle(out, (text_pos[0] - 5, text_pos[1] - 25), 
                     (text_pos[0] + 85, text_pos[1] + 5), (255, 255, 255), -1)
        draw_text(out, center_text, text_pos, (0, 0, 255), 0.8, 2)
    else:
        box = result["box"]
        cv2.drawContours(out, [box], 0, (0, 255, 0), 3)
        width_mm = result["width_mm"]
        height_mm = result["height_mm"]
        
        # Draw dimension text with background for better visibility
        text1 = f"Width: {width_mm:.1f} mm"
        text2 = f"Height: {height_mm:.1f} mm"
        
        # Add background rectangle for text
        text_size1 = cv2.getTextSize(text1, DRAW_FONT, 1.2, 3)[0]
        text_size2 = cv2.getTextSize(text2, DRAW_FONT, 1.2, 3)[0]
        max_width = max(text_size1[0], text_size2[0])
        cv2.rectangle(out, (10, 10), (20 + max_width, 90), (255, 255, 255), -1)
        draw_text(out, text1, (15, 35), (0, 0, 255), 1.2, 3)
        draw_text(out, text2, (15, 70), (0, 0, 255), 1.2, 3)
        
        # Draw dimension lines with measurements
        # Calculate center of box
        center_x = int(np.mean(box[:, 0]))
        center_y = int(np.mean(box[:, 1]))
        
        # Draw width dimension
        mid_left = ((box[0] + box[3]) / 2).astype(int)
        mid_right = ((box[1] + box[2]) / 2).astype(int)
        cv2.arrowedLine(out, tuple(mid_left), tuple(mid_right), (255, 0, 0), 2, tipLength=0.02)
        cv2.arrowedLine(out, tuple(mid_right), tuple(mid_left), (255, 0, 0), 2, tipLength=0.02)
        
        # Draw height dimension
        mid_top = ((box[0] + box[1]) / 2).astype(int)
        mid_bottom = ((box[2] + box[3]) / 2).astype(int)
        cv2.arrowedLine(out, tuple(mid_top), tuple(mid_bottom), (0, 0, 255), 2, tipLength=0.02)
        cv2.arrowedLine(out, tuple(mid_bottom), tuple(mid_top), (0, 0, 255), 2, tipLength=0.02)
    return out
