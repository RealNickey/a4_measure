
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
    y_text = 20
    idx = 0
    for res in results:
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
            text = f"#{idx+1} Circle: D={res['diameter_mm']:.1f}mm (R={res['diameter_mm']/2:.1f}mm)"
            # Background
            ts = cv2.getTextSize(text, DRAW_FONT, 0.9, 2)[0]
            cv2.rectangle(out, (10, y_text-18), (20 + ts[0], y_text+6), (255,255,255), -1)
            draw_text(out, text, (15, y_text), (0,0,0), 0.9, 2)
            # Near object label
            near = (center[0] + 5, max(20, center[1] - radius_px - 8))
            draw_text(out, f"D={res['diameter_mm']:.0f}mm", near, color, 0.8, 2)
            y_text += 28
        else:
            box = res["box"]
            cv2.drawContours(out, [box], 0, color, 3)
            text1 = f"#{idx+1} Rect: W={res['width_mm']:.1f}mm, H={res['height_mm']:.1f}mm"
            ts = cv2.getTextSize(text1, DRAW_FONT, 0.9, 2)[0]
            cv2.rectangle(out, (10, y_text-18), (20 + ts[0], y_text+6), (255,255,255), -1)
            draw_text(out, text1, (15, y_text), (0,0,0), 0.9, 2)
            y_text += 28
            # Dimension arrows
            mid_left = ((box[0] + box[3]) / 2).astype(int)
            mid_right = ((box[1] + box[2]) / 2).astype(int)
            cv2.arrowedLine(out, tuple(mid_left), tuple(mid_right), color, 2, tipLength=0.02)
            cv2.arrowedLine(out, tuple(mid_right), tuple(mid_left), color, 2, tipLength=0.02)
            mid_top = ((box[0] + box[1]) / 2).astype(int)
            mid_bottom = ((box[2] + box[3]) / 2).astype(int)
            cv2.arrowedLine(out, tuple(mid_top), tuple(mid_bottom), color, 2, tipLength=0.02)
            cv2.arrowedLine(out, tuple(mid_bottom), tuple(mid_top), color, 2, tipLength=0.02)
        idx += 1
    return out
