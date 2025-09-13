
import cv2
import numpy as np

def order_points(pts):
    # pts: (4,2) array
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def polygon_area(poly):
    # Shoelace formula
    x = poly[:,0]
    y = poly[:,1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def angle_between(v1, v2):
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def draw_text(img, text, org, color=(0,255,0), scale=0.7, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def approx_quad(contour, epsilon_ratio=0.02):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    if len(approx) == 4:
        return approx.reshape(4,2)
    return None

# Coordinate transformation utilities for interactive inspect mode
def calculate_display_scale(original_height, display_height):
    """Calculate scaling factor based on display window size"""
    return display_height / original_height

def display_to_original_coords(display_x, display_y, scale_factor):
    """Convert display window coordinates to original image coordinates"""
    original_x = int(display_x / scale_factor)
    original_y = int(display_y / scale_factor)
    return original_x, original_y

def original_to_display_coords(original_x, original_y, scale_factor):
    """Convert original image coordinates to display window coordinates"""
    display_x = int(original_x * scale_factor)
    display_y = int(original_y * scale_factor)
    return display_x, display_y

def is_point_in_bounds(x, y, image_width, image_height):
    """Check if coordinates are within valid image bounds"""
    return 0 <= x < image_width and 0 <= y < image_height

def clamp_coordinates(x, y, image_width, image_height):
    """Clamp coordinates to valid image bounds"""
    clamped_x = max(0, min(x, image_width - 1))
    clamped_y = max(0, min(y, image_height - 1))
    return clamped_x, clamped_y

def get_display_dimensions(original_width, original_height, display_height):
    """Calculate display window dimensions maintaining aspect ratio"""
    scale = calculate_display_scale(original_height, display_height)
    display_width = int(original_width * scale)
    return display_width, display_height, scale
