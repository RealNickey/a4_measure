
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from utils import order_points, approx_quad, polygon_area
from config import (CANNY_LOW, CANNY_HIGH, GAUSS_BLUR, ASPECT_MIN, ASPECT_MAX,
                    A4_WIDTH_MM, A4_HEIGHT_MM, PX_PER_MM, MIN_A4_AREA_RATIO, USE_CUDA_IF_AVAILABLE)

def try_cuda():
    try:
        return USE_CUDA_IF_AVAILABLE and hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

USE_CUDA = try_cuda()

def preprocess_edges(frame_gray):
    if USE_CUDA:
        gmat = cv2.cuda_GpuMat()
        gmat.upload(frame_gray)
        gmat = cv2.cuda.createGaussianFilter(gmat.type(), gmat.type(), (GAUSS_BLUR, GAUSS_BLUR), 0).apply(gmat)
        canny = cv2.cuda.createCannyEdgeDetector(CANNY_LOW, CANNY_HIGH).detect(gmat)
        edges = canny.download()
        return edges
    else:
        blur = cv2.GaussianBlur(frame_gray, (GAUSS_BLUR, GAUSS_BLUR), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        return edges

def refine_corners_subpixel(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Refine corner positions to sub-pixel accuracy.
    
    Args:
        gray: Grayscale image
        corners: Initial corner positions (4, 2) array
        
    Returns:
        Refined corner positions with sub-pixel accuracy
    """
    # cornerSubPix parameters: window size and termination criteria
    win_size = (5, 5)
    zero_zone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    
    # Reshape for cornerSubPix (needs float32 with shape (N, 1, 2))
    corners_input = corners.reshape(-1, 1, 2).astype(np.float32)
    
    # Refine corners
    refined = cv2.cornerSubPix(gray, corners_input, win_size, zero_zone, criteria)
    
    return refined.reshape(-1, 2)


def calculate_perspective_quality(quad: np.ndarray) -> float:
    """
    Calculate quality score for detected A4 perspective.
    
    Quality is based on:
    - How close the aspect ratio is to ideal A4 (1.414)
    - How rectangular the quadrilateral is (right angles)
    - How uniform the side lengths are
    
    Args:
        quad: Ordered quadrilateral points (4, 2)
        
    Returns:
        Quality score from 0.0 to 1.0 (higher is better)
    """
    (tl, tr, br, bl) = quad
    
    # Calculate side lengths
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    
    width = (widthA + widthB) * 0.5
    height = (heightA + heightB) * 0.5
    
    # Aspect ratio quality (ideal is sqrt(2) ≈ 1.414)
    ideal_ratio = np.sqrt(2)
    actual_ratio = max(width, height) / max(1.0, min(width, height))
    aspect_error = abs(actual_ratio - ideal_ratio) / ideal_ratio
    aspect_quality = max(0.0, 1.0 - aspect_error * 2.0)
    
    # Side length uniformity (parallel sides should be equal)
    width_uniformity = 1.0 - abs(widthA - widthB) / max(widthA, widthB, 1.0)
    height_uniformity = 1.0 - abs(heightA - heightB) / max(heightA, heightB, 1.0)
    uniformity_quality = (width_uniformity + height_uniformity) / 2.0
    
    # Check if angles are close to 90 degrees
    def angle_between_vectors(v1, v2):
        """Calculate angle between two vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    # Calculate angles at each corner
    v1 = tl - tr
    v2 = br - tr
    angle1 = angle_between_vectors(v1, v2)
    
    v1 = tr - br
    v2 = bl - br
    angle2 = angle_between_vectors(v1, v2)
    
    v1 = br - bl
    v2 = tl - bl
    angle3 = angle_between_vectors(v1, v2)
    
    v1 = bl - tl
    v2 = tr - tl
    angle4 = angle_between_vectors(v1, v2)
    
    # How close are angles to 90 degrees?
    angle_errors = [abs(90.0 - angle) for angle in [angle1, angle2, angle3, angle4]]
    avg_angle_error = np.mean(angle_errors)
    angle_quality = max(0.0, 1.0 - avg_angle_error / 45.0)  # 45 degrees tolerance
    
    # Combined quality score with weights
    quality = (aspect_quality * 0.4 + uniformity_quality * 0.3 + angle_quality * 0.3)
    
    return quality


def find_a4_quad(frame_bgr, enable_subpixel: bool = True) -> Optional[np.ndarray]:
    """
    Find A4 paper quadrilateral in frame with optional sub-pixel refinement.
    
    Args:
        frame_bgr: Input BGR frame
        enable_subpixel: Whether to apply sub-pixel corner refinement
        
    Returns:
        Ordered quadrilateral corners (4, 2) as float32 or None if not found
    """
    h, w = frame_bgr.shape[:2]
    area_frame = w * h

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = preprocess_edges(gray)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best = None
    best_area = 0

    for cnt in contours[:20]:
        area = cv2.contourArea(cnt)
        if area < MIN_A4_AREA_RATIO * area_frame:
            continue
        quad = approx_quad(cnt, epsilon_ratio=0.02)
        if quad is None:
            continue
        # Order and check aspect
        rect = order_points(quad.astype(np.float32))
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        width = (widthA + widthB) * 0.5
        height = (heightA + heightB) * 0.5
        if width < 10 or height < 10:
            continue
        ratio = max(width, height) / max(1.0, min(width, height))
        if ASPECT_MIN <= ratio <= ASPECT_MAX:
            if area > best_area:
                best = rect
                best_area = area

    # Apply sub-pixel refinement if enabled and quad was found
    if best is not None and enable_subpixel:
        best = refine_corners_subpixel(gray, best)
    
    return best  # None or (4,2) float32


def find_a4_quad_with_quality(frame_bgr, enable_subpixel: bool = True) -> Tuple[Optional[np.ndarray], float]:
    """
    Find A4 paper quadrilateral with quality score.
    
    Args:
        frame_bgr: Input BGR frame
        enable_subpixel: Whether to apply sub-pixel corner refinement
        
    Returns:
        Tuple of (quadrilateral corners or None, quality score)
    """
    quad = find_a4_quad(frame_bgr, enable_subpixel)
    
    if quad is None:
        return None, 0.0
    
    quality = calculate_perspective_quality(quad)
    return quad, quality

def warp_a4(frame_bgr, quad):
    # Decide orientation dynamically: map longer side of detected quad to 297mm,
    # shorter side to 210mm. This preserves aspect without vertical/horizontal stretching.
    q = quad.astype(np.float32)
    (tl, tr, br, bl) = q
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    est_w = (widthA + widthB) * 0.5
    est_h = (heightA + heightB) * 0.5

    # If the detected quad is wider than tall, treat as landscape A4 (297 x 210 mm)
    if est_w >= est_h:
        target_w_mm = A4_HEIGHT_MM  # 297mm
        target_h_mm = A4_WIDTH_MM   # 210mm
    else:
        target_w_mm = A4_WIDTH_MM   # 210mm
        target_h_mm = A4_HEIGHT_MM  # 297mm

    target_w = int(round(target_w_mm * PX_PER_MM))
    target_h = int(round(target_h_mm * PX_PER_MM))

    dst = np.array([[0, 0],
                    [target_w - 1, 0],
                    [target_w - 1, target_h - 1],
                    [0, target_h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
    return warped, M

def a4_scale_mm_per_px():
    # Because we warp to a fixed PX_PER_MM density, the scale is exact by construction.
    mm_per_px_x = 1.0 / PX_PER_MM
    mm_per_px_y = 1.0 / PX_PER_MM
    return mm_per_px_x, mm_per_px_y


class MultiFrameCalibration:
    """
    Multi-frame calibration for improved A4 detection accuracy.
    
    Collects multiple frames and selects the best one based on quality scores,
    or averages corner positions for improved accuracy.
    """
    
    def __init__(self, num_samples: int = 5, quality_threshold: float = 0.7):
        """
        Initialize multi-frame calibration.
        
        Args:
            num_samples: Number of frames to collect for calibration
            quality_threshold: Minimum quality score to accept a frame
        """
        self.num_samples = num_samples
        self.quality_threshold = quality_threshold
        self.frames = []
        self.quads = []
        self.qualities = []
        
    def add_frame(self, frame_bgr: np.ndarray, enable_subpixel: bool = True) -> bool:
        """
        Add a frame for calibration.
        
        Args:
            frame_bgr: Input BGR frame
            enable_subpixel: Whether to apply sub-pixel corner refinement
            
        Returns:
            True if frame was accepted, False otherwise
        """
        quad, quality = find_a4_quad_with_quality(frame_bgr, enable_subpixel)
        
        if quad is None or quality < self.quality_threshold:
            return False
        
        self.frames.append(frame_bgr.copy())
        self.quads.append(quad)
        self.qualities.append(quality)
        
        return True
    
    def is_ready(self) -> bool:
        """Check if enough samples have been collected."""
        return len(self.frames) >= self.num_samples
    
    def get_best_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Get the best frame based on quality score.
        
        Returns:
            Tuple of (best frame, best quad, best quality)
        """
        if not self.frames:
            return None, None, 0.0
        
        best_idx = np.argmax(self.qualities)
        return self.frames[best_idx], self.quads[best_idx], self.qualities[best_idx]
    
    def get_averaged_quad(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get averaged quadrilateral from all collected frames.
        
        Uses weighted averaging based on quality scores.
        
        Returns:
            Tuple of (averaged quad, average quality)
        """
        if not self.quads:
            return None, 0.0
        
        # Convert to numpy array for easier manipulation
        quads_array = np.array(self.quads)  # Shape: (n_samples, 4, 2)
        qualities_array = np.array(self.qualities)  # Shape: (n_samples,)
        
        # Normalize qualities to use as weights
        weights = qualities_array / np.sum(qualities_array)
        weights = weights.reshape(-1, 1, 1)  # Shape: (n_samples, 1, 1)
        
        # Weighted average of corner positions
        averaged_quad = np.sum(quads_array * weights, axis=0)
        avg_quality = np.mean(qualities_array)
        
        return averaged_quad.astype(np.float32), avg_quality
    
    def reset(self):
        """Reset calibration data."""
        self.frames.clear()
        self.quads.clear()
        self.qualities.clear()
    
    def get_sample_count(self) -> int:
        """Get the number of samples collected."""
        return len(self.frames)
    
    def get_quality_stats(self) -> Dict[str, float]:
        """
        Get quality statistics for collected samples.
        
        Returns:
            Dictionary with min, max, mean, and std of quality scores
        """
        if not self.qualities:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0
            }
        
        return {
            "min": float(np.min(self.qualities)),
            "max": float(np.max(self.qualities)),
            "mean": float(np.mean(self.qualities)),
            "std": float(np.std(self.qualities))
        }
