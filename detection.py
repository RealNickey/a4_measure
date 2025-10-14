
import cv2
import numpy as np
from utils import order_points, approx_quad, polygon_area, angle_between
from config import (CANNY_LOW, CANNY_HIGH, GAUSS_BLUR, ASPECT_MIN, ASPECT_MAX,
                    A4_WIDTH_MM, A4_HEIGHT_MM, PX_PER_MM, MIN_A4_AREA_RATIO, USE_CUDA_IF_AVAILABLE,
                    A4_CORNER_ANGLE_TOLERANCE, A4_MIN_CORNER_ANGLE, A4_MAX_CORNER_ANGLE,
                    A4_PERSPECTIVE_MAX_RATIO, A4_CONTOUR_COMPLEXITY_MAX)

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

def validate_corner_angles(quad):
    """
    Validate that the quadrilateral has approximately 90-degree corners.
    
    Args:
        quad: (4,2) array of corner points
        
    Returns:
        True if all corners are within acceptable range of 90 degrees
    """
    angles = []
    for i in range(4):
        # Get three consecutive points
        p1 = quad[i]
        p2 = quad[(i+1) % 4]
        p3 = quad[(i+2) % 4]
        
        # Create vectors from middle point
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        angle = angle_between(v1, v2)
        angles.append(angle)
    
    # Check if all angles are close to 90 degrees
    for angle in angles:
        if angle < A4_MIN_CORNER_ANGLE or angle > A4_MAX_CORNER_ANGLE:
            return False
    
    return True

def validate_perspective_distortion(quad):
    """
    Validate that perspective distortion is not too severe.
    Checks that opposite sides have similar lengths.
    
    Args:
        quad: (4,2) array of corner points (ordered: tl, tr, br, bl)
        
    Returns:
        True if perspective distortion is acceptable
    """
    (tl, tr, br, bl) = quad
    
    # Calculate lengths of opposite sides
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    
    # Check horizontal sides ratio
    if top > 1e-6 and bottom > 1e-6:
        h_ratio = max(top, bottom) / min(top, bottom)
        if h_ratio > A4_PERSPECTIVE_MAX_RATIO:
            return False
    
    # Check vertical sides ratio
    if left > 1e-6 and right > 1e-6:
        v_ratio = max(left, right) / min(left, right)
        if v_ratio > A4_PERSPECTIVE_MAX_RATIO:
            return False
    
    return True

def check_contour_hierarchy_simple(contour, all_contours, hierarchy, idx):
    """
    Check if contour is a good A4 candidate based on hierarchy.
    A4 should be a prominent outer contour without excessive internal complexity.
    
    Args:
        contour: The contour to check
        all_contours: List of all contours
        hierarchy: Contour hierarchy from cv2.findContours
        idx: Index of the contour in all_contours
        
    Returns:
        True if hierarchy suggests this is a valid A4 candidate
    """
    if hierarchy is None or len(hierarchy) == 0:
        return True  # No hierarchy info, can't validate
    
    # hierarchy[0][i] = [next, previous, first_child, parent]
    h = hierarchy[0][idx]
    
    # A4 paper should typically not have a parent (it's an outer contour)
    # However, we allow one level of nesting for flexibility
    parent_idx = h[3]
    if parent_idx >= 0:
        # Has a parent, check if the parent is much larger
        parent_area = cv2.contourArea(all_contours[parent_idx])
        this_area = cv2.contourArea(contour)
        if parent_area > 0 and this_area / parent_area < 0.8:
            # This contour is significantly smaller than parent, probably not A4
            return False
    
    # Check for excessive internal complexity
    # Count total area of child contours
    child_idx = h[2]
    child_area_total = 0
    child_count = 0
    
    while child_idx >= 0 and child_count < 100:  # limit iterations
        child_area_total += cv2.contourArea(all_contours[child_idx])
        child_count += 1
        child_idx = hierarchy[0][child_idx][0]  # next sibling
    
    # If children take up too much area, might be a complex shape, not clean A4
    this_area = cv2.contourArea(contour)
    if this_area > 0 and child_area_total / this_area > A4_CONTOUR_COMPLEXITY_MAX:
        return False
    
    return True

def score_a4_candidate(quad, area, frame_area, contour=None, all_contours=None, hierarchy=None, idx=None):
    """
    Score an A4 candidate based on multiple criteria.
    Higher score = better candidate.
    
    Args:
        quad: (4,2) ordered quadrilateral points
        area: contour area
        frame_area: total frame area
        contour: the contour (optional, for hierarchy check)
        all_contours: list of all contours (optional, for hierarchy check)
        hierarchy: contour hierarchy (optional, for hierarchy check)
        idx: contour index (optional, for hierarchy check)
        
    Returns:
        Float score (higher is better), or -1 if invalid
    """
    (tl, tr, br, bl) = quad
    
    # Calculate dimensions
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    width = (widthA + widthB) * 0.5
    height = (heightA + heightB) * 0.5
    
    if width < 10 or height < 10:
        return -1
    
    # Check aspect ratio
    ratio = max(width, height) / max(1.0, min(width, height))
    if ratio < ASPECT_MIN or ratio > ASPECT_MAX:
        return -1
    
    # Validate corner angles
    if not validate_corner_angles(quad):
        return -1
    
    # Validate perspective distortion
    if not validate_perspective_distortion(quad):
        return -1
    
    # Check hierarchy if available
    if contour is not None and all_contours is not None and hierarchy is not None and idx is not None:
        if not check_contour_hierarchy_simple(contour, all_contours, hierarchy, idx):
            return -1
    
    # Calculate score based on multiple factors
    score = 0.0
    
    # Factor 1: Area (larger is better, up to a point)
    area_ratio = area / frame_area
    if area_ratio >= MIN_A4_AREA_RATIO:
        score += min(area_ratio * 100, 50.0)  # Cap at 50 points
    
    # Factor 2: Aspect ratio proximity to ideal A4 (1.414)
    ideal_ratio = A4_HEIGHT_MM / A4_WIDTH_MM  # 1.414
    aspect_deviation = abs(ratio - ideal_ratio)
    aspect_score = max(0, 30.0 - aspect_deviation * 20.0)  # Up to 30 points
    score += aspect_score
    
    # Factor 3: Corner angle regularity (how close to 90 degrees)
    angle_score = 0.0
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i+1) % 4]
        p3 = quad[(i+2) % 4]
        v1 = p1 - p2
        v2 = p3 - p2
        angle = angle_between(v1, v2)
        angle_deviation = abs(angle - 90.0)
        angle_score += max(0, 5.0 - angle_deviation * 0.2)  # Up to 5 points per corner
    score += angle_score
    
    return score

def find_a4_quad(frame_bgr):
    """
    Find the A4 paper quadrilateral in the frame using enhanced multi-criteria validation.
    
    This function implements:
    - Corner angle validation (approximately 90 degrees)
    - Perspective distortion checking
    - Hierarchical contour filtering
    - Multi-criteria scoring to select the best candidate
    
    Args:
        frame_bgr: Input BGR frame
        
    Returns:
        (4,2) float32 array of ordered corner points (tl, tr, br, bl), or None if not found
    """
    h, w = frame_bgr.shape[:2]
    area_frame = w * h

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = preprocess_edges(gray)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    # Use RETR_TREE to get hierarchy information for better filtering
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Sort by area but keep original indices for hierarchy lookup
    contour_info = [(i, cnt, cv2.contourArea(cnt)) for i, cnt in enumerate(contours)]
    contour_info.sort(key=lambda x: x[2], reverse=True)

    best = None
    best_score = -1

    # Evaluate top candidates
    for idx, cnt, area in contour_info[:20]:
        if area < MIN_A4_AREA_RATIO * area_frame:
            continue
        
        # Try to approximate as quadrilateral
        quad = approx_quad(cnt, epsilon_ratio=0.02)
        if quad is None:
            continue
        
        # Order points consistently
        rect = order_points(quad.astype(np.float32))
        
        # Score this candidate using enhanced multi-criteria validation
        score = score_a4_candidate(rect, area, area_frame, cnt, contours, hierarchy, idx)
        
        if score > best_score:
            best = rect
            best_score = score

    return best  # None or (4,2) float32

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
