
import cv2
import numpy as np
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

def find_a4_quad(frame_bgr):
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
