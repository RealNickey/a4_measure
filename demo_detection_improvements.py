#!/usr/bin/env python3
"""
Demo script to showcase detection accuracy improvements.

This script demonstrates:
1. Sub-pixel corner refinement
2. Perspective quality scoring
3. Multi-frame calibration
4. Measurement confidence scoring
"""

import cv2
import numpy as np
from detection import (
    find_a4_quad, find_a4_quad_with_quality, 
    calculate_perspective_quality, MultiFrameCalibration
)
from measure import classify_and_measure, calculate_shape_confidence
from config import (
    ENABLE_SUBPIXEL_REFINEMENT, MIN_DETECTION_QUALITY,
    MULTI_FRAME_CALIBRATION_SAMPLES, CALIBRATION_QUALITY_THRESHOLD
)


def create_test_a4_image():
    """Create a synthetic A4 paper image for testing."""
    # Create a white canvas
    height, width = 600, 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Draw an A4-like rectangle with some perspective
    # A4 aspect ratio is approximately 1.414
    a4_width = 400
    a4_height = int(a4_width * 1.414)
    
    # Define corners with slight perspective distortion
    tl = (100, 50)
    tr = (100 + a4_width, 60)
    br = (100 + a4_width - 10, 60 + a4_height)
    bl = (90, 50 + a4_height)
    
    # Draw white A4 paper
    pts = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    cv2.polylines(img, [pts], True, (0, 0, 0), 3)
    
    # Add some objects on the paper
    # Circle
    circle_center = (int((tl[0] + tr[0]) / 2), int((tl[1] + bl[1]) / 2) - 80)
    cv2.circle(img, circle_center, 50, (100, 100, 100), -1)
    cv2.circle(img, circle_center, 50, (0, 0, 0), 2)
    
    # Rectangle
    rect_center = (int((tl[0] + tr[0]) / 2), int((tl[1] + bl[1]) / 2) + 80)
    rect_pts = np.array([
        [rect_center[0] - 60, rect_center[1] - 40],
        [rect_center[0] + 60, rect_center[1] - 40],
        [rect_center[0] + 60, rect_center[1] + 40],
        [rect_center[0] - 60, rect_center[1] + 40]
    ], dtype=np.int32)
    cv2.fillPoly(img, [rect_pts], (80, 80, 80))
    cv2.polylines(img, [rect_pts], True, (0, 0, 0), 2)
    
    return img


def demo_basic_detection():
    """Demo basic A4 detection without improvements."""
    print("\n" + "="*70)
    print("DEMO 1: Basic A4 Detection (Without Sub-pixel Refinement)")
    print("="*70)
    
    img = create_test_a4_image()
    
    # Detect without sub-pixel refinement
    quad = find_a4_quad(img, enable_subpixel=False)
    
    if quad is not None:
        print("✓ A4 paper detected")
        print(f"  Corner positions (pixel accuracy):")
        for i, corner in enumerate(quad):
            print(f"    Corner {i+1}: ({corner[0]:.2f}, {corner[1]:.2f})")
        
        # Draw detected quad
        display = img.copy()
        for i in range(4):
            p1 = tuple(quad[i].astype(int))
            p2 = tuple(quad[(i+1)%4].astype(int))
            cv2.line(display, p1, p2, (0, 255, 0), 2)
            cv2.circle(display, p1, 5, (255, 0, 0), -1)
        
        cv2.putText(display, "Basic Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return display
    else:
        print("✗ A4 paper not detected")
        return img


def demo_subpixel_detection():
    """Demo A4 detection with sub-pixel refinement."""
    print("\n" + "="*70)
    print("DEMO 2: A4 Detection WITH Sub-pixel Refinement")
    print("="*70)
    
    img = create_test_a4_image()
    
    # Detect with sub-pixel refinement
    quad = find_a4_quad(img, enable_subpixel=True)
    
    if quad is not None:
        print("✓ A4 paper detected with sub-pixel accuracy")
        print(f"  Corner positions (sub-pixel accuracy):")
        for i, corner in enumerate(quad):
            print(f"    Corner {i+1}: ({corner[0]:.4f}, {corner[1]:.4f})")
        
        # Draw detected quad
        display = img.copy()
        for i in range(4):
            p1 = tuple(quad[i].astype(int))
            p2 = tuple(quad[(i+1)%4].astype(int))
            cv2.line(display, p1, p2, (0, 255, 0), 2)
            cv2.circle(display, p1, 5, (255, 0, 0), -1)
        
        cv2.putText(display, "Sub-pixel Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return display
    else:
        print("✗ A4 paper not detected")
        return img


def demo_quality_scoring():
    """Demo perspective quality scoring."""
    print("\n" + "="*70)
    print("DEMO 3: Perspective Quality Scoring")
    print("="*70)
    
    img = create_test_a4_image()
    
    # Detect with quality score
    quad, quality = find_a4_quad_with_quality(img, enable_subpixel=True)
    
    if quad is not None:
        print(f"✓ A4 paper detected")
        print(f"  Detection Quality: {quality:.1%}")
        
        # Quality assessment
        if quality >= 0.8:
            quality_text = "Excellent"
            quality_color = (0, 255, 0)
        elif quality >= MIN_DETECTION_QUALITY:
            quality_text = "Good"
            quality_color = (0, 255, 255)
        else:
            quality_text = "Poor - Adjust position/lighting"
            quality_color = (0, 0, 255)
        
        print(f"  Quality Assessment: {quality_text}")
        
        # Draw detected quad with quality indicator
        display = img.copy()
        for i in range(4):
            p1 = tuple(quad[i].astype(int))
            p2 = tuple(quad[(i+1)%4].astype(int))
            cv2.line(display, p1, p2, quality_color, 2)
            cv2.circle(display, p1, 5, (255, 0, 0), -1)
        
        # Add quality text
        cv2.putText(display, f"Quality: {quality:.0%} - {quality_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, quality_color, 2)
        
        return display
    else:
        print("✗ A4 paper not detected")
        return img


def demo_multi_frame_calibration():
    """Demo multi-frame calibration."""
    print("\n" + "="*70)
    print("DEMO 4: Multi-Frame Calibration")
    print("="*70)
    
    # Create multiple slightly different frames (simulating camera jitter)
    calibration = MultiFrameCalibration(
        num_samples=MULTI_FRAME_CALIBRATION_SAMPLES,
        quality_threshold=CALIBRATION_QUALITY_THRESHOLD
    )
    
    print(f"Collecting {MULTI_FRAME_CALIBRATION_SAMPLES} calibration frames...")
    
    for i in range(MULTI_FRAME_CALIBRATION_SAMPLES + 2):  # Try a few extra
        # Create test image with slight variations
        img = create_test_a4_image()
        
        # Add some noise to simulate real-world conditions
        noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Try to add frame
        accepted = calibration.add_frame(img, enable_subpixel=True)
        
        if accepted:
            print(f"  Frame {calibration.get_sample_count()}/{MULTI_FRAME_CALIBRATION_SAMPLES} accepted")
        else:
            print(f"  Frame rejected (quality too low)")
        
        if calibration.is_ready():
            break
    
    if calibration.is_ready():
        # Get best frame
        best_frame, best_quad, best_quality = calibration.get_best_frame()
        print(f"\n✓ Multi-frame calibration complete")
        print(f"  Best frame quality: {best_quality:.1%}")
        
        # Get averaged quad
        avg_quad, avg_quality = calibration.get_averaged_quad()
        print(f"  Averaged calibration quality: {avg_quality:.1%}")
        
        # Get quality statistics
        stats = calibration.get_quality_stats()
        print(f"\n  Quality Statistics:")
        print(f"    Min: {stats['min']:.1%}")
        print(f"    Max: {stats['max']:.1%}")
        print(f"    Mean: {stats['mean']:.1%}")
        print(f"    Std Dev: {stats['std']:.3f}")
        
        # Display best frame
        display = best_frame.copy()
        for i in range(4):
            p1 = tuple(best_quad[i].astype(int))
            p2 = tuple(best_quad[(i+1)%4].astype(int))
            cv2.line(display, p1, p2, (0, 255, 0), 2)
            cv2.circle(display, p1, 5, (255, 0, 0), -1)
        
        cv2.putText(display, f"Best of {calibration.get_sample_count()} frames", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return display
    else:
        print("✗ Could not collect enough quality frames")
        return create_test_a4_image()


def demo_confidence_scoring():
    """Demo measurement confidence scoring."""
    print("\n" + "="*70)
    print("DEMO 5: Measurement Confidence Scoring")
    print("="*70)
    
    # Create a test contour (circle)
    center = (200, 200)
    radius = 50
    
    # Perfect circle
    angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
    circle_points = np.array([(int(center[0] + radius * np.cos(a)),
                              int(center[1] + radius * np.sin(a))) for a in angles])
    perfect_circle = circle_points.reshape(-1, 1, 2).astype(np.int32)
    
    # Calculate circularity for perfect circle
    area = cv2.contourArea(perfect_circle)
    peri = cv2.arcLength(perfect_circle, True)
    circularity = 4.0 * np.pi * area / (peri*peri + 1e-9)
    
    confidence = calculate_shape_confidence(perfect_circle, circularity, "circle")
    
    print(f"Perfect Circle:")
    print(f"  Circularity: {circularity:.3f}")
    print(f"  Confidence: {confidence:.1%}")
    
    # Imperfect circle (noisy)
    noisy_circle = perfect_circle.copy()
    noise = np.random.randint(-5, 5, noisy_circle.shape)
    noisy_circle = noisy_circle + noise
    
    area_noisy = cv2.contourArea(noisy_circle)
    peri_noisy = cv2.arcLength(noisy_circle, True)
    circularity_noisy = 4.0 * np.pi * area_noisy / (peri_noisy*peri_noisy + 1e-9)
    
    confidence_noisy = calculate_shape_confidence(noisy_circle, circularity_noisy, "circle")
    
    print(f"\nNoisy Circle:")
    print(f"  Circularity: {circularity_noisy:.3f}")
    print(f"  Confidence: {confidence_noisy:.1%}")
    
    # Create rectangle
    rect_points = np.array([
        [300, 150], [400, 150], [400, 250], [300, 250]
    ], dtype=np.int32).reshape(-1, 1, 2)
    
    area_rect = cv2.contourArea(rect_points)
    peri_rect = cv2.arcLength(rect_points, True)
    circularity_rect = 4.0 * np.pi * area_rect / (peri_rect*peri_rect + 1e-9)
    
    confidence_rect = calculate_shape_confidence(rect_points, circularity_rect, "rectangle")
    
    print(f"\nRectangle:")
    print(f"  Circularity: {circularity_rect:.3f}")
    print(f"  Confidence: {confidence_rect:.1%}")
    
    # Visualize
    img = np.ones((400, 500, 3), dtype=np.uint8) * 200
    
    cv2.drawContours(img, [perfect_circle], 0, (0, 255, 0), 2)
    cv2.putText(img, f"Conf: {confidence:.0%}", (center[0]-30, center[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.drawContours(img, [noisy_circle], 0, (0, 0, 255), 2)
    
    cv2.drawContours(img, [rect_points], 0, (255, 0, 0), 2)
    cv2.putText(img, f"Conf: {confidence_rect:.0%}", (330, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.putText(img, "Confidence Scoring Demo", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("DETECTION ACCURACY IMPROVEMENTS - DEMONSTRATION")
    print("="*70)
    print("\nThis demo showcases the following improvements:")
    print("1. Sub-pixel corner refinement for A4 detection")
    print("2. Perspective quality scoring")
    print("3. Multi-frame calibration")
    print("4. Measurement confidence scoring")
    print("\nPress any key to advance through demos, ESC to exit.")
    
    demos = [
        ("Basic Detection", demo_basic_detection),
        ("Sub-pixel Detection", demo_subpixel_detection),
        ("Quality Scoring", demo_quality_scoring),
        ("Multi-frame Calibration", demo_multi_frame_calibration),
        ("Confidence Scoring", demo_confidence_scoring)
    ]
    
    for demo_name, demo_func in demos:
        try:
            display = demo_func()
            
            # Show the demo
            cv2.namedWindow(demo_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(demo_name, 800, 600)
            cv2.imshow(demo_name, display)
            
            print(f"\n[Press any key to continue, ESC to exit]")
            key = cv2.waitKey(0)
            cv2.destroyWindow(demo_name)
            
            if key == 27:  # ESC
                print("\nDemo interrupted by user.")
                break
                
        except Exception as e:
            print(f"\n✗ Error in {demo_name}: {e}")
            import traceback
            traceback.print_exc()
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Improvements Demonstrated:")
    print("✓ Sub-pixel corner refinement improves detection precision")
    print("✓ Quality scoring helps identify good vs. poor detections")
    print("✓ Multi-frame calibration averages out noise and jitter")
    print("✓ Confidence scoring indicates measurement reliability")
    print("\nThese improvements address the core issues from the problem statement:")
    print("• Improved detection accuracy with sub-pixel precision")
    print("• Better handling of varying lighting and camera angles")
    print("• Quantified quality metrics for user feedback")
    print("• Reduced measurement variance through multi-frame averaging")


if __name__ == "__main__":
    main()
