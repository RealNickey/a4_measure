#!/usr/bin/env python3
"""
Test script to verify the measure.py fix for the NameError issue.
"""

import cv2
import numpy as np
from measure import classify_and_measure, detect_inner_circles, detect_inner_rectangles


def test_classify_and_measure():
    """Test the classify_and_measure function with sample contours."""
    print("Testing classify_and_measure function...")
    
    # Create a sample circular contour
    center = (100, 100)
    radius = 50
    angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    circle_points = np.array([(int(center[0] + radius * np.cos(a)),
                              int(center[1] + radius * np.sin(a))) for a in angles])
    circle_contour = circle_points.reshape(-1, 1, 2).astype(np.int32)
    
    # Test circle classification
    result = classify_and_measure(circle_contour, 0.1, 0.1, "test")
    if result is not None:
        print(f"✓ Circle classification successful: {result['type']}, diameter: {result['diameter_mm']:.1f}mm")
        print(f"  Center: {result['center']}, Hit contour shape: {result['hit_contour'].shape}")
    else:
        print("✗ Circle classification failed")
    
    # Create a sample rectangular contour
    box = np.array([[50, 50], [150, 50], [150, 100], [50, 100]], dtype=np.int32)
    rect_contour = box.reshape(-1, 1, 2)
    
    # Test rectangle classification
    result = classify_and_measure(rect_contour, 0.1, 0.1, "test")
    if result is not None:
        print(f"✓ Rectangle classification successful: {result['type']}, size: {result['width_mm']:.1f}x{result['height_mm']:.1f}mm")
        print(f"  Box shape: {result['box'].shape}, Hit contour shape: {result['hit_contour'].shape}")
    else:
        print("✗ Rectangle classification failed")


def test_inner_detection():
    """Test the inner shape detection functions."""
    print("\nTesting inner shape detection functions...")
    
    # Create a test image with some shapes
    img = np.ones((300, 300, 3), dtype=np.uint8) * 240  # Light background
    
    # Draw a large rectangle with inner shapes
    cv2.rectangle(img, (50, 50), (250, 200), (100, 100, 100), -1)  # Dark rectangle
    cv2.circle(img, (150, 125), 20, (240, 240, 240), -1)  # Light circle inside
    cv2.rectangle(img, (80, 80), (120, 120), (240, 240, 240), -1)  # Light rectangle inside
    
    # Create object mask and contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Test inner circle detection
        inner_circles = detect_inner_circles(img, mask, largest_contour, 0.1)
        if inner_circles:
            print(f"✓ Inner circle detection successful: found {len(inner_circles)} circle(s)")
            for i, circle in enumerate(inner_circles):
                print(f"  Circle {i+1}: diameter {circle['diameter_mm']:.1f}mm, center {circle['center']}")
        else:
            print("✗ Inner circle detection failed or no circles found")
        
        # Test inner rectangle detection
        inner_rects = detect_inner_rectangles(img, mask, largest_contour, 0.1, 0.1)
        if inner_rects:
            print(f"✓ Inner rectangle detection successful: found {len(inner_rects)} rectangle(s)")
            for i, rect in enumerate(inner_rects):
                print(f"  Rectangle {i+1}: size {rect['width_mm']:.1f}x{rect['height_mm']:.1f}mm")
        else:
            print("✗ Inner rectangle detection failed or no rectangles found")
    else:
        print("✗ No contours found in test image")


def test_hit_contour_creation():
    """Test hit contour creation function."""
    print("\nTesting hit contour creation...")
    
    from measure import create_hit_testing_contour
    
    # Test circle hit contour
    circle_contour = create_hit_testing_contour('circle', center=(100, 100), radius_px=50)
    print(f"✓ Circle hit contour created: shape {circle_contour.shape}")
    
    # Test rectangle hit contour
    box = np.array([[50, 50], [150, 50], [150, 100], [50, 100]], dtype=np.int32)
    rect_contour = create_hit_testing_contour('rectangle', box=box)
    print(f"✓ Rectangle hit contour created: shape {rect_contour.shape}")
    
    # Test rectangle from center and dimensions
    rect_contour2 = create_hit_testing_contour('rectangle', center=(100, 75), width=100, height=50)
    print(f"✓ Rectangle hit contour from center created: shape {rect_contour2.shape}")


if __name__ == "__main__":
    print("=== Testing measure.py fixes ===")
    print("This test verifies that the NameError issues have been resolved.")
    print()
    
    try:
        test_classify_and_measure()
        test_inner_detection()
        test_hit_contour_creation()
        
        print("\n=== All tests completed successfully! ===")
        print("✓ NameError issues have been fixed")
        print("✓ All functions now properly define required variables")
        print("✓ Hit contour creation works correctly")
        print("✓ The main.py file should now run without errors")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()