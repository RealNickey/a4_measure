"""
Demonstration of Manual Selection Integration with Measurement Pipeline

This script demonstrates how manual selection results are integrated with the
existing measurement pipeline to provide consistent data formats and measurements.
"""

import cv2
import numpy as np
from typing import List, Dict, Any

# Import measurement functions
from measure import (
    classify_and_measure, classify_and_measure_manual_selection,
    merge_automatic_and_manual_results, get_measurement_summary,
    validate_manual_measurement_result
)
from detection import a4_scale_mm_per_px


def create_test_image() -> np.ndarray:
    """Create a test image with various shapes."""
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw some shapes
    cv2.circle(image, (200, 200), 80, (0, 0, 0), -1)  # Large circle
    cv2.circle(image, (200, 200), 30, (255, 255, 255), -1)  # Inner circle (hole)
    
    cv2.rectangle(image, (400, 150), (550, 250), (0, 0, 0), -1)  # Rectangle
    cv2.rectangle(image, (420, 170), (530, 230), (255, 255, 255), -1)  # Inner rectangle
    
    cv2.circle(image, (600, 400), 60, (0, 0, 0), -1)  # Another circle
    
    return image


def demonstrate_automatic_detection(image: np.ndarray) -> List[Dict[str, Any]]:
    """Demonstrate automatic shape detection."""
    print("=== Automatic Detection ===")
    
    # Convert to grayscale and find contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get scale factors
    mm_per_px_x, mm_per_px_y = a4_scale_mm_per_px()
    
    # Process each contour
    automatic_results = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 1000:  # Filter small contours
            result = classify_and_measure(contour, mm_per_px_x, mm_per_px_y, "automatic")
            if result:
                automatic_results.append(result)
                print(f"  Shape {i+1}: {result['type']} - Detection: {result['detection_method']}")
                if result['type'] == 'circle':
                    print(f"    Diameter: {result['diameter_mm']:.1f} mm")
                else:
                    print(f"    Dimensions: {result['width_mm']:.1f} x {result['height_mm']:.1f} mm")
    
    return automatic_results


def demonstrate_manual_selection(image: np.ndarray) -> List[Dict[str, Any]]:
    """Demonstrate manual selection integration."""
    print("\n=== Manual Selection Integration ===")
    
    # Get scale factors
    mm_per_px_x, mm_per_px_y = a4_scale_mm_per_px()
    
    # Simulate manual selections with mock shape results
    manual_results = []
    
    # Mock manual circle selection
    circle_shape_result = {
        "type": "circle",
        "center": (200, 200),
        "radius": 80,
        "confidence_score": 0.92,
        "mode": "manual_circle"
    }
    
    circle_selection_rect = (120, 120, 160, 160)  # x, y, w, h
    
    manual_circle = classify_and_measure_manual_selection(
        image, circle_selection_rect, circle_shape_result, mm_per_px_x, mm_per_px_y
    )
    
    if manual_circle and validate_manual_measurement_result(manual_circle):
        manual_results.append(manual_circle)
        print(f"  Manual Circle: {manual_circle['type']} - Detection: {manual_circle['detection_method']}")
        print(f"    Diameter: {manual_circle['diameter_mm']:.1f} mm")
        print(f"    Confidence: {manual_circle['confidence_score']:.2f}")
        print(f"    Selection: {manual_circle['selection_rect']}")
    
    # Mock manual rectangle selection
    rectangle_shape_result = {
        "type": "rectangle",
        "center": (475, 200),
        "width": 150,
        "height": 100,
        "confidence_score": 0.88,
        "mode": "manual_rectangle"
    }
    
    rectangle_selection_rect = (380, 130, 190, 140)  # x, y, w, h
    
    manual_rectangle = classify_and_measure_manual_selection(
        image, rectangle_selection_rect, rectangle_shape_result, mm_per_px_x, mm_per_px_y
    )
    
    if manual_rectangle and validate_manual_measurement_result(manual_rectangle):
        manual_results.append(manual_rectangle)
        print(f"  Manual Rectangle: {manual_rectangle['type']} - Detection: {manual_rectangle['detection_method']}")
        print(f"    Dimensions: {manual_rectangle['width_mm']:.1f} x {manual_rectangle['height_mm']:.1f} mm")
        print(f"    Confidence: {manual_rectangle['confidence_score']:.2f}")
        print(f"    Selection: {manual_rectangle['selection_rect']}")
    
    return manual_results


def demonstrate_result_merging(automatic_results: List[Dict[str, Any]], 
                             manual_results: List[Dict[str, Any]]) -> None:
    """Demonstrate merging of automatic and manual results."""
    print("\n=== Result Merging and Analysis ===")
    
    # Merge results
    merged_results = merge_automatic_and_manual_results(automatic_results, manual_results)
    
    print(f"Automatic results: {len(automatic_results)}")
    print(f"Manual results: {len(manual_results)}")
    print(f"Merged results: {len(merged_results)}")
    
    # Generate summary
    summary = get_measurement_summary(merged_results)
    
    print(f"\nMeasurement Summary:")
    print(f"  Total shapes: {summary['total_shapes']}")
    print(f"  Automatic: {summary['automatic_count']}")
    print(f"  Manual: {summary['manual_count']}")
    print(f"  Circles: {summary['circles']}")
    print(f"  Rectangles: {summary['rectangles']}")
    print(f"  Inner shapes: {summary['inner_shapes']}")
    
    print(f"\nDetection Methods:")
    for method, count in summary['detection_methods'].items():
        print(f"  {method}: {count}")


def demonstrate_data_format_consistency() -> None:
    """Demonstrate that manual and automatic results have consistent data formats."""
    print("\n=== Data Format Consistency ===")
    
    # Create identical contours for comparison
    center = (300, 300)
    radius = 50
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    points = []
    for angle in angles:
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append([x, y])
    contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    
    # Get scale factors
    mm_per_px_x, mm_per_px_y = a4_scale_mm_per_px()
    
    # Automatic measurement
    auto_result = classify_and_measure(contour, mm_per_px_x, mm_per_px_y, "automatic")
    
    # Manual measurement (simulated)
    manual_shape_result = {
        "type": "circle",
        "center": center,
        "radius": radius,
        "confidence_score": 0.95
    }
    
    test_image = np.ones((600, 600, 3), dtype=np.uint8) * 255
    selection_rect = (250, 250, 100, 100)
    
    manual_result = classify_and_measure_manual_selection(
        test_image, selection_rect, manual_shape_result, mm_per_px_x, mm_per_px_y
    )
    
    # Compare data formats
    print("Automatic result fields:")
    for key in sorted(auto_result.keys()):
        print(f"  {key}: {type(auto_result[key]).__name__}")
    
    print("\nManual result fields:")
    for key in sorted(manual_result.keys()):
        print(f"  {key}: {type(manual_result[key]).__name__}")
    
    # Check common fields
    common_fields = set(auto_result.keys()) & set(manual_result.keys())
    manual_only_fields = set(manual_result.keys()) - set(auto_result.keys())
    
    print(f"\nCommon fields: {len(common_fields)}")
    print(f"Manual-only fields: {len(manual_only_fields)} - {list(manual_only_fields)}")
    
    # Compare measurements
    print(f"\nMeasurement Comparison:")
    print(f"  Automatic diameter: {auto_result['diameter_mm']:.2f} mm")
    print(f"  Manual diameter: {manual_result['diameter_mm']:.2f} mm")
    print(f"  Difference: {abs(auto_result['diameter_mm'] - manual_result['diameter_mm']):.2f} mm")


def main():
    """Main demonstration function."""
    print("Manual Selection Integration Demonstration")
    print("=" * 50)
    
    # Create test image
    test_image = create_test_image()
    
    # Demonstrate automatic detection
    automatic_results = demonstrate_automatic_detection(test_image)
    
    # Demonstrate manual selection
    manual_results = demonstrate_manual_selection(test_image)
    
    # Demonstrate result merging
    demonstrate_result_merging(automatic_results, manual_results)
    
    # Demonstrate data format consistency
    demonstrate_data_format_consistency()
    
    print("\n" + "=" * 50)
    print("Integration demonstration complete!")
    print("\nKey Integration Features:")
    print("✓ Manual selections use same data format as automatic detection")
    print("✓ Detection method field distinguishes manual from automatic")
    print("✓ Results can be merged and analyzed together")
    print("✓ Manual results include additional metadata (selection_rect, confidence)")
    print("✓ All results are compatible with existing rendering and interaction systems")


if __name__ == "__main__":
    main()