"""
Demo script for ShapeSnappingEngine

Demonstrates the shape snapping functionality with sample images
and different selection modes.
"""

import cv2
import numpy as np
from shape_snapping_engine import ShapeSnappingEngine
from enhanced_contour_analyzer import EnhancedContourAnalyzer
from selection_mode import SelectionMode


def create_demo_image():
    """Create a demo image with various shapes."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255  # White background
    
    # Draw various shapes
    # Large circle
    cv2.circle(image, (150, 150), 70, (0, 0, 0), -1)
    
    # Large rectangle
    cv2.rectangle(image, (300, 100), (450, 200), (0, 0, 0), -1)
    
    # Medium circle
    cv2.circle(image, (100, 350), 40, (0, 0, 0), -1)
    
    # Medium rectangle
    cv2.rectangle(image, (250, 300), (350, 400), (0, 0, 0), -1)
    
    # Small shapes
    cv2.circle(image, (400, 350), 25, (0, 0, 0), -1)
    cv2.rectangle(image, (50, 50), (90, 90), (0, 0, 0), -1)
    
    return image


def draw_selection_rect(image, rect, color=(0, 255, 0), thickness=2):
    """Draw selection rectangle on image."""
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def draw_detected_shape(image, result, color=(255, 0, 0), thickness=2):
    """Draw detected shape on image."""
    if result is None:
        return image
    
    if result["type"] == "circle":
        center = result["center"]
        radius = int(result["radius"])
        cv2.circle(image, center, radius, color, thickness)
        
        # Draw center point
        cv2.circle(image, center, 3, color, -1)
        
    elif result["type"] == "rectangle":
        # Draw minimum area rectangle
        contour = result["contour"]
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(image, [box], 0, color, thickness)
        
        # Draw center point
        center = result["center"]
        cv2.circle(image, center, 3, color, -1)
    
    return image


def add_text_info(image, result, selection_rect, y_offset=30):
    """Add text information about the detection."""
    if result is None:
        cv2.putText(image, "No shape detected", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return image
    
    # Shape type and confidence
    text1 = f"Type: {result['type']}, Confidence: {result['confidence_score']:.3f}"
    cv2.putText(image, text1, (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Dimensions
    if result["type"] == "circle":
        text2 = f"Radius: {result['radius']:.1f}px, Area: {result['area']:.0f}px²"
    else:
        text2 = f"Size: {result['width']:.1f}x{result['height']:.1f}px, Area: {result['area']:.0f}px²"
    
    cv2.putText(image, text2, (10, y_offset + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Scores
    text3 = f"Scores - Size: {result['size_score']:.3f}, Pos: {result['position_score']:.3f}, Total: {result['total_score']:.3f}"
    cv2.putText(image, text3, (10, y_offset + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image


def demo_circle_detection():
    """Demonstrate circle detection."""
    print("=== Circle Detection Demo ===")
    
    # Create demo image
    image = create_demo_image()
    
    # Initialize engine
    analyzer = EnhancedContourAnalyzer()
    engine = ShapeSnappingEngine(analyzer)
    
    # Test selections around different circles
    test_cases = [
        ("Large Circle", (80, 80, 140, 140)),
        ("Medium Circle", (60, 310, 80, 80)),
        ("Small Circle", (375, 325, 50, 50)),
    ]
    
    for name, selection_rect in test_cases:
        print(f"\nTesting {name} with selection {selection_rect}")
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Detect shape
        result = engine.snap_to_shape(image, selection_rect, SelectionMode.MANUAL_CIRCLE)
        
        # Visualize
        draw_selection_rect(vis_image, selection_rect, (0, 255, 0))
        draw_detected_shape(vis_image, result, (255, 0, 0))
        add_text_info(vis_image, result, selection_rect)
        
        if result:
            print(f"  Detected: {result['type']} at {result['center']} with radius {result['radius']:.1f}")
            print(f"  Confidence: {result['confidence_score']:.3f}, Total Score: {result['total_score']:.3f}")
        else:
            print("  No circle detected")
        
        # Save visualization
        filename = f"demo_circle_{name.lower().replace(' ', '_')}.png"
        cv2.imwrite(filename, vis_image)
        print(f"  Saved visualization: {filename}")


def demo_rectangle_detection():
    """Demonstrate rectangle detection."""
    print("\n=== Rectangle Detection Demo ===")
    
    # Create demo image
    image = create_demo_image()
    
    # Initialize engine
    analyzer = EnhancedContourAnalyzer()
    engine = ShapeSnappingEngine(analyzer)
    
    # Test selections around different rectangles
    test_cases = [
        ("Large Rectangle", (280, 80, 190, 140)),
        ("Medium Rectangle", (230, 280, 140, 140)),
        ("Small Rectangle", (30, 30, 80, 80)),
    ]
    
    for name, selection_rect in test_cases:
        print(f"\nTesting {name} with selection {selection_rect}")
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Detect shape
        result = engine.snap_to_shape(image, selection_rect, SelectionMode.MANUAL_RECTANGLE)
        
        # Visualize
        draw_selection_rect(vis_image, selection_rect, (0, 255, 0))
        draw_detected_shape(vis_image, result, (255, 0, 0))
        add_text_info(vis_image, result, selection_rect)
        
        if result:
            print(f"  Detected: {result['type']} at {result['center']} size {result['width']:.1f}x{result['height']:.1f}")
            print(f"  Confidence: {result['confidence_score']:.3f}, Total Score: {result['total_score']:.3f}")
        else:
            print("  No rectangle detected")
        
        # Save visualization
        filename = f"demo_rectangle_{name.lower().replace(' ', '_')}.png"
        cv2.imwrite(filename, vis_image)
        print(f"  Saved visualization: {filename}")


def demo_mixed_selection():
    """Demonstrate selection with mixed shapes."""
    print("\n=== Mixed Shape Selection Demo ===")
    
    # Create demo image
    image = create_demo_image()
    
    # Initialize engine
    analyzer = EnhancedContourAnalyzer()
    engine = ShapeSnappingEngine(analyzer)
    
    # Large selection containing both circle and rectangle
    selection_rect = (50, 50, 400, 350)
    
    print(f"Testing large mixed selection: {selection_rect}")
    
    # Test both modes
    for mode in [SelectionMode.MANUAL_CIRCLE, SelectionMode.MANUAL_RECTANGLE]:
        print(f"\n  Mode: {mode.value}")
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Detect shape
        result = engine.snap_to_shape(image, selection_rect, mode)
        
        # Visualize
        draw_selection_rect(vis_image, selection_rect, (0, 255, 0))
        draw_detected_shape(vis_image, result, (255, 0, 0))
        add_text_info(vis_image, result, selection_rect)
        
        if result:
            if result["type"] == "circle":
                print(f"    Detected: {result['type']} at {result['center']} with radius {result['radius']:.1f}")
            else:
                print(f"    Detected: {result['type']} at {result['center']} size {result['width']:.1f}x{result['height']:.1f}")
            print(f"    Confidence: {result['confidence_score']:.3f}, Total Score: {result['total_score']:.3f}")
        else:
            print(f"    No {mode.value.split('_')[1]} detected")
        
        # Save visualization
        filename = f"demo_mixed_{mode.value}.png"
        cv2.imwrite(filename, vis_image)
        print(f"    Saved visualization: {filename}")


def demo_engine_stats():
    """Demonstrate engine statistics."""
    print("\n=== Engine Statistics ===")
    
    analyzer = EnhancedContourAnalyzer()
    engine = ShapeSnappingEngine(analyzer)
    
    stats = engine.get_engine_stats()
    
    print("Engine Configuration:")
    print(f"  Min contour area: {stats['min_contour_area']}")
    print(f"  Max contour area ratio: {stats['max_contour_area_ratio']}")
    print(f"  Min circularity: {stats['min_circularity']}")
    print(f"  Min rectangularity: {stats['min_rectangularity']}")
    print(f"  Scoring weights: Size={stats['size_weight']}, Position={stats['position_weight']}, Quality={stats['quality_weight']}")
    
    print("\nCircle Detection Parameters:")
    circle_params = stats['circle_detection_params']
    for key, value in circle_params.items():
        print(f"  {key}: {value}")
    
    print("\nRectangle Detection Parameters:")
    rect_params = stats['rectangle_detection_params']
    for key, value in rect_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("Shape Snapping Engine Demo")
    print("=" * 50)
    
    try:
        # Run all demos
        demo_circle_detection()
        demo_rectangle_detection()
        demo_mixed_selection()
        demo_engine_stats()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("Check the generated PNG files for visual results.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()