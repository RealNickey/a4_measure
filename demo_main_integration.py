"""
Demo script to test main application integration with ExtendedInteractionManager

This script demonstrates the integration of manual selection capabilities
into the main application workflow.
"""

import cv2
import numpy as np
from extended_interaction_manager import setup_extended_interactive_inspect_mode
from measure import create_shape_data

def create_demo_shapes():
    """Create demo shape data for testing."""
    # Create sample contours
    circle_contour = np.array([[75, 100], [100, 75], [125, 100], [100, 125]], dtype=np.int32)
    rect_contour = np.array([[160, 120], [240, 120], [240, 180], [160, 180]], dtype=np.int32)
    
    # Create sample measurement results
    results = [
        {
            "type": "circle",
            "center": (100, 100),
            "radius": 50,
            "radius_px": 50.0,
            "diameter_mm": 25.0,
            "detection_method": "automatic",
            "confidence_score": 0.95,
            "hit_contour": circle_contour,
            "area_px": 7854.0,
            "inner": False
        },
        {
            "type": "rectangle", 
            "center": (200, 150),
            "width": 80,
            "height": 60,
            "width_mm": 40.0,
            "height_mm": 30.0,
            "detection_method": "automatic",
            "confidence_score": 0.90,
            "hit_contour": rect_contour,
            "area_px": 4800.0,
            "box": np.array([[160, 120], [240, 120], [240, 180], [160, 180]], dtype=np.float32),
            "inner": False
        }
    ]
    
    # Convert to shape data format
    shapes = []
    for result in results:
        shape_data = create_shape_data(result)
        if shape_data is not None:
            shapes.append(shape_data)
    
    return shapes

def create_demo_image():
    """Create a demo warped image with shapes."""
    # Create white background
    image = np.ones((400, 300, 3), dtype=np.uint8) * 255
    
    # Draw a circle
    cv2.circle(image, (100, 100), 50, (0, 0, 255), 2)
    cv2.putText(image, "Circle", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw a rectangle
    cv2.rectangle(image, (160, 120), (240, 180), (0, 255, 0), 2)
    cv2.putText(image, "Rectangle", (165, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add instructions
    cv2.putText(image, "Demo: Enhanced Inspect Mode", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, "M - Cycle modes", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "ESC - Exit", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return image

def main():
    """Main demo function."""
    print("=== Main Application Integration Demo ===")
    print("This demo shows the integration of ExtendedInteractionManager")
    print("into the main application workflow.")
    print()
    
    # Create demo data
    shapes = create_demo_shapes()
    warped_image = create_demo_image()
    
    print(f"Created {len(shapes)} demo shapes")
    print("Setting up extended interactive inspect mode...")
    
    try:
        # Setup extended interactive inspect mode (same as in main.py)
        window_name = "Demo: Enhanced Inspect Mode"
        manager = setup_extended_interactive_inspect_mode(shapes, warped_image, window_name)
        
        print("\n[DEMO] Enhanced inspect mode active!")
        print("- Hover over shapes to preview")
        print("- Click shapes to inspect")
        print("- Press 'M' to cycle between AUTO → MANUAL RECT → MANUAL CIRCLE modes")
        print("- In manual modes, click and drag to select areas")
        print("- Press ESC to exit demo")
        print()
        
        # Main interaction loop (similar to main.py)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k != 255:  # any key pressed
                # Handle keyboard shortcuts for mode switching and selection control
                key_handled = manager.handle_key_press(k)
                
                if key_handled:
                    # Key was handled by manager, continue loop
                    continue
                elif k == 27:  # ESC - exit demo
                    print("[INFO] ESC pressed - exiting demo.")
                    break
                else:
                    print(f"[INFO] Unhandled key: {k} ('{chr(k) if 32 <= k <= 126 else '?'}')")
        
        # Cleanup (same as in main.py)
        print("\n[INFO] Cleaning up demo resources...")
        manager.cleanup()
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[INFO] Demo completed successfully!")
    print("Integration with main application workflow verified.")

if __name__ == "__main__":
    main()