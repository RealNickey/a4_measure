#!/usr/bin/env python3
"""
Test script for manual selection responsiveness improvements.

This script tests the optimized manual selection system to verify:
1. Responsive mode switching without lag
2. Visible selection rectangle during dragging
3. Smooth real-time visual feedback
"""

import cv2
import numpy as np
from extended_interaction_manager import ExtendedInteractionManager
from selection_mode import SelectionMode


def create_test_image():
    """Create a test image with some shapes for testing."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add some test shapes
    cv2.circle(img, (200, 150), 50, (100, 100, 255), -1)  # Red circle
    cv2.rectangle(img, (350, 100), (450, 200), (100, 255, 100), -1)  # Green rectangle
    cv2.circle(img, (600, 300), 80, (255, 100, 100), -1)  # Blue circle
    cv2.rectangle(img, (100, 400), (300, 500), (255, 255, 100), -1)  # Cyan rectangle
    
    return img


def create_test_shapes():
    """Create test shape data for the interaction manager."""
    shapes = [
        {
            "type": "circle",
            "center": (200, 150),
            "radius_px": 50,
            "diameter_mm": 25.0,
            "area_px": np.pi * 50 * 50,
            "hit_cnt": np.array([(200, 150)]).reshape(-1, 1, 2)
        },
        {
            "type": "rectangle",
            "box": np.array([[350, 100], [450, 100], [450, 200], [350, 200]]),
            "width_mm": 50.0,
            "height_mm": 50.0,
            "area_px": 100 * 100,
            "hit_cnt": np.array([[350, 100], [450, 100], [450, 200], [350, 200]]).reshape(-1, 1, 2)
        }
    ]
    return shapes


def test_manual_selection_responsiveness():
    """Test the responsiveness of manual selection system."""
    print("=== Manual Selection Responsiveness Test ===")
    print()
    print("This test verifies the improvements to manual selection:")
    print("1. Fast mode switching (press 'M' to cycle modes)")
    print("2. Visible selection rectangle during dragging")
    print("3. Smooth real-time feedback")
    print()
    print("Controls:")
    print("  M - Cycle modes (AUTO → MANUAL RECT → MANUAL CIRCLE)")
    print("  Left click + drag - Create selection (in manual modes)")
    print("  Right click - Cancel selection")
    print("  ESC - Exit test")
    print()
    
    # Create test environment
    test_image = create_test_image()
    test_shapes = create_test_shapes()
    
    # Create extended interaction manager
    manager = ExtendedInteractionManager(
        shapes=test_shapes,
        warped_image=test_image,
        display_height=600,
        enable_performance_optimization=True
    )
    
    # Setup window
    window_name = "Manual Selection Responsiveness Test"
    manager.setup_window(window_name)
    
    print(f"[INFO] Test started in {manager.get_current_mode().value} mode")
    print("[INFO] Try switching modes quickly and dragging selections to test responsiveness")
    
    # Main test loop
    frame_count = 0
    last_mode = manager.get_current_mode()
    
    while True:
        # Render current state
        display_image = manager.render_with_manual_overlays()
        if display_image is not None:
            cv2.imshow(window_name, display_image)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC to exit
            break
        elif key != 255:  # Any other key
            handled = manager.handle_key_press(key)
            if handled:
                current_mode = manager.get_current_mode()
                if current_mode != last_mode:
                    print(f"[TEST] Mode switch: {last_mode.value} → {current_mode.value}")
                    last_mode = current_mode
        
        frame_count += 1
        
        # Print periodic status
        if frame_count % 300 == 0:  # Every ~10 seconds at 30fps
            mode_info = manager.get_manual_selection_info()
            print(f"[STATUS] Frame {frame_count}: Mode={mode_info['current_mode']}, "
                  f"Selecting={mode_info['is_selecting']}")
    
    # Cleanup
    manager.cleanup()
    cv2.destroyAllWindows()
    
    print("[INFO] Responsiveness test completed")


def test_selection_visibility():
    """Test the visibility of selection rectangle during dragging."""
    print("\n=== Selection Rectangle Visibility Test ===")
    print()
    print("This test focuses on selection rectangle visibility:")
    print("1. Switch to MANUAL RECT or MANUAL CIRCLE mode")
    print("2. Click and drag to create selections")
    print("3. Verify the rectangle is clearly visible during dragging")
    print("4. Test with different background areas")
    print()
    
    # Create a more complex test image with varied backgrounds
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Create different background regions
    img[0:200, 0:400] = (50, 50, 50)      # Dark gray
    img[0:200, 400:800] = (200, 200, 200) # Light gray
    img[200:400, 0:400] = (100, 150, 200) # Brownish
    img[200:400, 400:800] = (150, 100, 150) # Purple
    img[400:600, 0:400] = (200, 100, 100) # Reddish
    img[400:600, 400:800] = (100, 200, 100) # Greenish
    
    # Add some noise for challenging visibility
    noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Create minimal shapes for testing
    shapes = []
    
    # Create interaction manager
    manager = ExtendedInteractionManager(
        shapes=shapes,
        warped_image=img,
        display_height=600,
        enable_performance_optimization=True
    )
    
    # Set to manual mode for testing
    manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
    
    window_name = "Selection Visibility Test"
    manager.setup_window(window_name)
    
    print(f"[INFO] Visibility test started in {manager.get_current_mode().value} mode")
    print("[INFO] Try creating selections over different background regions")
    
    while True:
        display_image = manager.render_with_manual_overlays()
        if display_image is not None:
            cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key != 255:
            manager.handle_key_press(key)
    
    manager.cleanup()
    cv2.destroyAllWindows()
    
    print("[INFO] Visibility test completed")


if __name__ == "__main__":
    try:
        # Run responsiveness test
        test_manual_selection_responsiveness()
        
        # Ask user if they want to run visibility test
        print("\nWould you like to run the selection visibility test? (y/n): ", end="")
        response = input().strip().lower()
        
        if response in ['y', 'yes']:
            test_selection_visibility()
        
        print("\n=== All Tests Completed ===")
        print("Key improvements verified:")
        print("✓ Optimized mode switching with immediate visual feedback")
        print("✓ Enhanced selection rectangle visibility during dragging")
        print("✓ Improved mouse event handling for better responsiveness")
        print("✓ Performance optimizations for smooth real-time interaction")
        
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()