"""
Demo script for visual feedback and overlay rendering system.

This script demonstrates the SelectionOverlay functionality including:
- Real-time selection rectangle rendering
- Mode indicator display
- Shape confirmation feedback with animation
- Error feedback and instruction overlays
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Tuple

from selection_overlay import SelectionOverlay, render_complete_manual_feedback


def create_demo_image() -> np.ndarray:
    """Create a demo image with some basic shapes for testing."""
    image = np.ones((600, 800, 3), dtype=np.uint8) * 80  # Dark gray background
    
    # Add some visual elements to make the demo more interesting
    # Draw a grid pattern
    for i in range(0, 800, 50):
        cv2.line(image, (i, 0), (i, 600), (100, 100, 100), 1)
    for i in range(0, 600, 50):
        cv2.line(image, (0, i), (800, i), (100, 100, 100), 1)
    
    # Add some sample shapes
    cv2.circle(image, (200, 150), 40, (120, 120, 120), -1)
    cv2.rectangle(image, (350, 100), (450, 200), (120, 120, 120), -1)
    cv2.circle(image, (600, 300), 60, (120, 120, 120), -1)
    
    return image


def demo_selection_rectangle():
    """Demonstrate selection rectangle rendering."""
    print("Demo 1: Selection Rectangle Rendering")
    
    overlay = SelectionOverlay()
    base_image = create_demo_image()
    
    # Simulate different selection rectangles
    selections = [
        (100, 100, 150, 100),  # Small rectangle
        (300, 200, 200, 150),  # Medium rectangle
        (500, 350, 250, 200),  # Large rectangle
    ]
    
    for i, selection_rect in enumerate(selections):
        print(f"  Showing selection {i+1}: {selection_rect}")
        
        # Render selection rectangle
        result = overlay.render_selection_rectangle(base_image, selection_rect, active=True)
        
        # Add mode indicator
        result = overlay.render_mode_indicator(result, "MANUAL RECT", f"Selection {i+1}")
        
        # Display
        cv2.imshow("Selection Rectangle Demo", result)
        cv2.waitKey(2000)  # Show for 2 seconds
    
    cv2.destroyAllWindows()
    print("  Demo 1 complete\n")


def demo_mode_indicators():
    """Demonstrate mode indicator rendering."""
    print("Demo 2: Mode Indicator Display")
    
    overlay = SelectionOverlay()
    base_image = create_demo_image()
    
    modes = [
        ("AUTO", "Automatic detection"),
        ("MANUAL RECT", "Rectangle selection"),
        ("MANUAL CIRCLE", "Circle selection"),
        ("MANUAL RECT", "Selecting..."),
    ]
    
    for mode_text, additional_info in modes:
        print(f"  Showing mode: {mode_text} - {additional_info}")
        
        result = overlay.render_mode_indicator(base_image, mode_text, additional_info)
        
        cv2.imshow("Mode Indicator Demo", result)
        cv2.waitKey(1500)  # Show for 1.5 seconds
    
    cv2.destroyAllWindows()
    print("  Demo 2 complete\n")


def demo_shape_confirmation():
    """Demonstrate shape confirmation with animation."""
    print("Demo 3: Shape Confirmation with Animation")
    
    overlay = SelectionOverlay()
    base_image = create_demo_image()
    
    # Test circle confirmation
    circle_result = {
        "type": "circle",
        "center": (200, 150),
        "radius": 40,
        "confidence_score": 0.87
    }
    
    print("  Showing animated circle confirmation...")
    for frame in range(60):  # 60 frames of animation
        overlay.frame_counter = frame
        result = overlay.render_shape_confirmation(base_image, circle_result, animate=True)
        result = overlay.render_mode_indicator(result, "MANUAL CIRCLE", "Shape detected!")
        
        cv2.imshow("Shape Confirmation Demo", result)
        if cv2.waitKey(50) & 0xFF == ord('q'):  # 50ms per frame = ~20 FPS
            break
    
    # Test rectangle confirmation
    rectangle_result = {
        "type": "rectangle",
        "center": (400, 150),
        "width": 100,
        "height": 100,
        "contour": np.array([[350, 100], [450, 100], [450, 200], [350, 200]], dtype=np.int32),
        "confidence_score": 0.92
    }
    
    print("  Showing rectangle confirmation...")
    result = overlay.render_shape_confirmation(base_image, rectangle_result, animate=False)
    result = overlay.render_mode_indicator(result, "MANUAL RECT", "Rectangle found!")
    
    cv2.imshow("Shape Confirmation Demo", result)
    cv2.waitKey(2000)
    
    cv2.destroyAllWindows()
    print("  Demo 3 complete\n")


def demo_error_feedback():
    """Demonstrate error feedback rendering."""
    print("Demo 4: Error Feedback Display")
    
    overlay = SelectionOverlay()
    base_image = create_demo_image()
    
    error_messages = [
        "No shapes detected in selection",
        "Selection area too small",
        "Invalid selection bounds",
        "Shape detection failed"
    ]
    
    positions = [
        None,  # Center
        (100, 200),  # Custom position
        (500, 400),  # Another custom position
        None  # Center again
    ]
    
    for error_msg, position in zip(error_messages, positions):
        print(f"  Showing error: {error_msg}")
        
        result = overlay.render_error_feedback(base_image, error_msg, position)
        result = overlay.render_mode_indicator(result, "MANUAL RECT", "Error occurred")
        
        cv2.imshow("Error Feedback Demo", result)
        cv2.waitKey(2000)
    
    cv2.destroyAllWindows()
    print("  Demo 4 complete\n")


def demo_instruction_overlay():
    """Demonstrate instruction overlay rendering."""
    print("Demo 5: Instruction Overlay Display")
    
    overlay = SelectionOverlay()
    base_image = create_demo_image()
    
    instruction_sets = [
        [
            "Click and drag to select area",
            "Press M to cycle modes",
            "Press ESC to cancel"
        ],
        [
            "Manual Rectangle Mode",
            "Draw rectangle around target shape",
            "System will snap to best rectangle"
        ],
        [
            "Manual Circle Mode", 
            "Draw rectangle around circular object",
            "System will find best circle fit"
        ]
    ]
    
    for i, instructions in enumerate(instruction_sets):
        print(f"  Showing instruction set {i+1}")
        
        result = overlay.render_instruction_overlay(base_image, instructions)
        result = overlay.render_mode_indicator(result, "MANUAL MODE", f"Instructions {i+1}")
        
        cv2.imshow("Instruction Overlay Demo", result)
        cv2.waitKey(3000)
    
    cv2.destroyAllWindows()
    print("  Demo 5 complete\n")


def demo_complete_feedback():
    """Demonstrate complete manual feedback rendering."""
    print("Demo 6: Complete Manual Feedback System")
    
    base_image = create_demo_image()
    
    # Scenario 1: Active selection
    print("  Scenario 1: Active selection with instructions")
    result1 = render_complete_manual_feedback(
        base_image,
        selection_rect=(150, 120, 180, 140),
        mode_text="MANUAL RECT",
        instructions=[
            "Selecting rectangle area...",
            "Release mouse to complete selection"
        ]
    )
    cv2.imshow("Complete Feedback Demo", result1)
    cv2.waitKey(2500)
    
    # Scenario 2: Successful shape detection
    print("  Scenario 2: Successful shape detection")
    shape_result = {
        "type": "circle",
        "center": (200, 150),
        "radius": 40,
        "confidence_score": 0.89
    }
    result2 = render_complete_manual_feedback(
        base_image,
        selection_rect=None,
        mode_text="MANUAL CIRCLE",
        shape_result=shape_result
    )
    cv2.imshow("Complete Feedback Demo", result2)
    cv2.waitKey(2500)
    
    # Scenario 3: Error with instructions
    print("  Scenario 3: Error with recovery instructions")
    result3 = render_complete_manual_feedback(
        base_image,
        selection_rect=(300, 250, 50, 30),  # Small selection
        mode_text="MANUAL RECT",
        error_message="Selection too small - try again",
        instructions=[
            "Make a larger selection",
            "Minimum size: 50x50 pixels"
        ]
    )
    cv2.imshow("Complete Feedback Demo", result3)
    cv2.waitKey(3000)
    
    cv2.destroyAllWindows()
    print("  Demo 6 complete\n")


def demo_performance_test():
    """Demonstrate rendering performance."""
    print("Demo 7: Performance Test")
    
    overlay = SelectionOverlay()
    base_image = create_demo_image()
    
    selection_rect = (200, 150, 200, 150)
    mode_text = "PERFORMANCE TEST"
    
    print("  Running performance test (100 renders)...")
    start_time = time.time()
    
    for i in range(100):
        result = overlay.render_selection_rectangle(base_image, selection_rect)
        result = overlay.render_mode_indicator(result, mode_text, f"Frame {i+1}")
        
        # Simulate some processing
        if i % 10 == 0:
            cv2.imshow("Performance Test", result)
            cv2.waitKey(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / 100
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"  Performance Results:")
    print(f"    Total time: {total_time:.3f} seconds")
    print(f"    Average time per render: {avg_time:.4f} seconds")
    print(f"    Estimated FPS: {fps:.1f}")
    
    cv2.destroyAllWindows()
    print("  Demo 7 complete\n")


def main():
    """Run all visual feedback demos."""
    print("Visual Feedback and Overlay Rendering Demo")
    print("=" * 50)
    print("This demo showcases the SelectionOverlay functionality")
    print("Press 'q' during animation demos to skip to next demo")
    print("=" * 50)
    
    try:
        demo_selection_rectangle()
        demo_mode_indicators()
        demo_shape_confirmation()
        demo_error_feedback()
        demo_instruction_overlay()
        demo_complete_feedback()
        demo_performance_test()
        
        print("All demos completed successfully!")
        print("\nKey features demonstrated:")
        print("✓ Real-time selection rectangle rendering with semi-transparent overlay")
        print("✓ Mode indicator display in corner of inspection window")
        print("✓ Visual confirmation feedback with animation for detected shapes")
        print("✓ Error feedback with customizable positioning")
        print("✓ Instruction overlay for user guidance")
        print("✓ Complete integrated feedback system")
        print("✓ Performance optimization for responsive rendering")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()