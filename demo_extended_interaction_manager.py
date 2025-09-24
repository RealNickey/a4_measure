"""
Demo script for ExtendedInteractionManager

This script demonstrates the manual shape selection capabilities
integrated with the existing automatic detection system.
"""

import cv2
import numpy as np
from typing import List, Dict, Any

from extended_interaction_manager import setup_extended_interactive_inspect_mode
from detection import classify_and_measure


def create_demo_image_with_shapes() -> np.ndarray:
    """
    Create a demo image with various shapes for testing manual selection.
    
    Returns:
        Demo image with circles and rectangles
    """
    # Create a white background (simulating A4 paper)
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw some shapes
    # Large circle
    cv2.circle(image, (200, 200), 80, (100, 100, 100), -1)
    
    # Medium circle inside large circle (nested)
    cv2.circle(image, (200, 200), 40, (150, 150, 150), -1)
    
    # Rectangle
    cv2.rectangle(image, (400, 150), (550, 250), (120, 120, 120), -1)
    
    # Small rectangle inside large rectangle (nested)
    cv2.rectangle(image, (430, 180), (520, 220), (180, 180, 180), -1)
    
    # Complex shape with multiple nested elements
    cv2.rectangle(image, (100, 350), (300, 500), (80, 80, 80), -1)
    cv2.circle(image, (150, 400), 25, (200, 200, 200), -1)
    cv2.circle(image, (250, 450), 30, (200, 200, 200), -1)
    cv2.rectangle(image, (180, 420), (220, 460), (200, 200, 200), -1)
    
    # Add some noise/texture
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image


def create_demo_shapes() -> List[Dict[str, Any]]:
    """
    Create demo shape data for automatic detection.
    
    Returns:
        List of shape dictionaries
    """
    return [
        {
            "type": "circle",
            "center": (200, 200),
            "radius": 80,
            "diameter_mm": 40.0,
            "area": 20106.19,
            "contour": np.array([[[120, 200]], [[200, 120]], [[280, 200]], [[200, 280]]])
        },
        {
            "type": "rectangle",
            "center": (475, 200),
            "width": 150,
            "height": 100,
            "width_mm": 75.0,
            "height_mm": 50.0,
            "area": 15000,
            "contour": np.array([[[400, 150]], [[550, 150]], [[550, 250]], [[400, 250]]])
        },
        {
            "type": "rectangle",
            "center": (200, 425),
            "width": 200,
            "height": 150,
            "width_mm": 100.0,
            "height_mm": 75.0,
            "area": 30000,
            "contour": np.array([[[100, 350]], [[300, 350]], [[300, 500]], [[100, 500]]])
        }
    ]


def print_usage_instructions():
    """Print usage instructions for the demo."""
    print("\n" + "="*60)
    print("EXTENDED INTERACTION MANAGER DEMO")
    print("="*60)
    print("\nThis demo shows manual shape selection capabilities:")
    print("\n🔄 MODE SWITCHING:")
    print("  M - Cycle between modes:")
    print("      AUTO → MANUAL RECT → MANUAL CIRCLE → AUTO")
    print("\n🖱️  MANUAL SELECTION (in manual modes):")
    print("  Left Click + Drag - Draw selection rectangle")
    print("  Right Click - Cancel active selection")
    print("\n⌨️  KEYBOARD SHORTCUTS:")
    print("  ESC - Cancel selection or clear confirmation")
    print("  C - Toggle shape confirmation display")
    print("  Q - Quit demo")
    print("\n📋 FEATURES TO TEST:")
    print("  • Switch to MANUAL RECT mode and select the inner rectangle")
    print("  • Switch to MANUAL CIRCLE mode and select the inner circle")
    print("  • Try selecting nested shapes that auto-detection misses")
    print("  • Compare manual vs automatic detection results")
    print("\n" + "="*60)


def run_extended_interaction_demo():
    """Run the extended interaction manager demo."""
    print("Creating demo image with nested shapes...")
    
    # Create demo image and shapes
    demo_image = create_demo_image_with_shapes()
    demo_shapes = create_demo_shapes()
    
    print(f"Created demo with {len(demo_shapes)} automatically detected shapes")
    
    # Print usage instructions
    print_usage_instructions()
    
    try:
        # Setup extended interactive inspect mode
        manager = setup_extended_interactive_inspect_mode(
            demo_shapes, 
            demo_image,
            window_name="Extended Interaction Demo",
            enable_performance_optimization=True
        )
        
        print("\n🚀 Demo started! Try the manual selection features...")
        print("Current mode:", manager.get_current_mode().value.upper())
        
        # Main interaction loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC to quit
                print("\n👋 Demo ended by user")
                break
            elif key != 255:  # Any other key
                # Handle key press
                handled = manager.handle_key_press(key)
                if handled:
                    print(f"Current mode: {manager.get_current_mode().value.upper()}")
                    
                    # Show manual selection info
                    if manager.is_manual_mode():
                        info = manager.get_manual_selection_info()
                        if info["last_result"]:
                            result = info["last_result"]
                            print(f"Last manual detection: {result['type']} "
                                  f"(confidence: {result['confidence_score']:.2f})")
            
            # Check if window was closed
            if cv2.getWindowProperty("Extended Interaction Demo", cv2.WND_PROP_VISIBLE) < 1:
                print("\n👋 Demo ended (window closed)")
                break
        
        # Cleanup
        manager.cleanup()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n📊 FINAL STATISTICS:")
        stats = manager.get_performance_stats()
        if "manual_selection" in stats:
            manual_stats = stats["manual_selection"]
            print(f"  Final mode: {manual_stats['current_mode'].upper()}")
            print(f"  Had manual result: {manual_stats['has_manual_result']}")
        
        print("\n✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_extended_interaction_demo()