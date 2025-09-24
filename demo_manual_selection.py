"""
Demonstration of Manual Selection Engine Integration

This script shows how the ManualSelectionEngine integrates with the existing
interaction system and provides the foundation for manual shape selection.
"""

import cv2
import numpy as np
from manual_selection_engine import ManualSelectionEngine
from selection_mode import SelectionMode, ModeManager


def create_demo_image():
    """Create a demo image with some shapes for testing."""
    image = np.ones((600, 800, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw some test shapes
    # Rectangle
    cv2.rectangle(image, (100, 100), (250, 200), (0, 0, 255), 2)
    cv2.putText(image, "Rectangle", (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Circle
    cv2.circle(image, (400, 150), 60, (0, 255, 0), 2)
    cv2.putText(image, "Circle", (360, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Nested shapes
    cv2.rectangle(image, (500, 300), (700, 500), (255, 0, 0), 2)
    cv2.circle(image, (600, 400), 40, (0, 255, 255), 2)
    cv2.putText(image, "Nested Shapes", (520, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return image


def render_selection_overlay(image, engine, mode_manager):
    """Render selection overlay and mode indicator."""
    overlay = image.copy()
    
    # Draw current selection rectangle if selecting
    if engine.is_selecting():
        rect = engine.get_display_selection_rect()
        if rect:
            x, y, w, h = rect
            # Draw selection rectangle with semi-transparent overlay
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # Add semi-transparent fill
            selection_overlay = overlay.copy()
            cv2.rectangle(selection_overlay, (x, y), (x + w, y + h), (255, 255, 0), -1)
            overlay = cv2.addWeighted(overlay, 0.8, selection_overlay, 0.2, 0)
    
    # Draw mode indicator
    mode_text = f"Mode: {mode_manager.get_mode_indicator()}"
    cv2.putText(overlay, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Draw instructions
    instructions = [
        "M - Cycle modes (AUTO -> MANUAL RECT -> MANUAL CIRCLE)",
        "Left click + drag - Select area (in manual mode)",
        "Right click - Cancel selection",
        "ESC - Exit demo"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(overlay, instruction, (10, 70 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return overlay


def main():
    """Run the manual selection demo."""
    print("Manual Selection Engine Demo")
    print("=" * 40)
    print("This demo shows the core functionality of the manual selection engine.")
    print("The engine provides the foundation for manual shape selection and snapping.")
    print("\nControls:")
    print("- M: Cycle between modes")
    print("- Left click + drag: Create selection (in manual mode)")
    print("- Right click: Cancel selection")
    print("- ESC: Exit")
    print("\nPress any key to start...")
    input()
    
    # Create demo image and components
    demo_image = create_demo_image()
    mode_manager = ModeManager()
    engine = ManualSelectionEngine(display_scale=1.0, min_selection_size=20)
    
    # Set up callbacks
    def on_selection_start(x, y):
        print(f"Selection started at ({x}, {y})")
    
    def on_selection_update(x, y):
        print(f"Selection updated to ({x}, {y})")
    
    def on_selection_complete(rect):
        x, y, w, h = rect
        print(f"Selection completed: ({x}, {y}) size {w}x{h} (area: {w*h} pixels)")
        print("-> This selection would now be passed to the shape snapping engine")
    
    def on_selection_cancel():
        print("Selection cancelled")
    
    engine.set_callbacks(on_selection_start, on_selection_update, 
                        on_selection_complete, on_selection_cancel)
    
    # Mouse callback for OpenCV window
    def mouse_callback(event, x, y, flags, userdata):
        if mode_manager.is_manual_mode():
            # Handle manual selection events
            needs_render = engine.handle_mouse_event(event, x, y, flags, userdata)
            if needs_render:
                # Re-render with updated selection
                display_image = render_selection_overlay(demo_image, engine, mode_manager)
                cv2.imshow("Manual Selection Demo", display_image)
        else:
            # In auto mode, just show mouse position
            if event == cv2.EVENT_MOUSEMOVE:
                print(f"Auto mode: Mouse at ({x}, {y})")
    
    # Set up OpenCV window
    cv2.namedWindow("Manual Selection Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Manual Selection Demo", 800, 600)
    cv2.setMouseCallback("Manual Selection Demo", mouse_callback)
    
    # Main loop
    while True:
        # Render current state
        display_image = render_selection_overlay(demo_image, engine, mode_manager)
        cv2.imshow("Manual Selection Demo", display_image)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('m') or key == ord('M'):
            # Cycle mode
            old_mode = mode_manager.get_current_mode()
            new_mode = mode_manager.cycle_mode()
            print(f"Mode changed: {old_mode} -> {new_mode}")
            
            # Cancel any active selection when switching modes
            if engine.is_selecting():
                engine.cancel_selection()
    
    cv2.destroyAllWindows()
    print("\nDemo completed!")
    print("\nNext steps:")
    print("- Integrate with EnhancedContourAnalyzer for shape detection")
    print("- Add ShapeSnappingEngine for automatic shape snapping")
    print("- Extend InteractionManager to support manual selection modes")


if __name__ == '__main__':
    main()