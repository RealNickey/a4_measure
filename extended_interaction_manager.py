"""
Extended Interaction Manager for Manual Shape Selection

This module extends the existing InteractionManager to support manual shape selection
modes alongside automatic detection. It coordinates between automatic hit testing
and manual selection workflows with seamless mode switching.

Requirements addressed: 3.1, 3.4, 4.1, 4.2
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable

from interaction_manager import InteractionManager
from selection_mode import SelectionMode, ModeManager
from manual_selection_engine import ManualSelectionEngine
from shape_snapping_engine import ShapeSnappingEngine
from enhanced_contour_analyzer import EnhancedContourAnalyzer
from selection_overlay import SelectionOverlay


class ExtendedInteractionManager(InteractionManager):
    """
    Extended interaction manager that supports both automatic and manual selection modes.
    
    Inherits from InteractionManager and adds:
    - Mode management for switching between automatic and manual modes
    - Manual selection mouse event handling
    - Keyboard shortcuts for mode cycling and selection cancellation
    - Coordination between automatic hit testing and manual selection workflows
    - Visual feedback for manual selection operations
    """
    
    def __init__(self, shapes: List[Dict[str, Any]], warped_image: np.ndarray,
                 display_height: int = 800, hover_snap_distance_mm: float = 10.0,
                 enable_performance_optimization: bool = True):
        """
        Initialize the extended interaction manager.
        
        Args:
            shapes: List of detected shape data dictionaries
            warped_image: The warped A4 background image
            display_height: Height for the display window
            hover_snap_distance_mm: Distance threshold for hover snapping
            enable_performance_optimization: Enable performance optimizations
        """
        # Initialize parent class
        super().__init__(shapes, warped_image, display_height, 
                        hover_snap_distance_mm, enable_performance_optimization)
        
        # Initialize manual selection components
        self.mode_manager = ModeManager()
        self.manual_engine = ManualSelectionEngine(self.display_scale)
        self.selection_overlay = SelectionOverlay()
        
        # Initialize enhanced contour analyzer and shape snapping engine
        self.enhanced_analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.enhanced_analyzer)
        
        # Manual selection state
        self.last_manual_result: Optional[Dict[str, Any]] = None
        self.show_shape_confirmation = False
        self.confirmation_timer = 0
        self.confirmation_duration = 60  # frames to show confirmation
        
        # Keyboard shortcuts
        self.key_mode_cycle = ord('m')  # M key for mode cycling
        self.key_cancel_selection = 27  # ESC key for canceling selection
        self.key_toggle_confirmation = ord('c')  # C key to toggle confirmation display
        
        # Setup manual selection callbacks
        self._setup_manual_selection_callbacks()
        
        print(f"[INFO] Extended interaction manager initialized in {self.mode_manager.get_mode_indicator()} mode")
    
    def _setup_manual_selection_callbacks(self) -> None:
        """Setup callbacks for manual selection events."""
        self.manual_engine.set_callbacks(
            start_callback=self._on_manual_selection_start,
            update_callback=self._on_manual_selection_update,
            complete_callback=self._on_manual_selection_complete,
            cancel_callback=self._on_manual_selection_cancel
        )
    
    def handle_key_press(self, key: int) -> bool:
        """
        Handle keyboard input for mode switching and selection control.
        
        Args:
            key: Key code from OpenCV
            
        Returns:
            True if the key was handled and requires re-rendering
        """
        if key == self.key_mode_cycle:
            # Cycle to next mode
            old_mode = self.mode_manager.get_current_mode()
            new_mode = self.mode_manager.cycle_mode()
            
            # Cancel any active manual selection when switching modes
            if self.manual_engine.is_selecting():
                self.manual_engine.cancel_selection()
            
            # Clear any previous manual results when switching to auto mode
            if new_mode == SelectionMode.AUTO:
                self.last_manual_result = None
                self.show_shape_confirmation = False
            
            print(f"[INFO] Mode switched from {old_mode.value} to {new_mode.value}")
            return True
            
        elif key == self.key_cancel_selection:
            # Cancel active manual selection
            if self.manual_engine.is_selecting():
                self.manual_engine.cancel_selection()
                print("[INFO] Manual selection cancelled")
                return True
            
            # Clear shape confirmation
            if self.show_shape_confirmation:
                self.show_shape_confirmation = False
                self.last_manual_result = None
                print("[INFO] Shape confirmation cleared")
                return True
                
        elif key == self.key_toggle_confirmation:
            # Toggle shape confirmation display
            if self.last_manual_result is not None:
                self.show_shape_confirmation = not self.show_shape_confirmation
                print(f"[INFO] Shape confirmation {'enabled' if self.show_shape_confirmation else 'disabled'}")
                return True
        
        return False
    
    def handle_manual_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> bool:
        """
        Handle mouse events for manual selection mode.
        
        Args:
            event: OpenCV mouse event type
            x: X coordinate in display window
            y: Y coordinate in display window
            flags: OpenCV event flags
            param: User data (unused)
            
        Returns:
            True if the event was handled and requires re-rendering
        """
        if not self.mode_manager.is_manual_mode():
            return False
        
        # Handle manual selection mouse events
        handled = self.manual_engine.handle_mouse_event(event, x, y, flags, param)
        
        if handled:
            # Clear any previous shape confirmation when starting new selection
            if event == cv2.EVENT_LBUTTONDOWN:
                self.show_shape_confirmation = False
                self.last_manual_result = None
        
        return handled
    
    def render_with_manual_overlays(self) -> Optional[np.ndarray]:
        """
        Render the current state with manual selection overlays.
        
        Returns:
            Rendered image with manual overlays or None if rendering skipped
        """
        # Get base rendered image
        base_image = self.render_current_state()
        if base_image is None:
            return None
        
        result = base_image.copy()
        
        # Render mode indicator
        mode_text = self.mode_manager.get_mode_indicator()
        result = self.selection_overlay.render_mode_indicator(result, mode_text)
        
        # Render manual selection overlays if in manual mode
        if self.mode_manager.is_manual_mode():
            # Render active selection rectangle
            selection_rect = self.manual_engine.get_display_selection_rect()
            if selection_rect is not None:
                result = self.selection_overlay.render_selection_rectangle(result, selection_rect)
            
            # Render shape confirmation if available
            if self.show_shape_confirmation and self.last_manual_result is not None:
                # Transform shape result to display coordinates for rendering
                display_result = self._transform_shape_result_to_display(self.last_manual_result)
                result = self.selection_overlay.render_shape_confirmation(result, display_result)
                
                # Update confirmation timer
                self.confirmation_timer += 1
                if self.confirmation_timer >= self.confirmation_duration:
                    self.show_shape_confirmation = False
                    self.confirmation_timer = 0
        
        return result
    
    def _on_mouse_event(self, event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        """
        Enhanced mouse event handler that coordinates automatic and manual modes.
        
        Args:
            event: OpenCV mouse event type
            x: X coordinate in display window
            y: Y coordinate in display window
            flags: OpenCV event flags
            userdata: User data (unused)
        """
        needs_render = False
        
        # Handle manual selection events first if in manual mode
        if self.mode_manager.is_manual_mode():
            manual_handled = self.handle_manual_mouse_event(event, x, y, flags, userdata)
            if manual_handled:
                needs_render = True
            # Don't process automatic events if manual selection is active
            elif self.manual_engine.is_selecting():
                needs_render = False
            else:
                # Allow automatic hover in manual mode when not actively selecting
                if event == cv2.EVENT_MOUSEMOVE:
                    needs_render = self.handle_mouse_move(x, y)
        else:
            # Handle automatic mode events (original behavior)
            if event == cv2.EVENT_MOUSEMOVE:
                needs_render = self.handle_mouse_move(x, y)
            elif event == cv2.EVENT_LBUTTONDOWN:
                needs_render = self.handle_mouse_click(x, y)
        
        # Re-render if needed
        if needs_render and hasattr(self, 'window_name'):
            display_image = self.render_with_manual_overlays()
            if display_image is not None:
                cv2.imshow(self.window_name, display_image)
    
    def setup_window(self, window_name: str) -> None:
        """
        Setup the OpenCV window with enhanced mouse and keyboard callbacks.
        
        Args:
            window_name: Name of the OpenCV window
        """
        # Call parent setup
        super().setup_window(window_name)
        
        # Override mouse callback with extended version
        cv2.setMouseCallback(window_name, self._on_mouse_event)
        
        print(f"[INFO] Extended interaction window '{window_name}' setup complete")
        print("[INFO] Keyboard shortcuts:")
        print("  M - Cycle selection mode (AUTO → MANUAL RECT → MANUAL CIRCLE)")
        print("  ESC - Cancel active selection or clear confirmation")
        print("  C - Toggle shape confirmation display")
    
    def show_initial_render(self) -> None:
        """Display the initial rendered state with manual overlays."""
        if hasattr(self, 'window_name'):
            display_image = self.render_with_manual_overlays()
            if display_image is not None:
                cv2.imshow(self.window_name, display_image)
    
    def _on_manual_selection_start(self, x: int, y: int) -> None:
        """
        Callback for when manual selection starts.
        
        Args:
            x: Starting X coordinate in original image space
            y: Starting Y coordinate in original image space
        """
        mode = self.mode_manager.get_current_mode()
        shape_type = self.mode_manager.get_manual_shape_type()
        print(f"[INFO] Started manual {shape_type} selection at ({x}, {y})")
    
    def _on_manual_selection_update(self, x: int, y: int) -> None:
        """
        Callback for when manual selection is updated.
        
        Args:
            x: Current X coordinate in original image space
            y: Current Y coordinate in original image space
        """
        # This callback is called frequently during drag, so we don't log every update
        pass
    
    def _on_manual_selection_complete(self, selection_rect: Tuple[int, int, int, int]) -> None:
        """
        Callback for when manual selection is completed.
        
        Args:
            selection_rect: Final selection rectangle as (x, y, width, height)
        """
        mode = self.mode_manager.get_current_mode()
        shape_type = self.mode_manager.get_manual_shape_type()
        
        print(f"[INFO] Completed manual {shape_type} selection: {selection_rect}")
        
        # Attempt to snap to shape within the selection
        try:
            shape_result = self.snap_engine.snap_to_shape(
                self.warped_image, selection_rect, mode
            )
            
            if shape_result is not None:
                self.last_manual_result = shape_result
                self.show_shape_confirmation = True
                self.confirmation_timer = 0
                
                # Print shape information
                if shape_result["type"] == "circle":
                    print(f"[SUCCESS] Detected circle - Center: {shape_result['center']}, "
                          f"Radius: {shape_result['radius']:.1f}, "
                          f"Confidence: {shape_result['confidence_score']:.2f}")
                elif shape_result["type"] == "rectangle":
                    print(f"[SUCCESS] Detected rectangle - Center: {shape_result['center']}, "
                          f"Size: {shape_result['width']:.1f} x {shape_result['height']:.1f}, "
                          f"Confidence: {shape_result['confidence_score']:.2f}")
                
                # Call selection callback if provided (for integration with measurement system)
                if self.selection_callback:
                    # Convert manual result to format compatible with existing callback
                    self._call_selection_callback_for_manual_result(shape_result)
                    
            else:
                print(f"[INFO] No suitable {shape_type} found in selection area")
                
        except Exception as e:
            print(f"[ERROR] Shape snapping failed: {e}")
    
    def _on_manual_selection_cancel(self) -> None:
        """Callback for when manual selection is cancelled."""
        shape_type = self.mode_manager.get_manual_shape_type()
        print(f"[INFO] Manual {shape_type} selection cancelled")
    
    def _transform_shape_result_to_display(self, shape_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform shape result coordinates from original to display space.
        
        Args:
            shape_result: Shape result in original coordinates
            
        Returns:
            Shape result with display coordinates
        """
        result = shape_result.copy()
        
        # Transform center coordinates
        if "center" in result:
            orig_x, orig_y = result["center"]
            display_x = int(orig_x * self.display_scale)
            display_y = int(orig_y * self.display_scale)
            result["center"] = (display_x, display_y)
        
        # Transform radius for circles
        if result["type"] == "circle" and "radius" in result:
            result["radius"] = result["radius"] * self.display_scale
        
        # Transform dimensions for rectangles
        if result["type"] == "rectangle":
            if "width" in result and "height" in result:
                result["width"] = result["width"] * self.display_scale
                result["height"] = result["height"] * self.display_scale
        
        # Transform contour if present
        if "contour" in result and result["contour"] is not None:
            contour = result["contour"].copy()
            contour = (contour * self.display_scale).astype(np.int32)
            result["contour"] = contour
        
        return result
    
    def _call_selection_callback_for_manual_result(self, shape_result: Dict[str, Any]) -> None:
        """
        Call the selection callback with manual shape result.
        
        Args:
            shape_result: Manual shape detection result
        """
        # For manual selections, we don't have a shape index in the original shapes list
        # Instead, we pass None as the index and include the manual result in a temporary shapes list
        manual_shapes = [shape_result]
        
        if self.selection_callback:
            self.selection_callback(0, manual_shapes)  # Index 0 for the manual result
    
    def get_current_mode(self) -> SelectionMode:
        """
        Get the current selection mode.
        
        Returns:
            Current SelectionMode
        """
        return self.mode_manager.get_current_mode()
    
    def set_mode(self, mode: SelectionMode) -> None:
        """
        Set the current selection mode.
        
        Args:
            mode: SelectionMode to set
        """
        old_mode = self.mode_manager.get_current_mode()
        self.mode_manager.set_mode(mode)
        
        # Cancel any active manual selection when switching modes
        if self.manual_engine.is_selecting():
            self.manual_engine.cancel_selection()
        
        # Clear manual results when switching to auto mode
        if mode == SelectionMode.AUTO:
            self.last_manual_result = None
            self.show_shape_confirmation = False
        
        print(f"[INFO] Mode set from {old_mode.value} to {mode.value}")
    
    def is_manual_mode(self) -> bool:
        """
        Check if currently in manual selection mode.
        
        Returns:
            True if in manual mode, False if in automatic mode
        """
        return self.mode_manager.is_manual_mode()
    
    def get_manual_selection_info(self) -> Dict[str, Any]:
        """
        Get information about the current manual selection state.
        
        Returns:
            Dictionary with manual selection information
        """
        return {
            "current_mode": self.mode_manager.get_current_mode().value,
            "is_manual_mode": self.mode_manager.is_manual_mode(),
            "is_selecting": self.manual_engine.is_selecting(),
            "selection_info": self.manual_engine.get_selection_info(),
            "last_result": self.last_manual_result,
            "show_confirmation": self.show_shape_confirmation
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics including manual selection metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = super().get_performance_stats()
        
        # Add manual selection specific stats
        stats["manual_selection"] = {
            "current_mode": self.mode_manager.get_current_mode().value,
            "is_selecting": self.manual_engine.is_selecting(),
            "has_manual_result": self.last_manual_result is not None,
            "confirmation_active": self.show_shape_confirmation
        }
        
        # Add shape snapping engine stats
        if hasattr(self.snap_engine, 'get_engine_stats'):
            stats["shape_snapping"] = self.snap_engine.get_engine_stats()
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up resources including manual selection components."""
        try:
            # Cancel any active manual selection
            if self.manual_engine.is_selecting():
                self.manual_engine.cancel_selection()
            
            # Reset manual selection state
            self.manual_engine.reset()
            self.last_manual_result = None
            self.show_shape_confirmation = False
            
            print("[INFO] Manual selection components cleaned up")
            
        except Exception as e:
            print(f"[WARN] Error during manual selection cleanup: {e}")
        
        # Call parent cleanup
        super().cleanup()


# Utility functions for extended interaction management

def create_extended_interaction_manager(shapes: List[Dict[str, Any]], warped_image: np.ndarray,
                                      display_height: int = 800, 
                                      hover_snap_distance_mm: float = 10.0,
                                      enable_performance_optimization: bool = True) -> ExtendedInteractionManager:
    """
    Create and configure an extended interaction manager with manual selection support.
    
    Args:
        shapes: List of detected shape data dictionaries
        warped_image: The warped A4 background image
        display_height: Height for the display window
        hover_snap_distance_mm: Distance threshold for hover snapping
        enable_performance_optimization: Enable performance optimizations
        
    Returns:
        Configured ExtendedInteractionManager instance
    """
    from interaction_manager import default_selection_callback
    
    manager = ExtendedInteractionManager(shapes, warped_image, display_height, 
                                       hover_snap_distance_mm, enable_performance_optimization)
    manager.set_selection_callback(default_selection_callback)
    return manager


def setup_extended_interactive_inspect_mode(shapes: List[Dict[str, Any]], warped_image: np.ndarray,
                                          window_name: str = "Extended Inspect Mode",
                                          enable_performance_optimization: bool = True) -> ExtendedInteractionManager:
    """
    Complete setup for extended interactive inspect mode with manual selection support.
    
    Args:
        shapes: List of detected shape data dictionaries
        warped_image: The warped A4 background image
        window_name: Name for the OpenCV window
        enable_performance_optimization: Enable performance optimizations
        
    Returns:
        Configured and ready ExtendedInteractionManager instance
    """
    from interaction_manager import validate_shapes_for_interaction
    
    # Validate shapes
    valid_shapes = validate_shapes_for_interaction(shapes)
    
    # Create extended interaction manager
    manager = create_extended_interaction_manager(valid_shapes, warped_image, 
                                                enable_performance_optimization=enable_performance_optimization)
    
    # Setup window and display initial state
    manager.setup_window(window_name)
    manager.show_initial_render()
    
    # Print summary
    manager.print_shape_summary()
    
    if enable_performance_optimization:
        print("[INFO] Performance optimization enabled for smooth interaction")
    
    print("[INFO] Extended inspect mode ready - supports both automatic and manual selection")
    
    return manager