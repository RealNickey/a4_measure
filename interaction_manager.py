"""
Interaction Manager for Interactive Inspect Mode

This module coordinates between hit testing, state management, and rendering
to provide a complete interactive experience for shape inspection.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable

from hit_testing import HitTestingEngine
from interaction_state import InteractionState, transform_display_to_original_coords
from rendering import SelectiveRenderer
from performance_profiler import OptimizedRenderer, FrameRateOptimizer, PerformanceProfiler


class InteractionManager:
    """
    Manages the complete interaction workflow for interactive inspect mode.
    
    Coordinates between:
    - Hit testing engine for shape detection
    - Interaction state for tracking hover/selection
    - Selective renderer for visual feedback
    - Mouse event handling and coordinate transformation
    """
    
    def __init__(self, shapes: List[Dict[str, Any]], warped_image: np.ndarray,
                 display_height: int = 800, hover_snap_distance_mm: float = 10.0,
                 enable_performance_optimization: bool = True):
        """
        Initialize the interaction manager.
        
        Args:
            shapes: List of detected shape data dictionaries
            warped_image: The warped A4 background image
            display_height: Height for the display window
            hover_snap_distance_mm: Distance threshold for hover snapping
            enable_performance_optimization: Enable performance optimizations
        """
        self.warped_image = warped_image
        self.display_height = display_height
        self.enable_optimization = enable_performance_optimization
        
        # Calculate display scaling
        self.display_scale = display_height / warped_image.shape[0]
        self.display_width = int(warped_image.shape[1] * self.display_scale)
        
        # Initialize components
        self.hit_engine = HitTestingEngine(snap_distance_mm=hover_snap_distance_mm)
        self.state = InteractionState(shapes)
        self.state.set_display_scale(self.display_scale)
        
        # Initialize rendering components with optimization
        base_renderer = SelectiveRenderer()
        if self.enable_optimization:
            self.renderer = OptimizedRenderer(base_renderer)
            self.frame_optimizer = FrameRateOptimizer(target_fps=60.0, min_fps=30.0)
            self.profiler = PerformanceProfiler()
        else:
            self.renderer = base_renderer
            self.frame_optimizer = None
            self.profiler = None
        
        # Callback for console output
        self.selection_callback: Optional[Callable[[Optional[int], List[Dict[str, Any]]], None]] = None
        
        # Performance tracking
        self.last_render_time = 0
        self.render_skip_count = 0
    
    def set_selection_callback(self, callback: Callable[[Optional[int], List[Dict[str, Any]]], None]) -> None:
        """
        Set callback function for selection events.
        
        Args:
            callback: Function to call when selection changes, receives (shape_index, shapes)
        """
        self.selection_callback = callback
    
    def setup_window(self, window_name: str) -> None:
        """
        Setup the OpenCV window with proper sizing and mouse callback.
        
        Args:
            window_name: Name of the OpenCV window
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(window_name, self._on_mouse_event)
        self.window_name = window_name
    
    def render_current_state(self, force_render: bool = False) -> Optional[np.ndarray]:
        """
        Render the current interaction state with performance optimizations.
        
        Args:
            force_render: Force rendering regardless of optimization checks
            
        Returns:
            Rendered image showing current hover/selection state, or None if skipped
        """
        # Check if rendering should be skipped for performance
        if self.enable_optimization and not force_render:
            if self.frame_optimizer and not self.frame_optimizer.should_render_frame():
                self.render_skip_count += 1
                return None
            
            # Skip if no state changes and not forced
            if not self.state.needs_render and not force_render:
                return None
        
        # Start performance timing
        if self.profiler:
            self.profiler.start_timing("render")
        
        try:
            # Get current state as dictionary for renderer compatibility
            state_dict = self.state.get_state_dict()
            
            # Render with optimization if available
            if self.enable_optimization and hasattr(self.renderer, 'render_optimized'):
                rendered = self.renderer.render_optimized(
                    self.warped_image, state_dict, self.state.shapes, force_render
                )
            else:
                # Fallback to standard rendering
                rendered = self.renderer.render_complete_state(
                    self.warped_image, state_dict, self.state.shapes
                )
            
            # Scale for display
            display_image = cv2.resize(rendered, (self.display_width, self.display_height))
            
            # Clear render flag
            self.state.clear_render_flag()
            
            return display_image
            
        finally:
            # End performance timing
            if self.profiler:
                render_time = self.profiler.end_timing("render")
                if self.frame_optimizer:
                    self.frame_optimizer.frame_rendered(render_time)
    
    def show_initial_render(self) -> None:
        """Display the initial rendered state."""
        if hasattr(self, 'window_name'):
            display_image = self.render_current_state()
            cv2.imshow(self.window_name, display_image)
    
    def handle_mouse_move(self, display_x: int, display_y: int) -> bool:
        """
        Handle mouse movement and update hover state with performance optimization.
        
        Args:
            display_x: X coordinate in display window
            display_y: Y coordinate in display window
            
        Returns:
            True if re-rendering is needed
        """
        # Start performance timing
        if self.profiler:
            self.profiler.start_timing("mouse_event")
        
        try:
            # Transform to original coordinates
            orig_x, orig_y = transform_display_to_original_coords(
                display_x, display_y, self.display_scale
            )
            
            # Update mouse position
            self.state.update_mouse_position(orig_x, orig_y)
            
            # Find shape at current position
            hovered_shape = self.hit_engine.find_shape_at_point(self.state.shapes, orig_x, orig_y)
            
            # Update hover state
            hover_changed = self.state.update_hover(hovered_shape)
            
            return hover_changed
            
        finally:
            # End performance timing
            if self.profiler:
                self.profiler.end_timing("mouse_event")
    
    def handle_mouse_click(self, display_x: int, display_y: int) -> bool:
        """
        Handle mouse click and update selection state.
        
        Args:
            display_x: X coordinate in display window
            display_y: Y coordinate in display window
            
        Returns:
            True if re-rendering is needed
        """
        # Transform to original coordinates
        orig_x, orig_y = transform_display_to_original_coords(
            display_x, display_y, self.display_scale
        )
        
        # Find shape at click position
        clicked_shape = self.hit_engine.find_shape_at_point(self.state.shapes, orig_x, orig_y)
        
        # Update selection state
        selection_changed = self.state.update_selection(clicked_shape)
        
        # Call selection callback if provided
        if self.selection_callback:
            self.selection_callback(clicked_shape, self.state.shapes)
        
        return selection_changed
    
    def _on_mouse_event(self, event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        """
        Internal mouse event handler for OpenCV callback with performance optimization.
        
        Args:
            event: OpenCV mouse event type
            x: X coordinate in display window
            y: Y coordinate in display window
            flags: OpenCV event flags
            userdata: User data (unused)
        """
        needs_render = False
        
        if event == cv2.EVENT_MOUSEMOVE:
            needs_render = self.handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            needs_render = self.handle_mouse_click(x, y)
        
        # Re-render if needed with optimization
        if needs_render and hasattr(self, 'window_name'):
            display_image = self.render_current_state(force_render=True)
            if display_image is not None:
                cv2.imshow(self.window_name, display_image)
    
    def get_shape_summary(self) -> List[str]:
        """
        Get a summary of all detected shapes for console output.
        
        Returns:
            List of strings describing each shape
        """
        summary = []
        for i, shape in enumerate(self.state.shapes):
            if shape["type"] == "circle":
                summary.append(f"  {i+1}. Circle - Diameter: {shape['diameter_mm']:.1f} mm")
            else:
                summary.append(f"  {i+1}. Rectangle - {shape['width_mm']:.1f} x {shape['height_mm']:.1f} mm")
        return summary
    
    def print_shape_summary(self) -> None:
        """Print detected shapes summary to console."""
        print(f"\n[INFO] Detected {len(self.state.shapes)} shape(s):")
        for line in self.get_shape_summary():
            print(line)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "optimization_enabled": self.enable_optimization,
            "render_skip_count": self.render_skip_count
        }
        
        if self.profiler:
            stats.update(self.profiler.get_overall_stats())
        
        if self.frame_optimizer:
            stats["frame_optimizer"] = self.frame_optimizer.get_performance_metrics()
        
        if hasattr(self.renderer, 'get_performance_stats'):
            stats["renderer"] = self.renderer.get_performance_stats()
        
        return stats
    
    def print_performance_report(self) -> None:
        """Print comprehensive performance report."""
        if self.profiler:
            self.profiler.print_performance_report()
        
        if self.frame_optimizer:
            metrics = self.frame_optimizer.get_performance_metrics()
            print(f"\nFrame Rate Optimization:")
            print(f"  Frames Rendered: {metrics['frames_rendered']}")
            print(f"  Frames Skipped: {metrics['frames_skipped']}")
            print(f"  Skip Ratio: {metrics['skip_ratio']:.2%}")
            print(f"  Effective FPS: {metrics['effective_fps']:.1f}")
        
        if hasattr(self.renderer, 'print_performance_report'):
            self.renderer.print_performance_report()
    
    def enable_performance_monitoring(self, enable: bool = True) -> None:
        """
        Enable or disable performance monitoring.
        
        Args:
            enable: Whether to enable performance monitoring
        """
        if enable and not self.profiler:
            self.profiler = PerformanceProfiler()
        elif not enable:
            self.profiler = None
    
    def reset_performance_stats(self) -> None:
        """Reset all performance statistics."""
        if self.profiler:
            self.profiler.reset()
        
        if self.frame_optimizer:
            self.frame_optimizer.frames_rendered = 0
            self.frame_optimizer.frames_skipped = 0
            self.frame_optimizer.frame_times.clear()
        
        self.render_skip_count = 0
    
    def cleanup(self) -> None:
        """Clean up resources and reset state for proper mode transitions."""
        try:
            # Print performance report before cleanup if enabled
            if self.enable_optimization and self.profiler:
                print("\n[PERFORMANCE] Final performance report:")
                self.print_performance_report()
            
            # Reset interaction state
            self.state.reset()
            
            # Clear mouse callback to prevent dangling references
            if hasattr(self, 'window_name'):
                try:
                    cv2.setMouseCallback(self.window_name, lambda *args: None)
                except Exception as e:
                    print(f"[WARN] Error clearing mouse callback: {e}")
                
                # Destroy the window
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception as e:
                    print(f"[WARN] Error destroying window {self.window_name}: {e}")
            
            # Clear references to prevent memory leaks
            self.selection_callback = None
            
            print("[INFO] Interactive inspect mode cleaned up successfully.")
            
        except Exception as e:
            print(f"[WARN] Error during interaction manager cleanup: {e}")


# Default selection callback for console output
def default_selection_callback(shape_index: Optional[int], shapes: List[Dict[str, Any]]) -> None:
    """
    Default callback for selection events that prints to console.
    
    Args:
        shape_index: Index of selected shape or None
        shapes: List of all shapes
    """
    if shape_index is not None and 0 <= shape_index < len(shapes):
        shape = shapes[shape_index]
        if shape["type"] == "circle":
            print(f"[SELECTED] Circle - Diameter: {shape['diameter_mm']:.1f} mm")
        else:
            print(f"[SELECTED] Rectangle - Width: {shape['width_mm']:.1f} mm, Height: {shape['height_mm']:.1f} mm")
    else:
        print("[SELECTED] None (click on background)")


# Utility functions for interaction management

def create_interaction_manager(shapes: List[Dict[str, Any]], warped_image: np.ndarray,
                              display_height: int = 800, 
                              hover_snap_distance_mm: float = 10.0,
                              enable_performance_optimization: bool = True) -> InteractionManager:
    """
    Create and configure an interaction manager with performance optimization.
    
    Args:
        shapes: List of detected shape data dictionaries
        warped_image: The warped A4 background image
        display_height: Height for the display window
        hover_snap_distance_mm: Distance threshold for hover snapping
        enable_performance_optimization: Enable performance optimizations
        
    Returns:
        Configured InteractionManager instance
    """
    manager = InteractionManager(shapes, warped_image, display_height, 
                               hover_snap_distance_mm, enable_performance_optimization)
    manager.set_selection_callback(default_selection_callback)
    return manager


def validate_shapes_for_interaction(shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and filter shapes for interaction compatibility.
    
    Args:
        shapes: List of shape data dictionaries
        
    Returns:
        List of valid shapes that can be used for interaction
    """
    from rendering import validate_shape_for_rendering
    from hit_testing import validate_shape_data
    
    valid_shapes = []
    for shape in shapes:
        if validate_shape_for_rendering(shape) and validate_shape_data(shape):
            valid_shapes.append(shape)
    
    return valid_shapes


def setup_interactive_inspect_mode(shapes: List[Dict[str, Any]], warped_image: np.ndarray,
                                  window_name: str = "Inspect Mode",
                                  enable_performance_optimization: bool = True) -> InteractionManager:
    """
    Complete setup for interactive inspect mode with performance optimization.
    
    Args:
        shapes: List of detected shape data dictionaries
        warped_image: The warped A4 background image
        window_name: Name for the OpenCV window
        enable_performance_optimization: Enable performance optimizations
        
    Returns:
        Configured and ready InteractionManager instance
    """
    # Validate shapes
    valid_shapes = validate_shapes_for_interaction(shapes)
    
    # Create interaction manager with optimization
    manager = create_interaction_manager(valid_shapes, warped_image, 
                                       enable_performance_optimization=enable_performance_optimization)
    
    # Setup window and display initial state
    manager.setup_window(window_name)
    manager.show_initial_render()
    
    # Print summary
    manager.print_shape_summary()
    
    if enable_performance_optimization:
        print("[INFO] Performance optimization enabled for smooth interaction")
    
    return manager