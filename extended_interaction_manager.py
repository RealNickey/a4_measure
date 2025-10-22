"""
Extended Interaction Manager for Manual Shape Selection

This module extends the existing InteractionManager to support manual shape selection
modes alongside automatic detection. It coordinates between automatic hit testing
and manual selection workflows with seamless mode switching.

Requirements addressed: 3.1, 3.4, 4.1, 4.2
"""

import math

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable

from interaction_manager import InteractionManager
from selection_mode import SelectionMode, ModeManager
from manual_selection_engine import ManualSelectionEngine
from shape_snapping_engine import ShapeSnappingEngine
from enhanced_contour_analyzer import EnhancedContourAnalyzer
from selection_overlay import SelectionOverlay
import config


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
                 enable_performance_optimization: bool = True,
                 mm_per_px_x: Optional[float] = None,
                 mm_per_px_y: Optional[float] = None):
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

        # Calibration scale factors (default to config PX_PER_MM if not provided)
        px_per_mm = getattr(config, "PX_PER_MM", 1.0) or 1.0
        default_mm_per_px = 1.0 / float(px_per_mm)
        self.mm_per_px_x = mm_per_px_x if mm_per_px_x and mm_per_px_x > 0 else default_mm_per_px
        self.mm_per_px_y = mm_per_px_y if mm_per_px_y and mm_per_px_y > 0 else default_mm_per_px
        
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

        # Manual distance measurement state
        self.distance_measurements: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
        self.distance_point1: Optional[Tuple[int, int]] = None
        self.distance_point2: Optional[Tuple[int, int]] = None
        self.min_zoom_scale = 1.0
        self.max_zoom_scale = 5.0
        self.zoom_scale = 1.0
        self.zoom_center = (
            int(self.warped_image.shape[1] / 2),
            int(self.warped_image.shape[0] / 2)
        )
        h_img, w_img = self.warped_image.shape[:2]
        self.zoom_roi: Tuple[int, int, int, int] = (0, 0, w_img, h_img)

        # Display/window mapping state
        self.base_display_width = w_img
        self.base_display_height = h_img
        self.window_content_width = self.display_size[0]
        self.window_content_height = self.display_size[1]
        self.window_offset_x = 0
        self.window_offset_y = 0
        self.window_scale_x = w_img / max(1, self.window_content_width)
        self.window_scale_y = h_img / max(1, self.window_content_height)

        # Allow manual engine to reuse coordinate mapping
        self.manual_engine.set_coordinate_transform(self._display_to_original_coords)
        self.manual_engine.display_scale = 1.0
        
        # Keyboard shortcuts
        self.key_mode_cycle = ord('m')  # M key for mode cycling
        self.key_cancel_selection = 27  # ESC key for canceling selection / measurement
        self.key_toggle_confirmation = ord('c')  # C key to toggle confirmation / clear distances
        
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

    # ------------------------------------------------------------------
    # Internal helpers for rendering and manual distance mode
    # ------------------------------------------------------------------
    def _refresh_display(self) -> None:
        """Trigger a display refresh using the current rendering pipeline."""
        if not self.window_name:
            return
        display_image = self.render_with_manual_overlays()
        if display_image is None:
            return
        self.base_display_height, self.base_display_width = display_image.shape[:2]
        display = self._prepare_display_image(display_image)
        cv2.imshow(self.window_name, display)
        cv2.waitKey(1)

    def _prepare_display_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to current window size while tracking coordinate mapping."""
        target_w, target_h = self.display_size
        if self.window_name:
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(self.window_name)
                if win_w > 0 and win_h > 0:
                    target_w = win_w
                    target_h = win_h
            except Exception:
                pass

        img_h, img_w = image.shape[:2]
        scale = min(target_w / img_w, target_h / img_h) if img_w > 0 and img_h > 0 else 1.0
        if scale <= 0:
            scale = 1.0
        disp_w = max(1, int(round(img_w * scale)))
        disp_h = max(1, int(round(img_h * scale)))

        resized = image if (disp_w == img_w and disp_h == img_h) else cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        self.window_content_width = disp_w
        self.window_content_height = disp_h
        self.window_offset_x = max(0, (target_w - disp_w) // 2)
        self.window_offset_y = max(0, (target_h - disp_h) // 2)
        self.window_scale_x = img_w / float(disp_w) if disp_w > 0 else 1.0
        self.window_scale_y = img_h / float(disp_h) if disp_h > 0 else 1.0

        if disp_w == target_w and disp_h == target_h:
            return resized

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        canvas_shape = (target_h, target_w) if channels == 1 else (target_h, target_w, channels)
        canvas = np.zeros(canvas_shape, dtype=image.dtype)
        y0 = self.window_offset_y
        x0 = self.window_offset_x
        y1 = y0 + disp_h
        x1 = x0 + disp_w
        if channels == 1:
            canvas[y0:y1, x0:x1] = resized
        else:
            canvas[y0:y1, x0:x1, :] = resized
        return canvas

    def _clamp_zoom_center(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp zoom center coordinates to keep them within image bounds."""
        h, w = self.warped_image.shape[:2]
        cx = max(0, min(w - 1, x))
        cy = max(0, min(h - 1, y))
        return cx, cy

    def _apply_zoom_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Apply zoom to the provided image when manual distance mode is active."""
        h, w = image.shape[:2]

        if not self.mode_manager.is_manual_distance_mode() or self.zoom_scale <= 1.0:
            self.zoom_roi = (0, 0, w, h)
            return image

        zoom_w = max(20, int(round(w / self.zoom_scale)))
        zoom_h = max(20, int(round(h / self.zoom_scale)))

        cx, cy = self.zoom_center
        x1 = max(0, min(cx - zoom_w // 2, w - zoom_w))
        y1 = max(0, min(cy - zoom_h // 2, h - zoom_h))
        x2 = x1 + zoom_w
        y2 = y1 + zoom_h

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return image
        self.zoom_roi = (x1, y1, zoom_w, zoom_h)
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed

    def _calculate_distance_mm(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate calibrated distance in millimetres between two points."""
        dx_mm = (p2[0] - p1[0]) * self.mm_per_px_x
        dy_mm = (p2[1] - p1[1]) * self.mm_per_px_y
        return float(math.hypot(dx_mm, dy_mm))

    def _draw_distance_measurements(self, image: np.ndarray) -> None:
        """Render stored and in-progress distance measurements on the image."""
        # Completed measurements
        for p1, p2, dist_mm in self.distance_measurements:
            cv2.line(image, p1, p2, (0, 255, 255), 2)
            cv2.circle(image, p1, 5, (0, 255, 255), -1)
            cv2.circle(image, p2, 5, (0, 255, 255), -1)

            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            text = f"{dist_mm:.1f}mm"
            text_size = cv2.getTextSize(text, config.DRAW_FONT, 0.6, 2)[0]
            text_origin = (mid_x - text_size[0] // 2, mid_y - 10)
            cv2.rectangle(image,
                          (text_origin[0] - 5, text_origin[1] - text_size[1] - 5),
                          (text_origin[0] + text_size[0] + 5, text_origin[1] + 5),
                          (0, 0, 0), -1)
            cv2.putText(image, text, text_origin, config.DRAW_FONT, 0.6, (0, 255, 255), 2)

        # Measurement in progress
        if self.distance_point1 is not None:
            cv2.circle(image, self.distance_point1, 5, (255, 0, 255), -1)
            cv2.circle(image, self.distance_point1, 8, (255, 0, 255), 2)

            if self.distance_point2 is not None:
                cv2.line(image, self.distance_point1, self.distance_point2, (255, 0, 255), 2)
                cv2.circle(image, self.distance_point2, 5, (255, 0, 255), -1)
                cv2.circle(image, self.distance_point2, 8, (255, 0, 255), 2)

                dist_mm = self._calculate_distance_mm(self.distance_point1, self.distance_point2)
                mid_x = (self.distance_point1[0] + self.distance_point2[0]) // 2
                mid_y = (self.distance_point1[1] + self.distance_point2[1]) // 2
                text = f"{dist_mm:.1f}mm"
                text_size = cv2.getTextSize(text, config.DRAW_FONT, 0.6, 2)[0]
                text_origin = (mid_x - text_size[0] // 2, mid_y - 10)
                cv2.rectangle(image,
                              (text_origin[0] - 5, text_origin[1] - text_size[1] - 5),
                              (text_origin[0] + text_size[0] + 5, text_origin[1] + 5),
                              (0, 0, 0), -1)
                cv2.putText(image, text, text_origin, config.DRAW_FONT, 0.6, (255, 0, 255), 2)

    def _get_distance_instructions(self) -> List[str]:
        """Build instructional text for manual distance mode."""
        if self.distance_point1 is None:
            return [
                "Click first point to start measurement",
                "Scroll to zoom (1x-5x), press C to clear"
            ]
        return [
            "Click second point to complete measurement",
            "Right-click cancels current measurement"
        ]

    def _clear_distance_measurements(self) -> bool:
        """Clear all stored distance measurements."""
        cleared = bool(self.distance_measurements) or self.distance_point1 is not None
        self.distance_measurements.clear()
        self.distance_point1 = None
        self.distance_point2 = None
        return cleared

    def _cancel_active_distance_measurement(self) -> bool:
        """Cancel the active distance measurement without clearing history."""
        if self.distance_point1 is None:
            return False
        self.distance_point1 = None
        self.distance_point2 = None
        return True

    def _reset_distance_mode(self) -> None:
        """Reset manual distance state and zoom to defaults."""
        self.distance_point1 = None
        self.distance_point2 = None
        self.distance_measurements.clear()
        self.zoom_scale = 1.0
        self.zoom_center = (
            int(self.warped_image.shape[1] / 2),
            int(self.warped_image.shape[0] / 2)
        )
        h_img, w_img = self.warped_image.shape[:2]
        self.zoom_roi = (0, 0, w_img, h_img)

    def _handle_manual_distance_event(self, event: int, x: int, y: int, flags: int) -> bool:
        """Handle mouse events specific to manual distance mode."""
        if x is None or y is None:
            return False
        handled = False

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.distance_point1 is None:
                self.distance_point1 = (x, y)
                print(f"[DISTANCE] First point set at ({x}, {y})")
            else:
                self.distance_point2 = (x, y)
                dist_mm = self._calculate_distance_mm(self.distance_point1, self.distance_point2)
                self.distance_measurements.append((self.distance_point1, self.distance_point2, dist_mm))
                print(f"[DISTANCE] Second point ({x}, {y}) | Distance: {dist_mm:.1f}mm")
                self.distance_point1 = None
                self.distance_point2 = None
            handled = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.distance_point1 is not None:
                self.distance_point2 = (x, y)
                handled = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._cancel_active_distance_measurement():
                print("[DISTANCE] Measurement cancelled")
                handled = True

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = flags >> 16
            if delta > 0:
                self.zoom_scale = min(self.zoom_scale * 1.2, self.max_zoom_scale)
            else:
                self.zoom_scale = max(self.zoom_scale / 1.2, self.min_zoom_scale)
            self.zoom_center = self._clamp_zoom_center(x, y)
            print(f"[ZOOM] Scale: {self.zoom_scale:.1f}x at ({self.zoom_center[0]}, {self.zoom_center[1]})")
            handled = True

        return handled

    def _display_to_original_coords(self, display_x: int, display_y: int) -> Optional[Tuple[int, int]]:
        """Convert window coordinates to original image coordinates (handles zoom/letterbox)."""
        adj_x = display_x - self.window_offset_x
        adj_y = display_y - self.window_offset_y
        if adj_x < 0 or adj_y < 0 or adj_x >= self.window_content_width or adj_y >= self.window_content_height:
            return None

        if self.window_content_width <= 0 or self.window_content_height <= 0:
            return None

        norm_x = adj_x / self.window_content_width
        norm_y = adj_y / self.window_content_height

        if self.base_display_width <= 0 or self.base_display_height <= 0:
            return None

        img_x = norm_x * (self.base_display_width - 1)
        img_y = norm_y * (self.base_display_height - 1)

        if self.mode_manager.is_manual_distance_mode() and self.zoom_scale > 1.0:
            x1, y1, zoom_w, zoom_h = self.zoom_roi
            denom_x = max(1, self.base_display_width - 1)
            denom_y = max(1, self.base_display_height - 1)
            orig_x = x1 + (img_x / denom_x) * max(1, zoom_w - 1)
            orig_y = y1 + (img_y / denom_y) * max(1, zoom_h - 1)
        else:
            orig_x = img_x
            orig_y = img_y

        h, w = self.warped_image.shape[:2]
        orig_x = int(round(max(0, min(w - 1, orig_x))))
        orig_y = int(round(max(0, min(h - 1, orig_y))))
        return orig_x, orig_y
    
    def handle_key_press(self, key: int) -> bool:
        """
        Handle keyboard input for mode switching and selection control.
        
        Args:
            key: Key code from OpenCV
            
        Returns:
            True if the key was handled and requires re-rendering
        """
        if key == self.key_mode_cycle:
            # Cycle to next mode with optimized switching
            old_mode = self.mode_manager.get_current_mode()
            
            # Cancel any active manual selection before switching
            if self.manual_engine.is_selecting():
                self.manual_engine.cancel_selection()
            
            # Switch mode
            new_mode = self.mode_manager.cycle_mode()
            
            # Clear any previous manual results when switching to auto mode
            if new_mode == SelectionMode.AUTO:
                self.last_manual_result = None
                self.show_shape_confirmation = False
            
            # Reset manual distance state when leaving the mode
            if old_mode == SelectionMode.MANUAL_DISTANCE and new_mode != SelectionMode.MANUAL_DISTANCE:
                self._cancel_active_distance_measurement()
                self.zoom_scale = 1.0
                self.zoom_center = (
                    int(self.warped_image.shape[1] / 2),
                    int(self.warped_image.shape[0] / 2)
                )
                h_img, w_img = self.warped_image.shape[:2]
                self.zoom_roi = (0, 0, w_img, h_img)

            print(f"[INFO] Mode switched from {old_mode.value} to {new_mode.value}")

            if new_mode == SelectionMode.MANUAL_DISTANCE:
                print("[INFO] Manual distance mode controls:")
                print("  - Left click two points to measure distance")
                print("  - Mouse wheel to zoom (1x-5x)")
                print("  - Right click cancels current measurement")
                print("  - Press 'C' to clear stored measurements")

            self._refresh_display()
            return True

        elif key == self.key_cancel_selection:
            handled = False

            if self.mode_manager.is_manual_distance_mode():
                handled = self._cancel_active_distance_measurement()
                if handled:
                    print("[DISTANCE] Measurement cancelled")
            elif self.manual_engine.is_selecting():
                self.manual_engine.cancel_selection()
                print("[INFO] Manual selection cancelled")
                handled = True
            
            if not handled and self.show_shape_confirmation:
                self.show_shape_confirmation = False
                self.last_manual_result = None
                print("[INFO] Shape confirmation cleared")
                handled = True

            if handled:
                self._refresh_display()
            return handled
                
        elif key == self.key_toggle_confirmation:
            if self.mode_manager.is_manual_distance_mode():
                if self._clear_distance_measurements():
                    print("[DISTANCE] All measurements cleared")
                    self._refresh_display()
                    return True
                return False

            # Toggle shape confirmation display for manual shape mode
            if self.last_manual_result is not None:
                self.show_shape_confirmation = not self.show_shape_confirmation
                print(f"[INFO] Shape confirmation {'enabled' if self.show_shape_confirmation else 'disabled'}")
                self._refresh_display()
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
        Render the current state with manual selection overlays with performance optimization.
        
        Returns:
            Rendered image with manual overlays or None if rendering skipped
        """
        # Get base rendered image
        base_image = self.render_current_state()
        if base_image is None:
            return None
        
        result = base_image.copy()
        current_mode = self.mode_manager.get_current_mode()

        if current_mode == SelectionMode.MANUAL_DISTANCE:
            # Draw stored and in-progress measurements before zoom
            self._draw_distance_measurements(result)
            result = self._apply_zoom_if_needed(result)

            # Mode indicator with zoom info
            mode_text = self.mode_manager.get_mode_indicator()
            result = self.selection_overlay.render_mode_indicator(
                result,
                mode_text,
                additional_info=f"Zoom: {self.zoom_scale:.1f}x"
            )

            # Instruction overlay
            instructions = self._get_distance_instructions()
            if instructions:
                result = self.selection_overlay.render_instruction_overlay(result, instructions)

            return result

        # Manual rectangle/circle selection overlays
        if self.mode_manager.is_manual_mode():
            selection_rect = self.manual_engine.get_current_selection_rect()
            if selection_rect is not None:
                x, y, w, h = selection_rect
                result = self.selection_overlay.render_selection_rectangle(result, (int(x), int(y), int(w), int(h)), active=True)

            if self.show_shape_confirmation and self.last_manual_result is not None:
                try:
                    display_result = self._transform_shape_result_to_display(self.last_manual_result)
                    result = self.selection_overlay.render_shape_confirmation(result, display_result)

                    self.confirmation_timer += 1
                    if self.confirmation_timer >= self.confirmation_duration:
                        self.show_shape_confirmation = False
                        self.confirmation_timer = 0
                except Exception as e:
                    print(f"[WARN] Shape confirmation rendering error: {e}")
                    self.show_shape_confirmation = False

        # Render automatic hover/select highlights already baked into base image
        mode_text = self.mode_manager.get_mode_indicator()
        result = self.selection_overlay.render_mode_indicator(result, mode_text)

        # Instruction overlay for remaining modes
        instructions: List[str] = []
        if current_mode == SelectionMode.AUTO:
            instructions = [
                "Hover to preview, click to inspect",
                "Press 'M' to switch modes"
            ]
        elif current_mode == SelectionMode.MANUAL_RECTANGLE:
            instructions = [
                "Click and drag to select a rectangle",
                "Right-click cancels selection"
            ]
        elif current_mode == SelectionMode.MANUAL_CIRCLE:
            instructions = [
                "Click and drag to select a circle",
                "Right-click cancels selection"
            ]

        if instructions:
            result = self.selection_overlay.render_instruction_overlay(result, instructions)

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

        coord = self._display_to_original_coords(x, y)
        orig_x = coord[0] if coord is not None else None
        orig_y = coord[1] if coord is not None else None

        if self.mode_manager.is_manual_distance_mode():
            needs_render = self._handle_manual_distance_event(event, orig_x, orig_y, flags)

        # Handle manual selection events if in rectangle/circle modes
        elif self.mode_manager.is_manual_mode():
            manual_handled = self.handle_manual_mouse_event(event, x, y, flags, userdata)
            if manual_handled:
                needs_render = True
            # Don't process automatic events if manual selection is active
            elif self.manual_engine.is_selecting():
                needs_render = False
            else:
                # Allow automatic hover in manual mode when not actively selecting
                if event == cv2.EVENT_MOUSEMOVE and coord is not None:
                    needs_render = self.handle_mouse_move(orig_x, orig_y)
        else:
            # Handle automatic mode events (original behavior)
            if event == cv2.EVENT_MOUSEMOVE and coord is not None:
                needs_render = self.handle_mouse_move(orig_x, orig_y)
            elif event == cv2.EVENT_LBUTTONDOWN and coord is not None:
                needs_render = self.handle_mouse_click(orig_x, orig_y)
        
        # Re-render immediately for better responsiveness
        if needs_render:
            self._refresh_display()
    
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
        print("  M - Cycle selection mode (AUTO → MANUAL RECT → MANUAL CIRCLE → MANUAL DIST)")
        print("  ESC - Cancel active selection or measurement")
        print("  C - Toggle confirmation / clear distance measurements")
        print("  Mouse wheel (MANUAL DIST) - Zoom in/out (1x-5x)")
    
    def show_initial_render(self) -> None:
        """Display the initial rendered state with manual overlays."""
        self._refresh_display()
    
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

        if "center" in result:
            cx, cy = result["center"]
            result["center"] = (int(round(cx)), int(round(cy)))

        if result.get("type") == "circle" and "radius" in result:
            result["radius"] = float(result["radius"])

        if result.get("type") == "rectangle" and "contour" in result and result["contour"] is not None:
            contour = np.asarray(result["contour"], dtype=np.float32)
            result["contour"] = np.round(contour).astype(np.int32)

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
            self._reset_distance_mode()
            
            print("[INFO] Manual interaction components cleaned up")
            
        except Exception as e:
            print(f"[WARN] Error during manual selection cleanup: {e}")
        
        # Call parent cleanup
        super().cleanup()


# Utility functions for extended interaction management

def create_extended_interaction_manager(shapes: List[Dict[str, Any]], warped_image: np.ndarray,
                                      display_height: int = 800, 
                                      hover_snap_distance_mm: float = 10.0,
                                      enable_performance_optimization: bool = True,
                                      mm_per_px: Optional[Tuple[float, float]] = None) -> ExtendedInteractionManager:
    """
    Create and configure an extended interaction manager with manual selection support.
    
    Args:
        shapes: List of detected shape data dictionaries
        warped_image: The warped A4 background image
        display_height: Height for the display window
        hover_snap_distance_mm: Distance threshold for hover snapping
        enable_performance_optimization: Enable performance optimizations
        mm_per_px: Optional tuple of (mm_per_px_x, mm_per_px_y) for calibrated distance
        
    Returns:
        Configured ExtendedInteractionManager instance
    """
    from interaction_manager import default_selection_callback
    
    mm_per_px_x = mm_per_px[0] if mm_per_px else None
    mm_per_px_y = mm_per_px[1] if mm_per_px else None

    manager = ExtendedInteractionManager(
        shapes,
        warped_image,
        display_height,
        hover_snap_distance_mm,
        enable_performance_optimization,
        mm_per_px_x=mm_per_px_x,
        mm_per_px_y=mm_per_px_y
    )
    manager.set_selection_callback(default_selection_callback)
    return manager


def setup_extended_interactive_inspect_mode(shapes: List[Dict[str, Any]], warped_image: np.ndarray,
                                          window_name: str = "Extended Inspect Mode",
                                          enable_performance_optimization: bool = True,
                                          mm_per_px: Optional[Tuple[float, float]] = None) -> ExtendedInteractionManager:
    """
    Complete setup for extended interactive inspect mode with manual selection support.
    
    Args:
        shapes: List of detected shape data dictionaries
        warped_image: The warped A4 background image
        window_name: Name for the OpenCV window
        enable_performance_optimization: Enable performance optimizations
        mm_per_px: Optional tuple of (mm_per_px_x, mm_per_px_y) for calibrated distance
        
    Returns:
        Configured and ready ExtendedInteractionManager instance
    """
    from interaction_manager import validate_shapes_for_interaction
    
    # Validate shapes
    valid_shapes = validate_shapes_for_interaction(shapes)
    
    # Create extended interaction manager
    manager = create_extended_interaction_manager(
        valid_shapes,
        warped_image,
        enable_performance_optimization=enable_performance_optimization,
        mm_per_px=mm_per_px
    )
    
    # Setup window and display initial state
    manager.setup_window(window_name)
    manager.show_initial_render()
    
    # Print summary
    manager.print_shape_summary()
    
    if enable_performance_optimization:
        print("[INFO] Performance optimization enabled for smooth interaction")
    
    print("[INFO] Extended inspect mode ready - supports automatic, manual shape, and distance measurement modes")
    
    return manager