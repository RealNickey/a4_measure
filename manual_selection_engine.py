"""
Manual Selection Engine for Interactive Shape Selection

This module provides the core functionality for manual shape selection,
including selection area tracking, mouse event handling, and integration
with the shape snapping system.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable

from selection_mode import SelectionMode
from manual_selection_errors import (
    ManualSelectionError, SelectionValidationError, 
    ManualSelectionValidator, ErrorRecoveryManager, UserFeedbackManager,
    create_error_context, validate_manual_selection_operation
)
import config


@dataclass
class SelectionState:
    """
    Tracks the current state of manual selection operations.
    
    Manages selection coordinates, active selection state, and provides
    methods for calculating selection geometry.
    """
    is_selecting: bool = False
    start_point: Optional[Tuple[int, int]] = None
    current_point: Optional[Tuple[int, int]] = None
    selection_rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    
    def start_selection(self, x: int, y: int) -> None:
        """
        Start a new selection operation.
        
        Args:
            x: Starting X coordinate
            y: Starting Y coordinate
        """
        self.is_selecting = True
        self.start_point = (x, y)
        self.current_point = (x, y)
        self.selection_rect = None
    
    def update_selection(self, x: int, y: int) -> None:
        """
        Update the current selection area.
        
        Args:
            x: Current X coordinate
            y: Current Y coordinate
        """
        if self.is_selecting and self.start_point is not None:
            self.current_point = (x, y)
            self.selection_rect = self._calculate_selection_rect()
    
    def complete_selection(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Complete the selection operation and return the final rectangle.
        
        Returns:
            Selection rectangle as (x, y, width, height) or None if invalid
        """
        if self.is_selecting and self.selection_rect is not None:
            final_rect = self.selection_rect
            self.is_selecting = False
            return final_rect
        return None
    
    def cancel_selection(self) -> None:
        """Cancel the current selection operation."""
        self.is_selecting = False
        self.start_point = None
        self.current_point = None
        self.selection_rect = None
    
    def get_current_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the current selection rectangle.
        
        Returns:
            Current selection rectangle as (x, y, width, height) or None
        """
        return self.selection_rect
    
    def is_valid_selection(self, min_size: int = 20) -> bool:
        """
        Check if the current selection is valid (large enough).
        
        Args:
            min_size: Minimum width and height for valid selection
            
        Returns:
            True if selection is valid, False otherwise
        """
        if self.selection_rect is None:
            return False
        
        x, y, w, h = self.selection_rect
        return w >= min_size and h >= min_size
    
    def get_selection_area(self) -> int:
        """
        Get the area of the current selection.
        
        Returns:
            Area in pixels, or 0 if no valid selection
        """
        if self.selection_rect is None:
            return 0
        
        x, y, w, h = self.selection_rect
        return w * h
    
    def get_selection_center(self) -> Optional[Tuple[int, int]]:
        """
        Get the center point of the current selection.
        
        Returns:
            Center point as (x, y) or None if no selection
        """
        if self.selection_rect is None:
            return None
        
        x, y, w, h = self.selection_rect
        return (x + w // 2, y + h // 2)
    
    def _calculate_selection_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate the selection rectangle from start and current points.
        
        Returns:
            Rectangle as (x, y, width, height) or None if invalid
        """
        if self.start_point is None or self.current_point is None:
            return None
        
        x1, y1 = self.start_point
        x2, y2 = self.current_point
        
        # Calculate rectangle coordinates (top-left corner and dimensions)
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        return (x, y, w, h)
    
    def reset(self) -> None:
        """Reset the selection state to initial values."""
        self.is_selecting = False
        self.start_point = None
        self.current_point = None
        self.selection_rect = None


class ManualSelectionEngine:
    """
    Core engine for manual shape selection operations.
    
    Handles the complete workflow of manual selection including:
    - Mouse event processing for click-and-drag operations
    - Selection area tracking and validation
    - Integration with shape snapping algorithms
    - Coordinate transformation between display and original image space
    """
    
    def __init__(self, display_scale: float = 1.0, min_selection_size: int = 20):
        """
        Initialize the manual selection engine.
        
        Args:
            display_scale: Scaling factor between display and original coordinates
            min_selection_size: Minimum selection size in pixels
        """
        self.selection_state = SelectionState()
        self.display_scale = display_scale
        self.min_selection_size = min_selection_size
        
        # Error handling and validation
        self.validator = ManualSelectionValidator()
        self.error_recovery = ErrorRecoveryManager()
        self.feedback_manager = UserFeedbackManager()
        
        # Configuration parameters
        self.selection_timeout_ms = getattr(config, 'MANUAL_SELECTION_TIMEOUT_MS', 5000)
        self.enable_validation = getattr(config, 'ENABLE_SELECTION_VALIDATION', True)
        
        # Callbacks for external integration
        self.selection_start_callback: Optional[Callable[[int, int], None]] = None
        self.selection_update_callback: Optional[Callable[[int, int], None]] = None
        self.selection_complete_callback: Optional[Callable[[Tuple[int, int, int, int]], None]] = None
        self.selection_cancel_callback: Optional[Callable[[], None]] = None
        self.error_callback: Optional[Callable[[ManualSelectionError], None]] = None
    
    def set_display_scale(self, scale: float) -> None:
        """
        Set the display scaling factor.
        
        Args:
            scale: Scaling factor for coordinate transformation
        """
        self.display_scale = scale
    
    def set_callbacks(self, 
                     start_callback: Optional[Callable[[int, int], None]] = None,
                     update_callback: Optional[Callable[[int, int], None]] = None,
                     complete_callback: Optional[Callable[[Tuple[int, int, int, int]], None]] = None,
                     cancel_callback: Optional[Callable[[], None]] = None) -> None:
        """
        Set callback functions for selection events.
        
        Args:
            start_callback: Called when selection starts with (x, y)
            update_callback: Called when selection updates with (x, y)
            complete_callback: Called when selection completes with (x, y, w, h)
            cancel_callback: Called when selection is cancelled
        """
        self.selection_start_callback = start_callback
        self.selection_update_callback = update_callback
        self.selection_complete_callback = complete_callback
        self.selection_cancel_callback = cancel_callback
    
    def start_selection(self, display_x: int, display_y: int) -> None:
        """
        Start a new manual selection operation.
        
        Args:
            display_x: X coordinate in display window
            display_y: Y coordinate in display window
        """
        # Transform to original coordinates
        orig_x, orig_y = self._transform_display_to_original(display_x, display_y)
        
        # Start selection in original coordinate space
        self.selection_state.start_selection(orig_x, orig_y)
        
        # Call callback if provided
        if self.selection_start_callback:
            self.selection_start_callback(orig_x, orig_y)
    
    def update_selection(self, display_x: int, display_y: int) -> None:
        """
        Update the current selection area.
        
        Args:
            display_x: Current X coordinate in display window
            display_y: Current Y coordinate in display window
        """
        if not self.selection_state.is_selecting:
            return
        
        # Transform to original coordinates
        orig_x, orig_y = self._transform_display_to_original(display_x, display_y)
        
        # Update selection in original coordinate space
        self.selection_state.update_selection(orig_x, orig_y)
        
        # Call callback if provided
        if self.selection_update_callback:
            self.selection_update_callback(orig_x, orig_y)
    
    def complete_selection(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Complete the current selection operation with validation and error handling.
        
        Returns:
            Final selection rectangle as (x, y, width, height) in original coordinates,
            or None if selection is invalid
        """
        if not self.selection_state.is_selecting:
            return None
        
        try:
            # Get the current selection rectangle
            final_rect = self.selection_state.get_current_rect()
            if final_rect is None:
                raise SelectionValidationError("No selection rectangle available")
            
            # Validate selection if enabled
            if self.enable_validation:
                # We need image shape for validation, but we don't have it here
                # This will be validated by the caller with proper image context
                if not self.selection_state.is_valid_selection(self.min_selection_size):
                    raise SelectionValidationError(
                        f"Selection too small: minimum size is {self.min_selection_size}px",
                        final_rect
                    )
            
            # Complete the selection
            completed_rect = self.selection_state.complete_selection()
            
            # Add success feedback
            self.feedback_manager.add_success_message(
                f"Selection completed: {completed_rect[2]}x{completed_rect[3]}px",
                "selection_complete"
            )
            
            # Call callback if provided
            if completed_rect and self.selection_complete_callback:
                self.selection_complete_callback(completed_rect)
            
            return completed_rect
            
        except ManualSelectionError as e:
            # Handle the error with recovery
            context = create_error_context(
                "complete_selection",
                selection_rect=final_rect,
                retry_count=0
            )
            
            recovery_result = self.error_recovery.handle_selection_error(e, context)
            
            # Add error feedback
            self.feedback_manager.add_error_message(e, context)
            
            # Call error callback if provided
            if self.error_callback:
                self.error_callback(e)
            
            # Try to apply recovery if available
            if recovery_result and "adjusted_selection" in recovery_result:
                adjusted_rect = recovery_result["adjusted_selection"]
                self.selection_state.selection_rect = adjusted_rect
                return self.complete_selection()  # Retry with adjusted selection
            
            # Cancel selection on unrecoverable error
            self.cancel_selection()
            return None
    
    def cancel_selection(self) -> None:
        """Cancel the current selection operation."""
        self.selection_state.cancel_selection()
        
        # Call callback if provided
        if self.selection_cancel_callback:
            self.selection_cancel_callback()
    
    def handle_mouse_event(self, event: int, display_x: int, display_y: int, 
                          flags: int, userdata: Any = None) -> bool:
        """
        Handle mouse events for manual selection.
        
        Args:
            event: OpenCV mouse event type
            display_x: X coordinate in display window
            display_y: Y coordinate in display window
            flags: OpenCV event flags
            userdata: User data (unused)
            
        Returns:
            True if the event was handled and requires re-rendering
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start new selection
            self.start_selection(display_x, display_y)
            return True
            
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            # Update selection during drag
            self.update_selection(display_x, display_y)
            return True
            
        elif event == cv2.EVENT_LBUTTONUP:
            # Complete selection
            final_rect = self.complete_selection()
            return final_rect is not None
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Cancel selection on right click
            if self.selection_state.is_selecting:
                self.cancel_selection()
                return True
        
        return False
    
    def is_selecting(self) -> bool:
        """
        Check if currently in selection mode.
        
        Returns:
            True if actively selecting, False otherwise
        """
        return self.selection_state.is_selecting
    
    def get_current_selection_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the current selection rectangle in original coordinates.
        
        Returns:
            Current selection rectangle as (x, y, width, height) or None
        """
        return self.selection_state.get_current_rect()
    
    def get_display_selection_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the current selection rectangle in display coordinates.
        
        Returns:
            Current selection rectangle as (x, y, width, height) in display space or None
        """
        orig_rect = self.selection_state.get_current_rect()
        if orig_rect is None:
            return None
        
        x, y, w, h = orig_rect
        
        # Transform coordinates to display space
        display_x = int(x * self.display_scale)
        display_y = int(y * self.display_scale)
        display_w = int(w * self.display_scale)
        display_h = int(h * self.display_scale)
        
        return (display_x, display_y, display_w, display_h)
    
    def get_selection_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current selection.
        
        Returns:
            Dictionary with selection state information
        """
        rect = self.selection_state.get_current_rect()
        center = self.selection_state.get_selection_center()
        area = self.selection_state.get_selection_area()
        
        return {
            "is_selecting": self.selection_state.is_selecting,
            "selection_rect": rect,
            "selection_center": center,
            "selection_area": area,
            "is_valid": self.selection_state.is_valid_selection(self.min_selection_size),
            "display_scale": self.display_scale
        }
    
    def extract_selection_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the selected region from an image.
        
        Args:
            image: Source image to extract from
            
        Returns:
            Extracted image region or None if no valid selection
        """
        rect = self.selection_state.get_current_rect()
        if rect is None:
            return None
        
        x, y, w, h = rect
        
        # Ensure coordinates are within image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return None
        
        return image[y:y+h, x:x+w]
    
    def validate_selection_bounds(self, image_width: int, image_height: int) -> bool:
        """
        Validate that the current selection is within image bounds.
        
        Args:
            image_width: Width of the source image
            image_height: Height of the source image
            
        Returns:
            True if selection is within bounds, False otherwise
        """
        rect = self.selection_state.get_current_rect()
        if rect is None:
            return False
        
        x, y, w, h = rect
        
        # Check if selection is completely within image bounds
        return (x >= 0 and y >= 0 and 
                x + w <= image_width and 
                y + h <= image_height)
    
    def reset(self) -> None:
        """Reset the selection engine to initial state."""
        self.selection_state.reset()
    
    def _transform_display_to_original(self, display_x: int, display_y: int) -> Tuple[int, int]:
        """
        Transform display coordinates to original image coordinates.
        
        Args:
            display_x: X coordinate in display window
            display_y: Y coordinate in display window
            
        Returns:
            (x, y) coordinates in original image space
        """
        original_x = int(display_x / self.display_scale)
        original_y = int(display_y / self.display_scale)
        return original_x, original_y
    
    def _transform_original_to_display(self, original_x: int, original_y: int) -> Tuple[int, int]:
        """
        Transform original coordinates to display window coordinates.
        
        Args:
            original_x: X coordinate in original image
            original_y: Y coordinate in original image
            
        Returns:
            (x, y) coordinates in display window space
        """
        display_x = int(original_x * self.display_scale)
        display_y = int(original_y * self.display_scale)
        return display_x, display_y


# Utility functions for manual selection

def create_manual_selection_engine(display_scale: float = 1.0, 
                                 min_selection_size: int = 20) -> ManualSelectionEngine:
    """
    Create a configured manual selection engine.
    
    Args:
        display_scale: Scaling factor for coordinate transformation
        min_selection_size: Minimum selection size in pixels
        
    Returns:
        Configured ManualSelectionEngine instance
    """
    return ManualSelectionEngine(display_scale, min_selection_size)


def validate_selection_geometry(rect: Tuple[int, int, int, int], 
                              min_size: int = 20) -> bool:
    """
    Validate selection rectangle geometry.
    
    Args:
        rect: Rectangle as (x, y, width, height)
        min_size: Minimum width and height
        
    Returns:
        True if geometry is valid, False otherwise
    """
    if rect is None:
        return False
    
    x, y, w, h = rect
    
    # Check for valid coordinates and minimum size
    return (x >= 0 and y >= 0 and 
            w >= min_size and h >= min_size)


def calculate_selection_overlap(rect1: Tuple[int, int, int, int], 
                              rect2: Tuple[int, int, int, int]) -> float:
    """
    Calculate the overlap ratio between two selection rectangles.
    
    Args:
        rect1: First rectangle as (x, y, width, height)
        rect2: Second rectangle as (x, y, width, height)
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Calculate intersection
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left >= right or top >= bottom:
        return 0.0
    
    # Calculate areas
    intersection_area = (right - left) * (bottom - top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area