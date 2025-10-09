"""
Selection Overlay Rendering System for Manual Shape Selection

This module provides comprehensive visual feedback and overlay rendering for manual
shape selection operations, including selection rectangles, mode indicators, and
shape confirmation feedback.

Requirements addressed: 3.2, 4.3, 4.5
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from config import DRAW_FONT
from utils import draw_text


class SelectionOverlay:
    """
    Handles rendering of selection overlays and visual feedback for manual shape selection.
    
    Provides:
    - Real-time selection rectangle rendering with semi-transparent overlay
    - Mode indicator display in corner of inspection window
    - Visual confirmation feedback when shapes are successfully detected
    - Customizable colors and styling for different visual elements
    """
    
    def __init__(self):
        """Initialize the selection overlay renderer with default styling."""
        # Colors for different visual elements (BGR format for OpenCV)
        self.selection_color = (0, 255, 255)  # Yellow for selection rectangle
        self.mode_indicator_color = (255, 255, 255)  # White for mode text
        self.mode_background_color = (0, 0, 0)  # Black background for mode text
        self.confirmation_color = (0, 255, 0)  # Green for shape confirmation
        self.error_color = (0, 0, 255)  # Red for error feedback
        
        # Rendering parameters
        self.selection_thickness = 2
        self.selection_alpha = 0.2  # Semi-transparent overlay
        self.confirmation_thickness = 3
        self.text_font = DRAW_FONT
        self.text_scale = 0.7
        self.text_thickness = 2
        self.text_padding = 10
        
        # Animation parameters for visual feedback
        self.pulse_amplitude = 0.5
        self.pulse_frequency = 0.2
        self.frame_counter = 0
    
    def render_selection_rectangle(self, image: np.ndarray, 
                                 selection_rect: Optional[Tuple[int, int, int, int]],
                                 active: bool = True) -> np.ndarray:
        """
        Render the current selection rectangle with real-time visual feedback.
        
        Args:
            image: Image to render on
            selection_rect: Selection rectangle as (x, y, width, height) or None
            active: Whether the selection is currently active (affects styling)
            
        Returns:
            Image with selection rectangle rendered
        """
        if selection_rect is None:
            return image
        
        result = image.copy()
        x, y, w, h = selection_rect
        
        # Ensure coordinates are valid
        if w <= 0 or h <= 0:
            return result
        
        # Use brighter, more visible colors for active selection
        if active:
            # Bright cyan for high visibility during dragging
            color = (255, 255, 0)  # Bright cyan
            thickness = 3  # Thicker line for better visibility
        else:
            color = (128, 128, 128)  # Gray for inactive
            thickness = self.selection_thickness
        
        # Draw selection rectangle outline with enhanced visibility
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        # Add contrasting inner outline for better visibility
        if active:
            inner_color = (0, 0, 0)  # Black inner outline
            cv2.rectangle(result, (x + 1, y + 1), (x + w - 1, y + h - 1), inner_color, 1)
        
        # Add semi-transparent overlay for active selections
        if active and self.selection_alpha > 0:
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(result, 1 - self.selection_alpha, overlay, self.selection_alpha, 0, result)
        
        # Add corner markers for better visibility
        self._draw_corner_markers(result, x, y, w, h, color)
        
        # Add selection info text with better visibility
        if active:
            info_text = f"{w}x{h}"
            self._draw_selection_info(result, x, y, w, h, info_text, color)
        
        return result
    
    def render_mode_indicator(self, image: np.ndarray, mode_text: str, 
                            additional_info: Optional[str] = None) -> np.ndarray:
        """
        Render the current mode indicator in the corner of the image.
        
        Args:
            image: Image to render on
            mode_text: Text to display for current mode
            additional_info: Optional additional information to display
            
        Returns:
            Image with mode indicator rendered
        """
        result = image.copy()
        
        # Prepare text lines
        text_lines = [mode_text]
        if additional_info:
            text_lines.append(additional_info)
        
        # Calculate total text dimensions
        line_heights = []
        line_widths = []
        
        for line in text_lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                line, self.text_font, self.text_scale, self.text_thickness
            )
            line_widths.append(text_width)
            line_heights.append(text_height + baseline)
        
        max_width = max(line_widths) if line_widths else 0
        total_height = sum(line_heights) + (len(text_lines) - 1) * 5  # 5px spacing between lines
        
        # Position in top-right corner
        x = image.shape[1] - max_width - self.text_padding * 2
        y = self.text_padding
        
        # Draw background rectangle
        bg_rect = (x - 5, y - 5, max_width + self.text_padding * 2 + 10, total_height + 10)
        cv2.rectangle(result, 
                     (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]),
                     self.mode_background_color, -1)
        
        # Draw border
        cv2.rectangle(result, 
                     (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]),
                     self.mode_indicator_color, 1)
        
        # Draw text lines
        current_y = y + line_heights[0]
        for i, line in enumerate(text_lines):
            text_x = x + (max_width - line_widths[i]) // 2  # Center align
            cv2.putText(result, line, (text_x, current_y), 
                       self.text_font, self.text_scale, self.mode_indicator_color, self.text_thickness)
            if i < len(text_lines) - 1:
                current_y += line_heights[i] + 5  # Add spacing
        
        return result
    
    def render_shape_confirmation(self, image: np.ndarray, 
                                shape_result: Dict[str, Any],
                                animate: bool = True) -> np.ndarray:
        """
        Render visual confirmation when a shape is successfully detected.
        
        Args:
            image: Image to render on
            shape_result: Shape detection result
            animate: Whether to apply animation effects
            
        Returns:
            Image with shape confirmation rendered
        """
        result = image.copy()
        
        # Calculate animation factor if enabled
        animation_factor = 1.0
        if animate:
            animation_factor = 1.0 + self.pulse_amplitude * np.sin(
                self.frame_counter * self.pulse_frequency * 2 * np.pi
            )
            self.frame_counter += 1
        
        # Adjust color intensity based on animation
        base_color = np.array(self.confirmation_color, dtype=float)
        animated_color = base_color * animation_factor
        color = tuple(int(min(255, max(0, c))) for c in animated_color)
        
        if shape_result["type"] == "circle":
            self._render_circle_confirmation(result, shape_result, color)
        elif shape_result["type"] == "rectangle":
            self._render_rectangle_confirmation(result, shape_result, color)
        
        # Add confidence score display
        self._render_confidence_score(result, shape_result, color)
        
        return result
    
    def render_error_feedback(self, image: np.ndarray, error_message: str,
                            position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Render error feedback message on the image.
        
        Args:
            image: Image to render on
            error_message: Error message to display
            position: Optional position for the message (defaults to center)
            
        Returns:
            Image with error feedback rendered
        """
        result = image.copy()
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            error_message, self.text_font, self.text_scale, self.text_thickness
        )
        
        # Default to center position if not specified
        if position is None:
            x = (image.shape[1] - text_width) // 2
            y = (image.shape[0] + text_height) // 2
        else:
            x, y = position
        
        # Draw background rectangle
        padding = 10
        cv2.rectangle(result,
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     self.mode_background_color, -1)
        
        # Draw border
        cv2.rectangle(result,
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     self.error_color, 2)
        
        # Draw error text
        cv2.putText(result, error_message, (x, y), 
                   self.text_font, self.text_scale, self.error_color, self.text_thickness)
        
        return result
    
    def render_instruction_overlay(self, image: np.ndarray, instructions: List[str]) -> np.ndarray:
        """
        Render instruction text overlay for user guidance.
        
        Args:
            image: Image to render on
            instructions: List of instruction strings
            
        Returns:
            Image with instruction overlay rendered
        """
        result = image.copy()
        
        if not instructions:
            return result
        
        # Calculate dimensions for all instruction lines
        line_heights = []
        line_widths = []
        
        for instruction in instructions:
            (text_width, text_height), baseline = cv2.getTextSize(
                instruction, self.text_font, self.text_scale - 0.1, self.text_thickness - 1
            )
            line_widths.append(text_width)
            line_heights.append(text_height + baseline)
        
        max_width = max(line_widths) if line_widths else 0
        total_height = sum(line_heights) + (len(instructions) - 1) * 3  # 3px spacing
        
        # Position at bottom-left corner
        x = self.text_padding
        y = image.shape[0] - total_height - self.text_padding
        
        # Draw background
        bg_rect = (x - 5, y - 5, max_width + 20, total_height + 10)
        overlay = result.copy()
        cv2.rectangle(overlay, 
                     (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]),
                     self.mode_background_color, -1)
        cv2.addWeighted(result, 0.7, overlay, 0.3, 0, result)
        
        # Draw instruction text
        current_y = y + line_heights[0]
        for i, instruction in enumerate(instructions):
            cv2.putText(result, instruction, (x, current_y), 
                       self.text_font, self.text_scale - 0.1, 
                       self.mode_indicator_color, self.text_thickness - 1)
            if i < len(instructions) - 1:
                current_y += line_heights[i] + 3
        
        return result
    
    def _draw_corner_markers(self, image: np.ndarray, x: int, y: int, w: int, h: int, 
                           color: Tuple[int, int, int]) -> None:
        """
        Draw corner markers for selection rectangle.
        
        Args:
            image: Image to draw on (modified in place)
            x, y, w, h: Rectangle coordinates and dimensions
            color: Color for the markers
        """
        marker_size = 10
        thickness = 2
        
        # Top-left corner
        cv2.line(image, (x, y), (x + marker_size, y), color, thickness)
        cv2.line(image, (x, y), (x, y + marker_size), color, thickness)
        
        # Top-right corner
        cv2.line(image, (x + w, y), (x + w - marker_size, y), color, thickness)
        cv2.line(image, (x + w, y), (x + w, y + marker_size), color, thickness)
        
        # Bottom-left corner
        cv2.line(image, (x, y + h), (x + marker_size, y + h), color, thickness)
        cv2.line(image, (x, y + h), (x, y + h - marker_size), color, thickness)
        
        # Bottom-right corner
        cv2.line(image, (x + w, y + h), (x + w - marker_size, y + h), color, thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - marker_size), color, thickness)
    
    def _draw_selection_info(self, image: np.ndarray, x: int, y: int, w: int, h: int,
                           info_text: str, color: Tuple[int, int, int]) -> None:
        """
        Draw selection information text.
        
        Args:
            image: Image to draw on (modified in place)
            x, y, w, h: Rectangle coordinates and dimensions
            info_text: Information text to display
            color: Color for the text
        """
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            info_text, self.text_font, 0.5, 1
        )
        
        # Position text at top of selection rectangle
        text_x = x + (w - text_width) // 2
        text_y = max(y - 5, text_height + 5)  # Above rectangle or at top of image
        
        # Draw background
        cv2.rectangle(image,
                     (text_x - 3, text_y - text_height - 3),
                     (text_x + text_width + 3, text_y + 3),
                     self.mode_background_color, -1)
        
        # Draw text
        cv2.putText(image, info_text, (text_x, text_y), 
                   self.text_font, 0.5, color, 1)
    
    def _render_circle_confirmation(self, image: np.ndarray, shape_result: Dict[str, Any],
                                  color: Tuple[int, int, int]) -> None:
        """
        Render confirmation for detected circle.
        
        Args:
            image: Image to draw on (modified in place)
            shape_result: Circle detection result
            color: Color for rendering
        """
        center = tuple(map(int, shape_result["center"]))
        radius = int(shape_result["radius"])
        
        # Draw circle outline with confirmation color
        cv2.circle(image, center, radius, color, self.confirmation_thickness)
        
        # Draw center point with animated size
        center_size = int(3 * (1.0 + 0.5 * np.sin(self.frame_counter * 0.3)))
        cv2.circle(image, center, center_size, color, -1)
        
        # Draw diameter line
        cv2.line(image, 
                (center[0] - radius, center[1]), 
                (center[0] + radius, center[1]), 
                color, 2)
    
    def _render_rectangle_confirmation(self, image: np.ndarray, shape_result: Dict[str, Any],
                                     color: Tuple[int, int, int]) -> None:
        """
        Render confirmation for detected rectangle.
        
        Args:
            image: Image to draw on (modified in place)
            shape_result: Rectangle detection result
            color: Color for rendering
        """
        if "contour" in shape_result and shape_result["contour"] is not None:
            # Draw the detected rectangle contour
            cv2.drawContours(image, [shape_result["contour"]], -1, color, self.confirmation_thickness)
            
            # Draw center point
            if "center" in shape_result:
                center = tuple(map(int, shape_result["center"]))
                cv2.circle(image, center, 3, color, -1)
    
    def _render_confidence_score(self, image: np.ndarray, shape_result: Dict[str, Any],
                               color: Tuple[int, int, int]) -> None:
        """
        Render confidence score for detected shape.
        
        Args:
            image: Image to draw on (modified in place)
            shape_result: Shape detection result
            color: Color for rendering
        """
        if "confidence_score" not in shape_result:
            return
        
        confidence = shape_result["confidence_score"]
        confidence_text = f"Confidence: {confidence:.1%}"
        
        # Position near the detected shape
        if "center" in shape_result:
            center = shape_result["center"]
            text_x = int(center[0] - 50)
            text_y = int(center[1] + 30)
        else:
            text_x = 20
            text_y = image.shape[0] - 50
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            confidence_text, self.text_font, 0.6, 1
        )
        
        # Draw background
        cv2.rectangle(image,
                     (text_x - 3, text_y - text_height - 3),
                     (text_x + text_width + 3, text_y + 3),
                     self.mode_background_color, -1)
        
        # Draw confidence text
        cv2.putText(image, confidence_text, (text_x, text_y), 
                   self.text_font, 0.6, color, 1)
    
    def reset_animation(self) -> None:
        """Reset animation counters."""
        self.frame_counter = 0
    
    def set_colors(self, selection_color: Optional[Tuple[int, int, int]] = None,
                  confirmation_color: Optional[Tuple[int, int, int]] = None,
                  error_color: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Set custom colors for overlay elements.
        
        Args:
            selection_color: Color for selection rectangle (BGR)
            confirmation_color: Color for shape confirmation (BGR)
            error_color: Color for error feedback (BGR)
        """
        if selection_color is not None:
            self.selection_color = selection_color
        if confirmation_color is not None:
            self.confirmation_color = confirmation_color
        if error_color is not None:
            self.error_color = error_color
    
    def set_transparency(self, selection_alpha: float) -> None:
        """
        Set transparency level for selection overlay.
        
        Args:
            selection_alpha: Alpha value (0.0 to 1.0)
        """
        self.selection_alpha = max(0.0, min(1.0, selection_alpha))


# Utility functions for overlay rendering

def create_selection_overlay() -> SelectionOverlay:
    """
    Create a selection overlay with default settings.
    
    Returns:
        Configured SelectionOverlay instance
    """
    return SelectionOverlay()


def render_complete_manual_feedback(image: np.ndarray, 
                                  selection_rect: Optional[Tuple[int, int, int, int]],
                                  mode_text: str,
                                  shape_result: Optional[Dict[str, Any]] = None,
                                  error_message: Optional[str] = None,
                                  instructions: Optional[List[str]] = None) -> np.ndarray:
    """
    Render complete manual selection feedback on an image.
    
    Args:
        image: Base image to render on
        selection_rect: Current selection rectangle
        mode_text: Current mode indicator text
        shape_result: Optional shape detection result for confirmation
        error_message: Optional error message to display
        instructions: Optional instruction text
        
    Returns:
        Image with complete manual feedback rendered
    """
    overlay = create_selection_overlay()
    result = image.copy()
    
    # Render selection rectangle
    if selection_rect is not None:
        result = overlay.render_selection_rectangle(result, selection_rect)
    
    # Render mode indicator
    result = overlay.render_mode_indicator(result, mode_text)
    
    # Render shape confirmation
    if shape_result is not None:
        result = overlay.render_shape_confirmation(result, shape_result)
    
    # Render error feedback
    if error_message is not None:
        result = overlay.render_error_feedback(result, error_message)
    
    # Render instructions
    if instructions is not None:
        result = overlay.render_instruction_overlay(result, instructions)
    
    return result


def validate_overlay_parameters(selection_rect: Optional[Tuple[int, int, int, int]],
                              image_shape: Tuple[int, int]) -> bool:
    """
    Validate overlay rendering parameters.
    
    Args:
        selection_rect: Selection rectangle to validate
        image_shape: Shape of the target image (height, width)
        
    Returns:
        True if parameters are valid for rendering
    """
    if selection_rect is None:
        return True
    
    x, y, w, h = selection_rect
    img_h, img_w = image_shape[:2]
    
    # Check if rectangle is within image bounds
    return (x >= 0 and y >= 0 and 
            x + w <= img_w and y + h <= img_h and
            w > 0 and h > 0)