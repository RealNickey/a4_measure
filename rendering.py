"""
Selective Rendering Engine for Interactive Inspect Mode

This module provides rendering functionality for the interactive inspect mode,
including base rendering (clean A4 background), preview rendering (hover outlines),
selection rendering (full dimensions), and dynamic instruction text.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from config import DRAW_FONT, PREVIEW_COLOR, SELECTION_COLOR
from utils import draw_text


class SelectiveRenderer:
    """
    Rendering engine that provides selective display of shapes based on interaction state.
    
    Supports three rendering modes:
    - Base: Clean A4 background without any shape overlays
    - Preview: Outline-only rendering for hovered shapes
    - Selection: Full dimension rendering for selected shapes
    """
    
    def __init__(self):
        """Initialize the selective renderer."""
        pass
    
    def render_base(self, warped_image: np.ndarray) -> np.ndarray:
        """
        Create base renderer that displays clean A4 background without any shape overlays.
        
        Args:
            warped_image: The warped A4 background image
            
        Returns:
            Clean copy of the warped image without any annotations
        """
        return warped_image.copy()
    
    def render_preview(self, base_image: np.ndarray, shape: Dict[str, Any], 
                      color: Tuple[int, int, int] = None, thickness: int = 2) -> np.ndarray:
        """
        Build preview renderer for hover state that shows outline-only shape highlighting.
        
        Args:
            base_image: Base image to draw on
            shape: Shape data dictionary containing geometry information
            color: RGB color tuple for the outline (uses PREVIEW_COLOR if None)
            thickness: Line thickness for the outline
            
        Returns:
            Image with shape outline rendered
        """
        if color is None:
            color = PREVIEW_COLOR
            
        result = base_image.copy()
        self._draw_shape_outline(result, shape, color, thickness)
        return result
    
    def render_selection(self, base_image: np.ndarray, shape: Dict[str, Any],
                        color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Implement selection renderer that displays full dimensions and measurements for selected shapes.
        
        Args:
            base_image: Base image to draw on
            shape: Shape data dictionary containing geometry and measurement information
            color: RGB color tuple for the rendering (uses SELECTION_COLOR if None)
            
        Returns:
            Image with shape and full dimensions rendered
        """
        if color is None:
            color = SELECTION_COLOR
            
        result = base_image.copy()
        self._draw_shape_with_dimensions(result, shape, color)
        return result
    
    def render_instruction_text(self, image: np.ndarray, state: Dict[str, Any], 
                               shapes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Add dynamic instruction text rendering based on current interaction state.
        
        Args:
            image: Image to draw text on
            state: Current interaction state containing 'hovered' and 'selected' indices
            shapes: List of shape data for generating descriptive text
            
        Returns:
            Image with instruction text rendered
        """
        result = image.copy()
        
        # Generate instruction text based on current state
        text = self._generate_instruction_text(state, shapes)
        
        # Position text clearly at the top of the image
        text_position = (20, 40)
        text_color = (255, 255, 255)  # White text for visibility
        font_scale = 0.7
        thickness = 2
        
        draw_text(result, text, text_position, text_color, font_scale, thickness)
        return result
    
    def render_complete_state(self, warped_image: np.ndarray, state: Dict[str, Any],
                             shapes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Render the complete interaction state including base, preview, selection, and instructions.
        
        Args:
            warped_image: The warped A4 background image
            state: Current interaction state
            shapes: List of all detected shapes
            
        Returns:
            Fully rendered image showing current interaction state
        """
        # Start with clean base
        result = self.render_base(warped_image)
        
        # Add hover preview (if not selected)
        if (state.get("hovered") is not None and 
            state.get("hovered") != state.get("selected") and
            state["hovered"] < len(shapes)):
            result = self.render_preview(result, shapes[state["hovered"]])
        
        # Add selected shape with dimensions
        if (state.get("selected") is not None and 
            state["selected"] < len(shapes)):
            result = self.render_selection(result, shapes[state["selected"]])
        
        # Add instruction text
        result = self.render_instruction_text(result, state, shapes)
        
        return result
    
    def _draw_shape_outline(self, image: np.ndarray, shape: Dict[str, Any], 
                           color: Tuple[int, int, int], thickness: int) -> None:
        """
        Draw only the outline of a shape without dimensions.
        
        Args:
            image: Image to draw on (modified in place)
            shape: Shape data dictionary
            color: RGB color tuple
            thickness: Line thickness
        """
        if shape["type"] == "circle":
            center = tuple(shape["center"])
            radius = int(shape["radius_px"])
            cv2.circle(image, center, radius, color, thickness)
        elif shape["type"] == "rectangle":
            box = shape["box"]
            cv2.drawContours(image, [box], 0, color, thickness)
    
    def _draw_shape_with_dimensions(self, image: np.ndarray, shape: Dict[str, Any],
                                   color: Tuple[int, int, int]) -> None:
        """
        Draw shape with full dimensions and measurement annotations.
        
        Args:
            image: Image to draw on (modified in place)
            shape: Shape data dictionary
            color: RGB color tuple
        """
        if shape["type"] == "circle":
            self._draw_circle_with_dimensions(image, shape, color)
        elif shape["type"] == "rectangle":
            self._draw_rectangle_with_dimensions(image, shape, color)
    
    def _draw_circle_with_dimensions(self, image: np.ndarray, shape: Dict[str, Any],
                                    color: Tuple[int, int, int]) -> None:
        """
        Draw circle with diameter line and measurement text.
        
        Args:
            image: Image to draw on (modified in place)
            shape: Circle shape data dictionary
            color: RGB color tuple
        """
        center = tuple(shape["center"])
        radius = int(shape["radius_px"])
        
        # Draw circle outline
        cv2.circle(image, center, radius, color, 3)
        
        # Draw diameter line
        x0 = center[0] - radius
        x1 = center[0] + radius
        y = center[1]
        cv2.line(image, (x0, y), (x1, y), color, 2)
        
        # Draw dimension text centered inside the circle
        text = f"D={shape['diameter_mm']:.0f}mm"
        text_size = cv2.getTextSize(text, DRAW_FONT, 0.9, 2)[0]
        text_org = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))
        
        # White background for text readability
        cv2.rectangle(image,
                      (text_org[0] - 6, text_org[1] - text_size[1] - 6),
                      (text_org[0] + text_size[0] + 6, text_org[1] + 6),
                      (255, 255, 255), -1)
        
        draw_text(image, text, text_org, (0, 0, 0), 0.9, 2)
    
    def _draw_rectangle_with_dimensions(self, image: np.ndarray, shape: Dict[str, Any],
                                       color: Tuple[int, int, int]) -> None:
        """
        Draw rectangle with dimension arrows and measurement text.
        
        Args:
            image: Image to draw on (modified in place)
            shape: Rectangle shape data dictionary
            color: RGB color tuple
        """
        box = shape["box"]
        
        # Draw rectangle outline
        cv2.drawContours(image, [box], 0, color, 3)
        
        # Draw dimension arrows
        # Horizontal arrows (width)
        mid_left = ((box[0] + box[3]) / 2).astype(int)
        mid_right = ((box[1] + box[2]) / 2).astype(int)
        cv2.arrowedLine(image, tuple(mid_left), tuple(mid_right), color, 2, tipLength=0.02)
        cv2.arrowedLine(image, tuple(mid_right), tuple(mid_left), color, 2, tipLength=0.02)
        
        # Vertical arrows (height)
        mid_top = ((box[0] + box[1]) / 2).astype(int)
        mid_bottom = ((box[2] + box[3]) / 2).astype(int)
        cv2.arrowedLine(image, tuple(mid_top), tuple(mid_bottom), color, 2, tipLength=0.02)
        cv2.arrowedLine(image, tuple(mid_bottom), tuple(mid_top), color, 2, tipLength=0.02)
        
        # Draw dimension text centered inside the rectangle
        cx = int(np.mean(box[:, 0]))
        cy = int(np.mean(box[:, 1]))
        text = f"W={shape['width_mm']:.0f}mm  H={shape['height_mm']:.0f}mm"
        text_size = cv2.getTextSize(text, DRAW_FONT, 0.9, 2)[0]
        text_org = (int(cx - text_size[0] / 2), int(cy + text_size[1] / 2))
        
        # White background for text readability
        cv2.rectangle(image,
                      (text_org[0] - 6, text_org[1] - text_size[1] - 6),
                      (text_org[0] + text_size[0] + 6, text_org[1] + 6),
                      (255, 255, 255), -1)
        
        draw_text(image, text, text_org, (0, 0, 0), 0.9, 2)
    
    def _generate_instruction_text(self, state: Dict[str, Any], 
                                  shapes: List[Dict[str, Any]]) -> str:
        """
        Generate dynamic instruction text based on current interaction state.
        
        Args:
            state: Current interaction state
            shapes: List of shape data for generating descriptive text
            
        Returns:
            Instruction text string
        """
        selected_idx = state.get("selected")
        
        if selected_idx is not None and selected_idx < len(shapes):
            # Shape is selected - show shape type and dimensions
            shape = shapes[selected_idx]
            if shape["type"] == "circle":
                return f"Selected: Circle (D={shape['diameter_mm']:.0f}mm)"
            else:
                return f"Selected: Rectangle ({shape['width_mm']:.0f}x{shape['height_mm']:.0f}mm)"
        else:
            # No selection - show default instruction
            return "Hover to preview, click to inspect"


# Utility functions for rendering validation and debugging

def validate_shape_for_rendering(shape: Dict[str, Any]) -> bool:
    """
    Validate that a shape has the required fields for rendering.
    
    Args:
        shape: Shape data dictionary
        
    Returns:
        True if shape can be rendered
    """
    if not isinstance(shape, dict):
        return False
    
    if "type" not in shape:
        return False
    
    shape_type = shape["type"]
    
    if shape_type == "circle":
        required_fields = ["center", "radius_px", "diameter_mm"]
        return all(field in shape for field in required_fields)
    elif shape_type == "rectangle":
        required_fields = ["box", "width_mm", "height_mm"]
        return all(field in shape for field in required_fields)
    
    return False


def create_test_renderer() -> SelectiveRenderer:
    """
    Create a renderer instance for testing purposes.
    
    Returns:
        SelectiveRenderer instance
    """
    return SelectiveRenderer()


def render_debug_overlay(image: np.ndarray, debug_info: Dict[str, Any]) -> np.ndarray:
    """
    Render debug information overlay on an image.
    
    Args:
        image: Base image to draw on
        debug_info: Dictionary containing debug information
        
    Returns:
        Image with debug overlay
    """
    result = image.copy()
    
    # Draw debug text in top-right corner
    debug_text = f"Shapes: {debug_info.get('shape_count', 0)}"
    draw_text(result, debug_text, (image.shape[1] - 200, 30), (255, 255, 0), 0.6, 1)
    
    if "mouse_pos" in debug_info:
        mouse_pos = debug_info["mouse_pos"]
        debug_text2 = f"Mouse: ({mouse_pos[0]}, {mouse_pos[1]})"
        draw_text(result, debug_text2, (image.shape[1] - 200, 60), (255, 255, 0), 0.6, 1)
    
    return result