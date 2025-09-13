"""
Interaction State Management for Interactive Inspect Mode

This module provides state management functionality for tracking hover and selection
states, mouse position, and determining when re-rendering is needed.
"""

from typing import Dict, List, Optional, Any, Tuple


class InteractionState:
    """
    Manages the current interaction state for the interactive inspect mode.
    
    Tracks:
    - Currently hovered shape index
    - Currently selected shape index  
    - Mouse position
    - Shape data list
    - Display scaling information
    """
    
    def __init__(self, shapes: List[Dict[str, Any]] = None):
        """
        Initialize interaction state.
        
        Args:
            shapes: List of shape data dictionaries (optional)
        """
        self._hovered: Optional[int] = None
        self._selected: Optional[int] = None
        self._mouse_pos: Tuple[int, int] = (0, 0)
        self._shapes: List[Dict[str, Any]] = shapes or []
        self._display_scale: float = 1.0
        self._needs_render: bool = True
    
    @property
    def hovered(self) -> Optional[int]:
        """Get the currently hovered shape index."""
        return self._hovered
    
    @property
    def selected(self) -> Optional[int]:
        """Get the currently selected shape index."""
        return self._selected
    
    @property
    def mouse_pos(self) -> Tuple[int, int]:
        """Get the current mouse position."""
        return self._mouse_pos
    
    @property
    def shapes(self) -> List[Dict[str, Any]]:
        """Get the list of shapes."""
        return self._shapes
    
    @property
    def display_scale(self) -> float:
        """Get the display scaling factor."""
        return self._display_scale
    
    @property
    def needs_render(self) -> bool:
        """Check if re-rendering is needed."""
        return self._needs_render
    
    def update_hover(self, shape_index: Optional[int]) -> bool:
        """
        Update the hover state and determine if re-rendering is needed.
        
        Args:
            shape_index: Index of the shape being hovered, or None
            
        Returns:
            True if the hover state changed and re-rendering is needed
        """
        if self._hovered != shape_index:
            self._hovered = shape_index
            self._needs_render = True
            return True
        return False
    
    def update_hover_state(self, shape_index: Optional[int]) -> bool:
        """
        Update the hover state and determine if re-rendering is needed.
        Alias for update_hover for compatibility with tests.
        
        Args:
            shape_index: Index of the shape being hovered, or None
            
        Returns:
            True if the hover state changed and re-rendering is needed
        """
        return self.update_hover(shape_index)
    
    def update_selection(self, shape_index: Optional[int]) -> bool:
        """
        Update the selection state and determine if re-rendering is needed.
        
        Args:
            shape_index: Index of the shape being selected, or None
            
        Returns:
            True if the selection state changed and re-rendering is needed
        """
        if self._selected != shape_index:
            self._selected = shape_index
            self._needs_render = True
            return True
        return False
    
    def update_selection_state(self, shape_index: Optional[int]) -> bool:
        """
        Update the selection state and determine if re-rendering is needed.
        Alias for update_selection for compatibility with tests.
        
        Args:
            shape_index: Index of the shape being selected, or None
            
        Returns:
            True if the selection state changed and re-rendering is needed
        """
        return self.update_selection(shape_index)
    
    def update_mouse_position(self, x: int, y: int) -> bool:
        """
        Update mouse position tracking.
        
        Args:
            x: X coordinate in original image space
            y: Y coordinate in original image space
            
        Returns:
            True if the mouse position changed significantly
        """
        old_pos = self._mouse_pos
        self._mouse_pos = (x, y)
        
        # Consider position changed if moved more than 1 pixel
        changed = abs(old_pos[0] - x) > 1 or abs(old_pos[1] - y) > 1
        if changed:
            self._needs_render = True
        return changed
    
    def set_shapes(self, shapes: List[Dict[str, Any]]) -> None:
        """
        Set the list of shapes and reset interaction state.
        
        Args:
            shapes: List of shape data dictionaries
        """
        self._shapes = shapes
        self._hovered = None
        self._selected = None
        self._needs_render = True
    
    def set_display_scale(self, scale: float) -> None:
        """
        Set the display scaling factor.
        
        Args:
            scale: Scaling factor for display coordinates
        """
        self._display_scale = scale
    
    def clear_render_flag(self) -> None:
        """Clear the needs_render flag after rendering is complete."""
        self._needs_render = False
    
    def force_render(self) -> None:
        """Force a re-render on the next update."""
        self._needs_render = True
    
    def reset(self) -> None:
        """Reset all interaction state to initial values."""
        self._hovered = None
        self._selected = None
        self._mouse_pos = (0, 0)
        self._shapes = []
        self._display_scale = 1.0
        self._needs_render = True
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get current state as a dictionary for compatibility with existing code.
        
        Returns:
            Dictionary containing current state values
        """
        return {
            "hovered": self._hovered,
            "selected": self._selected,
            "mouse_pos": self._mouse_pos,
            "shapes": self._shapes,
            "display_scale": self._display_scale
        }
    
    def get_hovered_shape(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently hovered shape data.
        
        Returns:
            Shape data dictionary or None if no shape is hovered
        """
        if self._hovered is not None and 0 <= self._hovered < len(self._shapes):
            return self._shapes[self._hovered]
        return None
    
    def get_selected_shape(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently selected shape data.
        
        Returns:
            Shape data dictionary or None if no shape is selected
        """
        if self._selected is not None and 0 <= self._selected < len(self._shapes):
            return self._shapes[self._selected]
        return None
    
    def is_shape_hovered(self, shape_index: int) -> bool:
        """
        Check if a specific shape is currently hovered.
        
        Args:
            shape_index: Index of the shape to check
            
        Returns:
            True if the shape is currently hovered
        """
        return self._hovered == shape_index
    
    def is_shape_selected(self, shape_index: int) -> bool:
        """
        Check if a specific shape is currently selected.
        
        Args:
            shape_index: Index of the shape to check
            
        Returns:
            True if the shape is currently selected
        """
        return self._selected == shape_index
    
    def get_shape_count(self) -> int:
        """
        Get the total number of shapes.
        
        Returns:
            Number of shapes in the current state
        """
        return len(self._shapes)
    
    def is_valid_shape_index(self, index: Optional[int]) -> bool:
        """
        Check if a shape index is valid for the current shape list.
        
        Args:
            index: Shape index to validate
            
        Returns:
            True if the index is valid (within bounds and not None)
        """
        return index is not None and 0 <= index < len(self._shapes)
    
    def get_instruction_text(self) -> str:
        """
        Get instruction text based on current interaction state.
        
        Returns:
            Instruction text string for display
        """
        if self._selected is not None and self.is_valid_shape_index(self._selected):
            shape = self._shapes[self._selected]
            if shape["type"] == "circle":
                return f"Selected: Circle - Diameter: {shape['diameter_mm']:.0f}mm"
            else:
                return f"Selected: Rectangle - {shape['width_mm']:.0f}x{shape['height_mm']:.0f}mm"
        else:
            return "Hover to preview, click to inspect"
    
    def transform_display_to_original(self, display_x: int, display_y: int) -> Tuple[int, int]:
        """
        Transform display coordinates to original image coordinates.
        
        Args:
            display_x: X coordinate in display window
            display_y: Y coordinate in display window
            
        Returns:
            (x, y) coordinates in original image space
        """
        return transform_display_to_original_coords(display_x, display_y, self._display_scale)
    
    def transform_original_to_display(self, original_x: int, original_y: int) -> Tuple[int, int]:
        """
        Transform original coordinates to display window coordinates.
        
        Args:
            original_x: X coordinate in original image
            original_y: Y coordinate in original image
            
        Returns:
            (x, y) coordinates in display window space
        """
        return transform_original_to_display_coords(original_x, original_y, self._display_scale)
    
    def reset_interaction(self) -> None:
        """
        Reset interaction state (hover and selection) but keep shapes and scale.
        """
        self._hovered = None
        self._selected = None
        self._mouse_pos = (0, 0)
        self._needs_render = True
    
    def should_show_hover_preview(self, shape_index: int) -> bool:
        """
        Check if a shape should show hover preview.
        
        Args:
            shape_index: Index of the shape to check
            
        Returns:
            True if the shape should show hover preview
        """
        return (self._hovered == shape_index and 
                self._selected != shape_index and 
                self.is_valid_shape_index(shape_index))
    
    def should_show_selection(self, shape_index: int) -> bool:
        """
        Check if a shape should show selection rendering.
        
        Args:
            shape_index: Index of the shape to check
            
        Returns:
            True if the shape should show selection rendering
        """
        return (self._selected == shape_index and 
                self.is_valid_shape_index(shape_index))


# Utility functions for state management

def create_interaction_state(shapes: List[Dict[str, Any]] = None, scale: float = 1.0) -> InteractionState:
    """
    Create a new interaction state instance.
    
    Args:
        shapes: Optional list of shape data dictionaries
        scale: Display scaling factor
        
    Returns:
        New InteractionState instance
    """
    state = InteractionState(shapes)
    state.set_display_scale(scale)
    return state


def transform_display_to_original_coords(display_x: int, display_y: int, scale: float) -> Tuple[int, int]:
    """
    Transform display window coordinates to original image coordinates.
    
    Args:
        display_x: X coordinate in display window
        display_y: Y coordinate in display window
        scale: Display scaling factor
        
    Returns:
        (x, y) coordinates in original image space
    """
    original_x = int(display_x / scale)
    original_y = int(display_y / scale)
    return original_x, original_y


def transform_original_to_display_coords(original_x: int, original_y: int, scale: float) -> Tuple[int, int]:
    """
    Transform original image coordinates to display window coordinates.
    
    Args:
        original_x: X coordinate in original image
        original_y: Y coordinate in original image
        scale: Display scaling factor
        
    Returns:
        (x, y) coordinates in display window space
    """
    display_x = int(original_x * scale)
    display_y = int(original_y * scale)
    return display_x, display_y


def validate_mouse_coordinates(x: int, y: int, image_width: int, image_height: int) -> bool:
    """
    Validate that mouse coordinates are within image bounds.
    
    Args:
        x: X coordinate to validate
        y: Y coordinate to validate
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        True if coordinates are within bounds
    """
    return 0 <= x < image_width and 0 <= y < image_height


class StateChangeDetector:
    """
    Utility class for detecting changes in interaction state between updates.
    
    Tracks previous state values and compares them with current state to
    determine what has changed and whether re-rendering is needed.
    """
    
    def __init__(self):
        """Initialize the state change detector."""
        self._prev_hovered: Optional[int] = None
        self._prev_selected: Optional[int] = None
        self._prev_mouse_pos: Tuple[int, int] = (0, 0)
        self._initialized: bool = False
    
    def check_changes(self, state: InteractionState) -> Dict[str, bool]:
        """
        Check for changes in the interaction state.
        
        Args:
            state: Current interaction state to check
            
        Returns:
            Dictionary with change flags for different state aspects
        """
        changes = {
            'hover': False,
            'selection': False,
            'mouse_pos': False,
            'any': False
        }
        
        # Check for hover changes
        if self._prev_hovered != state.hovered:
            changes['hover'] = True
        
        # Check for selection changes
        if self._prev_selected != state.selected:
            changes['selection'] = True
        
        # Check for mouse position changes
        if self._prev_mouse_pos != state.mouse_pos:
            changes['mouse_pos'] = True
        
        # Set overall change flag
        changes['any'] = changes['hover'] or changes['selection'] or changes['mouse_pos']
        
        # Update previous values after checking
        self._prev_hovered = state.hovered
        self._prev_selected = state.selected
        self._prev_mouse_pos = state.mouse_pos
        self._initialized = True
        
        return changes
    
    def reset(self) -> None:
        """Reset the detector to initial state."""
        self._prev_hovered = None
        self._prev_selected = None
        self._prev_mouse_pos = (0, 0)
        self._initialized = False