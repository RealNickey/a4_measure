"""
Selection mode management system for manual shape selection.

This module provides the SelectionMode enum and ModeManager class for handling
mode switching between automatic detection and manual selection modes.
"""

from enum import Enum
from typing import Dict, Optional


class SelectionMode(Enum):
    """Enumeration of available selection modes."""
    AUTO = "auto"
    MANUAL_RECTANGLE = "manual_rect"
    MANUAL_CIRCLE = "manual_circle"

    def __str__(self) -> str:
        """Return string representation of the mode."""
        return self.value


class ModeManager:
    """
    Manages switching between automatic and manual selection modes.
    
    Provides functionality for mode cycling, state tracking, and mode indicators.
    """
    
    def __init__(self, initial_mode: SelectionMode = SelectionMode.AUTO):
        """
        Initialize the ModeManager.
        
        Args:
            initial_mode: The initial selection mode (defaults to AUTO)
        """
        self.current_mode = initial_mode
        self.mode_indicators = {
            SelectionMode.AUTO: "AUTO",
            SelectionMode.MANUAL_RECTANGLE: "MANUAL RECT",
            SelectionMode.MANUAL_CIRCLE: "MANUAL CIRCLE"
        }
        self.mode_cycle_order = [
            SelectionMode.AUTO,
            SelectionMode.MANUAL_RECTANGLE,
            SelectionMode.MANUAL_CIRCLE
        ]
    
    def cycle_mode(self) -> SelectionMode:
        """
        Cycle to the next mode in the sequence: AUTO → MANUAL_RECTANGLE → MANUAL_CIRCLE → AUTO.
        
        Returns:
            The new current mode after cycling
        """
        current_index = self.mode_cycle_order.index(self.current_mode)
        next_index = (current_index + 1) % len(self.mode_cycle_order)
        self.current_mode = self.mode_cycle_order[next_index]
        return self.current_mode
    
    def get_current_mode(self) -> SelectionMode:
        """
        Get the current selection mode.
        
        Returns:
            The current SelectionMode
        """
        return self.current_mode
    
    def set_mode(self, mode: SelectionMode) -> None:
        """
        Set the current mode to a specific value.
        
        Args:
            mode: The SelectionMode to set as current
        """
        if not isinstance(mode, SelectionMode):
            raise ValueError(f"Invalid mode type: {type(mode)}. Expected SelectionMode.")
        self.current_mode = mode
    
    def get_mode_indicator(self) -> str:
        """
        Get the display text for the current mode.
        
        Returns:
            String representation suitable for UI display
        """
        return self.mode_indicators[self.current_mode]
    
    def is_manual_mode(self) -> bool:
        """
        Check if currently in any manual selection mode.
        
        Returns:
            True if in MANUAL_RECTANGLE or MANUAL_CIRCLE mode, False otherwise
        """
        return self.current_mode in [SelectionMode.MANUAL_RECTANGLE, SelectionMode.MANUAL_CIRCLE]
    
    def is_auto_mode(self) -> bool:
        """
        Check if currently in automatic detection mode.
        
        Returns:
            True if in AUTO mode, False otherwise
        """
        return self.current_mode == SelectionMode.AUTO
    
    def get_manual_shape_type(self) -> Optional[str]:
        """
        Get the shape type for manual selection modes.
        
        Returns:
            "rectangle" for MANUAL_RECTANGLE mode, "circle" for MANUAL_CIRCLE mode,
            None for AUTO mode
        """
        if self.current_mode == SelectionMode.MANUAL_RECTANGLE:
            return "rectangle"
        elif self.current_mode == SelectionMode.MANUAL_CIRCLE:
            return "circle"
        return None