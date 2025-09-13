"""
Unit tests for the interaction state management system.
"""

import unittest
import numpy as np
from interaction_state import InteractionState, StateChangeDetector, create_interaction_state


class TestInteractionState(unittest.TestCase):
    """Test cases for InteractionState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_shapes = [
            {
                "type": "circle",
                "diameter_mm": 25.0,
                "center": (100, 100),
                "radius_px": 50.0,
                "area_px": 7853.98
            },
            {
                "type": "rectangle", 
                "width_mm": 30.0,
                "height_mm": 40.0,
                "box": np.array([[50, 50], [150, 50], [150, 150], [50, 150]]),
                "area_px": 10000.0
            }
        ]
        self.state = InteractionState(self.sample_shapes)
    
    def test_initialization(self):
        """Test proper initialization of InteractionState."""
        self.assertIsNone(self.state.hovered)
        self.assertIsNone(self.state.selected)
        self.assertEqual(self.state.mouse_pos, (0, 0))
        self.assertEqual(len(self.state.shapes), 2)
        self.assertEqual(self.state.display_scale, 1.0)
        self.assertTrue(self.state.needs_render)
    
    def test_empty_initialization(self):
        """Test initialization with no shapes."""
        empty_state = InteractionState()
        self.assertEqual(len(empty_state.shapes), 0)
        self.assertIsNone(empty_state.hovered)
        self.assertIsNone(empty_state.selected)
    
    def test_mouse_position_update(self):
        """Test mouse position tracking and change detection."""
        # Initial position change should return True
        changed = self.state.update_mouse_position(50, 75)
        self.assertTrue(changed)
        self.assertEqual(self.state.mouse_pos, (50, 75))
        self.assertTrue(self.state.needs_render)
        
        # Clear render flag
        self.state.clear_render_flag()
        self.assertFalse(self.state.needs_render)
        
        # Same position should return False
        changed = self.state.update_mouse_position(50, 75)
        self.assertFalse(changed)
        self.assertFalse(self.state.needs_render)
        
        # Different position should return True
        changed = self.state.update_mouse_position(100, 100)
        self.assertTrue(changed)
        self.assertTrue(self.state.needs_render)
    
    def test_hover_state_update(self):
        """Test hover state management."""
        # Set hover to first shape
        changed = self.state.update_hover_state(0)
        self.assertTrue(changed)
        self.assertEqual(self.state.hovered, 0)
        self.assertTrue(self.state.needs_render)
        
        # Clear render flag
        self.state.clear_render_flag()
        
        # Same hover should return False
        changed = self.state.update_hover_state(0)
        self.assertFalse(changed)
        self.assertFalse(self.state.needs_render)
        
        # Change to different shape
        changed = self.state.update_hover_state(1)
        self.assertTrue(changed)
        self.assertEqual(self.state.hovered, 1)
        self.assertTrue(self.state.needs_render)
        
        # Clear hover
        self.state.clear_render_flag()
        changed = self.state.update_hover_state(None)
        self.assertTrue(changed)
        self.assertIsNone(self.state.hovered)
        self.assertTrue(self.state.needs_render)
    
    def test_selection_state_update(self):
        """Test selection state management."""
        # Set selection to first shape
        changed = self.state.update_selection_state(0)
        self.assertTrue(changed)
        self.assertEqual(self.state.selected, 0)
        self.assertTrue(self.state.needs_render)
        
        # Clear render flag
        self.state.clear_render_flag()
        
        # Same selection should return False
        changed = self.state.update_selection_state(0)
        self.assertFalse(changed)
        self.assertFalse(self.state.needs_render)
        
        # Change to different shape
        changed = self.state.update_selection_state(1)
        self.assertTrue(changed)
        self.assertEqual(self.state.selected, 1)
        self.assertTrue(self.state.needs_render)
        
        # Clear selection
        self.state.clear_render_flag()
        changed = self.state.update_selection_state(None)
        self.assertTrue(changed)
        self.assertIsNone(self.state.selected)
        self.assertTrue(self.state.needs_render)
    
    def test_shape_queries(self):
        """Test shape query methods."""
        # Test with no selection/hover
        self.assertIsNone(self.state.get_hovered_shape())
        self.assertIsNone(self.state.get_selected_shape())
        
        # Set hover and selection
        self.state.update_hover_state(0)
        self.state.update_selection_state(1)
        
        # Test shape retrieval
        hovered_shape = self.state.get_hovered_shape()
        self.assertIsNotNone(hovered_shape)
        self.assertEqual(hovered_shape["type"], "circle")
        
        selected_shape = self.state.get_selected_shape()
        self.assertIsNotNone(selected_shape)
        self.assertEqual(selected_shape["type"], "rectangle")
        
        # Test invalid indices
        self.state.update_hover_state(99)  # Invalid index
        self.assertIsNone(self.state.get_hovered_shape())
    
    def test_shape_state_checks(self):
        """Test shape state checking methods."""
        self.state.update_hover_state(0)
        self.state.update_selection_state(1)
        
        # Test hover checks
        self.assertTrue(self.state.is_shape_hovered(0))
        self.assertFalse(self.state.is_shape_hovered(1))
        
        # Test selection checks
        self.assertTrue(self.state.is_shape_selected(1))
        self.assertFalse(self.state.is_shape_selected(0))
        
        # Test hover preview logic
        self.assertTrue(self.state.should_show_hover_preview(0))  # Hovered but not selected
        self.assertFalse(self.state.should_show_hover_preview(1))  # Selected (no hover preview)
    
    def test_coordinate_transformation(self):
        """Test coordinate transformation methods."""
        self.state.set_display_scale(2.0)
        
        # Test display to original
        orig_x, orig_y = self.state.transform_display_to_original(200, 300)
        self.assertEqual(orig_x, 100)
        self.assertEqual(orig_y, 150)
        
        # Test original to display
        disp_x, disp_y = self.state.transform_original_to_display(100, 150)
        self.assertEqual(disp_x, 200)
        self.assertEqual(disp_y, 300)
    
    def test_instruction_text(self):
        """Test instruction text generation."""
        # Default text
        text = self.state.get_instruction_text()
        self.assertEqual(text, "Hover to preview, click to inspect")
        
        # Circle selection
        self.state.update_selection_state(0)
        text = self.state.get_instruction_text()
        self.assertIn("Circle", text)
        self.assertIn("25mm", text)
        
        # Rectangle selection
        self.state.update_selection_state(1)
        text = self.state.get_instruction_text()
        self.assertIn("Rectangle", text)
        self.assertIn("30x40mm", text)
    
    def test_reset_interaction(self):
        """Test interaction state reset."""
        # Set some state
        self.state.update_hover_state(0)
        self.state.update_selection_state(1)
        self.state.clear_render_flag()
        
        # Reset should clear state and trigger render
        self.state.reset_interaction()
        self.assertIsNone(self.state.hovered)
        self.assertIsNone(self.state.selected)
        self.assertTrue(self.state.needs_render)
    
    def test_set_shapes(self):
        """Test updating shapes list."""
        new_shapes = [{"type": "circle", "diameter_mm": 15.0}]
        
        # Set some state first
        self.state.update_hover_state(0)
        self.state.update_selection_state(1)
        self.state.clear_render_flag()
        
        # Setting new shapes should reset interaction state
        self.state.set_shapes(new_shapes)
        self.assertEqual(len(self.state.shapes), 1)
        self.assertIsNone(self.state.hovered)
        self.assertIsNone(self.state.selected)
        self.assertTrue(self.state.needs_render)
    
    def test_force_render(self):
        """Test force render functionality."""
        self.state.clear_render_flag()
        self.assertFalse(self.state.needs_render)
        
        self.state.force_render()
        self.assertTrue(self.state.needs_render)


class TestStateChangeDetector(unittest.TestCase):
    """Test cases for StateChangeDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.shapes = [{"type": "circle", "diameter_mm": 25.0}]
        self.state = InteractionState(self.shapes)
        self.detector = StateChangeDetector()
    
    def test_initial_changes(self):
        """Test change detection on first call."""
        changes = self.detector.check_changes(self.state)
        
        # First call should not detect changes since both previous and current are None/default
        self.assertFalse(changes['hover'])  # Both None
        self.assertFalse(changes['selection'])  # Both None
        self.assertFalse(changes['mouse_pos'])  # Both (0, 0)
        self.assertFalse(changes['any'])
    
    def test_no_changes(self):
        """Test when no changes occur."""
        # First call to establish baseline
        self.detector.check_changes(self.state)
        
        # Second call with no changes
        changes = self.detector.check_changes(self.state)
        self.assertFalse(changes['hover'])
        self.assertFalse(changes['selection'])
        self.assertFalse(changes['mouse_pos'])
        self.assertFalse(changes['any'])
    
    def test_hover_changes(self):
        """Test hover change detection."""
        # Establish baseline
        self.detector.check_changes(self.state)
        
        # Change hover state
        self.state.update_hover_state(0)
        changes = self.detector.check_changes(self.state)
        
        self.assertTrue(changes['hover'])
        self.assertFalse(changes['selection'])
        self.assertFalse(changes['mouse_pos'])
        self.assertTrue(changes['any'])
    
    def test_selection_changes(self):
        """Test selection change detection."""
        # Establish baseline
        self.detector.check_changes(self.state)
        
        # Change selection state
        self.state.update_selection_state(0)
        changes = self.detector.check_changes(self.state)
        
        self.assertFalse(changes['hover'])
        self.assertTrue(changes['selection'])
        self.assertFalse(changes['mouse_pos'])
        self.assertTrue(changes['any'])
    
    def test_mouse_position_changes(self):
        """Test mouse position change detection."""
        # Establish baseline
        self.detector.check_changes(self.state)
        
        # Change mouse position
        self.state.update_mouse_position(100, 200)
        changes = self.detector.check_changes(self.state)
        
        self.assertFalse(changes['hover'])
        self.assertFalse(changes['selection'])
        self.assertTrue(changes['mouse_pos'])
        self.assertTrue(changes['any'])
    
    def test_reset(self):
        """Test detector reset functionality."""
        # Set some state and check changes
        self.state.update_hover_state(0)
        self.detector.check_changes(self.state)
        
        # Reset detector
        self.detector.reset()
        
        # Next check should detect changes again
        changes = self.detector.check_changes(self.state)
        self.assertTrue(changes['any'])


class TestFactoryFunction(unittest.TestCase):
    """Test cases for factory function."""
    
    def test_create_interaction_state(self):
        """Test factory function for creating InteractionState."""
        shapes = [{"type": "circle", "diameter_mm": 25.0}]
        scale = 1.5
        
        state = create_interaction_state(shapes, scale)
        
        self.assertIsInstance(state, InteractionState)
        self.assertEqual(len(state.shapes), 1)
        self.assertEqual(state.display_scale, scale)
        self.assertTrue(state.needs_render)
    
    def test_create_interaction_state_defaults(self):
        """Test factory function with default parameters."""
        shapes = []
        
        state = create_interaction_state(shapes)
        
        self.assertEqual(state.display_scale, 1.0)
        self.assertEqual(len(state.shapes), 0)


if __name__ == '__main__':
    unittest.main()