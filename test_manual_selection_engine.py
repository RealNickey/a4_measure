"""
Unit tests for Manual Selection Engine

Tests for SelectionState dataclass and ManualSelectionEngine class,
focusing on selection geometry calculations and state management.
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from manual_selection_engine import (
    SelectionState, 
    ManualSelectionEngine,
    create_manual_selection_engine,
    validate_selection_geometry,
    calculate_selection_overlap
)


class TestSelectionState(unittest.TestCase):
    """Test cases for SelectionState dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state = SelectionState()
    
    def test_initial_state(self):
        """Test initial state values."""
        self.assertFalse(self.state.is_selecting)
        self.assertIsNone(self.state.start_point)
        self.assertIsNone(self.state.current_point)
        self.assertIsNone(self.state.selection_rect)
    
    def test_start_selection(self):
        """Test starting a selection operation."""
        self.state.start_selection(100, 150)
        
        self.assertTrue(self.state.is_selecting)
        self.assertEqual(self.state.start_point, (100, 150))
        self.assertEqual(self.state.current_point, (100, 150))
        self.assertIsNone(self.state.selection_rect)
    
    def test_update_selection(self):
        """Test updating selection coordinates."""
        # Start selection first
        self.state.start_selection(100, 150)
        
        # Update selection
        self.state.update_selection(200, 250)
        
        self.assertTrue(self.state.is_selecting)
        self.assertEqual(self.state.start_point, (100, 150))
        self.assertEqual(self.state.current_point, (200, 250))
        self.assertEqual(self.state.selection_rect, (100, 150, 100, 100))
    
    def test_update_selection_without_start(self):
        """Test updating selection without starting first."""
        self.state.update_selection(200, 250)
        
        # Should not change state
        self.assertFalse(self.state.is_selecting)
        self.assertIsNone(self.state.start_point)
        self.assertIsNone(self.state.current_point)
        self.assertIsNone(self.state.selection_rect)
    
    def test_complete_selection(self):
        """Test completing a selection operation."""
        # Start and update selection
        self.state.start_selection(50, 75)
        self.state.update_selection(150, 175)
        
        # Complete selection
        result = self.state.complete_selection()
        
        self.assertFalse(self.state.is_selecting)
        self.assertEqual(result, (50, 75, 100, 100))
    
    def test_complete_selection_without_start(self):
        """Test completing selection without starting."""
        result = self.state.complete_selection()
        
        self.assertIsNone(result)
        self.assertFalse(self.state.is_selecting)
    
    def test_cancel_selection(self):
        """Test cancelling a selection operation."""
        # Start selection
        self.state.start_selection(100, 150)
        self.state.update_selection(200, 250)
        
        # Cancel selection
        self.state.cancel_selection()
        
        self.assertFalse(self.state.is_selecting)
        self.assertIsNone(self.state.start_point)
        self.assertIsNone(self.state.current_point)
        self.assertIsNone(self.state.selection_rect)
    
    def test_selection_rect_calculation_normal(self):
        """Test selection rectangle calculation for normal drag."""
        self.state.start_selection(100, 150)
        self.state.update_selection(200, 250)
        
        # Rectangle should be (100, 150, 100, 100)
        self.assertEqual(self.state.selection_rect, (100, 150, 100, 100))
    
    def test_selection_rect_calculation_reverse_drag(self):
        """Test selection rectangle calculation for reverse drag."""
        self.state.start_selection(200, 250)
        self.state.update_selection(100, 150)
        
        # Rectangle should normalize to (100, 150, 100, 100)
        self.assertEqual(self.state.selection_rect, (100, 150, 100, 100))
    
    def test_selection_rect_calculation_mixed_drag(self):
        """Test selection rectangle calculation for mixed direction drag."""
        self.state.start_selection(150, 200)
        self.state.update_selection(100, 250)
        
        # Rectangle should be (100, 200, 50, 50)
        self.assertEqual(self.state.selection_rect, (100, 200, 50, 50))
    
    def test_is_valid_selection(self):
        """Test selection validation."""
        # No selection
        self.assertFalse(self.state.is_valid_selection())
        
        # Valid selection
        self.state.start_selection(100, 150)
        self.state.update_selection(150, 200)
        self.assertTrue(self.state.is_valid_selection(20))
        
        # Too small selection
        self.state.start_selection(100, 150)
        self.state.update_selection(110, 160)
        self.assertFalse(self.state.is_valid_selection(20))
    
    def test_get_selection_area(self):
        """Test selection area calculation."""
        # No selection
        self.assertEqual(self.state.get_selection_area(), 0)
        
        # Valid selection
        self.state.start_selection(100, 150)
        self.state.update_selection(200, 250)
        self.assertEqual(self.state.get_selection_area(), 10000)  # 100 * 100
    
    def test_get_selection_center(self):
        """Test selection center calculation."""
        # No selection
        self.assertIsNone(self.state.get_selection_center())
        
        # Valid selection
        self.state.start_selection(100, 150)
        self.state.update_selection(200, 250)
        self.assertEqual(self.state.get_selection_center(), (150, 200))
    
    def test_reset(self):
        """Test resetting selection state."""
        # Set up some state
        self.state.start_selection(100, 150)
        self.state.update_selection(200, 250)
        
        # Reset
        self.state.reset()
        
        self.assertFalse(self.state.is_selecting)
        self.assertIsNone(self.state.start_point)
        self.assertIsNone(self.state.current_point)
        self.assertIsNone(self.state.selection_rect)


class TestManualSelectionEngine(unittest.TestCase):
    """Test cases for ManualSelectionEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = ManualSelectionEngine(display_scale=2.0, min_selection_size=20)
        
        # Mock callbacks
        self.start_callback = Mock()
        self.update_callback = Mock()
        self.complete_callback = Mock()
        self.cancel_callback = Mock()
        
        self.engine.set_callbacks(
            self.start_callback,
            self.update_callback,
            self.complete_callback,
            self.cancel_callback
        )
    
    def test_initial_state(self):
        """Test initial engine state."""
        self.assertFalse(self.engine.is_selecting())
        self.assertIsNone(self.engine.get_current_selection_rect())
        self.assertEqual(self.engine.display_scale, 2.0)
        self.assertEqual(self.engine.min_selection_size, 20)
    
    def test_set_display_scale(self):
        """Test setting display scale."""
        self.engine.set_display_scale(1.5)
        self.assertEqual(self.engine.display_scale, 1.5)
    
    def test_start_selection(self):
        """Test starting selection with coordinate transformation."""
        self.engine.start_selection(200, 300)  # Display coordinates
        
        self.assertTrue(self.engine.is_selecting())
        # Should transform to original coordinates (100, 150)
        self.start_callback.assert_called_once_with(100, 150)
    
    def test_update_selection(self):
        """Test updating selection with coordinate transformation."""
        self.engine.start_selection(200, 300)
        self.engine.update_selection(400, 500)  # Display coordinates
        
        # Should transform to original coordinates (200, 250)
        self.update_callback.assert_called_with(200, 250)
        
        rect = self.engine.get_current_selection_rect()
        self.assertEqual(rect, (100, 150, 100, 100))
    
    def test_update_selection_without_start(self):
        """Test updating selection without starting first."""
        self.engine.update_selection(400, 500)
        
        # Should not call callback
        self.update_callback.assert_not_called()
        self.assertFalse(self.engine.is_selecting())
    
    def test_complete_selection_valid(self):
        """Test completing a valid selection."""
        self.engine.start_selection(200, 300)
        self.engine.update_selection(400, 500)
        
        result = self.engine.complete_selection()
        
        self.assertFalse(self.engine.is_selecting())
        self.assertEqual(result, (100, 150, 100, 100))
        self.complete_callback.assert_called_once_with((100, 150, 100, 100))
    
    def test_complete_selection_invalid(self):
        """Test completing an invalid (too small) selection."""
        self.engine.start_selection(200, 300)
        self.engine.update_selection(220, 320)  # Too small
        
        result = self.engine.complete_selection()
        
        self.assertIsNone(result)
        self.assertFalse(self.engine.is_selecting())
        self.complete_callback.assert_not_called()
        self.cancel_callback.assert_called_once()
    
    def test_cancel_selection(self):
        """Test cancelling selection."""
        self.engine.start_selection(200, 300)
        self.engine.update_selection(400, 500)
        
        self.engine.cancel_selection()
        
        self.assertFalse(self.engine.is_selecting())
        self.assertIsNone(self.engine.get_current_selection_rect())
        self.cancel_callback.assert_called_once()
    
    def test_handle_mouse_event_start(self):
        """Test handling mouse button down event."""
        result = self.engine.handle_mouse_event(cv2.EVENT_LBUTTONDOWN, 200, 300, 0)
        
        self.assertTrue(result)
        self.assertTrue(self.engine.is_selecting())
        self.start_callback.assert_called_once_with(100, 150)
    
    def test_handle_mouse_event_drag(self):
        """Test handling mouse move with button down."""
        # Start selection
        self.engine.handle_mouse_event(cv2.EVENT_LBUTTONDOWN, 200, 300, 0)
        
        # Drag
        result = self.engine.handle_mouse_event(
            cv2.EVENT_MOUSEMOVE, 400, 500, cv2.EVENT_FLAG_LBUTTON
        )
        
        self.assertTrue(result)
        self.update_callback.assert_called_with(200, 250)
    
    def test_handle_mouse_event_complete(self):
        """Test handling mouse button up event."""
        # Start and drag
        self.engine.handle_mouse_event(cv2.EVENT_LBUTTONDOWN, 200, 300, 0)
        self.engine.handle_mouse_event(
            cv2.EVENT_MOUSEMOVE, 400, 500, cv2.EVENT_FLAG_LBUTTON
        )
        
        # Complete
        result = self.engine.handle_mouse_event(cv2.EVENT_LBUTTONUP, 400, 500, 0)
        
        self.assertTrue(result)
        self.assertFalse(self.engine.is_selecting())
        self.complete_callback.assert_called_once_with((100, 150, 100, 100))
    
    def test_handle_mouse_event_cancel(self):
        """Test handling right click to cancel."""
        # Start selection
        self.engine.handle_mouse_event(cv2.EVENT_LBUTTONDOWN, 200, 300, 0)
        
        # Right click to cancel
        result = self.engine.handle_mouse_event(cv2.EVENT_RBUTTONDOWN, 400, 500, 0)
        
        self.assertTrue(result)
        self.assertFalse(self.engine.is_selecting())
        self.cancel_callback.assert_called_once()
    
    def test_get_display_selection_rect(self):
        """Test getting selection rectangle in display coordinates."""
        self.engine.start_selection(200, 300)
        self.engine.update_selection(400, 500)
        
        display_rect = self.engine.get_display_selection_rect()
        # Original rect (100, 150, 100, 100) scaled by 2.0
        self.assertEqual(display_rect, (200, 300, 200, 200))
    
    def test_get_selection_info(self):
        """Test getting comprehensive selection information."""
        self.engine.start_selection(200, 300)
        self.engine.update_selection(400, 500)
        
        info = self.engine.get_selection_info()
        
        self.assertTrue(info["is_selecting"])
        self.assertEqual(info["selection_rect"], (100, 150, 100, 100))
        self.assertEqual(info["selection_center"], (150, 200))
        self.assertEqual(info["selection_area"], 10000)
        self.assertTrue(info["is_valid"])
        self.assertEqual(info["display_scale"], 2.0)
    
    def test_extract_selection_region(self):
        """Test extracting selection region from image."""
        # Create test image
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Set up selection
        self.engine.start_selection(200, 300)
        self.engine.update_selection(400, 500)
        
        # Extract region
        region = self.engine.extract_selection_region(image)
        
        self.assertIsNotNone(region)
        self.assertEqual(region.shape, (100, 100, 3))
    
    def test_extract_selection_region_out_of_bounds(self):
        """Test extracting selection region that goes out of bounds."""
        # Create small test image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Set up selection that goes out of bounds
        self.engine.start_selection(300, 300)  # Display coords
        self.engine.update_selection(600, 600)  # Display coords
        # Original coords: (150, 150) to (300, 300), but image is only 200x200
        
        region = self.engine.extract_selection_region(image)
        
        # Should extract what's available: from (150, 150) to (200, 200)
        self.assertIsNotNone(region)
        self.assertEqual(region.shape, (50, 50, 3))
    
    def test_validate_selection_bounds(self):
        """Test validating selection bounds against image size."""
        # Valid selection
        self.engine.start_selection(200, 300)
        self.engine.update_selection(400, 500)
        
        self.assertTrue(self.engine.validate_selection_bounds(400, 400))
        self.assertFalse(self.engine.validate_selection_bounds(150, 150))
    
    def test_reset(self):
        """Test resetting the engine."""
        self.engine.start_selection(200, 300)
        self.engine.update_selection(400, 500)
        
        self.engine.reset()
        
        self.assertFalse(self.engine.is_selecting())
        self.assertIsNone(self.engine.get_current_selection_rect())
    
    def test_coordinate_transformation(self):
        """Test coordinate transformation methods."""
        # Test display to original
        orig_x, orig_y = self.engine._transform_display_to_original(200, 300)
        self.assertEqual((orig_x, orig_y), (100, 150))
        
        # Test original to display
        disp_x, disp_y = self.engine._transform_original_to_display(100, 150)
        self.assertEqual((disp_x, disp_y), (200, 300))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_manual_selection_engine(self):
        """Test creating manual selection engine."""
        engine = create_manual_selection_engine(1.5, 30)
        
        self.assertEqual(engine.display_scale, 1.5)
        self.assertEqual(engine.min_selection_size, 30)
        self.assertFalse(engine.is_selecting())
    
    def test_validate_selection_geometry_valid(self):
        """Test validating valid selection geometry."""
        rect = (10, 20, 50, 60)
        self.assertTrue(validate_selection_geometry(rect, 20))
    
    def test_validate_selection_geometry_invalid(self):
        """Test validating invalid selection geometry."""
        # None rectangle
        self.assertFalse(validate_selection_geometry(None))
        
        # Negative coordinates
        self.assertFalse(validate_selection_geometry((-10, 20, 50, 60)))
        
        # Too small
        self.assertFalse(validate_selection_geometry((10, 20, 10, 15), 20))
    
    def test_calculate_selection_overlap_no_overlap(self):
        """Test calculating overlap for non-overlapping rectangles."""
        rect1 = (0, 0, 50, 50)
        rect2 = (100, 100, 50, 50)
        
        overlap = calculate_selection_overlap(rect1, rect2)
        self.assertEqual(overlap, 0.0)
    
    def test_calculate_selection_overlap_partial(self):
        """Test calculating overlap for partially overlapping rectangles."""
        rect1 = (0, 0, 50, 50)
        rect2 = (25, 25, 50, 50)
        
        overlap = calculate_selection_overlap(rect1, rect2)
        
        # Intersection: 25x25 = 625
        # Union: 50*50 + 50*50 - 625 = 4375
        # Overlap: 625/4375 ≈ 0.143
        self.assertAlmostEqual(overlap, 625/4375, places=3)
    
    def test_calculate_selection_overlap_complete(self):
        """Test calculating overlap for identical rectangles."""
        rect1 = (10, 20, 50, 60)
        rect2 = (10, 20, 50, 60)
        
        overlap = calculate_selection_overlap(rect1, rect2)
        self.assertEqual(overlap, 1.0)
    
    def test_calculate_selection_overlap_contained(self):
        """Test calculating overlap for contained rectangles."""
        rect1 = (0, 0, 100, 100)
        rect2 = (25, 25, 50, 50)
        
        overlap = calculate_selection_overlap(rect1, rect2)
        
        # Intersection: 50*50 = 2500
        # Union: 100*100 = 10000 (rect2 is contained in rect1)
        # Overlap: 2500/10000 = 0.25
        self.assertEqual(overlap, 0.25)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)