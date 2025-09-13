"""
Integration tests for complete hover-to-click workflow.

Tests cover:
- Complete mouse interaction workflow from hover to click
- State transitions and consistency
- Rendering triggers and updates
- Mouse event handling integration
- Real-world interaction scenarios
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

from interaction_manager import InteractionManager, create_interaction_manager
from hit_testing import create_hit_testing_contour
from interaction_state import InteractionState


class TestHoverClickWorkflow(unittest.TestCase):
    """Integration tests for hover-to-click workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test shapes
        self.test_shapes = self.create_test_shapes()
        
        # Create test warped image
        self.warped_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Create interaction manager
        self.manager = create_interaction_manager(
            self.test_shapes, 
            self.warped_image,
            display_height=400,
            hover_snap_distance_mm=10.0
        )
    
    def create_test_shapes(self):
        """Create test shapes for workflow testing."""
        shapes = []
        
        # Circle shape
        shapes.append({
            'type': 'circle',
            'center': (200, 150),
            'radius_px': 30.0,
            'diameter_mm': 25.0,
            'area_px': np.pi * 30 * 30,
            'hit_contour': create_hit_testing_contour('circle', center=(200, 150), radius_px=30.0),
            'inner': False
        })
        
        # Rectangle shape
        rect_box = np.array([[300, 100], [400, 100], [400, 200], [300, 200]], dtype=np.int32)
        shapes.append({
            'type': 'rectangle',
            'box': rect_box,
            'width_mm': 20.0,
            'height_mm': 15.0,
            'area_px': 100 * 100,
            'hit_contour': create_hit_testing_contour('rectangle', box=rect_box),
            'inner': False
        })
        
        # Small circle inside rectangle for overlap testing
        shapes.append({
            'type': 'circle',
            'center': (350, 150),
            'radius_px': 15.0,
            'diameter_mm': 12.0,
            'area_px': np.pi * 15 * 15,
            'hit_contour': create_hit_testing_contour('circle', center=(350, 150), radius_px=15.0),
            'inner': True
        })
        
        return shapes
    
    def test_initial_state(self):
        """Test initial state of interaction manager."""
        self.assertIsNone(self.manager.state.hovered)
        self.assertIsNone(self.manager.state.selected)
        self.assertEqual(self.manager.state.mouse_pos, (0, 0))
        self.assertEqual(len(self.manager.state.shapes), 3)
        self.assertTrue(self.manager.state.needs_render)
    
    def test_hover_workflow(self):
        """Test complete hover workflow."""
        # Initial state - no hover
        self.assertIsNone(self.manager.state.hovered)
        
        # Move mouse over first shape (circle at 200, 150)
        # Display coordinates need to be scaled
        display_x = int(200 * self.manager.display_scale)
        display_y = int(150 * self.manager.display_scale)
        
        needs_render = self.manager.handle_mouse_move(display_x, display_y)
        
        # Should trigger hover and need render
        self.assertTrue(needs_render)
        self.assertEqual(self.manager.state.hovered, 0)
        self.assertIsNone(self.manager.state.selected)
        
        # Move mouse away from shapes
        needs_render = self.manager.handle_mouse_move(10, 10)
        
        # Should clear hover and need render
        self.assertTrue(needs_render)
        self.assertIsNone(self.manager.state.hovered)
        self.assertIsNone(self.manager.state.selected)
    
    def test_click_workflow(self):
        """Test complete click workflow."""
        # Mock selection callback to track calls
        callback_mock = Mock()
        self.manager.set_selection_callback(callback_mock)
        
        # Initial state - no selection
        self.assertIsNone(self.manager.state.selected)
        
        # Click on first shape (circle)
        display_x = int(200 * self.manager.display_scale)
        display_y = int(150 * self.manager.display_scale)
        
        needs_render = self.manager.handle_mouse_click(display_x, display_y)
        
        # Should trigger selection and need render
        self.assertTrue(needs_render)
        self.assertEqual(self.manager.state.selected, 0)
        
        # Verify callback was called
        callback_mock.assert_called_once_with(0, self.test_shapes)
        
        # Click on background (no shape)
        callback_mock.reset_mock()
        needs_render = self.manager.handle_mouse_click(10, 10)
        
        # Should clear selection and need render
        self.assertTrue(needs_render)
        self.assertIsNone(self.manager.state.selected)
        
        # Verify callback was called with None
        callback_mock.assert_called_once_with(None, self.test_shapes)
    
    def test_hover_then_click_workflow(self):
        """Test complete hover-then-click workflow."""
        callback_mock = Mock()
        self.manager.set_selection_callback(callback_mock)
        
        # Step 1: Hover over shape
        display_x = int(200 * self.manager.display_scale)
        display_y = int(150 * self.manager.display_scale)
        
        hover_needs_render = self.manager.handle_mouse_move(display_x, display_y)
        self.assertTrue(hover_needs_render)
        self.assertEqual(self.manager.state.hovered, 0)
        self.assertIsNone(self.manager.state.selected)
        
        # Step 2: Click on the same shape
        click_needs_render = self.manager.handle_mouse_click(display_x, display_y)
        self.assertTrue(click_needs_render)
        self.assertEqual(self.manager.state.hovered, 0)  # Should still be hovered
        self.assertEqual(self.manager.state.selected, 0)  # Now also selected
        
        # Verify callback was called
        callback_mock.assert_called_once_with(0, self.test_shapes)
    
    def test_shape_switching_workflow(self):
        """Test workflow when switching between shapes."""
        callback_mock = Mock()
        self.manager.set_selection_callback(callback_mock)
        
        # Select first shape
        display_x1 = int(200 * self.manager.display_scale)
        display_y1 = int(150 * self.manager.display_scale)
        self.manager.handle_mouse_click(display_x1, display_y1)
        
        self.assertEqual(self.manager.state.selected, 0)
        callback_mock.assert_called_with(0, self.test_shapes)
        
        # Switch to second shape
        callback_mock.reset_mock()
        display_x2 = int(350 * self.manager.display_scale)
        display_y2 = int(150 * self.manager.display_scale)
        
        # First hover over second shape
        hover_needs_render = self.manager.handle_mouse_move(display_x2, display_y2)
        self.assertTrue(hover_needs_render)
        self.assertEqual(self.manager.state.hovered, 2)  # Small circle (highest priority)
        self.assertEqual(self.manager.state.selected, 0)  # Still first shape
        
        # Then click on second shape
        click_needs_render = self.manager.handle_mouse_click(display_x2, display_y2)
        self.assertTrue(click_needs_render)
        self.assertEqual(self.manager.state.selected, 2)  # Now second shape
        
        # Verify callback was called with new selection
        callback_mock.assert_called_with(2, self.test_shapes)
    
    def test_overlapping_shapes_workflow(self):
        """Test workflow with overlapping shapes (priority selection)."""
        # Click in area where rectangle and small circle overlap
        # Small circle should be selected due to smaller area
        display_x = int(350 * self.manager.display_scale)
        display_y = int(150 * self.manager.display_scale)
        
        # Hover first
        hover_needs_render = self.manager.handle_mouse_move(display_x, display_y)
        self.assertTrue(hover_needs_render)
        self.assertEqual(self.manager.state.hovered, 2)  # Small circle (index 2)
        
        # Click
        click_needs_render = self.manager.handle_mouse_click(display_x, display_y)
        self.assertTrue(click_needs_render)
        self.assertEqual(self.manager.state.selected, 2)  # Small circle selected
    
    def test_mouse_coordinate_transformation(self):
        """Test that mouse coordinates are properly transformed."""
        # Test with known coordinates
        original_x, original_y = 200, 150
        display_x = int(original_x * self.manager.display_scale)
        display_y = int(original_y * self.manager.display_scale)
        
        # Handle mouse move
        self.manager.handle_mouse_move(display_x, display_y)
        
        # Check that mouse position was recorded in original coordinates
        recorded_x, recorded_y = self.manager.state.mouse_pos
        
        # Should be close to original coordinates (within 1 pixel due to rounding)
        self.assertLessEqual(abs(recorded_x - original_x), 1)
        self.assertLessEqual(abs(recorded_y - original_y), 1)
    
    def test_render_flag_management(self):
        """Test that render flags are properly managed throughout workflow."""
        # Initial state should need render
        self.assertTrue(self.manager.state.needs_render)
        
        # Clear render flag
        self.manager.state.clear_render_flag()
        self.assertFalse(self.manager.state.needs_render)
        
        # Mouse move should set render flag if state changes
        display_x = int(200 * self.manager.display_scale)
        display_y = int(150 * self.manager.display_scale)
        
        needs_render = self.manager.handle_mouse_move(display_x, display_y)
        self.assertTrue(needs_render)
        self.assertTrue(self.manager.state.needs_render)
        
        # Clear flag again
        self.manager.state.clear_render_flag()
        
        # Same mouse position should not set render flag
        needs_render = self.manager.handle_mouse_move(display_x, display_y)
        self.assertFalse(needs_render)
        self.assertFalse(self.manager.state.needs_render)
    
    def test_state_consistency_throughout_workflow(self):
        """Test that state remains consistent throughout complex workflow."""
        callback_mock = Mock()
        self.manager.set_selection_callback(callback_mock)
        
        # Complex workflow: hover -> click -> hover different -> click different -> clear
        
        # Step 1: Hover first shape
        display_x1 = int(200 * self.manager.display_scale)
        display_y1 = int(150 * self.manager.display_scale)
        self.manager.handle_mouse_move(display_x1, display_y1)
        
        self.assertEqual(self.manager.state.hovered, 0)
        self.assertIsNone(self.manager.state.selected)
        
        # Step 2: Click first shape
        self.manager.handle_mouse_click(display_x1, display_y1)
        
        self.assertEqual(self.manager.state.hovered, 0)
        self.assertEqual(self.manager.state.selected, 0)
        
        # Step 3: Hover second shape
        display_x2 = int(350 * self.manager.display_scale)
        display_y2 = int(150 * self.manager.display_scale)
        self.manager.handle_mouse_move(display_x2, display_y2)
        
        self.assertEqual(self.manager.state.hovered, 2)  # Small circle
        self.assertEqual(self.manager.state.selected, 0)  # Still first shape
        
        # Step 4: Click second shape
        self.manager.handle_mouse_click(display_x2, display_y2)
        
        self.assertEqual(self.manager.state.hovered, 2)
        self.assertEqual(self.manager.state.selected, 2)  # Now second shape
        
        # Step 5: Move to empty area
        self.manager.handle_mouse_move(10, 10)
        
        self.assertIsNone(self.manager.state.hovered)
        self.assertEqual(self.manager.state.selected, 2)  # Selection persists
        
        # Step 6: Click empty area
        self.manager.handle_mouse_click(10, 10)
        
        self.assertIsNone(self.manager.state.hovered)
        self.assertIsNone(self.manager.state.selected)  # Selection cleared
        
        # Verify callback was called for each selection change
        expected_calls = [
            unittest.mock.call(0, self.test_shapes),  # First selection
            unittest.mock.call(2, self.test_shapes),  # Second selection
            unittest.mock.call(None, self.test_shapes)  # Clear selection
        ]
        callback_mock.assert_has_calls(expected_calls)
    
    @patch('cv2.imshow')
    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.setMouseCallback')
    def test_opencv_integration(self, mock_callback, mock_resize, mock_window, mock_imshow):
        """Test integration with OpenCV window and mouse callbacks."""
        # Setup window
        window_name = "Test Window"
        self.manager.setup_window(window_name)
        
        # Verify OpenCV calls
        mock_window.assert_called_once_with(window_name, cv2.WINDOW_NORMAL)
        mock_resize.assert_called_once_with(window_name, self.manager.display_width, self.manager.display_height)
        mock_callback.assert_called_once()
        
        # Test initial render
        self.manager.show_initial_render()
        mock_imshow.assert_called_once_with(window_name, unittest.mock.ANY)
    
    def test_cleanup_workflow(self):
        """Test cleanup process and resource management."""
        # Setup window and state
        with patch('cv2.namedWindow'), patch('cv2.resizeWindow'), patch('cv2.setMouseCallback'):
            self.manager.setup_window("Test Window")
        
        # Set some state
        self.manager.state.update_hover(0)
        self.manager.state.update_selection(1)
        callback_mock = Mock()
        self.manager.set_selection_callback(callback_mock)
        
        # Cleanup
        with patch('cv2.destroyWindow') as mock_destroy, \
             patch('cv2.setMouseCallback') as mock_clear_callback:
            
            self.manager.cleanup()
            
            # Verify cleanup actions
            mock_destroy.assert_called_once_with("Test Window")
            mock_clear_callback.assert_called_once()
        
        # Verify state was reset
        self.assertIsNone(self.manager.state.hovered)
        self.assertIsNone(self.manager.state.selected)
        self.assertEqual(self.manager.state.mouse_pos, (0, 0))
        self.assertIsNone(self.manager.selection_callback)
    
    def test_error_handling_in_workflow(self):
        """Test error handling during workflow operations."""
        # Test with invalid shape data
        invalid_shapes = [{'type': 'invalid', 'area_px': 100}]  # Missing hit_contour
        
        manager = InteractionManager(invalid_shapes, self.warped_image)
        
        # Should handle invalid shapes gracefully
        needs_render = manager.handle_mouse_move(100, 100)
        self.assertFalse(needs_render)  # No valid shapes to hover
        
        needs_render = manager.handle_mouse_click(100, 100)
        self.assertFalse(needs_render)  # No valid shapes to select
    
    def test_performance_with_rapid_mouse_movement(self):
        """Test performance with rapid mouse movement simulation."""
        import time
        
        # Simulate rapid mouse movement
        start_time = time.time()
        
        for i in range(100):
            x = 100 + i
            y = 100 + i
            self.manager.handle_mouse_move(x, y)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Should handle 100 mouse moves quickly (under 100ms)
        self.assertLess(elapsed_time, 0.1, f"Rapid mouse movement took too long: {elapsed_time:.4f}s")
    
    def test_instruction_text_workflow(self):
        """Test instruction text changes throughout workflow."""
        # Initial instruction text
        text = self.manager.state.get_instruction_text()
        self.assertEqual(text, "Hover to preview, click to inspect")
        
        # Select circle
        display_x = int(200 * self.manager.display_scale)
        display_y = int(150 * self.manager.display_scale)
        self.manager.handle_mouse_click(display_x, display_y)
        
        text = self.manager.state.get_instruction_text()
        self.assertIn("Circle", text)
        self.assertIn("25mm", text)
        
        # Select rectangle
        display_x = int(350 * self.manager.display_scale)
        display_y = int(120 * self.manager.display_scale)
        self.manager.handle_mouse_click(display_x, display_y)
        
        text = self.manager.state.get_instruction_text()
        self.assertIn("Rectangle", text)
        self.assertIn("20x15mm", text)
        
        # Clear selection
        self.manager.handle_mouse_click(10, 10)
        
        text = self.manager.state.get_instruction_text()
        self.assertEqual(text, "Hover to preview, click to inspect")


class TestWorkflowEdgeCases(unittest.TestCase):
    """Test edge cases in hover-click workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Single shape for edge case testing
        self.single_shape = [{
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 25.0,
            'diameter_mm': 20.0,
            'area_px': np.pi * 25 * 25,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=25.0)
        }]
        
        self.warped_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        self.manager = create_interaction_manager(self.single_shape, self.warped_image)
    
    def test_empty_shapes_workflow(self):
        """Test workflow with no shapes."""
        empty_manager = create_interaction_manager([], self.warped_image)
        
        # Mouse move should not trigger hover
        needs_render = empty_manager.handle_mouse_move(100, 100)
        self.assertFalse(needs_render)
        self.assertIsNone(empty_manager.state.hovered)
        
        # Mouse click should not trigger selection
        needs_render = empty_manager.handle_mouse_click(100, 100)
        self.assertFalse(needs_render)
        self.assertIsNone(empty_manager.state.selected)
    
    def test_boundary_coordinates_workflow(self):
        """Test workflow with boundary coordinates."""
        # Test at image boundaries
        boundary_coords = [
            (0, 0),  # Top-left
            (self.manager.display_width - 1, 0),  # Top-right
            (0, self.manager.display_height - 1),  # Bottom-left
            (self.manager.display_width - 1, self.manager.display_height - 1)  # Bottom-right
        ]
        
        for x, y in boundary_coords:
            # Should handle boundary coordinates without error
            try:
                self.manager.handle_mouse_move(x, y)
                self.manager.handle_mouse_click(x, y)
            except Exception as e:
                self.fail(f"Boundary coordinate ({x}, {y}) caused error: {e}")
    
    def test_rapid_state_changes(self):
        """Test rapid state changes in workflow."""
        display_x = int(100 * self.manager.display_scale)
        display_y = int(100 * self.manager.display_scale)
        
        # Rapid hover/unhover
        for _ in range(10):
            self.manager.handle_mouse_move(display_x, display_y)  # Hover
            self.manager.handle_mouse_move(10, 10)  # Unhover
        
        # Rapid select/deselect
        for _ in range(10):
            self.manager.handle_mouse_click(display_x, display_y)  # Select
            self.manager.handle_mouse_click(10, 10)  # Deselect
        
        # Final state should be consistent
        self.assertIsNone(self.manager.state.hovered)
        self.assertIsNone(self.manager.state.selected)


if __name__ == '__main__':
    unittest.main(verbosity=2)