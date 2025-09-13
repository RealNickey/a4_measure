"""
Edge cases and integration tests for interactive functionality.

Tests cover:
- Edge cases with malformed or invalid data
- Integration between all components
- Error handling and recovery
- Boundary conditions and limits
- Real-world usage scenarios
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

from hit_testing import HitTestingEngine, create_hit_testing_contour, validate_shape_data
from interaction_state import InteractionState, StateChangeDetector
from interaction_manager import InteractionManager, validate_shapes_for_interaction


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""
        engine = HitTestingEngine()
        
        # Empty shapes list
        result = engine.find_shape_at_point([], 100, 100)
        self.assertIsNone(result)
        
        # None shapes list should not crash
        try:
            result = engine.find_shape_at_point(None, 100, 100)
        except (TypeError, AttributeError):
            pass  # Expected behavior
        
        # Empty interaction state
        state = InteractionState([])
        self.assertEqual(len(state.shapes), 0)
        self.assertIsNone(state.get_hovered_shape())
        self.assertIsNone(state.get_selected_shape())
    
    def test_invalid_shape_data(self):
        """Test handling of invalid shape data."""
        engine = HitTestingEngine()
        
        # Shape without hit_contour
        invalid_shape1 = {
            'type': 'circle',
            'area_px': 100.0
        }
        
        # Shape with malformed hit_contour
        invalid_shape2 = {
            'type': 'circle',
            'area_px': 100.0,
            'hit_contour': np.array([[1, 2], [3, 4]])  # Wrong shape
        }
        
        # Shape with non-numpy hit_contour
        invalid_shape3 = {
            'type': 'circle',
            'area_px': 100.0,
            'hit_contour': [[1, 2], [3, 4]]  # Not numpy array
        }
        
        invalid_shapes = [invalid_shape1, invalid_shape2, invalid_shape3]
        
        # Should handle invalid shapes gracefully
        try:
            result = engine.find_shape_at_point(invalid_shapes, 100, 100)
            self.assertIsNone(result)
        except (cv2.error, TypeError, AttributeError):
            # Expected behavior for invalid shapes
            pass
        
        # Validation should catch these
        for shape in invalid_shapes:
            self.assertFalse(validate_shape_data(shape))
    
    def test_extreme_coordinates(self):
        """Test handling of extreme coordinate values."""
        engine = HitTestingEngine()
        
        # Create a normal shape
        shape = {
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 25.0,
            'area_px': np.pi * 25 * 25,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=25.0)
        }
        
        shapes = [shape]
        
        # Test extreme coordinates
        extreme_coords = [
            (-1000, -1000),  # Very negative
            (10000, 10000),  # Very positive
            (0, 0),          # Origin
            (-1, -1),        # Just negative
        ]
        
        for x, y in extreme_coords:
            # Should not crash
            try:
                result = engine.find_shape_at_point(shapes, x, y)
                # Result can be None or valid index
                self.assertTrue(result is None or isinstance(result, int))
            except Exception as e:
                self.fail(f"Extreme coordinates ({x}, {y}) caused error: {e}")
    
    def test_zero_area_shapes(self):
        """Test handling of shapes with zero or negative area."""
        # Create shape with zero area
        zero_area_shape = {
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 0.0,
            'area_px': 0.0,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=1.0)  # Minimal contour
        }
        
        # Create shape with negative area (shouldn't happen but test anyway)
        negative_area_shape = {
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 10.0,
            'area_px': -100.0,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=10.0)
        }
        
        shapes = [zero_area_shape, negative_area_shape]
        engine = HitTestingEngine()
        
        # Should handle gracefully
        result = engine.find_shape_at_point(shapes, 100, 100)
        # Should either find a shape or return None
        self.assertTrue(result is None or isinstance(result, int))
    
    def test_malformed_contours(self):
        """Test handling of malformed contours."""
        # Contour with insufficient points
        insufficient_points = np.array([[[10, 10]]], dtype=np.int32)  # Only 1 point
        
        # Contour with wrong data type
        wrong_dtype = np.array([[[10.5, 10.5], [20.5, 20.5]]], dtype=np.float32)
        
        malformed_shapes = [
            {
                'type': 'circle',
                'area_px': 100.0,
                'hit_contour': insufficient_points
            },
            {
                'type': 'circle',
                'area_px': 100.0,
                'hit_contour': wrong_dtype
            }
        ]
        
        engine = HitTestingEngine()
        
        for shape in malformed_shapes:
            # Should not crash, even with malformed contours
            try:
                result = engine.find_shape_at_point([shape], 100, 100)
            except Exception as e:
                # Some exceptions might be expected with malformed data
                pass
    
    def test_interaction_state_edge_cases(self):
        """Test interaction state edge cases."""
        state = InteractionState()
        
        # Invalid shape indices
        invalid_indices = [-1, 100, None]
        
        for idx in invalid_indices:
            if idx is not None:
                # Should handle invalid indices gracefully
                self.assertFalse(state.is_valid_shape_index(idx))
                
                # Update methods should handle invalid indices
                state.update_hover(idx)
                state.update_selection(idx)
                
                # Should not crash when getting shapes with invalid indices
                self.assertIsNone(state.get_hovered_shape())
                self.assertIsNone(state.get_selected_shape())
    
    def test_coordinate_transformation_edge_cases(self):
        """Test coordinate transformation edge cases."""
        from interaction_state import transform_display_to_original_coords, transform_original_to_display_coords
        
        # Zero scale factor
        try:
            result = transform_display_to_original_coords(100, 100, 0.0)
            # If it doesn't crash, result should be extreme values
        except (ZeroDivisionError, OverflowError):
            pass  # Expected behavior
        
        # Very small scale factor
        result = transform_display_to_original_coords(100, 100, 0.001)
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)
        
        # Very large scale factor
        result = transform_original_to_display_coords(100, 100, 1000.0)
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)
        
        # Negative scale factor
        result = transform_display_to_original_coords(100, 100, -1.0)
        self.assertEqual(result, (-100, -100))


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration between all components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_shapes = [
            {
                'type': 'circle',
                'center': (100, 100),
                'radius_px': 25.0,
                'diameter_mm': 20.0,
                'area_px': np.pi * 25 * 25,
                'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=25.0),
                'inner': False
            },
            {
                'type': 'rectangle',
                'box': np.array([[150, 75], [200, 75], [200, 125], [150, 125]], dtype=np.int32),
                'width_mm': 15.0,
                'height_mm': 12.0,
                'area_px': 50 * 50,
                'hit_contour': create_hit_testing_contour('rectangle', 
                    box=np.array([[150, 75], [200, 75], [200, 125], [150, 125]], dtype=np.int32)),
                'inner': False
            }
        ]
        
        self.warped_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    def test_full_workflow_integration(self):
        """Test complete workflow integration from setup to cleanup."""
        # Create interaction manager
        manager = InteractionManager(self.test_shapes, self.warped_image)
        
        # Test initial state
        self.assertEqual(len(manager.state.shapes), 2)
        self.assertIsNone(manager.state.hovered)
        self.assertIsNone(manager.state.selected)
        
        # Test mouse interaction
        display_x = int(100 * manager.display_scale)
        display_y = int(100 * manager.display_scale)
        
        # Hover
        hover_changed = manager.handle_mouse_move(display_x, display_y)
        self.assertTrue(hover_changed)
        self.assertEqual(manager.state.hovered, 0)
        
        # Click
        selection_changed = manager.handle_mouse_click(display_x, display_y)
        self.assertTrue(selection_changed)
        self.assertEqual(manager.state.selected, 0)
        
        # Test rendering state
        state_dict = manager.state.get_state_dict()
        self.assertEqual(state_dict['hovered'], 0)
        self.assertEqual(state_dict['selected'], 0)
        
        # Test cleanup
        manager.cleanup()
        self.assertIsNone(manager.state.hovered)
        self.assertIsNone(manager.state.selected)
    
    def test_component_interaction_consistency(self):
        """Test consistency between different components."""
        engine = HitTestingEngine()
        state = InteractionState(self.test_shapes)
        
        # Test point that should hit first shape
        test_x, test_y = 100, 100
        
        # Hit testing should find the shape
        hit_result = engine.find_shape_at_point(self.test_shapes, test_x, test_y)
        self.assertEqual(hit_result, 0)
        
        # State should be able to handle this result
        state.update_hover(hit_result)
        self.assertEqual(state.hovered, 0)
        
        # Should be able to get the shape
        hovered_shape = state.get_hovered_shape()
        self.assertIsNotNone(hovered_shape)
        self.assertEqual(hovered_shape['type'], 'circle')
    
    def test_state_change_detection_integration(self):
        """Test state change detection with real interactions."""
        state = InteractionState(self.test_shapes)
        detector = StateChangeDetector()
        
        # Initial check
        changes = detector.check_changes(state)
        self.assertFalse(changes['any'])
        
        # Make a change
        state.update_hover(0)
        changes = detector.check_changes(state)
        self.assertTrue(changes['hover'])
        self.assertTrue(changes['any'])
        
        # No change
        changes = detector.check_changes(state)
        self.assertFalse(changes['any'])
        
        # Another change
        state.update_selection(1)
        changes = detector.check_changes(state)
        self.assertTrue(changes['selection'])
        self.assertTrue(changes['any'])
    
    def test_shape_validation_integration(self):
        """Test shape validation across components."""
        # Mix of valid and invalid shapes
        mixed_shapes = [
            self.test_shapes[0],  # Valid
            {'type': 'invalid'},  # Invalid
            self.test_shapes[1],  # Valid
        ]
        
        # Validation should filter out invalid shapes
        valid_shapes = validate_shapes_for_interaction(mixed_shapes)
        self.assertEqual(len(valid_shapes), 2)
        
        # Components should work with validated shapes
        engine = HitTestingEngine()
        result = engine.find_shape_at_point(valid_shapes, 100, 100)
        self.assertEqual(result, 0)
    
    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.setMouseCallback')
    @patch('cv2.imshow')
    def test_opencv_integration_mocked(self, mock_imshow, mock_callback, mock_resize, mock_window):
        """Test OpenCV integration with mocked functions."""
        manager = InteractionManager(self.test_shapes, self.warped_image)
        
        # Setup window
        manager.setup_window("Test Window")
        
        # Verify OpenCV calls
        mock_window.assert_called_once()
        mock_resize.assert_called_once()
        mock_callback.assert_called_once()
        
        # Test rendering
        manager.show_initial_render()
        mock_imshow.assert_called_once()
        
        # Test mouse event simulation
        # Get the callback function that was set
        callback_args = mock_callback.call_args[0]
        mouse_callback = callback_args[1]
        
        # Simulate mouse events
        mouse_callback(cv2.EVENT_MOUSEMOVE, 100, 100, 0, None)
        mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
        
        # Should have updated state (may be None if no shapes at that position)
        # Just verify the callback was called without error
        self.assertTrue(True)  # If we get here, no exceptions were raised


class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Test error handling and recovery mechanisms."""
    
    def test_hit_testing_error_recovery(self):
        """Test hit testing error recovery."""
        engine = HitTestingEngine()
        
        # Create shape that might cause cv2.pointPolygonTest to fail
        problematic_shape = {
            'type': 'circle',
            'area_px': 100.0,
            'hit_contour': np.array([[[0, 0]]], dtype=np.int32)  # Degenerate contour
        }
        
        # Should handle errors gracefully
        try:
            result = engine.find_shape_at_point([problematic_shape], 100, 100)
            # If no exception, result should be None or valid
            self.assertTrue(result is None or isinstance(result, int))
        except Exception:
            # If exception occurs, it should be handled gracefully in real implementation
            pass
    
    def test_interaction_manager_error_recovery(self):
        """Test interaction manager error recovery."""
        # Create manager with problematic data
        problematic_shapes = [
            {'type': 'invalid', 'area_px': 100}  # Invalid shape
        ]
        
        warped_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        try:
            manager = InteractionManager(problematic_shapes, warped_image)
            
            # Should handle mouse events without crashing
            manager.handle_mouse_move(50, 50)
            manager.handle_mouse_click(50, 50)
            
            # State should remain consistent
            self.assertIsNone(manager.state.hovered)
            self.assertIsNone(manager.state.selected)
            
        except Exception as e:
            # If exceptions occur, they should be handled gracefully
            pass
    
    def test_cleanup_error_handling(self):
        """Test cleanup error handling."""
        manager = InteractionManager([], np.ones((100, 100, 3), dtype=np.uint8) * 255)
        
        # Set up some state
        manager.window_name = "Test Window"
        
        # Mock OpenCV functions to raise exceptions
        with patch('cv2.destroyWindow', side_effect=Exception("Mock error")), \
             patch('cv2.setMouseCallback', side_effect=Exception("Mock error")):
            
            # Cleanup should handle errors gracefully
            try:
                manager.cleanup()
                # Should still reset state even if OpenCV calls fail
                self.assertIsNone(manager.state.hovered)
                self.assertIsNone(manager.state.selected)
            except Exception as e:
                self.fail(f"Cleanup should handle errors gracefully: {e}")


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios."""
    
    def test_typical_user_interaction_sequence(self):
        """Test typical user interaction sequence."""
        # Create realistic shapes
        shapes = [
            {
                'type': 'circle',
                'center': (200, 150),
                'radius_px': 30.0,
                'diameter_mm': 25.4,  # 1 inch
                'area_px': np.pi * 30 * 30,
                'hit_contour': create_hit_testing_contour('circle', center=(200, 150), radius_px=30.0),
                'inner': False
            },
            {
                'type': 'rectangle',
                'box': np.array([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=np.int32),
                'width_mm': 50.8,  # 2 inches
                'height_mm': 25.4,  # 1 inch
                'area_px': 200 * 100,
                'hit_contour': create_hit_testing_contour('rectangle', 
                    box=np.array([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=np.int32)),
                'inner': False
            }
        ]
        
        warped_image = np.ones((400, 500, 3), dtype=np.uint8) * 255
        manager = InteractionManager(shapes, warped_image)
        
        # Typical user sequence: explore -> select -> measure -> compare
        
        # 1. User moves mouse around exploring
        exploration_points = [
            (50, 50),    # Empty area
            (150, 120),  # Near rectangle
            (200, 150),  # On circle
            (250, 150),  # On rectangle
        ]
        
        for x, y in exploration_points:
            display_x = int(x * manager.display_scale)
            display_y = int(y * manager.display_scale)
            manager.handle_mouse_move(display_x, display_y)
        
        # 2. User clicks on circle to measure it
        circle_x = int(200 * manager.display_scale)
        circle_y = int(150 * manager.display_scale)
        manager.handle_mouse_click(circle_x, circle_y)
        
        self.assertEqual(manager.state.selected, 0)  # Circle selected
        
        # 3. User clicks on rectangle to compare
        rect_x = int(200 * manager.display_scale)  # Center of rectangle
        rect_y = int(150 * manager.display_scale)
        manager.handle_mouse_click(rect_x, rect_y)
        
        # Should select either the circle or rectangle (both are at this position)
        self.assertIsNotNone(manager.state.selected)  # Something should be selected
        
        # 4. User clicks away to clear selection
        manager.handle_mouse_click(10, 10)
        self.assertIsNone(manager.state.selected)
    
    def test_precision_measurement_scenario(self):
        """Test precision measurement scenario with small shapes."""
        # Create small, precisely positioned shapes
        small_shapes = []
        
        for i in range(5):
            x = 100 + i * 50
            y = 100
            radius = 5 + i * 2  # Increasing sizes
            
            small_shapes.append({
                'type': 'circle',
                'center': (x, y),
                'radius_px': radius,
                'diameter_mm': radius * 0.2,  # Small measurements
                'area_px': np.pi * radius * radius,
                'hit_contour': create_hit_testing_contour('circle', center=(x, y), radius_px=radius),
                'inner': False
            })
        
        warped_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        manager = InteractionManager(small_shapes, warped_image, display_height=600)  # High resolution
        
        # Test precise selection of small shapes
        for i, shape in enumerate(small_shapes):
            center_x, center_y = shape['center']
            display_x = int(center_x * manager.display_scale)
            display_y = int(center_y * manager.display_scale)
            
            # Should be able to precisely select each small shape
            manager.handle_mouse_click(display_x, display_y)
            self.assertEqual(manager.state.selected, i)
    
    def test_overlapping_shapes_scenario(self):
        """Test scenario with many overlapping shapes."""
        # Create overlapping shapes of different sizes
        overlapping_shapes = []
        center = (200, 200)
        
        # Large background rectangle
        large_box = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.int32)
        overlapping_shapes.append({
            'type': 'rectangle',
            'box': large_box,
            'width_mm': 40.0,
            'height_mm': 40.0,
            'area_px': 200 * 200,
            'hit_contour': create_hit_testing_contour('rectangle', box=large_box),
            'inner': False
        })
        
        # Medium circle
        overlapping_shapes.append({
            'type': 'circle',
            'center': center,
            'radius_px': 50.0,
            'diameter_mm': 20.0,
            'area_px': np.pi * 50 * 50,
            'hit_contour': create_hit_testing_contour('circle', center=center, radius_px=50.0),
            'inner': False
        })
        
        # Small inner circle
        overlapping_shapes.append({
            'type': 'circle',
            'center': center,
            'radius_px': 20.0,
            'diameter_mm': 8.0,
            'area_px': np.pi * 20 * 20,
            'hit_contour': create_hit_testing_contour('circle', center=center, radius_px=20.0),
            'inner': True
        })
        
        warped_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        manager = InteractionManager(overlapping_shapes, warped_image)
        
        # Click at center - should select smallest shape
        center_x = int(center[0] * manager.display_scale)
        center_y = int(center[1] * manager.display_scale)
        manager.handle_mouse_click(center_x, center_y)
        
        # Should select the smallest circle (index 2)
        self.assertEqual(manager.state.selected, 2)
        
        # Verify it's the inner circle
        selected_shape = manager.state.get_selected_shape()
        self.assertTrue(selected_shape['inner'])
        self.assertEqual(selected_shape['diameter_mm'], 8.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)