"""
Integration tests for ExtendedInteractionManager

Tests seamless mode switching, event handling coordination, and integration
between automatic hit testing and manual selection workflows.

Requirements tested: 3.1, 3.4, 4.1, 4.2
"""

import unittest
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Tuple

from extended_interaction_manager import ExtendedInteractionManager, SelectionOverlay
from selection_mode import SelectionMode
from manual_selection_engine import ManualSelectionEngine


class TestExtendedInteractionManager(unittest.TestCase):
    """Test cases for ExtendedInteractionManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test image and shapes
        self.test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        self.test_shapes = [
            {
                "type": "circle",
                "center": (200, 200),
                "radius": 50,
                "diameter_mm": 25.0,
                "area": 7853.98,
                "contour": np.array([[[150, 200]], [[200, 150]], [[250, 200]], [[200, 250]]])
            },
            {
                "type": "rectangle",
                "center": (400, 300),
                "width": 80,
                "height": 60,
                "width_mm": 40.0,
                "height_mm": 30.0,
                "area": 4800,
                "contour": np.array([[[360, 270]], [[440, 270]], [[440, 330]], [[360, 330]]])
            }
        ]
        
        # Create manager instance
        self.manager = ExtendedInteractionManager(
            self.test_shapes, self.test_image, display_height=400
        )
        
        # Mock the window setup to avoid actual OpenCV windows in tests
        self.manager.window_name = "test_window"
    
    def test_initialization(self):
        """Test proper initialization of ExtendedInteractionManager."""
        # Check that all components are initialized
        self.assertIsNotNone(self.manager.mode_manager)
        self.assertIsNotNone(self.manager.manual_engine)
        self.assertIsNotNone(self.manager.selection_overlay)
        self.assertIsNotNone(self.manager.enhanced_analyzer)
        self.assertIsNotNone(self.manager.snap_engine)
        
        # Check initial mode is AUTO
        self.assertEqual(self.manager.get_current_mode(), SelectionMode.AUTO)
        
        # Check that manual selection state is properly initialized
        self.assertFalse(self.manager.manual_engine.is_selecting())
        self.assertIsNone(self.manager.last_manual_result)
        self.assertFalse(self.manager.show_shape_confirmation)
    
    def test_mode_cycling(self):
        """Test mode cycling functionality."""
        # Test cycling through all modes
        initial_mode = self.manager.get_current_mode()
        self.assertEqual(initial_mode, SelectionMode.AUTO)
        
        # Cycle to MANUAL_RECTANGLE
        handled = self.manager.handle_key_press(ord('m'))
        self.assertTrue(handled)
        self.assertEqual(self.manager.get_current_mode(), SelectionMode.MANUAL_RECTANGLE)
        
        # Cycle to MANUAL_CIRCLE
        handled = self.manager.handle_key_press(ord('m'))
        self.assertTrue(handled)
        self.assertEqual(self.manager.get_current_mode(), SelectionMode.MANUAL_CIRCLE)
        
        # Cycle back to AUTO
        handled = self.manager.handle_key_press(ord('m'))
        self.assertTrue(handled)
        self.assertEqual(self.manager.get_current_mode(), SelectionMode.AUTO)
    
    def test_keyboard_shortcuts(self):
        """Test keyboard shortcut handling."""
        # Test mode cycling key
        handled = self.manager.handle_key_press(ord('m'))
        self.assertTrue(handled)
        
        # Test ESC key (should return False when no active selection)
        handled = self.manager.handle_key_press(27)  # ESC
        self.assertFalse(handled)
        
        # Test unknown key
        handled = self.manager.handle_key_press(ord('x'))
        self.assertFalse(handled)
    
    def test_manual_mode_detection(self):
        """Test manual mode detection methods."""
        # Initially in AUTO mode
        self.assertFalse(self.manager.is_manual_mode())
        
        # Switch to manual rectangle mode
        self.manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        self.assertTrue(self.manager.is_manual_mode())
        
        # Switch to manual circle mode
        self.manager.set_mode(SelectionMode.MANUAL_CIRCLE)
        self.assertTrue(self.manager.is_manual_mode())
        
        # Switch back to auto mode
        self.manager.set_mode(SelectionMode.AUTO)
        self.assertFalse(self.manager.is_manual_mode())
    
    def test_manual_selection_mouse_events(self):
        """Test manual selection mouse event handling."""
        # Switch to manual mode
        self.manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        
        # Test mouse down event (start selection)
        handled = self.manager.handle_manual_mouse_event(
            cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None
        )
        self.assertTrue(handled)
        self.assertTrue(self.manager.manual_engine.is_selecting())
        
        # Test mouse move event (update selection)
        handled = self.manager.handle_manual_mouse_event(
            cv2.EVENT_MOUSEMOVE, 200, 200, cv2.EVENT_FLAG_LBUTTON, None
        )
        self.assertTrue(handled)
        
        # Test mouse up event (complete selection)
        with patch.object(self.manager.snap_engine, 'snap_to_shape') as mock_snap:
            mock_snap.return_value = {
                "type": "rectangle",
                "center": (150, 150),
                "width": 80,
                "height": 60,
                "confidence_score": 0.85
            }
            
            handled = self.manager.handle_manual_mouse_event(
                cv2.EVENT_LBUTTONUP, 200, 200, 0, None
            )
            self.assertTrue(handled)
            self.assertFalse(self.manager.manual_engine.is_selecting())
    
    def test_automatic_mode_mouse_events(self):
        """Test that automatic mode mouse events work correctly."""
        # Ensure we're in AUTO mode
        self.manager.set_mode(SelectionMode.AUTO)
        
        # Mock the parent class methods
        with patch.object(self.manager, 'handle_mouse_move') as mock_move, \
             patch.object(self.manager, 'handle_mouse_click') as mock_click:
            
            mock_move.return_value = True
            mock_click.return_value = True
            
            # Test mouse move in auto mode
            self.manager._on_mouse_event(cv2.EVENT_MOUSEMOVE, 100, 100, 0, None)
            mock_move.assert_called_once_with(100, 100)
            
            # Test mouse click in auto mode
            self.manager._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
            mock_click.assert_called_once_with(100, 100)
    
    def test_mode_switching_cancels_selection(self):
        """Test that switching modes cancels active manual selection."""
        # Switch to manual mode and start selection
        self.manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        self.manager.manual_engine.start_selection(100, 100)
        self.assertTrue(self.manager.manual_engine.is_selecting())
        
        # Switch modes - should cancel selection
        self.manager.handle_key_press(ord('m'))
        self.assertFalse(self.manager.manual_engine.is_selecting())
    
    def test_esc_cancels_active_selection(self):
        """Test that ESC key cancels active manual selection."""
        # Switch to manual mode and start selection
        self.manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        self.manager.manual_engine.start_selection(100, 100)
        self.assertTrue(self.manager.manual_engine.is_selecting())
        
        # Press ESC - should cancel selection
        handled = self.manager.handle_key_press(27)  # ESC
        self.assertTrue(handled)
        self.assertFalse(self.manager.manual_engine.is_selecting())
    
    def test_manual_selection_callbacks(self):
        """Test manual selection callback integration."""
        # Mock the selection callback
        mock_callback = Mock()
        self.manager.set_selection_callback(mock_callback)
        
        # Switch to manual mode
        self.manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        
        # Mock successful shape detection
        mock_shape_result = {
            "type": "rectangle",
            "center": (150, 150),
            "width": 80,
            "height": 60,
            "confidence_score": 0.85
        }
        
        with patch.object(self.manager.snap_engine, 'snap_to_shape') as mock_snap:
            mock_snap.return_value = mock_shape_result
            
            # Complete a manual selection
            selection_rect = (100, 100, 100, 100)
            self.manager._on_manual_selection_complete(selection_rect)
            
            # Check that callback was called
            mock_callback.assert_called_once()
            
            # Check that manual result was stored
            self.assertEqual(self.manager.last_manual_result, mock_shape_result)
            self.assertTrue(self.manager.show_shape_confirmation)
    
    def test_render_with_manual_overlays(self):
        """Test rendering with manual selection overlays."""
        # Mock the base render method
        base_image = np.zeros((400, 600, 3), dtype=np.uint8)
        
        with patch.object(self.manager, 'render_current_state') as mock_render:
            mock_render.return_value = base_image
            
            # Test rendering in AUTO mode
            self.manager.set_mode(SelectionMode.AUTO)
            result = self.manager.render_with_manual_overlays()
            self.assertIsNotNone(result)
            
            # Test rendering in MANUAL mode
            self.manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
            result = self.manager.render_with_manual_overlays()
            self.assertIsNotNone(result)
    
    def test_coordinate_transformation_integration(self):
        """Test coordinate transformation between display and original space."""
        # Test with known display scale
        display_scale = 0.5  # 50% scale
        self.manager.manual_engine.set_display_scale(display_scale)
        
        # Start selection in display coordinates
        display_x, display_y = 200, 150
        self.manager.manual_engine.start_selection(display_x, display_y)
        
        # Check that coordinates are properly transformed
        selection_info = self.manager.manual_engine.get_selection_info()
        self.assertTrue(selection_info["is_selecting"])
        
        # The internal coordinates should be in original space
        # (transformed from display coordinates)
        expected_orig_x = int(display_x / display_scale)
        expected_orig_y = int(display_y / display_scale)
        
        # Complete selection to get final rectangle
        self.manager.manual_engine.update_selection(display_x + 100, display_y + 100)
        final_rect = self.manager.manual_engine.complete_selection()
        
        self.assertIsNotNone(final_rect)
        # Check that the rectangle is in original coordinate space
        x, y, w, h = final_rect
        self.assertGreater(w, 100)  # Should be larger due to inverse scaling
        self.assertGreater(h, 100)
    
    def test_performance_stats_integration(self):
        """Test performance statistics collection."""
        stats = self.manager.get_performance_stats()
        
        # Check that manual selection stats are included
        self.assertIn("manual_selection", stats)
        manual_stats = stats["manual_selection"]
        
        self.assertIn("current_mode", manual_stats)
        self.assertIn("is_selecting", manual_stats)
        self.assertIn("has_manual_result", manual_stats)
        self.assertIn("confirmation_active", manual_stats)
        
        # Check that shape snapping stats are included
        self.assertIn("shape_snapping", stats)
    
    def test_cleanup_integration(self):
        """Test cleanup of manual selection components."""
        # Start a manual selection
        self.manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        self.manager.manual_engine.start_selection(100, 100)
        self.manager.last_manual_result = {"type": "test"}
        self.manager.show_shape_confirmation = True
        
        # Cleanup should reset everything
        self.manager.cleanup()
        
        self.assertFalse(self.manager.manual_engine.is_selecting())
        self.assertIsNone(self.manager.last_manual_result)
        self.assertFalse(self.manager.show_shape_confirmation)
    
    def test_manual_selection_info(self):
        """Test manual selection information retrieval."""
        info = self.manager.get_manual_selection_info()
        
        # Check required fields
        self.assertIn("current_mode", info)
        self.assertIn("is_manual_mode", info)
        self.assertIn("is_selecting", info)
        self.assertIn("selection_info", info)
        self.assertIn("last_result", info)
        self.assertIn("show_confirmation", info)
        
        # Check initial values
        self.assertEqual(info["current_mode"], "auto")
        self.assertFalse(info["is_manual_mode"])
        self.assertFalse(info["is_selecting"])
        self.assertIsNone(info["last_result"])
        self.assertFalse(info["show_confirmation"])


class TestSelectionOverlay(unittest.TestCase):
    """Test cases for SelectionOverlay rendering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.overlay = SelectionOverlay()
        self.test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    def test_render_selection_rectangle(self):
        """Test selection rectangle rendering."""
        selection_rect = (100, 100, 200, 150)
        
        result = self.overlay.render_selection_rectangle(self.test_image, selection_rect)
        
        # Check that result is not None and has correct shape
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that the image was modified (not identical to original)
        self.assertFalse(np.array_equal(result, self.test_image))
    
    def test_render_mode_indicator(self):
        """Test mode indicator rendering."""
        mode_text = "MANUAL RECT"
        
        result = self.overlay.render_mode_indicator(self.test_image, mode_text)
        
        # Check that result is not None and has correct shape
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that the image was modified
        self.assertFalse(np.array_equal(result, self.test_image))
    
    def test_render_shape_confirmation_circle(self):
        """Test shape confirmation rendering for circles."""
        shape_result = {
            "type": "circle",
            "center": (200, 200),
            "radius": 50
        }
        
        result = self.overlay.render_shape_confirmation(self.test_image, shape_result)
        
        # Check that result is not None and has correct shape
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that the image was modified
        self.assertFalse(np.array_equal(result, self.test_image))
    
    def test_render_shape_confirmation_rectangle(self):
        """Test shape confirmation rendering for rectangles."""
        shape_result = {
            "type": "rectangle",
            "contour": np.array([[[150, 150]], [[250, 150]], [[250, 250]], [[150, 250]]])
        }
        
        result = self.overlay.render_shape_confirmation(self.test_image, shape_result)
        
        # Check that result is not None and has correct shape
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that the image was modified
        self.assertFalse(np.array_equal(result, self.test_image))
    
    def test_render_with_none_selection(self):
        """Test rendering with None selection rectangle."""
        result = self.overlay.render_selection_rectangle(self.test_image, None)
        
        # Should return original image unchanged
        self.assertTrue(np.array_equal(result, self.test_image))


class TestExtendedInteractionManagerIntegration(unittest.TestCase):
    """Integration tests for ExtendedInteractionManager with real components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create a more realistic test image
        self.test_image = np.ones((600, 800, 3), dtype=np.uint8) * 128  # Gray background
        
        # Draw some test shapes on the image
        cv2.circle(self.test_image, (200, 200), 50, (255, 255, 255), -1)  # White circle
        cv2.rectangle(self.test_image, (350, 250), (450, 350), (255, 255, 255), -1)  # White rectangle
        
        self.test_shapes = [
            {
                "type": "circle",
                "center": (200, 200),
                "radius": 50,
                "diameter_mm": 25.0,
                "area": 7853.98,
                "contour": np.array([[[150, 200]], [[200, 150]], [[250, 200]], [[200, 250]]])
            },
            {
                "type": "rectangle",
                "center": (400, 300),
                "width": 100,
                "height": 100,
                "width_mm": 50.0,
                "height_mm": 50.0,
                "area": 10000,
                "contour": np.array([[[350, 250]], [[450, 250]], [[450, 350]], [[350, 350]]])
            }
        ]
        
        self.manager = ExtendedInteractionManager(
            self.test_shapes, self.test_image, display_height=400
        )
    
    def test_end_to_end_manual_selection_workflow(self):
        """Test complete manual selection workflow."""
        # Switch to manual rectangle mode
        self.manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        self.assertTrue(self.manager.is_manual_mode())
        
        # Simulate manual selection around the rectangle
        selection_rect = (340, 240, 120, 120)  # Around the white rectangle
        
        # Mock the shape snapping to return a valid result
        with patch.object(self.manager.snap_engine, 'snap_to_shape') as mock_snap:
            mock_snap.return_value = {
                "type": "rectangle",
                "center": (400, 300),
                "width": 100,
                "height": 100,
                "confidence_score": 0.9,
                "contour": self.test_shapes[1]["contour"]
            }
            
            # Complete the selection
            self.manager._on_manual_selection_complete(selection_rect)
            
            # Check that shape was detected and stored
            self.assertIsNotNone(self.manager.last_manual_result)
            self.assertTrue(self.manager.show_shape_confirmation)
            self.assertEqual(self.manager.last_manual_result["type"], "rectangle")
    
    def test_mode_switching_workflow(self):
        """Test seamless mode switching workflow."""
        # Start in AUTO mode
        self.assertEqual(self.manager.get_current_mode(), SelectionMode.AUTO)
        
        # Switch to MANUAL_RECTANGLE
        self.manager.handle_key_press(ord('m'))
        self.assertEqual(self.manager.get_current_mode(), SelectionMode.MANUAL_RECTANGLE)
        
        # Start a manual selection
        self.manager.manual_engine.start_selection(100, 100)
        self.assertTrue(self.manager.manual_engine.is_selecting())
        
        # Switch modes - should cancel selection
        self.manager.handle_key_press(ord('m'))
        self.assertEqual(self.manager.get_current_mode(), SelectionMode.MANUAL_CIRCLE)
        self.assertFalse(self.manager.manual_engine.is_selecting())
        
        # Switch back to AUTO
        self.manager.handle_key_press(ord('m'))
        self.assertEqual(self.manager.get_current_mode(), SelectionMode.AUTO)
        self.assertIsNone(self.manager.last_manual_result)
    
    def test_rendering_integration(self):
        """Test rendering integration with all components."""
        # Test rendering in different modes
        modes_to_test = [SelectionMode.AUTO, SelectionMode.MANUAL_RECTANGLE, SelectionMode.MANUAL_CIRCLE]
        
        # Mock the base render method to ensure it returns an image
        base_image = np.ones((400, 533, 3), dtype=np.uint8) * 128  # Gray image with correct dimensions
        
        with patch.object(self.manager, 'render_current_state') as mock_render:
            mock_render.return_value = base_image
            
            for mode in modes_to_test:
                self.manager.set_mode(mode)
                
                # Render should work without errors
                result = self.manager.render_with_manual_overlays()
                self.assertIsNotNone(result)
                # Check that height matches display height (400)
                self.assertEqual(result.shape[0], 400)  # Display height scaling
                # Width should be calculated based on aspect ratio, so just check it's reasonable
                self.assertGreater(result.shape[1], 300)  # Should be reasonable width
    
    def test_coordinate_system_consistency(self):
        """Test coordinate system consistency across components."""
        # Set a specific display scale
        display_scale = 0.5
        self.manager.manual_engine.set_display_scale(display_scale)
        
        # Test coordinate transformation
        display_coords = (200, 150)
        original_coords = (int(display_coords[0] / display_scale), int(display_coords[1] / display_scale))
        
        # Start selection and verify coordinates
        self.manager.manual_engine.start_selection(*display_coords)
        selection_info = self.manager.manual_engine.get_selection_info()
        
        # The selection should be active
        self.assertTrue(selection_info["is_selecting"])
        
        # Complete selection and check rectangle
        self.manager.manual_engine.update_selection(display_coords[0] + 100, display_coords[1] + 100)
        final_rect = self.manager.manual_engine.complete_selection()
        
        self.assertIsNotNone(final_rect)
        x, y, w, h = final_rect
        
        # Coordinates should be in original space (larger due to inverse scaling)
        self.assertGreaterEqual(w, 200)  # 100 * 2 (inverse of 0.5 scale)
        self.assertGreaterEqual(h, 200)


def run_integration_tests():
    """Run all integration tests for ExtendedInteractionManager."""
    # Create test loader
    loader = unittest.TestLoader()
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestExtendedInteractionManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSelectionOverlay))
    suite.addTests(loader.loadTestsFromTestCase(TestExtendedInteractionManagerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running ExtendedInteractionManager integration tests...")
    success = run_integration_tests()
    
    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed!")
    
    exit(0 if success else 1)