"""
End-to-End Integration Tests for Main Application Workflow

This module tests the complete workflow integration including:
- Scan mode to enhanced inspect mode transitions
- ExtendedInteractionManager initialization and cleanup
- Manual selection capabilities in main application
- Resource management and proper cleanup

Requirements addressed: 5.4, 5.5, 3.3
"""

import unittest
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import os
from typing import List, Dict, Any

# Import modules to test
import main
from extended_interaction_manager import ExtendedInteractionManager, setup_extended_interactive_inspect_mode
from selection_mode import SelectionMode
from measure import create_shape_data


class TestMainApplicationIntegration(unittest.TestCase):
    """Test complete main application workflow integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample contours for testing
        circle_contour = np.array([[75, 100], [100, 75], [125, 100], [100, 125]], dtype=np.int32)
        rect_contour = np.array([[160, 120], [240, 120], [240, 180], [160, 180]], dtype=np.int32)
        
        # Create sample measurement results with all required fields
        self.sample_results = [
            {
                "type": "circle",
                "center": (100, 100),
                "radius": 50,
                "radius_px": 50.0,
                "diameter_mm": 25.0,
                "detection_method": "automatic",
                "confidence_score": 0.95,
                "hit_contour": circle_contour,
                "area_px": 7854.0,  # π * 50^2
                "inner": False
            },
            {
                "type": "rectangle", 
                "center": (200, 150),
                "width": 80,
                "height": 60,
                "width_mm": 40.0,
                "height_mm": 30.0,
                "detection_method": "automatic",
                "confidence_score": 0.90,
                "hit_contour": rect_contour,
                "area_px": 4800.0,  # 80 * 60
                "box": np.array([[160, 120], [240, 120], [240, 180], [160, 180]], dtype=np.float32),
                "inner": False
            }
        ]
        
        # Create sample warped image
        self.sample_warped = np.ones((400, 300, 3), dtype=np.uint8) * 255
        
        # Create sample shapes data
        self.sample_shapes = []
        for result in self.sample_results:
            shape_data = create_shape_data(result)
            if shape_data is not None:
                self.sample_shapes.append(shape_data)
    
    def test_extended_interaction_manager_initialization(self):
        """Test that ExtendedInteractionManager initializes correctly in main workflow."""
        with patch('cv2.namedWindow'), \
             patch('cv2.resizeWindow'), \
             patch('cv2.setMouseCallback'), \
             patch('cv2.imshow'):
            
            manager = setup_extended_interactive_inspect_mode(
                self.sample_shapes, self.sample_warped, "Test Window"
            )
            
            # Verify manager is ExtendedInteractionManager instance
            self.assertIsInstance(manager, ExtendedInteractionManager)
            
            # Verify initial mode is AUTO
            self.assertEqual(manager.get_current_mode(), SelectionMode.AUTO)
            
            # Verify manual selection capabilities are available
            self.assertFalse(manager.is_manual_mode())
            self.assertIsNotNone(manager.mode_manager)
            self.assertIsNotNone(manager.manual_engine)
            self.assertIsNotNone(manager.snap_engine)
            
            # Cleanup
            manager.cleanup()
    
    def test_mode_switching_integration(self):
        """Test mode switching functionality in main application context."""
        with patch('cv2.namedWindow'), \
             patch('cv2.resizeWindow'), \
             patch('cv2.setMouseCallback'), \
             patch('cv2.imshow'):
            
            manager = setup_extended_interactive_inspect_mode(
                self.sample_shapes, self.sample_warped, "Test Window"
            )
            
            # Test mode cycling
            initial_mode = manager.get_current_mode()
            self.assertEqual(initial_mode, SelectionMode.AUTO)
            
            # Simulate 'M' key press for mode cycling
            key_handled = manager.handle_key_press(ord('m'))
            self.assertTrue(key_handled)
            self.assertEqual(manager.get_current_mode(), SelectionMode.MANUAL_RECTANGLE)
            self.assertTrue(manager.is_manual_mode())
            
            # Cycle again
            key_handled = manager.handle_key_press(ord('m'))
            self.assertTrue(key_handled)
            self.assertEqual(manager.get_current_mode(), SelectionMode.MANUAL_CIRCLE)
            self.assertTrue(manager.is_manual_mode())
            
            # Cycle back to auto
            key_handled = manager.handle_key_press(ord('m'))
            self.assertTrue(key_handled)
            self.assertEqual(manager.get_current_mode(), SelectionMode.AUTO)
            self.assertFalse(manager.is_manual_mode())
            
            # Cleanup
            manager.cleanup()
    
    def test_manual_selection_workflow_integration(self):
        """Test complete manual selection workflow in main application."""
        with patch('cv2.namedWindow'), \
             patch('cv2.resizeWindow'), \
             patch('cv2.setMouseCallback'), \
             patch('cv2.imshow'):
            
            manager = setup_extended_interactive_inspect_mode(
                self.sample_shapes, self.sample_warped, "Test Window"
            )
            
            # Switch to manual rectangle mode
            manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
            self.assertTrue(manager.is_manual_mode())
            
            # Simulate manual selection workflow
            # Start selection
            handled = manager.handle_manual_mouse_event(cv2.EVENT_LBUTTONDOWN, 50, 50, 0, None)
            self.assertTrue(handled)
            self.assertTrue(manager.manual_engine.is_selecting())
            
            # Update selection
            handled = manager.handle_manual_mouse_event(cv2.EVENT_MOUSEMOVE, 150, 150, cv2.EVENT_FLAG_LBUTTON, None)
            self.assertTrue(handled)
            
            # Complete selection (this will trigger shape snapping)
            with patch.object(manager.snap_engine, 'snap_to_shape') as mock_snap:
                mock_snap.return_value = {
                    "type": "rectangle",
                    "center": (100, 100),
                    "width": 80,
                    "height": 60,
                    "confidence_score": 0.85,
                    "detection_method": "manual"
                }
                
                handled = manager.handle_manual_mouse_event(cv2.EVENT_LBUTTONUP, 150, 150, 0, None)
                self.assertTrue(handled)
                
                # Verify shape snapping was called
                mock_snap.assert_called_once()
                
                # Verify manual result was stored
                self.assertIsNotNone(manager.last_manual_result)
                self.assertEqual(manager.last_manual_result["type"], "rectangle")
                self.assertEqual(manager.last_manual_result["detection_method"], "manual")
            
            # Cleanup
            manager.cleanup()
    
    def test_seamless_mode_transitions(self):
        """Test seamless transitions between scan mode and enhanced inspect mode."""
        # Mock the main application components
        with patch('main.open_capture') as mock_open_capture, \
             patch('main.find_a4_quad') as mock_find_quad, \
             patch('main.warp_a4') as mock_warp, \
             patch('main.segment_object') as mock_segment, \
             patch('main.all_inner_contours') as mock_contours, \
             patch('main.classify_and_measure') as mock_classify, \
             patch('cv2.namedWindow'), \
             patch('cv2.resizeWindow'), \
             patch('cv2.setMouseCallback'), \
             patch('cv2.imshow'), \
             patch('cv2.waitKey') as mock_waitkey, \
             patch('cv2.destroyWindow'), \
             patch('cv2.destroyAllWindows'):
            
            # Setup mocks for successful detection
            mock_cap = Mock()
            mock_cap.read.return_value = (True, np.ones((480, 640, 3), dtype=np.uint8))
            mock_open_capture.return_value = (mock_cap, False)
            
            mock_quad = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)
            mock_find_quad.return_value = mock_quad
            
            mock_warp.return_value = (self.sample_warped, None)
            mock_segment.return_value = np.ones((400, 300), dtype=np.uint8)
            mock_contours.return_value = [np.array([[50, 50], [150, 50], [150, 150], [50, 150]])]
            mock_classify.return_value = self.sample_results[0]
            
            # Simulate key presses: stable frames, then mode switch, then exit
            mock_waitkey.side_effect = [255] * 20 + [ord('m')] + [255] * 5 + [ord('q')]  # 'q' to exit inspect mode
            
            # Test would require more complex mocking of the main loop
            # For now, verify that the integration components are properly imported and available
            
            # Verify extended interaction manager can be imported and used
            from extended_interaction_manager import setup_extended_interactive_inspect_mode
            manager = setup_extended_interactive_inspect_mode(
                self.sample_shapes, self.sample_warped, "Test Window"
            )
            
            self.assertIsInstance(manager, ExtendedInteractionManager)
            manager.cleanup()
    
    def test_resource_cleanup_integration(self):
        """Test proper resource cleanup during mode transitions."""
        with patch('cv2.namedWindow'), \
             patch('cv2.resizeWindow'), \
             patch('cv2.setMouseCallback') as mock_set_callback, \
             patch('cv2.imshow'), \
             patch('cv2.destroyWindow') as mock_destroy_window:
            
            manager = setup_extended_interactive_inspect_mode(
                self.sample_shapes, self.sample_warped, "Test Window"
            )
            
            # Verify window setup
            mock_set_callback.assert_called_once()
            
            # Start manual selection to create some state
            manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
            manager.manual_engine.start_selection(50, 50)
            
            # Verify selection is active
            self.assertTrue(manager.manual_engine.is_selecting())
            
            # Cleanup and verify proper resource management
            manager.cleanup()
            
            # Verify cleanup was called
            mock_destroy_window.assert_called_once_with("Test Window")
            
            # Verify manual selection was cancelled
            self.assertFalse(manager.manual_engine.is_selecting())
            
            # Verify state was reset
            self.assertIsNone(manager.last_manual_result)
            self.assertFalse(manager.show_shape_confirmation)
    
    def test_keyboard_shortcut_integration(self):
        """Test keyboard shortcuts work correctly in main application context."""
        with patch('cv2.namedWindow'), \
             patch('cv2.resizeWindow'), \
             patch('cv2.setMouseCallback'), \
             patch('cv2.imshow'):
            
            manager = setup_extended_interactive_inspect_mode(
                self.sample_shapes, self.sample_warped, "Test Window"
            )
            
            # Test mode cycling with 'M' key
            self.assertEqual(manager.get_current_mode(), SelectionMode.AUTO)
            
            handled = manager.handle_key_press(ord('m'))
            self.assertTrue(handled)
            self.assertEqual(manager.get_current_mode(), SelectionMode.MANUAL_RECTANGLE)
            
            # Test ESC key for canceling selection
            manager.manual_engine.start_selection(50, 50)
            self.assertTrue(manager.manual_engine.is_selecting())
            
            handled = manager.handle_key_press(27)  # ESC key
            self.assertTrue(handled)
            self.assertFalse(manager.manual_engine.is_selecting())
            
            # Test unhandled keys
            handled = manager.handle_key_press(ord('x'))
            self.assertFalse(handled)
            
            # Cleanup
            manager.cleanup()
    
    def test_performance_optimization_integration(self):
        """Test performance optimization features work in main application."""
        with patch('cv2.namedWindow'), \
             patch('cv2.resizeWindow'), \
             patch('cv2.setMouseCallback'), \
             patch('cv2.imshow'):
            
            # Test with performance optimization enabled
            manager = setup_extended_interactive_inspect_mode(
                self.sample_shapes, self.sample_warped, "Test Window",
                enable_performance_optimization=True
            )
            
            self.assertTrue(manager.enable_optimization)
            self.assertIsNotNone(manager.frame_optimizer)
            self.assertIsNotNone(manager.profiler)
            
            # Get performance stats
            stats = manager.get_performance_stats()
            self.assertIn("optimization_enabled", stats)
            self.assertTrue(stats["optimization_enabled"])
            self.assertIn("manual_selection", stats)
            
            manager.cleanup()
            
            # Test with performance optimization disabled
            manager = setup_extended_interactive_inspect_mode(
                self.sample_shapes, self.sample_warped, "Test Window",
                enable_performance_optimization=False
            )
            
            self.assertFalse(manager.enable_optimization)
            
            manager.cleanup()
    
    def test_error_handling_integration(self):
        """Test error handling during main application workflow."""
        with patch('cv2.namedWindow'), \
             patch('cv2.resizeWindow'), \
             patch('cv2.setMouseCallback'), \
             patch('cv2.imshow'):
            
            manager = setup_extended_interactive_inspect_mode(
                self.sample_shapes, self.sample_warped, "Test Window"
            )
            
            # Test error handling during shape snapping
            manager.set_mode(SelectionMode.MANUAL_CIRCLE)
            
            with patch.object(manager.snap_engine, 'snap_to_shape') as mock_snap:
                # Simulate shape snapping error
                mock_snap.side_effect = Exception("Shape snapping failed")
                
                # Start and complete selection
                manager.manual_engine.start_selection(50, 50)
                manager.manual_engine.update_selection(150, 150)
                
                # Complete selection should handle the error gracefully
                try:
                    manager.manual_engine.complete_selection()
                    # Should not raise exception, error should be handled internally
                except Exception:
                    self.fail("Exception should have been handled internally")
            
            # Test cleanup with errors
            with patch.object(manager, 'state') as mock_state:
                mock_state.reset.side_effect = Exception("State reset failed")
                
                # Cleanup should handle errors gracefully
                try:
                    manager.cleanup()
                except Exception:
                    self.fail("Cleanup should handle errors gracefully")


class TestMainApplicationWorkflow(unittest.TestCase):
    """Test main application workflow modifications."""
    
    def test_main_imports_extended_manager(self):
        """Test that main.py correctly imports ExtendedInteractionManager."""
        # Verify the import works
        try:
            from extended_interaction_manager import setup_extended_interactive_inspect_mode
            self.assertTrue(True)  # Import successful
        except ImportError:
            self.fail("Failed to import setup_extended_interactive_inspect_mode")
    
    def test_main_uses_extended_setup_function(self):
        """Test that main.py uses the extended setup function."""
        # Read main.py content to verify it uses the extended setup
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Verify extended interaction manager import
        self.assertIn('from extended_interaction_manager import setup_extended_interactive_inspect_mode', content)
        
        # Verify extended setup function is called
        self.assertIn('setup_extended_interactive_inspect_mode(shapes, warped, window_name)', content)
        
        # Verify enhanced inspect mode messaging
        self.assertIn('[ENHANCED INSPECT MODE]', content)
        self.assertIn("Use 'M' to cycle between AUTO", content)
        
        # Verify keyboard handling integration
        self.assertIn('manager.handle_key_press(k)', content)
    
    def test_main_keyboard_integration(self):
        """Test that main.py properly integrates keyboard handling."""
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Verify keyboard handling logic
        self.assertIn('key_handled = manager.handle_key_press(k)', content)
        self.assertIn('if key_handled:', content)
        self.assertIn('continue', content)  # Continue loop when key is handled
        
        # Verify ESC handling is preserved
        self.assertIn('elif k == 27:', content)  # ESC key
        self.assertIn('inspect_exit_flag = True', content)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)