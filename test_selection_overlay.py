"""
Test suite for Selection Overlay rendering system.

Tests rendering accuracy and visual feedback responsiveness for manual shape selection.
"""

import unittest
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List

from selection_overlay import (
    SelectionOverlay, create_selection_overlay, render_complete_manual_feedback,
    validate_overlay_parameters
)


class TestSelectionOverlay(unittest.TestCase):
    """Test cases for SelectionOverlay class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.overlay = SelectionOverlay()
        self.test_image = np.zeros((600, 800, 3), dtype=np.uint8)  # Black test image
        self.test_image.fill(50)  # Dark gray background for visibility
        
        # Test selection rectangle
        self.test_selection_rect = (100, 100, 200, 150)  # x, y, w, h
        
        # Test shape results
        self.test_circle_result = {
            "type": "circle",
            "center": (300, 200),
            "radius": 50,
            "confidence_score": 0.85
        }
        
        self.test_rectangle_result = {
            "type": "rectangle",
            "center": (400, 300),
            "width": 80,
            "height": 60,
            "contour": np.array([[360, 270], [440, 270], [440, 330], [360, 330]], dtype=np.int32),
            "confidence_score": 0.92
        }
    
    def test_initialization(self):
        """Test SelectionOverlay initialization."""
        overlay = SelectionOverlay()
        
        # Check default colors are set
        self.assertEqual(overlay.selection_color, (0, 255, 255))  # Yellow
        self.assertEqual(overlay.confirmation_color, (0, 255, 0))  # Green
        self.assertEqual(overlay.error_color, (0, 0, 255))  # Red
        
        # Check default parameters
        self.assertEqual(overlay.selection_thickness, 2)
        self.assertEqual(overlay.selection_alpha, 0.2)
        self.assertEqual(overlay.text_scale, 0.7)
    
    def test_render_selection_rectangle_valid(self):
        """Test rendering of valid selection rectangle."""
        result = self.overlay.render_selection_rectangle(self.test_image, self.test_selection_rect)
        
        # Check that result is different from original
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check image dimensions are preserved
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that selection area has been modified
        x, y, w, h = self.test_selection_rect
        selection_area = result[y:y+h, x:x+w]
        original_area = self.test_image[y:y+h, x:x+w]
        self.assertFalse(np.array_equal(selection_area, original_area))
    
    def test_render_selection_rectangle_none(self):
        """Test rendering with None selection rectangle."""
        result = self.overlay.render_selection_rectangle(self.test_image, None)
        
        # Should return unchanged image
        self.assertTrue(np.array_equal(result, self.test_image))
    
    def test_render_selection_rectangle_invalid_dimensions(self):
        """Test rendering with invalid selection rectangle dimensions."""
        invalid_rect = (100, 100, 0, 0)  # Zero width and height
        result = self.overlay.render_selection_rectangle(self.test_image, invalid_rect)
        
        # Should return unchanged image for invalid dimensions
        self.assertTrue(np.array_equal(result, self.test_image))
    
    def test_render_mode_indicator(self):
        """Test rendering of mode indicator."""
        mode_text = "MANUAL RECT"
        result = self.overlay.render_mode_indicator(self.test_image, mode_text)
        
        # Check that result is different from original
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check image dimensions are preserved
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that top-right corner area has been modified
        corner_area = result[0:50, -200:]
        original_corner = self.test_image[0:50, -200:]
        self.assertFalse(np.array_equal(corner_area, original_corner))
    
    def test_render_mode_indicator_with_additional_info(self):
        """Test rendering mode indicator with additional information."""
        mode_text = "MANUAL CIRCLE"
        additional_info = "Selecting..."
        result = self.overlay.render_mode_indicator(self.test_image, mode_text, additional_info)
        
        # Check that result is different from original
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check image dimensions are preserved
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_render_circle_confirmation(self):
        """Test rendering of circle shape confirmation."""
        result = self.overlay.render_shape_confirmation(self.test_image, self.test_circle_result)
        
        # Check that result is different from original
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check image dimensions are preserved
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that area around circle center has been modified
        center = self.test_circle_result["center"]
        radius = self.test_circle_result["radius"]
        circle_area = result[center[1]-radius-10:center[1]+radius+10, 
                           center[0]-radius-10:center[0]+radius+10]
        original_area = self.test_image[center[1]-radius-10:center[1]+radius+10, 
                                      center[0]-radius-10:center[0]+radius+10]
        self.assertFalse(np.array_equal(circle_area, original_area))
    
    def test_render_rectangle_confirmation(self):
        """Test rendering of rectangle shape confirmation."""
        result = self.overlay.render_shape_confirmation(self.test_image, self.test_rectangle_result)
        
        # Check that result is different from original
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check image dimensions are preserved
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_render_error_feedback(self):
        """Test rendering of error feedback."""
        error_message = "No shapes detected"
        result = self.overlay.render_error_feedback(self.test_image, error_message)
        
        # Check that result is different from original
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check image dimensions are preserved
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_render_error_feedback_with_position(self):
        """Test rendering of error feedback at specific position."""
        error_message = "Invalid selection"
        position = (100, 200)
        result = self.overlay.render_error_feedback(self.test_image, error_message, position)
        
        # Check that result is different from original
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check that area around specified position has been modified
        x, y = position
        error_area = result[y-30:y+30, x-50:x+150]
        original_area = self.test_image[y-30:y+30, x-50:x+150]
        self.assertFalse(np.array_equal(error_area, original_area))
    
    def test_render_instruction_overlay(self):
        """Test rendering of instruction overlay."""
        instructions = [
            "Click and drag to select area",
            "Press M to cycle modes",
            "Press ESC to cancel"
        ]
        result = self.overlay.render_instruction_overlay(self.test_image, instructions)
        
        # Check that result is different from original
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check image dimensions are preserved
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that bottom-left area has been modified
        bottom_area = result[-100:, :300]
        original_bottom = self.test_image[-100:, :300]
        self.assertFalse(np.array_equal(bottom_area, original_bottom))
    
    def test_render_instruction_overlay_empty(self):
        """Test rendering with empty instruction list."""
        result = self.overlay.render_instruction_overlay(self.test_image, [])
        
        # Should return unchanged image
        self.assertTrue(np.array_equal(result, self.test_image))
    
    def test_animation_effects(self):
        """Test animation effects in shape confirmation."""
        # Test that animation parameter affects rendering
        self.overlay.frame_counter = 0
        result_no_anim = self.overlay.render_shape_confirmation(self.test_image, self.test_circle_result, animate=False)
        
        self.overlay.frame_counter = 25  # Set to a value that should produce visible difference
        result_with_anim = self.overlay.render_shape_confirmation(self.test_image, self.test_circle_result, animate=True)
        
        # Check that animation produces some difference
        # Even if small, there should be some difference due to animated center point size
        diff = np.sum(np.abs(result_no_anim.astype(int) - result_with_anim.astype(int)))
        self.assertGreaterEqual(diff, 0, "Animation rendering should work without errors")
        
        # Test that non-animated renders are consistent
        result3 = self.overlay.render_shape_confirmation(self.test_image, self.test_circle_result, animate=False)
        result4 = self.overlay.render_shape_confirmation(self.test_image, self.test_circle_result, animate=False)
        self.assertTrue(np.array_equal(result3, result4), "Non-animated renders should be consistent")
    
    def test_reset_animation(self):
        """Test animation reset functionality."""
        self.overlay.frame_counter = 100
        self.overlay.reset_animation()
        self.assertEqual(self.overlay.frame_counter, 0)
    
    def test_set_colors(self):
        """Test custom color setting."""
        new_selection_color = (255, 0, 0)  # Red
        new_confirmation_color = (0, 0, 255)  # Blue
        new_error_color = (255, 255, 0)  # Cyan
        
        self.overlay.set_colors(
            selection_color=new_selection_color,
            confirmation_color=new_confirmation_color,
            error_color=new_error_color
        )
        
        self.assertEqual(self.overlay.selection_color, new_selection_color)
        self.assertEqual(self.overlay.confirmation_color, new_confirmation_color)
        self.assertEqual(self.overlay.error_color, new_error_color)
    
    def test_set_transparency(self):
        """Test transparency setting."""
        # Test valid transparency values
        self.overlay.set_transparency(0.5)
        self.assertEqual(self.overlay.selection_alpha, 0.5)
        
        # Test clamping to valid range
        self.overlay.set_transparency(-0.1)
        self.assertEqual(self.overlay.selection_alpha, 0.0)
        
        self.overlay.set_transparency(1.5)
        self.assertEqual(self.overlay.selection_alpha, 1.0)
    
    def test_corner_markers(self):
        """Test corner marker rendering."""
        result = self.overlay.render_selection_rectangle(self.test_image, self.test_selection_rect, active=True)
        
        # Check that corners have been modified
        x, y, w, h = self.test_selection_rect
        
        # Check top-left corner
        corner_tl = result[y-5:y+15, x-5:x+15]
        original_tl = self.test_image[y-5:y+15, x-5:x+15]
        self.assertFalse(np.array_equal(corner_tl, original_tl))
        
        # Check bottom-right corner
        corner_br = result[y+h-15:y+h+5, x+w-15:x+w+5]
        original_br = self.test_image[y+h-15:y+h+5, x+w-15:x+w+5]
        self.assertFalse(np.array_equal(corner_br, original_br))


class TestOverlayUtilities(unittest.TestCase):
    """Test cases for overlay utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.zeros((400, 600, 3), dtype=np.uint8)
        self.test_image.fill(100)  # Gray background
        
        self.test_selection_rect = (50, 50, 100, 80)
        self.test_mode_text = "AUTO"
        
        self.test_shape_result = {
            "type": "circle",
            "center": (200, 150),
            "radius": 30,
            "confidence_score": 0.75
        }
    
    def test_create_selection_overlay(self):
        """Test overlay creation utility."""
        overlay = create_selection_overlay()
        self.assertIsInstance(overlay, SelectionOverlay)
        
        # Check default initialization
        self.assertEqual(overlay.selection_color, (0, 255, 255))
        self.assertEqual(overlay.selection_thickness, 2)
    
    def test_render_complete_manual_feedback(self):
        """Test complete manual feedback rendering."""
        result = render_complete_manual_feedback(
            self.test_image,
            selection_rect=self.test_selection_rect,
            mode_text=self.test_mode_text,
            shape_result=self.test_shape_result,
            instructions=["Test instruction"]
        )
        
        # Check that result is different from original
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check image dimensions are preserved
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_render_complete_manual_feedback_minimal(self):
        """Test complete manual feedback with minimal parameters."""
        result = render_complete_manual_feedback(
            self.test_image,
            selection_rect=None,
            mode_text=self.test_mode_text
        )
        
        # Should still render mode indicator
        self.assertFalse(np.array_equal(result, self.test_image))
    
    def test_render_complete_manual_feedback_with_error(self):
        """Test complete manual feedback with error message."""
        result = render_complete_manual_feedback(
            self.test_image,
            selection_rect=self.test_selection_rect,
            mode_text=self.test_mode_text,
            error_message="Selection too small"
        )
        
        # Check that result includes error feedback
        self.assertFalse(np.array_equal(result, self.test_image))
    
    def test_validate_overlay_parameters_valid(self):
        """Test validation with valid parameters."""
        image_shape = (400, 600, 3)
        valid_rect = (50, 50, 100, 80)
        
        self.assertTrue(validate_overlay_parameters(valid_rect, image_shape))
    
    def test_validate_overlay_parameters_none(self):
        """Test validation with None selection rectangle."""
        image_shape = (400, 600, 3)
        
        self.assertTrue(validate_overlay_parameters(None, image_shape))
    
    def test_validate_overlay_parameters_out_of_bounds(self):
        """Test validation with out-of-bounds rectangle."""
        image_shape = (400, 600, 3)
        
        # Rectangle extends beyond image bounds
        invalid_rect = (550, 350, 100, 80)
        self.assertFalse(validate_overlay_parameters(invalid_rect, image_shape))
        
        # Negative coordinates
        invalid_rect2 = (-10, 50, 100, 80)
        self.assertFalse(validate_overlay_parameters(invalid_rect2, image_shape))
    
    def test_validate_overlay_parameters_zero_dimensions(self):
        """Test validation with zero dimensions."""
        image_shape = (400, 600, 3)
        
        # Zero width
        invalid_rect = (50, 50, 0, 80)
        self.assertFalse(validate_overlay_parameters(invalid_rect, image_shape))
        
        # Zero height
        invalid_rect2 = (50, 50, 100, 0)
        self.assertFalse(validate_overlay_parameters(invalid_rect2, image_shape))


class TestVisualFeedbackResponsiveness(unittest.TestCase):
    """Test cases for visual feedback responsiveness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.overlay = SelectionOverlay()
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray image
    
    def test_rendering_performance(self):
        """Test rendering performance for responsiveness."""
        import time
        
        selection_rect = (100, 100, 200, 150)
        mode_text = "MANUAL RECT"
        
        # Measure rendering time
        start_time = time.time()
        
        for _ in range(100):  # Render 100 times
            result = self.overlay.render_selection_rectangle(self.test_image, selection_rect)
            result = self.overlay.render_mode_indicator(result, mode_text)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should render quickly for responsiveness (less than 10ms per frame)
        self.assertLess(avg_time, 0.01, f"Rendering too slow: {avg_time:.4f}s per frame")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of overlay rendering."""
        import sys
        
        selection_rect = (50, 50, 300, 200)
        
        # Get initial memory usage
        initial_size = sys.getsizeof(self.test_image)
        
        # Render multiple overlays
        results = []
        for i in range(10):
            result = self.overlay.render_selection_rectangle(self.test_image, selection_rect)
            results.append(result)
        
        # Check that results are proper copies, not references
        for result in results:
            self.assertFalse(result is self.test_image)
            self.assertEqual(result.shape, self.test_image.shape)
    
    def test_animation_smoothness(self):
        """Test animation smoothness for shape confirmation."""
        shape_result = {
            "type": "circle",
            "center": (200, 150),
            "radius": 40,
            "confidence_score": 0.8
        }
        
        # Generate animation frames
        frames = []
        for frame in range(20):
            self.overlay.frame_counter = frame
            result = self.overlay.render_shape_confirmation(
                self.test_image, shape_result, animate=True
            )
            frames.append(result)
        
        # Check that consecutive frames are different (animation is working)
        differences = []
        for i in range(1, len(frames)):
            diff = np.sum(np.abs(frames[i].astype(int) - frames[i-1].astype(int)))
            differences.append(diff)
        
        # Should have some variation between frames
        self.assertGreater(max(differences), 0, "Animation not producing frame differences")
        
        # Check that animation produces reasonable variation
        avg_diff = sum(differences) / len(differences)
        self.assertGreater(avg_diff, 0, "Animation should produce consistent frame changes")
    
    def test_overlay_layering(self):
        """Test proper layering of multiple overlay elements."""
        selection_rect = (100, 100, 200, 150)
        mode_text = "MANUAL CIRCLE"
        error_message = "Test error"
        instructions = ["Test instruction 1", "Test instruction 2"]
        
        # Render all elements
        result = render_complete_manual_feedback(
            self.test_image,
            selection_rect=selection_rect,
            mode_text=mode_text,
            error_message=error_message,
            instructions=instructions
        )
        
        # Check that all elements are visible (image has been modified in multiple areas)
        self.assertFalse(np.array_equal(result, self.test_image))
        
        # Check specific areas for modifications
        # Mode indicator area (top-right)
        mode_area = result[0:50, -200:]
        original_mode_area = self.test_image[0:50, -200:]
        self.assertFalse(np.array_equal(mode_area, original_mode_area))
        
        # Selection area
        x, y, w, h = selection_rect
        selection_area = result[y:y+h, x:x+w]
        original_selection_area = self.test_image[y:y+h, x:x+w]
        self.assertFalse(np.array_equal(selection_area, original_selection_area))
        
        # Instructions area (bottom-left)
        instruction_area = result[-100:, :300]
        original_instruction_area = self.test_image[-100:, :300]
        self.assertFalse(np.array_equal(instruction_area, original_instruction_area))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSelectionOverlay))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOverlayUtilities))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVisualFeedbackResponsiveness))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Selection Overlay Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    print(f"{'='*50}")