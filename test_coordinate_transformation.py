"""
Unit tests for coordinate transformation accuracy.

Tests cover:
- Display to original coordinate transformation
- Original to display coordinate transformation
- Boundary checking and validation
- Scaling factor accuracy
- Edge cases and precision
"""

import unittest
import numpy as np
from interaction_state import (
    transform_display_to_original_coords,
    transform_original_to_display_coords,
    validate_mouse_coordinates,
    InteractionState
)


class TestCoordinateTransformation(unittest.TestCase):
    """Test cases for coordinate transformation functions."""
    
    def test_display_to_original_basic(self):
        """Test basic display to original coordinate transformation."""
        # Scale factor of 2.0 (display is 2x larger than original)
        scale = 2.0
        
        # Test center point
        orig_x, orig_y = transform_display_to_original_coords(200, 300, scale)
        self.assertEqual(orig_x, 100)
        self.assertEqual(orig_y, 150)
        
        # Test origin
        orig_x, orig_y = transform_display_to_original_coords(0, 0, scale)
        self.assertEqual(orig_x, 0)
        self.assertEqual(orig_y, 0)
        
        # Test with fractional results (should be rounded down)
        orig_x, orig_y = transform_display_to_original_coords(101, 201, scale)
        self.assertEqual(orig_x, 50)  # 101/2 = 50.5 -> 50
        self.assertEqual(orig_y, 100)  # 201/2 = 100.5 -> 100
    
    def test_original_to_display_basic(self):
        """Test basic original to display coordinate transformation."""
        # Scale factor of 1.5 (display is 1.5x larger than original)
        scale = 1.5
        
        # Test center point
        disp_x, disp_y = transform_original_to_display_coords(100, 200, scale)
        self.assertEqual(disp_x, 150)
        self.assertEqual(disp_y, 300)
        
        # Test origin
        disp_x, disp_y = transform_original_to_display_coords(0, 0, scale)
        self.assertEqual(disp_x, 0)
        self.assertEqual(disp_y, 0)
        
        # Test with fractional results (should be rounded down)
        disp_x, disp_y = transform_original_to_display_coords(33, 67, scale)
        self.assertEqual(disp_x, 49)  # 33*1.5 = 49.5 -> 49
        self.assertEqual(disp_y, 100)  # 67*1.5 = 100.5 -> 100
    
    def test_transformation_roundtrip_accuracy(self):
        """Test that transformations are consistent in both directions."""
        scale = 2.5
        
        # Test multiple points
        test_points = [(0, 0), (100, 150), (50, 75), (200, 300), (1, 1)]
        
        for orig_x, orig_y in test_points:
            # Original -> Display -> Original
            disp_x, disp_y = transform_original_to_display_coords(orig_x, orig_y, scale)
            back_x, back_y = transform_display_to_original_coords(disp_x, disp_y, scale)
            
            # Should be very close (within 1 pixel due to integer rounding)
            self.assertLessEqual(abs(back_x - orig_x), 1, 
                               f"X roundtrip failed: {orig_x} -> {disp_x} -> {back_x}")
            self.assertLessEqual(abs(back_y - orig_y), 1,
                               f"Y roundtrip failed: {orig_y} -> {disp_y} -> {back_y}")
    
    def test_scale_factor_edge_cases(self):
        """Test transformation with edge case scale factors."""
        # Scale factor of 1.0 (no scaling)
        orig_x, orig_y = transform_display_to_original_coords(100, 200, 1.0)
        self.assertEqual(orig_x, 100)
        self.assertEqual(orig_y, 200)
        
        disp_x, disp_y = transform_original_to_display_coords(100, 200, 1.0)
        self.assertEqual(disp_x, 100)
        self.assertEqual(disp_y, 200)
        
        # Very small scale factor
        orig_x, orig_y = transform_display_to_original_coords(100, 200, 0.1)
        self.assertEqual(orig_x, 1000)
        self.assertEqual(orig_y, 2000)
        
        # Very large scale factor
        disp_x, disp_y = transform_original_to_display_coords(10, 20, 10.0)
        self.assertEqual(disp_x, 100)
        self.assertEqual(disp_y, 200)
    
    def test_negative_coordinates(self):
        """Test transformation with negative coordinates."""
        scale = 2.0
        
        # Negative original coordinates
        disp_x, disp_y = transform_original_to_display_coords(-50, -100, scale)
        self.assertEqual(disp_x, -100)
        self.assertEqual(disp_y, -200)
        
        # Negative display coordinates
        orig_x, orig_y = transform_display_to_original_coords(-100, -200, scale)
        self.assertEqual(orig_x, -50)
        self.assertEqual(orig_y, -100)
    
    def test_precision_with_realistic_scales(self):
        """Test precision with realistic scaling factors from actual usage."""
        # Typical scale factors based on A4 image sizes
        realistic_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
        
        # Test points representing typical mouse positions
        test_points = [
            (0, 0),      # Top-left corner
            (595, 842),  # A4 dimensions in pixels (approximate)
            (297, 421),  # Center of A4
            (100, 100),  # Typical shape position
            (500, 700),  # Near bottom-right
        ]
        
        for scale in realistic_scales:
            for orig_x, orig_y in test_points:
                # Test forward and backward transformation
                disp_x, disp_y = transform_original_to_display_coords(orig_x, orig_y, scale)
                back_x, back_y = transform_display_to_original_coords(disp_x, disp_y, scale)
                
                # Verify precision (should be within 1 pixel)
                self.assertLessEqual(abs(back_x - orig_x), 1,
                                   f"Scale {scale}, point ({orig_x}, {orig_y}): precision error in X")
                self.assertLessEqual(abs(back_y - orig_y), 1,
                                   f"Scale {scale}, point ({orig_x}, {orig_y}): precision error in Y")


class TestInteractionStateTransformation(unittest.TestCase):
    """Test coordinate transformation methods in InteractionState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state = InteractionState()
        self.state.set_display_scale(2.0)
    
    def test_state_transform_display_to_original(self):
        """Test InteractionState display to original transformation."""
        orig_x, orig_y = self.state.transform_display_to_original(200, 400)
        self.assertEqual(orig_x, 100)
        self.assertEqual(orig_y, 200)
    
    def test_state_transform_original_to_display(self):
        """Test InteractionState original to display transformation."""
        disp_x, disp_y = self.state.transform_original_to_display(100, 200)
        self.assertEqual(disp_x, 200)
        self.assertEqual(disp_y, 400)
    
    def test_state_scale_change(self):
        """Test transformation after changing scale factor."""
        # Initial scale
        orig_x, orig_y = self.state.transform_display_to_original(100, 200)
        self.assertEqual(orig_x, 50)
        self.assertEqual(orig_y, 100)
        
        # Change scale
        self.state.set_display_scale(0.5)
        orig_x, orig_y = self.state.transform_display_to_original(100, 200)
        self.assertEqual(orig_x, 200)
        self.assertEqual(orig_y, 400)


class TestMouseCoordinateValidation(unittest.TestCase):
    """Test mouse coordinate validation functions."""
    
    def test_valid_coordinates(self):
        """Test validation of valid mouse coordinates."""
        # Test coordinates within bounds
        self.assertTrue(validate_mouse_coordinates(0, 0, 800, 600))
        self.assertTrue(validate_mouse_coordinates(400, 300, 800, 600))
        self.assertTrue(validate_mouse_coordinates(799, 599, 800, 600))
    
    def test_invalid_coordinates(self):
        """Test validation of invalid mouse coordinates."""
        # Test coordinates outside bounds
        self.assertFalse(validate_mouse_coordinates(-1, 0, 800, 600))
        self.assertFalse(validate_mouse_coordinates(0, -1, 800, 600))
        self.assertFalse(validate_mouse_coordinates(800, 300, 800, 600))
        self.assertFalse(validate_mouse_coordinates(400, 600, 800, 600))
        self.assertFalse(validate_mouse_coordinates(800, 600, 800, 600))
    
    def test_edge_coordinates(self):
        """Test validation of edge case coordinates."""
        # Test exact boundary coordinates
        self.assertTrue(validate_mouse_coordinates(0, 0, 1, 1))
        self.assertFalse(validate_mouse_coordinates(1, 0, 1, 1))
        self.assertFalse(validate_mouse_coordinates(0, 1, 1, 1))
    
    def test_zero_dimensions(self):
        """Test validation with zero image dimensions."""
        self.assertFalse(validate_mouse_coordinates(0, 0, 0, 0))
        self.assertFalse(validate_mouse_coordinates(0, 0, 0, 100))
        self.assertFalse(validate_mouse_coordinates(0, 0, 100, 0))


class TestTransformationAccuracyBenchmark(unittest.TestCase):
    """Benchmark tests for transformation accuracy with large datasets."""
    
    def test_large_scale_accuracy(self):
        """Test transformation accuracy with many points and various scales."""
        # Generate test points
        np.random.seed(42)  # For reproducible results
        test_points = [(int(x), int(y)) for x, y in 
                      zip(np.random.randint(0, 1000, 100), 
                          np.random.randint(0, 1000, 100))]
        
        # Test various scale factors
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        
        max_error = 0
        total_error = 0
        test_count = 0
        
        for scale in scales:
            for orig_x, orig_y in test_points:
                # Forward and backward transformation
                disp_x, disp_y = transform_original_to_display_coords(orig_x, orig_y, scale)
                back_x, back_y = transform_display_to_original_coords(disp_x, disp_y, scale)
                
                # Calculate error
                error_x = abs(back_x - orig_x)
                error_y = abs(back_y - orig_y)
                max_error_point = max(error_x, error_y)
                
                max_error = max(max_error, max_error_point)
                total_error += max_error_point
                test_count += 1
        
        # Verify accuracy constraints (allow up to 3 pixels due to integer rounding with various scales)
        self.assertLessEqual(max_error, 3, "Maximum transformation error exceeds 3 pixels")
        
        avg_error = total_error / test_count
        self.assertLess(avg_error, 1.0, f"Average transformation error too high: {avg_error}")
        
        print(f"Transformation accuracy test: Max error = {max_error}, Avg error = {avg_error:.3f}")
    
    def test_mouse_interaction_simulation(self):
        """Simulate realistic mouse interaction patterns and test accuracy."""
        # Simulate mouse movement patterns
        mouse_paths = [
            # Straight line movement
            [(i, 100) for i in range(0, 500, 10)],
            # Diagonal movement
            [(i, i) for i in range(0, 300, 5)],
            # Circular movement
            [(int(150 + 50 * np.cos(t)), int(150 + 50 * np.sin(t))) 
             for t in np.linspace(0, 2*np.pi, 36)],
        ]
        
        scale = 1.75  # Realistic scale factor
        
        for path in mouse_paths:
            for display_x, display_y in path:
                # Transform to original coordinates (as would happen in mouse handler)
                orig_x, orig_y = transform_display_to_original_coords(display_x, display_y, scale)
                
                # Transform back to display (as would happen in rendering)
                back_disp_x, back_disp_y = transform_original_to_display_coords(orig_x, orig_y, scale)
                
                # Verify accuracy (allow up to 2 pixels due to integer rounding)
                error_x = abs(back_disp_x - display_x)
                error_y = abs(back_disp_y - display_y)
                
                self.assertLessEqual(error_x, 2, 
                                   f"Mouse simulation X error: {display_x} -> {orig_x} -> {back_disp_x}")
                self.assertLessEqual(error_y, 2,
                                   f"Mouse simulation Y error: {display_y} -> {orig_y} -> {back_disp_y}")


if __name__ == '__main__':
    unittest.main(verbosity=2)