#!/usr/bin/env python3
"""
Unit tests for dimension conversion functions in measure.py.

This test suite covers:
- _convert_manual_circle_to_measurement() with known scaling factors
- _convert_manual_rectangle_to_measurement() with axis-specific scaling
- Scaling factor validation and error handling
- Measurement result data structure consistency

Requirements: 1.3, 2.4, 4.5
"""

import numpy as np
import sys
import traceback
import unittest
from typing import Dict, Any

# Import the functions we're testing
from measure import (
    _convert_manual_circle_to_measurement,
    _convert_manual_rectangle_to_measurement
)


class TestCircleDimensionConversion(unittest.TestCase):
    """Test suite for manual circle dimension conversion."""
    
    def setUp(self):
        """Set up common test data."""
        self.basic_circle_shape = {
            "type": "circle",
            "center": (100, 100),
            "radius": 50.0,
            "confidence_score": 0.95,
            "mode": "manual_circle"
        }
        self.selection_rect = (50, 50, 100, 100)
    
    def test_basic_circle_conversion(self):
        """Test circle pixel-to-mm conversion with known scaling factors."""
        mm_per_px_x = 0.2  # 0.2 mm per pixel
        
        result = _convert_manual_circle_to_measurement(
            self.basic_circle_shape, mm_per_px_x, self.selection_rect
        )
        
        # Verify calculations
        expected_diameter_px = 2.0 * 50.0  # 100 pixels
        expected_diameter_mm = expected_diameter_px * mm_per_px_x  # 20.0 mm
        expected_area_px = np.pi * (50.0 ** 2)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "circle")
        self.assertEqual(result["diameter_mm"], expected_diameter_mm)
        self.assertEqual(result["radius_px"], 50.0)
        self.assertEqual(result["center"], (100, 100))
        self.assertEqual(result["detection_method"], "manual")
        self.assertAlmostEqual(result["area_px"], expected_area_px, places=2)
    
    def test_different_scaling_factors(self):
        """Test circle conversion with different scaling factors."""
        test_cases = [
            (0.1, 10.0),   # 0.1 mm/px → 10.0 mm diameter
            (0.5, 50.0),   # 0.5 mm/px → 50.0 mm diameter
            (1.0, 100.0),  # 1.0 mm/px → 100.0 mm diameter
        ]
        
        for mm_per_px_x, expected_diameter_mm in test_cases:
            with self.subTest(scaling_factor=mm_per_px_x):
                result = _convert_manual_circle_to_measurement(
                    self.basic_circle_shape, mm_per_px_x, self.selection_rect
                )
                self.assertEqual(result["diameter_mm"], expected_diameter_mm)
                # Radius should remain in pixels for rendering
                self.assertEqual(result["radius_px"], 50.0)
    
    def test_circle_scaling_factor_validation(self):
        """Test scaling factor validation and error handling for circles."""
        # Test None scaling factor
        with self.assertRaises(ValueError) as cm:
            _convert_manual_circle_to_measurement(
                self.basic_circle_shape, None, self.selection_rect
            )
        self.assertIn("cannot be None", str(cm.exception))
        
        # Test zero scaling factor
        with self.assertRaises(ValueError) as cm:
            _convert_manual_circle_to_measurement(
                self.basic_circle_shape, 0.0, self.selection_rect
            )
        self.assertIn("must be positive", str(cm.exception))
        
        # Test negative scaling factor
        with self.assertRaises(ValueError) as cm:
            _convert_manual_circle_to_measurement(
                self.basic_circle_shape, -0.1, self.selection_rect
            )
        self.assertIn("must be positive", str(cm.exception))
        
        # Test out of range scaling factors
        with self.assertRaises(ValueError) as cm:
            _convert_manual_circle_to_measurement(
                self.basic_circle_shape, 0.001, self.selection_rect
            )
        self.assertIn("out of reasonable range", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            _convert_manual_circle_to_measurement(
                self.basic_circle_shape, 200.0, self.selection_rect
            )
        self.assertIn("out of reasonable range", str(cm.exception))
    
    def test_circle_data_structure_consistency(self):
        """Test that circle measurement result data structure is consistent."""
        mm_per_px_x = 0.15
        result = _convert_manual_circle_to_measurement(
            self.basic_circle_shape, mm_per_px_x, self.selection_rect
        )
        
        # Check all required fields are present
        required_fields = [
            "type", "diameter_mm", "center", "radius_px", "hit_contour",
            "area_px", "inner", "detection_method", "selection_rect",
            "confidence_score", "manual_mode"
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
        
        # Check field types and values
        self.assertIsInstance(result["diameter_mm"], (int, float))
        self.assertIsInstance(result["radius_px"], (int, float))
        self.assertIsInstance(result["center"], tuple)
        self.assertIsInstance(result["hit_contour"], np.ndarray)
        self.assertIsInstance(result["area_px"], (int, float))
        self.assertEqual(result["inner"], False)
        self.assertEqual(result["detection_method"], "manual")
        
        # Verify hit contour format (should be 3D array for OpenCV compatibility)
        hit_contour = result["hit_contour"]
        self.assertEqual(len(hit_contour.shape), 3)
        self.assertEqual(hit_contour.shape[1], 1)
        self.assertEqual(hit_contour.shape[2], 2)


class TestRectangleDimensionConversion(unittest.TestCase):
    """Test suite for manual rectangle dimension conversion."""
    
    def setUp(self):
        """Set up common test data."""
        self.basic_rectangle_shape = {
            "type": "rectangle",
            "width": 100,  # 100 pixels width
            "height": 200,  # 200 pixels height
            "center": (150, 150),
            "confidence_score": 0.9,
            "mode": "manual_rectangle"
        }
        self.selection_rect = (100, 100, 100, 100)
    
    def test_basic_rectangle_conversion_axis_specific(self):
        """Test rectangle pixel-to-mm conversion with axis-specific scaling."""
        mm_per_px_x = 0.5  # 0.5 mm per pixel in X direction
        mm_per_px_y = 0.3  # 0.3 mm per pixel in Y direction
        
        result = _convert_manual_rectangle_to_measurement(
            self.basic_rectangle_shape, mm_per_px_x, mm_per_px_y, self.selection_rect
        )
        
        # Function normalizes width < height, so min(100,200)=100, max(100,200)=200
        expected_width_mm = 100 * mm_per_px_x  # 50 mm (width uses mm_per_px_x)
        expected_height_mm = 200 * mm_per_px_y  # 60 mm (height uses mm_per_px_y)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "rectangle")
        self.assertEqual(result["width_mm"], expected_width_mm)
        self.assertEqual(result["height_mm"], expected_height_mm)
        self.assertEqual(result["detection_method"], "manual")
    
    def test_rectangle_dimensions_format_handling(self):
        """Test that dimensions are extracted correctly from different formats."""
        # Test with dimensions tuple format
        shape_with_dimensions = {
            "type": "rectangle",
            "dimensions": (80, 120),  # width, height
            "center": (150, 150),
            "confidence_score": 0.85
        }
        
        mm_per_px_x = 0.4
        mm_per_px_y = 0.6
        
        result = _convert_manual_rectangle_to_measurement(
            shape_with_dimensions, mm_per_px_x, mm_per_px_y, self.selection_rect
        )
        
        # Function normalizes width < height, so min(80,120)=80, max(80,120)=120
        expected_width_mm = 80 * mm_per_px_x  # 32 mm
        expected_height_mm = 120 * mm_per_px_y  # 72 mm
        
        self.assertEqual(result["width_mm"], expected_width_mm)
        self.assertEqual(result["height_mm"], expected_height_mm)
    
    def test_rectangle_scaling_factor_validation(self):
        """Test scaling factor validation and error handling for rectangles."""
        # Test None values
        with self.assertRaises(ValueError) as cm:
            _convert_manual_rectangle_to_measurement(
                self.basic_rectangle_shape, None, 0.5, self.selection_rect
            )
        self.assertIn("mm_per_px_x cannot be None", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            _convert_manual_rectangle_to_measurement(
                self.basic_rectangle_shape, 0.5, None, self.selection_rect
            )
        self.assertIn("mm_per_px_y cannot be None", str(cm.exception))
        
        # Test negative values
        with self.assertRaises(ValueError) as cm:
            _convert_manual_rectangle_to_measurement(
                self.basic_rectangle_shape, -0.5, 0.5, self.selection_rect
            )
        self.assertIn("mm_per_px_x must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            _convert_manual_rectangle_to_measurement(
                self.basic_rectangle_shape, 0.5, -0.3, self.selection_rect
            )
        self.assertIn("mm_per_px_y must be positive", str(cm.exception))
        
        # Test zero values
        with self.assertRaises(ValueError) as cm:
            _convert_manual_rectangle_to_measurement(
                self.basic_rectangle_shape, 0.0, 0.5, self.selection_rect
            )
        self.assertIn("mm_per_px_x must be positive", str(cm.exception))
        
        # Test out of range values
        with self.assertRaises(ValueError) as cm:
            _convert_manual_rectangle_to_measurement(
                self.basic_rectangle_shape, 0.001, 0.5, self.selection_rect
            )
        self.assertIn("out of reasonable range", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            _convert_manual_rectangle_to_measurement(
                self.basic_rectangle_shape, 0.5, 200.0, self.selection_rect
            )
        self.assertIn("out of reasonable range", str(cm.exception))
    
    def test_box_coordinates_remain_pixels(self):
        """Test that box coordinates remain in pixels for rendering purposes."""
        # Create shape result with explicit box
        box_pixels = np.array([
            [100, 100],
            [200, 100],
            [200, 300],
            [100, 300]
        ], dtype=int)
        
        shape_with_box = {
            "type": "rectangle",
            "width": 100,
            "height": 200,
            "box": box_pixels,
            "center": (150, 200),
            "confidence_score": 0.9
        }
        
        mm_per_px_x = 0.5
        mm_per_px_y = 0.3
        
        result = _convert_manual_rectangle_to_measurement(
            shape_with_box, mm_per_px_x, mm_per_px_y, self.selection_rect
        )
        
        # Box should remain in pixels (unchanged)
        np.testing.assert_array_equal(result["box"], box_pixels)
        self.assertTrue(result["box"].dtype in [np.int32, int])
        
        # But measurements should be in mm
        self.assertEqual(result["width_mm"], 100 * mm_per_px_x)  # 50 mm
        self.assertEqual(result["height_mm"], 200 * mm_per_px_y)  # 60 mm
    
    def test_rectangle_data_structure_consistency(self):
        """Test that rectangle measurement result data structure is consistent."""
        mm_per_px_x = 0.4
        mm_per_px_y = 0.6
        
        result = _convert_manual_rectangle_to_measurement(
            self.basic_rectangle_shape, mm_per_px_x, mm_per_px_y, self.selection_rect
        )
        
        # Check all required fields are present
        required_fields = [
            "type", "width_mm", "height_mm", "box", "hit_contour",
            "area_px", "inner", "detection_method", "selection_rect",
            "confidence_score", "manual_mode"
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
        
        # Check field types and values
        self.assertIsInstance(result["width_mm"], (int, float))
        self.assertIsInstance(result["height_mm"], (int, float))
        self.assertIsInstance(result["box"], np.ndarray)
        self.assertIsInstance(result["hit_contour"], np.ndarray)
        self.assertIsInstance(result["area_px"], (int, float))
        self.assertEqual(result["inner"], False)
        self.assertEqual(result["detection_method"], "manual")
        
        # Verify hit contour format
        hit_contour = result["hit_contour"]
        self.assertEqual(len(hit_contour.shape), 3)
        self.assertEqual(hit_contour.shape[1], 1)
        self.assertEqual(hit_contour.shape[2], 2)


class TestMeasurementResultConsistency(unittest.TestCase):
    """Test suite for measurement result data structure consistency."""
    
    def test_circle_vs_rectangle_field_consistency(self):
        """Test that circle and rectangle results have consistent base fields."""
        # Create circle result
        circle_shape = {
            "type": "circle",
            "center": (100, 100),
            "radius": 30.0,
            "confidence_score": 0.9
        }
        
        circle_result = _convert_manual_circle_to_measurement(
            circle_shape, 0.2, (70, 70, 60, 60)
        )
        
        # Create rectangle result
        rectangle_shape = {
            "type": "rectangle",
            "width": 60,
            "height": 80,
            "center": (100, 100),
            "confidence_score": 0.9
        }
        
        rectangle_result = _convert_manual_rectangle_to_measurement(
            rectangle_shape, 0.2, 0.2, (70, 70, 60, 60)
        )
        
        # Common fields that should exist in both
        common_fields = [
            "type", "hit_contour", "area_px", "inner", 
            "detection_method", "selection_rect", "confidence_score"
        ]
        
        for field in common_fields:
            self.assertIn(field, circle_result, f"Circle missing common field: {field}")
            self.assertIn(field, rectangle_result, f"Rectangle missing common field: {field}")
        
        # Verify common field values
        self.assertEqual(circle_result["inner"], False)
        self.assertEqual(rectangle_result["inner"], False)
        self.assertEqual(circle_result["detection_method"], "manual")
        self.assertEqual(rectangle_result["detection_method"], "manual")
    
    def test_measurement_precision_consistency(self):
        """Test that measurements maintain consistent precision."""
        # Test with various scaling factors to check precision handling
        test_cases = [
            (0.123456789, 50.0),  # High precision scaling
            (0.1, 25.5),          # Decimal radius
            (1.0, 33.333),        # Repeating decimal
        ]
        
        for mm_per_px_x, radius in test_cases:
            with self.subTest(scaling=mm_per_px_x, radius=radius):
                circle_shape = {
                    "type": "circle",
                    "center": (100, 100),
                    "radius": radius,
                    "confidence_score": 0.9
                }
                
                result = _convert_manual_circle_to_measurement(
                    circle_shape, mm_per_px_x, (50, 50, 100, 100)
                )
                
                # Verify calculation precision
                expected_diameter_mm = 2.0 * radius * mm_per_px_x
                self.assertAlmostEqual(result["diameter_mm"], expected_diameter_mm, places=10)


def run_all_tests():
    """Run all dimension conversion tests."""
    print("=" * 70)
    print("Unit Tests for Dimension Conversion Functions")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCircleDimensionConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestRectangleDimensionConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestMeasurementResultConsistency))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print("Dimension conversion functions are working correctly:")
        print("- Circle conversion with proper scaling factors ✓")
        print("- Rectangle conversion with axis-specific scaling ✓") 
        print("- Scaling factor validation and error handling ✓")
        print("- Measurement result data structure consistency ✓")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)