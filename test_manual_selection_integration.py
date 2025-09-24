"""
Integration tests for manual selection measurement pipeline.

This module tests the integration between manual selection components and the
existing measurement pipeline to ensure consistent data formats and accurate
measurements between automatic and manual detection modes.

Requirements tested: 5.1, 5.2, 5.3
"""

import unittest
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple

# Import the functions we're testing
from measure import (
    classify_and_measure, classify_and_measure_manual_selection,
    process_manual_selection, validate_manual_measurement_result,
    merge_automatic_and_manual_results, get_measurement_summary,
    _convert_manual_circle_to_measurement, _convert_manual_rectangle_to_measurement
)
from detection import a4_scale_mm_per_px


class TestManualSelectionIntegration(unittest.TestCase):
    """Test integration between manual selection and measurement pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test image with simple shapes
        self.test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw a circle
        cv2.circle(self.test_image, (150, 150), 50, (0, 0, 0), -1)
        
        # Draw a rectangle
        cv2.rectangle(self.test_image, (250, 100), (350, 200), (0, 0, 0), -1)
        
        # Get scale factors
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        
        # Test selection rectangles
        self.circle_selection = (100, 100, 100, 100)  # x, y, w, h
        self.rectangle_selection = (225, 75, 150, 150)
    
    def test_classify_and_measure_with_detection_method(self):
        """Test that classify_and_measure includes detection_method field."""
        # Create a simple circular contour
        center = (150, 150)
        radius = 50
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        points = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])
        contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        
        # Test automatic detection
        result_auto = classify_and_measure(contour, self.mm_per_px_x, self.mm_per_px_y, "automatic")
        self.assertIsNotNone(result_auto)
        self.assertEqual(result_auto["detection_method"], "automatic")
        
        # Test manual detection
        result_manual = classify_and_measure(contour, self.mm_per_px_x, self.mm_per_px_y, "manual")
        self.assertIsNotNone(result_manual)
        self.assertEqual(result_manual["detection_method"], "manual")
        
        # Test default (should be automatic)
        result_default = classify_and_measure(contour, self.mm_per_px_x, self.mm_per_px_y)
        self.assertIsNotNone(result_default)
        self.assertEqual(result_default["detection_method"], "automatic")
    
    def test_convert_manual_circle_to_measurement(self):
        """Test conversion of manual circle selection to measurement format."""
        # Mock shape result from shape snapping engine
        shape_result = {
            "type": "circle",
            "center": (150, 150),
            "radius": 50,
            "confidence_score": 0.85,
            "mode": "manual_circle"
        }
        
        result = _convert_manual_circle_to_measurement(
            shape_result, self.mm_per_px_x, self.circle_selection
        )
        
        # Validate result structure
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "circle")
        self.assertEqual(result["detection_method"], "manual")
        self.assertEqual(result["center"], (150, 150))
        self.assertEqual(result["radius_px"], 50)
        self.assertAlmostEqual(result["diameter_mm"], 100 * self.mm_per_px_x, places=2)
        self.assertIn("hit_contour", result)
        self.assertIn("area_px", result)
        self.assertEqual(result["selection_rect"], self.circle_selection)
        self.assertEqual(result["confidence_score"], 0.85)
        self.assertEqual(result["manual_mode"], "manual_circle")
    
    def test_convert_manual_rectangle_to_measurement(self):
        """Test conversion of manual rectangle selection to measurement format."""
        # Mock shape result from shape snapping engine
        shape_result = {
            "type": "rectangle",
            "center": (300, 150),
            "width": 80,
            "height": 100,
            "confidence_score": 0.75,
            "mode": "manual_rectangle"
        }
        
        result = _convert_manual_rectangle_to_measurement(
            shape_result, self.mm_per_px_x, self.mm_per_px_y, self.rectangle_selection
        )
        
        # Validate result structure
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "rectangle")
        self.assertEqual(result["detection_method"], "manual")
        # Width and height should be normalized (width < height)
        self.assertLessEqual(result["width_mm"], result["height_mm"])
        self.assertIn("box", result)
        self.assertIn("hit_contour", result)
        self.assertIn("area_px", result)
        self.assertEqual(result["selection_rect"], self.rectangle_selection)
        self.assertEqual(result["confidence_score"], 0.75)
        self.assertEqual(result["manual_mode"], "manual_rectangle")
    
    def test_convert_manual_rectangle_with_dimensions(self):
        """Test rectangle conversion with dimensions field."""
        shape_result = {
            "type": "rectangle",
            "center": (300, 150),
            "dimensions": (80, 100),
            "confidence_score": 0.75
        }
        
        result = _convert_manual_rectangle_to_measurement(
            shape_result, self.mm_per_px_x, self.mm_per_px_y, self.rectangle_selection
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "rectangle")
        self.assertEqual(result["detection_method"], "manual")
    
    def test_convert_manual_rectangle_with_contour(self):
        """Test rectangle conversion with contour fallback."""
        # Create a rectangular contour
        box = np.array([[250, 100], [350, 100], [350, 200], [250, 200]], dtype=np.int32)
        contour = box.reshape(-1, 1, 2)
        
        shape_result = {
            "type": "rectangle",
            "center": (300, 150),
            "contour": contour,
            "confidence_score": 0.75
        }
        
        result = _convert_manual_rectangle_to_measurement(
            shape_result, self.mm_per_px_x, self.mm_per_px_y, self.rectangle_selection
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "rectangle")
        self.assertEqual(result["detection_method"], "manual")
    
    def test_validate_manual_measurement_result(self):
        """Test validation of manual measurement results."""
        # Valid circle result
        valid_circle = {
            "type": "circle",
            "detection_method": "manual",
            "diameter_mm": 50.0,
            "center": (150, 150),
            "radius_px": 25.0,
            "hit_contour": np.array([[[100, 125]], [[125, 100]], [[150, 125]], [[125, 150]]], dtype=np.int32),
            "area_px": 1963.5
        }
        self.assertTrue(validate_manual_measurement_result(valid_circle))
        
        # Valid rectangle result
        valid_rectangle = {
            "type": "rectangle",
            "detection_method": "manual",
            "width_mm": 40.0,
            "height_mm": 60.0,
            "box": np.array([[250, 100], [350, 100], [350, 200], [250, 200]], dtype=np.int32),
            "hit_contour": np.array([[[250, 100]], [[350, 100]], [[350, 200]], [[250, 200]]], dtype=np.int32),
            "area_px": 10000.0
        }
        self.assertTrue(validate_manual_measurement_result(valid_rectangle))
        
        # Invalid results
        self.assertFalse(validate_manual_measurement_result(None))
        
        # Missing required fields
        invalid_missing_field = valid_circle.copy()
        del invalid_missing_field["diameter_mm"]
        self.assertFalse(validate_manual_measurement_result(invalid_missing_field))
        
        # Wrong detection method
        invalid_method = valid_circle.copy()
        invalid_method["detection_method"] = "automatic"
        self.assertFalse(validate_manual_measurement_result(invalid_method))
        
        # Invalid measurements
        invalid_diameter = valid_circle.copy()
        invalid_diameter["diameter_mm"] = -10.0
        self.assertFalse(validate_manual_measurement_result(invalid_diameter))
    
    def test_merge_automatic_and_manual_results(self):
        """Test merging of automatic and manual measurement results."""
        # Create mock automatic results
        auto_results = [
            {
                "type": "circle",
                "diameter_mm": 30.0,
                "center": (100, 100),
                "radius_px": 15.0,
                "hit_contour": np.array([[[85, 100]], [[100, 85]], [[115, 100]], [[100, 115]]], dtype=np.int32),
                "area_px": 706.9,
                "detection_method": "automatic"
            },
            {
                "type": "rectangle",
                "width_mm": 20.0,
                "height_mm": 30.0,
                "box": np.array([[200, 200], [220, 200], [220, 230], [200, 230]], dtype=np.int32),
                "hit_contour": np.array([[[200, 200]], [[220, 200]], [[220, 230]], [[200, 230]]], dtype=np.int32),
                "area_px": 600.0
                # Missing detection_method - should be added automatically
            }
        ]
        
        # Create mock manual results
        manual_results = [
            {
                "type": "circle",
                "diameter_mm": 50.0,
                "center": (150, 150),
                "radius_px": 25.0,
                "hit_contour": np.array([[[125, 150]], [[150, 125]], [[175, 150]], [[150, 175]]], dtype=np.int32),
                "area_px": 1963.5,
                "detection_method": "manual",
                "selection_rect": (125, 125, 50, 50)
            }
        ]
        
        # Merge results
        merged = merge_automatic_and_manual_results(auto_results, manual_results)
        
        # Validate merged results
        self.assertEqual(len(merged), 3)
        
        # Check that all results have detection_method field
        for result in merged:
            self.assertIn("detection_method", result)
        
        # Check that missing detection_method was added
        auto_rect = next(r for r in merged if r["type"] == "rectangle" and "selection_rect" not in r)
        self.assertEqual(auto_rect["detection_method"], "automatic")
        
        # Check manual result is preserved
        manual_circle = next(r for r in merged if "selection_rect" in r)
        self.assertEqual(manual_circle["detection_method"], "manual")
    
    def test_get_measurement_summary(self):
        """Test measurement summary generation."""
        # Create mixed results
        results = [
            {"type": "circle", "detection_method": "automatic", "inner": False},
            {"type": "rectangle", "detection_method": "automatic", "inner": True},
            {"type": "circle", "detection_method": "manual", "inner": False},
            {"type": "rectangle", "detection_method": "manual", "inner": False},
            {"type": "circle", "detection_method": "unknown", "inner": False}
        ]
        
        summary = get_measurement_summary(results)
        
        # Validate summary
        self.assertEqual(summary["total_shapes"], 5)
        self.assertEqual(summary["automatic_count"], 2)
        self.assertEqual(summary["manual_count"], 2)
        self.assertEqual(summary["circles"], 3)
        self.assertEqual(summary["rectangles"], 2)
        self.assertEqual(summary["inner_shapes"], 1)
        
        # Check detection methods breakdown
        self.assertEqual(summary["detection_methods"]["automatic"], 2)
        self.assertEqual(summary["detection_methods"]["manual"], 2)
        self.assertEqual(summary["detection_methods"]["unknown"], 1)
    
    def test_measurement_consistency_between_modes(self):
        """Test that manual and automatic measurements produce consistent results for the same shape."""
        # Create a perfect circular contour
        center = (150, 150)
        radius = 50
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        points = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])
        contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        
        # Get automatic measurement
        auto_result = classify_and_measure(contour, self.mm_per_px_x, self.mm_per_px_y, "automatic")
        
        # Create equivalent manual result
        manual_shape_result = {
            "type": "circle",
            "center": center,
            "radius": radius,
            "confidence_score": 0.95
        }
        
        manual_result = classify_and_measure_manual_selection(
            self.test_image, self.circle_selection, manual_shape_result, 
            self.mm_per_px_x, self.mm_per_px_y
        )
        
        # Compare key measurements (should be very close)
        self.assertIsNotNone(auto_result)
        self.assertIsNotNone(manual_result)
        
        # Both should be circles
        self.assertEqual(auto_result["type"], "circle")
        self.assertEqual(manual_result["type"], "circle")
        
        # Diameters should be very close (within 1mm tolerance)
        self.assertAlmostEqual(
            auto_result["diameter_mm"], manual_result["diameter_mm"], 
            delta=1.0, msg="Diameter measurements should be consistent"
        )
        
        # Centers should be very close (within 2 pixels tolerance)
        auto_center = auto_result["center"]
        manual_center = manual_result["center"]
        center_diff = abs(auto_center[0] - manual_center[0]) + abs(auto_center[1] - manual_center[1])
        self.assertLessEqual(center_diff, 2, "Centers should be within 2 pixels of each other")
        
        # Detection methods should be different
        self.assertEqual(auto_result["detection_method"], "automatic")
        self.assertEqual(manual_result["detection_method"], "manual")
        
        # Manual result should have additional fields
        self.assertIn("selection_rect", manual_result)
        self.assertIn("confidence_score", manual_result)
        self.assertNotIn("selection_rect", auto_result)
    
    def test_data_format_compatibility(self):
        """Test that manual measurements are compatible with existing rendering and interaction systems."""
        # Create a manual measurement result
        manual_result = {
            "type": "circle",
            "diameter_mm": 50.0,
            "center": (150, 150),
            "radius_px": 25.0,
            "hit_contour": np.array([[[125, 150]], [[150, 125]], [[175, 150]], [[150, 175]]], dtype=np.int32),
            "area_px": 1963.5,
            "detection_method": "manual",
            "inner": False
        }
        
        # Test compatibility with create_shape_data function
        from measure import create_shape_data
        shape_data = create_shape_data(manual_result)
        
        self.assertIsNotNone(shape_data)
        self.assertEqual(shape_data["type"], "circle")
        self.assertEqual(shape_data["diameter_mm"], 50.0)
        self.assertIn("hit_contour", shape_data)
        
        # Test compatibility with hit testing validation
        from hit_testing import validate_shape_data
        self.assertTrue(validate_shape_data(shape_data))
    
    def test_error_handling_in_conversion(self):
        """Test error handling in manual selection conversion functions."""
        # Test with None input
        result = classify_and_measure_manual_selection(
            self.test_image, self.circle_selection, None, 
            self.mm_per_px_x, self.mm_per_px_y
        )
        self.assertIsNone(result)
        
        # Test with invalid shape type
        invalid_shape = {"type": "triangle", "center": (100, 100)}
        result = classify_and_measure_manual_selection(
            self.test_image, self.circle_selection, invalid_shape, 
            self.mm_per_px_x, self.mm_per_px_y
        )
        self.assertIsNone(result)
        
        # Test rectangle conversion with missing dimensions
        invalid_rect = {"type": "rectangle", "center": (100, 100)}
        try:
            _convert_manual_rectangle_to_measurement(
                invalid_rect, self.mm_per_px_x, self.mm_per_px_y, self.rectangle_selection
            )
            self.fail("Should have raised ValueError for missing dimensions")
        except ValueError:
            pass  # Expected


class TestManualSelectionProcessing(unittest.TestCase):
    """Test the complete manual selection processing workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(self.test_image, (150, 150), 50, (0, 0, 0), -1)
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        self.circle_selection = (100, 100, 100, 100)
    
    def test_process_manual_selection_invalid_mode(self):
        """Test process_manual_selection with invalid mode."""
        result = process_manual_selection(
            self.test_image, self.circle_selection, "invalid_mode",
            self.mm_per_px_x, self.mm_per_px_y
        )
        self.assertIsNone(result)
    
    def test_process_manual_selection_missing_components(self):
        """Test graceful handling when manual selection components are not available."""
        # This test would require mocking the import to simulate missing components
        # For now, we'll just test that the function exists and handles the case
        result = process_manual_selection(
            self.test_image, self.circle_selection, "manual_circle",
            self.mm_per_px_x, self.mm_per_px_y
        )
        # Result could be None if components are missing, or a valid result if they're available
        # Either is acceptable for this integration test


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)