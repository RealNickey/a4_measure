"""
Integration tests for ShapeSnappingEngine with real image processing

Tests the complete workflow of shape detection and snapping using actual
image processing operations and contour detection.
"""

import unittest
import numpy as np
import cv2
import math

from shape_snapping_engine import ShapeSnappingEngine
from enhanced_contour_analyzer import EnhancedContourAnalyzer
from selection_mode import SelectionMode


class TestShapeSnappingIntegration(unittest.TestCase):
    """Integration tests for shape snapping with real image processing."""
    
    def setUp(self):
        """Set up test fixtures with real analyzer."""
        self.analyzer = EnhancedContourAnalyzer()
        self.engine = ShapeSnappingEngine(self.analyzer)
        
        # Create test images with actual shapes
        self.circle_image = self._create_circle_test_image()
        self.rectangle_image = self._create_rectangle_test_image()
        self.mixed_image = self._create_mixed_shapes_image()
    
    def _create_circle_test_image(self):
        """Create test image with a clear circle."""
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Draw a black circle
        cv2.circle(image, (200, 200), 80, (0, 0, 0), -1)
        
        # Add some noise circles
        cv2.circle(image, (100, 100), 20, (0, 0, 0), -1)
        cv2.circle(image, (300, 300), 15, (0, 0, 0), -1)
        
        return image
    
    def _create_rectangle_test_image(self):
        """Create test image with a clear rectangle."""
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Draw a black rectangle
        cv2.rectangle(image, (150, 150), (250, 250), (0, 0, 0), -1)
        
        # Add some noise rectangles
        cv2.rectangle(image, (50, 50), (80, 80), (0, 0, 0), -1)
        cv2.rectangle(image, (320, 320), (350, 350), (0, 0, 0), -1)
        
        return image
    
    def _create_mixed_shapes_image(self):
        """Create test image with both circles and rectangles."""
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Draw shapes
        cv2.circle(image, (150, 150), 60, (0, 0, 0), -1)  # Large circle
        cv2.rectangle(image, (220, 220), (320, 320), (0, 0, 0), -1)  # Large rectangle
        cv2.circle(image, (300, 100), 25, (0, 0, 0), -1)  # Small circle
        
        return image
    
    def test_circle_detection_integration(self):
        """Test complete circle detection workflow."""
        selection_rect = (120, 120, 160, 160)  # Around the large circle
        
        result = self.engine.snap_to_shape(
            self.circle_image, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        self.assertIsNotNone(result, "Should detect circle in selection")
        self.assertEqual(result["type"], "circle")
        self.assertEqual(result["detection_method"], "manual")
        
        # Check that detected circle is reasonable
        center = result["center"]
        radius = result["radius"]
        
        # Center should be roughly in the middle of our drawn circle
        self.assertAlmostEqual(center[0], 200, delta=20)
        self.assertAlmostEqual(center[1], 200, delta=20)
        
        # Radius should be roughly 80 pixels
        self.assertAlmostEqual(radius, 80, delta=15)
        
        # Confidence should be high for a clear circle
        self.assertGreater(result["confidence_score"], 0.7)
    
    def test_rectangle_detection_integration(self):
        """Test complete rectangle detection workflow."""
        selection_rect = (130, 130, 140, 140)  # Around the large rectangle
        
        result = self.engine.snap_to_shape(
            self.rectangle_image, selection_rect, SelectionMode.MANUAL_RECTANGLE
        )
        
        self.assertIsNotNone(result, "Should detect rectangle in selection")
        self.assertEqual(result["type"], "rectangle")
        self.assertEqual(result["detection_method"], "manual")
        
        # Check that detected rectangle is reasonable
        center = result["center"]
        width = result["width"]
        height = result["height"]
        
        # Center should be roughly in the middle of our drawn rectangle
        self.assertAlmostEqual(center[0], 200, delta=20)
        self.assertAlmostEqual(center[1], 200, delta=20)
        
        # Dimensions should be roughly 100x100 pixels
        self.assertAlmostEqual(width, 100, delta=20)
        self.assertAlmostEqual(height, 100, delta=20)
        
        # Confidence should be high for a clear rectangle
        self.assertGreater(result["confidence_score"], 0.7)
    
    def test_shape_prioritization_in_mixed_image(self):
        """Test that the engine prioritizes the correct shape type."""
        # Selection around area with both circle and rectangle
        selection_rect = (100, 100, 200, 200)
        
        # Test circle mode - should prefer the circle
        circle_result = self.engine.snap_to_shape(
            self.mixed_image, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        self.assertIsNotNone(circle_result)
        self.assertEqual(circle_result["type"], "circle")
        
        # Test rectangle mode - should prefer the rectangle
        rect_selection = (200, 200, 140, 140)  # Around rectangle area
        rectangle_result = self.engine.snap_to_shape(
            self.mixed_image, rect_selection, SelectionMode.MANUAL_RECTANGLE
        )
        
        self.assertIsNotNone(rectangle_result)
        self.assertEqual(rectangle_result["type"], "rectangle")
    
    def test_size_based_selection(self):
        """Test that larger shapes are preferred when multiple candidates exist."""
        # Selection that includes both large and small circles
        selection_rect = (50, 50, 300, 300)
        
        result = self.engine.snap_to_shape(
            self.circle_image, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        self.assertIsNotNone(result)
        
        # Should select the larger circle (radius ~80) not the smaller ones (~20, ~15)
        radius = result["radius"]
        self.assertGreater(radius, 50, "Should select the larger circle")
    
    def test_position_based_selection(self):
        """Test that centered shapes are preferred."""
        # Create image with off-center and centered circles
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Off-center circle (same size)
        cv2.circle(image, (120, 120), 40, (0, 0, 0), -1)
        
        # Centered circle (same size)
        cv2.circle(image, (200, 200), 40, (0, 0, 0), -1)
        
        # Selection centered on (200, 200)
        selection_rect = (150, 150, 100, 100)
        
        result = self.engine.snap_to_shape(
            image, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        self.assertIsNotNone(result)
        
        # Should select the centered circle
        center = result["center"]
        distance_to_selection_center = math.sqrt(
            (center[0] - 200)**2 + (center[1] - 200)**2
        )
        
        self.assertLess(distance_to_selection_center, 30, 
                       "Should select the more centered circle")
    
    def test_no_shapes_in_selection(self):
        """Test behavior when no suitable shapes are in selection."""
        # Selection in empty area
        selection_rect = (10, 10, 50, 50)
        
        result = self.engine.snap_to_shape(
            self.circle_image, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        self.assertIsNone(result, "Should return None when no shapes found")
    
    def test_shape_quality_filtering(self):
        """Test that low-quality shapes are filtered out."""
        # Create image with very irregular shape
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw an irregular blob that's not circular or rectangular
        points = np.array([
            [180, 150], [220, 160], [250, 180], [240, 220], 
            [200, 240], [160, 230], [150, 190], [170, 160]
        ], np.int32)
        cv2.fillPoly(image, [points], (0, 0, 0))
        
        selection_rect = (140, 140, 120, 120)
        
        # Should not detect this as a good circle
        circle_result = self.engine.snap_to_shape(
            image, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        # Might return None or a result with lower confidence than a perfect circle
        if circle_result is not None:
            self.assertLess(circle_result["confidence_score"], 0.95,
                           "Irregular shape should have lower confidence than perfect circle")
    
    def test_edge_case_tiny_selection(self):
        """Test behavior with very small selection."""
        tiny_selection = (200, 200, 10, 10)
        
        result = self.engine.snap_to_shape(
            self.circle_image, tiny_selection, SelectionMode.MANUAL_CIRCLE
        )
        
        # Should handle gracefully (either None or valid result)
        if result is not None:
            self.assertTrue(self.engine.validate_shape_result(result))
    
    def test_edge_case_huge_selection(self):
        """Test behavior with selection covering entire image."""
        huge_selection = (0, 0, 400, 400)
        
        result = self.engine.snap_to_shape(
            self.circle_image, huge_selection, SelectionMode.MANUAL_CIRCLE
        )
        
        # Should still work and select the best circle
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "circle")
        
        # Should select the largest circle
        radius = result["radius"]
        self.assertGreater(radius, 50)
    
    def test_result_validation_integration(self):
        """Test that all results pass validation."""
        test_cases = [
            (self.circle_image, (120, 120, 160, 160), SelectionMode.MANUAL_CIRCLE),
            (self.rectangle_image, (130, 130, 140, 140), SelectionMode.MANUAL_RECTANGLE),
            (self.mixed_image, (100, 100, 200, 200), SelectionMode.MANUAL_CIRCLE),
            (self.mixed_image, (200, 200, 140, 140), SelectionMode.MANUAL_RECTANGLE),
        ]
        
        for image, selection_rect, mode in test_cases:
            with self.subTest(mode=mode.value):
                result = self.engine.snap_to_shape(image, selection_rect, mode)
                
                if result is not None:
                    self.assertTrue(
                        self.engine.validate_shape_result(result),
                        f"Result should pass validation for mode {mode.value}"
                    )
                    
                    # Check that all expected fields are present
                    required_fields = [
                        "type", "detection_method", "center", "dimensions",
                        "area", "confidence_score", "total_score"
                    ]
                    
                    for field in required_fields:
                        self.assertIn(field, result, f"Missing field: {field}")
    
    def test_performance_reasonable(self):
        """Test that shape detection completes in reasonable time."""
        import time
        
        selection_rect = (120, 120, 160, 160)
        
        start_time = time.time()
        result = self.engine.snap_to_shape(
            self.circle_image, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within 1 second for a simple case
        self.assertLess(processing_time, 1.0, 
                       f"Processing took {processing_time:.3f}s, should be faster")
        
        self.assertIsNotNone(result, "Should still produce valid result")


if __name__ == '__main__':
    unittest.main()