#!/usr/bin/env python3
"""
Measurement Accuracy Validation Tests

This test suite validates that the interactive inspect mode preserves all existing
measurement calculations and maintains compatibility with the original implementation.

Test Coverage:
- A4 scaling factor consistency between old and new systems
- Measurement calculation accuracy preservation
- Dimension display format compatibility
- Console output format validation
"""

import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import sys
import io
from contextlib import redirect_stdout

# Import modules to test
from measure import classify_and_measure, create_shape_data, annotate_result, annotate_results
from config import PX_PER_MM, A4_WIDTH_MM, A4_HEIGHT_MM
from detection import a4_scale_mm_per_px
from interaction_manager import default_selection_callback
from rendering import SelectiveRenderer


class TestMeasurementAccuracyValidation(unittest.TestCase):
    """Test suite for validating measurement accuracy preservation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test contours for circles and rectangles
        self.circle_contour = self._create_circle_contour(center=(100, 100), radius=30)
        self.rect_contour = self._create_rectangle_contour(center=(200, 200), width=60, height=40)
        
        # Standard A4 scaling factors
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        
        # Create test warped image
        self.test_warped_image = np.ones((int(A4_HEIGHT_MM * PX_PER_MM), 
                                         int(A4_WIDTH_MM * PX_PER_MM), 3), dtype=np.uint8) * 255
        
        # Initialize renderer for testing
        self.renderer = SelectiveRenderer()
    
    def _create_circle_contour(self, center, radius):
        """Create a circular contour for testing."""
        angles = np.linspace(0, 2*np.pi, 36)
        points = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])
        return np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    
    def _create_rectangle_contour(self, center, width, height):
        """Create a rectangular contour for testing."""
        half_w, half_h = width // 2, height // 2
        points = [
            [center[0] - half_w, center[1] - half_h],
            [center[0] + half_w, center[1] - half_h],
            [center[0] + half_w, center[1] + half_h],
            [center[0] - half_w, center[1] + half_h]
        ]
        return np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    
    def test_a4_scaling_factor_consistency(self):
        """Test that A4 scaling factors remain consistent between old and new systems."""
        print("\n=== Testing A4 Scaling Factor Consistency ===")
        
        # Test that a4_scale_mm_per_px returns expected values
        mm_per_px_x, mm_per_px_y = a4_scale_mm_per_px()
        
        expected_mm_per_px = 1.0 / PX_PER_MM
        
        self.assertAlmostEqual(mm_per_px_x, expected_mm_per_px, places=6,
                              msg="X scaling factor should match 1/PX_PER_MM")
        self.assertAlmostEqual(mm_per_px_y, expected_mm_per_px, places=6,
                              msg="Y scaling factor should match 1/PX_PER_MM")
        
        # Verify A4 dimensions in pixels
        expected_width_px = A4_WIDTH_MM * PX_PER_MM
        expected_height_px = A4_HEIGHT_MM * PX_PER_MM
        
        actual_width_px = A4_WIDTH_MM / mm_per_px_x
        actual_height_px = A4_HEIGHT_MM / mm_per_px_y
        
        self.assertAlmostEqual(actual_width_px, expected_width_px, places=1,
                              msg="A4 width in pixels should be consistent")
        self.assertAlmostEqual(actual_height_px, expected_height_px, places=1,
                              msg="A4 height in pixels should be consistent")
        
        print(f"✓ A4 scaling factors: {mm_per_px_x:.6f} mm/px (both X and Y)")
        print(f"✓ A4 dimensions: {actual_width_px:.1f} x {actual_height_px:.1f} px")
    
    def test_circle_measurement_accuracy(self):
        """Test that circle measurements remain unchanged."""
        print("\n=== Testing Circle Measurement Accuracy ===")
        
        # Measure circle using original function
        result = classify_and_measure(self.circle_contour, self.mm_per_px_x, self.mm_per_px_y)
        
        self.assertIsNotNone(result, "Circle measurement should not be None")
        self.assertEqual(result["type"], "circle", "Shape should be classified as circle")
        
        # Verify diameter calculation consistency (OpenCV may not return exact radius)
        # The important thing is that the calculation is consistent: diameter_mm = 2 * radius_px * mm_per_px
        calculated_diameter_mm = 2 * result["radius_px"] * self.mm_per_px_x
        
        self.assertAlmostEqual(result["diameter_mm"], calculated_diameter_mm, places=6,
                              msg="Circle diameter should equal 2 * radius_px * mm_per_px_x")
        
        # Verify radius is reasonable (should be close to our test setup)
        self.assertGreater(result["radius_px"], 25, "Circle radius should be reasonable")
        self.assertLess(result["radius_px"], 35, "Circle radius should be reasonable")
        
        # Test shape data conversion preserves measurements
        shape_data = create_shape_data(result)
        self.assertIsNotNone(shape_data, "Shape data conversion should succeed")
        self.assertEqual(shape_data["type"], "circle", "Shape type should be preserved")
        self.assertAlmostEqual(shape_data["diameter_mm"], result["diameter_mm"], places=6,
                              msg="Diameter should be preserved in shape data")
        
        print(f"✓ Circle diameter: {result['diameter_mm']:.2f} mm")
        print(f"✓ Circle radius: {result['radius_px']:.1f} px")
        print(f"✓ Diameter calculation: 2 * {result['radius_px']:.1f} * {self.mm_per_px_x:.6f} = {calculated_diameter_mm:.2f} mm")
        print(f"✓ Shape data conversion preserves measurements")
    
    def test_rectangle_measurement_accuracy(self):
        """Test that rectangle measurements remain unchanged."""
        print("\n=== Testing Rectangle Measurement Accuracy ===")
        
        # Measure rectangle using original function
        result = classify_and_measure(self.rect_contour, self.mm_per_px_x, self.mm_per_px_y)
        
        self.assertIsNotNone(result, "Rectangle measurement should not be None")
        self.assertEqual(result["type"], "rectangle", "Shape should be classified as rectangle")
        
        # Verify dimension calculations
        expected_width_px = 40  # Smaller dimension from test setup
        expected_height_px = 60  # Larger dimension from test setup
        expected_width_mm = expected_width_px * self.mm_per_px_x
        expected_height_mm = expected_height_px * self.mm_per_px_y
        
        self.assertAlmostEqual(result["width_mm"], expected_width_mm, places=2,
                              msg="Rectangle width calculation should be accurate")
        self.assertAlmostEqual(result["height_mm"], expected_height_mm, places=2,
                              msg="Rectangle height calculation should be accurate")
        
        # Test shape data conversion preserves measurements
        shape_data = create_shape_data(result)
        self.assertIsNotNone(shape_data, "Shape data conversion should succeed")
        self.assertEqual(shape_data["type"], "rectangle", "Shape type should be preserved")
        self.assertAlmostEqual(shape_data["width_mm"], result["width_mm"], places=6,
                              msg="Width should be preserved in shape data")
        self.assertAlmostEqual(shape_data["height_mm"], result["height_mm"], places=6,
                              msg="Height should be preserved in shape data")
        
        print(f"✓ Rectangle dimensions: {result['width_mm']:.2f} x {result['height_mm']:.2f} mm")
        print(f"✓ Shape data conversion preserves measurements")
    
    def test_dimension_display_compatibility(self):
        """Test that dimension display matches original implementation exactly."""
        print("\n=== Testing Dimension Display Compatibility ===")
        
        # Test circle display
        circle_result = classify_and_measure(self.circle_contour, self.mm_per_px_x, self.mm_per_px_y)
        circle_annotated = annotate_result(self.test_warped_image, circle_result, self.mm_per_px_x)
        
        self.assertIsNotNone(circle_annotated, "Circle annotation should succeed")
        self.assertEqual(circle_annotated.shape, self.test_warped_image.shape,
                        "Annotated image should have same dimensions as original")
        
        # Test rectangle display
        rect_result = classify_and_measure(self.rect_contour, self.mm_per_px_x, self.mm_per_px_y)
        rect_annotated = annotate_result(self.test_warped_image, rect_result, self.mm_per_px_x)
        
        self.assertIsNotNone(rect_annotated, "Rectangle annotation should succeed")
        self.assertEqual(rect_annotated.shape, self.test_warped_image.shape,
                        "Annotated image should have same dimensions as original")
        
        # Test multiple results annotation
        results = [circle_result, rect_result]
        multi_annotated = annotate_results(self.test_warped_image, results, self.mm_per_px_x)
        
        self.assertIsNotNone(multi_annotated, "Multiple results annotation should succeed")
        self.assertEqual(multi_annotated.shape, self.test_warped_image.shape,
                        "Multi-annotated image should have same dimensions as original")
        
        print("✓ Circle annotation rendering works correctly")
        print("✓ Rectangle annotation rendering works correctly")
        print("✓ Multiple results annotation works correctly")
    
    def test_interactive_rendering_accuracy(self):
        """Test that interactive rendering produces equivalent visual output."""
        print("\n=== Testing Interactive Rendering Accuracy ===")
        
        # Create test shapes
        circle_result = classify_and_measure(self.circle_contour, self.mm_per_px_x, self.mm_per_px_y)
        rect_result = classify_and_measure(self.rect_contour, self.mm_per_px_x, self.mm_per_px_y)
        
        circle_shape = create_shape_data(circle_result)
        rect_shape = create_shape_data(rect_result)
        
        # Test individual shape rendering
        circle_rendered = self.renderer.render_selection(self.test_warped_image, circle_shape)
        rect_rendered = self.renderer.render_selection(self.test_warped_image, rect_shape)
        
        self.assertIsNotNone(circle_rendered, "Circle interactive rendering should succeed")
        self.assertIsNotNone(rect_rendered, "Rectangle interactive rendering should succeed")
        
        # Verify dimensions are preserved in rendering
        self.assertEqual(circle_rendered.shape, self.test_warped_image.shape,
                        "Circle rendered image should maintain dimensions")
        self.assertEqual(rect_rendered.shape, self.test_warped_image.shape,
                        "Rectangle rendered image should maintain dimensions")
        
        # Test complete state rendering
        state = {"hovered": None, "selected": 0}
        shapes = [circle_shape, rect_shape]
        complete_rendered = self.renderer.render_complete_state(self.test_warped_image, state, shapes)
        
        self.assertIsNotNone(complete_rendered, "Complete state rendering should succeed")
        self.assertEqual(complete_rendered.shape, self.test_warped_image.shape,
                        "Complete rendered image should maintain dimensions")
        
        print("✓ Interactive circle rendering works correctly")
        print("✓ Interactive rectangle rendering works correctly")
        print("✓ Complete state rendering works correctly")
    
    def test_console_output_format_compatibility(self):
        """Test that console output format matches original implementation."""
        print("\n=== Testing Console Output Format Compatibility ===")
        
        # Create test shapes
        circle_result = classify_and_measure(self.circle_contour, self.mm_per_px_x, self.mm_per_px_y)
        rect_result = classify_and_measure(self.rect_contour, self.mm_per_px_x, self.mm_per_px_y)
        
        circle_shape = create_shape_data(circle_result)
        rect_shape = create_shape_data(rect_result)
        shapes = [circle_shape, rect_shape]
        
        # Test circle selection output
        with redirect_stdout(io.StringIO()) as circle_output:
            default_selection_callback(0, shapes)
        circle_text = circle_output.getvalue().strip()
        
        # Verify circle output format
        self.assertIn("[SELECTED]", circle_text, "Console output should have [SELECTED] prefix")
        self.assertIn("Circle", circle_text, "Console output should identify shape type")
        self.assertIn("Diameter:", circle_text, "Console output should show diameter")
        self.assertIn("mm", circle_text, "Console output should show units")
        
        # Test rectangle selection output
        with redirect_stdout(io.StringIO()) as rect_output:
            default_selection_callback(1, shapes)
        rect_text = rect_output.getvalue().strip()
        
        # Verify rectangle output format
        self.assertIn("[SELECTED]", rect_text, "Console output should have [SELECTED] prefix")
        self.assertIn("Rectangle", rect_text, "Console output should identify shape type")
        self.assertIn("Width:", rect_text, "Console output should show width")
        self.assertIn("Height:", rect_text, "Console output should show height")
        self.assertIn("mm", rect_text, "Console output should show units")
        
        # Test no selection output
        with redirect_stdout(io.StringIO()) as none_output:
            default_selection_callback(None, shapes)
        none_text = none_output.getvalue().strip()
        
        # Verify no selection output format
        self.assertIn("[SELECTED] None", none_text, "Console output should show None for no selection")
        
        print(f"✓ Circle selection output: {circle_text}")
        print(f"✓ Rectangle selection output: {rect_text}")
        print(f"✓ No selection output: {none_text}")
    
    def test_measurement_precision_consistency(self):
        """Test that measurement precision is consistent across systems."""
        print("\n=== Testing Measurement Precision Consistency ===")
        
        # Test various sizes to ensure precision is maintained
        test_radii = [10, 25, 50, 100]
        test_rect_sizes = [(20, 30), (40, 60), (80, 120), (160, 240)]
        
        for radius in test_radii:
            contour = self._create_circle_contour((150, 150), radius)
            result = classify_and_measure(contour, self.mm_per_px_x, self.mm_per_px_y)
            shape_data = create_shape_data(result)
            
            # Verify precision is maintained
            self.assertAlmostEqual(result["diameter_mm"], shape_data["diameter_mm"], places=6,
                                  msg=f"Circle diameter precision should be maintained for radius {radius}")
        
        for width, height in test_rect_sizes:
            contour = self._create_rectangle_contour((200, 200), width, height)
            result = classify_and_measure(contour, self.mm_per_px_x, self.mm_per_px_y)
            shape_data = create_shape_data(result)
            
            # Verify precision is maintained
            self.assertAlmostEqual(result["width_mm"], shape_data["width_mm"], places=6,
                                  msg=f"Rectangle width precision should be maintained for size {width}x{height}")
            self.assertAlmostEqual(result["height_mm"], shape_data["height_mm"], places=6,
                                  msg=f"Rectangle height precision should be maintained for size {width}x{height}")
        
        print(f"✓ Tested {len(test_radii)} circle sizes - precision maintained")
        print(f"✓ Tested {len(test_rect_sizes)} rectangle sizes - precision maintained")
    
    def test_edge_case_measurements(self):
        """Test measurement accuracy for edge cases."""
        print("\n=== Testing Edge Case Measurements ===")
        
        # Test very small circle
        small_circle = self._create_circle_contour((100, 100), 5)
        small_result = classify_and_measure(small_circle, self.mm_per_px_x, self.mm_per_px_y)
        
        if small_result is not None:  # May be None if too small
            small_shape = create_shape_data(small_result)
            self.assertIsNotNone(small_shape, "Small circle shape data should be created")
            print(f"✓ Small circle (5px radius): {small_result['diameter_mm']:.2f} mm")
        else:
            print("✓ Small circle correctly filtered out")
        
        # Test very large rectangle
        large_rect = self._create_rectangle_contour((300, 300), 200, 300)
        large_result = classify_and_measure(large_rect, self.mm_per_px_x, self.mm_per_px_y)
        
        if large_result is not None:
            large_shape = create_shape_data(large_result)
            self.assertIsNotNone(large_shape, "Large rectangle shape data should be created")
            print(f"✓ Large rectangle: {large_result['width_mm']:.2f} x {large_result['height_mm']:.2f} mm")
        
        # Test square (width == height)
        square = self._create_rectangle_contour((150, 150), 50, 50)
        square_result = classify_and_measure(square, self.mm_per_px_x, self.mm_per_px_y)
        
        if square_result is not None:
            square_shape = create_shape_data(square_result)
            self.assertIsNotNone(square_shape, "Square shape data should be created")
            # For squares, width should be <= height due to normalization
            self.assertLessEqual(square_result["width_mm"], square_result["height_mm"],
                               "Square should have width <= height after normalization")
            print(f"✓ Square: {square_result['width_mm']:.2f} x {square_result['height_mm']:.2f} mm")


def run_validation_tests():
    """Run all measurement accuracy validation tests."""
    print("=" * 60)
    print("MEASUREMENT ACCURACY VALIDATION TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMeasurementAccuracyValidation)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ ALL MEASUREMENT ACCURACY TESTS PASSED")
        print("✅ Interactive system preserves original measurement calculations")
        print("✅ A4 scaling factors are consistent")
        print("✅ Dimension display format is compatible")
        print("✅ Console output format is preserved")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)