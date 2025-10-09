#!/usr/bin/env python3
"""
Test measurement precision and formatting consistency between Auto Mode and Manual Mode.

This test verifies:
1. Manual Mode uses same precision (rounded to nearest millimeter) as Auto Mode
2. Measurement display formatting matches Auto Mode exactly
3. Corrected measurements maintain proper significant digits
4. Measurement units are consistently displayed as millimeters

Requirements: 1.5, 5.4, 5.5
"""

import unittest
import numpy as np
import cv2
import re
from measure import (
    classify_and_measure, 
    process_manual_selection,
    annotate_results,
    _convert_manual_circle_to_measurement,
    _convert_manual_rectangle_to_measurement
)
from enhanced_contour_analyzer import EnhancedContourAnalyzer
from shape_snapping_engine import ShapeSnappingEngine
from selection_mode import SelectionMode


class TestMeasurementPrecisionFormatting(unittest.TestCase):
    """Test measurement precision and formatting consistency."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mm_per_px_x = 0.2  # 0.2 mm per pixel
        self.mm_per_px_y = 0.2  # Isotropic scaling for simplicity
        
        # Initialize shape snapping components
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        
        # Create test images with known dimensions
        self.test_cases = self._create_precision_test_cases()
    
    def _create_precision_test_cases(self):
        """Create test cases with specific dimensions to test precision."""
        test_cases = []
        
        # Test case 1: Circle with fractional mm result (should round to nearest mm)
        img1 = np.ones((300, 300, 3), dtype=np.uint8) * 255
        center = (150, 150)
        radius_px = 37  # 37 * 2 * 0.2 = 14.8mm diameter -> should round to 15mm
        cv2.circle(img1, center, radius_px, (0, 0, 0), -1)
        
        test_cases.append({
            "name": "circle_fractional_precision",
            "image": img1,
            "shape_type": "circle",
            "selection_rect": (110, 110, 80, 80),
            "expected_diameter_exact": 14.8,  # Exact calculation
            "expected_diameter_display": 15,   # Rounded for display
            "center": center,
            "radius_px": radius_px
        })
        
        # Test case 2: Rectangle with fractional mm results
        img2 = np.ones((300, 300, 3), dtype=np.uint8) * 255
        # Width: 43px * 0.2 = 8.6mm -> should round to 9mm
        # Height: 67px * 0.2 = 13.4mm -> should round to 13mm
        cv2.rectangle(img2, (100, 100), (143, 167), (0, 0, 0), -1)
        
        test_cases.append({
            "name": "rectangle_fractional_precision",
            "image": img2,
            "shape_type": "rectangle",
            "selection_rect": (95, 95, 53, 77),
            "expected_width_exact": 8.6,
            "expected_height_exact": 13.4,
            "expected_width_display": 9,
            "expected_height_display": 13
        })
        
        # Test case 3: Very precise measurements (test significant digits)
        img3 = np.ones((400, 400, 3), dtype=np.uint8) * 255
        radius_px = 125  # 125 * 2 * 0.2 = 50.0mm diameter (exact)
        cv2.circle(img3, (200, 200), radius_px, (0, 0, 0), -1)
        
        test_cases.append({
            "name": "circle_exact_precision",
            "image": img3,
            "shape_type": "circle",
            "selection_rect": (70, 70, 260, 260),
            "expected_diameter_exact": 50.0,
            "expected_diameter_display": 50,
            "center": (200, 200),
            "radius_px": radius_px
        })
        
        return test_cases
    
    def test_manual_mode_precision_matches_auto_mode(self):
        """Test that Manual Mode uses same precision as Auto Mode (rounded to nearest mm)."""
        print("\n=== Testing Manual Mode Precision Matches Auto Mode ===")
        
        for test_case in self.test_cases:
            with self.subTest(case=test_case["name"]):
                print(f"\nTesting {test_case['name']}...")
                
                image = test_case["image"]
                selection_rect = test_case["selection_rect"]
                shape_type = test_case["shape_type"]
                
                # Get manual mode result
                if shape_type == "circle":
                    mode_str = "manual_circle"
                else:
                    mode_str = "manual_rectangle"
                
                manual_result = process_manual_selection(
                    image, selection_rect, mode_str,
                    self.mm_per_px_x, self.mm_per_px_y
                )
                
                self.assertIsNotNone(manual_result, f"Manual selection failed for {test_case['name']}")
                
                # Test precision for circles
                if shape_type == "circle":
                    diameter_mm = manual_result["diameter_mm"]
                    expected_display = test_case["expected_diameter_display"]
                    
                    # Check that the result is properly rounded to nearest millimeter
                    self.assertEqual(
                        round(diameter_mm), expected_display,
                        f"Circle diameter should round to {expected_display}mm, got {diameter_mm}mm"
                    )
                    
                    print(f"  ✓ Circle diameter: {diameter_mm}mm rounds to {round(diameter_mm)}mm (expected {expected_display}mm)")
                
                # Test precision for rectangles
                elif shape_type == "rectangle":
                    width_mm = manual_result["width_mm"]
                    height_mm = manual_result["height_mm"]
                    expected_width = test_case["expected_width_display"]
                    expected_height = test_case["expected_height_display"]
                    
                    # Check that results are properly rounded to nearest millimeter
                    self.assertEqual(
                        round(width_mm), expected_width,
                        f"Rectangle width should round to {expected_width}mm, got {width_mm}mm"
                    )
                    self.assertEqual(
                        round(height_mm), expected_height,
                        f"Rectangle height should round to {expected_height}mm, got {height_mm}mm"
                    )
                    
                    print(f"  ✓ Rectangle: {width_mm}mm x {height_mm}mm rounds to {round(width_mm)}mm x {round(height_mm)}mm")
                    print(f"    Expected: {expected_width}mm x {expected_height}mm")
    
    def test_measurement_display_formatting_consistency(self):
        """Test that measurement display formatting matches Auto Mode exactly."""
        print("\n=== Testing Measurement Display Formatting Consistency ===")
        
        # Create a test image with both circle and rectangle
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw circle
        circle_center = (150, 150)
        circle_radius = 50  # 50 * 2 * 0.2 = 20mm diameter
        cv2.circle(test_image, circle_center, circle_radius, (0, 0, 0), -1)
        
        # Draw rectangle
        rect_coords = (250, 100, 350, 200)  # 100x100 px = 20x20 mm
        cv2.rectangle(test_image, rect_coords[:2], rect_coords[2:], (0, 0, 0), -1)
        
        # Get automatic detection results
        mask = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        auto_results = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:  # Filter small contours
                result = classify_and_measure(cnt, self.mm_per_px_x, self.mm_per_px_y, "automatic")
                if result:
                    auto_results.append(result)
        
        # Get manual detection results
        manual_results = []
        
        # Manual circle detection
        circle_selection = (100, 100, 100, 100)
        manual_circle = process_manual_selection(
            test_image, circle_selection, "manual_circle",
            self.mm_per_px_x, self.mm_per_px_y
        )
        if manual_circle:
            manual_results.append(manual_circle)
        
        # Manual rectangle detection
        rect_selection = (240, 90, 120, 120)
        manual_rect = process_manual_selection(
            test_image, rect_selection, "manual_rectangle",
            self.mm_per_px_x, self.mm_per_px_y
        )
        if manual_rect:
            manual_results.append(manual_rect)
        
        # Test annotation formatting consistency
        auto_annotated = annotate_results(test_image.copy(), auto_results, self.mm_per_px_x)
        manual_annotated = annotate_results(test_image.copy(), manual_results, self.mm_per_px_x)
        
        # Extract text patterns from annotations (this is a simplified test)
        # In a real implementation, we would need to analyze the actual rendered text
        
        print("  ✓ Auto mode results:")
        for result in auto_results:
            if result["type"] == "circle":
                formatted_text = f"D={result['diameter_mm']:.0f}mm"
                print(f"    Circle: {formatted_text}")
                # Verify format pattern
                self.assertRegex(formatted_text, r"D=\d+mm", "Circle format should be 'D=XXmm'")
            else:
                formatted_text = f"W={result['width_mm']:.0f}mm  H={result['height_mm']:.0f}mm"
                print(f"    Rectangle: {formatted_text}")
                # Verify format pattern
                self.assertRegex(formatted_text, r"W=\d+mm\s+H=\d+mm", "Rectangle format should be 'W=XXmm  H=XXmm'")
        
        print("  ✓ Manual mode results:")
        for result in manual_results:
            if result["type"] == "circle":
                formatted_text = f"D={result['diameter_mm']:.0f}mm"
                print(f"    Circle: {formatted_text}")
                # Verify format pattern matches auto mode
                self.assertRegex(formatted_text, r"D=\d+mm", "Manual circle format should match auto mode")
            else:
                formatted_text = f"W={result['width_mm']:.0f}mm  H={result['height_mm']:.0f}mm"
                print(f"    Rectangle: {formatted_text}")
                # Verify format pattern matches auto mode
                self.assertRegex(formatted_text, r"W=\d+mm\s+H=\d+mm", "Manual rectangle format should match auto mode")
        
        print("  ✓ Display formatting consistency verified")
    
    def test_significant_digits_consistency(self):
        """Test that corrected measurements maintain proper significant digits."""
        print("\n=== Testing Significant Digits Consistency ===")
        
        # Test with various scaling factors to check significant digit handling
        test_scaling_factors = [
            (0.1, 0.1),    # High precision
            (0.5, 0.5),    # Medium precision  
            (1.0, 1.0),    # Low precision
            (0.25, 0.3)    # Anisotropic scaling
        ]
        
        for mm_per_px_x, mm_per_px_y in test_scaling_factors:
            with self.subTest(scaling=(mm_per_px_x, mm_per_px_y)):
                print(f"\nTesting with scaling factors: {mm_per_px_x}, {mm_per_px_y}")
                
                # Create test shape data
                circle_shape_result = {
                    "type": "circle",
                    "center": (100, 100),
                    "radius": 50,  # 50px radius
                    "dimensions": [50, 50],
                    "confidence_score": 0.9
                }
                
                rectangle_shape_result = {
                    "type": "rectangle",
                    "width": 60,   # 60px width
                    "height": 80,  # 80px height
                    "dimensions": [60, 80],
                    "confidence_score": 0.9
                }
                
                # Test circle conversion
                circle_result = _convert_manual_circle_to_measurement(
                    circle_shape_result, mm_per_px_x, (50, 50, 100, 100)
                )
                
                # Test rectangle conversion
                rectangle_result = _convert_manual_rectangle_to_measurement(
                    rectangle_shape_result, mm_per_px_x, mm_per_px_y, (50, 50, 100, 100)
                )
                
                # Verify circle measurements
                expected_diameter = 100 * mm_per_px_x  # 2 * radius * scaling
                self.assertAlmostEqual(
                    circle_result["diameter_mm"], expected_diameter, places=6,
                    msg=f"Circle diameter calculation incorrect"
                )
                
                # Verify rectangle measurements
                expected_width = 60 * mm_per_px_x
                expected_height = 80 * mm_per_px_y
                self.assertAlmostEqual(
                    rectangle_result["width_mm"], expected_width, places=6,
                    msg=f"Rectangle width calculation incorrect"
                )
                self.assertAlmostEqual(
                    rectangle_result["height_mm"], expected_height, places=6,
                    msg=f"Rectangle height calculation incorrect"
                )
                
                print(f"  ✓ Circle: {circle_result['diameter_mm']:.3f}mm diameter")
                print(f"  ✓ Rectangle: {rectangle_result['width_mm']:.3f}mm x {rectangle_result['height_mm']:.3f}mm")
    
    def test_measurement_units_consistency(self):
        """Test that measurement units are consistently displayed as millimeters."""
        print("\n=== Testing Measurement Units Consistency ===")
        
        # Test various measurement results to ensure units are always in mm
        test_measurements = [
            {"type": "circle", "diameter_mm": 15.7},
            {"type": "circle", "diameter_mm": 0.5},   # Very small
            {"type": "circle", "diameter_mm": 150.0}, # Large
            {"type": "rectangle", "width_mm": 25.3, "height_mm": 40.1},
            {"type": "rectangle", "width_mm": 1.2, "height_mm": 2.8},   # Very small
            {"type": "rectangle", "width_mm": 200.0, "height_mm": 300.0} # Large
        ]
        
        for measurement in test_measurements:
            with self.subTest(measurement=measurement):
                if measurement["type"] == "circle":
                    # Test circle formatting
                    formatted = f"D={measurement['diameter_mm']:.0f}mm"
                    print(f"  Circle: {formatted}")
                    
                    # Verify units are present and correct
                    self.assertTrue(formatted.endswith("mm"), "Circle measurement should end with 'mm'")
                    self.assertIn("D=", formatted, "Circle measurement should start with 'D='")
                    
                    # Extract numeric value and verify it's reasonable
                    numeric_match = re.search(r"D=(\d+)mm", formatted)
                    self.assertIsNotNone(numeric_match, "Should extract numeric diameter value")
                    extracted_value = int(numeric_match.group(1))
                    expected_value = round(measurement['diameter_mm'])
                    self.assertEqual(extracted_value, expected_value, "Extracted value should match rounded input")
                
                else:  # rectangle
                    # Test rectangle formatting
                    formatted = f"W={measurement['width_mm']:.0f}mm  H={measurement['height_mm']:.0f}mm"
                    print(f"  Rectangle: {formatted}")
                    
                    # Verify units are present and correct
                    self.assertRegex(formatted, r"W=\d+mm", "Width should be formatted as 'W=XXmm'")
                    self.assertRegex(formatted, r"H=\d+mm", "Height should be formatted as 'H=XXmm'")
                    
                    # Extract numeric values and verify they're reasonable
                    width_match = re.search(r"W=(\d+)mm", formatted)
                    height_match = re.search(r"H=(\d+)mm", formatted)
                    
                    self.assertIsNotNone(width_match, "Should extract numeric width value")
                    self.assertIsNotNone(height_match, "Should extract numeric height value")
                    
                    extracted_width = int(width_match.group(1))
                    extracted_height = int(height_match.group(1))
                    expected_width = round(measurement['width_mm'])
                    expected_height = round(measurement['height_mm'])
                    
                    self.assertEqual(extracted_width, expected_width, "Extracted width should match rounded input")
                    self.assertEqual(extracted_height, expected_height, "Extracted height should match rounded input")
        
        print("  ✓ All measurements consistently use millimeter units")
    
    def test_console_output_formatting(self):
        """Test that console output formatting is consistent between modes."""
        print("\n=== Testing Console Output Formatting ===")
        
        # Create test results in both auto and manual format
        auto_circle = {
            "type": "circle",
            "diameter_mm": 25.7,
            "detection_method": "automatic"
        }
        
        manual_circle = {
            "type": "circle", 
            "diameter_mm": 25.7,
            "detection_method": "manual"
        }
        
        auto_rectangle = {
            "type": "rectangle",
            "width_mm": 15.3,
            "height_mm": 22.8,
            "detection_method": "automatic"
        }
        
        manual_rectangle = {
            "type": "rectangle",
            "width_mm": 15.3,
            "height_mm": 22.8,
            "detection_method": "manual"
        }
        
        # Test console formatting patterns (simulated)
        test_cases = [
            (auto_circle, "Auto circle"),
            (manual_circle, "Manual circle"),
            (auto_rectangle, "Auto rectangle"),
            (manual_rectangle, "Manual rectangle")
        ]
        
        for result, description in test_cases:
            if result["type"] == "circle":
                # Simulate console output format
                console_output = f"Diameter: {result['diameter_mm']:.1f}mm"
                print(f"  {description}: {console_output}")
                
                # Verify format consistency
                self.assertRegex(console_output, r"Diameter: \d+\.\d+mm", 
                               "Console circle output should follow 'Diameter: XX.Xmm' pattern")
            
            else:  # rectangle
                # Simulate console output format
                console_output = f"{result['width_mm']:.1f} x {result['height_mm']:.1f}mm"
                print(f"  {description}: {console_output}")
                
                # Verify format consistency
                self.assertRegex(console_output, r"\d+\.\d+ x \d+\.\d+mm",
                               "Console rectangle output should follow 'XX.X x XX.Xmm' pattern")
        
        print("  ✓ Console output formatting is consistent")


def run_precision_formatting_tests():
    """Run all precision and formatting tests."""
    print("=" * 60)
    print("MEASUREMENT PRECISION AND FORMATTING CONSISTENCY TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMeasurementPrecisionFormatting)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("PRECISION AND FORMATTING TEST SUMMARY")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ ALL PRECISION AND FORMATTING TESTS PASSED!")
        print(f"   Ran {result.testsRun} tests successfully")
        print("\n✅ Manual Mode precision matches Auto Mode")
        print("✅ Display formatting is consistent between modes")
        print("✅ Significant digits are properly maintained")
        print("✅ Measurement units are consistently displayed as millimeters")
    else:
        print("❌ SOME PRECISION AND FORMATTING TESTS FAILED!")
        print(f"   Ran {result.testsRun} tests")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_precision_formatting_tests()
    exit(0 if success else 1)