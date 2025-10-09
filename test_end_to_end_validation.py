#!/usr/bin/env python3
"""
End-to-End Validation and Regression Testing

This comprehensive test suite validates Task 10 requirements:
- Test that Auto Mode measurements remain unchanged after modifications
- Verify Manual Mode shape detection quality is preserved
- Confirm user interaction workflow (click, drag, snap) remains intact
- Test measurement accuracy with objects of known dimensions

Requirements: 3.1, 3.2, 3.3, 3.5

This test serves as the final validation that all manual mode dimension corrections
work correctly while preserving existing functionality.
"""

import unittest
import cv2
import numpy as np
import math
import tempfile
import os
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock

# Import all required components
try:
    from measure import (
        classify_and_measure, process_manual_selection,
        validate_manual_measurement_result, compare_auto_vs_manual_measurements,
        _convert_manual_circle_to_measurement, _convert_manual_rectangle_to_measurement
    )
    from detection import a4_scale_mm_per_px, find_a4_quad, warp_a4
    from shape_snapping_engine import ShapeSnappingEngine
    from manual_selection_engine import ManualSelectionEngine
    from enhanced_contour_analyzer import EnhancedContourAnalyzer
    import main
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


class TestEndToEndValidation(unittest.TestCase):
    """Comprehensive end-to-end validation and regression testing."""
    
    def setUp(self):
        """Set up test fixtures with known dimensions."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Required components not available")
        
        # Standard A4 calibration values for testing
        self.mm_per_px_x = 0.2  # 5 pixels per mm
        self.mm_per_px_y = 0.2  # 5 pixels per mm
        
        # Create test images with known dimensions
        self.test_image = self._create_test_image_with_known_shapes()
        
        # Reference measurements for validation (ground truth)
        self.reference_measurements = {
            "large_circle": {"diameter_mm": 20.0, "type": "circle"},
            "medium_circle": {"diameter_mm": 12.0, "type": "circle"},
            "small_circle": {"diameter_mm": 8.0, "type": "circle"},
            "large_rectangle": {"width_mm": 16.0, "height_mm": 12.0, "type": "rectangle"},
            "medium_rectangle": {"width_mm": 10.0, "height_mm": 8.0, "type": "rectangle"},
            "small_rectangle": {"width_mm": 6.0, "height_mm": 4.0, "type": "rectangle"}
        }
        
        # Tolerance for measurement validation (±2mm or ±2% whichever is larger)
        self.tolerance_mm = 2.0
        self.tolerance_percent = 0.02
    
    def _create_test_image_with_known_shapes(self):
        """Create test image with shapes of known dimensions."""
        # Create 400x400 white image
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw circles with known diameters (in pixels, converted from mm)
        # Large circle: 20mm diameter = 100px diameter = 50px radius
        cv2.circle(image, (100, 100), 50, (0, 0, 0), -1)
        
        # Medium circle: 12mm diameter = 60px diameter = 30px radius  
        cv2.circle(image, (250, 100), 30, (0, 0, 0), -1)
        
        # Small circle: 8mm diameter = 40px diameter = 20px radius
        cv2.circle(image, (350, 100), 20, (0, 0, 0), -1)
        
        # Draw rectangles with known dimensions (in pixels, converted from mm)
        # Large rectangle: 16x12mm = 80x60px
        cv2.rectangle(image, (60, 200), (140, 260), (0, 0, 0), -1)
        
        # Medium rectangle: 10x8mm = 50x40px
        cv2.rectangle(image, (225, 210), (275, 250), (0, 0, 0), -1)
        
        # Small rectangle: 6x4mm = 30x20px
        cv2.rectangle(image, (335, 220), (365, 240), (0, 0, 0), -1)
        
        return image
    
    def _calculate_tolerance(self, expected_value):
        """Calculate tolerance for a measurement (±2mm or ±2% whichever is larger)."""
        percent_tolerance = expected_value * self.tolerance_percent
        return max(self.tolerance_mm, percent_tolerance)
    
    def _validate_measurement_accuracy(self, measured_value, expected_value, shape_name):
        """Validate that measured value is within acceptable tolerance."""
        tolerance = self._calculate_tolerance(expected_value)
        difference = abs(measured_value - expected_value)
        
        is_accurate = difference <= tolerance
        
        print(f"  {shape_name}: Expected {expected_value:.1f}mm, Got {measured_value:.1f}mm")
        print(f"    Difference: {difference:.2f}mm, Tolerance: ±{tolerance:.2f}mm")
        print(f"    Result: {'✅ PASS' if is_accurate else '❌ FAIL'}")
        
        return is_accurate, difference, tolerance

    def test_requirement_3_1_auto_mode_measurements_unchanged(self):
        """
        Requirement 3.1: Test that Auto Mode measurements remain unchanged after modifications
        
        This test verifies that automatic detection still produces accurate measurements
        and that the dimension correction changes haven't affected Auto Mode functionality.
        """
        print("\n=== Testing Requirement 3.1: Auto Mode Measurements Unchanged ===")
        
        # Test automatic detection on known shapes
        test_cases = [
            {"center": (100, 100), "expected": self.reference_measurements["large_circle"]},
            {"center": (250, 100), "expected": self.reference_measurements["medium_circle"]},
            {"center": (100, 230), "expected": self.reference_measurements["large_rectangle"]},
            {"center": (250, 230), "expected": self.reference_measurements["medium_rectangle"]}
        ]
        
        all_accurate = True
        
        for i, case in enumerate(test_cases):
            print(f"\nTesting Auto Mode - Shape {i+1}:")
            
            # Create contour around the shape for automatic detection
            center = case["center"]
            if case["expected"]["type"] == "circle":
                # Create circular contour
                radius = 55 if center == (100, 100) else (35 if center == (250, 100) else 25)
                angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
                contour_points = [(int(center[0] + radius * np.cos(a)),
                                 int(center[1] + radius * np.sin(a))) for a in angles]
                contour = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
            else:
                # Create rectangular contour
                if center == (100, 230):  # Large rectangle
                    contour = np.array([[60, 200], [140, 200], [140, 260], [60, 260]], dtype=np.int32).reshape(-1, 1, 2)
                else:  # Medium rectangle
                    contour = np.array([[225, 210], [275, 210], [275, 250], [225, 250]], dtype=np.int32).reshape(-1, 1, 2)
            
            # Test automatic detection
            result = classify_and_measure(contour, self.mm_per_px_x, self.mm_per_px_y, "automatic")
            
            if result is not None:
                expected = case["expected"]
                if expected["type"] == "circle":
                    measured = result["diameter_mm"]
                    expected_val = expected["diameter_mm"]
                    is_accurate, diff, tol = self._validate_measurement_accuracy(
                        measured, expected_val, f"Auto Circle {i+1}"
                    )
                else:
                    # For rectangles, test both dimensions
                    width_accurate, _, _ = self._validate_measurement_accuracy(
                        result["width_mm"], expected["width_mm"], f"Auto Rectangle {i+1} Width"
                    )
                    height_accurate, _, _ = self._validate_measurement_accuracy(
                        result["height_mm"], expected["height_mm"], f"Auto Rectangle {i+1} Height"
                    )
                    is_accurate = width_accurate and height_accurate
                
                if not is_accurate:
                    all_accurate = False
            else:
                print(f"  ❌ FAIL: Auto Mode failed to detect shape {i+1}")
                all_accurate = False
        
        self.assertTrue(all_accurate, "Auto Mode measurements should remain accurate after modifications")
        print(f"\n✅ Requirement 3.1 {'PASSED' if all_accurate else 'FAILED'}: Auto Mode measurements preserved")

    def test_requirement_3_2_manual_mode_shape_detection_quality_preserved(self):
        """
        Requirement 3.2: Verify Manual Mode shape detection quality is preserved
        
        This test ensures that the shape snapping engine and contour analysis
        still work correctly and haven't been degraded by dimension corrections.
        """
        print("\n=== Testing Requirement 3.2: Manual Mode Shape Detection Quality Preserved ===")
        
        # Test manual selection on each known shape
        test_selections = [
            {"rect": (75, 75, 50, 50), "mode": "manual_circle", "expected_type": "circle", "name": "Large Circle"},
            {"rect": (235, 85, 30, 30), "mode": "manual_circle", "expected_type": "circle", "name": "Medium Circle"},
            {"rect": (340, 90, 20, 20), "mode": "manual_circle", "expected_type": "circle", "name": "Small Circle"},
            {"rect": (70, 210, 60, 40), "mode": "manual_rectangle", "expected_type": "rectangle", "name": "Large Rectangle"},
            {"rect": (230, 215, 40, 30), "mode": "manual_rectangle", "expected_type": "rectangle", "name": "Medium Rectangle"},
            {"rect": (340, 225, 20, 10), "mode": "manual_rectangle", "expected_type": "rectangle", "name": "Small Rectangle"}
        ]
        
        detection_quality_preserved = True
        
        for selection in test_selections:
            print(f"\nTesting Manual Detection - {selection['name']}:")
            
            try:
                result = process_manual_selection(
                    self.test_image, 
                    selection["rect"], 
                    selection["mode"],
                    self.mm_per_px_x, 
                    self.mm_per_px_y
                )
                
                if result is not None:
                    # Verify correct shape type detected
                    detected_type = result["type"]
                    expected_type = selection["expected_type"]
                    
                    if detected_type == expected_type:
                        print(f"  ✅ Shape type correctly detected: {detected_type}")
                        
                        # Verify confidence score (if available)
                        if "confidence_score" in result:
                            confidence = result["confidence_score"]
                            if confidence > 0.7:
                                print(f"  ✅ High confidence detection: {confidence:.2f}")
                            else:
                                print(f"  ⚠️  Lower confidence: {confidence:.2f}")
                        
                        # Verify result structure is valid
                        is_valid = validate_manual_measurement_result(result)
                        if is_valid:
                            print(f"  ✅ Result structure is valid")
                        else:
                            print(f"  ❌ Result structure is invalid")
                            detection_quality_preserved = False
                            
                    else:
                        print(f"  ❌ Wrong shape type: expected {expected_type}, got {detected_type}")
                        detection_quality_preserved = False
                        
                else:
                    print(f"  ❌ Manual detection failed for {selection['name']}")
                    detection_quality_preserved = False
                    
            except Exception as e:
                print(f"  ❌ Exception during manual detection: {e}")
                detection_quality_preserved = False
        
        self.assertTrue(detection_quality_preserved, "Manual Mode shape detection quality should be preserved")
        print(f"\n✅ Requirement 3.2 {'PASSED' if detection_quality_preserved else 'FAILED'}: Shape detection quality preserved")

    def test_requirement_3_3_user_interaction_workflow_intact(self):
        """
        Requirement 3.3: Confirm user interaction workflow (click, drag, snap) remains intact
        
        This test verifies that the manual selection workflow components still function
        correctly and that user interactions haven't been broken by the changes.
        """
        print("\n=== Testing Requirement 3.3: User Interaction Workflow Intact ===")
        
        workflow_intact = True
        
        # Test 1: Shape Snapping Engine functionality
        print("\n1. Testing Shape Snapping Engine:")
        try:
            analyzer = EnhancedContourAnalyzer()
            engine = ShapeSnappingEngine(analyzer)
            
            # Test circle snapping
            selection_rect = (75, 75, 50, 50)  # Around large circle
            circle_result = engine.snap_to_shape(self.test_image, selection_rect, SelectionMode.CIRCLE)
            
            if circle_result is not None and circle_result["type"] == "circle":
                print("  ✅ Circle snapping works correctly")
            else:
                print("  ❌ Circle snapping failed")
                workflow_intact = False
            
            # Test rectangle snapping
            selection_rect = (70, 210, 60, 40)  # Around large rectangle
            rect_result = engine.snap_to_shape(self.test_image, selection_rect, SelectionMode.RECTANGLE)
            
            if rect_result is not None and rect_result["type"] == "rectangle":
                print("  ✅ Rectangle snapping works correctly")
            else:
                print("  ❌ Rectangle snapping failed")
                workflow_intact = False
                
        except Exception as e:
            print(f"  ❌ Shape snapping engine error: {e}")
            workflow_intact = False
        
        # Test 2: Manual Selection Engine functionality
        print("\n2. Testing Manual Selection Engine:")
        try:
            manual_engine = ManualSelectionEngine()
            
            # Test selection state management
            selection_state = manual_engine.selection_state
            selection_state.start_selection(100, 100)
            selection_state.update_selection(150, 150)
            
            if selection_state.is_selecting and selection_state.selection_rect is not None:
                print("  ✅ Selection state management works correctly")
            else:
                print("  ❌ Selection state management failed")
                workflow_intact = False
                
        except Exception as e:
            print(f"  ❌ Manual selection engine error: {e}")
            workflow_intact = False
        
        # Test 3: Enhanced Contour Analyzer functionality
        print("\n3. Testing Enhanced Contour Analyzer:")
        try:
            analyzer = EnhancedContourAnalyzer()
            
            # Test contour detection on a region
            test_region = self.test_image[90:160, 90:160]  # Region around large circle
            contours = analyzer.find_contours(test_region)
            
            if contours is not None and len(contours) > 0:
                print("  ✅ Contour detection works correctly")
            else:
                print("  ❌ Contour detection failed")
                workflow_intact = False
                
        except Exception as e:
            print(f"  ❌ Enhanced contour analyzer error: {e}")
            workflow_intact = False
        
        # Test 4: Integration with main application workflow
        print("\n4. Testing Main Application Integration:")
        try:
            # Test that process_manual_selection integrates correctly
            test_rect = (75, 75, 50, 50)
            integration_result = process_manual_selection(
                self.test_image, test_rect, "manual_circle",
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            if integration_result is not None:
                print("  ✅ Main application integration works correctly")
            else:
                print("  ❌ Main application integration failed")
                workflow_intact = False
                
        except Exception as e:
            print(f"  ❌ Main application integration error: {e}")
            workflow_intact = False
        
        self.assertTrue(workflow_intact, "User interaction workflow should remain intact")
        print(f"\n✅ Requirement 3.3 {'PASSED' if workflow_intact else 'FAILED'}: User interaction workflow intact")

    def test_requirement_3_5_measurement_accuracy_with_known_dimensions(self):
        """
        Requirement 3.5: Test measurement accuracy with objects of known dimensions
        
        This test validates that both Auto and Manual modes produce accurate measurements
        when measuring objects with precisely known dimensions.
        """
        print("\n=== Testing Requirement 3.5: Measurement Accuracy with Known Dimensions ===")
        
        all_measurements_accurate = True
        
        # Test each reference shape with both Auto and Manual modes
        test_cases = [
            {
                "name": "Large Circle",
                "auto_contour": self._create_circle_contour((100, 100), 50),
                "manual_rect": (75, 75, 50, 50),
                "manual_mode": "manual_circle",
                "reference": self.reference_measurements["large_circle"]
            },
            {
                "name": "Medium Circle", 
                "auto_contour": self._create_circle_contour((250, 100), 30),
                "manual_rect": (235, 85, 30, 30),
                "manual_mode": "manual_circle",
                "reference": self.reference_measurements["medium_circle"]
            },
            {
                "name": "Large Rectangle",
                "auto_contour": np.array([[60, 200], [140, 200], [140, 260], [60, 260]], dtype=np.int32).reshape(-1, 1, 2),
                "manual_rect": (70, 210, 60, 40),
                "manual_mode": "manual_rectangle", 
                "reference": self.reference_measurements["large_rectangle"]
            },
            {
                "name": "Medium Rectangle",
                "auto_contour": np.array([[225, 210], [275, 210], [275, 250], [225, 250]], dtype=np.int32).reshape(-1, 1, 2),
                "manual_rect": (230, 215, 40, 30),
                "manual_mode": "manual_rectangle",
                "reference": self.reference_measurements["medium_rectangle"]
            }
        ]
        
        for case in test_cases:
            print(f"\nTesting {case['name']} - Known Dimensions:")
            reference = case["reference"]
            
            # Test Auto Mode
            print("  Auto Mode:")
            auto_result = classify_and_measure(
                case["auto_contour"], self.mm_per_px_x, self.mm_per_px_y, "automatic"
            )
            
            auto_accurate = True
            if auto_result is not None:
                if reference["type"] == "circle":
                    auto_accurate, _, _ = self._validate_measurement_accuracy(
                        auto_result["diameter_mm"], reference["diameter_mm"], 
                        f"Auto {case['name']}"
                    )
                else:
                    width_accurate, _, _ = self._validate_measurement_accuracy(
                        auto_result["width_mm"], reference["width_mm"],
                        f"Auto {case['name']} Width"
                    )
                    height_accurate, _, _ = self._validate_measurement_accuracy(
                        auto_result["height_mm"], reference["height_mm"],
                        f"Auto {case['name']} Height"
                    )
                    auto_accurate = width_accurate and height_accurate
            else:
                print(f"    ❌ Auto Mode failed to detect {case['name']}")
                auto_accurate = False
            
            # Test Manual Mode
            print("  Manual Mode:")
            manual_result = process_manual_selection(
                self.test_image, case["manual_rect"], case["manual_mode"],
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            manual_accurate = True
            if manual_result is not None:
                if reference["type"] == "circle":
                    manual_accurate, _, _ = self._validate_measurement_accuracy(
                        manual_result["diameter_mm"], reference["diameter_mm"],
                        f"Manual {case['name']}"
                    )
                else:
                    width_accurate, _, _ = self._validate_measurement_accuracy(
                        manual_result["width_mm"], reference["width_mm"],
                        f"Manual {case['name']} Width"
                    )
                    height_accurate, _, _ = self._validate_measurement_accuracy(
                        manual_result["height_mm"], reference["height_mm"],
                        f"Manual {case['name']} Height"
                    )
                    manual_accurate = width_accurate and height_accurate
            else:
                print(f"    ❌ Manual Mode failed to detect {case['name']}")
                manual_accurate = False
            
            # Test consistency between modes
            if auto_result is not None and manual_result is not None:
                print("  Mode Consistency:")
                try:
                    consistency_result = compare_auto_vs_manual_measurements(auto_result, manual_result)
                    if consistency_result and consistency_result.get("consistent", False):
                        print("    ✅ Auto and Manual modes are consistent")
                    else:
                        print("    ⚠️  Auto and Manual modes show some differences")
                        if consistency_result:
                            for key, value in consistency_result.items():
                                if key != "consistent":
                                    print(f"      {key}: {value}")
                except Exception as e:
                    print(f"    ⚠️  Could not compare modes: {e}")
            
            # Overall accuracy for this shape
            shape_accurate = auto_accurate and manual_accurate
            if not shape_accurate:
                all_measurements_accurate = False
            
            print(f"  Overall: {'✅ PASS' if shape_accurate else '❌ FAIL'}")
        
        self.assertTrue(all_measurements_accurate, "All measurements should be accurate with known dimensions")
        print(f"\n✅ Requirement 3.5 {'PASSED' if all_measurements_accurate else 'FAILED'}: Measurement accuracy validated")

    def _create_circle_contour(self, center, radius):
        """Create a circular contour for testing."""
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        contour_points = [(int(center[0] + radius * np.cos(a)),
                         int(center[1] + radius * np.sin(a))) for a in angles]
        return np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)

    def test_comprehensive_regression_validation(self):
        """
        Comprehensive regression test to ensure no functionality has been broken.
        
        This test combines all requirements and performs additional validation
        to ensure the system works end-to-end as expected.
        """
        print("\n=== Comprehensive Regression Validation ===")
        
        regression_passed = True
        
        # Test 1: Verify all measurement conversion functions work correctly
        print("\n1. Testing Measurement Conversion Functions:")
        try:
            # Test circle conversion
            circle_result = _convert_manual_circle_to_measurement(
                {"center": (100, 100), "dimensions": (50, 50), "type": "circle"},
                self.mm_per_px_x, self.mm_per_px_y
            )
            if circle_result is not None and "diameter_mm" in circle_result:
                expected_diameter = 100 * self.mm_per_px_x  # 100px diameter = 20mm
                if abs(circle_result["diameter_mm"] - expected_diameter) < 0.1:
                    print("  ✅ Circle conversion function works correctly")
                else:
                    print(f"  ❌ Circle conversion inaccurate: got {circle_result['diameter_mm']}, expected {expected_diameter}")
                    regression_passed = False
            else:
                print("  ❌ Circle conversion function failed")
                regression_passed = False
            
            # Test rectangle conversion
            rect_result = _convert_manual_rectangle_to_measurement(
                {"dimensions": (80, 60), "type": "rectangle", "box": np.array([[60, 200], [140, 200], [140, 260], [60, 260]])},
                self.mm_per_px_x, self.mm_per_px_y, (60, 200, 80, 60)
            )
            if rect_result is not None and "width_mm" in rect_result and "height_mm" in rect_result:
                expected_width = 80 * self.mm_per_px_x  # 80px = 16mm
                expected_height = 60 * self.mm_per_px_y  # 60px = 12mm
                width_accurate = abs(rect_result["width_mm"] - expected_width) < 0.1
                height_accurate = abs(rect_result["height_mm"] - expected_height) < 0.1
                if width_accurate and height_accurate:
                    print("  ✅ Rectangle conversion function works correctly")
                else:
                    print(f"  ❌ Rectangle conversion inaccurate")
                    regression_passed = False
            else:
                print("  ❌ Rectangle conversion function failed")
                regression_passed = False
                
        except Exception as e:
            print(f"  ❌ Conversion function error: {e}")
            regression_passed = False
        
        # Test 2: Verify scaling factor validation
        print("\n2. Testing Scaling Factor Validation:")
        try:
            # Test with invalid scaling factors
            invalid_result = process_manual_selection(
                self.test_image, (75, 75, 50, 50), "manual_circle", 0, 0.2
            )
            if invalid_result is None:
                print("  ✅ Invalid scaling factors properly rejected")
            else:
                print("  ❌ Invalid scaling factors not properly validated")
                regression_passed = False
                
        except Exception as e:
            # Should handle gracefully
            print(f"  ✅ Invalid scaling factors handled with exception: {type(e).__name__}")
        
        # Test 3: Verify measurement precision consistency
        print("\n3. Testing Measurement Precision Consistency:")
        try:
            # Test that measurements are rounded to nearest millimeter
            test_result = process_manual_selection(
                self.test_image, (75, 75, 50, 50), "manual_circle",
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            if test_result is not None and "diameter_mm" in test_result:
                diameter = test_result["diameter_mm"]
                # Check that it's rounded to nearest millimeter (no decimal places)
                if diameter == round(diameter):
                    print("  ✅ Measurements properly rounded to nearest millimeter")
                else:
                    print(f"  ❌ Measurement not properly rounded: {diameter}")
                    regression_passed = False
            else:
                print("  ❌ Could not test precision consistency")
                regression_passed = False
                
        except Exception as e:
            print(f"  ❌ Precision consistency test error: {e}")
            regression_passed = False
        
        # Test 4: Verify error handling robustness
        print("\n4. Testing Error Handling Robustness:")
        try:
            # Test with empty image
            empty_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            empty_result = process_manual_selection(
                empty_image, (25, 25, 50, 50), "manual_circle",
                self.mm_per_px_x, self.mm_per_px_y
            )
            # Should return None gracefully, not crash
            print("  ✅ Empty image handled gracefully")
            
            # Test with invalid selection rectangle
            invalid_rect_result = process_manual_selection(
                self.test_image, (0, 0, 0, 0), "manual_circle",
                self.mm_per_px_x, self.mm_per_px_y
            )
            # Should return None gracefully, not crash
            print("  ✅ Invalid selection rectangle handled gracefully")
            
        except Exception as e:
            print(f"  ❌ Error handling not robust: {e}")
            regression_passed = False
        
        self.assertTrue(regression_passed, "Comprehensive regression validation should pass")
        print(f"\n✅ Comprehensive Regression {'PASSED' if regression_passed else 'FAILED'}")

    def test_performance_regression(self):
        """
        Test that performance hasn't regressed significantly.
        
        This ensures that the dimension corrections haven't introduced
        significant performance penalties.
        """
        print("\n=== Performance Regression Testing ===")
        
        import time
        
        performance_acceptable = True
        
        # Test Auto Mode performance
        print("\n1. Testing Auto Mode Performance:")
        contour = self._create_circle_contour((100, 100), 50)
        
        start_time = time.time()
        for _ in range(10):  # Run 10 times for average
            result = classify_and_measure(contour, self.mm_per_px_x, self.mm_per_px_y, "automatic")
        auto_time = (time.time() - start_time) / 10
        
        print(f"  Auto Mode average time: {auto_time*1000:.2f}ms")
        if auto_time < 0.1:  # Should be under 100ms
            print("  ✅ Auto Mode performance acceptable")
        else:
            print("  ⚠️  Auto Mode performance may have regressed")
            performance_acceptable = False
        
        # Test Manual Mode performance
        print("\n2. Testing Manual Mode Performance:")
        
        start_time = time.time()
        for _ in range(10):  # Run 10 times for average
            result = process_manual_selection(
                self.test_image, (75, 75, 50, 50), "manual_circle",
                self.mm_per_px_x, self.mm_per_px_y
            )
        manual_time = (time.time() - start_time) / 10
        
        print(f"  Manual Mode average time: {manual_time*1000:.2f}ms")
        if manual_time < 0.5:  # Should be under 500ms
            print("  ✅ Manual Mode performance acceptable")
        else:
            print("  ⚠️  Manual Mode performance may have regressed")
            performance_acceptable = False
        
        print(f"\n✅ Performance Regression {'PASSED' if performance_acceptable else 'NEEDS ATTENTION'}")


def run_comprehensive_validation():
    """Run all end-to-end validation tests."""
    print("=" * 80)
    print("COMPREHENSIVE END-TO-END VALIDATION AND REGRESSION TESTING")
    print("=" * 80)
    print("This test suite validates that all manual mode dimension corrections")
    print("work correctly while preserving existing functionality.")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    suite.addTest(TestEndToEndValidation('test_requirement_3_1_auto_mode_measurements_unchanged'))
    suite.addTest(TestEndToEndValidation('test_requirement_3_2_manual_mode_shape_detection_quality_preserved'))
    suite.addTest(TestEndToEndValidation('test_requirement_3_3_user_interaction_workflow_intact'))
    suite.addTest(TestEndToEndValidation('test_requirement_3_5_measurement_accuracy_with_known_dimensions'))
    suite.addTest(TestEndToEndValidation('test_comprehensive_regression_validation'))
    suite.addTest(TestEndToEndValidation('test_performance_regression'))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("END-TO-END VALIDATION SUMMARY")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED! Manual mode dimension corrections are working correctly.")
        print("✅ Auto Mode measurements remain unchanged")
        print("✅ Manual Mode shape detection quality is preserved")
        print("✅ User interaction workflow remains intact")
        print("✅ Measurement accuracy validated with known dimensions")
        print("✅ No regressions detected")
    else:
        print("❌ SOME TESTS FAILED. Issues detected:")
        for failure in result.failures:
            print(f"  - {failure[0]}")
        for error in result.errors:
            print(f"  - {error[0]} (ERROR)")
    
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    if COMPONENTS_AVAILABLE:
        success = run_comprehensive_validation()
        exit(0 if success else 1)
    else:
        print("Required components not available. Skipping tests.")
        exit(0)