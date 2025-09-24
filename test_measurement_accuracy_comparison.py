"""
Measurement accuracy validation between automatic and manual modes.

This module validates that manual selection measurements are consistent
with automatic detection measurements for the same shapes.

Requirements tested: 1.4, 1.5, 4.4, 4.5
"""

import unittest
import cv2
import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional

# Import measurement and detection components
try:
    from measure import classify_and_measure, classify_and_measure_manual_selection
    from detection import a4_scale_mm_per_px, detect_objects
    from shape_snapping_engine import ShapeSnappingEngine
    from enhanced_contour_analyzer import EnhancedContourAnalyzer
    from selection_mode import SelectionMode
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


class TestMeasurementAccuracyComparison(unittest.TestCase):
    """Test measurement accuracy between automatic and manual detection modes."""
    
    def setUp(self):
        """Set up measurement accuracy test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Required components not available")
            
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        
        # Create calibrated test images with known ground truth
        self.calibrated_test_cases = self._create_calibrated_test_cases()
        
        # Tolerance for measurement comparisons (in mm)
        self.measurement_tolerance = 2.0  # 2mm tolerance
        self.percentage_tolerance = 0.05  # 5% tolerance
    
    def _create_calibrated_test_cases(self) -> List[Dict]:
        """Create test cases with known ground truth measurements."""
        test_cases = []
        
        # Test Case 1: Perfect circles of various sizes
        for radius_px in [30, 50, 80, 120]:
            img = np.ones((400, 400, 3), dtype=np.uint8) * 255
            center = (200, 200)
            cv2.circle(img, center, radius_px, (0, 0, 0), -1)
            
            # Calculate ground truth
            diameter_mm = radius_px * 2 * self.mm_per_px_x
            
            test_cases.append({
                "name": f"circle_r{radius_px}",
                "image": img,
                "shape_type": "circle",
                "center": center,
                "ground_truth": {
                    "radius_px": radius_px,
                    "diameter_mm": diameter_mm,
                    "area_mm2": math.pi * (diameter_mm / 2) ** 2
                },
                "selection_rect": (center[0] - radius_px - 20, center[1] - radius_px - 20,
                                 (radius_px + 20) * 2, (radius_px + 20) * 2)
            })
        
        # Test Case 2: Perfect rectangles of various sizes
        for width_px, height_px in [(60, 40), (100, 80), (150, 100), (200, 120)]:
            img = np.ones((400, 400, 3), dtype=np.uint8) * 255
            center = (200, 200)
            x1, y1 = center[0] - width_px//2, center[1] - height_px//2
            x2, y2 = center[0] + width_px//2, center[1] + height_px//2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            
            # Calculate ground truth
            width_mm = width_px * self.mm_per_px_x
            height_mm = height_px * self.mm_per_px_y
            
            test_cases.append({
                "name": f"rectangle_{width_px}x{height_px}",
                "image": img,
                "shape_type": "rectangle",
                "center": center,
                "ground_truth": {
                    "width_px": width_px,
                    "height_px": height_px,
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "area_mm2": width_mm * height_mm
                },
                "selection_rect": (x1 - 10, y1 - 10, width_px + 20, height_px + 20)
            })
        
        # Test Case 3: Shapes with different orientations
        for angle in [0, 30, 45, 60]:
            img = np.ones((400, 400, 3), dtype=np.uint8) * 255
            center = (200, 200)
            
            # Create rotated rectangle
            width_px, height_px = 100, 60
            
            # Calculate rotated rectangle points
            cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            corners = [
                (-width_px//2, -height_px//2),
                (width_px//2, -height_px//2),
                (width_px//2, height_px//2),
                (-width_px//2, height_px//2)
            ]
            
            rotated_corners = []
            for x, y in corners:
                rx = int(center[0] + x * cos_a - y * sin_a)
                ry = int(center[1] + x * sin_a + y * cos_a)
                rotated_corners.append((rx, ry))
            
            # Draw rotated rectangle
            points = np.array(rotated_corners, dtype=np.int32)
            cv2.fillPoly(img, [points], (0, 0, 0))
            
            # Ground truth (dimensions don't change with rotation)
            width_mm = width_px * self.mm_per_px_x
            height_mm = height_px * self.mm_per_px_y
            
            test_cases.append({
                "name": f"rotated_rect_{angle}deg",
                "image": img,
                "shape_type": "rectangle",
                "center": center,
                "ground_truth": {
                    "width_px": width_px,
                    "height_px": height_px,
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "area_mm2": width_mm * height_mm,
                    "rotation": angle
                },
                "selection_rect": (120, 120, 160, 160)  # Large enough to contain rotated shape
            })
        
        return test_cases
    
    def _create_contour_from_image(self, image: np.ndarray, shape_type: str) -> Optional[np.ndarray]:
        """Extract contour from test image for automatic detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Return the largest contour
        return max(contours, key=cv2.contourArea)
    
    def test_circle_measurement_consistency(self):
        """Test consistency of circle measurements between automatic and manual modes."""
        circle_cases = [case for case in self.calibrated_test_cases if case["shape_type"] == "circle"]
        
        for test_case in circle_cases:
            with self.subTest(case=test_case["name"]):
                image = test_case["image"]
                ground_truth = test_case["ground_truth"]
                selection_rect = test_case["selection_rect"]
                
                # Get automatic measurement
                contour = self._create_contour_from_image(image, "circle")
                self.assertIsNotNone(contour, f"Could not extract contour for {test_case['name']}")
                
                auto_result = classify_and_measure(
                    contour, self.mm_per_px_x, self.mm_per_px_y, "automatic"
                )
                self.assertIsNotNone(auto_result, f"Automatic detection failed for {test_case['name']}")
                self.assertEqual(auto_result["type"], "circle")
                
                # Get manual measurement
                manual_shape_result = self.snap_engine.snap_to_shape(
                    image, selection_rect, SelectionMode.MANUAL_CIRCLE
                )
                self.assertIsNotNone(manual_shape_result, f"Manual detection failed for {test_case['name']}")
                
                manual_result = classify_and_measure_manual_selection(
                    image, selection_rect, manual_shape_result,
                    self.mm_per_px_x, self.mm_per_px_y
                )
                self.assertIsNotNone(manual_result, f"Manual measurement failed for {test_case['name']}")
                self.assertEqual(manual_result["type"], "circle")
                
                # Compare measurements
                auto_diameter = auto_result["diameter_mm"]
                manual_diameter = manual_result["diameter_mm"]
                expected_diameter = ground_truth["diameter_mm"]
                
                # Check accuracy against ground truth
                auto_error = abs(auto_diameter - expected_diameter)
                manual_error = abs(manual_diameter - expected_diameter)
                
                self.assertLess(auto_error, self.measurement_tolerance,
                               f"Automatic diameter error {auto_error:.2f}mm > {self.measurement_tolerance}mm")
                self.assertLess(manual_error, self.measurement_tolerance,
                               f"Manual diameter error {manual_error:.2f}mm > {self.measurement_tolerance}mm")
                
                # Check consistency between modes
                diameter_diff = abs(auto_diameter - manual_diameter)
                self.assertLess(diameter_diff, self.measurement_tolerance,
                               f"Diameter difference {diameter_diff:.2f}mm between modes")
                
                # Check percentage difference
                percentage_diff = diameter_diff / expected_diameter
                self.assertLess(percentage_diff, self.percentage_tolerance,
                               f"Diameter percentage difference {percentage_diff:.1%} too large")
                
                print(f"Circle {test_case['name']}: Ground truth {expected_diameter:.2f}mm, "
                      f"Auto {auto_diameter:.2f}mm, Manual {manual_diameter:.2f}mm")
    
    def test_rectangle_measurement_consistency(self):
        """Test consistency of rectangle measurements between automatic and manual modes."""
        rect_cases = [case for case in self.calibrated_test_cases 
                     if case["shape_type"] == "rectangle" and "rotation" not in case["ground_truth"]]
        
        for test_case in rect_cases:
            with self.subTest(case=test_case["name"]):
                image = test_case["image"]
                ground_truth = test_case["ground_truth"]
                selection_rect = test_case["selection_rect"]
                
                # Get automatic measurement
                contour = self._create_contour_from_image(image, "rectangle")
                self.assertIsNotNone(contour, f"Could not extract contour for {test_case['name']}")
                
                auto_result = classify_and_measure(
                    contour, self.mm_per_px_x, self.mm_per_px_y, "automatic"
                )
                self.assertIsNotNone(auto_result, f"Automatic detection failed for {test_case['name']}")
                self.assertEqual(auto_result["type"], "rectangle")
                
                # Get manual measurement
                manual_shape_result = self.snap_engine.snap_to_shape(
                    image, selection_rect, SelectionMode.MANUAL_RECTANGLE
                )
                self.assertIsNotNone(manual_shape_result, f"Manual detection failed for {test_case['name']}")
                
                manual_result = classify_and_measure_manual_selection(
                    image, selection_rect, manual_shape_result,
                    self.mm_per_px_x, self.mm_per_px_y
                )
                self.assertIsNotNone(manual_result, f"Manual measurement failed for {test_case['name']}")
                self.assertEqual(manual_result["type"], "rectangle")
                
                # Compare measurements (dimensions might be swapped due to normalization)
                auto_dims = sorted([auto_result["width_mm"], auto_result["height_mm"]])
                manual_dims = sorted([manual_result["width_mm"], manual_result["height_mm"]])
                expected_dims = sorted([ground_truth["width_mm"], ground_truth["height_mm"]])
                
                for i, (auto_dim, manual_dim, expected_dim) in enumerate(zip(auto_dims, manual_dims, expected_dims)):
                    # Check accuracy against ground truth
                    auto_error = abs(auto_dim - expected_dim)
                    manual_error = abs(manual_dim - expected_dim)
                    
                    self.assertLess(auto_error, self.measurement_tolerance,
                                   f"Automatic dimension {i} error {auto_error:.2f}mm > {self.measurement_tolerance}mm")
                    self.assertLess(manual_error, self.measurement_tolerance,
                                   f"Manual dimension {i} error {manual_error:.2f}mm > {self.measurement_tolerance}mm")
                    
                    # Check consistency between modes
                    dim_diff = abs(auto_dim - manual_dim)
                    self.assertLess(dim_diff, self.measurement_tolerance,
                                   f"Dimension {i} difference {dim_diff:.2f}mm between modes")
                    
                    # Check percentage difference
                    percentage_diff = dim_diff / expected_dim
                    self.assertLess(percentage_diff, self.percentage_tolerance,
                                   f"Dimension {i} percentage difference {percentage_diff:.1%} too large")
                
                print(f"Rectangle {test_case['name']}: Ground truth {expected_dims}, "
                      f"Auto {auto_dims}, Manual {manual_dims}")
    
    def test_area_calculation_consistency(self):
        """Test consistency of area calculations between modes."""
        for test_case in self.calibrated_test_cases:
            if "rotation" in test_case["ground_truth"]:
                continue  # Skip rotated shapes for area comparison
                
            with self.subTest(case=test_case["name"]):
                image = test_case["image"]
                ground_truth = test_case["ground_truth"]
                selection_rect = test_case["selection_rect"]
                shape_type = test_case["shape_type"]
                
                # Get automatic measurement
                contour = self._create_contour_from_image(image, shape_type)
                auto_result = classify_and_measure(
                    contour, self.mm_per_px_x, self.mm_per_px_y, "automatic"
                )
                
                # Get manual measurement
                mode = SelectionMode.MANUAL_CIRCLE if shape_type == "circle" else SelectionMode.MANUAL_RECTANGLE
                manual_shape_result = self.snap_engine.snap_to_shape(image, selection_rect, mode)
                manual_result = classify_and_measure_manual_selection(
                    image, selection_rect, manual_shape_result,
                    self.mm_per_px_x, self.mm_per_px_y
                )
                
                if auto_result and manual_result:
                    # Calculate areas
                    if shape_type == "circle":
                        auto_area = math.pi * (auto_result["diameter_mm"] / 2) ** 2
                        manual_area = math.pi * (manual_result["diameter_mm"] / 2) ** 2
                    else:
                        auto_area = auto_result["width_mm"] * auto_result["height_mm"]
                        manual_area = manual_result["width_mm"] * manual_result["height_mm"]
                    
                    expected_area = ground_truth["area_mm2"]
                    
                    # Check accuracy
                    auto_area_error = abs(auto_area - expected_area)
                    manual_area_error = abs(manual_area - expected_area)
                    
                    # Allow larger tolerance for area (compound error)
                    area_tolerance = self.measurement_tolerance * 10
                    
                    self.assertLess(auto_area_error, area_tolerance,
                                   f"Automatic area error {auto_area_error:.2f}mm² too large")
                    self.assertLess(manual_area_error, area_tolerance,
                                   f"Manual area error {manual_area_error:.2f}mm² too large")
                    
                    # Check consistency between modes
                    area_diff = abs(auto_area - manual_area)
                    self.assertLess(area_diff, area_tolerance,
                                   f"Area difference {area_diff:.2f}mm² between modes")
                    
                    print(f"Area {test_case['name']}: Ground truth {expected_area:.2f}mm², "
                          f"Auto {auto_area:.2f}mm², Manual {manual_area:.2f}mm²")
    
    def test_center_position_consistency(self):
        """Test consistency of center position detection between modes."""
        for test_case in self.calibrated_test_cases:
            with self.subTest(case=test_case["name"]):
                image = test_case["image"]
                expected_center = test_case["center"]
                selection_rect = test_case["selection_rect"]
                shape_type = test_case["shape_type"]
                
                # Get automatic measurement
                contour = self._create_contour_from_image(image, shape_type)
                auto_result = classify_and_measure(
                    contour, self.mm_per_px_x, self.mm_per_px_y, "automatic"
                )
                
                # Get manual measurement
                mode = SelectionMode.MANUAL_CIRCLE if shape_type == "circle" else SelectionMode.MANUAL_RECTANGLE
                manual_shape_result = self.snap_engine.snap_to_shape(image, selection_rect, mode)
                manual_result = classify_and_measure_manual_selection(
                    image, selection_rect, manual_shape_result,
                    self.mm_per_px_x, self.mm_per_px_y
                )
                
                if auto_result and manual_result:
                    auto_center = auto_result["center"]
                    manual_center = manual_result["center"]
                    
                    # Check accuracy against ground truth
                    auto_center_error = math.sqrt(
                        (auto_center[0] - expected_center[0])**2 + 
                        (auto_center[1] - expected_center[1])**2
                    )
                    manual_center_error = math.sqrt(
                        (manual_center[0] - expected_center[0])**2 + 
                        (manual_center[1] - expected_center[1])**2
                    )
                    
                    # Centers should be within a few pixels
                    center_tolerance = 5  # pixels
                    
                    self.assertLess(auto_center_error, center_tolerance,
                                   f"Automatic center error {auto_center_error:.1f}px too large")
                    self.assertLess(manual_center_error, center_tolerance,
                                   f"Manual center error {manual_center_error:.1f}px too large")
                    
                    # Check consistency between modes
                    center_diff = math.sqrt(
                        (auto_center[0] - manual_center[0])**2 + 
                        (auto_center[1] - manual_center[1])**2
                    )
                    self.assertLess(center_diff, center_tolerance,
                                   f"Center difference {center_diff:.1f}px between modes")
                    
                    print(f"Center {test_case['name']}: Expected {expected_center}, "
                          f"Auto {auto_center}, Manual {manual_center}")
    
    def test_measurement_precision_stability(self):
        """Test that measurements are stable across multiple runs."""
        # Use a simple test case
        test_case = self.calibrated_test_cases[0]  # First circle
        image = test_case["image"]
        selection_rect = test_case["selection_rect"]
        
        # Perform multiple measurements
        num_runs = 10
        manual_measurements = []
        
        for i in range(num_runs):
            manual_shape_result = self.snap_engine.snap_to_shape(
                image, selection_rect, SelectionMode.MANUAL_CIRCLE
            )
            
            if manual_shape_result:
                manual_result = classify_and_measure_manual_selection(
                    image, selection_rect, manual_shape_result,
                    self.mm_per_px_x, self.mm_per_px_y
                )
                
                if manual_result:
                    manual_measurements.append(manual_result["diameter_mm"])
        
        # Check stability
        if len(manual_measurements) >= 2:
            avg_measurement = sum(manual_measurements) / len(manual_measurements)
            max_deviation = max(abs(m - avg_measurement) for m in manual_measurements)
            
            # Measurements should be very stable (within 0.1mm)
            self.assertLess(max_deviation, 0.1,
                           f"Measurement instability: {max_deviation:.3f}mm deviation")
            
            print(f"Stability test: {len(manual_measurements)} runs, "
                  f"avg {avg_measurement:.3f}mm, max deviation {max_deviation:.3f}mm")
    
    def test_edge_case_measurement_handling(self):
        """Test measurement handling for edge cases."""
        # Very small shape
        small_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(small_img, (100, 100), 10, (0, 0, 0), -1)
        
        small_selection = (85, 85, 30, 30)
        small_result = self.snap_engine.snap_to_shape(
            small_img, small_selection, SelectionMode.MANUAL_CIRCLE
        )
        
        if small_result:
            small_measurement = classify_and_measure_manual_selection(
                small_img, small_selection, small_result,
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            if small_measurement:
                # Should detect reasonable size
                self.assertGreater(small_measurement["diameter_mm"], 5)
                self.assertLess(small_measurement["diameter_mm"], 50)
        
        # Very large shape (near image boundaries)
        large_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(large_img, (200, 200), 180, (0, 0, 0), -1)
        
        large_selection = (50, 50, 300, 300)
        large_result = self.snap_engine.snap_to_shape(
            large_img, large_selection, SelectionMode.MANUAL_CIRCLE
        )
        
        if large_result:
            large_measurement = classify_and_measure_manual_selection(
                large_img, large_selection, large_result,
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            if large_measurement:
                # Should detect reasonable size
                self.assertGreater(large_measurement["diameter_mm"], 100)
                self.assertLess(large_measurement["diameter_mm"], 1000)


class TestMeasurementDataFormatConsistency(unittest.TestCase):
    """Test that measurement data formats are consistent between modes."""
    
    def setUp(self):
        """Set up data format consistency tests."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Required components not available")
            
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        
        # Create simple test image
        self.test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(self.test_image, (200, 200), 50, (0, 0, 0), -1)
        cv2.rectangle(self.test_image, (100, 300), (200, 350), (0, 0, 0), -1)
    
    def test_circle_data_format_consistency(self):
        """Test that circle measurement data formats are consistent."""
        # Get automatic measurement
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circle_contour = max(contours, key=cv2.contourArea)
        auto_result = classify_and_measure(
            circle_contour, self.mm_per_px_x, self.mm_per_px_y, "automatic"
        )
        
        # Get manual measurement
        selection_rect = (150, 150, 100, 100)
        manual_shape_result = self.snap_engine.snap_to_shape(
            self.test_image, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        if manual_shape_result:
            manual_result = classify_and_measure_manual_selection(
                self.test_image, selection_rect, manual_shape_result,
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            if auto_result and manual_result:
                # Check that both have required fields
                required_fields = ["type", "center", "diameter_mm", "radius_px", "area_px", "hit_contour"]
                
                for field in required_fields:
                    self.assertIn(field, auto_result, f"Automatic result missing {field}")
                    self.assertIn(field, manual_result, f"Manual result missing {field}")
                
                # Check data types
                self.assertEqual(type(auto_result["type"]), type(manual_result["type"]))
                self.assertEqual(type(auto_result["center"]), type(manual_result["center"]))
                self.assertEqual(type(auto_result["diameter_mm"]), type(manual_result["diameter_mm"]))
                
                # Manual result should have additional fields
                self.assertIn("detection_method", manual_result)
                self.assertEqual(manual_result["detection_method"], "manual")
                
                # Automatic result should have detection_method too (if added)
                if "detection_method" in auto_result:
                    self.assertEqual(auto_result["detection_method"], "automatic")
    
    def test_rectangle_data_format_consistency(self):
        """Test that rectangle measurement data formats are consistent."""
        # Get automatic measurement
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find rectangle contour (smaller area)
        rect_contour = min(contours, key=cv2.contourArea)
        auto_result = classify_and_measure(
            rect_contour, self.mm_per_px_x, self.mm_per_px_y, "automatic"
        )
        
        # Get manual measurement
        selection_rect = (80, 280, 140, 90)
        manual_shape_result = self.snap_engine.snap_to_shape(
            self.test_image, selection_rect, SelectionMode.MANUAL_RECTANGLE
        )
        
        if manual_shape_result:
            manual_result = classify_and_measure_manual_selection(
                self.test_image, selection_rect, manual_shape_result,
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            if auto_result and manual_result:
                # Check that both have required fields
                required_fields = ["type", "width_mm", "height_mm", "area_px", "hit_contour", "box"]
                
                for field in required_fields:
                    self.assertIn(field, auto_result, f"Automatic result missing {field}")
                    self.assertIn(field, manual_result, f"Manual result missing {field}")
                
                # Check data types
                self.assertEqual(type(auto_result["type"]), type(manual_result["type"]))
                self.assertEqual(type(auto_result["width_mm"]), type(manual_result["width_mm"]))
                self.assertEqual(type(auto_result["height_mm"]), type(manual_result["height_mm"]))
                
                # Manual result should have additional fields
                self.assertIn("detection_method", manual_result)
                self.assertEqual(manual_result["detection_method"], "manual")


if __name__ == '__main__':
    # Run measurement accuracy comparison tests
    print("="*60)
    print("MEASUREMENT ACCURACY COMPARISON TEST SUITE")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMeasurementAccuracyComparison))
    suite.addTest(unittest.makeSuite(TestMeasurementDataFormatConsistency))
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MEASUREMENT ACCURACY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nACCURACY ISSUES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    if not result.failures and not result.errors:
        print("\n✅ All measurement accuracy tests passed!")
        print("Manual and automatic measurements are consistent.")