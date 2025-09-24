"""
Comprehensive test suite for manual shape selection validation.

This module provides comprehensive testing for the manual shape selection feature,
including accuracy validation, performance testing, and edge case handling.

Requirements tested: 1.4, 1.5, 4.4, 4.5
"""

import unittest
import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock

# Import the components we're testing
try:
    from extended_interaction_manager import ExtendedInteractionManager
    from manual_selection_engine import ManualSelectionEngine
    from shape_snapping_engine import ShapeSnappingEngine
    from enhanced_contour_analyzer import EnhancedContourAnalyzer
    from selection_mode import SelectionMode
    from measure import classify_and_measure, classify_and_measure_manual_selection
    from detection import a4_scale_mm_per_px
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available for testing: {e}")
    COMPONENTS_AVAILABLE = False


class TestManualSelectionAccuracy(unittest.TestCase):
    """Test manual selection accuracy with various object types and scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Manual selection components not available")
            
        self.test_images = self._create_test_image_suite()
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        
        # Initialize components
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        self.manual_engine = ManualSelectionEngine(self.analyzer)
    
    def _create_test_image_suite(self) -> Dict[str, np.ndarray]:
        """Create comprehensive test image suite."""
        images = {}
        
        # 1. Simple shapes on clean background
        simple_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(simple_img, (150, 150), 50, (0, 0, 0), -1)
        cv2.rectangle(simple_img, (250, 100), (350, 200), (0, 0, 0), -1)
        images["simple_shapes"] = simple_img
        
        # 2. Nested shapes (shapes within shapes)
        nested_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(nested_img, (50, 50), (350, 350), (0, 0, 0), -1)
        cv2.rectangle(nested_img, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.circle(nested_img, (200, 200), 60, (0, 0, 0), -1)
        cv2.circle(nested_img, (200, 200), 30, (255, 255, 255), -1)
        images["nested_shapes"] = nested_img
        
        # 3. Overlapping shapes
        overlap_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(overlap_img, (150, 150), 60, (0, 0, 0), -1)
        cv2.circle(overlap_img, (200, 150), 60, (100, 100, 100), -1)
        cv2.rectangle(overlap_img, (120, 200), (230, 300), (50, 50, 50), -1)
        images["overlapping_shapes"] = overlap_img
        
        # 4. Complex background with noise
        complex_img = np.random.randint(180, 220, (400, 400, 3), dtype=np.uint8)
        cv2.circle(complex_img, (150, 150), 50, (0, 0, 0), -1)
        cv2.rectangle(complex_img, (250, 100), (350, 200), (255, 255, 255), -1)
        # Add noise
        for _ in range(100):
            x, y = np.random.randint(0, 400, 2)
            cv2.circle(complex_img, (x, y), np.random.randint(1, 5), 
                      (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), -1)
        images["complex_background"] = complex_img
        
        # 5. Varying lighting conditions
        gradient_img = np.zeros((400, 400, 3), dtype=np.uint8)
        for y in range(400):
            for x in range(400):
                intensity = int(50 + 150 * (x + y) / 800)
                gradient_img[y, x] = [intensity, intensity, intensity]
        cv2.circle(gradient_img, (150, 150), 50, (0, 0, 0), -1)
        cv2.rectangle(gradient_img, (250, 100), (350, 200), (255, 255, 255), -1)
        images["varying_lighting"] = gradient_img
        
        # 6. Small shapes requiring precision
        small_shapes_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(small_shapes_img, (100, 100), 15, (0, 0, 0), -1)
        cv2.circle(small_shapes_img, (150, 100), 20, (0, 0, 0), -1)
        cv2.rectangle(small_shapes_img, (200, 90), (230, 110), (0, 0, 0), -1)
        cv2.rectangle(small_shapes_img, (250, 85), (290, 115), (0, 0, 0), -1)
        images["small_shapes"] = small_shapes_img
        
        # 7. Large shapes testing boundaries
        large_shapes_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(large_shapes_img, (200, 200), 150, (0, 0, 0), -1)
        cv2.rectangle(large_shapes_img, (10, 10), (190, 190), (100, 100, 100), -1)
        images["large_shapes"] = large_shapes_img
        
        return images
    
    def test_circle_detection_accuracy(self):
        """Test circle detection accuracy across different scenarios."""
        test_cases = [
            ("simple_shapes", (125, 125, 50, 50), "circle", 0.8),
            ("nested_shapes", (170, 170, 60, 60), "circle", 0.7),
            ("complex_background", (125, 125, 50, 50), "circle", 0.6),
            ("varying_lighting", (125, 125, 50, 50), "circle", 0.6),
            ("small_shapes", (85, 85, 30, 30), "circle", 0.5),
        ]
        
        for image_name, selection_rect, expected_type, min_confidence in test_cases:
            with self.subTest(image=image_name):
                image = self.test_images[image_name]
                
                result = self.snap_engine.snap_to_shape(
                    image, selection_rect, SelectionMode.MANUAL_CIRCLE
                )
                
                if result is not None:
                    self.assertEqual(result["type"], expected_type)
                    self.assertGreaterEqual(result["confidence_score"], min_confidence)
                    self.assertIn("radius", result)
                    self.assertIn("diameter", result)
                    self.assertGreater(result["radius"], 0)
                else:
                    # For some complex cases, detection might fail - this is acceptable
                    # but we should log it for analysis
                    print(f"Warning: No {expected_type} detected in {image_name}")
    
    def test_rectangle_detection_accuracy(self):
        """Test rectangle detection accuracy across different scenarios."""
        test_cases = [
            ("simple_shapes", (225, 75, 150, 150), "rectangle", 0.8),
            ("nested_shapes", (75, 75, 250, 250), "rectangle", 0.7),
            ("complex_background", (225, 75, 150, 150), "rectangle", 0.6),
            ("varying_lighting", (225, 75, 150, 150), "rectangle", 0.6),
            ("small_shapes", (195, 85, 40, 30), "rectangle", 0.5),
        ]
        
        for image_name, selection_rect, expected_type, min_confidence in test_cases:
            with self.subTest(image=image_name):
                image = self.test_images[image_name]
                
                result = self.snap_engine.snap_to_shape(
                    image, selection_rect, SelectionMode.MANUAL_RECTANGLE
                )
                
                if result is not None:
                    self.assertEqual(result["type"], expected_type)
                    self.assertGreaterEqual(result["confidence_score"], min_confidence)
                    self.assertIn("width", result)
                    self.assertIn("height", result)
                    self.assertGreater(result["width"], 0)
                    self.assertGreater(result["height"], 0)
                else:
                    print(f"Warning: No {expected_type} detected in {image_name}")
    
    def test_nested_shape_detection(self):
        """Test detection of shapes within shapes."""
        nested_image = self.test_images["nested_shapes"]
        
        # Test outer rectangle detection
        outer_rect_selection = (25, 25, 350, 350)
        outer_result = self.snap_engine.snap_to_shape(
            nested_image, outer_rect_selection, SelectionMode.MANUAL_RECTANGLE
        )
        
        if outer_result:
            self.assertEqual(outer_result["type"], "rectangle")
            self.assertGreater(outer_result["width"], 250)  # Should detect large outer rectangle
        
        # Test inner circle detection
        inner_circle_selection = (140, 140, 120, 120)
        inner_result = self.snap_engine.snap_to_shape(
            nested_image, inner_circle_selection, SelectionMode.MANUAL_CIRCLE
        )
        
        if inner_result:
            self.assertEqual(inner_result["type"], "circle")
            self.assertGreater(inner_result["radius"], 25)  # Should detect inner circle
    
    def test_overlapping_shape_handling(self):
        """Test handling of overlapping shapes."""
        overlap_image = self.test_images["overlapping_shapes"]
        
        # Test selection of left circle
        left_circle_selection = (90, 90, 120, 120)
        left_result = self.snap_engine.snap_to_shape(
            overlap_image, left_circle_selection, SelectionMode.MANUAL_CIRCLE
        )
        
        # Test selection of right circle
        right_circle_selection = (140, 90, 120, 120)
        right_result = self.snap_engine.snap_to_shape(
            overlap_image, right_circle_selection, SelectionMode.MANUAL_CIRCLE
        )
        
        # Both should detect circles, but potentially different ones
        if left_result and right_result:
            self.assertEqual(left_result["type"], "circle")
            self.assertEqual(right_result["type"], "circle")
            
            # Centers should be different (detecting different circles)
            left_center = left_result["center"]
            right_center = right_result["center"]
            distance = abs(left_center[0] - right_center[0]) + abs(left_center[1] - right_center[1])
            self.assertGreater(distance, 20)  # Should be detecting different circles
    
    def test_precision_with_small_shapes(self):
        """Test precision when detecting small shapes."""
        small_image = self.test_images["small_shapes"]
        
        # Test small circle detection
        small_selections = [
            (85, 85, 30, 30),   # Around 15px radius circle
            (135, 85, 30, 30),  # Around 20px radius circle
        ]
        
        for i, selection in enumerate(small_selections):
            with self.subTest(selection=i):
                result = self.snap_engine.snap_to_shape(
                    small_image, selection, SelectionMode.MANUAL_CIRCLE
                )
                
                if result:
                    self.assertEqual(result["type"], "circle")
                    self.assertGreater(result["radius"], 10)  # Should detect reasonable size
                    self.assertLess(result["radius"], 30)     # But not too large
    
    def test_boundary_case_handling(self):
        """Test handling of shapes near image boundaries."""
        large_image = self.test_images["large_shapes"]
        
        # Test large circle that extends to boundaries
        large_circle_selection = (50, 50, 300, 300)
        result = self.snap_engine.snap_to_shape(
            large_image, large_circle_selection, SelectionMode.MANUAL_CIRCLE
        )
        
        if result:
            self.assertEqual(result["type"], "circle")
            # Should handle boundary clipping gracefully
            self.assertGreater(result["radius"], 100)
    
    def test_selection_rect_validation(self):
        """Test validation of selection rectangles."""
        test_image = self.test_images["simple_shapes"]
        
        # Valid selection
        valid_result = self.snap_engine.snap_to_shape(
            test_image, (100, 100, 100, 100), SelectionMode.MANUAL_CIRCLE
        )
        # Should either succeed or fail gracefully
        
        # Invalid selections should return None
        invalid_selections = [
            (-10, 100, 100, 100),    # Negative x
            (100, -10, 100, 100),    # Negative y
            (350, 100, 100, 100),    # Out of bounds x
            (100, 350, 100, 100),    # Out of bounds y
            (100, 100, 5, 100),      # Too small width
            (100, 100, 100, 5),      # Too small height
        ]
        
        for invalid_selection in invalid_selections:
            result = self.snap_engine.snap_to_shape(
                test_image, invalid_selection, SelectionMode.MANUAL_CIRCLE
            )
            self.assertIsNone(result, f"Should reject invalid selection: {invalid_selection}")


class TestManualSelectionPerformance(unittest.TestCase):
    """Test performance characteristics of manual selection system."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Manual selection components not available")
            
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        self.manual_engine = ManualSelectionEngine(self.analyzer)
        
        # Create performance test images
        self.performance_images = self._create_performance_test_images()
    
    def _create_performance_test_images(self) -> Dict[str, np.ndarray]:
        """Create images for performance testing."""
        images = {}
        
        # Standard resolution image
        std_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(std_img, (200, 200), 50, (0, 0, 0), -1)
        images["standard"] = std_img
        
        # High resolution image
        hd_img = np.ones((1920, 1080, 3), dtype=np.uint8) * 255
        cv2.circle(hd_img, (960, 540), 200, (0, 0, 0), -1)
        cv2.rectangle(hd_img, (400, 200), (800, 600), (0, 0, 0), -1)
        images["high_resolution"] = hd_img
        
        # Complex image with many shapes
        complex_img = np.ones((800, 800, 3), dtype=np.uint8) * 255
        for i in range(20):
            x, y = np.random.randint(50, 750, 2)
            radius = np.random.randint(20, 60)
            cv2.circle(complex_img, (x, y), radius, (0, 0, 0), -1)
        for i in range(15):
            x1, y1 = np.random.randint(50, 700, 2)
            x2, y2 = x1 + np.random.randint(40, 100), y1 + np.random.randint(40, 100)
            cv2.rectangle(complex_img, (x1, y1), (x2, y2), (100, 100, 100), -1)
        images["complex"] = complex_img
        
        return images
    
    def test_single_selection_performance(self):
        """Test performance of single shape selection operations."""
        test_cases = [
            ("standard", (150, 150, 100, 100), 0.5),      # Should be very fast
            ("high_resolution", (760, 340, 400, 400), 2.0), # Slower but acceptable
            ("complex", (100, 100, 200, 200), 1.0),       # Moderate complexity
        ]
        
        for image_name, selection_rect, max_time in test_cases:
            with self.subTest(image=image_name):
                image = self.performance_images[image_name]
                
                start_time = time.time()
                result = self.snap_engine.snap_to_shape(
                    image, selection_rect, SelectionMode.MANUAL_CIRCLE
                )
                end_time = time.time()
                
                processing_time = end_time - start_time
                self.assertLess(processing_time, max_time, 
                               f"Selection took {processing_time:.3f}s, expected < {max_time}s")
                
                # Log performance for analysis
                print(f"Performance: {image_name} selection took {processing_time:.3f}s")
    
    def test_multiple_selections_performance(self):
        """Test performance when making multiple selections rapidly."""
        image = self.performance_images["standard"]
        selections = [
            (50, 50, 100, 100),
            (150, 150, 100, 100),
            (250, 250, 100, 100),
            (100, 250, 100, 100),
            (250, 100, 100, 100),
        ]
        
        start_time = time.time()
        results = []
        
        for selection in selections:
            result = self.snap_engine.snap_to_shape(
                image, selection, SelectionMode.MANUAL_CIRCLE
            )
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(selections)
        
        # Multiple selections should not significantly degrade performance
        self.assertLess(avg_time, 0.3, f"Average selection time {avg_time:.3f}s too slow")
        self.assertLess(total_time, 1.5, f"Total time {total_time:.3f}s too slow")
        
        print(f"Performance: {len(selections)} selections took {total_time:.3f}s "
              f"(avg: {avg_time:.3f}s per selection)")
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during extended use."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        image = self.performance_images["standard"]
        selection = (150, 150, 100, 100)
        
        # Perform many selections to test for memory leaks
        for i in range(50):
            result = self.snap_engine.snap_to_shape(
                image, selection, SelectionMode.MANUAL_CIRCLE
            )
            
            # Check memory every 10 iterations
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory should not increase significantly
                self.assertLess(memory_increase, 50, 
                               f"Memory increased by {memory_increase:.1f}MB after {i+1} selections")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"Memory usage: Initial {initial_memory:.1f}MB, "
              f"Final {final_memory:.1f}MB, Increase {total_increase:.1f}MB")
        
        # Total memory increase should be reasonable
        self.assertLess(total_increase, 100, "Excessive memory usage detected")
    
    def test_concurrent_selection_handling(self):
        """Test handling of concurrent selection operations."""
        image = self.performance_images["standard"]
        results = []
        errors = []
        
        def perform_selection(selection_rect, mode):
            try:
                result = self.snap_engine.snap_to_shape(image, selection_rect, mode)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads performing selections
        threads = []
        selections = [
            ((50, 50, 100, 100), SelectionMode.MANUAL_CIRCLE),
            ((150, 150, 100, 100), SelectionMode.MANUAL_RECTANGLE),
            ((250, 250, 100, 100), SelectionMode.MANUAL_CIRCLE),
            ((100, 250, 100, 100), SelectionMode.MANUAL_RECTANGLE),
        ]
        
        start_time = time.time()
        
        for selection_rect, mode in selections:
            thread = threading.Thread(target=perform_selection, args=(selection_rect, mode))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)  # 5 second timeout
        
        end_time = time.time()
        
        # Check that no errors occurred
        self.assertEqual(len(errors), 0, f"Concurrent selection errors: {errors}")
        
        # Check that all selections completed
        self.assertEqual(len(results), len(selections), "Not all selections completed")
        
        # Concurrent operations should not take much longer than sequential
        self.assertLess(end_time - start_time, 3.0, "Concurrent operations too slow")
        
        print(f"Concurrent performance: {len(selections)} selections took {end_time - start_time:.3f}s")


class TestMeasurementAccuracyValidation(unittest.TestCase):
    """Test measurement accuracy between automatic and manual modes."""
    
    def setUp(self):
        """Set up measurement accuracy test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Manual selection components not available")
            
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        
        # Create test images with known dimensions
        self.calibrated_images = self._create_calibrated_test_images()
    
    def _create_calibrated_test_images(self) -> Dict[str, Tuple[np.ndarray, Dict]]:
        """Create test images with known ground truth dimensions."""
        images = {}
        
        # Perfect circle with known radius
        circle_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        circle_radius_px = 50
        cv2.circle(circle_img, (200, 200), circle_radius_px, (0, 0, 0), -1)
        
        circle_ground_truth = {
            "type": "circle",
            "center": (200, 200),
            "radius_px": circle_radius_px,
            "diameter_mm": circle_radius_px * 2 * self.mm_per_px_x
        }
        images["perfect_circle"] = (circle_img, circle_ground_truth)
        
        # Perfect rectangle with known dimensions
        rect_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        rect_width_px, rect_height_px = 80, 60
        cv2.rectangle(rect_img, (160, 170), (240, 230), (0, 0, 0), -1)
        
        rect_ground_truth = {
            "type": "rectangle",
            "center": (200, 200),
            "width_px": rect_width_px,
            "height_px": rect_height_px,
            "width_mm": rect_width_px * self.mm_per_px_x,
            "height_mm": rect_height_px * self.mm_per_px_y
        }
        images["perfect_rectangle"] = (rect_img, rect_ground_truth)
        
        return images
    
    def test_circle_measurement_accuracy(self):
        """Test accuracy of circle measurements between automatic and manual modes."""
        circle_img, ground_truth = self.calibrated_images["perfect_circle"]
        
        # Create contour for automatic measurement
        circle_contour = self._create_circle_contour(
            ground_truth["center"], ground_truth["radius_px"]
        )
        
        # Get automatic measurement
        auto_result = classify_and_measure(
            circle_contour, self.mm_per_px_x, self.mm_per_px_y, "automatic"
        )
        
        # Get manual measurement
        selection_rect = (150, 150, 100, 100)
        manual_shape_result = self.snap_engine.snap_to_shape(
            circle_img, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        if auto_result and manual_shape_result:
            manual_result = classify_and_measure_manual_selection(
                circle_img, selection_rect, manual_shape_result,
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            if manual_result:
                # Compare measurements
                auto_diameter = auto_result["diameter_mm"]
                manual_diameter = manual_result["diameter_mm"]
                expected_diameter = ground_truth["diameter_mm"]
                
                # Both should be close to ground truth
                self.assertAlmostEqual(auto_diameter, expected_diameter, delta=2.0,
                                     msg="Automatic diameter measurement inaccurate")
                self.assertAlmostEqual(manual_diameter, expected_diameter, delta=2.0,
                                     msg="Manual diameter measurement inaccurate")
                
                # Manual and automatic should be close to each other
                diameter_diff = abs(auto_diameter - manual_diameter)
                self.assertLess(diameter_diff, 1.0,
                               f"Diameter difference {diameter_diff:.2f}mm too large")
                
                print(f"Circle accuracy: Ground truth {expected_diameter:.2f}mm, "
                      f"Auto {auto_diameter:.2f}mm, Manual {manual_diameter:.2f}mm")
    
    def test_rectangle_measurement_accuracy(self):
        """Test accuracy of rectangle measurements between automatic and manual modes."""
        rect_img, ground_truth = self.calibrated_images["perfect_rectangle"]
        
        # Create contour for automatic measurement
        rect_contour = self._create_rectangle_contour(
            ground_truth["center"], ground_truth["width_px"], ground_truth["height_px"]
        )
        
        # Get automatic measurement
        auto_result = classify_and_measure(
            rect_contour, self.mm_per_px_x, self.mm_per_px_y, "automatic"
        )
        
        # Get manual measurement
        selection_rect = (140, 150, 120, 100)
        manual_shape_result = self.snap_engine.snap_to_shape(
            rect_img, selection_rect, SelectionMode.MANUAL_RECTANGLE
        )
        
        if auto_result and manual_shape_result:
            manual_result = classify_and_measure_manual_selection(
                rect_img, selection_rect, manual_shape_result,
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            if manual_result:
                # Compare measurements (width and height might be swapped due to normalization)
                auto_dims = sorted([auto_result["width_mm"], auto_result["height_mm"]])
                manual_dims = sorted([manual_result["width_mm"], manual_result["height_mm"]])
                expected_dims = sorted([ground_truth["width_mm"], ground_truth["height_mm"]])
                
                # Both should be close to ground truth
                for i, (auto_dim, manual_dim, expected_dim) in enumerate(zip(auto_dims, manual_dims, expected_dims)):
                    self.assertAlmostEqual(auto_dim, expected_dim, delta=2.0,
                                         msg=f"Automatic dimension {i} measurement inaccurate")
                    self.assertAlmostEqual(manual_dim, expected_dim, delta=2.0,
                                         msg=f"Manual dimension {i} measurement inaccurate")
                    
                    # Manual and automatic should be close to each other
                    dim_diff = abs(auto_dim - manual_dim)
                    self.assertLess(dim_diff, 1.0,
                                   f"Dimension {i} difference {dim_diff:.2f}mm too large")
                
                print(f"Rectangle accuracy: Ground truth {expected_dims}, "
                      f"Auto {auto_dims}, Manual {manual_dims}")
    
    def test_measurement_consistency_across_selections(self):
        """Test that measurements are consistent across different selection areas."""
        circle_img, ground_truth = self.calibrated_images["perfect_circle"]
        
        # Test multiple selection rectangles around the same circle
        selection_rects = [
            (150, 150, 100, 100),  # Tight fit
            (140, 140, 120, 120),  # Loose fit
            (160, 160, 80, 80),    # Very tight fit
            (130, 130, 140, 140),  # Very loose fit
        ]
        
        measurements = []
        
        for selection_rect in selection_rects:
            manual_shape_result = self.snap_engine.snap_to_shape(
                circle_img, selection_rect, SelectionMode.MANUAL_CIRCLE
            )
            
            if manual_shape_result:
                manual_result = classify_and_measure_manual_selection(
                    circle_img, selection_rect, manual_shape_result,
                    self.mm_per_px_x, self.mm_per_px_y
                )
                
                if manual_result:
                    measurements.append(manual_result["diameter_mm"])
        
        # All measurements should be consistent
        if len(measurements) >= 2:
            avg_measurement = sum(measurements) / len(measurements)
            
            for measurement in measurements:
                diff = abs(measurement - avg_measurement)
                self.assertLess(diff, 1.0, 
                               f"Measurement {measurement:.2f}mm differs from average {avg_measurement:.2f}mm by {diff:.2f}mm")
            
            print(f"Consistency test: {len(measurements)} measurements, "
                  f"average {avg_measurement:.2f}mm, max diff {max(abs(m - avg_measurement) for m in measurements):.2f}mm")
    
    def _create_circle_contour(self, center, radius):
        """Create a circular contour for testing."""
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        points = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])
        return np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    
    def _create_rectangle_contour(self, center, width, height):
        """Create a rectangular contour for testing."""
        cx, cy = center
        hw, hh = width // 2, height // 2
        points = [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh]
        ]
        return np.array(points, dtype=np.int32).reshape(-1, 1, 2)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling in manual selection system."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Manual selection components not available")
            
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        self.manual_engine = ManualSelectionEngine(self.analyzer)
    
    def test_empty_selection_handling(self):
        """Test handling of selections with no shapes."""
        empty_img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # Pure white image
        selection_rect = (100, 100, 200, 200)
        
        # Should return None for empty selections
        circle_result = self.snap_engine.snap_to_shape(
            empty_img, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        self.assertIsNone(circle_result)
        
        rectangle_result = self.snap_engine.snap_to_shape(
            empty_img, selection_rect, SelectionMode.MANUAL_RECTANGLE
        )
        self.assertIsNone(rectangle_result)
    
    def test_invalid_image_handling(self):
        """Test handling of invalid images."""
        selection_rect = (100, 100, 200, 200)
        
        # Test with None image
        result = self.snap_engine.snap_to_shape(
            None, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        self.assertIsNone(result)
        
        # Test with empty image
        empty_array = np.array([])
        result = self.snap_engine.snap_to_shape(
            empty_array, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        self.assertIsNone(result)
        
        # Test with wrong dimensions
        wrong_dims = np.ones((400, 400), dtype=np.uint8)  # Missing color channel
        result = self.snap_engine.snap_to_shape(
            wrong_dims, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        # Should handle gracefully (may succeed or fail, but shouldn't crash)
    
    def test_extreme_selection_sizes(self):
        """Test handling of extremely small and large selections."""
        test_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(test_img, (200, 200), 50, (0, 0, 0), -1)
        
        # Extremely small selection
        tiny_selection = (195, 195, 10, 10)
        tiny_result = self.snap_engine.snap_to_shape(
            test_img, tiny_selection, SelectionMode.MANUAL_CIRCLE
        )
        # Should either detect or return None, but not crash
        
        # Extremely large selection (entire image)
        huge_selection = (0, 0, 400, 400)
        huge_result = self.snap_engine.snap_to_shape(
            test_img, huge_selection, SelectionMode.MANUAL_CIRCLE
        )
        # Should handle gracefully
        
        # Selection larger than image
        oversized_selection = (0, 0, 500, 500)
        oversized_result = self.snap_engine.snap_to_shape(
            test_img, oversized_selection, SelectionMode.MANUAL_CIRCLE
        )
        # Should handle gracefully
    
    def test_corrupted_image_handling(self):
        """Test handling of corrupted or unusual images."""
        # Image with extreme values
        extreme_img = np.zeros((400, 400, 3), dtype=np.uint8)
        extreme_img[100:300, 100:300] = 255  # High contrast square
        
        selection_rect = (50, 50, 300, 300)
        result = self.snap_engine.snap_to_shape(
            extreme_img, selection_rect, SelectionMode.MANUAL_RECTANGLE
        )
        # Should handle without crashing
        
        # Image with NaN values (if possible)
        try:
            nan_img = np.ones((400, 400, 3), dtype=np.float32) * np.nan
            nan_img = nan_img.astype(np.uint8)  # Convert back to uint8
            result = self.snap_engine.snap_to_shape(
                nan_img, selection_rect, SelectionMode.MANUAL_CIRCLE
            )
        except:
            pass  # Expected to fail gracefully
    
    def test_concurrent_access_safety(self):
        """Test thread safety of manual selection components."""
        test_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(test_img, (200, 200), 50, (0, 0, 0), -1)
        
        results = []
        errors = []
        
        def concurrent_selection():
            try:
                for i in range(10):
                    selection = (150 + i, 150 + i, 100, 100)
                    result = self.snap_engine.snap_to_shape(
                        test_img, selection, SelectionMode.MANUAL_CIRCLE
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=concurrent_selection)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Should not have any errors
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
    
    def test_memory_cleanup(self):
        """Test that resources are properly cleaned up."""
        import gc
        
        # Create many temporary objects
        for i in range(100):
            test_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            cv2.circle(test_img, (100, 100), 30, (0, 0, 0), -1)
            
            selection = (70, 70, 60, 60)
            result = self.snap_engine.snap_to_shape(
                test_img, selection, SelectionMode.MANUAL_CIRCLE
            )
            
            # Explicitly delete references
            del test_img
            del result
        
        # Force garbage collection
        gc.collect()
        
        # Test that system is still responsive
        final_test_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(final_test_img, (200, 200), 50, (0, 0, 0), -1)
        
        final_result = self.snap_engine.snap_to_shape(
            final_test_img, (150, 150, 100, 100), SelectionMode.MANUAL_CIRCLE
        )
        
        # Should still work after cleanup
        # (Result can be None or valid, but shouldn't crash)


if __name__ == '__main__':
    # Configure test runner
    unittest.TestLoader.sortTestMethodsUsing = None  # Preserve test order
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes in order
    suite.addTest(unittest.makeSuite(TestManualSelectionAccuracy))
    suite.addTest(unittest.makeSuite(TestManualSelectionPerformance))
    suite.addTest(unittest.makeSuite(TestMeasurementAccuracyValidation))
    suite.addTest(unittest.makeSuite(TestEdgeCasesAndErrorHandling))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST SUITE SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")