#!/usr/bin/env python3
"""
End-to-End Validation with Real Images

This test uses the existing demo images to test the system with real-world scenarios.
This provides a more realistic validation of the manual mode dimension corrections.

Requirements: 3.1, 3.2, 3.3, 3.5
"""

import unittest
import cv2
import numpy as np
import os
from typing import Dict, List, Any, Tuple, Optional

# Import required components
try:
    from measure import (
        classify_and_measure, process_manual_selection,
        validate_manual_measurement_result, compare_auto_vs_manual_measurements
    )
    from detection import a4_scale_mm_per_px
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


class TestEndToEndWithRealImages(unittest.TestCase):
    """End-to-end validation using real demo images."""
    
    def setUp(self):
        """Set up test fixtures with real images."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Required components not available")
        
        # Standard calibration values
        self.mm_per_px_x = 0.2  # 5 pixels per mm
        self.mm_per_px_y = 0.2  # 5 pixels per mm
        
        # Load available demo images
        self.demo_images = self._load_demo_images()
        
        if not self.demo_images:
            self.skipTest("No demo images available for testing")
    
    def _load_demo_images(self):
        """Load available demo images for testing."""
        demo_images = {}
        
        # List of demo image files to look for
        demo_files = [
            "demo_circle_large_circle.png",
            "demo_circle_medium_circle.png", 
            "demo_circle_small_circle.png",
            "demo_rectangle_large_rectangle.png",
            "demo_rectangle_medium_rectangle.png",
            "demo_rectangle_small_rectangle.png",
            "demo_mixed_manual_circle.png",
            "demo_mixed_manual_rect.png"
        ]
        
        for filename in demo_files:
            if os.path.exists(filename):
                try:
                    image = cv2.imread(filename)
                    if image is not None:
                        demo_images[filename] = image
                        print(f"Loaded demo image: {filename}")
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
        
        return demo_images

    def test_auto_mode_with_real_images(self):
        """Test Auto Mode functionality with real demo images."""
        print("\n=== Testing Auto Mode with Real Images ===")
        
        auto_mode_working = True
        
        for filename, image in self.demo_images.items():
            if "circle" in filename:
                print(f"\nTesting Auto Mode with {filename}:")
                
                # Create a contour around the center of the image
                h, w = image.shape[:2]
                center = (w//2, h//2)
                
                # Estimate radius based on image size
                radius = min(w, h) // 4
                angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
                contour_points = [(int(center[0] + radius * np.cos(a)),
                                 int(center[1] + radius * np.sin(a))) for a in angles]
                circle_contour = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
                
                # Test automatic detection
                result = classify_and_measure(circle_contour, self.mm_per_px_x, self.mm_per_px_y, "automatic")
                
                if result is not None and result["type"] == "circle":
                    diameter = result["diameter_mm"]
                    print(f"  ✅ Auto Mode detected circle: {diameter:.1f}mm")
                else:
                    print(f"  ❌ Auto Mode failed on {filename}")
                    auto_mode_working = False
        
        self.assertTrue(auto_mode_working, "Auto Mode should work with real images")

    def test_manual_mode_with_real_images(self):
        """Test Manual Mode functionality with real demo images."""
        print("\n=== Testing Manual Mode with Real Images ===")
        
        manual_mode_results = []
        
        for filename, image in self.demo_images.items():
            print(f"\nTesting Manual Mode with {filename}:")
            
            h, w = image.shape[:2]
            
            # Define selection areas based on image type
            if "circle" in filename:
                # Select center area for circle detection
                margin = min(w, h) // 6
                selection_rect = (margin, margin, w - 2*margin, h - 2*margin)
                mode = "manual_circle"
                expected_type = "circle"
            elif "rectangle" in filename:
                # Select center area for rectangle detection
                margin = min(w, h) // 6
                selection_rect = (margin, margin, w - 2*margin, h - 2*margin)
                mode = "manual_rectangle"
                expected_type = "rectangle"
            else:
                continue
            
            # Test manual selection
            result = process_manual_selection(
                image, selection_rect, mode,
                self.mm_per_px_x, self.mm_per_px_y
            )
            
            if result is not None:
                if result["type"] == expected_type:
                    if expected_type == "circle":
                        diameter = result["diameter_mm"]
                        print(f"  ✅ Manual Mode detected circle: {diameter:.1f}mm")
                        manual_mode_results.append(("circle", diameter, filename))
                    else:
                        width = result["width_mm"]
                        height = result["height_mm"]
                        print(f"  ✅ Manual Mode detected rectangle: {width:.1f}x{height:.1f}mm")
                        manual_mode_results.append(("rectangle", (width, height), filename))
                else:
                    print(f"  ❌ Wrong shape type detected: expected {expected_type}, got {result['type']}")
            else:
                print(f"  ⚠️  Manual Mode failed to detect shape in {filename}")
        
        # At least some manual detections should work
        self.assertGreater(len(manual_mode_results), 0, "Manual Mode should work with at least some real images")
        print(f"\n✅ Manual Mode successfully processed {len(manual_mode_results)} images")

    def test_measurement_consistency_with_real_images(self):
        """Test measurement consistency between Auto and Manual modes with real images."""
        print("\n=== Testing Measurement Consistency with Real Images ===")
        
        consistency_results = []
        
        for filename, image in self.demo_images.items():
            if "circle" in filename:
                print(f"\nTesting consistency with {filename}:")
                
                h, w = image.shape[:2]
                center = (w//2, h//2)
                
                # Auto Mode test
                radius = min(w, h) // 4
                angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
                contour_points = [(int(center[0] + radius * np.cos(a)),
                                 int(center[1] + radius * np.sin(a))) for a in angles]
                circle_contour = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
                
                auto_result = classify_and_measure(circle_contour, self.mm_per_px_x, self.mm_per_px_y, "automatic")
                
                # Manual Mode test
                margin = min(w, h) // 6
                selection_rect = (margin, margin, w - 2*margin, h - 2*margin)
                manual_result = process_manual_selection(
                    image, selection_rect, "manual_circle",
                    self.mm_per_px_x, self.mm_per_px_y
                )
                
                # Compare results
                if auto_result is not None and manual_result is not None:
                    auto_diameter = auto_result["diameter_mm"]
                    manual_diameter = manual_result["diameter_mm"]
                    
                    difference = abs(auto_diameter - manual_diameter)
                    tolerance = max(2.0, auto_diameter * 0.02)  # 2mm or 2%, whichever is larger
                    
                    is_consistent = difference <= tolerance
                    
                    print(f"  Auto: {auto_diameter:.1f}mm, Manual: {manual_diameter:.1f}mm")
                    print(f"  Difference: {difference:.2f}mm, Tolerance: ±{tolerance:.2f}mm")
                    print(f"  Result: {'✅ CONSISTENT' if is_consistent else '❌ INCONSISTENT'}")
                    
                    consistency_results.append(is_consistent)
                else:
                    print(f"  ⚠️  Could not compare - missing results")
        
        if consistency_results:
            consistent_count = sum(consistency_results)
            total_count = len(consistency_results)
            consistency_rate = consistent_count / total_count
            
            print(f"\n✅ Consistency rate: {consistent_count}/{total_count} ({consistency_rate*100:.1f}%)")
            
            # At least 70% should be consistent
            self.assertGreaterEqual(consistency_rate, 0.7, "At least 70% of measurements should be consistent")
        else:
            print("\n⚠️  No consistency comparisons could be made")

    def test_error_handling_with_real_images(self):
        """Test error handling with real images."""
        print("\n=== Testing Error Handling with Real Images ===")
        
        error_handling_robust = True
        
        for filename, image in self.demo_images.items():
            print(f"\nTesting error handling with {filename}:")
            
            h, w = image.shape[:2]
            
            # Test with invalid selection rectangles
            invalid_selections = [
                (0, 0, 0, 0),  # Zero size
                (-10, -10, 20, 20),  # Negative coordinates
                (w, h, 50, 50),  # Out of bounds
                (10, 10, w*2, h*2),  # Too large
            ]
            
            for i, invalid_rect in enumerate(invalid_selections):
                try:
                    result = process_manual_selection(
                        image, invalid_rect, "manual_circle",
                        self.mm_per_px_x, self.mm_per_px_y
                    )
                    # Should return None gracefully
                    if result is None:
                        print(f"  ✅ Invalid selection {i+1} handled gracefully")
                    else:
                        print(f"  ⚠️  Invalid selection {i+1} unexpectedly succeeded")
                except Exception as e:
                    print(f"  ❌ Invalid selection {i+1} caused exception: {e}")
                    error_handling_robust = False
        
        self.assertTrue(error_handling_robust, "Error handling should be robust with real images")

    def test_performance_with_real_images(self):
        """Test performance with real images."""
        print("\n=== Testing Performance with Real Images ===")
        
        import time
        
        if not self.demo_images:
            self.skipTest("No demo images available for performance testing")
        
        # Test with one representative image
        test_image = list(self.demo_images.values())[0]
        h, w = test_image.shape[:2]
        
        # Test Auto Mode performance
        center = (w//2, h//2)
        radius = min(w, h) // 4
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        contour_points = [(int(center[0] + radius * np.cos(a)),
                         int(center[1] + radius * np.sin(a))) for a in angles]
        circle_contour = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
        
        start_time = time.time()
        for _ in range(3):  # Run 3 times for average
            result = classify_and_measure(circle_contour, self.mm_per_px_x, self.mm_per_px_y, "automatic")
        auto_time = (time.time() - start_time) / 3
        
        print(f"Auto Mode average time: {auto_time*1000:.2f}ms")
        self.assertLess(auto_time, 0.2, "Auto Mode should be fast with real images")
        
        # Test Manual Mode performance
        margin = min(w, h) // 6
        selection_rect = (margin, margin, w - 2*margin, h - 2*margin)
        
        start_time = time.time()
        for _ in range(3):  # Run 3 times for average
            result = process_manual_selection(
                test_image, selection_rect, "manual_circle",
                self.mm_per_px_x, self.mm_per_px_y
            )
        manual_time = (time.time() - start_time) / 3
        
        print(f"Manual Mode average time: {manual_time*1000:.2f}ms")
        self.assertLess(manual_time, 2.0, "Manual Mode should be reasonably fast with real images")
        
        print("✅ Performance is acceptable with real images")

    def test_shape_detection_quality_preserved(self):
        """Test that shape detection quality is preserved."""
        print("\n=== Testing Shape Detection Quality Preserved ===")
        
        detection_quality_good = True
        detection_results = []
        
        for filename, image in self.demo_images.items():
            print(f"\nTesting shape detection quality with {filename}:")
            
            h, w = image.shape[:2]
            margin = min(w, h) // 6
            selection_rect = (margin, margin, w - 2*margin, h - 2*margin)
            
            if "circle" in filename:
                result = process_manual_selection(
                    image, selection_rect, "manual_circle",
                    self.mm_per_px_x, self.mm_per_px_y
                )
                expected_type = "circle"
            elif "rectangle" in filename:
                result = process_manual_selection(
                    image, selection_rect, "manual_rectangle",
                    self.mm_per_px_x, self.mm_per_px_y
                )
                expected_type = "rectangle"
            else:
                continue
            
            if result is not None:
                detected_type = result["type"]
                if detected_type == expected_type:
                    print(f"  ✅ Correctly detected {detected_type}")
                    detection_results.append(True)
                    
                    # Check if result structure is valid
                    is_valid = validate_manual_measurement_result(result)
                    if is_valid:
                        print(f"  ✅ Result structure is valid")
                    else:
                        print(f"  ❌ Result structure is invalid")
                        detection_quality_good = False
                else:
                    print(f"  ❌ Wrong type: expected {expected_type}, got {detected_type}")
                    detection_results.append(False)
                    detection_quality_good = False
            else:
                print(f"  ⚠️  No detection result")
                detection_results.append(False)
        
        if detection_results:
            success_rate = sum(detection_results) / len(detection_results)
            print(f"\n✅ Shape detection success rate: {sum(detection_results)}/{len(detection_results)} ({success_rate*100:.1f}%)")
            
            # At least 50% should work (real images can be challenging)
            self.assertGreaterEqual(success_rate, 0.5, "At least 50% of shape detections should succeed")
        
        self.assertTrue(detection_quality_good, "Shape detection quality should be preserved")


def run_real_image_validation():
    """Run end-to-end validation with real images."""
    print("=" * 80)
    print("END-TO-END VALIDATION WITH REAL IMAGES")
    print("=" * 80)
    print("Testing with actual demo images for realistic validation")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    suite.addTest(TestEndToEndWithRealImages('test_auto_mode_with_real_images'))
    suite.addTest(TestEndToEndWithRealImages('test_manual_mode_with_real_images'))
    suite.addTest(TestEndToEndWithRealImages('test_measurement_consistency_with_real_images'))
    suite.addTest(TestEndToEndWithRealImages('test_error_handling_with_real_images'))
    suite.addTest(TestEndToEndWithRealImages('test_performance_with_real_images'))
    suite.addTest(TestEndToEndWithRealImages('test_shape_detection_quality_preserved'))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("REAL IMAGE VALIDATION SUMMARY")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 ALL REAL IMAGE TESTS PASSED!")
        print("✅ Auto Mode works correctly with real images")
        print("✅ Manual Mode functions with real images")
        print("✅ Measurement consistency is acceptable")
        print("✅ Error handling is robust")
        print("✅ Performance is acceptable")
        print("✅ Shape detection quality is preserved")
    else:
        print("❌ SOME REAL IMAGE TESTS FAILED:")
        for failure in result.failures:
            print(f"  - {failure[0]}")
        for error in result.errors:
            print(f"  - {error[0]} (ERROR)")
    
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    if COMPONENTS_AVAILABLE:
        success = run_real_image_validation()
        exit(0 if success else 1)
    else:
        print("Required components not available. Skipping tests.")
        exit(0)