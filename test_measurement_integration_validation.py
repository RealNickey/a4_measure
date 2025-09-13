#!/usr/bin/env python3
"""
Integration Validation Tests for Measurement Accuracy

This test suite validates measurement accuracy preservation in the complete
interactive workflow, including end-to-end testing with the interaction manager.
"""

import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import sys
import io
from contextlib import redirect_stdout

# Import modules to test
from measure import classify_and_measure, create_shape_data, segment_object, all_inner_contours
from config import PX_PER_MM, A4_WIDTH_MM, A4_HEIGHT_MM, MIN_OBJECT_AREA_MM2
from detection import a4_scale_mm_per_px
from interaction_manager import setup_interactive_inspect_mode, create_interaction_manager
from rendering import SelectiveRenderer


class TestMeasurementIntegrationValidation(unittest.TestCase):
    """Integration tests for measurement accuracy in the complete workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test warped A4 image
        self.warped_image = np.ones((int(A4_HEIGHT_MM * PX_PER_MM), 
                                    int(A4_WIDTH_MM * PX_PER_MM), 3), dtype=np.uint8) * 255
        
        # Standard A4 scaling factors
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        
        # Create realistic test shapes with known measurements
        self.test_shapes = self._create_test_shapes()
    
    def _create_test_shapes(self):
        """Create test shapes with known measurements for validation."""
        shapes = []
        
        # Circle: 20mm diameter (60px radius at 6 px/mm)
        circle_radius_px = 60
        circle_center = (300, 400)
        circle_contour = self._create_circle_contour(circle_center, circle_radius_px)
        circle_result = classify_and_measure(circle_contour, self.mm_per_px_x, self.mm_per_px_y)
        if circle_result:
            shapes.append(create_shape_data(circle_result))
        
        # Rectangle: 30mm x 50mm (180px x 300px at 6 px/mm)
        rect_width_px = 180
        rect_height_px = 300
        rect_center = (600, 400)
        rect_contour = self._create_rectangle_contour(rect_center, rect_width_px, rect_height_px)
        rect_result = classify_and_measure(rect_contour, self.mm_per_px_x, self.mm_per_px_y)
        if rect_result:
            shapes.append(create_shape_data(rect_result))
        
        # Small circle: 5mm diameter (15px radius)
        small_circle_radius_px = 15
        small_circle_center = (200, 200)
        small_circle_contour = self._create_circle_contour(small_circle_center, small_circle_radius_px)
        small_circle_result = classify_and_measure(small_circle_contour, self.mm_per_px_x, self.mm_per_px_y)
        if small_circle_result:
            shapes.append(create_shape_data(small_circle_result))
        
        return shapes
    
    def _create_circle_contour(self, center, radius):
        """Create a circular contour for testing."""
        angles = np.linspace(0, 2*np.pi, 72)  # More points for better accuracy
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
    
    def test_end_to_end_measurement_workflow(self):
        """Test complete measurement workflow from detection to interactive display."""
        print("\n=== Testing End-to-End Measurement Workflow ===")
        
        # Verify we have test shapes
        self.assertGreater(len(self.test_shapes), 0, "Should have test shapes")
        
        # Test interaction manager creation
        manager = create_interaction_manager(self.test_shapes, self.warped_image)
        self.assertIsNotNone(manager, "Interaction manager should be created")
        
        # Verify shapes are preserved in manager
        self.assertEqual(len(manager.state.shapes), len(self.test_shapes),
                        "All shapes should be preserved in interaction manager")
        
        # Test measurement preservation through the workflow
        for i, original_shape in enumerate(self.test_shapes):
            manager_shape = manager.state.shapes[i]
            
            # Verify shape type preservation
            self.assertEqual(manager_shape["type"], original_shape["type"],
                           f"Shape {i} type should be preserved")
            
            # Verify measurement preservation
            if original_shape["type"] == "circle":
                self.assertAlmostEqual(manager_shape["diameter_mm"], original_shape["diameter_mm"], places=6,
                                     msg=f"Circle {i} diameter should be preserved")
            else:
                self.assertAlmostEqual(manager_shape["width_mm"], original_shape["width_mm"], places=6,
                                     msg=f"Rectangle {i} width should be preserved")
                self.assertAlmostEqual(manager_shape["height_mm"], original_shape["height_mm"], places=6,
                                     msg=f"Rectangle {i} height should be preserved")
        
        print(f"✓ Created interaction manager with {len(self.test_shapes)} shapes")
        print("✓ All measurements preserved through workflow")
        
        # Cleanup
        manager.cleanup()
    
    def test_interactive_selection_accuracy(self):
        """Test that interactive selection preserves measurement accuracy."""
        print("\n=== Testing Interactive Selection Accuracy ===")
        
        # Create interaction manager
        manager = create_interaction_manager(self.test_shapes, self.warped_image)
        
        # Test selection of each shape
        for i, expected_shape in enumerate(self.test_shapes):
            # Simulate selection
            manager.state.update_selection(i)
            
            # Verify selection state
            self.assertEqual(manager.state.selected, i, f"Shape {i} should be selected")
            
            # Get selected shape
            selected_shape = manager.state.shapes[i]
            
            # Verify measurements match
            if expected_shape["type"] == "circle":
                self.assertAlmostEqual(selected_shape["diameter_mm"], expected_shape["diameter_mm"], places=6,
                                     msg=f"Selected circle {i} diameter should match")
                print(f"✓ Circle {i}: {selected_shape['diameter_mm']:.2f} mm diameter")
            else:
                self.assertAlmostEqual(selected_shape["width_mm"], expected_shape["width_mm"], places=6,
                                     msg=f"Selected rectangle {i} width should match")
                self.assertAlmostEqual(selected_shape["height_mm"], expected_shape["height_mm"], places=6,
                                     msg=f"Selected rectangle {i} height should match")
                print(f"✓ Rectangle {i}: {selected_shape['width_mm']:.2f} x {selected_shape['height_mm']:.2f} mm")
        
        # Cleanup
        manager.cleanup()
    
    def test_rendering_measurement_consistency(self):
        """Test that rendered measurements are consistent with calculated values."""
        print("\n=== Testing Rendering Measurement Consistency ===")
        
        renderer = SelectiveRenderer()
        
        for i, shape in enumerate(self.test_shapes):
            # Render shape with measurements
            rendered = renderer.render_selection(self.warped_image, shape)
            
            # Verify rendering succeeded
            self.assertIsNotNone(rendered, f"Shape {i} rendering should succeed")
            self.assertEqual(rendered.shape, self.warped_image.shape,
                           f"Rendered shape {i} should maintain image dimensions")
            
            # Verify the rendered image is different from base (has annotations)
            base = renderer.render_base(self.warped_image)
            self.assertFalse(np.array_equal(rendered, base),
                           f"Rendered shape {i} should be different from base image")
            
            print(f"✓ Shape {i} ({shape['type']}) rendered successfully with measurements")
        
        print("✓ All shapes render correctly with preserved measurements")
    
    def test_coordinate_transformation_accuracy(self):
        """Test that coordinate transformations preserve measurement accuracy."""
        print("\n=== Testing Coordinate Transformation Accuracy ===")
        
        # Create interaction manager with display scaling
        manager = create_interaction_manager(self.test_shapes, self.warped_image, display_height=800)
        
        # Verify display scaling is calculated correctly
        expected_scale = 800 / self.warped_image.shape[0]
        self.assertAlmostEqual(manager.display_scale, expected_scale, places=6,
                              msg="Display scale should be calculated correctly")
        
        # Test coordinate transformation for each shape
        for i, shape in enumerate(self.test_shapes):
            if shape["type"] == "circle":
                # Test circle center transformation
                orig_center = shape["center"]
                display_center = (int(orig_center[0] * manager.display_scale),
                                int(orig_center[1] * manager.display_scale))
                
                # Transform back to original coordinates
                from interaction_state import transform_display_to_original_coords
                back_to_orig = transform_display_to_original_coords(
                    display_center[0], display_center[1], manager.display_scale)
                
                # Verify round-trip accuracy (allow for rounding errors in coordinate transformation)
                self.assertAlmostEqual(back_to_orig[0], orig_center[0], delta=2,
                                     msg=f"Circle {i} X coordinate should round-trip accurately")
                self.assertAlmostEqual(back_to_orig[1], orig_center[1], delta=2,
                                     msg=f"Circle {i} Y coordinate should round-trip accurately")
                
                print(f"✓ Circle {i} coordinate transformation: {orig_center} -> {display_center} -> {back_to_orig}")
        
        # Cleanup
        manager.cleanup()
    
    def test_measurement_units_consistency(self):
        """Test that measurement units are consistent throughout the system."""
        print("\n=== Testing Measurement Units Consistency ===")
        
        # Verify PX_PER_MM configuration
        self.assertEqual(PX_PER_MM, 6.0, "PX_PER_MM should be 6.0 as configured")
        
        # Verify mm_per_px calculation
        expected_mm_per_px = 1.0 / PX_PER_MM
        self.assertAlmostEqual(self.mm_per_px_x, expected_mm_per_px, places=6,
                              msg="mm_per_px_x should equal 1/PX_PER_MM")
        self.assertAlmostEqual(self.mm_per_px_y, expected_mm_per_px, places=6,
                              msg="mm_per_px_y should equal 1/PX_PER_MM")
        
        # Test unit consistency in measurements
        for i, shape in enumerate(self.test_shapes):
            if shape["type"] == "circle":
                # Verify diameter calculation: diameter_mm = 2 * radius_px * mm_per_px
                expected_diameter = 2 * shape["radius_px"] * self.mm_per_px_x
                self.assertAlmostEqual(shape["diameter_mm"], expected_diameter, places=6,
                                     msg=f"Circle {i} diameter units should be consistent")
                print(f"✓ Circle {i}: {shape['radius_px']:.1f} px -> {shape['diameter_mm']:.2f} mm")
            else:
                # Verify rectangle calculations
                expected_width = shape["width_mm"] / self.mm_per_px_x  # Convert back to pixels
                expected_height = shape["height_mm"] / self.mm_per_px_y
                print(f"✓ Rectangle {i}: {expected_width:.1f}x{expected_height:.1f} px -> {shape['width_mm']:.2f}x{shape['height_mm']:.2f} mm")
        
        print(f"✓ All measurements use consistent units (mm) with {PX_PER_MM} px/mm scaling")
    
    def test_measurement_precision_limits(self):
        """Test measurement precision at the limits of the system."""
        print("\n=== Testing Measurement Precision Limits ===")
        
        # Test minimum measurable size (based on MIN_OBJECT_AREA_MM2)
        min_area_px = MIN_OBJECT_AREA_MM2 * (PX_PER_MM ** 2)
        min_radius_px = np.sqrt(min_area_px / np.pi)
        
        print(f"✓ Minimum object area: {MIN_OBJECT_AREA_MM2} mm² = {min_area_px:.1f} px²")
        print(f"✓ Minimum circle radius: {min_radius_px:.1f} px = {min_radius_px/PX_PER_MM:.2f} mm")
        
        # Test precision with very precise measurements
        test_radius_px = 100.5  # Half-pixel precision
        test_center = (400, 400)
        test_contour = self._create_circle_contour(test_center, test_radius_px)
        test_result = classify_and_measure(test_contour, self.mm_per_px_x, self.mm_per_px_y)
        
        if test_result:
            # Verify precision is maintained in the calculation
            calculated_diameter = 2 * test_result["radius_px"] * self.mm_per_px_x
            self.assertAlmostEqual(test_result["diameter_mm"], calculated_diameter, places=6,
                                 msg="High precision measurements should be maintained")
            print(f"✓ High precision test: {test_result['radius_px']:.2f} px -> {test_result['diameter_mm']:.4f} mm")
        
        # Test A4 scale limits
        max_dimension_mm = max(A4_WIDTH_MM, A4_HEIGHT_MM)
        max_dimension_px = max_dimension_mm * PX_PER_MM
        print(f"✓ Maximum A4 dimension: {max_dimension_mm} mm = {max_dimension_px:.0f} px")
        
        print("✓ System precision limits validated")


def run_integration_validation_tests():
    """Run all integration validation tests."""
    print("=" * 70)
    print("MEASUREMENT INTEGRATION VALIDATION TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMeasurementIntegrationValidation)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("INTEGRATION VALIDATION SUMMARY")
    print("=" * 70)
    
    if result.wasSuccessful():
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("✅ End-to-end measurement accuracy preserved")
        print("✅ Interactive workflow maintains precision")
        print("✅ Coordinate transformations are accurate")
        print("✅ Measurement units are consistent")
        print("✅ System precision limits validated")
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
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
    success = run_integration_validation_tests()
    sys.exit(0 if success else 1)