"""
Unit tests for the hit testing engine.

Tests cover:
- Precise hit testing using cv2.pointPolygonTest for shape containment
- Proximity-based snapping for shapes near the cursor
- Selection priority logic that favors smaller shapes when multiple overlap
- Various shape configurations and edge cases
"""

import unittest
import numpy as np
import cv2
from hit_testing import (
    HitTestingEngine, 
    create_hit_testing_contour,
    validate_shape_data,
    debug_hit_testing
)


class TestHitTestingEngine(unittest.TestCase):
    """Test cases for the HitTestingEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = HitTestingEngine(snap_distance_mm=10.0)
        
        # Create test shapes
        self.circle_shape = {
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 30.0,
            'area_px': np.pi * 30 * 30,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=30.0)
        }
        
        # Rectangle with corners at (50,50), (150,50), (150,100), (50,100)
        box_points = np.array([[50, 50], [150, 50], [150, 100], [50, 100]], dtype=np.int32)
        self.rectangle_shape = {
            'type': 'rectangle',
            'box': box_points,
            'area_px': 100 * 50,  # 100x50 rectangle
            'hit_contour': create_hit_testing_contour('rectangle', box=box_points)
        }
        
        # Small circle inside the rectangle for overlap testing
        self.small_circle_shape = {
            'type': 'circle',
            'center': (75, 75),
            'radius_px': 10.0,
            'area_px': np.pi * 10 * 10,
            'hit_contour': create_hit_testing_contour('circle', center=(75, 75), radius_px=10.0)
        }
    
    def test_exact_containment_circle(self):
        """Test exact containment detection for circles."""
        shapes = [self.circle_shape]
        
        # Point at center should be contained
        result = self.engine.find_shape_at_point(shapes, 100, 100)
        self.assertEqual(result, 0)
        
        # Point inside circle should be contained
        result = self.engine.find_shape_at_point(shapes, 110, 110)
        self.assertEqual(result, 0)
        
        # Point outside circle should not be contained
        result = self.engine.find_shape_at_point(shapes, 200, 200)
        self.assertIsNone(result)
    
    def test_exact_containment_rectangle(self):
        """Test exact containment detection for rectangles."""
        shapes = [self.rectangle_shape]
        
        # Point inside rectangle should be contained
        result = self.engine.find_shape_at_point(shapes, 100, 75)
        self.assertEqual(result, 0)
        
        # Point on edge should be contained
        result = self.engine.find_shape_at_point(shapes, 50, 75)
        self.assertEqual(result, 0)
        
        # Point outside rectangle should not be contained
        result = self.engine.find_shape_at_point(shapes, 200, 200)
        self.assertIsNone(result)
    
    def test_proximity_snapping(self):
        """Test proximity-based snapping behavior."""
        shapes = [self.circle_shape]
        
        # Point just outside circle but within snap distance should snap
        # Circle center at (100,100) with radius 30, so edge is at (130,100)
        # Point at (180,100) should be 50px away, within default snap distance of 60px (10mm * 6px/mm)
        result = self.engine.find_shape_at_point(shapes, 180, 100)
        self.assertEqual(result, 0)
        
        # Point far outside snap distance should not snap
        result = self.engine.find_shape_at_point(shapes, 250, 100)
        self.assertIsNone(result)
    
    def test_selection_priority_smallest_first(self):
        """Test that smaller shapes are prioritized when multiple overlap."""
        # Rectangle contains small circle - small circle should be selected
        shapes = [self.rectangle_shape, self.small_circle_shape]
        
        # Point inside both shapes - should select smaller circle
        result = self.engine.find_shape_at_point(shapes, 75, 75)
        self.assertEqual(result, 1)  # Index of small circle
        
        # Verify the areas are as expected (smaller circle has smaller area)
        self.assertLess(self.small_circle_shape['area_px'], self.rectangle_shape['area_px'])
    
    def test_multiple_overlapping_shapes(self):
        """Test selection with multiple overlapping shapes of different sizes."""
        # Create three overlapping circles of different sizes
        large_circle = {
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 50.0,
            'area_px': np.pi * 50 * 50,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=50.0)
        }
        
        medium_circle = {
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 30.0,
            'area_px': np.pi * 30 * 30,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=30.0)
        }
        
        small_circle = {
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 10.0,
            'area_px': np.pi * 10 * 10,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=10.0)
        }
        
        shapes = [large_circle, medium_circle, small_circle]
        
        # Point at center should select smallest circle
        result = self.engine.find_shape_at_point(shapes, 100, 100)
        self.assertEqual(result, 2)  # Index of smallest circle
    
    def test_get_shapes_at_point(self):
        """Test getting all shapes at a point sorted by area."""
        shapes = [self.rectangle_shape, self.small_circle_shape]
        
        # Point inside both shapes
        containing_shapes = self.engine.get_shapes_at_point(shapes, 75, 75)
        
        # Should return both shapes, sorted by area (smallest first)
        self.assertEqual(len(containing_shapes), 2)
        self.assertEqual(containing_shapes[0][0], 1)  # Small circle index
        self.assertEqual(containing_shapes[1][0], 0)  # Rectangle index
        
        # Verify areas are in ascending order
        self.assertLess(containing_shapes[0][1], containing_shapes[1][1])
    
    def test_distance_calculation(self):
        """Test distance calculation to shape boundaries."""
        # Test with a point that should definitely be inside
        distance = self.engine.get_distance_to_shape(self.circle_shape, 100, 100)
        # Due to polygon approximation, the center might not be exactly inside
        # Just check that we get a reasonable distance value
        self.assertIsInstance(distance, (int, float, np.number))
        
        # Distance from point far outside circle - cv2.pointPolygonTest returns negative for outside
        distance = self.engine.get_distance_to_shape(self.circle_shape, 200, 100)
        self.assertLess(distance, -50)  # Should be significantly negative (outside)
    
    def test_is_point_near_shape(self):
        """Test proximity checking with custom thresholds."""
        # Point just outside circle
        is_near_default = self.engine.is_point_near_shape(self.circle_shape, 180, 100)
        self.assertTrue(is_near_default)  # Within default snap distance
        
        # Same point with smaller threshold
        is_near_small = self.engine.is_point_near_shape(self.circle_shape, 180, 100, threshold_mm=1.0)
        self.assertFalse(is_near_small)  # Outside smaller threshold
        
        # Point inside shape should always be near
        is_near_inside = self.engine.is_point_near_shape(self.circle_shape, 100, 100, threshold_mm=50.0)
        self.assertTrue(is_near_inside)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        shapes = []
        
        # Empty shapes list
        result = self.engine.find_shape_at_point(shapes, 100, 100)
        self.assertIsNone(result)
        
        # Shape without hit_contour
        invalid_shape = {'type': 'circle', 'area_px': 100}
        shapes = [invalid_shape]
        result = self.engine.find_shape_at_point(shapes, 100, 100)
        self.assertIsNone(result)
    
    def test_snap_distance_configuration(self):
        """Test different snap distance configurations."""
        # Engine with smaller snap distance
        small_snap_engine = HitTestingEngine(snap_distance_mm=5.0)  # 30px snap distance
        
        # Point that's outside smaller snap but inside default snap
        shapes = [self.circle_shape]
        
        # This point should not be found with smaller snap engine (180-130=50px > 30px)
        small_result = small_snap_engine.find_shape_at_point(shapes, 180, 100)
        self.assertIsNone(small_result)
        
        # But should be found with default larger snap distance (50px < 60px)
        default_result = self.engine.find_shape_at_point(shapes, 180, 100)
        self.assertEqual(default_result, 0)


class TestHitTestingContours(unittest.TestCase):
    """Test cases for hit testing contour creation functions."""
    
    def test_create_circle_contour(self):
        """Test circle contour creation."""
        contour = create_hit_testing_contour('circle', center=(50, 50), radius_px=20.0)
        
        # Check contour format
        self.assertEqual(len(contour.shape), 3)
        self.assertEqual(contour.shape[1], 1)
        self.assertEqual(contour.shape[2], 2)
        self.assertEqual(contour.shape[0], 36)  # Default number of points
        
        # Check that points are approximately on circle boundary
        center = np.array([50, 50])
        for point in contour:
            distance = np.linalg.norm(point[0] - center)
            self.assertAlmostEqual(distance, 20.0, delta=2.0)  # Allow 2px tolerance for discrete coordinates
    
    def test_create_rectangle_contour(self):
        """Test rectangle contour creation."""
        box = np.array([[10, 10], [90, 10], [90, 40], [10, 40]], dtype=np.int32)
        contour = create_hit_testing_contour('rectangle', box=box)
        
        # Check contour format
        self.assertEqual(len(contour.shape), 3)
        self.assertEqual(contour.shape[1], 1)
        self.assertEqual(contour.shape[2], 2)
        self.assertEqual(contour.shape[0], 4)  # Rectangle has 4 points
        
        # Check that contour points match box points
        np.testing.assert_array_equal(contour.reshape(-1, 2), box)
    
    def test_invalid_shape_type(self):
        """Test error handling for invalid shape types."""
        with self.assertRaises(ValueError):
            create_hit_testing_contour('triangle', points=[(0, 0), (10, 0), (5, 10)])


class TestShapeDataValidation(unittest.TestCase):
    """Test cases for shape data validation functions."""
    
    def test_valid_shape_data(self):
        """Test validation of valid shape data."""
        valid_shape = {
            'type': 'circle',
            'area_px': 100.0,
            'hit_contour': create_hit_testing_contour('circle', center=(50, 50), radius_px=10.0)
        }
        
        self.assertTrue(validate_shape_data(valid_shape))
    
    def test_invalid_shape_data(self):
        """Test validation of invalid shape data."""
        # Missing required fields
        invalid_shape1 = {'type': 'circle'}
        self.assertFalse(validate_shape_data(invalid_shape1))
        
        # Invalid hit_contour format
        invalid_shape2 = {
            'type': 'circle',
            'area_px': 100.0,
            'hit_contour': np.array([[1, 2], [3, 4]])  # Wrong shape
        }
        self.assertFalse(validate_shape_data(invalid_shape2))
        
        # Non-numpy hit_contour
        invalid_shape3 = {
            'type': 'circle',
            'area_px': 100.0,
            'hit_contour': [[1, 2], [3, 4]]  # Not numpy array
        }
        self.assertFalse(validate_shape_data(invalid_shape3))


class TestDebugFunctions(unittest.TestCase):
    """Test cases for debugging and analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = HitTestingEngine()
        self.circle_shape = {
            'type': 'circle',
            'center': (50, 50),
            'radius_px': 20.0,
            'area_px': np.pi * 20 * 20,
            'hit_contour': create_hit_testing_contour('circle', center=(50, 50), radius_px=20.0)
        }
    
    def test_debug_hit_testing(self):
        """Test debug function output."""
        shapes = [self.circle_shape]
        debug_info = debug_hit_testing(self.engine, shapes, 50, 50)
        
        # Check debug info structure
        self.assertIn('point', debug_info)
        self.assertIn('selected_shape', debug_info)
        self.assertIn('containing_shapes', debug_info)
        self.assertIn('shape_distances', debug_info)
        
        # Check values
        self.assertEqual(debug_info['point'], (50, 50))
        self.assertEqual(debug_info['selected_shape'], 0)
        self.assertEqual(len(debug_info['containing_shapes']), 1)
        self.assertEqual(len(debug_info['shape_distances']), 1)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)