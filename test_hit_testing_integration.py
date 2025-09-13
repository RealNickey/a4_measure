#!/usr/bin/env python3
"""
Integration test for hit testing engine with various shape configurations.
This script tests the hit testing functionality with realistic shape data.
"""

import unittest
import numpy as np
from hit_testing import HitTestingEngine, create_hit_testing_contour, debug_hit_testing


class TestHitTestingIntegration(unittest.TestCase):
    """Integration tests for hit testing engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = HitTestingEngine()
        self.shapes = self.create_test_shapes()
    
    def create_test_shapes(self):
        """Create a set of test shapes for integration testing."""
        shapes = []
        
        # Large rectangle
        large_rect_box = np.array([[50, 50], [200, 50], [200, 150], [50, 150]], dtype=np.int32)
        shapes.append({
            'type': 'rectangle',
            'width_mm': 25.0,
            'height_mm': 16.7,
            'box': large_rect_box,
            'area_px': 150 * 100,  # 150x100 rectangle
            'hit_contour': create_hit_testing_contour('rectangle', box=large_rect_box),
            'inner': False
        })
        
        # Small circle inside the rectangle
        shapes.append({
            'type': 'circle',
            'center': (125, 100),
            'radius_px': 20.0,
            'diameter_mm': 6.7,
            'area_px': np.pi * 20 * 20,
            'hit_contour': create_hit_testing_contour('circle', center=(125, 100), radius_px=20.0),
            'inner': True
        })
        
        # Medium circle outside the rectangle
        shapes.append({
            'type': 'circle',
            'center': (300, 100),
            'radius_px': 30.0,
            'diameter_mm': 10.0,
            'area_px': np.pi * 30 * 30,
            'hit_contour': create_hit_testing_contour('circle', center=(300, 100), radius_px=30.0),
            'inner': False
        })
        
        # Small rectangle overlapping with large rectangle
        small_rect_box = np.array([[180, 120], [250, 120], [250, 180], [180, 180]], dtype=np.int32)
        shapes.append({
            'type': 'rectangle',
            'width_mm': 11.7,
            'height_mm': 10.0,
            'box': small_rect_box,
            'area_px': 70 * 60,  # 70x60 rectangle
            'hit_contour': create_hit_testing_contour('rectangle', box=small_rect_box),
            'inner': False
        })
        
        return shapes
    
    def test_exact_containment(self):
        """Test exact containment detection."""
        # Test points inside each shape
        test_cases = [
            (125, 100, 1, "Small circle center"),  # Should select small circle (index 1)
            (100, 100, 0, "Large rectangle center"),  # Should select large rectangle (index 0)
            (300, 100, 2, "Medium circle center"),  # Should select medium circle (index 2)
            (215, 150, 3, "Small rectangle center"),  # Should select small rectangle (index 3)
        ]
        
        for x, y, expected_idx, description in test_cases:
            result = self.engine.find_shape_at_point(self.shapes, x, y)
            self.assertEqual(result, expected_idx, f"{description}: Expected {expected_idx}, got {result}")
    
    def test_selection_priority(self):
        """Test selection priority with overlapping shapes."""
        # Point in overlapping area between large rectangle and small rectangle
        x, y = 190, 140
        result = self.engine.find_shape_at_point(self.shapes, x, y)
        
        # Should select smaller rectangle (index 3) over larger rectangle (index 0)
        expected = 3
        self.assertEqual(result, expected, f"Overlapping area: Expected {expected}, got {result}")
        
        # Get all shapes at this point for verification
        all_shapes = self.engine.get_shapes_at_point(self.shapes, x, y)
        self.assertGreater(len(all_shapes), 1, "Should find multiple overlapping shapes")
    
    def test_proximity_snapping(self):
        """Test proximity-based snapping behavior."""
        # Test points near shape boundaries
        test_cases = [
            (145, 100, 1, "Near small circle edge"),  # 20px from center, should snap to circle
            (360, 100, 2, "Near medium circle edge"),  # 30px from center, should snap to circle
            (40, 100, 0, "Near large rectangle edge"),  # Should snap to rectangle
            (400, 100, None, "Far from all shapes"),  # Should not snap to anything
        ]
        
        for x, y, expected_idx, description in test_cases:
            result = self.engine.find_shape_at_point(self.shapes, x, y)
            self.assertEqual(result, expected_idx, f"{description}: Expected {expected_idx}, got {result}")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty point (no shapes nearby)
        result = self.engine.find_shape_at_point(self.shapes, 500, 500)
        self.assertIsNone(result, "Empty area should return None")
        
        # Point exactly on shape boundary
        result = self.engine.find_shape_at_point(self.shapes, 50, 100)  # Left edge of large rectangle
        self.assertEqual(result, 0, "Shape boundary should be detected")
        
        # Test with empty shapes list
        result = self.engine.find_shape_at_point([], 100, 100)
        self.assertIsNone(result, "Empty shapes list should return None")
    
    def test_debug_functionality(self):
        """Test debug and analysis functions."""
        # Debug a point in overlapping area
        debug_info = debug_hit_testing(self.engine, self.shapes, 190, 140)
        
        # Verify debug info structure
        required_keys = ['point', 'selected_shape', 'containing_shapes', 'shape_distances']
        for key in required_keys:
            self.assertIn(key, debug_info, f"Debug info missing key: {key}")
        
        self.assertEqual(debug_info['point'], (190, 140))
        self.assertIsNotNone(debug_info['selected_shape'])
        self.assertGreater(len(debug_info['containing_shapes']), 0)
        self.assertEqual(len(debug_info['shape_distances']), len(self.shapes))


if __name__ == '__main__':
    unittest.main(verbosity=2)

