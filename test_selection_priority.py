"""
Unit tests for selection priority logic with overlapping shapes.

Tests cover:
- Selection priority based on shape area (smallest first)
- Complex overlapping scenarios with multiple shapes
- Edge cases with identical areas
- Priority consistency across different shape types
- Performance with many overlapping shapes
"""

import unittest
import numpy as np
from hit_testing import HitTestingEngine, create_hit_testing_contour


class TestSelectionPriority(unittest.TestCase):
    """Test cases for selection priority logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = HitTestingEngine()
    
    def create_overlapping_circles(self):
        """Create a set of overlapping circles with different sizes."""
        circles = []
        
        # Large circle (area = π * 50²)
        circles.append({
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 50.0,
            'area_px': np.pi * 50 * 50,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=50.0)
        })
        
        # Medium circle (area = π * 30²)
        circles.append({
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 30.0,
            'area_px': np.pi * 30 * 30,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=30.0)
        })
        
        # Small circle (area = π * 15²)
        circles.append({
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 15.0,
            'area_px': np.pi * 15 * 15,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=15.0)
        })
        
        return circles
    
    def create_overlapping_rectangles(self):
        """Create a set of overlapping rectangles with different sizes."""
        rectangles = []
        
        # Large rectangle (area = 200 * 150)
        large_box = np.array([[0, 0], [200, 0], [200, 150], [0, 150]], dtype=np.int32)
        rectangles.append({
            'type': 'rectangle',
            'box': large_box,
            'area_px': 200 * 150,
            'hit_contour': create_hit_testing_contour('rectangle', box=large_box)
        })
        
        # Medium rectangle (area = 120 * 80)
        medium_box = np.array([[40, 35], [160, 35], [160, 115], [40, 115]], dtype=np.int32)
        rectangles.append({
            'type': 'rectangle',
            'box': medium_box,
            'area_px': 120 * 80,
            'hit_contour': create_hit_testing_contour('rectangle', box=medium_box)
        })
        
        # Small rectangle (area = 60 * 40)
        small_box = np.array([[70, 55], [130, 55], [130, 95], [70, 95]], dtype=np.int32)
        rectangles.append({
            'type': 'rectangle',
            'box': small_box,
            'area_px': 60 * 40,
            'hit_contour': create_hit_testing_contour('rectangle', box=small_box)
        })
        
        return rectangles
    
    def create_mixed_overlapping_shapes(self):
        """Create overlapping shapes of different types."""
        shapes = []
        
        # Large rectangle
        large_box = np.array([[50, 50], [200, 50], [200, 150], [50, 150]], dtype=np.int32)
        shapes.append({
            'type': 'rectangle',
            'box': large_box,
            'area_px': 150 * 100,  # 15000
            'hit_contour': create_hit_testing_contour('rectangle', box=large_box)
        })
        
        # Medium circle (smaller area than rectangle)
        shapes.append({
            'type': 'circle',
            'center': (125, 100),
            'radius_px': 40.0,
            'area_px': np.pi * 40 * 40,  # ≈ 5027
            'hit_contour': create_hit_testing_contour('circle', center=(125, 100), radius_px=40.0)
        })
        
        # Small rectangle (smallest area)
        small_box = np.array([[100, 80], [150, 80], [150, 120], [100, 120]], dtype=np.int32)
        shapes.append({
            'type': 'rectangle',
            'box': small_box,
            'area_px': 50 * 40,  # 2000
            'hit_contour': create_hit_testing_contour('rectangle', box=small_box)
        })
        
        return shapes
    
    def test_concentric_circles_priority(self):
        """Test priority selection with concentric circles."""
        circles = self.create_overlapping_circles()
        
        # Point at center should select smallest circle (index 2)
        result = self.engine.find_shape_at_point(circles, 100, 100)
        self.assertEqual(result, 2, "Should select smallest circle at center")
        
        # Point in medium circle but outside small circle
        result = self.engine.find_shape_at_point(circles, 120, 100)  # 20px from center
        self.assertEqual(result, 1, "Should select medium circle in its exclusive area")
        
        # Point in large circle but outside medium circle
        result = self.engine.find_shape_at_point(circles, 140, 100)  # 40px from center
        self.assertEqual(result, 0, "Should select large circle in its exclusive area")
    
    def test_nested_rectangles_priority(self):
        """Test priority selection with nested rectangles."""
        rectangles = self.create_overlapping_rectangles()
        
        # Point in all three rectangles should select smallest
        result = self.engine.find_shape_at_point(rectangles, 100, 75)
        self.assertEqual(result, 2, "Should select smallest rectangle in overlapping area")
        
        # Point in large and medium but not small
        result = self.engine.find_shape_at_point(rectangles, 50, 50)
        self.assertEqual(result, 1, "Should select medium rectangle when small is not available")
        
        # Point only in large rectangle
        result = self.engine.find_shape_at_point(rectangles, 20, 20)
        self.assertEqual(result, 0, "Should select large rectangle in its exclusive area")
    
    def test_mixed_shape_types_priority(self):
        """Test priority selection with different shape types."""
        shapes = self.create_mixed_overlapping_shapes()
        
        # Point in all three shapes should select smallest (small rectangle, index 2)
        result = self.engine.find_shape_at_point(shapes, 125, 100)
        self.assertEqual(result, 2, "Should select smallest shape regardless of type")
        
        # Point in rectangle and circle but not small rectangle
        result = self.engine.find_shape_at_point(shapes, 90, 100)
        self.assertEqual(result, 1, "Should select circle over rectangle when it's smaller")
        
        # Point only in large rectangle
        result = self.engine.find_shape_at_point(shapes, 60, 60)
        self.assertEqual(result, 0, "Should select large rectangle in its exclusive area")
    
    def test_get_shapes_at_point_ordering(self):
        """Test that get_shapes_at_point returns shapes in correct order."""
        shapes = self.create_mixed_overlapping_shapes()
        
        # Point in all three shapes
        containing_shapes = self.engine.get_shapes_at_point(shapes, 125, 100)
        
        # Should return all three shapes, ordered by area (smallest first)
        self.assertEqual(len(containing_shapes), 3)
        
        # Verify ordering by area
        areas = [area for _, area in containing_shapes]
        self.assertEqual(areas, sorted(areas), "Shapes should be ordered by area (smallest first)")
        
        # Verify the specific order
        self.assertEqual(containing_shapes[0][0], 2)  # Small rectangle
        self.assertEqual(containing_shapes[1][0], 1)  # Medium circle
        self.assertEqual(containing_shapes[2][0], 0)  # Large rectangle
    
    def test_identical_areas_priority(self):
        """Test priority when shapes have identical areas."""
        # Create two circles with identical areas
        identical_circles = [
            {
                'type': 'circle',
                'center': (100, 100),
                'radius_px': 25.0,
                'area_px': np.pi * 25 * 25,
                'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=25.0)
            },
            {
                'type': 'circle',
                'center': (100, 100),
                'radius_px': 25.0,
                'area_px': np.pi * 25 * 25,  # Identical area
                'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=25.0)
            }
        ]
        
        # Should consistently select the first one (stable sort)
        result = self.engine.find_shape_at_point(identical_circles, 100, 100)
        self.assertEqual(result, 0, "Should consistently select first shape when areas are identical")
        
        # Test multiple times to ensure consistency
        for _ in range(10):
            result = self.engine.find_shape_at_point(identical_circles, 100, 100)
            self.assertEqual(result, 0, "Selection should be consistent with identical areas")
    
    def test_area_calculation_accuracy(self):
        """Test that area calculations are accurate for priority decisions."""
        # Create shapes with precisely calculated areas
        shapes = []
        
        # Circle with radius 20 (area = π * 400 ≈ 1256.64)
        shapes.append({
            'type': 'circle',
            'center': (100, 100),
            'radius_px': 20.0,
            'area_px': np.pi * 20 * 20,
            'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=20.0)
        })
        
        # Rectangle 35x36 (area = 1260, slightly larger than circle)
        rect_box = np.array([[82, 82], [117, 82], [117, 118], [82, 118]], dtype=np.int32)
        shapes.append({
            'type': 'rectangle',
            'box': rect_box,
            'area_px': 35 * 36,  # 1260
            'hit_contour': create_hit_testing_contour('rectangle', box=rect_box)
        })
        
        # Point in overlapping area should select circle (smaller area)
        result = self.engine.find_shape_at_point(shapes, 100, 100)
        self.assertEqual(result, 0, "Should select circle with smaller area")
        
        # Verify the area comparison is correct
        circle_area = shapes[0]['area_px']
        rect_area = shapes[1]['area_px']
        self.assertLess(circle_area, rect_area, "Circle should have smaller area than rectangle")
    
    def test_complex_overlapping_scenario(self):
        """Test complex scenario with many overlapping shapes."""
        shapes = []
        
        # Create 5 overlapping shapes with different areas
        areas_and_shapes = [
            (10000, 'large_rect'),
            (5000, 'medium_rect'),
            (3000, 'circle'),
            (1500, 'small_rect'),
            (500, 'tiny_circle')
        ]
        
        for i, (area, shape_type) in enumerate(areas_and_shapes):
            if 'rect' in shape_type:
                # Create rectangle with approximately the target area
                width = int(np.sqrt(area * 1.5))
                height = int(area / width)
                x1, y1 = 100 - width//2, 100 - height//2
                x2, y2 = x1 + width, y1 + height
                box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                
                shapes.append({
                    'type': 'rectangle',
                    'box': box,
                    'area_px': float(area),
                    'hit_contour': create_hit_testing_contour('rectangle', box=box)
                })
            else:
                # Create circle with approximately the target area
                radius = np.sqrt(area / np.pi)
                shapes.append({
                    'type': 'circle',
                    'center': (100, 100),
                    'radius_px': radius,
                    'area_px': float(area),
                    'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=radius)
                })
        
        # Point at center should select smallest shape (index 4 - tiny circle)
        result = self.engine.find_shape_at_point(shapes, 100, 100)
        self.assertEqual(result, 4, "Should select smallest shape in complex overlapping scenario")
        
        # Verify all shapes are detected at the center point
        containing_shapes = self.engine.get_shapes_at_point(shapes, 100, 100)
        self.assertEqual(len(containing_shapes), 5, "All shapes should contain the center point")
        
        # Verify correct ordering
        for i in range(len(containing_shapes) - 1):
            current_area = containing_shapes[i][1]
            next_area = containing_shapes[i + 1][1]
            self.assertLessEqual(current_area, next_area, 
                               f"Shape areas should be in ascending order: {current_area} <= {next_area}")
    
    def test_priority_with_inner_shapes(self):
        """Test priority logic with inner shapes (circles inside rectangles)."""
        shapes = []
        
        # Large rectangle
        large_box = np.array([[50, 50], [200, 50], [200, 150], [50, 150]], dtype=np.int32)
        shapes.append({
            'type': 'rectangle',
            'area_px': 150 * 100,
            'inner': False,
            'hit_contour': create_hit_testing_contour('rectangle', box=large_box)
        })
        
        # Inner circle (should be prioritized as it's smaller)
        shapes.append({
            'type': 'circle',
            'center': (125, 100),
            'radius_px': 20.0,
            'area_px': np.pi * 20 * 20,
            'inner': True,
            'hit_contour': create_hit_testing_contour('circle', center=(125, 100), radius_px=20.0)
        })
        
        # Point in both shapes should select inner circle
        result = self.engine.find_shape_at_point(shapes, 125, 100)
        self.assertEqual(result, 1, "Should select inner circle over outer rectangle")
    
    def test_performance_with_many_overlapping_shapes(self):
        """Test performance and correctness with many overlapping shapes."""
        import time
        
        # Create 20 overlapping circles with random sizes
        np.random.seed(42)  # For reproducible results
        shapes = []
        
        for i in range(20):
            radius = np.random.uniform(10, 50)
            area = np.pi * radius * radius
            
            shapes.append({
                'type': 'circle',
                'center': (100, 100),
                'radius_px': radius,
                'area_px': area,
                'hit_contour': create_hit_testing_contour('circle', center=(100, 100), radius_px=radius)
            })
        
        # Time the selection process
        start_time = time.time()
        result = self.engine.find_shape_at_point(shapes, 100, 100)
        end_time = time.time()
        
        # Should complete quickly (under 10ms)
        elapsed_time = end_time - start_time
        self.assertLess(elapsed_time, 0.01, f"Selection took too long: {elapsed_time:.4f}s")
        
        # Should select the shape with smallest area
        selected_area = shapes[result]['area_px']
        all_areas = [shape['area_px'] for shape in shapes]
        min_area = min(all_areas)
        
        self.assertEqual(selected_area, min_area, "Should select shape with minimum area")
        
        # Verify get_shapes_at_point returns all shapes in correct order
        containing_shapes = self.engine.get_shapes_at_point(shapes, 100, 100)
        self.assertEqual(len(containing_shapes), 20, "Should find all overlapping shapes")
        
        # Verify ordering
        areas = [area for _, area in containing_shapes]
        self.assertEqual(areas, sorted(areas), "Areas should be in ascending order")


if __name__ == '__main__':
    unittest.main(verbosity=2)