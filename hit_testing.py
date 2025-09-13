"""
Hit Testing Engine for Interactive Shape Selection

This module provides precise hit testing functionality using cv2.pointPolygonTest
for shape containment detection, proximity-based snapping for shapes near the cursor,
and selection priority logic that favors smaller shapes when multiple overlap.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from config import PX_PER_MM


class HitTestingEngine:
    """
    Engine for precise hit testing and shape selection with snapping behavior.
    
    Provides methods for:
    - Precise hit testing using cv2.pointPolygonTest
    - Proximity-based snapping for nearby shapes
    - Selection priority logic favoring smaller shapes
    """
    
    def __init__(self, snap_distance_mm: float = 10.0):
        """
        Initialize the hit testing engine.
        
        Args:
            snap_distance_mm: Distance threshold in millimeters for snapping behavior
        """
        self.snap_distance_px = int(snap_distance_mm * PX_PER_MM)
    
    def find_shape_at_point(self, shapes: List[Dict[str, Any]], x: int, y: int) -> Optional[int]:
        """
        Find the most appropriate shape at or near the given point.
        
        Uses a two-phase approach:
        1. Check for exact containment (point inside shape)
        2. Check for proximity-based snapping (point near shape boundary)
        
        When multiple shapes contain the point, prioritizes the smallest shape
        for more precise selection.
        
        Args:
            shapes: List of shape data dictionaries with 'hit_contour' and 'area_px'
            x: X coordinate in image space
            y: Y coordinate in image space
            
        Returns:
            Index of the selected shape, or None if no shape is found
        """
        point = (x, y)
        
        # Phase 1: Check for exact containment
        containment_candidates = []
        
        for i, shape in enumerate(shapes):
            if 'hit_contour' not in shape:
                continue
                
            # Use cv2.pointPolygonTest for precise containment checking
            distance = cv2.pointPolygonTest(shape['hit_contour'], point, True)
            
            if distance >= 0:  # Point is inside or on the boundary
                area = shape.get('area_px', float('inf'))
                containment_candidates.append((area, i, distance))
        
        if containment_candidates:
            # Return smallest shape containing the point (most specific selection)
            containment_candidates.sort(key=lambda candidate: candidate[0])
            return containment_candidates[0][1]
        
        # Phase 2: Check for proximity-based snapping
        return self._find_nearest_shape_within_snap_distance(shapes, point)
    
    def _find_nearest_shape_within_snap_distance(self, shapes: List[Dict[str, Any]], point: Tuple[int, int]) -> Optional[int]:
        """
        Find the nearest shape within snapping distance.
        
        Args:
            shapes: List of shape data dictionaries
            point: (x, y) coordinate tuple
            
        Returns:
            Index of the nearest shape within snap distance, or None
        """
        min_distance = float('inf')
        closest_shape_idx = None
        
        for i, shape in enumerate(shapes):
            if 'hit_contour' not in shape:
                continue
            
            # Get absolute distance to shape boundary
            distance = abs(cv2.pointPolygonTest(shape['hit_contour'], point, True))
            
            if distance <= self.snap_distance_px and distance < min_distance:
                min_distance = distance
                closest_shape_idx = i
        
        return closest_shape_idx
    
    def get_shapes_at_point(self, shapes: List[Dict[str, Any]], x: int, y: int) -> List[Tuple[int, float]]:
        """
        Get all shapes that contain the given point, sorted by area (smallest first).
        
        Useful for debugging and understanding selection priority.
        
        Args:
            shapes: List of shape data dictionaries
            x: X coordinate in image space
            y: Y coordinate in image space
            
        Returns:
            List of (shape_index, area) tuples for shapes containing the point,
            sorted by area (smallest first)
        """
        point = (x, y)
        containing_shapes = []
        
        for i, shape in enumerate(shapes):
            if 'hit_contour' not in shape:
                continue
                
            distance = cv2.pointPolygonTest(shape['hit_contour'], point, True)
            
            if distance >= 0:  # Point is inside or on boundary
                area = shape.get('area_px', float('inf'))
                containing_shapes.append((i, area))
        
        # Sort by area (smallest first for priority selection)
        containing_shapes.sort(key=lambda item: item[1])
        return containing_shapes
    
    def get_distance_to_shape(self, shape: Dict[str, Any], x: int, y: int) -> float:
        """
        Get the distance from a point to a shape's boundary.
        
        Args:
            shape: Shape data dictionary with 'hit_contour'
            x: X coordinate in image space
            y: Y coordinate in image space
            
        Returns:
            Distance to shape boundary (positive if outside, negative if inside)
        """
        if 'hit_contour' not in shape:
            return float('inf')
        
        return cv2.pointPolygonTest(shape['hit_contour'], (x, y), True)
    
    def is_point_near_shape(self, shape: Dict[str, Any], x: int, y: int, threshold_mm: float = None) -> bool:
        """
        Check if a point is within a specified distance of a shape.
        
        Args:
            shape: Shape data dictionary with 'hit_contour'
            x: X coordinate in image space
            y: Y coordinate in image space
            threshold_mm: Distance threshold in millimeters (uses snap_distance if None)
            
        Returns:
            True if point is within threshold distance of the shape
        """
        if threshold_mm is None:
            threshold_px = self.snap_distance_px
        else:
            threshold_px = int(threshold_mm * PX_PER_MM)
        
        distance = abs(self.get_distance_to_shape(shape, x, y))
        return distance <= threshold_px


def create_hit_testing_contour(shape_type: str, **kwargs) -> np.ndarray:
    """
    Create a hit testing contour for a given shape type.
    
    Args:
        shape_type: Either "circle" or "rectangle"
        **kwargs: Shape-specific parameters
            For circles: center (tuple), radius_px (float)
            For rectangles: box (np.ndarray of 4 points)
    
    Returns:
        Contour array suitable for cv2.pointPolygonTest
    """
    if shape_type == "circle":
        return _create_circle_hit_contour(kwargs['center'], kwargs['radius_px'])
    elif shape_type == "rectangle":
        return _create_rectangle_hit_contour(kwargs['box'])
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")


def _create_circle_hit_contour(center: Tuple[int, int], radius_px: float, num_points: int = 36) -> np.ndarray:
    """
    Generate hit testing polygon for a circle using multiple points.
    
    Args:
        center: (x, y) center coordinates
        radius_px: Radius in pixels
        num_points: Number of points to use for polygon approximation
        
    Returns:
        Contour array in the format expected by cv2.pointPolygonTest
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    
    for angle in angles:
        x = int(center[0] + radius_px * np.cos(angle))
        y = int(center[1] + radius_px * np.sin(angle))
        points.append([x, y])
    
    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)


def _create_rectangle_hit_contour(box: np.ndarray) -> np.ndarray:
    """
    Generate hit testing polygon for a rectangle from box points.
    
    Args:
        box: Array of 4 corner points from cv2.boxPoints()
        
    Returns:
        Contour array in the format expected by cv2.pointPolygonTest
    """
    return box.reshape(-1, 1, 2).astype(np.int32)


# Utility functions for shape data validation and debugging

def validate_shape_data(shape: Dict[str, Any]) -> bool:
    """
    Validate that a shape data dictionary has the required fields for hit testing.
    
    Args:
        shape: Shape data dictionary
        
    Returns:
        True if shape data is valid for hit testing
    """
    required_fields = ['type', 'hit_contour', 'area_px']
    
    for field in required_fields:
        if field not in shape:
            return False
    
    # Validate hit_contour format
    hit_contour = shape['hit_contour']
    if not isinstance(hit_contour, np.ndarray):
        return False
    
    if len(hit_contour.shape) != 3 or hit_contour.shape[1] != 1 or hit_contour.shape[2] != 2:
        return False
    
    return True


def debug_hit_testing(engine: HitTestingEngine, shapes: List[Dict[str, Any]], x: int, y: int) -> Dict[str, Any]:
    """
    Debug function to analyze hit testing results at a specific point.
    
    Args:
        engine: HitTestingEngine instance
        shapes: List of shape data dictionaries
        x: X coordinate to test
        y: Y coordinate to test
        
    Returns:
        Dictionary with debug information about hit testing at the point
    """
    result = {
        'point': (x, y),
        'selected_shape': engine.find_shape_at_point(shapes, x, y),
        'containing_shapes': engine.get_shapes_at_point(shapes, x, y),
        'shape_distances': []
    }
    
    for i, shape in enumerate(shapes):
        if validate_shape_data(shape):
            distance = engine.get_distance_to_shape(shape, x, y)
            is_near = engine.is_point_near_shape(shape, x, y)
            result['shape_distances'].append({
                'shape_index': i,
                'distance': distance,
                'is_near': is_near,
                'area_px': shape.get('area_px', 0)
            })
    
    return result