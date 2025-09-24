"""
Unit tests for ShapeSnappingEngine

Tests shape detection accuracy, scoring consistency, and validation functionality
for both circle and rectangle detection within manual selections.
"""

import unittest
import numpy as np
import cv2
import math
from unittest.mock import Mock, patch

from shape_snapping_engine import ShapeSnappingEngine, ShapeCandidate
from enhanced_contour_analyzer import EnhancedContourAnalyzer
from selection_mode import SelectionMode


class TestShapeSnappingEngine(unittest.TestCase):
    """Test cases for ShapeSnappingEngine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Mock(spec=EnhancedContourAnalyzer)
        self.engine = ShapeSnappingEngine(self.analyzer)
        
        # Create test images
        self.test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        self.test_selection_rect = (100, 100, 200, 200)  # x, y, w, h
        
        # Create test contours
        self.circle_contour = self._create_test_circle_contour((200, 200), 50)
        self.rectangle_contour = self._create_test_rectangle_contour((200, 200), 80, 60)
        self.invalid_contour = np.array([[[150, 150]], [[151, 150]], [[151, 151]]], dtype=np.int32)
    
    def _create_test_circle_contour(self, center, radius):
        """Create a circular contour for testing."""
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        points = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])
        return np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    
    def _create_test_rectangle_contour(self, center, width, height):
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
    
    def test_initialization(self):
        """Test engine initialization with default parameters."""
        self.assertEqual(self.engine.analyzer, self.analyzer)
        self.assertEqual(self.engine.min_contour_area, 100)
        self.assertEqual(self.engine.max_contour_area_ratio, 0.8)
        self.assertEqual(self.engine.min_circularity, 0.6)
        self.assertEqual(self.engine.min_rectangularity, 0.7)
        
        # Test scoring weights sum to reasonable value
        total_weight = (self.engine.size_weight + 
                       self.engine.position_weight + 
                       self.engine.quality_weight)
        self.assertAlmostEqual(total_weight, 1.0, places=1)
    
    def test_validate_selection_rect(self):
        """Test selection rectangle validation."""
        # Valid selection
        valid_rect = (50, 50, 100, 100)
        self.assertTrue(self.engine._validate_selection_rect(self.test_image, valid_rect))
        
        # Invalid selections
        self.assertFalse(self.engine._validate_selection_rect(self.test_image, (-10, 50, 100, 100)))  # Negative x
        self.assertFalse(self.engine._validate_selection_rect(self.test_image, (50, -10, 100, 100)))  # Negative y
        self.assertFalse(self.engine._validate_selection_rect(self.test_image, (350, 50, 100, 100)))  # Out of bounds
        self.assertFalse(self.engine._validate_selection_rect(self.test_image, (50, 50, 10, 100)))    # Too small width
        self.assertFalse(self.engine._validate_selection_rect(self.test_image, (50, 50, 100, 10)))    # Too small height
        
        # Test with None image
        self.assertFalse(self.engine._validate_selection_rect(None, valid_rect))
    
    def test_extract_roi(self):
        """Test region of interest extraction."""
        # Valid extraction
        roi = self.engine._extract_roi(self.test_image, self.test_selection_rect)
        self.assertIsNotNone(roi)
        self.assertEqual(roi.shape, (200, 200, 3))
        
        # Invalid extraction (out of bounds)
        invalid_rect = (350, 350, 100, 100)
        roi = self.engine._extract_roi(self.test_image, invalid_rect)
        self.assertIsNotNone(roi)  # Should still work but with adjusted dimensions
        
        # Zero-size extraction
        zero_rect = (100, 100, 0, 0)
        roi = self.engine._extract_roi(self.test_image, zero_rect)
        self.assertIsNone(roi)
    
    def test_calculate_circularity(self):
        """Test circularity calculation."""
        # Perfect circle should have high circularity
        circularity = self.engine._calculate_circularity(self.circle_contour)
        self.assertGreater(circularity, 0.8)
        self.assertLessEqual(circularity, 1.0)
        
        # Rectangle should have lower circularity
        rect_circularity = self.engine._calculate_circularity(self.rectangle_contour)
        self.assertLess(rect_circularity, circularity)
        
        # Invalid contour
        invalid_circularity = self.engine._calculate_circularity(self.invalid_contour)
        self.assertGreaterEqual(invalid_circularity, 0.0)
        self.assertLessEqual(invalid_circularity, 1.0)
    
    def test_calculate_rectangularity(self):
        """Test rectangularity calculation."""
        # Rectangle should have high rectangularity
        rectangularity = self.engine._calculate_rectangularity(self.rectangle_contour)
        self.assertGreater(rectangularity, 0.7)
        self.assertLessEqual(rectangularity, 1.0)
        
        # Circle should have lower rectangularity
        circle_rectangularity = self.engine._calculate_rectangularity(self.circle_contour)
        self.assertLess(circle_rectangularity, rectangularity)
        
        # Invalid contour
        invalid_rectangularity = self.engine._calculate_rectangularity(self.invalid_contour)
        self.assertGreaterEqual(invalid_rectangularity, 0.0)
        self.assertLessEqual(invalid_rectangularity, 1.0)
    
    def test_calculate_size_score(self):
        """Test size scoring algorithm."""
        selection_area = 40000  # 200x200
        
        # Optimal size (30% of selection)
        optimal_area = selection_area * 0.3
        optimal_score = self.engine._calculate_size_score(optimal_area, selection_area)
        self.assertEqual(optimal_score, 1.0)
        
        # Small size (5% of selection)
        small_area = selection_area * 0.05
        small_score = self.engine._calculate_size_score(small_area, selection_area)
        self.assertLess(small_score, optimal_score)
        self.assertGreater(small_score, 0.0)
        
        # Large size (80% of selection)
        large_area = selection_area * 0.8
        large_score = self.engine._calculate_size_score(large_area, selection_area)
        self.assertLess(large_score, optimal_score)
        
        # Zero selection area
        zero_score = self.engine._calculate_size_score(1000, 0)
        self.assertEqual(zero_score, 0.0)
    
    def test_calculate_position_score(self):
        """Test position scoring algorithm."""
        selection_rect = (100, 100, 200, 200)
        selection_center = (200, 200)
        
        # Centered shape should get perfect score
        centered_score = self.engine._calculate_position_score(
            selection_center, selection_center, selection_rect
        )
        self.assertEqual(centered_score, 1.0)
        
        # Off-center shape should get lower score
        off_center = (250, 250)
        off_center_score = self.engine._calculate_position_score(
            off_center, selection_center, selection_rect
        )
        self.assertLess(off_center_score, centered_score)
        self.assertGreaterEqual(off_center_score, 0.0)
        
        # Far off-center shape should get very low score
        far_off_center = (300, 300)
        far_score = self.engine._calculate_position_score(
            far_off_center, selection_center, selection_rect
        )
        self.assertLessEqual(far_score, off_center_score)
    
    def test_is_point_in_selection(self):
        """Test point-in-selection checking."""
        selection_rect = (100, 100, 200, 200)
        
        # Points inside selection
        self.assertTrue(self.engine._is_point_in_selection((150, 150), selection_rect))
        self.assertTrue(self.engine._is_point_in_selection((200, 200), selection_rect))
        self.assertTrue(self.engine._is_point_in_selection((100, 100), selection_rect))  # Edge
        self.assertTrue(self.engine._is_point_in_selection((300, 300), selection_rect))  # Edge
        
        # Points outside selection
        self.assertFalse(self.engine._is_point_in_selection((50, 150), selection_rect))
        self.assertFalse(self.engine._is_point_in_selection((150, 50), selection_rect))
        self.assertFalse(self.engine._is_point_in_selection((350, 200), selection_rect))
        self.assertFalse(self.engine._is_point_in_selection((200, 350), selection_rect))
    
    def test_create_circle_contour(self):
        """Test circular contour creation."""
        center = (200, 200)
        radius = 50
        
        contour = self.engine._create_circle_contour(center, radius)
        
        # Check contour properties
        self.assertIsInstance(contour, np.ndarray)
        self.assertEqual(len(contour.shape), 3)
        self.assertEqual(contour.shape[1], 1)
        self.assertEqual(contour.shape[2], 2)
        self.assertGreater(contour.shape[0], 8)  # Should have reasonable number of points
        
        # Check that points are roughly on the circle
        for point in contour:
            x, y = point[0]
            distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            self.assertAlmostEqual(distance, radius, delta=2)  # Allow small tolerance
    
    def test_find_circle_candidates(self):
        """Test circle candidate detection."""
        # Mock analyzer to return test contours
        self.analyzer.analyze_region.return_value = [self.circle_contour, self.rectangle_contour]
        
        candidates = self.engine._find_circle_candidates(
            [self.circle_contour, self.rectangle_contour], 
            self.test_selection_rect
        )
        
        # Should find at least one circle candidate
        self.assertGreater(len(candidates), 0)
        
        # Check candidate properties
        circle_candidates = [c for c in candidates if c.shape_type == "circle"]
        self.assertGreater(len(circle_candidates), 0)
        
        for candidate in circle_candidates:
            self.assertEqual(candidate.shape_type, "circle")
            self.assertGreater(candidate.confidence_score, 0.0)
            self.assertLessEqual(candidate.confidence_score, 1.0)
            self.assertGreater(candidate.total_score, 0.0)
    
    def test_find_rectangle_candidates(self):
        """Test rectangle candidate detection."""
        candidates = self.engine._find_rectangle_candidates(
            [self.rectangle_contour, self.circle_contour], 
            self.test_selection_rect
        )
        
        # Should find at least one rectangle candidate
        self.assertGreater(len(candidates), 0)
        
        # Check candidate properties
        rect_candidates = [c for c in candidates if c.shape_type == "rectangle"]
        self.assertGreater(len(rect_candidates), 0)
        
        for candidate in rect_candidates:
            self.assertEqual(candidate.shape_type, "rectangle")
            self.assertGreater(candidate.confidence_score, 0.0)
            self.assertLessEqual(candidate.confidence_score, 1.0)
            self.assertGreater(candidate.total_score, 0.0)
    
    def test_select_best_candidate(self):
        """Test best candidate selection."""
        # Create test candidates with different scores
        candidate1 = ShapeCandidate(
            contour=self.circle_contour,
            shape_type="circle",
            center=(200, 200),
            dimensions=(50, 50),
            area=7854,
            confidence_score=0.9,
            position_score=1.0,
            size_score=0.8,
            quality_score=0.9,
            total_score=0.9,
            bounding_rect=(150, 150, 100, 100)
        )
        
        candidate2 = ShapeCandidate(
            contour=self.rectangle_contour,
            shape_type="rectangle",
            center=(180, 180),
            dimensions=(80, 60),
            area=4800,
            confidence_score=0.8,
            position_score=0.8,
            size_score=0.7,
            quality_score=0.8,
            total_score=0.77,
            bounding_rect=(140, 150, 80, 60)
        )
        
        candidates = [candidate2, candidate1]  # Intentionally out of order
        
        best = self.engine._select_best_candidate(candidates, self.test_selection_rect)
        
        self.assertIsNotNone(best)
        self.assertEqual(best, candidate1)  # Should select the one with higher total_score
        
        # Test with empty list
        empty_best = self.engine._select_best_candidate([], self.test_selection_rect)
        self.assertIsNone(empty_best)
    
    def test_create_shape_result(self):
        """Test shape result creation."""
        candidate = ShapeCandidate(
            contour=self.circle_contour,
            shape_type="circle",
            center=(200, 200),
            dimensions=(50, 50),
            area=7854,
            confidence_score=0.9,
            position_score=1.0,
            size_score=0.8,
            quality_score=0.9,
            total_score=0.9,
            bounding_rect=(150, 150, 100, 100)
        )
        
        result = self.engine._create_shape_result(
            candidate, self.test_selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        # Check required fields
        self.assertEqual(result["type"], "circle")
        self.assertEqual(result["detection_method"], "manual")
        self.assertEqual(result["mode"], "manual_circle")
        self.assertEqual(result["center"], (200, 200))
        self.assertEqual(result["dimensions"], (50, 50))
        self.assertEqual(result["area"], 7854)
        self.assertEqual(result["confidence_score"], 0.9)
        
        # Check circle-specific fields
        self.assertEqual(result["radius"], 50)
        self.assertEqual(result["diameter"], 100)
        
        # Test rectangle result
        rect_candidate = ShapeCandidate(
            contour=self.rectangle_contour,
            shape_type="rectangle",
            center=(200, 200),
            dimensions=(80, 60),
            area=4800,
            confidence_score=0.8,
            position_score=0.8,
            size_score=0.7,
            quality_score=0.8,
            total_score=0.77,
            bounding_rect=(160, 170, 80, 60)
        )
        
        rect_result = self.engine._create_shape_result(
            rect_candidate, self.test_selection_rect, SelectionMode.MANUAL_RECTANGLE
        )
        
        self.assertEqual(rect_result["type"], "rectangle")
        self.assertEqual(rect_result["width"], 80)
        self.assertEqual(rect_result["height"], 60)
    
    def test_validate_shape_result(self):
        """Test shape result validation."""
        # Valid result
        valid_result = {
            "type": "circle",
            "detection_method": "manual",
            "center": (200, 200),
            "dimensions": (50, 50),
            "area": 7854,
            "contour": self.circle_contour,
            "confidence_score": 0.9
        }
        
        self.assertTrue(self.engine.validate_shape_result(valid_result))
        
        # Invalid results
        invalid_results = [
            {},  # Empty
            {**valid_result, "type": "invalid"},  # Invalid type
            {**valid_result, "confidence_score": 1.5},  # Invalid confidence
            {**valid_result, "dimensions": (0, 50)},  # Invalid dimensions
            {k: v for k, v in valid_result.items() if k != "type"}  # Missing required field
        ]
        
        for invalid_result in invalid_results:
            self.assertFalse(self.engine.validate_shape_result(invalid_result))
    
    def test_snap_to_shape_circle_mode(self):
        """Test shape snapping in circle mode."""
        # Mock analyzer to return circle contour
        self.analyzer.analyze_region.return_value = [self.circle_contour]
        
        result = self.engine.snap_to_shape(
            self.test_image, self.test_selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "circle")
        self.assertEqual(result["detection_method"], "manual")
        self.assertIn("radius", result)
        self.assertIn("diameter", result)
    
    def test_snap_to_shape_rectangle_mode(self):
        """Test shape snapping in rectangle mode."""
        # Mock analyzer to return rectangle contour
        self.analyzer.analyze_region.return_value = [self.rectangle_contour]
        
        result = self.engine.snap_to_shape(
            self.test_image, self.test_selection_rect, SelectionMode.MANUAL_RECTANGLE
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "rectangle")
        self.assertEqual(result["detection_method"], "manual")
        self.assertIn("width", result)
        self.assertIn("height", result)
    
    def test_snap_to_shape_auto_mode_error(self):
        """Test that AUTO mode raises error."""
        with self.assertRaises(ValueError):
            self.engine.snap_to_shape(
                self.test_image, self.test_selection_rect, SelectionMode.AUTO
            )
    
    def test_snap_to_shape_no_contours(self):
        """Test shape snapping with no contours found."""
        # Mock analyzer to return empty list
        self.analyzer.analyze_region.return_value = []
        
        result = self.engine.snap_to_shape(
            self.test_image, self.test_selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        self.assertIsNone(result)
    
    def test_snap_to_shape_invalid_selection(self):
        """Test shape snapping with invalid selection rectangle."""
        invalid_rect = (-10, -10, 5, 5)
        
        result = self.engine.snap_to_shape(
            self.test_image, invalid_rect, SelectionMode.MANUAL_CIRCLE
        )
        
        self.assertIsNone(result)
    
    def test_find_best_circle(self):
        """Test find_best_circle method."""
        contours = [self.circle_contour, self.rectangle_contour]
        
        result = self.engine.find_best_circle(contours, self.test_selection_rect)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "circle")
    
    def test_find_best_rectangle(self):
        """Test find_best_rectangle method."""
        contours = [self.rectangle_contour, self.circle_contour]
        
        result = self.engine.find_best_rectangle(contours, self.test_selection_rect)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "rectangle")
    
    def test_get_engine_stats(self):
        """Test engine statistics retrieval."""
        stats = self.engine.get_engine_stats()
        
        # Check that all expected keys are present
        expected_keys = [
            "min_contour_area", "max_contour_area_ratio", "min_circularity",
            "min_rectangularity", "size_weight", "position_weight", "quality_weight",
            "circle_detection_params", "rectangle_detection_params"
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check nested dictionaries
        self.assertIn("dp", stats["circle_detection_params"])
        self.assertIn("epsilon_ratio", stats["rectangle_detection_params"])
    
    @patch('cv2.HoughCircles')
    def test_find_hough_circles(self, mock_hough):
        """Test Hough circle detection."""
        # Mock HoughCircles to return test circles
        mock_hough.return_value = np.array([[[100, 100, 50], [150, 150, 30]]])
        
        roi_image = np.zeros((200, 200, 3), dtype=np.uint8)
        candidates = self.engine._find_hough_circles(roi_image, self.test_selection_rect)
        
        self.assertGreater(len(candidates), 0)
        
        for candidate in candidates:
            self.assertEqual(candidate.shape_type, "circle")
            self.assertGreater(candidate.confidence_score, 0.0)
    
    def test_scoring_consistency(self):
        """Test that scoring algorithms produce consistent results."""
        # Create multiple similar candidates
        candidates = []
        for i in range(5):
            center = (200 + i * 10, 200 + i * 10)
            contour = self._create_test_circle_contour(center, 50)
            
            candidate = ShapeCandidate(
                contour=contour,
                shape_type="circle",
                center=center,
                dimensions=(50, 50),
                area=7854,
                confidence_score=0.9,
                position_score=self.engine._calculate_position_score(
                    center, (200, 200), self.test_selection_rect
                ),
                size_score=0.8,
                quality_score=0.9,
                total_score=0.0,  # Will be calculated
                bounding_rect=(center[0] - 50, center[1] - 50, 100, 100)
            )
            
            # Calculate total score
            candidate.total_score = (
                self.engine.size_weight * candidate.size_score +
                self.engine.position_weight * candidate.position_score +
                self.engine.quality_weight * candidate.quality_score
            )
            
            candidates.append(candidate)
        
        # Scores should decrease as shapes move away from center
        for i in range(len(candidates) - 1):
            self.assertGreaterEqual(candidates[i].total_score, candidates[i + 1].total_score)


class TestShapeCandidate(unittest.TestCase):
    """Test cases for ShapeCandidate dataclass."""
    
    def test_shape_candidate_creation(self):
        """Test ShapeCandidate creation and properties."""
        contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32)
        
        candidate = ShapeCandidate(
            contour=contour,
            shape_type="rectangle",
            center=(150, 150),
            dimensions=(100, 100),
            area=10000,
            confidence_score=0.8,
            position_score=0.9,
            size_score=0.7,
            quality_score=0.8,
            total_score=0.8,
            bounding_rect=(100, 100, 100, 100)
        )
        
        self.assertEqual(candidate.shape_type, "rectangle")
        self.assertEqual(candidate.center, (150, 150))
        self.assertEqual(candidate.dimensions, (100, 100))
        self.assertEqual(candidate.area, 10000)
        self.assertEqual(candidate.confidence_score, 0.8)
        self.assertEqual(candidate.total_score, 0.8)


if __name__ == '__main__':
    unittest.main()