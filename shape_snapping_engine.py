"""
Shape Snapping Engine for Manual Selection

This module provides the ShapeSnappingEngine class that analyzes selected regions
and snaps to the most prominent circle or rectangle within the selection area.
It includes shape scoring algorithms that prioritize size, position, and shape quality.

Requirements addressed: 1.2, 1.3, 4.4
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from enhanced_contour_analyzer import EnhancedContourAnalyzer
from selection_mode import SelectionMode
from manual_selection_errors import (
    ManualSelectionError, ShapeDetectionError, EnhancedAnalysisError,
    ManualSelectionValidator, ErrorRecoveryManager, UserFeedbackManager,
    create_error_context
)
import config


@dataclass
class ShapeCandidate:
    """
    Represents a potential shape candidate with scoring information.
    """
    contour: np.ndarray
    shape_type: str  # "circle" or "rectangle"
    center: Tuple[int, int]
    dimensions: Tuple[float, float]  # (width, height) or (radius, radius) for circles
    area: float
    confidence_score: float
    position_score: float
    size_score: float
    quality_score: float
    total_score: float
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h


class ShapeSnappingEngine:
    """
    Engine for analyzing selected regions and snapping to the most prominent shapes.
    
    Provides shape detection, scoring, and validation for both circles and rectangles
    within manually selected areas.
    """
    
    def __init__(self, analyzer: EnhancedContourAnalyzer):
        """
        Initialize the shape snapping engine.
        
        Args:
            analyzer: Enhanced contour analyzer for image processing
        """
        self.analyzer = analyzer
        
        # Error handling and validation
        self.validator = ManualSelectionValidator()
        self.error_recovery = ErrorRecoveryManager()
        self.feedback_manager = UserFeedbackManager()
        
        # Shape detection parameters from config
        self.min_contour_area = getattr(config, 'ENHANCED_MIN_CONTOUR_AREA', 100)
        self.max_contour_area_ratio = getattr(config, 'MAX_SELECTION_AREA_RATIO', 0.8)
        self.min_circularity = getattr(config, 'MIN_CIRCULARITY_THRESHOLD', 0.6)
        self.min_rectangularity = getattr(config, 'MIN_RECTANGULARITY_THRESHOLD', 0.7)
        self.confidence_threshold = getattr(config, 'SHAPE_CONFIDENCE_THRESHOLD', 0.5)
        
        # Scoring weights
        self.size_weight = 0.4
        self.position_weight = 0.3
        self.quality_weight = 0.3
        
        # Circle detection parameters
        self.circle_dp = 1
        self.circle_min_dist_ratio = 0.1  # Minimum distance between circles as ratio of selection size
        self.circle_param1 = 50
        self.circle_param2 = 30
        
        # Rectangle approximation parameters
        self.rect_epsilon_ratio = 0.02  # Contour approximation epsilon as ratio of perimeter
        self.rect_min_vertices = 4
        self.rect_max_vertices = 8
        
        # Fallback configuration
        self.enable_fallback = getattr(config, 'FALLBACK_TO_STANDARD_THRESHOLD', True)
    
    def snap_to_shape(self, image: np.ndarray, selection_rect: Tuple[int, int, int, int], 
                     mode: SelectionMode) -> Optional[Dict[str, Any]]:
        """
        Analyze a selected region and snap to the most prominent shape with error handling.
        
        Args:
            image: Source image (BGR format)
            selection_rect: Selection rectangle as (x, y, width, height)
            mode: Selection mode (MANUAL_RECTANGLE or MANUAL_CIRCLE)
            
        Returns:
            Dictionary with shape information or None if no suitable shape found
        """
        context = create_error_context(
            "snap_to_shape",
            selection_rect=selection_rect,
            image_shape=image.shape[:2],
            mode=mode.value
        )
        
        try:
            if mode == SelectionMode.AUTO:
                raise ValueError("Shape snapping requires manual selection mode")
            
            # Validate inputs
            self.validator.validate_image_parameters(image)
            self.validator.validate_selection_rect(selection_rect, image.shape)
            
            # Extract region of interest
            roi_image = self._extract_roi(image, selection_rect)
            if roi_image is None:
                raise ShapeDetectionError("Failed to extract region of interest")
            
            # Get contours from the selected region with fallback handling
            contours = self._get_contours_with_fallback(image, selection_rect)
            if not contours:
                shape_type = "circle" if mode == SelectionMode.MANUAL_CIRCLE else "rectangle"
                raise ShapeDetectionError(f"No contours found in selection", shape_type)
            
            # Find shape candidates based on mode
            candidates = self._find_shape_candidates(contours, selection_rect, roi_image, mode)
            
            if not candidates:
                shape_type = "circle" if mode == SelectionMode.MANUAL_CIRCLE else "rectangle"
                raise ShapeDetectionError(f"No {shape_type} candidates found", shape_type)
            
            # Score and select the best candidate
            best_candidate = self._select_best_candidate(candidates, selection_rect)
            if best_candidate is None:
                raise ShapeDetectionError("No suitable shape candidate found")
            
            # Validate the result
            result = self._create_shape_result(best_candidate, selection_rect, mode)
            self.validator.validate_shape_result(result)
            
            # Add success feedback
            self.feedback_manager.add_success_message(
                f"{result['type'].title()} detected with {result['confidence_score']:.1%} confidence",
                "shape_detection"
            )
            
            return result
            
        except ManualSelectionError as e:
            # Handle the error with recovery
            recovery_result = self.error_recovery.handle_selection_error(e, context)
            
            # Add error feedback
            self.feedback_manager.add_error_message(e, context)
            
            # Try fallback strategies if available
            if recovery_result and recovery_result.get("use_fallback"):
                try:
                    return self._snap_with_fallback(image, selection_rect, mode)
                except Exception:
                    pass  # Fallback failed, return None
            
            return None
        
        except Exception as e:
            # Handle unexpected errors
            error = ManualSelectionError(f"Unexpected error in shape snapping: {str(e)}")
            self.feedback_manager.add_error_message(error, context)
            return None
    
    def find_best_circle(self, contours: List[np.ndarray], 
                        selection_rect: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
        """
        Find the best circle within the given contours and selection area.
        
        Args:
            contours: List of contours to analyze
            selection_rect: Selection rectangle as (x, y, width, height)
            
        Returns:
            Dictionary with circle information or None if no suitable circle found
        """
        candidates = self._find_circle_candidates(contours, selection_rect)
        if not candidates:
            return None
        
        best_candidate = self._select_best_candidate(candidates, selection_rect)
        if best_candidate is None:
            return None
        
        return self._create_shape_result(best_candidate, selection_rect, SelectionMode.MANUAL_CIRCLE)
    
    def find_best_rectangle(self, contours: List[np.ndarray], 
                           selection_rect: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
        """
        Find the best rectangle within the given contours and selection area.
        
        Args:
            contours: List of contours to analyze
            selection_rect: Selection rectangle as (x, y, width, height)
            
        Returns:
            Dictionary with rectangle information or None if no suitable rectangle found
        """
        candidates = self._find_rectangle_candidates(contours, selection_rect)
        if not candidates:
            return None
        
        best_candidate = self._select_best_candidate(candidates, selection_rect)
        if best_candidate is None:
            return None
        
        return self._create_shape_result(best_candidate, selection_rect, SelectionMode.MANUAL_RECTANGLE)
    
    def _validate_selection_rect(self, image: np.ndarray, 
                                selection_rect: Tuple[int, int, int, int]) -> bool:
        """
        Validate that the selection rectangle is within image bounds and has valid dimensions.
        
        Args:
            image: Source image
            selection_rect: Selection rectangle to validate
            
        Returns:
            True if valid, False otherwise
        """
        if image is None or len(image.shape) < 2:
            return False
        
        x, y, w, h = selection_rect
        img_h, img_w = image.shape[:2]
        
        # Check bounds and minimum size
        return (x >= 0 and y >= 0 and 
                x + w <= img_w and y + h <= img_h and
                w >= 20 and h >= 20)
    
    def _extract_roi(self, image: np.ndarray, 
                    selection_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract region of interest from the image.
        
        Args:
            image: Source image
            selection_rect: Selection rectangle
            
        Returns:
            Extracted ROI image or None if invalid
        """
        x, y, w, h = selection_rect
        
        # Ensure coordinates are within bounds
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return None
        
        return image[y:y+h, x:x+w].copy()
    
    def _find_circle_candidates(self, contours: List[np.ndarray], 
                               selection_rect: Tuple[int, int, int, int],
                               roi_image: Optional[np.ndarray] = None) -> List[ShapeCandidate]:
        """
        Find circle candidates from contours and Hough circle detection.
        
        Args:
            contours: List of contours to analyze
            selection_rect: Selection rectangle
            roi_image: Region of interest image for Hough detection
            
        Returns:
            List of circle candidates
        """
        candidates = []
        sel_x, sel_y, sel_w, sel_h = selection_rect
        selection_area = sel_w * sel_h
        selection_center = (sel_x + sel_w // 2, sel_y + sel_h // 2)
        
        # Method 1: Analyze contours for circular shapes
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_contour_area or area > selection_area * self.max_contour_area_ratio:
                continue
            
            # Check circularity
            circularity = self._calculate_circularity(contour)
            if circularity < self.min_circularity:
                continue
            
            # Get circle properties
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            center = (int(center_x), int(center_y))
            
            # Check if circle is within selection bounds
            if not self._is_point_in_selection(center, selection_rect):
                continue
            
            # Calculate scores
            size_score = self._calculate_size_score(area, selection_area)
            position_score = self._calculate_position_score(center, selection_center, selection_rect)
            quality_score = circularity
            
            total_score = (self.size_weight * size_score + 
                          self.position_weight * position_score + 
                          self.quality_weight * quality_score)
            
            # Create candidate
            candidate = ShapeCandidate(
                contour=contour,
                shape_type="circle",
                center=center,
                dimensions=(radius, radius),
                area=area,
                confidence_score=circularity,
                position_score=position_score,
                size_score=size_score,
                quality_score=quality_score,
                total_score=total_score,
                bounding_rect=cv2.boundingRect(contour)
            )
            
            candidates.append(candidate)
        
        # Method 2: Hough Circle Detection (if ROI image provided)
        if roi_image is not None:
            hough_candidates = self._find_hough_circles(roi_image, selection_rect)
            candidates.extend(hough_candidates)
        
        return candidates
    
    def _find_hough_circles(self, roi_image: np.ndarray, 
                           selection_rect: Tuple[int, int, int, int]) -> List[ShapeCandidate]:
        """
        Find circles using Hough Circle Transform.
        
        Args:
            roi_image: Region of interest image
            selection_rect: Selection rectangle
            
        Returns:
            List of circle candidates from Hough detection
        """
        candidates = []
        sel_x, sel_y, sel_w, sel_h = selection_rect
        selection_area = sel_w * sel_h
        selection_center = (sel_x + sel_w // 2, sel_y + sel_h // 2)
        
        # Convert to grayscale
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Calculate parameters
        min_radius = max(10, min(sel_w, sel_h) // 10)
        max_radius = min(sel_w, sel_h) // 2
        min_dist = int(min(sel_w, sel_h) * self.circle_min_dist_ratio)
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_dp,
            minDist=min_dist,
            param1=self.circle_param1,
            param2=self.circle_param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (roi_x, roi_y, radius) in circles:
                # Convert ROI coordinates to image coordinates
                center_x = sel_x + roi_x
                center_y = sel_y + roi_y
                center = (center_x, center_y)
                
                # Calculate area and scores
                area = math.pi * radius * radius
                
                # Create a circular contour for consistency
                contour = self._create_circle_contour(center, radius)
                
                # Calculate scores
                size_score = self._calculate_size_score(area, selection_area)
                position_score = self._calculate_position_score(center, selection_center, selection_rect)
                quality_score = 0.9  # High quality score for Hough circles
                
                total_score = (self.size_weight * size_score + 
                              self.position_weight * position_score + 
                              self.quality_weight * quality_score)
                
                # Create candidate
                candidate = ShapeCandidate(
                    contour=contour,
                    shape_type="circle",
                    center=center,
                    dimensions=(radius, radius),
                    area=area,
                    confidence_score=quality_score,
                    position_score=position_score,
                    size_score=size_score,
                    quality_score=quality_score,
                    total_score=total_score,
                    bounding_rect=(center_x - radius, center_y - radius, 2 * radius, 2 * radius)
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _find_rectangle_candidates(self, contours: List[np.ndarray], 
                                  selection_rect: Tuple[int, int, int, int]) -> List[ShapeCandidate]:
        """
        Find rectangle candidates from contours.
        
        Args:
            contours: List of contours to analyze
            selection_rect: Selection rectangle
            
        Returns:
            List of rectangle candidates
        """
        candidates = []
        sel_x, sel_y, sel_w, sel_h = selection_rect
        selection_area = sel_w * sel_h
        selection_center = (sel_x + sel_w // 2, sel_y + sel_h // 2)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_contour_area or area > selection_area * self.max_contour_area_ratio:
                continue
            
            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.rect_epsilon_ratio * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4-8 vertices)
            if len(approx) < self.rect_min_vertices or len(approx) > self.rect_max_vertices:
                continue
            
            # Calculate rectangularity
            rectangularity = self._calculate_rectangularity(contour)
            if rectangularity < self.min_rectangularity:
                continue
            
            # Get rectangle properties
            rect = cv2.minAreaRect(contour)
            center = (int(rect[0][0]), int(rect[0][1]))
            width, height = rect[1]
            
            # Check if rectangle center is within selection bounds
            if not self._is_point_in_selection(center, selection_rect):
                continue
            
            # Calculate scores
            size_score = self._calculate_size_score(area, selection_area)
            position_score = self._calculate_position_score(center, selection_center, selection_rect)
            quality_score = rectangularity
            
            total_score = (self.size_weight * size_score + 
                          self.position_weight * position_score + 
                          self.quality_weight * quality_score)
            
            # Create candidate
            candidate = ShapeCandidate(
                contour=contour,
                shape_type="rectangle",
                center=center,
                dimensions=(width, height),
                area=area,
                confidence_score=rectangularity,
                position_score=position_score,
                size_score=size_score,
                quality_score=quality_score,
                total_score=total_score,
                bounding_rect=cv2.boundingRect(contour)
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _select_best_candidate(self, candidates: List[ShapeCandidate], 
                              selection_rect: Tuple[int, int, int, int]) -> Optional[ShapeCandidate]:
        """
        Select the best candidate from the list based on total score.
        
        Args:
            candidates: List of shape candidates
            selection_rect: Selection rectangle
            
        Returns:
            Best candidate or None if no valid candidates
        """
        if not candidates:
            return None
        
        # Sort by total score (highest first)
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        
        # Return the best candidate
        return candidates[0]
    
    def _calculate_circularity(self, contour: np.ndarray) -> float:
        """
        Calculate the circularity of a contour.
        
        Args:
            contour: Input contour
            
        Returns:
            Circularity value (0.0 to 1.0, where 1.0 is perfect circle)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity = 4π * area / perimeter²
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        return min(1.0, circularity)
    
    def _calculate_rectangularity(self, contour: np.ndarray) -> float:
        """
        Calculate the rectangularity of a contour.
        
        Args:
            contour: Input contour
            
        Returns:
            Rectangularity value (0.0 to 1.0, where 1.0 is perfect rectangle)
        """
        area = cv2.contourArea(contour)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bounding_area = w * h
        
        if bounding_area == 0:
            return 0.0
        
        # Rectangularity = contour area / bounding rectangle area
        rectangularity = area / bounding_area
        return min(1.0, rectangularity)
    
    def _calculate_size_score(self, shape_area: float, selection_area: float) -> float:
        """
        Calculate size score based on shape area relative to selection area.
        
        Args:
            shape_area: Area of the shape
            selection_area: Area of the selection rectangle
            
        Returns:
            Size score (0.0 to 1.0, where larger shapes get higher scores)
        """
        if selection_area == 0:
            return 0.0
        
        # Prefer shapes that are 20-60% of selection area
        ratio = shape_area / selection_area
        
        if ratio < 0.1:
            return ratio / 0.1  # Linear increase from 0 to 1
        elif ratio <= 0.6:
            return 1.0  # Optimal range
        else:
            return max(0.0, 1.0 - (ratio - 0.6) / 0.4)  # Linear decrease
    
    def _calculate_position_score(self, shape_center: Tuple[int, int], 
                                 selection_center: Tuple[int, int],
                                 selection_rect: Tuple[int, int, int, int]) -> float:
        """
        Calculate position score based on how centered the shape is in the selection.
        
        Args:
            shape_center: Center point of the shape
            selection_center: Center point of the selection
            selection_rect: Selection rectangle
            
        Returns:
            Position score (0.0 to 1.0, where centered shapes get higher scores)
        """
        sel_x, sel_y, sel_w, sel_h = selection_rect
        
        # Calculate distance from selection center
        dx = abs(shape_center[0] - selection_center[0])
        dy = abs(shape_center[1] - selection_center[1])
        
        # Normalize by selection dimensions
        max_dx = sel_w / 2
        max_dy = sel_h / 2
        
        if max_dx == 0 or max_dy == 0:
            return 0.0
        
        # Calculate normalized distance
        norm_dx = dx / max_dx
        norm_dy = dy / max_dy
        distance = math.sqrt(norm_dx * norm_dx + norm_dy * norm_dy)
        
        # Convert to score (closer to center = higher score)
        return max(0.0, 1.0 - distance)
    
    def _is_point_in_selection(self, point: Tuple[int, int], 
                              selection_rect: Tuple[int, int, int, int]) -> bool:
        """
        Check if a point is within the selection rectangle.
        
        Args:
            point: Point to check
            selection_rect: Selection rectangle
            
        Returns:
            True if point is within selection, False otherwise
        """
        x, y = point
        sel_x, sel_y, sel_w, sel_h = selection_rect
        
        return (sel_x <= x <= sel_x + sel_w and 
                sel_y <= y <= sel_y + sel_h)
    
    def _create_circle_contour(self, center: Tuple[int, int], radius: int) -> np.ndarray:
        """
        Create a circular contour for Hough circle results.
        
        Args:
            center: Circle center
            radius: Circle radius
            
        Returns:
            Circular contour as numpy array
        """
        # Create points around the circle
        num_points = max(8, int(2 * math.pi * radius / 5))  # Adaptive point count
        angles = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
        
        points = []
        for angle in angles:
            x = int(center[0] + radius * math.cos(angle))
            y = int(center[1] + radius * math.sin(angle))
            points.append([x, y])
        
        return np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    
    def _create_shape_result(self, candidate: ShapeCandidate, 
                            selection_rect: Tuple[int, int, int, int],
                            mode: SelectionMode) -> Dict[str, Any]:
        """
        Create a shape result dictionary from a candidate.
        
        Args:
            candidate: Selected shape candidate
            selection_rect: Original selection rectangle
            mode: Selection mode used
            
        Returns:
            Dictionary with shape result information
        """
        result = {
            "type": candidate.shape_type,
            "detection_method": "manual",
            "mode": mode.value,
            "center": candidate.center,
            "dimensions": candidate.dimensions,
            "area": candidate.area,
            "contour": candidate.contour,
            "bounding_rect": candidate.bounding_rect,
            "selection_rect": selection_rect,
            "confidence_score": candidate.confidence_score,
            "position_score": candidate.position_score,
            "size_score": candidate.size_score,
            "quality_score": candidate.quality_score,
            "total_score": candidate.total_score
        }
        
        # Add shape-specific information
        if candidate.shape_type == "circle":
            result["radius"] = candidate.dimensions[0]
            result["diameter"] = candidate.dimensions[0] * 2
        elif candidate.shape_type == "rectangle":
            result["width"] = candidate.dimensions[0]
            result["height"] = candidate.dimensions[1]
        
        return result
    
    def validate_shape_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate a shape result for completeness and consistency.
        
        Args:
            result: Shape result dictionary
            
        Returns:
            True if result is valid, False otherwise
        """
        required_fields = [
            "type", "detection_method", "center", "dimensions", 
            "area", "contour", "confidence_score"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in result:
                return False
        
        # Validate shape type
        if result["type"] not in ["circle", "rectangle"]:
            return False
        
        # Validate confidence score
        if not (0.0 <= result["confidence_score"] <= 1.0):
            return False
        
        # Validate dimensions
        dimensions = result["dimensions"]
        if len(dimensions) != 2 or any(d <= 0 for d in dimensions):
            return False
        
        return True
    
    def _get_contours_with_fallback(self, image: np.ndarray, 
                                   selection_rect: Tuple[int, int, int, int]) -> List[np.ndarray]:
        """
        Get contours with fallback to standard analysis if enhanced analysis fails.
        
        Args:
            image: Source image
            selection_rect: Selection rectangle
            
        Returns:
            List of contours
        """
        try:
            # Try enhanced analysis first
            contours = self.analyzer.analyze_region(image, selection_rect)
            if contours:
                return contours
        except Exception as e:
            if self.enable_fallback:
                # Log the enhanced analysis failure
                error = EnhancedAnalysisError(f"Enhanced analysis failed: {str(e)}")
                context = create_error_context("enhanced_analysis", selection_rect=selection_rect)
                self.error_recovery.handle_selection_error(error, context)
                
                # Try fallback to standard contour detection
                return self._fallback_contour_detection(image, selection_rect)
            else:
                raise
        
        # If enhanced analysis returned empty results, try fallback
        if self.enable_fallback:
            return self._fallback_contour_detection(image, selection_rect)
        
        return []
    
    def _fallback_contour_detection(self, image: np.ndarray, 
                                   selection_rect: Tuple[int, int, int, int]) -> List[np.ndarray]:
        """
        Fallback contour detection using standard OpenCV methods.
        
        Args:
            image: Source image
            selection_rect: Selection rectangle
            
        Returns:
            List of contours from standard detection
        """
        try:
            # Extract ROI
            roi = self._extract_roi(image, selection_rect)
            if roi is None:
                return []
            
            # Convert to grayscale
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()
            
            # Apply standard thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Adjust contour coordinates to image space
            x, y, _, _ = selection_rect
            adjusted_contours = []
            for contour in contours:
                adjusted_contour = contour.copy()
                adjusted_contour[:, 0, 0] += x
                adjusted_contour[:, 0, 1] += y
                adjusted_contours.append(adjusted_contour)
            
            return adjusted_contours
            
        except Exception:
            return []
    
    def _find_shape_candidates(self, contours: List[np.ndarray], 
                              selection_rect: Tuple[int, int, int, int],
                              roi_image: np.ndarray, mode: SelectionMode) -> List[ShapeCandidate]:
        """
        Find shape candidates based on the selection mode.
        
        Args:
            contours: List of contours to analyze
            selection_rect: Selection rectangle
            roi_image: Region of interest image
            mode: Selection mode
            
        Returns:
            List of shape candidates
        """
        if mode == SelectionMode.MANUAL_CIRCLE:
            return self._find_circle_candidates(contours, selection_rect, roi_image)
        elif mode == SelectionMode.MANUAL_RECTANGLE:
            return self._find_rectangle_candidates(contours, selection_rect)
        else:
            return []
    
    def _snap_with_fallback(self, image: np.ndarray, selection_rect: Tuple[int, int, int, int],
                           mode: SelectionMode) -> Optional[Dict[str, Any]]:
        """
        Attempt shape snapping with fallback methods.
        
        Args:
            image: Source image
            selection_rect: Selection rectangle
            mode: Selection mode
            
        Returns:
            Shape result or None if fallback fails
        """
        try:
            # Use fallback contour detection
            contours = self._fallback_contour_detection(image, selection_rect)
            if not contours:
                return None
            
            # Extract ROI for Hough detection if needed
            roi_image = self._extract_roi(image, selection_rect)
            
            # Find candidates with relaxed thresholds
            original_circularity = self.min_circularity
            original_rectangularity = self.min_rectangularity
            
            # Relax thresholds for fallback
            self.min_circularity *= 0.8
            self.min_rectangularity *= 0.8
            
            try:
                candidates = self._find_shape_candidates(contours, selection_rect, roi_image, mode)
                
                if candidates:
                    best_candidate = self._select_best_candidate(candidates, selection_rect)
                    if best_candidate:
                        result = self._create_shape_result(best_candidate, selection_rect, mode)
                        result["fallback_used"] = True
                        return result
            finally:
                # Restore original thresholds
                self.min_circularity = original_circularity
                self.min_rectangularity = original_rectangularity
            
            return None
            
        except Exception:
            return None
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get engine configuration and statistics.
        
        Returns:
            Dictionary with engine information
        """
        stats = {
            "min_contour_area": self.min_contour_area,
            "max_contour_area_ratio": self.max_contour_area_ratio,
            "min_circularity": self.min_circularity,
            "min_rectangularity": self.min_rectangularity,
            "confidence_threshold": self.confidence_threshold,
            "size_weight": self.size_weight,
            "position_weight": self.position_weight,
            "quality_weight": self.quality_weight,
            "enable_fallback": self.enable_fallback,
            "circle_detection_params": {
                "dp": self.circle_dp,
                "min_dist_ratio": self.circle_min_dist_ratio,
                "param1": self.circle_param1,
                "param2": self.circle_param2
            },
            "rectangle_detection_params": {
                "epsilon_ratio": self.rect_epsilon_ratio,
                "min_vertices": self.rect_min_vertices,
                "max_vertices": self.rect_max_vertices
            }
        }
        
        # Add error recovery statistics
        if hasattr(self, 'error_recovery'):
            stats["error_recovery"] = self.error_recovery.get_recovery_stats()
        
        return stats