"""
Unit tests for Enhanced Contour Analyzer

Tests thresholding accuracy and contour detection quality for the
EnhancedContourAnalyzer class.
"""

import unittest
import cv2
import numpy as np
from enhanced_contour_analyzer import EnhancedContourAnalyzer


class TestEnhancedContourAnalyzer(unittest.TestCase):
    """Test cases for EnhancedContourAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = EnhancedContourAnalyzer()
        
        # Create test images
        self.test_image_simple = self._create_simple_test_image()
        self.test_image_complex = self._create_complex_test_image()
        self.test_image_varying_light = self._create_varying_light_image()
        self.test_image_nested = self._create_nested_shapes_image()
    
    def _create_simple_test_image(self) -> np.ndarray:
        """Create a simple test image with basic shapes."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img.fill(255)  # White background
        
        # Draw a black rectangle
        cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)
        
        # Draw a black circle
        cv2.circle(img, (300, 100), 50, (0, 0, 0), -1)
        
        return img
    
    def _create_complex_test_image(self) -> np.ndarray:
        """Create a complex test image with multiple shapes."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img.fill(200)  # Light gray background
        
        # Multiple rectangles
        cv2.rectangle(img, (50, 50), (150, 100), (0, 0, 0), -1)
        cv2.rectangle(img, (200, 200), (350, 280), (50, 50, 50), -1)
        
        # Multiple circles
        cv2.circle(img, (100, 300), 40, (0, 0, 0), -1)
        cv2.circle(img, (400, 150), 60, (30, 30, 30), -1)
        
        # Overlapping shapes
        cv2.rectangle(img, (300, 300), (450, 400), (0, 0, 0), -1)
        cv2.circle(img, (375, 350), 30, (255, 255, 255), -1)
        
        return img
    
    def _create_varying_light_image(self) -> np.ndarray:
        """Create test image with varying lighting conditions."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Create gradient background (varying lighting)
        for y in range(400):
            for x in range(400):
                intensity = int(100 + 100 * (x + y) / 800)
                img[y, x] = [intensity, intensity, intensity]
        
        # Add shapes with different contrasts
        cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)  # High contrast
        cv2.circle(img, (300, 100), 50, (150, 150, 150), -1)     # Medium contrast
        cv2.rectangle(img, (200, 250), (350, 350), (50, 50, 50), -1)  # Low contrast
        
        return img
    
    def _create_nested_shapes_image(self) -> np.ndarray:
        """Create test image with nested shapes (shapes within shapes)."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img.fill(255)  # White background
        
        # Outer rectangle
        cv2.rectangle(img, (50, 50), (350, 350), (0, 0, 0), -1)
        
        # Inner white rectangle
        cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
        
        # Inner black circle
        cv2.circle(img, (200, 200), 50, (0, 0, 0), -1)
        
        # Small white circle inside black circle
        cv2.circle(img, (200, 200), 20, (255, 255, 255), -1)
        
        return img
    
    def test_initialization(self):
        """Test EnhancedContourAnalyzer initialization."""
        # Test default initialization
        analyzer = EnhancedContourAnalyzer()
        self.assertEqual(analyzer.gaussian_block_size, 31)
        self.assertEqual(analyzer.gaussian_c, 7.0)
        
        # Test custom initialization
        analyzer_custom = EnhancedContourAnalyzer(gaussian_block_size=21, gaussian_c=5.0)
        self.assertEqual(analyzer_custom.gaussian_block_size, 21)
        self.assertEqual(analyzer_custom.gaussian_c, 5.0)
        
        # Test even block size correction
        analyzer_even = EnhancedContourAnalyzer(gaussian_block_size=20)
        self.assertEqual(analyzer_even.gaussian_block_size, 21)  # Should be corrected to odd
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion functionality."""
        # Test BGR to grayscale
        gray = self.analyzer._convert_to_grayscale(self.test_image_simple)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (400, 400))
        
        # Test already grayscale image
        gray_input = cv2.cvtColor(self.test_image_simple, cv2.COLOR_BGR2GRAY)
        gray_output = self.analyzer._convert_to_grayscale(gray_input)
        np.testing.assert_array_equal(gray_input, gray_output)
    
    def test_adaptive_gaussian_threshold(self):
        """Test adaptive Gaussian thresholding accuracy."""
        gray = cv2.cvtColor(self.test_image_varying_light, cv2.COLOR_BGR2GRAY)
        binary = self.analyzer.apply_adaptive_gaussian_threshold(gray)
        
        # Check output properties
        self.assertEqual(binary.dtype, np.uint8)
        self.assertEqual(binary.shape, gray.shape)
        
        # Check that output is binary (only 0 and 255 values)
        unique_values = np.unique(binary)
        self.assertTrue(len(unique_values) <= 2)
        self.assertTrue(all(val in [0, 255] for val in unique_values))
        
        # Test that thresholding produces reasonable results
        # Should have both black and white regions
        self.assertGreater(np.sum(binary == 0), 1000)  # Some black pixels
        self.assertGreater(np.sum(binary == 255), 1000)  # Some white pixels
    
    def test_noise_reduction(self):
        """Test morphological noise reduction."""
        # Create noisy binary image
        noisy_binary = np.zeros((200, 200), dtype=np.uint8)
        noisy_binary.fill(255)
        
        # Add noise (small black spots)
        for i in range(50):
            x, y = np.random.randint(0, 200, 2)
            noisy_binary[y, x] = 0
        
        # Add main shape
        cv2.rectangle(noisy_binary, (50, 50), (150, 150), 0, -1)
        
        # Apply noise reduction
        denoised = self.analyzer._reduce_noise(noisy_binary)
        
        # Check that noise is reduced
        self.assertEqual(denoised.dtype, np.uint8)
        self.assertEqual(denoised.shape, noisy_binary.shape)
        
        # Main shape should still be present
        roi = denoised[50:150, 50:150]
        self.assertGreater(np.sum(roi == 0), 5000)  # Most of rectangle should be black
    
    def test_contour_detection_simple(self):
        """Test contour detection on simple shapes."""
        contours = self.analyzer.analyze_region(self.test_image_simple)
        
        # Should detect at least 2 contours (rectangle and circle)
        self.assertGreaterEqual(len(contours), 2)
        
        # Check that contours have reasonable areas
        areas = [cv2.contourArea(c) for c in contours]
        self.assertTrue(all(area > 100 for area in areas))  # All contours should be substantial
    
    def test_contour_detection_complex(self):
        """Test contour detection on complex image."""
        contours = self.analyzer.analyze_region(self.test_image_complex)
        
        # Should detect multiple contours
        self.assertGreaterEqual(len(contours), 3)
        
        # Contours should be sorted by area (largest first)
        areas = [cv2.contourArea(c) for c in contours]
        self.assertEqual(areas, sorted(areas, reverse=True))
    
    def test_contour_detection_nested(self):
        """Test contour detection on nested shapes."""
        contours = self.analyzer.analyze_region(self.test_image_nested)
        
        # Should detect nested contours
        self.assertGreaterEqual(len(contours), 2)
        
        # Check that we can detect both outer and inner shapes
        areas = [cv2.contourArea(c) for c in contours]
        self.assertGreater(max(areas), 10000)  # Large outer shape
        self.assertGreater(min([a for a in areas if a > 1000]), 1000)  # Smaller inner shapes
    
    def test_roi_analysis(self):
        """Test region of interest analysis."""
        # Define ROI around the rectangle in simple test image
        roi = (25, 25, 150, 150)  # x, y, width, height
        contours = self.analyzer.analyze_region(self.test_image_simple, roi)
        
        # Should detect the rectangle
        self.assertGreaterEqual(len(contours), 1)
        
        # Check that contour coordinates are adjusted for ROI
        if contours:
            contour = contours[0]
            x_coords = contour[:, 0, 0]
            y_coords = contour[:, 0, 1]
            
            # Coordinates should be in original image space, not ROI space
            self.assertGreaterEqual(min(x_coords), 25)  # Should be >= ROI x offset
            self.assertGreaterEqual(min(y_coords), 25)  # Should be >= ROI y offset
    
    def test_roi_bounds_checking(self):
        """Test ROI bounds checking and handling."""
        # Test ROI outside image bounds
        roi_outside = (500, 500, 100, 100)
        contours = self.analyzer.analyze_region(self.test_image_simple, roi_outside)
        self.assertEqual(len(contours), 0)
        
        # Test ROI partially outside bounds
        roi_partial = (350, 350, 100, 100)
        contours = self.analyzer.analyze_region(self.test_image_simple, roi_partial)
        # Should handle gracefully (may or may not find contours)
        self.assertIsInstance(contours, list)
    
    def test_contour_validation(self):
        """Test contour validation logic."""
        # Create valid contour (rectangle)
        valid_contour = np.array([[[50, 50]], [[150, 50]], [[150, 150]], [[50, 150]]], dtype=np.int32)
        self.assertTrue(self.analyzer._is_valid_contour(valid_contour))
        
        # Create invalid contour (too small area)
        small_contour = np.array([[[0, 0]], [[5, 0]], [[5, 5]], [[0, 5]]], dtype=np.int32)
        self.assertFalse(self.analyzer._is_valid_contour(small_contour))
        
        # Create invalid contour (too few points)
        few_points = np.array([[[0, 0]], [[10, 10]]], dtype=np.int32)
        self.assertFalse(self.analyzer._is_valid_contour(few_points))
    
    def test_processing_stats(self):
        """Test processing statistics retrieval."""
        stats = self.analyzer.get_processing_stats()
        
        # Check that all expected keys are present
        expected_keys = ['gaussian_block_size', 'gaussian_c', 'cuda_enabled', 
                        'noise_kernel_size', 'closing_kernel_size']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check data types
        self.assertIsInstance(stats['gaussian_block_size'], int)
        self.assertIsInstance(stats['gaussian_c'], float)
        self.assertIsInstance(stats['cuda_enabled'], bool)
    
    def test_thresholding_quality_varying_light(self):
        """Test thresholding quality under varying lighting conditions."""
        gray = cv2.cvtColor(self.test_image_varying_light, cv2.COLOR_BGR2GRAY)
        binary = self.analyzer.apply_adaptive_gaussian_threshold(gray)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Should detect at least some shapes despite varying lighting
        self.assertGreaterEqual(len(contours), 1)
        
        # Check that detected contours have reasonable areas
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
        self.assertGreaterEqual(len(areas), 1)  # At least 1 substantial shape
    
    def test_enhanced_vs_standard_detection(self):
        """Compare enhanced detection with standard thresholding."""
        gray = cv2.cvtColor(self.test_image_varying_light, cv2.COLOR_BGR2GRAY)
        
        # Enhanced thresholding
        enhanced_binary = self.analyzer.apply_adaptive_gaussian_threshold(gray)
        enhanced_contours, _ = cv2.findContours(enhanced_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Standard global thresholding
        _, standard_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        standard_contours, _ = cv2.findContours(standard_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Enhanced should perform better or equal in varying light conditions
        enhanced_count = len([c for c in enhanced_contours if cv2.contourArea(c) > 100])
        standard_count = len([c for c in standard_contours if cv2.contourArea(c) > 100])
        
        # Enhanced should detect at least as many meaningful contours
        self.assertGreaterEqual(enhanced_count, standard_count)


if __name__ == '__main__':
    unittest.main()