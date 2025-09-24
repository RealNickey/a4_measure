"""
Integration tests for Enhanced Contour Analyzer

Tests the integration of the enhanced contour analyzer with real-world scenarios
and validates the improvements over standard detection methods.
"""

import unittest
import cv2
import numpy as np
from enhanced_contour_analyzer import EnhancedContourAnalyzer


class TestEnhancedContourIntegration(unittest.TestCase):
    """Integration test cases for EnhancedContourAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = EnhancedContourAnalyzer()
    
    def test_real_world_scenario_simulation(self):
        """Test enhanced analyzer with simulated real-world conditions."""
        # Create a realistic test image with multiple challenges
        img = self._create_realistic_test_image()
        
        # Analyze the entire image
        contours = self.analyzer.analyze_region(img)
        
        # Should detect multiple meaningful contours
        self.assertGreater(len(contours), 0)
        
        # Verify contours have reasonable properties
        for contour in contours:
            area = cv2.contourArea(contour)
            self.assertGreater(area, 100)  # Minimum meaningful area
            
            # Check contour is not degenerate
            perimeter = cv2.arcLength(contour, True)
            self.assertGreater(perimeter, 20)
    
    def test_performance_comparison(self):
        """Test performance comparison between enhanced and standard methods."""
        img = self._create_realistic_test_image()
        
        # Enhanced method
        enhanced_contours = self.analyzer.analyze_region(img)
        
        # Standard method for comparison
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        standard_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter standard contours by same criteria
        filtered_standard = [c for c in standard_contours if cv2.contourArea(c) > 100]
        
        # Enhanced method should perform at least as well
        self.assertGreaterEqual(len(enhanced_contours), len(filtered_standard))
    
    def test_roi_processing_accuracy(self):
        """Test ROI processing maintains accuracy."""
        img = self._create_realistic_test_image()
        
        # Process entire image
        full_contours = self.analyzer.analyze_region(img)
        
        # Process specific ROI containing known shapes
        roi = (100, 100, 200, 200)  # x, y, w, h
        roi_contours = self.analyzer.analyze_region(img, roi)
        
        # ROI processing should find contours within the region
        if roi_contours:
            for contour in roi_contours:
                # Check that contour coordinates are reasonable
                x_coords = contour[:, 0, 0]
                y_coords = contour[:, 0, 1]
                
                # Should be within or near the ROI bounds
                self.assertGreaterEqual(min(x_coords), roi[0] - 10)  # Allow small margin
                self.assertGreaterEqual(min(y_coords), roi[1] - 10)
                self.assertLessEqual(max(x_coords), roi[0] + roi[2] + 10)
                self.assertLessEqual(max(y_coords), roi[1] + roi[3] + 10)
    
    def test_adaptive_threshold_effectiveness(self):
        """Test that adaptive thresholding is more effective than global thresholding."""
        # Create image with varying illumination
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Create gradient background
        for y in range(300):
            for x in range(300):
                intensity = int(50 + 150 * x / 300)
                img[y, x] = [intensity, intensity, intensity]
        
        # Add shapes with different local contrasts
        cv2.rectangle(img, (50, 50), (100, 100), (0, 0, 0), -1)      # Left side - high contrast
        cv2.rectangle(img, (200, 200), (250, 250), (100, 100, 100), -1)  # Right side - low contrast
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced adaptive thresholding
        adaptive_binary = self.analyzer.apply_adaptive_gaussian_threshold(gray)
        adaptive_contours, _ = cv2.findContours(adaptive_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Global thresholding
        _, global_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        global_contours, _ = cv2.findContours(global_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by meaningful area
        adaptive_meaningful = [c for c in adaptive_contours if cv2.contourArea(c) > 500]
        global_meaningful = [c for c in global_contours if cv2.contourArea(c) > 500]
        
        # Adaptive should detect both rectangles better
        self.assertGreaterEqual(len(adaptive_meaningful), len(global_meaningful))
    
    def test_noise_handling(self):
        """Test that noise reduction improves contour quality."""
        # Create image with noise
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img.fill(255)  # White background
        
        # Add main shape
        cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)
        
        # Add random noise
        noise = np.random.randint(0, 50, (200, 200, 3), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)
        
        # Analyze with enhanced method (includes noise reduction)
        contours = self.analyzer.analyze_region(img)
        
        # Should still detect the main rectangle despite noise
        self.assertGreater(len(contours), 0)
        
        # Largest contour should be the rectangle
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            self.assertGreater(area, 5000)  # Rectangle should be substantial
    
    def test_configuration_impact(self):
        """Test that different configurations produce different results."""
        img = self._create_realistic_test_image()
        
        # Test with different block sizes
        analyzer_small = EnhancedContourAnalyzer(gaussian_block_size=11)
        analyzer_large = EnhancedContourAnalyzer(gaussian_block_size=51)
        
        contours_small = analyzer_small.analyze_region(img)
        contours_large = analyzer_large.analyze_region(img)
        
        # Both should detect contours, but potentially different numbers
        self.assertGreater(len(contours_small), 0)
        self.assertGreater(len(contours_large), 0)
        
        # Test with different C values
        analyzer_low_c = EnhancedContourAnalyzer(gaussian_c=2.0)
        analyzer_high_c = EnhancedContourAnalyzer(gaussian_c=15.0)
        
        contours_low_c = analyzer_low_c.analyze_region(img)
        contours_high_c = analyzer_high_c.analyze_region(img)
        
        # Both should work
        self.assertGreater(len(contours_low_c), 0)
        self.assertGreater(len(contours_high_c), 0)
    
    def _create_realistic_test_image(self) -> np.ndarray:
        """Create a realistic test image with various challenges."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Create non-uniform background
        for y in range(400):
            for x in range(400):
                # Gradient + some texture
                base_intensity = int(80 + 100 * (x + y) / 800)
                noise = np.random.randint(-20, 20)
                intensity = np.clip(base_intensity + noise, 0, 255)
                img[y, x] = [intensity, intensity, intensity]
        
        # Add shapes with varying contrasts
        cv2.rectangle(img, (50, 50), (120, 120), (20, 20, 20), -1)     # High contrast
        cv2.circle(img, (300, 100), 40, (150, 150, 150), -1)          # Medium contrast
        cv2.rectangle(img, (200, 250), (280, 330), (100, 100, 100), -1)  # Low contrast
        
        # Add some overlapping shapes
        cv2.circle(img, (150, 300), 50, (0, 0, 0), -1)
        cv2.rectangle(img, (120, 270), (180, 330), (200, 200, 200), -1)
        
        return img


if __name__ == '__main__':
    unittest.main()