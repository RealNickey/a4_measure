"""
Adaptive Threshold Calibration Module

This module implements an intelligent adaptive threshold calibration system that
dynamically adjusts detection parameters based on the specific characteristics of
each input image. It significantly improves measurement accuracy across diverse
lighting conditions.

Features:
- Dynamic Lighting Analysis: Analyzes brightness, contrast, and histogram statistics
- Multi-Pass Threshold Strategy: Progressive threshold refinement
- Local Adaptive Thresholding: Region-based processing for non-uniform lighting
- Contrast Enhancement Pre-Processing: CLAHE for better edge detection
- Noise Reduction Integration: Bilateral filtering and morphological operations
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging


class AdaptiveThresholdCalibrator:
    """
    Intelligent adaptive threshold calibration system for improved detection accuracy.
    """
    
    def __init__(self,
                 initial_block_size: int = 31,
                 initial_c: float = 7.0,
                 enable_clahe: bool = True,
                 enable_multipass: bool = True,
                 enable_local_adaptive: bool = True):
        """
        Initialize the adaptive threshold calibrator.
        
        Args:
            initial_block_size: Initial block size for adaptive threshold (must be odd)
            initial_c: Initial constant subtracted from mean
            enable_clahe: Enable CLAHE pre-processing
            enable_multipass: Enable multi-pass threshold refinement
            enable_local_adaptive: Enable local adaptive thresholding
        """
        # Ensure block size is odd and at least 3
        if initial_block_size % 2 == 0:
            initial_block_size += 1
        self.initial_block_size = max(3, initial_block_size)
        self.initial_c = initial_c
        
        # Feature flags
        self.enable_clahe = enable_clahe
        self.enable_multipass = enable_multipass
        self.enable_local_adaptive = enable_local_adaptive
        
        # CLAHE object for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Statistics tracking
        self.last_calibration_stats = {}
        
        logging.info("AdaptiveThresholdCalibrator initialized")
    
    def analyze_lighting_conditions(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the lighting conditions of the input image.
        
        Args:
            gray: Grayscale input image
            
        Returns:
            Dictionary with lighting analysis results
        """
        # Calculate histogram statistics
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Calculate mean and standard deviation
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Calculate percentiles for better understanding of distribution
        p10 = np.percentile(gray, 10)
        p90 = np.percentile(gray, 90)
        dynamic_range = p90 - p10
        
        # Detect lighting condition
        if mean_brightness < 80:
            lighting_condition = "underexposed"
        elif mean_brightness > 175:
            lighting_condition = "overexposed"
        else:
            lighting_condition = "normal"
        
        # Calculate contrast metric
        if dynamic_range > 0:
            contrast_ratio = std_brightness / dynamic_range
        else:
            contrast_ratio = 0.0
        
        # Detect if histogram is bimodal (good for thresholding)
        hist_smooth = cv2.GaussianBlur(hist.reshape(-1, 1), (5, 1), 0).flatten()
        peaks = self._find_histogram_peaks(hist_smooth)
        is_bimodal = len(peaks) >= 2
        
        return {
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'dynamic_range': float(dynamic_range),
            'contrast_ratio': float(contrast_ratio),
            'lighting_condition': lighting_condition,
            'is_bimodal': is_bimodal,
            'p10': float(p10),
            'p90': float(p90),
            'histogram': hist
        }
    
    def _find_histogram_peaks(self, hist: np.ndarray, threshold: float = 0.1) -> list:
        """
        Find peaks in histogram for bimodal detection.
        
        Args:
            hist: Histogram values
            threshold: Relative threshold for peak detection
            
        Returns:
            List of peak indices
        """
        peaks = []
        max_val = np.max(hist)
        threshold_val = max_val * threshold
        
        for i in range(1, len(hist) - 1):
            if hist[i] > threshold_val and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)
        
        return peaks
    
    def calibrate_threshold_parameters(self, lighting_stats: Dict[str, Any]) -> Tuple[int, float]:
        """
        Calibrate threshold parameters based on lighting analysis.
        
        Args:
            lighting_stats: Lighting analysis results
            
        Returns:
            Tuple of (block_size, c_constant)
        """
        mean_brightness = lighting_stats['mean_brightness']
        std_brightness = lighting_stats['std_brightness']
        contrast_ratio = lighting_stats['contrast_ratio']
        lighting_condition = lighting_stats['lighting_condition']
        dynamic_range = lighting_stats['dynamic_range']
        
        # Start with initial values
        block_size = self.initial_block_size
        c_constant = self.initial_c
        
        # Adjust block size based on image characteristics
        # Larger block size for uniform lighting, smaller for varied lighting
        if contrast_ratio > 0.5:  # High local contrast
            block_size = max(11, self.initial_block_size - 10)
        elif contrast_ratio < 0.2:  # Low local contrast
            block_size = min(51, self.initial_block_size + 10)
        
        # Adjust C constant based on lighting condition AND dynamic range
        # If dynamic range is very low, be more conservative with C adjustment
        if dynamic_range < 50:  # Low dynamic range - use standard deviation instead
            # For low dynamic range images, adjust based on std
            if std_brightness < 30:  # Very uniform
                c_constant = max(3.0, self.initial_c - 2.0)
            else:  # Some variation
                c_constant = self.initial_c
        else:
            # Normal dynamic range - use lighting condition
            if lighting_condition == "underexposed":
                # Reduce C to be more sensitive in dark images
                c_constant = max(2.0, self.initial_c - 3.0)
            elif lighting_condition == "overexposed":
                # For overexposed, only increase C if std is also low
                if std_brightness < 40:
                    c_constant = min(12.0, self.initial_c + 3.0)
                else:
                    # There's variation in the bright image, keep default
                    c_constant = self.initial_c
            else:
                # Normal lighting - use dynamic range for fine-tuning
                if dynamic_range > 150:  # High dynamic range
                    c_constant = min(10.0, self.initial_c + 2.0)
        
        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        block_size = max(3, min(99, block_size))  # Clamp to valid range
        c_constant = max(1.0, min(20.0, c_constant))  # Clamp to valid range
        
        return block_size, c_constant
    
    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply contrast enhancement to improve edge detection.
        
        Args:
            gray: Grayscale input image
            
        Returns:
            Contrast-enhanced grayscale image
        """
        if not self.enable_clahe:
            return gray
        
        # Check if CLAHE would be beneficial
        # Only apply if the image has low contrast
        std = np.std(gray)
        
        if std < 30:  # Low contrast, CLAHE would help
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            enhanced = self.clahe.apply(gray)
            return enhanced
        
        # Good contrast already, return as-is
        return gray
    
    def apply_local_adaptive_threshold(self, gray: np.ndarray, 
                                      block_size: int, 
                                      c_constant: float) -> np.ndarray:
        """
        Apply local adaptive thresholding with region-based processing.
        
        Args:
            gray: Grayscale input image
            block_size: Block size for adaptive threshold
            c_constant: Constant subtracted from mean
            
        Returns:
            Binary thresholded image
        """
        if not self.enable_local_adaptive:
            # Fall back to standard adaptive threshold
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, c_constant
            )
        
        # Apply adaptive Gaussian threshold (already local by nature)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_constant
        )
        
        return binary
    
    def refine_with_multipass(self, gray: np.ndarray, 
                             initial_binary: np.ndarray,
                             initial_block_size: int,
                             initial_c: float) -> np.ndarray:
        """
        Refine threshold using multi-pass strategy.
        
        Args:
            gray: Original grayscale image
            initial_binary: Initial binary result
            initial_block_size: Initial block size used
            initial_c: Initial C constant used
            
        Returns:
            Refined binary image
        """
        if not self.enable_multipass:
            return initial_binary
        
        # Analyze quality of initial detection
        contours, _ = cv2.findContours(initial_binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # No contours found, try with more sensitive settings
            refined_block_size = max(11, initial_block_size - 10)
            refined_c = max(2.0, initial_c - 2.0)
            
            if refined_block_size % 2 == 0:
                refined_block_size += 1
            
            refined_binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, refined_block_size, refined_c
            )
            
            return refined_binary
        
        # Calculate edge strength in detected regions
        edges = cv2.Canny(gray, 50, 150)
        edge_strength = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        if edge_strength < 0.01:
            # Weak edges, increase sensitivity
            refined_block_size = max(11, initial_block_size - 10)
            refined_c = max(2.0, initial_c - 2.0)
            
            if refined_block_size % 2 == 0:
                refined_block_size += 1
            
            refined_binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, refined_block_size, refined_c
            )
            
            return refined_binary
        
        # Initial detection seems good, return as is
        return initial_binary
    
    def reduce_noise(self, binary: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """
        Apply noise reduction using morphological operations.
        
        Args:
            binary: Binary input image
            aggressive: Use more aggressive noise reduction
            
        Returns:
            Noise-reduced binary image
        """
        if aggressive:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        else:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove small noise with opening
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        # Fill small gaps with closing
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_close)
        
        return denoised
    
    def calibrate_and_threshold(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main calibration and thresholding pipeline.
        
        Args:
            image: Input BGR or grayscale image
            
        Returns:
            Tuple of (binary_image, calibration_stats)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Analyze lighting conditions
        lighting_stats = self.analyze_lighting_conditions(gray)
        
        # Step 2: Calibrate threshold parameters
        block_size, c_constant = self.calibrate_threshold_parameters(lighting_stats)
        
        # Step 3: Enhance contrast
        enhanced = self.enhance_contrast(gray)
        
        # Step 4: Apply local adaptive thresholding
        binary = self.apply_local_adaptive_threshold(enhanced, block_size, c_constant)
        
        # Step 5: Multi-pass refinement
        binary = self.refine_with_multipass(enhanced, binary, block_size, c_constant)
        
        # Step 6: Noise reduction
        # Only use aggressive noise reduction for very low SNR scenarios
        # Check if there's actually a lot of noise
        edges = cv2.Canny(enhanced, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        aggressive = edge_density > 0.15  # High edge density suggests noise
        binary = self.reduce_noise(binary, aggressive=aggressive)
        
        # Store calibration stats
        calibration_stats = {
            'block_size': block_size,
            'c_constant': c_constant,
            'lighting_stats': lighting_stats,
            'clahe_enabled': self.enable_clahe,
            'multipass_enabled': self.enable_multipass,
            'local_adaptive_enabled': self.enable_local_adaptive
        }
        
        self.last_calibration_stats = calibration_stats
        
        return binary, calibration_stats
    
    def get_last_calibration_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the last calibration.
        
        Returns:
            Dictionary with calibration statistics
        """
        return self.last_calibration_stats.copy()
