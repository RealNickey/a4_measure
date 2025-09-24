"""
Enhanced Contour Analysis Module

This module provides improved contour detection using Adaptive Gaussian thresholding
and advanced image processing techniques for better shape detection in challenging
lighting conditions.

Requirements addressed: 2.1, 2.2, 2.3
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import USE_CUDA_IF_AVAILABLE


class EnhancedContourAnalyzer:
    """
    Enhanced contour analyzer with Adaptive Gaussian thresholding for improved
    shape detection in varying lighting conditions.
    """
    
    def __init__(self, gaussian_block_size: int = 31, gaussian_c: float = 7.0):
        """
        Initialize the enhanced contour analyzer.
        
        Args:
            gaussian_block_size: Block size for adaptive Gaussian threshold (must be odd)
            gaussian_c: Constant subtracted from the mean for adaptive threshold
        """
        # Ensure block size is odd and at least 3
        if gaussian_block_size % 2 == 0:
            gaussian_block_size += 1
        self.gaussian_block_size = max(3, gaussian_block_size)
        self.gaussian_c = gaussian_c
        
        # Morphological operation kernels for noise reduction
        self.noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Check CUDA availability
        self.use_cuda = self._check_cuda_availability()
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available for OpenCV operations."""
        try:
            return (USE_CUDA_IF_AVAILABLE and 
                    hasattr(cv2, 'cuda') and 
                    cv2.cuda.getCudaEnabledDeviceCount() > 0)
        except Exception:
            return False
    
    def analyze_region(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> List[np.ndarray]:
        """
        Analyze a region of the image for contours using enhanced processing.
        
        Args:
            image: Input BGR image
            roi: Region of interest as (x, y, width, height). If None, analyze entire image
            
        Returns:
            List of detected contours
        """
        # Extract region of interest if specified
        if roi is not None:
            x, y, w, h = roi
            # Ensure ROI is within image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return []
                
            region = image[y:y+h, x:x+w].copy()
        else:
            region = image.copy()
        
        # Convert to grayscale
        gray = self._convert_to_grayscale(region)
        
        # Apply adaptive Gaussian thresholding
        binary = self.apply_adaptive_gaussian_threshold(gray)
        
        # Apply morphological operations for noise reduction
        binary = self._reduce_noise(binary)
        
        # Find enhanced contours
        contours = self.find_enhanced_contours(binary)
        
        # Adjust contour coordinates if ROI was used
        if roi is not None:
            x, y, _, _ = roi
            adjusted_contours = []
            for contour in contours:
                adjusted_contour = contour.copy()
                adjusted_contour[:, :, 0] += x  # Adjust x coordinates
                adjusted_contour[:, :, 1] += y  # Adjust y coordinates
                adjusted_contours.append(adjusted_contour)
            return adjusted_contours
        
        return contours
    
    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale with CUDA acceleration if available.
        
        Args:
            image: Input BGR image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 2:
            return image  # Already grayscale
        
        if self.use_cuda:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
                gray = gpu_gray.download()
                return gray
            except Exception:
                # Fall back to CPU processing
                pass
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def apply_adaptive_gaussian_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Adaptive Gaussian thresholding for better edge detection
        in varying lighting conditions.
        
        Args:
            gray: Grayscale input image
            
        Returns:
            Binary thresholded image
        """
        if self.use_cuda:
            try:
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray)
                
                # CUDA adaptive threshold
                gpu_binary = cv2.cuda.threshold(
                    gpu_gray, 0, 255, 
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]
                
                binary = gpu_binary.download()
                return binary
            except Exception:
                # Fall back to CPU processing
                pass
        
        # CPU-based adaptive Gaussian threshold
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.gaussian_block_size,
            self.gaussian_c
        )
        
        return binary
    
    def _reduce_noise(self, binary: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to reduce noise in the binary image.
        
        Args:
            binary: Binary input image
            
        Returns:
            Noise-reduced binary image
        """
        # Remove small noise with opening
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.noise_kernel)
        
        # Fill small gaps with closing
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, self.closing_kernel)
        
        return denoised
    
    def find_enhanced_contours(self, binary: np.ndarray) -> List[np.ndarray]:
        """
        Find contours using enhanced detection suitable for nested shapes.
        
        Args:
            binary: Binary input image
            
        Returns:
            List of detected contours
        """
        # Use RETR_TREE to detect nested shapes (shapes within shapes)
        contours, hierarchy = cv2.findContours(
            binary, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return []
        
        # Filter contours by minimum area and validity
        filtered_contours = []
        min_area = 100  # Minimum contour area in pixels
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Skip very small contours
            if area < min_area:
                continue
            
            # Skip contours that are too simple (less than 5 points)
            if len(contour) < 5:
                continue
            
            # Check if contour is valid (not degenerate)
            if self._is_valid_contour(contour):
                filtered_contours.append(contour)
        
        # Sort by area (largest first) for better shape detection
        filtered_contours.sort(key=cv2.contourArea, reverse=True)
        
        return filtered_contours
    
    def _is_valid_contour(self, contour: np.ndarray) -> bool:
        """
        Check if a contour is valid for shape detection.
        
        Args:
            contour: Input contour
            
        Returns:
            True if contour is valid, False otherwise
        """
        # Check minimum area
        area = cv2.contourArea(contour)
        if area < 100:
            return False
        
        # Check perimeter
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 20:
            return False
        
        # Check aspect ratio (avoid extremely elongated shapes)
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return False
        
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 10:  # Too elongated
            return False
        
        return True
    
    def get_processing_stats(self) -> dict:
        """
        Get processing statistics and configuration.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            'gaussian_block_size': self.gaussian_block_size,
            'gaussian_c': self.gaussian_c,
            'cuda_enabled': self.use_cuda,
            'noise_kernel_size': self.noise_kernel.shape,
            'closing_kernel_size': self.closing_kernel.shape
        }