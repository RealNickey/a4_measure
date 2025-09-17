"""
High Resolution Optimizer for 4K+ Video Processing

This module provides GPU acceleration and adaptive processing optimizations
for handling high-resolution video inputs (4K, 8K) efficiently using CUDA
and intelligent downsampling strategies.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any, List
from threading import Thread, Lock
import queue
from dataclasses import dataclass

from config import USE_CUDA_IF_AVAILABLE, PX_PER_MM


@dataclass
class ResolutionProfile:
    """Configuration profile for different input resolutions."""
    name: str
    max_width: int
    max_height: int
    detection_scale: float  # Scale factor for detection processing
    processing_scale: float  # Scale factor for object processing
    target_fps: int
    use_gpu: bool
    enable_threading: bool


class HighResolutionOptimizer:
    """
    Optimizer for handling high-resolution video inputs efficiently.
    
    Features:
    - GPU acceleration using CUDA when available
    - Adaptive resolution scaling based on input size
    - Multi-threaded processing pipeline
    - Intelligent caching and frame skipping
    """
    
    def __init__(self):
        """Initialize the high-resolution optimizer."""
        self.gpu_available = self._check_gpu_availability()
        self.current_profile = None
        self.frame_cache = {}
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.processing_lock = Lock()
        self.is_processing = False
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        
        # Define resolution profiles
        self.profiles = {
            "1080p": ResolutionProfile("1080p", 1920, 1080, 1.0, 1.0, 60, False, False),
            "1440p": ResolutionProfile("1440p", 2560, 1440, 0.75, 0.85, 45, True, True),
            "4k": ResolutionProfile("4K", 3840, 2160, 0.5, 0.7, 30, True, True),
            "5k": ResolutionProfile("5K", 5120, 2880, 0.4, 0.6, 24, True, True),
            "8k": ResolutionProfile("8K", 7680, 4320, 0.25, 0.4, 15, True, True)
        }
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            if not USE_CUDA_IF_AVAILABLE:
                return False
            
            if not hasattr(cv2, 'cuda'):
                print("[INFO] OpenCV CUDA not available")
                return False
            
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count == 0:
                print("[INFO] No CUDA-enabled devices found")
                return False
            
            # Test GPU functionality
            test_mat = cv2.cuda_GpuMat()
            test_array = np.ones((100, 100), dtype=np.uint8)
            test_mat.upload(test_array)
            test_result = test_mat.download()
            
            print(f"[INFO] GPU acceleration available with {device_count} CUDA device(s)")
            return True
            
        except Exception as e:
            print(f"[WARN] GPU acceleration not available: {e}")
            return False
    
    def select_optimal_profile(self, frame_width: int, frame_height: int) -> ResolutionProfile:
        """
        Select the optimal processing profile based on input resolution.
        
        Args:
            frame_width: Input frame width
            frame_height: Input frame height
            
        Returns:
            Optimal ResolutionProfile for the input resolution
        """
        # Find the best matching profile based on resolution
        selected_profile = None
        
        if frame_width >= 7680 or frame_height >= 4320:
            selected_profile = self.profiles["8k"]
        elif frame_width >= 5120 or frame_height >= 2880:
            selected_profile = self.profiles["5k"]
        elif frame_width >= 3840 or frame_height >= 2160:
            selected_profile = self.profiles["4k"]
        elif frame_width >= 2560 or frame_height >= 1440:
            selected_profile = self.profiles["1440p"]
        else:
            selected_profile = self.profiles["1080p"]
        
        # Create a copy to avoid modifying the original
        import copy
        selected_profile = copy.deepcopy(selected_profile)
        
        # Disable GPU if not available
        if not self.gpu_available:
            selected_profile.use_gpu = False
        
        print(f"[INFO] Selected {selected_profile.name} profile for {frame_width}x{frame_height} input")
        print(f"[INFO] Detection scale: {selected_profile.detection_scale:.2f}, "
              f"Processing scale: {selected_profile.processing_scale:.2f}")
        
        return selected_profile    

    def optimize_frame_for_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Optimize frame for A4 detection by scaling and GPU processing.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (optimized_frame, scale_factor)
        """
        if self.current_profile is None:
            self.current_profile = self.select_optimal_profile(frame.shape[1], frame.shape[0])
        
        scale = self.current_profile.detection_scale
        
        if scale == 1.0:
            return frame, 1.0
        
        # Calculate new dimensions
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        
        # Use GPU resize if available
        if self.current_profile.use_gpu and self.gpu_available:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_resized = cv2.cuda.resize(gpu_frame, (new_width, new_height))
                resized_frame = gpu_resized.download()
                return resized_frame, scale
            except Exception as e:
                print(f"[WARN] GPU resize failed, falling back to CPU: {e}")
        
        # CPU fallback
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame, scale
    
    def scale_detection_result(self, quad: Optional[np.ndarray], scale_factor: float) -> Optional[np.ndarray]:
        """
        Scale detection results back to original resolution.
        
        Args:
            quad: Detected quad coordinates (or None)
            scale_factor: Scale factor used for detection
            
        Returns:
            Scaled quad coordinates in original resolution
        """
        if quad is None or scale_factor == 1.0:
            return quad
        
        return quad / scale_factor
    
    def gpu_preprocess_edges(self, frame_gray: np.ndarray, gauss_blur: int = 5, 
                           canny_low: int = 50, canny_high: int = 150) -> np.ndarray:
        """
        GPU-accelerated edge preprocessing.
        
        Args:
            frame_gray: Grayscale input frame
            gauss_blur: Gaussian blur kernel size
            canny_low: Canny low threshold
            canny_high: Canny high threshold
            
        Returns:
            Edge-detected image
        """
        if not self.gpu_available:
            # CPU fallback
            blur = cv2.GaussianBlur(frame_gray, (gauss_blur, gauss_blur), 0)
            edges = cv2.Canny(blur, canny_low, canny_high)
            return edges
        
        try:
            # GPU processing
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame_gray)
            
            # Gaussian blur
            gaussian_filter = cv2.cuda.createGaussianFilter(
                gpu_frame.type(), gpu_frame.type(), (gauss_blur, gauss_blur), 0
            )
            gpu_blurred = gaussian_filter.apply(gpu_frame)
            
            # Canny edge detection
            canny_detector = cv2.cuda.createCannyEdgeDetector(canny_low, canny_high)
            gpu_edges = canny_detector.detect(gpu_blurred)
            
            # Download result
            edges = gpu_edges.download()
            return edges
            
        except Exception as e:
            print(f"[WARN] GPU edge processing failed: {e}")
            # CPU fallback
            blur = cv2.GaussianBlur(frame_gray, (gauss_blur, gauss_blur), 0)
            edges = cv2.Canny(blur, canny_low, canny_high)
            return edges
    
    def record_performance(self, operation: str, duration_ms: float):
        """
        Record performance metrics for adaptive optimization.
        
        Args:
            operation: Operation name ('frame' or 'detection')
            duration_ms: Duration in milliseconds
        """
        if operation == "frame":
            self.frame_times.append(duration_ms)
            if len(self.frame_times) > 50:
                self.frame_times.pop(0)
        elif operation == "detection":
            self.detection_times.append(duration_ms)
            if len(self.detection_times) > 30:
                self.detection_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "gpu_available": self.gpu_available,
            "current_profile": self.current_profile.name if self.current_profile else None
        }
        
        if self.frame_times:
            stats["frame_processing"] = {
                "avg_ms": sum(self.frame_times) / len(self.frame_times),
                "min_ms": min(self.frame_times),
                "max_ms": max(self.frame_times),
                "count": len(self.frame_times)
            }
        
        if self.detection_times:
            stats["detection_processing"] = {
                "avg_ms": sum(self.detection_times) / len(self.detection_times),
                "min_ms": min(self.detection_times),
                "max_ms": max(self.detection_times),
                "count": len(self.detection_times)
            }
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        self.frame_cache.clear()
        self.frame_times.clear()
        self.detection_times.clear()
        self.current_profile = None


class GPUAcceleratedDetection:
    """GPU-accelerated A4 detection for high-resolution inputs."""
    
    def __init__(self, optimizer: HighResolutionOptimizer):
        """
        Initialize GPU-accelerated detection.
        
        Args:
            optimizer: High-resolution optimizer instance
        """
        self.optimizer = optimizer
        self.gpu_available = optimizer.gpu_available
    
    def find_a4_quad_optimized(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        GPU-optimized A4 quad detection for high-resolution frames.
        
        Args:
            frame_bgr: Input BGR frame
            
        Returns:
            Detected A4 quad coordinates or None
        """
        start_time = time.perf_counter()
        
        try:
            # Optimize frame for detection
            detection_frame, scale_factor = self.optimizer.optimize_frame_for_detection(frame_bgr)
            
            # Convert to grayscale
            if self.gpu_available:
                try:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(detection_frame)
                    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                    gray = gpu_gray.download()
                except Exception:
                    gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            
            # GPU-accelerated edge detection
            edges = self.optimizer.gpu_preprocess_edges(gray)
            
            # Dilate edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours (CPU operation)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Process contours to find A4
            quad = self._find_best_a4_quad(contours, detection_frame.shape)
            
            # Scale result back to original resolution
            result = self.optimizer.scale_detection_result(quad, scale_factor)
            
            # Record performance
            duration = (time.perf_counter() - start_time) * 1000
            self.optimizer.record_performance("detection", duration)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] GPU-optimized detection failed: {e}")
            return None
    
    def _find_best_a4_quad(self, contours: List[np.ndarray], frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """
        Find the best A4 quad from contours.
        
        Args:
            contours: List of contours
            frame_shape: Shape of the frame (h, w, c)
            
        Returns:
            Best A4 quad or None
        """
        from utils import order_points, approx_quad
        from config import MIN_A4_AREA_RATIO, ASPECT_MIN, ASPECT_MAX
        
        h, w = frame_shape[:2]
        area_frame = w * h
        
        best = None
        best_area = 0
        
        for cnt in contours[:20]:  # Check top 20 contours
            area = cv2.contourArea(cnt)
            if area < MIN_A4_AREA_RATIO * area_frame:
                continue
            
            quad = approx_quad(cnt, epsilon_ratio=0.02)
            if quad is None:
                continue
            
            # Order and check aspect ratio
            rect = order_points(quad.astype(np.float32))
            (tl, tr, br, bl) = rect
            
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            
            width = (widthA + widthB) * 0.5
            height = (heightA + heightB) * 0.5
            
            if width < 10 or height < 10:
                continue
            
            ratio = max(width, height) / max(1.0, min(width, height))
            if ASPECT_MIN <= ratio <= ASPECT_MAX:
                if area > best_area:
                    best = rect
                    best_area = area
        
        return best


def create_high_resolution_optimizer() -> HighResolutionOptimizer:
    """
    Create and initialize a high-resolution optimizer.
    
    Returns:
        Configured HighResolutionOptimizer instance
    """
    optimizer = HighResolutionOptimizer()
    
    # Print initialization info
    print(f"[INFO] High-resolution optimizer initialized")
    print(f"[INFO] GPU acceleration: {'Available' if optimizer.gpu_available else 'Not available'}")
    
    return optimizer