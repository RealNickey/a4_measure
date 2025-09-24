"""
Performance test suite for manual selection system.

This module tests that manual selection mode doesn't degrade system responsiveness
and maintains acceptable performance characteristics.

Requirements tested: 1.4, 1.5, 4.4, 4.5
"""

import unittest
import cv2
import numpy as np
import time
import threading
import psutil
import os
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import manual selection components
try:
    from extended_interaction_manager import ExtendedInteractionManager
    from manual_selection_engine import ManualSelectionEngine
    from shape_snapping_engine import ShapeSnappingEngine
    from enhanced_contour_analyzer import EnhancedContourAnalyzer
    from selection_mode import SelectionMode
    from measure import classify_and_measure_manual_selection
    from detection import a4_scale_mm_per_px
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


class PerformanceMonitor:
    """Monitor system performance during testing."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.initial_cpu_percent = self.process.cpu_percent()
        
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Performance monitoring loop."""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                self.memory_samples.append(memory_mb)
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(0.1)  # Sample every 100ms
            except:
                break
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.memory_samples:
            return {}
        
        return {
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": self.memory_samples[-1] if self.memory_samples else self.initial_memory,
            "max_memory_mb": max(self.memory_samples) if self.memory_samples else self.initial_memory,
            "avg_memory_mb": sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else self.initial_memory,
            "memory_increase_mb": (self.memory_samples[-1] - self.initial_memory) if self.memory_samples else 0,
            "max_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0,
            "avg_cpu_percent": sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            "sample_count": len(self.memory_samples)
        }


class TestManualSelectionPerformance(unittest.TestCase):
    """Test performance characteristics of manual selection system."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Manual selection components not available")
            
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        self.manual_engine = ManualSelectionEngine(self.analyzer)
        
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        
        # Create test images of various sizes and complexities
        self.test_images = self._create_performance_test_images()
        
        # Performance thresholds
        self.max_single_selection_time = 1.0  # seconds
        self.max_memory_increase = 100  # MB
        self.max_cpu_usage = 80  # percent
    
    def _create_performance_test_images(self) -> Dict[str, np.ndarray]:
        """Create test images for performance testing."""
        images = {}
        
        # Small image (typical webcam resolution)
        small_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.circle(small_img, (320, 240), 50, (0, 0, 0), -1)
        cv2.rectangle(small_img, (100, 100), (200, 200), (0, 0, 0), -1)
        images["small"] = small_img
        
        # Medium image (HD resolution)
        medium_img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        cv2.circle(medium_img, (640, 360), 100, (0, 0, 0), -1)
        cv2.rectangle(medium_img, (200, 200), (400, 400), (0, 0, 0), -1)
        images["medium"] = medium_img
        
        # Large image (Full HD resolution)
        large_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        cv2.circle(large_img, (960, 540), 150, (0, 0, 0), -1)
        cv2.rectangle(large_img, (300, 300), (600, 600), (0, 0, 0), -1)
        images["large"] = large_img
        
        # Complex image with many shapes
        complex_img = np.ones((800, 800, 3), dtype=np.uint8) * 255
        
        # Add many random shapes
        np.random.seed(42)  # For reproducible results
        for i in range(50):
            x, y = np.random.randint(50, 750, 2)
            if i % 2 == 0:
                radius = np.random.randint(20, 60)
                cv2.circle(complex_img, (x, y), radius, (0, 0, 0), -1)
            else:
                w, h = np.random.randint(40, 100, 2)
                cv2.rectangle(complex_img, (x - w//2, y - h//2), (x + w//2, y + h//2), (100, 100, 100), -1)
        
        images["complex"] = complex_img
        
        # Noisy image
        noisy_img = np.random.randint(100, 200, (600, 800, 3), dtype=np.uint8)
        cv2.circle(noisy_img, (400, 300), 80, (0, 0, 0), -1)
        cv2.rectangle(noisy_img, (100, 100), (300, 250), (255, 255, 255), -1)
        
        # Add noise
        for _ in range(500):
            x, y = np.random.randint(0, 800), np.random.randint(0, 600)
            radius = np.random.randint(1, 5)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(noisy_img, (x, y), radius, color, -1)
        
        images["noisy"] = noisy_img
        
        return images
    
    def test_single_selection_performance(self):
        """Test performance of single shape selection operations."""
        test_cases = [
            ("small", (270, 190, 100, 100), 0.3),
            ("medium", (590, 310, 100, 100), 0.5),
            ("large", (910, 490, 100, 100), 0.8),
            ("complex", (100, 100, 200, 200), 1.0),
            ("noisy", (350, 250, 160, 160), 1.2),
        ]
        
        for image_name, selection_rect, max_time in test_cases:
            with self.subTest(image=image_name):
                image = self.test_images[image_name]
                
                # Test circle detection
                start_time = time.time()
                circle_result = self.snap_engine.snap_to_shape(
                    image, selection_rect, SelectionMode.MANUAL_CIRCLE
                )
                circle_time = time.time() - start_time
                
                # Test rectangle detection
                start_time = time.time()
                rect_result = self.snap_engine.snap_to_shape(
                    image, selection_rect, SelectionMode.MANUAL_RECTANGLE
                )
                rect_time = time.time() - start_time
                
                # Check performance
                self.assertLess(circle_time, max_time, 
                               f"Circle detection took {circle_time:.3f}s, expected < {max_time}s")
                self.assertLess(rect_time, max_time, 
                               f"Rectangle detection took {rect_time:.3f}s, expected < {max_time}s")
                
                print(f"Performance {image_name}: Circle {circle_time:.3f}s, Rectangle {rect_time:.3f}s")
    
    def test_rapid_successive_selections(self):
        """Test performance when making rapid successive selections."""
        image = self.test_images["medium"]
        
        # Define multiple selection areas
        selections = [
            (100, 100, 150, 150),
            (300, 200, 150, 150),
            (500, 300, 150, 150),
            (700, 100, 150, 150),
            (200, 400, 150, 150),
            (600, 500, 150, 150),
            (100, 500, 150, 150),
            (800, 400, 150, 150),
        ]
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        results = []
        
        for i, selection in enumerate(selections):
            mode = SelectionMode.MANUAL_CIRCLE if i % 2 == 0 else SelectionMode.MANUAL_RECTANGLE
            
            selection_start = time.time()
            result = self.snap_engine.snap_to_shape(image, selection, mode)
            selection_time = time.time() - selection_start
            
            results.append({
                "selection": selection,
                "mode": mode,
                "result": result,
                "time": selection_time
            })
            
            # Each individual selection should be fast
            self.assertLess(selection_time, self.max_single_selection_time,
                           f"Selection {i} took {selection_time:.3f}s")
        
        total_time = time.time() - start_time
        monitor.stop_monitoring()
        
        # Check overall performance
        avg_time = total_time / len(selections)
        self.assertLess(avg_time, 0.5, f"Average selection time {avg_time:.3f}s too slow")
        
        # Check system resources
        stats = monitor.get_stats()
        if stats:
            self.assertLess(stats["memory_increase_mb"], self.max_memory_increase,
                           f"Memory increased by {stats['memory_increase_mb']:.1f}MB")
            self.assertLess(stats["max_cpu_percent"], self.max_cpu_usage,
                           f"CPU usage peaked at {stats['max_cpu_percent']:.1f}%")
        
        print(f"Rapid selections: {len(selections)} selections in {total_time:.3f}s "
              f"(avg: {avg_time:.3f}s)")
        if stats:
            print(f"Memory: {stats['memory_increase_mb']:.1f}MB increase, "
                  f"CPU: {stats['max_cpu_percent']:.1f}% peak")
    
    def test_concurrent_selections(self):
        """Test performance with concurrent selection operations."""
        image = self.test_images["large"]
        
        # Define selections for concurrent processing
        concurrent_selections = [
            ((200, 200, 200, 200), SelectionMode.MANUAL_CIRCLE),
            ((600, 200, 200, 200), SelectionMode.MANUAL_RECTANGLE),
            ((1000, 200, 200, 200), SelectionMode.MANUAL_CIRCLE),
            ((200, 600, 200, 200), SelectionMode.MANUAL_RECTANGLE),
            ((600, 600, 200, 200), SelectionMode.MANUAL_CIRCLE),
            ((1000, 600, 200, 200), SelectionMode.MANUAL_RECTANGLE),
        ]
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        def perform_selection(selection_data):
            selection_rect, mode = selection_data
            start_time = time.time()
            result = self.snap_engine.snap_to_shape(image, selection_rect, mode)
            end_time = time.time()
            return {
                "selection": selection_rect,
                "mode": mode,
                "result": result,
                "time": end_time - start_time
            }
        
        # Execute concurrent selections
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_selection = {
                executor.submit(perform_selection, selection_data): selection_data
                for selection_data in concurrent_selections
            }
            
            results = []
            for future in as_completed(future_to_selection):
                try:
                    result = future.result(timeout=5.0)
                    results.append(result)
                except Exception as e:
                    self.fail(f"Concurrent selection failed: {e}")
        
        total_time = time.time() - start_time
        monitor.stop_monitoring()
        
        # Check that all selections completed
        self.assertEqual(len(results), len(concurrent_selections))
        
        # Check individual selection times
        for result in results:
            self.assertLess(result["time"], self.max_single_selection_time,
                           f"Concurrent selection took {result['time']:.3f}s")
        
        # Concurrent execution should be faster than sequential
        sequential_time_estimate = sum(result["time"] for result in results)
        self.assertLess(total_time, sequential_time_estimate * 0.8,
                       f"Concurrent execution not efficient: {total_time:.3f}s vs estimated {sequential_time_estimate:.3f}s")
        
        # Check system resources
        stats = monitor.get_stats()
        if stats:
            self.assertLess(stats["memory_increase_mb"], self.max_memory_increase * 2,  # Allow more for concurrent
                           f"Memory increased by {stats['memory_increase_mb']:.1f}MB")
        
        print(f"Concurrent selections: {len(results)} selections in {total_time:.3f}s")
        if stats:
            print(f"Memory: {stats['memory_increase_mb']:.1f}MB increase")
    
    def test_memory_stability_extended_use(self):
        """Test memory stability during extended use."""
        image = self.test_images["medium"]
        selection = (500, 300, 200, 200)
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Perform many selections to test for memory leaks
        num_iterations = 100
        memory_samples = []
        
        for i in range(num_iterations):
            # Alternate between modes
            mode = SelectionMode.MANUAL_CIRCLE if i % 2 == 0 else SelectionMode.MANUAL_RECTANGLE
            
            result = self.snap_engine.snap_to_shape(image, selection, mode)
            
            # Sample memory every 10 iterations
            if i % 10 == 0:
                current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Check for excessive memory growth
                if len(memory_samples) > 1:
                    memory_increase = current_memory - memory_samples[0]
                    self.assertLess(memory_increase, self.max_memory_increase,
                                   f"Memory increased by {memory_increase:.1f}MB after {i+1} iterations")
        
        monitor.stop_monitoring()
        
        # Check final memory state
        stats = monitor.get_stats()
        if stats and memory_samples:
            final_increase = memory_samples[-1] - memory_samples[0]
            self.assertLess(final_increase, 50,  # Allow some increase for caching
                           f"Memory increased by {final_increase:.1f}MB over {num_iterations} iterations")
            
            print(f"Memory stability: {final_increase:.1f}MB increase over {num_iterations} iterations")
    
    def test_cpu_usage_efficiency(self):
        """Test CPU usage efficiency during selection operations."""
        image = self.test_images["complex"]
        selections = [
            (50, 50, 150, 150),
            (200, 200, 150, 150),
            (350, 350, 150, 150),
            (500, 100, 150, 150),
            (100, 500, 150, 150),
        ]
        
        monitor = PerformanceMonitor()
        
        # Baseline CPU usage
        time.sleep(0.5)  # Let system settle
        baseline_cpu = psutil.Process(os.getpid()).cpu_percent(interval=0.1)
        
        monitor.start_monitoring()
        
        # Perform selections
        for selection in selections:
            self.snap_engine.snap_to_shape(image, selection, SelectionMode.MANUAL_CIRCLE)
            self.snap_engine.snap_to_shape(image, selection, SelectionMode.MANUAL_RECTANGLE)
        
        monitor.stop_monitoring()
        
        stats = monitor.get_stats()
        if stats:
            # CPU usage should be reasonable
            self.assertLess(stats["max_cpu_percent"], self.max_cpu_usage,
                           f"CPU usage peaked at {stats['max_cpu_percent']:.1f}%")
            
            # Average CPU usage should be moderate
            self.assertLess(stats["avg_cpu_percent"], 50,
                           f"Average CPU usage {stats['avg_cpu_percent']:.1f}% too high")
            
            print(f"CPU efficiency: Peak {stats['max_cpu_percent']:.1f}%, "
                  f"Average {stats['avg_cpu_percent']:.1f}%")
    
    def test_responsiveness_during_processing(self):
        """Test system responsiveness during shape processing."""
        image = self.test_images["large"]
        selection = (800, 400, 300, 300)
        
        # Measure responsiveness by timing simple operations during processing
        responsiveness_times = []
        
        def measure_responsiveness():
            """Measure time for simple operations."""
            for _ in range(10):
                start = time.time()
                # Simple operation that should be fast
                _ = np.ones((100, 100), dtype=np.uint8)
                end = time.time()
                responsiveness_times.append(end - start)
                time.sleep(0.01)
        
        # Start responsiveness measurement in background
        responsiveness_thread = threading.Thread(target=measure_responsiveness)
        responsiveness_thread.start()
        
        # Perform shape detection
        start_time = time.time()
        result = self.snap_engine.snap_to_shape(image, selection, SelectionMode.MANUAL_CIRCLE)
        processing_time = time.time() - start_time
        
        # Wait for responsiveness measurement to complete
        responsiveness_thread.join()
        
        # Check that simple operations remained fast
        if responsiveness_times:
            max_responsiveness_time = max(responsiveness_times)
            avg_responsiveness_time = sum(responsiveness_times) / len(responsiveness_times)
            
            self.assertLess(max_responsiveness_time, 0.01,  # 10ms
                           f"System became unresponsive: {max_responsiveness_time:.4f}s")
            self.assertLess(avg_responsiveness_time, 0.005,  # 5ms average
                           f"Average responsiveness degraded: {avg_responsiveness_time:.4f}s")
            
            print(f"Responsiveness: Max {max_responsiveness_time:.4f}s, "
                  f"Avg {avg_responsiveness_time:.4f}s during {processing_time:.3f}s processing")
    
    def test_scalability_with_image_size(self):
        """Test performance scalability with different image sizes."""
        selection_rect = (100, 100, 200, 200)
        
        # Test with different image sizes
        size_tests = [
            ("small", 480 * 640),
            ("medium", 720 * 1280),
            ("large", 1080 * 1920),
        ]
        
        performance_data = []
        
        for image_name, pixel_count in size_tests:
            image = self.test_images[image_name]
            
            # Measure processing time
            start_time = time.time()
            result = self.snap_engine.snap_to_shape(image, selection_rect, SelectionMode.MANUAL_CIRCLE)
            processing_time = time.time() - start_time
            
            performance_data.append({
                "name": image_name,
                "pixel_count": pixel_count,
                "processing_time": processing_time
            })
            
            print(f"Scalability {image_name}: {processing_time:.3f}s for {pixel_count:,} pixels")
        
        # Check that processing time scales reasonably with image size
        # (Should be roughly linear or sub-linear, not exponential)
        if len(performance_data) >= 2:
            small_data = performance_data[0]
            large_data = performance_data[-1]
            
            size_ratio = large_data["pixel_count"] / small_data["pixel_count"]
            time_ratio = large_data["processing_time"] / small_data["processing_time"]
            
            # Time ratio should not be much larger than size ratio
            self.assertLess(time_ratio, size_ratio * 2,
                           f"Processing time scales poorly: {time_ratio:.2f}x time for {size_ratio:.2f}x pixels")
            
            print(f"Scalability ratio: {time_ratio:.2f}x time for {size_ratio:.2f}x pixels")


class TestSystemIntegrationPerformance(unittest.TestCase):
    """Test performance of integrated manual selection system."""
    
    def setUp(self):
        """Set up integration performance tests."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Manual selection components not available")
            
        # Initialize full system
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        self.manual_engine = ManualSelectionEngine(self.analyzer)
        
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        
        # Create test image
        self.test_image = np.ones((800, 800, 3), dtype=np.uint8) * 255
        cv2.circle(self.test_image, (300, 300), 80, (0, 0, 0), -1)
        cv2.rectangle(self.test_image, (500, 200), (700, 400), (0, 0, 0), -1)
    
    def test_end_to_end_workflow_performance(self):
        """Test performance of complete manual selection workflow."""
        selection_rect = (250, 250, 100, 100)
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Complete workflow: detection + measurement
        start_time = time.time()
        
        # 1. Shape detection
        shape_result = self.snap_engine.snap_to_shape(
            self.test_image, selection_rect, SelectionMode.MANUAL_CIRCLE
        )
        detection_time = time.time() - start_time
        
        # 2. Measurement (if shape detected)
        measurement_result = None
        if shape_result:
            measurement_start = time.time()
            measurement_result = classify_and_measure_manual_selection(
                self.test_image, selection_rect, shape_result,
                self.mm_per_px_x, self.mm_per_px_y
            )
            measurement_time = time.time() - measurement_start
        else:
            measurement_time = 0
        
        total_time = time.time() - start_time
        monitor.stop_monitoring()
        
        # Check performance
        self.assertLess(detection_time, 1.0, f"Detection took {detection_time:.3f}s")
        self.assertLess(measurement_time, 0.1, f"Measurement took {measurement_time:.3f}s")
        self.assertLess(total_time, 1.1, f"Total workflow took {total_time:.3f}s")
        
        # Check system resources
        stats = monitor.get_stats()
        if stats:
            self.assertLess(stats["memory_increase_mb"], 50,
                           f"Workflow used {stats['memory_increase_mb']:.1f}MB")
        
        print(f"End-to-end performance: Detection {detection_time:.3f}s, "
              f"Measurement {measurement_time:.3f}s, Total {total_time:.3f}s")
    
    def test_mode_switching_performance(self):
        """Test performance when rapidly switching between modes."""
        selection_rect = (450, 150, 200, 200)
        
        # Test rapid mode switching
        modes = [SelectionMode.MANUAL_CIRCLE, SelectionMode.MANUAL_RECTANGLE] * 10
        
        start_time = time.time()
        results = []
        
        for mode in modes:
            mode_start = time.time()
            result = self.snap_engine.snap_to_shape(self.test_image, selection_rect, mode)
            mode_time = time.time() - mode_start
            
            results.append({
                "mode": mode,
                "result": result,
                "time": mode_time
            })
            
            # Each mode switch should be fast
            self.assertLess(mode_time, 0.5, f"Mode {mode.value} took {mode_time:.3f}s")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(modes)
        
        # Mode switching should not degrade performance
        self.assertLess(avg_time, 0.3, f"Average mode switch time {avg_time:.3f}s")
        
        print(f"Mode switching: {len(modes)} switches in {total_time:.3f}s "
              f"(avg: {avg_time:.3f}s)")


if __name__ == '__main__':
    # Configure test runner for performance testing
    unittest.TestLoader.sortTestMethodsUsing = None
    
    # Run performance tests
    print("="*60)
    print("MANUAL SELECTION PERFORMANCE TEST SUITE")
    print("="*60)
    
    # Check system specs
    print(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f}GB RAM")
    print(f"Python process: PID {os.getpid()}")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestManualSelectionPerformance))
    suite.addTest(unittest.makeSuite(TestSystemIntegrationPerformance))
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print performance summary
    print(f"\n{'='*60}")
    print(f"PERFORMANCE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nPERFORMANCE ISSUES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    if not result.failures and not result.errors:
        print("\n✅ All performance tests passed!")
        print("Manual selection system meets performance requirements.")