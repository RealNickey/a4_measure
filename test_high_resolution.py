#!/usr/bin/env python3
"""
High Resolution Performance Test

This script tests the high-resolution optimizations for 4K+ video processing,
including GPU acceleration and adaptive scaling.
"""

import cv2
import numpy as np
import time
import sys
from typing import Dict, Any

from high_resolution_optimizer import create_high_resolution_optimizer, GPUAcceleratedDetection
from detection import find_a4_quad, get_detection_performance_stats, cleanup_detection_resources


def create_test_frames() -> Dict[str, np.ndarray]:
    """Create test frames at different resolutions."""
    test_frames = {}
    
    # Create test patterns that simulate A4 detection scenarios
    resolutions = {
        "1080p": (1920, 1080),
        "1440p": (2560, 1440), 
        "4K": (3840, 2160),
        "5K": (5120, 2880),
        "8K": (7680, 4320)
    }
    
    for name, (width, height) in resolutions.items():
        print(f"Creating {name} test frame ({width}x{height})...")
        
        # Create white background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add A4-like rectangle in the center
        a4_width = int(width * 0.6)
        a4_height = int(a4_width * 297 / 210)  # A4 aspect ratio
        
        if a4_height > height * 0.8:
            a4_height = int(height * 0.8)
            a4_width = int(a4_height * 210 / 297)
        
        x_center = width // 2
        y_center = height // 2
        
        x1 = x_center - a4_width // 2
        x2 = x_center + a4_width // 2
        y1 = y_center - a4_height // 2
        y2 = y_center + a4_height // 2
        
        # Draw A4 rectangle with black border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 8)
        cv2.rectangle(frame, (x1+8, y1+8), (x2-8, y2-8), (255, 255, 255), -1)
        
        # Add some objects inside
        obj_size = max(20, min(width, height) // 100)
        
        # Circle
        circle_center = (x_center - a4_width//4, y_center)
        cv2.circle(frame, circle_center, obj_size, (100, 100, 100), -1)
        
        # Rectangle
        rect_x = x_center + a4_width//4 - obj_size
        rect_y = y_center - obj_size//2
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + obj_size*2, rect_y + obj_size), (100, 100, 100), -1)
        
        # Add some noise for realism
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        test_frames[name] = frame
    
    return test_frames


def test_detection_performance():
    """Test A4 detection performance at different resolutions."""
    print("\n=== Detection Performance Test ===")
    
    test_frames = create_test_frames()
    results = {}
    
    for resolution, frame in test_frames.items():
        print(f"\nTesting {resolution} ({frame.shape[1]}x{frame.shape[0]})...")
        
        # Test multiple runs for average
        times = []
        detections = []
        
        for run in range(5):
            start_time = time.perf_counter()
            quad = find_a4_quad(frame)
            end_time = time.perf_counter()
            
            detection_time = (end_time - start_time) * 1000
            times.append(detection_time)
            detections.append(quad is not None)
        
        avg_time = sum(times) / len(times)
        detection_rate = sum(detections) / len(detections)
        
        results[resolution] = {
            "avg_time_ms": avg_time,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "detection_rate": detection_rate,
            "frame_size": frame.shape
        }
        
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Detection rate: {detection_rate:.1%}")
        print(f"  Time range: {min(times):.2f}ms - {max(times):.2f}ms")
    
    return results


def test_gpu_acceleration():
    """Test GPU acceleration performance."""
    print("\n=== GPU Acceleration Test ===")
    
    optimizer = create_high_resolution_optimizer()
    
    if not optimizer.gpu_available:
        print("[WARN] GPU acceleration not available, skipping GPU tests")
        return {}
    
    gpu_detector = GPUAcceleratedDetection(optimizer)
    test_frames = create_test_frames()
    
    results = {}
    
    for resolution, frame in test_frames.items():
        if resolution in ["1080p", "1440p"]:  # Skip lower resolutions for GPU test
            continue
            
        print(f"\nTesting GPU acceleration on {resolution}...")
        
        # Test GPU detection
        gpu_times = []
        gpu_detections = []
        
        for run in range(3):
            start_time = time.perf_counter()
            quad = gpu_detector.find_a4_quad_optimized(frame)
            end_time = time.perf_counter()
            
            gpu_time = (end_time - start_time) * 1000
            gpu_times.append(gpu_time)
            gpu_detections.append(quad is not None)
        
        # Test CPU detection for comparison
        cpu_times = []
        cpu_detections = []
        
        for run in range(3):
            start_time = time.perf_counter()
            # Force CPU detection by using standard function
            from detection import _find_a4_quad_standard
            quad = _find_a4_quad_standard(frame)
            end_time = time.perf_counter()
            
            cpu_time = (end_time - start_time) * 1000
            cpu_times.append(cpu_time)
            cpu_detections.append(quad is not None)
        
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
        
        results[resolution] = {
            "gpu_time_ms": avg_gpu_time,
            "cpu_time_ms": avg_cpu_time,
            "speedup": speedup,
            "gpu_detection_rate": sum(gpu_detections) / len(gpu_detections),
            "cpu_detection_rate": sum(cpu_detections) / len(cpu_detections)
        }
        
        print(f"  GPU time: {avg_gpu_time:.2f}ms")
        print(f"  CPU time: {avg_cpu_time:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    return results


def test_adaptive_scaling():
    """Test adaptive scaling performance."""
    print("\n=== Adaptive Scaling Test ===")
    
    optimizer = create_high_resolution_optimizer()
    test_frames = create_test_frames()
    
    results = {}
    
    for resolution, frame in test_frames.items():
        print(f"\nTesting adaptive scaling on {resolution}...")
        
        # Test frame optimization
        start_time = time.perf_counter()
        optimized_frame, scale_factor = optimizer.optimize_frame_for_detection(frame)
        optimization_time = (time.perf_counter() - start_time) * 1000
        
        # Calculate size reduction
        original_pixels = frame.shape[0] * frame.shape[1]
        optimized_pixels = optimized_frame.shape[0] * optimized_frame.shape[1]
        size_reduction = 1 - (optimized_pixels / original_pixels)
        
        results[resolution] = {
            "original_size": frame.shape[:2][::-1],
            "optimized_size": optimized_frame.shape[:2][::-1],
            "scale_factor": scale_factor,
            "size_reduction": size_reduction,
            "optimization_time_ms": optimization_time
        }
        
        print(f"  Scale factor: {scale_factor:.2f}")
        print(f"  Size reduction: {size_reduction:.1%}")
        print(f"  Optimization time: {optimization_time:.2f}ms")
        print(f"  Original: {frame.shape[1]}x{frame.shape[0]}")
        print(f"  Optimized: {optimized_frame.shape[1]}x{optimized_frame.shape[0]}")
    
    return results


def test_memory_usage():
    """Test memory usage with high-resolution frames."""
    print("\n=== Memory Usage Test ===")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Create and process multiple high-resolution frames
        test_frames = create_test_frames()
        
        for i in range(3):  # Process frames multiple times
            print(f"\nProcessing cycle {i+1}/3...")
            
            for resolution, frame in test_frames.items():
                if resolution in ["4K", "5K", "8K"]:  # Only test high-res
                    quad = find_a4_quad(frame)
                    
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"  {resolution}: {current_memory:.1f} MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"\nFinal memory usage: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        return {
            "initial_mb": initial_memory,
            "final_mb": final_memory,
            "increase_mb": memory_increase
        }
        
    except ImportError:
        print("[WARN] psutil not available, skipping detailed memory test")
        
        # Simple memory test without psutil
        test_frames = create_test_frames()
        
        for resolution, frame in test_frames.items():
            if resolution in ["4K", "5K", "8K"]:
                quad = find_a4_quad(frame)
                print(f"  Processed {resolution} frame")
        
        return {
            "initial_mb": 0,
            "final_mb": 0,
            "increase_mb": 0
        }


def run_comprehensive_high_res_test():
    """Run comprehensive high-resolution performance tests."""
    print("=" * 70)
    print("HIGH RESOLUTION OPTIMIZATION - COMPREHENSIVE TEST")
    print("=" * 70)
    
    try:
        # Test 1: Detection performance
        detection_results = test_detection_performance()
        
        # Test 2: GPU acceleration
        gpu_results = test_gpu_acceleration()
        
        # Test 3: Adaptive scaling
        scaling_results = test_adaptive_scaling()
        
        # Test 4: Memory usage
        memory_results = test_memory_usage()
        
        # Summary
        print("\n" + "=" * 70)
        print("HIGH RESOLUTION TEST SUMMARY")
        print("=" * 70)
        
        print("\nDetection Performance:")
        for resolution, result in detection_results.items():
            print(f"  {resolution}: {result['avg_time_ms']:.2f}ms avg, "
                  f"{result['detection_rate']:.1%} success rate")
        
        if gpu_results:
            print("\nGPU Acceleration:")
            for resolution, result in gpu_results.items():
                print(f"  {resolution}: {result['speedup']:.2f}x speedup "
                      f"({result['cpu_time_ms']:.2f}ms → {result['gpu_time_ms']:.2f}ms)")
        
        print("\nAdaptive Scaling:")
        for resolution, result in scaling_results.items():
            print(f"  {resolution}: {result['scale_factor']:.2f}x scale, "
                  f"{result['size_reduction']:.1%} size reduction")
        
        print(f"\nMemory Usage: {memory_results['increase_mb']:.1f} MB increase")
        
        # Get final performance stats
        perf_stats = get_detection_performance_stats()
        if perf_stats:
            print(f"\nOptimizer Status:")
            print(f"  GPU Available: {perf_stats.get('gpu_available', False)}")
            if "current_profile" in perf_stats:
                print(f"  Profile Used: {perf_stats['current_profile']}")
        
        print("\nAll high-resolution tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during high-resolution testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        cleanup_detection_resources()


if __name__ == "__main__":
    success = run_comprehensive_high_res_test()
    sys.exit(0 if success else 1)