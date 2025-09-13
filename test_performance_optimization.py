#!/usr/bin/env python3
"""
Performance Optimization Test Suite

This script tests the performance optimizations implemented for the interactive
inspect mode, including rendering performance, mouse event handling, and frame
rate optimization.
"""

import cv2
import numpy as np
import time
import sys
from typing import List, Dict, Any

# Import the modules we need to test
from performance_profiler import (
    PerformanceProfiler, OptimizedRenderer, FrameRateOptimizer,
    create_performance_test_suite, run_performance_benchmark
)
from rendering import SelectiveRenderer
from interaction_manager import InteractionManager, setup_interactive_inspect_mode
from measure import create_shape_data


def create_test_shapes() -> List[Dict[str, Any]]:
    """Create test shapes for performance testing."""
    test_shapes = []
    
    # Create test circle
    circle_data = {
        "type": "circle",
        "center": (200, 200),
        "radius_px": 50,
        "diameter_mm": 25.0,
        "area_px": 7854,
        "inner": False
    }
    test_shapes.append(circle_data)
    
    # Create test rectangle
    rectangle_data = {
        "type": "rectangle",
        "box": np.array([[100, 100], [300, 100], [300, 200], [100, 200]]),
        "width_mm": 40.0,
        "height_mm": 20.0,
        "area_px": 8000,
        "inner": False
    }
    test_shapes.append(rectangle_data)
    
    # Create multiple smaller shapes for stress testing
    for i in range(8):
        x = 50 + (i % 4) * 100
        y = 300 + (i // 4) * 100
        
        if i % 2 == 0:
            # Circle
            shape = {
                "type": "circle",
                "center": (x, y),
                "radius_px": 20,
                "diameter_mm": 10.0,
                "area_px": 1256,
                "inner": False
            }
        else:
            # Rectangle
            shape = {
                "type": "rectangle",
                "box": np.array([[x-15, y-10], [x+15, y-10], [x+15, y+10], [x-15, y+10]]),
                "width_mm": 15.0,
                "height_mm": 10.0,
                "area_px": 600,
                "inner": False
            }
        
        test_shapes.append(shape)
    
    return test_shapes


def create_test_image() -> np.ndarray:
    """Create a test A4 image for performance testing."""
    # Create a white A4-sized image (approximately 2480x3508 pixels at 300 DPI)
    # For testing, we'll use a smaller size
    height, width = 800, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some texture to make it more realistic
    noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
    image = cv2.subtract(image, noise)
    
    return image


def test_basic_rendering_performance():
    """Test basic rendering performance without optimization."""
    print("\n=== Basic Rendering Performance Test ===")
    
    shapes = create_test_shapes()
    image = create_test_image()
    
    renderer = SelectiveRenderer()
    profiler = PerformanceProfiler()
    
    # Test rendering performance
    num_renders = 100
    print(f"Performing {num_renders} renders...")
    
    for i in range(num_renders):
        profiler.start_timing("render")
        
        state = {
            "hovered": i % len(shapes) if shapes else None,
            "selected": (i // 2) % len(shapes) if shapes else None
        }
        
        result = renderer.render_complete_state(image, state, shapes)
        profiler.end_timing("render")
    
    stats = profiler.get_render_stats()
    print(f"Average render time: {stats['avg_ms']:.2f}ms ({stats['avg_fps']:.1f} FPS)")
    print(f"Min/Max render time: {stats['min_ms']:.2f}ms / {stats['max_ms']:.2f}ms")
    
    return stats


def test_optimized_rendering_performance():
    """Test optimized rendering performance."""
    print("\n=== Optimized Rendering Performance Test ===")
    
    shapes = create_test_shapes()
    image = create_test_image()
    
    base_renderer = SelectiveRenderer()
    optimized_renderer = OptimizedRenderer(base_renderer)
    
    # Test optimized rendering performance
    num_renders = 100
    print(f"Performing {num_renders} optimized renders...")
    
    for i in range(num_renders):
        state = {
            "hovered": i % len(shapes) if shapes else None,
            "selected": (i // 2) % len(shapes) if shapes else None
        }
        
        # Force render for first few frames, then let optimization kick in
        force_render = i < 5
        result = optimized_renderer.render_optimized(image, state, shapes, force_render)
    
    stats = optimized_renderer.get_performance_stats()
    if "render_stats" in stats:
        rs = stats["render_stats"]
        print(f"Average render time: {rs['avg_ms']:.2f}ms ({rs['avg_fps']:.1f} FPS)")
        print(f"Min/Max render time: {rs['min_ms']:.2f}ms / {rs['max_ms']:.2f}ms")
    
    optimized_renderer.print_performance_report()
    return stats


def test_frame_rate_optimization():
    """Test frame rate optimization."""
    print("\n=== Frame Rate Optimization Test ===")
    
    optimizer = FrameRateOptimizer(target_fps=60.0, min_fps=30.0)
    
    # Simulate frame rendering with varying render times
    render_times = [10, 15, 20, 25, 30, 35, 40, 45, 50, 20, 15, 10]  # milliseconds
    
    frames_rendered = 0
    frames_skipped = 0
    
    print("Simulating frame rendering with varying render times...")
    
    for i, render_time in enumerate(render_times * 10):  # Repeat pattern
        if optimizer.should_render_frame():
            optimizer.frame_rendered(render_time)
            frames_rendered += 1
            time.sleep(render_time / 1000.0)  # Simulate render time
        else:
            frames_skipped += 1
        
        time.sleep(0.001)  # Small delay between frames
    
    metrics = optimizer.get_performance_metrics()
    print(f"Frames rendered: {metrics['frames_rendered']}")
    print(f"Frames skipped: {metrics['frames_skipped']}")
    print(f"Skip ratio: {metrics['skip_ratio']:.2%}")
    print(f"Effective FPS: {metrics['effective_fps']:.1f}")
    print(f"Adaptive threshold: {metrics['adaptive_threshold_ms']:.1f}ms")
    
    return metrics


def test_interaction_manager_performance():
    """Test interaction manager performance with optimization."""
    print("\n=== Interaction Manager Performance Test ===")
    
    shapes = create_test_shapes()
    image = create_test_image()
    
    # Test with optimization enabled
    print("Testing with optimization enabled...")
    manager_opt = InteractionManager(shapes, image, enable_performance_optimization=True)
    
    # Simulate mouse movements
    num_moves = 200
    for i in range(num_moves):
        x = (i * 3) % image.shape[1]
        y = (i * 2) % image.shape[0]
        manager_opt.handle_mouse_move(x, y)
        
        # Render occasionally
        if i % 10 == 0:
            result = manager_opt.render_current_state()
    
    opt_stats = manager_opt.get_performance_stats()
    print("Optimized manager performance:")
    if "render_stats" in opt_stats:
        rs = opt_stats["render_stats"]
        print(f"  Renders: {rs['count']}, Avg time: {rs['avg_ms']:.2f}ms")
    if "mouse_stats" in opt_stats:
        ms = opt_stats["mouse_stats"]
        print(f"  Mouse events: {ms['count']}, Avg time: {ms['avg_ms']:.2f}ms")
    
    manager_opt.cleanup()
    
    # Test without optimization for comparison
    print("\nTesting without optimization...")
    manager_std = InteractionManager(shapes, image, enable_performance_optimization=False)
    
    # Simulate same mouse movements
    for i in range(num_moves):
        x = (i * 3) % image.shape[1]
        y = (i * 2) % image.shape[0]
        manager_std.handle_mouse_move(x, y)
        
        # Render occasionally
        if i % 10 == 0:
            result = manager_std.render_current_state()
    
    std_stats = manager_std.get_performance_stats()
    print("Standard manager performance:")
    print(f"  Optimization enabled: {std_stats['optimization_enabled']}")
    
    manager_std.cleanup()
    
    return opt_stats, std_stats


def test_multiple_shapes_performance():
    """Test performance with multiple detected shapes."""
    print("\n=== Multiple Shapes Performance Test ===")
    
    # Create many shapes for stress testing
    shapes = []
    for i in range(50):  # 50 shapes
        x = 50 + (i % 10) * 50
        y = 50 + (i // 10) * 80
        
        if i % 2 == 0:
            shape = {
                "type": "circle",
                "center": (x, y),
                "radius_px": 15,
                "diameter_mm": 7.5,
                "area_px": 706,
                "inner": False
            }
        else:
            shape = {
                "type": "rectangle",
                "box": np.array([[x-10, y-8], [x+10, y-8], [x+10, y+8], [x-10, y+8]]),
                "width_mm": 10.0,
                "height_mm": 8.0,
                "area_px": 320,
                "inner": False
            }
        
        shapes.append(shape)
    
    image = create_test_image()
    
    print(f"Testing with {len(shapes)} shapes...")
    
    # Run benchmark
    run_performance_benchmark(shapes, image)
    
    return len(shapes)


def run_comprehensive_performance_test():
    """Run comprehensive performance test suite."""
    print("=" * 60)
    print("INTERACTIVE INSPECT MODE - PERFORMANCE TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Basic rendering performance
        basic_stats = test_basic_rendering_performance()
        
        # Test 2: Optimized rendering performance
        opt_stats = test_optimized_rendering_performance()
        
        # Test 3: Frame rate optimization
        frame_stats = test_frame_rate_optimization()
        
        # Test 4: Interaction manager performance
        manager_opt_stats, manager_std_stats = test_interaction_manager_performance()
        
        # Test 5: Multiple shapes performance
        shape_count = test_multiple_shapes_performance()
        
        # Summary
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        
        if basic_stats and opt_stats:
            basic_fps = basic_stats.get('avg_fps', 0)
            opt_fps = opt_stats.get('render_stats', {}).get('avg_fps', 0)
            if basic_fps > 0 and opt_fps > 0:
                improvement = ((opt_fps - basic_fps) / basic_fps) * 100
                print(f"Rendering Performance Improvement: {improvement:+.1f}%")
                print(f"  Basic: {basic_fps:.1f} FPS")
                print(f"  Optimized: {opt_fps:.1f} FPS")
        
        print(f"Frame Rate Optimization: {frame_stats['skip_ratio']:.1%} frames skipped")
        print(f"Stress Test: Successfully handled {shape_count} shapes")
        
        print("\nAll performance tests completed successfully!")
        
    except Exception as e:
        print(f"Error during performance testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_comprehensive_performance_test()
    sys.exit(0 if success else 1)