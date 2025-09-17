#!/usr/bin/env python3
"""
4K Performance Demo

This script demonstrates the performance improvement for 4K video processing
with and without high-resolution optimization.
"""

import cv2
import numpy as np
import time
from typing import Tuple

def create_realistic_4k_frame() -> np.ndarray:
    """Create a realistic 4K frame with A4 sheet for testing."""
    width, height = 3840, 2160
    
    # Create realistic background with texture
    frame = np.random.randint(200, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some realistic elements
    # Desk texture
    desk_color = [180, 160, 140]
    for i in range(3):
        frame[:, :, i] = np.clip(frame[:, :, i] * 0.7 + desk_color[i] * 0.3, 0, 255)
    
    # Add A4 sheet in center
    a4_width = int(width * 0.4)  # A4 takes 40% of frame width
    a4_height = int(a4_width * 297 / 210)  # A4 aspect ratio
    
    x_center = width // 2
    y_center = height // 2
    
    x1 = x_center - a4_width // 2
    x2 = x_center + a4_width // 2
    y1 = y_center - a4_height // 2
    y2 = y_center + a4_height // 2
    
    # Draw A4 sheet
    cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 0), 10)  # Shadow
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)     # White paper
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)            # Border
    
    # Add some objects on the A4 sheet
    obj_size = max(30, a4_width // 40)
    
    # Circle object
    circle_x = x_center - a4_width // 4
    circle_y = y_center
    cv2.circle(frame, (circle_x, circle_y), obj_size, (80, 80, 80), -1)
    cv2.circle(frame, (circle_x, circle_y), obj_size, (0, 0, 0), 2)
    
    # Rectangle object
    rect_x = x_center + a4_width // 4 - obj_size
    rect_y = y_center - obj_size // 2
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + obj_size*2, rect_y + obj_size), (80, 80, 80), -1)
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + obj_size*2, rect_y + obj_size), (0, 0, 0), 2)
    
    # Add some realistic noise and compression artifacts
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return frame

def benchmark_detection_methods(frame: np.ndarray, num_runs: int = 5) -> Tuple[float, float, bool, bool]:
    """
    Benchmark both standard and optimized detection methods.
    
    Returns:
        Tuple of (standard_time, optimized_time, standard_success, optimized_success)
    """
    print(f"Benchmarking detection on {frame.shape[1]}x{frame.shape[0]} frame...")
    
    # Test standard detection
    print("  Testing standard detection...")
    standard_times = []
    standard_results = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        # Use the standard detection method
        from detection import _find_a4_quad_standard
        quad = _find_a4_quad_standard(frame)
        
        end_time = time.perf_counter()
        
        detection_time = (end_time - start_time) * 1000
        standard_times.append(detection_time)
        standard_results.append(quad is not None)
        
        print(f"    Run {i+1}: {detection_time:.2f}ms, Success: {quad is not None}")
    
    # Test optimized detection
    print("  Testing optimized detection...")
    optimized_times = []
    optimized_results = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        # Use the optimized detection method
        from detection import find_a4_quad
        quad = find_a4_quad(frame)
        
        end_time = time.perf_counter()
        
        detection_time = (end_time - start_time) * 1000
        optimized_times.append(detection_time)
        optimized_results.append(quad is not None)
        
        print(f"    Run {i+1}: {detection_time:.2f}ms, Success: {quad is not None}")
    
    avg_standard = sum(standard_times) / len(standard_times)
    avg_optimized = sum(optimized_times) / len(optimized_times)
    standard_success = all(standard_results)
    optimized_success = all(optimized_results)
    
    return avg_standard, avg_optimized, standard_success, optimized_success

def demo_interactive_performance():
    """Demonstrate interactive performance with 4K frames."""
    print("\n=== Interactive Performance Demo ===")
    
    frame = create_realistic_4k_frame()
    
    # Simulate interactive session
    print("Simulating interactive session with mouse movements...")
    
    from interaction_manager import setup_interactive_inspect_mode
    from measure import create_shape_data
    
    # Create some test shapes
    shapes = [
        {
            "type": "circle",
            "center": (1920//2 - 400, 1080//2),
            "radius_px": 50,
            "diameter_mm": 25.0,
            "area_px": 7854,
            "inner": False
        },
        {
            "type": "rectangle",
            "box": np.array([[1920//2 + 200, 1080//2 - 30], [1920//2 + 400, 1080//2 - 30], 
                           [1920//2 + 400, 1080//2 + 30], [1920//2 + 200, 1080//2 + 30]]),
            "width_mm": 40.0,
            "height_mm": 12.0,
            "area_px": 7200,
            "inner": False
        }
    ]
    
    # Test with optimization enabled
    print("\nTesting with optimization enabled...")
    start_time = time.perf_counter()
    
    manager = setup_interactive_inspect_mode(
        shapes, frame, "4K Performance Demo", enable_performance_optimization=True
    )
    
    # Simulate mouse movements
    mouse_positions = [
        (960, 540), (1000, 540), (1040, 540), (1080, 540), (1120, 540),
        (1160, 540), (1200, 540), (1240, 540), (1280, 540), (1320, 540)
    ]
    
    render_times = []
    for x, y in mouse_positions:
        render_start = time.perf_counter()
        needs_render = manager.handle_mouse_move(x, y)
        if needs_render:
            result = manager.render_current_state()
        render_end = time.perf_counter()
        
        render_time = (render_end - render_start) * 1000
        render_times.append(render_time)
    
    setup_time = (time.perf_counter() - start_time) * 1000
    avg_render_time = sum(render_times) / len(render_times) if render_times else 0
    
    print(f"  Setup time: {setup_time:.2f}ms")
    print(f"  Average render time: {avg_render_time:.2f}ms")
    print(f"  Total mouse events: {len(mouse_positions)}")
    
    # Get performance stats
    stats = manager.get_performance_stats()
    manager.cleanup()
    
    return stats

def main():
    """Run the 4K performance demonstration."""
    print("=" * 60)
    print("4K PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    print("Creating realistic 4K test frame...")
    frame = create_realistic_4k_frame()
    print(f"Created {frame.shape[1]}x{frame.shape[0]} frame ({frame.nbytes / 1024 / 1024:.1f} MB)")
    
    # Benchmark detection methods
    print("\n=== Detection Performance Comparison ===")
    standard_time, optimized_time, standard_success, optimized_success = benchmark_detection_methods(frame)
    
    print(f"\nResults:")
    print(f"  Standard Detection: {standard_time:.2f}ms avg, Success: {standard_success}")
    print(f"  Optimized Detection: {optimized_time:.2f}ms avg, Success: {optimized_success}")
    
    if optimized_time > 0:
        speedup = standard_time / optimized_time
        improvement = ((standard_time - optimized_time) / standard_time) * 100
        print(f"  Performance Improvement: {improvement:.1f}% faster ({speedup:.2f}x speedup)")
    
    # Demo interactive performance
    interactive_stats = demo_interactive_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE DEMO SUMMARY")
    print("=" * 60)
    
    print(f"4K Detection Performance:")
    print(f"  Standard: {standard_time:.2f}ms")
    print(f"  Optimized: {optimized_time:.2f}ms")
    print(f"  Improvement: {((standard_time - optimized_time) / standard_time) * 100:.1f}%")
    
    if interactive_stats:
        print(f"\nInteractive Performance:")
        if "render_stats" in interactive_stats:
            rs = interactive_stats["render_stats"]
            print(f"  Render FPS: {rs['avg_fps']:.1f}")
            print(f"  Render time: {rs['avg_ms']:.2f}ms")
        
        print(f"  Optimization enabled: {interactive_stats.get('optimization_enabled', False)}")
        
        if "frame_optimizer" in interactive_stats:
            fo = interactive_stats["frame_optimizer"]
            print(f"  Frame skip ratio: {fo['skip_ratio']:.1%}")
    
    print(f"\n✅ 4K performance optimization successfully demonstrated!")
    
    # Cleanup
    from detection import cleanup_detection_resources
    cleanup_detection_resources()

if __name__ == "__main__":
    main()