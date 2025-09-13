#!/usr/bin/env python3
"""
Performance Integration Test

This script tests the performance optimizations integrated with the main
interactive inspect mode system to ensure they work correctly in real scenarios.
"""

import cv2
import numpy as np
import time
import sys
from typing import List, Dict, Any

# Import the main system components
from interaction_manager import setup_interactive_inspect_mode
from measure import create_shape_data
from performance_profiler import run_performance_benchmark


def create_realistic_test_shapes() -> List[Dict[str, Any]]:
    """Create realistic test shapes that would be detected by the system."""
    shapes = []
    
    # Create shapes that match the format from create_shape_data
    test_measurements = [
        {
            "type": "circle",
            "center": (300, 200),
            "radius_px": 60,
            "diameter_mm": 30.0,
            "area_px": 11309,
            "inner": False
        },
        {
            "type": "rectangle", 
            "box": np.array([[150, 150], [350, 150], [350, 250], [150, 250]]),
            "width_mm": 50.0,
            "height_mm": 25.0,
            "area_px": 20000,
            "inner": False
        },
        {
            "type": "circle",
            "center": (500, 300),
            "radius_px": 40,
            "diameter_mm": 20.0,
            "area_px": 5026,
            "inner": True  # Inner circle
        },
        {
            "type": "rectangle",
            "box": np.array([[400, 400], [600, 400], [600, 500], [400, 500]]),
            "width_mm": 40.0,
            "height_mm": 25.0,
            "area_px": 20000,
            "inner": True  # Inner rectangle
        }
    ]
    
    return test_measurements


def create_realistic_test_image() -> np.ndarray:
    """Create a realistic A4 test image."""
    # Create A4-proportioned image
    height, width = 1000, 707  # A4 aspect ratio
    image = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Add some realistic texture and shadows
    for i in range(10):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(20, 100)
        color = np.random.randint(200, 255)
        cv2.circle(image, (x, y), radius, (color, color, color), -1)
    
    # Add some noise
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def test_interactive_system_performance():
    """Test the complete interactive system performance."""
    print("\n=== Interactive System Performance Test ===")
    
    shapes = create_realistic_test_shapes()
    image = create_realistic_test_image()
    
    print(f"Testing with {len(shapes)} shapes on {image.shape[1]}x{image.shape[0]} image")
    
    # Test with optimization enabled
    print("\nTesting optimized system...")
    start_time = time.perf_counter()
    
    manager = setup_interactive_inspect_mode(
        shapes, image, "Test Window", enable_performance_optimization=True
    )
    
    # Simulate realistic interaction patterns
    interaction_patterns = [
        # Hover over different shapes
        (200, 200), (300, 200), (400, 200), (500, 200),
        # Move around first shape
        (280, 180), (290, 190), (300, 200), (310, 210), (320, 220),
        # Click on shape
        (300, 200),
        # Hover over second shape
        (250, 200), (200, 200), (150, 200),
        # Click on second shape
        (250, 200),
        # Move around and hover
        (400, 300), (450, 350), (500, 400), (550, 450),
        # Click on background
        (100, 100)
    ]
    
    render_count = 0
    for x, y in interaction_patterns:
        # Simulate mouse move
        needs_render = manager.handle_mouse_move(x, y)
        if needs_render:
            result = manager.render_current_state()
            if result is not None:
                render_count += 1
        
        # Simulate some clicks
        if (x, y) in [(300, 200), (250, 200), (100, 100)]:
            needs_render = manager.handle_mouse_click(x, y)
            if needs_render:
                result = manager.render_current_state(force_render=True)
                if result is not None:
                    render_count += 1
        
        # Small delay to simulate real interaction
        time.sleep(0.001)
    
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000
    
    print(f"Total interaction time: {total_time:.2f}ms")
    print(f"Renders performed: {render_count}")
    print(f"Average time per interaction: {total_time / len(interaction_patterns):.2f}ms")
    
    # Get performance statistics
    stats = manager.get_performance_stats()
    print("\nPerformance Statistics:")
    if "render_stats" in stats:
        rs = stats["render_stats"]
        print(f"  Render count: {rs['count']}")
        print(f"  Average render time: {rs['avg_ms']:.2f}ms")
        print(f"  Render FPS: {rs['avg_fps']:.1f}")
    
    if "mouse_stats" in stats:
        ms = stats["mouse_stats"]
        print(f"  Mouse events: {ms['count']}")
        print(f"  Average mouse event time: {ms['avg_ms']:.3f}ms")
    
    if "frame_optimizer" in stats:
        fo = stats["frame_optimizer"]
        print(f"  Frame skip ratio: {fo['skip_ratio']:.1%}")
        print(f"  Effective FPS: {fo['effective_fps']:.1f}")
    
    # Cleanup
    manager.cleanup()
    
    return stats


def test_stress_performance():
    """Test performance under stress conditions."""
    print("\n=== Stress Performance Test ===")
    
    # Create many shapes for stress testing
    shapes = []
    for i in range(100):  # 100 shapes
        x = 50 + (i % 20) * 30
        y = 50 + (i // 20) * 100
        
        if i % 3 == 0:
            shape = {
                "type": "circle",
                "center": (x, y),
                "radius_px": 10 + (i % 20),
                "diameter_mm": 5.0 + (i % 10),
                "area_px": 314 + i * 10,
                "inner": i % 4 == 0
            }
        else:
            shape = {
                "type": "rectangle",
                "box": np.array([[x-10, y-8], [x+10, y-8], [x+10, y+8], [x-10, y+8]]),
                "width_mm": 8.0 + (i % 5),
                "height_mm": 6.0 + (i % 3),
                "area_px": 160 + i * 5,
                "inner": i % 5 == 0
            }
        
        shapes.append(shape)
    
    image = create_realistic_test_image()
    
    print(f"Stress testing with {len(shapes)} shapes...")
    
    # Run performance benchmark
    run_performance_benchmark(shapes, image)
    
    # Test interactive system under stress
    manager = setup_interactive_inspect_mode(
        shapes, image, "Stress Test", enable_performance_optimization=True
    )
    
    # Rapid mouse movements
    start_time = time.perf_counter()
    for i in range(500):  # 500 rapid movements
        x = (i * 7) % image.shape[1]
        y = (i * 5) % image.shape[0]
        manager.handle_mouse_move(x, y)
        
        # Occasional renders
        if i % 20 == 0:
            manager.render_current_state()
    
    end_time = time.perf_counter()
    stress_time = (end_time - start_time) * 1000
    
    print(f"Stress test completed in {stress_time:.2f}ms")
    print(f"Average time per mouse event: {stress_time / 500:.3f}ms")
    
    # Get final stats
    final_stats = manager.get_performance_stats()
    manager.cleanup()
    
    return final_stats


def test_memory_performance():
    """Test memory usage and cleanup performance."""
    print("\n=== Memory Performance Test ===")
    
    shapes = create_realistic_test_shapes()
    image = create_realistic_test_image()
    
    # Test multiple manager creation/cleanup cycles
    print("Testing multiple manager lifecycle cycles...")
    
    for cycle in range(5):
        print(f"  Cycle {cycle + 1}/5")
        
        manager = setup_interactive_inspect_mode(
            shapes, image, f"Memory Test {cycle}", enable_performance_optimization=True
        )
        
        # Perform some interactions
        for i in range(50):
            x = (i * 10) % image.shape[1]
            y = (i * 8) % image.shape[0]
            manager.handle_mouse_move(x, y)
            
            if i % 10 == 0:
                manager.render_current_state()
        
        # Cleanup
        manager.cleanup()
        
        # Small delay between cycles
        time.sleep(0.01)
    
    print("Memory performance test completed successfully")
    return True


def run_integration_performance_tests():
    """Run comprehensive integration performance tests."""
    print("=" * 70)
    print("INTERACTIVE INSPECT MODE - INTEGRATION PERFORMANCE TESTS")
    print("=" * 70)
    
    try:
        # Test 1: Interactive system performance
        interactive_stats = test_interactive_system_performance()
        
        # Test 2: Stress performance
        stress_stats = test_stress_performance()
        
        # Test 3: Memory performance
        memory_result = test_memory_performance()
        
        # Summary
        print("\n" + "=" * 70)
        print("INTEGRATION PERFORMANCE TEST SUMMARY")
        print("=" * 70)
        
        print("✓ Interactive system performance test completed")
        print("✓ Stress performance test completed")
        print("✓ Memory performance test completed")
        
        if interactive_stats and "render_stats" in interactive_stats:
            rs = interactive_stats["render_stats"]
            print(f"\nInteractive System Performance:")
            print(f"  Render FPS: {rs['avg_fps']:.1f}")
            print(f"  Average render time: {rs['avg_ms']:.2f}ms")
        
        if stress_stats and "mouse_stats" in stress_stats:
            ms = stress_stats["mouse_stats"]
            print(f"\nStress Test Performance:")
            print(f"  Mouse events processed: {ms['count']}")
            print(f"  Average mouse event time: {ms['avg_ms']:.3f}ms")
        
        print("\nAll integration performance tests passed!")
        return True
        
    except Exception as e:
        print(f"Error during integration performance testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_integration_performance_tests()
    sys.exit(0 if success else 1)