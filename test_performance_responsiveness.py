"""
Performance tests for real-time mouse interaction responsiveness.

Tests cover:
- Real-time mouse tracking performance
- Rendering performance during interaction
- Memory usage during extended interaction
- Scalability with many shapes
- Response time benchmarks
"""

import unittest
import time
import numpy as np
import os
from typing import List, Dict, Any

from interaction_manager import InteractionManager, create_interaction_manager
from hit_testing import HitTestingEngine, create_hit_testing_contour
from interaction_state import InteractionState

# Try to import psutil, but make it optional
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class TestPerformanceResponsiveness(unittest.TestCase):
    """Performance tests for interactive functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if HAS_PSUTIL:
            self.process = psutil.Process(os.getpid())
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        else:
            self.process = None
            self.initial_memory = 0
    
    def create_many_shapes(self, count: int) -> List[Dict[str, Any]]:
        """Create many test shapes for performance testing."""
        shapes = []
        np.random.seed(42)  # For reproducible results
        
        for i in range(count):
            if i % 2 == 0:
                # Create circle
                center_x = np.random.randint(50, 750)
                center_y = np.random.randint(50, 550)
                radius = np.random.uniform(10, 50)
                
                shapes.append({
                    'type': 'circle',
                    'center': (center_x, center_y),
                    'radius_px': radius,
                    'diameter_mm': radius * 2 * 0.2,  # Approximate mm conversion
                    'area_px': np.pi * radius * radius,
                    'hit_contour': create_hit_testing_contour('circle', center=(center_x, center_y), radius_px=radius),
                    'inner': False
                })
            else:
                # Create rectangle
                x1 = np.random.randint(50, 700)
                y1 = np.random.randint(50, 500)
                width = np.random.randint(20, 100)
                height = np.random.randint(20, 100)
                
                box = np.array([[x1, y1], [x1 + width, y1], [x1 + width, y1 + height], [x1, y1 + height]], dtype=np.int32)
                
                shapes.append({
                    'type': 'rectangle',
                    'box': box,
                    'width_mm': width * 0.2,  # Approximate mm conversion
                    'height_mm': height * 0.2,
                    'area_px': width * height,
                    'hit_contour': create_hit_testing_contour('rectangle', box=box),
                    'inner': False
                })
        
        return shapes
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, (end_time - start_time) * 1000  # Return time in milliseconds
    
    def test_hit_testing_performance(self):
        """Test hit testing performance with many shapes."""
        print("\n=== Hit Testing Performance ===")
        
        shape_counts = [10, 50, 100, 200, 500]
        max_acceptable_time = 5.0  # 5ms maximum for real-time interaction
        
        for count in shape_counts:
            shapes = self.create_many_shapes(count)
            engine = HitTestingEngine()
            
            # Test multiple points
            test_points = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
            
            total_time = 0
            for point in test_points:
                _, exec_time = self.measure_execution_time(
                    engine.find_shape_at_point, shapes, point[0], point[1]
                )
                total_time += exec_time
            
            avg_time = total_time / len(test_points)
            print(f"  {count:3d} shapes: {avg_time:.2f}ms average hit testing time")
            
            # Performance assertion
            self.assertLess(avg_time, max_acceptable_time, 
                          f"Hit testing with {count} shapes took {avg_time:.2f}ms (max: {max_acceptable_time}ms)")
    
    def test_mouse_move_performance(self):
        """Test mouse movement handling performance."""
        print("\n=== Mouse Movement Performance ===")
        
        shapes = self.create_many_shapes(100)
        warped_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        manager = create_interaction_manager(shapes, warped_image)
        
        # Simulate mouse movement path
        mouse_path = []
        for i in range(100):
            x = int(400 + 200 * np.sin(i * 0.1))
            y = int(300 + 150 * np.cos(i * 0.1))
            mouse_path.append((x, y))
        
        # Measure mouse movement handling
        total_time = 0
        for x, y in mouse_path:
            _, exec_time = self.measure_execution_time(
                manager.handle_mouse_move, x, y
            )
            total_time += exec_time
        
        avg_time = total_time / len(mouse_path)
        max_acceptable_time = 2.0  # 2ms for smooth 60fps interaction
        
        print(f"  Average mouse move time: {avg_time:.2f}ms")
        print(f"  Total time for 100 moves: {total_time:.2f}ms")
        
        self.assertLess(avg_time, max_acceptable_time,
                       f"Mouse movement handling took {avg_time:.2f}ms (max: {max_acceptable_time}ms)")
    
    def test_mouse_click_performance(self):
        """Test mouse click handling performance."""
        print("\n=== Mouse Click Performance ===")
        
        shapes = self.create_many_shapes(100)
        warped_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        manager = create_interaction_manager(shapes, warped_image)
        
        # Test clicks at various positions
        click_positions = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
        
        total_time = 0
        for x, y in click_positions:
            _, exec_time = self.measure_execution_time(
                manager.handle_mouse_click, x, y
            )
            total_time += exec_time
        
        avg_time = total_time / len(click_positions)
        max_acceptable_time = 3.0  # 3ms for click response
        
        print(f"  Average click time: {avg_time:.2f}ms")
        
        self.assertLess(avg_time, max_acceptable_time,
                       f"Mouse click handling took {avg_time:.2f}ms (max: {max_acceptable_time}ms)")
    
    def test_state_update_performance(self):
        """Test interaction state update performance."""
        print("\n=== State Update Performance ===")
        
        shapes = self.create_many_shapes(50)
        state = InteractionState(shapes)
        
        # Test hover state updates
        hover_times = []
        for i in range(100):
            shape_idx = i % len(shapes) if i % 2 == 0 else None
            _, exec_time = self.measure_execution_time(
                state.update_hover, shape_idx
            )
            hover_times.append(exec_time)
        
        # Test selection state updates
        selection_times = []
        for i in range(100):
            shape_idx = i % len(shapes) if i % 3 == 0 else None
            _, exec_time = self.measure_execution_time(
                state.update_selection, shape_idx
            )
            selection_times.append(exec_time)
        
        avg_hover_time = sum(hover_times) / len(hover_times)
        avg_selection_time = sum(selection_times) / len(selection_times)
        
        print(f"  Average hover update time: {avg_hover_time:.3f}ms")
        print(f"  Average selection update time: {avg_selection_time:.3f}ms")
        
        # State updates should be very fast (under 0.1ms)
        self.assertLess(avg_hover_time, 0.1, f"Hover updates too slow: {avg_hover_time:.3f}ms")
        self.assertLess(avg_selection_time, 0.1, f"Selection updates too slow: {avg_selection_time:.3f}ms")
    
    def test_coordinate_transformation_performance(self):
        """Test coordinate transformation performance."""
        print("\n=== Coordinate Transformation Performance ===")
        
        from interaction_state import transform_display_to_original_coords, transform_original_to_display_coords
        
        # Test many coordinate transformations
        coordinates = [(i, j) for i in range(0, 800, 10) for j in range(0, 600, 10)]
        scale = 1.5
        
        # Test display to original transformation
        total_time = 0
        for x, y in coordinates:
            _, exec_time = self.measure_execution_time(
                transform_display_to_original_coords, x, y, scale
            )
            total_time += exec_time
        
        avg_time = total_time / len(coordinates)
        print(f"  Average transformation time: {avg_time:.4f}ms ({len(coordinates)} transformations)")
        
        # Transformations should be extremely fast
        self.assertLess(avg_time, 0.01, f"Coordinate transformations too slow: {avg_time:.4f}ms")
    
    def test_memory_usage_during_interaction(self):
        """Test memory usage during extended interaction."""
        print("\n=== Memory Usage Test ===")
        
        if not HAS_PSUTIL:
            print("  Skipping memory test - psutil not available")
            return
        
        shapes = self.create_many_shapes(200)
        warped_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        manager = create_interaction_manager(shapes, warped_image)
        
        # Record initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        print(f"  Initial memory: {initial_memory:.1f} MB")
        
        # Simulate extended interaction
        for cycle in range(10):
            # Simulate mouse movement and clicks
            for i in range(50):
                x = int(400 + 200 * np.sin(i * 0.1 + cycle))
                y = int(300 + 150 * np.cos(i * 0.1 + cycle))
                
                manager.handle_mouse_move(x, y)
                
                if i % 10 == 0:
                    manager.handle_mouse_click(x, y)
            
            # Check memory usage
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            print(f"  Cycle {cycle + 1}: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
            
            # Memory increase should be reasonable (less than 50MB)
            self.assertLess(memory_increase, 50, 
                          f"Memory usage increased too much: {memory_increase:.1f} MB")
        
        # Cleanup and check for memory leaks
        manager.cleanup()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = self.process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - self.initial_memory
        
        print(f"  Final memory: {final_memory:.1f} MB (+{total_increase:.1f} MB from start)")
        
        # Total memory increase should be reasonable
        self.assertLess(total_increase, 100, 
                       f"Total memory increase too high: {total_increase:.1f} MB")
    
    def test_scalability_with_shape_count(self):
        """Test performance scalability with increasing shape count."""
        print("\n=== Scalability Test ===")
        
        shape_counts = [10, 25, 50, 100, 200, 500]
        warped_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        results = []
        
        for count in shape_counts:
            shapes = self.create_many_shapes(count)
            manager = create_interaction_manager(shapes, warped_image)
            
            # Test standard interaction sequence
            test_sequence = [
                (100, 100), (200, 200), (300, 300), (400, 400), (500, 500)
            ]
            
            total_time = 0
            for x, y in test_sequence:
                # Mouse move + click
                _, move_time = self.measure_execution_time(manager.handle_mouse_move, x, y)
                _, click_time = self.measure_execution_time(manager.handle_mouse_click, x, y)
                total_time += move_time + click_time
            
            avg_time = total_time / len(test_sequence)
            results.append((count, avg_time))
            
            print(f"  {count:3d} shapes: {avg_time:.2f}ms average interaction time")
            
            manager.cleanup()
        
        # Check that performance doesn't degrade too much with more shapes
        # Allow some increase but not exponential
        for i in range(1, len(results)):
            prev_count, prev_time = results[i-1]
            curr_count, curr_time = results[i]
            
            # Performance should not increase more than linearly with shape count
            ratio_shapes = curr_count / prev_count
            ratio_time = curr_time / prev_time
            
            # Allow up to 3x time increase for significant shape count increase
            max_acceptable_ratio = min(3.0, ratio_shapes)
            
            self.assertLess(ratio_time, max_acceptable_ratio,
                          f"Performance degraded too much: {prev_count} -> {curr_count} shapes, "
                          f"{prev_time:.2f} -> {curr_time:.2f}ms (ratio: {ratio_time:.2f})")
    
    def test_rendering_performance_simulation(self):
        """Test rendering performance simulation (without actual OpenCV rendering)."""
        print("\n=== Rendering Performance Simulation ===")
        
        shapes = self.create_many_shapes(100)
        warped_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        manager = create_interaction_manager(shapes, warped_image)
        
        # Simulate rendering calls
        render_times = []
        
        for i in range(50):
            # Set some interaction state
            shape_idx = i % len(shapes) if i % 3 == 0 else None
            manager.state.update_hover(shape_idx)
            manager.state.update_selection(shape_idx)
            
            # Simulate render call (just the state preparation part)
            start_time = time.perf_counter()
            
            # Get state for rendering
            state_dict = manager.state.get_state_dict()
            instruction_text = manager.state.get_instruction_text()
            
            # Simulate some rendering work
            _ = len(shapes)  # Simple operation to simulate work
            
            end_time = time.perf_counter()
            render_times.append((end_time - start_time) * 1000)
        
        avg_render_time = sum(render_times) / len(render_times)
        max_render_time = max(render_times)
        
        print(f"  Average render preparation time: {avg_render_time:.3f}ms")
        print(f"  Maximum render preparation time: {max_render_time:.3f}ms")
        
        # Render preparation should be very fast
        self.assertLess(avg_render_time, 1.0, f"Render preparation too slow: {avg_render_time:.3f}ms")
        self.assertLess(max_render_time, 2.0, f"Max render preparation too slow: {max_render_time:.3f}ms")
    
    def test_concurrent_operations_performance(self):
        """Test performance when multiple operations happen simultaneously."""
        print("\n=== Concurrent Operations Test ===")
        
        shapes = self.create_many_shapes(100)
        warped_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        manager = create_interaction_manager(shapes, warped_image)
        
        # Simulate rapid concurrent operations
        start_time = time.perf_counter()
        
        for i in range(100):
            x = 100 + i
            y = 100 + i
            
            # Rapid sequence of operations
            manager.handle_mouse_move(x, y)
            manager.state.get_instruction_text()
            manager.handle_mouse_click(x, y)
            manager.state.get_state_dict()
            
            if i % 10 == 0:
                manager.state.clear_render_flag()
                manager.state.force_render()
        
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        avg_time_per_cycle = total_time / 100
        
        print(f"  Total time for 100 operation cycles: {total_time:.2f}ms")
        print(f"  Average time per cycle: {avg_time_per_cycle:.2f}ms")
        
        # Should handle concurrent operations efficiently
        self.assertLess(avg_time_per_cycle, 5.0, 
                       f"Concurrent operations too slow: {avg_time_per_cycle:.2f}ms per cycle")
    
    def test_edge_case_performance(self):
        """Test performance with edge cases."""
        print("\n=== Edge Case Performance ===")
        
        # Test with many overlapping shapes at same location
        overlapping_shapes = []
        for i in range(50):
            radius = 10 + i
            overlapping_shapes.append({
                'type': 'circle',
                'center': (200, 200),
                'radius_px': radius,
                'diameter_mm': radius * 0.4,
                'area_px': np.pi * radius * radius,
                'hit_contour': create_hit_testing_contour('circle', center=(200, 200), radius_px=radius),
                'inner': False
            })
        
        engine = HitTestingEngine()
        
        # Test hit testing performance with many overlapping shapes
        _, exec_time = self.measure_execution_time(
            engine.find_shape_at_point, overlapping_shapes, 200, 200
        )
        
        print(f"  Hit testing with 50 overlapping shapes: {exec_time:.2f}ms")
        
        # Should handle overlapping shapes efficiently
        self.assertLess(exec_time, 10.0, 
                       f"Overlapping shapes hit testing too slow: {exec_time:.2f}ms")
        
        # Test get_shapes_at_point performance
        _, exec_time = self.measure_execution_time(
            engine.get_shapes_at_point, overlapping_shapes, 200, 200
        )
        
        print(f"  Getting all shapes at point: {exec_time:.2f}ms")
        
        self.assertLess(exec_time, 15.0,
                       f"Getting all shapes at point too slow: {exec_time:.2f}ms")


class TestRealTimeResponsiveness(unittest.TestCase):
    """Test real-time responsiveness requirements."""
    
    def test_60fps_mouse_tracking(self):
        """Test that mouse tracking can maintain 60fps."""
        print("\n=== 60fps Mouse Tracking Test ===")
        
        shapes = []
        for i in range(20):  # Moderate number of shapes
            shapes.append({
                'type': 'circle',
                'center': (100 + i * 30, 100 + i * 20),
                'radius_px': 20.0,
                'diameter_mm': 16.0,
                'area_px': np.pi * 20 * 20,
                'hit_contour': create_hit_testing_contour('circle', center=(100 + i * 30, 100 + i * 20), radius_px=20.0),
                'inner': False
            })
        
        warped_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        manager = create_interaction_manager(shapes, warped_image)
        
        # 60fps means 16.67ms per frame
        target_frame_time = 16.67  # milliseconds
        
        # Simulate mouse tracking for one second at 60fps
        frame_times = []
        
        for frame in range(60):
            frame_start = time.perf_counter()
            
            # Simulate mouse position for this frame
            t = frame / 60.0
            x = int(400 + 200 * np.sin(t * 2 * np.pi))
            y = int(300 + 150 * np.cos(t * 2 * np.pi))
            
            # Handle mouse movement (this is what needs to be fast)
            manager.handle_mouse_move(x, y)
            
            # Simulate getting render state
            state_dict = manager.state.get_state_dict()
            needs_render = manager.state.needs_render
            
            if needs_render:
                manager.state.clear_render_flag()
            
            frame_end = time.perf_counter()
            frame_time = (frame_end - frame_start) * 1000
            frame_times.append(frame_time)
        
        avg_frame_time = sum(frame_times) / len(frame_times)
        max_frame_time = max(frame_times)
        frames_over_budget = sum(1 for t in frame_times if t > target_frame_time)
        
        print(f"  Average frame time: {avg_frame_time:.2f}ms (target: {target_frame_time:.2f}ms)")
        print(f"  Maximum frame time: {max_frame_time:.2f}ms")
        print(f"  Frames over budget: {frames_over_budget}/60 ({frames_over_budget/60*100:.1f}%)")
        
        # Most frames should be within budget
        self.assertLess(frames_over_budget, 6, f"Too many frames over budget: {frames_over_budget}/60")
        self.assertLess(avg_frame_time, target_frame_time * 0.8, 
                       f"Average frame time too high: {avg_frame_time:.2f}ms")
    
    def test_click_response_time(self):
        """Test click response time requirements."""
        print("\n=== Click Response Time Test ===")
        
        shapes = []
        for i in range(50):
            shapes.append({
                'type': 'circle',
                'center': (np.random.randint(50, 750), np.random.randint(50, 550)),
                'radius_px': 25.0,
                'diameter_mm': 20.0,
                'area_px': np.pi * 25 * 25,
                'hit_contour': create_hit_testing_contour('circle', 
                    center=(np.random.randint(50, 750), np.random.randint(50, 550)), radius_px=25.0),
                'inner': False
            })
        
        warped_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        manager = create_interaction_manager(shapes, warped_image)
        
        # Test click response times
        click_times = []
        target_click_time = 5.0  # 5ms maximum for good responsiveness
        
        for i in range(20):
            x = np.random.randint(50, 750)
            y = np.random.randint(50, 550)
            
            start_time = time.perf_counter()
            manager.handle_mouse_click(x, y)
            end_time = time.perf_counter()
            
            click_time = (end_time - start_time) * 1000
            click_times.append(click_time)
        
        avg_click_time = sum(click_times) / len(click_times)
        max_click_time = max(click_times)
        
        print(f"  Average click response time: {avg_click_time:.2f}ms")
        print(f"  Maximum click response time: {max_click_time:.2f}ms")
        
        self.assertLess(avg_click_time, target_click_time,
                       f"Average click response too slow: {avg_click_time:.2f}ms")
        self.assertLess(max_click_time, target_click_time * 2,
                       f"Maximum click response too slow: {max_click_time:.2f}ms")


if __name__ == '__main__':
    # Run with higher verbosity to see performance results
    unittest.main(verbosity=2)