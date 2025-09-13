"""
Performance Profiler for Interactive Inspect Mode

This module provides performance profiling and optimization tools for the interactive
inspect mode, focusing on rendering performance during mouse movement and real-time
interaction responsiveness.
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import deque
import statistics


class PerformanceProfiler:
    """
    Profiler for measuring and analyzing rendering performance during mouse interaction.
    
    Tracks:
    - Frame rendering times
    - Mouse event processing times
    - State update times
    - Overall interaction responsiveness
    """
    
    def __init__(self, max_samples: int = 100):
        """
        Initialize the performance profiler.
        
        Args:
            max_samples: Maximum number of samples to keep for analysis
        """
        self.max_samples = max_samples
        self.render_times = deque(maxlen=max_samples)
        self.mouse_event_times = deque(maxlen=max_samples)
        self.state_update_times = deque(maxlen=max_samples)
        self.total_frame_times = deque(maxlen=max_samples)
        
        # Performance counters
        self.frame_count = 0
        self.render_count = 0
        self.mouse_event_count = 0
        
        # Timing contexts
        self._start_times = {}
        
        # Performance thresholds (in milliseconds)
        self.target_frame_time = 16.67  # 60 FPS
        self.warning_frame_time = 33.33  # 30 FPS
        self.critical_frame_time = 50.0  # 20 FPS
    
    def start_timing(self, operation: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation being timed
        """
        self._start_times[operation] = time.perf_counter()
    
    def end_timing(self, operation: str) -> float:
        """
        End timing an operation and record the duration.
        
        Args:
            operation: Name of the operation being timed
            
        Returns:
            Duration in milliseconds
        """
        if operation not in self._start_times:
            return 0.0
        
        duration_ms = (time.perf_counter() - self._start_times[operation]) * 1000
        del self._start_times[operation]
        
        # Record timing based on operation type
        if operation == "render":
            self.render_times.append(duration_ms)
            self.render_count += 1
        elif operation == "mouse_event":
            self.mouse_event_times.append(duration_ms)
            self.mouse_event_count += 1
        elif operation == "state_update":
            self.state_update_times.append(duration_ms)
        elif operation == "total_frame":
            self.total_frame_times.append(duration_ms)
            self.frame_count += 1
        
        return duration_ms
    
    def get_render_stats(self) -> Dict[str, float]:
        """
        Get rendering performance statistics.
        
        Returns:
            Dictionary with rendering performance metrics
        """
        if not self.render_times:
            return {"count": 0}
        
        times = list(self.render_times)
        return {
            "count": len(times),
            "avg_ms": statistics.mean(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "median_ms": statistics.median(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "avg_fps": 1000.0 / statistics.mean(times) if statistics.mean(times) > 0 else 0.0
        }
    
    def get_mouse_event_stats(self) -> Dict[str, float]:
        """
        Get mouse event processing performance statistics.
        
        Returns:
            Dictionary with mouse event performance metrics
        """
        if not self.mouse_event_times:
            return {"count": 0}
        
        times = list(self.mouse_event_times)
        return {
            "count": len(times),
            "avg_ms": statistics.mean(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "median_ms": statistics.median(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall performance statistics.
        
        Returns:
            Dictionary with comprehensive performance metrics
        """
        render_stats = self.get_render_stats()
        mouse_stats = self.get_mouse_event_stats()
        
        overall_stats = {
            "total_frames": self.frame_count,
            "total_renders": self.render_count,
            "total_mouse_events": self.mouse_event_count,
            "render_stats": render_stats,
            "mouse_stats": mouse_stats
        }
        
        # Add frame time statistics if available
        if self.total_frame_times:
            times = list(self.total_frame_times)
            overall_stats["frame_stats"] = {
                "avg_ms": statistics.mean(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "avg_fps": 1000.0 / statistics.mean(times) if statistics.mean(times) > 0 else 0.0
            }
        
        return overall_stats
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze performance and identify potential issues.
        
        Returns:
            Dictionary with performance analysis and recommendations
        """
        stats = self.get_overall_stats()
        analysis = {
            "status": "good",
            "issues": [],
            "recommendations": []
        }
        
        # Analyze rendering performance
        if stats["render_stats"].get("count", 0) > 0:
            avg_render_time = stats["render_stats"]["avg_ms"]
            max_render_time = stats["render_stats"]["max_ms"]
            
            if avg_render_time > self.critical_frame_time:
                analysis["status"] = "critical"
                analysis["issues"].append(f"Average render time ({avg_render_time:.1f}ms) exceeds critical threshold")
                analysis["recommendations"].append("Consider reducing rendering complexity or optimizing algorithms")
            elif avg_render_time > self.warning_frame_time:
                analysis["status"] = "warning"
                analysis["issues"].append(f"Average render time ({avg_render_time:.1f}ms) exceeds warning threshold")
                analysis["recommendations"].append("Monitor rendering performance and consider optimizations")
            
            if max_render_time > self.critical_frame_time * 2:
                analysis["issues"].append(f"Maximum render time ({max_render_time:.1f}ms) indicates frame drops")
                analysis["recommendations"].append("Investigate causes of render time spikes")
        
        # Analyze mouse event performance
        if stats["mouse_stats"].get("count", 0) > 0:
            avg_mouse_time = stats["mouse_stats"]["avg_ms"]
            
            if avg_mouse_time > 5.0:  # Mouse events should be very fast
                analysis["issues"].append(f"Mouse event processing ({avg_mouse_time:.1f}ms) is slow")
                analysis["recommendations"].append("Optimize mouse event handling and hit testing")
        
        return analysis
    
    def print_performance_report(self) -> None:
        """Print a comprehensive performance report to console."""
        stats = self.get_overall_stats()
        analysis = self.analyze_performance()
        
        print("\n=== Performance Report ===")
        print(f"Status: {analysis['status'].upper()}")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Total Renders: {stats['total_renders']}")
        print(f"Total Mouse Events: {stats['total_mouse_events']}")
        
        # Rendering statistics
        if stats["render_stats"].get("count", 0) > 0:
            rs = stats["render_stats"]
            print(f"\nRendering Performance:")
            print(f"  Average: {rs['avg_ms']:.2f}ms ({rs['avg_fps']:.1f} FPS)")
            print(f"  Range: {rs['min_ms']:.2f}ms - {rs['max_ms']:.2f}ms")
            print(f"  Median: {rs['median_ms']:.2f}ms")
            print(f"  Std Dev: {rs['std_dev']:.2f}ms")
        
        # Mouse event statistics
        if stats["mouse_stats"].get("count", 0) > 0:
            ms = stats["mouse_stats"]
            print(f"\nMouse Event Performance:")
            print(f"  Average: {ms['avg_ms']:.2f}ms")
            print(f"  Range: {ms['min_ms']:.2f}ms - {ms['max_ms']:.2f}ms")
            print(f"  Median: {ms['median_ms']:.2f}ms")
        
        # Frame statistics
        if "frame_stats" in stats:
            fs = stats["frame_stats"]
            print(f"\nOverall Frame Performance:")
            print(f"  Average: {fs['avg_ms']:.2f}ms ({fs['avg_fps']:.1f} FPS)")
            print(f"  Range: {fs['min_ms']:.2f}ms - {fs['max_ms']:.2f}ms")
        
        # Issues and recommendations
        if analysis["issues"]:
            print(f"\nIssues Identified:")
            for issue in analysis["issues"]:
                print(f"  - {issue}")
        
        if analysis["recommendations"]:
            print(f"\nRecommendations:")
            for rec in analysis["recommendations"]:
                print(f"  - {rec}")
        
        print("=" * 30)
    
    def reset(self) -> None:
        """Reset all performance counters and statistics."""
        self.render_times.clear()
        self.mouse_event_times.clear()
        self.state_update_times.clear()
        self.total_frame_times.clear()
        
        self.frame_count = 0
        self.render_count = 0
        self.mouse_event_count = 0
        
        self._start_times.clear()


class OptimizedRenderer:
    """
    Optimized renderer that implements efficient re-rendering strategies.
    
    Features:
    - State change detection to avoid unnecessary renders
    - Cached rendering for static elements
    - Efficient update regions for partial rendering
    """
    
    def __init__(self, base_renderer):
        """
        Initialize the optimized renderer.
        
        Args:
            base_renderer: The base SelectiveRenderer instance to optimize
        """
        self.base_renderer = base_renderer
        self.cached_base_image = None
        self.cached_instruction_text = None
        self.last_state_hash = None
        self.profiler = PerformanceProfiler()
        
        # Optimization flags
        self.enable_caching = True
        self.enable_partial_updates = True
        self.enable_state_diffing = True
    
    def render_optimized(self, warped_image: np.ndarray, state: Dict[str, Any],
                        shapes: List[Dict[str, Any]], force_full_render: bool = False) -> np.ndarray:
        """
        Render with optimizations for performance.
        
        Args:
            warped_image: The warped A4 background image
            state: Current interaction state
            shapes: List of all detected shapes
            force_full_render: Force a complete re-render ignoring optimizations
            
        Returns:
            Optimized rendered image
        """
        self.profiler.start_timing("render")
        
        try:
            # Check if we can use cached rendering
            if not force_full_render and self.enable_state_diffing:
                state_hash = self._compute_state_hash(state, shapes)
                if state_hash == self.last_state_hash and self.cached_base_image is not None:
                    # No changes, return cached result (avoid copy for better performance)
                    return self.cached_base_image
                self.last_state_hash = state_hash
            
            # Start with clean base image
            result = self.base_renderer.render_base(warped_image)
            
            # Add interactive elements efficiently
            result = self._render_interactive_elements_optimized(result, state, shapes)
            
            # Cache the result if caching is enabled
            if self.enable_caching:
                self.cached_base_image = result.copy()
            
            return result
            
        finally:
            self.profiler.end_timing("render")
    
    def _compute_state_hash(self, state: Dict[str, Any], shapes: List[Dict[str, Any]]) -> int:
        """
        Compute a hash of the current state for change detection.
        
        Args:
            state: Current interaction state
            shapes: List of shapes
            
        Returns:
            Hash value representing the current state
        """
        # Create a tuple of relevant state values that affect rendering
        state_tuple = (
            state.get("hovered"),
            state.get("selected"),
            len(shapes),
            # Include mouse position for more granular change detection
            state.get("mouse_pos", (0, 0))[0] // 10,  # Quantize to reduce sensitivity
            state.get("mouse_pos", (0, 0))[1] // 10
        )
        return hash(state_tuple)
    
    def _render_interactive_elements(self, base_image: np.ndarray, state: Dict[str, Any],
                                   shapes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Render only the interactive elements (hover, selection, text).
        
        Args:
            base_image: Base image to draw on
            state: Current interaction state
            shapes: List of shapes
            
        Returns:
            Image with interactive elements added
        """
        result = base_image.copy()
        
        # Add hover preview (if not selected)
        hovered_idx = state.get("hovered")
        selected_idx = state.get("selected")
        
        if (hovered_idx is not None and 
            hovered_idx != selected_idx and
            hovered_idx < len(shapes)):
            result = self.base_renderer.render_preview(result, shapes[hovered_idx])
        
        # Add selected shape with dimensions
        if (selected_idx is not None and selected_idx < len(shapes)):
            result = self.base_renderer.render_selection(result, shapes[selected_idx])
        
        # Add instruction text
        result = self.base_renderer.render_instruction_text(result, state, shapes)
        
        return result
    
    def _render_interactive_elements_optimized(self, base_image: np.ndarray, state: Dict[str, Any],
                                             shapes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Optimized rendering of interactive elements with minimal copying.
        
        Args:
            base_image: Base image to draw on (modified in place)
            state: Current interaction state
            shapes: List of shapes
            
        Returns:
            Image with interactive elements added
        """
        # Modify base image in place for better performance
        hovered_idx = state.get("hovered")
        selected_idx = state.get("selected")
        
        # Add hover preview (if not selected)
        if (hovered_idx is not None and 
            hovered_idx != selected_idx and
            hovered_idx < len(shapes)):
            from config import PREVIEW_COLOR
            self.base_renderer._draw_shape_outline(
                base_image, shapes[hovered_idx], PREVIEW_COLOR, 2
            )
        
        # Add selected shape with dimensions
        if (selected_idx is not None and selected_idx < len(shapes)):
            from config import SELECTION_COLOR
            self.base_renderer._draw_shape_with_dimensions(
                base_image, shapes[selected_idx], SELECTION_COLOR
            )
        
        # Add instruction text efficiently
        text = self._generate_instruction_text_cached(state, shapes)
        from utils import draw_text
        draw_text(base_image, text, (20, 40), (255, 255, 255), 0.7, 2)
        
        return base_image
    
    def _generate_instruction_text_cached(self, state: Dict[str, Any], 
                                        shapes: List[Dict[str, Any]]) -> str:
        """
        Generate instruction text with caching for performance.
        
        Args:
            state: Current interaction state
            shapes: List of shapes
            
        Returns:
            Cached or generated instruction text
        """
        selected_idx = state.get("selected")
        
        # Use cached text if available and state hasn't changed
        cache_key = (selected_idx, len(shapes))
        if hasattr(self, '_text_cache') and self._text_cache.get('key') == cache_key:
            return self._text_cache['text']
        
        # Generate new text
        if selected_idx is not None and selected_idx < len(shapes):
            shape = shapes[selected_idx]
            if shape["type"] == "circle":
                text = f"Selected: Circle (D={shape['diameter_mm']:.0f}mm)"
            else:
                text = f"Selected: Rectangle ({shape['width_mm']:.0f}x{shape['height_mm']:.0f}mm)"
        else:
            text = "Hover to preview, click to inspect"
        
        # Cache the result
        self._text_cache = {'key': cache_key, 'text': text}
        return text
    
    def invalidate_cache(self) -> None:
        """Invalidate cached rendering data to force fresh render."""
        self.cached_base_image = None
        self.cached_instruction_text = None
        self.last_state_hash = None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the profiler."""
        return self.profiler.get_overall_stats()
    
    def print_performance_report(self) -> None:
        """Print performance report from the profiler."""
        self.profiler.print_performance_report()


class FrameRateOptimizer:
    """
    Frame rate optimizer for smooth real-time mouse tracking.
    
    Implements adaptive frame rate control and efficient update scheduling
    to maintain smooth interaction while minimizing CPU usage.
    """
    
    def __init__(self, target_fps: float = 60.0, min_fps: float = 30.0):
        """
        Initialize the frame rate optimizer.
        
        Args:
            target_fps: Target frame rate for smooth interaction
            min_fps: Minimum acceptable frame rate
        """
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.target_frame_time = 1000.0 / target_fps  # milliseconds
        self.min_frame_time = 1000.0 / min_fps
        
        self.last_render_time = 0
        self.frame_times = deque(maxlen=30)  # Track recent frame times
        self.adaptive_threshold = self.target_frame_time
        
        # Performance tracking
        self.frames_rendered = 0
        self.frames_skipped = 0
    
    def should_render_frame(self, force_render: bool = False) -> bool:
        """
        Determine if a frame should be rendered based on timing constraints.
        
        Args:
            force_render: Force rendering regardless of timing
            
        Returns:
            True if frame should be rendered
        """
        if force_render:
            return True
        
        current_time = time.perf_counter() * 1000  # milliseconds
        time_since_last = current_time - self.last_render_time
        
        # Always render if enough time has passed
        if time_since_last >= self.adaptive_threshold:
            return True
        
        # Skip frame if too soon
        self.frames_skipped += 1
        return False
    
    def frame_rendered(self, render_time_ms: float) -> None:
        """
        Record that a frame was rendered and update timing statistics.
        
        Args:
            render_time_ms: Time taken to render the frame in milliseconds
        """
        current_time = time.perf_counter() * 1000
        self.last_render_time = current_time
        self.frames_rendered += 1
        
        # Track frame times for adaptive adjustment
        self.frame_times.append(render_time_ms)
        
        # Adjust adaptive threshold based on recent performance
        if len(self.frame_times) >= 10:
            avg_render_time = sum(self.frame_times) / len(self.frame_times)
            
            # If rendering is fast, allow higher frame rate
            if avg_render_time < self.target_frame_time * 0.5:
                self.adaptive_threshold = max(self.target_frame_time * 0.8, 8.33)  # Up to 120 FPS
            # If rendering is slow, reduce frame rate
            elif avg_render_time > self.target_frame_time:
                self.adaptive_threshold = min(self.min_frame_time, avg_render_time * 1.5)
            else:
                self.adaptive_threshold = self.target_frame_time
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get frame rate optimization performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        total_frames = self.frames_rendered + self.frames_skipped
        skip_ratio = self.frames_skipped / total_frames if total_frames > 0 else 0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        effective_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            "frames_rendered": self.frames_rendered,
            "frames_skipped": self.frames_skipped,
            "skip_ratio": skip_ratio,
            "adaptive_threshold_ms": self.adaptive_threshold,
            "avg_frame_time_ms": avg_frame_time,
            "effective_fps": effective_fps
        }


def create_performance_test_suite(shapes: List[Dict[str, Any]], warped_image: np.ndarray) -> Dict[str, Callable]:
    """
    Create a suite of performance tests for the interactive system.
    
    Args:
        shapes: List of shapes for testing
        warped_image: Test image
        
    Returns:
        Dictionary of test functions
    """
    def test_rendering_performance():
        """Test basic rendering performance."""
        from rendering import SelectiveRenderer
        renderer = SelectiveRenderer()
        profiler = PerformanceProfiler()
        
        # Test multiple render cycles
        for i in range(50):
            profiler.start_timing("render")
            state = {"hovered": i % len(shapes) if shapes else None, "selected": None}
            result = renderer.render_complete_state(warped_image, state, shapes)
            profiler.end_timing("render")
        
        return profiler.get_render_stats()
    
    def test_mouse_event_performance():
        """Test mouse event processing performance."""
        from hit_testing import HitTestingEngine
        hit_engine = HitTestingEngine()
        profiler = PerformanceProfiler()
        
        # Simulate mouse movement across the image
        for x in range(0, warped_image.shape[1], 10):
            for y in range(0, warped_image.shape[0], 10):
                profiler.start_timing("mouse_event")
                hit_engine.find_shape_at_point(shapes, x, y)
                profiler.end_timing("mouse_event")
        
        return profiler.get_mouse_event_stats()
    
    def test_state_update_performance():
        """Test interaction state update performance."""
        from interaction_state import InteractionState
        state = InteractionState(shapes)
        profiler = PerformanceProfiler()
        
        # Test state updates
        for i in range(100):
            profiler.start_timing("state_update")
            state.update_hover(i % len(shapes) if shapes else None)
            state.update_selection(i % len(shapes) if shapes else None)
            profiler.end_timing("state_update")
        
        return profiler.get_overall_stats()
    
    return {
        "rendering": test_rendering_performance,
        "mouse_events": test_mouse_event_performance,
        "state_updates": test_state_update_performance
    }


def run_performance_benchmark(shapes: List[Dict[str, Any]], warped_image: np.ndarray) -> None:
    """
    Run comprehensive performance benchmark for the interactive system.
    
    Args:
        shapes: List of shapes for testing
        warped_image: Test image
    """
    print("\n=== Performance Benchmark ===")
    print(f"Testing with {len(shapes)} shapes on {warped_image.shape[1]}x{warped_image.shape[0]} image")
    
    test_suite = create_performance_test_suite(shapes, warped_image)
    
    for test_name, test_func in test_suite.items():
        print(f"\nRunning {test_name} test...")
        try:
            start_time = time.perf_counter()
            results = test_func()
            end_time = time.perf_counter()
            
            print(f"  Test completed in {(end_time - start_time) * 1000:.2f}ms")
            if isinstance(results, dict) and "avg_ms" in results:
                print(f"  Average operation time: {results['avg_ms']:.2f}ms")
                if "avg_fps" in results:
                    print(f"  Effective FPS: {results['avg_fps']:.1f}")
        except Exception as e:
            print(f"  Test failed: {e}")
    
    print("\n=== Benchmark Complete ===")