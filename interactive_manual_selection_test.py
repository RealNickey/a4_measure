"""
Interactive test script for manual selection accuracy validation.

This script provides an interactive interface for testing and validating
the accuracy of manual shape selection across various scenarios.

Requirements tested: 1.4, 1.5, 4.4, 4.5
"""

import cv2
import numpy as np
import time
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

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


class InteractiveManualSelectionTester:
    """Interactive tester for manual selection accuracy validation."""
    
    def __init__(self):
        """Initialize the interactive tester."""
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Manual selection components not available")
            
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        self.manual_engine = ManualSelectionEngine(self.analyzer)
        
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        
        # Test results storage
        self.test_results = []
        self.current_test_session = {
            "start_time": datetime.now().isoformat(),
            "test_cases": [],
            "summary": {}
        }
        
        # Create test scenarios
        self.test_scenarios = self._create_test_scenarios()
        self.current_scenario_index = 0
        
        # UI state
        self.current_image = None
        self.current_scenario = None
        self.selection_start = None
        self.selection_current = None
        self.is_selecting = False
        self.current_mode = SelectionMode.MANUAL_CIRCLE
        
        print("Interactive Manual Selection Tester initialized")
        print("Press 'h' for help, 'q' to quit")
    
    def _create_test_scenarios(self) -> List[Dict]:
        """Create comprehensive test scenarios."""
        scenarios = []
        
        # Scenario 1: Simple shapes on clean background
        simple_img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.circle(simple_img, (200, 200), 60, (0, 0, 0), -1)
        cv2.rectangle(simple_img, (400, 150), (550, 250), (0, 0, 0), -1)
        cv2.circle(simple_img, (200, 400), 40, (100, 100, 100), -1)
        cv2.rectangle(simple_img, (400, 350), (500, 450), (50, 50, 50), -1)
        
        scenarios.append({
            "name": "Simple Shapes",
            "description": "Basic circles and rectangles on clean background",
            "image": simple_img,
            "expected_shapes": [
                {"type": "circle", "center": (200, 200), "radius": 60},
                {"type": "rectangle", "center": (475, 200), "width": 150, "height": 100},
                {"type": "circle", "center": (200, 400), "radius": 40},
                {"type": "rectangle", "center": (450, 400), "width": 100, "height": 100},
            ],
            "difficulty": "Easy"
        })
        
        # Scenario 2: Nested shapes
        nested_img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.rectangle(nested_img, (100, 100), (500, 500), (0, 0, 0), -1)
        cv2.rectangle(nested_img, (150, 150), (450, 450), (255, 255, 255), -1)
        cv2.circle(nested_img, (300, 300), 80, (0, 0, 0), -1)
        cv2.circle(nested_img, (300, 300), 40, (255, 255, 255), -1)
        cv2.rectangle(nested_img, (550, 200), (750, 400), (100, 100, 100), -1)
        cv2.circle(nested_img, (650, 300), 50, (255, 255, 255), -1)
        
        scenarios.append({
            "name": "Nested Shapes",
            "description": "Shapes within shapes - test precision",
            "image": nested_img,
            "expected_shapes": [
                {"type": "rectangle", "center": (300, 300), "width": 400, "height": 400},
                {"type": "rectangle", "center": (300, 300), "width": 300, "height": 300},
                {"type": "circle", "center": (300, 300), "radius": 80},
                {"type": "circle", "center": (300, 300), "radius": 40},
                {"type": "rectangle", "center": (650, 300), "width": 200, "height": 200},
                {"type": "circle", "center": (650, 300), "radius": 50},
            ],
            "difficulty": "Hard"
        })
        
        # Scenario 3: Overlapping shapes
        overlap_img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.circle(overlap_img, (200, 200), 70, (0, 0, 0), -1)
        cv2.circle(overlap_img, (280, 200), 70, (100, 100, 100), -1)
        cv2.rectangle(overlap_img, (150, 300), (330, 450), (50, 50, 50), -1)
        cv2.circle(overlap_img, (240, 375), 60, (200, 200, 200), -1)
        cv2.rectangle(overlap_img, (400, 150), (600, 250), (0, 0, 0), -1)
        cv2.rectangle(overlap_img, (500, 200), (700, 300), (150, 150, 150), -1)
        
        scenarios.append({
            "name": "Overlapping Shapes",
            "description": "Overlapping circles and rectangles",
            "image": overlap_img,
            "expected_shapes": [
                {"type": "circle", "center": (200, 200), "radius": 70},
                {"type": "circle", "center": (280, 200), "radius": 70},
                {"type": "rectangle", "center": (240, 375), "width": 180, "height": 150},
                {"type": "circle", "center": (240, 375), "radius": 60},
                {"type": "rectangle", "center": (500, 200), "width": 200, "height": 100},
                {"type": "rectangle", "center": (600, 250), "width": 200, "height": 100},
            ],
            "difficulty": "Medium"
        })
        
        # Scenario 4: Complex background with noise
        complex_img = np.random.randint(150, 200, (600, 800, 3), dtype=np.uint8)
        cv2.circle(complex_img, (200, 200), 50, (0, 0, 0), -1)
        cv2.rectangle(complex_img, (400, 150), (550, 250), (255, 255, 255), -1)
        cv2.circle(complex_img, (200, 400), 45, (255, 255, 255), -1)
        cv2.rectangle(complex_img, (500, 350), (650, 450), (0, 0, 0), -1)
        
        # Add noise
        for _ in range(200):
            x, y = np.random.randint(0, 800), np.random.randint(0, 600)
            radius = np.random.randint(1, 8)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(complex_img, (x, y), radius, color, -1)
        
        scenarios.append({
            "name": "Complex Background",
            "description": "Shapes on noisy background - test robustness",
            "image": complex_img,
            "expected_shapes": [
                {"type": "circle", "center": (200, 200), "radius": 50},
                {"type": "rectangle", "center": (475, 200), "width": 150, "height": 100},
                {"type": "circle", "center": (200, 400), "radius": 45},
                {"type": "rectangle", "center": (575, 400), "width": 150, "height": 100},
            ],
            "difficulty": "Hard"
        })
        
        # Scenario 5: Small shapes requiring precision
        small_img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        positions = [(150, 150), (300, 150), (450, 150), (600, 150),
                    (150, 300), (300, 300), (450, 300), (600, 300)]
        
        for i, (x, y) in enumerate(positions):
            if i % 2 == 0:
                radius = 15 + i * 2
                cv2.circle(small_img, (x, y), radius, (0, 0, 0), -1)
            else:
                size = 20 + i * 3
                cv2.rectangle(small_img, (x - size//2, y - size//2), 
                            (x + size//2, y + size//2), (0, 0, 0), -1)
        
        scenarios.append({
            "name": "Small Shapes",
            "description": "Small shapes requiring precision",
            "image": small_img,
            "expected_shapes": [
                {"type": "circle", "center": (150, 150), "radius": 15},
                {"type": "rectangle", "center": (300, 150), "width": 23, "height": 23},
                {"type": "circle", "center": (450, 150), "radius": 19},
                {"type": "rectangle", "center": (600, 150), "width": 29, "height": 29},
                {"type": "circle", "center": (150, 300), "radius": 23},
                {"type": "rectangle", "center": (300, 300), "width": 35, "height": 35},
                {"type": "circle", "center": (450, 300), "radius": 27},
                {"type": "rectangle", "center": (600, 300), "width": 41, "height": 41},
            ],
            "difficulty": "Hard"
        })
        
        return scenarios
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for manual selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)
            self.selection_current = (x, y)
            self.is_selecting = True
            
        elif event == cv2.EVENT_MOUSEMOVE and self.is_selecting:
            self.selection_current = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP and self.is_selecting:
            self.selection_current = (x, y)
            self.is_selecting = False
            self._process_selection()
    
    def _process_selection(self):
        """Process the completed selection."""
        if not self.selection_start or not self.selection_current:
            return
            
        # Calculate selection rectangle
        x1, y1 = self.selection_start
        x2, y2 = self.selection_current
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        if w < 10 or h < 10:
            print("Selection too small, try again")
            return
        
        selection_rect = (x, y, w, h)
        
        # Perform shape snapping
        start_time = time.time()
        result = self.snap_engine.snap_to_shape(
            self.current_image, selection_rect, self.current_mode
        )
        processing_time = time.time() - start_time
        
        # Record test result
        test_case = {
            "timestamp": datetime.now().isoformat(),
            "scenario": self.current_scenario["name"],
            "mode": self.current_mode.value,
            "selection_rect": selection_rect,
            "processing_time": processing_time,
            "result": result,
            "success": result is not None
        }
        
        if result:
            # Get measurement
            measurement = classify_and_measure_manual_selection(
                self.current_image, selection_rect, result,
                self.mm_per_px_x, self.mm_per_px_y
            )
            test_case["measurement"] = measurement
            
            # Display result
            print(f"\\nDetected {result['type']}:")
            print(f"  Center: {result['center']}")
            print(f"  Confidence: {result['confidence_score']:.3f}")
            print(f"  Processing time: {processing_time:.3f}s")
            
            if measurement:
                if result['type'] == 'circle':
                    print(f"  Diameter: {measurement['diameter_mm']:.2f}mm")
                else:
                    print(f"  Dimensions: {measurement['width_mm']:.2f}mm x {measurement['height_mm']:.2f}mm")
        else:
            print(f"\\nNo {self.current_mode.value} detected in selection")
            print(f"Processing time: {processing_time:.3f}s")
        
        self.current_test_session["test_cases"].append(test_case)
        
        # Clear selection
        self.selection_start = None
        self.selection_current = None
    
    def _draw_overlay(self, image):
        """Draw UI overlay on the image."""
        overlay = image.copy()
        
        # Draw current selection
        if self.is_selecting and self.selection_start and self.selection_current:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_current
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw mode indicator
        mode_text = f"Mode: {self.current_mode.value}"
        cv2.putText(overlay, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw scenario info
        scenario_text = f"Scenario: {self.current_scenario['name']} ({self.current_scenario['difficulty']})"
        cv2.putText(overlay, scenario_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw instructions
        instructions = [
            "Click and drag to select area",
            "M: Toggle mode (Circle/Rectangle)",
            "N: Next scenario",
            "P: Previous scenario",
            "R: Reset current scenario",
            "S: Save results",
            "H: Help",
            "Q: Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = overlay.shape[0] - 20 - (len(instructions) - i - 1) * 25
            cv2.putText(overlay, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw expected shapes (optional - can be toggled)
        if hasattr(self, 'show_expected') and self.show_expected:
            for shape in self.current_scenario['expected_shapes']:
                center = shape['center']
                if shape['type'] == 'circle':
                    radius = shape['radius']
                    cv2.circle(overlay, center, radius, (255, 0, 0), 2)
                else:
                    w, h = shape['width'], shape['height']
                    x1, y1 = center[0] - w//2, center[1] - h//2
                    x2, y2 = center[0] + w//2, center[1] + h//2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return overlay
    
    def run_scenario(self, scenario_index):
        """Run a specific test scenario."""
        if scenario_index < 0 or scenario_index >= len(self.test_scenarios):
            print(f"Invalid scenario index: {scenario_index}")
            return
            
        self.current_scenario_index = scenario_index
        self.current_scenario = self.test_scenarios[scenario_index]
        self.current_image = self.current_scenario["image"].copy()
        
        print(f"\\n{'='*60}")
        print(f"SCENARIO: {self.current_scenario['name']}")
        print(f"DIFFICULTY: {self.current_scenario['difficulty']}")
        print(f"DESCRIPTION: {self.current_scenario['description']}")
        print(f"EXPECTED SHAPES: {len(self.current_scenario['expected_shapes'])}")
        print(f"{'='*60}")
        
        window_name = f"Manual Selection Test - {self.current_scenario['name']}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            display_image = self._draw_overlay(self.current_image)
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Toggle mode
                if self.current_mode == SelectionMode.MANUAL_CIRCLE:
                    self.current_mode = SelectionMode.MANUAL_RECTANGLE
                else:
                    self.current_mode = SelectionMode.MANUAL_CIRCLE
                print(f"Switched to {self.current_mode.value} mode")
            elif key == ord('n'):
                # Next scenario
                if self.current_scenario_index < len(self.test_scenarios) - 1:
                    cv2.destroyWindow(window_name)
                    self.run_scenario(self.current_scenario_index + 1)
                    break
                else:
                    print("Already at last scenario")
            elif key == ord('p'):
                # Previous scenario
                if self.current_scenario_index > 0:
                    cv2.destroyWindow(window_name)
                    self.run_scenario(self.current_scenario_index - 1)
                    break
                else:
                    print("Already at first scenario")
            elif key == ord('r'):
                # Reset scenario
                self.current_image = self.current_scenario["image"].copy()
                print("Scenario reset")
            elif key == ord('s'):
                # Save results
                self._save_results()
            elif key == ord('e'):
                # Toggle expected shapes display
                self.show_expected = not getattr(self, 'show_expected', False)
                print(f"Expected shapes display: {'ON' if self.show_expected else 'OFF'}")
            elif key == ord('h'):
                self._show_help()
        
        cv2.destroyWindow(window_name)
    
    def _show_help(self):
        """Display help information."""
        help_text = """
        INTERACTIVE MANUAL SELECTION TESTER HELP
        ========================================
        
        MOUSE CONTROLS:
        - Click and drag to create selection rectangle
        - Release to process selection
        
        KEYBOARD CONTROLS:
        - M: Toggle between Circle and Rectangle modes
        - N: Next scenario
        - P: Previous scenario
        - R: Reset current scenario
        - S: Save test results to file
        - E: Toggle expected shapes display
        - H: Show this help
        - Q: Quit current scenario
        
        TESTING WORKFLOW:
        1. Select a shape by clicking and dragging around it
        2. The system will attempt to detect and snap to the shape
        3. Results are displayed in the console
        4. Try different selection sizes and positions
        5. Test both circle and rectangle modes
        6. Move between scenarios to test different cases
        
        EVALUATION CRITERIA:
        - Detection accuracy (correct shape type)
        - Measurement precision (size and position)
        - Processing speed (< 1 second preferred)
        - Robustness (handles edge cases gracefully)
        """
        print(help_text)
    
    def _save_results(self):
        """Save test results to file."""
        self.current_test_session["end_time"] = datetime.now().isoformat()
        self.current_test_session["summary"] = self._generate_summary()
        
        filename = f"manual_selection_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_test_session, f, indent=2, default=str)
            print(f"\\nTest results saved to: {filename}")
            print(f"Total test cases: {len(self.current_test_session['test_cases'])}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _generate_summary(self):
        """Generate test session summary."""
        test_cases = self.current_test_session["test_cases"]
        
        if not test_cases:
            return {"message": "No test cases recorded"}
        
        total_cases = len(test_cases)
        successful_cases = sum(1 for case in test_cases if case["success"])
        
        # Group by scenario and mode
        by_scenario = {}
        by_mode = {}
        processing_times = []
        
        for case in test_cases:
            scenario = case["scenario"]
            mode = case["mode"]
            
            if scenario not in by_scenario:
                by_scenario[scenario] = {"total": 0, "success": 0}
            by_scenario[scenario]["total"] += 1
            if case["success"]:
                by_scenario[scenario]["success"] += 1
            
            if mode not in by_mode:
                by_mode[mode] = {"total": 0, "success": 0}
            by_mode[mode]["total"] += 1
            if case["success"]:
                by_mode[mode]["success"] += 1
            
            processing_times.append(case["processing_time"])
        
        summary = {
            "total_test_cases": total_cases,
            "successful_detections": successful_cases,
            "success_rate": successful_cases / total_cases if total_cases > 0 else 0,
            "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "max_processing_time": max(processing_times) if processing_times else 0,
            "min_processing_time": min(processing_times) if processing_times else 0,
            "by_scenario": by_scenario,
            "by_mode": by_mode
        }
        
        return summary
    
    def run_all_scenarios(self):
        """Run all test scenarios in sequence."""
        print("Starting comprehensive manual selection testing...")
        print(f"Total scenarios: {len(self.test_scenarios)}")
        
        for i in range(len(self.test_scenarios)):
            self.run_scenario(i)
            
            # Ask if user wants to continue
            response = input("\\nContinue to next scenario? (y/n/q): ").lower()
            if response == 'q':
                break
            elif response == 'n':
                continue
        
        # Generate final report
        print("\\n" + "="*60)
        print("TESTING SESSION COMPLETE")
        print("="*60)
        
        summary = self._generate_summary()
        print(f"Total test cases: {summary['total_test_cases']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Average processing time: {summary['average_processing_time']:.3f}s")
        
        # Save results
        self._save_results()


def main():
    """Main function to run interactive testing."""
    if not COMPONENTS_AVAILABLE:
        print("Error: Manual selection components not available")
        print("Please ensure all required modules are installed and working")
        return
    
    try:
        tester = InteractiveManualSelectionTester()
        
        print("\\nInteractive Manual Selection Tester")
        print("===================================")
        print("Choose an option:")
        print("1. Run all scenarios sequentially")
        print("2. Select specific scenario")
        print("3. Quick test (scenario 1)")
        print("4. Exit")
        
        while True:
            choice = input("\\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                tester.run_all_scenarios()
                break
            elif choice == '2':
                print("\\nAvailable scenarios:")
                for i, scenario in enumerate(tester.test_scenarios):
                    print(f"{i+1}. {scenario['name']} ({scenario['difficulty']})")
                
                try:
                    scenario_num = int(input("Enter scenario number: ")) - 1
                    tester.run_scenario(scenario_num)
                except (ValueError, IndexError):
                    print("Invalid scenario number")
                    continue
                break
            elif choice == '3':
                tester.run_scenario(0)
                break
            elif choice == '4':
                break
            else:
                print("Invalid choice, please try again")
    
    except KeyboardInterrupt:
        print("\\nTesting interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()