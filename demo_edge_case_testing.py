"""
Demo script for edge case testing in manual selection system.

This script demonstrates how the manual selection system handles
various edge cases and challenging scenarios.

Requirements tested: 1.4, 1.5, 4.4, 4.5
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional

# Import manual selection components
try:
    from shape_snapping_engine import ShapeSnappingEngine
    from enhanced_contour_analyzer import EnhancedContourAnalyzer
    from selection_mode import SelectionMode
    from measure import classify_and_measure_manual_selection
    from detection import a4_scale_mm_per_px
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


class EdgeCaseTestDemo:
    """Demo class for edge case testing."""
    
    def __init__(self):
        """Initialize the edge case test demo."""
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Manual selection components not available")
            
        self.analyzer = EnhancedContourAnalyzer()
        self.snap_engine = ShapeSnappingEngine(self.analyzer)
        self.mm_per_px_x, self.mm_per_px_y = a4_scale_mm_per_px()
        
        # Create edge case test images
        self.edge_case_images = self._create_edge_case_images()
        
        print("Edge Case Test Demo initialized")
        print(f"Created {len(self.edge_case_images)} edge case scenarios")
    
    def _create_edge_case_images(self) -> Dict[str, Dict]:
        """Create images for edge case testing."""
        images = {}
        
        # 1. Overlapping circles
        overlap_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(overlap_img, (150, 200), 60, (0, 0, 0), -1)
        cv2.circle(overlap_img, (200, 200), 60, (100, 100, 100), -1)
        cv2.circle(overlap_img, (175, 150), 40, (50, 50, 50), -1)
        
        images["overlapping_circles"] = {
            "image": overlap_img,
            "description": "Multiple overlapping circles",
            "test_selections": [
                ((120, 170, 60, 60), SelectionMode.MANUAL_CIRCLE, "Left circle"),
                ((170, 170, 60, 60), SelectionMode.MANUAL_CIRCLE, "Right circle"),
                ((145, 120, 60, 60), SelectionMode.MANUAL_CIRCLE, "Top circle"),
                ((120, 120, 120, 120), SelectionMode.MANUAL_CIRCLE, "All circles")
            ],
            "difficulty": "Hard"
        }
        
        # 2. Shapes near boundaries
        boundary_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.circle(boundary_img, (50, 50), 45, (0, 0, 0), -1)  # Top-left corner
        cv2.circle(boundary_img, (350, 50), 45, (0, 0, 0), -1)  # Top-right corner
        cv2.rectangle(boundary_img, (5, 200), (95, 290), (0, 0, 0), -1)  # Bottom-left
        cv2.rectangle(boundary_img, (305, 200), (395, 290), (0, 0, 0), -1)  # Bottom-right
        
        images["boundary_shapes"] = {
            "image": boundary_img,
            "description": "Shapes near image boundaries",
            "test_selections": [
                ((5, 5, 90, 90), SelectionMode.MANUAL_CIRCLE, "Top-left circle"),
                ((305, 5, 90, 90), SelectionMode.MANUAL_CIRCLE, "Top-right circle"),
                ((5, 170, 90, 120), SelectionMode.MANUAL_RECTANGLE, "Bottom-left rect"),
                ((305, 170, 90, 120), SelectionMode.MANUAL_RECTANGLE, "Bottom-right rect")
            ],
            "difficulty": "Medium"
        }
        
        # 3. Very small shapes
        small_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        positions = [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200), (300, 200)]
        
        for i, (x, y) in enumerate(positions):
            if i % 2 == 0:
                radius = 8 + i * 2
                cv2.circle(small_img, (x, y), radius, (0, 0, 0), -1)
            else:
                size = 12 + i * 2
                cv2.rectangle(small_img, (x - size//2, y - size//2), 
                            (x + size//2, y + size//2), (0, 0, 0), -1)
        
        images["tiny_shapes"] = {
            "image": small_img,
            "description": "Very small shapes requiring precision",
            "test_selections": [
                ((85, 85, 30, 30), SelectionMode.MANUAL_CIRCLE, "8px circle"),
                ((185, 85, 30, 30), SelectionMode.MANUAL_RECTANGLE, "14px square"),
                ((285, 85, 30, 30), SelectionMode.MANUAL_CIRCLE, "12px circle"),
                ((85, 185, 30, 30), SelectionMode.MANUAL_RECTANGLE, "18px square")
            ],
            "difficulty": "Hard"
        }
        
        # 4. Complex background with noise
        noisy_img = np.random.randint(120, 180, (400, 400, 3), dtype=np.uint8)
        
        # Add main shapes
        cv2.circle(noisy_img, (150, 150), 40, (0, 0, 0), -1)
        cv2.rectangle(noisy_img, (250, 100), (350, 200), (255, 255, 255), -1)
        cv2.circle(noisy_img, (150, 300), 35, (255, 255, 255), -1)
        
        # Add noise
        for _ in range(300):
            x, y = np.random.randint(0, 400, 2)
            radius = np.random.randint(1, 6)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(noisy_img, (x, y), radius, color, -1)
        
        images["noisy_background"] = {
            "image": noisy_img,
            "description": "Shapes on very noisy background",
            "test_selections": [
                ((110, 110, 80, 80), SelectionMode.MANUAL_CIRCLE, "Black circle"),
                ((210, 60, 180, 180), SelectionMode.MANUAL_RECTANGLE, "White rectangle"),
                ((115, 265, 70, 70), SelectionMode.MANUAL_CIRCLE, "White circle")
            ],
            "difficulty": "Hard"
        }
        
        # 5. Partial shapes (cut off by selection)
        partial_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(partial_img, (200, 200), 80, (0, 0, 0), -1)
        cv2.rectangle(partial_img, (100, 300), (300, 500), (0, 0, 0), -1)  # Extends beyond image
        
        images["partial_shapes"] = {
            "image": partial_img,
            "description": "Partially visible shapes",
            "test_selections": [
                ((150, 150, 60, 60), SelectionMode.MANUAL_CIRCLE, "Partial circle"),
                ((180, 180, 80, 80), SelectionMode.MANUAL_CIRCLE, "Most of circle"),
                ((150, 320, 100, 80), SelectionMode.MANUAL_RECTANGLE, "Partial rectangle")
            ],
            "difficulty": "Medium"
        }
        
        # 6. Low contrast shapes
        low_contrast_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
        cv2.circle(low_contrast_img, (150, 150), 50, (180, 180, 180), -1)  # Light gray on light background
        cv2.rectangle(low_contrast_img, (250, 100), (350, 200), (220, 220, 220), -1)  # Very light gray
        cv2.circle(low_contrast_img, (150, 300), 40, (160, 160, 160), -1)  # Medium gray
        
        images["low_contrast"] = {
            "image": low_contrast_img,
            "description": "Low contrast shapes",
            "test_selections": [
                ((100, 100, 100, 100), SelectionMode.MANUAL_CIRCLE, "Light gray circle"),
                ((200, 50, 200, 200), SelectionMode.MANUAL_RECTANGLE, "Very light rectangle"),
                ((110, 260, 80, 80), SelectionMode.MANUAL_CIRCLE, "Medium gray circle")
            ],
            "difficulty": "Hard"
        }
        
        # 7. Irregular shapes that might confuse detection
        irregular_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Create irregular "circle-like" shape
        center = (150, 150)
        angles = np.linspace(0, 2*np.pi, 20)
        points = []
        for angle in angles:
            radius = 40 + 10 * np.sin(5 * angle)  # Wavy circle
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])
        cv2.fillPoly(irregular_img, [np.array(points, dtype=np.int32)], (0, 0, 0))
        
        # Create irregular "rectangle-like" shape
        rect_points = np.array([[250, 100], [350, 110], [340, 200], [260, 190]], dtype=np.int32)
        cv2.fillPoly(irregular_img, [rect_points], (0, 0, 0))
        
        images["irregular_shapes"] = {
            "image": irregular_img,
            "description": "Irregular shapes that approximate circles/rectangles",
            "test_selections": [
                ((100, 100, 100, 100), SelectionMode.MANUAL_CIRCLE, "Wavy circle"),
                ((220, 70, 160, 160), SelectionMode.MANUAL_RECTANGLE, "Irregular quadrilateral")
            ],
            "difficulty": "Medium"
        }
        
        return images
    
    def test_edge_case(self, case_name: str, verbose: bool = True) -> Dict[str, Any]:
        """Test a specific edge case scenario."""
        if case_name not in self.edge_case_images:
            return {"error": f"Edge case '{case_name}' not found"}
        
        case_data = self.edge_case_images[case_name]
        image = case_data["image"]
        test_selections = case_data["test_selections"]
        
        results = {
            "case_name": case_name,
            "description": case_data["description"],
            "difficulty": case_data["difficulty"],
            "test_results": []
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Testing: {case_data['description']}")
            print(f"Difficulty: {case_data['difficulty']}")
            print(f"{'='*50}")
        
        for selection_rect, mode, description in test_selections:
            if verbose:
                print(f"\nTesting: {description}")
                print(f"Selection: {selection_rect}")
                print(f"Mode: {mode.value}")
            
            # Perform shape detection
            start_time = time.time()
            try:
                shape_result = self.snap_engine.snap_to_shape(image, selection_rect, mode)
                detection_time = time.time() - start_time
                
                test_result = {
                    "description": description,
                    "selection_rect": selection_rect,
                    "mode": mode.value,
                    "detection_time": detection_time,
                    "success": shape_result is not None,
                    "shape_result": shape_result
                }
                
                if shape_result:
                    # Get measurement
                    measurement_start = time.time()
                    measurement = classify_and_measure_manual_selection(
                        image, selection_rect, shape_result,
                        self.mm_per_px_x, self.mm_per_px_y
                    )
                    measurement_time = time.time() - measurement_start
                    
                    test_result["measurement"] = measurement
                    test_result["measurement_time"] = measurement_time
                    
                    if verbose:
                        print(f"  ✅ Detected {shape_result['type']}")
                        print(f"  Center: {shape_result['center']}")
                        print(f"  Confidence: {shape_result['confidence_score']:.3f}")
                        print(f"  Detection time: {detection_time:.3f}s")
                        
                        if measurement:
                            if shape_result['type'] == 'circle':
                                print(f"  Diameter: {measurement['diameter_mm']:.2f}mm")
                            else:
                                print(f"  Dimensions: {measurement['width_mm']:.2f}mm x {measurement['height_mm']:.2f}mm")
                else:
                    if verbose:
                        print(f"  ❌ No {mode.value} detected")
                        print(f"  Detection time: {detection_time:.3f}s")
                
                results["test_results"].append(test_result)
                
            except Exception as e:
                test_result = {
                    "description": description,
                    "selection_rect": selection_rect,
                    "mode": mode.value,
                    "detection_time": 0,
                    "success": False,
                    "error": str(e)
                }
                results["test_results"].append(test_result)
                
                if verbose:
                    print(f"  ❌ Error: {e}")
        
        # Calculate summary statistics
        successful_tests = sum(1 for r in results["test_results"] if r["success"])
        total_tests = len(results["test_results"])
        avg_detection_time = sum(r["detection_time"] for r in results["test_results"]) / max(total_tests, 1)
        
        results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / max(total_tests, 1),
            "average_detection_time": avg_detection_time
        }
        
        if verbose:
            print(f"\n{'='*30}")
            print(f"SUMMARY: {case_name}")
            print(f"{'='*30}")
            print(f"Success rate: {successful_tests}/{total_tests} ({results['summary']['success_rate']:.1%})")
            print(f"Average detection time: {avg_detection_time:.3f}s")
        
        return results
    
    def run_all_edge_cases(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all edge case tests."""
        if verbose:
            print("RUNNING ALL EDGE CASE TESTS")
            print("="*60)
        
        all_results = {
            "timestamp": time.time(),
            "total_cases": len(self.edge_case_images),
            "case_results": {},
            "overall_summary": {}
        }
        
        for case_name in self.edge_case_images.keys():
            result = self.test_edge_case(case_name, verbose)
            all_results["case_results"][case_name] = result
        
        # Calculate overall statistics
        total_tests = sum(r["summary"]["total_tests"] for r in all_results["case_results"].values())
        total_successful = sum(r["summary"]["successful_tests"] for r in all_results["case_results"].values())
        avg_success_rate = sum(r["summary"]["success_rate"] for r in all_results["case_results"].values()) / len(all_results["case_results"])
        avg_detection_time = sum(r["summary"]["average_detection_time"] for r in all_results["case_results"].values()) / len(all_results["case_results"])
        
        all_results["overall_summary"] = {
            "total_test_cases": len(self.edge_case_images),
            "total_individual_tests": total_tests,
            "total_successful_tests": total_successful,
            "overall_success_rate": total_successful / max(total_tests, 1),
            "average_case_success_rate": avg_success_rate,
            "average_detection_time": avg_detection_time
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print("OVERALL EDGE CASE TEST SUMMARY")
            print(f"{'='*60}")
            print(f"Test cases run: {len(self.edge_case_images)}")
            print(f"Individual tests: {total_tests}")
            print(f"Successful tests: {total_successful}")
            print(f"Overall success rate: {all_results['overall_summary']['overall_success_rate']:.1%}")
            print(f"Average detection time: {avg_detection_time:.3f}s")
            
            print(f"\nPER-CASE BREAKDOWN:")
            for case_name, result in all_results["case_results"].items():
                summary = result["summary"]
                print(f"  {case_name}: {summary['successful_tests']}/{summary['total_tests']} "
                      f"({summary['success_rate']:.1%}) - {result['difficulty']}")
        
        return all_results
    
    def demonstrate_interactive(self):
        """Interactive demonstration of edge cases."""
        print("\nINTERACTIVE EDGE CASE DEMONSTRATION")
        print("="*50)
        print("Available edge cases:")
        
        for i, (case_name, case_data) in enumerate(self.edge_case_images.items(), 1):
            print(f"{i}. {case_name}: {case_data['description']} ({case_data['difficulty']})")
        
        while True:
            try:
                choice = input(f"\nEnter case number (1-{len(self.edge_case_images)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    break
                
                case_index = int(choice) - 1
                case_names = list(self.edge_case_images.keys())
                
                if 0 <= case_index < len(case_names):
                    case_name = case_names[case_index]
                    self.test_edge_case(case_name, verbose=True)
                else:
                    print("Invalid choice. Please try again.")
                    
            except ValueError:
                print("Invalid input. Please enter a number or 'q'.")
            except KeyboardInterrupt:
                print("\nDemo interrupted.")
                break


def main():
    """Main function for edge case testing demo."""
    if not COMPONENTS_AVAILABLE:
        print("Error: Manual selection components not available")
        print("Please ensure all required modules are installed and working")
        return
    
    try:
        demo = EdgeCaseTestDemo()
        
        print("\nEdge Case Testing Demo")
        print("="*30)
        print("Choose an option:")
        print("1. Run all edge cases automatically")
        print("2. Interactive edge case selection")
        print("3. Quick test (overlapping circles)")
        print("4. Exit")
        
        while True:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                demo.run_all_edge_cases(verbose=True)
                break
            elif choice == '2':
                demo.demonstrate_interactive()
                break
            elif choice == '3':
                demo.test_edge_case("overlapping_circles", verbose=True)
                break
            elif choice == '4':
                break
            else:
                print("Invalid choice, please try again")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()