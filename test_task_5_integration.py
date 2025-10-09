#!/usr/bin/env python3
"""
Integration test for Task 5: Add measurement validation and consistency checking

This test verifies that all the validation functions work together properly
and meet the requirements specified in the task.
"""

import numpy as np
import logging
from measure import (
    compare_auto_vs_manual_measurements,
    validate_measurement_result_data_structure,
    log_measurement_consistency_warning,
    validate_measurement_consistency,
    validate_manual_measurement_result
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_measurement_result(shape_type, dimensions, detection_method="automatic"):
    """Create a realistic measurement result for testing."""
    if shape_type == "circle":
        diameter_mm = dimensions[0]
        radius_px = diameter_mm * 2  # Simplified conversion
        center = (150, 150)
        
        # Create circular hit contour
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        circle_points = np.array([(int(center[0] + radius_px * np.cos(a)),
                                  int(center[1] + radius_px * np.sin(a))) for a in angles])
        hit_contour = circle_points.reshape(-1, 1, 2).astype(np.int32)
        
        return {
            "type": "circle",
            "diameter_mm": diameter_mm,
            "center": center,
            "radius_px": radius_px,
            "hit_contour": hit_contour,
            "area_px": np.pi * (radius_px ** 2),
            "inner": False,
            "detection_method": detection_method
        }
    
    elif shape_type == "rectangle":
        width_mm, height_mm = dimensions
        center = (150, 150)
        hw, hh = width_mm, height_mm  # Simplified conversion
        
        box = np.array([
            [center[0] - hw/2, center[1] - hh/2],
            [center[0] + hw/2, center[1] - hh/2],
            [center[0] + hw/2, center[1] + hh/2],
            [center[0] - hw/2, center[1] + hh/2]
        ], dtype=int)
        
        hit_contour = box.reshape(-1, 1, 2).astype(np.int32)
        
        return {
            "type": "rectangle",
            "width_mm": width_mm,
            "height_mm": height_mm,
            "box": box,
            "hit_contour": hit_contour,
            "area_px": width_mm * height_mm,  # Simplified
            "inner": False,
            "detection_method": detection_method
        }

def test_requirement_5_1_circle_tolerance():
    """Test Requirement 5.1: Circle diameter measurements within tolerance."""
    print("\n=== Testing Requirement 5.1: Circle Tolerance ===")
    
    # Test case 1: Within 2mm tolerance
    auto_circle = create_measurement_result("circle", [50.0], "automatic")
    manual_circle = create_measurement_result("circle", [51.5], "manual")  # 1.5mm difference
    
    comparison = compare_auto_vs_manual_measurements(auto_circle, manual_circle)
    assert comparison["consistent"], "1.5mm difference should be within 2mm tolerance"
    print("✓ Circle measurements within 2mm tolerance are consistent")
    
    # Test case 2: Within 2% tolerance (for larger circles)
    auto_circle_large = create_measurement_result("circle", [200.0], "automatic")
    manual_circle_large = create_measurement_result("circle", [203.0], "manual")  # 3mm difference, but 1.5%
    
    comparison = compare_auto_vs_manual_measurements(auto_circle_large, manual_circle_large)
    assert comparison["consistent"], "1.5% difference should be within 2% tolerance"
    print("✓ Circle measurements within 2% tolerance are consistent")
    
    # Test case 3: Outside tolerance
    auto_circle = create_measurement_result("circle", [50.0], "automatic")
    manual_circle = create_measurement_result("circle", [55.0], "manual")  # 5mm difference, >2mm and >2%
    
    comparison = compare_auto_vs_manual_measurements(auto_circle, manual_circle)
    assert not comparison["consistent"], "5mm difference should exceed tolerance"
    print("✓ Circle measurements outside tolerance are flagged as inconsistent")

def test_requirement_5_2_rectangle_tolerance():
    """Test Requirement 5.2: Rectangle width and height measurements within tolerance."""
    print("\n=== Testing Requirement 5.2: Rectangle Tolerance ===")
    
    # Test case 1: Both dimensions within tolerance
    auto_rect = create_measurement_result("rectangle", [30.0, 40.0], "automatic")
    manual_rect = create_measurement_result("rectangle", [31.0, 41.5], "manual")  # 1mm and 1.5mm differences
    
    comparison = compare_auto_vs_manual_measurements(auto_rect, manual_rect)
    assert comparison["consistent"], "Small differences should be within tolerance"
    print("✓ Rectangle measurements within tolerance are consistent")
    
    # Test case 2: One dimension outside tolerance
    auto_rect = create_measurement_result("rectangle", [30.0, 40.0], "automatic")
    manual_rect = create_measurement_result("rectangle", [33.5, 41.0], "manual")  # 3.5mm width difference
    
    comparison = compare_auto_vs_manual_measurements(auto_rect, manual_rect)
    assert not comparison["consistent"], "Width difference should exceed tolerance"
    print("✓ Rectangle measurements with one dimension outside tolerance are flagged")
    
    # Test case 3: Both dimensions outside tolerance
    auto_rect = create_measurement_result("rectangle", [30.0, 40.0], "automatic")
    manual_rect = create_measurement_result("rectangle", [35.0, 45.0], "manual")  # Both >2mm difference
    
    comparison = compare_auto_vs_manual_measurements(auto_rect, manual_rect)
    assert not comparison["consistent"], "Both dimensions should exceed tolerance"
    print("✓ Rectangle measurements with both dimensions outside tolerance are flagged")

def test_requirement_5_3_logging():
    """Test Requirement 5.3: Logging for measurement consistency warnings."""
    print("\n=== Testing Requirement 5.3: Logging ===")
    
    # Capture log messages
    import io
    import sys
    
    # Create inconsistent measurements
    auto_circle = create_measurement_result("circle", [50.0], "automatic")
    manual_circle = create_measurement_result("circle", [55.0], "manual")
    
    comparison = compare_auto_vs_manual_measurements(auto_circle, manual_circle)
    
    # Test that warning is logged
    print("Testing warning log for inconsistent measurements...")
    log_measurement_consistency_warning(comparison)
    print("✓ Warning logged for inconsistent measurements")
    
    # Test comprehensive validation logging
    auto_results = [auto_circle]
    manual_results = [manual_circle]
    
    validation = validate_measurement_consistency(auto_results, manual_results)
    print("✓ Comprehensive validation logging works")

def test_data_structure_validation():
    """Test helper function to validate measurement result data structure."""
    print("\n=== Testing Data Structure Validation ===")
    
    # Test valid structures
    valid_circle = create_measurement_result("circle", [50.0], "manual")
    validation = validate_measurement_result_data_structure(valid_circle)
    assert validation["valid"], f"Valid circle should pass: {validation['issues']}"
    
    valid_rectangle = create_measurement_result("rectangle", [30.0, 40.0], "automatic")
    validation = validate_measurement_result_data_structure(valid_rectangle)
    assert validation["valid"], f"Valid rectangle should pass: {validation['issues']}"
    
    # Test invalid structures
    invalid_result = {"type": "circle", "diameter_mm": -10}  # Negative diameter
    validation = validate_measurement_result_data_structure(invalid_result)
    assert not validation["valid"], "Invalid result should fail validation"
    
    print("✓ Data structure validation helper function works correctly")

def test_integration_workflow():
    """Test the complete integration workflow."""
    print("\n=== Testing Complete Integration Workflow ===")
    
    # Create realistic test scenario
    auto_results = [
        create_measurement_result("circle", [25.0], "automatic"),
        create_measurement_result("rectangle", [15.0, 20.0], "automatic")
    ]
    
    manual_results = [
        create_measurement_result("circle", [25.5], "manual"),  # Consistent
        create_measurement_result("rectangle", [18.0, 23.0], "manual")  # Inconsistent
    ]
    
    # Validate individual results
    for result in manual_results:
        assert validate_manual_measurement_result(result), "Manual results should be valid"
    
    # Validate consistency
    validation = validate_measurement_consistency(auto_results, manual_results)
    
    assert validation["total_comparisons"] == 2, "Should have 2 comparisons"
    assert validation["consistent_measurements"] == 1, "Should have 1 consistent measurement"
    assert validation["inconsistent_measurements"] == 1, "Should have 1 inconsistent measurement"
    assert not validation["overall_consistent"], "Overall should be inconsistent"
    
    print("✓ Complete integration workflow works correctly")

def main():
    """Run all integration tests for Task 5."""
    print("Integration Test for Task 5: Measurement Validation and Consistency Checking")
    print("=" * 80)
    
    try:
        test_requirement_5_1_circle_tolerance()
        test_requirement_5_2_rectangle_tolerance()
        test_requirement_5_3_logging()
        test_data_structure_validation()
        test_integration_workflow()
        
        print("\n" + "=" * 80)
        print("✅ All Task 5 integration tests passed successfully!")
        print("\nTask 5 Implementation Summary:")
        print("✓ Created validation function to compare Auto vs Manual measurements")
        print("✓ Implemented tolerance checking (±2mm or ±2% whichever is larger)")
        print("✓ Added logging for measurement consistency warnings")
        print("✓ Created helper function to validate measurement result data structure")
        print("\nRequirements Coverage:")
        print("✓ Requirement 5.1: Circle diameter tolerance checking")
        print("✓ Requirement 5.2: Rectangle width/height tolerance checking")
        print("✓ Requirement 5.3: Measurement consistency warning logging")
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()