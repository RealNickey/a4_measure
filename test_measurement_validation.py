#!/usr/bin/env python3
"""
Test script for measurement validation and consistency checking functions.

This script tests the new validation functions added to measure.py for task 5:
- compare_auto_vs_manual_measurements
- validate_measurement_result_data_structure  
- log_measurement_consistency_warning
- validate_measurement_consistency
"""

import numpy as np
import logging
from measure import (
    compare_auto_vs_manual_measurements,
    validate_measurement_result_data_structure,
    log_measurement_consistency_warning,
    validate_measurement_consistency
)

# Configure logging to see warning messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_test_circle_result(diameter_mm, detection_method="automatic"):
    """Create a test circle measurement result."""
    radius_px = diameter_mm * 2  # Simplified conversion
    center = (100, 100)
    
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

def create_test_rectangle_result(width_mm, height_mm, detection_method="automatic"):
    """Create a test rectangle measurement result."""
    center = (100, 100)
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

def test_data_structure_validation():
    """Test the validate_measurement_result_data_structure function."""
    print("\n=== Testing Data Structure Validation ===")
    
    # Test valid circle result
    valid_circle = create_test_circle_result(50.0, "manual")
    validation = validate_measurement_result_data_structure(valid_circle)
    print(f"Valid circle validation: {validation['valid']}")
    assert validation["valid"], f"Valid circle should pass validation: {validation['issues']}"
    
    # Test valid rectangle result
    valid_rectangle = create_test_rectangle_result(30.0, 40.0, "automatic")
    validation = validate_measurement_result_data_structure(valid_rectangle)
    print(f"Valid rectangle validation: {validation['valid']}")
    assert validation["valid"], f"Valid rectangle should pass validation: {validation['issues']}"
    
    # Test invalid result (missing fields)
    invalid_result = {"type": "circle"}
    validation = validate_measurement_result_data_structure(invalid_result)
    print(f"Invalid result validation: {validation['valid']} (expected False)")
    assert not validation["valid"], "Invalid result should fail validation"
    print(f"Issues found: {validation['issues']}")
    
    # Test None result
    validation = validate_measurement_result_data_structure(None)
    print(f"None result validation: {validation['valid']} (expected False)")
    assert not validation["valid"], "None result should fail validation"
    
    print("✓ Data structure validation tests passed!")

def test_measurement_comparison():
    """Test the compare_auto_vs_manual_measurements function."""
    print("\n=== Testing Measurement Comparison ===")
    
    # Test consistent circle measurements (within tolerance)
    auto_circle = create_test_circle_result(50.0, "automatic")
    manual_circle = create_test_circle_result(50.5, "manual")  # 0.5mm difference, within 2mm tolerance
    
    comparison = compare_auto_vs_manual_measurements(auto_circle, manual_circle)
    print(f"Consistent circle comparison: {comparison['consistent']}")
    assert comparison["consistent"], "Small difference should be within tolerance"
    
    # Test inconsistent circle measurements (outside tolerance)
    auto_circle = create_test_circle_result(50.0, "automatic")
    manual_circle = create_test_circle_result(55.0, "manual")  # 5mm difference, outside 2mm tolerance
    
    comparison = compare_auto_vs_manual_measurements(auto_circle, manual_circle)
    print(f"Inconsistent circle comparison: {comparison['consistent']} (expected False)")
    assert not comparison["consistent"], "Large difference should exceed tolerance"
    print(f"Diameter difference: {comparison['differences']['diameter_mm']:.1f}mm")
    
    # Test consistent rectangle measurements
    auto_rect = create_test_rectangle_result(30.0, 40.0, "automatic")
    manual_rect = create_test_rectangle_result(30.5, 40.8, "manual")  # Small differences
    
    comparison = compare_auto_vs_manual_measurements(auto_rect, manual_rect)
    print(f"Consistent rectangle comparison: {comparison['consistent']}")
    assert comparison["consistent"], "Small differences should be within tolerance"
    
    # Test inconsistent rectangle measurements
    auto_rect = create_test_rectangle_result(30.0, 40.0, "automatic")
    manual_rect = create_test_rectangle_result(35.0, 45.0, "manual")  # Large differences
    
    comparison = compare_auto_vs_manual_measurements(auto_rect, manual_rect)
    print(f"Inconsistent rectangle comparison: {comparison['consistent']} (expected False)")
    assert not comparison["consistent"], "Large differences should exceed tolerance"
    print(f"Width difference: {comparison['differences']['width_mm']:.1f}mm")
    print(f"Height difference: {comparison['differences']['height_mm']:.1f}mm")
    
    # Test shape type mismatch
    comparison = compare_auto_vs_manual_measurements(auto_circle, manual_rect)
    print(f"Shape mismatch comparison: {comparison['consistent']} (expected False)")
    assert not comparison["consistent"], "Different shape types should not be consistent"
    
    print("✓ Measurement comparison tests passed!")

def test_consistency_validation():
    """Test the validate_measurement_consistency function."""
    print("\n=== Testing Consistency Validation ===")
    
    # Create test data sets
    auto_results = [
        create_test_circle_result(50.0, "automatic"),
        create_test_rectangle_result(30.0, 40.0, "automatic")
    ]
    
    manual_results = [
        create_test_circle_result(50.5, "manual"),  # Consistent
        create_test_rectangle_result(35.0, 45.0, "manual")  # Inconsistent
    ]
    
    validation = validate_measurement_consistency(auto_results, manual_results)
    
    print(f"Total comparisons: {validation['total_comparisons']}")
    print(f"Consistent measurements: {validation['consistent_measurements']}")
    print(f"Inconsistent measurements: {validation['inconsistent_measurements']}")
    print(f"Overall consistent: {validation['overall_consistent']}")
    
    assert validation["total_comparisons"] == 2, "Should have 2 comparisons"
    assert validation["consistent_measurements"] == 1, "Should have 1 consistent measurement"
    assert validation["inconsistent_measurements"] == 1, "Should have 1 inconsistent measurement"
    assert not validation["overall_consistent"], "Overall should be inconsistent due to rectangle"
    
    print("✓ Consistency validation tests passed!")

def test_logging_functionality():
    """Test the log_measurement_consistency_warning function."""
    print("\n=== Testing Logging Functionality ===")
    
    # Create an inconsistent comparison result
    auto_circle = create_test_circle_result(50.0, "automatic")
    manual_circle = create_test_circle_result(55.0, "manual")
    
    comparison = compare_auto_vs_manual_measurements(auto_circle, manual_circle)
    
    print("Testing warning log (should see WARNING message above):")
    log_measurement_consistency_warning(comparison)
    
    # Test with consistent measurement (should not log warning)
    auto_circle_2 = create_test_circle_result(50.0, "automatic")
    manual_circle_2 = create_test_circle_result(50.5, "manual")
    
    comparison_2 = compare_auto_vs_manual_measurements(auto_circle_2, manual_circle_2)
    print("Testing with consistent measurement (should not see warning):")
    log_measurement_consistency_warning(comparison_2)
    
    print("✓ Logging functionality tests passed!")

def main():
    """Run all validation tests."""
    print("Testing Measurement Validation and Consistency Checking Functions")
    print("=" * 70)
    
    try:
        test_data_structure_validation()
        test_measurement_comparison()
        test_consistency_validation()
        test_logging_functionality()
        
        print("\n" + "=" * 70)
        print("✅ All validation tests passed successfully!")
        print("\nThe following functions have been implemented and tested:")
        print("- compare_auto_vs_manual_measurements: ✓")
        print("- validate_measurement_result_data_structure: ✓")
        print("- log_measurement_consistency_warning: ✓")
        print("- validate_measurement_consistency: ✓")
        print("\nThese functions implement:")
        print("- Tolerance checking (±2mm or ±2% whichever is larger)")
        print("- Logging for measurement consistency warnings")
        print("- Helper function to validate measurement result data structure")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()