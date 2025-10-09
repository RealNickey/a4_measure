#!/usr/bin/env python3
"""
Test script for scaling factor validation in manual selection processing.

This test verifies that the enhanced error handling for invalid scaling factors
works correctly and provides appropriate error messages for calibration issues.
"""

import sys
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Import the functions we want to test
from measure import (
    _validate_scaling_factors,
    process_manual_selection,
    classify_and_measure_manual_selection,
    _convert_manual_circle_to_measurement,
    _convert_manual_rectangle_to_measurement
)


def test_validate_scaling_factors():
    """Test the _validate_scaling_factors function with various inputs."""
    print("Testing _validate_scaling_factors function...")
    
    # Test valid scaling factors
    result = _validate_scaling_factors(0.5, 0.5)
    assert result["valid"] == True, "Valid scaling factors should pass validation"
    assert len(result["errors"]) == 0, "Valid scaling factors should have no errors"
    print("✓ Valid scaling factors pass validation")
    
    # Test None values
    result = _validate_scaling_factors(None, 0.5)
    assert result["valid"] == False, "None mm_per_px_x should fail validation"
    assert any("mm_per_px_x cannot be None" in error for error in result["errors"]), "Should have None error for X"
    print("✓ None mm_per_px_x fails validation with appropriate error")
    
    result = _validate_scaling_factors(0.5, None)
    assert result["valid"] == False, "None mm_per_px_y should fail validation"
    assert any("mm_per_px_y cannot be None" in error for error in result["errors"]), "Should have None error for Y"
    print("✓ None mm_per_px_y fails validation with appropriate error")
    
    # Test zero values
    result = _validate_scaling_factors(0.0, 0.5)
    assert result["valid"] == False, "Zero mm_per_px_x should fail validation"
    assert any("must be positive" in error for error in result["errors"]), "Should have positive error"
    print("✓ Zero scaling factors fail validation")
    
    # Test negative values
    result = _validate_scaling_factors(-0.5, 0.5)
    assert result["valid"] == False, "Negative mm_per_px_x should fail validation"
    assert any("must be positive" in error for error in result["errors"]), "Should have positive error"
    print("✓ Negative scaling factors fail validation")
    
    # Test out of range values
    result = _validate_scaling_factors(200.0, 0.5)
    assert result["valid"] == False, "Out of range mm_per_px_x should fail validation"
    assert any("out of reasonable range" in error for error in result["errors"]), "Should have range error"
    print("✓ Out of range scaling factors fail validation")
    
    # Test unusual but valid values (should generate warnings)
    result = _validate_scaling_factors(15.0, 0.5)
    assert result["valid"] == True, "Unusual but valid scaling factors should pass"
    assert len(result["warnings"]) > 0, "Unusual values should generate warnings"
    print("✓ Unusual scaling factors generate warnings")
    
    # Test significant anisotropy
    result = _validate_scaling_factors(1.0, 0.5)
    assert result["valid"] == True, "Anisotropic scaling should be valid"
    assert any("scaling difference" in warning for warning in result["warnings"]), "Should warn about anisotropy"
    print("✓ Anisotropic scaling generates warnings")
    
    print("All _validate_scaling_factors tests passed!\n")


def test_convert_manual_circle_error_handling():
    """Test error handling in _convert_manual_circle_to_measurement."""
    print("Testing _convert_manual_circle_to_measurement error handling...")
    
    # Valid shape result for testing
    valid_shape_result = {
        "type": "circle",
        "center": (100, 100),
        "radius": 50,
        "confidence_score": 0.8
    }
    selection_rect = (50, 50, 100, 100)
    
    # Test None scaling factor
    try:
        _convert_manual_circle_to_measurement(valid_shape_result, None, selection_rect)
        assert False, "Should raise ValueError for None scaling factor"
    except ValueError as e:
        assert "cannot be None" in str(e), f"Should mention None error: {e}"
        assert "calibration" in str(e).lower(), f"Should mention calibration: {e}"
        print("✓ None scaling factor raises appropriate ValueError")
    
    # Test zero scaling factor
    try:
        _convert_manual_circle_to_measurement(valid_shape_result, 0.0, selection_rect)
        assert False, "Should raise ValueError for zero scaling factor"
    except ValueError as e:
        assert "must be positive" in str(e), f"Should mention positive requirement: {e}"
        print("✓ Zero scaling factor raises appropriate ValueError")
    
    # Test negative scaling factor
    try:
        _convert_manual_circle_to_measurement(valid_shape_result, -0.5, selection_rect)
        assert False, "Should raise ValueError for negative scaling factor"
    except ValueError as e:
        assert "must be positive" in str(e), f"Should mention positive requirement: {e}"
        print("✓ Negative scaling factor raises appropriate ValueError")
    
    # Test out of range scaling factor
    try:
        _convert_manual_circle_to_measurement(valid_shape_result, 200.0, selection_rect)
        assert False, "Should raise ValueError for out of range scaling factor"
    except ValueError as e:
        assert "out of reasonable range" in str(e), f"Should mention range error: {e}"
        print("✓ Out of range scaling factor raises appropriate ValueError")
    
    # Test invalid shape data
    invalid_shape_result = {
        "type": "circle",
        "center": (100, 100)
        # Missing radius/dimensions
    }
    try:
        _convert_manual_circle_to_measurement(invalid_shape_result, 0.5, selection_rect)
        assert False, "Should raise ValueError for missing shape data"
    except ValueError as e:
        assert "Missing required shape data" in str(e) or "Invalid circle radius" in str(e), f"Should mention missing data: {e}"
        print("✓ Missing shape data raises appropriate ValueError")
    
    print("All _convert_manual_circle_to_measurement error handling tests passed!\n")


def test_convert_manual_rectangle_error_handling():
    """Test error handling in _convert_manual_rectangle_to_measurement."""
    print("Testing _convert_manual_rectangle_to_measurement error handling...")
    
    # Valid shape result for testing
    valid_shape_result = {
        "type": "rectangle",
        "center": (100, 100),
        "width": 80,
        "height": 60,
        "confidence_score": 0.8
    }
    selection_rect = (50, 50, 100, 100)
    
    # Test None scaling factors
    try:
        _convert_manual_rectangle_to_measurement(valid_shape_result, None, 0.5, selection_rect)
        assert False, "Should raise ValueError for None mm_per_px_x"
    except ValueError as e:
        assert "mm_per_px_x cannot be None" in str(e), f"Should mention None error: {e}"
        print("✓ None mm_per_px_x raises appropriate ValueError")
    
    try:
        _convert_manual_rectangle_to_measurement(valid_shape_result, 0.5, None, selection_rect)
        assert False, "Should raise ValueError for None mm_per_px_y"
    except ValueError as e:
        assert "mm_per_px_y cannot be None" in str(e), f"Should mention None error: {e}"
        print("✓ None mm_per_px_y raises appropriate ValueError")
    
    # Test zero scaling factors
    try:
        _convert_manual_rectangle_to_measurement(valid_shape_result, 0.0, 0.5, selection_rect)
        assert False, "Should raise ValueError for zero mm_per_px_x"
    except ValueError as e:
        assert "must be positive" in str(e), f"Should mention positive requirement: {e}"
        print("✓ Zero mm_per_px_x raises appropriate ValueError")
    
    # Test invalid shape data
    invalid_shape_result = {
        "type": "rectangle",
        "center": (100, 100)
        # Missing width/height/dimensions
    }
    try:
        _convert_manual_rectangle_to_measurement(invalid_shape_result, 0.5, 0.5, selection_rect)
        assert False, "Should raise ValueError for missing shape data"
    except ValueError as e:
        assert "Cannot determine rectangle dimensions" in str(e) or "Missing required shape data" in str(e), f"Should mention missing data: {e}"
        print("✓ Missing shape data raises appropriate ValueError")
    
    print("All _convert_manual_rectangle_to_measurement error handling tests passed!\n")


def test_classify_and_measure_manual_selection_error_handling():
    """Test error handling in classify_and_measure_manual_selection."""
    print("Testing classify_and_measure_manual_selection error handling...")
    
    # Create a dummy image
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    selection_rect = (50, 50, 100, 100)
    
    # Test None shape result
    result = classify_and_measure_manual_selection(image, selection_rect, None, 0.5, 0.5)
    assert result is None, "None shape result should return None"
    print("✓ None shape result returns None")
    
    # Test shape result without type
    invalid_shape_result = {"center": (100, 100)}
    result = classify_and_measure_manual_selection(image, selection_rect, invalid_shape_result, 0.5, 0.5)
    assert result is None, "Shape result without type should return None"
    print("✓ Shape result without type returns None")
    
    # Test invalid scaling factors
    valid_shape_result = {
        "type": "circle",
        "center": (100, 100),
        "radius": 50
    }
    result = classify_and_measure_manual_selection(image, selection_rect, valid_shape_result, None, 0.5)
    assert result is None, "Invalid scaling factors should return None"
    print("✓ Invalid scaling factors return None")
    
    # Test unsupported shape type
    unsupported_shape_result = {
        "type": "triangle",
        "center": (100, 100)
    }
    result = classify_and_measure_manual_selection(image, selection_rect, unsupported_shape_result, 0.5, 0.5)
    assert result is None, "Unsupported shape type should return None"
    print("✓ Unsupported shape type returns None")
    
    print("All classify_and_measure_manual_selection error handling tests passed!\n")


def test_graceful_degradation():
    """Test that error handling ensures graceful degradation."""
    print("Testing graceful degradation...")
    
    # Create a dummy image
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    selection_rect = (50, 50, 100, 100)
    
    # Test that invalid inputs don't crash the system
    test_cases = [
        # (mm_per_px_x, mm_per_px_y, expected_result)
        (None, None, None),
        (0.0, 0.5, None),
        (-0.5, 0.5, None),
        (200.0, 0.5, None),
        ("invalid", 0.5, None),
        (0.5, "invalid", None),
    ]
    
    for mm_per_px_x, mm_per_px_y, expected in test_cases:
        try:
            # This should not crash, even with invalid inputs
            result = process_manual_selection(image, selection_rect, "manual_circle", mm_per_px_x, mm_per_px_y)
            assert result == expected, f"Expected {expected}, got {result} for inputs ({mm_per_px_x}, {mm_per_px_y})"
            print(f"✓ Graceful handling of inputs ({mm_per_px_x}, {mm_per_px_y})")
        except Exception as e:
            print(f"✗ Unexpected exception for inputs ({mm_per_px_x}, {mm_per_px_y}): {e}")
            raise
    
    print("All graceful degradation tests passed!\n")


def main():
    """Run all error handling tests."""
    print("Running scaling factor validation tests...\n")
    
    try:
        test_validate_scaling_factors()
        test_convert_manual_circle_error_handling()
        test_convert_manual_rectangle_error_handling()
        test_classify_and_measure_manual_selection_error_handling()
        test_graceful_degradation()
        
        print("🎉 All error handling tests passed successfully!")
        print("\nError handling implementation verified:")
        print("✓ Validation for None, zero, or negative scaling factors")
        print("✓ Clear error messages for calibration issues")
        print("✓ Graceful degradation when scaling factors are invalid")
        print("✓ Existing shape detection error handling maintained")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)