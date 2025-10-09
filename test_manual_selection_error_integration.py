#!/usr/bin/env python3
"""
Integration test for manual selection error handling.

This test simulates real-world scenarios where A4 calibration might fail
and verifies that the system handles these gracefully.
"""

import sys
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional

# Import the functions we want to test
from measure import process_manual_selection


def create_test_image() -> np.ndarray:
    """Create a test image with a simple circle for testing."""
    image = np.ones((300, 300, 3), dtype=np.uint8) * 255  # White background
    
    # Draw a black circle
    center = (150, 150)
    radius = 50
    cv2.circle(image, center, radius, (0, 0, 0), -1)
    
    return image


def test_calibration_failure_scenarios():
    """Test various calibration failure scenarios."""
    print("Testing calibration failure scenarios...")
    
    image = create_test_image()
    selection_rect = (100, 100, 100, 100)  # Selection around the circle
    
    # Scenario 1: A4 paper not detected (None scaling factors)
    print("\n1. Testing A4 paper not detected scenario...")
    result = process_manual_selection(image, selection_rect, "manual_circle", None, None)
    assert result is None, "Should return None when A4 calibration fails"
    print("✓ Gracefully handles missing A4 calibration")
    
    # Scenario 2: Invalid calibration data (zero scaling factors)
    print("\n2. Testing invalid calibration data scenario...")
    result = process_manual_selection(image, selection_rect, "manual_circle", 0.0, 0.0)
    assert result is None, "Should return None when scaling factors are zero"
    print("✓ Gracefully handles invalid calibration data")
    
    # Scenario 3: Corrupted calibration (negative scaling factors)
    print("\n3. Testing corrupted calibration scenario...")
    result = process_manual_selection(image, selection_rect, "manual_circle", -0.5, -0.5)
    assert result is None, "Should return None when scaling factors are negative"
    print("✓ Gracefully handles corrupted calibration")
    
    # Scenario 4: Extreme calibration values (out of range)
    print("\n4. Testing extreme calibration values scenario...")
    result = process_manual_selection(image, selection_rect, "manual_circle", 1000.0, 1000.0)
    assert result is None, "Should return None when scaling factors are out of range"
    print("✓ Gracefully handles extreme calibration values")
    
    # Scenario 5: Mixed valid/invalid scaling factors
    print("\n5. Testing mixed valid/invalid scaling factors...")
    result = process_manual_selection(image, selection_rect, "manual_circle", 0.5, None)
    assert result is None, "Should return None when one scaling factor is invalid"
    print("✓ Gracefully handles mixed valid/invalid scaling factors")
    
    # Scenario 6: Valid calibration (should work)
    print("\n6. Testing valid calibration scenario...")
    try:
        result = process_manual_selection(image, selection_rect, "manual_circle", 0.5, 0.5)
        # Note: This might still return None if shape snapping fails, but it shouldn't crash
        print("✓ Valid calibration doesn't crash the system")
    except Exception as e:
        print(f"✗ Valid calibration caused unexpected error: {e}")
        raise
    
    print("\nAll calibration failure scenarios handled gracefully!")


def test_error_message_quality():
    """Test that error messages provide helpful guidance."""
    print("\nTesting error message quality...")
    
    image = create_test_image()
    selection_rect = (100, 100, 100, 100)
    
    # Capture output to verify error messages
    import io
    import contextlib
    
    # Test None scaling factors
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = process_manual_selection(image, selection_rect, "manual_circle", None, None)
    
    output = f.getvalue()
    assert "A4 calibration" in output, "Should mention A4 calibration in error message"
    assert "failed" in output.lower(), "Should mention failure in error message"
    assert "ensure A4 paper" in output, "Should provide guidance about A4 paper"
    print("✓ Error messages mention A4 calibration and provide guidance")
    
    # Test zero scaling factors
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = process_manual_selection(image, selection_rect, "manual_circle", 0.0, 0.5)
    
    output = f.getvalue()
    assert "positive" in output, "Should mention positive requirement"
    assert "calibration" in output.lower(), "Should mention calibration"
    print("✓ Error messages explain validation requirements")
    
    print("Error message quality verified!")


def test_system_stability():
    """Test that the system remains stable after errors."""
    print("\nTesting system stability after errors...")
    
    image = create_test_image()
    selection_rect = (100, 100, 100, 100)
    
    # Cause multiple errors in sequence
    error_scenarios = [
        (None, None),
        (0.0, 0.5),
        (-0.5, 0.5),
        (1000.0, 0.5),
        ("invalid", 0.5),
    ]
    
    for i, (mm_per_px_x, mm_per_px_y) in enumerate(error_scenarios):
        try:
            result = process_manual_selection(image, selection_rect, "manual_circle", mm_per_px_x, mm_per_px_y)
            assert result is None, f"Error scenario {i+1} should return None"
        except Exception as e:
            print(f"✗ Error scenario {i+1} caused unexpected exception: {e}")
            raise
    
    # After all errors, system should still work with valid inputs
    try:
        result = process_manual_selection(image, selection_rect, "manual_circle", 0.5, 0.5)
        print("✓ System remains stable after multiple errors")
    except Exception as e:
        print(f"✗ System became unstable after errors: {e}")
        raise
    
    print("System stability verified!")


def main():
    """Run all integration tests."""
    print("Running manual selection error handling integration tests...\n")
    
    try:
        test_calibration_failure_scenarios()
        test_error_message_quality()
        test_system_stability()
        
        print("\n🎉 All integration tests passed successfully!")
        print("\nIntegration test results:")
        print("✓ System gracefully handles all calibration failure scenarios")
        print("✓ Error messages provide clear guidance for users")
        print("✓ System remains stable after encountering errors")
        print("✓ Manual selection error handling is robust and user-friendly")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)