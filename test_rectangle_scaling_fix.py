#!/usr/bin/env python3
"""
Test script to verify the manual rectangle dimension conversion fix.
Tests that the function properly applies axis-specific scaling factors.
"""

import numpy as np
import sys
import traceback

# Import the function we're testing
from measure import _convert_manual_rectangle_to_measurement

def test_rectangle_scaling_basic():
    """Test basic rectangle scaling with different X and Y factors"""
    print("Testing basic rectangle scaling...")
    
    # Create a mock shape result
    shape_result = {
        "type": "rectangle",
        "width": 100,  # 100 pixels width
        "height": 200,  # 200 pixels height
        "center": (150, 150),
        "confidence_score": 0.9,
        "mode": "manual_rectangle"
    }
    
    # Different scaling factors for X and Y
    mm_per_px_x = 0.5  # 0.5 mm per pixel in X direction
    mm_per_px_y = 0.3  # 0.3 mm per pixel in Y direction
    selection_rect = (100, 100, 100, 100)
    
    result = _convert_manual_rectangle_to_measurement(
        shape_result, mm_per_px_x, mm_per_px_y, selection_rect
    )
    
    # Verify the conversion
    expected_width_mm = 100 * 0.5  # 50 mm (width uses mm_per_px_x)
    expected_height_mm = 200 * 0.3  # 60 mm (height uses mm_per_px_y)
    
    assert result["width_mm"] == expected_width_mm, f"Expected width {expected_width_mm}, got {result['width_mm']}"
    assert result["height_mm"] == expected_height_mm, f"Expected height {expected_height_mm}, got {result['height_mm']}"
    assert result["type"] == "rectangle"
    assert result["detection_method"] == "manual"
    
    print(f"✓ Width: {result['width_mm']} mm (expected {expected_width_mm})")
    print(f"✓ Height: {result['height_mm']} mm (expected {expected_height_mm})")
    print("✓ Basic scaling test passed!")

def test_rectangle_scaling_validation():
    """Test input validation for scaling factors"""
    print("\nTesting scaling factor validation...")
    
    shape_result = {
        "type": "rectangle",
        "width": 100,
        "height": 200,
        "center": (150, 150)
    }
    selection_rect = (100, 100, 100, 100)
    
    # Test None values
    try:
        _convert_manual_rectangle_to_measurement(shape_result, None, 0.5, selection_rect)
        assert False, "Should have raised ValueError for None mm_per_px_x"
    except ValueError as e:
        print(f"✓ Correctly caught None mm_per_px_x: {e}")
    
    try:
        _convert_manual_rectangle_to_measurement(shape_result, 0.5, None, selection_rect)
        assert False, "Should have raised ValueError for None mm_per_px_y"
    except ValueError as e:
        print(f"✓ Correctly caught None mm_per_px_y: {e}")
    
    # Test negative values
    try:
        _convert_manual_rectangle_to_measurement(shape_result, -0.5, 0.5, selection_rect)
        assert False, "Should have raised ValueError for negative mm_per_px_x"
    except ValueError as e:
        print(f"✓ Correctly caught negative mm_per_px_x: {e}")
    
    try:
        _convert_manual_rectangle_to_measurement(shape_result, 0.5, -0.3, selection_rect)
        assert False, "Should have raised ValueError for negative mm_per_px_y"
    except ValueError as e:
        print(f"✓ Correctly caught negative mm_per_px_y: {e}")
    
    # Test zero values
    try:
        _convert_manual_rectangle_to_measurement(shape_result, 0.0, 0.5, selection_rect)
        assert False, "Should have raised ValueError for zero mm_per_px_x"
    except ValueError as e:
        print(f"✓ Correctly caught zero mm_per_px_x: {e}")
    
    print("✓ Validation tests passed!")

def test_rectangle_dimensions_format():
    """Test that dimensions are extracted correctly from different formats"""
    print("\nTesting dimension format handling...")
    
    # Test with dimensions tuple
    shape_result_dims = {
        "type": "rectangle",
        "dimensions": (80, 120),  # width, height
        "center": (150, 150)
    }
    
    mm_per_px_x = 0.4
    mm_per_px_y = 0.6
    selection_rect = (100, 100, 100, 100)
    
    result = _convert_manual_rectangle_to_measurement(
        shape_result_dims, mm_per_px_x, mm_per_px_y, selection_rect
    )
    
    # Note: function normalizes width < height, so min(80,120)=80, max(80,120)=120
    expected_width_mm = 80 * 0.4  # 32 mm
    expected_height_mm = 120 * 0.6  # 72 mm
    
    assert result["width_mm"] == expected_width_mm
    assert result["height_mm"] == expected_height_mm
    
    print(f"✓ Dimensions format test passed: {result['width_mm']}mm x {result['height_mm']}mm")

def test_box_coordinates_remain_pixels():
    """Test that box coordinates remain in pixels for rendering"""
    print("\nTesting box coordinates remain in pixels...")
    
    # Create shape result with explicit box
    box_pixels = np.array([
        [100, 100],
        [200, 100], 
        [200, 300],
        [100, 300]
    ], dtype=int)
    
    shape_result = {
        "type": "rectangle",
        "width": 100,
        "height": 200,
        "box": box_pixels,
        "center": (150, 200)
    }
    
    mm_per_px_x = 0.5
    mm_per_px_y = 0.3
    selection_rect = (90, 90, 120, 220)
    
    result = _convert_manual_rectangle_to_measurement(
        shape_result, mm_per_px_x, mm_per_px_y, selection_rect
    )
    
    # Box should remain in pixels (unchanged)
    assert np.array_equal(result["box"], box_pixels), "Box coordinates should remain in pixels"
    assert result["box"].dtype == np.int32 or result["box"].dtype == int, "Box should be integer type"
    
    # But measurements should be in mm
    assert result["width_mm"] == 100 * 0.5  # 50 mm
    assert result["height_mm"] == 200 * 0.3  # 60 mm
    
    print("✓ Box coordinates correctly remain in pixels for rendering")
    print(f"✓ Measurements correctly converted to mm: {result['width_mm']}mm x {result['height_mm']}mm")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Manual Rectangle Dimension Conversion Fix")
    print("=" * 60)
    
    try:
        test_rectangle_scaling_basic()
        test_rectangle_scaling_validation()
        test_rectangle_dimensions_format()
        test_box_coordinates_remain_pixels()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Manual rectangle dimension conversion is working correctly.")
        print("- Axis-specific scaling factors are properly applied")
        print("- Input validation prevents invalid scaling factors")
        print("- Box coordinates remain in pixels for rendering")
        print("- Different dimension formats are handled correctly")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)