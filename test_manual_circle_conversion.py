#!/usr/bin/env python3
"""
Test script for manual circle dimension conversion fix.
Tests the _convert_manual_circle_to_measurement function with proper scaling factors.
"""

import numpy as np
from measure import _convert_manual_circle_to_measurement

def test_manual_circle_dimension_conversion():
    """Test circle pixel-to-mm conversion with known scaling factors"""
    print("Testing manual circle dimension conversion...")
    
    # Test case 1: Basic conversion with known scaling factor
    shape_result = {
        "type": "circle",
        "center": (100, 100),
        "radius": 50.0,  # 50 pixels radius
        "confidence_score": 0.95,
        "mode": "manual_circle"
    }
    
    mm_per_px_x = 0.2  # 0.2 mm per pixel
    selection_rect = (50, 50, 100, 100)
    
    result = _convert_manual_circle_to_measurement(shape_result, mm_per_px_x, selection_rect)
    
    # Verify results
    expected_diameter_px = 2.0 * 50.0  # 100 pixels
    expected_diameter_mm = expected_diameter_px * mm_per_px_x  # 100 * 0.2 = 20.0 mm
    
    assert result is not None, "Result should not be None"
    assert result["type"] == "circle", f"Expected circle, got {result['type']}"
    assert result["diameter_mm"] == expected_diameter_mm, f"Expected diameter {expected_diameter_mm}mm, got {result['diameter_mm']}mm"
    assert result["radius_px"] == 50.0, f"Expected radius_px 50.0, got {result['radius_px']}"
    assert result["center"] == (100, 100), f"Expected center (100, 100), got {result['center']}"
    assert result["detection_method"] == "manual", f"Expected manual detection method, got {result['detection_method']}"
    
    print(f"✓ Test 1 passed: {expected_diameter_px}px diameter → {expected_diameter_mm}mm")
    
    # Test case 2: Different scaling factor
    mm_per_px_x = 0.1  # 0.1 mm per pixel
    result2 = _convert_manual_circle_to_measurement(shape_result, mm_per_px_x, selection_rect)
    expected_diameter_mm2 = 100.0 * 0.1  # 10.0 mm
    
    assert result2["diameter_mm"] == expected_diameter_mm2, f"Expected diameter {expected_diameter_mm2}mm, got {result2['diameter_mm']}mm"
    assert result2["radius_px"] == 50.0, "Radius_px should remain in pixels for rendering"
    
    print(f"✓ Test 2 passed: {expected_diameter_px}px diameter → {expected_diameter_mm2}mm with different scaling")
    
    print("All basic conversion tests passed!")

def test_scaling_factor_validation():
    """Test scaling factor validation and error handling"""
    print("\nTesting scaling factor validation...")
    
    shape_result = {
        "type": "circle",
        "center": (100, 100),
        "radius": 50.0,
        "confidence_score": 0.95
    }
    selection_rect = (50, 50, 100, 100)
    
    # Test None scaling factor
    try:
        _convert_manual_circle_to_measurement(shape_result, None, selection_rect)
        assert False, "Should have raised ValueError for None scaling factor"
    except ValueError as e:
        assert "cannot be None" in str(e)
        print("✓ Test 3 passed: None scaling factor properly rejected")
    
    # Test zero scaling factor
    try:
        _convert_manual_circle_to_measurement(shape_result, 0.0, selection_rect)
        assert False, "Should have raised ValueError for zero scaling factor"
    except ValueError as e:
        assert "must be positive" in str(e)
        print("✓ Test 4 passed: Zero scaling factor properly rejected")
    
    # Test negative scaling factor
    try:
        _convert_manual_circle_to_measurement(shape_result, -0.1, selection_rect)
        assert False, "Should have raised ValueError for negative scaling factor"
    except ValueError as e:
        assert "must be positive" in str(e)
        print("✓ Test 5 passed: Negative scaling factor properly rejected")
    
    # Test out of range scaling factor (too small)
    try:
        _convert_manual_circle_to_measurement(shape_result, 0.001, selection_rect)
        assert False, "Should have raised ValueError for out of range scaling factor"
    except ValueError as e:
        assert "out of reasonable range" in str(e)
        print("✓ Test 6 passed: Too small scaling factor properly rejected")
    
    # Test out of range scaling factor (too large)
    try:
        _convert_manual_circle_to_measurement(shape_result, 200.0, selection_rect)
        assert False, "Should have raised ValueError for out of range scaling factor"
    except ValueError as e:
        assert "out of reasonable range" in str(e)
        print("✓ Test 7 passed: Too large scaling factor properly rejected")
    
    print("All validation tests passed!")

def test_measurement_result_data_structure():
    """Test that measurement result data structure is consistent"""
    print("\nTesting measurement result data structure...")
    
    shape_result = {
        "type": "circle",
        "center": (150, 200),
        "radius": 25.0,
        "confidence_score": 0.88,
        "mode": "manual_circle"
    }
    
    mm_per_px_x = 0.15
    selection_rect = (125, 175, 50, 50)
    
    result = _convert_manual_circle_to_measurement(shape_result, mm_per_px_x, selection_rect)
    
    # Check all required fields are present
    required_fields = [
        "type", "diameter_mm", "center", "radius_px", "hit_contour", 
        "area_px", "inner", "detection_method", "selection_rect",
        "confidence_score", "manual_mode"
    ]
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Check field types and values
    assert isinstance(result["diameter_mm"], (int, float)), "diameter_mm should be numeric"
    assert isinstance(result["radius_px"], (int, float)), "radius_px should be numeric"
    assert isinstance(result["center"], tuple), "center should be tuple"
    assert isinstance(result["hit_contour"], np.ndarray), "hit_contour should be numpy array"
    assert isinstance(result["area_px"], (int, float)), "area_px should be numeric"
    assert result["inner"] == False, "Manual selections should not be marked as inner"
    assert result["detection_method"] == "manual", "Detection method should be manual"
    
    print("✓ Test 8 passed: All required fields present with correct types")
    
    # Verify hit contour format
    hit_contour = result["hit_contour"]
    assert len(hit_contour.shape) == 3, f"Hit contour should be 3D array, got {len(hit_contour.shape)}D"
    assert hit_contour.shape[1] == 1, f"Hit contour second dimension should be 1, got {hit_contour.shape[1]}"
    assert hit_contour.shape[2] == 2, f"Hit contour third dimension should be 2, got {hit_contour.shape[2]}"
    
    print("✓ Test 9 passed: Hit contour has correct format")
    print("All data structure tests passed!")

if __name__ == "__main__":
    test_manual_circle_dimension_conversion()
    test_scaling_factor_validation()
    test_measurement_result_data_structure()
    print("\n🎉 All tests passed! Manual circle dimension conversion is working correctly.")