#!/usr/bin/env python3
"""
Integration test to verify the rectangle scaling fix works with the full pipeline.
"""

import numpy as np
import cv2
from measure import classify_and_measure_manual_selection, validate_manual_measurement_result

def test_integration_with_manual_selection():
    """Test integration with the manual selection pipeline"""
    print("Testing integration with manual selection pipeline...")
    
    # Create a mock image
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Create a mock shape result that would come from the shape snapping engine
    shape_result = {
        "type": "rectangle",
        "width": 120,  # pixels
        "height": 180,  # pixels
        "center": (200, 200),
        "confidence_score": 0.85,
        "mode": "manual_rectangle",
        "contour": np.array([
            [[140, 110]], [[260, 110]], [[260, 290]], [[140, 290]]
        ], dtype=np.int32)
    }
    
    selection_rect = (130, 100, 140, 200)
    mm_per_px_x = 0.4  # Different scaling factors
    mm_per_px_y = 0.6
    
    # Test the full conversion pipeline
    result = classify_and_measure_manual_selection(
        image, selection_rect, shape_result, mm_per_px_x, mm_per_px_y
    )
    
    assert result is not None, "Result should not be None"
    assert result["type"] == "rectangle"
    assert result["detection_method"] == "manual"
    
    # Check axis-specific scaling
    expected_width_mm = 120 * 0.4  # 48 mm
    expected_height_mm = 180 * 0.6  # 108 mm
    
    assert result["width_mm"] == expected_width_mm, f"Expected width {expected_width_mm}, got {result['width_mm']}"
    assert result["height_mm"] == expected_height_mm, f"Expected height {expected_height_mm}, got {result['height_mm']}"
    
    # Validate the result structure
    assert validate_manual_measurement_result(result), "Result should pass validation"
    
    print(f"✓ Integration test passed!")
    print(f"  Width: {result['width_mm']} mm (expected {expected_width_mm})")
    print(f"  Height: {result['height_mm']} mm (expected {expected_height_mm})")
    print(f"  Detection method: {result['detection_method']}")
    
    return result

def test_consistency_with_auto_mode():
    """Test that manual mode produces consistent results with auto mode scaling"""
    print("\nTesting consistency with auto mode scaling...")
    
    # Create a simple rectangle contour
    contour = np.array([
        [[100, 100]], [[200, 100]], [[200, 250]], [[100, 250]]
    ], dtype=np.int32)
    
    # Same scaling factors for both modes
    mm_per_px_x = 0.5
    mm_per_px_y = 0.5
    
    # Test auto mode
    from measure import classify_and_measure
    auto_result = classify_and_measure(contour, mm_per_px_x, mm_per_px_y, "automatic")
    
    # Test manual mode with equivalent shape result
    shape_result = {
        "type": "rectangle",
        "width": 100,   # Same dimensions as the contour
        "height": 150,
        "center": (150, 175),
        "contour": contour
    }
    
    manual_result = classify_and_measure_manual_selection(
        np.zeros((400, 400, 3), dtype=np.uint8),
        (90, 90, 120, 170),
        shape_result,
        mm_per_px_x,
        mm_per_px_y
    )
    
    # Both should produce similar measurements (within reasonable tolerance)
    width_diff = abs(auto_result["width_mm"] - manual_result["width_mm"])
    height_diff = abs(auto_result["height_mm"] - manual_result["height_mm"])
    
    print(f"  Auto mode: {auto_result['width_mm']:.1f}mm x {auto_result['height_mm']:.1f}mm")
    print(f"  Manual mode: {manual_result['width_mm']:.1f}mm x {manual_result['height_mm']:.1f}mm")
    print(f"  Differences: {width_diff:.1f}mm width, {height_diff:.1f}mm height")
    
    # Should be very close (allowing for minor differences in dimension extraction)
    assert width_diff < 5.0, f"Width difference too large: {width_diff}mm"
    assert height_diff < 5.0, f"Height difference too large: {height_diff}mm"
    
    print("✓ Manual and auto modes produce consistent measurements!")

def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("Integration Tests for Rectangle Scaling Fix")
    print("=" * 60)
    
    try:
        test_integration_with_manual_selection()
        test_consistency_with_auto_mode()
        
        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("Rectangle scaling fix integrates correctly with:")
        print("- Manual selection pipeline")
        print("- Measurement validation")
        print("- Auto mode consistency")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = run_integration_tests()
    sys.exit(0 if success else 1)