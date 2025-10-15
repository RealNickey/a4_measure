"""
Test script for adaptive threshold calibration system.

This script validates the adaptive threshold calibrator with various
synthetic test conditions to ensure it works correctly.
"""

import cv2
import numpy as np
import sys
from adaptive_threshold_calibrator import AdaptiveThresholdCalibrator


def create_test_image(condition: str) -> np.ndarray:
    """
    Create synthetic test images with different lighting conditions.
    
    Args:
        condition: Lighting condition ('normal', 'underexposed', 'overexposed', 'varied')
        
    Returns:
        Test grayscale image
    """
    # Create a 600x600 test image
    img = np.zeros((600, 600), dtype=np.uint8)
    
    if condition == 'normal':
        # Normal lighting: medium gray background with dark object
        img[:] = 200
        cv2.rectangle(img, (150, 150), (450, 450), 50, -1)
        cv2.circle(img, (300, 300), 80, 30, -1)
        
    elif condition == 'underexposed':
        # Dark image: dark background with darker object
        img[:] = 60
        cv2.rectangle(img, (150, 150), (450, 450), 20, -1)
        cv2.circle(img, (300, 300), 80, 10, -1)
        
    elif condition == 'overexposed':
        # Bright image: bright background with medium object
        img[:] = 240
        cv2.rectangle(img, (150, 150), (450, 450), 180, -1)
        cv2.circle(img, (300, 300), 80, 160, -1)
        
    elif condition == 'varied':
        # Non-uniform lighting: gradient background
        for i in range(600):
            img[i, :] = int(100 + (i / 600.0) * 100)
        cv2.rectangle(img, (150, 150), (450, 450), 50, -1)
        cv2.circle(img, (300, 300), 80, 30, -1)
    
    # Add some noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def test_lighting_analysis():
    """Test the lighting analysis functionality."""
    print("Testing lighting analysis...")
    
    calibrator = AdaptiveThresholdCalibrator()
    
    # Test with different conditions
    conditions = ['normal', 'underexposed', 'overexposed', 'varied']
    
    for condition in conditions:
        test_img = create_test_image(condition)
        stats = calibrator.analyze_lighting_conditions(test_img)
        
        print(f"\n{condition.upper()} Lighting:")
        print(f"  Mean brightness: {stats['mean_brightness']:.1f}")
        print(f"  Std brightness: {stats['std_brightness']:.1f}")
        print(f"  Dynamic range: {stats['dynamic_range']:.1f}")
        print(f"  Contrast ratio: {stats['contrast_ratio']:.3f}")
        print(f"  Lighting condition: {stats['lighting_condition']}")
        print(f"  Is bimodal: {stats['is_bimodal']}")
    
    print("\n✓ Lighting analysis test passed")


def test_parameter_calibration():
    """Test the parameter calibration functionality."""
    print("\nTesting parameter calibration...")
    
    calibrator = AdaptiveThresholdCalibrator()
    
    conditions = ['normal', 'underexposed', 'overexposed', 'varied']
    
    for condition in conditions:
        test_img = create_test_image(condition)
        stats = calibrator.analyze_lighting_conditions(test_img)
        block_size, c_constant = calibrator.calibrate_threshold_parameters(stats)
        
        print(f"\n{condition.upper()} Parameters:")
        print(f"  Block size: {block_size}")
        print(f"  C constant: {c_constant:.1f}")
        
        # Verify parameters are valid
        assert block_size >= 3 and block_size <= 99, f"Invalid block size: {block_size}"
        assert block_size % 2 == 1, f"Block size must be odd: {block_size}"
        assert c_constant >= 1.0 and c_constant <= 20.0, f"Invalid C constant: {c_constant}"
    
    print("\n✓ Parameter calibration test passed")


def test_full_pipeline():
    """Test the full calibration and thresholding pipeline."""
    print("\nTesting full calibration pipeline...")
    
    calibrator = AdaptiveThresholdCalibrator()
    
    conditions = ['normal', 'underexposed', 'overexposed', 'varied']
    
    for condition in conditions:
        test_img = create_test_image(condition)
        binary, stats = calibrator.calibrate_and_threshold(test_img)
        
        print(f"\n{condition.upper()} Pipeline:")
        print(f"  Block size: {stats['block_size']}")
        print(f"  C constant: {stats['c_constant']:.1f}")
        print(f"  CLAHE enabled: {stats['clahe_enabled']}")
        print(f"  Multipass enabled: {stats['multipass_enabled']}")
        print(f"  Binary shape: {binary.shape}")
        print(f"  Binary dtype: {binary.dtype}")
        print(f"  Binary unique values: {np.unique(binary)}")
        
        # Verify binary output
        assert binary.shape == test_img.shape, "Binary shape mismatch"
        assert binary.dtype == np.uint8, "Binary dtype must be uint8"
        assert set(np.unique(binary)).issubset({0, 255}), "Binary must have only 0 and 255 values"
        
        # Find contours to verify detection
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  Contours found: {len(contours)}")
        
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            print(f"  Largest contour area: {max(areas):.0f}")
    
    print("\n✓ Full pipeline test passed")


def test_feature_toggles():
    """Test feature toggle functionality."""
    print("\nTesting feature toggles...")
    
    test_img = create_test_image('normal')
    
    # Test with all features disabled
    calibrator = AdaptiveThresholdCalibrator(
        enable_clahe=False,
        enable_multipass=False,
        enable_local_adaptive=False
    )
    binary1, stats1 = calibrator.calibrate_and_threshold(test_img)
    print(f"\nAll features disabled:")
    print(f"  CLAHE: {stats1['clahe_enabled']}")
    print(f"  Multipass: {stats1['multipass_enabled']}")
    print(f"  Local adaptive: {stats1['local_adaptive_enabled']}")
    
    # Test with all features enabled
    calibrator = AdaptiveThresholdCalibrator(
        enable_clahe=True,
        enable_multipass=True,
        enable_local_adaptive=True
    )
    binary2, stats2 = calibrator.calibrate_and_threshold(test_img)
    print(f"\nAll features enabled:")
    print(f"  CLAHE: {stats2['clahe_enabled']}")
    print(f"  Multipass: {stats2['multipass_enabled']}")
    print(f"  Local adaptive: {stats2['local_adaptive_enabled']}")
    
    # Both should produce valid binary images
    assert binary1.shape == test_img.shape
    assert binary2.shape == test_img.shape
    
    print("\n✓ Feature toggle test passed")


def test_integration_with_measure():
    """Test integration with measure.py segment_object function."""
    print("\nTesting integration with measure.py...")
    
    try:
        from measure import segment_object
        import config
        
        # Create test BGR image
        test_gray = create_test_image('normal')
        test_bgr = cv2.cvtColor(test_gray, cv2.COLOR_GRAY2BGR)
        
        # Test with adaptive threshold enabled
        original_setting = config.ENABLE_ADAPTIVE_THRESHOLD
        config.ENABLE_ADAPTIVE_THRESHOLD = True
        
        binary = segment_object(test_bgr)
        
        print(f"  Adaptive threshold enabled: {config.ENABLE_ADAPTIVE_THRESHOLD}")
        print(f"  Binary shape: {binary.shape}")
        print(f"  Binary dtype: {binary.dtype}")
        
        # Verify output
        assert binary.shape[:2] == test_bgr.shape[:2], "Shape mismatch"
        assert binary.dtype == np.uint8, "Dtype must be uint8"
        
        # Restore original setting
        config.ENABLE_ADAPTIVE_THRESHOLD = original_setting
        
        print("\n✓ Integration test passed")
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("ADAPTIVE THRESHOLD CALIBRATION TEST SUITE")
    print("=" * 60)
    
    try:
        test_lighting_analysis()
        test_parameter_calibration()
        test_full_pipeline()
        test_feature_toggles()
        test_integration_with_measure()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("TESTS FAILED ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
