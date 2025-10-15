"""
Visual demonstration of adaptive threshold calibration.

This script creates side-by-side comparisons of standard vs adaptive thresholding
under various lighting conditions to demonstrate the improvements.
"""

import cv2
import numpy as np
import sys
from adaptive_threshold_calibrator import AdaptiveThresholdCalibrator
from config import BINARY_BLOCK_SIZE, BINARY_C


def create_demo_image(condition: str, size: tuple = (800, 800)) -> np.ndarray:
    """
    Create demo images with various lighting conditions.
    
    Args:
        condition: Lighting condition
        size: Image size (height, width)
        
    Returns:
        Demo grayscale image
    """
    h, w = size
    img = np.zeros((h, w), dtype=np.uint8)
    
    if condition == 'normal':
        # Normal lighting with clear contrast
        img[:] = 200
        # Large rectangle
        cv2.rectangle(img, (100, 100), (400, 400), 50, -1)
        # Circle
        cv2.circle(img, (600, 250), 120, 30, -1)
        # Small rectangle
        cv2.rectangle(img, (500, 500), (700, 650), 50, -1)
        
    elif condition == 'underexposed':
        # Dark/underexposed scene
        img[:] = 50
        cv2.rectangle(img, (100, 100), (400, 400), 15, -1)
        cv2.circle(img, (600, 250), 120, 5, -1)
        cv2.rectangle(img, (500, 500), (700, 650), 15, -1)
        
    elif condition == 'overexposed':
        # Bright/overexposed scene
        img[:] = 245
        cv2.rectangle(img, (100, 100), (400, 400), 190, -1)
        cv2.circle(img, (600, 250), 120, 180, -1)
        cv2.rectangle(img, (500, 500), (700, 650), 190, -1)
        
    elif condition == 'gradient':
        # Non-uniform lighting with gradient
        for i in range(h):
            img[i, :] = int(80 + (i / h) * 140)
        cv2.rectangle(img, (100, 100), (400, 400), 40, -1)
        cv2.circle(img, (600, 250), 120, 25, -1)
        cv2.rectangle(img, (500, 500), (700, 650), 40, -1)
        
    elif condition == 'shadow':
        # Scene with shadows (left bright, right dark)
        for j in range(w):
            intensity = int(220 - (j / w) * 150)
            img[:, j] = intensity
        cv2.rectangle(img, (100, 100), (400, 400), 50, -1)
        cv2.circle(img, (600, 250), 120, 30, -1)
        cv2.rectangle(img, (500, 500), (700, 650), 50, -1)
    
    # Add realistic noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def apply_standard_threshold(gray: np.ndarray) -> np.ndarray:
    """Apply standard fixed-parameter adaptive threshold."""
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, BINARY_BLOCK_SIZE, BINARY_C
    )
    # Morph cleanup
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return bw


def add_text_overlay(img: np.ndarray, text: str, position: str = 'top') -> np.ndarray:
    """Add text overlay to image."""
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if position == 'top':
        y_pos = 30
    else:
        y_pos = img.shape[0] - 20
    
    cv2.putText(img_color, text, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_color


def create_comparison_image(condition: str) -> np.ndarray:
    """
    Create side-by-side comparison of standard vs adaptive thresholding.
    
    Args:
        condition: Lighting condition
        
    Returns:
        Comparison image (BGR)
    """
    # Create test image
    test_img = create_demo_image(condition, size=(800, 800))
    
    # Apply standard threshold
    standard_binary = apply_standard_threshold(test_img)
    
    # Apply adaptive threshold
    calibrator = AdaptiveThresholdCalibrator()
    adaptive_binary, stats = calibrator.calibrate_and_threshold(test_img)
    
    # Add labels
    original = add_text_overlay(test_img, "Original", 'top')
    standard = add_text_overlay(standard_binary, 
                               f"Standard (B={BINARY_BLOCK_SIZE}, C={BINARY_C})", 'top')
    adaptive = add_text_overlay(adaptive_binary,
                               f"Adaptive (B={stats['block_size']}, C={stats['c_constant']:.1f})", 'top')
    
    # Add lighting info to original
    lighting_info = stats['lighting_stats']
    info_text = f"Lighting: {lighting_info['lighting_condition']}, Brightness: {lighting_info['mean_brightness']:.0f}"
    cv2.putText(original, info_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Stack horizontally
    comparison = np.hstack([original, standard, adaptive])
    
    # Add title bar
    title_bar = np.zeros((80, comparison.shape[1], 3), dtype=np.uint8)
    title_text = f"Condition: {condition.upper()}"
    cv2.putText(title_bar, title_text, (comparison.shape[1]//2 - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    final = np.vstack([title_bar, comparison])
    
    return final


def run_demo():
    """Run the visual demonstration."""
    print("=" * 70)
    print("ADAPTIVE THRESHOLD CALIBRATION - VISUAL DEMONSTRATION")
    print("=" * 70)
    print("\nGenerating comparison images for various lighting conditions...")
    
    conditions = [
        'normal',
        'underexposed', 
        'overexposed',
        'gradient',
        'shadow'
    ]
    
    # Create comparison images
    all_comparisons = []
    
    for condition in conditions:
        print(f"\nProcessing: {condition}")
        comparison = create_comparison_image(condition)
        all_comparisons.append(comparison)
        
        # Save individual comparison
        filename = f"demo_adaptive_{condition}.png"
        cv2.imwrite(filename, comparison)
        print(f"  Saved: {filename}")
    
    # Create final stacked comparison
    print("\nCreating final stacked comparison...")
    
    # Stack all comparisons vertically
    final_demo = np.vstack(all_comparisons)
    
    # Save final image
    cv2.imwrite("demo_adaptive_threshold_comparison.png", final_demo)
    print("\nSaved: demo_adaptive_threshold_comparison.png")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - demo_adaptive_normal.png")
    print("  - demo_adaptive_underexposed.png")
    print("  - demo_adaptive_overexposed.png")
    print("  - demo_adaptive_gradient.png")
    print("  - demo_adaptive_shadow.png")
    print("  - demo_adaptive_threshold_comparison.png (all conditions)")
    print("\nThe adaptive threshold calibration automatically adjusts parameters")
    print("based on lighting conditions, resulting in more consistent detection")
    print("across various scenarios.")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
