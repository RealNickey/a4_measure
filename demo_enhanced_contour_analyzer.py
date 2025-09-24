"""
Demonstration script for Enhanced Contour Analyzer

This script demonstrates the capabilities of the EnhancedContourAnalyzer
and shows the improvements over standard contour detection methods.
"""

import cv2
import numpy as np
from enhanced_contour_analyzer import EnhancedContourAnalyzer


def create_demo_image():
    """Create a demonstration image with challenging conditions."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Create gradient background (varying lighting)
    for y in range(500):
        for x in range(500):
            intensity = int(60 + 120 * (x + y) / 1000)
            img[y, x] = [intensity, intensity, intensity]
    
    # Add shapes with different contrasts
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)        # High contrast rectangle
    cv2.circle(img, (350, 100), 60, (180, 180, 180), -1)          # Medium contrast circle
    cv2.rectangle(img, (200, 300), (350, 400), (80, 80, 80), -1)  # Low contrast rectangle
    
    # Add nested shapes
    cv2.rectangle(img, (100, 250), (200, 350), (20, 20, 20), -1)  # Outer rectangle
    cv2.circle(img, (150, 300), 30, (200, 200, 200), -1)          # Inner circle
    
    return img


def demonstrate_enhanced_analysis():
    """Demonstrate the enhanced contour analysis capabilities."""
    print("Enhanced Contour Analyzer Demonstration")
    print("=" * 50)
    
    # Create analyzer
    analyzer = EnhancedContourAnalyzer()
    
    # Display configuration
    stats = analyzer.get_processing_stats()
    print(f"Configuration:")
    print(f"  Gaussian Block Size: {stats['gaussian_block_size']}")
    print(f"  Gaussian C: {stats['gaussian_c']}")
    print(f"  CUDA Enabled: {stats['cuda_enabled']}")
    print()
    
    # Create demo image
    demo_img = create_demo_image()
    print(f"Created demo image: {demo_img.shape}")
    
    # Analyze entire image
    print("\n1. Full Image Analysis:")
    contours = analyzer.analyze_region(demo_img)
    print(f"   Detected {len(contours)} contours")
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        print(f"   Contour {i+1}: Area={area:.1f}, Perimeter={perimeter:.1f}")
    
    # Analyze specific ROI
    print("\n2. ROI Analysis (top-left region):")
    roi = (25, 25, 200, 200)
    roi_contours = analyzer.analyze_region(demo_img, roi)
    print(f"   Detected {len(roi_contours)} contours in ROI")
    
    for i, contour in enumerate(roi_contours):
        area = cv2.contourArea(contour)
        print(f"   ROI Contour {i+1}: Area={area:.1f}")
    
    # Compare with standard thresholding
    print("\n3. Comparison with Standard Thresholding:")
    gray = cv2.cvtColor(demo_img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced thresholding
    enhanced_binary = analyzer.apply_adaptive_gaussian_threshold(gray)
    enhanced_contours, _ = cv2.findContours(enhanced_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    enhanced_filtered = [c for c in enhanced_contours if cv2.contourArea(c) > 100]
    
    # Standard thresholding
    _, standard_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    standard_contours, _ = cv2.findContours(standard_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    standard_filtered = [c for c in standard_contours if cv2.contourArea(c) > 100]
    
    print(f"   Enhanced method: {len(enhanced_filtered)} meaningful contours")
    print(f"   Standard method: {len(standard_filtered)} meaningful contours")
    
    # Test different configurations
    print("\n4. Configuration Impact:")
    analyzer_fine = EnhancedContourAnalyzer(gaussian_block_size=15, gaussian_c=3.0)
    analyzer_coarse = EnhancedContourAnalyzer(gaussian_block_size=51, gaussian_c=12.0)
    
    fine_contours = analyzer_fine.analyze_region(demo_img)
    coarse_contours = analyzer_coarse.analyze_region(demo_img)
    
    print(f"   Fine settings: {len(fine_contours)} contours")
    print(f"   Coarse settings: {len(coarse_contours)} contours")
    
    print("\nDemonstration completed successfully!")
    return demo_img, contours


if __name__ == "__main__":
    try:
        demo_img, contours = demonstrate_enhanced_analysis()
        print(f"\nDemo image shape: {demo_img.shape}")
        print(f"Total contours detected: {len(contours)}")
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()