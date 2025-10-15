# Adaptive Threshold Calibration System

## Overview

The Adaptive Threshold Calibration system is an intelligent image processing feature that dynamically adjusts detection parameters based on the specific characteristics of each input image. This significantly improves measurement accuracy across diverse lighting conditions, from underexposed dark scenes to overexposed bright environments.

## Key Features

### 1. Dynamic Lighting Analysis

The system analyzes the overall brightness and contrast levels of input images before processing:

- **Histogram Statistics**: Calculates histogram distribution to understand pixel intensity patterns
- **Brightness Detection**: Identifies whether the image is underexposed, overexposed, or normally lit
- **Contrast Analysis**: Measures local and global contrast ratios
- **Bimodal Detection**: Determines if the image has clear foreground/background separation

### 2. Multi-Pass Threshold Strategy

Implements progressive threshold refinement:

- **Initial Pass**: Performs preliminary detection with calibrated parameters
- **Quality Assessment**: Analyzes the quality of initial detection results
- **Adaptive Refinement**: Adjusts thresholds based on detection quality
- **Second Pass**: Executes refined detection for improved accuracy

### 3. Local Adaptive Thresholding

Handles non-uniform lighting conditions:

- **Region-Based Processing**: Uses Gaussian-weighted neighborhood analysis
- **Independent Calibration**: Calculates optimal thresholds for each local region
- **Automatic Adjustment**: Adapts to lighting variations across the image

### 4. Contrast Enhancement Pre-Processing

Enhances image quality before thresholding:

- **CLAHE**: Applies Contrast Limited Adaptive Histogram Equalization
- **Selective Enhancement**: Only applies when beneficial for detection
- **Edge Preservation**: Maintains sharp edges for accurate contour detection

### 5. Noise Reduction Integration

Filters noise while preserving important features:

- **Adaptive Filtering**: Adjusts filtering strength based on detected noise levels
- **Morphological Operations**: Uses opening and closing to clean binary results
- **Context-Aware**: More aggressive filtering for challenging lighting conditions

## Configuration

The system can be configured through `config.py`:

```python
# Enable/disable adaptive threshold calibration
ENABLE_ADAPTIVE_THRESHOLD = True

# Feature toggles
ADAPTIVE_THRESHOLD_ENABLE_CLAHE = True      # Contrast enhancement
ADAPTIVE_THRESHOLD_ENABLE_MULTIPASS = True  # Multi-pass refinement
ADAPTIVE_THRESHOLD_ENABLE_LOCAL = True      # Local adaptive processing

# CLAHE parameters
ADAPTIVE_THRESHOLD_CLAHE_CLIP_LIMIT = 2.0   # Clip limit for contrast
ADAPTIVE_THRESHOLD_CLAHE_TILE_SIZE = 8      # Tile grid size
```

## Usage

### Basic Usage

The adaptive threshold calibration is automatically integrated into the measurement pipeline:

```python
from measure import segment_object

# segment_object now uses adaptive thresholding automatically
# if ENABLE_ADAPTIVE_THRESHOLD is True in config
binary_mask = segment_object(a4_bgr_image)
```

### Advanced Usage

For custom processing, you can use the calibrator directly:

```python
from adaptive_threshold_calibrator import AdaptiveThresholdCalibrator

# Create calibrator instance
calibrator = AdaptiveThresholdCalibrator(
    initial_block_size=31,
    initial_c=7.0,
    enable_clahe=True,
    enable_multipass=True,
    enable_local_adaptive=True
)

# Analyze lighting conditions
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
lighting_stats = calibrator.analyze_lighting_conditions(gray_image)

print(f"Mean brightness: {lighting_stats['mean_brightness']}")
print(f"Lighting condition: {lighting_stats['lighting_condition']}")

# Apply calibrated threshold
binary_image, stats = calibrator.calibrate_and_threshold(bgr_image)

print(f"Calibrated block size: {stats['block_size']}")
print(f"Calibrated C constant: {stats['c_constant']}")
```

## How It Works

### 1. Lighting Analysis Phase

```
Input Image → Grayscale Conversion → Histogram Calculation
                                   ↓
                          Calculate Statistics:
                          - Mean brightness
                          - Standard deviation
                          - Dynamic range (P10-P90)
                          - Contrast ratio
                          - Bimodal detection
                                   ↓
                          Classify Lighting:
                          - Underexposed (mean < 80)
                          - Overexposed (mean > 175)
                          - Normal (80-175)
```

### 2. Parameter Calibration Phase

```
Lighting Statistics → Analyze Characteristics → Adjust Block Size:
                                               - High contrast: smaller (11-21)
                                               - Low contrast: larger (41-51)
                                               - Default: 31
                                               ↓
                                          Adjust C Constant:
                                               - Underexposed: reduce (2-4)
                                               - Overexposed: increase (10-12)
                                               - Default: 7
```

### 3. Processing Phase

```
Input Image → CLAHE Enhancement (optional)
                     ↓
         Adaptive Gaussian Threshold
         (with calibrated parameters)
                     ↓
         Multi-pass Refinement (optional)
                     ↓
         Morphological Noise Reduction
                     ↓
         Binary Output
```

## Parameter Ranges

### Block Size
- **Range**: 3 to 99 (must be odd)
- **Default**: 31
- **Underexposed**: Typically 21-31
- **Normal**: 21-41
- **Overexposed**: 31-51

### C Constant
- **Range**: 1.0 to 20.0
- **Default**: 7.0
- **Underexposed**: 2.0-5.0 (more sensitive)
- **Normal**: 5.0-9.0
- **Overexposed**: 8.0-12.0 (less sensitive)

## Performance Characteristics

### Computational Cost

- **Lighting Analysis**: ~5-10ms per image
- **CLAHE Enhancement**: ~10-20ms per image
- **Adaptive Threshold**: ~15-25ms per image
- **Multi-pass Refinement**: +5-10ms per image
- **Total Overhead**: ~35-65ms per image (still real-time at 15-30 FPS)

### Memory Usage

- Minimal additional memory: ~2-3 MB for calibrator instance
- No significant increase in per-frame memory usage

## Testing

### Run Tests

```bash
python3 test_adaptive_threshold.py
```

This will run comprehensive tests covering:
- Lighting analysis accuracy
- Parameter calibration correctness
- Full pipeline functionality
- Feature toggle behavior
- Integration with measure.py

### Visual Demo

```bash
python3 demo_adaptive_threshold.py
```

This generates comparison images showing:
- Standard vs adaptive thresholding
- Performance across different lighting conditions
- Parameter adjustments for each scenario

## Benefits

### Improved Accuracy

- **Consistent Detection**: Works reliably across varying lighting conditions
- **Better Edge Detection**: More accurate edge identification leads to precise measurements
- **Reduced False Positives**: Better distinguishes actual objects from background clutter

### Robustness

- **Handles Shadows**: Adapts to non-uniform lighting with shadows
- **Deals with Reflections**: Reduces impact of reflections on detection
- **Works with Texture**: Better handling of textured backgrounds

### Consistency

- **Predictable Results**: More consistent measurements across different image qualities
- **Camera Independence**: Less dependent on specific camera settings
- **Environmental Tolerance**: Works in various lighting environments

## Troubleshooting

### Low Detection Accuracy

If detection accuracy is still low:

1. Check if adaptive threshold is enabled:
   ```python
   from config import ENABLE_ADAPTIVE_THRESHOLD
   print(f"Adaptive threshold: {ENABLE_ADAPTIVE_THRESHOLD}")
   ```

2. Review calibration stats:
   ```python
   from measure import _get_adaptive_calibrator
   calibrator = _get_adaptive_calibrator()
   stats = calibrator.get_last_calibration_stats()
   print(stats)
   ```

3. Try adjusting feature flags in config.py

### Performance Issues

If processing is too slow:

1. Disable multipass refinement:
   ```python
   ADAPTIVE_THRESHOLD_ENABLE_MULTIPASS = False
   ```

2. Disable CLAHE for simple scenes:
   ```python
   ADAPTIVE_THRESHOLD_ENABLE_CLAHE = False
   ```

### Fallback Behavior

The system automatically falls back to standard thresholding if:
- Adaptive calibrator initialization fails
- An error occurs during adaptive processing
- `ENABLE_ADAPTIVE_THRESHOLD = False` in config

## Future Enhancements

Potential future improvements:

1. **Machine Learning Integration**: Train a model to predict optimal parameters
2. **Per-Region Calibration**: Different parameters for different image regions
3. **Temporal Smoothing**: Smooth parameter changes across video frames
4. **Advanced Edge Detection**: Integration with specialized edge detection algorithms
5. **GPU Acceleration**: CUDA-based processing for real-time performance

## References

- [Adaptive Thresholding in OpenCV](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [CLAHE Algorithm](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- [Morphological Operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)

## License

MIT License - See main project LICENSE file for details.
