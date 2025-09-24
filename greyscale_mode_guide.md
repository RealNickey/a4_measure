# Greyscale Processing in Manual Shape Selection System

## Overview

The manual shape selection system uses greyscale processing internally for improved shape detection and analysis. While there isn't a user-selectable "greyscale mode", the system automatically converts color images to greyscale for various processing steps.

## How Greyscale Processing Works

### 1. Automatic Greyscale Conversion

The system automatically converts color images to greyscale in several components:

#### Enhanced Contour Analyzer
```python
# In enhanced_contour_analyzer.py
def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale with CUDA acceleration if available."""
    if len(image.shape) == 2:
        return image  # Already grayscale
    
    if self.use_cuda:
        # GPU-accelerated conversion
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        return gpu_gray.download()
    
    # CPU conversion
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

#### Shape Snapping Engine
```python
# In shape_snapping_engine.py
# Convert to grayscale for better edge detection
if len(roi_image.shape) == 3:
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
```

### 2. When Greyscale Processing is Used

Greyscale conversion happens automatically in these scenarios:

1. **Contour Detection**: For finding shape boundaries
2. **Adaptive Thresholding**: For better edge detection in varying lighting
3. **Shape Analysis**: For calculating shape properties
4. **Manual Selection**: During shape snapping operations

### 3. Benefits of Greyscale Processing

- **Better Edge Detection**: Removes color noise for cleaner edges
- **Improved Performance**: Faster processing with single-channel images
- **Lighting Robustness**: Less sensitive to color variations
- **Enhanced Accuracy**: More consistent shape detection

## Available Selection Modes

The system has three main selection modes (not greyscale-specific):

### 1. AUTO Mode
- Automatic shape detection
- Uses greyscale processing internally
- **Key**: Press `M` to cycle modes

### 2. MANUAL RECTANGLE Mode
- Manual rectangle selection
- Greyscale processing for shape snapping
- **Key**: Press `M` to cycle to this mode

### 3. MANUAL CIRCLE Mode
- Manual circle selection
- Greyscale processing for shape snapping
- **Key**: Press `M` to cycle to this mode

## Using the System

### Basic Usage

1. **Start the Application**:
   ```bash
   python main.py
   ```

2. **Mode Switching**:
   - Press `M` to cycle between modes: AUTO → MANUAL RECT → MANUAL CIRCLE → AUTO
   - Current mode is displayed in the top-left corner

3. **Manual Selection**:
   - In manual modes, click and drag to select an area
   - The system uses greyscale processing to snap to shapes
   - Release to complete selection

### Advanced Configuration

You can configure greyscale processing parameters in `config.py`:

```python
# Enhanced contour analysis parameters
ENHANCED_GAUSSIAN_BLOCK_SIZE = 31  # Block size for adaptive thresholding
ENHANCED_GAUSSIAN_C = 7.0          # Constant for adaptive thresholding
ENHANCED_MIN_CONTOUR_AREA = 100    # Minimum contour area

# Shape detection thresholds
MIN_CIRCULARITY_THRESHOLD = 0.6    # Circle detection sensitivity
MIN_RECTANGULARITY_THRESHOLD = 0.7 # Rectangle detection sensitivity
```

## Troubleshooting Greyscale Processing

### Poor Shape Detection

If shapes aren't being detected well:

1. **Adjust Lighting**: Ensure good contrast between shapes and background
2. **Modify Thresholds**: Edit `ENHANCED_GAUSSIAN_C` in config.py
3. **Change Block Size**: Adjust `ENHANCED_GAUSSIAN_BLOCK_SIZE` for different lighting conditions

### Performance Issues

For better performance:

1. **Enable CUDA**: Set `USE_CUDA_IF_AVAILABLE = True` in config.py
2. **Reduce Image Size**: Use lower resolution if detection is slow
3. **Optimize Camera Settings**: The system automatically optimizes camera saturation for better greyscale conversion

## Testing Greyscale Processing

### Run Comprehensive Tests

```bash
# Test all greyscale processing components
python test_enhanced_contour_analyzer.py

# Test shape snapping with greyscale processing
python test_shape_snapping_engine.py

# Run comprehensive manual selection tests
python run_comprehensive_manual_selection_tests.py
```

### Interactive Testing

```bash
# Interactive testing with various scenarios
python interactive_manual_selection_test.py

# Edge case testing including low contrast scenarios
python demo_edge_case_testing.py
```

## Code Examples

### Custom Greyscale Processing

If you need to customize greyscale processing:

```python
from enhanced_contour_analyzer import EnhancedContourAnalyzer

# Initialize with custom parameters
analyzer = EnhancedContourAnalyzer(
    gaussian_block_size=25,  # Smaller for fine details
    gaussian_c=5.0          # Lower for high contrast
)

# Process image
contours = analyzer.analyze_region(image, roi=(x, y, w, h))
```

### Manual Shape Detection with Greyscale

```python
from shape_snapping_engine import ShapeSnappingEngine
from selection_mode import SelectionMode

# Initialize components
snap_engine = ShapeSnappingEngine(analyzer)

# Detect circle using greyscale processing
result = snap_engine.snap_to_shape(
    image, 
    selection_rect=(x, y, w, h), 
    mode=SelectionMode.MANUAL_CIRCLE
)
```

## Performance Monitoring

The system provides performance statistics for greyscale processing:

```python
# Get processing statistics
stats = analyzer.get_processing_stats()
print(f"CUDA enabled: {stats['cuda_enabled']}")
print(f"Gaussian block size: {stats['gaussian_block_size']}")
```

## Summary

While there's no explicit "greyscale mode" button, the system extensively uses greyscale processing internally for optimal shape detection. The three selection modes (AUTO, MANUAL RECT, MANUAL CIRCLE) all benefit from sophisticated greyscale processing algorithms that adapt to different lighting conditions and provide robust shape detection capabilities.

To effectively use the system:
1. Use `M` key to cycle between selection modes
2. Ensure good lighting and contrast
3. Adjust configuration parameters if needed
4. Run tests to validate performance in your specific use case