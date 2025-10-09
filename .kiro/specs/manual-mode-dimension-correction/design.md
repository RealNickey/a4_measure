# Design Document

## Overview

This design addresses the dimension calculation discrepancy between Auto Mode and Manual Mode in the A4 object dimension scanner. The core issue is that Manual Mode's shape snapping engine calculates dimensions in pixel space but doesn't properly convert them to millimeters using the same A4 calibration scaling factors as Auto Mode.

The solution involves modifying the measurement conversion layer in `measure.py` to ensure Manual Mode receives and properly applies the same `mm_per_px_x` and `mm_per_px_y` scaling factors that Auto Mode uses. This approach preserves Manual Mode's superior shape detection capabilities while achieving measurement accuracy parity with Auto Mode.

## Architecture

### Current Architecture Analysis

**Auto Mode Flow:**
1. `segment_object()` → `all_inner_contours()` → `classify_and_measure()`
2. `classify_and_measure()` receives `mm_per_px_x`, `mm_per_px_y` parameters
3. Pixel measurements are converted to millimeters using scaling factors
4. Results include accurate millimeter dimensions

**Manual Mode Flow (Current - Problematic):**
1. User selection → `ShapeSnappingEngine.snap_to_shape()`
2. Shape snapping returns pixel-based dimensions
3. `classify_and_measure_manual_selection()` converts to measurement format
4. **Missing:** Proper application of `mm_per_px_x`, `mm_per_px_y` scaling factors

### Proposed Architecture

**Manual Mode Flow (Corrected):**
1. User selection → `ShapeSnappingEngine.snap_to_shape()` (unchanged)
2. Shape snapping returns pixel-based dimensions (unchanged)
3. `process_manual_selection()` receives `mm_per_px_x`, `mm_per_px_y` parameters
4. `classify_and_measure_manual_selection()` applies proper scaling conversion
5. Results include accurate millimeter dimensions matching Auto Mode

## Components and Interfaces

### Modified Components

#### 1. `measure.py` - Manual Selection Processing Functions

**Function: `process_manual_selection()`**
- **Current Signature:** `process_manual_selection(image, selection_rect, mode, mm_per_px_x, mm_per_px_y)`
- **Modification:** Ensure scaling factors are properly passed to conversion functions
- **Responsibility:** Coordinate manual selection workflow with proper scaling

**Function: `classify_and_measure_manual_selection()`**
- **Current Signature:** `classify_and_measure_manual_selection(image, selection_rect, shape_result, mm_per_px_x, mm_per_px_y)`
- **Modification:** Fix dimension conversion logic to properly apply scaling factors
- **Responsibility:** Convert shape snapping results to measurement format with accurate scaling

**Functions: `_convert_manual_circle_to_measurement()` and `_convert_manual_rectangle_to_measurement()`**
- **Modification:** Implement proper pixel-to-millimeter conversion
- **Responsibility:** Apply scaling factors correctly for each shape type

#### 2. `main.py` - Integration Point

**Manual Selection Integration:**
- **Current:** Manual selection workflow exists but may not pass scaling factors
- **Modification:** Ensure `mm_per_px_x`, `mm_per_px_y` are passed to manual processing functions
- **Responsibility:** Provide consistent scaling factors to both Auto and Manual modes

### Unchanged Components

#### 1. `shape_snapping_engine.py`
- **Preservation:** All shape detection and scoring algorithms remain unchanged
- **Rationale:** Shape detection quality is already superior; only dimension conversion needs fixing

#### 2. `manual_selection_engine.py`
- **Preservation:** User interaction and selection area tracking remain unchanged
- **Rationale:** User experience and selection workflow are working correctly

#### 3. Enhanced Contour Analyzer
- **Preservation:** Contour analysis and shape identification remain unchanged
- **Rationale:** Shape detection accuracy is the strength of Manual Mode

## Data Models

### Shape Result Data Structure

**Current Shape Snapping Result:**
```python
{
    "type": "circle" | "rectangle",
    "center": (x, y),  # pixels
    "dimensions": (radius, radius) | (width, height),  # pixels
    "area": float,  # pixels²
    "contour": np.ndarray,
    "confidence_score": float
}
```

**Target Measurement Result (Consistent with Auto Mode):**
```python
{
    "type": "circle" | "rectangle",
    "detection_method": "manual",
    "hit_contour": np.ndarray,
    "area_px": float,
    
    # Circle-specific (millimeters)
    "diameter_mm": float,
    "center": (x, y),  # pixels (for rendering)
    "radius_px": float,  # pixels (for rendering)
    
    # Rectangle-specific (millimeters)
    "width_mm": float,
    "height_mm": float,
    "box": np.ndarray  # pixels (for rendering)
}
```

### Scaling Factor Application

**Circle Dimension Conversion:**
```python
# Current (incorrect): Uses arbitrary or missing scaling
diameter_mm = diameter_px * some_scaling_factor

# Corrected: Uses A4 calibration scaling
diameter_mm = diameter_px * mm_per_px_x  # Assumes isotropic scaling
```

**Rectangle Dimension Conversion:**
```python
# Current (incorrect): May not apply proper scaling
width_mm = width_px * scaling_factor
height_mm = height_px * scaling_factor

# Corrected: Uses A4 calibration scaling with axis-specific factors
width_mm = width_px * mm_per_px_x
height_mm = height_px * mm_per_px_y
```

## Error Handling

### Scaling Factor Validation

**Input Validation:**
- Verify `mm_per_px_x` and `mm_per_px_y` are positive values
- Ensure scaling factors are within reasonable ranges (e.g., 0.1 to 10.0 mm/px)
- Handle cases where scaling factors are None or invalid

**Fallback Strategies:**
- If scaling factors are invalid, log error and return None result
- Provide clear error messages indicating calibration issues
- Maintain existing error handling for shape detection failures

### Measurement Consistency Validation

**Cross-Mode Validation:**
- Add optional validation function to compare Auto vs Manual measurements
- Implement tolerance checking (±2mm or ±2% whichever is larger)
- Log warnings when measurements differ significantly

## Testing Strategy

### Unit Tests

**Dimension Conversion Tests:**
```python
def test_manual_circle_dimension_conversion():
    # Test circle pixel-to-mm conversion with known scaling factors
    
def test_manual_rectangle_dimension_conversion():
    # Test rectangle pixel-to-mm conversion with axis-specific scaling
    
def test_scaling_factor_validation():
    # Test handling of invalid scaling factors
```

**Integration Tests:**
```python
def test_manual_vs_auto_measurement_consistency():
    # Compare measurements of same object using both modes
    
def test_end_to_end_manual_selection_workflow():
    # Test complete manual selection with proper dimension output
```

### Validation Tests

**Measurement Accuracy Tests:**
- Use objects with known dimensions (e.g., coins, standard shapes)
- Compare Manual Mode results against Auto Mode results
- Verify measurements are within acceptable tolerance

**Regression Tests:**
- Ensure Auto Mode measurements remain unchanged
- Verify Manual Mode shape detection quality is preserved
- Confirm user interaction workflow remains intact

## Implementation Approach

### Phase 1: Core Dimension Conversion Fix

1. **Modify `_convert_manual_circle_to_measurement()`:**
   - Apply `mm_per_px_x` to diameter calculation
   - Ensure radius_px remains in pixels for rendering

2. **Modify `_convert_manual_rectangle_to_measurement()`:**
   - Apply `mm_per_px_x` to width calculation
   - Apply `mm_per_px_y` to height calculation
   - Ensure box coordinates remain in pixels for rendering

3. **Update `process_manual_selection()`:**
   - Verify scaling factors are properly passed through
   - Add input validation for scaling factors

### Phase 2: Integration and Validation

1. **Update `main.py` integration:**
   - Ensure manual selection receives same scaling factors as auto detection
   - Verify measurement results are properly integrated

2. **Add measurement validation:**
   - Implement optional cross-mode comparison
   - Add logging for measurement consistency checks

### Phase 3: Testing and Refinement

1. **Comprehensive testing:**
   - Unit tests for conversion functions
   - Integration tests for end-to-end workflow
   - Accuracy validation with known objects

2. **Performance verification:**
   - Ensure changes don't impact Manual Mode responsiveness
   - Verify Auto Mode performance remains unchanged

## Risk Mitigation

### Backward Compatibility

**Risk:** Changes might break existing workflows
**Mitigation:** 
- Preserve all existing function signatures
- Maintain existing data structures where possible
- Add comprehensive regression tests

### Measurement Precision

**Risk:** Rounding or precision errors in conversion
**Mitigation:**
- Use consistent precision handling with Auto Mode
- Implement proper rounding at display layer only
- Maintain full precision in intermediate calculations

### Shape Detection Quality

**Risk:** Changes might inadvertently affect shape detection
**Mitigation:**
- Isolate changes to measurement conversion layer only
- Preserve all shape snapping and detection algorithms
- Add tests to verify shape detection quality is maintained