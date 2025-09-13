# Design Document

## Overview

The interactive inspect mode will be implemented as an enhancement to the existing OpenCV object detection system. The design maintains the current A4 detection and measurement pipeline while adding a new interactive layer that provides hover previews and click-to-inspect functionality. The system will separate shape detection from shape rendering, allowing for selective display based on user interaction.

## Architecture

### Current System Flow
1. Camera capture → A4 detection → Perspective correction → Object segmentation → Shape classification → Measurement → Display all shapes

### New Interactive Flow  
1. Camera capture → A4 detection → Perspective correction → Object segmentation → Shape classification → Measurement → **Interactive Display Mode**
   - Store all detected shapes internally
   - Render clean background by default
   - Handle mouse events for hover/click interactions
   - Selectively render shapes based on interaction state

### Key Design Principles
- **Separation of Concerns**: Decouple shape detection from shape rendering
- **State Management**: Track hover and selection states independently  
- **Event-Driven Rendering**: Update display only when interaction state changes
- **Backward Compatibility**: Preserve existing measurement accuracy and A4 scaling

## Components and Interfaces

### 1. Shape Data Structure
```python
ShapeData = {
    "type": str,  # "circle" or "rectangle"
    "measurements": dict,  # width_mm, height_mm, diameter_mm, etc.
    "geometry": dict,  # center, radius_px, box points, etc.
    "hit_contour": np.ndarray,  # Polygon for hit testing
    "area_px": float,  # For selection priority
    "inner": bool  # Whether this is an inner shape
}
```

### 2. Interaction State Manager
```python
InteractionState = {
    "hovered": Optional[int],  # Index of currently hovered shape
    "selected": Optional[int],  # Index of currently selected shape  
    "mouse_pos": Tuple[int, int],  # Current mouse position
    "shapes": List[ShapeData],  # All detected shapes
    "display_scale": float  # Scale factor for display window
}
```

### 3. Hit Testing Engine
- **Primary Method**: `cv2.pointPolygonTest()` for precise containment checking
- **Fallback Method**: Distance-based proximity for snapping behavior
- **Priority Logic**: Smallest area shape wins when multiple shapes overlap
- **Snapping Distance**: Configurable threshold (default: 10 * PX_PER_MM)

### 4. Rendering Engine
- **Base Renderer**: Clean A4 background without any overlays
- **Preview Renderer**: Outline-only rendering for hovered shapes
- **Selection Renderer**: Full dimension rendering for selected shapes
- **Text Renderer**: Dynamic instruction text based on current state

### 5. Mouse Event Handler
- **Move Events**: Update hover state and trigger re-render if changed
- **Click Events**: Update selection state and trigger re-render
- **Coordinate Transformation**: Convert display coordinates to original image coordinates

## Data Models

### Shape Detection Pipeline Enhancement
The existing `classify_and_measure()` function will be enhanced to return additional data needed for interaction:

```python
def create_shape_data(measurement_result, contour):
    """Convert measurement result to interactive shape data"""
    shape_data = {
        "type": measurement_result["type"],
        "measurements": extract_measurements(measurement_result),
        "geometry": extract_geometry(measurement_result),
        "hit_contour": create_hit_polygon(measurement_result, contour),
        "area_px": cv2.contourArea(contour),
        "inner": measurement_result.get("inner", False)
    }
    return shape_data
```

### Hit Testing Polygon Generation
- **Circles**: Generate 36-point polygon approximation around circle boundary
- **Rectangles**: Use the existing box points from `cv2.minAreaRect()`
- **Coordinate System**: All polygons in original warped image coordinates

### Display Coordinate Transformation
```python
def transform_coordinates(original_pos, display_scale):
    """Transform between display window and original image coordinates"""
    return (int(original_pos[0] * display_scale), int(original_pos[1] * display_scale))
```

## Error Handling

### Mouse Event Robustness
- **Boundary Checking**: Ensure mouse coordinates are within image bounds
- **Invalid Shape Handling**: Skip shapes with malformed geometry data
- **Rendering Failures**: Graceful degradation if shape rendering fails

### Memory Management
- **Shape Data Cleanup**: Clear shape list when returning to scan mode
- **OpenCV Window Management**: Proper cleanup of mouse callbacks and windows
- **Resource Leaks**: Ensure camera capture is properly released and re-initialized

### Edge Cases
- **No Shapes Detected**: Display appropriate message and handle empty shape list
- **Overlapping Shapes**: Consistent selection priority based on area
- **Rapid Mouse Movement**: Debounce hover state changes to prevent flicker

## Testing Strategy

### Unit Testing
- **Hit Testing Accuracy**: Verify `cv2.pointPolygonTest()` results for various shapes
- **Coordinate Transformation**: Test display-to-original coordinate mapping
- **Shape Data Creation**: Validate shape data structure generation from measurements
- **Priority Selection**: Test smallest-area-first selection logic

### Integration Testing  
- **Mouse Event Flow**: Test complete hover → click → selection workflow
- **Rendering Pipeline**: Verify correct shape rendering in different states
- **Mode Transitions**: Test scan mode → inspect mode → scan mode transitions
- **Memory Management**: Verify proper cleanup between detection cycles

### Visual Testing
- **Hover Accuracy**: Manual verification that hover highlights correct shapes
- **Selection Precision**: Verify clicked shapes match visual expectations  
- **Dimension Display**: Confirm measurements match original implementation
- **Snapping Behavior**: Test edge-case proximity detection

### Performance Testing
- **Real-time Responsiveness**: Ensure smooth mouse tracking without lag
- **Rendering Performance**: Verify acceptable frame rates during interaction
- **Memory Usage**: Monitor memory consumption during extended use
- **Large Shape Count**: Test performance with many detected shapes

## Implementation Notes

### Existing Code Integration Points
- **main.py**: Modify the inspect mode section (lines ~180-350) to use new interactive system
- **measure.py**: Enhance shape classification to return hit testing data
- **No changes needed**: camera.py, detection.py, config.py, utils.py remain unchanged

### Configuration Extensions
Add new configuration parameters to config.py:
- `HOVER_SNAP_DISTANCE_MM`: Distance threshold for hover snapping (default: 10mm)
- `PREVIEW_COLOR`: Color for hover preview outlines (default: (0, 200, 200))
- `SELECTION_COLOR`: Color for selected shape rendering (default: (0, 255, 0))

### Backward Compatibility
- All existing measurement accuracy preserved
- Original A4 detection and scaling unchanged  
- Console output format maintained for selected shapes
- ESC key behavior and mode transitions preserved