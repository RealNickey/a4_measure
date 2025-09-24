# How main.py Works Across All Modes

## Overview

`main.py` is the entry point for the A4 Object Dimension Scanner. It operates in **two main phases**: the **Scan Mode** and the **Inspect Mode**. The Inspect Mode supports three different selection modes that can be switched dynamically.

## Complete Workflow

### Phase 1: Initialization and Setup

```python
def main():
    print("=== A4 Object Dimension Scanner ===")
    
    # 1. Camera Setup
    ip_base = input("Enter IP camera base URL or leave empty for webcam: ")
    high_res_input = input("Enable high-resolution mode for 4K+ cameras? (y/N): ")
    
    # 2. Initialize Camera
    cap, tried_ip = open_capture(ip_base, prefer_high_resolution=prefer_high_res)
    
    # 3. Optimize Camera Settings
    camera_info = get_camera_info(cap)
    optimize_camera_for_detection(cap)  # Optimizes saturation for better greyscale conversion
    
    # 4. Create Display Window
    cv2.namedWindow("Scan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scan", 800, 600)
```

### Phase 2: Scan Mode (A4 Detection)

This phase runs continuously until a stable A4 sheet is detected:

```python
while True:
    # 1. Capture Frame
    ok, frame = cap.read()
    display = frame.copy()
    
    # 2. Detect A4 Sheet
    quad = find_a4_quad(frame)
    
    # 3. Check Stability
    if quad is not None:
        # Draw A4 boundary
        for i in range(4):
            p1 = tuple(quad[i].astype(int))
            p2 = tuple(quad[(i+1)%4].astype(int))
            cv2.line(display, p1, p2, (0,255,0), 2)
        
        # Check if corners are stable across frames
        if corners_stable(last_quad, quad, MAX_CORNER_JITTER):
            stable_count += 1
        else:
            stable_count = 1
    
    # 4. Display Status
    if quad is None:
        draw_text(display, "Show a full A4 sheet in view...", (20, 40), (0,0,255))
    else:
        draw_text(display, f"A4 detected. Stabilizing... ({stable_count}/{STABLE_FRAMES})", 
                 (20, 40), (0,255,0))
    
    # 5. Check for Exit
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break
    
    # 6. Transition to Inspect Mode when stable
    if quad is not None and stable_count >= STABLE_FRAMES:
        # Proceed to Phase 3
        break
```

### Phase 3: Object Processing and Analysis

When A4 is stable, the system processes the captured frame:

```python
# 1. Capture High-Quality Frame
ok2, frame2 = cap.read()
cleanup_resources(cap=cap)  # Free camera resources during processing

# 2. Warp A4 to Standard View
warped, _ = warp_a4(frame2, quad)
mm_per_px_x, mm_per_px_y = a4_scale_mm_per_px()

# 3. Segment Objects
mask = segment_object(warped)
cnts = all_inner_contours(mask)

# 4. Automatic Shape Detection
results = []
if cnts:
    min_area_px = MIN_OBJECT_AREA_MM2 * (PX_PER_MM**2)
    for cnt in cnts:
        if cv2.contourArea(cnt) >= min_area_px:
            # Outer shape detection
            r = classify_and_measure(cnt, mm_per_px_x, mm_per_px_y, "automatic")
            if r is not None:
                results.append(r)
            
            # Inner shape detection
            inner_c = detect_inner_circles(warped, mask, cnt, mm_per_px_x)
            if inner_c:
                results.extend(inner_c)
            
            inner_r = detect_inner_rectangles(warped, mask, cnt, mm_per_px_x, mm_per_px_y)
            if inner_r:
                results.extend(inner_r)
```

### Phase 4: Inspect Mode (Multi-Mode Interactive)

This is where all three selection modes operate:

#### 4A: No Objects Detected
```python
if not results:
    print("[RESULT] No valid object found fully inside A4.")
    # Display message and wait for user input
    # ESC exits, any other key returns to scan mode
```

#### 4B: Objects Detected - Enhanced Interactive Mode
```python
else:
    # 1. Convert Results to Shape Data
    shapes = []
    for r in results:
        shape_data = create_shape_data(r)
        if shape_data is not None:
            shapes.append(shape_data)
    
    # 2. Setup Extended Interactive Manager
    manager = setup_extended_interactive_inspect_mode(shapes, warped, window_name)
    
    # 3. Enter Interactive Loop
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k != 255:  # Key pressed
            key_handled = manager.handle_key_press(k)
            
            if key_handled:
                continue  # Manager handled the key
            elif k == 27:  # ESC - exit application
                inspect_exit_flag = True
                break
            else:
                break  # Return to scan mode
```

## The Three Selection Modes

### Mode 1: AUTO Mode (Default)

**How it works:**
- Uses automatically detected shapes from Phase 3
- Hover over shapes to preview measurements
- Click on shapes to inspect details
- No manual selection - works with pre-detected objects

**User interactions:**
- **Mouse hover**: Preview shape measurements
- **Mouse click**: Select and inspect shape
- **M key**: Switch to MANUAL RECT mode

### Mode 2: MANUAL RECTANGLE Mode

**How it works:**
- Allows manual selection of rectangular areas
- Uses shape snapping engine with greyscale processing
- Detects best rectangle within selected area

**User interactions:**
- **Click and drag**: Create selection rectangle
- **Release mouse**: System snaps to best rectangle in area
- **M key**: Switch to MANUAL CIRCLE mode
- **ESC**: Cancel current selection

**Internal processing:**
```python
# When user completes selection
selection_rect = (x, y, width, height)
shape_result = snap_engine.snap_to_shape(
    warped_image, 
    selection_rect, 
    SelectionMode.MANUAL_RECTANGLE
)
# System uses greyscale processing to find best rectangle
```

### Mode 3: MANUAL CIRCLE Mode

**How it works:**
- Allows manual selection of circular areas
- Uses shape snapping engine with greyscale processing
- Detects best circle within selected area

**User interactions:**
- **Click and drag**: Create selection rectangle
- **Release mouse**: System snaps to best circle in area
- **M key**: Switch back to AUTO mode
- **ESC**: Cancel current selection

**Internal processing:**
```python
# When user completes selection
selection_rect = (x, y, width, height)
shape_result = snap_engine.snap_to_shape(
    warped_image, 
    selection_rect, 
    SelectionMode.MANUAL_CIRCLE
)
# System uses greyscale processing to find best circle
```

## Mode Switching Mechanism

The mode switching is handled by the `ExtendedInteractionManager`:

```python
def handle_key_press(self, key: int) -> bool:
    if key == ord('m'):  # M key
        old_mode = self.mode_manager.get_current_mode()
        new_mode = self.mode_manager.cycle_mode()
        
        # Mode cycle: AUTO → MANUAL_RECTANGLE → MANUAL_CIRCLE → AUTO
        
        # Cancel any active manual selection
        self.manual_engine.cancel_selection()
        
        # Clear previous results when switching to auto
        if new_mode == SelectionMode.AUTO:
            self.last_manual_result = None
            self.show_shape_confirmation = False
        
        print(f"[INFO] Mode switched: {old_mode.value} → {new_mode.value}")
        return True
```

## Visual Feedback System

Each mode provides different visual feedback:

### AUTO Mode Display
- Shows automatically detected shapes with colored outlines
- Displays measurements on hover
- Mode indicator: "AUTO" in top-left corner

### MANUAL RECTANGLE Mode Display
- Shows selection rectangle while dragging
- Displays detected rectangle with confidence score
- Mode indicator: "MANUAL RECT" in top-left corner
- Shows shape confirmation after successful detection

### MANUAL CIRCLE Mode Display
- Shows selection rectangle while dragging
- Displays detected circle with confidence score
- Mode indicator: "MANUAL CIRCLE" in top-left corner
- Shows shape confirmation after successful detection

## Error Handling and Recovery

The system includes robust error handling:

```python
try:
    # Interactive mode operations
    manager = setup_extended_interactive_inspect_mode(shapes, warped, window_name)
    # ... interactive loop ...
except Exception as e:
    print(f"[ERROR] Error in inspect mode: {e}")
    inspect_exit_flag = True
finally:
    # Always cleanup resources
    cleanup_resources(window_names=[window_name])
    try:
        manager.cleanup()
    except Exception as e:
        print(f"[WARN] Error during manager cleanup: {e}")
```

## Return to Scan Mode

After inspect mode, the system returns to scan mode:

```python
# Re-initialize camera
print("[INFO] Re-initializing camera for scan mode...")
cap, tried_ip = reinitialize_camera(ip_base)

# Reset scan state
stable_count, last_quad = reset_scan_state()
print("[INFO] Scan state reset. Ready for new detection.")
```

## Performance Optimization

The system includes several performance optimizations:

1. **Camera resource management**: Releases camera during processing
2. **GPU acceleration**: Uses CUDA when available for greyscale processing
3. **Efficient rendering**: Optimizes display updates
4. **Memory management**: Proper cleanup of resources

## Summary

`main.py` orchestrates a sophisticated workflow that seamlessly transitions between:

1. **Scan Mode**: Detects and stabilizes A4 sheet
2. **Processing**: Automatically detects shapes using greyscale processing
3. **Inspect Mode**: Provides three interactive modes:
   - **AUTO**: Work with automatically detected shapes
   - **MANUAL RECT**: Manually select rectangular areas
   - **MANUAL CIRCLE**: Manually select circular areas

The system uses advanced greyscale processing throughout all modes to ensure accurate shape detection and measurement, while providing an intuitive user interface for switching between different interaction paradigms.