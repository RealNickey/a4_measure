
import cv2
import numpy as np
from camera import open_capture
from detection import find_a4_quad, find_a4_quad_with_quality, warp_a4, a4_scale_mm_per_px
from measure import segment_object, largest_inner_contour, all_inner_contours, classify_and_measure, annotate_results, annotate_result, detect_inner_circles, detect_inner_rectangles, process_manual_selection
from utils import draw_text
from config import STABLE_FRAMES, MAX_CORNER_JITTER, DRAW_THICKNESS, DRAW_FONT, ENABLE_SUBPIXEL_REFINEMENT, MIN_DETECTION_QUALITY

def corners_stable(prev, curr, tol):
    if prev is None or curr is None:
        return False
    d = np.linalg.norm(prev - curr, axis=1).mean()
    return d <= tol

def main():
    print("=== A4 Object Dimension Scanner ===")
    ip_base = input("Enter IP camera base URL (e.g. http://192.168.1.7:8080) or leave empty to use webcam: ").strip()
    cap, tried_ip = open_capture(ip_base if ip_base else None)
    print("[INFO] Video source:", "IP Camera" if tried_ip else "Webcam")

    # Create windows with proper sizing
    cv2.namedWindow("Scan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scan", 800, 600)  # Set default size for scan window

    stable_count = 0
    last_quad = None
    pause_for_processing = False
    detection_quality = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame. Reconnecting / Exiting...")
            break

        display = frame.copy()

        # Use improved detection with quality scoring
        quad, detection_quality = find_a4_quad_with_quality(frame, enable_subpixel=ENABLE_SUBPIXEL_REFINEMENT)
        if quad is not None:
            # draw quad on display
            for i in range(4):
                p1 = tuple(quad[i].astype(int))
                p2 = tuple(quad[(i+1)%4].astype(int))
                cv2.line(display, p1, p2, (0,255,0), 2)

            if last_quad is not None and corners_stable(last_quad, quad, MAX_CORNER_JITTER):
                stable_count += 1
            else:
                stable_count = 1
            last_quad = quad.copy()
        else:
            last_quad = None
            stable_count = 0

        if quad is None:
            draw_text(display, "Show a full A4 sheet in view...", (20, 40), (0,0,255), 0.8, 2)
        else:
            # Show quality indicator with color based on quality
            quality_text = f"A4 detected (Q: {detection_quality:.0%}). Stabilizing... ({stable_count}/{STABLE_FRAMES})"
            if detection_quality < MIN_DETECTION_QUALITY:
                quality_color = (0, 165, 255)  # Orange for low quality
            elif detection_quality < 0.8:
                quality_color = (0, 255, 255)  # Yellow for medium quality
            else:
                quality_color = (0, 255, 0)  # Green for high quality
            draw_text(display, quality_text, (20, 40), quality_color, 0.8, 2)
            
            # Show warning if quality is low
            if detection_quality < MIN_DETECTION_QUALITY:
                draw_text(display, "Warning: Low detection quality. Adjust position/lighting.", 
                         (20, 70), (0, 0, 255), 0.6, 2)

        cv2.imshow("Scan", display)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit anytime
            break

        # Once A4 is stable across frames, take a high-quality frame & process once.
        if quad is not None and stable_count >= STABLE_FRAMES:
            # Freeze: capture one more frame for processing
            ok2, frame2 = cap.read()
            if not ok2:
                frame2 = frame.copy()
            # Stop/flush the video to free resources during processing for consistency
            cap.release()

            # Report detection quality
            print(f"\n[INFO] A4 Detection Quality: {detection_quality:.1%}")
            if detection_quality < MIN_DETECTION_QUALITY:
                print(f"[WARN] Detection quality is below threshold ({MIN_DETECTION_QUALITY:.0%})")
                print("[WARN] Results may be less accurate. Consider adjusting camera position or lighting.")
            
            warped, _ = warp_a4(frame2, quad)
            mm_per_px_x, mm_per_px_y = a4_scale_mm_per_px()
            mask = segment_object(warped)
            cnts = all_inner_contours(mask)
            results = []
            if cnts:
                from config import MIN_OBJECT_AREA_MM2, PX_PER_MM
                min_area_px = MIN_OBJECT_AREA_MM2 * (PX_PER_MM**2)
                for cnt in cnts:
                    # Filter out contours smaller than minimum area (in pixels)
                    if cv2.contourArea(cnt) >= min_area_px:
                        # Add outer shape (rectangle/circle) of the object
                        r = classify_and_measure(cnt, mm_per_px_x, mm_per_px_y, "automatic")
                        if r is not None:
                            results.append(r)
                        # Detect inner prominent shapes (one circle and one rectangle)
                        inner_c = detect_inner_circles(warped, mask, cnt, mm_per_px_x)
                        if inner_c:
                            results.extend(inner_c)
                        inner_r = detect_inner_rectangles(warped, mask, cnt, mm_per_px_x, mm_per_px_y)
                        if inner_r:
                            results.extend(inner_r)

            window_name = "Inspect Mode"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            if not results:
                print("[RESULT] No valid object found fully inside A4.")
                base = warped.copy()
                draw_text(base, "No valid object detected.", (20, 40), (0,0,255), 0.9, 2)
                display_height = 800
                scale = display_height / base.shape[0]
                display_width = int(base.shape[1] * scale)
                cv2.resizeWindow(window_name, display_width, display_height)
                disp = cv2.resize(base, (display_width, display_height))
                cv2.imshow(window_name, disp)
                print("[INFO] Press any key in the window to resume scanning, or ESC to exit.")
                key = cv2.waitKey(0) & 0xFF
                if key == 27:
                    break
                try:
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass
            else:
                # Build interactive 'inspect' mode with manual selection support
                shapes = []
                for r in results:
                    entry = {
                        "type": r["type"],
                        "inner": r.get("inner", False)
                    }
                    if r["type"] == "rectangle":
                        box = np.array(r["box"], dtype=np.int32)
                        entry["box"] = box
                        entry["width_mm"] = float(r["width_mm"])
                        entry["height_mm"] = float(r["height_mm"])
                        entry["area_px"] = float(cv2.contourArea(box))
                        entry["hit_cnt"] = box.reshape(-1, 1, 2).astype(np.int32)
                    else:
                        center = (int(r["center"][0]), int(r["center"][1]))
                        radius_px = int(round(r["radius_px"]))
                        entry["center"] = center
                        entry["radius_px"] = radius_px
                        entry["diameter_mm"] = float(r["diameter_mm"])
                        entry["area_px"] = float(np.pi * (radius_px ** 2))
                        # Create polygon for hit testing
                        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
                        poly = np.array([(int(center[0] + radius_px * np.cos(a)),
                                         int(center[1] + radius_px * np.sin(a))) for a in angles])
                        entry["hit_cnt"] = poly.reshape(-1, 1, 2).astype(np.int32)
                    shapes.append(entry)

                # Enhanced interaction mode with manual selection support
                # Manual selection mode management
                current_mode = "AUTO"  # AUTO, MANUAL_RECT, MANUAL_CIRCLE
                mode_cycle = ["AUTO", "MANUAL_RECT", "MANUAL_CIRCLE"]
                
                # Manual selection state
                manual_selecting = False
                manual_start_point = None
                manual_current_point = None
                manual_selection_rect = None
                
                def cycle_mode():
                    nonlocal current_mode
                    current_index = mode_cycle.index(current_mode)
                    next_index = (current_index + 1) % len(mode_cycle)
                    current_mode = mode_cycle[next_index]
                    print(f"[INFO] Mode switched to: {current_mode}")
                    return current_mode
                
                def start_manual_selection(x, y):
                    nonlocal manual_selecting, manual_start_point, manual_current_point, manual_selection_rect
                    manual_selecting = True
                    manual_start_point = (x, y)
                    manual_current_point = (x, y)
                    manual_selection_rect = None
                    
                def update_manual_selection(x, y):
                    nonlocal manual_current_point, manual_selection_rect
                    if manual_selecting and manual_start_point:
                        manual_current_point = (x, y)
                        # Calculate selection rectangle
                        x1, y1 = manual_start_point
                        x2, y2 = manual_current_point
                        manual_selection_rect = (
                            min(x1, x2), min(y1, y2),
                            abs(x2 - x1), abs(y2 - y1)
                        )
                
                def complete_manual_selection():
                    nonlocal manual_selecting, manual_selection_rect, shapes
                    if manual_selecting and manual_selection_rect:
                        x, y, w, h = manual_selection_rect
                        if w > 10 and h > 10:  # Minimum size check
                            print(f"[MANUAL] Selected area: {w}x{h} pixels at ({x}, {y})")
                            
                            # Convert mode to the format expected by process_manual_selection
                            if current_mode == "MANUAL_CIRCLE":
                                mode_str = "manual_circle"
                                print(f"[MANUAL] Looking for circle in selection...")
                            elif current_mode == "MANUAL_RECT":
                                mode_str = "manual_rectangle"
                                print(f"[MANUAL] Looking for rectangle in selection...")
                            else:
                                print(f"[MANUAL] Invalid mode: {current_mode}")
                                manual_selecting = False
                                manual_selection_rect = None
                                return
                            
                            # Call process_manual_selection with proper scaling factors
                            try:
                                # Validate scaling factors before processing
                                if mm_per_px_x is None or mm_per_px_y is None:
                                    print("[MANUAL] ERROR: A4 calibration failed - scaling factors are None")
                                    print("[MANUAL] Please ensure A4 paper is properly detected before using manual selection")
                                    manual_selecting = False
                                    manual_selection_rect = None
                                    return
                                
                                if mm_per_px_x <= 0 or mm_per_px_y <= 0:
                                    print(f"[MANUAL] ERROR: Invalid scaling factors - X: {mm_per_px_x}, Y: {mm_per_px_y}")
                                    print("[MANUAL] A4 calibration appears to be invalid")
                                    manual_selecting = False
                                    manual_selection_rect = None
                                    return
                                
                                manual_result = process_manual_selection(
                                    warped, manual_selection_rect, mode_str, 
                                    mm_per_px_x, mm_per_px_y
                                )
                                
                                if manual_result is not None:
                                    # Convert manual result to the same format as automatic detection results
                                    entry = {
                                        "type": manual_result["type"],
                                        "inner": False  # Manual selections are treated as outer shapes
                                    }
                                    
                                    if manual_result["type"] == "rectangle":
                                        box = np.array(manual_result["box"], dtype=np.int32)
                                        entry["box"] = box
                                        entry["width_mm"] = float(manual_result["width_mm"])
                                        entry["height_mm"] = float(manual_result["height_mm"])
                                        entry["area_px"] = float(cv2.contourArea(box))
                                        entry["hit_cnt"] = box.reshape(-1, 1, 2).astype(np.int32)
                                        print(f"[MANUAL] Found rectangle - {entry['width_mm']:.1f} x {entry['height_mm']:.1f} mm")
                                    else:  # circle
                                        center = (int(manual_result["center"][0]), int(manual_result["center"][1]))
                                        radius_px = int(round(manual_result["radius_px"]))
                                        entry["center"] = center
                                        entry["radius_px"] = radius_px
                                        entry["diameter_mm"] = float(manual_result["diameter_mm"])
                                        entry["area_px"] = float(np.pi * (radius_px ** 2))
                                        # Create polygon for hit testing
                                        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
                                        poly = np.array([(int(center[0] + radius_px * np.cos(a)),
                                                         int(center[1] + radius_px * np.sin(a))) for a in angles])
                                        entry["hit_cnt"] = poly.reshape(-1, 1, 2).astype(np.int32)
                                        print(f"[MANUAL] Found circle - Diameter: {entry['diameter_mm']:.1f} mm")
                                    
                                    # Add to shapes list
                                    shapes.append(entry)
                                    print(f"[MANUAL] Added manual detection result to shapes list")
                                    
                                else:
                                    print(f"[MANUAL] No suitable shape found in selection")
                                    
                            except ValueError as e:
                                # Enhanced error handling for calibration issues
                                if "calibration" in str(e).lower() or "scaling" in str(e).lower():
                                    print(f"[MANUAL] CALIBRATION ERROR: {e}")
                                    print("[MANUAL] Please recalibrate by ensuring A4 paper is properly positioned")
                                else:
                                    print(f"[MANUAL] VALIDATION ERROR: {e}")
                            except Exception as e:
                                # Maintain existing shape detection error handling
                                print(f"[MANUAL] ERROR: Error processing selection: {e}")
                                print("[MANUAL] Manual selection failed, but system remains stable")
                                
                    manual_selecting = False
                    manual_selection_rect = None
                
                def cancel_manual_selection():
                    nonlocal manual_selecting, manual_selection_rect
                    manual_selecting = False
                    manual_selection_rect = None
                    print("[MANUAL] Selection cancelled")
                    manual_selection_rect = None
                    print("[MANUAL] Selection cancelled")
                
                # State tracking for hover and selection
                state = {
                    "hovered": None,   # Index of hovered shape
                    "selected": None,  # Index of selected shape
                    "mouse_pos": (0, 0)
                }

                # Rendering functions
                def draw_shape_outline(img, sh, color, thickness=2):
                    """Draw only the outline of a shape"""
                    if sh["type"] == "circle":
                        cv2.circle(img, sh["center"], int(sh["radius_px"]), color, thickness)
                    else:
                        cv2.drawContours(img, [sh["box"]], 0, color, thickness)

                def draw_shape_with_dims(img, sh, color):
                    """Draw shape with full dimensions and annotations"""
                    if sh["type"] == "circle":
                        center = sh["center"]
                        radius_px = int(sh["radius_px"])
                        # Draw circle
                        cv2.circle(img, center, radius_px, color, 3)
                        # Draw diameter line
                        x0 = center[0] - radius_px
                        x1 = center[0] + radius_px
                        y = center[1]
                        cv2.line(img, (x0, y), (x1, y), color, 2)
                        # Draw dimension text
                        text_inside = f"D={sh['diameter_mm']:.0f}mm"
                        ts = cv2.getTextSize(text_inside, DRAW_FONT, 0.9, 2)[0]
                        text_org = (int(center[0] - ts[0] / 2), int(center[1] + ts[1] / 2))
                        cv2.rectangle(img,
                                      (text_org[0] - 6, text_org[1] - ts[1] - 6),
                                      (text_org[0] + ts[0] + 6, text_org[1] + 6),
                                      (255, 255, 255), -1)
                        draw_text(img, text_inside, text_org, (0, 0, 0), 0.9, 2)
                    else:
                        box = sh["box"]
                        # Draw rectangle
                        cv2.drawContours(img, [box], 0, color, 3)
                        # Draw dimension arrows
                        mid_left = ((box[0] + box[3]) / 2).astype(int)
                        mid_right = ((box[1] + box[2]) / 2).astype(int)
                        cv2.arrowedLine(img, tuple(mid_left), tuple(mid_right), color, 2, tipLength=0.02)
                        cv2.arrowedLine(img, tuple(mid_right), tuple(mid_left), color, 2, tipLength=0.02)
                        mid_top = ((box[0] + box[1]) / 2).astype(int)
                        mid_bottom = ((box[2] + box[3]) / 2).astype(int)
                        cv2.arrowedLine(img, tuple(mid_top), tuple(mid_bottom), color, 2, tipLength=0.02)
                        cv2.arrowedLine(img, tuple(mid_bottom), tuple(mid_top), color, 2, tipLength=0.02)
                        # Draw dimension text
                        cx = int(np.mean(box[:, 0]))
                        cy = int(np.mean(box[:, 1]))
                        text_inside = f"W={sh['width_mm']:.0f}mm  H={sh['height_mm']:.0f}mm"
                        ts = cv2.getTextSize(text_inside, DRAW_FONT, 0.9, 2)[0]
                        text_org = (int(cx - ts[0] / 2), int(cy + ts[1] / 2))
                        cv2.rectangle(img,
                                      (text_org[0] - 6, text_org[1] - ts[1] - 6),
                                      (text_org[0] + ts[0] + 6, text_org[1] + 6),
                                      (255, 255, 255), -1)
                        draw_text(img, text_inside, text_org, (0, 0, 0), 0.9, 2)

                def draw_manual_selection_rect(img, rect):
                    """Draw manual selection rectangle with high visibility"""
                    if rect is None:
                        return
                    x, y, w, h = rect
                    if w <= 0 or h <= 0:
                        return
                    
                    # Bright cyan color for high visibility
                    color = (255, 255, 0)
                    thickness = 3
                    
                    # Draw main rectangle
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
                    
                    # Draw contrasting inner outline
                    cv2.rectangle(img, (x + 1, y + 1), (x + w - 1, y + h - 1), (0, 0, 0), 1)
                    
                    # Draw corner markers
                    marker_size = 10
                    # Top-left
                    cv2.line(img, (x, y), (x + marker_size, y), color, thickness)
                    cv2.line(img, (x, y), (x, y + marker_size), color, thickness)
                    # Top-right
                    cv2.line(img, (x + w, y), (x + w - marker_size, y), color, thickness)
                    cv2.line(img, (x + w, y), (x + w, y + marker_size), color, thickness)
                    # Bottom-left
                    cv2.line(img, (x, y + h), (x + marker_size, y + h), color, thickness)
                    cv2.line(img, (x, y + h), (x, y + h - marker_size), color, thickness)
                    # Bottom-right
                    cv2.line(img, (x + w, y + h), (x + w - marker_size, y + h), color, thickness)
                    cv2.line(img, (x + w, y + h), (x + w, y + h - marker_size), color, thickness)
                    
                    # Add size info
                    info_text = f"{w}x{h}"
                    text_size = cv2.getTextSize(info_text, DRAW_FONT, 0.5, 1)[0]
                    text_x = x + (w - text_size[0]) // 2
                    text_y = max(y - 5, text_size[1] + 5)
                    
                    # Background for text
                    cv2.rectangle(img, (text_x - 3, text_y - text_size[1] - 3),
                                 (text_x + text_size[0] + 3, text_y + 3), (0, 0, 0), -1)
                    cv2.putText(img, info_text, (text_x, text_y), DRAW_FONT, 0.5, color, 1)

                def find_shape_at_point(x, y, distance_threshold=15):
                    """Find shape at or near the given point, with snapping"""
                    # First check for exact containment
                    candidates = []
                    for i, sh in enumerate(shapes):
                        dist = cv2.pointPolygonTest(sh["hit_cnt"], (x, y), True)
                        if dist >= 0:  # Inside shape
                            candidates.append((sh["area_px"], i, dist))
                    
                    if candidates:
                        # Return smallest shape containing point (most specific)
                        candidates.sort(key=lambda t: t[0])
                        return candidates[0][1]
                    
                    # Check for nearby shapes (snapping)
                    min_dist = float('inf')
                    closest_idx = None
                    for i, sh in enumerate(shapes):
                        dist = abs(cv2.pointPolygonTest(sh["hit_cnt"], (x, y), True))
                        if dist < min_dist and dist <= distance_threshold:
                            min_dist = dist
                            closest_idx = i
                    
                    return closest_idx

                def render():
                    """Render current state with manual selection support"""
                    base = warped.copy()
                    
                    # Draw hover preview (if not selected and not in manual selection)
                    if (current_mode == "AUTO" and not manual_selecting and 
                        state["hovered"] is not None and state["hovered"] != state["selected"]):
                        draw_shape_outline(base, shapes[state["hovered"]], (0, 200, 200), 2)
                    
                    # Draw selected shape with dimensions
                    if current_mode == "AUTO" and state["selected"] is not None:
                        draw_shape_with_dims(base, shapes[state["selected"]], (0, 255, 0))
                    
                    # Draw manual selection rectangle
                    if manual_selection_rect is not None:
                        draw_manual_selection_rect(base, manual_selection_rect)
                    
                    # Mode indicator in top-right corner
                    mode_text = f"MODE: {current_mode}"
                    text_size = cv2.getTextSize(mode_text, DRAW_FONT, 0.7, 2)[0]
                    mode_x = base.shape[1] - text_size[0] - 20
                    mode_y = 30
                    
                    # Background for mode indicator
                    cv2.rectangle(base, (mode_x - 10, mode_y - text_size[1] - 10),
                                 (mode_x + text_size[0] + 10, mode_y + 10), (0, 0, 0), -1)
                    cv2.rectangle(base, (mode_x - 10, mode_y - text_size[1] - 10),
                                 (mode_x + text_size[0] + 10, mode_y + 10), (255, 255, 255), 1)
                    cv2.putText(base, mode_text, (mode_x, mode_y), DRAW_FONT, 0.7, (255, 255, 255), 2)
                    
                    # Add instruction text
                    if current_mode == "AUTO":
                        if state["selected"] is not None:
                            sh = shapes[state["selected"]]
                            if sh["type"] == "circle":
                                text = f"Selected: Circle (D={sh['diameter_mm']:.0f}mm)"
                            else:
                                text = f"Selected: Rectangle ({sh['width_mm']:.0f}x{sh['height_mm']:.0f}mm)"
                        else:
                            text = "Hover to preview, click to inspect. Press 'M' to switch modes."
                    else:
                        if manual_selecting:
                            text = f"Drag to select area for {current_mode.replace('MANUAL_', '').lower()} detection"
                        else:
                            text = f"Click and drag to select area for {current_mode.replace('MANUAL_', '').lower()} detection"
                    
                    draw_text(base, text, (20, 40), (255, 255, 255), 0.7, 2)
                    
                    return base

                # Setup display
                display_height = 800
                scale = display_height / warped.shape[0]
                display_width = int(warped.shape[1] * scale)
                cv2.resizeWindow(window_name, display_width, display_height)

                def on_mouse(event, x, y, flags, userdata):
                    # Convert to original coordinates
                    ox = int(x / scale)
                    oy = int(y / scale)
                    needs_redraw = False
                    
                    if current_mode == "AUTO":
                        # Automatic mode - original behavior
                        if event == cv2.EVENT_MOUSEMOVE:
                            # Update hover state
                            state["mouse_pos"] = (ox, oy)
                            old_hover = state["hovered"]
                            state["hovered"] = find_shape_at_point(ox, oy, distance_threshold=int(10 * PX_PER_MM))
                            
                            # Redraw if hover changed
                            if old_hover != state["hovered"]:
                                needs_redraw = True
                        
                        elif event == cv2.EVENT_LBUTTONDOWN:
                            # Select shape at click position
                            clicked_shape = find_shape_at_point(ox, oy, distance_threshold=int(10 * PX_PER_MM))
                            state["selected"] = clicked_shape
                            
                            # Print to console
                            if clicked_shape is not None:
                                sh = shapes[clicked_shape]
                                if sh["type"] == "circle":
                                    print(f"[SELECTED] Circle - Diameter: {sh['diameter_mm']:.1f} mm")
                                else:
                                    print(f"[SELECTED] Rectangle - Width: {sh['width_mm']:.1f} mm, Height: {sh['height_mm']:.1f} mm")
                            else:
                                print("[SELECTED] None (click on background)")
                            
                            needs_redraw = True
                    
                    else:
                        # Manual mode - selection behavior
                        if event == cv2.EVENT_LBUTTONDOWN:
                            # Start manual selection
                            start_manual_selection(ox, oy)
                            needs_redraw = True
                            
                        elif event == cv2.EVENT_MOUSEMOVE:
                            # Update manual selection during drag
                            if flags & cv2.EVENT_FLAG_LBUTTON and manual_selecting:
                                update_manual_selection(ox, oy)
                                needs_redraw = True
                                
                        elif event == cv2.EVENT_LBUTTONUP:
                            # Complete manual selection
                            if manual_selecting:
                                complete_manual_selection()
                                needs_redraw = True
                                
                        elif event == cv2.EVENT_RBUTTONDOWN:
                            # Cancel manual selection
                            if manual_selecting:
                                cancel_manual_selection()
                                needs_redraw = True
                    
                    # Redraw if needed
                    if needs_redraw:
                        img = render()
                        disp = cv2.resize(img, (display_width, display_height))
                        cv2.imshow(window_name, disp)
                        cv2.waitKey(1)  # Force immediate update

                cv2.setMouseCallback(window_name, on_mouse)

                # Initial render
                img0 = render()
                disp0 = cv2.resize(img0, (display_width, display_height))
                cv2.imshow(window_name, disp0)

                # Print detected shapes summary
                print(f"\n[INFO] Detected {len(shapes)} shape(s):")
                for i, sh in enumerate(shapes):
                    confidence_str = ""
                    if "confidence_score" in results[i]:
                        confidence_str = f" (confidence: {results[i]['confidence_score']:.0%})"
                    if sh["type"] == "circle":
                        print(f"  {i+1}. Circle - Diameter: {sh['diameter_mm']:.1f} mm{confidence_str}")
                    else:
                        print(f"  {i+1}. Rectangle - {sh['width_mm']:.1f} x {sh['height_mm']:.1f} mm{confidence_str}")

                print(f"\n[ENHANCED INSPECT MODE] Current mode: {current_mode}")
                print("Controls:")
                print("• Hover over shapes to preview (AUTO mode)")
                print("• Click shapes to inspect (AUTO mode)")
                print("• Press 'M' to cycle between AUTO → MANUAL RECT → MANUAL CIRCLE modes")
                print("• In manual modes, click and drag to select areas")
                print("• Right-click to cancel manual selections")
                print("• Press ESC to exit, any other key to resume scanning")
                print()
                
                while True:
                    k = cv2.waitKey(20) & 0xFF
                    if k != 255:  # any key pressed
                        if k == 27:  # ESC
                            cv2.destroyWindow(window_name)
                            exit_flag = True
                            break
                        elif k == ord('m') or k == ord('M'):  # M key for mode switching
                            # Cancel any active manual selection
                            if manual_selecting:
                                cancel_manual_selection()
                            
                            # Cycle to next mode
                            cycle_mode()
                            
                            # Force immediate re-render
                            img = render()
                            disp = cv2.resize(img, (display_width, display_height))
                            cv2.imshow(window_name, disp)
                            cv2.waitKey(1)  # Force display update
                        else:
                            # Other keys exit inspect mode
                            exit_flag = False
                            break
                if exit_flag:
                    break

            # Re-open capture and reset stability
            cap, tried_ip = open_capture(ip_base if ip_base else None)
            stable_count = 0
            last_quad = None

    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
