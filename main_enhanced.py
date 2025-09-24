import cv2
import numpy as np
from camera import open_capture
from detection import find_a4_quad, warp_a4, a4_scale_mm_per_px
from measure import segment_object, largest_inner_contour, all_inner_contours, classify_and_measure, annotate_results, annotate_result, detect_inner_circles, detect_inner_rectangles
from utils import draw_text
from config import STABLE_FRAMES, MAX_CORNER_JITTER, DRAW_THICKNESS, DRAW_FONT, MIN_OBJECT_AREA_MM2, PX_PER_MM

def corners_stable(prev, curr, tol):
    if prev is None or curr is None:
        return False
    d = np.linalg.norm(prev - curr, axis=1).mean()
    return d <= tol

def main():
    print("=== A4 Object Dimension Scanner (Enhanced Inspect Mode) ===")
    ip_base = input("Enter IP camera base URL (e.g. http://192.168.1.7:8080) or leave empty to use webcam: ").strip()
    cap, tried_ip = open_capture(ip_base if ip_base else None)
    print("[INFO] Video source:", "IP Camera" if tried_ip else "Webcam")

    # Create windows with proper sizing
    cv2.namedWindow("Scan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scan", 800, 600)  # Set default size for scan window

    stable_count = 0
    last_quad = None
    pause_for_processing = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame. Reconnecting / Exiting...")
            break

        display = frame.copy()

        quad = find_a4_quad(frame)
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
            draw_text(display, f"A4 detected. Stabilizing... ({stable_count}/{STABLE_FRAMES})", (20, 40), (0,255,0), 0.8, 2)

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

            warped, _ = warp_a4(frame2, quad)
            mm_per_px_x, mm_per_px_y = a4_scale_mm_per_px()
            mask = segment_object(warped)
            cnts = all_inner_contours(mask)
            results = []
            if cnts:
                min_area_px = MIN_OBJECT_AREA_MM2 * (PX_PER_MM**2)
                for cnt in cnts:
                    # Filter out contours smaller than minimum area (in pixels)
                    if cv2.contourArea(cnt) >= min_area_px:
                        # Add outer shape (rectangle/circle) of the object
                        r = classify_and_measure(cnt, mm_per_px_x, mm_per_px_y)
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
                # Build interactive 'inspect' mode with enhanced visual feedback
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
                        # Create polygon for hit testing with more points for better accuracy
                        angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
                        poly = np.array([(int(center[0] + radius_px * np.cos(a)),
                                         int(center[1] + radius_px * np.sin(a))) for a in angles])
                        entry["hit_cnt"] = poly.reshape(-1, 1, 2).astype(np.int32)
                    shapes.append(entry)

                # State tracking for hover and selection
                state = {
                    "hovered": None,   # Index of hovered shape
                    "selected": None,  # Index of selected shape
                    "mouse_pos": (0, 0),
                    "show_cursor": True  # Show cursor crosshair
                }

                # Enhanced color scheme
                COLORS = {
                    "hover": (255, 200, 0),      # Cyan/yellow for hover
                    "selected": (0, 255, 0),      # Green for selected
                    "selected_dim": (0, 200, 0),  # Darker green for dimensions
                    "cursor": (128, 128, 128),    # Gray for cursor
                    "bg_overlay": (240, 240, 240) # Light background
                }

                # Rendering functions
                def draw_shape_outline(img, sh, color, thickness=2, dashed=False):
                    """Draw only the outline of a shape with optional dashed line"""
                    if dashed:
                        # Create dashed effect by drawing segments
                        if sh["type"] == "circle":
                            center = sh["center"]
                            radius = int(sh["radius_px"])
                            # Draw dashed circle using arc segments
                            for i in range(0, 360, 15):
                                if i % 30 == 0:
                                    cv2.ellipse(img, center, (radius, radius), 0, i, i+10, color, thickness)
                        else:
                            box = sh["box"]
                            # Draw dashed rectangle
                            for i in range(4):
                                p1 = box[i]
                                p2 = box[(i+1)%4]
                                # Draw dashed line
                                dist = np.linalg.norm(p2 - p1)
                                num_dashes = int(dist / 20)
                                for j in range(0, num_dashes, 2):
                                    t1 = j / num_dashes
                                    t2 = min((j + 1) / num_dashes, 1.0)
                                    pt1 = (p1 + t1 * (p2 - p1)).astype(int)
                                    pt2 = (p1 + t2 * (p2 - p1)).astype(int)
                                    cv2.line(img, tuple(pt1), tuple(pt2), color, thickness)
                    else:
                        if sh["type"] == "circle":
                            cv2.circle(img, sh["center"], int(sh["radius_px"]), color, thickness)
                        else:
                            cv2.drawContours(img, [sh["box"]], 0, color, thickness)

                def draw_shape_with_dims(img, sh, color):
                    """Draw shape with full dimensions and annotations"""
                    # Add subtle highlight effect
                    overlay = img.copy()
                    
                    if sh["type"] == "circle":
                        center = sh["center"]
                        radius_px = int(sh["radius_px"])
                        
                        # Draw filled semi-transparent circle for highlight
                        cv2.circle(overlay, center, radius_px, color, -1)
                        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
                        
                        # Draw circle outline
                        cv2.circle(img, center, radius_px, color, 3)
                        
                        # Draw diameter line with markers
                        x0 = center[0] - radius_px
                        x1 = center[0] + radius_px
                        y = center[1]
                        cv2.line(img, (x0, y), (x1, y), color, 2)
                        # Add end markers
                        cv2.circle(img, (x0, y), 4, color, -1)
                        cv2.circle(img, (x1, y), 4, color, -1)
                        
                        # Draw dimension text with better background
                        text_inside = f"⌀ {sh['diameter_mm']:.1f} mm"
                        font_scale = 0.8
                        thickness = 2
                        ts = cv2.getTextSize(text_inside, DRAW_FONT, font_scale, thickness)[0]
                        text_org = (int(center[0] - ts[0] / 2), int(center[1] + ts[1] / 2))
                        
                        # Draw text background with rounded corners effect
                        padding = 8
                        cv2.rectangle(img,
                                      (text_org[0] - padding, text_org[1] - ts[1] - padding),
                                      (text_org[0] + ts[0] + padding, text_org[1] + padding),
                                      (255, 255, 255), -1)
                        cv2.rectangle(img,
                                      (text_org[0] - padding, text_org[1] - ts[1] - padding),
                                      (text_org[0] + ts[0] + padding, text_org[1] + padding),
                                      color, 1)
                        draw_text(img, text_inside, text_org, (0, 0, 0), font_scale, thickness)
                    else:
                        box = sh["box"]
                        
                        # Draw filled semi-transparent rectangle for highlight
                        cv2.drawContours(overlay, [box], 0, color, -1)
                        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
                        
                        # Draw rectangle outline
                        cv2.drawContours(img, [box], 0, color, 3)
                        
                        # Draw dimension arrows with better visibility
                        mid_left = ((box[0] + box[3]) / 2).astype(int)
                        mid_right = ((box[1] + box[2]) / 2).astype(int)
                        cv2.arrowedLine(img, tuple(mid_left), tuple(mid_right), color, 2, tipLength=0.03)
                        cv2.arrowedLine(img, tuple(mid_right), tuple(mid_left), color, 2, tipLength=0.03)
                        
                        mid_top = ((box[0] + box[1]) / 2).astype(int)
                        mid_bottom = ((box[2] + box[3]) / 2).astype(int)
                        cv2.arrowedLine(img, tuple(mid_top), tuple(mid_bottom), color, 2, tipLength=0.03)
                        cv2.arrowedLine(img, tuple(mid_bottom), tuple(mid_top), color, 2, tipLength=0.03)
                        
                        # Draw dimension text with icons
                        cx = int(np.mean(box[:, 0]))
                        cy = int(np.mean(box[:, 1]))
                        text_width = f"W: {sh['width_mm']:.1f} mm"
                        text_height = f"H: {sh['height_mm']:.1f} mm"
                        
                        font_scale = 0.7
                        thickness = 2
                        
                        # Calculate text sizes
                        ts_w = cv2.getTextSize(text_width, DRAW_FONT, font_scale, thickness)[0]
                        ts_h = cv2.getTextSize(text_height, DRAW_FONT, font_scale, thickness)[0]
                        
                        # Draw text background
                        padding = 8
                        total_height = ts_w[1] + ts_h[1] + padding * 3
                        total_width = max(ts_w[0], ts_h[0]) + padding * 2
                        
                        bg_x1 = cx - total_width // 2
                        bg_y1 = cy - total_height // 2
                        bg_x2 = cx + total_width // 2
                        bg_y2 = cy + total_height // 2
                        
                        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
                        
                        # Draw dimension texts
                        text_x = cx - max(ts_w[0], ts_h[0]) // 2
                        text_y1 = cy - ts_h[1] // 2
                        text_y2 = cy + ts_h[1] // 2 + padding
                        
                        draw_text(img, text_width, (text_x, text_y1), (0, 0, 0), font_scale, thickness)
                        draw_text(img, text_height, (text_x, text_y2), (0, 0, 0), font_scale, thickness)

                def find_shape_at_point(x, y, distance_threshold=20):
                    """Find shape at or near the given point, with improved snapping"""
                    # First check for exact containment
                    candidates = []
                    for i, sh in enumerate(shapes):
                        dist = cv2.pointPolygonTest(sh["hit_cnt"], (x, y), True)
                        if dist >= 0:  # Inside shape
                            # Prioritize inner shapes when nested
                            priority = 1 if sh.get("inner", False) else 2
                            candidates.append((priority, sh["area_px"], i, dist))
                    
                    if candidates:
                        # Sort by priority first, then by area (smallest first for most specific)
                        candidates.sort(key=lambda t: (t[0], t[1]))
                        return candidates[0][2]
                    
                    # Check for nearby shapes (snapping) with dynamic threshold
                    min_dist = float('inf')
                    closest_idx = None
                    
                    # Adjust threshold based on shape size
                    for i, sh in enumerate(shapes):
                        dist = abs(cv2.pointPolygonTest(sh["hit_cnt"], (x, y), True))
                        # Dynamic threshold based on shape size
                        shape_size = np.sqrt(sh["area_px"])
                        adaptive_threshold = min(distance_threshold, shape_size * 0.1)
                        
                        if dist < min_dist and dist <= adaptive_threshold:
                            min_dist = dist
                            closest_idx = i
                    
                    return closest_idx

                def draw_cursor(img, x, y):
                    """Draw a subtle crosshair cursor"""
                    color = COLORS["cursor"]
                    # Draw crosshair
                    cv2.line(img, (x - 10, y), (x - 3, y), color, 1)
                    cv2.line(img, (x + 3, y), (x + 10, y), color, 1)
                    cv2.line(img, (x, y - 10), (x, y - 3), color, 1)
                    cv2.line(img, (x, y + 3), (x, y + 10), color, 1)

                def render():
                    """Render current state with enhanced visual feedback"""
                    base = warped.copy()
                    
                    # Apply subtle background dimming when shape is selected
                    if state["selected"] is not None:
                        overlay = np.ones_like(base) * 245
                        cv2.addWeighted(overlay, 0.3, base, 0.7, 0, base)
                    
                    # Draw hover preview (if not selected)
                    if state["hovered"] is not None and state["hovered"] != state["selected"]:
                        # Draw with dashed outline for hover effect
                        draw_shape_outline(base, shapes[state["hovered"]], COLORS["hover"], 2, dashed=True)
                        
                        # Show quick info tooltip near cursor
                        sh = shapes[state["hovered"]]
                        mx, my = state["mouse_pos"]
                        if sh["type"] == "circle":
                            tooltip = f"Circle: {sh['diameter_mm']:.0f}mm"
                        else:
                            tooltip = f"Rect: {sh['width_mm']:.0f}x{sh['height_mm']:.0f}mm"
                        
                        ts = cv2.getTextSize(tooltip, DRAW_FONT, 0.6, 1)[0]
                        tooltip_x = min(mx + 15, base.shape[1] - ts[0] - 10)
                        tooltip_y = max(my - 10, ts[1] + 10)
                        
                        # Draw tooltip background
                        cv2.rectangle(base,
                                    (tooltip_x - 4, tooltip_y - ts[1] - 4),
                                    (tooltip_x + ts[0] + 4, tooltip_y + 4),
                                    (255, 255, 200), -1)
                        cv2.rectangle(base,
                                    (tooltip_x - 4, tooltip_y - ts[1] - 4),
                                    (tooltip_x + ts[0] + 4, tooltip_y + 4),
                                    COLORS["hover"], 1)
                        draw_text(base, tooltip, (tooltip_x, tooltip_y), (0, 0, 0), 0.6, 1)
                    
                    # Draw selected shape with dimensions
                    if state["selected"] is not None:
                        draw_shape_with_dims(base, shapes[state["selected"]], COLORS["selected"])
                    
                    # Draw cursor if enabled
                    if state["show_cursor"] and state["mouse_pos"] != (0, 0):
                        draw_cursor(base, state["mouse_pos"][0], state["mouse_pos"][1])
                    
                    # Add instruction bar at top
                    bar_height = 60
                    cv2.rectangle(base, (0, 0), (base.shape[1], bar_height), (250, 250, 250), -1)
                    
                    # Status text
                    if state["selected"] is not None:
                        sh = shapes[state["selected"]]
                        if sh["type"] == "circle":
                            status = f"SELECTED: Circle (Diameter: {sh['diameter_mm']:.1f} mm)"
                        else:
                            status = f"SELECTED: Rectangle ({sh['width_mm']:.1f} × {sh['height_mm']:.1f} mm)"
                        color = COLORS["selected"]
                    elif state["hovered"] is not None:
                        status = "HOVERING: Click to inspect this shape"
                        color = COLORS["hover"]
                    else:
                        status = "INSPECT MODE: Hover to preview, click to select"
                        color = (100, 100, 100)
                    
                    draw_text(base, status, (15, 25), color, 0.7, 2)
                    draw_text(base, "Press ESC to exit, any other key to rescan", (15, 45), (150, 150, 150), 0.5, 1)
                    
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
                    
                    if event == cv2.EVENT_MOUSEMOVE:
                        # Update hover state
                        state["mouse_pos"] = (ox, oy)
                        old_hover = state["hovered"]
                        
                        # Use improved snapping with adaptive threshold
                        state["hovered"] = find_shape_at_point(ox, oy, distance_threshold=int(15 * PX_PER_MM))
                        
                        # Always redraw on mouse move for cursor update
                        img = render()
                        disp = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                        cv2.imshow(window_name, disp)
                    
                    elif event == cv2.EVENT_LBUTTONDOWN:
                        # Select shape at click position
                        clicked_shape = find_shape_at_point(ox, oy, distance_threshold=int(15 * PX_PER_MM))
                        
                        # Toggle selection if clicking same shape
                        if clicked_shape == state["selected"]:
                            state["selected"] = None
                        else:
                            state["selected"] = clicked_shape
                        
                        # Print to console
                        if state["selected"] is not None:
                            sh = shapes[state["selected"]]
                            inner_tag = " (inner)" if sh.get("inner", False) else ""
                            if sh["type"] == "circle":
                                print(f"[SELECTED] Circle{inner_tag} - Diameter: {sh['diameter_mm']:.1f} mm")
                            else:
                                print(f"[SELECTED] Rectangle{inner_tag} - Width: {sh['width_mm']:.1f} mm, Height: {sh['height_mm']:.1f} mm")
                        else:
                            print("[DESELECTED] Click on background or same shape to deselect")
                        
                        # Redraw
                        img = render()
                        disp = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                        cv2.imshow(window_name, disp)
                    
                    elif event == cv2.EVENT_RBUTTONDOWN:
                        # Right click to clear selection
                        state["selected"] = None
                        print("[CLEARED] Selection cleared")
                        img = render()
                        disp = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                        cv2.imshow(window_name, disp)

                cv2.setMouseCallback(window_name, on_mouse)

                # Initial render
                img0 = render()
                disp0 = cv2.resize(img0, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                cv2.imshow(window_name, disp0)

                # Print detected shapes summary
                print(f"\n[INFO] Detected {len(shapes)} shape(s):")
                for i, sh in enumerate(shapes):
                    inner_tag = " (inner)" if sh.get("inner", False) else ""
                    if sh["type"] == "circle":
                        print(f"  {i+1}. Circle{inner_tag} - Diameter: {sh['diameter_mm']:.1f} mm")
                    else:
                        print(f"  {i+1}. Rectangle{inner_tag} - {sh['width_mm']:.1f} × {sh['height_mm']:.1f} mm")

                print("\n========== INSPECT MODE CONTROLS ==========")
                print("  • HOVER: Preview shape outline")
                print("  • LEFT CLICK: Select/deselect shape")
                print("  • RIGHT CLICK: Clear selection")
                print("  • ANY KEY: Resume scanning")
                print("  • ESC: Exit program")
                print("==========================================\n")
                
                exit_flag = False
                while True:
                    k = cv2.waitKey(20) & 0xFF
                    if k != 255:  # any key pressed
                        if k == 27:  # ESC
                            cv2.destroyWindow(window_name)
                            exit_flag = True
                        else:
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
