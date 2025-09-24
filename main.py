
import cv2
import numpy as np
from camera import open_capture
from detection import find_a4_quad, warp_a4, a4_scale_mm_per_px
from measure import segment_object, largest_inner_contour, all_inner_contours, classify_and_measure, annotate_results, annotate_result, detect_inner_circles, detect_inner_rectangles
from utils import draw_text
from config import STABLE_FRAMES, MAX_CORNER_JITTER, DRAW_THICKNESS, DRAW_FONT

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
                # Build interactive 'inspect' mode
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
                    """Render current state"""
                    base = warped.copy()
                    
                    # Draw hover preview (if not selected)
                    if state["hovered"] is not None and state["hovered"] != state["selected"]:
                        draw_shape_outline(base, shapes[state["hovered"]], (0, 200, 200), 2)
                    
                    # Draw selected shape with dimensions
                    if state["selected"] is not None:
                        draw_shape_with_dims(base, shapes[state["selected"]], (0, 255, 0))
                    
                    # Add instruction text
                    text = "Hover to preview, click to inspect"
                    if state["selected"] is not None:
                        sh = shapes[state["selected"]]
                        if sh["type"] == "circle":
                            text = f"Selected: Circle (D={sh['diameter_mm']:.0f}mm)"
                        else:
                            text = f"Selected: Rectangle ({sh['width_mm']:.0f}x{sh['height_mm']:.0f}mm)"
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
                    
                    if event == cv2.EVENT_MOUSEMOVE:
                        # Update hover state
                        state["mouse_pos"] = (ox, oy)
                        old_hover = state["hovered"]
                        state["hovered"] = find_shape_at_point(ox, oy, distance_threshold=int(10 * PX_PER_MM))
                        
                        # Redraw if hover changed
                        if old_hover != state["hovered"]:
                            img = render()
                            disp = cv2.resize(img, (display_width, display_height))
                            cv2.imshow(window_name, disp)
                    
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
                        
                        # Redraw
                        img = render()
                        disp = cv2.resize(img, (display_width, display_height))
                        cv2.imshow(window_name, disp)

                cv2.setMouseCallback(window_name, on_mouse)

                # Initial render
                img0 = render()
                disp0 = cv2.resize(img0, (display_width, display_height))
                cv2.imshow(window_name, disp0)

                # Print detected shapes summary
                print(f"\n[INFO] Detected {len(shapes)} shape(s):")
                for i, sh in enumerate(shapes):
                    if sh["type"] == "circle":
                        print(f"  {i+1}. Circle - Diameter: {sh['diameter_mm']:.1f} mm")
                    else:
                        print(f"  {i+1}. Rectangle - {sh['width_mm']:.1f} x {sh['height_mm']:.1f} mm")

                print("\n[INSPECT MODE] Hover over shapes to preview, click to inspect.")
                print("Press any key to resume scanning (ESC to exit).")
                
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
