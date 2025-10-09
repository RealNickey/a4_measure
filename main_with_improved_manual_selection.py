#!/usr/bin/env python3
"""
Improved main.py with optimized manual selection system.

This version includes the fixes for:
1. Laggy mode switching
2. Invisible rectangle during manual dragging
3. Better visual feedback and responsiveness
"""

import cv2
import numpy as np
from camera import open_capture
from detection import find_a4_quad, warp_a4, a4_scale_mm_per_px
from measure import segment_object, all_inner_contours, classify_and_measure, detect_inner_circles, detect_inner_rectangles
from utils import draw_text
from config import STABLE_FRAMES, MAX_CORNER_JITTER, DRAW_THICKNESS, DRAW_FONT, MIN_OBJECT_AREA_MM2, PX_PER_MM

# Import improved manual selection components
from extended_interaction_manager import ExtendedInteractionManager, setup_extended_interactive_inspect_mode


def corners_stable(prev, curr, tol):
    if prev is None or curr is None:
        return False
    d = np.linalg.norm(prev - curr, axis=1).mean()
    return d <= tol


def main():
    print("=== A4 Object Dimension Scanner (Improved Manual Selection) ===")
    ip_base = input("Enter IP camera base URL (e.g. http://192.168.1.7:8080) or leave empty to use webcam: ").strip()
    cap, tried_ip = open_capture(ip_base if ip_base else None)
    print("[INFO] Video source:", "IP Camera" if tried_ip else "Webcam")

    # Create windows with proper sizing
    cv2.namedWindow("Scan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scan", 800, 600)

    stable_count = 0
    last_quad = None

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

            window_name = "Enhanced Inspect Mode"

            if not results:
                print("[RESULT] No valid object found fully inside A4.")
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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
                # Build interactive 'inspect' mode with improved manual selection
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

                # Use the improved extended interaction manager
                try:
                    manager = setup_extended_interactive_inspect_mode(
                        shapes=shapes,
                        warped_image=warped,
                        window_name=window_name,
                        enable_performance_optimization=True
                    )

                    # Print detected shapes summary
                    print(f"\n[INFO] Detected {len(shapes)} shape(s):")
                    for i, sh in enumerate(shapes):
                        if sh["type"] == "circle":
                            print(f"  {i+1}. Circle - Diameter: {sh['diameter_mm']:.1f} mm")
                        else:
                            print(f"  {i+1}. Rectangle - {sh['width_mm']:.1f} x {sh['height_mm']:.1f} mm")

                    print("\n[ENHANCED INSPECT MODE] Features:")
                    print("• Hover over shapes to preview")
                    print("• Click shapes to inspect")
                    print("• Press 'M' to cycle between AUTO → MANUAL RECT → MANUAL CIRCLE modes")
                    print("• In manual modes, click and drag to select areas")
                    print("• Right-click to cancel manual selections")
                    print("• Press ESC to exit")
                    print()
                    
                    # Main interaction loop
                    exit_flag = False
                    while True:
                        k = cv2.waitKey(20) & 0xFF
                        if k != 255:  # any key pressed
                            if k == 27:  # ESC
                                exit_flag = True
                                break
                            else:
                                # Handle other keys through the manager
                                manager.handle_key_press(k)
                    
                    # Cleanup
                    manager.cleanup()
                    
                    if exit_flag:
                        break

                except Exception as e:
                    print(f"[ERROR] Enhanced inspect mode failed: {e}")
                    print("[INFO] Falling back to basic mode...")
                    
                    # Fallback to basic display
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    base = warped.copy()
                    draw_text(base, f"Enhanced mode error: {str(e)[:50]}...", (20, 40), (0,0,255), 0.7, 2)
                    display_height = 800
                    scale = display_height / base.shape[0]
                    display_width = int(base.shape[1] * scale)
                    cv2.resizeWindow(window_name, display_width, display_height)
                    disp = cv2.resize(base, (display_width, display_height))
                    cv2.imshow(window_name, disp)
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27:
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