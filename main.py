
import cv2
import numpy as np
from camera import open_capture
from detection import find_a4_quad, warp_a4, a4_scale_mm_per_px
from measure import segment_object, largest_inner_contour, all_inner_contours, classify_and_measure, annotate_results, annotate_result, detect_inner_circles, detect_inner_rectangles
from utils import draw_text
from config import STABLE_FRAMES, MAX_CORNER_JITTER, DRAW_THICKNESS

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

            if not results:
                print("[RESULT] No valid object found fully inside A4.")
                overlay = warped.copy()
                draw_text(overlay, "No valid object detected.", (20, 40), (0,0,255), 0.9, 2)
                # Resize for display
                display_height = 800  # Target display height
                scale = display_height / overlay.shape[0]
                display_width = int(overlay.shape[1] * scale)
                overlay_resized = cv2.resize(overlay, (display_width, display_height))
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Result", display_width, display_height)
                cv2.imshow("Result", overlay_resized)
            else:
                # Print summary
                for r in results:
                    if r["type"] == "circle":
                        print(f"[RESULT] Circle - Diameter: {r['diameter_mm']:.2f} mm")
                    else:
                        print(f"[RESULT] Rectangle - Width: {r['width_mm']:.2f} mm, Height: {r['height_mm']:.2f} mm")

                annotated = annotate_results(warped, results, (mm_per_px_x, mm_per_px_y))
                # Resize for display
                display_height = 800  # Target display height
                scale = display_height / annotated.shape[0]
                display_width = int(annotated.shape[1] * scale)
                annotated_resized = cv2.resize(annotated, (display_width, display_height))
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Result", display_width, display_height)
                cv2.imshow("Result", annotated_resized)

            print("[INFO] Press any key in the window to resume scanning, or ESC to exit.")
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
            # Close result window
            try:
                cv2.destroyWindow("Result")
            except Exception:
                pass

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
