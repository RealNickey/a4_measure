
import cv2
import numpy as np
from camera import open_capture
from detection import find_a4_quad, warp_a4, a4_scale_mm_per_px
from measure import segment_object, largest_inner_contour, all_inner_contours, classify_and_measure, annotate_results, annotate_result, detect_inner_circles, detect_inner_rectangles
from hit_testing import HitTestingEngine
from utils import draw_text
from config import STABLE_FRAMES, MAX_CORNER_JITTER, DRAW_THICKNESS, DRAW_FONT, HOVER_SNAP_DISTANCE_MM, PREVIEW_COLOR, SELECTION_COLOR

def corners_stable(prev, curr, tol):
    if prev is None or curr is None:
        return False
    d = np.linalg.norm(prev - curr, axis=1).mean()
    return d <= tol

def reset_scan_state():
    """Reset all scanning state variables for clean mode transitions."""
    return 0, None  # stable_count, last_quad

def cleanup_resources(cap=None, window_names=None):
    """Properly clean up camera and OpenCV resources."""
    if cap is not None:
        try:
            cap.release()
        except Exception as e:
            print(f"[WARN] Error releasing camera: {e}")
    
    if window_names:
        for window_name in window_names:
            try:
                cv2.destroyWindow(window_name)
            except Exception as e:
                print(f"[WARN] Error destroying window {window_name}: {e}")
    else:
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[WARN] Error destroying all windows: {e}")

def reinitialize_camera(ip_base):
    """Re-initialize camera capture with proper error handling."""
    try:
        cap, tried_ip = open_capture(ip_base if ip_base else None)
        print(f"[INFO] Camera re-initialized: {'IP Camera' if tried_ip else 'Webcam'}")
        return cap, tried_ip
    except Exception as e:
        print(f"[ERROR] Failed to re-initialize camera: {e}")
        return None, False

def main():
    print("=== A4 Object Dimension Scanner ===")
    ip_base = input("Enter IP camera base URL (e.g. http://192.168.1.7:8080) or leave empty to use webcam: ").strip()
    
    # Ask about high-resolution preference
    high_res_input = input("Enable high-resolution mode for 4K+ cameras? (y/N): ").strip().lower()
    prefer_high_res = high_res_input in ['y', 'yes']
    
    cap, tried_ip = open_capture(ip_base if ip_base else None, prefer_high_resolution=prefer_high_res)
    print("[INFO] Video source:", "IP Camera" if tried_ip else "Webcam")
    
    # Get and display camera information
    from camera import get_camera_info, optimize_camera_for_detection
    camera_info = get_camera_info(cap)
    if camera_info:
        print(f"[INFO] Camera resolution: {camera_info['width']}x{camera_info['height']} "
              f"({camera_info['category']}, {camera_info['megapixels']:.1f}MP)")
        print(f"[INFO] Camera FPS: {camera_info['fps']:.1f}")
        
        # Optimize camera settings
        optimize_camera_for_detection(cap)

    # Create windows with proper sizing
    cv2.namedWindow("Scan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scan", 800, 600)  # Set default size for scan window

    stable_count, last_quad = reset_scan_state()

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
            cleanup_resources(cap=cap)
            cap = None

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

            window_name = "Inspect Mode"
            inspect_exit_flag = False
            
            try:
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
                        inspect_exit_flag = True
                else:
                    # Build interactive 'inspect' mode using new selective rendering engine
                    from measure import create_shape_data
                    from interaction_manager import setup_interactive_inspect_mode
                    
                    # Convert measurement results to shape data format
                    shapes = []
                    for r in results:
                        shape_data = create_shape_data(r)
                        if shape_data is not None:
                            shapes.append(shape_data)

                    # Setup interactive inspect mode with new rendering engine
                    manager = setup_interactive_inspect_mode(shapes, warped, window_name)
                    
                    print("\n[INSPECT MODE] Hover over shapes to preview, click to inspect.")
                    print("Press any key to resume scanning (ESC to exit).")
                    
                    while True:
                        k = cv2.waitKey(20) & 0xFF
                        if k != 255:  # any key pressed
                            if k == 27:  # ESC - exit application entirely
                                inspect_exit_flag = True
                                print("[INFO] ESC pressed - exiting application.")
                            else:
                                print("[INFO] Returning to scan mode.")
                            break
                    
                    # Cleanup interactive mode resources
                    try:
                        manager.cleanup()
                    except Exception as e:
                        print(f"[WARN] Error during manager cleanup: {e}")

            except Exception as e:
                print(f"[ERROR] Error in inspect mode: {e}")
                inspect_exit_flag = True
            finally:
                # Ensure inspect mode window is properly destroyed
                cleanup_resources(window_names=[window_name])
            
            # Exit application if ESC was pressed in inspect mode
            if inspect_exit_flag:
                break
            
            # Re-initialize camera and reset scan state for clean transition back to scan mode
            print("[INFO] Re-initializing camera for scan mode...")
            cap, tried_ip = reinitialize_camera(ip_base)
            if cap is None:
                print("[ERROR] Failed to re-initialize camera. Exiting.")
                break
            
            # Reset all scan state variables for clean transition
            stable_count, last_quad = reset_scan_state()
            print("[INFO] Scan state reset. Ready for new detection.")

    # Final cleanup
    cleanup_resources(cap=cap)
    
    # Cleanup detection resources
    from detection import cleanup_detection_resources, get_detection_performance_stats
    
    # Print performance stats if available
    perf_stats = get_detection_performance_stats()
    if perf_stats:
        print("\n[PERFORMANCE] Detection performance summary:")
        if "detection_processing" in perf_stats:
            dp = perf_stats["detection_processing"]
            print(f"  Average detection time: {dp['avg_ms']:.2f}ms")
            print(f"  Detection count: {dp['count']}")
        
        if "current_profile" in perf_stats:
            print(f"  Resolution profile used: {perf_stats['current_profile']}")
        
        if "gpu_available" in perf_stats:
            print(f"  GPU acceleration: {'Used' if perf_stats['gpu_available'] else 'Not available'}")
    
    cleanup_detection_resources()

if __name__ == "__main__":
    main()
