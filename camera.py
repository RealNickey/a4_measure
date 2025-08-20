
import cv2

def build_ip_stream_url(base_url: str) -> str:
    base = base_url.rstrip('/')
    # If user gave a full path, honor it; else append /video
    if base.endswith(('/video', '/video/')) or base.count('/') > 2 and base.split('/')[-1] != '':
        return base  # assume full URL
    return base + '/video'

def open_capture(ip_base_url: str | None):
    # Try IP camera first (if provided), else fallback to webcam.
    cap = None
    tried_ip = False
    if ip_base_url:
        tried_ip = True
        stream_url = build_ip_stream_url(ip_base_url)
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            cap.release()
            cap = None

    if cap is None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(0)

    if not cap or not cap.isOpened():
        raise RuntimeError("Failed to open video source (IP cam and webcam both failed).")

    # Try to set a reasonable resolution
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Reduce buffer if supported
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass

    return cap, tried_ip
