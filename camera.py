
import cv2

def build_ip_stream_url(base_url: str) -> str:
    base = base_url.rstrip('/')
    # If user gave a full path, honor it; else append /video
    if base.endswith(('/video', '/video/')) or base.count('/') > 2 and base.split('/')[-1] != '':
        return base  # assume full URL
    return base + '/video'

def open_capture(ip_base_url: str | None, prefer_high_resolution: bool = False):
    """
    Open video capture with optional high-resolution support.
    
    Args:
        ip_base_url: IP camera base URL or None for webcam
        prefer_high_resolution: Whether to prefer high resolution if available
        
    Returns:
        Tuple of (capture, tried_ip)
    """
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

    # Configure resolution based on preference
    try:
        if prefer_high_resolution:
            # Try to set high resolution first
            resolutions_to_try = [
                (3840, 2160),  # 4K
                (2560, 1440),  # 1440p
                (1920, 1080),  # 1080p
                (1280, 720)    # 720p fallback
            ]
        else:
            # Standard resolution preference
            resolutions_to_try = [
                (1920, 1080),  # 1080p
                (1280, 720),   # 720p
                (1024, 768),   # XGA
                (640, 480)     # VGA fallback
            ]
        
        # Try resolutions in order
        for width, height in resolutions_to_try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify the resolution was set
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if actual_width >= width * 0.9 and actual_height >= height * 0.9:
                print(f"[INFO] Set resolution to {int(actual_width)}x{int(actual_height)}")
                break
        
        # Set other properties
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Reduce buffer for lower latency
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Enable auto-exposure and auto-focus if available
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # Auto focus
        except Exception:
            pass
            
    except Exception as e:
        print(f"[WARN] Failed to configure camera properties: {e}")

    return cap, tried_ip

def get_camera_info(cap):
    """
    Get camera information and capabilities.
    
    Args:
        cap: OpenCV VideoCapture object
        
    Returns:
        Dictionary with camera information
    """
    if not cap or not cap.isOpened():
        return {}
    
    info = {}
    
    try:
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        info['fourcc'] = int(cap.get(cv2.CAP_PROP_FOURCC))
        info['buffer_size'] = cap.get(cv2.CAP_PROP_BUFFERSIZE)
        
        # Calculate megapixels
        info['megapixels'] = (info['width'] * info['height']) / 1_000_000
        
        # Determine resolution category
        if info['width'] >= 3840:
            info['category'] = '4K+'
        elif info['width'] >= 2560:
            info['category'] = '1440p'
        elif info['width'] >= 1920:
            info['category'] = '1080p'
        elif info['width'] >= 1280:
            info['category'] = '720p'
        else:
            info['category'] = 'SD'
            
    except Exception as e:
        print(f"[WARN] Error getting camera info: {e}")
    
    return info

def optimize_camera_for_detection(cap):
    """
    Optimize camera settings for A4 detection.
    
    Args:
        cap: OpenCV VideoCapture object
    """
    if not cap or not cap.isOpened():
        return
    
    try:
        # Optimize for detection accuracy
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # Slightly underexpose for better edge detection
        cap.set(cv2.CAP_PROP_CONTRAST, 1.2)       # Increase contrast
        cap.set(cv2.CAP_PROP_SHARPNESS, 1.1)      # Increase sharpness
        cap.set(cv2.CAP_PROP_SATURATION, 0.8)     # Reduce saturation for better grayscale conversion
        
        print("[INFO] Optimized camera settings for A4 detection")
        
    except Exception as e:
        print(f"[WARN] Could not optimize camera settings: {e}")
