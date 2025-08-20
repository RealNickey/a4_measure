
# Tunable parameters for the A4 measurement pipeline

# Desired rendering density after warping the A4 sheet.
# Higher values = more pixels = potentially better accuracy but more compute.
PX_PER_MM = 6.0  # pixels per millimeter (A4 -> 1260 x 1782 px)

# A4 dimensions in millimeters
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

# Edge detection / contour tuning
CANNY_LOW = 50
CANNY_HIGH = 150
GAUSS_BLUR = 5  # kernel size for GaussianBlur

# Aspect ratio tolerance for A4 detection (portrait or landscape)
# A4 aspect = 297/210 ≈ 1.414
ASPECT_MIN = 1.25
ASPECT_MAX = 1.60

# Minimum area (portion of frame) for a valid A4 candidate
MIN_A4_AREA_RATIO = 0.08

# Stability: require this many consecutive frames with similar A4 corners + object presence
STABLE_FRAMES = 6
MAX_CORNER_JITTER = 12.0  # pixels allowed movement between frames for the A4 corners

# Object segmentation
BINARY_BLOCK_SIZE = 31  # for adaptive threshold; must be odd
BINARY_C = 7            # for adaptive threshold
MIN_OBJECT_AREA_MM2 = 300.0  # ignore tiny specks (in mm^2 after warp -> converted to px^2 inside code)

# Shape classification
CIRCULARITY_CUTOFF = 0.80  # > => circle-like
RECT_ANGLE_EPS_DEG = 12.0  # tolerance for right angle check

# Visualization
DRAW_FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
DRAW_THICKNESS = 2

# If CUDA is available in OpenCV, we can use GPU for Canny/resize.
USE_CUDA_IF_AVAILABLE = True
