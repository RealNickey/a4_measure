
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

# Interactive hit testing configuration
HOVER_SNAP_DISTANCE_MM = 10.0  # Distance threshold for hover snapping (in mm)
PREVIEW_COLOR = (0, 200, 200)  # Color for hover preview outlines (cyan)
SELECTION_COLOR = (0, 255, 0)  # Color for selected shape rendering (green)

# Manual selection configuration
MIN_SELECTION_SIZE_PX = 20  # Minimum selection rectangle size in pixels
MAX_SELECTION_AREA_RATIO = 0.8  # Maximum shape area as ratio of selection area
SELECTION_SNAP_DISTANCE_PX = 15  # Distance threshold for shape snapping in pixels
MANUAL_SELECTION_TIMEOUT_MS = 5000  # Timeout for manual selection operations in milliseconds

# Enhanced contour analysis parameters
ENHANCED_GAUSSIAN_BLOCK_SIZE = 31  # Block size for adaptive Gaussian thresholding (must be odd)
ENHANCED_GAUSSIAN_C = 7.0  # Constant subtracted from mean in adaptive thresholding
ENHANCED_MIN_CONTOUR_AREA = 100  # Minimum contour area for enhanced analysis
ENHANCED_MORPHOLOGY_KERNEL_SIZE = 3  # Kernel size for morphological operations

# Shape detection thresholds
MIN_CIRCULARITY_THRESHOLD = 0.6  # Minimum circularity for circle detection (0.0-1.0)
MIN_RECTANGULARITY_THRESHOLD = 0.7  # Minimum rectangularity for rectangle detection (0.0-1.0)
SHAPE_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for shape acceptance (0.0-1.0)

# Error handling and fallback parameters
MAX_SELECTION_RETRIES = 3  # Maximum number of selection retry attempts
FALLBACK_TO_STANDARD_THRESHOLD = True  # Enable fallback to standard thresholding
ERROR_MESSAGE_DISPLAY_TIME_MS = 3000  # Duration to display error messages in milliseconds
ENABLE_SELECTION_VALIDATION = True  # Enable comprehensive selection validation

# Detection accuracy improvements
ENABLE_SUBPIXEL_REFINEMENT = True  # Enable sub-pixel corner refinement for A4 detection
MIN_DETECTION_QUALITY = 0.6  # Minimum quality score for A4 detection (0.0-1.0)
MULTI_FRAME_CALIBRATION_SAMPLES = 5  # Number of frames to sample for multi-frame calibration
CALIBRATION_QUALITY_THRESHOLD = 0.7  # Quality threshold for accepting calibration frames
