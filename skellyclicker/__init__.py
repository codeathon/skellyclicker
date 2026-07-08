
__url__ = "https://github.com/freemocap/skellyclicker"

import logging
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

PointNameString = str
VideoNameString = str
VideoPathString = str
ClickDataCSVPathString = str

MAX_WINDOW_SIZE = (1920, 1080)
# Scrub-preview composite cap (committed labeler uses native per-camera grid).
PREVIEW_MAX_WINDOW_SIZE = (1920, 1080)
LABELER_JPEG_QUALITY_COMMITTED = 92
LABELER_JPEG_QUALITY_PREVIEW = 65
# Train & Analyze: diverse performance-sample frames (1% of video, clamped).
PERF_SAMPLE_FRACTION = 0.01
PERF_SAMPLE_MIN_FRAMES = 50
PERF_SAMPLE_MAX_FRAMES = 200
# Labeler left panel: include machine-only nav when CSV is sparse (not full analyze).
MAX_NAV_MACHINE_FRAMES = 500
ZOOM_STEP = 1.1
ZOOM_MIN = 1.0
ZOOM_MAX = 10.0
POSITION_EPSILON = 1e-6  # Small threshold for position changes
