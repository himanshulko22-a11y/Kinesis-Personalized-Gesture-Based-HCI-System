"""All settings and file paths for Kinesis. Tweak the numbers here."""
import os
import sys

# Data files are kept next to the app, whichever folder it is started from.
_frozen = getattr(sys, "frozen", False)  # True when built into an .exe with PyInstaller
BASE_DIR = os.path.dirname(os.path.abspath(sys.executable if _frozen else __file__))


def _path(name):
    return os.path.join(BASE_DIR, name)


def _backup_of(path):
    root, ext = os.path.splitext(path)
    return f"{root}_backup{ext}"


CSV_PATH = _path("gesture_data.csv")
MODEL_PATH = _path("gesture_model.keras")
ENCODER_PATH = _path("label_encoder.pkl")
MAPPING_PATH = _path("gesture_mapping.json")
LOG_PATH = _path("activity_log.txt")
SCREENSHOT_DIR = _path("screenshots")
ICON_PATH = os.path.join(getattr(sys, "_MEIPASS", BASE_DIR), "favicon.ico")

DATA_FILES = [CSV_PATH, MODEL_PATH, ENCODER_PATH, MAPPING_PATH]
BACKUP_PAIRS = [(path, _backup_of(path)) for path in DATA_FILES]  # (current file, saved default)

# Recognition
CONFIDENCE_THRESHOLD = 0.70   # minimum model confidence before an action fires
SMOOTHING_WINDOW = 5          # frames in a row that must agree on a gesture
RELEASE_FRAMES = 6            # frames without a gesture before a one-shot action can fire again
ACTION_COOLDOWN = 1.0         # seconds before the same one-shot action can repeat
REPEAT_COOLDOWN = 0.3         # seconds between repeats of volume/scroll while held
FRAMES_PER_RECORDING = 150

# Actions
VOLUME_PRESSES = 3
SCROLL_AMOUNT = 300

# Cursor: the part of the camera image that maps to the whole screen
ACTIVE_ZONE_X = (0.15, 0.85)
ACTIVE_ZONE_Y = (0.15, 0.65)
EDGE_BUFFER = 5               # pixels kept away from the screen edge

# Cursor smoothing (One Euro filter + hold zone, see CursorSmoother in gestures.py)
CURSOR_MIN_CUTOFF = 1.0       # lower = smoother but slower to follow slow movements
CURSOR_BETA = 0.01            # higher = less lag during fast movements
CURSOR_STILL_SPEED = 120      # pixels/second; slower than this counts as stopped
CURSOR_HOLD_RADIUS = (4, 24)  # pixels a still cursor ignores, for steadiness slider 0..100
CURSOR_STEADINESS = 50        # default slider value

# Two-hand zoom
ZOOM_DEADZONE = 0.0015        # hand-distance changes smaller than this are noise
ZOOM_SENSITIVITY = 6000       # turns hand-distance change into a scroll amount
ZOOM_MAX_STEP = 150           # largest scroll sent in one frame

# Idle lock
IDLE_LOCK_TIMEOUT = 30        # seconds with no hand before the app locks
IDLE_LOCK_WARNING = 10
UNLOCK_HOLD_SECONDS = 1.5

# Activity log
LOG_MAX_ENTRIES = 50
LOG_MAX_FILE_BYTES = 200_000

FRAME_INTERVAL_MS = 15        # the loop waits for each new camera frame anyway
CAMERA_INDEX = 0
CAMERA_FPS = 30
# Run camera_test.py to find the fastest settings for your webcam.
CAMERA_BACKEND = "msmf"       # "dshow", "msmf" or "any" (msmf: ~30 fps on this webcam vs ~8 with dshow)
CAMERA_MJPG = True            # MJPG mode lets many webcams reach full frame rate
CAMERA_EXPOSURE = None        # None = automatic; a number like -6 = fixed exposure (faster in dim light, darker image)

DEFAULT_MAPPING = {
    "fist": "double_click",
    "open_palm": "none",
    "peace": "screenshot",
    "thumbs_up": "volume_up",
    "thumbs_down": "volume_down",
    "point": "move_cursor",
    "ok_sign": "right_click",
}
