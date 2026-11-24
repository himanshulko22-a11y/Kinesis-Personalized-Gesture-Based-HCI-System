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
MOVEMENT_DATA_PATH = _path("movement_data.json")
LOG_PATH = _path("activity_log.txt")
SCREENSHOT_DIR = _path("screenshots")
SETTINGS_PATH = _path("settings.json")   # choices made in the app, e.g. the mistake hotkey
ICON_PATH = os.path.join(getattr(sys, "_MEIPASS", BASE_DIR), "favicon.ico")

DATA_FILES = [CSV_PATH, MODEL_PATH, ENCODER_PATH, MAPPING_PATH, MOVEMENT_DATA_PATH]
TRASH_CSV_PATH = os.path.join(BASE_DIR, "deleted_gestures.csv")          # deleted poses, for Undo / Restore
TRASH_MOVEMENT_PATH = os.path.join(BASE_DIR, "deleted_movements.json")  # deleted movements, for Undo / Restore
UNDO_SECONDS = 15             # how long the Undo link stays after a delete
TRASH_KEEP_DAYS = 30          # deleted gestures are removed for good after this many days
LITE_TRACKING = False         # True = MediaPipe's lighter, faster hand model (landmarks differ slightly)
BACKUP_PAIRS = [(path, _backup_of(path)) for path in DATA_FILES]  # (current file, saved default)

# Recognition
CONFIDENCE_THRESHOLD = 0.70   # minimum model confidence before an action fires
SMOOTHING_WINDOW = 5          # frames in a row that must agree on a gesture
CURSOR_AGREE_FRAMES = 3       # Move Cursor only needs this many of those frames (a wobble is harmless)
RELEASE_FRAMES = 6            # frames without a gesture before a one-shot action can fire again
ACTION_COOLDOWN = 1.0         # seconds before the same one-shot action can repeat
REPEAT_COOLDOWN = 0.3         # seconds between repeats of volume/scroll while held
FRAMES_PER_RECORDING = 150

# Training
LIVE_TEST_READY = 2.5         # seconds of countdown before each pose in the Live Test
LIVE_TEST_HOLD = 3.0          # seconds each pose is scored for
LIVE_TEST_PATH = os.path.join(BASE_DIR, "live_test_results.csv")   # Live Test results, for the report
LIVE_TEST_KEEP_RUNS = 2       # only the newest runs are kept in that file (the latest and the one before)
TEST_GAP_FRAMES = 15          # frames (~0.5 s) skipped between each gesture's training and test frames
TRAIN_WITH_AUGMENTATION = False   # starting state of the "Train with data augmentation" checkbox

# Similar-gesture warning (hand-shape distance, see gestures.similarity_warning)
SIMILAR_SAME = 0.6            # below this, a new gesture looks almost the same as an existing one
SIMILAR_CLOSE = 1.0           # below this, it is quite similar (two takes of one gesture are ~0.2-0.5)

# Learning from mistakes
MISTAKE_FRAMES = 10           # hand frames saved when an action is marked as a mistake
MISTAKE_WINDOW = 5.0          # seconds after an action during which it can be marked as a mistake
MISTAKE_HOTKEY = "Z"          # default; change it in the app (e.g. "F9", "Ctrl+Alt+Z")
MIN_MISTAKE_FRAMES = 20       # saved mistake frames needed before training uses them

# Movements (dynamic gestures, see movements.py). Speeds are in hand-sizes per second.
MOVEMENT_NEEDS_OPEN_HAND = True  # movements only count with an open hand (so pointing/cursor moves don't trigger them)
MOVE_OPEN_FINGERS = 4            # long fingers that must be stretched out for a hand to count as open
MOVE_OPEN_SHARE = 0.6            # share of a movement's frames in which the hand must be open
MOVE_SPEED_FRAMES = 4         # speed is measured over this many frames (averages out jitter)
MOVE_START_SPEED = 1.5        # hand faster than this -> a movement starts
MOVE_STOP_SPEED = 1.0         # slower than this for MOVE_STOP_FRAMES frames -> it has ended
MOVE_STOP_FRAMES = 6
MOVE_MIN_SECONDS = 0.12        # a fast flick can be this short
MOVE_STILL_SPEED = 1.2        # slower than this counts as a still hand
MOVE_REST_WINDOW = 15         # frames (about 0.5 s) looked at just before a movement starts...
MOVE_REST_FRAMES = 3          # ...of which this many must be still (a deliberate start, not casual motion)
TRAIL_SECONDS = 1.0           # how long a finished movement's trail stays on the video
MOVE_REFRACTORY = 0.7         # seconds after a movement during which new ones are ignored (the return stroke)
MOVE_MAX_GAP = 4              # frames the hand may vanish (motion blur) without losing the movement
MOVE_MAX_SECONDS = 2.5
MOVE_MIN_TRAVEL = 1.0         # hand-sizes the palm must travel (unless it moved towards/away)
MOVE_MIN_DEPTH_RATIO = 1.3    # hand size change that counts as a push/pull on its own
MOVE_DEPTH_WEIGHT = 3.0       # how much towards/away movement counts compared with sideways
MOVE_POINTS = 32              # every path is resampled to this many points
MOVE_DTW_BAND = 10            # how far DTW may stretch time (in points)
MOVE_SHORTEST_PART = 0.6      # also try the beginning of a movement, down to 60% of your shortest recording
MOVE_END_TRIES = 12           # how many different end points to try (the hand often comes straight back)
MOVE_RETURN_SHARE = 0.5       # ...but only cut off an ending that brings the hand at least half-way back
MOVE_RETURN_STRAIGHT = 1.4     # the return may be at most this x longer than a straight line (a circle's is ~1.6)
MOVE_REJECT_FACTOR = 4.0      # accept a movement up to this x the usual gap between your recordings...
MOVE_AMBIGUITY_RATIO = 0.8    # ...and only if the next-closest movement is clearly further away (25%+)
MOVE_REJECT_MIN = 0.08
MOVE_REJECT_DEFAULT = 0.35    # used until each movement has 2+ recordings
MOVE_COUNTDOWN = 3            # seconds before a movement recording starts
MOVE_RECORD_TIMEOUT = 8       # seconds to wait for a movement after the countdown

# Movement Live Test (see movetest.py)
MOVE_TEST_REPS = 10           # default number of attempts per movement (you can change it when starting)
MOVE_TEST_READY = 2.0         # seconds of countdown before each attempt (keep the hand still)
MOVE_TEST_TIMEOUT = 6.0       # seconds to make the movement after GO, then it counts as "not detected"
MOVE_TEST_SHOW = 1.2          # seconds each attempt's result stays on screen
MOVE_TEST_PATH = os.path.join(BASE_DIR, "movement_live_test_results.csv")   # Movement Test results, for the report
MOVE_TEST_KEEP_RUNS = 2       # only the newest runs are kept in that file

# Actions
VOLUME_PRESSES = 3
BRIGHTNESS_STEP = 5           # percent per step; keeps repeating while the gesture is held (like volume)
SCROLL_AMOUNT = 300

# Cursor: the part of the camera image that maps to the whole screen
ACTIVE_ZONE_X = (0.15, 0.85)
ACTIVE_ZONE_Y = (0.15, 0.65)
EDGE_BUFFER = 5               # pixels kept away from the screen edge

CURSOR_SENSITIVITY = 70       # starting value of the Cursor Sensitivity slider (1-100)

# Cursor smoothing (One Euro filter + hold zone, see CursorSmoother in gestures.py)
CURSOR_MIN_CUTOFF = 1.0       # lower = smoother but slower to follow slow movements
CURSOR_BETA = 0.01            # higher = less lag during fast movements
CURSOR_STILL_SPEED = 120      # pixels/second; slower than this counts as stopped
CURSOR_HOLD_RADIUS = 14       # pixels of hand shake a resting cursor ignores (bigger = steadier, needs a bigger move to start)

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
    "pinch": "left_click",
    "swipe_left": "prev_slide",
    "swipe_right": "next_slide",
    "circle_cw": "next_track",
    "circle_ccw": "prev_track",
    "wave": "show_desktop",
    "push": "play_pause",
    "pull": "mute",
}
