"""What each gesture can do on the computer."""
import json
import os
import subprocess
import sys
import threading
import time
from collections import namedtuple

import pyautogui

import config as cfg

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0   # the default 0.1 s pause after every call made the cursor stutter

# kind: "once"      fires once when the gesture starts; holding it does not repeat
#       "repeat"    keeps firing while the gesture is held (volume, scroll)
#       "cursor"    moves the mouse every frame while held
#       "hold"      keeps a mouse button pressed while the gesture is held (fn = which button)
#       "hold_move" the same, and also moves the cursor - for one-handed drag and drop
Action = namedtuple("Action", "label kind fn")


def _press(key, times=cfg.VOLUME_PRESSES):
    for _ in range(times):
        pyautogui.press(key)


messages = []   # notes for the app's activity log (e.g. a brightness change that failed)

# Laptop brightness keys (like Fn+F12) are handled by the laptop's hardware, so pressing F12
# from a program doesn't change brightness. Windows' own brightness setting is changed instead.
_READ_BRIGHTNESS = ("(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness | "
                    "Select-Object -First 1).CurrentBrightness")
_SET_BRIGHTNESS = ("(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods | "
                   "Select-Object -First 1).WmiSetBrightness(1, {level}) | Out-Null; 'ok'")


def _powershell(command):
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, timeout=10, creationflags=0x08000000)  # no console window
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr.strip()[:150] or "no response")
    return result.stdout.strip().splitlines()[-1]


class _Brightness:
    """Changes brightness in the background. Holding a gesture asks for a step every ~0.3 s but
    each Windows call takes ~0.5 s, so the steps are added up and only the newest level is sent."""

    def __init__(self):
        self.lock = threading.Lock()
        self.level = None          # brightness Kinesis is aiming for
        self.pending = 0           # steps asked for but not sent yet
        self.busy = False
        self.last_change = 0.0

    def change(self, step):
        if sys.platform != "win32":
            messages.append("Brightness control only works on Windows")
            return
        with self.lock:
            self.pending += step
            if self.busy:
                return             # the running worker will pick this step up
            self.busy = True
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        try:
            if self.level is None or time.time() - self.last_change > 5:
                self.level = int(float(_powershell(_READ_BRIGHTNESS)))   # re-read: keys may have changed it
            while True:
                with self.lock:
                    if not self.pending:
                        self.busy = False
                        break
                    self.level = max(0, min(100, self.level + self.pending))
                    self.pending = 0
                    level = self.level
                _powershell(_SET_BRIGHTNESS.format(level=level))
                self.last_change = time.time()
            messages.append(f"Brightness {self.level}%")
        except Exception as e:
            with self.lock:
                self.busy, self.pending, self.level = False, 0, None
            messages.append("Brightness change failed (works on a laptop's built-in screen, "
                            f"not most external monitors): {e}")


_brightness = _Brightness()


def _change_brightness(step):
    """Laptop screen brightness up/down by `step` percent, in the background."""
    _brightness.change(step)


def _screenshot():
    os.makedirs(cfg.SCREENSHOT_DIR, exist_ok=True)
    name = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
    pyautogui.screenshot(os.path.join(cfg.SCREENSHOT_DIR, name))


# ---------- Mistake hotkey (Windows virtual-key codes, checked from any app) ----------

MODIFIER_KEYS = {"Ctrl": 0x11, "Alt": 0x12, "Shift": 0x10, "Meta": 0x5B}
NAMED_KEYS = {"Space": 0x20, "Esc": 0x1B, "Backspace": 0x08, "Tab": 0x09, "Return": 0x0D, "Enter": 0x0D,
              "Print": 0x2C, "Ins": 0x2D, "Del": 0x2E, "Home": 0x24, "End": 0x23,
              "PgUp": 0x21, "PgDown": 0x22, "Left": 0x25, "Up": 0x26, "Right": 0x27, "Down": 0x28,
              "Pause": 0x13, "ScrollLock": 0x91}


def parse_hotkey(text):
    """'Ctrl+Alt+Z' -> [0x11, 0x12, 0x5A]. Raises ValueError for keys it can't watch."""
    parts = [p for p in text.replace(" ", "").split("+") if p]
    if not parts:
        raise ValueError("No key given.")
    *modifiers, key = parts
    codes = []
    for m in modifiers:
        if m not in MODIFIER_KEYS:
            raise ValueError(f"'{m}' can't be used as a modifier.")
        codes.append(MODIFIER_KEYS[m])
    if len(key) == 1 and key.upper().isalnum():
        codes.append(ord(key.upper()))                 # letters and digits
    elif key.upper().startswith("F") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        codes.append(0x70 + int(key[1:]) - 1)          # F1-F24
    elif key in NAMED_KEYS:
        codes.append(NAMED_KEYS[key])
    else:
        raise ValueError(f"'{key}' isn't supported - use a letter, digit, F1-F24, or a key like "
                         f"Space, Backspace, Tab, Enter, Home or Del.")
    return codes


def hotkey_pressed(codes):
    """True while exactly this combination is held (other modifiers must be up, so 'Z' doesn't fire on Ctrl+Z)."""
    import ctypes
    state = ctypes.windll.user32.GetAsyncKeyState
    if not all(state(code) & 0x8000 for code in codes):
        return False
    return not any(state(m) & 0x8000 for m in MODIFIER_KEYS.values() if m not in codes)


def load_settings():
    try:
        with open(cfg.SETTINGS_PATH) as f:
            settings = json.load(f)
        return settings if isinstance(settings, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    with open(cfg.SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


ACTIONS = {
    "none":               Action("None (unhooked)", "once", None),
    "left_click":         Action("Left Click", "once", lambda: pyautogui.click(button="left")),
    "right_click":        Action("Right Click", "once", lambda: pyautogui.click(button="right")),
    "double_click":       Action("Double Click", "once", pyautogui.doubleClick),
    "volume_up":          Action("Volume Up", "repeat", lambda: _press("volumeup")),
    "volume_down":        Action("Volume Down", "repeat", lambda: _press("volumedown")),
    "screenshot":         Action("Screenshot", "once", _screenshot),
    "move_cursor":        Action("Move Cursor", "cursor", None),
    "hold_left":          Action("Hold Left Click", "hold", "left"),
    "hold_right":         Action("Hold Right Click", "hold", "right"),
    "drag":               Action("Drag (hold left click + move cursor)", "hold_move", "left"),
    "switch_window_next": Action("Switch Window (Next)", "once", lambda: pyautogui.hotkey("alt", "tab")),
    "switch_window_prev": Action("Switch Window (Previous)", "once", lambda: pyautogui.hotkey("alt", "shift", "tab")),
    "scroll_up":          Action("Scroll Up", "repeat", lambda: pyautogui.scroll(cfg.SCROLL_AMOUNT)),
    "scroll_down":        Action("Scroll Down", "repeat", lambda: pyautogui.scroll(-cfg.SCROLL_AMOUNT)),
    "mark_mistake":       Action("Mark Last Action as Mistake", "once", None),   # handled by the app
    # Ready-made keyboard shortcuts
    "play_pause":         Action("Media: Play / Pause", "once", lambda: pyautogui.press("playpause")),
    "next_track":         Action("Media: Next Track", "once", lambda: pyautogui.press("nexttrack")),
    "prev_track":         Action("Media: Previous Track", "once", lambda: pyautogui.press("prevtrack")),
    "mute":               Action("Mute / Unmute", "once", lambda: pyautogui.press("volumemute")),
    "brightness_up":      Action("Brightness Up", "repeat", lambda: _change_brightness(cfg.BRIGHTNESS_STEP)),
    "brightness_down":    Action("Brightness Down", "repeat", lambda: _change_brightness(-cfg.BRIGHTNESS_STEP)),
    "next_slide":         Action("Next Slide (Right arrow)", "once", lambda: pyautogui.press("right")),
    "prev_slide":         Action("Previous Slide (Left arrow)", "once", lambda: pyautogui.press("left")),
    "show_desktop":       Action("Show Desktop (Win+D)", "once", lambda: pyautogui.hotkey("win", "d")),
    "task_view":          Action("Task View (Win+Tab)", "once", lambda: pyautogui.hotkey("win", "tab")),
    "copy":               Action("Copy (Ctrl+C)", "once", lambda: pyautogui.hotkey("ctrl", "c")),
    "paste":              Action("Paste (Ctrl+V)", "once", lambda: pyautogui.hotkey("ctrl", "v")),
    "undo":               Action("Undo (Ctrl+Z)", "once", lambda: pyautogui.hotkey("ctrl", "z")),
    "enter":              Action("Enter", "once", lambda: pyautogui.press("enter")),
    "escape":             Action("Escape", "once", lambda: pyautogui.press("esc")),
}


# ---------- Custom keyboard shortcuts (made in the app, saved in settings.json) ----------

SHORTCUT_PREFIX = "shortcut:"
QT_TO_PYAUTOGUI = {"Ctrl": "ctrl", "Alt": "alt", "Shift": "shift", "Meta": "win", "Space": "space",
                   "Esc": "esc", "Return": "enter", "Enter": "enter", "Tab": "tab", "Backspace": "backspace",
                   "Ins": "insert", "Del": "delete", "Home": "home", "End": "end", "PgUp": "pageup",
                   "PgDown": "pagedown", "Left": "left", "Right": "right", "Up": "up", "Down": "down",
                   "Print": "printscreen", "Pause": "pause"}


def shortcut_keys(text):
    """'Ctrl+Shift+S' -> ['ctrl', 'shift', 's'] for pyautogui. Raises ValueError for unknown keys."""
    parts = [p for p in text.replace(" ", "").split("+") if p] if text != "+" else ["+"]
    if not parts:
        raise ValueError("No keys given.")
    keys = []
    for part in parts:
        if part in QT_TO_PYAUTOGUI:
            keys.append(QT_TO_PYAUTOGUI[part])
        elif part.upper().startswith("F") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
            keys.append(part.lower())
        elif len(part) == 1 and part.lower() in pyautogui.KEYBOARD_KEYS:
            keys.append(part.lower())
        else:
            raise ValueError(f"'{part}' isn't a key Kinesis can press.")
    return keys


def add_shortcut(text):
    """Registers 'Ctrl+C' etc. as an action. Returns its action key."""
    keys = shortcut_keys(text)
    key = SHORTCUT_PREFIX + text
    ACTIONS[key] = Action(f"Shortcut: {text}", "once", lambda: pyautogui.hotkey(*keys))
    return key


def remove_shortcut(text):
    ACTIONS.pop(SHORTCUT_PREFIX + text, None)


def custom_shortcuts():
    return [key[len(SHORTCUT_PREFIX):] for key in ACTIONS if key.startswith(SHORTCUT_PREFIX)]
