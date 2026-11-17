"""What each gesture can do on the computer."""
import os
import time
from collections import namedtuple

import pyautogui

import config as cfg

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0   # the default 0.1 s pause after every call made the cursor stutter

# kind: "once"   fires once when the gesture starts; holding it does not repeat
#       "repeat" keeps firing while the gesture is held (volume, scroll)
#       "cursor" moves the mouse every frame while held
Action = namedtuple("Action", "label kind fn")


def _press(key, times=cfg.VOLUME_PRESSES):
    for _ in range(times):
        pyautogui.press(key)


def _screenshot():
    os.makedirs(cfg.SCREENSHOT_DIR, exist_ok=True)
    name = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
    pyautogui.screenshot(os.path.join(cfg.SCREENSHOT_DIR, name))


ACTIONS = {
    "none":               Action("None (unhooked)", "once", None),
    "left_click":         Action("Left Click", "once", lambda: pyautogui.click(button="left")),
    "right_click":        Action("Right Click", "once", lambda: pyautogui.click(button="right")),
    "double_click":       Action("Double Click", "once", pyautogui.doubleClick),
    "volume_up":          Action("Volume Up", "repeat", lambda: _press("volumeup")),
    "volume_down":        Action("Volume Down", "repeat", lambda: _press("volumedown")),
    "screenshot":         Action("Screenshot", "once", _screenshot),
    "move_cursor":        Action("Move Cursor", "cursor", None),
    "switch_window_next": Action("Switch Window (Next)", "once", lambda: pyautogui.hotkey("alt", "tab")),
    "switch_window_prev": Action("Switch Window (Previous)", "once", lambda: pyautogui.hotkey("alt", "shift", "tab")),
    "scroll_up":          Action("Scroll Up", "repeat", lambda: pyautogui.scroll(cfg.SCROLL_AMOUNT)),
    "scroll_down":        Action("Scroll Down", "repeat", lambda: pyautogui.scroll(-cfg.SCROLL_AMOUNT)),
}
