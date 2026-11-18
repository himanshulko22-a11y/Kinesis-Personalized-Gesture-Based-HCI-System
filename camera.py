"""Webcam reader.

Opens and reads the camera in the app's main thread, exactly like camera_test.py
(on Windows, using the camera from a background thread gave only black frames).
It tries the camera modes in config.py order, and if a mode turns black it
switches to another one by itself. What happens is written to the activity log.
"""
import sys
import time

import cv2

import config as cfg

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
BLACK = 5            # average brightness below this counts as a black frame
BLACK_SECONDS = 2.0  # this long of only-black frames -> try another camera mode


def open_camera(backend="dshow", mjpg=True, exposure=None):
    """Opens the webcam with the given settings (also used by camera_test.py)."""
    if sys.platform != "win32":
        backend = "any"   # dshow/msmf only exist on Windows
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX, BACKENDS[backend])
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # set before the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, cfg.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual mode for DirectShow
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    elif BACKENDS[backend] == cv2.CAP_DSHOW:
        # Webcams remember manual exposure between programs; always switch it back to automatic.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # 0.75 = automatic for DirectShow
    return cap


def reset_exposure(seconds=3.0):
    """Switches the webcam back to automatic exposure. Returns the picture brightness after."""
    cap = open_camera("dshow", mjpg=True, exposure=None)
    level = 0.0
    if cap.isOpened():
        start = time.time()
        while time.time() - start < seconds:   # give auto exposure time to adjust
            ok, frame = cap.read()
            if ok and frame is not None:
                level = brightness(frame)
    cap.release()
    time.sleep(1)
    return level


def brightness(frame):
    return float(frame[::16, ::16].mean())   # every 16th pixel is plenty and fast


def picture_brightness(cap, seconds=3.0):
    """Brightest picture seen in the first few seconds (0 = only black frames)."""
    best, start = 0.0, time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if ok and frame is not None:
            best = max(best, brightness(frame))
            if best > BLACK:
                break
    return best


class Camera:
    def __init__(self):
        self.fps = 0.0               # frames per second actually received
        self.backend = None          # camera mode in use, e.g. "msmf"
        self.messages = []           # notes for the app's activity log
        self.last_frame_time = time.time()
        self.cap = None
        self._frame_id = 0
        self._previous = time.perf_counter()
        self._black_since = None
        self._skip = set()
        self._switches = 0           # mode switches this run (limited, each one pauses the app briefly)
        self._open_best()

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def _open_best(self):
        """Opens the first camera mode that shows a picture."""
        if not self._skip and cfg.CAMERA_EXPOSURE is None and sys.platform == "win32":
            self.messages.append(f"Camera: automatic exposure on (brightness {reset_exposure(1.5):.0f}/255)")
        order = [cfg.CAMERA_BACKEND] + [b for b in ("msmf", "dshow") if b != cfg.CAMERA_BACKEND]
        order = [b for b in order if b not in self._skip] + [b for b in order if b in self._skip]
        for backend in order:
            cap = open_camera(backend, cfg.CAMERA_MJPG, cfg.CAMERA_EXPOSURE)
            if not cap.isOpened():
                cap.release()
                self.messages.append(f"Camera mode {backend}: could not open")
                time.sleep(1)
                continue
            level = picture_brightness(cap)
            self.messages.append(f"Camera mode {backend}: picture brightness {level:.0f}/255")
            if level > BLACK:
                self.cap, self.backend = cap, backend
                self.messages.append(f"Camera: using {backend}")
                return
            cap.release()        # release before trying the next mode, or it can't open
            time.sleep(1)
        self.cap = open_camera(cfg.CAMERA_BACKEND, cfg.CAMERA_MJPG, cfg.CAMERA_EXPOSURE)
        self.backend = cfg.CAMERA_BACKEND if self.cap.isOpened() else None
        self.messages.append("Camera: no mode showed a picture - is the camera covered, switched off "
                             "(privacy key/shutter), or in use by another app?")

    def latest(self, seen_id):
        """Reads the next frame. Returns (frame, frame_id), or (None, seen_id) if there is none."""
        if self.cap is None:
            return None, seen_id
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, seen_id
        now = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self._previous, 1e-3)
        self._previous = now
        self.last_frame_time = time.time()

        if brightness(frame) <= BLACK:
            self._black_since = self._black_since or now
            if now - self._black_since > BLACK_SECONDS and self._switches < 2:
                self.messages.append(f"Camera mode {self.backend} went black - switching mode")
                self._switches += 1
                self._skip.add(self.backend)
                self._black_since = None
                self.cap.release()
                time.sleep(1)
                self._open_best()
                return None, seen_id
        else:
            self._black_since = None

        self._frame_id += 1
        return frame, self._frame_id

    def release(self):
        if self.cap is not None:
            self.cap.release()
