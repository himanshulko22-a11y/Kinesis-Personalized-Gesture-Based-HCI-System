"""Kinesis Receiver - lets Kinesis on another PC control THIS PC over local Wi-Fi or a hotspot.

Run on the PC you want to control:   python receiver.py
(or build it into an .exe:            pyinstaller --onefile --windowed receiver.py)

No camera and no machine learning here: this small window just
  1. listens on the local network (TCP port 50607, encrypted with TLS 1.2+),
  2. shows this PC's IP address and a random 6-digit pairing code,
  3. accepts ONE Kinesis PC that sends the right code,
  4. performs only the actions you allow below (an allow-list), using PyAutoGUI,
  5. logs every action it received, and stops the moment you press "Stop receiving".

Needs: PyQt5 and PyAutoGUI (pip install PyQt5 PyAutoGUI). Shares remote.py with Kinesis.
Windows Firewall asks the first time it runs: tick "Private networks" (and "Public" if your
hotspot is set as a public network), then click Allow.
"""
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time

import pyautogui
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget,
                             QPushButton, QVBoxLayout, QWidget)

from remote import DEFAULT_PORT, PROTOCOL_VERSION, LineReader, encode, local_ip_addresses, server_tls_context, short_fingerprint

pyautogui.FAILSAFE = False   # Kinesis may move the cursor into a corner on purpose
pyautogui.PAUSE = 0          # no 0.1 s pause after every action (that made the cursor stutter)

MAX_WRONG_CODES = 5          # after this many wrong codes from one IP, it's blocked until a new code is made
WRONG_CODE_DELAY = 1.0       # seconds to wait before answering a wrong code (slows down guessing)
SILENCE_LIMIT = 10.0         # Kinesis pings every 2 s; if it's silent this long, it's treated as gone
VOLUME_PRESSES = 3           # same as Kinesis' config.py
SCROLL_AMOUNT = 300
BRIGHTNESS_STEP = 5
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(
    sys.executable if getattr(sys, "frozen", False) else __file__)), "screenshots")


# ---------- What this PC is willing to do (the allow-list) ----------

def _press(key, times=1):
    for _ in range(times):
        pyautogui.press(key)


def _screenshot():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    pyautogui.screenshot(os.path.join(SCREENSHOT_DIR, time.strftime("screenshot_%Y%m%d_%H%M%S.png")))


class Brightness:
    """Screen brightness up/down on Windows, Linux and macOS. Each system has its own way, so the
    first one that works is used (and remembered). If none works, the error explains what to install."""

    def __init__(self):
        self.lock = threading.Lock()   # two quick gestures must not change the level at the same time
        self.method = None             # the way that worked last time
        self.failed = set()            # ways that didn't work on this PC (not tried again)
        self.xrandr_level = None       # Linux software dimming: the level we set (1.0 = normal)

    def change(self, step):
        """step = percent, e.g. +5 or -5. Raises RuntimeError with a clear message if nothing works."""
        with self.lock:
            if sys.platform == "win32":
                ways = [self._windows]
            elif sys.platform == "darwin":
                ways = [self._mac_cli, self._mac_keys]
            else:
                ways = [self._brightnessctl, self._xrandr]
            if self.method in ways:
                ways.remove(self.method)
                ways.insert(0, self.method)   # try what worked last time first
            problems = []
            for way in ways:
                if way in self.failed:
                    continue
                try:
                    result = way(step)
                    self.method = way
                    return result
                except Exception as e:
                    self.failed.add(way)
                    problems.append(str(e))
            self.failed.clear()   # let the user fix it (e.g. install a tool) and try again
            raise RuntimeError("brightness can't be changed here: " + "; ".join(problems))

    @staticmethod
    def _run(command):
        """Runs a command and returns its output. Raises RuntimeError if it fails."""
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"{command[0]}: {e}")
        if result.returncode != 0:
            raise RuntimeError(f"{command[0]}: {(result.stderr or result.stdout).strip()[:100] or 'failed'}")
        return result.stdout

    # ----- Windows: the laptop screen's own setting (same idea as Kinesis' actions.py) -----
    @staticmethod
    def _windows(step):
        command = ("$m = Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness | Select-Object -First 1; "
                   f"$level = [Math]::Max(0, [Math]::Min(100, $m.CurrentBrightness + ({step}))); "
                   "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods | Select-Object -First 1)"
                   ".WmiSetBrightness(1, $level) | Out-Null")
        subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                         creationflags=0x08000000)   # in the background, no console window
        return None

    # ----- Linux 1: brightnessctl changes the real backlight (laptops) -----
    def _brightnessctl(self, step):
        if not shutil.which("brightnessctl"):
            raise RuntimeError("brightnessctl not installed (sudo apt install brightnessctl)")
        self._run(["brightnessctl", "-q", "set", f"{abs(step)}%{'+' if step > 0 else '-'}"])
        return "backlight"

    # ----- Linux 2: xrandr dims the picture in software (works in a VM and on monitors) -----
    def _xrandr(self, step):
        if not shutil.which("xrandr"):
            raise RuntimeError("xrandr not installed (sudo apt install x11-xserver-utils)")
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("xrandr needs an X11 desktop (DISPLAY isn't set; Wayland isn't supported)")
        info = self._run(["xrandr", "--verbose"])
        outputs, current = [], None
        for line in info.splitlines():
            if " connected" in line and not line.startswith(" "):
                outputs.append(line.split()[0])           # e.g. "Virtual-1" or "HDMI-1"
            elif line.strip().startswith("Brightness:") and current is None:
                current = float(line.split(":")[1])
        if not outputs:
            raise RuntimeError("xrandr found no connected screen")
        if self.xrandr_level is None:
            self.xrandr_level = current if current is not None else 1.0
        # Keep at least 20%, so the screen never goes completely black.
        self.xrandr_level = round(max(0.2, min(1.0, self.xrandr_level + step / 100)), 2)
        for output in outputs:
            self._run(["xrandr", "--output", output, "--brightness", str(self.xrandr_level)])
        return f"{self.xrandr_level:.0%}, software dimming"

    # ----- macOS 1: the free `brightness` tool (brew install brightness) -----
    def _mac_cli(self, step):
        if not shutil.which("brightness"):
            raise RuntimeError("'brightness' tool not installed (brew install brightness)")
        listing = self._run(["brightness", "-l"])
        levels = [float(line.rsplit(" ", 1)[1]) for line in listing.splitlines() if "brightness " in line]
        if not levels:
            raise RuntimeError("brightness: no display found")
        level = max(0.0, min(1.0, levels[0] + step / 100))
        self._run(["brightness", f"{level:.2f}"])
        return f"{level:.0%}"

    # ----- macOS 2: press the brightness keys (needs Accessibility permission for Terminal/Python) -----
    def _mac_keys(self, step):
        key_code = 144 if step > 0 else 145   # Apple's brightness up / down keys
        self._run(["osascript", "-e", f'tell application "System Events" to key code {key_code}'])
        return None


_brightness = Brightness()


# Each group is one tick box in the window. key -> (label shown in the log, what to do)
GROUPS = [
    ("clicks", "Mouse clicks, hold-click and drag", True, {
        "left_click": ("Left Click", lambda: pyautogui.click(button="left")),
        "right_click": ("Right Click", lambda: pyautogui.click(button="right")),
        "double_click": ("Double Click", pyautogui.doubleClick),
    }),
    ("cursor", "Move the cursor", True, {}),
    ("scroll", "Scroll and zoom", True, {
        "scroll_up": ("Scroll Up", lambda: pyautogui.scroll(SCROLL_AMOUNT)),
        "scroll_down": ("Scroll Down", lambda: pyautogui.scroll(-SCROLL_AMOUNT)),
    }),
    ("media", "Volume and media keys", True, {
        "volume_up": ("Volume Up", lambda: _press("volumeup", VOLUME_PRESSES)),
        "volume_down": ("Volume Down", lambda: _press("volumedown", VOLUME_PRESSES)),
        "mute": ("Mute / Unmute", lambda: _press("volumemute")),
        "play_pause": ("Play / Pause", lambda: _press("playpause")),
        "next_track": ("Next Track", lambda: _press("nexttrack")),
        "prev_track": ("Previous Track", lambda: _press("prevtrack")),
    }),
    ("slides", "Slides, windows and Enter/Esc", True, {
        "next_slide": ("Next Slide", lambda: _press("right")),
        "prev_slide": ("Previous Slide", lambda: _press("left")),
        "enter": ("Enter", lambda: _press("enter")),
        "escape": ("Escape", lambda: _press("esc")),
        "switch_window_next": ("Switch Window (Next)", lambda: pyautogui.hotkey("alt", "tab")),
        "switch_window_prev": ("Switch Window (Previous)", lambda: pyautogui.hotkey("alt", "shift", "tab")),
        "show_desktop": ("Show Desktop", lambda: pyautogui.hotkey("win", "d")),
        "task_view": ("Task View", lambda: pyautogui.hotkey("win", "tab")),
    }),
    ("editing", "Copy, paste and undo", True, {
        "copy": ("Copy", lambda: pyautogui.hotkey("ctrl", "c")),
        "paste": ("Paste", lambda: pyautogui.hotkey("ctrl", "v")),
        "undo": ("Undo", lambda: pyautogui.hotkey("ctrl", "z")),
    }),
    ("screenshot", "Screenshots (saved on this PC)", True, {
        "screenshot": ("Screenshot", _screenshot),
    }),
    ("brightness", "Screen brightness", True, {
        "brightness_up": ("Brightness Up", lambda: _brightness.change(BRIGHTNESS_STEP)),
        "brightness_down": ("Brightness Down", lambda: _brightness.change(-BRIGHTNESS_STEP)),
    }),
    # Off by default: a custom shortcut can be any key combination (even Alt+F4).
    ("shortcuts", "Custom keyboard shortcuts (any key combination)", False, {}),
]
ACTION_GROUP = {key: group for group, _, _, actions in GROUPS for key in actions}
ACTION_INFO = {key: info for _, _, _, actions in GROUPS for key, info in actions.items()}
MOUSE_BUTTONS = ("left", "right")


# ---------- Network part (runs in background threads) ----------

class Receiver(QObject):
    """Listens for Kinesis, checks the pairing code and performs allowed actions.
    Talks to the window only through Qt signals, which are safe to send from any thread."""
    log = pyqtSignal(str)
    status = pyqtSignal(str, str)     # text, colour name: "idle" | "waiting" | "connected" | "error"

    def __init__(self, port=DEFAULT_PORT):
        super().__init__()
        self.port = port
        self.code = self.new_code()
        self.allowed = {group for group, _, default_on, _ in GROUPS if default_on}
        self.server = None
        self.accept_thread = None
        self.running = False
        self.controller = None          # (socket, name, ip) of the Kinesis PC in control, if any
        self.wrong_codes = {}           # ip -> wrong codes sent
        self.held = set()               # mouse buttons Kinesis is holding down right now
        self.lock = threading.Lock()

    @staticmethod
    def new_code():
        return f"{secrets.randbelow(1_000_000):06d}"

    def refresh_code(self):
        self.code = self.new_code()
        self.wrong_codes.clear()        # a new code gives blocked PCs another chance
        return self.code

    # ----- start / stop -----

    def start(self):
        if self.accept_thread is not None:
            self.accept_thread.join(timeout=2.0)   # the previous run lets go of the port first
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform != "win32":
            # Linux/macOS: allow restarting straight away. (On Windows this option would let another
            # program share the port, so it is left off; Windows allows the restart anyway.)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("0.0.0.0", self.port))   # 0.0.0.0 = every network this PC is on
            server.listen(2)
            server.settimeout(0.5)        # wake up twice a second to see if we should stop
        except OSError as e:
            server.close()
            self.status.emit(f"Can't listen on port {self.port}: {e.strerror or e}. "
                             "Is another receiver already open?", "error")
            return False
        self.server, self.running = server, True
        self.accept_thread = threading.Thread(target=self._accept_loop, args=(server,), daemon=True)
        self.accept_thread.start()
        self.status.emit("Waiting for Kinesis to connect...", "waiting")
        self.log.emit(f"Started receiving on port {self.port}")
        return True

    def stop(self):
        self.running = False
        with self.lock:
            controller = self.controller
        if controller is not None:
            try:   # cut the link; the connection's own thread then closes it (TLS sockets aren't shared)
                socket.socket.shutdown(controller[0], socket.SHUT_RDWR)
            except OSError:
                pass
            controller = None
        if self.accept_thread is not None:
            self.accept_thread.join(timeout=2.0)   # it closes the listening socket as it ends
        self.server = None
        self._release_buttons()
        self.status.emit("Stopped - nobody can control this PC.", "idle")
        self.log.emit("Stopped receiving")

    @staticmethod
    def _close(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    # ----- connections -----

    def _accept_loop(self, server):
        try:
            while self.running:
                try:
                    sock, (ip, _) = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not self.running:
                    self._close(sock)
                    break
                threading.Thread(target=self._serve_secure, args=(sock, ip), daemon=True).start()
        finally:
            server.close()                  # stop listening: nobody else can connect

    def _serve(self, sock, ip):
        """Handles one incoming connection from start to end."""
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        reader = LineReader()
        try:
            hello = self._read_hello(sock, reader)
            if hello is None:
                return
            if not self._check_pairing(sock, ip, hello):
                return
            name = str(hello.get("name", ip))[:40]
            sock.sendall(encode({"type": "welcome", "name": socket.gethostname(),
                                 "version": PROTOCOL_VERSION}))
            self.status.emit(f"Controlled by {name} ({ip})", "connected")
            self.log.emit(f"{name} ({ip}) connected - it can now control this PC")
            self._handle_messages(sock, reader)
            self.log.emit(f"{name} ({ip}) disconnected")
        except (OSError, ValueError) as e:
            if self.running:
                self.log.emit(f"Connection from {ip} ended: {e}")
            else:
                self.log.emit(f"{ip} disconnected (receiving stopped)")
        finally:
            with self.lock:
                was_controller = self.controller is not None and self.controller[0] is sock
                if was_controller:
                    self.controller = None
            if was_controller:
                self._release_buttons()     # never leave a mouse button stuck down
                if self.running:
                    self.status.emit("Waiting for Kinesis to connect...", "waiting")
            self._close(sock)

    def _read_hello(self, sock, reader):
        sock.settimeout(5.0)
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    return None
                messages = reader.feed(data)
                if messages:
                    return messages[0]
        except socket.timeout:
            return None

    def _check_pairing(self, sock, ip, hello):
        """Returns True if this connection may take control."""
        def refuse(reason):
            sock.sendall(encode({"type": "refused", "reason": reason}))
            return False

        if hello.get("type") != "hello":
            return refuse("Not a Kinesis connection.")
        if self.wrong_codes.get(ip, 0) >= MAX_WRONG_CODES:
            self.log.emit(f"Blocked {ip}: too many wrong codes (click 'New code' to let it try again)")
            return refuse("Too many wrong pairing codes. Ask the receiver's owner for a new code.")
        if not secrets.compare_digest(str(hello.get("code", "")).strip().encode(), self.code.encode()):
            self.wrong_codes[ip] = self.wrong_codes.get(ip, 0) + 1
            self.log.emit(f"Wrong pairing code from {ip} ({self.wrong_codes[ip]}/{MAX_WRONG_CODES})")
            time.sleep(WRONG_CODE_DELAY)
            return refuse("Wrong pairing code - check the 6 digits in the receiver window.")
        with self.lock:
            if self.controller is not None:
                busy_with = self.controller[1]
            else:
                busy_with = None
                self.controller = (sock, str(hello.get("name", ip))[:40], ip)
        if busy_with is not None:
            self.log.emit(f"Refused {ip}: already controlled by {busy_with}")
            return refuse(f"This PC is already being controlled by {busy_with}.")
        self.wrong_codes.pop(ip, None)
        return True

    def _handle_messages(self, sock, reader):
        sock.settimeout(1.0)
        last_heard = time.time()
        while self.running:
            try:
                data = sock.recv(65536)
            except socket.timeout:
                if time.time() - last_heard > SILENCE_LIMIT:
                    self.log.emit("Kinesis went silent - disconnecting")
                    return
                continue
            if not data:
                return                       # Kinesis closed the connection
            last_heard = time.time()
            messages = reader.feed(data)
            # If several cursor positions arrived together, only the newest matters.
            cursor_moves = [m for m in messages if m.get("type") == "cursor"]
            for message in messages:
                kind = message.get("type")
                if kind == "bye":
                    return
                if kind == "cursor" and message is not cursor_moves[-1]:
                    continue
                reply = self._perform(message)
                if reply is not None:
                    sock.sendall(encode(reply))

    # ----- doing the actions -----

    def _perform(self, message):
        """Runs one message from Kinesis if it is allowed. Returns the reply to send (or None)."""
        kind, msg_id = message.get("type"), message.get("id")
        gesture = str(message.get("gesture") or "")[:40]
        if kind == "ping":
            return {"type": "pong"}
        if kind == "cursor":
            if "cursor" in self.allowed:
                try:
                    width, height = pyautogui.size()
                    x = min(max(float(message["x"]), 0.0), 1.0) * (width - 1)
                    y = min(max(float(message["y"]), 0.0), 1.0) * (height - 1)
                    pyautogui.moveTo(int(x), int(y))
                except (KeyError, TypeError, ValueError):
                    pass
            return None                      # cursor moves are not answered (there are ~30 a second)

        label, run, group = self._lookup(message)
        if run is None:
            self.log.emit(f"Ignored: {label}")
            return {"type": "done", "id": msg_id, "ok": False, "reason": label}
        if group not in self.allowed:
            self.log.emit(f"Blocked (not allowed): {gesture} -> {label}")
            return {"type": "done", "id": msg_id, "ok": False, "reason": f"'{label}' is not allowed on that PC"}
        try:
            result = run()   # brightness returns the new level as text, e.g. "75%, software dimming"
        except Exception as e:
            self.log.emit(f"FAILED: {gesture} -> {label}: {e}")
            return {"type": "done", "id": msg_id, "ok": False, "reason": str(e)[:200]}
        detail = f" ({result})" if isinstance(result, str) else ""
        self.log.emit((f"{gesture} -> {label}" if gesture else label) + detail)
        return {"type": "done", "id": msg_id, "ok": True}

    def _lookup(self, message):
        """-> (label, function to run or None, allow-list group). Anything unknown is refused."""
        kind = message.get("type")
        if kind == "action":
            key = message.get("action")
            if key not in ACTION_INFO:
                return f"unknown action '{str(key)[:40]}'", None, None
            label, run = ACTION_INFO[key]
            return label, run, ACTION_GROUP[key]
        if kind == "shortcut":
            keys = message.get("keys")
            if (not isinstance(keys, list) or not 1 <= len(keys) <= 4
                    or not all(isinstance(k, str) and k in pyautogui.KEYBOARD_KEYS for k in keys)):
                return "a shortcut with unknown keys", None, None
            return "Shortcut " + "+".join(keys), lambda: pyautogui.hotkey(*keys), "shortcuts"
        if kind == "mouse":
            button, state = message.get("button"), message.get("state")
            if button not in MOUSE_BUTTONS or state not in ("down", "up"):
                return "a bad mouse message", None, None
            if state == "down":
                def run():
                    pyautogui.mouseDown(button=button)
                    self.held.add(button)
            else:
                def run():
                    pyautogui.mouseUp(button=button)
                    self.held.discard(button)
            return f"{button.title()} button {'pressed' if state == 'down' else 'released'}", run, "clicks"
        if kind == "zoom":
            try:
                amount = max(-150, min(150, int(message.get("amount", 0))))
            except (TypeError, ValueError):
                return "a bad zoom message", None, None

            def run():
                pyautogui.keyDown("ctrl")
                try:
                    pyautogui.scroll(amount)
                finally:
                    pyautogui.keyUp("ctrl")
            return "Zoom in" if amount > 0 else "Zoom out", run, "scroll"
        return f"unknown message type '{str(kind)[:20]}'", None, None

    def _release_buttons(self):
        for button in list(self.held):
            try:
                pyautogui.mouseUp(button=button)
            except Exception:
                pass
        self.held.clear()

    # ----- encryption (TLS) -----

    tls = None              # ssl context, set by the window before start() (see remote.server_tls_context)

    def _serve_secure(self, raw, ip):
        """Runs the TLS handshake for a new connection, then handles it like before (_serve)."""
        if self.tls is None:
            self._close(raw)
            return
        raw.settimeout(5.0)                  # a PC that connects but never finishes the handshake
        try:
            sock = self.tls.wrap_socket(raw, server_side=True)
        except (OSError, ValueError) as e:  # ssl.SSLError is an OSError
            self.log.emit(f"Refused {ip}: not an encrypted Kinesis connection (old Kinesis version?) - {e}")
            self._close(raw)
            return
        self._serve(sock, ip)


# ---------- Window ----------

STYLESHEET = """
QWidget { background-color: #120c20; color: #ece6fa; font-family: "Segoe UI", Arial; font-size: 13px; }
QFrame#card { background-color: #1b1430; border: 1px solid #2f2550; border-radius: 12px; }
QLabel { background: transparent; }
QLabel#title { font-size: 22px; font-weight: 800; color: #b794ff; }
QLabel#caption { font-size: 11px; font-weight: bold; color: #8a7bb0; }
QLabel#big { font-size: 26px; font-weight: bold; color: #ffffff; }
QLabel#fingerprint { font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 15px; color: #5ee39a; }
QLabel#note { font-size: 11px; color: #9d8ec2; }
QPushButton { background-color: #7c4dff; color: white; border: none; border-radius: 8px; padding: 8px 14px;
              font-weight: bold; }
QPushButton:hover { background-color: #9068ff; }
QPushButton#secondary { background-color: #241a3d; border: 1px solid #3a2e63; color: #d2bcff; }
QPushButton#stop { background-color: #3a1626; border: 1px solid #ff7a93; color: #ff7a93; }
QCheckBox { spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #5a4b8a;
                       background-color: #160f29; }
QCheckBox::indicator:checked { background-color: #7c4dff; border-color: #7c4dff; }
QListWidget { background-color: #160f29; border: 1px solid #2f2550; border-radius: 8px; padding: 4px;
              font-size: 12px; }
"""
STATUS_COLOURS = {"idle": "#9d8ec2", "waiting": "#ffb86b", "connected": "#5ee39a", "error": "#ff7a93"}


class ReceiverWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinesis Receiver")
        self.resize(560, 760)
        self.receiver = Receiver()
        self.receiver.log.connect(self.add_log)
        self.receiver.status.connect(self.set_status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        title = QLabel("KINESIS RECEIVER")
        title.setObjectName("title")
        layout.addWidget(title)
        info = QLabel("Lets Kinesis on another PC control this PC over the same Wi-Fi or hotspot. "
                      "Only the actions ticked below are performed.")
        info.setObjectName("note")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Encryption: load this PC's certificate, or make one the first time (saved next to receiver.py)
        tls_error, made = None, False
        try:
            self.receiver.tls, cert_fingerprint, made = server_tls_context(os.path.dirname(SCREENSHOT_DIR))
        except Exception as e:
            tls_error, cert_fingerprint = str(e), None

        # Address and pairing code: type these into Kinesis' Remote Control panel
        card, grid = self._card(layout, QGridLayout)
        ips = local_ip_addresses()
        address = ips[0] if self.receiver.port == DEFAULT_PORT else f"{ips[0]}:{self.receiver.port}"
        for column, (caption, value) in enumerate((("THIS PC'S IP ADDRESS", address),
                                                   ("PAIRING CODE", self.receiver.code))):
            label = QLabel(caption)
            label.setObjectName("caption")
            grid.addWidget(label, 0, column)
            big = QLabel(value)
            big.setObjectName("big")
            big.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(big, 1, column)
            if column == 1:
                self.code_label = big
        if len(ips) > 1:
            other = QLabel("Other addresses of this PC: " + ", ".join(ips[1:]) +
                           "  (use the one on the same network as Kinesis)")
            other.setObjectName("note")
            other.setWordWrap(True)
            grid.addWidget(other, 2, 0, 1, 2)
        if cert_fingerprint:
            caption = QLabel("ENCRYPTED (TLS) - CERTIFICATE FINGERPRINT")
            caption.setObjectName("caption")
            grid.addWidget(caption, 3, 0, 1, 3)
            fp_label = QLabel(short_fingerprint(cert_fingerprint))
            fp_label.setObjectName("fingerprint")
            fp_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            fp_label.setToolTip("Kinesis shows the same code when it connects. If they differ, don't continue.")
            grid.addWidget(fp_label, 4, 0, 1, 3)
        code_btn = QPushButton("New code")
        code_btn.setObjectName("secondary")
        code_btn.setToolTip("Makes a new pairing code. The PC in control stays connected.")
        code_btn.clicked.connect(self.new_code)
        grid.addWidget(code_btn, 1, 2)

        # Status and the start/stop button
        card, box = self._card(layout, QVBoxLayout)
        row = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        font = QFont()
        font.setBold(True)
        self.status_label.setFont(font)
        row.addWidget(self.status_label, 1)
        self.toggle_btn = QPushButton("Stop receiving")
        self.toggle_btn.setObjectName("stop")
        self.toggle_btn.clicked.connect(self.toggle)
        row.addWidget(self.toggle_btn)
        box.addLayout(row)

        # Allow-list
        card, box = self._card(layout, QVBoxLayout)
        caption = QLabel("ALLOWED ACTIONS")
        caption.setObjectName("caption")
        box.addWidget(caption)
        for group, text, default_on, _ in GROUPS:
            check = QCheckBox(text)
            check.setChecked(default_on)
            check.stateChanged.connect(lambda state, group=group: self.set_allowed(group, state == Qt.Checked))
            box.addWidget(check)

        # Log of everything received
        caption = QLabel("RECEIVED ACTIONS")
        caption.setObjectName("caption")
        layout.addWidget(caption)
        self.log_list = QListWidget()
        layout.addWidget(self.log_list, 1)
        tip = QLabel("Not connecting? Both PCs must be on the same Wi-Fi or hotspot, and Windows Firewall must "
                     "allow this app. College or office Wi-Fi often blocks PC-to-PC traffic: use a phone "
                     "hotspot instead.")
        tip.setObjectName("note")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        self.setStyleSheet(STYLESHEET)

        if tls_error:
            self.set_status(f"Encryption isn't available, so receiving is off: {tls_error}", "error")
            self.add_log(f"Couldn't set up encryption: {tls_error}")
            self._show_stopped()
            return
        self.add_log(("Made a new encryption certificate" if made else "Loaded the encryption certificate") +
                     f" (fingerprint {short_fingerprint(cert_fingerprint)})")
        if not self.receiver.start():
            self._show_stopped()

    @staticmethod
    def _card(parent_layout, layout_class):
        card = QFrame()
        card.setObjectName("card")
        inner = layout_class(card)
        inner.setContentsMargins(14, 12, 14, 12)
        parent_layout.addWidget(card)
        return card, inner

    def set_allowed(self, group, on):
        allowed = set(self.receiver.allowed)
        (allowed.add if on else allowed.discard)(group)
        self.receiver.allowed = allowed          # swapped in one go, so the network thread never sees half
        self.add_log(f"{'Allowed' if on else 'Blocked'}: {next(t for g, t, _, _ in GROUPS if g == group)}")

    def new_code(self):
        self.code_label.setText(self.receiver.refresh_code())
        self.add_log("New pairing code made")

    def toggle(self):
        if self.receiver.running:
            self.receiver.stop()
            self._show_stopped()
        elif self.receiver.tls is not None and self.receiver.start():   # never without encryption
            self._set_button("Stop receiving", "stop")

    def _show_stopped(self):
        self._set_button("Start receiving", "")

    def _set_button(self, text, style_name):
        self.toggle_btn.setText(text)
        self.toggle_btn.setObjectName(style_name)
        self.toggle_btn.style().unpolish(self.toggle_btn)   # re-apply the style for the new name
        self.toggle_btn.style().polish(self.toggle_btn)

    def set_status(self, text, colour):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {STATUS_COLOURS.get(colour, '#ece6fa')};")

    def add_log(self, text):
        self.log_list.insertItem(0, f"[{time.strftime('%H:%M:%S')}] {text}")
        while self.log_list.count() > 200:
            self.log_list.takeItem(self.log_list.count() - 1)

    def closeEvent(self, event):
        if self.receiver.running:
            self.receiver.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ReceiverWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
