"""Remote control over local Wi-Fi: Kinesis on this PC sends its actions to another PC.

How it works (in plain words):
  * The OTHER PC runs receiver.py. It listens on a TCP port and shows its IP address and a
    6-digit pairing code.
  * Kinesis connects to that IP, sends the pairing code, and from then on, instead of pressing
    keys or clicking on this PC, it sends a small JSON message like
        {"type": "action", "id": 7, "action": "volume_up", "gesture": "thumbs_up"}
    The receiver checks the action is on its allow-list and performs it with PyAutoGUI.
  * The receiver answers {"type": "done", "id": 7, "ok": true}, so Kinesis can show the
    round-trip time (latency) in milliseconds.

Each message is one line of JSON text ending in a newline ("\n"). This file only uses Python's
standard library, so receiver.py can share it and still be built into a small .exe.

All network work happens in a background thread, so the Kinesis window never freezes. The window
reads what happened from RemoteClient.events (a thread-safe queue) on every frame.
"""
import json
import queue
import select
import socket
import threading
import time

DEFAULT_PORT = 50607          # any free port works; both sides must use the same one
PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 4096         # longer messages are refused (nothing we send comes close)
CONNECT_TIMEOUT = 4.0         # seconds to reach the receiver before giving up
HANDSHAKE_TIMEOUT = 6.0       # seconds to wait for the receiver to accept the pairing code
PING_EVERY = 2.0              # Kinesis says "still here" this often...
SILENCE_LIMIT = 8.0           # ...and if the other side is silent this long, the link is treated as lost


# ---------- Shared helpers (used by both Kinesis and receiver.py) ----------

def encode(message):
    """dict -> one line of JSON as bytes."""
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


class LineReader:
    """Collects bytes from a socket and hands back complete JSON messages, one per line.
    (TCP is a stream: one recv() may hold half a message or several messages.)"""

    def __init__(self):
        self.buffer = b""

    def feed(self, data):
        """Adds received bytes. Returns the list of complete messages (dicts) found so far."""
        self.buffer += data
        messages = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                message = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue                     # ignore garbage instead of crashing
            if isinstance(message, dict):
                messages.append(message)
        if len(self.buffer) > MAX_LINE_BYTES:
            raise ValueError("message too long")
        return messages


def local_ip_addresses():
    """This PC's IP addresses on the local network (Wi-Fi / hotspot / LAN), best guess first."""
    found = []
    try:
        # "Connecting" a UDP socket sends nothing; it just asks Windows which network card it would use.
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        found.append(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass                                 # no default route (e.g. a hotspot with no internet)
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in found:
                found.append(ip)
    except OSError:
        pass
    usable = [ip for ip in found if not ip.startswith(("127.", "169.254."))]
    return usable or ["127.0.0.1"]


def parse_address(text):
    """'192.168.1.5' or '192.168.1.5:50607' -> ('192.168.1.5', 50607). Raises ValueError."""
    text = text.strip()
    if not text:
        raise ValueError("Type the receiver's IP address (shown in the receiver window).")
    host, port = text, DEFAULT_PORT
    if text.count(":") == 1:                 # host:port (IPv6 addresses aren't used here)
        host, port_text = text.split(":")
        if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
            raise ValueError(f"'{port_text}' isn't a valid port number.")
        port = int(port_text)
    if " " in host or not host:
        raise ValueError(f"'{text}' doesn't look like an IP address.")
    return host, port


# ---------- Kinesis side: the client ----------

class RemoteClient:
    """Connects to receiver.py on another PC and sends it actions, without blocking the UI.

    Use from the UI thread:
        client.connect("192.168.43.20", 50607, "482913")
        client.send_action("volume_up", gesture="thumbs_up", label="Volume Up")
        ... every frame:  while not client.events.empty(): handle(client.events.get())
        client.disconnect()

    Events put on client.events (tuples):
        ("connected", receiver_name)
        ("failed", reason)             could not connect / wrong code
        ("lost", reason)               was connected, then the link dropped
        ("disconnected",)              you pressed Disconnect
        ("done", info)                 the receiver finished an action; info has
                                       gesture, label, ok, reason, latency_ms
    """

    def __init__(self):
        self.events = queue.Queue()
        self.state = "idle"                  # idle | connecting | connected
        self.receiver_name = ""
        self.address = None
        self._outbox = queue.Queue()         # messages waiting to be sent by the thread
        self._cursor = None                  # newest cursor position not sent yet (older ones are skipped)
        self._cursor_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._next_id = 0
        self._sent = {}                      # id -> (time sent, gesture, label), to measure latency

    @property
    def connected(self):
        return self.state == "connected"

    # ----- called from the UI thread -----

    def connect(self, host, port, code):
        """Starts connecting in the background. The result arrives as an event."""
        self.disconnect(quiet=True)
        self._stop = threading.Event()
        self._outbox = queue.Queue()
        self._sent = {}
        self._cursor = None
        self.address = (host, port)
        self.state = "connecting"
        self._thread = threading.Thread(target=self._run, args=(host, port, code, self._stop), daemon=True)
        self._thread.start()

    def disconnect(self, quiet=False):
        """Closes the connection (the receiver lets go of any held mouse button)."""
        if self._thread is None:
            return
        if self.state == "connected":
            self._outbox.put({"type": "bye"})
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        was_open = self.state != "idle"
        self.state = "idle"
        if was_open and not quiet:
            self.events.put(("disconnected",))

    def send_action(self, action_key, gesture="", label=""):
        """One of Kinesis' actions by name (e.g. 'volume_up'). The receiver decides if it's allowed."""
        self._send({"type": "action", "action": action_key}, gesture, label)

    def send_shortcut(self, keys, gesture="", label=""):
        """A custom keyboard shortcut, e.g. ['ctrl', 's']."""
        self._send({"type": "shortcut", "keys": list(keys)}, gesture, label)

    def send_mouse(self, button, pressed, gesture="", label=""):
        """Press (pressed=True) or release a mouse button, for hold-click and drag."""
        self._send({"type": "mouse", "button": button, "state": "down" if pressed else "up"}, gesture, label)

    def send_zoom(self, amount):
        """Ctrl+scroll: positive = zoom in, negative = zoom out."""
        self._send({"type": "zoom", "amount": int(amount)}, "two hands", "Zoom", track=False)

    def move_cursor(self, fx, fy):
        """Moves the remote cursor. fx, fy are 0-1 fractions of the screen, so screens of any
        size match. Only the newest position is sent, so a slow network never builds up a lag."""
        with self._cursor_lock:
            self._cursor = (round(min(max(fx, 0.0), 1.0), 4), round(min(max(fy, 0.0), 1.0), 4))

    def _send(self, message, gesture, label, track=True):
        if not self.connected:
            return
        self._next_id += 1
        message["id"] = self._next_id
        message["gesture"] = gesture
        if track:
            self._sent[self._next_id] = (time.perf_counter(), gesture, label)
        self._outbox.put(message)

    # ----- background thread -----

    def _run(self, host, port, code, stop):
        sock = None
        try:
            try:
                sock = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)
            except socket.timeout:
                raise ConnectionError(f"No answer from {host}:{port}. Check the IP, that the receiver is "
                                      "running, that both PCs are on the same Wi-Fi/hotspot, and that "
                                      "Windows Firewall allows the receiver.")
            except ConnectionRefusedError:
                raise ConnectionError(f"{host} refused the connection - is the receiver running "
                                      f"(and listening on port {port})?")
            except OSError as e:
                raise ConnectionError(f"Can't reach {host}:{port} ({e.strerror or e}).")
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # send small messages at once
            reader = LineReader()
            sock.sendall(encode({"type": "hello", "code": code, "name": socket.gethostname(),
                                 "version": PROTOCOL_VERSION}))
            reply = self._wait_for_reply(sock, reader, stop)
            if reply.get("type") != "welcome":
                raise ConnectionError(reply.get("reason", "The receiver refused the connection."))
            self.receiver_name = str(reply.get("name", host))[:60]
            self.state = "connected"
            self.events.put(("connected", self.receiver_name))
            self._loop(sock, reader, stop)
        except ConnectionError as e:
            if not stop.is_set():
                kind = "lost" if self.state == "connected" else "failed"
                self.state = "idle"
                self.events.put((kind, str(e)))
        except Exception as e:                # never let the thread die silently
            if not stop.is_set():
                kind = "lost" if self.state == "connected" else "failed"
                self.state = "idle"
                self.events.put((kind, f"Unexpected error: {e}"))
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    @staticmethod
    def _wait_for_reply(sock, reader, stop):
        deadline = time.time() + HANDSHAKE_TIMEOUT
        sock.settimeout(0.2)
        while time.time() < deadline and not stop.is_set():
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError as e:
                raise ConnectionError(f"Connection closed during pairing ({e}).")
            if not data:
                raise ConnectionError("The receiver closed the connection during pairing.")
            messages = reader.feed(data)
            if messages:
                return messages[0]
        raise ConnectionError("The receiver didn't answer the pairing request in time.")

    def _loop(self, sock, reader, stop):
        sock.setblocking(False)
        last_heard = last_ping = time.time()
        while not stop.is_set():
            # 1. Send everything waiting: actions first, then the newest cursor position.
            outgoing = []
            while True:
                try:
                    outgoing.append(self._outbox.get_nowait())
                except queue.Empty:
                    break
            with self._cursor_lock:
                cursor, self._cursor = self._cursor, None
            if cursor is not None:
                outgoing.append({"type": "cursor", "x": cursor[0], "y": cursor[1]})
            now = time.time()
            if now - last_ping >= PING_EVERY:
                outgoing.append({"type": "ping"})
                last_ping = now
            if outgoing:
                data = b"".join(encode(m) for m in outgoing)
                sock.setblocking(True)
                sock.settimeout(3.0)
                try:
                    sock.sendall(data)
                except OSError as e:
                    raise ConnectionError(f"Sending failed - the link to the receiver dropped ({e}).")
                sock.setblocking(False)
                if any(m.get("type") == "bye" for m in outgoing):
                    return

            # 2. Read replies (wait at most 5 ms, so new actions go out quickly).
            readable, _, _ = select.select([sock], [], [], 0.005)
            if readable:
                try:
                    data = sock.recv(65536)
                except BlockingIOError:
                    data = None
                except OSError as e:
                    raise ConnectionError(f"The link to the receiver dropped ({e}).")
                if data == b"":
                    raise ConnectionError("The receiver closed the connection (it was stopped or closed).")
                if data:
                    last_heard = time.time()
                    try:
                        replies = reader.feed(data)
                    except ValueError:
                        raise ConnectionError("The receiver sent something unexpected.")
                    for reply in replies:
                        self._handle_reply(reply)
            if time.time() - last_heard > SILENCE_LIMIT:
                raise ConnectionError(f"No reply from the receiver for {SILENCE_LIMIT:.0f} s - "
                                      "Wi-Fi dropped or the other PC went to sleep.")

    def _handle_reply(self, reply):
        kind = reply.get("type")
        if kind == "done":
            sent = self._sent.pop(reply.get("id"), None)
            if sent is None:
                return
            started, gesture, label = sent
            self.events.put(("done", {"gesture": gesture, "label": label, "ok": bool(reply.get("ok")),
                                      "reason": str(reply.get("reason", "")),
                                      "latency_ms": (time.perf_counter() - started) * 1000}))
        elif kind == "stopping":
            raise ConnectionError("The other PC pressed 'Stop receiving'.")
