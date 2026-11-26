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

Encryption (TLS 1.2 or newer): everything, including the pairing code, travels inside a TLS
connection, so nobody else on the Wi-Fi can read or change it. The receiver makes its own
"self-signed" certificate the first time it runs (no certificate authority is involved). Instead,
Kinesis checks the certificate's fingerprint, a short code shown in the receiver window:
  * first connection: Kinesis remembers the fingerprint ("trust on first use") and shows it, so you
    can check it matches the receiver window;
  * later connections: if the fingerprint is different, Kinesis refuses to connect and warns you
    (someone may be pretending to be that PC), until you say the new certificate is fine.

All network work happens in a background thread, so the Kinesis window never freezes. The window
reads what happened from RemoteClient.events (a thread-safe queue) on every frame.
"""
import hashlib
import json
import os
import queue
import select
import shutil
import socket
import ssl
import subprocess
import threading
import time

DEFAULT_PORT = 50607          # any free port works; both sides must use the same one
PROTOCOL_VERSION = 2          # 2 = encrypted with TLS (version 1 was plain text)
CERT_FILE = "receiver_cert.pem"   # the receiver's certificate and private key, made on first run
KEY_FILE = "receiver_key.pem"     # (kept next to receiver.py; never share the key file)
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


# ---------- Encryption (TLS) helpers ----------

def fingerprint(cert_der):
    """SHA-256 of the certificate, as 64 hex digits. Both sides compute it the same way."""
    return hashlib.sha256(cert_der).hexdigest().upper()


def short_fingerprint(full):
    """The first 16 hex digits in groups of 4, e.g. '3FA2 91C0 7B4E 22D1' - easy to compare by eye."""
    return " ".join(full[i:i + 4] for i in range(0, 16, 4))


def _make_certificate(cert_path, key_path):
    """Creates a self-signed certificate + private key (EC P-256, valid 10 years).
    Uses the 'cryptography' package if installed, otherwise the openssl command (Linux/macOS have it)."""
    name = socket.gethostname()[:60] or "kinesis-receiver"
    try:
        import datetime
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError:
        if not shutil.which("openssl"):
            raise RuntimeError("Can't make an encryption certificate: install the 'cryptography' package "
                               "(pip install cryptography) or the openssl command.")
        result = subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt",
                                 "ec_paramgen_curve:prime256v1", "-nodes", "-days", "3650",
                                 "-subj", f"/CN={name}", "-keyout", key_path, "-out", cert_path],
                                capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"openssl couldn't make a certificate: {result.stderr.strip()[:200]}")
    else:
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)   # self-signed
                .public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=3650))
                .sign(key, hashes.SHA256()))
        with open(key_path, "wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
    try:
        os.chmod(key_path, 0o600)   # only this user may read the private key (Linux/macOS)
    except OSError:
        pass


def server_tls_context(folder):
    """For receiver.py: loads (or makes, the first time) its certificate.
    Returns (ssl context, full fingerprint, True if the certificate was just made)."""
    cert_path, key_path = os.path.join(folder, CERT_FILE), os.path.join(folder, KEY_FILE)
    made = not (os.path.exists(cert_path) and os.path.exists(key_path))
    if made:
        _make_certificate(cert_path, key_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2   # no old, weak SSL/TLS versions
    context.load_cert_chain(cert_path, key_path)
    with open(cert_path) as f:
        der = ssl.PEM_cert_to_DER_cert(f.read())
    return context, fingerprint(der), made


def client_tls_context():
    """For Kinesis. The certificate is self-signed, so Python's usual check (signed by a known
    authority, matching a domain name) can't work; the fingerprint is checked instead (see RemoteClient)."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


class UntrustedCertificate(ConnectionError):
    """The receiver's fingerprint isn't the one Kinesis remembered."""

    def __init__(self, new_fingerprint):
        super().__init__("certificate changed")
        self.new_fingerprint = new_fingerprint


# ---------- Kinesis side: the client ----------

class RemoteClient:
    """Connects to receiver.py on another PC and sends it actions, without blocking the UI.

    Use from the UI thread:
        client.connect("192.168.43.20", 50607, "482913")
        client.send_action("volume_up", gesture="thumbs_up", label="Volume Up")
        ... every frame:  while not client.events.empty(): handle(client.events.get())
        client.disconnect()

    Events put on client.events (tuples):
        ("connected", receiver_name, fingerprint, tls_version)
        ("failed", reason)             could not connect / wrong code
        ("untrusted", new_fingerprint) the receiver's certificate isn't the one we remembered
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

    def connect(self, host, port, code, trusted_fingerprint=None):
        """Starts connecting in the background. The result arrives as an event.
        trusted_fingerprint: the receiver's fingerprint from last time (None = first connection)."""
        self.disconnect(quiet=True)
        self._stop = threading.Event()
        self._outbox = queue.Queue()
        self._sent = {}
        self._cursor = None
        self.address = (host, port)
        self.state = "connecting"
        self._thread = threading.Thread(target=self._run, args=(host, port, code, trusted_fingerprint, self._stop),
                                        daemon=True)
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

    def _run(self, host, port, code, trusted_fingerprint, stop):
        sock = None
        try:
            try:
                raw = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)
            except socket.timeout:
                raise ConnectionError(f"No answer from {host}:{port}. Check the IP, that the receiver is "
                                      "running, that both PCs are on the same Wi-Fi/hotspot, and that "
                                      "Windows Firewall allows the receiver.")
            except ConnectionRefusedError:
                raise ConnectionError(f"{host} refused the connection - is the receiver running "
                                      f"(and listening on port {port})?")
            except OSError as e:
                raise ConnectionError(f"Can't reach {host}:{port} ({e.strerror or e}).")
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # send small messages at once
            try:
                # TLS handshake: agree on keys; from here on everything is encrypted
                sock = client_tls_context().wrap_socket(raw)
            except (ssl.SSLError, OSError) as e:
                raw.close()
                raise ConnectionError(f"Encrypted (TLS) connection failed ({e}). If that PC runs an older "
                                      "receiver.py, copy the new receiver.py and remote.py to it.")
            # Is this the same receiver as last time? Checked BEFORE the pairing code is sent.
            seen = fingerprint(sock.getpeercert(binary_form=True))
            if trusted_fingerprint and seen != trusted_fingerprint:
                raise UntrustedCertificate(seen)
            reader = LineReader()
            sock.sendall(encode({"type": "hello", "code": code, "name": socket.gethostname(),
                                 "version": PROTOCOL_VERSION}))
            reply = self._wait_for_reply(sock, reader, stop)
            if reply.get("type") != "welcome":
                raise ConnectionError(reply.get("reason", "The receiver refused the connection."))
            self.receiver_name = str(reply.get("name", host))[:60]
            self.state = "connected"
            self.events.put(("connected", self.receiver_name, seen, sock.version()))
            self._loop(sock, reader, stop)
        except UntrustedCertificate as e:
            if not stop.is_set():
                self.state = "idle"
                self.events.put(("untrusted", e.new_fingerprint))
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
        sock.settimeout(3.0)                  # a send that takes longer than this means the link is gone
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
                try:
                    sock.sendall(b"".join(encode(m) for m in outgoing))
                except OSError as e:
                    raise ConnectionError(f"Sending failed - the link to the receiver dropped ({e}).")
                if any(m.get("type") == "bye" for m in outgoing):
                    return

            # 2. Read replies (wait at most 5 ms, so new actions go out quickly). TLS may already hold
            #    decrypted bytes that select() can't see, so sock.pending() is checked too.
            if sock.pending() or select.select([sock], [], [], 0.005)[0]:
                sock.settimeout(0.2)
                try:
                    data = sock.recv(65536)
                except (socket.timeout, ssl.SSLWantReadError):
                    data = None               # only part of an encrypted record so far
                except OSError as e:
                    raise ConnectionError(f"The link to the receiver dropped ({e}).")
                finally:
                    sock.settimeout(3.0)
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
