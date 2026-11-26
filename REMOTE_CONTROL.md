# Remote control over Wi-Fi

Kinesis can control a **second PC** on the same Wi-Fi network or phone hotspot. The webcam and the gesture model stay on your laptop; only the action (for example "next slide") is sent to the other PC, which performs it.

```
 Your laptop (Kinesis + webcam)                       Other PC (receiver.py)
 gesture -> action  --- {"type":"action","action":"next_slide"} --->  checks the code and the allow-list
                    <-- {"type":"done","ok":true} -------------------  presses the Right arrow key
 shows "Last sent: swipe_right -> Next Slide (8 ms)"
```

Everything between the two PCs is **encrypted with TLS** (1.2 or newer), including the pairing code.

## The two parts

| Part | Runs on | What it does |
|------|---------|--------------|
| `receiver.py` | The PC you want to control | Small window: shows that PC's **IP address** and a random 6-digit **pairing code**, who is controlling it, which actions are **allowed** (tick boxes), a log of every action received, and a **Stop receiving** button. No camera, no ML. |
| Remote Control panel in Kinesis | Your laptop | Type the receiver's IP and code, press **Connect**, then choose **Send actions to: Remote PC / This PC**. Shows the connection status and *Last sent: gesture -> action (latency ms)*. |

## Demo with two PCs on a phone hotspot

1. Turn on your phone's hotspot and connect **both** PCs to it.
2. Copy the project folder to the other PC (it only needs `receiver.py` and `remote.py`) and install the two libraries it uses:
   `pip install PyQt5 PyAutoGUI`
3. On the other PC run: `python receiver.py`
   - The first time, **Windows Firewall** asks whether to allow Python. Tick **Private networks** and also **Public networks** (Windows often marks a phone hotspot as public), then click **Allow access**.
   - The window shows something like `IP 192.168.43.20`, `Pairing code 482913` and a green **certificate fingerprint** such as `EDDE 9056 3E6D 5FEB`.
4. On your laptop run Kinesis as usual (`python Kinesis_app.py`). Open the **Remote Control** section, type the IP and the code, and press **Connect**.
5. Kinesis says *Connected ... encrypted with TLSv1.3. Fingerprint EDDE 9056 3E6D 5FEB (first connection: check it matches the receiver window)*. Check the fingerprint is the same as on the receiver. When it says **Connected**, *Send actions to* switches to **Remote PC** and the status pill reads **LIVE → REMOTE PC**. Your gestures now act on the other PC (for example, open a PowerPoint there and swipe to change slides). The receiver's log shows each action as it arrives.
6. Switch back with **This PC**, press **Disconnect**, or press **Stop receiving** on the other PC. If the link drops, Kinesis goes back to controlling your own laptop by itself and says so.

**Practice on one PC first:** run `python receiver.py` and Kinesis on the same laptop, connect to `127.0.0.1` with the code shown. Everything works the same, just on one screen.

## Safety

- **Encrypted:** all traffic uses TLS 1.2+, so other devices on the Wi-Fi can't read the pairing code or the actions, or change them on the way. On its first run the receiver makes its own certificate (`receiver_cert.pem` + `receiver_key.pem`, next to `receiver.py`; nothing to install). Kinesis remembers the receiver's fingerprint on the first connection ("trust on first use"). If the fingerprint is ever different, Kinesis refuses to connect **before sending the pairing code** and shows a warning. If you know why it changed (you deleted the two .pem files, or the receiver moved to another PC) and the fingerprint matches the receiver window, click **Trust the new certificate and connect**.
- The receiver must be started by the other PC's owner. Nothing can connect to a PC that isn't running it.
- The 6-digit pairing code is required. After 5 wrong codes, that PC is blocked until **New code** is pressed. Only one Kinesis PC can be in control at a time.
- **Allow-list only:** the receiver performs only Kinesis' known actions in the ticked groups (clicks, cursor, scroll/zoom, volume/media, slides/windows, copy/paste/undo, screenshots, brightness). Anything else is refused and logged. Custom keyboard shortcuts are **off** by default because a shortcut can be any key combination.
- If the connection drops, the receiver lets go of any mouse button Kinesis was holding (so a drag never gets stuck).

## Receiver on Linux or macOS

`receiver.py` also runs on Linux (X11 desktops such as Xfce, e.g. Kali in VMware) and macOS. It needs the same two files, `receiver.py` and `remote.py`. Copy **both**, and recopy them whenever they change. The certificate is made with Python's `cryptography` package if installed (Kali has it), otherwise with the `openssl` command (always on Kali), so no extra install is needed for encryption.

```bash
# Linux (Debian / Ubuntu / Kali)
sudo apt install python3-pyqt5 python3-tk python3-dev scrot   # if PyQt5 or PyAutoGUI complain
pip install PyAutoGUI
sudo apt install brightnessctl x11-xserver-utils               # for brightness (see below)
python3 receiver.py
```

**Brightness** works differently on each system, so the receiver tries these in order and uses the first that works:

| System | 1st try | If that fails |
|--------|---------|---------------|
| Windows | Windows' own brightness setting (laptop screens) | - |
| Linux | `brightnessctl`: the real backlight (laptops; you may need `sudo usermod -aG video $USER`, then log out and in) | `xrandr --brightness`: dims the picture in software. Works in a VM and on external monitors, never below 20% |
| macOS | the `brightness` tool (`brew install brightness`) | presses the brightness keys (allow Terminal under System Settings -> Privacy -> Accessibility) |

In a virtual machine there is no real backlight, so brightnessctl fails and xrandr's software dimming is used. The receiver's log shows the new level, e.g. `thumbs_up -> Brightness Up (85%, software dimming)`. If nothing works, the log (and Kinesis' activity log) says what to install. xrandr needs an X11 session; it doesn't work on Wayland.

## If it won't connect

- Both PCs must be on the **same** network. Use the IP shown in the receiver window (if it lists several, pick the one that starts like your laptop's IP, e.g. both `192.168.43.x`).
- *"Encrypted (TLS) connection failed"* in Kinesis, or *"not an encrypted Kinesis connection"* in the receiver log: one PC has the new files and the other the old ones. Copy the new `receiver.py` **and** `remote.py` to the receiver PC.
- *"Encryption isn't available"* in the receiver: it couldn't make a certificate. Run `pip install cryptography` (Windows) or `sudo apt install openssl` (Linux).
- Check Windows Firewall allowed the receiver (Windows Security -> Firewall -> *Allow an app through firewall* -> Python).
- **College or office Wi-Fi often blocks PC-to-PC traffic** ("client isolation"). Use a phone hotspot instead.

## Technical notes (for the report or viva)

- **Transport:** TCP on port `50607`, wrapped in **TLS 1.2+** (Python's `ssl` module), one JSON message per line. The receiver listens on all network cards (`0.0.0.0`).
- **Certificate and trust:** the receiver's certificate is self-signed (elliptic-curve P-256 key, valid 10 years), so there's no certificate authority to vouch for it. Kinesis instead pins its **SHA-256 fingerprint** on first use, the same idea SSH uses for host keys. A changed fingerprint (a possible man-in-the-middle) stops the connection before the pairing code is sent.
- **Messages:** `hello` (with the code) -> `welcome` / `refused`; then `action`, `shortcut`, `mouse` (press/release for drag), `zoom` and `cursor`. Every action gets a `done` reply, used to measure the **round-trip latency**. Kinesis sends a `ping` every 2 s; if either side hears nothing for 8-10 s, the link is treated as lost.
- **Cursor:** sent as a fraction of the screen (0-1), so it lands in the same place on a screen of any resolution. Only the newest position is sent, so a slow network never builds up a lag.
- **Never freezes:** all network work runs in a background thread; the Kinesis window reads its results once per frame. On the receiver, network threads talk to the window through Qt signals.
- **Over the internet later:** install [Tailscale](https://tailscale.com) on both PCs and use the receiver's Tailscale IP (`100.x.x.x`). No code changes are needed.
- **Make an .exe of the receiver:** `pip install pyinstaller`, then `pyinstaller --onefile --windowed --name "Kinesis Receiver" receiver.py`. The .exe appears in `dist/`; copy it to the other PC (no Python needed there).
