"""Finds the fastest camera settings for your webcam.

Run:  python camera_test.py   (close Kinesis first - only one app can use the camera)
Sit where you normally use Kinesis. Takes about 40 seconds.
"""
import time

import cv2

from camera import open_camera

SETTINGS = [
    # (name, backend, mjpg, exposure)
    ("DirectShow + MJPG", "dshow", True, None),
    ("DirectShow, default format", "dshow", False, None),
    ("Media Foundation", "msmf", False, None),
    ("Media Foundation + MJPG", "msmf", True, None),
]


def measure(backend, mjpg, exposure, seconds=2.5):
    for _ in range(4):   # Windows needs a moment to release the camera between tests
        cap = open_camera(backend, mjpg, exposure)
        if cap.isOpened():
            break
        cap.release()
        time.sleep(1.5)
    else:
        return None
    try:
        start = time.perf_counter()
        while time.perf_counter() - start < 1.0:   # let the camera settle
            cap.read()
        frames, brightness = 0, 0.0
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            ok, frame = cap.read()
            if ok:
                frames += 1
                brightness = frame.mean()
        fps = frames / (time.perf_counter() - start)
        size = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        return fps, brightness, size
    finally:
        cap.release()
        time.sleep(2)


def main():
    print("Testing camera settings... keep your face/hand in view.\n")
    results = []
    for name, backend, mjpg, exposure in SETTINGS:
        result = measure(backend, mjpg, exposure)
        if result is None:
            print(f"  {name:42s} could not open camera")
            continue
        fps, brightness, size = result
        results.append((fps, brightness, name, backend, mjpg, exposure))
        print(f"  {name:42s} {fps:5.1f} fps   brightness {brightness:5.1f}/255   {size}")

    usable = [r for r in results if r[1] > 60]   # too dark to track hands otherwise
    if not usable:
        print("\nAll pictures are too dark. Run:  python camera_reset.py")
    else:
        fps, brightness, name, backend, mjpg, exposure = max(usable)
        print(f"\nFastest usable: {name} ({fps:.1f} fps).")
        print("Put these lines in config.py (or send this output to Claude):")
        print(f'    CAMERA_BACKEND = "{backend}"')
        print(f"    CAMERA_MJPG = {mjpg}")
        print(f"    CAMERA_EXPOSURE = {exposure}")
    print("\nTip: more light on your hand usually raises the frame rate the most.")


if __name__ == "__main__":
    main()
