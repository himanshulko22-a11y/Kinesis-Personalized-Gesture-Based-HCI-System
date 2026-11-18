"""Switches the webcam back to automatic exposure (fixes a dark or black picture).

Run:  python camera_reset.py   (close Kinesis and other camera apps first)
"""
from camera import reset_exposure

print("Switching the webcam to automatic exposure... keep your face in view.")
level = reset_exposure(seconds=5)
print(f"Picture brightness now: {level:.0f}/255")
if level > 60:
    print("Done - the picture is normal again.")
else:
    print("Still dark. Unplug and replug the webcam (or restart the PC), then run this again.")
