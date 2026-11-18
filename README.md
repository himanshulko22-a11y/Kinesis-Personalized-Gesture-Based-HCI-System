# Kinesis: Personalized Gesture-Based HCI System

Control your computer with hand gestures through your webcam. Kinesis lets you **record your own gestures, train a small neural network on them, and map each gesture to an action** such as clicking, scrolling, changing the volume, or taking a screenshot.

## Screenshots

<img width="960" height="507" alt="123" src="https://github.com/user-attachments/assets/f7f543c7-1d98-4944-a111-887d6b0e4377" />

<img width="960" height="506" alt="321" src="https://github.com/user-attachments/assets/14579d4a-c012-4320-ac4b-a974fe147fe9" />

## Features

- **Custom gestures:** record any gesture you like and train the model in a few clicks. No pre-built gesture set is forced on you.
- **Gesture-to-action mapping:** choose what each gesture does from the app's mapping list.
- **Cursor control:** point with your hand to move the mouse, with an adjustable sensitivity slider. The cursor follows the base of the middle finger (landmark 9), which shakes less than the wrist or a fingertip, and is smoothed with a One Euro filter plus a hold zone, so it stays still when your hand rests and keeps up when you move fast. A **Cursor Steadiness** slider adjusts the hold zone.
- **Two-hand zoom:** move both hands apart or together to zoom in and out (can be turned on or off).
- **Idle lock and unlock:** the app locks itself when no hand is seen for a while, and you unlock it by holding a chosen gesture (or with a button).
- **Confidence threshold:** set how sure the model must be before an action fires.
- **Built-in model tools:** dataset info, dataset balancing, optional data augmentation, a self-test for accuracy, and a confusion matrix.
- **Similar-gesture warning:** right after you record a gesture, Kinesis compares its hand shape with your other gestures and warns you if they're too alike, before you train.
- **Learning from mistakes:** if a gesture fires by accident, mark it within 5 seconds (button, the mistake hotkey - **Z** by default, changeable in the app - or a gesture mapped to *Mark Last Action as Mistake*). That hand shape is saved as "not a gesture", and the model learns not to fire on it at the next retrain.
- **Speed readout:** camera FPS, app FPS and time per step, shown on the video.
- **Activity log:** see what the app did and when.
- **Safety options:** pause recognition at any time, reset all data, and save or restore a default setup. Gestures never fire while a dialog is open.

## Available actions

Left click, right click, double click, move cursor, volume up/down, scroll up/down, switch window (next/previous), take screenshot, mark last action as mistake, or none. Ready-made shortcuts: media play/pause, next/previous track, mute, next/previous slide, show desktop, task view, copy, paste, undo, Enter and Escape. You can also add **any keyboard shortcut** of your own (e.g. Ctrl+S, Win+E, F5) under **Custom Shortcuts**; it is saved in `settings.json`.

## How it works

1. Your webcam feed goes through [MediaPipe Hands](https://developers.google.com/mediapipe), which finds 21 landmarks on each hand.
2. The landmarks are normalized relative to your wrist, so the position of your hand in the frame doesn't matter.
3. A small neural network built with TensorFlow/Keras (dense layers) classifies the gesture.
4. A gesture must be stable across several frames, and stay above the confidence threshold, before its action runs. Clicks, screenshots and window switching fire **once** per gesture: holding the gesture doesn't repeat them. Volume and scroll keep repeating while held.
5. Actions are performed with PyAutoGUI. The interface is built with PyQt5.

## Requirements

- **Windows** (the app uses the DirectShow camera backend and Windows-specific features)
- Python 3.11 (TensorFlow 2.15 supports up to 3.11 and pandas 3 needs 3.11 or newer)
- A webcam

## Installation

```bash
git clone https://github.com/himanshulko22-a11y/Kinesis-Personalized-Gesture-Based-HCI-System.git
cd Kinesis-Personalized-Gesture-Based-HCI-System
pip install -r requirements.txt
```

## Usage

```bash
python Kinesis_app.py
```

### First-time setup

No trained model is included, so you start by teaching the app your gestures:

1. Click **Record Gesture**, type a name (for example `fist` or `thumbs_up`), and hold that gesture in front of the camera until recording finishes.
2. Record at least two different gestures.
3. Click **Retrain Model**.
4. Under **Gesture Mapping**, pick an action for each gesture.
5. Use **Self-Test Accuracy** to check how well the model recognizes your gestures. If two gestures get confused, record more samples with the **+** button.

### Tips

- Record each gesture in good lighting, and move your hand slightly while recording so the model sees some variety.
- Keep the number of samples roughly equal across gestures. **Balance Dataset** can help.
- Raise the confidence threshold if you get accidental actions, and lower it if gestures aren't detected.

## Project files

| File | Contents |
|------|----------|
| `Kinesis_app.py` | Main window and the per-frame loop (run this) |
| `gestures.py` | Landmarks, smoothing, dataset, model training, gesture-to-action mapping |
| `actions.py` | What each action does (click, volume, scroll...) |
| `widgets.py` | Small UI pieces: buttons, mapping row, confusion matrix |
| `camera.py` | Reads the webcam in a background thread so the app always gets the newest frame |
| `config.py` | All settings and file paths |
| `camera_test.py` | Measures which camera mode is fastest on your webcam |
| `camera_reset.py` | Switches the webcam back to automatic exposure |
| `style.py` | Window theme |

## Included default dataset

The repository includes a ready-made default setup: **15 static hand poses** (9,000 frames, 600 per pose, recorded with both hands) with a trained model and action mapping. After downloading, click **Restore Defaults** to use it straight away, or record your own gestures instead.

Poses: open_palm, fist, point, pinch, ok_sign, thumbs_up, thumbs_down, peace, three, four, rock, call_me, L_sign, point_down, pinky_up.

The data contains only MediaPipe hand-landmark coordinates (21 points per frame, relative to the wrist), with no images or video. These are the `*_backup` files; **Save as Default** replaces them with your own setup.

## Files the app creates

These are made next to `Kinesis_app.py` while you use the app and are not uploaded:

| File | Purpose |
|------|---------|
| `gesture_data.csv` | Your recorded gesture data |
| `gesture_model.keras` | Your trained model |
| `label_encoder.pkl` | Gesture name encoder |
| `gesture_mapping.json` | Your gesture-to-action settings |
| `activity_log.txt` | Activity log |
| `screenshots/` | Screenshots taken by the screenshot gesture |
| `settings.json` | Your mistake hotkey and custom shortcuts |

The `*_backup` files (your saved default setup) **are** part of the repository - see *Included default dataset* above.

## Built with

Python, OpenCV, MediaPipe, TensorFlow/Keras, scikit-learn, pandas, NumPy, PyAutoGUI, PyQt5

## Author

Made by **Himanshu Chauhan** ([GitHub](https://github.com/himanshulko22-a11y))
