# Kinesis: Personalized Gesture-Based HCI System

Control your computer with hand gestures through your webcam. Kinesis lets you **record your own gestures, train a small neural network on them, and map each gesture to an action** such as clicking, scrolling, changing the volume, or taking a screenshot.

## Features

- **Custom gestures:** record any gesture you like and train the model in a few clicks. No pre-built gesture set is forced on you.
- **Gesture-to-action mapping:** choose what each gesture does from the app's mapping list.
- **Cursor control:** point with your hand to move the mouse, with an adjustable sensitivity slider.
- **Two-hand zoom:** move both hands apart or together to zoom in and out (can be turned on or off).
- **Idle lock and unlock:** the app locks itself when no hand is seen for a while, and you unlock it by holding a chosen gesture (or with a button).
- **Confidence threshold:** set how sure the model must be before an action fires.
- **Built-in model tools:** dataset info, dataset balancing, optional data augmentation, a self-test for accuracy, and a confusion matrix.
- **Activity log:** see what the app did and when.
- **Safety options:** pause recognition at any time, reset all data, or restore default backup files.

## Available actions

Left click, right click, double click, move cursor, volume up/down, scroll up/down, switch window (next/previous), take screenshot, or none.

## How it works

1. Your webcam feed goes through [MediaPipe Hands](https://developers.google.com/mediapipe), which finds 21 landmarks on each hand.
2. The landmarks are normalized relative to your wrist, so the position of your hand in the frame doesn't matter.
3. A small neural network built with TensorFlow/Keras (dense layers) classifies the gesture.
4. A gesture must be stable across several frames, and stay above the confidence threshold, before its action runs. There is also a cooldown so actions don't repeat by accident.
5. Actions are performed with PyAutoGUI. The interface is built with PyQt5.

## Requirements

- **Windows** (the app uses the DirectShow camera backend and Windows-specific features)
- Python 3.9 to 3.11 recommended
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

## Files the app creates

These are made while you use the app and are not part of the repository:

| File | Purpose |
|------|---------|
| `gesture_data.csv` | Your recorded gesture data |
| `gesture_model.keras` | Your trained model |
| `label_encoder.pkl` | Gesture name encoder |
| `gesture_mapping.json` | Your gesture-to-action settings |
| `activity_log.txt` | Activity log |
| `screenshot_*.png` | Screenshots taken by the screenshot gesture |

## Built with

Python, OpenCV, MediaPipe, TensorFlow/Keras, scikit-learn, pandas, NumPy, PyAutoGUI, PyQt5

## Author

Made by **Himanshu Chauhan** ([GitHub](https://github.com/himanshulko22-a11y))
